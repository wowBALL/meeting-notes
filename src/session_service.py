"""service ควบคุมการอัดที่หน้าเว็บคุยด้วย -- bind 127.0.0.1 เท่านั้น

ตัวอัดรันเป็น thread ในตัว service ไม่ใช่ process ลูก เพราะ record_streams_to_session
รับ threading.Event เป็นสัญญาณหยุดอยู่แล้ว ปุ่มปิดห้องจึงเป็น stop_event.set() ตรง ๆ
ไม่ต้องปลอมสัญญาณ Ctrl+C ข้าม process บน Windows

ผลพลอยได้ที่ตั้งใจ: ปิดหน้าเว็บระหว่างประชุมแล้วยังอัดต่อ เพราะการอัดไม่ได้อยู่
ในหน้าเว็บ เปิดใหม่เมื่อไหร่ก็เห็นสถานะเดิม
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src import activity
from src.record import run_recording

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WORKER_PROBE_CACHE_SECONDS = 10
ACTIVITY_LIMIT = 200

# คำสั่งเดียวกับที่ start-meeting.bat:36 ใช้ตรวจว่า watcher รันอยู่หรือไม่
_WORKER_PROBE_COMMAND = [
    "powershell",
    "-NoProfile",
    "-Command",
    "if (Get-CimInstance Win32_Process | Where-Object { "
    "$_.Name -match 'python' -and $_.CommandLine -like '*src.main*' }) "
    "{ exit 0 } exit 1",
]


def probe_worker() -> bool:
    """watcher รันอยู่หรือไม่

    เลือกเช็ค process แทนการให้ watcher เขียน heartbeat เพราะไม่ต้องแตะ
    src/watcher.py ซึ่งต้องทำงานเหมือนเดิมสำหรับทางเข้าเดิม ถ้าภายหลังอยากได้
    heartbeat ที่ถูกกว่านี้ เปลี่ยนได้โดยไม่กระทบสัญญาของ endpoint
    """
    try:
        return (
            subprocess.run(
                _WORKER_PROBE_COMMAND, capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


class RecorderState:
    """สถานะของการอัดที่ service เป็นเจ้าของ

    มีแค่สามค่าโดยตั้งใจ เพราะ service เป็นเจ้าของแค่การอัด ความคืบหน้าหลังส่ง
    เข้า inbox/ เป็นของ watcher ซึ่งมาทาง activity log -- ให้ service เดาแทน
    watcher คือการสร้างสถานะที่โกหกได้
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | recording | stopping
        self.room = None
        self.model = None
        self.started_at = None
        self.stop_event = None
        self.thread = None
        self.warnings = []
        self.last_result = None

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = int(time.monotonic() - self.started_at) if self.started_at else 0
            return {
                "recorder": self.status,
                "room": self.room,
                "model": self.model,
                "elapsed_seconds": elapsed,
                "warnings": list(self.warnings),
                "last_result": self.last_result,
            }


def create_app(config, recorder=run_recording, worker_probe=probe_worker) -> Flask:
    app = Flask(__name__, static_folder=None)
    state = RecorderState()
    activity.trim(config.base_dir)
    probe_cache = {"value": False, "at": 0.0}

    def worker_ready() -> bool:
        # การเช็คเป็นการ spawn powershell หนึ่งตัว หน้าเว็บ poll ทุกวินาที
        # จึง cache ไว้ ไม่งั้นจะ spawn 60 ตัวต่อนาทีเพื่อตอบคำถามเดิม
        now = time.monotonic()
        if now - probe_cache["at"] > WORKER_PROBE_CACHE_SECONDS:
            probe_cache["value"] = worker_probe()
            probe_cache["at"] = now
        return probe_cache["value"]

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/state")
    def get_state():
        body = state.snapshot()
        body["worker_ready"] = worker_ready()
        body["activity"] = activity.tail(config.base_dir, ACTIVITY_LIMIT)
        body["lang"] = config.ui_lang
        return jsonify(body)

    @app.post("/api/session")
    def open_room():
        payload = request.get_json(silent=True) or {}
        with state.lock:
            if state.status != "idle":
                return jsonify({"error": "already_recording"}), 409
            state.status = "recording"
            state.room = (payload.get("name") or "").strip() or None
            state.model = payload.get("model")
            state.started_at = time.monotonic()
            state.stop_event = threading.Event()
            state.warnings = []
            state.last_result = None
            stop_event = state.stop_event
            room, model = state.room, state.model

        def on_event(code, params=None, level="info"):
            if level in ("warn", "error"):
                with state.lock:
                    state.warnings.append({"code": code, "params": params or {}})
            activity.append(config.base_dir, room or "unnamed", code, level, params)

        def work():
            # ตัวอัดที่ระเบิดต้องไม่ทิ้งหน้าจอค้างที่ "กำลังอัด" ตลอดไป -- สถานะ
            # ต้องกลับไป idle ไม่ว่าจะจบทางไหน
            try:
                result = recorder(room, model, config, stop_event, on_event)
            except Exception:
                logger.exception("ตัวอัดล้มระหว่างทำงาน")
                result = None
            with state.lock:
                state.status = "idle"
                state.started_at = None
                state.last_result = str(result) if result else None
                state.stop_event = None

        thread = threading.Thread(target=work, daemon=True)
        with state.lock:
            state.thread = thread
        thread.start()
        return jsonify({"ok": True}), 201

    @app.post("/api/session/stop")
    def stop_room():
        with state.lock:
            if state.status != "recording" or state.stop_event is None:
                return jsonify({"error": "not_recording"}), 409
            state.status = "stopping"
            state.stop_event.set()
        return jsonify({"ok": True}), 202

    @app.get("/<path:filename>")
    def static_file(filename):
        return send_from_directory(WEB_DIR, filename)

    return app


def main() -> None:
    from src.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    app = create_app(config)
    # 127.0.0.1 เท่านั้น -- ขอบเขตความปลอดภัยของ service นี้คือการไม่รับจากนอกเครื่อง
    app.run(host="127.0.0.1", port=config.ui_port, threaded=True)


if __name__ == "__main__":
    main()
