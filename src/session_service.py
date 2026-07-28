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

from src import activity, pending, speakers
from src.messages import render
from src.record import run_recording
from src.storage import rename_speaker_in_transcript, safe_meeting_dir

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
        self.devices = {}

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
                "devices": dict(self.devices),
            }


def _public_speaker(speaker: dict) -> dict:
    """ผู้พูดหนึ่งคนในรูปที่ส่งออกหน้าเว็บได้

    ตัด embedding ออกเสมอ: หน้าเว็บไม่ได้ใช้ และเวกเตอร์เสียงเป็นข้อมูล biometric
    ที่ไม่ควรมีสำเนาเพิ่มในที่ที่ไม่จำเป็น
    """
    return {key: value for key, value in speaker.items() if key != "embedding"}


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
        # ภาษามาจากหน้าเว็บ แล้ว service เป็นคนแปลงรหัสเป็นคำพูด -- ถ้าให้ JS
        # มี catalog ของตัวเองจะกลายเป็นสองชุดที่ต้องแก้พร้อมกันเสมอ
        lang = request.args.get("lang") or config.ui_lang
        body = state.snapshot()
        body["worker_ready"] = worker_ready()
        body["lang"] = lang
        body["warnings"] = [
            {**w, "text": render(w["code"], w.get("params"), lang)}
            for w in body["warnings"]
        ]
        body["activity"] = [
            {**e, "text": render(e.get("code", ""), e.get("params"), lang)}
            for e in activity.tail(config.base_dir, ACTIVITY_LIMIT)
        ]
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
            state.devices = {}
            stop_event = state.stop_event
            room, model = state.room, state.model

        def on_event(code, params=None, level="info"):
            # ไมค์/ลำโพงที่ถูกดักฟังจริงต้องเห็นได้ตลอดการอัด ไม่ใช่ไปขุดใน log --
            # การอัดจากอุปกรณ์ผิดตัวคือสาเหตุอันดับหนึ่งของเคส "ไม่มีเสียง"
            if code == "devices_selected":
                with state.lock:
                    state.devices = dict(params or {})
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

    @app.get("/api/speakers/pending")
    def list_pending_speakers():
        meetings = [
            {**meeting, "speakers": [_public_speaker(s) for s in meeting["speakers"]]}
            for meeting in pending.load_all_pending(config.base_dir)
        ]
        return jsonify({"meetings": meetings})

    @app.get("/api/speakers")
    def list_speakers():
        return jsonify(
            {
                "speakers": [
                    {
                        "id": speaker["id"],
                        "name": speaker["name"],
                        "sample_count": len(speaker.get("samples", [])),
                    }
                    for speaker in speakers.load_registry(config.base_dir)
                ]
            }
        )

    @app.delete("/api/speakers/<speaker_id>")
    def delete_speaker(speaker_id):
        registry = speakers.load_registry(config.base_dir)
        remaining = speakers.remove_speaker(registry, speaker_id)
        if len(remaining) == len(registry):
            return jsonify({"error": "not_found"}), 404
        speakers.save_registry(config.base_dir, remaining)
        return jsonify({"ok": True})

    @app.post("/api/speakers/confirm")
    def confirm_speaker():
        """ตั้งชื่อผู้พูดหนึ่งคน หรือข้ามไป

        ลำดับสำคัญ: อ่านข้อมูลมาตรวจให้ผ่านและบันทึกทะเบียนให้สำเร็จก่อน ค่อยตัดคนนั้น
        ออกจากคิว -- ตัดออกก่อนแล้วบันทึกพลาดคือการทำให้ผู้ใช้เสียทั้งชื่อที่เพิ่งพิมพ์
        และรายการที่จะกลับมาพิมพ์ใหม่
        """
        payload = request.get_json(silent=True) or {}
        meeting = payload.get("meeting")
        label = payload.get("label")
        if not isinstance(meeting, str) or not isinstance(label, str) or not label:
            return jsonify({"error": "bad_request"}), 400

        entry = pending.find_pending(config.base_dir, meeting, label)
        if entry is None:
            return jsonify({"error": "not_found"}), 404

        if payload.get("skip") is True:
            # ทางนี้ไม่ได้ทำอะไรอย่างอื่นเลย ถ้าตัดคิวไม่สำเร็จก็แปลว่าไม่มีอะไรเกิดขึ้น
            # จริง ๆ การตอบ ok จึงเป็นการโกหก
            if not pending.resolve_pending(config.base_dir, meeting, label):
                return jsonify({"error": "resolve_failed"}), 500
            return jsonify({"ok": True, "renamed": False, "name": None})

        registry = speakers.load_registry(config.base_dir)
        name = payload.get("name")
        speaker_id = payload.get("speaker_id")
        if isinstance(speaker_id, str) and speaker_id:
            existing = next((s for s in registry if s["id"] == speaker_id), None)
            if existing is None:
                return jsonify({"error": "unknown_speaker"}), 404
            name = existing["name"]
        if not isinstance(name, str) or not speakers.clean_name(name):
            return jsonify({"error": "bad_name"}), 400
        cleaned = speakers.clean_name(name)

        speakers.save_registry(
            config.base_dir,
            speakers.add_sample(registry, cleaned, entry["embedding"], source=meeting),
        )

        # การแก้ไฟล์เก่าเป็นของแถม ทะเบียนคือของจริงเพราะมันไปออกดอกที่การประชุม
        # ครั้งหน้า -- โฟลเดอร์ที่ผู้ใช้ย้ายไปแล้วจึงต้องไม่ทำให้คำขอนี้ล้มเหลว
        meeting_dir = safe_meeting_dir(config.meetings_dir, meeting)
        renamed = (
            rename_speaker_in_transcript(meeting_dir, label, cleaned)
            if meeting_dir is not None
            else False
        )
        # ทะเบียนถูกเซฟไปแล้วตรงนี้ = เสียงถูกจำแล้วจริง การตัดคิวไม่สำเร็จจึงต้องไม่
        # กลายเป็น 500 ที่บอกผู้ใช้ว่าล้มเหลว เพราะเขาจะพิมพ์ใหม่ และถ้าพิมพ์ชื่อไม่
        # เหมือนเดิมจะได้คนซ้ำในทะเบียนสำหรับเสียงเดียวกัน ปล่อยให้รายการค้างไว้แล้ว
        # โผล่ในหน้าเว็บรอบหน้าดีกว่า -- ผู้ใช้กดข้ามได้ และเสียงก็จำไปแล้ว
        if not pending.resolve_pending(config.base_dir, meeting, label):
            logger.warning(
                "จำเสียง %s สำเร็จแล้วแต่ตัดออกจากคิวไม่ได้ (%s/%s)", cleaned, meeting, label
            )
        return jsonify({"ok": True, "renamed": renamed, "name": cleaned})

    @app.get("/api/speakers/audio/<path:meeting>")
    def speaker_audio(meeting):
        """ไฟล์เสียงของการประชุมที่ยังมีคนรอตั้งชื่อ ให้กดฟังเพื่อนึกออกว่าใครพูด

        ชื่อไฟล์อ่านจากรายการในคิว ไม่ใช่จากการไล่ดูว่ามีไฟล์อะไรในโฟลเดอร์ -- ผลพลอยได้
        คือ endpoint นี้เสิร์ฟได้เฉพาะการประชุมที่มีงานค้างจริง ไม่ได้กลายเป็นช่องอ่าน
        ไฟล์ทั่ว meetings/
        """
        # อ่านเฉพาะไฟล์ของการประชุมนี้ ไม่ใช่ load_all_pending: ตัวเล่นเสียงยิง Range
        # request หลายครั้งตอน seek และ load_all_pending จะ glob + parse ไฟล์คิว
        # "ทุกไฟล์" (พร้อมเวกเตอร์เสียงข้างใน) ใหม่ทุกครั้งเพื่อหาสตริงเดียว
        record = pending.load_pending(config.base_dir, meeting)
        if record is None:
            return jsonify({"error": "not_found"}), 404
        audio_file = record.get("audio_file")
        if not isinstance(audio_file, str) or audio_file != Path(audio_file).name:
            return jsonify({"error": "not_found"}), 404
        directory = safe_meeting_dir(config.meetings_dir, meeting)
        if directory is None or not (directory / audio_file).is_file():
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(directory, audio_file, conditional=True)

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
