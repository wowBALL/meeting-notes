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

from src import activity, enroll, pending, speakers
from src.messages import render
from src.record import run_recording
from src.storage import rename_speaker_in_transcript, safe_meeting_dir

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WORKER_PROBE_CACHE_SECONDS = 10
ACTIVITY_LIMIT = 200

# ทะเบียนเสียงถูกอ่าน-แก้-เขียนทับใน confirm, การตัดคิวของ skip, และ delete โดยไม่มี
# ตัวกัน ทั้งที่ service รันด้วย threaded=True -- สอง request ที่แก้พร้อมกันชนะกันได้
# ทำให้ฝ่ายที่แพ้เสียชื่อที่เพิ่งพิมพ์ไปทั้งที่ได้รับ ok กลับไปแล้ว ล็อกเดียวพอเพราะ
# ผู้ใช้มีคนเดียวและเว็บนี้ bind 127.0.0.1 เท่านั้น -- ไม่ต้องถึงขั้นล็อกระดับไฟล์
_registry_lock = threading.Lock()

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


def _speaker_summary(speaker: dict) -> dict:
    """คนหนึ่งคนในทะเบียน ในรูปสรุปที่ /api/speakers และ /api/enroll ใช้ร่วมกัน (finding C
    ของรีวิวรอบสุดท้าย -- คนละรอบกับ finding 1-6 ที่แก้ไปก่อนหน้านี้)

    ไม่ใช้ _public_speaker ตรง ๆ เพราะ shape ต่างกัน: _public_speaker เก็บทุกคีย์ยกเว้น
    embedding ซึ่งใช้ได้กับผู้พูดที่รอตั้งชื่อ (embedding อยู่ระดับบนสุด) แต่ entry ใน
    ทะเบียนไม่มี embedding ระดับบนสุดเลย -- มันซ้อนอยู่ใน samples[].embedding ถ้าเอา
    _public_speaker มาใช้ตรงนี้ "samples" ทั้งก้อน (พร้อมเวกเตอร์เสียงทุกตัวข้างใน) จะ
    หลุดออกไปทั้งดุ้นแทนที่จะเหลือแค่ sample_count
    """
    return {
        "id": speaker["id"],
        "name": speaker["name"],
        "sample_count": len(speaker.get("samples", [])),
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
                    _speaker_summary(speaker)
                    for speaker in speakers.load_registry(config.base_dir)
                ]
            }
        )

    @app.delete("/api/speakers/<speaker_id>")
    def delete_speaker(speaker_id):
        with _registry_lock:
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
            with _registry_lock:
                dequeued = pending.resolve_pending(config.base_dir, meeting, label)
            if not dequeued:
                return jsonify({"error": "resolve_failed"}), 500
            return jsonify({"ok": True, "renamed": False, "name": None})

        # find_pending คืนสิ่งที่อ่านจากไฟล์คิวตรง ๆ โดยไม่ตรวจอะไรเลย (ดู docstring
        # ของมัน) -- ไฟล์คิวที่ถูกแก้มือหรือมาจากเวอร์ชันเก่ากว่านี้อาจไม่มีคีย์
        # embedding เลย หรือมีแต่เป็นเวกเตอร์ศูนย์ (pyannote pad เข้ามาเมื่อ label เกิน
        # จำนวน centroid) ปล่อยให้หลุดไปถึง save_registry จะได้ KeyError (500 ที่ไม่มี
        # ใครอธิบาย) หรือแย่กว่านั้นคือเก็บเวกเตอร์ศูนย์ลงทะเบียนถาวรซึ่ง "เหมือน" กับ
        # เวกเตอร์ศูนย์อื่นทุกตัว ต้องเช็คให้ผ่านก่อนแตะทะเบียนเลย
        if not speakers.is_usable_embedding(entry.get("embedding")):
            return jsonify({"error": "bad_embedding"}), 400

        name = payload.get("name")
        speaker_id = payload.get("speaker_id")
        # ล็อกคุมช่วง อ่านทะเบียน -> แก้ -> เขียนทับ เท่านั้น: service รันด้วย
        # threaded=True และสอง request ที่แก้พร้อมกันแบบไม่มีล็อกจะชนะกันได้ ทำให้ฝ่าย
        # แพ้เสียชื่อที่เพิ่งพิมพ์ไปทั้งที่ได้ ok กลับไปแล้ว
        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
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
        # ครั้งหน้า -- โฟลเดอร์ที่ผู้ใช้ย้ายไปแล้วจึงต้องไม่ทำให้คำขอนี้ล้มเหลว การรีไรต์
        # transcript ไม่ต้องอยู่ในล็อก: มันแตะไฟล์คนละไฟล์กับทะเบียน/คิว ไม่มีอะไรแข่งกัน
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
        with _registry_lock:
            dequeued = pending.resolve_pending(config.base_dir, meeting, label)
        if not dequeued:
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
        # หมายเหตุที่คนอ่านต้องรู้: วันนี้ safe_meeting_dir ปฏิเสธอะไรไม่ได้เลยที่นี่
        # เพราะ load_pending ข้างบนผ่าน _is_safe_name มาแล้ว ซึ่งบังคับให้ meeting เป็น
        # ชื่อไฟล์เปล่า ๆ (วัดแล้ว: ทุกชื่อที่ผ่าน _is_safe_name ผ่านตัวนี้หมด) เก็บไว้
        # เป็นชั้นที่สองโดยตั้งใจ เพราะสองด่านนี้อยู่คนละไฟล์และถูกแก้แยกกันได้ -- แต่
        # อย่าเข้าใจผิดว่านี่คือด่านหลัก ถ้าจะแก้ _is_safe_name ให้หลวมลง ต้องกลับมา
        # ดูตรงนี้ว่ายังกันได้จริงไหม
        directory = safe_meeting_dir(config.meetings_dir, meeting)
        if directory is None or not (directory / audio_file).is_file():
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(directory, audio_file, conditional=True)

    @app.get("/enroll")
    def enroll_page():
        # หน้าแยกโดยเจตนา ไม่มีลิงก์จากหน้าจอ idle -- หน้าจอเดิมไม่ถูกแตะเลยแม้แต่
        # บรรทัดเดียว ต้องมาก่อน route catch-all ข้างล่างไม่งั้นถูกกลืน
        return send_from_directory(WEB_DIR, "enroll.html")

    @app.get("/api/enroll")
    def list_enroll():
        registry = speakers.load_registry(config.base_dir)
        entries = enroll.list_entries(config.base_dir)
        # เทียบกับคนที่มีอยู่แล้วในทะเบียนสำหรับไฟล์ที่ลงทะเบียนได้ (finding B ของรีวิว
        # รอบสุดท้าย: สเปกต้องการคะแนนความคล้ายแต่ไม่เคยถูกสร้างจริง) ทำที่นี่บนเซิร์ฟเวอร์
        # เท่านั้น -- entries ที่ enroll.list_entries คืนมาตัด embedding ออกไปแล้ว จึงต้อง
        # ไปอ่าน result.json ดิบอีกรอบเพื่อเอาเวกเตอร์มาเทียบ แล้วส่งกลับไปแค่ชื่อกับคะแนน
        # ที่ปัดแล้ว -- เวกเตอร์ตัวจริงต้องไม่ออกจากฟังก์ชันนี้เลย ใช้ match_known/
        # cosine_similarity ของ src/speakers.py ตัวเดิม ไม่เขียนตัวเทียบความเหมือนซ้ำ
        # และใช้เกณฑ์ config.speaker_match_high/low ตัวเดิม ไม่ตั้งเกณฑ์ใหม่
        for entry in entries:
            if entry.get("status") != "ok":
                continue
            result = enroll.read_result(config.base_dir, entry["audio_file"])
            embedding = result.get("embedding") if result else None
            if not speakers.is_usable_embedding(embedding):
                continue
            matches = speakers.match_known(
                {entry["audio_file"]: embedding},
                registry,
                config.speaker_match_high,
                config.speaker_match_low,
            )
            match = matches.get(entry["audio_file"])
            if match is not None:
                entry["match"] = {
                    "name": match.name,
                    "score": round(match.score, 2),
                    "confident": match.confident,
                }
        return jsonify(
            {
                "files": entries,
                # หน้านี้สั่งงานให้ watcher ทำ ถ้า watcher ไม่ได้รัน ไฟล์จะค้างที่
                # "กำลังวิเคราะห์" ตลอดไปโดยไม่มีอะไรบอกผู้ใช้ว่าทำไม
                "worker": worker_ready(),
                # ค่าเดียวกับที่ enroll.analyze ใช้ตัดสินสถานะ too_short ส่งมาให้หน้าเว็บ
                # แสดงในข้อความ แทนที่จะฝังตัวเลขซ้ำไว้เป็นสตริงคงที่อีกชุดหนึ่งในสอง
                # ภาษา (finding 5 ของรีวิวรอบสุดท้าย) -- ค่าคงที่มีแหล่งเดียวคือ
                # src/speakers.py
                "min_speaking_seconds": speakers.MIN_SPEAKING_SECONDS,
                "speakers": [_speaker_summary(speaker) for speaker in registry],
            }
        )

    @app.post("/api/enroll/analyze")
    def analyze_enroll():
        """สั่งวิเคราะห์ -- เขียนแค่ใบสั่งงาน งานจริงทำที่ watcher

        ตอบ 202 ไม่ใช่ 200 เพราะยังไม่มีอะไรเสร็จ หน้าเว็บต้อง poll ต่อ
        """
        payload = request.get_json(silent=True) or {}
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            return jsonify({"error": "bad_request"}), 400
        if not all(enroll.is_safe_filename(name) for name in files):
            return jsonify({"error": "bad_request"}), 400
        queued = [
            name for name in files if enroll.write_request(config.base_dir, name)
        ]
        if not queued:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"ok": True, "queued": queued}), 202

    @app.post("/api/enroll/confirm")
    def confirm_enroll():
        """ยืนยันชื่อแล้วเก็บเสียงเข้าทะเบียน

        ลำดับเดียวกับ /api/speakers/confirm: ตรวจให้ผ่านและบันทึกทะเบียนให้สำเร็จก่อน
        แล้วค่อยย้ายไฟล์ -- ย้ายก่อนแล้วบันทึกพลาดคือผู้ใช้เสียทั้งชื่อที่พิมพ์และไฟล์
        ที่จะลองใหม่
        """
        payload = request.get_json(silent=True) or {}
        audio_file = payload.get("audio_file")
        if not enroll.is_safe_filename(audio_file):
            return jsonify({"error": "bad_request"}), 400

        result = enroll.read_result(config.base_dir, audio_file)
        if result is None:
            return jsonify({"error": "not_found"}), 404
        if result.get("status") != "ok":
            return jsonify({"error": "not_enrollable"}), 400
        # ไฟล์ผลถูกแก้มือได้และมาจากเวอร์ชันเก่ากว่านี้ได้ -- เวกเตอร์ศูนย์ที่หลุดเข้า
        # ทะเบียนจะ "เหมือน" กับเวกเตอร์ศูนย์อื่นทุกตัว ต้องเช็คก่อนแตะทะเบียนเลย
        if not speakers.is_usable_embedding(result.get("embedding")):
            return jsonify({"error": "bad_embedding"}), 400

        name = payload.get("name")
        if not isinstance(name, str) or not speakers.clean_name(name):
            return jsonify({"error": "bad_name"}), 400
        cleaned = speakers.clean_name(name)

        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
            try:
                speakers.save_registry(
                    config.base_dir,
                    speakers.add_sample(
                        registry,
                        cleaned,
                        result["embedding"],
                        source=f"enroll:{audio_file}",
                    ),
                )
            except OSError as e:
                # ทะเบียนยังไม่ถูกแก้ ไฟล์ต้องอยู่ที่เดิมให้ลองใหม่ได้
                logger.warning("บันทึกทะเบียนจากการลงทะเบียนเสียงไม่ได้: %s", e)
                return jsonify({"error": "save_failed"}), 500

        # ทะเบียนถูกเซฟไปแล้ว = เสียงถูกจำแล้วจริง การย้ายไฟล์ไม่สำเร็จจึงต้องไม่กลายเป็น
        # 500 ที่บอกผู้ใช้ว่าล้มเหลว เพราะเขาจะกดใหม่แล้วได้ตัวอย่างซ้ำในทะเบียน
        try:
            enroll.archive(config.base_dir, audio_file)
        except OSError as e:
            logger.warning("ย้าย %s เข้า done/ ไม่ได้ แต่เสียงถูกจำแล้ว: %s", audio_file, e)
            # archive() ปกติเรียก clear() เองหลังย้ายไฟล์สำเร็จ แต่พังก่อนถึงตรงนั้น --
            # ถ้าปล่อย result.json เดิมไว้ การ์ดนี้จะยังโชว์สถานะ "พร้อมบันทึก" ในรอบหน้า
            # กดซ้ำได้ตัวอย่างซ้ำเข้าทะเบียนคนเดิม (finding 2 ของรีวิวรอบสุดท้าย) ล้าง
            # แบบ best-effort พอ -- ล้างไม่สำเร็จก็ไม่ใช่เรื่องที่ทำให้ request นี้ล้มเหลว
            # เพราะทะเบียนถูกบันทึกไปแล้วจริง
            enroll.clear(config.base_dir, audio_file)
            return jsonify(
                {"ok": True, "name": cleaned, "warning": "archive_failed"}
            )
        return jsonify({"ok": True, "name": cleaned})

    @app.delete("/api/enroll/<path:audio_file>")
    def dismiss_enroll(audio_file):
        """เอาไฟล์ออกจากรายการโดยไม่ลงทะเบียน -- ย้ายเข้า done/ ไม่ลบทิ้ง"""
        if not enroll.is_safe_filename(audio_file):
            return jsonify({"error": "bad_request"}), 400
        if enroll.archive(config.base_dir, audio_file) is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"ok": True})

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
