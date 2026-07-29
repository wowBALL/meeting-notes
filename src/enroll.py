"""ลงทะเบียนน้ำเสียงล่วงหน้าจากไฟล์ที่ผู้ใช้วางไว้ในโฟลเดอร์ enroll/

แยกจาก speakers.py ด้วยเหตุผลเดียวกับที่ pending.py แยก: อายุของข้อมูลต่างกันคนละแบบ
ทะเบียนคือความรู้ระยะยาวที่ห้ามหาย ส่วนนี่คืองานค้างที่ "ต้อง" หายไปเมื่อทำเสร็จ

โมดูลนี้ไม่เคยเขียน registry.json เลยแม้แต่บรรทัดเดียวโดยเจตนา -- watcher เป็นผู้เรียก
หลักของมันและ watcher ไม่มีมนุษย์นั่งดูอยู่ การเขียนทะเบียนอยู่ที่ session_service
ซึ่งเรียกได้ก็ต่อเมื่อมีคนกดยืนยันเท่านั้น
"""

import json
import logging
import shutil
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

from src.diarize import diarize_audio
from src.speakers import MIN_SPEAKING_SECONDS, clean_name, is_usable_embedding
from src.storage import replace_with_retry

logger = logging.getLogger(__name__)

ENROLL_DIRNAME = "enroll"
DONE_DIRNAME = "done"

# ชุดเดียวกับ watcher.AUDIO_EXTENSIONS โดยตั้งใจ: ไฟล์ที่วางลง inbox/ ได้ ต้องวางลง
# enroll/ ได้ด้วย ไม่งั้นผู้ใช้ต้องจำว่าโฟลเดอร์ไหนรับอะไร
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}

REQUEST_SUFFIX = ".request.json"
RESULT_SUFFIX = ".result.json"


def enroll_dir(base_dir: Path) -> Path:
    return Path(base_dir) / ENROLL_DIRNAME


def done_dir(base_dir: Path) -> Path:
    return enroll_dir(base_dir) / DONE_DIRNAME


def is_safe_filename(name) -> bool:
    """ชื่อไฟล์ที่ใช้ต่อกับ enroll/ ได้โดยไม่พาออกนอกโฟลเดอร์

    ชื่อนี้เดินทางมาจาก HTTP request ได้ การต่อสตริงตรง ๆ กับ ".." หรือ path สัมบูรณ์
    คือช่องอ่าน/เขียนไฟล์นอกโปรเจกต์ -- กฎเดียวกับ pending._is_safe_name
    """
    return (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and name == Path(name).name
    )


def suggested_name_from(filename: str) -> str:
    """ชื่อไฟล์ -> ชื่อคนที่เขียนลง transcript.md ได้โดยไม่ทำให้ markdown เสียรูป

    ผ่าน speakers.clean_name ตัวเดียวกับที่ทะเบียนใช้ ไม่ใช่กฎชุดที่สอง -- ชื่อที่หน้าเว็บ
    เติมให้ต้องเป็นชื่อเดียวกับที่จะถูกบันทึกจริง ไม่งั้นผู้ใช้เห็นอย่างหนึ่งแต่ได้อีกอย่าง
    """
    return clean_name(Path(filename).stem)


def scan_audio(base_dir: Path) -> list[Path]:
    """ไฟล์เสียงที่รอลงทะเบียน เรียงตามชื่อ

    ไม่ลงไปใน done/ เพราะไฟล์ในนั้นลงทะเบียนไปแล้ว การเห็นมันโผล่ในรายการอีกรอบ
    แปลว่าผู้ใช้จะลงทะเบียนคนเดิมซ้ำโดยไม่ได้ตั้งใจ
    """
    directory = enroll_dir(base_dir)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _seconds_by_speaker(turns: list[dict]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for turn in turns:
        key = turn["speaker"]
        seconds[key] = seconds.get(key, 0.0) + max(0.0, turn["end"] - turn["start"])
    return seconds


def analyze(audio_path: Path, pipeline: Any) -> dict:
    """ไฟล์เสียงหนึ่งไฟล์ -> ผลที่พร้อมเขียนลง <ชื่อ>.result.json

    รับ pipeline เข้ามาไม่โหลดเอง: ผู้เรียกคือ watcher ซึ่งโหลด pyannote ค้างไว้แล้ว
    และการโหลดซ้ำต่อไฟล์กินเวลา 10-20 วินาทีโดยไม่ได้อะไรกลับมา ผลพลอยได้ที่ตั้งใจ
    คือเทสต์ทุกตัวรันได้โดยไม่ต้องมี GPU และไม่ต้องมี HF_TOKEN

    ไม่ raise เลยไม่ว่าเกิดอะไรขึ้น -- ผู้เรียกต้องมีผลไปเขียนไฟล์เสมอ เพราะไฟล์ผลคือ
    สิ่งเดียวที่ทำให้หน้าเว็บเลิกแสดง "กำลังวิเคราะห์" ได้ ความล้มเหลวที่เงียบคือหน้าจอ
    ที่ค้างตลอดกาลโดยไม่มีอะไรบอกผู้ใช้ว่าต้องทำอะไรต่อ
    """
    suggested_name = suggested_name_from(audio_path.name)
    try:
        # hf_token ไม่ถูกใช้เมื่อส่ง pipeline มาแล้ว (ดู diarize.diarize_audio) ส่ง ""
        # ไปเพื่อไม่ให้โมดูลนี้ต้องรู้จัก token เลย
        result = diarize_audio(audio_path, "", pipeline)
    except Exception as e:
        # กว้างโดยตั้งใจ: pyannote/torch/ffmpeg โยนอะไรออกมาก็ได้ และไม่ว่าตัวไหน
        # ผู้ใช้ต้องได้เห็นว่าไฟล์นี้วิเคราะห์ไม่ผ่านพร้อมเหตุผล
        logger.warning("วิเคราะห์เสียง %s ไม่สำเร็จ: %s", audio_path.name, e)
        return {
            "status": "rejected",
            "reason": "analysis_failed",
            "suggested_name": suggested_name,
            "detail": str(e),
        }

    seconds = _seconds_by_speaker(result.turns)
    base = {
        "suggested_name": suggested_name,
        "speaker_count": len(seconds),
        "speaking_seconds": round(sum(seconds.values()), 1),
    }

    if len(seconds) > 1:
        return {**base, "status": "rejected", "reason": "multiple_speakers"}
    # ถึงตรงนี้มีผู้พูดไม่เกินหนึ่งคน ผลรวมจึงเท่ากับเวลาพูดของคนนั้นพอดี ไฟล์ที่ไม่มี
    # เสียงพูดเลยตกที่นี่ด้วย (0.0 วินาที) ซึ่งเป็นคำตอบที่ถูกต้องอยู่แล้ว
    if base["speaking_seconds"] < MIN_SPEAKING_SECONDS:
        return {**base, "status": "rejected", "reason": "too_short"}

    label = next(iter(seconds))
    embedding = result.embeddings.get(label)
    if not is_usable_embedding(embedding):
        return {**base, "status": "rejected", "reason": "unusable_embedding"}

    return {**base, "status": "ok", "embedding": [float(value) for value in embedding]}


def _sidecar_path(base_dir: Path, audio_file: str, suffix: str) -> Path | None:
    """ชื่อไฟล์ประกอบ -- เก็บนามสกุลของไฟล์เสียงไว้เต็ม ๆ ไม่ตัดทิ้งเหมือน suggested_name_from

    ถ้าตัดนามสกุลออก call.wav กับ call.mp3 จะได้ sidecar ชื่อเดียวกัน (call.request.json,
    call.result.json) เขียนไฟล์หนึ่งจะทับอีกไฟล์แบบเงียบ ๆ แล้ว list_entries จะเอาผลของ
    ไฟล์หนึ่งไปแปะให้อีกไฟล์ -- เวกเตอร์เสียงของคนผิดหลุดไปให้กดยืนยันภายใต้ชื่อคนอื่น
    """
    if not is_safe_filename(audio_file):
        return None
    return enroll_dir(base_dir) / (audio_file + suffix)


def request_path(base_dir: Path, audio_file: str) -> Path | None:
    return _sidecar_path(base_dir, audio_file, REQUEST_SUFFIX)


def result_path(base_dir: Path, audio_file: str) -> Path | None:
    return _sidecar_path(base_dir, audio_file, RESULT_SUFFIX)


def _write_json(path: Path, payload: dict) -> None:
    """เขียนผ่านไฟล์ชั่วคราวแล้วค่อยสลับ แบบเดียวกับ save_registry และ write_pending

    การเขียนทับตรง ๆ (write_text โหมด "w") ตัดไฟล์เดิมทิ้งก่อนเขียน ถ้าล้มกลางทางจะได้
    ไฟล์พังที่ผู้อ่านต้องข้าม -- และ Windows บนเครื่องนี้ล็อกไฟล์ที่เพิ่งเขียนได้จริง
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        replace_with_retry(temp, path)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def write_request(base_dir: Path, audio_file: str, now: datetime | None = None) -> Path | None:
    """ใบสั่งงานให้ watcher หยิบไปทำ คืน None เมื่อชื่อไม่ปลอดภัยหรือไฟล์ไม่มีจริง

    ตรวจว่าไฟล์เสียงมีอยู่จริงก่อนเขียน เพราะใบสั่งงานที่ไม่มีไฟล์คู่กันจะทำให้ watcher
    ต้องรับมือกับสภาพที่ป้องกันได้ตั้งแต่ต้นทาง
    """
    path = request_path(base_dir, audio_file)
    if path is None:
        return None
    if not (enroll_dir(base_dir) / audio_file).is_file():
        return None
    _write_json(
        path,
        {
            "audio_file": audio_file,
            "requested": (now or datetime.now()).isoformat(timespec="seconds"),
        },
    )
    return path


def pending_requests(base_dir: Path) -> list[str]:
    """ชื่อไฟล์เสียงที่สั่งแล้วแต่ยังไม่มีผล เรียงตามชื่อ

    "ยังไม่มีผล" คือเงื่อนไขที่ทำให้งานไม่ถูกทำซ้ำ: watcher เขียนผลเป็นสิ่งสุดท้าย
    ใบสั่งงานที่ค้างเพราะเครื่องดับกลางทางจึงถูกหยิบไปทำใหม่รอบหน้าเอง
    """
    directory = enroll_dir(base_dir)
    if not directory.is_dir():
        return []
    audio_files = {path.name for path in scan_audio(base_dir)}
    waiting = []
    for audio_file in sorted(audio_files):
        request = request_path(base_dir, audio_file)
        result = result_path(base_dir, audio_file)
        if request is not None and request.is_file() and not result.is_file():
            waiting.append(audio_file)
    return waiting


def write_result(
    base_dir: Path, audio_file: str, analyzed: dict, now: datetime | None = None
) -> Path | None:
    path = result_path(base_dir, audio_file)
    if path is None:
        return None
    _write_json(
        path,
        {
            "audio_file": audio_file,
            "analyzed": (now or datetime.now()).isoformat(timespec="seconds"),
            **analyzed,
        },
    )
    return path


def read_result(base_dir: Path, audio_file: str) -> dict | None:
    """ผลวิเคราะห์ของไฟล์เดียว ไฟล์พัง/หาย = None ไม่ raise

    ผลที่อ่านไม่ออกต้องไม่ทำให้ไฟล์อื่นในรายการหายตามไปด้วย -- แบบเดียวกับ
    pending._read_pending_file
    """
    path = result_path(base_dir, audio_file)
    if path is None or not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("ข้ามผลวิเคราะห์ที่อ่านไม่ได้ (%s): %s", path.name, e)
        return None
    return parsed if isinstance(parsed, dict) else None


def clear(base_dir: Path, audio_file: str) -> None:
    for path in (request_path(base_dir, audio_file), result_path(base_dir, audio_file)):
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("ลบไฟล์ประกอบของ %s ไม่ได้: %s", audio_file, e)


def archive(base_dir: Path, audio_file: str) -> Path | None:
    """ย้ายไฟล์เสียงเข้า done/ แล้วเก็บกวาดไฟล์ประกอบ คืน path ปลายทาง

    ไม่ลบไฟล์ทิ้งเพราะผู้ใช้อาจอยากลงทะเบียนเสียงเดิมซ้ำ (เช่นตั้งชื่อผิด) และไม่เขียนทับ
    ของเดิมที่ชื่อชนกัน -- ต่อท้ายด้วย -2, -3 แบบเดียวกับ storage.create_meeting_folder
    """
    if not is_safe_filename(audio_file):
        return None
    source = enroll_dir(base_dir) / audio_file
    if not source.is_file():
        return None
    destination_dir = done_dir(base_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_file).stem
    suffix = Path(audio_file).suffix
    for attempt in count(1):
        candidate = destination_dir / (
            audio_file if attempt == 1 else f"{stem}-{attempt}{suffix}"
        )
        if not candidate.exists():
            break
    shutil.move(str(source), str(candidate))
    clear(base_dir, audio_file)
    return candidate


def list_entries(base_dir: Path) -> list[dict]:
    """ทุกไฟล์ที่รอลงทะเบียน พร้อมสถานะ ในรูปที่ส่งออกหน้าเว็บได้

    ตัด embedding ออกเสมอแบบเดียวกับ session_service._public_speaker -- หน้าเว็บไม่ได้ใช้
    และเวกเตอร์เสียงเป็นข้อมูล biometric ที่ไม่ควรมีสำเนาเพิ่มในที่ที่ไม่จำเป็น
    """
    entries = []
    for path in scan_audio(base_dir):
        audio_file = path.name
        request = request_path(base_dir, audio_file)
        result = read_result(base_dir, audio_file)
        if result is not None:
            state = "done"
        elif request is not None and request.is_file():
            state = "queued"
        else:
            state = "idle"
        entry = {
            "audio_file": audio_file,
            "state": state,
            "size_bytes": path.stat().st_size,
            "suggested_name": suggested_name_from(audio_file),
        }
        if result is not None:
            entry.update(
                {key: value for key, value in result.items() if key != "embedding"}
            )
        entries.append(entry)
    return entries
