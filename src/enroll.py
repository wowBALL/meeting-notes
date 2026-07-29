"""ลงทะเบียนน้ำเสียงล่วงหน้าจากไฟล์ที่ผู้ใช้วางไว้ในโฟลเดอร์ enroll/

แยกจาก speakers.py ด้วยเหตุผลเดียวกับที่ pending.py แยก: อายุของข้อมูลต่างกันคนละแบบ
ทะเบียนคือความรู้ระยะยาวที่ห้ามหาย ส่วนนี่คืองานค้างที่ "ต้อง" หายไปเมื่อทำเสร็จ

โมดูลนี้ไม่เคยเขียน registry.json เลยแม้แต่บรรทัดเดียวโดยเจตนา -- watcher เป็นผู้เรียก
หลักของมันและ watcher ไม่มีมนุษย์นั่งดูอยู่ การเขียนทะเบียนอยู่ที่ session_service
ซึ่งเรียกได้ก็ต่อเมื่อมีคนกดยืนยันเท่านั้น
"""

import logging
from pathlib import Path
from typing import Any

from src.diarize import diarize_audio
from src.speakers import MIN_SPEAKING_SECONDS, clean_name, is_usable_embedding

logger = logging.getLogger(__name__)

ENROLL_DIRNAME = "enroll"
DONE_DIRNAME = "done"

# ชุดเดียวกับ watcher.AUDIO_EXTENSIONS โดยตั้งใจ: ไฟล์ที่วางลง inbox/ ได้ ต้องวางลง
# enroll/ ได้ด้วย ไม่งั้นผู้ใช้ต้องจำว่าโฟลเดอร์ไหนรับอะไร
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}


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
