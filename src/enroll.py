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
import tempfile
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

from src.audio_convert import convert_to_wav
from src.diarize import diarize_audio
from src.speakers import MIN_SPEAKING_SECONDS, clean_name
from src.storage import replace_with_retry
from src.voiceprint import clean_intervals, extract_voiceprints, select_intervals

logger = logging.getLogger(__name__)

ENROLL_DIRNAME = "enroll"
DONE_DIRNAME = "done"

# ความคลาดเคลื่อนของ mtime ที่ยอมรับได้ตอนผูกผลกับไฟล์เสียง (ดู read_result) -- การ
# คัดลอกไฟล์ (แทนที่จะย้าย) ขยับ mtime ได้เป็นวินาที ไม่ใช่แค่ปัดเศษ nanosecond และ
# FAT32/exFAT (การ์ด SD, USB drive ที่ผู้ใช้อาจก็อปไฟล์มาจาก) มีความละเอียดเวลาแค่ 2
# วินาที ตั้งไว้ 2.0 วินาทีให้พอดีกับกรณีนั้นโดยไม่หลวมจนรับไฟล์ที่เนื้อหาเปลี่ยนจริง
_MTIME_TOLERANCE_SECONDS = 2.0

# ชุดเดียวกับ watcher.AUDIO_EXTENSIONS โดยตั้งใจ: ไฟล์ที่วางลง inbox/ ได้ ต้องวางลง
# enroll/ ได้ด้วย ไม่งั้นผู้ใช้ต้องจำว่าโฟลเดอร์ไหนรับอะไร
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}

REQUEST_SUFFIX = ".request.json"
RESULT_SUFFIX = ".result.json"
# finding 5: เครื่องหมายชั่วคราวบอกว่าผล "ถูกล้างทิ้งเพราะผูกกับไฟล์เสียงไม่ได้" (ดู
# write_result) ไม่ใช่ sidecar งาน -- ไม่มีใครอ่านมันเพื่อสั่งงานอะไรเลย มีไว้ให้ list_entries
# อ่านครั้งเดียวแล้วลบทิ้งทันที (ดู _consume_changed_marker) เพื่อบอกผู้ใช้ว่าทำไมการ์ดถึง
# เด้งกลับไป "รอวิเคราะห์" เฉย ๆ โดยไม่มีคำอธิบาย ไม่มี state ค้างถาวรเลยแม้แต่บิตเดียว
CHANGED_SUFFIX = ".changed.json"


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

    Minor C / finding 2-3: Path.is_file()/is_dir() ใน python เวอร์ชันนี้กลืนเฉพาะ
    ENOENT/ENOTDIR/EBADF/WSAELOOP แล้วคืน False -- PermissionError (สแกนไวรัส/ตัวซิงก์
    ไฟล์ล็อกไฟล์ไว้ชั่วครู่) หลุดออกไปเป็น exception จริง ไม่ถูกกลืน ถ้าปล่อยให้ลอยขึ้นไปตรง ๆ
    /api/enroll ทั้งหน้าจะ 500 เพราะไฟล์ (หรือทั้งโฟลเดอร์) ตัวเดียวที่ถูกล็อกชั่วคราว -- ทั้ง
    directory.is_dir() และ path.is_file() ต่อไฟล์จึงต้องอยู่ใน try/except OSError ของตัวเอง
    ข้ามไปแค่รอบ poll นี้ดีกว่า รอบถัดไปจะเห็นใหม่เองเมื่อคลายล็อก และต้อง log ไว้ (finding 3)
    ไม่งั้นไฟล์ที่ล็อกค้างถาวรจะไม่โผล่ใน /enroll ตลอดกาลโดยไม่มีร่องรอยอะไรให้ตามหาเลย
    """
    directory = enroll_dir(base_dir)
    try:
        is_dir = directory.is_dir()
    except OSError as e:
        logger.warning(
            "stat โฟลเดอร์ enroll/ ไม่สำเร็จชั่วคราว (%s) -- ถือว่ายังไม่มีไฟล์ให้เห็นรอบ poll นี้",
            e,
        )
        return []
    if not is_dir:
        return []
    audio_files = []
    for path in directory.iterdir():
        try:
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(path)
        except OSError as e:
            logger.warning(
                "stat ไฟล์ %s ใน enroll/ ไม่สำเร็จชั่วคราว (%s) -- ข้ามไฟล์นี้ไปรอบ poll นี้",
                path.name,
                e,
            )
            continue
    return sorted(audio_files)


def _seconds_by_speaker(turns: list[dict]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for turn in turns:
        key = turn["speaker"]
        seconds[key] = seconds.get(key, 0.0) + max(0.0, turn["end"] - turn["start"])
    return seconds


def analyze(audio_path: Path, pipeline: Any, model: str, embedder) -> dict:
    """ไฟล์เสียงหนึ่งไฟล์ -> ผลที่พร้อมเขียนลง <ชื่อ>.result.json

    `model` คือชื่อ checkpoint ที่ `pipeline` (แยกผู้พูด) ถูกโหลดมา -- ถูกบันทึกไปกับผล
    เพื่อให้ตัวอย่างที่ถูกเก็บเข้าทะเบียนทีหลังติดป้ายโมเดลที่สร้างมันจริง ๆ ไม่ใช่โมเดลที่
    ตั้งอยู่ ณ ตอนที่ผู้ใช้กดยืนยัน (ดู speakers.add_sample) ผลที่วิเคราะห์ค้างไว้ข้าม
    การสลับโมเดลจึงยังผูกกับพื้นที่เวกเตอร์ที่ถูกต้องเสมอ

    `embedder` คือตัวคำนวณ voiceprint จริง (ดู voiceprint.PyannoteEmbedder) -- ผลลัพธ์
    เก็บ `embedder.checkpoint` ไว้เป็น `embedding_model` เพราะนั่นคือป้ายที่บอกว่า
    เวกเตอร์นี้อยู่ในพื้นที่ไหนจริง ๆ ไม่ใช่ค่าใน config ตอนที่ผู้ใช้กดยืนยันชื่อทีหลัง
    (เหตุผลเดียวกับ `model` ด้านบน แต่คนละโมเดลกัน: `model` คือโมเดลแยกผู้พูด ส่วน
    `embedder.checkpoint` คือโมเดล speaker verification ที่สร้างเวกเตอร์)

    รับ pipeline เข้ามาไม่โหลดเอง: ผู้เรียกคือ watcher ซึ่งโหลด pyannote ค้างไว้แล้ว
    และการโหลดซ้ำต่อไฟล์กินเวลา 10-20 วินาทีโดยไม่ได้อะไรกลับมา ผลพลอยได้ที่ตั้งใจ
    คือเทสต์ทุกตัวรันได้โดยไม่ต้องมี GPU และไม่ต้องมี HF_TOKEN

    ไม่ raise เลยไม่ว่าเกิดอะไรขึ้น -- ผู้เรียกต้องมีผลไปเขียนไฟล์เสมอ เพราะไฟล์ผลคือ
    สิ่งเดียวที่ทำให้หน้าเว็บเลิกแสดง "กำลังวิเคราะห์" ได้ ความล้มเหลวที่เงียบคือหน้าจอ
    ที่ค้างตลอดกาลโดยไม่มีอะไรบอกผู้ใช้ว่าต้องทำอะไรต่อ

    แปลงเป็น wav ก่อนเสมอ (finding 4 ของรีวิวรอบสุดท้าย) แบบเดียวกับที่
    pipeline.process_file ทำก่อนถอดเทปประชุม -- ไม่งั้น pyannote ได้ container ดิบ
    (.mp3/.m4a/.ogg) ที่ decode ด้วย backend ที่เดาไม่ได้ (soundfile ในไฟล์ requirements
    ถอดรหัส AAC ไม่ได้) แทนที่จะผ่าน ffmpeg ซึ่งเป็นตัวถอดรหัสเดียวที่โปรเจกต์นี้รับประกัน
    และทำให้เวกเตอร์เสียงของการลงทะเบียนกับของการประชุมมาจากการปรับสภาพเสียงแบบเดียวกัน
    """
    suggested_name = suggested_name_from(audio_path.name)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
            convert_to_wav(audio_path, wav_path)
            # hf_token ไม่ถูกใช้เมื่อส่ง pipeline มาแล้ว (ดู diarize.diarize_audio) ส่ง ""
            # ไปเพื่อไม่ให้โมดูลนี้ต้องรู้จัก token เลย
            result = diarize_audio(wav_path, "", pipeline)
    except Exception as e:
        # กว้างโดยตั้งใจ: ffmpeg/pyannote/torch โยนอะไรออกมาก็ได้ และไม่ว่าตัวไหน
        # ผู้ใช้ต้องได้เห็นว่าไฟล์นี้วิเคราะห์ไม่ผ่านพร้อมเหตุผล
        logger.warning("วิเคราะห์เสียง %s ไม่สำเร็จ: %s", audio_path.name, e)
        return {
            "status": "rejected",
            "reason": "analysis_failed",
            "suggested_name": suggested_name,
            "detail": str(e),
        }

    seconds = _seconds_by_speaker(result.turns)
    # ตัวนับ "กี่คน" เคยเป็น "label ที่มี centroid ที่ใช้ได้" (pyannote pad ศูนย์เข้ามา
    # เมื่อจำนวน label มากกว่าจำนวน centroid) ซึ่งไม่มีอีกแล้วหลังย้าย voiceprint ออกจาก
    # pipeline ไปเป็นโมเดล speaker verification แยกต่างหาก (ดู src/voiceprint.py) --
    # ตอนนี้คือ "label ที่มีเสียงสะอาดไม่มีใครพูดทับถึง 1.5 วินาที" คำนวณจาก turns ล้วน
    # ไม่ผ่านโมเดล embedding เลย จึงไม่ปนกับความสำเร็จของการ extract ด้านล่าง
    #
    # คอมเมนต์เดิมที่นี่ห้ามใส่เกณฑ์เวลาไว้ที่ด่านนี้ เพราะกลัวกลืนคนที่สองที่พูดจริงแค่
    # แวบเดียว -- เหตุผลนั้นหมดอายุแล้ว: สิ่งที่ด่านนี้มีไว้กันคือ voiceprint ที่ปนเสียง
    # คนอื่น และการตัดช่วงทับ (clean_intervals) + ทิ้งท่อนสั้นกว่า 1.5 วิ (select_intervals)
    # กันตรงนั้นตรง ๆ อยู่แล้วโดยไม่ต้องพึ่งด่านนี้เลย คนที่สองที่พูดสะอาด 5 วินาทีขึ้นไป
    # ยังมีคีย์ในผลลัพธ์และยังถูกนับเป็นคน ยังถูกปฏิเสธเหมือนเดิมทุกประการ ที่หลุดผ่าน
    # ด่านนี้ไปได้คือคนที่พูดทับคนอื่นตลอดหรือพูดสั้นกว่า 1.5 วินาที ซึ่งเสียงของเขาไม่เคย
    # เข้าไปอยู่ใน voiceprint อยู่แล้วไม่ว่าด่านนี้จะนับเขาเป็นคนหรือไม่
    usable_labels = select_intervals(clean_intervals(result.turns))
    ignored = len(seconds) - len(usable_labels)
    if ignored > 0:
        logger.info(
            "ข้ามป้ายผู้พูดที่ไม่มีเสียงสะอาดยาวพอ %d ป้ายในไฟล์ %s (เหลือ %d ป้ายที่นับเป็นคนจริง)",
            ignored,
            audio_path.name,
            len(usable_labels),
        )

    base = {
        "suggested_name": suggested_name,
        "speaker_count": len(usable_labels),
        # นับเวลาจากทุกป้าย ไม่ใช่เฉพาะที่นับเป็นคน: เศษที่ถูกข้ามไปทับอยู่บนช่วงที่คน
        # จริงพูดอยู่แล้ว เวลานั้นจึงเป็นเสียงของคนนั้นจริง ๆ ตัดออกเท่ากับรายงานต่ำกว่า
        # ความจริง -- และที่สำคัญกว่านั้น ถ้านับเฉพาะป้ายที่ใช้ได้ ไฟล์ที่มีคนพูด 45
        # วินาทีแต่ extract เสียจะรายงานเป็น 0.0 วินาที แล้วตกด้วยเหตุผล "สั้นเกินไป"
        # ซึ่งเป็นคำอธิบายที่ผิดและแก้ตามไม่ได้
        "speaking_seconds": round(sum(seconds.values()), 1),
    }
    if ignored > 0:
        base["ignored_short_labels"] = ignored

    if len(usable_labels) > 1:
        return {**base, "status": "rejected", "reason": "multiple_speakers"}
    # เรียงก่อน unusable_embedding โดยเจตนา: ไฟล์ที่ไม่มีเสียงพูดเลย (0.0 วินาที) ต้อง
    # ได้เหตุผล "สั้นเกินไป" ซึ่งผู้ใช้ทำตามได้ ไม่ใช่ "เวกเตอร์ใช้ไม่ได้" ที่เป็นภาษา
    # ของระบบและไม่บอกว่าต้องทำอะไรต่อ
    if base["speaking_seconds"] < MIN_SPEAKING_SECONDS:
        return {**base, "status": "rejected", "reason": "too_short"}
    # มีเสียงพูดยาวพอ แต่ไม่มีป้ายไหนมีเสียงสะอาดพอจะนับเป็นคนเลย
    if not usable_labels:
        return {**base, "status": "rejected", "reason": "unusable_embedding"}

    # ด่านนับคนใช้ turns ล้วน ไม่รู้จักโมเดล embedding เลย -- การคำนวณเวกเตอร์จริง (ซึ่ง
    # พึ่ง GPU/โมเดลภายนอก) จึงเพิ่งเริ่มตรงนี้ และยังพังได้อิสระจากด่านข้างบน แปลงเป็น
    # wav อีกครั้งเป็นครั้งที่สอง (ครั้งแรกอยู่ในบล็อก diarize ด้านบน) แทนที่จะรวมเป็น
    # ครั้งเดียว: ffmpeg บนไฟล์ enroll สั้น ๆ ราคาไม่กี่ร้อย ms ส่วนการรวมเป็นครั้งเดียว
    # ต้องขยายอายุของ TemporaryDirectory ให้ครอบทั้งฟังก์ชัน ซึ่งจะทำให้ path การ reject
    # ทุกทางด้านบน (multiple_speakers/too_short/unusable_embedding) ถือ temp dir ไว้โดย
    # ไม่จำเป็น -- ถ้าวัดแล้วช้าจริงค่อยรวมทีหลัง
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        try:
            convert_to_wav(audio_path, wav_path)
        except Exception as e:
            # finding 4 ของรีวิวรอบนี้: การแปลงรอบสองนี้คือ convert_to_wav ตัวเดียวกับรอบแรก
            # เป๊ะ ๆ (ถอดรหัส container เดิมด้วย ffmpeg ตัวเดียวกัน) ต่างกันแค่เวลาที่เรียก --
            # สาเหตุที่มันพังจึงเป็นสาเหตุเดียวกับตอนพังรอบแรกทุกประการ (ffmpeg ถอดรหัสไม่ได้/
            # ไฟล์เสีย) เดิมโค้ดตรงนี้ปนอยู่ในบล็อกเดียวกับ extract_voiceprints แล้วรายงาน
            # "unusable_embedding" ซึ่งบอกผู้ใช้ผิด ๆ ว่า "ลองอัดใหม่ให้เสียงชัดกว่านี้" ทั้งที่
            # ปัญหาคือแปลงไฟล์ไม่ได้ เหมือนรอบแรกที่รายงาน "analysis_failed" -- เหตุเดียวกัน
            # ต้องได้เหตุผลเดียวกัน แยก try นี้ออกมาต่างหากจาก extract_voiceprints ด้านล่าง
            logger.warning(
                "แปลงเสียงของ %s เป็น wav ไม่สำเร็จ (รอบที่สอง ก่อนสร้าง voiceprint): %s",
                audio_path.name,
                e,
            )
            return {
                **base,
                "status": "rejected",
                "reason": "analysis_failed",
                "detail": str(e),
            }
        try:
            voiceprints = extract_voiceprints(wav_path, result.turns, embedder)
            # finding 3 ของรีวิวรอบนี้: เดิม embedder.checkpoint ถูกอ่านนอก try ทั้งหมด (ตอน
            # ประกอบผล "ok" ด้านล่าง) -- embedder ที่ใช้งานได้ปกติทุกอย่างแต่ไม่มี attribute นี้
            # ทำให้ analyze() raise AttributeError หลุดออกไปตรง ๆ ทั้งที่ docstring ของ
            # ฟังก์ชันนี้เองสัญญาไว้ว่าไม่ raise ไม่ว่าเกิดอะไรขึ้น เพราะผู้เรียกคือ watcher ที่
            # ไม่มีมนุษย์นั่งดูอยู่ -- ไฟล์ผลที่ไม่ถูกเขียนเลยคือหน้าเว็บค้างที่ "กำลังวิเคราะห์"
            # ตลอดกาล อ่านมันไว้ในนี้ ในโซนที่มี except คุ้มกันอยู่แล้ว แทนที่จะอ่านทีหลังตอน
            # ไม่มีการ์ดอะไรคุ้มกันเลย
            embedding_model = embedder.checkpoint
        except Exception as e:
            logger.warning("สร้าง voiceprint ของ %s ไม่สำเร็จ: %s", audio_path.name, e)
            return {**base, "status": "rejected", "reason": "unusable_embedding"}

    voiceprint = voiceprints.get(next(iter(usable_labels)))
    if voiceprint is None:
        return {**base, "status": "rejected", "reason": "unusable_embedding"}

    # finding 1 (ตัดสินใจแล้วโดยเจ้าของโปรเจกต์): MIN_SPEAKING_SECONDS เดิมผูกกับเวลาพูด
    # ทั้งหมดได้จริง เพราะเวกเตอร์เคยมาจาก centroid ที่ครอบคลุมเสียงพูดทั้งหมดของคนคนนั้น --
    # ตอนนี้เวกเตอร์มาจากเฉพาะช่วงสะอาดที่ select_intervals คัดแล้วเท่านั้น ซึ่งเป็นเซตย่อย
    # ของเวลาพูดทั้งหมดเสมอ ไฟล์ที่มี speaking_seconds ผ่านด่านข้างบน (>=10) จึงยังมีทางได้
    # เวกเตอร์จากเสียงแค่ไม่กี่วินาที (ทำซ้ำจาก: turns เก้าท่อน 1.0 วิ บวกอีกหนึ่งท่อน 2.0 วิ
    # -- speaking_seconds 11.0 ผ่านด่าน แต่ embedding_seconds เหลือแค่ 2.0) กฎ 10 วินาทีเดิม
    # จึงต้องถูกนำมาใช้กับเสียงที่ *เข้าไปในเวกเตอร์จริง* (voiceprint.seconds) ด้วย ไม่ใช่แค่
    # เวลาพูดทั้งหมดอีกต่อไป -- นี่คือสิ่งที่กฎ 10 วินาทีพยายามพูดมาตลอด ใช้ค่าคงที่ตัวเดิม
    # ไม่เพิ่มตัวที่สอง สำหรับคลิปพูดคนเดียว เสียงสะอาดกับเวลาพูดทั้งหมดใกล้เคียงกันจนแทบไม่
    # กระทบ ส่วนคลิปที่ส่วนใหญ่พูดทับกัน ด่านนี้จะปฏิเสธถูกต้อง
    if voiceprint.seconds < MIN_SPEAKING_SECONDS:
        return {**base, "status": "rejected", "reason": "too_short"}

    return {
        **base,
        "status": "ok",
        "model": model,
        "embedding_model": embedding_model,
        "embedding": list(voiceprint.embedding),
        "embedding_seconds": voiceprint.seconds,
        "segment_count": voiceprint.segment_count,
    }


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


def _changed_marker_path(base_dir: Path, audio_file: str) -> Path | None:
    return _sidecar_path(base_dir, audio_file, CHANGED_SUFFIX)


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

    finding 6: /api/enroll/analyze เดิมเช็คแค่ is_safe_filename ก่อนเรียกฟังก์ชันนี้ -- ชื่อ
    ที่ปลอดภัยแต่ไม่ใช่ไฟล์เสียง (เช่น notes.txt ที่มีอยู่จริงใน enroll/) ผ่านเข้ามาเขียน
    notes.txt.request.json ได้ สิ่งนี้ scan_audio ไม่มีวันเห็น (ไม่ใช่ AUDIO_EXTENSIONS) และ
    orphan sweep ก็ไม่ลบให้ (ไฟล์ต้นทางยังอยู่จริง) -- กลายเป็นขยะค้างตลอดกาล เช็คนามสกุลที่
    ชั้นเดียวกับที่ตรวจชื่อไปแล้ว (ก่อนแตะดิสก์เลย) กันไม่ให้ sidecar กินไม่ได้แบบนี้เกิดขึ้น
    ตั้งแต่ต้น ไม่ต้องพึ่งการเช็คใน scan_audio/analyze เพียงอย่างเดียว
    """
    path = request_path(base_dir, audio_file)
    if path is None:
        return None
    if Path(audio_file).suffix.lower() not in AUDIO_EXTENSIONS:
        return None
    audio_path = enroll_dir(base_dir) / audio_file
    try:
        exists = audio_path.is_file()
    except OSError as e:
        # finding 2: PermissionError ชั่วคราวหลุดออกจาก is_file() ได้ (Minor C) -- ไม่รู้
        # แน่ชัดว่าไฟล์มีอยู่จริงไหม อย่าฟันธงว่า "ไม่มี" ผิด ๆ แค่ไม่สั่งงานรอบนี้ ผู้ใช้
        # กดวิเคราะห์ใหม่ได้เองเมื่อล็อกคลาย
        logger.warning(
            "stat ไฟล์ %s ก่อนสั่งวิเคราะห์ไม่สำเร็จชั่วคราว (%s) -- ไม่สั่งงานรอบนี้",
            audio_file,
            e,
        )
        return None
    if not exists:
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

    Minor C: จุดนี้เคยเป็นที่เดียวในไฟล์ที่ยังเรียก .is_dir()/.is_file() ดิบ ๆ อยู่ (ทำให้
    คำกล่าวอ้างของ _confirmed_missing ที่ว่า "idiom เดียวกันทุกจุดในไฟล์นี้" ไม่จริง) --
    PermissionError ชั่วคราวจาก sidecar ตัวเดียวหลุดออกไปเป็น exception จริง แล้ว
    watcher.process_enroll_requests ครอบการเรียกฟังก์ชันนี้ทั้งก้อนด้วย try/except Exception
    เดียว ผลคือไฟล์ที่ค้างอยู่ *ทุกไฟล์* ในรอบ poll นั้นถูกข้ามไปหมด ไม่ใช่แค่ไฟล์ที่ถูกล็อก
    ต้องใช้ _confirmed_missing แบบเดียวกับ list_entries/read_result/_sweep_orphan_sidecars

    sidecar ผลที่อ่านไม่ชัดเจน (ล็อกชั่วคราว) ต้องถือว่า "มีผลแล้ว" เสมอ ไม่ใช่ "ยังไม่มีผล" --
    ถ้าตีความผิดเป็นฝั่งหลัง ไฟล์ที่วิเคราะห์เสร็จสมบูรณ์แล้วจริงจะถูกส่งวิเคราะห์ซ้ำ (ถอดเสียง
    เต็มรอบบน GPU) ทั้งที่ผลที่ถูกต้องนอนอยู่บนดิสก์แล้ว แค่ sidecar ถูกล็อกชั่วครู่เท่านั้น
    """
    directory = enroll_dir(base_dir)
    try:
        is_dir = directory.is_dir()
    except OSError as e:
        logger.warning(
            "stat โฟลเดอร์ enroll/ ไม่สำเร็จชั่วคราว (%s) -- ถือว่ายังไม่มีใบสั่งงานค้างให้เห็น"
            "รอบ poll นี้",
            e,
        )
        return []
    if not is_dir:
        return []
    audio_files = {path.name for path in scan_audio(base_dir)}
    waiting = []
    for audio_file in sorted(audio_files):
        request = request_path(base_dir, audio_file)
        result = result_path(base_dir, audio_file)
        if request is None or result is None:
            continue
        request_queued = not _confirmed_missing(request)
        # ambiguous (ล็อกชั่วคราว) ต้องนับเป็น "มีผลแล้ว" เสมอ (finding 5 ของรีวิวรอบนี้) --
        # กันไม่ให้ไฟล์ที่วิเคราะห์เสร็จแล้วจริงถูกส่งวิเคราะห์ซ้ำเพียงเพราะอ่าน sidecar ผล
        # ไม่ทันจังหวะ poll นี้
        has_result = not _confirmed_missing(result)
        if request_queued and not has_result:
            waiting.append(audio_file)
    return waiting


def write_result(
    base_dir: Path,
    audio_file: str,
    analyzed: dict,
    pre_analysis_stat: tuple[int, float] | None = None,
    now: datetime | None = None,
) -> Path | None:
    """ผลวิเคราะห์ พร้อมผูกติดกับขนาด/mtime ของไฟล์เสียง ณ ตอนเขียน (CRITICAL A/B, finding 1/4)

    เดิมฟังก์ชันนี้ stat() ไฟล์แค่ตอนเขียนผล (T3 นาทีให้หลัง) แล้วเชื่อว่า "ตรงกับไฟล์บน
    ดิสก์ตอนนี้" คือพอแล้ว -- แต่ analyze() อ่านไบต์ไปตั้งแต่ T1 diarization ใช้เวลานาน
    ระหว่างนั้นผู้ใช้แทนที่ enroll/<ชื่อ>.ogg ด้วยการอัดคนละคนได้ พอถึง T3 ไฟล์ใหม่ "ตรง"
    กับตัวมันเอง 100% เสมอ -- เช็คแบบเดิมจึง "ผ่าน" ทุกครั้งไม่ว่าไฟล์จะถูกแทนที่หรือไม่
    ผู้เรียก (watcher.process_enroll_requests) จึงต้อง stat() ไฟล์ *ก่อน* ส่งเข้า analyze()
    แล้วส่ง (size, mtime) นั้นมาเป็น pre_analysis_stat -- ถ้าไม่ตรงกับ stat ตอนเขียนผล
    แปลว่าไฟล์ถูกแทนที่ระหว่างวิเคราะห์ ผล "ok" ที่ได้จึงบรรยายไบต์ที่ไม่มีอยู่บนดิสก์แล้ว
    ต้องลดสถานะแทน ไม่ปล่อยให้ embedding ของคนเดิมไปแอบอยู่ใต้ชื่อไฟล์ใหม่

    finding 1 (defense in depth): pre_analysis_stat=None ต้องถูกปฏิบัติเหมือน "ผูกไม่ได้"
    ด้วยเช่นกัน ไม่ใช่ "ข้ามการเช็คคู่นี้ไปเงียบ ๆ" อย่างเดิม -- คอมเมนต์เก่าที่นี่เคยให้เหตุผล
    ว่าต้อง "คงพฤติกรรมเดิมไว้ให้ผู้เรียกอื่นที่ไม่ได้อยู่ในเส้นทาง watcher" แต่นั่นคือช่องโหว่
    เอง: ไม่มี production caller ตัวไหนส่ง payload status "ok" เข้ามาโดยไม่ผ่าน
    watcher.process_enroll_requests (ซึ่งตอนนี้ stat ไม่สำเร็จก็ continue ไม่วิเคราะห์เลย
    ดู watcher.py) -- pre_analysis_stat ที่เป็น None พร้อม status "ok" จึงแปลว่าไม่มีทางรู้
    เลยว่าผลนี้เป็นของไฟล์เสียงไบต์ชุดไหน ต้องถือว่ายืนยันไม่ได้เหมือนกับกรณี stat ไม่สำเร็จ
    ตอนเขียนผล ไม่ใช่ "ผ่านเพราะไม่มีอะไรให้เทียบ"

    finding 4: ทุกกรณีที่ผูกไม่ได้ (stat ไม่สำเร็จตอนเขียนผล / pre_analysis_stat เป็น None /
    ไฟล์ถูกแทนที่ระหว่างวิเคราะห์) ตอนนี้ "ล้าง" ทั้งสอง sidecar ทิ้งแทนการเขียน rejected
    ค้างไว้เหมือนเดิม -- ถ้าเขียน rejected ค้างไว้ (โดยเฉพาะกรณีไฟล์ถูกแทนที่: ไฟล์ใหม่ยังอยู่
    บนดิสก์จริง) audio_size/audio_mtime ที่บันทึกจะ "ตรง" กับไฟล์ใหม่ที่อยู่ ณ ตอนนี้พอดี
    ทำให้ rejected นั้น "ยืนยันได้" ตลอดกาลสำหรับไฟล์ใหม่ การ์ดค้างที่ "ใช้ไม่ได้" (state
    "done") ไปตลอด เพราะปุ่มวิเคราะห์เก็บเฉพาะไฟล์ state "idle" -- ไฟล์ที่วางใหม่ถูกต้องทุก
    ไบต์กลายเป็นทางตันที่ทำได้แค่เอาออกจากรายการ (ดู done/) ล้างทิ้งทั้งคู่ (request+result)
    ให้การ์ดกลับไป idle ทันทีดีกว่า: ผู้ใช้กดวิเคราะห์ไฟล์ที่วางใหม่ได้เองเป็นการกระทำใหม่
    ของมนุษย์ ไม่มีอะไรสั่งวิเคราะห์ซ้ำอัตโนมัติ (ไม่มี request.json ค้างให้ pending_requests
    เห็น) จึงไม่มีทางวนซ้ำไม่รู้จบ

    finding 5 (รอบก่อนหน้า): เดิมกรณี "ผูกไม่ได้" เขียนข้อความ detail อธิบายเหตุผลไว้ในไฟล์
    (เช่น "ไฟล์เสียงถูกแทนที่ระหว่างการวิเคราะห์") แต่ read_result ล้าง sidecar ที่ audio_
    size/audio_mtime เป็น null ทิ้งแบบไม่มีเงื่อนไขก่อนคืนค่าเสมอ -- ข้อความนั้นจึงไม่มีทาง
    ถูกอ่านโดยใครเลยแม้แต่ครั้งเดียว (unreachable message) ตอนนี้เขียนคำอธิบายแยกไว้เป็น
    เครื่องหมายชั่วคราวต่างหาก (_mark_changed_during_analysis) แทน ซึ่ง list_entries เป็น
    ผู้อ่านโดยตรง ไม่ผ่าน read_result เลย จึงไม่ชนปัญหาเดิมอีก (ดู finding 5 ของรีวิวรอบนี้
    ด้านล่าง ตรงจุดที่เรียก clear() แล้ว mark)

    finding 2 ของรีวิวรอบนี้: การ์ดกันไม่ให้ embedding หลุดออกไปโดยไม่ผ่านการยืนยัน binding
    เดิมเช็คจาก analyzed.get("status") == "ok" -- แต่สิ่งที่ต้องกันจริง ๆ คือ "embedding"
    ต่างหาก ไม่ใช่สตริง "ok" ซึ่งเป็นแค่ตัวแทน (proxy) ของมัน วันนี้ analyze() รับประกันว่า
    เฉพาะ branch status "ok" เท่านั้นที่ใส่ embedding มาด้วย จึงยังไม่มีทางเรียกจริงที่ทำให้
    ต่างกัน (unreachable) แต่ผู้เรียกในอนาคต (หรือ analyzed ที่ถูกแก้ไขระหว่างทาง) ที่ส่ง
    embedding มาพร้อม status อื่นจะหลุดผ่านการเช็คแบบเดิมไปได้เงียบ ๆ -- เป็นรูปทรงเดียวกับ
    บั๊ก fail-open ที่ใช้เวลาแก้สามรอบกว่าจะปิดสนิท (ดู docstring ของ CRITICAL A/B ด้านบน)
    เช็คทั้งสองเงื่อนไขไว้ด้วยกันกันไม่ให้กลับไปเป็นแบบเดิมโดยไม่ได้ตั้งใจ

    finding 3 ของรีวิวรอบนี้: payload เดิม spread **analyzed หลัง audio_size/audio_mtime --
    ถ้า analyzed มีคีย์ audio_file/analyzed/audio_size/audio_mtime ปนมาด้วย (ไม่เกิดขึ้นจริง
    วันนี้เพราะ analyze() ไม่เคยใส่คีย์เหล่านี้) มันจะทับค่าที่คำนวณไว้แบบเงียบ ๆ กลายเป็นว่า
    ผลที่เขียนลงดิสก์ผูกกับ binding ปลอมที่ analyzed ใส่มาเอง ไม่ใช่ binding จริงที่เพิ่ง
    ยืนยันผ่านด้านบน -- ต้อง spread **analyzed ก่อน แล้วให้คีย์ bookkeeping ตามหลัง ค่าที่
    คำนวณเองจึงชนะเสมอไม่ว่า analyzed จะมีอะไรปนมา
    """
    path = result_path(base_dir, audio_file)
    if path is None:
        return None
    try:
        stat = (enroll_dir(base_dir) / audio_file).stat()
        audio_size: int | None = stat.st_size
        audio_mtime: float | None = stat.st_mtime
    except OSError:
        audio_size = None
        audio_mtime = None

    # finding 2: เช็คทั้ง status "ok" (เดิม) และการมี "embedding" ตรง ๆ -- คีย์ตายตัวคือ
    # embedding ไม่ใช่สตริง status ที่เป็นแค่ตัวแทน กันไว้สองชั้นไม่ให้ใครทำให้ embedding
    # หลุดผ่านการยืนยัน binding ไปได้โดยไม่ตั้งใจในอนาคต
    if analyzed.get("status") == "ok" or "embedding" in analyzed:
        cannot_bind_reason = None
        if audio_size is None or audio_mtime is None:
            # CRITICAL B: stat ไม่สำเร็จตอนเขียนผล -- ไม่มีทางผูกกับไฟล์ไหนได้เลย
            cannot_bind_reason = "stat ไฟล์เสียงไม่สำเร็จตอนเขียนผล (ไฟล์เสียงหายไปแล้ว)"
        elif pre_analysis_stat is None:
            # finding 1 defense in depth: ไม่มี stat ก่อนวิเคราะห์ให้เทียบเลย
            cannot_bind_reason = "ไม่มี pre_analysis_stat ให้เทียบ ยืนยันไม่ได้ว่าผลนี้เป็นของไฟล์ไหน"
        elif pre_analysis_stat != (audio_size, audio_mtime):
            # CRITICAL A: ไฟล์เสียงถูกแทนที่ระหว่างที่กำลังวิเคราะห์อยู่พอดี
            cannot_bind_reason = "ไฟล์เสียงถูกแทนที่ระหว่างการวิเคราะห์"

        if cannot_bind_reason is not None:
            logger.warning(
                "ผลวิเคราะห์ของ %s ผูกกับไฟล์เสียงไม่ได้ (%s) -- ล้าง sidecar ทิ้งทั้งคู่แทน "
                "การเขียน rejected ค้างไว้ (finding 4) ให้การ์ดกลับไป idle",
                audio_file,
                cannot_bind_reason,
            )
            clear(base_dir, audio_file)
            # finding 5 ของรีวิวรอบนี้: clear() ทำให้การ์ดเงียบ ๆ เด้งจาก "กำลังวิเคราะห์"
            # กลับไป "รอวิเคราะห์" โดยไม่มีร่องรอยอะไรเลยว่าทำไม -- ทิ้งเครื่องหมายไว้ให้
            # list_entries อ่านครั้งเดียวแล้วลบทิ้ง (ดู _mark_changed_during_analysis กับ
            # _consume_changed_marker) หน้าเว็บจะได้อธิบายเหตุผลให้ผู้ใช้เห็นสักครั้งหนึ่ง
            _mark_changed_during_analysis(base_dir, audio_file, now)
            return None

    _write_json(
        path,
        {
            **analyzed,
            "audio_file": audio_file,
            "analyzed": (now or datetime.now()).isoformat(timespec="seconds"),
            "audio_size": audio_size,
            "audio_mtime": audio_mtime,
        },
    )
    return path


def _mark_changed_during_analysis(
    base_dir: Path, audio_file: str, now: datetime | None
) -> None:
    """ทิ้งเครื่องหมายไว้บอกว่าผลของไฟล์นี้เพิ่งถูกล้างเพราะผูกกับไบต์ไหนไม่ได้ (finding 5)

    ไม่มี state ค้างถาวร: list_entries เป็นผู้เดียวที่อ่านเครื่องหมายนี้ และลบทิ้งทันทีที่อ่าน
    เจอ (ดู _consume_changed_marker) -- ส่งคำเตือนให้ /api/enroll เห็นครั้งเดียวแล้วหายไปเอง
    การ์ดจึงไม่ค้างสถานะเตือนตลอดกาล และไม่มีการวิเคราะห์ซ้ำเกิดขึ้นจากเครื่องหมายนี้เลยไม่ว่า
    กรณีใด -- มันเป็นแค่ข้อมูลสำหรับแสดงผล ไม่ใช่ใบสั่งงานที่ pending_requests มองเห็น

    เขียนแบบ best-effort เท่านั้น: เขียนไม่สำเร็จก็แค่ไม่มีคำเตือนโผล่ในรอบนี้ ไม่ทำให้การ
    เคลียร์ sidecar หลักที่เพิ่งทำไปด้านบน (ซึ่งสำคัญกว่า) เสียหายหรือ raise ย้อนขึ้นไป
    """
    path = _changed_marker_path(base_dir, audio_file)
    if path is None:
        return
    try:
        _write_json(
            path,
            {
                "audio_file": audio_file,
                "changed": (now or datetime.now()).isoformat(timespec="seconds"),
            },
        )
    except OSError as e:
        logger.warning(
            "เขียนเครื่องหมายแจ้งเตือน 'ไฟล์เปลี่ยนระหว่างวิเคราะห์' ของ %s ไม่สำเร็จ (%s) -- "
            "ผู้ใช้จะไม่เห็นคำเตือนรอบนี้ แต่การ์ดยังกลับไป idle ให้วิเคราะห์ใหม่ได้ตามปกติ",
            audio_file,
            e,
        )


def _consume_changed_marker(base_dir: Path, audio_file: str) -> bool:
    """True ถ้ามีเครื่องหมาย "เปลี่ยนระหว่างวิเคราะห์" ค้างอยู่ -- อ่านแล้วลบทิ้งทันที

    "รายงานครั้งเดียวแล้วหาย" โดยเจตนา (finding 5): ไม่เก็บ state ถาวรใด ๆ นอกจากไฟล์นี้เอง
    ซึ่งลบตัวเองทิ้งทันทีที่ list_entries อ่านเจอ -- รอบ poll ถัดไปจะไม่เห็นคำเตือนซ้ำอีก และ
    ไฟล์ยังเป็น state "idle" ตามปกติ วิเคราะห์ใหม่ได้ทันที ไม่ใช่ทางตันแบบ rejected ค้างไว้

    ล็อกชั่วคราวที่ทำให้อ่านไม่ชัดเจนถูกปฏิบัติแบบเดียวกับจุดอื่นในไฟล์นี้: ข้ามไปก่อน ไม่ลบ
    ไม่ฟันธง แค่ไม่แสดงคำเตือนรอบนี้ -- เครื่องหมายที่เขียนไม่สำเร็จ (best-effort) ก็ทำให้
    ฟังก์ชันนี้คืน False เป็นปกติอยู่แล้วเช่นกัน (ไม่มีไฟล์ให้เจอ)
    """
    path = _changed_marker_path(base_dir, audio_file)
    if path is None:
        return False
    try:
        exists = path.is_file()
    except OSError as e:
        logger.warning(
            "stat เครื่องหมาย 'ไฟล์เปลี่ยนระหว่างวิเคราะห์' ของ %s ไม่สำเร็จชั่วคราว (%s) -- "
            "ข้ามการแสดงคำเตือนรอบนี้ ไม่ลบทิ้ง",
            audio_file,
            e,
        )
        return False
    if not exists:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(
            "ลบเครื่องหมาย 'ไฟล์เปลี่ยนระหว่างวิเคราะห์' ของ %s ไม่ได้ (%s)", audio_file, e
        )
    return True


def read_result(base_dir: Path, audio_file: str) -> dict | None:
    """ผลวิเคราะห์ของไฟล์เดียว ไฟล์พัง/หาย/ไม่ตรงกับไฟล์เสียงบนดิสก์ = None ไม่ raise

    ผลที่อ่านไม่ออกต้องไม่ทำให้ไฟล์อื่นในรายการหายตามไปด้วย -- แบบเดียวกับ
    pending._read_pending_file

    เช็คขนาด/mtime กับไฟล์เสียงตัวจริงก่อนคืนผลเสมอ (finding 1): ผลที่ไม่ตรงคือผลของ
    ไฟล์เสียงคนละไฟล์ (ถูกลบแล้ววางไฟล์ใหม่ชื่อเดียวกันทับ) ปล่อยให้หลุดออกไปเท่ากับเอา
    เวกเตอร์เสียงของคนเดิมไปเสนอให้ยืนยันภายใต้ชื่อไฟล์ใหม่ -- ต้องถือว่า "ไม่มีผล" และ
    เก็บกวาด sidecar เก่าทิ้งไปเลย ไม่ปล่อยค้างให้ผูกผิดซ้ำได้อีก

    CRITICAL B: audio_size/audio_mtime เป็น null ต้องถือว่า "ยืนยันไม่ได้" ไม่ใช่ "ผ่านการ
    เช็คเพราะไม่มีอะไรให้เทียบ" ไม่งั้นเท่ากับปิดการป้องกันทั้งหมดไว้พอดีตอนที่ต้องการมัน
    ที่สุด -- ปกติแล้ว write_result เองจะไม่เขียน null แบบนี้ค้างไว้เลยด้วยซ้ำตั้งแต่
    finding 4 (มันล้าง sidecar ทิ้งทั้งคู่แทนตอนผูกไม่ได้ ดู write_result ด้านบน) แต่
    sidecar เก่าจากก่อนแก้ไข หรือไฟล์ที่ถูกแก้มือ ยังมีทางมี audio_size/audio_mtime เป็น
    null ค้างอยู่บนดิสก์ได้อยู่ดี -- ต้องกันตรงนี้ด้วยอีกชั้น (defense in depth) เช่นกัน

    finding 2: path.is_file() ด้านล่างต้องอยู่ใน try/except OSError -- PermissionError
    ชั่วคราวหลุดออกมาได้ (Minor C) ไม่ถูกกลืนเป็น False เหมือน ENOENT
    """
    path = result_path(base_dir, audio_file)
    if path is None:
        return None
    try:
        exists = path.is_file()
    except OSError as e:
        logger.warning(
            "stat sidecar ผล %s ไม่สำเร็จชั่วคราว (%s) -- ถือว่ายืนยันไม่ได้รอบนี้ ไม่ลบทิ้ง",
            path.name,
            e,
        )
        return None
    if not exists:
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("ข้ามผลวิเคราะห์ที่อ่านไม่ได้ (%s): %s", path.name, e)
        return None
    if not isinstance(parsed, dict):
        return None

    try:
        stat = (enroll_dir(base_dir) / audio_file).stat()
    except (FileNotFoundError, NotADirectoryError):
        # ไฟล์เสียงหายไปแล้วจริง ๆ (ลบผ่าน Explorer) -- ผลนี้กำพร้า ไม่มีไฟล์ให้ผูกอีกต่อไป
        clear(base_dir, audio_file)
        return None
    except OSError as e:
        # Minor C: PermissionError ชั่วคราว (โปรแกรมสแกนไวรัส/ตัวซิงก์ไฟล์ล็อกไฟล์ไว้
        # ชั่วครู่ -- ปัญหาที่โปรเจกต์นี้ยอมรับอยู่แล้ว ดู storage.replace_with_retry) ไม่ใช่
        # "ไฟล์หาย" -- ไม่รู้แน่ชัดว่าเกิดอะไรขึ้น เก็บ sidecar ไว้ก่อน ไม่ล้าง แค่คืน None
        # ให้รอบ poll นี้ ผลวิเคราะห์ที่เสร็จสมบูรณ์แล้วต้องไม่ถูกลบทิ้งฟรี ๆ เพราะสาเหตุ
        # ชั่วคราวที่ตัวมันเองไม่ได้ทำอะไรผิด
        logger.warning(
            "stat ไฟล์เสียง %s ไม่สำเร็จชั่วคราว (%s) -- เก็บ sidecar ไว้ก่อน", audio_file, e
        )
        return None

    recorded_size = parsed.get("audio_size")
    recorded_mtime = parsed.get("audio_mtime")
    if recorded_size is None or recorded_mtime is None:
        logger.warning(
            "ผลวิเคราะห์ของ %s ไม่มี binding ให้ยืนยัน (audio_size/audio_mtime เป็น null) "
            "-- ถือว่ายืนยันไม่ได้ ล้างทิ้ง",
            audio_file,
        )
        clear(base_dir, audio_file)
        return None
    size_matches = recorded_size == stat.st_size
    mtime_matches = abs(stat.st_mtime - recorded_mtime) <= _MTIME_TOLERANCE_SECONDS
    if not (size_matches and mtime_matches):
        logger.warning(
            "ผลวิเคราะห์ของ %s ไม่ตรงกับไฟล์เสียงบนดิสก์แล้ว (ไฟล์ถูกแทนที่) ล้างทิ้ง",
            audio_file,
        )
        clear(base_dir, audio_file)
        return None
    return parsed


def clear(base_dir: Path, audio_file: str) -> None:
    # เครื่องหมาย "เปลี่ยนระหว่างวิเคราะห์" (finding 5) รวมอยู่ในชุดที่ต้องล้างด้วย -- ไม่งั้น
    # เครื่องหมายเก่าจากรอบก่อนหน้าจะค้างอยู่ให้ list_entries เห็นซ้ำผิดรอบ เช่นตอนผู้ใช้กด
    # "เอาออกจากรายการ" ก่อนที่จะทันเห็นคำเตือนนั้นเลยด้วยซ้ำ
    for path in (
        request_path(base_dir, audio_file),
        result_path(base_dir, audio_file),
        _changed_marker_path(base_dir, audio_file),
    ):
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("ลบไฟล์ประกอบของ %s ไม่ได้: %s", audio_file, e)


def _confirmed_missing(path: Path) -> bool:
    """True ต่อเมื่อ path หายไปจริง (FileNotFoundError/NotADirectoryError) เท่านั้น

    Minor C / finding 2: ใช้ idiom เดียวกันทุกจุดในไฟล์นี้ที่ต้องตัดสินใจว่าไฟล์/โฟลเดอร์
    "หายไปแล้วจริง" หรือแค่ "ไม่รู้แน่ชัด" -- Path.is_file()/is_dir()/exists() กลืนเฉพาะ
    ENOENT/ENOTDIR/EBADF/WSAELOOP แล้วคืน False เงียบ ๆ ส่วน PermissionError ชั่วคราว
    (โปรแกรมสแกนไวรัส/ตัวซิงก์ไฟล์ล็อกไฟล์ไว้ชั่วครู่ ดู storage.replace_with_retry) หลุด
    ออกไปเป็น exception จริง ถ้าตีความ False ของ .is_file() ว่า "ไม่มีอยู่" ตรง ๆ จะลบ
    sidecar ของผลวิเคราะห์ที่เสร็จสมบูรณ์แล้วทิ้งไปฟรี ๆ ต้องแยกให้ชัดว่า "หายจริง" (True)
    กับ "มีอยู่จริง หรือไม่รู้แน่ชัด" (False ทั้งคู่) ไม่ใช่อย่างเดียวกัน -- ผู้เรียกต้อง
    ปฏิบัติกับทั้งสองกรณีที่คืน False เหมือนกันคือ "อย่าลบ อย่าฟันธง ข้ามรอบนี้ไปก่อน"

    เดิมชื่อ _audio_confirmed_missing เฉพาะไฟล์เสียง -- เปลี่ยนชื่อเป็นกลาง ๆ เพื่อใช้ซ้ำกับ
    การเช็คไฟล์/โฟลเดอร์อื่นในโมดูลนี้ด้วย (finding 2) แทนที่จะเขียน idiom เดิมซ้ำอีกชุด
    """
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _sweep_orphan_sidecars(base_dir: Path) -> None:
    """ลบ .request.json/.result.json/.changed.json ที่ไฟล์เสียงต้นทางหายไปแล้ว

    list_entries กวาดเฉพาะไฟล์เสียง sidecar กำพร้า (เกิดจากผู้ใช้ลบไฟล์เสียงผ่าน Explorer
    เพราะหน้าเว็บไม่เคยมีปุ่มเอาไฟล์ที่วิเคราะห์แล้วออกโดยไม่ลงทะเบียน -- ดู finding 1)
    จึงมองไม่เห็นในรายการเลย แต่ยังนอนอยู่บนดิสก์พร้อมผูกเข้ากับไฟล์ใหม่ชื่อเดียวกันที่ถูก
    วางเข้ามาทีหลัง ลบทิ้งที่นี่ตัดปัญหาตั้งแต่ต้น ไม่ต้องพึ่งการเช็ค size/mtime ใน
    read_result เพียงอย่างเดียว

    finding 2: ทั้ง directory.is_dir() และ path.is_file() ต่อ sidecar ต้องอยู่ใน
    try/except OSError ของตัวเอง (PermissionError ชั่วคราวหลุดออกไปเป็น exception จริง
    ไม่ถูกกลืน) -- ไม่งั้นไฟล์ที่ถูกล็อกชั่วคราวหนึ่งตัวทำให้ /api/enroll ทั้งหน้า 500
    """
    directory = enroll_dir(base_dir)
    try:
        is_dir = directory.is_dir()
    except OSError as e:
        logger.warning(
            "stat โฟลเดอร์ enroll/ ไม่สำเร็จชั่วคราว (%s) -- ข้ามการกวาด sidecar กำพร้ารอบนี้",
            e,
        )
        return
    if not is_dir:
        return
    for path in directory.iterdir():
        # เช็คชื่อก่อน (ไม่มี I/O) แล้วค่อย is_file() เฉพาะไฟล์ที่ชื่อลงท้ายแบบ sidecar
        # จริง ๆ -- กันไม่ให้ต้อง stat() ไฟล์เสียงทุกไฟล์ในโฟลเดอร์โดยไม่จำเป็น
        for suffix in (REQUEST_SUFFIX, RESULT_SUFFIX, CHANGED_SUFFIX):
            if not path.name.endswith(suffix):
                continue
            try:
                exists = path.is_file()
            except OSError as e:
                logger.warning(
                    "stat sidecar %s ไม่สำเร็จชั่วคราว (%s) -- ข้ามไปรอบนี้ ไม่ลบทิ้ง",
                    path.name,
                    e,
                )
                break
            if not exists:
                break
            audio_file = path.name[: -len(suffix)]
            if _confirmed_missing(directory / audio_file):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("ลบไฟล์ประกอบกำพร้า %s ไม่ได้: %s", path.name, e)
            break


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

    กวาด sidecar กำพร้าทิ้งทุกครั้งที่เรียก (finding 1) เพราะจุดนี้กำลังแจกแจงโฟลเดอร์
    enroll/ อยู่แล้ว และเป็นจุดที่หน้าเว็บ poll ถี่ที่สุด

    finding 2: path.stat() เดิมดัก FileNotFoundError อย่างเดียว -- PermissionError ชั่วคราว
    (Minor C) หลุดออกไปเป็น 500 ทั้งหน้าได้ ต้องข้ามแถวนั้นไปแค่รอบนี้เหมือนกัน ไม่ใช่ raise
    และ request.is_file() ก็ต้องกันแบบเดียวกัน (ไม่ใช่ is_file() ตรง ๆ)

    finding 5: จุดนี้เป็นผู้เดียวที่อ่าน "เครื่องหมายเปลี่ยนระหว่างวิเคราะห์" ที่ write_result
    ทิ้งไว้ (ดู _mark_changed_during_analysis) แล้วลบทิ้งทันที (_consume_changed_marker) --
    เพราะเป็นจุดที่หน้าเว็บ poll ถี่ที่สุดอยู่แล้วเช่นเดียวกับเหตุผลของการกวาด sidecar กำพร้า
    ข้างบน คำเตือนจึงโผล่ให้ผู้ใช้เห็นแค่ครั้งเดียวแล้วหายไปเอง ไม่ค้างสถานะตลอดไป
    """
    _sweep_orphan_sidecars(base_dir)
    entries = []
    for path in scan_audio(base_dir):
        audio_file = path.name
        try:
            # ไฟล์นี้ถูก archive ไปพอดีโดย request อื่นระหว่างที่กำลังแจกแจงรายการอยู่ได้
            # (finding 5 เดิม) -- ข้ามแถวนี้ไปเฉย ๆ ดีกว่าปล่อยให้ FileNotFoundError หลุดออกไป
            # เป็น 500 ทั้งหน้า
            size_bytes = path.stat().st_size
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as e:
            logger.warning(
                "stat ไฟล์ %s ไม่สำเร็จชั่วคราว (%s) -- ข้ามแถวนี้ไปรอบนี้", audio_file, e
            )
            continue
        request = request_path(base_dir, audio_file)
        try:
            request_queued = request is not None and request.is_file()
        except OSError as e:
            logger.warning(
                "stat ใบสั่งงานของ %s ไม่สำเร็จชั่วคราว (%s) -- ถือว่ายังไม่มีใบสั่งงานรอบนี้",
                audio_file,
                e,
            )
            request_queued = False
        result = read_result(base_dir, audio_file)
        if result is not None:
            state = "done"
        elif request_queued:
            state = "queued"
        else:
            state = "idle"
        entry = {
            "audio_file": audio_file,
            "state": state,
            "size_bytes": size_bytes,
            "suggested_name": suggested_name_from(audio_file),
        }
        if _consume_changed_marker(base_dir, audio_file):
            entry["changed_during_analysis"] = True
        if result is not None:
            entry.update(
                {key: value for key, value in result.items() if key != "embedding"}
            )
        entries.append(entry)
    return entries
