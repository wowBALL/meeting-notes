"""คิว "ผู้พูดที่ยังไม่รู้จัก" ที่ watcher เขียนไว้ให้คนกลับมาตั้งชื่อทีหลัง

แยกจาก speakers.py เพราะอายุของข้อมูลต่างกันคนละแบบ: ทะเบียนคือความรู้ระยะยาวที่
ห้ามหาย ส่วนนี่คือรายการงานค้างของการประชุมหนึ่งครั้งที่ "ต้อง" หายไปเมื่อทำเสร็จ
เอากฎสองอย่างนี้มาไว้ไฟล์เดียวกันเมื่อไหร่ ก็มีวันที่การลบอย่างหลังไปลบอย่างแรก

ไฟล์ต่อการประชุมหนึ่งครั้ง ไม่ใช่ไฟล์รวม: watcher เขียนตอนจบการประชุม ส่วน UI
service ลบทีละคน สองกระบวนการนี้จึงไม่มีวันแตะไฟล์เดียวกันพร้อมกัน
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.speakers import MIN_SPEAKING_SECONDS, REGISTRY_DIRNAME, is_usable_embedding
from src.storage import replace_with_retry

logger = logging.getLogger(__name__)

PENDING_DIRNAME = "pending"

# ประโยคตัวอย่างที่แสดงให้ผู้ใช้ดูเพื่อนึกออกว่าใครพูด -- สามอันพอให้จับบริบทได้
# โดยไม่ท่วมหน้าจอ
SAMPLE_COUNT = 3


def pending_dir(base_dir: Path) -> Path:
    return Path(base_dir) / REGISTRY_DIRNAME / PENDING_DIRNAME


def _is_safe_name(meeting_name: str) -> bool:
    """ชื่อการประชุมที่ใช้เป็นชื่อไฟล์ได้โดยไม่พาออกนอกโฟลเดอร์

    ชื่อนี้เดินทางมาจาก HTTP request ได้ (ดู session_service) การต่อสตริงตรง ๆ กับ
    ".." หรือ path สัมบูรณ์คือช่องอ่าน/เขียนไฟล์นอกโปรเจกต์
    """
    return (
        isinstance(meeting_name, str)
        and bool(meeting_name)
        and meeting_name not in (".", "..")
        and meeting_name == Path(meeting_name).name
    )


def _pending_path(base_dir: Path, meeting_name: str) -> Path | None:
    if not _is_safe_name(meeting_name):
        return None
    return pending_dir(base_dir) / f"{meeting_name}.json"


def _longest_samples(segments: list[dict], count: int) -> list[dict]:
    """ประโยคที่ยาวที่สุด แต่คืนเรียงตามเวลา

    ยาวที่สุดเพราะประโยคยาวช่วยให้นึกออกว่าใครพูดมากกว่า "ครับ" -- แต่เรียงตามเวลา
    เพราะผู้อ่านกำลังไล่ตามบทสนทนา ไม่ได้กำลังเรียงความยาว
    """
    longest = sorted(segments, key=lambda item: len(item["text"]), reverse=True)[:count]
    return sorted(longest, key=lambda item: item["start"])


def build_pending_speakers(
    merged_segments: list[dict],
    labels: dict[str, str],
    embeddings: dict[str, list[float]],
    matches: dict | None = None,
    min_seconds: float = MIN_SPEAKING_SECONDS,
) -> list[dict]:
    """ผู้พูดในการประชุมนี้ที่ควรถามผู้ใช้ว่าเป็นใคร

    ข้ามสามกลุ่ม: คนที่จำได้แล้วอย่างมั่นใจ (ไม่ต้องถามซ้ำ), คนที่ไม่มีเวกเตอร์ที่ใช้ได้
    (ถามไปก็เก็บเข้าทะเบียนไม่ได้), และคนที่พูดสั้นเกินกว่าจะเชื่อเวกเตอร์ได้
    """
    matches = matches or {}
    seconds: dict[str, float] = {}
    lines: dict[str, list[dict]] = {}
    for segment in merged_segments:
        key = segment["speaker"]
        seconds[key] = seconds.get(key, 0.0) + max(0.0, segment["end"] - segment["start"])
        lines.setdefault(key, []).append(
            {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        )

    pending = []
    for key, total in seconds.items():
        match = matches.get(key)
        if match is not None and match.confident:
            continue
        embedding = embeddings.get(key)
        if not is_usable_embedding(embedding):
            continue
        if total < min_seconds:
            continue
        pending.append(
            {
                "label": labels.get(key, key),
                "diarization_id": key,
                "embedding": [float(value) for value in embedding],
                "speaking_seconds": round(total, 1),
                "samples": _longest_samples(lines[key], SAMPLE_COUNT),
                "guess": None,
                "suggested": (
                    {
                        "speaker_id": match.speaker_id,
                        "name": match.name,
                        "score": round(match.score, 3),
                    }
                    if match is not None
                    else None
                ),
            }
        )
    return pending


def write_pending(
    base_dir: Path, meeting_name: str, audio_file: str, speakers: list[dict]
) -> Path | None:
    """คืน path ที่เขียน หรือ None เมื่อไม่มีใครต้องตั้งชื่อ

    ไม่เขียนไฟล์เปล่าเพราะไฟล์เปล่าจะโผล่ในหน้าเว็บเป็นการประชุมที่ "รอทำอะไรบางอย่าง"
    ทั้งที่ไม่มีอะไรให้ทำ
    """
    if not speakers:
        return None
    path = _pending_path(base_dir, meeting_name)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "meeting_dir": meeting_name,
            "audio_file": audio_file,
            "created": datetime.now().isoformat(timespec="seconds"),
            "speakers": speakers,
        },
        ensure_ascii=False,
        indent=2,
    )
    # เขียนผ่านไฟล์ชั่วคราวแล้วค่อยสลับ แบบเดียวกับ save_registry และ
    # rename_speaker_in_transcript: การเขียนทับตรง ๆ (write_text โหมด "w") ตัดไฟล์
    # เดิมทิ้งก่อนเขียน ถ้าล้มกลางทาง (OSError) ไฟล์คิวจะพังแล้ว _read_pending_file
    # จะข้ามมันไป -- ผู้พูดที่ยังไม่ได้ตั้งชื่อคนอื่น ๆ ในการประชุมนี้จะหายจากหน้าเว็บไป
    # พร้อมเวกเตอร์เสียงที่กู้กลับมาไม่ได้แล้ว
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(payload, encoding="utf-8")
        replace_with_retry(temp, path)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    return path


def _read_pending_file(path: Path) -> dict | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("ข้ามรายการรอตั้งชื่อที่อ่านไม่ได้ (%s): %s", path.name, e)
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("speakers"), list):
        return None
    # รายการที่ไม่เหลือใครแล้วถือว่าไม่มีอยู่ ไฟล์แบบนี้เกิดได้เมื่อ resolve_pending
    # ลบไฟล์ไม่สำเร็จแล้วเขียนทับด้วยรายการว่างแทน -- ถ้าไม่ข้ามตรงนี้ หน้าเว็บจะขึ้น
    # การประชุมที่ "รอทำอะไรบางอย่าง" ทั้งที่ไม่มีอะไรให้ทำ
    if not parsed["speakers"]:
        return None
    return parsed


def load_all_pending(base_dir: Path) -> list[dict]:
    directory = pending_dir(base_dir)
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        parsed = _read_pending_file(path)
        if parsed is not None:
            entries.append(parsed)
    return entries


def load_pending(base_dir: Path, meeting_name: str) -> dict | None:
    """ข้อมูลคิวของการประชุมเดียว ไม่แตะหรือ parse ไฟล์ของการประชุมอื่นเลย

    ต่างจาก load_all_pending ตรงที่ผู้เรียกรู้อยู่แล้วว่าอยากได้การประชุมไหน จึงไม่มี
    เหตุผลต้อง glob และ parse ไฟล์คิวทุกไฟล์ (พร้อมเวกเตอร์เสียงของคนอื่นข้างใน) เพื่อ
    หาไฟล์เดียว
    """
    path = _pending_path(base_dir, meeting_name)
    if path is None or not path.is_file():
        return None
    return _read_pending_file(path)


def find_pending(base_dir: Path, meeting_name: str, label: str) -> dict | None:
    """ข้อมูลของผู้พูดหนึ่งคนโดยไม่แตะไฟล์

    แยกจาก resolve_pending เพราะผู้เรียกต้องอ่านเวกเตอร์มาตรวจและบันทึกทะเบียนให้
    สำเร็จ *ก่อน* ที่จะตัดคนนั้นออกจากคิว -- ตัดออกก่อนแล้วบันทึกพลาดคือการทำให้
    ผู้ใช้เสียทั้งชื่อที่พิมพ์และรายการที่จะพิมพ์ใหม่
    """
    path = _pending_path(base_dir, meeting_name)
    if path is None or not path.is_file():
        return None
    parsed = _read_pending_file(path)
    if parsed is None:
        return None
    for speaker in parsed["speakers"]:
        if isinstance(speaker, dict) and speaker.get("label") == label:
            return speaker
    return None


def resolve_pending(base_dir: Path, meeting_name: str, label: str) -> bool:
    """ตัดผู้พูดคนนี้ออกจากคิว ลบไฟล์ทิ้งเมื่อไม่เหลือใคร"""
    path = _pending_path(base_dir, meeting_name)
    if path is None or not path.is_file():
        return False
    parsed = _read_pending_file(path)
    if parsed is None:
        return False
    remaining = [
        speaker
        for speaker in parsed["speakers"]
        if not (isinstance(speaker, dict) and speaker.get("label") == label)
    ]
    if len(remaining) == len(parsed["speakers"]):
        return False
    if not remaining:
        try:
            path.unlink()
        except OSError as e:
            # ลบไฟล์ไม่สำเร็จแล้วตอบ True ทั้งที่ไฟล์เดิมยังมีชื่อคนนี้อยู่ครบ = โกหกผู้เรียก
            # และผู้ใช้จะเห็นคนที่เพิ่งตั้งชื่อไปแล้วโผล่ในคิวอีกรอบ Windows บนเครื่องนี้
            # จับไฟล์ที่เพิ่งเขียนค้างได้จริง (วัดมาแล้ว) จึงต้องมีทางถอย: เขียนทับด้วย
            # รายการว่าง เนื้อหาบนดิสก์จะได้ตรงกับความจริง แล้ว _read_pending_file
            # ข้ามไฟล์ที่ไม่เหลือใครให้เอง
            logger.warning("ลบรายการรอตั้งชื่อที่ทำครบแล้วไม่ได้ (%s): %s", path.name, e)
            parsed["speakers"] = []
            try:
                path.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as write_error:
                logger.warning(
                    "เขียนทับรายการรอตั้งชื่อก็ไม่ได้ (%s): %s", path.name, write_error
                )
                return False
        return True
    parsed["speakers"] = remaining
    # เขียนผ่านไฟล์ชั่วคราวแล้วค่อยสลับ (แบบเดียวกับ write_pending/save_registry)
    # แทนการเขียนทับตรง ๆ -- ล้มกลางทางแบบเก่าจะทำให้ผู้พูดที่ "เหลือ" ของการประชุมนี้
    # หายไปด้วย ทั้งที่คนพวกนั้นยังไม่ถูกตัดออกจากคิวจริง ๆ
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        replace_with_retry(temp, path)
    except OSError as e:
        # กิ่งนี้เป็นกรณีที่พบบ่อยที่สุด (ตั้งชื่อทีละคนจากหลายคน) แต่เดิมไม่มี try เลย
        # ทั้งที่กิ่ง unlink ข้างบนมี -- ผู้เรียกเซฟทะเบียนสำเร็จไปแล้วก่อนถึงตรงนี้
        # exception ที่หลุดออกไปจึงกลายเป็น HTTP 500 ทั้งที่เสียงถูกจำไปเรียบร้อย
        logger.warning("ตัดผู้พูดออกจากรายการรอตั้งชื่อไม่ได้ (%s): %s", path.name, e)
        try:
            temp.unlink()
        except OSError:
            pass
        return False
    return True
