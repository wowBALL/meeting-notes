"""บันทึกเหตุการณ์แบบ append-only ที่ทั้งตัวอัดและ watcher เขียนได้

watcher เป็นคนละ process กับ service และต้องทำงานได้แม้ไม่มี service อยู่เลย
(วันนี้มันทำงานคู่กับ start-meeting.bat และต้องทำงานแบบนั้นต่อไป) ไฟล์จึงเป็น
ช่องทางที่ถูก: ไม่มีใครต้องรู้จักใคร และไม่มีใครล้มเพราะอีกฝั่งไม่อยู่
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ACTIVITY_DIRNAME = "state"
ACTIVITY_FILENAME = "activity.jsonl"


def activity_path(base_dir: Path) -> Path:
    return Path(base_dir) / ACTIVITY_DIRNAME / ACTIVITY_FILENAME


def append(
    base_dir: Path,
    job: str,
    code: str,
    level: str = "info",
    params: dict | None = None,
) -> None:
    """เขียนหนึ่งเหตุการณ์ ไม่ raise ไม่ว่าเกิดอะไร

    ตัวนี้ถูกเรียกจากกลางท่อประมวลผลประชุม การเขียนสถานะที่ล้มเหลวต้องไม่มีทาง
    ทำให้ประชุมที่อัดซ้ำไม่ได้พังตามไปด้วย

    เปิด-เขียนบรรทัดเดียว-ปิด: บน Windows การ append สั้น ๆ แบบนี้ไม่ฉีกกันเอง
    และความถี่อยู่ระดับไม่กี่บรรทัดต่อประชุม จึงไม่ต้องมี lock ให้เป็นหนี้เพิ่ม
    """
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "job": job,
        "code": code,
        "level": level,
        "params": params or {},
    }
    try:
        path = activity_path(base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("เขียน activity ไม่ได้ (%s): %s", code, e)


def tail(base_dir: Path, limit: int = 200) -> list[dict]:
    """เหตุการณ์ล่าสุดไม่เกิน limit บรรทัด ข้ามบรรทัดที่อ่านไม่ออก

    บรรทัดที่พังเกิดได้จาก process ที่ถูกฆ่ากลางการเขียน มันต้องไม่ทำให้
    ประวัติทั้งไฟล์อ่านไม่ได้
    """
    path = activity_path(base_dir)
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in raw[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def trim(base_dir: Path, keep: int = 2000) -> None:
    """ตัดไฟล์ให้เหลือ keep บรรทัดท้าย

    เรียกตอน service start เท่านั้น ห้ามเรียกตอนเขียน: การเขียนทับไฟล์ขณะอีก
    process กำลัง append คือวิธีทำให้ข้อมูลหายที่แน่นอนที่สุด
    """
    path = activity_path(base_dir)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= keep:
            return
        path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
    except OSError:
        return
