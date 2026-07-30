"""ส่งเรื่องค้างจากประชุมครั้งก่อนเข้าไปในสรุปครั้งนี้

ประชุม dev ถี่ 3 ครั้ง/สัปดาห์ เรื่องที่ค้างจากวันจันทร์จะถูกยกมาคุยต่อวันพุธเสมอ
ต้นทุนแค่อ่านไฟล์เดียวเพิ่ม แต่ได้ความต่อเนื่องข้ามประชุมทันที

**ต้อง profile เดียวกันเท่านั้น** เรื่องค้างของประชุมข้ามฝ่ายไม่ควรไปโผล่ในสรุป
dev ล้วน มันเป็นคนละวงคุย

ข้อควรระวังที่ต้องรู้ก่อนเปิดใช้: กลไกนี้ผูกสรุปเข้าด้วยกันเป็นลูกโซ่ ถ้าสรุปครั้งหนึ่ง
เพี้ยน เรื่องค้างที่ผิดจะถูกส่งต่อไปครั้งถัดไปด้วย ปิดได้ที่ CARRYOVER_ENABLED
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ต้องตรงกับหัวข้อใน prompts/reduce.md และ prompts/single.md
OPEN_ITEMS_HEADING = "## ต้องคุยต่อครั้งหน้า"
# ต้องตรงกับที่ storage.save_summary เขียนท้ายไฟล์
PROFILE_FOOTER_PREFIX = "ประเภทประชุม:"

SUMMARY_NAME = "summary.md"


def _profile_of(summary_text: str) -> str | None:
    for line in summary_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(PROFILE_FOOTER_PREFIX):
            value = stripped[len(PROFILE_FOOTER_PREFIX) :].strip()
            return value or None
    return None


def _section_body(summary_text: str, heading: str) -> str:
    """เนื้อในของหัวข้อ -- สตริงว่างเมื่อไม่มีหัวข้อนั้นหรือหัวข้อนั้นว่าง

    ไม่ใช้ regex ที่ยึด `$` โดยเจตนา: summary.md ที่เขียนบน Windows เป็น CRLF
    (write_text แปลง \\n เป็น os.linesep) แม้ read_text จะ normalize กลับให้แล้ว
    การเดินทีละบรรทัดก็ไม่มีทางพลาดเพราะเรื่องขอบบรรทัดอยู่ดี
    """
    lines = summary_text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        # หัวข้อถัดไป หรือเส้นคั่นก่อน footer = จบหัวข้อนี้
        if stripped.startswith("## ") or stripped == "---":
            break
        body.append(line.rstrip())
    return "\n".join(body).strip()


def previous_open_items(
    meetings_dir: Path, profile: str, exclude_dir: Path | None = None
) -> str:
    """เรื่องค้างจากสรุปล่าสุดของ profile เดียวกัน -- สตริงว่างเมื่อไม่มี

    ไม่มีประชุมก่อนหน้า อ่านไม่ได้ ไม่มีหัวข้อนั้น หรือหัวข้อนั้นว่าง = คืนสตริงว่าง
    ทุกกรณี ความต่อเนื่องข้ามประชุมเป็นของแถม ไม่ใช่ของที่ควรทำให้สรุปครั้งนี้ล้ม

    `exclude_dir` ต้องส่งโฟลเดอร์ของประชุมที่กำลังสรุปมาด้วย: path ลองใหม่เข้ามาที่
    โฟลเดอร์เดิมที่อาจมี summary.md จากรอบก่อนอยู่แล้ว ไม่กันไว้ประชุมจะอ่านเรื่องค้าง
    ของตัวเองมาเป็น carryover
    """
    try:
        candidates = [d for d in meetings_dir.iterdir() if d.is_dir()]
    except OSError:
        return ""

    if exclude_dir is not None:
        excluded = exclude_dir.resolve()
        candidates = [d for d in candidates if d.resolve() != excluded]

    # ชื่อโฟลเดอร์คือ YYYY-MM-DD_HH-MM-<ชื่อ> จึงเรียงตามตัวอักษรได้ผลเท่ากับเรียง
    # ตามเวลา และนิ่งกว่า mtime ที่ขยับได้เมื่อมีคนเปิดแก้ไฟล์ในโฟลเดอร์
    for directory in sorted(candidates, key=lambda d: d.name, reverse=True):
        path = directory / SUMMARY_NAME
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            if path.exists():
                logger.warning("อ่าน %s ไม่ได้ ข้ามไป: %s", path, e)
            continue
        if _profile_of(text) != profile:
            continue
        body = _section_body(text, OPEN_ITEMS_HEADING)
        if body:
            logger.info("ยกเรื่องค้างจาก %s มาต่อในสรุปครั้งนี้", directory.name)
            return body
        # เจอประชุมล่าสุดของ profile นี้แล้วแต่ไม่มีเรื่องค้าง = ไม่มีอะไรต้องส่งต่อ
        # ไม่ต้องไล่ย้อนไปครั้งก่อนหน้านั้น เพราะเรื่องค้างเก่ากว่านั้นถือว่าถูกสะสาง
        # หรือถูกยกมาเขียนใหม่แล้วในครั้งล่าสุด
        return ""
    return ""


def format_for_prompt(open_items: str) -> str:
    """บล็อกที่ยัดเข้า {carryover} -- สตริงว่างเมื่อไม่มีเรื่องค้าง

    ใส่ทั้งข้อมูลและคำสั่งไว้ในบล็อกเดียวกันโดยเจตนา: หัวข้อ "คืบหน้าจากครั้งก่อน"
    จะปรากฏในสรุปเฉพาะตอนที่มีเรื่องค้างจริง ไม่ใช่หัวข้อว่างที่โผล่มาทุกครั้ง
    """
    if not open_items.strip():
        return ""
    return (
        "## เรื่องค้างจากประชุมครั้งก่อน (ประเภทเดียวกัน)\n"
        f"{open_items.strip()}\n\n"
        'เพิ่มหัวข้อนี้ในสรุป วางไว้หลัง "## หัวข้อที่คุยกัน":\n\n'
        "```\n"
        "## คืบหน้าจากครั้งก่อน\n"
        "- <เรื่องค้างข้างบน> → <ครั้งนี้ได้ข้อสรุปว่า... / ยังไม่ได้แตะ>\n"
        "```\n\n"
        "ทุกเรื่องในรายการข้างบนต้องมีบรรทัดของตัวเอง เรื่องที่ประชุมครั้งนี้ไม่ได้พูดถึงเลย "
        'ให้เขียนว่า "ยังไม่ได้แตะ" **ห้ามแต่งความคืบหน้าที่ไม่มีใครพูดถึงในประชุมนี้** '
        "และห้ามเอาเรื่องค้างเก่าไปใส่ในหมวด \"ตกลงแล้ว\" เว้นแต่ประชุมครั้งนี้สรุปปิดมันจริง"
    )
