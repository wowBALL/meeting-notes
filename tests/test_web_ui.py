"""ตรวจว่า catalog ข้อความสองภาษาใน web/app.js ไม่หลุดจากกัน

src/messages.py มีเทสต์แบบนี้อยู่แล้ว (tests/test_messages.py) แต่ UI ฝั่งเว็บมี
catalog ของตัวเองใน web/app.js (อ็อบเจกต์ UI.th / UI.en) ซึ่งไม่ได้ผ่าน src/messages.py
เลย -- คีย์ที่เพิ่มเข้า th อย่างเดียวโดยลืม en จะเรนเดอร์เป็น undefined บนหน้าจอภาษา
อังกฤษเงียบ ๆ โดยไม่มีอะไรจับ
"""

import re
from pathlib import Path

APP_JS_PATH = Path(__file__).resolve().parent.parent / "web" / "app.js"


def _extract_object_block(text: str, key: str) -> str:
    """เนื้อหาระหว่างวงเล็บปีกกาของ `<key>: { ... }` ในระดับบนสุดของอ็อบเจกต์ UI

    นับวงเล็บปีกกาเอาเองแทน regex ตัวเดียวจบ เพราะเนื้อหาข้างในมีอาร์เรย์ซ้อนหลายชั้น
    (เช่น models) การ match แบบ non-greedy เสี่ยงหยุดที่ `}` แรกที่เจอซึ่งอาจอยู่ใน
    อาร์เรย์ ไม่ใช่ตัวปิดจริงของอ็อบเจกต์
    """
    marker = f"{key}: {{"
    start = text.index(marker) + len(marker)
    depth = 1
    i = start
    while depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def _top_level_keys(block: str) -> set[str]:
    # คีย์ระดับบนสุดเท่านั้น -- เยื้อง 4 ช่องตรงกับ property ของ th:/en: ในไฟล์จริง
    # ส่วนเนื้อหาอาร์เรย์ที่ซ้อนอยู่ (เยื้องมากกว่านี้ และไม่ได้ขึ้นต้นด้วยชื่อคีย์) จะ
    # ไม่ถูกจับมาปนด้วย
    return set(re.findall(r"(?m)^ {4}(\w+):", block))


def test_ui_catalogs_have_the_same_keys_in_both_languages():
    text = APP_JS_PATH.read_text(encoding="utf-8")

    th_keys = _top_level_keys(_extract_object_block(text, "th"))
    en_keys = _top_level_keys(_extract_object_block(text, "en"))

    # ยืนยันว่า parser จับอะไรได้จริง ไม่ใช่ผ่านเทสต์เพราะ regex ไม่ match อะไรเลย
    # (เช่นถ้าโครงสร้างไฟล์เปลี่ยนไปจนไม่เข้ารูปแบบที่ parser รู้จัก)
    assert th_keys, "ไม่พบคีย์ใดใน UI.th เลย -- โครงสร้าง web/app.js อาจเปลี่ยนไป"
    assert en_keys, "ไม่พบคีย์ใดใน UI.en เลย -- โครงสร้าง web/app.js อาจเปลี่ยนไป"
    assert "spkTitle" in th_keys
    assert "spkTitle" in en_keys

    assert th_keys == en_keys, (
        f"คีย์ไม่ตรงกันระหว่าง th กับ en: "
        f"th only={th_keys - en_keys} en only={en_keys - th_keys}"
    )
