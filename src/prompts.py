"""โหลด prompt จากไฟล์ใน prompts/ แทนการฝังไว้ในโค้ด

เหตุผลที่แยกออกมา: การจูนคำสั่งสรุปเป็นงานที่ทำบ่อยที่สุดหลังระบบเดินได้ ถ้า prompt
ฝังใน .py ทุกครั้งที่อยากขยับถ้อยคำต้องแก้โค้ดแล้วรันเทสต์ใหม่ทั้งชุด

`{profile_rules}` คือกฎที่ใช้เฉพาะประเภทประชุม แทรกจาก prompts/profiles/<profile>.md
เข้า base เดียวกัน -- ไม่ทำ prompt สองชุดเต็ม เพราะกฎที่ใช้ร่วมกันจะ drift ออกจากกัน
แน่นอนเมื่อแก้ชุดหนึ่งแล้วลืมอีกชุด

ไฟล์หาย/อ่านไม่ได้ = ตกไปใช้ prompt ที่ฝังใน FALLBACKS ด้านล่าง ไม่ crash
transcript ที่ถอดด้วย GPU มาแล้วสำคัญกว่ารูปแบบของสรุป
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# prompts/ อยู่ที่ root ของ repo ข้างๆ src/ -- ยึดจากตำแหน่งไฟล์นี้ ไม่ใช่ cwd
# เพราะ watcher ถูกสั่งรันจากที่ไหนก็ได้
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

DEFAULT_PROFILE = "dev"
# ประชุมข้ามฝ่าย -- แยกเป็นชื่อของตัวเองเพราะมีโค้ดนอกไฟล์นี้ต้องเทียบกับมัน
# (pipeline เปิดตาราง ambiguous/teams ตามค่านี้) สตริงลอยๆ กระจายหลายที่คือทางที่
# วันหนึ่งจะมีที่หนึ่งพิมพ์ต่างจากที่อื่นแล้วฟีเจอร์ครึ่งเดียวทำงาน
CROSS_TEAM_PROFILE = "cross"
KNOWN_PROFILES = (DEFAULT_PROFILE, CROSS_TEAM_PROFILE)

# prompt ที่ใช้มาก่อนจะย้ายออกเป็นไฟล์ -- คงไว้เป็นตัวสำรองเมื่อ prompts/ หาย
# ห้ามแก้ที่นี่เพื่อจูนถ้อยคำ ให้ไปแก้ไฟล์ใน prompts/ ตัวนี้มีไว้กันระบบตายเท่านั้น
_SINGLE_FALLBACK = """คุณเป็นผู้ช่วยสรุปการประชุม อ่าน transcript ที่ให้มาแล้วสรุปเป็นภาษาไทยในรูปแบบ Markdown ประกอบด้วย:

## ประเด็นสำคัญ
(สรุปหัวข้อและประเด็นหลักที่พูดคุยกัน เป็น bullet point)

## Action Items
(รายการสิ่งที่ต้องทำ พร้อมระบุผู้รับผิดชอบถ้าอ้างอิงได้จากบทสนทนา ถ้าไม่ระบุชัดเจนให้เขียนว่า "ไม่ระบุผู้รับผิดชอบ")

ถ้าจากบริบทการสนทนาพอเดาชื่อจริงของผู้พูดแต่ละคนได้ (เช่นมีการเอ่ยชื่อกัน) ให้ใช้ชื่อจริงแทน label "ผู้พูด N" ในสรุป ถ้าเดาไม่ได้ให้คงป้าย "ผู้พูด N" ไว้"""

_MAP_FALLBACK = """คุณเป็นผู้ช่วยสรุปการประชุม ข้อความที่ให้มาคือ transcript "เพียงบางช่วง" ของการประชุมที่ยาวกว่านี้ ไม่ใช่ทั้งการประชุม

สรุปเฉพาะเนื้อหาในช่วงนี้เป็นภาษาไทยแบบ Markdown เป็น bullet point เก็บรายละเอียดให้ครบ ทั้งประเด็นที่คุยกัน ข้อสรุป และสิ่งที่ต้องทำพร้อมผู้รับผิดชอบถ้าระบุได้

ห้ามเดาเนื้อหาช่วงอื่นที่ไม่ได้ให้มา และไม่ต้องเขียนคำนำหรือคำลงท้าย ใช้ bullet point เท่านั้น ห้ามใส่ markdown heading (เช่น ## หรือ ###)"""

_REDUCE_FALLBACK = """ข้อความที่ให้มาคือสรุปย่อยของการประชุมเดียวกัน เรียงตามช่วงเวลา

รวมทั้งหมดเป็นสรุปฉบับเดียวเป็นภาษาไทยแบบ Markdown ประกอบด้วย:

## ประเด็นสำคัญ
(รวมประเด็นจากทุกช่วง จัดกลุ่มตามหัวข้อไม่ใช่ตามเวลา ยุบเรื่องที่ซ้ำกันเข้าด้วยกัน)

## Action Items
(รวมสิ่งที่ต้องทำจากทุกช่วง พร้อมผู้รับผิดชอบถ้าอ้างอิงได้ ถ้าไม่ระบุชัดเจนให้เขียนว่า "ไม่ระบุผู้รับผิดชอบ")

ถ้าพอเดาชื่อจริงของผู้พูดได้จากบริบท ให้ใช้ชื่อจริงแทน label "ผู้พูด N" เก็บเนื้อหาสำคัญให้ครบ อย่าตัดทิ้งเพียงเพราะอยากให้สั้น"""

FALLBACKS = {
    "map": _MAP_FALLBACK,
    "reduce": _REDUCE_FALLBACK,
    "single": _SINGLE_FALLBACK,
}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _profile_rules(prompts_dir: Path, profile: str) -> str:
    """กฎเฉพาะประเภทประชุม -- สตริงว่างเมื่อหาไฟล์ไม่เจอ

    profile ที่ไม่รู้จักต้องเตือนแล้วใช้ dev ต่อ ไม่ใช่ล้ม: ถ้าพิมพ์ผิดใน .env
    หรือใน .job.json แล้วระบบตาย ประชุมนั้นจะไม่ได้สรุปเลย ทั้งที่ transcript
    ออกมาครบและเสียเงินถอดเสียงไปแล้ว
    """
    if profile not in KNOWN_PROFILES:
        logger.warning(
            "ไม่รู้จักประเภทประชุม %r (ที่รองรับ: %s) ใช้ %r แทน / "
            "unknown meeting profile %r, falling back to %r",
            profile,
            ", ".join(KNOWN_PROFILES),
            DEFAULT_PROFILE,
            profile,
            DEFAULT_PROFILE,
        )
        profile = DEFAULT_PROFILE
    text = _read(prompts_dir / "profiles" / f"{profile}.md")
    if text is None:
        logger.warning(
            "ไม่พบ prompts/profiles/%s.md -- สรุปต่อโดยไม่มีกฎเฉพาะประเภทประชุม",
            profile,
        )
        return ""
    return text.strip()


def render(
    name: str,
    *,
    profile: str = DEFAULT_PROFILE,
    glossary_text: str = "",
    prompts_dir: Path | None = None,
) -> str:
    """system prompt ที่พร้อมส่งให้โมเดล

    แทนที่ด้วย str.replace ไม่ใช่ str.format โดยเจตนา: prompt เป็นข้อความที่คนเขียน
    และอาจมี `{` `}` ปนอยู่ตามธรรมชาติ (ตัวอย่าง JSON, วงเล็บปีกกาในคำอธิบาย)
    str.format จะ raise KeyError ใส่ทั้งที่ไฟล์ไม่ได้ผิดอะไร
    """
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
    base = _read(prompts_dir / f"{name}.md")
    if base is None:
        logger.warning(
            "ไม่พบ prompts/%s.md ใช้ prompt ที่ฝังในโค้ดแทน (จูนถ้อยคำจะไม่มีผล "
            "จนกว่าไฟล์จะกลับมา) / falling back to the embedded %r prompt",
            name,
            name,
        )
        return FALLBACKS[name]
    return (
        base.replace("{glossary}", glossary_text.strip())
        .replace("{profile_rules}", _profile_rules(prompts_dir, profile))
        .strip()
    )
