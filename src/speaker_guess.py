"""ให้โมเดลเดาว่า "ผู้พูด N" แต่ละคนคือใคร จากการที่คนในห้องเอ่ยชื่อกัน

ผลจากที่นี่เป็น "คำแนะนำ" เท่านั้น มันไปโผล่ในหน้าเว็บให้คนกดยืนยัน และไม่มีทาง
เข้าทะเบียนเสียงเองโดยไม่มีมนุษย์ตัดสิน -- โมเดลที่เดาชื่อผิดแล้วถูกจำถาวรจะทำให้
สรุปของทุกการประชุมหลังจากนั้นระบุผู้รับผิดชอบผิดคน

ไม่มี retry ที่นี่โดยตั้งใจ: การเดาที่ล้มเหลวแปลว่าไม่มีคำใบ้ให้ผู้ใช้ ซึ่งเขาก็ยังตั้งชื่อ
เองได้อยู่ดี ไม่คุ้มกับการยิงซ้ำ
"""

import json
import logging
import re

from src.job import NO_SUMMARY_MODEL
from src.llm import resolve

logger = logging.getLogger(__name__)

GUESS_SYSTEM_PROMPT = """คุณกำลังช่วยหาว่า "ผู้พูด N" แต่ละคนใน transcript การประชุมคือใคร

ดูจากบทสนทนาว่ามีการเอ่ยชื่อ เรียกชื่อ หรือแนะนำตัวหรือไม่ ตอบเป็น JSON เท่านั้น ไม่ต้องมีคำอธิบายนอก JSON:

{"ผู้พูด 2": {"name": "ชื่อที่เดาได้", "evidence": "ข้อความในบทสนทนาที่ทำให้เดาแบบนี้"}}

กติกา:
- ใช้เฉพาะชื่อที่ปรากฏในบทสนทนาจริงเท่านั้น ห้ามแต่งชื่อขึ้นเอง
- ป้ายไหนเดาไม่ได้ ให้ตอบ null ตรงป้ายนั้น หรือไม่ต้องใส่ป้ายนั้นเลย
- ถ้าไม่มีป้ายไหนเดาได้เลย ให้ตอบ {}"""

# ครอบคลุมทั้ง JSON เปล่า ๆ, JSON ใน code fence และ JSON ที่มีคำอธิบายขนาบข้าง
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def guess_speaker_names(
    transcript_markdown: str, labels: list[str], model: str
) -> dict[str, dict]:
    """ชื่อที่โมเดลเดาให้กับป้ายแต่ละอัน ป้ายที่เดาไม่ได้จะไม่มีใน dict

    ผู้เรียกต้องห่อ try/except เอง -- ฟังก์ชันนี้ปล่อย exception จากชั้น provider
    ขึ้นไปตามปกติ เพื่อไม่ให้ความล้มเหลวจริงของ endpoint หายเข้ากลีบเมฆ
    """
    if model == NO_SUMMARY_MODEL or not labels:
        return {}
    provider = resolve(model)
    content = (
        "ป้ายผู้พูดที่ยังไม่รู้ว่าเป็นใคร: "
        + ", ".join(labels)
        + "\n\n"
        + transcript_markdown
    )
    completion = provider.complete(
        GUESS_SYSTEM_PROMPT, content, provider.map_max_tokens
    )
    return _parse_guesses(completion.text, labels)


def _parse_guesses(text: str, labels: list[str]) -> dict[str, dict]:
    match = _JSON_BLOCK_RE.search(text or "")
    if match is None:
        logger.info("โมเดลไม่ได้ตอบ JSON กลับมา ไม่มีคำใบ้ชื่อผู้พูดรอบนี้")
        return {}
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        logger.info("คำตอบของโมเดลไม่ใช่ JSON ที่อ่านได้ ไม่มีคำใบ้ชื่อผู้พูดรอบนี้")
        return {}
    if not isinstance(parsed, dict):
        return {}
    guesses: dict[str, dict] = {}
    for label in labels:
        entry = parsed.get(label)
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        evidence = entry.get("evidence")
        guesses[label] = {
            "name": name.strip(),
            "evidence": evidence.strip() if isinstance(evidence, str) else "",
        }
    return guesses
