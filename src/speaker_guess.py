"""ให้โมเดลเดาว่า "ผู้พูด N" แต่ละคนคือใคร จากการที่คนในห้องเอ่ยชื่อกัน

ผลจากที่นี่เป็น "คำแนะนำ" เท่านั้น มันไปโผล่ในหน้าเว็บให้คนกดยืนยัน และไม่มีทาง
เข้าทะเบียนเสียงเองโดยไม่มีมนุษย์ตัดสิน -- โมเดลที่เดาชื่อผิดแล้วถูกจำถาวรจะทำให้
สรุปของทุกการประชุมหลังจากนั้นระบุผู้รับผิดชอบผิดคน

ไม่มี retry ที่นี่โดยตั้งใจ: การเดาที่ล้มเหลวแปลว่าไม่มีคำใบ้ให้ผู้ใช้ ซึ่งเขาก็ยังตั้งชื่อ
เองได้อยู่ดี ไม่คุ้มกับการยิงซ้ำ
"""

import json
import logging

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


def _extract_first_json_object(text: str) -> dict | None:
    """สแกนหา "{" ทีละตำแหน่ง คืน dict แรกที่ decode ผ่านตรงนั้น ครอบคลุมทั้ง JSON
    เปล่า ๆ, JSON ใน code fence และ JSON ที่มีคำอธิบายขนาบข้าง

    เดิมใช้ regex `\\{.*\\}` แบบ greedy ซึ่งจับตั้งแต่ "{" ตัวแรกถึง "}" ตัวสุดท้าย --
    ถ้าคำตอบมีวงเล็บปีกกาที่สองแทรกอยู่ (อีก JSON object หนึ่ง หรือ "{" หลุดมาในคำอธิบาย)
    ช่วงที่จับได้จะไม่ใช่ JSON ที่ถูกต้อง ทำให้คำตอบทั้งก้อนถูกทิ้งทั้งที่ตัวแรกอ่านได้ปกติ
    การสแกนทีละตำแหน่งด้วย raw_decode แก้ปัญหานี้เพราะรู้ตำแหน่งที่ object แต่ละอันจบ
    จริง ๆ ไม่ใช่เดาจากตำแหน่ง "}" ตัวสุดท้ายในข้อความ
    """
    decoder = json.JSONDecoder()
    index = 0
    while True:
        brace = text.find("{", index)
        if brace == -1:
            return None
        try:
            candidate, _end = decoder.raw_decode(text, brace)
        except ValueError:
            index = brace + 1
            continue
        if isinstance(candidate, dict):
            return candidate
        index = brace + 1


def _parse_guesses(text: str, labels: list[str]) -> dict[str, dict]:
    parsed = _extract_first_json_object(text or "")
    if parsed is None:
        logger.info("คำตอบของโมเดลไม่มี JSON ที่อ่านได้ ไม่มีคำใบ้ชื่อผู้พูดรอบนี้")
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
