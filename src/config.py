import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ค่าเริ่มต้นเป็น GLM-5.2 บน endpoint ของบริษัท ไม่ใช่ Claude เพราะเหตุผลเรื่องความเป็น
# ส่วนตัวของข้อมูล: ทางที่ transcript ไม่ออกนอกบริษัทต้องเป็นทางที่ไม่ต้องคิด ถ้า Claude
# เป็นค่าเริ่มต้น การกด Enter ผ่านเมนูจะส่งข้อมูลออกไปทุกครั้งโดยไม่มีใครตัดสินใจเรื่องนั้น
#
# preflight อ่านค่านี้ด้วยโดยไม่ผ่าน load_config -- จึงต้องอยู่ตรงนี้ ไม่ใช่ฝังใน default
# ของ dataclass ที่แห่งเดียว
#
# เคยมี DEFAULT_CLAUDE_MODEL เป็นชื่อของค่านี้ ลบทิ้งแล้วไม่เก็บเป็น alias เพราะชื่อที่
# บอกว่า Claude แต่ถือค่า GLM คือชื่อที่โกหก ส่วนชื่อ field claude_model ใน Config และ
# ใน job.json คงไว้ตามเดิมโดยเจตนา -- การรีเนมกระทบ sidecar ที่ค้างอยู่ใน inbox/
DEFAULT_SUMMARY_MODEL = "GLM-5.2"

# ใช้โดย session_service และเป็นภาษาตั้งต้นของข้อความฝั่ง CLI -- หน้าเว็บจำภาษา
# ของตัวเองใน localStorage จึงไม่ได้อ่านค่านี้
DEFAULT_UI_PORT = 8765
DEFAULT_UI_LANG = "th"

# เกณฑ์ความเหมือนของน้ำเสียง (cosine similarity): สูงกว่า HIGH = ใส่ชื่อให้เลย,
# ระหว่าง LOW กับ HIGH = เสนอให้คนยืนยัน, ต่ำกว่า LOW = ถือว่าไม่รู้จัก
#
# ค่าพวกนี้ยังไม่ผ่านการวัดกับข้อมูลจริงของเครื่องนี้ ตั้งไว้ฝั่งอนุรักษ์นิยมโดยตั้งใจ
# เพราะการไม่ใส่ชื่อเสียหายน้อยกว่าการใส่ชื่อผิดคนลงสรุปที่ระบุผู้รับผิดชอบ
DEFAULT_SPEAKER_MATCH_HIGH = 0.70
DEFAULT_SPEAKER_MATCH_LOW = 0.50


@dataclass
class Config:
    base_dir: Path
    inbox_dir: Path
    failed_dir: Path
    meetings_dir: Path
    hf_token: str
    claude_model: str = DEFAULT_SUMMARY_MODEL
    whisper_model: str = "small"
    ui_port: int = DEFAULT_UI_PORT
    ui_lang: str = DEFAULT_UI_LANG
    speaker_match_high: float = DEFAULT_SPEAKER_MATCH_HIGH
    speaker_match_low: float = DEFAULT_SPEAKER_MATCH_LOW


def _read_float(name: str, default: float) -> float:
    """ค่าที่พิมพ์ผิดใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- แบบเดียวกับ UI_PORT"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def load_config(base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    # load-bearing เกินหน้าที่ของ Config เอง: นี่คือ call เดียวที่ทำให้ LLM_API_KEY
    # กับ LLM_BASE_URL (และ ANTHROPIC_API_KEY) เข้าไปอยู่ใน os.environ ให้
    # src/llm.py อ่านตรงๆ ได้ (llm.py ไม่เรียก load_dotenv เอง) ลบ/ย้ายบรรทัดนี้แล้ว
    # provider จะหา key ตัวเองไม่เจอเลย ทั้งที่ไม่มีอะไรใน config.py เรียกใช้ค่าพวกนี้
    # ให้เห็นตรงๆ
    load_dotenv(base_dir / ".env")

    hf_token = os.environ["HF_TOKEN"]
    claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_SUMMARY_MODEL)
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    # พอร์ตที่ตั้งมาผิดต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- ตกกลับไปที่ default
    try:
        ui_port = int(os.environ.get("UI_PORT", DEFAULT_UI_PORT))
    except ValueError:
        ui_port = DEFAULT_UI_PORT
    ui_lang = os.environ.get("UI_LANG", DEFAULT_UI_LANG)
    speaker_match_high = _read_float("SPEAKER_MATCH_HIGH", DEFAULT_SPEAKER_MATCH_HIGH)
    speaker_match_low = _read_float("SPEAKER_MATCH_LOW", DEFAULT_SPEAKER_MATCH_LOW)
    # HIGH ต้องไม่ต่ำกว่า LOW -- ถ้ากลับกัน ทุกคนที่ผ่านเกณฑ์ LOW จะผ่านเกณฑ์ HIGH ไปด้วย
    # และ Match.confident จะกลายเป็น True ของทุกคน ระบบจะเริ่มใส่ชื่อจริงลง transcript
    # ให้อัตโนมัติที่ความมั่นใจต่ำแบบเงียบ ๆ -- ตรงข้ามกับที่ค่าสองเกณฑ์นี้มีไว้กัน
    # ค่าที่พิมพ์กลับกันใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- แบบเดียวกับ UI_PORT
    # และ _read_float จึงตกกลับไปที่ default ทั้งคู่ ไม่ใช่แค่ตัวเดียว เพราะ default
    # คู่นี้ถูกวัดมาคู่กัน ผสมค่าที่ผู้ใช้ตั้งเข้ากับ default ตัวเดียวอาจได้คู่ที่ยังกลับกันอยู่
    if speaker_match_high < speaker_match_low:
        logger.warning(
            "SPEAKER_MATCH_HIGH (%s) ต่ำกว่า SPEAKER_MATCH_LOW (%s) ซึ่งกลับด้านกัน "
            "ใช้ค่าเริ่มต้นทั้งคู่แทน (HIGH=%s, LOW=%s) / "
            "SPEAKER_MATCH_HIGH (%s) is lower than SPEAKER_MATCH_LOW (%s), which is "
            "inverted -- falling back to both defaults (HIGH=%s, LOW=%s)",
            speaker_match_high,
            speaker_match_low,
            DEFAULT_SPEAKER_MATCH_HIGH,
            DEFAULT_SPEAKER_MATCH_LOW,
            speaker_match_high,
            speaker_match_low,
            DEFAULT_SPEAKER_MATCH_HIGH,
            DEFAULT_SPEAKER_MATCH_LOW,
        )
        speaker_match_high = DEFAULT_SPEAKER_MATCH_HIGH
        speaker_match_low = DEFAULT_SPEAKER_MATCH_LOW

    return Config(
        base_dir=base_dir,
        inbox_dir=base_dir / "inbox",
        failed_dir=base_dir / "failed",
        meetings_dir=base_dir / "meetings",
        hf_token=hf_token,
        claude_model=claude_model,
        whisper_model=whisper_model,
        ui_port=ui_port,
        ui_lang=ui_lang,
        speaker_match_high=speaker_match_high,
        speaker_match_low=speaker_match_low,
    )
