import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.llm import DEFAULT_LLM_BASE_URL

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


@dataclass
class Config:
    base_dir: Path
    inbox_dir: Path
    failed_dir: Path
    meetings_dir: Path
    anthropic_api_key: str
    hf_token: str
    llm_api_key: str = ""
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    claude_model: str = DEFAULT_SUMMARY_MODEL
    whisper_model: str = "small"
    ui_port: int = DEFAULT_UI_PORT
    ui_lang: str = DEFAULT_UI_LANG


def load_config(base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    load_dotenv(base_dir / ".env")

    # ไม่บังคับ: ค่าเริ่มต้นคือ GLM ซึ่งไม่ใช้ key ของ Anthropic เลย คนตั้งเครื่องใหม่จึง
    # ต้องเริ่มงานได้โดยไม่มีมัน error ย้ายไปเกิดตอน llm.resolve() ของ provider ที่ต้อง
    # ใช้จริง ซึ่งบอกได้ว่าต้องตั้ง env var ตัวไหน
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    hf_token = os.environ["HF_TOKEN"]
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL
    claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_SUMMARY_MODEL)
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    # พอร์ตที่ตั้งมาผิดต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- ตกกลับไปที่ default
    try:
        ui_port = int(os.environ.get("UI_PORT", DEFAULT_UI_PORT))
    except ValueError:
        ui_port = DEFAULT_UI_PORT
    ui_lang = os.environ.get("UI_LANG", DEFAULT_UI_LANG)

    return Config(
        base_dir=base_dir,
        inbox_dir=base_dir / "inbox",
        failed_dir=base_dir / "failed",
        meetings_dir=base_dir / "meetings",
        anthropic_api_key=anthropic_api_key,
        hf_token=hf_token,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        claude_model=claude_model,
        whisper_model=whisper_model,
        ui_port=ui_port,
        ui_lang=ui_lang,
    )
