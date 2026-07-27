import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.llm import DEFAULT_LLM_BASE_URL

# preflight อ่านค่านี้ด้วย โดยไม่ผ่าน load_config -- จึงต้องอยู่ตรงนี้ ไม่ใช่ฝังใน default
# ของ dataclass ที่แห่งเดียว
DEFAULT_CLAUDE_MODEL = "claude-opus-5"

# ชื่อใหม่ของค่าเดียวกัน: มันไม่ใช่ "Claude model" อีกแล้วเมื่อ provider มีมากกว่าหนึ่ง
# ชื่อ field claude_model ใน Config และใน job.json คงไว้ตามเดิมโดยเจตนา -- การรีเนม
# ไฟล์ sidecar ที่ค้างอยู่ใน inbox/ ไม่คุ้มกับความสวยของชื่อ
DEFAULT_SUMMARY_MODEL = DEFAULT_CLAUDE_MODEL

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
    claude_model: str = DEFAULT_CLAUDE_MODEL
    whisper_model: str = "small"
    ui_port: int = DEFAULT_UI_PORT
    ui_lang: str = DEFAULT_UI_LANG


def load_config(base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    load_dotenv(base_dir / ".env")

    # ไม่บังคับอีกแล้ว: ค่าเริ่มต้นคือ GLM ซึ่งไม่ใช้ key ของ Anthropic เลย คนตั้งเครื่อง
    # ใหม่ต้องเริ่มงานได้โดยไม่มี key ตัวนี้ error ย้ายไปเกิดตอน llm.resolve() ของ
    # provider ที่ต้องใช้จริง ซึ่งบอกได้ว่าต้องตั้งตัวไหน
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    hf_token = os.environ["HF_TOKEN"]
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL
    claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
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
