import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    base_dir: Path
    inbox_dir: Path
    failed_dir: Path
    meetings_dir: Path
    anthropic_api_key: str
    hf_token: str
    claude_model: str = "claude-opus-5"
    whisper_model: str = "small"


def load_config(base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    load_dotenv(base_dir / ".env")

    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    hf_token = os.environ["HF_TOKEN"]
    claude_model = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    whisper_model = os.environ.get("WHISPER_MODEL", "small")

    return Config(
        base_dir=base_dir,
        inbox_dir=base_dir / "inbox",
        failed_dir=base_dir / "failed",
        meetings_dir=base_dir / "meetings",
        anthropic_api_key=anthropic_api_key,
        hf_token=hf_token,
        claude_model=claude_model,
        whisper_model=whisper_model,
    )
