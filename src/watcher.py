import logging
import time
from pathlib import Path
from typing import Any

from src.config import Config
from src.pipeline import process_file

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}


def scan_inbox(inbox_dir: Path) -> list[Path]:
    if not inbox_dir.exists():
        return []
    return sorted(
        p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def is_file_stable(path: Path, check_interval: float = 1.0) -> bool:
    size_before = path.stat().st_size
    time.sleep(check_interval)
    size_after = path.stat().st_size
    return size_before == size_after and size_before > 0


def watch_loop(
    config: Config,
    poll_interval: float = 5.0,
    single_pass: bool = False,
    diarization_pipeline: Any = None,
    whisper_model: Any = None,
) -> None:
    while True:
        for audio_path in scan_inbox(config.inbox_dir):
            if is_file_stable(audio_path, check_interval=0.5):
                try:
                    meeting_dir = process_file(
                        audio_path,
                        config,
                        diarization_pipeline=diarization_pipeline,
                        whisper_model=whisper_model,
                    )
                    logger.info("Processed %s -> %s", audio_path.name, meeting_dir)
                except Exception:
                    logger.exception("Failed to process %s", audio_path.name)
        if single_pass:
            return
        time.sleep(poll_interval)
