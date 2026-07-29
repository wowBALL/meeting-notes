import logging
import time
from pathlib import Path
from typing import Any

from src import enroll
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


def process_enroll_requests(config: Config, diarization_pipeline: Any = None) -> None:
    """วิเคราะห์ไฟล์ลงทะเบียนเสียงที่มีใบสั่งงานค้างอยู่

    ใช้ pipeline ตัวเดียวกับที่ถอดเทปประชุมโดยเจตนา: เวกเตอร์ที่จะเอาไปเทียบกันต้องมา
    จากโมเดลเดียวกัน ไม่งั้น cosine similarity ระหว่างสองฝั่งไม่มีความหมายเลย

    ไม่เคยเขียน registry.json -- เขียนแค่ผลวิเคราะห์ไว้ให้คนมากดยืนยันที่หน้าเว็บ
    """
    for audio_file in enroll.pending_requests(config.base_dir):
        audio_path = enroll.enroll_dir(config.base_dir) / audio_file
        try:
            analyzed = enroll.analyze(audio_path, pipeline=diarization_pipeline)
            enroll.write_result(config.base_dir, audio_file, analyzed)
            logger.info(
                "Analyzed enrollment clip %s -> %s", audio_file, analyzed.get("status")
            )
        except Exception:
            # enroll.analyze ไม่ raise อยู่แล้ว ตัวนี้กันความล้มเหลวของ "การเขียนไฟล์ผล"
            # โดยเฉพาะ -- ไฟล์เดียวที่เขียนไม่ได้ต้องไม่ทำให้ไฟล์อื่นในคิวไม่ถูกทำ
            logger.exception("Failed to analyze enrollment clip %s", audio_file)


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
        # หลัง inbox เสมอ: การประชุมที่อัดซ้ำไม่ได้ต้องได้ GPU ก่อนงานลงทะเบียนเสียง
        # ซึ่งผู้ใช้สั่งใหม่ได้ตลอด try/except ของตัวเองอยู่ข้างในแล้ว
        process_enroll_requests(config, diarization_pipeline=diarization_pipeline)
        if single_pass:
            return
        time.sleep(poll_interval)
