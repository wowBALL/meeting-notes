import logging
import os
from pathlib import Path

if os.name == "nt":
    # Since Python 3.8, PATH is no longer consulted when ctypes/torch resolve
    # dependency DLLs (e.g. torchcodec loading ffmpeg's avcodec-*.dll), even
    # though PATH itself is set correctly. Re-registering every PATH entry
    # here restores that lookup regardless of how the process was launched.
    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        if _dir and os.path.isdir(_dir):
            try:
                os.add_dll_directory(_dir)
            except OSError:
                pass

from src.config import load_config
from src.diarize import load_diarization_pipeline
from src.enroll import enroll_dir
from src.transcribe import load_whisper_model
from src.watcher import watch_loop

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(base_dir: Path = PROJECT_ROOT) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(base_dir=base_dir)
    config.inbox_dir.mkdir(parents=True, exist_ok=True)
    config.failed_dir.mkdir(parents=True, exist_ok=True)
    config.meetings_dir.mkdir(parents=True, exist_ok=True)
    # enroll\ ไม่เคยถูกสร้างมาก่อน (finding 5 ของรีวิวรอบสุดท้าย) -- write_request
    # ต้องการให้ไฟล์เสียงมีอยู่จริงก่อนเขียนใบสั่งงาน ผู้ใช้จึงต้องมีโฟลเดอร์นี้อยู่แล้ว
    # ตั้งแต่ checkout ใหม่ ๆ ไม่งั้น README บอกให้วางไฟล์ในโฟลเดอร์ที่ไม่มีอยู่จริง
    enroll_dir(config.base_dir).mkdir(parents=True, exist_ok=True)

    # Load both models once at startup, then reuse them for every file,
    # instead of reloading from disk on each meeting.
    logging.info("Loading speaker-diarization model (%s)...", config.diarization_model)
    diarization_pipeline = load_diarization_pipeline(
        config.hf_token, config.diarization_model
    )

    logging.info("Loading Whisper model (%s)...", config.whisper_model)
    whisper_model = load_whisper_model(config.whisper_model)

    logging.info("Watching %s for new audio files...", config.inbox_dir)
    watch_loop(
        config,
        diarization_pipeline=diarization_pipeline,
        whisper_model=whisper_model,
    )


if __name__ == "__main__":
    main()
