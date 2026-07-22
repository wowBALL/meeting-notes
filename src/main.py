import logging
from pathlib import Path

from src.config import load_config
from src.transcribe import load_whisper_model
from src.watcher import watch_loop

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(base_dir: Path = PROJECT_ROOT) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(base_dir=base_dir)
    config.inbox_dir.mkdir(parents=True, exist_ok=True)
    config.failed_dir.mkdir(parents=True, exist_ok=True)
    config.meetings_dir.mkdir(parents=True, exist_ok=True)

    # Load both models once at startup, then reuse them for every file,
    # instead of reloading from disk on each meeting.
    from pyannote.audio import Pipeline

    logging.info("Loading speaker-diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=config.hf_token
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
