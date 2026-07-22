import logging
from pathlib import Path

from src.config import load_config
from src.watcher import watch_loop

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(base_dir: Path = PROJECT_ROOT) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(base_dir=base_dir)
    config.inbox_dir.mkdir(parents=True, exist_ok=True)
    config.failed_dir.mkdir(parents=True, exist_ok=True)
    config.meetings_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Watching %s for new audio files...", config.inbox_dir)
    watch_loop(config)


if __name__ == "__main__":
    main()
