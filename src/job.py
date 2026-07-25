import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Sits beside <stem>.ogg in inbox/. watcher.scan_inbox() filters on audio
# extensions (src/watcher.py:11), so this file is invisible to the scan and can
# never be mistaken for a recording waiting to be processed.
JOB_SUFFIX = ".job.json"


def job_path_for(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}{JOB_SUFFIX}")


def write_job(inbox_dir: Path, stem: str, claude_model: str) -> Path:
    path = inbox_dir / f"{stem}{JOB_SUFFIX}"
    path.write_text(
        json.dumps({"claude_model": claude_model}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_model(audio_path: Path) -> str | None:
    """The model chosen for this recording, or None to fall back to the config default."""
    path = job_path_for(audio_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("claude_model")
    except (OSError, ValueError) as e:
        # By the time this is read, the transcript already exists -- a full GPU
        # pass over the recording. Failing the run over a few unreadable bytes is
        # never the right trade, so fall back to the configured default instead.
        logger.warning("Ignoring unreadable job file %s: %s", path.name, e)
        return None


def discard_job(audio_path: Path) -> None:
    # Swallowed like the part-file cleanup in segments.finish_session
    # (src/segments.py:203-210): on Windows an AV scanner can hold a handle for a
    # moment, and the meeting is already summarized by now -- a cleanup that
    # cannot run must not fail the run.
    try:
        job_path_for(audio_path).unlink()
    except OSError:
        pass


def move_job(audio_path: Path, destination_dir: Path) -> None:
    path = job_path_for(audio_path)
    if not path.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.replace(destination_dir / path.name)
    except OSError:
        pass
