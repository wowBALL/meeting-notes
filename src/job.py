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


def _load(audio_path: Path) -> dict:
    """The job file's contents, or {} when it is missing, unreadable, or not an object.

    Every field in here is an optimisation -- which model to summarize with, where
    an earlier run left the transcript. Losing one costs a re-run of work that is
    cheap next to the recording itself, so a few unreadable bytes must never fail
    the run; the callers fall back to their defaults instead.
    """
    path = job_path_for(audio_path)
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable job file %s: %s", path.name, e)
        return {}
    # Guard against JSON that is not an object (null, arrays, primitives)
    return parsed if isinstance(parsed, dict) else {}


def read_model(audio_path: Path) -> str | None:
    """The model chosen for this recording, or None to fall back to the config default."""
    model = _load(audio_path).get("claude_model")
    # Guard against non-string values to honor the return type annotation
    return model if isinstance(model, str) else None


def record_transcript(audio_path: Path, transcript_path: Path) -> None:
    """Remember where this recording's transcript was written.

    Read back by a later run so that a recording returning from failed/ is
    summarized against the transcript that already exists rather than being put
    through the GPU a second time for an identical result.
    """
    path = job_path_for(audio_path)
    data = _load(audio_path)
    data["transcript_path"] = str(transcript_path)
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        # Forgetting the pointer only costs a re-transcription on the next
        # attempt; the summary this run is about to produce is unaffected.
        logger.warning("Could not record the transcript path in %s: %s", path.name, e)


def read_transcript(audio_path: Path) -> Path | None:
    """The transcript an earlier run saved for this recording, if it is still there.

    The pointer outlives the file it names -- meetings/ is the user's folder and
    they may have moved or deleted it -- so the file is checked, not just the path.
    """
    value = _load(audio_path).get("transcript_path")
    if not isinstance(value, str):
        return None
    path = Path(value)
    return path if path.is_file() else None


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
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        path.replace(destination_dir / path.name)
    except OSError:
        pass
