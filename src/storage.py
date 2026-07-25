import re
import shutil
from datetime import date
from pathlib import Path

from src.job import move_job

# Stems produced by record.build_output_filename:
#   named:   "<topic>-HH-MM-SS"
#   unnamed: "YYYY-MM-DD_HH-MM-SS"
# A ':' is illegal in a Windows path, so the requested HH:MM separator between
# hour and minute is written as HH-MM.
_UNNAMED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-\d{2}$")
_NAMED_RE = re.compile(r"^(?P<topic>.+)-(\d{2})-(\d{2})-\d{2}$")


def meeting_folder_name(stem: str, today: date) -> str:
    """Build 'YYYY-MM-DD_HH-MM-<topic>' from a recording's file stem."""
    unnamed = _UNNAMED_RE.match(stem)
    if unnamed:
        day, hh, mm = unnamed.groups()
        return f"{day}_{hh}-{mm}"
    named = _NAMED_RE.match(stem)
    if named:
        return f"{today.isoformat()}_{named.group(2)}-{named.group(3)}-{named.group('topic')}"
    # A file the user dropped into inbox/ themselves carries no recorder
    # timestamp to parse, so keep the whole name and just date-stamp it.
    return f"{today.isoformat()}_{stem}"


def create_meeting_folder(
    audio_path: Path, meetings_dir: Path, today: date | None = None
) -> Path:
    today = today or date.today()
    meeting_dir = meetings_dir / meeting_folder_name(audio_path.stem, today)
    meeting_dir.mkdir(parents=True, exist_ok=True)
    return meeting_dir


# Saved separately because the pipeline writes them at different moments: the
# transcript goes to disk before summarizing, so a summarization failure can
# never discard a transcript that cost a full GPU pass over the recording.
def save_transcript(meeting_dir: Path, transcript_markdown: str) -> Path:
    path = meeting_dir / "transcript.md"
    path.write_text(transcript_markdown, encoding="utf-8")
    return path


def save_summary(meeting_dir: Path, summary_markdown: str, model: str) -> Path:
    # `model` is required, not optional: the point of choosing a model per meeting
    # is being able to judge afterwards whether the pricier one was worth it, and
    # a summary.md with no attribution cannot be judged at all.
    path = meeting_dir / "summary.md"
    body = summary_markdown.rstrip("\n")
    path.write_text(f"{body}\n\n---\nสรุปด้วย {model}\n", encoding="utf-8")
    return path


def archive_audio(meeting_dir: Path, audio_path: Path) -> Path:
    destination = meeting_dir / audio_path.name
    shutil.move(str(audio_path), str(destination))
    return destination


def move_to_failed(audio_path: Path, failed_dir: Path, error_message: str) -> Path:
    failed_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_dir / audio_path.name
    shutil.move(str(audio_path), str(destination))
    error_log = failed_dir / f"{audio_path.stem}.error.log"
    error_log.write_text(error_message, encoding="utf-8")
    # The job file follows the recording so a later retry summarizes with the
    # model the user actually picked. Handled here rather than at each of
    # process_file's six failure branches, where a seventh would eventually
    # forget it.
    move_job(audio_path, failed_dir)
    return destination
