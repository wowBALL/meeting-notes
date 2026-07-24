import shutil
from datetime import date
from pathlib import Path


def create_meeting_folder(
    audio_path: Path, meetings_dir: Path, today: date | None = None
) -> Path:
    today = today or date.today()
    slug = audio_path.stem
    folder_name = f"{today.isoformat()}-{slug}"
    meeting_dir = meetings_dir / folder_name
    meeting_dir.mkdir(parents=True, exist_ok=True)
    return meeting_dir


# Saved separately because the pipeline writes them at different moments: the
# transcript goes to disk before summarizing, so a summarization failure can
# never discard a transcript that cost a full GPU pass over the recording.
def save_transcript(meeting_dir: Path, transcript_markdown: str) -> Path:
    path = meeting_dir / "transcript.md"
    path.write_text(transcript_markdown, encoding="utf-8")
    return path


def save_summary(meeting_dir: Path, summary_markdown: str) -> Path:
    path = meeting_dir / "summary.md"
    path.write_text(summary_markdown, encoding="utf-8")
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
    return destination
