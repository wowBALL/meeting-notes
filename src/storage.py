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


def save_outputs(
    meeting_dir: Path, audio_path: Path, transcript_markdown: str, summary_markdown: str
) -> None:
    (meeting_dir / "transcript.md").write_text(transcript_markdown, encoding="utf-8")
    (meeting_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")
    shutil.move(str(audio_path), str(meeting_dir / audio_path.name))


def move_to_failed(audio_path: Path, failed_dir: Path, error_message: str) -> Path:
    failed_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_dir / audio_path.name
    shutil.move(str(audio_path), str(destination))
    error_log = failed_dir / f"{audio_path.stem}.error.log"
    error_log.write_text(error_message, encoding="utf-8")
    return destination
