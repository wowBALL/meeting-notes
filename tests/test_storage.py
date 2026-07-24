from datetime import date

from src.storage import (
    archive_audio,
    create_meeting_folder,
    move_to_failed,
    save_summary,
    save_transcript,
)


def test_create_meeting_folder_uses_date_and_filename_slug(tmp_path):
    meetings_dir = tmp_path / "meetings"
    audio_path = tmp_path / "inbox" / "weekly-standup.mp3"

    result = create_meeting_folder(audio_path, meetings_dir, today=date(2026, 7, 22))

    assert result == meetings_dir / "2026-07-22-weekly-standup"
    assert result.is_dir()


def test_save_transcript_writes_the_transcript_file(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_transcript(meeting_dir, "# Transcript")

    assert path == meeting_dir / "transcript.md"
    assert path.read_text(encoding="utf-8") == "# Transcript"


def test_save_summary_writes_the_summary_file(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, "# Summary")

    assert path == meeting_dir / "summary.md"
    assert path.read_text(encoding="utf-8") == "# Summary"


def test_archive_audio_moves_the_recording_into_the_meeting_folder(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = archive_audio(meeting_dir, audio_path)

    assert destination == meeting_dir / "weekly-standup.mp3"
    assert destination.exists()
    assert not audio_path.exists()


def test_move_to_failed_moves_file_and_writes_error_log(tmp_path):
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = move_to_failed(audio_path, failed_dir, "Transcription failed: network error")

    assert destination == failed_dir / "broken.mp3"
    assert destination.exists()
    assert not audio_path.exists()
    error_log = failed_dir / "broken.error.log"
    assert error_log.read_text(encoding="utf-8") == "Transcription failed: network error"
