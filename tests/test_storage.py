from datetime import date

from src.job import JOB_SUFFIX, read_model, write_job
from src.storage import (
    archive_audio,
    create_meeting_folder,
    meeting_folder_name,
    move_to_failed,
    save_summary,
    save_transcript,
)

TODAY = date(2026, 7, 22)


def test_folder_name_for_a_named_recording_is_date_time_topic():
    # recorder stem: "<topic>-HH-MM-SS" -> "YYYY-MM-DD_HH-MM-<topic>"
    # (':' is illegal in a Windows path, so HH:MM is written HH-MM)
    assert meeting_folder_name("Meet1900-19-01-45", TODAY) == "2026-07-22_19-01-Meet1900"


def test_folder_name_keeps_a_topic_that_contains_dashes():
    assert (
        meeting_folder_name("Q3-2026-Review-09-30-00", TODAY)
        == "2026-07-22_09-30-Q3-2026-Review"
    )


def test_folder_name_for_an_unnamed_recording_has_no_topic():
    # unnamed recorder stem: "YYYY-MM-DD_HH-MM-SS" -> use its own date, drop the topic
    assert meeting_folder_name("2026-07-24_19-01-45", TODAY) == "2026-07-24_19-01"


def test_folder_name_for_a_user_dropped_file_keeps_the_whole_name():
    # no recorder timestamp to parse; just date-stamp whatever was dropped in
    assert meeting_folder_name("weekly-standup", TODAY) == "2026-07-22_weekly-standup"


def test_create_meeting_folder_builds_the_new_format_and_makes_the_dir(tmp_path):
    meetings_dir = tmp_path / "meetings"
    audio_path = tmp_path / "inbox" / "Meet1900-19-01-45.ogg"

    result = create_meeting_folder(audio_path, meetings_dir, today=TODAY)

    assert result == meetings_dir / "2026-07-22_19-01-Meet1900"
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


def test_move_to_failed_takes_the_job_file_along(tmp_path):
    # the next attempt must summarize with the model the user actually picked
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(inbox_dir, "broken", "claude-sonnet-5")

    move_to_failed(audio_path, failed_dir, "Summarization failed: boom")

    assert not (inbox_dir / f"broken{JOB_SUFFIX}").exists()
    assert read_model(failed_dir / "broken.mp3") == "claude-sonnet-5"


def test_move_to_failed_works_when_there_is_no_job_file(tmp_path):
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "dropped.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = move_to_failed(audio_path, failed_dir, "Transcription failed: boom")

    assert destination.exists()
