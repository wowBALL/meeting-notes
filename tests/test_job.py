from pathlib import Path

from src.job import (
    JOB_SUFFIX,
    discard_job,
    job_path_for,
    move_job,
    read_model,
    read_profile,
    read_transcript,
    record_transcript,
    write_job,
)


def test_job_path_sits_beside_the_audio_with_the_job_suffix():
    audio_path = Path("inbox") / "meet1.ogg"

    assert job_path_for(audio_path) == Path("inbox") / f"meet1{JOB_SUFFIX}"


def test_write_job_then_read_model_round_trips(tmp_path):
    write_job(tmp_path, "meet1", "claude-sonnet-5")

    assert read_model(tmp_path / "meet1.ogg") == "claude-sonnet-5"


def test_write_job_then_read_profile_round_trips(tmp_path):
    write_job(tmp_path, "meet1", "claude-sonnet-5", profile="cross")

    assert read_profile(tmp_path / "meet1.ogg") == "cross"
    # ต้องเดินทางคู่กับ model ในไฟล์เดียวกัน คิวสลับกันแล้วจะไม่ผิดอัน
    assert read_model(tmp_path / "meet1.ogg") == "claude-sonnet-5"


def test_write_job_without_a_profile_stores_nothing_for_it(tmp_path):
    write_job(tmp_path, "meet1", "claude-sonnet-5")

    assert read_profile(tmp_path / "meet1.ogg") is None
    assert read_model(tmp_path / "meet1.ogg") == "claude-sonnet-5"


def test_read_profile_returns_none_for_a_job_file_written_before_the_feature(tmp_path):
    """ไฟล์ที่ค้างใน inbox/ ตอนอัปเดตโค้ดต้องไม่พัง -- ผู้เรียกตกไปใช้ค่าจาก .env"""
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text(
        '{"claude_model": "GLM-5.2"}', encoding="utf-8"
    )

    assert read_profile(tmp_path / "meet1.ogg") is None


def test_read_profile_returns_none_when_the_value_is_not_a_string(tmp_path):
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text('{"profile": 7}', encoding="utf-8")

    assert read_profile(tmp_path / "meet1.ogg") is None


def test_read_profile_returns_none_when_there_is_no_job_file(tmp_path):
    assert read_profile(tmp_path / "dropped.mp3") is None


def test_read_model_returns_none_when_there_is_no_job_file(tmp_path):
    # a file the user dropped into inbox/ themselves carries no job file
    assert read_model(tmp_path / "dropped.mp3") is None


def test_read_model_returns_none_when_the_job_file_is_corrupt(tmp_path):
    # the transcript is already on disk by the time this is read -- a few
    # unreadable bytes must not cost a whole GPU pass over the recording
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text("{not json", encoding="utf-8")

    assert read_model(tmp_path / "meet1.ogg") is None


def test_read_model_returns_none_when_the_key_is_missing(tmp_path):
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text("{}", encoding="utf-8")

    assert read_model(tmp_path / "meet1.ogg") is None


def test_discard_job_removes_the_file(tmp_path):
    write_job(tmp_path, "meet1", "claude-opus-5")

    discard_job(tmp_path / "meet1.ogg")

    assert not (tmp_path / f"meet1{JOB_SUFFIX}").exists()


def test_discard_job_tolerates_a_missing_file(tmp_path):
    # the meeting is already summarized by this point; a cleanup that cannot run
    # must not raise. Reaching the assert at all is the behaviour under test.
    discard_job(tmp_path / "meet1.ogg")

    assert not (tmp_path / f"meet1{JOB_SUFFIX}").exists()


def test_move_job_moves_the_file_into_the_destination(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    failed = tmp_path / "failed"
    write_job(inbox, "meet1", "claude-sonnet-5")

    move_job(inbox / "meet1.ogg", failed)

    assert not (inbox / f"meet1{JOB_SUFFIX}").exists()
    assert read_model(failed / "meet1.ogg") == "claude-sonnet-5"


def test_move_job_is_a_no_op_when_there_is_no_job_file(tmp_path):
    failed = tmp_path / "failed"

    move_job(tmp_path / "dropped.mp3", failed)

    assert not failed.exists()


def test_move_job_does_not_raise_when_the_destination_cannot_be_created(tmp_path):
    # move_job is called from move_to_failed, which itself runs inside
    # process_file's except blocks -- a raise here would mask the original
    # pipeline error the user actually needs to see. A plain file sitting where
    # the destination directory should go makes mkdir fail in a way that
    # reproduces the same on every platform.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    failed = tmp_path / "failed"
    failed.write_text("not a directory", encoding="utf-8")
    write_job(inbox, "meet1", "claude-sonnet-5")

    move_job(inbox / "meet1.ogg", failed)

    assert (inbox / f"meet1{JOB_SUFFIX}").exists()


def test_read_model_returns_none_when_the_job_file_contains_null(tmp_path):
    # Valid JSON but not an object; must not raise AttributeError
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text("null", encoding="utf-8")

    assert read_model(tmp_path / "meet1.ogg") is None


def test_read_model_returns_none_when_the_job_file_is_a_json_array(tmp_path):
    # Valid JSON but not an object; must not raise AttributeError
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text("[1, 2]", encoding="utf-8")

    assert read_model(tmp_path / "meet1.ogg") is None


def test_read_model_returns_none_when_claude_model_value_is_not_a_string(tmp_path):
    # claude_model is a number instead of string; must not return wrong type
    (tmp_path / f"meet1{JOB_SUFFIX}").write_text('{"claude_model": 5}', encoding="utf-8")

    assert read_model(tmp_path / "meet1.ogg") is None


def _transcript_at(tmp_path: Path) -> Path:
    meeting_dir = tmp_path / "meetings" / "2026-07-25_14-30"
    meeting_dir.mkdir(parents=True)
    transcript = meeting_dir / "transcript.md"
    transcript.write_text("# Transcript\n", encoding="utf-8")
    return transcript


def test_record_transcript_then_read_transcript_round_trips(tmp_path):
    audio_path = tmp_path / "meet1.ogg"
    transcript = _transcript_at(tmp_path)

    record_transcript(audio_path, transcript)

    assert read_transcript(audio_path) == transcript


def test_record_transcript_keeps_the_model_already_in_the_job_file(tmp_path):
    # the model choice and the transcript pointer share one file; writing either
    # one must not erase the other
    write_job(tmp_path, "meet1", "claude-sonnet-5")
    audio_path = tmp_path / "meet1.ogg"

    record_transcript(audio_path, _transcript_at(tmp_path))

    assert read_model(audio_path) == "claude-sonnet-5"


def test_record_transcript_creates_the_job_file_when_there_is_none(tmp_path):
    # a file dropped into inbox/ by hand never had a job file written for it
    audio_path = tmp_path / "meet1.ogg"

    record_transcript(audio_path, _transcript_at(tmp_path))

    assert read_transcript(audio_path) is not None
    assert read_model(audio_path) is None


def test_read_transcript_returns_none_when_there_is_no_job_file(tmp_path):
    assert read_transcript(tmp_path / "meet1.ogg") is None


def test_read_transcript_returns_none_when_the_transcript_was_deleted(tmp_path):
    # the pointer outlives the file it points at, so existence has to be checked
    audio_path = tmp_path / "meet1.ogg"
    transcript = _transcript_at(tmp_path)
    record_transcript(audio_path, transcript)
    transcript.unlink()

    assert read_transcript(audio_path) is None


def test_read_transcript_returns_none_when_the_value_is_not_a_string(tmp_path):
    audio_path = tmp_path / "meet1.ogg"
    job_path_for(audio_path).write_text('{"transcript_path": 42}', encoding="utf-8")

    assert read_transcript(audio_path) is None
