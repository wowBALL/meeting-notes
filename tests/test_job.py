from pathlib import Path

from src.job import (
    JOB_SUFFIX,
    discard_job,
    job_path_for,
    move_job,
    read_model,
    write_job,
)


def test_job_path_sits_beside_the_audio_with_the_job_suffix():
    audio_path = Path("inbox") / "meet1.ogg"

    assert job_path_for(audio_path) == Path("inbox") / f"meet1{JOB_SUFFIX}"


def test_write_job_then_read_model_round_trips(tmp_path):
    write_job(tmp_path, "meet1", "claude-sonnet-5")

    assert read_model(tmp_path / "meet1.ogg") == "claude-sonnet-5"


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
