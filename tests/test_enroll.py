from pathlib import Path

from src import enroll


def test_enroll_dir_and_done_dir_are_derived_from_base_dir(tmp_path):
    assert enroll.enroll_dir(tmp_path) == tmp_path / "enroll"
    assert enroll.done_dir(tmp_path) == tmp_path / "enroll" / "done"


def test_scan_audio_returns_only_audio_files_sorted(tmp_path):
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "b.wav").write_bytes(b"x")
    (directory / "a.ogg").write_bytes(b"x")
    (directory / "a.request.json").write_text("{}", encoding="utf-8")
    (directory / "notes.txt").write_bytes(b"x")

    assert enroll.scan_audio(tmp_path) == [directory / "a.ogg", directory / "b.wav"]


def test_scan_audio_ignores_the_done_subfolder(tmp_path):
    directory = tmp_path / "enroll"
    (directory / "done").mkdir(parents=True)
    (directory / "done" / "archived.ogg").write_bytes(b"x")
    (directory / "live.ogg").write_bytes(b"x")

    assert enroll.scan_audio(tmp_path) == [directory / "live.ogg"]


def test_scan_audio_returns_empty_list_when_dir_missing(tmp_path):
    assert enroll.scan_audio(tmp_path) == []


def test_is_safe_filename_rejects_paths_that_escape_the_folder():
    assert enroll.is_safe_filename("สมชาย.ogg") is True
    assert enroll.is_safe_filename("../../evil.ogg") is False
    assert enroll.is_safe_filename("sub/dir.ogg") is False
    assert enroll.is_safe_filename("C:\\Windows\\evil.ogg") is False
    assert enroll.is_safe_filename("") is False
    assert enroll.is_safe_filename(".") is False
    assert enroll.is_safe_filename("..") is False
    assert enroll.is_safe_filename(None) is False


def test_suggested_name_strips_extension_and_markdown_characters():
    assert enroll.suggested_name_from("สมชาย.ogg") == "สมชาย"
    assert enroll.suggested_name_from("พี่ *เอ* [1].wav") == "พี่ เอ 1"
    assert enroll.suggested_name_from("  a   b .m4a") == "a b"


def test_suggested_name_is_empty_when_nothing_usable_remains():
    assert enroll.suggested_name_from("***.ogg") == ""
