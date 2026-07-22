from unittest.mock import patch

from src.config import Config
from src.watcher import is_file_stable, scan_inbox, watch_loop


def make_config(tmp_path) -> Config:
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        anthropic_api_key="sk-ant-test",
        hf_token="hf-test-token",
    )


def test_scan_inbox_returns_only_audio_files_sorted(tmp_path):
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    (inbox_dir / "b.mp3").write_bytes(b"x")
    (inbox_dir / "a.wav").write_bytes(b"x")
    (inbox_dir / "notes.txt").write_bytes(b"x")

    result = scan_inbox(inbox_dir)

    assert result == [inbox_dir / "a.wav", inbox_dir / "b.mp3"]


def test_scan_inbox_returns_empty_list_when_dir_missing(tmp_path):
    assert scan_inbox(tmp_path / "does-not-exist") == []


def test_is_file_stable_true_for_unchanging_file(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is True


def test_is_file_stable_false_for_empty_file(tmp_path):
    audio_path = tmp_path / "empty.mp3"
    audio_path.write_bytes(b"")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is False


def test_watch_loop_processes_stable_files_once_with_single_pass(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=None
    )


def test_watch_loop_threads_diarization_pipeline_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_pipeline = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, diarization_pipeline=sentinel_pipeline)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=sentinel_pipeline, whisper_model=None
    )


def test_watch_loop_threads_whisper_model_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_model = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, whisper_model=sentinel_model)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=sentinel_model
    )


def test_watch_loop_skips_unstable_files(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=False),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_not_called()
