from unittest.mock import patch

from src.main import main


def test_main_creates_required_directories_and_starts_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch("src.main.load_whisper_model", return_value=object()),
        patch("src.main.load_diarization_pipeline", return_value=object()),
    ):
        main(base_dir=tmp_path)

    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "failed").is_dir()
    assert (tmp_path / "meetings").is_dir()
    mock_watch_loop.assert_called_once()


def test_main_loads_diarization_pipeline_once_and_passes_to_watch_loop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    loaded_pipeline = object()

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch("src.main.load_whisper_model", return_value=object()),
        patch(
            "src.main.load_diarization_pipeline", return_value=loaded_pipeline
        ) as mock_load,
    ):
        main(base_dir=tmp_path)

    # the GPU/CPU placement decision lives in load_diarization_pipeline, so the
    # watcher's long-lived pipeline must come from it -- not a bare from_pretrained
    mock_load.assert_called_once_with("hf-test-token")
    assert (
        mock_watch_loop.call_args.kwargs["diarization_pipeline"] is loaded_pipeline
    )


def test_main_loads_whisper_model_once_and_passes_to_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("WHISPER_MODEL", "medium")

    loaded_whisper_model = object()

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch(
            "src.main.load_whisper_model", return_value=loaded_whisper_model
        ) as mock_load_whisper,
        patch("src.main.load_diarization_pipeline", return_value=object()),
    ):
        main(base_dir=tmp_path)

    mock_load_whisper.assert_called_once_with("medium")
    assert mock_watch_loop.call_args.kwargs["whisper_model"] is loaded_whisper_model
