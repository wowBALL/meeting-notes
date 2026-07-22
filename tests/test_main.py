import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from src.main import main


def _fake_pyannote(mock_pipeline_cls):
    """Fake `pyannote.audio` module so the test never loads the real model."""
    pyannote_pkg = ModuleType("pyannote")
    audio_mod = ModuleType("pyannote.audio")
    audio_mod.Pipeline = mock_pipeline_cls
    pyannote_pkg.audio = audio_mod
    return {"pyannote": pyannote_pkg, "pyannote.audio": audio_mod}


def test_main_creates_required_directories_and_starts_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    mock_pipeline_cls = MagicMock()
    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch.dict(sys.modules, _fake_pyannote(mock_pipeline_cls)),
    ):
        main(base_dir=tmp_path)

    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "failed").is_dir()
    assert (tmp_path / "meetings").is_dir()
    mock_watch_loop.assert_called_once()


def test_main_loads_diarization_pipeline_once_and_passes_to_watch_loop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    loaded_pipeline = object()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded_pipeline

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch.dict(sys.modules, _fake_pyannote(mock_pipeline_cls)),
    ):
        main(base_dir=tmp_path)

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", token="hf-test-token"
    )
    assert (
        mock_watch_loop.call_args.kwargs["diarization_pipeline"] is loaded_pipeline
    )
