from unittest.mock import patch

from src.main import main


def test_main_creates_required_directories_and_starts_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    with patch("src.main.watch_loop") as mock_watch_loop:
        main(base_dir=tmp_path)

    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "failed").is_dir()
    assert (tmp_path / "meetings").is_dir()
    mock_watch_loop.assert_called_once()
