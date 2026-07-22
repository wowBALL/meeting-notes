import pytest

from src.config import load_config


def test_load_config_reads_required_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    config = load_config(base_dir=tmp_path)

    assert config.anthropic_api_key == "sk-ant-test"
    assert config.hf_token == "hf-test-token"
    assert config.claude_model == "claude-opus-4-8"
    assert config.whisper_model == "small"
    assert config.inbox_dir == tmp_path / "inbox"
    assert config.failed_dir == tmp_path / "failed"
    assert config.meetings_dir == tmp_path / "meetings"


def test_load_config_reads_claude_model_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")

    config = load_config(base_dir=tmp_path)

    assert config.claude_model == "claude-sonnet-5"


def test_load_config_reads_whisper_model_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("WHISPER_MODEL", "medium")

    config = load_config(base_dir=tmp_path)

    assert config.whisper_model == "medium"


def test_load_config_raises_when_required_env_var_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(KeyError):
        load_config(base_dir=tmp_path)
