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
    assert config.claude_model == "claude-opus-5"
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


def test_load_config_defaults_the_ui_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.delenv("UI_PORT", raising=False)
    monkeypatch.delenv("UI_LANG", raising=False)

    config = load_config(base_dir=tmp_path)

    assert config.ui_port == 8765
    assert config.ui_lang == "th"


def test_load_config_reads_the_ui_settings_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setenv("UI_PORT", "9000")
    monkeypatch.setenv("UI_LANG", "en")

    config = load_config(base_dir=tmp_path)

    assert config.ui_port == 9000
    assert config.ui_lang == "en"


def test_load_config_falls_back_when_the_port_is_not_a_number(tmp_path, monkeypatch):
    # พอร์ตที่พิมพ์ผิดใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "h")
    monkeypatch.setenv("UI_PORT", "not-a-number")

    config = load_config(base_dir=tmp_path)

    assert config.ui_port == 8765
