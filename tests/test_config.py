import pytest

from src.config import load_config


def test_load_config_reads_required_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    config = load_config(base_dir=tmp_path)

    assert config.hf_token == "hf-test-token"
    assert config.claude_model == "GLM-5.2"
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


def test_load_config_works_without_an_anthropic_key(tmp_path, monkeypatch):
    """GLM จะเป็นค่าเริ่มต้น -- คนที่ตั้งเครื่องใหม่ต้องเริ่มงานได้โดยไม่มี key ของ Anthropic

    Config ไม่มี field anthropic_api_key อีกแล้ว (ไม่มีใครอ่านมันในโค้ดจริง --
    ดู src/llm.py ที่อ่าน ANTHROPIC_API_KEY ตรงจาก os.environ เอง) เทสต์นี้จึงยืนยัน
    แค่ว่า load_config ไม่ raise และยังได้ค่าเริ่มต้น GLM ตามเดิมแม้ไม่มี key นี้เลย
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    (tmp_path / ".env").write_text("", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.claude_model == "GLM-5.2"


def test_the_default_summary_model_is_the_one_that_keeps_data_in_house():
    """ทางที่เป็นส่วนตัวต้องเป็นทางที่ไม่ต้องคิด

    ถ้า Claude เป็นค่าเริ่มต้น การกด Enter ผ่านเมนูจะส่ง transcript ออกนอกบริษัททุกครั้ง
    โดยไม่มีใครตัดสินใจเรื่องนั้นเลย -- ค่าเริ่มต้นจึงต้องเป็น provider ที่ข้อมูลไม่ออกไปไหน
    """
    from src.config import DEFAULT_SUMMARY_MODEL
    from src.llm import resolve

    assert DEFAULT_SUMMARY_MODEL == "GLM-5.2"
    # ต้องอยู่ใน registry จริง ไม่ใช่แค่สตริงที่พิมพ์ถูก
    assert resolve(DEFAULT_SUMMARY_MODEL).model_id == "GLM-5.2"


def test_the_misnamed_claude_default_constant_is_gone():
    """DEFAULT_CLAUDE_MODEL ถือค่าที่ไม่ใช่ Claude แล้วจะเป็นชื่อที่โกหก -- ลบทิ้ง
    ไม่เก็บเป็น alias ส่วนชื่อ field claude_model กับใน job.json คงไว้ตามที่ตัดสินใจ
    เพราะการรีเนมกระทบ sidecar ที่ค้างอยู่ใน inbox/"""
    import src.config as config

    assert not hasattr(config, "DEFAULT_CLAUDE_MODEL")
