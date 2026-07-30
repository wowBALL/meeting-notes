import pytest

from src.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_SPEAKER_MATCH_HIGH,
    DEFAULT_SPEAKER_MATCH_LOW,
    LEGACY_DIARIZATION_MODEL,
    load_config,
)


def test_load_config_defaults_to_the_measured_diarization_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("DIARIZATION_MODEL", raising=False)

    config = load_config(base_dir=tmp_path)

    # เขียนชื่อเต็มตรง ๆ ที่นี่ที่เดียว ไม่อ้างค่าคงที่: การเปลี่ยนโมเดลตั้งต้นต้องเป็น
    # การตัดสินใจที่มีคนแก้เทสต์ตาม ไม่ใช่สิ่งที่เลื่อนตามค่าคงที่ไปเงียบ ๆ
    assert config.diarization_model == "pyannote/speaker-diarization-community-1"


def test_load_config_reads_the_diarization_model_override(tmp_path, monkeypatch):
    """การกลับไปใช้ 3.1 ต้องทำได้ด้วย .env บรรทัดเดียว ไม่ต้อง revert โค้ด"""
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("DIARIZATION_MODEL", LEGACY_DIARIZATION_MODEL)

    config = load_config(base_dir=tmp_path)

    assert config.diarization_model == LEGACY_DIARIZATION_MODEL


def test_load_config_falls_back_when_the_diarization_model_is_blank(tmp_path, monkeypatch):
    # "DIARIZATION_MODEL=" ที่ลืมเติมค่า -- สตริงว่างที่หลุดไปถึง from_pretrained
    # ทำให้เปิดโปรแกรมไม่ได้เลย แบบเดียวกับที่ UI_PORT ที่พิมพ์ผิดต้องไม่ทำ
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("DIARIZATION_MODEL", "   ")

    config = load_config(base_dir=tmp_path)

    assert config.diarization_model == DEFAULT_DIARIZATION_MODEL


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


def test_load_config_reads_speaker_match_thresholds(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("SPEAKER_MATCH_HIGH", "0.82")
    monkeypatch.setenv("SPEAKER_MATCH_LOW", "0.61")

    config = load_config(base_dir=tmp_path)

    assert config.speaker_match_high == 0.82
    assert config.speaker_match_low == 0.61


def test_load_config_uses_default_speaker_thresholds(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("SPEAKER_MATCH_HIGH", raising=False)
    monkeypatch.delenv("SPEAKER_MATCH_LOW", raising=False)

    config = load_config(base_dir=tmp_path)

    # ค่าตัวเลขเขียนตรง ๆ ที่นี่ที่เดียว: การขยับเกณฑ์ต้องเป็นการตัดสินใจที่มีคนแก้
    # เทสต์ตาม ไม่ใช่สิ่งที่เลื่อนไปเองเพราะทุกที่อ้างค่าคงที่ตัวเดียวกันหมด
    assert config.speaker_match_high == 0.80
    assert config.speaker_match_low == 0.50


def test_load_config_falls_back_when_a_threshold_is_not_a_number(tmp_path, monkeypatch):
    # ค่าที่พิมพ์ผิดใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- แบบเดียวกับ UI_PORT
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("SPEAKER_MATCH_HIGH", "สูงมาก")

    config = load_config(base_dir=tmp_path)

    # เทสต์นี้ถามว่า "ตกกลับไปใช้ค่าเริ่มต้นไหม" ไม่ใช่ "ค่าเริ่มต้นเป็นเท่าไร"
    assert config.speaker_match_high == DEFAULT_SPEAKER_MATCH_HIGH


def test_load_config_falls_back_to_both_defaults_when_the_thresholds_are_inverted(
    tmp_path, monkeypatch, caplog
):
    # HIGH < LOW แปลว่าทุกคนที่ผ่าน LOW จะผ่าน HIGH ไปด้วย -- Match.confident จะเป็น
    # True ของทุกคน และระบบจะเริ่มใส่ชื่อจริงอัตโนมัติที่ความมั่นใจต่ำแบบเงียบ ๆ
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("SPEAKER_MATCH_HIGH", "0.4")
    monkeypatch.setenv("SPEAKER_MATCH_LOW", "0.6")

    with caplog.at_level("WARNING"):
        config = load_config(base_dir=tmp_path)

    assert config.speaker_match_high == DEFAULT_SPEAKER_MATCH_HIGH
    assert config.speaker_match_low == DEFAULT_SPEAKER_MATCH_LOW
    assert "0.4" in caplog.text
    assert "0.6" in caplog.text


def test_load_config_keeps_a_valid_threshold_pair_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("SPEAKER_MATCH_HIGH", "0.8")
    monkeypatch.setenv("SPEAKER_MATCH_LOW", "0.3")

    config = load_config(base_dir=tmp_path)

    assert config.speaker_match_high == 0.8
    assert config.speaker_match_low == 0.3


def test_load_config_accepts_equal_thresholds_as_the_degenerate_but_coherent_case(
    tmp_path, monkeypatch
):
    # high == low ไม่ใช่ "กลับด้าน" -- ทุกคนจะเป็น "มั่นใจ" หรือ "ไม่รู้จัก" อย่างใดอย่าง
    # หนึ่งเสมอ ไม่มีช่วงเสนอให้ยืนยัน ซึ่งเป็นค่าที่แปลกแต่สอดคล้องในตัวเอง ไม่ใช่ค่าที่
    # ต้องกันเหมือนกรณีกลับด้าน
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("SPEAKER_MATCH_HIGH", "0.6")
    monkeypatch.setenv("SPEAKER_MATCH_LOW", "0.6")

    config = load_config(base_dir=tmp_path)

    assert config.speaker_match_high == 0.6
    assert config.speaker_match_low == 0.6
