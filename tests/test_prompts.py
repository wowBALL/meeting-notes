import logging

import pytest

from src.prompts import FALLBACKS, render


@pytest.fixture
def prompts_dir(tmp_path):
    root = tmp_path / "prompts"
    (root / "profiles").mkdir(parents=True)
    (root / "map.md").write_text(
        "สรุปช่วงนี้\n\n{glossary}\n\n{profile_rules}\n", encoding="utf-8"
    )
    (root / "reduce.md").write_text("รวมสรุป\n\n{profile_rules}\n", encoding="utf-8")
    (root / "single.md").write_text("สรุปทั้งไฟล์\n\n{glossary}\n", encoding="utf-8")
    (root / "profiles" / "dev.md").write_text("กฎ dev ล้วน", encoding="utf-8")
    (root / "profiles" / "cross.md").write_text("กฎข้ามฝ่าย", encoding="utf-8")
    return root


def test_render_reads_the_prompt_file(prompts_dir):
    result = render("reduce", prompts_dir=prompts_dir)

    assert result.startswith("รวมสรุป")


def test_render_substitutes_the_glossary(prompts_dir):
    result = render("map", glossary_text="## ตารางศัพท์\n- Electron", prompts_dir=prompts_dir)

    assert "## ตารางศัพท์\n- Electron" in result
    assert "{glossary}" not in result


def test_an_empty_glossary_leaves_no_placeholder_behind(prompts_dir):
    result = render("map", glossary_text="", prompts_dir=prompts_dir)

    assert "{glossary}" not in result


def test_render_injects_the_profile_rules(prompts_dir):
    dev = render("map", profile="dev", prompts_dir=prompts_dir)
    cross = render("map", profile="cross", prompts_dir=prompts_dir)

    assert "กฎ dev ล้วน" in dev
    assert "กฎข้ามฝ่าย" not in dev
    assert "กฎข้ามฝ่าย" in cross
    assert "{profile_rules}" not in cross


def test_an_unknown_profile_warns_and_falls_back_to_dev(prompts_dir, caplog):
    with caplog.at_level(logging.WARNING):
        result = render("map", profile="ไม่มีจริง", prompts_dir=prompts_dir)

    assert "กฎ dev ล้วน" in result
    assert "ไม่มีจริง" in caplog.text


def test_a_missing_prompt_file_falls_back_to_the_embedded_prompt(tmp_path, caplog):
    """prompt หายต้องไม่ทำให้สรุปไม่ได้ -- transcript ถอดด้วย GPU มาแล้ว
    สำคัญกว่ารูปแบบของสรุป"""
    with caplog.at_level(logging.WARNING):
        result = render("map", prompts_dir=tmp_path / "ไม่มีโฟลเดอร์นี้")

    assert result == FALLBACKS["map"]
    assert "map" in caplog.text


def test_a_missing_profile_file_still_renders_the_base_prompt(prompts_dir, caplog):
    (prompts_dir / "profiles" / "dev.md").unlink()

    with caplog.at_level(logging.WARNING):
        result = render("map", profile="dev", prompts_dir=prompts_dir)

    assert result.startswith("สรุปช่วงนี้")
    assert "{profile_rules}" not in result


def test_braces_in_a_prompt_file_survive_untouched(prompts_dir):
    """ต้องแทนที่ด้วย str.replace ไม่ใช่ str.format -- prompt ที่มีตัวอย่าง JSON
    หรือ {} ตัวอื่นจะทำให้ format ระเบิด (KeyError) ทั้งที่ไฟล์ถูกต้อง"""
    (prompts_dir / "reduce.md").write_text(
        'ตอบเป็น JSON เช่น {"topic": "x"} และ {ไม่ใช่ placeholder}\n\n{profile_rules}',
        encoding="utf-8",
    )

    result = render("reduce", profile="dev", prompts_dir=prompts_dir)

    assert '{"topic": "x"}' in result
    assert "{ไม่ใช่ placeholder}" in result
    assert "กฎ dev ล้วน" in result


def test_every_prompt_name_has_an_embedded_fallback():
    assert set(FALLBACKS) == {"map", "reduce", "single"}
    for name, text in FALLBACKS.items():
        assert text.strip(), f"fallback ของ {name} ว่าง"


def test_the_shipped_prompt_files_render_for_both_profiles():
    """เทสต์ทุกข้อข้างบนใช้ไฟล์ปลอมใน tmp_path -- ข้อนี้ยิงกับ prompts/ ของจริงที่
    ship ไปจริง เพราะไฟล์ที่เขียนเองใน fixture ไม่เคยพลาดแบบเดียวกับไฟล์จริง"""
    for name in FALLBACKS:
        for profile in ("dev", "cross"):
            result = render(name, profile=profile, glossary_text="ตาราง")

            assert result != FALLBACKS[name], f"{name}.md ที่ ship ไปอ่านไม่ได้"
            assert "{glossary}" not in result
            assert "{profile_rules}" not in result
            assert "ตาราง" in result or "{glossary}" not in result
