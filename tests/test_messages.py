from src.messages import LANGUAGES, MESSAGES, render


def test_every_code_exists_in_every_language():
    reference = set(MESSAGES["th"])
    for lang in LANGUAGES:
        assert set(MESSAGES[lang]) == reference, f"{lang} ไม่ตรงกับ th"


def test_render_fills_in_the_parameters():
    assert render("device_changed", {"old": "A", "new": "B"}, "en") == (
        'Audio device changed from "A" to "B" — the recorder is still listening '
        "to the old one, so the far end will not be captured."
    )


def test_render_falls_back_to_thai_for_an_unknown_language():
    assert render("room_closed", {}, "de") == render("room_closed", {}, "th")


def test_render_returns_the_code_itself_when_it_is_unknown():
    assert render("no_such_code", {}, "th") == "no_such_code"


def test_render_returns_the_raw_template_when_a_parameter_is_missing():
    # ข้อความที่เติมค่าไม่ครบต้องไม่ทำให้ประชุมล่ม
    assert "{old}" in render("device_changed", {}, "th")


def test_render_accepts_no_params_at_all():
    assert render("room_closed") == MESSAGES["th"]["room_closed"]


def test_new_speaker_codes_exist_in_every_language():
    from src.messages import LANGUAGES, MESSAGES

    for code in ("speakers_matched", "speakers_pending", "speakers_failed"):
        for language in LANGUAGES:
            assert code in MESSAGES[language], f"{code} หายไปจากภาษา {language}"
