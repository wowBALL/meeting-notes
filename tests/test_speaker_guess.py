from unittest.mock import MagicMock, patch

from src.job import NO_SUMMARY_MODEL
from src.llm import Completion, Provider
from src.speaker_guess import guess_speaker_names


def _provider(text: str) -> Provider:
    return Provider(
        model_id="fake-model",
        map_max_tokens=1234,
        reduce_max_tokens=2048,
        complete=MagicMock(return_value=Completion(text=text, truncated=False)),
    )


def test_guess_speaker_names_reads_the_json_the_model_returns():
    provider = _provider(
        '{"ผู้พูด 2": {"name": "พี่เอ็ม", "evidence": "มีคนเรียกพี่เอ็มตอน 12:40"}}'
    )

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# Transcript", ["ผู้พูด 2"], model="fake-model")

    assert result == {
        "ผู้พูด 2": {"name": "พี่เอ็ม", "evidence": "มีคนเรียกพี่เอ็มตอน 12:40"}
    }


def test_guess_speaker_names_pulls_the_json_out_of_surrounding_prose():
    # โมเดลชอบห่อ JSON ด้วยคำอธิบายหรือ code fence แม้จะสั่งว่าอย่าทำ
    provider = _provider('นี่คือผลลัพธ์:\n```json\n{"ผู้พูด 1": {"name": "บี"}}\n```\nหวังว่าจะช่วยได้')

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# Transcript", ["ผู้พูด 1"], model="fake-model")

    assert result == {"ผู้พูด 1": {"name": "บี", "evidence": ""}}


def test_guess_speaker_names_drops_labels_it_was_not_asked_about():
    provider = _provider('{"ผู้พูด 1": {"name": "บี"}, "ผู้พูด 9": {"name": "ผี"}}')

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# Transcript", ["ผู้พูด 1"], model="fake-model")

    assert list(result) == ["ผู้พูด 1"]


def test_guess_speaker_names_ignores_a_null_or_empty_guess():
    provider = _provider('{"ผู้พูด 1": null, "ผู้พูด 2": {"name": "   "}}')

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# T", ["ผู้พูด 1", "ผู้พูด 2"], model="fake-model")

    assert result == {}


def test_guess_speaker_names_uses_the_first_object_when_the_answer_has_two():
    # regex เดิมที่ greedy จะจับตั้งแต่ "{" ตัวแรกถึง "}" ตัวสุดท้าย ซึ่งไม่ใช่ JSON
    # ที่ถูกต้องเมื่อมีสอง object แยกกัน แล้วทิ้งคำตอบทั้งก้อนทั้งที่ตัวแรกอ่านได้ปกติ
    provider = _provider(
        '{"ผู้พูด 1": {"name": "เอ"}} ลองอีกแบบด้วย {"ผู้พูด 1": {"name": "บี"}}'
    )

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# T", ["ผู้พูด 1"], model="fake-model")

    assert result == {"ผู้พูด 1": {"name": "เอ", "evidence": ""}}


def test_guess_speaker_names_skips_a_stray_brace_in_prose_before_the_real_object():
    provider = _provider(
        'เขาพิมพ์ { เฉย ๆ ตรงนี้ แล้วสรุปว่า {"ผู้พูด 1": {"name": "ซี"}}'
    )

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# T", ["ผู้พูด 1"], model="fake-model")

    assert result == {"ผู้พูด 1": {"name": "ซี", "evidence": ""}}


def test_guess_speaker_names_returns_nothing_when_the_answer_is_not_json():
    provider = _provider("ผมเดาไม่ได้ครับ")

    with patch("src.speaker_guess.resolve", return_value=provider):
        result = guess_speaker_names("# T", ["ผู้พูด 1"], model="fake-model")

    assert result == {}


def test_guess_speaker_names_skips_the_call_entirely_in_transcript_only_mode():
    # transcript-only แปลว่าผู้ใช้เลือกไม่ให้ข้อความออกนอกเครื่อง การแอบส่งไปเดาชื่อ
    # คือการละเมิดสิ่งที่เขาเลือก ด้วยเหตุผลเดียวกับที่ระบบนี้ไม่มี cross-provider fallback
    with patch("src.speaker_guess.resolve") as resolve:
        result = guess_speaker_names("# T", ["ผู้พูด 1"], model=NO_SUMMARY_MODEL)

    assert result == {}
    resolve.assert_not_called()


def test_guess_speaker_names_skips_the_call_when_there_is_nobody_to_guess():
    with patch("src.speaker_guess.resolve") as resolve:
        assert guess_speaker_names("# T", [], model="fake-model") == {}

    resolve.assert_not_called()


def test_guess_speaker_names_asks_with_the_model_own_map_budget():
    provider = _provider("{}")

    with patch("src.speaker_guess.resolve", return_value=provider):
        guess_speaker_names("# Transcript ทั้งไฟล์", ["ผู้พูด 1"], model="fake-model")

    system, content, max_tokens = provider.complete.call_args.args
    assert max_tokens == 1234
    assert "ผู้พูด 1" in content
    assert "# Transcript ทั้งไฟล์" in content
