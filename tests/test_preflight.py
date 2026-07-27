import json
import math
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.preflight as preflight
from src.config import DEFAULT_SUMMARY_MODEL
from src.llm import LLM_TIMEOUT_SECONDS, Completion, MissingApiKeyError
from src.preflight import (
    LOOPBACK_SILENT_DBFS,
    MIC_GOOD_DBFS,
    MIC_WEAK_DBFS,
    PROBE_TIMEOUT_SECONDS,
    CheckResult,
    check_api_key,
    check_summary_model,
    check_summary_readiness,
    classify_probe_error,
    evaluate_loopback,
    evaluate_mic,
    evaluate_samplerate,
    format_report,
    peak_dbfs,
    probe_summary_model,
    read_summary_settings,
    run_preflight,
)


def test_peak_dbfs_converts_amplitude_to_decibels():
    assert peak_dbfs(1.0) == 0.0
    assert round(peak_dbfs(0.5), 1) == -6.0
    assert peak_dbfs(0.0) == -math.inf


def test_evaluate_mic_passes_on_speech_level_signal():
    result = evaluate_mic(-15.0)

    assert result.status == "ok"
    assert "ไมค์" in result.name


def test_evaluate_mic_warns_when_signal_is_weak():
    # audible but far below speech level: the exact state that produced an empty
    # transcript on 2026-07-24 -- worth a warning, not a hard stop
    result = evaluate_mic((MIC_GOOD_DBFS + MIC_WEAK_DBFS) / 2)

    assert result.status == "warn"


def test_evaluate_mic_fails_on_silence():
    result = evaluate_mic(-math.inf)

    assert result.status == "fail"


def test_evaluate_loopback_passes_when_audio_is_flowing():
    result = evaluate_loopback(-20.0, "Speakers (NX-S2)")

    assert result.status == "ok"
    assert "NX-S2" in result.detail


def test_evaluate_loopback_warns_when_the_default_output_is_silent():
    # cannot distinguish "nothing playing" from "the meeting app targets another
    # device", and the second one silently loses the far end of the meeting
    result = evaluate_loopback(LOOPBACK_SILENT_DBFS - 10, "Speakers (NX-S2)")

    assert result.status == "warn"
    assert "NX-S2" in result.detail


def test_format_report_marks_every_check_and_ends_with_a_verdict():
    report = format_report(
        [
            CheckResult("ไมค์", "ok", "peak -15.0 dB"),
            CheckResult("ลำโพง", "warn", "ไม่มีเสียง"),
        ]
    )

    assert "ไมค์" in report and "ลำโพง" in report
    assert "peak -15.0 dB" in report
    assert report.strip().splitlines()[-1].startswith("สรุป")


def test_format_report_verdict_is_ready_only_when_nothing_failed():
    ok_report = format_report([CheckResult("ไมค์", "ok", "")])
    failed_report = format_report([CheckResult("ไมค์", "fail", "")])

    assert "พร้อมอัด" in ok_report
    assert "พร้อมอัด" not in failed_report


def test_evaluate_samplerate_passes_when_both_devices_agree():
    result = evaluate_samplerate(
        {"defaultSampleRate": 48000.0}, {"defaultSampleRate": 48000.0}
    )

    assert result.status == "ok"
    assert "48000" in result.detail


def test_evaluate_samplerate_fails_when_the_two_rates_differ():
    # record.py ปฏิเสธการอัดในกรณีนี้ ถ้า preflight บอกว่าผ่าน คนใช้จะเสียเวลาตรวจเสียง
    # เลือกโมเดล ใส่ชื่อประชุม แล้วค่อยเจอ error ตอนที่จอขึ้นคำว่าเริ่มอัดไปแล้ว
    result = evaluate_samplerate(
        {"defaultSampleRate": 44100.0}, {"defaultSampleRate": 48000.0}
    )

    assert result.status == "fail"
    assert "44100" in result.detail and "48000" in result.detail
    assert "mmsys.cpl" in result.detail


def test_run_preflight_measures_both_devices_and_checks_the_samplerate():
    mic_device = {
        "name": "Microphone (Realtek)",
        "maxInputChannels": 2,
        "index": 1,
        "defaultSampleRate": 48000.0,
    }
    loopback_device = {
        "name": "Speakers (NX-S2) [Loopback]",
        "maxInputChannels": 2,
        "index": 9,
        "defaultSampleRate": 48000.0,
    }

    with (
        patch("src.preflight.pyaudio_instance", return_value=MagicMock()),
        patch("src.preflight.get_wasapi_mic_device", return_value=mic_device),
        patch("src.preflight.get_wasapi_loopback_device", return_value=loopback_device),
        patch("src.preflight.measure_peaks", return_value=(0.2, 0.001)),
    ):
        results = run_preflight(seconds=1)

    assert [r.name for r in results] == ["ไมค์", "ลำโพง (คู่สนทนา)", "sample rate"]
    assert results[0].status == "ok"  # -14 dB
    assert results[1].status == "warn"  # -60 dB
    assert results[2].status == "ok"  # ทั้งคู่ 48000 Hz


def test_run_preflight_reports_a_missing_device_as_a_failure():
    with (
        patch("src.preflight.pyaudio_instance", return_value=MagicMock()),
        patch(
            "src.preflight.get_wasapi_mic_device",
            side_effect=RuntimeError("ไม่พบไมค์ default"),
        ),
    ):
        results = run_preflight(seconds=1)

    assert results[0].status == "fail"
    assert "ไม่พบไมค์ default" in results[0].detail


def test_measure_peaks_returns_the_loudest_block_from_each_stream():
    # the callbacks pyaudio would invoke, driven by hand
    captured = {}

    def fake_open(**kwargs):
        captured[kwargs["input_device_index"]] = kwargs["stream_callback"]
        return MagicMock()

    audio = MagicMock()
    audio.open.side_effect = fake_open

    def drive(_seconds):
        quiet = np.array([0.01, -0.02], dtype=np.float32).tobytes()
        loud = np.array([0.5, -0.9], dtype=np.float32).tobytes()
        captured[1](quiet, 1, None, None)
        captured[1](loud, 1, None, None)
        captured[2](quiet, 1, None, None)

    with patch("src.preflight.time.sleep", side_effect=drive):
        mic_peak, loopback_peak = measure_peaks_for_test(audio)

    # float32 round-trip, so compare with a tolerance
    assert mic_peak == pytest.approx(0.9)
    assert loopback_peak == pytest.approx(0.02)


def measure_peaks_for_test(audio):
    mic_device = {"maxInputChannels": 1, "index": 1, "defaultSampleRate": 48000.0}
    loopback_device = {"maxInputChannels": 1, "index": 2, "defaultSampleRate": 48000.0}
    return preflight.measure_peaks(audio, mic_device, loopback_device, seconds=1)


class FakeAPIError(Exception):
    """รูปร่างเดียวกับ anthropic.APIStatusError: exception ที่พก .status_code มาด้วย"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def failing_probe(error):
    def probe(api_key, model):
        raise error

    return probe


def test_check_api_key_reports_a_missing_key_without_calling_the_api():
    called = []

    result = check_api_key("", "claude-opus-5", probe=lambda k, m: called.append(k))

    assert result.status == "warn"
    assert "ANTHROPIC_API_KEY" in result.detail
    assert called == []


def test_check_api_key_passes_when_the_probe_succeeds():
    # บรรทัดเดียวสั้น ๆ -- รายงานนี้พิมพ์ลง console กว้าง 80 คอลัมน์ ข้อความยาวจะตัดบรรทัด
    # จนอ่านยาก และกรณีผ่านคือกรณีที่ไม่ต้องให้ใครอ่านอะไรเพิ่ม
    result = check_api_key("sk-ant-test", "claude-opus-5", probe=lambda k, m: None)

    assert result.status == "ok"
    assert result.detail == "ใช้งานได้ปกติ"


def test_check_api_key_reports_an_expired_key():
    result = check_api_key(
        "sk-ant-test",
        "claude-opus-5",
        probe=failing_probe(FakeAPIError(401, "invalid x-api-key")),
    )

    assert result.status == "warn"
    assert "หมดอายุ" in result.detail


@pytest.mark.parametrize("status_code", [400, 403])
def test_check_api_key_reports_an_exhausted_credit_balance(status_code):
    result = check_api_key(
        "sk-ant-test",
        "claude-opus-5",
        probe=failing_probe(
            FakeAPIError(status_code, "Your credit balance is too low to access the API")
        ),
    )

    assert result.status == "warn"
    assert "เครดิต" in result.detail


def test_check_api_key_treats_a_rate_limit_as_a_working_key():
    # 429 แปลว่า key ผ่าน auth และมีเครดิตแล้ว แค่ยิงถี่เกินไปตอนนี้
    result = check_api_key(
        "sk-ant-test",
        "claude-opus-5",
        probe=failing_probe(FakeAPIError(429, "rate limit exceeded")),
    )

    assert result.status == "ok"


def test_check_api_key_surfaces_an_unrecognised_failure_verbatim():
    result = check_api_key(
        "sk-ant-test",
        "claude-opus-5",
        probe=failing_probe(OSError("getaddrinfo failed")),
    )

    assert result.status == "warn"
    assert "getaddrinfo failed" in result.detail


@pytest.mark.parametrize(
    "error",
    [
        None,
        FakeAPIError(401, "invalid x-api-key"),
        FakeAPIError(400, "Your credit balance is too low"),
        FakeAPIError(429, "rate limit exceeded"),
        FakeAPIError(500, "internal server error"),
        OSError("network unreachable"),
    ],
)
def test_check_api_key_never_blocks_recording(error):
    # start-meeting.bat ถามยกเลิกการอัดเมื่อเจอ "fail" แต่ key เสียไม่ใช่เหตุให้ไม่อัด:
    # transcript ยังได้ครบ เอาไปให้ Claude สรุปทีหลังได้ ส่วนประชุมที่ไม่ได้อัดนั้นหายถาวร
    def probe(api_key, model):
        if error is not None:
            raise error

    for api_key in ("", "sk-ant-test"):
        assert check_api_key(api_key, "claude-opus-5", probe=probe).status != "fail"


def test_check_summary_model_warns_on_a_model_outside_the_registry():
    # เคยเป็น "ส่ง CLAUDE_MODEL ตรงเข้า API" มาก่อน -- id อะไรที่ API รับก็ใช้ได้ พอ
    # summarize_transcript หันมาเรียกผ่าน resolve() แล้ว id ที่ไม่อยู่ใน registry จะ
    # โยน UnknownModelError กลางท่อ หลังถอดเสียงเสร็จไปแล้ว ต้องรู้ตอนนี้ ไม่ใช่ตอนนั้น
    result = check_summary_model("claude-opus-4-8")

    assert result.status == "warn"
    assert "claude-opus-4-8" in result.detail
    assert "claude-opus-5" in result.detail
    assert "claude-sonnet-5" in result.detail


def test_check_summary_model_passes_for_a_registered_model():
    result = check_summary_model("claude-opus-5")

    assert result.status == "ok"


def test_check_summary_model_never_fails():
    # เหตุผลเดียวกับ check_api_key: resolve ไม่ผ่านไม่ได้ทำให้อัดหรือถอดเสียงไม่ได้
    # transcript ยังออกมาครบ เอาไปสรุปด้วยมือทีหลังได้
    assert check_summary_model("not-a-real-model").status != "fail"


# --- probe_summary_model / check_api_key ผ่าน provider ที่ resolve ได้จริง ----------


def test_probe_summary_model_uses_the_provider_of_the_given_model():
    # ค่าเริ่มต้นเป็น GLM แล้ว การตรวจ key ของ Anthropic เสมอจะบอกผิดฝั่งทุกครั้ง
    called = {}

    def fake_complete(system, content, max_tokens, timeout=None):
        called["max_tokens"] = max_tokens
        called["timeout"] = timeout
        return Completion(text="hi", truncated=False)

    provider = MagicMock()
    provider.complete = fake_complete

    with patch("src.preflight.resolve", return_value=provider) as resolve_mock:
        probe_summary_model("GLM-5.2")

    resolve_mock.assert_called_once_with("GLM-5.2")
    assert called["max_tokens"] == 1


def test_probe_summary_model_passes_its_own_short_timeout_not_the_providers():
    # ตัวเก่ามี max_retries=0 กับ timeout สั้นเฉพาะของ probe -- Task 5 ทำให้ probe ยิงผ่าน
    # resolve(model).complete() ที่ปรับแต่งมาสำหรับสรุปทั้งก้อน (LLM_TIMEOUT_SECONDS =
    # 900) ต้องส่ง timeout สั้นของตัวเองเข้าไปแทนเสมอ ไม่งั้น preflight ค้างได้ถึง 15 นาที
    # ก่อนเข้าประชุม
    called = {}

    def fake_complete(system, content, max_tokens, timeout=None):
        called["timeout"] = timeout
        return Completion(text="hi", truncated=False)

    provider = MagicMock()
    provider.complete = fake_complete

    with patch("src.preflight.resolve", return_value=provider):
        probe_summary_model("claude-opus-5")

    assert called["timeout"] == PROBE_TIMEOUT_SECONDS
    assert called["timeout"] != LLM_TIMEOUT_SECONDS


def test_probe_forwards_the_probe_timeout_to_the_openai_compatible_transport():
    # จบที่ transport จริง (urlopen) ไม่ใช่แค่ที่ตัว fake -- กัน regression ที่ตัว
    # completer เองเมินพารามิเตอร์ timeout ที่ส่งเข้าไป
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
            ).encode("utf-8")
        )

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
    ):
        probe_summary_model("GLM-5.2")

    assert captured["timeout"] == PROBE_TIMEOUT_SECONDS
    assert captured["timeout"] != LLM_TIMEOUT_SECONDS


def test_probe_forwards_the_probe_timeout_to_the_anthropic_sdk_client():
    # จบที่ constructor ของ Anthropic client จริง -- นี่คือจุดที่ max_retries=0 ต้องติด
    # ไปด้วยเสมอตอน probe เพื่อไม่ให้ SDK ลองซ้ำเอง
    block = MagicMock()
    block.type = "text"
    block.text = "hi"
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = response

    with (
        patch("anthropic.Anthropic", return_value=client) as anthropic_cls,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        probe_summary_model("claude-opus-5")

    kwargs = anthropic_cls.call_args.kwargs
    assert kwargs["timeout"] == PROBE_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == 0


def test_probe_does_not_retry_the_openai_compatible_transport_on_failure():
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise OSError("getaddrinfo failed")

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(OSError),
    ):
        probe_summary_model("GLM-5.2")

    assert calls["n"] == 1


def test_an_empty_answer_from_the_probe_still_passes():
    # probe ถามว่า key ใช้ได้และโมเดลมีอยู่ ไม่ได้ถามคุณภาพคำตอบ -- GLM ที่ max_tokens=1
    # ใช้โควตาหมดไปกับ reasoning แล้วคืน content ว่าง ซึ่งพิสูจน์แล้วว่าไปถึงโมเดลจริง
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError(
        "GLM-5.2 returned no text (finish_reason='length')"
    )

    with patch("src.preflight.resolve", return_value=provider):
        result = check_api_key("sk-test", "GLM-5.2")

    assert result.status == "ok"


def test_a_missing_key_names_the_env_var_of_that_provider():
    provider = MagicMock()
    provider.complete.side_effect = MissingApiKeyError(
        "ไม่ได้ตั้ง LLM_API_KEY ใน .env"
    )

    with patch("src.preflight.resolve", return_value=provider):
        result = check_api_key("sk-test", "GLM-5.2")

    assert result.status == "warn"
    assert "LLM_API_KEY" in result.detail


def test_probe_summary_model_reraises_an_http_status_error():
    # HttpStatusError พก .status_code มาให้ classify_probe_error อ่านต่อ -- ต้องไม่ถูก
    # probe_summary_model กลืนเป็น "ผ่าน" เหมือน RuntimeError ธรรมดา
    class FakeHttpError(RuntimeError):
        def __init__(self):
            super().__init__("HTTP 401: invalid key")
            self.status_code = 401

    provider = MagicMock()
    provider.complete.side_effect = FakeHttpError()

    with patch("src.preflight.resolve", return_value=provider):
        result = check_api_key("sk-test", "GLM-5.2")

    assert result.status == "warn"
    assert "401" in result.detail


def test_check_api_key_reports_the_right_env_var_when_the_glm_key_is_missing():
    # api_no_key เดิมพูดถึง ANTHROPIC_API_KEY เสมอไม่ว่าโมเดลจะเป็นอะไร -- ผิดฝั่งทันที
    # ที่ GLM เป็นโมเดลที่ตั้งไว้
    result = check_api_key("", "GLM-5.2")

    assert result.status == "warn"
    assert "LLM_API_KEY" in result.detail
    assert "ANTHROPIC_API_KEY" not in result.detail


# --- check_summary_readiness: หนึ่งสาเหตุ หนึ่งรายงาน -----------------------------


def test_check_summary_readiness_skips_the_key_probe_when_the_model_is_unregistered():
    # โมเดลไม่อยู่ใน registry แปลว่าไม่มี provider ให้ยิงคำขอตรวจ key เลย -- ต้องได้
    # คำเตือนข้อเดียวจาก check_summary_model ไม่ใช่สองข้อสำหรับสาเหตุเดียวกัน
    results = check_summary_readiness("sk-test", "not-a-real-model")

    assert [r.code for r in results] == ["model_unresolvable"]


def test_check_summary_readiness_checks_the_key_when_the_model_resolves():
    provider = MagicMock()
    provider.complete.return_value = Completion(text="hi", truncated=False)

    with patch("src.preflight.resolve", return_value=provider):
        results = check_summary_readiness("sk-test", "claude-opus-5")

    assert [r.code for r in results] == ["model_ok", "api_ok"]


# --- read_summary_settings: ไม่รู้จัก provider ใด ๆ เป็นการเฉพาะ --------------------


def test_read_summary_settings_returns_an_empty_key_instead_of_raising(tmp_path, monkeypatch):
    # load_config โยน KeyError เมื่อไม่มี ANTHROPIC_API_KEY -- ซึ่งคือกรณีที่ต้องรายงาน ไม่ใช่พัง
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    api_key, model = read_summary_settings(base_dir=tmp_path)

    assert api_key == ""
    assert model == DEFAULT_SUMMARY_MODEL


def test_read_summary_settings_reads_the_configured_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")

    assert read_summary_settings(base_dir=tmp_path) == ("sk-ant-test", "claude-sonnet-5")


def test_read_summary_settings_reads_the_glm_key_when_glm_is_configured(tmp_path, monkeypatch):
    # จุดที่ทั้งบั๊กเดิมอยู่: เดิมโค้ดนี้อ่าน ANTHROPIC_API_KEY เสมอไม่ว่าโมเดลจะเป็นอะไร
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "GLM-5.2")

    assert read_summary_settings(base_dir=tmp_path) == ("llm-test-key", "GLM-5.2")


def test_read_summary_settings_returns_an_empty_key_for_an_unregistered_model(
    tmp_path, monkeypatch
):
    # ไม่มี provider ให้ถามว่า key_env ชื่ออะไร -- ปล่อยว่างไว้ ไม่ใช่เดาว่าเป็น Anthropic
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "not-a-real-model")

    api_key, model = read_summary_settings(base_dir=tmp_path)

    assert api_key == ""
    assert model == "not-a-real-model"


# --- สลับภาษา ------------------------------------------------------------


def test_every_check_carries_a_message_code(tmp_path):
    """ผลตรวจต้องพก "รหัส" ไปด้วย ไม่ใช่มีแต่ข้อความที่ render แล้ว

    ถ้าข้อไหนไม่มีรหัส ข้อนั้นจะแปลไม่ได้และค้างเป็นไทยอยู่ในรายงานภาษาอังกฤษ
    """
    results = [
        evaluate_mic(-15.0),
        evaluate_loopback(-20.0, "Speakers"),
        check_api_key("", "claude-opus-5"),
    ]

    assert all(r.code for r in results)
    assert all(r.name_code for r in results)


def test_evaluate_mic_renders_in_english():
    result = evaluate_mic(-15.0, "en")

    assert result.name == "Microphone"
    assert "normal speaking level" in result.detail
    assert "peak -15.0 dB" in result.detail


def test_evaluate_loopback_keeps_the_device_name_untranslated():
    # ชื่ออุปกรณ์เป็นของ Windows ไม่ใช่ข้อความของเรา -- ห้ามแตะ
    result = evaluate_loopback(-60.0, "Speakers (3- NX-S2)", "en")

    assert "Speakers (3- NX-S2)" in result.detail
    assert result.status == "warn"


def test_api_failures_render_in_english():
    class Unauthorized(Exception):
        status_code = 401

    result = classify_probe_error(Unauthorized("nope"), "claude-opus-5", "en")

    assert result.name == "Summary model API key"
    assert "401" in result.detail
    assert "expired" in result.detail


def test_format_report_translates_results_that_were_built_in_thai():
    """รายงานภาษาอังกฤษต้องแปลผลที่สร้างไว้เป็นไทยแล้วได้

    ผลตรวจถูกสร้างตอนวัดเสียง ส่วนภาษาถูกเลือกตอนพิมพ์รายงาน -- สองจังหวะนี้
    คนละเวลากัน
    """
    results = [evaluate_mic(-15.0), evaluate_mic(-60.0)]

    report = format_report(results, "en")

    assert "[ PASS ]" in report
    assert "[ FAIL ]" in report
    assert "Microphone" in report
    assert "Verdict: do not start recording yet" in report
    assert "ไมค์" not in report


def test_format_report_leaves_a_codeless_result_alone():
    # ผลที่ประกอบเองโดยไม่ผ่าน _result (เช่นในเทสต์เก่า) ต้องไม่ทำให้รายงานพัง
    report = format_report([CheckResult("ไมค์", "ok", "peak -15.0 dB")], "en")

    assert "ไมค์" in report
    assert "Verdict: ready to record" in report
