import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.preflight as preflight
from src.config import DEFAULT_CLAUDE_MODEL
from src.preflight import (
    LOOPBACK_SILENT_DBFS,
    MIC_GOOD_DBFS,
    MIC_WEAK_DBFS,
    CheckResult,
    check_api_key,
    evaluate_loopback,
    evaluate_mic,
    format_report,
    peak_dbfs,
    read_api_settings,
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


def test_run_preflight_measures_both_devices_and_returns_two_checks():
    mic_device = {"name": "Microphone (Realtek)", "maxInputChannels": 2, "index": 1}
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

    assert [r.name for r in results] == ["ไมค์", "ลำโพง (คู่สนทนา)"]
    assert results[0].status == "ok"  # -14 dB
    assert results[1].status == "warn"  # -60 dB


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
    result = check_api_key("sk-ant-test", "claude-opus-5", probe=lambda k, m: None)

    assert result.status == "ok"


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


def test_read_api_settings_returns_an_empty_key_instead_of_raising(tmp_path, monkeypatch):
    # load_config โยน KeyError เมื่อไม่มี ANTHROPIC_API_KEY -- ซึ่งคือกรณีที่ต้องรายงาน ไม่ใช่พัง
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    api_key, model = read_api_settings(base_dir=tmp_path)

    assert api_key == ""
    assert model == DEFAULT_CLAUDE_MODEL


def test_read_api_settings_reads_the_configured_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")

    assert read_api_settings(base_dir=tmp_path) == ("sk-ant-test", "claude-sonnet-5")
