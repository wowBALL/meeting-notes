import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.preflight as preflight
from src.preflight import (
    LOOPBACK_SILENT_DBFS,
    MIC_GOOD_DBFS,
    MIC_WEAK_DBFS,
    CheckResult,
    evaluate_loopback,
    evaluate_mic,
    format_report,
    peak_dbfs,
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
