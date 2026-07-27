import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.preflight as preflight
from src.config import DEFAULT_SUMMARY_MODEL
from src.preflight import (
    LOOPBACK_SILENT_DBFS,
    MIC_GOOD_DBFS,
    MIC_WEAK_DBFS,
    CheckResult,
    check_summary_model,
    evaluate_loopback,
    evaluate_mic,
    evaluate_samplerate,
    format_report,
    peak_dbfs,
    read_summary_model,
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


# --- check_summary_model: ไม่มีการเรียก API จริง ----------------------------


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
    # โมเดลไม่อยู่ใน registry ไม่ได้ทำให้อัดหรือถอดเสียงไม่ได้ -- transcript ยังออกมา
    # ครบ เอาไปสรุปด้วยมือทีหลังได้
    assert check_summary_model("not-a-real-model").status != "fail"


# --- read_summary_model: อ่านค่าจาก .env ตรง ๆ ไม่ตรวจสอบอะไรเพิ่ม -----------------


def test_read_summary_model_returns_the_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    assert read_summary_model(base_dir=tmp_path) == DEFAULT_SUMMARY_MODEL


def test_read_summary_model_reads_the_configured_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")

    assert read_summary_model(base_dir=tmp_path) == "claude-sonnet-5"


def test_read_summary_model_does_not_validate_against_the_registry(tmp_path, monkeypatch):
    # การตรวจว่าโมเดลอยู่ใน registry ไหมเป็นหน้าที่ของ check_summary_model แยกต่างหาก --
    # read_summary_model แค่อ่านค่าดิบจาก .env เท่านั้น
    monkeypatch.setenv("CLAUDE_MODEL", "not-a-real-model")

    assert read_summary_model(base_dir=tmp_path) == "not-a-real-model"


# --- สลับภาษา ------------------------------------------------------------


def test_every_check_carries_a_message_code():
    """ผลตรวจต้องพก "รหัส" ไปด้วย ไม่ใช่มีแต่ข้อความที่ render แล้ว

    ถ้าข้อไหนไม่มีรหัส ข้อนั้นจะแปลไม่ได้และค้างเป็นไทยอยู่ในรายงานภาษาอังกฤษ
    """
    results = [
        evaluate_mic(-15.0),
        evaluate_loopback(-20.0, "Speakers"),
        check_summary_model("claude-opus-5"),
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


def test_check_summary_model_renders_in_english():
    result = check_summary_model("claude-opus-4-8", "en")

    assert result.name == "Summary model"
    assert "unknown model" in result.detail
    assert "claude-opus-4-8" in result.detail


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
