from datetime import datetime
import queue
import threading
from unittest.mock import patch

import numpy as np
import pyaudiowpatch as pyaudio
import pytest
import soundfile as sf

import src.record as record
from src.record import to_mono, mix_recordings, build_output_filename


def test_to_mono_returns_input_unchanged_when_already_mono():
    frames = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = to_mono(frames)

    assert np.array_equal(result, frames)


def test_to_mono_averages_stereo_channels():
    frames = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    result = to_mono(frames)

    assert np.allclose(result, [0.5, 0.5])


def test_mix_recordings_averages_equal_length_signals():
    mic = np.array([1.0, 1.0], dtype=np.float32)
    speaker = np.array([0.0, 0.0], dtype=np.float32)

    result = mix_recordings(mic, speaker)

    assert np.allclose(result, [0.5, 0.5])


def test_mix_recordings_pads_shorter_signal_with_silence():
    mic = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    speaker = np.array([1.0], dtype=np.float32)

    result = mix_recordings(mic, speaker)

    assert np.allclose(result, [1.0, 0.5, 0.5])


def test_build_output_filename_with_name():
    now = datetime(2026, 7, 22, 16, 30, 5)

    result = build_output_filename("weekly-standup", now)

    assert result == "weekly-standup-16-30-05.ogg"


def test_build_output_filename_without_name():
    now = datetime(2026, 7, 22, 16, 30, 5)

    result = build_output_filename(None, now)

    assert result == "2026-07-22_16-30-05.ogg"


def test_recover_orphan_sessions_finishes_each_leftover_session(tmp_path):
    finished = []

    with (
        patch("src.record.find_orphan_sessions", return_value=[tmp_path / "a", tmp_path / "b"]),
        patch("src.record.finish_session", side_effect=lambda d, inbox: finished.append(d) or d),
    ):
        recovered = record.recover_orphan_sessions(tmp_path)

    assert finished == [tmp_path / "a", tmp_path / "b"]
    assert recovered == [tmp_path / "a", tmp_path / "b"]


def test_recover_orphan_sessions_keeps_going_when_one_fails(tmp_path):
    def flaky(session_dir, inbox_dir):
        if session_dir.name == "a":
            raise RuntimeError("ffmpeg boom")
        return session_dir

    with (
        patch("src.record.find_orphan_sessions", return_value=[tmp_path / "a", tmp_path / "b"]),
        patch("src.record.finish_session", side_effect=flaky),
    ):
        recovered = record.recover_orphan_sessions(tmp_path)

    # the failure must not block recovery of the others, nor abort the new recording
    assert recovered == [tmp_path / "b"]


def test_finish_or_discard_returns_the_destination_on_success(tmp_path):
    with patch("src.record.finish_session", return_value=tmp_path / "meet1.ogg"):
        assert record.finish_or_discard(tmp_path / ".session-meet1", tmp_path) == (
            tmp_path / "meet1.ogg"
        )


def test_finish_or_discard_removes_a_session_that_recorded_no_audio(tmp_path):
    # Ctrl+C before the first block leaves a header-only part. Keeping that
    # directory means every later startup tries to recover it and fails forever.
    session_dir = tmp_path / ".session-silent"
    session_dir.mkdir()

    with patch("src.record.finish_session", side_effect=record.NoAudioRecorded("no audio")):
        assert record.finish_or_discard(session_dir, tmp_path) is None

    assert not session_dir.exists()


def test_finish_or_discard_propagates_a_real_encode_failure(tmp_path):
    # An ffmpeg failure is different: the parts are real and must be kept for the
    # next recovery pass, so the error has to reach the caller.
    session_dir = tmp_path / ".session-meet1"
    session_dir.mkdir()

    with patch("src.record.finish_session", side_effect=RuntimeError("ffmpeg boom")):
        with pytest.raises(RuntimeError, match="ffmpeg boom"):
            record.finish_or_discard(session_dir, tmp_path)

    assert session_dir.exists()


class _FakePyAudio:
    def __init__(self, host_api, devices, loopback_devices):
        self._host_api = host_api
        self._devices = devices
        self._loopback_devices = loopback_devices

    def get_host_api_info_by_type(self, api_type):
        return self._host_api

    def get_device_info_by_index(self, index):
        return self._devices[index]

    def get_loopback_device_info_generator(self):
        return iter(self._loopback_devices)


def test_get_wasapi_mic_device_returns_default_input():
    fake = _FakePyAudio(
        host_api={"defaultInputDevice": 5, "defaultOutputDevice": 3},
        devices={
            5: {
                "index": 5,
                "name": "Microphone (Realtek)",
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }
        },
        loopback_devices=[],
    )

    result = record.get_wasapi_mic_device(fake)

    assert result["name"] == "Microphone (Realtek)"


def test_get_wasapi_loopback_device_returns_output_device_when_already_loopback():
    fake = _FakePyAudio(
        host_api={"defaultInputDevice": 5, "defaultOutputDevice": 3},
        devices={
            3: {
                "index": 3,
                "name": "Speakers [Loopback]",
                "isLoopbackDevice": True,
                "maxInputChannels": 2,
                "defaultSampleRate": 48000.0,
            }
        },
        loopback_devices=[],
    )

    result = record.get_wasapi_loopback_device(fake)

    assert result["name"] == "Speakers [Loopback]"


def test_get_wasapi_loopback_device_searches_loopback_generator_by_name():
    fake = _FakePyAudio(
        host_api={"defaultInputDevice": 5, "defaultOutputDevice": 3},
        devices={
            3: {
                "index": 3,
                "name": "Speakers (Realtek)",
                "isLoopbackDevice": False,
                "maxInputChannels": 0,
                "defaultSampleRate": 48000.0,
            }
        },
        loopback_devices=[
            {"index": 20, "name": "Other Device [Loopback]"},
            {"index": 21, "name": "Speakers (Realtek) [Loopback]"},
        ],
    )

    result = record.get_wasapi_loopback_device(fake)

    assert result["index"] == 21


def test_get_wasapi_loopback_device_raises_when_not_found():
    fake = _FakePyAudio(
        host_api={"defaultInputDevice": 5, "defaultOutputDevice": 3},
        devices={
            3: {
                "index": 3,
                "name": "Speakers (Realtek)",
                "isLoopbackDevice": False,
                "maxInputChannels": 0,
                "defaultSampleRate": 48000.0,
            }
        },
        loopback_devices=[{"index": 20, "name": "Other Device [Loopback]"}],
    )

    with pytest.raises(RuntimeError, match="loopback"):
        record.get_wasapi_loopback_device(fake)


def test_get_common_samplerate_returns_shared_rate():
    result = record.get_common_samplerate(
        {"defaultSampleRate": 48000.0}, {"defaultSampleRate": 48000.0}
    )

    assert result == 48000


def test_get_common_samplerate_raises_when_rates_differ():
    with pytest.raises(RuntimeError, match="sample rate"):
        record.get_common_samplerate(
            {"defaultSampleRate": 44100.0}, {"defaultSampleRate": 48000.0}
        )


def test_drain_next_block_returns_item_when_available():
    q = queue.Queue()
    block = np.array([0.1, 0.2], dtype=np.float32)
    q.put(block)

    result = record.drain_next_block(q, timeout=0.1)

    assert np.array_equal(result, block)


def test_drain_next_block_returns_empty_array_on_timeout():
    q = queue.Queue()

    result = record.drain_next_block(q, timeout=0.05)

    assert len(result) == 0


def test_make_callback_converts_stereo_block_to_mono_and_queues_it():
    q = queue.Queue()
    callback = record.make_callback(q, channels=2)
    in_data = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32).tobytes()

    result, flag = callback(in_data, 2, {}, 0)

    assert result is None
    assert flag == pyaudio.paContinue
    queued = q.get_nowait()
    assert np.allclose(queued, [0.5, 0.5])


def _queue_with(*blocks):
    q: "queue.Queue[np.ndarray]" = queue.Queue()
    for block in blocks:
        q.put(block)
    return q


def test_record_streams_to_session_writes_a_single_part_when_short(tmp_path):
    stop_event = threading.Event()
    stop_event.set()
    block = np.array([0.5, 0.5], dtype=np.float32)

    parts = record.record_streams_to_session(
        _queue_with(block),
        _queue_with(block),
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=1000,
        block_timeout=0.01,
    )

    assert parts == ["part0001.wav"]
    written, _ = sf.read(str(tmp_path / "part0001.wav"), dtype="float32")
    assert len(written) == 2


def test_record_streams_to_session_rotates_after_the_sample_budget(tmp_path):
    stop_event = threading.Event()
    stop_event.set()
    block = np.ones(4, dtype=np.float32)

    parts = record.record_streams_to_session(
        _queue_with(block, block, block),
        _queue_with(block, block, block),
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=4,
        block_timeout=0.01,
    )

    # each 4-sample block fills the budget exactly, so every block gets its own part
    assert parts == ["part0001.wav", "part0002.wav", "part0003.wav"]
    for name in parts:
        written, _ = sf.read(str(tmp_path / name), dtype="float32")
        assert len(written) == 4


def test_record_streams_to_session_reports_each_closed_part(tmp_path):
    stop_event = threading.Event()
    stop_event.set()
    block = np.ones(4, dtype=np.float32)
    seen = []

    record.record_streams_to_session(
        _queue_with(block, block),
        _queue_with(block, block),
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=4,
        on_part_closed=lambda parts: seen.append(list(parts)),
        block_timeout=0.01,
    )

    assert seen == [["part0001.wav"], ["part0001.wav", "part0002.wav"]]


def test_record_streams_to_session_drops_a_trailing_empty_part(tmp_path):
    # rotation landing exactly on the last block must not leave a 0-sample file
    stop_event = threading.Event()
    stop_event.set()
    block = np.ones(4, dtype=np.float32)

    parts = record.record_streams_to_session(
        _queue_with(block),
        _queue_with(block),
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=4,
        block_timeout=0.01,
    )

    assert parts == ["part0001.wav"]
    assert not (tmp_path / "part0002.wav").exists()


def test_record_streams_to_session_drains_queues_after_stop_requested(tmp_path):
    stop_event = threading.Event()
    stop_event.set()
    block = np.ones(2, dtype=np.float32)

    parts = record.record_streams_to_session(
        _queue_with(block, block),
        _queue_with(block, block),
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=10_000,
        block_timeout=0.01,
    )

    written, _ = sf.read(str(tmp_path / parts[0]), dtype="float32")
    assert len(written) == 4
