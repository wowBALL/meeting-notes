from datetime import datetime

import numpy as np
import pytest

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

    assert result == "weekly-standup-16-30-05.wav"


def test_build_output_filename_without_name():
    now = datetime(2026, 7, 22, 16, 30, 5)

    result = build_output_filename(None, now)

    assert result == "2026-07-22_16-30-05.wav"


class _FakeRecorder:
    """Fake recorder context manager. Yields real chunks in order; once
    exhausted, the next record() call raises KeyboardInterrupt, simulating
    the user pressing Ctrl+C right as the stream runs dry."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def record(self, numframes=None):
        if not self._chunks:
            raise KeyboardInterrupt
        return self._chunks.pop(0)


class _FakeDevice:
    def __init__(self, chunks):
        self._chunks = chunks

    def recorder(self, samplerate=None, channels=None, blocksize=None, exclusive_mode=False):
        return _FakeRecorder(self._chunks)


class _FailingRecorder:
    def __enter__(self):
        raise RuntimeError("permission denied")

    def __exit__(self, *args):
        return False


class _FailingDevice:
    def recorder(self, samplerate=None, channels=None, blocksize=None, exclusive_mode=False):
        return _FailingRecorder()


def test_record_until_interrupted_returns_concatenated_frames_when_stopped_normally():
    mic_chunk_1 = np.array([1.0, 1.0], dtype=np.float32)
    mic_chunk_2 = np.array([2.0, 2.0], dtype=np.float32)
    speaker_chunk_1 = np.array([3.0, 3.0], dtype=np.float32)
    speaker_chunk_2 = np.array([4.0, 4.0], dtype=np.float32)

    mic = _FakeDevice([mic_chunk_1, mic_chunk_2])
    speaker = _FakeDevice([speaker_chunk_1, speaker_chunk_2])

    mic_frames, speaker_frames = record.record_until_interrupted(
        mic, speaker, samplerate=16000, blocksize=2
    )

    assert np.array_equal(mic_frames, np.concatenate([mic_chunk_1, mic_chunk_2]))
    assert np.array_equal(speaker_frames, np.concatenate([speaker_chunk_1, speaker_chunk_2]))


def test_record_until_interrupted_raises_when_a_device_fails():
    mic = _FailingDevice()
    speaker = _FakeDevice([np.zeros(2, dtype=np.float32)])

    with pytest.raises(RuntimeError, match="permission denied"):
        record.record_until_interrupted(mic, speaker, samplerate=16000, blocksize=2)


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
