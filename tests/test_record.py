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
