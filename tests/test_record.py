import threading
import time as time_module
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
    """Fake recorder context manager. Yields real chunks first, then keeps
    returning empty (zero-length) arrays forever instead of raising or
    blocking, so a background _record_loop thread never errors out and never
    hangs on its own -- it only stops when the caller sets stop_event.

    If a `drained_event` is supplied, it is set right after the last real
    chunk has been popped, so a test can synchronize on "all real chunks
    have been consumed" instead of guessing with a fixed sleep."""

    def __init__(self, chunks, drained_event=None):
        self._chunks = list(chunks)
        self._drained_event = drained_event

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def record(self, numframes=None):
        if self._chunks:
            chunk = self._chunks.pop(0)
            if not self._chunks and self._drained_event is not None:
                self._drained_event.set()
            return chunk
        return np.zeros(0, dtype=np.float32)


class _FakeDevice:
    def __init__(self, chunks, drained_event=None):
        self._chunks = chunks
        self._drained_event = drained_event

    def recorder(self, samplerate=None, channels=None, blocksize=None, exclusive_mode=False):
        return _FakeRecorder(self._chunks, drained_event=self._drained_event)


class _FailingRecorder:
    def __enter__(self):
        raise RuntimeError("permission denied")

    def __exit__(self, *args):
        return False


class _FailingDevice:
    def recorder(self, samplerate=None, channels=None, blocksize=None, exclusive_mode=False):
        return _FailingRecorder()


def test_record_until_interrupted_returns_concatenated_frames_when_stopped_normally(monkeypatch):
    # Fake devices each hand out two small chunks, then run dry (empty
    # arrays) forever -- np.concatenate ignores the empty arrays, so the
    # assertion below is deterministic regardless of exactly how many extra
    # empty reads happen before the stop is noticed.
    mic_chunk_1 = np.array([1.0, 1.0], dtype=np.float32)
    mic_chunk_2 = np.array([2.0, 2.0], dtype=np.float32)
    speaker_chunk_1 = np.array([3.0, 3.0], dtype=np.float32)
    speaker_chunk_2 = np.array([4.0, 4.0], dtype=np.float32)

    mic_drained = threading.Event()
    speaker_drained = threading.Event()
    mic = _FakeDevice([mic_chunk_1, mic_chunk_2], drained_event=mic_drained)
    speaker = _FakeDevice([speaker_chunk_1, speaker_chunk_2], drained_event=speaker_drained)

    def fake_sleep_then_interrupt(seconds):
        # Wait for both background threads to signal that they've popped
        # their last real chunk (a generous but bounded timeout avoids ever
        # hanging forever if something is broken), then simulate the user
        # hitting Ctrl+C -- this is what the main polling loop's
        # `except KeyboardInterrupt` is designed to handle. This replaces a
        # fixed-duration sleep, which was a latent source of flakiness under
        # slow/loaded test runners.
        assert mic_drained.wait(timeout=2), "mic fake never drained its chunks"
        assert speaker_drained.wait(timeout=2), "speaker fake never drained its chunks"
        raise KeyboardInterrupt

    monkeypatch.setattr(record.time, "sleep", fake_sleep_then_interrupt)

    mic_frames, speaker_frames = record.record_until_interrupted(
        mic, speaker, samplerate=16000, blocksize=2
    )

    assert np.array_equal(mic_frames, np.concatenate([mic_chunk_1, mic_chunk_2]))
    assert np.array_equal(speaker_frames, np.concatenate([speaker_chunk_1, speaker_chunk_2]))


def test_record_until_interrupted_raises_when_a_device_fails(monkeypatch):
    # Regression test for the two bugs fixed during task review: a failing
    # recorder's exception must propagate out of record_until_interrupted
    # (not be swallowed by the thread), and the main loop must notice the
    # failure promptly via stop_event rather than hanging until Ctrl+C.
    mic = _FailingDevice()
    speaker = _FakeDevice([np.zeros(2, dtype=np.float32)])

    real_sleep = time_module.sleep

    def fast_sleep(seconds):
        real_sleep(0.01)

    monkeypatch.setattr(record.time, "sleep", fast_sleep)

    with pytest.raises(RuntimeError, match="permission denied"):
        record.record_until_interrupted(mic, speaker, samplerate=16000, blocksize=2)
