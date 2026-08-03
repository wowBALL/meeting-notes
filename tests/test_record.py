from datetime import datetime
import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pyaudiowpatch as pyaudio
import pytest
import soundfile as sf

import src.record as record
from src.job import NO_SUMMARY_MODEL
from src.record import to_mono, mix_recordings, build_output_filename, parse_args


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


class _FakeStopEvent:
    """Lets a test drive the poll loop: is_set() stays False for `waits` polls."""

    def __init__(self, waits: int):
        self.remaining = waits

    def wait(self, _timeout):
        if self.remaining <= 0:
            return True
        self.remaining -= 1
        return False


def test_default_output_name_reads_from_a_fresh_audio_instance():
    # PortAudio snapshots the device list at initialization, so a long-lived
    # instance keeps reporting the device that was default when recording began.
    # Only a fresh instance sees a switch that happened mid-meeting.
    audio = MagicMock()
    audio.get_host_api_info_by_type.return_value = {"index": 0, "defaultOutputDevice": 7}
    audio.get_device_info_by_index.return_value = {"name": "Headphones (FreeBuds)"}

    with patch("src.record.pyaudio_instance", return_value=audio) as mock_new:
        assert record.default_output_name() == "Headphones (FreeBuds)"

    mock_new.assert_called_once()
    audio.terminate.assert_called_once()  # must not leak an instance every poll


def test_watch_output_device_reports_each_switch():
    names = iter(["Speakers (NX-S2)", "Headphones (FreeBuds)", "Headphones (FreeBuds)"])
    changes = []

    record.watch_output_device(
        get_name=lambda: next(names),
        initial_name="Speakers (NX-S2)",
        stop_event=_FakeStopEvent(3),
        on_change=lambda old, new: changes.append((old, new)),
        poll_seconds=0,
    )

    assert changes == [("Speakers (NX-S2)", "Headphones (FreeBuds)")]


def test_watch_output_device_stays_quiet_when_nothing_changes():
    changes = []

    record.watch_output_device(
        get_name=lambda: "Speakers (NX-S2)",
        initial_name="Speakers (NX-S2)",
        stop_event=_FakeStopEvent(3),
        on_change=lambda old, new: changes.append((old, new)),
        poll_seconds=0,
    )

    assert changes == []


def test_watch_output_device_survives_a_failed_probe():
    # a device switch can make enumeration throw for an instant; one bad poll
    # must not kill the watch for the rest of the meeting
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("device in flux")
        return "Headphones (FreeBuds)"

    changes = []
    record.watch_output_device(
        get_name=flaky,
        initial_name="Speakers (NX-S2)",
        stop_event=_FakeStopEvent(3),
        on_change=lambda old, new: changes.append((old, new)),
        poll_seconds=0,
    )

    assert changes == [("Speakers (NX-S2)", "Headphones (FreeBuds)")]


def test_recover_orphan_sessions_sweeps_manifest_less_debris_first(tmp_path):
    # Debris left when a locked part blocked cleanup: no manifest, so it can
    # never be finished -- startup must remove it, not report it forever.
    debris = tmp_path / ".session-debris"
    debris.mkdir()
    (debris / "part0001.wav").write_bytes(b"x" * 4096)

    with (
        patch("src.record.find_orphan_sessions", return_value=[]),
        patch("src.record.finish_session"),
    ):
        record.recover_orphan_sessions(tmp_path)

    assert not debris.exists()


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


def test_make_callback_passes_audio_through_when_not_muted():
    q = queue.Queue()
    callback = record.make_callback(q, channels=1, muted=threading.Event())
    in_data = np.array([0.8, -0.6], dtype=np.float32).tobytes()

    callback(in_data, 2, {}, 0)

    assert np.allclose(q.get_nowait(), [0.8, -0.6])


def test_make_callback_defaults_to_unmuted_when_no_event_is_given():
    q = queue.Queue()
    callback = record.make_callback(q, channels=1)
    in_data = np.array([0.3], dtype=np.float32).tobytes()

    callback(in_data, 1, {}, 0)

    assert np.allclose(q.get_nowait(), [0.3])


def test_make_callback_queues_silence_when_muted():
    # เช็คก่อนใส่คิว ไม่ใช่ตอน mix -- เสียงที่ปิดไมค์ไว้ต้องไม่มีทางถูกเขียนลงดิสก์
    # เลยแม้แต่บล็อกเดียว
    q = queue.Queue()
    muted = threading.Event()
    muted.set()
    callback = record.make_callback(q, channels=1, muted=muted)
    in_data = np.array([0.8, -0.6], dtype=np.float32).tobytes()

    callback(in_data, 2, {}, 0)

    assert np.array_equal(q.get_nowait(), np.zeros(2, dtype=np.float32))


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


def test_record_streams_to_session_does_not_wait_per_queue_once_stopped(tmp_path):
    """หยุดแล้วต้องระบายคิวที่ค้างจนหมดโดยไม่รอ timeout เป็นราย ๆ คิว

    ของจริงที่พังคือ: loopback ที่ไม่ส่ง callback เลยทำให้ speaker_queue ว่างตลอด
    แต่ละรอบจึงไปนอนรอ block_timeout เต็ม ๆ ที่คิวว่างนั้น ทั้งที่ mic มีของค้างอยู่
    เป็นพันบล็อก การระบายจึงกินเวลา (จำนวนบล็อก x block_timeout) ซึ่งกลายเป็น
    "กดปิดแล้วไม่ปิด" ในสายตาผู้ใช้
    """
    stop_event = threading.Event()
    stop_event.set()
    block = np.ones(2, dtype=np.float32)
    backlog = 20

    started = time.monotonic()
    parts = record.record_streams_to_session(
        _queue_with(*[block] * backlog),
        queue.Queue(),  # loopback ที่ไม่เคยส่งอะไรมาเลย
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=10_000,
        block_timeout=0.2,
    )
    elapsed = time.monotonic() - started

    written, _ = sf.read(str(tmp_path / parts[0]), dtype="float32")
    assert len(written) == backlog * 2, "ต้องไม่ทิ้งเสียงที่ค้างอยู่ในคิว"
    # โค้ดเดิมใช้เวลา backlog x block_timeout = 4 วินาที เผื่อไว้กว้าง ๆ ให้
    # เทสต์ไม่ flaky บนเครื่องที่โหลดหนัก แต่ยังห่างจาก 4 วินาทีอยู่หลายเท่า
    assert elapsed < backlog * 0.2 / 4, f"ระบายคิวช้าเกินไป: {elapsed:.2f}s"


def test_record_streams_to_session_keeps_up_when_one_stream_is_dead(tmp_path):
    """สตรีมที่เปิดอยู่แต่ไม่ส่งอะไรเลย ต้องไม่ฉุดให้ตัวเขียนตามเสียงจริงไม่ทัน

    ถ้าตัวเขียนช้ากว่าที่ callback ป้อนเข้ามา คิวจะโตไม่มีเพดาน เสียงที่เขียนลงดิสก์
    จะตามหลังเวลาจริงเรื่อย ๆ และถ้า process ตายไปก่อนหยุด ส่วนที่ค้างในแรมหายหมด
    """
    stop_event = threading.Event()
    mic_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    block = np.ones(2, dtype=np.float32)
    produced = 40
    interval = 0.01  # ป้อนเข้ามาเรื่อย ๆ เหมือน callback ของไมค์

    def feed_mic():
        for _ in range(produced):
            mic_queue.put(block)
            time.sleep(interval)
        stop_event.set()

    feeder = threading.Thread(target=feed_mic)
    started = time.monotonic()
    feeder.start()
    parts = record.record_streams_to_session(
        mic_queue,
        queue.Queue(),  # loopback ที่ตายอยู่
        tmp_path,
        samplerate=16000,
        stop_event=stop_event,
        rotate_samples=10_000,
        block_timeout=0.2,
    )
    elapsed = time.monotonic() - started
    feeder.join()

    written, _ = sf.read(str(tmp_path / parts[0]), dtype="float32")
    assert len(written) == produced * 2, "เขียนไม่ครบ แปลว่าตัวเขียนตามไม่ทัน"
    # ป้อนเข้ามาทั้งหมด produced x interval = 0.4 วินาที ถ้าตัวเขียนตามทันก็ต้องจบ
    # ไม่ห่างจากนั้นมาก โค้ดเดิมเขียนได้รอบละหนึ่งบล็อกต่อ block_timeout จึงใช้
    # produced x 0.2 = 8 วินาที เพดาน 2 วินาทีนี้จับความต่างนั้นได้โดยไม่ flaky
    assert elapsed < 2.0, f"ตัวเขียนตามเสียงจริงไม่ทัน: ใช้ไป {elapsed:.2f}s"


def test_run_recording_closes_the_streams_before_it_waits_for_the_drain(
    tmp_path, monkeypatch
):
    """ต้องปิดสตรีมก่อน แล้วจึงรอตัวอัดระบายคิว ไม่ใช่ลำดับกลับกัน

    ถ้ารอตัวอัดก่อนปิดสตรีม callback ยังป้อนคิวอยู่ตลอดเวลาที่รอ เงื่อนไขจบของ
    ตัวอัด (คิวว่างทั้งสองข้าง) จึงอาจไม่เป็นจริงเลย -- การกดปิดจะค้างถาวร
    """
    calls = _fake_audio(monkeypatch)
    config = _config(tmp_path)
    mic_stopped = threading.Event()
    outcome = {}

    original_open = record.pyaudio.PyAudio

    def audio_that_signals_on_stop():
        instance = original_open()
        real_open = instance.open

        def open_and_wrap(**kwargs):
            stream = real_open(**kwargs)
            real_stop = stream.stop_stream

            def stop_and_signal():
                real_stop()
                mic_stopped.set()

            stream.stop_stream = stop_and_signal
            return stream

        instance.open = open_and_wrap
        return instance

    monkeypatch.setattr(record.pyaudio, "PyAudio", audio_that_signals_on_stop)

    def fake_record(*a, **k):
        # ยืนอยู่จนกว่าสตรีมจะถูกปิด เลียนแบบของจริงที่ระบายคิวไม่จบตราบใดที่
        # callback ยังป้อนเข้ามา ถ้าผู้เรียก join ก่อนปิดสตรีม อันนี้จะไม่มีวันหลุด
        outcome["streams_closed_first"] = mic_stopped.wait(timeout=3.0)
        return ["part0001.wav"]

    monkeypatch.setattr(record, "record_streams_to_session", fake_record)
    monkeypatch.setattr(
        record, "finish_or_discard", lambda s, i: config.inbox_dir / "meet.ogg"
    )

    stop_event = threading.Event()
    stop_event.set()
    record.run_recording("meet", "claude-opus-5", config, stop_event)

    assert outcome["streams_closed_first"] is True, (
        "ตัวอัดถูกรอจนจบก่อนสตรีมจะถูกปิด -- นี่คือลำดับที่ทำให้กดปิดแล้วค้าง"
    )
    assert [s.closed for s in calls["streams"]] == [True, True]


def test_run_recording_keeps_the_profile_when_a_part_closes(tmp_path, monkeypatch):
    """ปิด part แล้ว manifest ต้องยังจดประเภทประชุมไว้ ไม่ใช่ถูกล้างเป็น None

    on_part_closed เขียน manifest ทับทั้งไฟล์ ถ้าไม่ส่ง profile ไปด้วย ค่าที่เขียนไว้
    ตอนเริ่มอัดจะหายไปเงียบๆ (write_manifest มี default เป็น None) แล้ว finish_session
    ที่อ่านไฟล์นี้ต่อจะเขียน job.json โดยไม่มี profile -- ฝั่งสรุปจึงตกไปใช้ค่าจาก .env
    ทุกครั้ง ไม่ว่าผู้ใช้จะเลือกอะไรไว้

    ทุกการอัดปิด part อย่างน้อยหนึ่งครั้งตอนหยุด จึงไม่ใช่เคสมุม -- วัดจากของจริง
    (activity.jsonl 2026-07-30 12:51:29) part_closed ยิงก่อน encode_started ทันที
    """
    _fake_audio(monkeypatch)
    config = _config(tmp_path)

    def fake_record(*a, **k):
        # เลียนแบบของจริง: part แรกปิดก่อนตัวอัดจะคืนค่า
        # on_part_closed ถูกส่งเป็น positional ตัวที่ 7 (ดู args= ของ recorder_thread)
        on_part_closed = k.get("on_part_closed") or (a[6] if len(a) > 6 else None)
        assert on_part_closed is not None, (
            "ไม่ได้รับ on_part_closed -- เทสนี้จะไม่ได้พิสูจน์อะไรเลย"
        )
        on_part_closed(["part0001.wav"])
        return ["part0001.wav"]

    monkeypatch.setattr(record, "record_streams_to_session", fake_record)
    # กันไม่ให้ finish_session กินโฟลเดอร์ทิ้งก่อนอ่าน manifest
    monkeypatch.setattr(
        record, "finish_or_discard", lambda s, i: config.inbox_dir / "meet.ogg"
    )

    stop_event = threading.Event()
    stop_event.set()
    record.run_recording(
        "meet", "claude-opus-5", config, stop_event, profile="cross"
    )

    from src.segments import read_manifest

    # stem มีเวลาต่อท้าย (build_output_filename) จึงหาโฟลเดอร์จาก prefix ไม่ใช่เดาชื่อ
    sessions = list(config.inbox_dir.glob(".session-*"))
    assert len(sessions) == 1, f"คาดว่าจะมี session เดียว ได้ {sessions}"
    manifest = read_manifest(sessions[0])
    assert manifest["profile"] == "cross"
    # claude_model ผ่านมาได้อยู่แล้ว -- assert คู่กันไว้เพื่อให้เห็นว่าเทสนี้จับ
    # การหายของ profile จริง ไม่ใช่ manifest ที่ว่างทั้งไฟล์
    assert manifest["claude_model"] == "claude-opus-5"


def test_run_recording_writes_the_asr_engine_into_the_manifest(
    tmp_path, monkeypatch
):
    """asr_engine เดินทางเข้า manifest แบบเดียวกับ profile -- กันไม่ให้หายเงียบๆ
    เหมือนที่ profile เคยหายไปก่อนมีเทสต์คู่นี้ (ดู test ด้านบน)"""
    config = _config(tmp_path)

    def fake_record(*a, **k):
        on_part_closed = k.get("on_part_closed") or (a[6] if len(a) > 6 else None)
        assert on_part_closed is not None
        on_part_closed(["part0001.wav"])
        return ["part0001.wav"]

    monkeypatch.setattr(record, "record_streams_to_session", fake_record)
    monkeypatch.setattr(
        record, "finish_or_discard", lambda s, i: config.inbox_dir / "meet.ogg"
    )

    stop_event = threading.Event()
    stop_event.set()
    record.run_recording(
        "meet", "claude-opus-5", config, stop_event, asr_engine="typhoon"
    )

    from src.segments import read_manifest

    sessions = list(config.inbox_dir.glob(".session-*"))
    assert len(sessions) == 1, f"คาดว่าจะมี session เดียว ได้ {sessions}"
    manifest = read_manifest(sessions[0])
    assert manifest["asr_engine"] == "typhoon"


def test_parse_args_reads_the_name_and_the_model():
    assert parse_args(["weekly-standup", "--model", "claude-sonnet-5"]) == (
        "weekly-standup",
        "claude-sonnet-5",
        None,
        None,
    )


def test_parse_args_allows_a_model_with_no_name():
    assert parse_args(["--model", "claude-opus-5"]) == (
        None,
        "claude-opus-5",
        None,
        None,
    )


def test_parse_args_defaults_all_to_none():
    assert parse_args([]) == (None, None, None, None)


def test_parse_args_passes_the_transcript_only_sentinel_through_untouched():
    # ตัวอัดไม่ควรรู้จักโหมดนี้เลย -- มันแค่ส่งต่อสิ่งที่ .bat ให้มา เทสต์นี้จับกรณี
    # ที่มีคนเผลอไปเพิ่ม validation ชื่อโมเดลใน record แล้วโหมดนี้ตายเงียบ
    assert parse_args(["--model", NO_SUMMARY_MODEL]) == (
        None,
        NO_SUMMARY_MODEL,
        None,
        None,
    )


def test_parse_args_reads_the_meeting_profile():
    assert parse_args(["--model", "GLM-5.2", "--profile", "cross"]) == (
        None,
        "GLM-5.2",
        "cross",
        None,
    )


def test_parse_args_passes_an_unknown_profile_through_untouched():
    """ตัวอัดไม่ validate ชื่อ profile -- มันแค่ส่งต่อสิ่งที่ .bat ให้มา แบบเดียวกับ
    ชื่อโมเดล คนที่ตัดสินใจว่าจะทำอะไรกับค่าที่ไม่รู้จักคือฝั่งสรุป และมันเตือนแล้วใช้
    dev ต่อ การ validate ที่นี่จะทำให้ประชุมอัดไม่ได้เพราะพิมพ์ผิดในเมนู"""
    assert parse_args(["--profile", "พิมพ์ผิด"]) == (None, None, "พิมพ์ผิด", None)


def test_an_empty_profile_string_behaves_like_no_profile():
    """.bat ส่งสตริงว่างมาได้เมื่อ set /p ถูกกด Enter ผ่าน -- แบบเดียวกับชื่อประชุม"""
    assert parse_args(["--profile", ""]) == (None, None, None, None)


def test_parse_args_treats_an_empty_name_as_no_name():
    # start-meeting.bat passes "" through when the user skips the name prompt
    assert parse_args([""]) == (None, None, None, None)


def test_parse_args_accepts_a_name_starting_with_a_dash_after_the_separator():
    # Without "--", argparse takes "-standup" for an unrecognized option and
    # exits with an error instead of recording. start-meeting.bat relies on the
    # "--" end-of-options separator to keep such names working.
    assert parse_args(["--model", "claude-opus-5", "--", "-standup"]) == (
        "-standup",
        "claude-opus-5",
        None,
        None,
    )


def test_parse_args_reads_the_asr_engine():
    assert parse_args(["--asr-engine", "typhoon"]) == (None, None, None, "typhoon")


def test_an_empty_asr_engine_string_behaves_like_none():
    assert parse_args(["--asr-engine", ""]) == (None, None, None, None)


# --- orchestration ของ run_recording -------------------------------------
# main() ไม่เคยมีเทสต์คุมมาก่อน เทสต์ชุดนี้จึงเป็นนั่งร้านที่ต้องขึ้นก่อนแยก
# ฟังก์ชันออกมา ไม่ใช่หลังจากนั้น


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _fake_audio(monkeypatch):
    """แทนที่ทุกอย่างที่แตะฮาร์ดแวร์เสียง คืน dict ของสิ่งที่ถูกเรียก"""
    calls = {"streams": [], "terminated": False}

    mic = {
        "index": 1,
        "name": "FakeMic",
        "maxInputChannels": 1,
        "defaultSampleRate": 48000.0,
    }
    speaker = {
        "index": 2,
        "name": "FakeLoopback",
        "maxInputChannels": 1,
        "defaultSampleRate": 48000.0,
        "isLoopbackDevice": True,
    }

    class FakePyAudio:
        def open(self, **kwargs):
            stream = FakeStream()
            calls["streams"].append(stream)
            return stream

        def terminate(self):
            calls["terminated"] = True

    monkeypatch.setattr(record.pyaudio, "PyAudio", lambda: FakePyAudio())
    monkeypatch.setattr(record, "get_wasapi_mic_device", lambda p: mic)
    monkeypatch.setattr(record, "get_wasapi_loopback_device", lambda p: speaker)
    monkeypatch.setattr(record, "default_output_name", lambda: "FakeLoopback")
    return calls


def _config(tmp_path):
    from src.config import Config

    inbox = tmp_path / "inbox"
    inbox.mkdir(exist_ok=True)
    return Config(
        base_dir=tmp_path,
        inbox_dir=inbox,
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        hf_token="h",
    )


def test_run_recording_stops_when_the_event_is_set(tmp_path, monkeypatch):
    calls = _fake_audio(monkeypatch)
    config = _config(tmp_path)
    expected = config.inbox_dir / "meet.ogg"

    # ตัวอัดจริงถูกแทนที่: เทสต์นี้ตรวจลำดับการประกอบ ไม่ใช่การเขียนไฟล์เสียง
    monkeypatch.setattr(
        record, "record_streams_to_session", lambda *a, **k: ["part0001.wav"]
    )
    monkeypatch.setattr(record, "finish_or_discard", lambda s, i: expected)

    stop_event = threading.Event()
    stop_event.set()
    events = []
    result = record.run_recording(
        "meet",
        "claude-opus-5",
        config,
        stop_event,
        on_event=lambda code, params=None, level="info": events.append(code),
    )

    assert result == expected
    assert [s.closed for s in calls["streams"]] == [True, True]
    assert calls["terminated"] is True
    assert "encode_done" in events


def test_run_recording_wires_mic_muted_into_the_mic_stream_only(tmp_path, monkeypatch):
    """ปิดไมค์ต้องไม่แตะเสียงคู่สนทนาเลย -- ถ้า Event เดียวกันหลุดไปโดนลำโพงด้วย
    การปิดไมค์หนึ่งครั้งจะทำให้ครึ่งการประชุมหายไปเงียบ ๆ โดยไม่มีใครสังเกตจนจบ
    """
    calls = _fake_audio(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        record, "record_streams_to_session", lambda *a, **k: ["part0001.wav"]
    )
    monkeypatch.setattr(record, "finish_or_discard", lambda s, i: tmp_path / "meet.ogg")

    seen_muted = []
    real_make_callback = record.make_callback

    def spy(block_queue, channels, muted=None):
        seen_muted.append(muted)
        return real_make_callback(block_queue, channels, muted)

    monkeypatch.setattr(record, "make_callback", spy)

    stop_event = threading.Event()
    stop_event.set()
    mic_muted = threading.Event()

    record.run_recording("meet", "claude-opus-5", config, stop_event, mic_muted=mic_muted)

    # mic stream is opened before the speaker/loopback stream
    assert seen_muted[0] is mic_muted
    assert seen_muted[1] is not mic_muted


def test_run_recording_writes_the_manifest_before_recording_starts(
    tmp_path, monkeypatch
):
    """manifest ต้องมีอยู่ตอน record_streams_to_session เริ่มทำงาน

    ลำดับนี้คือสิ่งที่ทำให้ session ที่ค้างจากเครื่องดับกู้กลับมาได้ ถ้า manifest
    ถูกเขียนทีหลัง เสียงที่อัดไว้จะกลายเป็นโฟลเดอร์ที่ไม่มีใครกู้ได้
    """
    _fake_audio(monkeypatch)
    config = _config(tmp_path)
    seen = {}

    def fake_record(*args, **kwargs):
        session_dir = args[2]
        seen["manifest_exists"] = (session_dir / "session.json").exists()
        seen["manifest"] = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        return ["part0001.wav"]

    monkeypatch.setattr(record, "record_streams_to_session", fake_record)
    monkeypatch.setattr(
        record, "finish_or_discard", lambda s, i: config.inbox_dir / "meet.ogg"
    )

    stop_event = threading.Event()
    stop_event.set()
    record.run_recording("meet", NO_SUMMARY_MODEL, config, stop_event)

    assert seen["manifest_exists"] is True
    assert seen["manifest"]["claude_model"] == NO_SUMMARY_MODEL
    assert seen["manifest"]["status"] == "recording"
    assert seen["manifest"]["devices"] == {
        "mic": "FakeMic",
        "loopback": "FakeLoopback",
    }


def test_run_recording_reports_no_audio_and_returns_none(tmp_path, monkeypatch):
    _fake_audio(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(record, "record_streams_to_session", lambda *a, **k: [])
    monkeypatch.setattr(record, "finish_or_discard", lambda s, i: None)

    stop_event = threading.Event()
    stop_event.set()
    events = []
    result = record.run_recording(
        None,
        "claude-opus-5",
        config,
        stop_event,
        on_event=lambda code, params=None, level="info": events.append(code),
    )

    assert result is None
    assert "no_audio" in events
    assert "encode_done" not in events


def test_run_recording_reports_a_device_open_failure_without_raising(
    tmp_path, monkeypatch
):
    _fake_audio(monkeypatch)

    def boom(p):
        raise RuntimeError("ไม่มีไมค์")

    monkeypatch.setattr(record, "get_wasapi_mic_device", boom)
    config = _config(tmp_path)
    events = []
    result = record.run_recording(
        "meet",
        "claude-opus-5",
        config,
        threading.Event(),
        on_event=lambda code, params=None, level="info": events.append(code),
    )

    assert result is None
    assert "device_open_failed" in events


def test_run_recording_reports_an_encode_failure_and_keeps_the_session(
    tmp_path, monkeypatch
):
    """encode ที่ล้มต้องไม่ทิ้งชิ้นส่วน -- มันคือเสียงประชุมที่อัดซ้ำไม่ได้"""
    _fake_audio(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        record, "record_streams_to_session", lambda *a, **k: ["part0001.wav"]
    )

    def boom(session_dir, inbox_dir):
        raise RuntimeError("ffmpeg พัง")

    monkeypatch.setattr(record, "finish_or_discard", boom)

    stop_event = threading.Event()
    stop_event.set()
    events = []
    result = record.run_recording(
        "meet",
        "claude-opus-5",
        config,
        stop_event,
        on_event=lambda code, params=None, level="info": events.append(code),
    )

    assert result is None
    assert "encode_failed" in events
    assert list(config.inbox_dir.glob(".session-*"))
