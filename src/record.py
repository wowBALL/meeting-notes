import argparse
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

from src.config import load_config
from src.messages import render
from src.segments import (  # noqa: F401 -- write_manifest re-exported for tests
    NoAudioRecorded,
    find_orphan_sessions,
    finish_session,
    part_filename,
    session_dir_for,
    sweep_unrecoverable_sessions,
    write_manifest,
)

ROTATE_SECONDS = 1800
DEVICE_POLL_SECONDS = 5.0
# How long to keep waiting for the block that pairs with one we already have. Two
# capture callbacks never fire in step, so the counterpart is usually already
# queued and this wait costs nothing -- it only has to cover the gap between them.
PAIR_GRACE_SECONDS = 0.05


def _ffmpeg_error_detail(error: Exception) -> str:
    stderr = getattr(error, "stderr", None)
    if not stderr:
        return str(error)
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return f"{error}\n{stderr[-2000:]}"


def to_mono(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return frames
    return frames.mean(axis=1)


def mix_recordings(mic_frames: np.ndarray, speaker_frames: np.ndarray) -> np.ndarray:
    length = max(len(mic_frames), len(speaker_frames))
    mic_padded = np.zeros(length, dtype=np.float32)
    speaker_padded = np.zeros(length, dtype=np.float32)
    mic_padded[: len(mic_frames)] = mic_frames
    speaker_padded[: len(speaker_frames)] = speaker_frames
    return (mic_padded * 0.5) + (speaker_padded * 0.5)


def build_output_filename(name: str | None, now: datetime) -> str:
    if name:
        return f"{name}-{now.strftime('%H-%M-%S')}.ogg"
    return f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.ogg"


def drain_next_block(block_queue: "queue.Queue[np.ndarray]", timeout: float) -> np.ndarray:
    try:
        return block_queue.get(timeout=timeout)
    except queue.Empty:
        return np.zeros(0, dtype=np.float32)


def make_callback(
    block_queue: "queue.Queue[np.ndarray]",
    channels: int,
    muted: threading.Event | None = None,
):
    """`muted` ถูกเช็คก่อนใส่คิวเสมอ ไม่ใช่ตอน mix -- เสียงที่ปิดไมค์ไว้จะไม่มีทาง
    ถูกเขียนลงดิสก์เลยแม้แต่บล็อกเดียว ไม่ใช่แค่ถูกทำให้เงียบทีหลัง ซึ่งสำคัญเพราะ
    นี่คือเสียงคนพูดจริงที่ผู้ใช้เลือกจะไม่ให้เข้าประชุม
    """

    def callback(in_data, frame_count, time_info, status):
        block = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
        mono = to_mono(block.copy())
        if muted is not None and muted.is_set():
            mono = np.zeros_like(mono)
        block_queue.put(mono)
        return (None, pyaudio.paContinue)

    return callback


def record_streams_to_session(
    mic_queue: "queue.Queue[np.ndarray]",
    speaker_queue: "queue.Queue[np.ndarray]",
    session_dir: Path,
    samplerate: int,
    stop_event: threading.Event,
    rotate_samples: int,
    on_part_closed=None,
    block_timeout: float = 0.5,
    flush_every: int = 60,
) -> list[str]:
    # Writes mixed audio into rotating part files and returns their names in order.
    # Rotating bounds how much a crash can cost and keeps every file far below the
    # 4GB RIFF ceiling; the parts are concatenated and encoded once, on stop.
    parts: list[str] = []
    handle = None
    samples_in_part = 0
    blocks_written = 0

    def open_next_part() -> None:
        nonlocal handle, samples_in_part
        name = part_filename(len(parts) + 1)
        parts.append(name)
        handle = sf.SoundFile(
            str(session_dir / name),
            mode="w",
            samplerate=samplerate,
            channels=1,
            subtype="FLOAT",
        )
        samples_in_part = 0

    def close_current_part() -> None:
        nonlocal handle
        try:
            handle.flush()
        finally:
            handle.close()
        handle = None
        if on_part_closed is not None:
            on_part_closed(list(parts))

    # Waiting block_timeout on EACH queue in turn is what made one dead stream
    # fatal. An open WASAPI loopback whose endpoint is idle can deliver no
    # callbacks at all -- measured on this machine, two of four loopback devices
    # deliver silence blocks and the other two deliver nothing -- so its queue
    # stays empty forever. The loop then burned a whole timeout per iteration on
    # that queue while the mic kept filling its own at full rate: the recording
    # ran at ~19% of real time, the mic queue grew without bound, and the exit
    # condition below (both queues empty) could never become true, so stopping
    # never finished.
    #
    # So the two streams are paired only while both are actually delivering. A
    # stream that has produced nothing for a whole block_timeout is treated as
    # dead and read without waiting. "In the last block_timeout" rather than "on
    # the previous iteration" on purpose: a live stream legitimately misses
    # individual iterations, and dropping the pairing wait for it would write each
    # stream's blocks on their own and double the length of the recording.
    last_delivery = {"mic": time.monotonic(), "speaker": time.monotonic()}

    def read_pair() -> tuple[np.ndarray, np.ndarray]:
        if stop_event.is_set():
            # Nothing can arrive any more -- run_recording closes both streams
            # before it waits for this drain -- so take what is queued and never
            # wait. Every iteration then removes at least one block, which is what
            # makes the drain guaranteed to finish rather than merely likely to.
            return (
                drain_next_block(mic_queue, 0.0),
                drain_next_block(speaker_queue, 0.0),
            )
        now = time.monotonic()
        mic_live = now - last_delivery["mic"] < block_timeout
        speaker_live = now - last_delivery["speaker"] < block_timeout
        # Someone has to wait, or a lull on both streams turns into a busy loop.
        # The mic takes that role unless it is the dead one.
        mic_wait = block_timeout if mic_live or not speaker_live else PAIR_GRACE_SECONDS
        mic_block = drain_next_block(mic_queue, mic_wait)
        speaker_block = drain_next_block(
            speaker_queue, PAIR_GRACE_SECONDS if speaker_live else 0.0
        )
        if len(mic_block):
            last_delivery["mic"] = time.monotonic()
        if len(speaker_block):
            last_delivery["speaker"] = time.monotonic()
        return mic_block, speaker_block

    open_next_part()
    try:
        while True:
            if stop_event.is_set() and mic_queue.empty() and speaker_queue.empty():
                break
            mic_block, speaker_block = read_pair()
            if len(mic_block) == 0 and len(speaker_block) == 0:
                continue
            mixed = mix_recordings(mic_block, speaker_block)
            handle.write(mixed)
            samples_in_part += len(mixed)
            blocks_written += 1
            if blocks_written % flush_every == 0:
                handle.flush()
            # checked only after a whole block is written, so no block is ever split
            if samples_in_part >= rotate_samples:
                close_current_part()
                open_next_part()
    finally:
        if handle is not None:
            # Rotation can land exactly on the final block, leaving a freshly opened
            # part with nothing in it. Drop that file instead of closing it, and do
            # NOT fire on_part_closed: the previous close already reported the true
            # part list, and firing again would report a part that no longer exists.
            trailing_empty = samples_in_part == 0 and len(parts) > 1
            try:
                handle.flush()
            finally:
                handle.close()
            handle = None
            if trailing_empty:
                (session_dir / parts[-1]).unlink()
                parts.pop()
            elif on_part_closed is not None:
                on_part_closed(list(parts))
    return parts


def _close_quietly(stream) -> None:
    # Closing a stream is now on the path to waiting for the recorder to drain, so
    # a failure here must not leave the other stream open: its callbacks would keep
    # refilling a queue the drain is waiting to see empty. PortAudio close errors
    # are not actionable and must not replace the recording's own result.
    for step in (stream.stop_stream, stream.close):
        try:
            step()
        except Exception:
            pass


def pyaudio_instance():
    return pyaudio.PyAudio()


def default_output_name() -> str:
    # A FRESH instance on purpose: PortAudio snapshots the device list when it
    # initializes, so the instance that is busy recording keeps reporting the
    # device that was default when the meeting started -- exactly the switch we
    # need to notice.
    audio = pyaudio_instance()
    try:
        wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        return audio.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
    finally:
        audio.terminate()


def watch_output_device(
    get_name,
    initial_name: str,
    stop_event,
    on_change,
    poll_seconds: float = DEVICE_POLL_SECONDS,
) -> None:
    # The loopback stream is bound to one device for the whole recording. If the
    # default output moves (headset connected, speaker unplugged), the far end of
    # the meeting stops being captured and nothing else would ever say so.
    last = initial_name
    while not stop_event.wait(poll_seconds):
        try:
            current = get_name()
        except Exception:
            # enumeration can throw for an instant while a device is switching;
            # one bad poll must not end the watch for the rest of the meeting
            continue
        if current != last:
            on_change(last, current)
            last = current


def get_wasapi_mic_device(p) -> dict:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    return p.get_device_info_by_index(wasapi_info["defaultInputDevice"])


def get_wasapi_loopback_device(p) -> dict:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speaker = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    if default_speaker["isLoopbackDevice"]:
        return default_speaker
    for loopback in p.get_loopback_device_info_generator():
        if default_speaker["name"] in loopback["name"]:
            return loopback
    raise RuntimeError(
        f"ไม่พบ loopback device ของลำโพง default ({default_speaker['name']})"
    )


def get_common_samplerate(mic_device: dict, speaker_device: dict) -> int:
    mic_rate = int(mic_device["defaultSampleRate"])
    speaker_rate = int(speaker_device["defaultSampleRate"])
    if mic_rate != speaker_rate:
        raise RuntimeError(
            f"sample rate ของไมค์ ({mic_rate} Hz) กับลำโพง ({speaker_rate} Hz) ไม่ตรงกัน "
            "กรุณาไปที่ mmsys.cpl > อุปกรณ์ > Advanced แล้วตั้งให้เท่ากัน"
        )
    return mic_rate


def _noop_event(code: str, params: dict | None = None, level: str = "info") -> None:
    pass


def finish_or_discard(session_dir: Path, inbox_dir: Path) -> Path | None:
    # A session that captured no audio at all (stopped before the first block was
    # written) can never be encoded. Keeping it would make every later startup try
    # to recover it and fail again, forever -- so discard it and report nothing.
    # A genuine encode failure is the opposite case: those parts hold real audio,
    # so the error propagates and the directory survives for the next attempt.
    # Returning None rather than reporting it here: the caller owns the output
    # channel, which is a console for the CLI and a web page for the UI.
    try:
        return finish_session(session_dir, inbox_dir)
    except NoAudioRecorded:
        shutil.rmtree(session_dir, ignore_errors=True)
        return None


def recover_orphan_sessions(inbox_dir: Path, emit=None) -> list[Path]:
    # A leftover session directory means a previous run died before encoding.
    # Finish what we can; one bad session must not stop the others or the new run.
    emit = emit or _noop_event
    for swept in sweep_unrecoverable_sessions(inbox_dir):
        emit("orphan_swept", {"name": swept.name})
    recovered = []
    for session_dir in find_orphan_sessions(inbox_dir):
        emit("orphan_found", {"name": session_dir.name})
        try:
            destination = finish_session(session_dir, inbox_dir)
        except Exception as e:
            emit(
                "orphan_failed",
                {"name": session_dir.name, "error": _ffmpeg_error_detail(e)},
                "error",
            )
            continue
        emit("orphan_recovered", {"path": str(destination)})
        recovered.append(session_dir)
    return recovered


def parse_args(argv: list[str]) -> tuple[str | None, str | None]:
    """(meeting name, chosen summary model) from the command line."""
    parser = argparse.ArgumentParser(prog="src.record")
    parser.add_argument("name", nargs="?", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(argv)
    # start-meeting.bat passes an empty string through when the user skips the
    # prompt, and an empty meeting name must behave exactly like no name at all.
    # เหตุผลเดียวกันกับ --profile: set /p ที่ถูกกด Enter ผ่านส่งสตริงว่างมา
    #
    # ไม่ validate ชื่อ profile ที่นี่โดยเจตนา เหมือนที่ไม่ validate ชื่อโมเดล --
    # ตัวอัดแค่ส่งต่อ ฝั่งสรุปเป็นคนตัดสินใจกับค่าที่ไม่รู้จัก (เตือนแล้วใช้ dev)
    # ถ้า validate ที่นี่ การพิมพ์ผิดในเมนูจะทำให้ประชุมอัดไม่ได้เลย
    return (args.name or None), args.model, (args.profile or None)


def run_recording(
    name: str | None,
    claude_model: str | None,
    config,
    stop_event: threading.Event,
    on_event=None,
    mic_muted: threading.Event | None = None,
    profile: str | None = None,
) -> Path | None:
    """อัดจนกว่า stop_event จะถูก set แล้วคืน path ของไฟล์ที่ encode แล้ว

    คืน None เมื่อไม่ได้ไฟล์ (ไม่มีเสียง, เปิดอุปกรณ์ไม่ได้, encode ล้ม) — ทุกกรณี
    รายงานผ่าน on_event แทนการ raise เพราะผู้เรียกคือ CLI หรือหน้าจอที่ต้องแสดงผล
    ต่อได้ ไม่ใช่ล้มทั้ง process

    stop_event รับมาจากผู้เรียก นั่นคือทั้งหมดที่ทำให้ปุ่มบนหน้าจอหยุดการอัดได้
    โดยไม่ต้องปลอมสัญญาณ Ctrl+C ข้าม process บน Windows

    mic_muted เป็น Event เดียวกัน: ผู้เรียกอยู่นอกฟังก์ชันนี้เป็นคนสั่ง set/clear
    ระหว่างอัด ไม่มี event ส่งมา (ทางเข้า CLI ที่ไม่มีปุ่ม) แปลว่าไม่มีทางถูกปิดไมค์
    เลยตลอดการอัด -- สร้าง Event สดที่ไม่มีใครแตะแทน ไม่ใช่ None ที่ทุกจุดต้องเช็ค
    """
    emit = on_event or _noop_event
    if mic_muted is None:
        mic_muted = threading.Event()

    config.inbox_dir.mkdir(parents=True, exist_ok=True)
    recover_orphan_sessions(config.inbox_dir, emit)

    p = pyaudio.PyAudio()
    recorder_thread = None
    thread_started = False
    try:
        try:
            mic_device = get_wasapi_mic_device(p)
            speaker_device = get_wasapi_loopback_device(p)
        except Exception as e:
            emit("device_open_failed", {"error": str(e)}, "error")
            return None

        try:
            samplerate = get_common_samplerate(mic_device, speaker_device)
        except RuntimeError as e:
            emit("samplerate_mismatch", {"error": str(e)}, "error")
            return None

        mic_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        speaker_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        now = datetime.now()
        stem = Path(build_output_filename(name, now)).stem
        session_dir = session_dir_for(config.inbox_dir, stem)

        try:
            mic_stream = p.open(
                format=pyaudio.paFloat32,
                channels=int(mic_device["maxInputChannels"]),
                rate=samplerate,
                input=True,
                input_device_index=mic_device["index"],
                frames_per_buffer=4096,
                stream_callback=make_callback(
                    mic_queue, int(mic_device["maxInputChannels"]), mic_muted
                ),
            )
            speaker_stream = p.open(
                format=pyaudio.paFloat32,
                channels=int(speaker_device["maxInputChannels"]),
                rate=samplerate,
                input=True,
                input_device_index=speaker_device["index"],
                frames_per_buffer=4096,
                stream_callback=make_callback(
                    speaker_queue, int(speaker_device["maxInputChannels"])
                ),
            )
        except Exception as e:
            emit("stream_open_failed", {"error": str(e)}, "error")
            return None

        # Only create the session directory (and let the watcher's orphan-recovery
        # machinery become aware of it) once both streams are confirmed open --
        # otherwise a stream-open failure above would leave an empty, permanently
        # unrecoverable .session-* directory behind (find_orphan_sessions requires
        # at least one part file, so nothing would ever clean it up).
        session_dir.mkdir(parents=True, exist_ok=True)
        devices = {"mic": mic_device["name"], "loopback": speaker_device["name"]}
        write_manifest(
            session_dir,
            stem,
            now.isoformat(),
            samplerate,
            [],
            "recording",
            devices,
            claude_model=claude_model,
            profile=profile,
        )

        def on_part_closed(parts: list[str]) -> None:
            # profile ต้องมาคู่กับ claude_model ทุกที่ที่เขียน manifest -- ไฟล์นี้ถูกเขียน
            # ทับทั้งไฟล์ ไม่ใช่ patch ทีละคีย์ คีย์ที่ไม่ส่งจึงกลายเป็น None (default ของ
            # write_manifest) แล้ว finish_session ที่อ่านต่อเขียน job.json โดยไม่มี profile
            # ฝั่งสรุปจึงตกไปใช้ค่าจาก .env ทุกครั้งไม่ว่าผู้ใช้จะเลือกอะไรไว้
            #
            # ทุกการอัดปิด part อย่างน้อยหนึ่งครั้งตอนหยุด จึงพลาดทุกประชุม ไม่ใช่เคสมุม
            # (วัดจาก activity.jsonl ของจริง 2026-07-30 12:51:29: part_closed ยิงก่อน
            # encode_started ทันที และ job.json ที่ได้มีแต่ claude_model)
            write_manifest(
                session_dir,
                stem,
                now.isoformat(),
                samplerate,
                parts,
                "recording",
                devices,
                claude_model=claude_model,
                profile=profile,
            )
            emit("part_closed", {"count": len(parts)})

        def warn_output_changed(old_name: str, new_name: str) -> None:
            emit("device_changed", {"old": old_name, "new": new_name}, "warn")

        # Daemon: an audio device probe must never be the reason a stop hangs.
        device_watch_thread = threading.Thread(
            target=watch_output_device,
            args=(
                default_output_name,
                default_output_name(),
                stop_event,
                warn_output_changed,
            ),
            daemon=True,
        )
        device_watch_thread.start()

        try:
            recorder_thread = threading.Thread(
                target=record_streams_to_session,
                args=(
                    mic_queue,
                    speaker_queue,
                    session_dir,
                    samplerate,
                    stop_event,
                    ROTATE_SECONDS * samplerate,
                    on_part_closed,
                ),
            )
            recorder_thread.start()
            thread_started = True

            emit(
                "room_opened",
                {"room": name or "", "model": claude_model or ""},
            )
            emit(
                "devices_selected",
                {"mic": mic_device["name"], "loopback": speaker_device["name"]},
            )
            emit("recording_started", {})
            # Wait for the stop request, NOT for the recorder to finish. The
            # recorder cannot finish until the capture callbacks stop pushing into
            # its queues, and closing the streams to make that happen is this
            # thread's job -- see the finally below. Waiting on the recorder here
            # was the deadlock: it can only end once the streams are closed, and
            # they were only closed once it ended.
            # is_alive() stays in the condition so a recorder that dies on its own
            # (a disk error mid-meeting) still releases this thread.
            while recorder_thread.is_alive() and not stop_event.wait(timeout=0.2):
                pass
        except KeyboardInterrupt:
            # Swallowed on purpose, exactly as before: Ctrl+C is a request to STOP
            # the meeting, not to abandon it. Letting it propagate would skip the
            # encode below and leave the audio as raw parts every single time.
            # The join belongs to the finally below, which closes the streams
            # first -- joining here would wait for a drain that is still being fed.
            stop_event.set()
        finally:
            stop_event.set()
            # Close the streams BEFORE waiting for the drain. The capture callbacks
            # keep pushing into the queues until the streams are closed, and the
            # recorder only finishes once both queues are empty, so waiting first
            # means waiting for something that is still being refilled.
            _close_quietly(mic_stream)
            _close_quietly(speaker_stream)
            if thread_started:
                recorder_thread.join()

        emit("encode_started", {})
        try:
            destination = finish_or_discard(session_dir, config.inbox_dir)
        except Exception as e:
            emit(
                "encode_failed",
                {"path": str(session_dir), "error": _ffmpeg_error_detail(e)},
                "error",
            )
            return None
        if destination is None:
            emit("no_audio", {}, "warn")
            return None
        emit("encode_done", {"path": str(destination)})
        return destination
    finally:
        stop_event.set()
        if thread_started:
            recorder_thread.join()
        p.terminate()


def main() -> None:
    """ทางเข้าแบบ CLI ที่ start-meeting.bat เรียก: Ctrl+C หยุด ข้อความออก stdout"""
    name, claude_model, profile = parse_args(sys.argv[1:])
    config = load_config()
    lang = config.ui_lang

    def emit(code, params=None, level="info"):
        print(render(code, params, lang))
        # The hint belongs right after the recorder actually starts, not before
        # the device lines -- it is the last thing the user reads before waiting.
        if code == "recording_started":
            print(render("press_ctrl_c", {}, lang))

    run_recording(name, claude_model, config, threading.Event(), emit, profile=profile)


if __name__ == "__main__":
    main()
