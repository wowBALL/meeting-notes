import queue
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

from src.config import load_config
from src.segments import (
    NoAudioRecorded,
    find_orphan_sessions,
    finish_session,
    part_filename,
    session_dir_for,
    write_manifest,
)

ROTATE_SECONDS = 1800


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


def make_callback(block_queue: "queue.Queue[np.ndarray]", channels: int):
    def callback(in_data, frame_count, time_info, status):
        block = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
        block_queue.put(to_mono(block.copy()))
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

    open_next_part()
    try:
        while not stop_event.is_set() or not mic_queue.empty() or not speaker_queue.empty():
            mic_block = drain_next_block(mic_queue, block_timeout)
            speaker_block = drain_next_block(speaker_queue, block_timeout)
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


def finish_or_discard(session_dir: Path, inbox_dir: Path) -> Path | None:
    # A session that captured no audio at all (stopped before the first block was
    # written) can never be encoded. Keeping it would make every later startup try
    # to recover it and fail again, forever -- so discard it and report nothing.
    # A genuine encode failure is the opposite case: those parts hold real audio,
    # so the error propagates and the directory survives for the next attempt.
    try:
        return finish_session(session_dir, inbox_dir)
    except NoAudioRecorded:
        shutil.rmtree(session_dir, ignore_errors=True)
        print("ไม่มีเสียงถูกบันทึกในรอบนี้ (หยุดเร็วเกินไปหรือไมค์ไม่ส่งเสียงมา) ไม่ได้สร้างไฟล์")
        return None


def recover_orphan_sessions(inbox_dir: Path) -> list[Path]:
    # A leftover session directory means a previous run died before encoding.
    # Finish what we can; one bad session must not stop the others or the new run.
    recovered = []
    for session_dir in find_orphan_sessions(inbox_dir):
        print(f"พบการอัดค้างจากรอบก่อน กำลังกู้ {session_dir.name} ...")
        try:
            destination = finish_session(session_dir, inbox_dir)
        except Exception as e:
            print(
                f"กู้ {session_dir.name} ไม่สำเร็จ ข้ามไปก่อน (ชิ้นส่วนยังอยู่): "
                f"{_ffmpeg_error_detail(e)}"
            )
            continue
        print(f"กู้สำเร็จ: {destination}")
        recovered.append(session_dir)
    return recovered


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else None

    config = load_config()
    config.inbox_dir.mkdir(parents=True, exist_ok=True)
    recover_orphan_sessions(config.inbox_dir)

    p = pyaudio.PyAudio()
    recorder_thread = None
    thread_started = False
    stop_event = threading.Event()
    try:
        try:
            mic_device = get_wasapi_mic_device(p)
            speaker_device = get_wasapi_loopback_device(p)
        except Exception as e:
            print(f"ไม่พบไมค์/ลำโพง default กรุณาตรวจสอบการตั้งค่าเสียงของ Windows: {e}")
            return

        try:
            samplerate = get_common_samplerate(mic_device, speaker_device)
        except RuntimeError as e:
            print(str(e))
            return

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
                stream_callback=make_callback(mic_queue, int(mic_device["maxInputChannels"])),
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
            print(
                "อัดเสียงไม่สำเร็จ (อาจไม่มีสิทธิ์เข้าถึงไมค์ - ตรวจสอบที่ "
                f"Settings > Privacy > Microphone): {e}"
            )
            return

        # Only create the session directory (and let the watcher's orphan-recovery
        # machinery become aware of it) once both streams are confirmed open --
        # otherwise a stream-open failure above would leave an empty, permanently
        # unrecoverable .session-* directory behind (find_orphan_sessions requires
        # at least one part file, so nothing would ever clean it up).
        session_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(session_dir, stem, now.isoformat(), samplerate, [], "recording")

        def on_part_closed(parts: list[str]) -> None:
            write_manifest(
                session_dir, stem, now.isoformat(), samplerate, parts, "recording"
            )

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

            print("กำลังอัดเสียง... กด Ctrl+C เพื่อหยุด")
            while recorder_thread.is_alive():
                recorder_thread.join(timeout=0.2)
        except KeyboardInterrupt:
            stop_event.set()
            if thread_started:
                recorder_thread.join()
        finally:
            stop_event.set()
            if thread_started:
                recorder_thread.join()
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            speaker_stream.close()

        print("กำลังบีบอัดไฟล์เสียง...")
        try:
            destination = finish_or_discard(session_dir, config.inbox_dir)
        except Exception as e:
            print(
                f"บีบอัดไม่สำเร็จ ชิ้นส่วนเสียงยังอยู่ที่ {session_dir} : "
                f"{_ffmpeg_error_detail(e)}"
            )
            return
        if destination is None:
            return
        print(f"หยุดอัดแล้ว บันทึกไปที่ {destination}")
    finally:
        stop_event.set()
        if thread_started:
            recorder_thread.join()
        p.terminate()


if __name__ == "__main__":
    main()
