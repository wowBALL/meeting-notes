import sys
from datetime import datetime

import numpy as np
import soundcard as sc
import soundfile as sf

from src.config import load_config

DEFAULT_SAMPLERATE = 48000


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
        return f"{name}-{now.strftime('%H-%M-%S')}.wav"
    return f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.wav"


def get_default_mic():
    return sc.default_microphone()


def get_default_speaker_loopback():
    speaker = sc.default_speaker()
    return sc.get_microphone(id=str(speaker.name), include_loopback=True)


def record_until_interrupted(
    mic, speaker, samplerate: int = DEFAULT_SAMPLERATE, blocksize: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    # soundcard only initializes COM (CoInitializeEx) on the thread that
    # first imports it. Recording mic and speaker concurrently from two
    # separate threads means calling WASAPI from a thread that never joined
    # a COM apartment, which is undefined behavior on Windows -- in
    # practice it reproducibly corrupted the process heap. Polling both
    # streams alternately from this single thread avoids the problem
    # entirely, at the cost of relying on each stream's own internal buffer
    # to absorb the time spent blocked on the other.
    # When `channels` isn't given, soundcard auto-detects it by reading a
    # COM property blob (IPropertyStore PKEY_AudioEngine_DeviceFormat) that
    # is unreliable in practice: it intermittently returns garbage channel
    # counts (observed: 87508, ~24940), which then corrupt buffer-size
    # math downstream -- sometimes as a catchable OverflowError, sometimes
    # as a native heap corruption crash with no Python-level output at
    # all. Passing channels explicitly skips that code path entirely.
    # WASAPI's shared-mode auto-convert flags (set unconditionally by
    # soundcard) handle any actual channel-count mismatch.
    mic_chunks: list[np.ndarray] = []
    speaker_chunks: list[np.ndarray] = []

    with mic.recorder(samplerate=samplerate, channels=2) as mic_recorder:
        with speaker.recorder(samplerate=samplerate, channels=2) as speaker_recorder:
            try:
                while True:
                    mic_chunks.append(mic_recorder.record(numframes=blocksize))
                    speaker_chunks.append(speaker_recorder.record(numframes=blocksize))
            except KeyboardInterrupt:
                pass

    mic_frames = np.concatenate(mic_chunks) if mic_chunks else np.zeros(0, dtype=np.float32)
    speaker_frames = (
        np.concatenate(speaker_chunks) if speaker_chunks else np.zeros(0, dtype=np.float32)
    )
    return mic_frames, speaker_frames


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else None

    config = load_config()
    config.inbox_dir.mkdir(parents=True, exist_ok=True)

    try:
        mic = get_default_mic()
        speaker = get_default_speaker_loopback()
    except Exception as e:
        print(f"ไม่พบไมค์/ลำโพง default กรุณาตรวจสอบการตั้งค่าเสียงของ Windows: {e}")
        return

    print("กำลังอัดเสียง... กด Ctrl+C เพื่อหยุด")
    try:
        mic_frames, speaker_frames = record_until_interrupted(mic, speaker)
    except Exception as e:
        print(
            "อัดเสียงไม่สำเร็จ (อาจไม่มีสิทธิ์เข้าถึงไมค์ - ตรวจสอบที่ "
            f"Settings > Privacy > Microphone): {e}"
        )
        return

    mixed = mix_recordings(to_mono(mic_frames), to_mono(speaker_frames))

    now = datetime.now()
    filename = build_output_filename(name, now)
    output_path = config.inbox_dir / filename
    sf.write(str(output_path), mixed, DEFAULT_SAMPLERATE)

    print(f"หยุดอัดแล้ว บันทึกไปที่ {output_path}")


if __name__ == "__main__":
    main()
