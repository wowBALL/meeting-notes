from datetime import datetime

import numpy as np


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
