from pathlib import Path
from typing import Any


def diarize_audio(audio_path: Path, hf_token: str, pipeline: Any = None) -> list[dict]:
    if pipeline is None:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        )
    diarization = pipeline(str(audio_path))
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns
