from pathlib import Path
from typing import Any


def load_diarization_pipeline(hf_token: str) -> Any:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    # Pipeline.from_pretrained leaves the model on the CPU; nothing warns about
    # it, the work just runs an order of magnitude slower (measured: 15+ minutes
    # of diarization for a 50-minute meeting vs ~2 on the GPU). Same model, same
    # output -- only the device changes. Falls back to CPU exactly as before
    # when torch/CUDA is unavailable.
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    return pipeline


def diarize_audio(audio_path: Path, hf_token: str, pipeline: Any = None) -> list[dict]:
    if pipeline is None:
        pipeline = load_diarization_pipeline(hf_token)
    result = pipeline(str(audio_path))
    diarization = result.speaker_diarization
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns
