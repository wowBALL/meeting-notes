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
            # faster-whisper's ctranslate2 loads the cu12 cuDNN DLLs while this
            # torch build bundles its own cu13 cuDNN under the same basenames.
            # Windows keeps one DLL per basename per process, so whichever stack
            # initializes first poisons the other: pyannote's first GPU forward
            # after whisper died with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
            # (observed 2026-07-24). With cuDNN off, torch uses its native CUDA
            # kernels -- same outputs, still far faster than CPU, and the two
            # stacks no longer share any cuDNN state.
            torch.backends.cudnn.enabled = False
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
