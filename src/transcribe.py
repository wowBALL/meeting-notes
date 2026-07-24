import os
import pathlib
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}

# Batch of VAD-segmented chunks decoded in parallel per forward pass -- same
# large-v3 weights, ~3x faster than sequential. Sized for the RTX 3060 6GB:
# the model resident at int8_float16 (~3.2GB) plus the pyannote pipeline now
# also living on the GPU (~1GB) leaves no headroom for a larger batch.
BATCH_SIZE = 4


def _register_cuda_dll_dirs() -> None:
    # faster-whisper's CTranslate2 backend loads CUDA 12 cuBLAS/cuDNN DLLs that
    # ship in the nvidia-*-cu12 pip packages. Since Python 3.8 on Windows, DLL
    # resolution ignores PATH, so each provider's bin dir must be registered as
    # a DLL search directory before the first device="cuda" model is created.
    # No-op on non-Windows and when the nvidia packages aren't installed (CPU).
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
            d = pathlib.Path(root) / sub
            if d.is_dir():
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass


def _select_device_and_compute() -> tuple[str, str]:
    # int8_float16 fits large-v3 in ~4GB (RTX 3060 6GB); CPU uses int8.
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def load_whisper_model(model_size: str) -> Any:
    if model_size not in _MODEL_CACHE:
        _register_cuda_dll_dirs()
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device, compute_type = _select_device_and_compute()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _MODEL_CACHE[model_size] = BatchedInferencePipeline(model=model)
    return _MODEL_CACHE[model_size]


def transcribe_audio(
    audio_path: Path, model_size: str = "large-v3", model: Any = None
) -> list[dict]:
    if model is None:
        model = load_whisper_model(model_size)
    segments, _info = model.transcribe(
        str(audio_path), language="th", vad_filter=True, batch_size=BATCH_SIZE
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
