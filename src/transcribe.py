import os
import pathlib
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}


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
    except ImportError:
        pass
    return "cpu", "int8"


def load_whisper_model(model_size: str) -> Any:
    if model_size not in _MODEL_CACHE:
        _register_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        device, compute_type = _select_device_and_compute()
        _MODEL_CACHE[model_size] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _MODEL_CACHE[model_size]


def transcribe_audio(
    audio_path: Path, model_size: str = "large-v3", model: Any = None
) -> list[dict]:
    if model is None:
        model = load_whisper_model(model_size)
    segments, _info = model.transcribe(str(audio_path), language="th", vad_filter=True)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
