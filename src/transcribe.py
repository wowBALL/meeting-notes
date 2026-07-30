import os
import pathlib
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}

# Batch of VAD-segmented chunks decoded in parallel per forward pass -- same
# large-v3 weights, ~4x faster than sequential. Sized for the RTX 3060 6GB:
# the model resident at int8_float16 (~3.2GB) plus the pyannote pipeline now
# also living on the GPU (~1GB) leaves no headroom for a larger batch.
#
# *** ปิดเป็นค่าเริ่มต้นแล้ว (WHISPER_BATCHED=false) *** วัดกับเสียงประชุมจริงของ
# เครื่องนี้ (2026-07-30, ไฟล์ 4:41 นาที, ภาษาไทย 2 ผู้พูด):
#
#   int8_float16 + batched     1403 chars   76 คำ   โทเคนซ้ำมากสุด 31.6%   15s
#   float16      + batched     1570 chars   59 คำ                   5.1%   93s
#   float16      + sequential  1659 chars   85 คำ                   2.4%  165s
#   int8_float16 + sequential  1677 chars   96 คำ                   3.1%   61s
#
# แบบ batched ทำให้ faster-whisper วนซ้ำคำเดิม ("ทํา" 24 ครั้ง, "ปลอดภัณฑ์" 7 ครั้ง)
# กินพื้นที่ไปเกือบหนึ่งในสามของ transcript และเนื้อหาช่วงนั้นหายไปทั้งหมด
# sequential ได้คำมากกว่า 26% และคำซ้ำลดลง 10 เท่า แลกกับ 46 วินาทีต่อประชุม 5 นาที
#
# quantization ไม่ใช่ต้นเหตุ: float16 ไม่ได้ดีกว่า int8_float16 เลย (และ batched
# ที่ float16 ยังแย่กว่า int8 sequential) จึงคง int8_float16 ไว้ตามเดิม
#
# condition_on_previous_text=False ไม่ช่วยอะไร -- ทดลองแล้วได้ผลเท่ากันเป๊ะทุกไบต์
# ทั้งที่มันอยู่ใน signature ของ BatchedInferencePipeline.transcribe (1.2.1)
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


def load_whisper_model(model_size: str, batched: bool = False) -> Any:
    """โมเดล whisper ที่ cache ไว้ -- ห่อด้วย BatchedInferencePipeline เมื่อ batched=True

    ค่าเริ่มต้นไม่ห่อ เพราะวัดกับเสียงประชุมจริงแล้วตัวห่อทำให้คุณภาพแย่ลงมาก
    (ดูหมายเหตุใต้ BATCH_SIZE) cache key รวม batched ไว้ด้วย ไม่งั้นสลับค่าแล้วจะได้
    โมเดลผิดชนิดจาก cache
    """
    key = (model_size, batched)
    if key not in _MODEL_CACHE:
        _register_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        device, compute_type = _select_device_and_compute()
        model: Any = WhisperModel(model_size, device=device, compute_type=compute_type)
        if batched:
            from faster_whisper import BatchedInferencePipeline

            model = BatchedInferencePipeline(model=model)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def transcribe_audio(
    audio_path: Path,
    model_size: str = "large-v3",
    model: Any = None,
    batched: bool = False,
) -> list[dict]:
    if model is None:
        model = load_whisper_model(model_size, batched=batched)
    # batch_size ส่งได้เฉพาะกับ BatchedInferencePipeline -- WhisperModel.transcribe
    # ไม่รู้จักพารามิเตอร์นี้
    extra = {"batch_size": BATCH_SIZE} if batched else {}
    segments, _info = model.transcribe(
        str(audio_path), language="th", vad_filter=True, **extra
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
