import logging
import os
import pathlib
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
# condition_on_previous_text=False ไม่ช่วยอะไร *บน BatchedInferencePipeline* --
# ทดลองแล้วได้ผลเท่ากันเป๊ะทุกไบต์ ทั้งที่มันอยู่ใน signature (1.2.1)
#
# *** ข้อสรุปนั้นใช้กับ path ที่ระบบใช้จริงไม่ได้ *** วัดใหม่บน WhisperModel แบบ
# sequential (2026-07-31, ช่วง 80:00-90:00 ของประชุม 92 นาที, large-v3 int8_float16)
# เทียบกับ hotwords สามขนาด -- คอลัมน์ "ถูก%" คืออัตราส่วนคำในตาราง glossary ที่ถอด
# ถูก เทียบกับที่ถอดผิด:
#
#   hotwords   cond   segs   chars   ซ้ำสุด   ถูก%
#   ปิด        เปิด    170    5325    3.5%    34.7%   <- ค่าที่ระบบใช้อยู่วันนี้
#   ปิด        ปิด     112    5104    1.7%    41.2%
#   14 คำ      เปิด    124    5109    2.3%    50.0%
#   14 คำ      ปิด     105    5181    1.7%    44.2%
#   54 คำ      เปิด     90    2139    5.8%      --    <- พัง เนื้อหาหาย 60%
#   54 คำ      ปิด     108    5034    2.5%    56.1%
#
# แถว "54 คำ + cond เปิด" คือโมเดลตกวงวนซ้ำคำ ("PPU" 80 ครั้ง กินบรรทัด 58-132 จาก
# ทั้งหมด 133) แล้วไม่กลับมาอีกเลย เพราะข้อความที่ซ้ำถูกป้อนกลับเป็น prompt ของ
# หน้าต่างถัดไป -- ทำซ้ำได้สองรอบ (อีกรอบเหลือ 1685 chars, ซ้ำสุด 33.8%)
# ปิด conditioning แล้วอาการหายไปทั้งหมด สองปุ่มนี้จึงแยกกันวัดไม่ได้
#
# ยังไม่พลิกค่าเริ่มต้นเพราะวัดจากหน้าต่างเดียว (คำในตารางโผล่ ~50-70 ครั้ง) และ
# การถอดซ้ำ config เดิมให้ segment 220 กับ 170 -- ต่างกันพอที่จะยังไม่แยก 41% ออกจาก
# 56% ได้อย่างมั่นใจ ต้องยืนยันด้วยหน้าต่างที่สองก่อน
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
    hotwords: str | None = None,
    condition_on_previous_text: bool = True,
) -> list[dict]:
    if model is None:
        model = load_whisper_model(model_size, batched=batched)
    # batch_size ส่งได้เฉพาะกับ BatchedInferencePipeline -- WhisperModel.transcribe
    # ไม่รู้จักพารามิเตอร์นี้
    extra: dict[str, Any] = {"batch_size": BATCH_SIZE} if batched else {}
    extra["hotwords"] = hotwords
    # ค่าเริ่มต้น True ตรงกับ faster-whisper -- ปุ่มนี้มีไว้ตัดวงวนซ้ำคำ ซึ่งข้อความที่
    # ซ้ำถูกป้อนกลับเป็น prompt ของหน้าต่างถัดไปจนออกไม่ได้ (ดูการวัดใต้ BATCH_SIZE)
    extra["condition_on_previous_text"] = condition_on_previous_text
    segments, _info = model.transcribe(
        str(audio_path), language="th", vad_filter=True, **extra
    )
    # faster-whisper yields segments lazily as decoding proceeds -- logging as we
    # consume the generator is the only way to see how far a long transcription
    # has gotten (or whether it's stuck repeating the same point in the audio,
    # a known failure mode on noisy/spliced recordings) without waiting for the
    # whole call to return.
    result = []
    started = time.monotonic()
    for i, seg in enumerate(segments, start=1):
        result.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if i == 1 or i % 10 == 0:
            logger.info(
                "Transcribed segment %d, now at %.0fs of audio (%.0fs elapsed)",
                i,
                seg.end,
                time.monotonic() - started,
            )
    return result
