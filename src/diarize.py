import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class DiarizationResult:
    """ผลของการแยกผู้พูดหนึ่งไฟล์

    `embeddings` คีย์ด้วยชื่อ label ตรง ๆ ไม่ใช่ array คู่ขนานกับ turns เพราะ pyannote
    เรียง array ตาม `diarization.labels()` ซึ่งไม่ใช่ลำดับที่ผู้พูดโผล่ครั้งแรก การผูก
    เป็นคีย์ตั้งแต่ตรงนี้ทำให้ลำดับที่เคลื่อนไปหนึ่งตำแหน่ง (= จำเสียงผิดคน) ไม่มีทาง
    เกิดขึ้นเงียบ ๆ ที่ปลายทาง
    """

    turns: list[dict]
    embeddings: dict[str, list[float]] = field(default_factory=dict)


def _speaker_embeddings(result: Any, diarization: Any) -> dict[str, list[float]]:
    """เวกเตอร์เสียงหนึ่งตัวต่อผู้พูดหนึ่งคน คีย์ด้วย label

    pyannote คำนวณเวกเตอร์ชุดนี้อยู่แล้วเพื่อใช้ clustering เอง เราแค่เก็บมันไว้ --
    ไม่มีงาน GPU เพิ่มจากบรรทัดนี้เลย

    อ่านไม่ออกแปลว่าการประชุมนี้จำเสียงไม่ได้ ซึ่งยอมเสียได้ ต่างจากการแยกผู้พูด
    ที่ยังต้องได้ผลตามปกติ จึงกลืน exception ไว้ที่นี่แทนที่จะปล่อยขึ้นไป
    """
    embeddings = getattr(result, "speaker_embeddings", None)
    if embeddings is None:
        return {}
    try:
        labels = list(diarization.labels())
        return {
            label: [float(value) for value in embeddings[index]]
            for index, label in enumerate(labels)
            if index < len(embeddings)
        }
    except (TypeError, ValueError, IndexError, KeyError) as e:
        logger.warning("อ่าน speaker embeddings ไม่ได้ ไปต่อโดยไม่จำเสียง: %s", e)
        return {}


def diarize_audio(
    audio_path: Path, hf_token: str, pipeline: Any = None
) -> DiarizationResult:
    if pipeline is None:
        pipeline = load_diarization_pipeline(hf_token)
    result = pipeline(str(audio_path))
    diarization = result.speaker_diarization
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return DiarizationResult(
        turns=turns, embeddings=_speaker_embeddings(result, diarization)
    )
