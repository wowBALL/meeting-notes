import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import DEFAULT_DIARIZATION_MODEL
from src.gpu import cuda_device
from src.waveform import load_waveform

logger = logging.getLogger(__name__)


def load_diarization_pipeline(
    hf_token: str, checkpoint: str = DEFAULT_DIARIZATION_MODEL
) -> Any:
    """โหลด pipeline แยกผู้พูดตาม checkpoint ที่เลือก (ดู config.DIARIZATION_MODEL)

    ค่า default อยู่ที่ config.py ที่เดียว ไม่ทำสำเนาไว้ที่นี่ -- โมดูลนี้กับ .env ต้อง
    ตอบคำถาม "ตกลงใช้โมเดลไหน" ตรงกันเสมอ
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(checkpoint, token=hf_token)
    # Pipeline.from_pretrained ทิ้งโมเดลไว้บน CPU โดยไม่เตือนอะไร งานแค่ช้าลงระดับ
    # ลำดับขั้น (วัดจริง: diarization ของประชุม 50 นาที 15+ นาทีบน CPU เทียบกับ ~2 บน GPU)
    # เหตุผลเรื่อง cuDNN อยู่ใน src/gpu.py
    device = cuda_device()
    if device is not None:
        pipeline.to(device)
    return pipeline


@dataclass(frozen=True)
class DiarizationResult:
    """ผลของการแยกผู้พูดหนึ่งไฟล์ -- ช่วงเวลาว่าใครพูดเมื่อไหร่ เท่านั้น

    เคยมี `embeddings` (centroid ที่ pyannote คำนวณเพื่อ clustering ของตัวเอง) อยู่ที่นี่
    ด้วย ย้ายออกไปเป็น src/voiceprint.py เพราะ centroid นั้นอยู่ในพื้นที่ที่เปลี่ยนไปตาม
    DIARIZATION_MODEL -- 3.1 กับ community-1 ใช้โมเดล embedding คนละตัว ทะเบียนเสียงจึงล้ม
    ทั้งใบทุกครั้งที่สลับโมเดล
    """

    turns: list[dict]


def diarize_audio(
    audio_path: Path,
    hf_token: str,
    pipeline: Any = None,
    checkpoint: str = DEFAULT_DIARIZATION_MODEL,
) -> DiarizationResult:
    if pipeline is None:
        pipeline = load_diarization_pipeline(hf_token, checkpoint)
    # ป้อนเสียงเป็น waveform ในหน่วยความจำ ไม่ใช่ path -- pyannote 4.x อ่านไฟล์เองผ่าน
    # torchcodec เท่านั้น ซึ่งโหลดไม่ขึ้นบนเครื่องนี้และทำให้การแยกผู้พูดตายทุกครั้งแบบ
    # ที่ผู้ใช้ไม่เห็นอะไรเลย (ดูเหตุผลเต็มใน src/waveform.py)
    result = pipeline(load_waveform(audio_path))
    turns = []
    for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return DiarizationResult(turns=turns)
