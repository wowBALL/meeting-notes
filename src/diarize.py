import logging
from dataclasses import dataclass, field
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
    try:
        embeddings = getattr(result, "speaker_embeddings", None)
        if embeddings is None:
            return {}
        labels = list(diarization.labels())
        return {
            label: [float(value) for value in embeddings[index]]
            for index, label in enumerate(labels)
            if index < len(embeddings)
        }
    except Exception as e:
        # กว้างโดยตั้งใจ ไม่ใช่ความมักง่าย: ฟังก์ชันนี้ต้องไม่ปล่อยอะไรออกไปเลย
        # เพราะ diarize_audio สร้าง turns เสร็จแล้วก่อนเรียกมัน และ exception ที่
        # หลุดออกไปจะไปโดน except ของ pipeline ซึ่งเซ็ต speaker_turns = [] --
        # ความล้มเหลวของ "การจำเสียง" จะไปทำลาย "การแยกผู้พูด" ของประชุมที่อัดซ้ำ
        # ไม่ได้ ซึ่งเป็นความไม่สมมาตรที่ทั้ง spec และ docstring ข้างบนห้ามไว้ --
        # getattr(..., None) เองก็อยู่ในนี้ด้วย เพราะ default ของมันกันแค่
        # AttributeError จากการหา attribute เท่านั้น ถ้า speaker_embeddings เคย
        # กลายเป็น property ที่ raise อย่างอื่น มันต้องโดนกลืนที่นี่เหมือนกัน
        logger.warning("อ่าน speaker embeddings ไม่ได้ ไปต่อโดยไม่จำเสียง: %s", e)
        return {}


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
    diarization = result.speaker_diarization
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return DiarizationResult(
        turns=turns, embeddings=_speaker_embeddings(result, diarization)
    )
