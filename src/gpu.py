"""การตัดสินใจเรื่อง GPU ที่เดียวของโปรเจกต์นี้

มีสองโมเดลที่ต้องขึ้น GPU (diarization pipeline กับโมเดล embedding ของ voiceprint) และ
ทั้งคู่อยู่ใต้ข้อจำกัดเดียวกันเป๊ะ ๆ: ctranslate2 ของ faster-whisper โหลด cuDNN cu12 ขณะที่
torch build นี้ bundle cu13 มาใต้ชื่อ DLL เดียวกัน Windows เก็บ DLL ชื่อละหนึ่งตัวต่อ
process ตัวที่ init ก่อนจึงทำให้อีกตัวตายด้วย CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
(วัดจริงกับ watcher 2026-07-24)

การคัดลอกเงื่อนไขนี้ไปไว้ที่ผู้เรียกทั้งสองคือการเปิดทางให้สองที่นั้นเลื่อนออกจากกันในวันที่
มีคนแก้ที่เดียว -- ผู้เรียกสองรายที่มีข้อจำกัดเดียวกันคือเหตุผลที่พอสำหรับโมดูลขนาดนี้
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def cuda_device() -> Any:
    """torch.device("cuda") ถ้าใช้ได้ ไม่งั้น None

    คืน `None` แทนการ raise เพราะการไม่มี GPU ไม่ใช่ความผิดพลาด -- README รองรับการรันบน
    เครื่องที่ไม่มี CUDA อยู่แล้ว และผู้เรียกทุกรายแปล `None` ว่า "อยู่บน CPU ต่อไป"

    ปิด cuDNN *ก่อน* คืน device ไม่ใช่หลังจากผู้เรียกย้ายโมเดลขึ้นไปแล้ว: forward ครั้งแรก
    เกิดขึ้นได้ทันทีหลัง .to() และถ้า cuDNN ยังเปิดอยู่ตอนนั้นก็สายเกินไป

    except กว้างโดยเจตนา: torch โยนอะไรออกมาก็ได้ตอนถามเรื่องไดรเวอร์ และไม่ว่าตัวไหน
    คำตอบที่ถูกคือ "ใช้ CPU" ไม่ใช่ "เปิดโปรแกรมไม่ได้"
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.backends.cudnn.enabled = False
        return torch.device("cuda")
    except Exception as e:
        logger.info("ใช้ GPU ไม่ได้ (%s) รันบน CPU ต่อ", e)
        return None
