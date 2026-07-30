"""ทางเดียวที่เสียงเข้าไปถึง pyannote ในโปรเจกต์นี้

pyannote 4.x อ่านไฟล์เสียงด้วย torchcodec เท่านั้น และ torchcodec โหลด DLL ของตัวเอง
ไม่ขึ้นบนเครื่องนี้ (torchcodec 0.15.0 กับ torch 2.13.0+cu130) -- ผลคือการส่ง path เข้า
pipeline ตาย `RuntimeError: torchcodec is not available. Cannot read audio file.` ทุกครั้ง
ซึ่ง pipeline.py ดักไว้แล้วเดินต่อด้วย speaker_turns = [] แบบเงียบ ๆ ประชุมทั้งครั้งจึงได้
ป้าย "ผู้พูด 1" ก้อนเดียวโดยไม่มีอะไรบอกว่าการแยกผู้พูดไม่ได้ทำงานเลย

pyannote รับ dict `{'waveform': (channel, time) tensor, 'sample_rate': int}` เป็นทางเลือก
อยู่แล้ว และทางนั้นไม่แตะ torchcodec เลย -- อ่านไบต์เองด้วย soundfile (dependency เดิมของ
โปรเจกต์) แล้วส่ง dict เข้าไปจึงตัดปัญหาทั้งชั้นออกไป ไม่ใช่แค่แก้อาการ

ไม่เสียอะไรจากการทำแบบนี้: ทุกเส้นทางที่เรียก pyannote (pipeline.process_file และ
enroll.analyze) แปลงเป็น wav 16k โมโนด้วย ffmpeg มาก่อนแล้ว -- ffmpeg คือตัวถอดรหัสเดียว
ที่โปรเจกต์นี้รับประกันอยู่ดี ส่วน soundfile อ่าน wav ได้โดยไม่ต้องพึ่ง DLL ของใคร
"""

from pathlib import Path

import soundfile as sf
import torch


def load_waveform(audio_path: Path) -> dict:
    """ไฟล์เสียง -> dict ที่ pyannote รับตรง ๆ

    `always_2d` เพราะ soundfile คืน (time,) สำหรับโมโนและ (time, channel) สำหรับหลายช่อง
    -- รูปที่ต่างกันตามเนื้อไฟล์คือที่มาของการสลับแกนผิดที่จับได้ยาก บังคับให้เป็น 2 มิติ
    เสมอแล้วสลับเป็น (channel, time) ครั้งเดียวจึงถูกทุกกรณี

    ไม่ resample และไม่ downmix ที่นี่: pyannote ทำทั้งสองอย่างเองจาก sample_rate ที่เรา
    บอกมัน (ดู Audio.downmix_and_resample) การทำซ้ำที่นี่มีแต่จะเพิ่มรอบการแปลงและโอกาส
    ที่สองที่จะไม่ตรงกัน

    ปล่อยให้ exception ลอยขึ้นไป ไม่กลืน: ผู้เรียกคือ diarize_audio ซึ่ง pipeline.py ดัก
    ไว้แล้วและบันทึกเหตุผลลง activity log -- การคืนเสียงว่างเงียบ ๆ จะกลายเป็นการประชุม
    ที่ "แยกผู้พูดสำเร็จแต่ไม่มีใครพูด" ซึ่งไม่มีใครตามหาสาเหตุได้
    """
    samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return {
        "waveform": torch.from_numpy(samples.T.copy()),
        "sample_rate": int(sample_rate),
    }
