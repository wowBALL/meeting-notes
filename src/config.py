import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ชุดประเภทประชุมที่รองรับมาจากที่เดียว: ไฟล์ใน prompts/profiles/ คือของจริง
# ถ้า config มีลิสต์ของตัวเอง วันหนึ่งจะมี profile ที่ผ่าน config แต่หา prompt ไม่เจอ
from src.prompts import KNOWN_PROFILES

logger = logging.getLogger(__name__)

# ค่าเริ่มต้นเป็นโมเดลบน endpoint ของบริษัท ไม่ใช่ Claude เพราะเหตุผลเรื่องความเป็น
# ส่วนตัวของข้อมูล: ทางที่ transcript ไม่ออกนอกบริษัทต้องเป็นทางที่ไม่ต้องคิด ถ้า Claude
# เป็นค่าเริ่มต้น การกด Enter ผ่านเมนูจะส่งข้อมูลออกไปทุกครั้งโดยไม่มีใครตัดสินใจเรื่องนั้น
#
# ในสามตัวที่ข้อมูลไม่ออกนอกบริษัท ไม่ใช่ GLM-5.2 เพราะ GLM ใช้โควตาหมดไปกับ
# reasoning แล้วคืน content ว่างเปล่า 10 ครั้งจาก 10 ครั้งกับงานสรุป (วัด 2026-07-31 --
# ดูคอมเมนต์ใน src/llm.py) ค่าเริ่มต้นที่เป็นส่วนตัวแต่ไม่คืนสรุปคือค่าเริ่มต้นที่ใช้ไม่ได้
#
# gemma4 ถูกลองเป็นค่าเริ่มต้นแทน Qwen เมื่อ 2026-08-05 ตามที่ผู้ใช้เลือกเอง แล้วถอยกลับมา
# Qwen วันเดียวกันหลังเทียบกับประชุมจริง (84 นาที, 3 chunks): gemma4 ยุบขั้น reduce เหลือ
# สรุปแค่ chunk แรก (ถึง 38:24) เงียบๆ -- finish_reason ไม่ใช่ "length" และไม่มี
# TRUNCATION_NOTICE/REDUCE_FAILURE_NOTICE ติดมาด้วยเลย (ดู src/summarize.py) แปลว่ามันไม่ใช่
# โควตาไม่พอแบบที่ retry สองเท่าจะช่วยได้ แต่เป็นโมเดลไม่ทำตามคำสั่ง reduce ครบเมื่อ input
# ยาว -- อันตรายกว่าการตัดกลางคันเพราะไม่มีสัญญาณอะไรบอกผู้ใช้เลยว่าสรุปหายไปสองในสาม
# ส่วน Qwen สรุปครบทั้ง 3 chunks ในเวลาใกล้เคียงกัน (69s vs gemma4 60s) -- ถ้าจะลอง gemma4
# เป็นค่าเริ่มต้นอีกครั้งต้องแก้ปัญหานี้ก่อน ไม่ใช่แค่วัดก้อนเล็กแล้วตั้งเป็นค่าเริ่มต้นตรงๆ
#
# อัปเดต: อาการ "หายเงียบ" ที่ย่อหน้าบนบรรยายไว้เกิดไม่ได้อีกแล้ว -- gemma4 ตั้ง
# can_map_reduce=False (ดู src/llm.py) ประชุมที่เกินเพดาน single-call ของมันจะล้มให้เห็น
# พร้อมข้อความบอกให้เลือกโมเดลอื่น แทนที่จะเดินเข้า reduce แล้วคืนสรุปที่ขาดไปสองในสาม
#
# แต่นั่นยังไม่ใช่เหตุผลให้ตั้งเป็นค่าเริ่มต้น: gemma4 ตอนนี้ "ปฏิเสธ" ประชุมที่ยาวเกิน
# 150,000 token ขณะที่ Qwen สรุปให้ได้จริง ค่าเริ่มต้นที่ปฏิเสธงานยังแย่กว่าค่าเริ่มต้นที่
# ทำงานได้ทุกขนาด -- เปลี่ยนจาก "อันตรายเกินกว่าจะเป็นค่าเริ่มต้น" เป็น "ยังครอบไม่ครบ
# เท่า Qwen" เท่านั้น (ประชุมจริงยาวสุดที่เคยมีคือ ~150 นาที ~92,650 token ยังห่างเพดานอยู่)
#
# preflight อ่านค่านี้ด้วยโดยไม่ผ่าน load_config -- จึงต้องอยู่ตรงนี้ ไม่ใช่ฝังใน default
# ของ dataclass ที่แห่งเดียว
#
# ต้องตรงกับค่าเริ่มต้นของ start-meeting.bat และ web/app.js ทั้งสองที่ ไม่งั้น preflight
# จะไปตรวจโมเดลคนละตัวกับที่การกด Enter เลือกจริง
#
# เคยมี DEFAULT_CLAUDE_MODEL เป็นชื่อของค่านี้ ลบทิ้งแล้วไม่เก็บเป็น alias เพราะชื่อที่
# บอกว่า Claude แต่ถือค่าที่ไม่ใช่ Claude คือชื่อที่โกหก ส่วนชื่อ field claude_model ใน
# Config และใน job.json คงไว้ตามเดิมโดยเจตนา -- การรีเนมกระทบ sidecar ที่ค้างใน inbox/
DEFAULT_SUMMARY_MODEL = "Qwen/Qwen3.6-35B-A3B"

# ใช้โดย session_service และเป็นภาษาตั้งต้นของข้อความฝั่ง CLI -- หน้าเว็บจำภาษา
# ของตัวเองใน localStorage จึงไม่ได้อ่านค่านี้
DEFAULT_UI_PORT = 8765
DEFAULT_UI_LANG = "th"

# โมเดลแยกผู้พูด เปลี่ยนได้ด้วย DIARIZATION_MODEL ใน .env โดยไม่ต้องแก้โค้ด
#
# community-1 (VBxClustering) เป็นค่าเริ่มต้นเพราะวัดกับไฟล์ประชุมจริงของเครื่องนี้แล้ว
# (2026-07-29, Meet1900 80 นาที): 3.1 ยัดคนหลายคนรวมเป็นป้ายเดียว 94 จาก 145 ท่อนเป็น
# "ผู้พูด 1" ก้อนใหญ่ของมันตกเกณฑ์ "คนเดียวจริงไหม" 2 ใน 3 ท่อนทดสอบ ส่วน community-1
# กระจายเป็น 34/22/62/19/1/7 และก้อนใหญ่ผ่านทั้ง 3 ท่อน -- รันสดทั้งท่อ (Meet1903)
# ใช้เวลา 10m54s เทียบกับ 3.1 ที่ 11m30s บนไฟล์เดียวกัน จึงไม่ได้แลกความเร็วมา
#
# 3.1 คือค่าที่ใช้มาก่อนหน้านี้ทั้งหมด กลับไปได้ด้วย DIARIZATION_MODEL=<ค่านี้> ใน .env
# แต่ต้องอ่านหมายเหตุเรื่องทะเบียนใต้ DEFAULT_SPEAKER_MATCH_HIGH ก่อน
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# โมเดลที่สร้าง "voiceprint" สำหรับจำเสียงข้ามการประชุม -- แยกจาก DIARIZATION_MODEL โดย
# สิ้นเชิงและนั่นคือเหตุผลทั้งหมดที่มันมีอยู่: 3.1 กับ community-1 ใช้โมเดล embedding คนละ
# ตัว (ตรวจจาก config.yaml ของทั้งคู่) การเอา centroid ของ pipeline มาใช้จึงทำให้การสลับ
# โมเดลแยกผู้พูดล้มทะเบียนทั้งใบ ตรึงตัวนี้ไว้แล้วสลับ DIARIZATION_MODEL ได้อย่างอิสระ
#
# wespeaker-voxceleb-resnet34-LM เป็นโมเดล speaker verification โดยตรง (VoxCeleb EER
# ~0.9% ดีกว่า ECAPA-TDNN) ไม่ต้องเพิ่ม dependency เพราะ pyannote.audio โหลดมันได้อยู่แล้ว
#
# 26MB และ **ไม่ใช่ gated** (ตรวจ 2026-07-31: gated=False ต่างจาก community-1 กับ
# segmentation-3.0 ที่ต้องกด Agree) ดาวน์โหลดเองอัตโนมัติครั้งแรกที่ใช้ ไม่ต้องตั้งค่าอะไร
# -- เครื่องที่เคยใช้ diarization 3.1 จะมีมันในแคชอยู่แล้วเพราะ 3.1 ใช้ตัวเดียวกัน แต่
# เครื่องที่ใช้ community-1 มาตลอด (ค่าเริ่มต้นตอนนี้) จะต้องโหลดใหม่ครั้งเดียว
#
# *** การเปลี่ยนค่านี้ทำให้ทุกคนในทะเบียนต้อง enroll ใหม่ *** เวกเตอร์จากคนละโมเดลอยู่คนละ
# พื้นที่ cosine ระหว่างสองฝั่งได้ตัวเลขที่ "ดูใช้ได้" แต่ไม่มีความหมาย sample ทุกตัวจึงถูก
# ติดป้าย embedding_model และ speakers.match_known ใช้เฉพาะตัวที่ป้ายตรง (ข้ามตัวที่ไม่มี
# ป้ายด้วย ไม่มีการเดา) ข้อมูลเดิมไม่หาย สลับกลับมาก็ใช้ได้เหมือนเดิม
DEFAULT_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# เกณฑ์ความเหมือนของน้ำเสียง (cosine similarity บน voiceprint ของ EMBEDDING_MODEL):
# สูงกว่า HIGH = ใส่ชื่อให้เลย, ระหว่าง LOW กับ HIGH = เสนอให้คนยืนยัน, ต่ำกว่า LOW =
# ถือว่าไม่รู้จัก
#
# วัดจริง 2026-07-31 ด้วยท่อทั้งเส้นของ src/voiceprint.py บน Meet1900 เต็มไฟล์ (81 นาที,
# community-1, 2248 turn, 6 label) บวกคลิป enroll 20 วินาที -- วัดซ้ำได้ด้วย
# `python -m tools.calibrate_voiceprints speakers-truth.json --cache .calib-cache`:
#
#   กองคนเดียวกัน (ข้ามการบันทึก)   n=1    0.6679
#   กองคนละคน (ทุกคู่)              n=20   0.0454 - 0.5711  มัธยฐาน 0.2904
#   ช่องว่างระหว่างสองกอง                  +0.0967
#
# **ทั้ง 6 ป้ายในประชุมยืนยันด้วยหูโดยเจ้าของเสียงเอง** (2026-07-31 ฟังจากคลิปที่
# `--extract-clips` ตัดให้) ไม่ใช่อนุมานจากคะแนน -- การเอาคะแนนสูงสุดมาสรุปว่า "ต้องคือเขา"
# แล้วตั้งเกณฑ์จากข้อสรุปนั้น คือการเอาคำตอบมาพิสูจน์ตัวมันเอง ทั้งคู่ที่บังคับขอบทั้งสองฝั่ง
# จึงเป็นคู่ที่รู้จริงว่าใครเป็นใคร:
#
#   0.6679  คนเดียวกัน  คลิป enroll <-> ป้ายของเจ้าตัวในประชุม     <- ขอบบน
#   0.5711  คนละคน      สองคนที่เสียงคล้ายกันที่สุดในทีม           <- ขอบล่าง
#   0.4372  คนละคน      คู่ที่สูงสุดในบรรดาคู่ที่มีเจ้าของทะเบียนอยู่ด้วย
#
# HIGH = 0.62 และ LOW = 0.58 เป็นค่าที่ tools/calibrate_voiceprints คำนวณออกมาเอง
# ไม่ใช่ค่าที่เลือกด้วยมือแล้วหาเหตุผลมารองรับทีหลัง -- รันซ้ำได้ ได้เลขเดิม exit code 0
#
# HIGH อยู่กลางช่องว่าง ห่างสองฝั่งข้างละ ~0.048 ทิศทางของความเสียหายสองฝั่งไม่เท่ากัน
# (สูงเกินไป = ไม่ใส่ชื่อคนที่จำได้ คลิกแก้ได้ / ต่ำเกินไป = ใส่ชื่อผิดคน แก้ไม่ได้ถ้าไม่มี
# ใครสังเกต) แต่ที่เลือกกลางแทนที่จะเอียงขึ้น เพราะขอบบนมาจากคู่เดียว (n=1) เอียงขึ้นคือการ
# ปฏิเสธหลักฐานเดียวที่มี
#
# LOW = 0.58 อยู่เหนือเพดานของกองคนละคน (0.5711) เล็กน้อย เพื่อไม่ให้คนที่ไม่ใช่โผล่ในโซน
# "เสนอให้ยืนยัน" เลย -- ช่องระหว่าง LOW กับ HIGH จึงแคบมาก (0.04) ซึ่งเป็นคำตอบที่ตรงกับ
# ข้อมูล: สองกองแยกกันชัด ช่องเสนอให้ยืนยันจึงไม่มีอะไรให้ทำมาก คู่ที่ตกใต้ 0.58 จะไปโผล่ใน
# คิวตั้งชื่อแบบไม่มีชื่อเสนอ ซึ่งเป็นทางที่ปลอดภัยกว่าการเสนอชื่อผิด ช่องนี้จะกว้างขึ้นเอง
# เมื่อมีการประชุมมากขึ้นและสองกองเริ่มกระจาย -- อย่ากว้างมันด้วยมือก่อนถึงตอนนั้น
#
# *** n=1 คือข้อจำกัดที่ใหญ่ที่สุดของชุดนี้ *** กองคนเดียวกันมีจุดเดียว ขอบบนจึงเป็นจุด
# ไม่ใช่การกระจาย ประชุมครั้งถัดไปที่รู้ว่าใครพูดจะเพิ่มคู่ข้ามการบันทึกอีกหลายคู่ -- เติมลง
# speakers-truth.json แล้วรันใหม่ ถ้ามีคู่คนเดียวกันตัวไหนต่ำกว่า 0.62 ต้องลด HIGH ตาม และ
# เขียนชุดวัดใหม่ต่อท้ายที่นี่ โดย **ไม่ลบชุดเดิมทิ้ง** -- ประวัติของการวัดคือสิ่งเดียวที่
# บอกได้ว่าเกณฑ์นิ่งหรือกำลังเลื่อน
#
# *** ทำไมไม่ใช่ 0.80 เหมือนก่อนหน้านี้ ***: 0.80 วัดมาจากพื้นที่ของ centroid ที่ pyannote
# คำนวณเพื่อ clustering ซึ่งไม่ใช่พื้นที่นี้ ในพื้นที่ใหม่ไม่มีคู่ไหนแตะ 0.70 ด้วยซ้ำ 0.80
# จึงแปลว่า "ไม่ใส่ชื่อให้ใครเลยตลอดกาล" -- ตัวเลขเก่าที่ยกมาใช้ต่อในพื้นที่ใหม่คือค่าที่ดู
# สมเหตุสมผลแต่ผิดอย่างเงียบที่สุด
#
# และ 0.70 ที่เคยเขียนไว้ตรงนี้ระหว่างที่ตัวตนของ SPEAKER_03 ยังไม่ยืนยัน ก็จะปฏิเสธ
# 0.6679 เหมือนกัน -- ตอนนั้นเป็นค่าที่ถูกสำหรับหลักฐานที่มี ณ ตอนนั้น (ไม่มีคู่ยืนยันเลย
# จึงต้องไม่ใส่ชื่อใครอัตโนมัติ) พอมีคู่ยืนยันแล้ว ค่าที่ถูกก็เปลี่ยนตาม
#
# *** และตัวเลขชุดแรกที่เคยเขียนไว้ตรงนี้ (0.7493/0.6313/0.4310) ก็ผิดด้วยเหตุผลเดียวกัน ***
# มันวัดจากการป้อน .ogg 48 kHz เข้า pyannote ตรง ๆ แต่ระบบจริงแปลงเป็น wav 16 kHz ก่อนเสมอ
# (pipeline.process_file, enroll.analyze) ไฟล์เดียวกันโมเดลเดียวกันแต่ได้ turn คนละชุด
# (2143 กับ 2248) เลขป้ายเลื่อนทั้งชุด และคะแนนต่างกันราว 0.08 -- ทุกครั้งที่วัด ต้องวัดผ่าน
# ทางที่ระบบเดินจริงเท่านั้น ดูรายละเอียดที่ tools/README.md หัวข้อเรื่องเลขป้ายไม่ถาวร
#
# *** ทะเบียนเสียงกับการสลับโมเดล ***: เวกเตอร์จากคนละโมเดลอยู่คนละพื้นที่ เอามาหา
# cosine กันได้ตัวเลขที่ "ดูใช้ได้" แต่ไม่มีความหมาย -- ปล่อยไว้เท่ากับใส่ชื่อผิดคนลง
# transcript เงียบ ๆ ซึ่งเป็นอันตรายตัวเดียวกับที่เกณฑ์คู่นี้ตั้งมากัน ตัวอย่างเสียง
# ทุกตัวในทะเบียนจึงถูกติดป้าย embedding_model ที่สร้างมันไว้ และ speakers.match_known()
# ใช้เฉพาะตัวอย่างที่ป้ายตรงกับโมเดลปัจจุบัน สลับ EMBEDDING_MODEL แล้วคนที่ enroll ไว้ใน
# อีกฝั่งจะไม่ถูกจำ (กลับไปเป็นป้าย "ผู้พูด N") จนกว่าจะ enroll ใหม่ -- แต่ข้อมูลเดิมไม่หาย
# สลับกลับมาก็ใช้ได้เหมือนเดิม เกณฑ์คู่นี้ผูกกับ EMBEDDING_MODEL ไม่ใช่ DIARIZATION_MODEL
# อีกต่อไป: สลับโมเดลแยกผู้พูดไม่กระทบเลขคู่นี้
DEFAULT_SPEAKER_MATCH_HIGH = 0.62
DEFAULT_SPEAKER_MATCH_LOW = 0.58

# ประเภทประชุม เลือกได้ต่อประชุมจากเมนูตอนเริ่มอัด ค่านี้เป็นค่าที่ใช้เมื่อไม่ได้เลือก
# (ไฟล์ที่ลากใส่ inbox/ เอง หรือ .job.json ที่เขียนไว้ก่อนจะมีฟีเจอร์นี้)
#
# dev เป็นค่าเริ่มต้นเพราะสัดส่วนจริงคือ dev ล้วน 3 ครั้ง / ข้ามฝ่าย 1 ครั้ง ต่อสัปดาห์
# และการเผลอใช้ cross กับประชุม dev ล้วนไม่ใช่แค่เปลืองโทเคน: prompt จะบอกโมเดลว่า
# คำอย่าง "เสร็จ" กำกวมระหว่างสองฝ่าย ทั้งที่ในห้องมีแต่ dev โมเดลจะไป qualify
# คำพูดปกติเกินจำเป็นจนได้สรุปที่อ่านแล้วอ้อมค้อม
DEFAULT_MEETING_PROFILE = "dev"

# ส่งหัวข้อ "ต้องคุยต่อครั้งหน้า" ของประชุมครั้งก่อน (ประเภทเดียวกัน) เข้าไปในสรุปครั้งนี้
# เปิดเป็นค่าเริ่มต้นเพราะประชุม dev ถี่ 3 ครั้ง/สัปดาห์ เรื่องค้างถูกยกมาคุยต่อเสมอ
#
# ปิดเมื่อไหร่: กลไกนี้ผูกสรุปเข้าด้วยกันเป็นลูกโซ่ ถ้าสรุปครั้งหนึ่งเพี้ยน เรื่องค้าง
# ที่ผิดจะถูกส่งต่อไปครั้งถัดไป ช่วงที่ยังจูน prompt อยู่ควรอ่านหัวข้อนั้นทุกครั้งก่อน
# ปล่อยผ่าน หรือปิดสวิตช์นี้ไปก่อน
DEFAULT_CARRYOVER_ENABLED = True

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})

# ถอดเสียงแบบ batch (BatchedInferencePipeline) -- ปิดไว้เพราะวัดกับเสียงประชุมจริง
# ของเครื่องนี้แล้วว่ามันทำให้ faster-whisper วนซ้ำคำเดิมจนเนื้อหาหาย (โทเคนที่ซ้ำ
# มากสุดกินพื้นที่ 31.6% ของ transcript) ตัวเลขการวัดทั้งสี่ config อยู่ใต้ BATCH_SIZE
# ใน src/transcribe.py
#
# เปิดกลับได้ด้วย WHISPER_BATCHED=true ถ้ายอมแลกคุณภาพกับความเร็ว ~4 เท่า
DEFAULT_WHISPER_BATCHED = False

# ป้อนคำถูกจาก glossary.md ให้ decoder ตั้งแต่ตอนถอดเสียง (faster-whisper `hotwords`)
#
# ปิดไว้เพราะวัดแล้วว่ามันทำลาย transcript ทั้งไฟล์ ไม่ใช่แค่ "ยังไม่รู้ว่าดีขึ้นไหม"
# -- กฎเดียวกับที่ WHISPER_BATCHED ถูกปิดหลังวัด (ดูใต้ BATCH_SIZE ใน transcribe.py)
#
# *** วัดจริง 2026-07-31 บนประชุม Stanup (84 นาที, ไทย, 8 ผู้พูด, large-v3,
# sequential) ไฟล์เสียงเดียวกันเป๊ะ รันสองรอบสลับค่านี้: ***
#
#   hotwords=false   1618 บรรทัด   84,916 อักขระ   "สวัสดีครับพี่" 2 บรรทัด (0.1%)
#   hotwords=true    1059 บรรทัด   43,816 อักขระ   "สวัสดีครับพี่" 1044 บรรทัด (98.6%)
#
# รอบที่เปิด decoder ล็อกอยู่กับประโยคทักทายจากนาทีแรกตั้งแต่ [01:01] แล้ววนซ้ำจนจบไฟล์
# เหลือเนื้อหาจริงราว 15 บรรทัด ศัพท์เทคนิคที่อยู่ใน hotwords เองหายเกลี้ยง (Migration
# 16 -> 0, Merge Request 3 -> 0, BRD 2 -> 0) คือคำที่ป้อนเข้าไปเพื่อให้ถอดถูก กลับไม่
# ปรากฏในผลลัพธ์เลยสักคำ
#
# อาการนี้คือ repetition loop ตัวเดียวกับที่ batched=True ทำ (โทเคนซ้ำ 31.6%) แต่หนักกว่า
# สามเท่า และต่างจากความเสี่ยงที่เคยเดาไว้ ("โมเดลแต่งคำในลิสต์ขึ้นมาในช่วงเงียบ") --
# ของจริงคือมันไม่แต่งคำในลิสต์เลย มันหยุดถอดเสียงไปทั้งไฟล์
#
# ผลข้างเคียงที่หลอกตา: รอบที่เปิดใช้เวลา 9 นาที ส่วนรอบที่ปิดใช้ 28 นาที -- เร็วขึ้น
# สามเท่าเพราะมันเลิกถอดเสียงตั้งแต่นาทีแรก ไม่ใช่เพราะ decoder ทำงานดีขึ้น อย่าอ่าน
# เวลาที่ลดลงเป็นสัญญาณบวก
#
# สองข้อที่ "ยังไม่ได้ลอง" ข้างบนถูกลองแล้ว (2026-07-31, สไลซ์ 10 นาทีสองช่วงของประชุม
# 92 นาที ช่วง 80:00-90:00 และ 10:00-20:00) ตัวเลขคือส่วนต่างของอัตราส่วน "คำในตาราง
# glossary ที่ถอดถูก : ที่ถอดผิด" เทียบกับค่าเริ่มต้นของหน้าต่างเดียวกัน:
#
#                                  80:00-90:00   10:00-20:00
#   hotwords เต็ม + cond ปิด          +21.4         +20.4
#   hotwords 14-18 คำ + cond เปิด     +15.3          +3.0
#   hotwords 14-18 คำ + cond ปิด       +9.5          +2.0
#   cond ปิด อย่างเดียว                +6.5          -3.6   <- กลับเครื่องหมาย
#
# ปิด conditioning ทำให้อาการวนซ้ำหายไปจริงบนสไลซ์ทั้งสอง (hotwords เต็ม + cond เปิด
# เหลือเนื้อหา 40% ส่วน + cond ปิด เหลือ 95%) และ hotwords ที่สั้นลงไม่ใช่คำตอบ --
# ปุ่มเดี่ยว ๆ ทั้งสองตัวไม่นิ่งข้ามหน้าต่าง มีแต่คู่เต็มที่นิ่ง
#
# *** ยืนยันบนไฟล์เต็มแล้ว *** (2026-07-31, Stanup 84 นาที ไฟล์เดียวกับที่รอบพังใช้
# ให้คะแนนใหม่ทั้งสองฝั่งด้วย glossary.md ตัวปัจจุบัน เพราะตารางโตขึ้นระหว่างทาง):
#
#                            บรรทัด   อักขระ   บรรทัดซ้ำสุด   ถูก%
#   hw ปิด / cond เปิด (เดิม)   1618    40,580       0.5%      28.9%
#   hw เปิด / cond ปิด           877    39,154       0.6%      54.2%   <- ค่าปัจจุบัน
#   hw เปิด / cond เปิด         1059    43,816      98.6%        --    <- รอบที่พัง
#
# +25.3 จุด สอดคล้องกับสไลซ์ทั้งสอง (+21.4, +20.4) เนื้อหาหายเพียง 3.5% และไม่มีวงวน
# ระดับไฟล์อีกเลย ใช้เวลา 27 นาที (รอบที่พังใช้ 9 นาทีเพราะมันเลิกถอดเสียง)
#
# สิ่งที่แลกมา วัดไว้ให้ครบ ไม่ใช่แค่ด้านที่ดีขึ้น:
#   * segment ยาวขึ้นเท่าตัว (เฉลี่ย 24.1 -> 43.6 อักขระ, p99 80 -> 271) เพราะโมเดล
#     ไม่มีบริบทประโยคก่อนหน้ามาช่วยตัด -- ดันสัดส่วนบรรทัดที่มีสองคนพูดจาก 0.2%
#     เป็น 0.5% (ยังเล็กมาก แต่เป็นทิศทางที่ทำให้ป้ายผู้พูดพังง่ายขึ้น)
#   * วงวนซ้ำคำยังเกิดได้ แค่ถูกกักไว้ในหน้าต่างเดียวแทนที่จะลามทั้งไฟล์ -- บรรทัดที่
#     มีวลีซ้ำติดกัน 3+ ครั้ง กินพื้นที่ 1.4% -> 3.7% ของ transcript
#   * อักขระจากภาษาอื่นที่หลุดมา 0.067% -> 0.112%
DEFAULT_WHISPER_HOTWORDS = True

# ตัวตัดวงวนซ้ำคำ -- ค่านี้ต่างจาก faster-whisper (ซึ่งเป็น True) โดยเจตนา
#
# ปิดไว้เพราะมันคือกลไกที่ทำให้วงวนซ้ำ "ติด" ข้ามหน้าต่าง 30 วินาที: ข้อความที่ซ้ำถูก
# ป้อนกลับเป็น prompt ของหน้าต่างถัดไป ซึ่งผลิตซ้ำอีก วนจนจบไฟล์ ปิดแล้วแต่ละหน้าต่าง
# ตัดสินจากเสียงล้วน วงวนจึงจบในตัวมันเอง
#
# คอมเมนต์เก่าใต้ BATCH_SIZE เคยบันทึกว่าค่านี้ "ไม่ช่วยอะไร ได้ผลเท่ากันเป๊ะทุกไบต์"
# -- นั่นวัดบน BatchedInferencePipeline ซึ่งเป็น path ที่ปิดไปแล้ว บน WhisperModel
# แบบ sequential ที่ระบบใช้จริง มันเปลี่ยนผลลัพธ์ชัดเจน (ดูตารางใต้ BATCH_SIZE)
#
# ลำพังตัวเดียวไม่ได้ดีขึ้นเสมอไป (+6.5 จุดหน้าต่างหนึ่ง แต่ -3.6 อีกหน้าต่าง) ที่ปิดไว้
# เป็นค่าเริ่มต้นเพราะมันลดอัตราวนซ้ำได้ทุกรอบที่วัด และเป็นเงื่อนไขจำเป็นของ hotwords
DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

# ขนาด chunk ตอนหั่น transcript ยาว และส่วนที่ให้ chunk คาบเกี่ยวกัน
#
# ค่าคู่นี้อยู่ที่นี่ไม่ใช่ใน summarize.py เพราะ summarize.py import config อยู่แล้ว
# ทางกลับกันจะเป็น import วน และการตรวจว่า overlap < max ต้องเห็นทั้งสองค่า
#
# overlap มีไว้กันบทสนทนาขาดกลางประโยคตรงรอยต่อ chunk 1_500/15_000 = 10%
# ผลข้างเคียงที่ต้องรู้: segment ท้าย chunk ถูกเล่นซ้ำที่หัว chunk ถัดไป จึงเป็นเหตุผล
# ที่ตัวกรองศัพท์ต้องทำงานครั้งเดียวก่อนหั่น chunk ไม่ใช่ทีละ chunk (ไม่งั้นนับซ้ำ)
DEFAULT_CHUNK_MAX_TOKENS = 15_000
DEFAULT_CHUNK_OVERLAP_TOKENS = 1_500

# รวมบล็อกของผู้พูดคนเดียวกันที่อยู่ติดกัน ก่อนส่งข้อความให้โมเดล (transcript.md ไม่เปลี่ยน)
#
# เปิดเป็นค่าเริ่มต้นเพราะวัดจริงแล้วทั้ง 9 ประชุม: token ที่ส่ง -33% และจำนวนคำขอ 28->20
# บนประชุม 84 นาทีคือ -43% กับ 6->4 คำขอ ซึ่งสำคัญกว่าค่า token ที่ประหยัดได้ -- เครื่อง
# inference ที่ปลายทางถูกแชร์กันทั้งองค์กร และวัดไว้แล้วว่าคิวที่นั่นตันเป็นช่วง ๆ จนคำขอ
# ขนาด 16 token ใช้เวลา 36 วินาที ทุกคำขอที่เราไม่ต้องยิงคือคิวที่คนอื่นไม่ต้องรอ
#
# ปิดเมื่อไหร่: ตอนอยากเทียบคุณภาพสรุปกับข้อความดิบ หรือเมื่อ diarization ของประชุมนั้น
# มั่วจนการรวมบล็อกยิ่งกลบความผิดพลาด (ดู docstring ของ merge_speaker_turns)
DEFAULT_MERGE_SPEAKER_TURNS = True

# "typhoon" ต้องมี .typhoon_venv ตั้งไว้แล้ว (ดู src/transcribe_typhoon.py) -- เร็วกว่า
# large-v3 บน CPU ล้วนและหลอนน้อยกว่าตามที่วัดไว้ แต่ไม่มี hotwords ทำให้พลาดศัพท์
# glossary หนักกว่า large-v3+hotwords มาก ยังเป็นตัวเลือกทดลอง ไม่ใช่ค่าเริ่มต้น
DEFAULT_ASR_ENGINE = "whisper"
KNOWN_ASR_ENGINES = (DEFAULT_ASR_ENGINE, "typhoon")


@dataclass
class Config:
    base_dir: Path
    inbox_dir: Path
    failed_dir: Path
    meetings_dir: Path
    hf_token: str
    claude_model: str = DEFAULT_SUMMARY_MODEL
    whisper_model: str = "small"
    ui_port: int = DEFAULT_UI_PORT
    ui_lang: str = DEFAULT_UI_LANG
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    speaker_match_high: float = DEFAULT_SPEAKER_MATCH_HIGH
    speaker_match_low: float = DEFAULT_SPEAKER_MATCH_LOW
    meeting_profile: str = DEFAULT_MEETING_PROFILE
    carryover_enabled: bool = DEFAULT_CARRYOVER_ENABLED
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    merge_speaker_turns: bool = DEFAULT_MERGE_SPEAKER_TURNS
    whisper_batched: bool = DEFAULT_WHISPER_BATCHED
    whisper_hotwords: bool = DEFAULT_WHISPER_HOTWORDS
    whisper_condition_on_previous_text: bool = (
        DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT
    )
    asr_engine: str = DEFAULT_ASR_ENGINE


def _read_bool(name: str, default: bool) -> bool:
    """สวิตช์เปิดปิดจาก .env -- ค่าที่อ่านไม่ออกต้องเตือน ไม่ใช่เงียบแล้วใช้ default

    ต่างจาก _read_float ตรงที่เตือนโดยเจตนา: คนที่พิมพ์ `CARRYOVER_ENABLED=flase`
    กำลังพยายาม "ปิด" อยู่ ถ้าเงียบแล้วเปิดต่อ เขาจะไม่รู้ว่ามันยังทำงาน ซึ่งเป็นฝั่ง
    ที่เสียหายกว่า เพราะเหตุผลที่คนอยากปิดคือสรุปครั้งก่อนเพี้ยนแล้วไม่อยากให้ส่งต่อ
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_WORDS:
        return True
    if value in _FALSE_WORDS:
        return False
    logger.warning(
        "%s=%r อ่านไม่ออก (ใช้ได้: %s) ใช้ค่าเริ่มต้น %s แทน / "
        "unreadable %s value %r, falling back to %s",
        name,
        raw,
        ", ".join(sorted(_TRUE_WORDS | _FALSE_WORDS)),
        default,
        name,
        raw,
        default,
    )
    return default


def _read_chunk_overlap() -> int:
    """overlap จาก .env -- ตกกลับไปที่ default พร้อมเตือนเมื่อใช้ไม่ได้

    ต้องดักที่นี่ ไม่ใช่ปล่อยไปถึง split_into_chunks: ที่นั่น overlap >= max_tokens
    raise ValueError ซึ่งจะทำให้ประชุมนั้นตกไป failed/ ทั้งที่ถอดเสียงด้วย GPU เสร็จ
    และจ่ายไปแล้ว ค่าที่พิมพ์ผิดใน .env ต้องไม่แพงขนาดนั้น (เหมือน UI_PORT)

    0 ผ่านได้: การไม่ให้ chunk คาบเกี่ยวกันเลยเป็นค่าที่ตั้งใจตั้งได้
    """
    raw = os.environ.get("CHUNK_OVERLAP_TOKENS", "").strip()
    if not raw:
        return DEFAULT_CHUNK_OVERLAP_TOKENS
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or value < 0 or value >= DEFAULT_CHUNK_MAX_TOKENS:
        logger.warning(
            "CHUNK_OVERLAP_TOKENS=%r ใช้ไม่ได้ (ต้องเป็นจำนวนเต็ม 0 ถึง %d) "
            "ใช้ค่าเริ่มต้น %d แทน / unusable CHUNK_OVERLAP_TOKENS %r, falling back to %d",
            raw,
            DEFAULT_CHUNK_MAX_TOKENS - 1,
            DEFAULT_CHUNK_OVERLAP_TOKENS,
            raw,
            DEFAULT_CHUNK_OVERLAP_TOKENS,
        )
        return DEFAULT_CHUNK_OVERLAP_TOKENS
    # เกินครึ่งของขนาด chunk = จำนวน chunk เริ่มระเบิด วัดกับ transcript 300 segment:
    # overlap 1500 -> 14 chunk, 7500 (ครึ่ง) -> 24 chunk, 14999 -> 277 chunk
    # ค่านี้ยัง "ถูกต้อง" ตามกฎ overlap < max จึงไม่ปฏิเสธ (ผู้ใช้ตั้งมาเองย่อมมีเหตุผล)
    # แต่ต้องเตือน เพราะแต่ละ chunk คือ API call ที่จ่ายจริง ไม่ควรมาเซอร์ไพรส์ตอนดูบิล
    if value > DEFAULT_CHUNK_MAX_TOKENS // 2:
        logger.warning(
            "CHUNK_OVERLAP_TOKENS=%d เกินครึ่งของขนาด chunk (%d) จำนวน chunk จะเพิ่มขึ้น "
            "อย่างรวดเร็วและแต่ละ chunk คือหนึ่ง API call ที่จ่ายเงินจริง / "
            "overlap %d exceeds half the chunk size, expect many more paid API calls",
            value,
            DEFAULT_CHUNK_MAX_TOKENS,
            value,
        )
    return value


def _read_float(name: str, default: float) -> float:
    """ค่าที่พิมพ์ผิดใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- แบบเดียวกับ UI_PORT"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def load_config(base_dir: Path | None = None) -> Config:
    base_dir = base_dir or Path.cwd()
    # load-bearing เกินหน้าที่ของ Config เอง: นี่คือ call เดียวที่ทำให้ LLM_API_KEY
    # กับ LLM_BASE_URL (และ ANTHROPIC_API_KEY) เข้าไปอยู่ใน os.environ ให้
    # src/llm.py อ่านตรงๆ ได้ (llm.py ไม่เรียก load_dotenv เอง) ลบ/ย้ายบรรทัดนี้แล้ว
    # provider จะหา key ตัวเองไม่เจอเลย ทั้งที่ไม่มีอะไรใน config.py เรียกใช้ค่าพวกนี้
    # ให้เห็นตรงๆ
    load_dotenv(base_dir / ".env")

    hf_token = os.environ["HF_TOKEN"]
    claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_SUMMARY_MODEL)
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    # พอร์ตที่ตั้งมาผิดต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- ตกกลับไปที่ default
    try:
        ui_port = int(os.environ.get("UI_PORT", DEFAULT_UI_PORT))
    except ValueError:
        ui_port = DEFAULT_UI_PORT
    ui_lang = os.environ.get("UI_LANG", DEFAULT_UI_LANG)
    # ค่าว่าง/ช่องว่างล้วนใน .env (DIARIZATION_MODEL= ที่ลืมเติมค่า) ต้องตกกลับไปที่
    # default ไม่ใช่ส่งสตริงว่างไปให้ Pipeline.from_pretrained ตายเอาตอนเปิดโปรแกรม
    diarization_model = os.environ.get("DIARIZATION_MODEL", "").strip() or DEFAULT_DIARIZATION_MODEL
    embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    # ค่าที่พิมพ์ผิดต้องรู้ตัวตอนนี้ ไม่ใช่ไปเจอตอนอ่านสรุปแล้วสงสัยว่าทำไมหน้าตาแปลก
    # -- แต่ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ แบบเดียวกับ UI_PORT และ DIARIZATION_MODEL
    meeting_profile = os.environ.get("MEETING_PROFILE", "").strip() or DEFAULT_MEETING_PROFILE
    if meeting_profile not in KNOWN_PROFILES:
        logger.warning(
            "MEETING_PROFILE=%r ไม่ใช่ประเภทประชุมที่รองรับ (%s) ใช้ %r แทน / "
            "unknown MEETING_PROFILE %r, falling back to %r",
            meeting_profile,
            ", ".join(KNOWN_PROFILES),
            DEFAULT_MEETING_PROFILE,
            meeting_profile,
            DEFAULT_MEETING_PROFILE,
        )
        meeting_profile = DEFAULT_MEETING_PROFILE
    carryover_enabled = _read_bool("CARRYOVER_ENABLED", DEFAULT_CARRYOVER_ENABLED)
    chunk_overlap_tokens = _read_chunk_overlap()
    merge_speaker_turns = _read_bool(
        "MERGE_SPEAKER_TURNS", DEFAULT_MERGE_SPEAKER_TURNS
    )
    whisper_batched = _read_bool("WHISPER_BATCHED", DEFAULT_WHISPER_BATCHED)
    whisper_hotwords = _read_bool("WHISPER_HOTWORDS", DEFAULT_WHISPER_HOTWORDS)
    whisper_condition_on_previous_text = _read_bool(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    )
    if whisper_hotwords and whisper_condition_on_previous_text:
        # ไม่ห้าม เพราะเจ้าของเครื่องอาจตั้งใจวัดซ้ำเอง แต่ต้องไม่ใช่กับดักเงียบ:
        # คู่นี้วัดแล้วสองรอบว่าทำให้ decoder ล็อกอยู่กับประโยคเดียววนจนจบไฟล์
        # (ไฟล์เต็ม 84 นาที: 98.6% ของบรรทัดเป็นประโยคทักทายเดียวกัน)
        logger.warning(
            "WHISPER_HOTWORDS=true พร้อมกับ WHISPER_CONDITION_ON_PREVIOUS_TEXT=true "
            "เป็นคู่ที่วัดแล้วว่าทำลาย transcript ทั้งไฟล์ (decoder ล็อกอยู่กับประโยค "
            "เดียววนจนจบ) -- ตั้งใจจริงให้ปล่อยไว้ ไม่งั้นตั้ง "
            "WHISPER_CONDITION_ON_PREVIOUS_TEXT=false"
        )
    asr_engine = os.environ.get("ASR_ENGINE", "").strip().lower() or DEFAULT_ASR_ENGINE
    if asr_engine not in KNOWN_ASR_ENGINES:
        logger.warning(
            "ASR_ENGINE=%r ไม่ใช่ตัวถอดเสียงที่รองรับ (%s) ใช้ %r แทน / "
            "unknown ASR_ENGINE %r, falling back to %r",
            asr_engine,
            ", ".join(KNOWN_ASR_ENGINES),
            DEFAULT_ASR_ENGINE,
            asr_engine,
            DEFAULT_ASR_ENGINE,
        )
        asr_engine = DEFAULT_ASR_ENGINE
    speaker_match_high = _read_float("SPEAKER_MATCH_HIGH", DEFAULT_SPEAKER_MATCH_HIGH)
    speaker_match_low = _read_float("SPEAKER_MATCH_LOW", DEFAULT_SPEAKER_MATCH_LOW)
    # HIGH ต้องไม่ต่ำกว่า LOW -- ถ้ากลับกัน ทุกคนที่ผ่านเกณฑ์ LOW จะผ่านเกณฑ์ HIGH ไปด้วย
    # และ Match.confident จะกลายเป็น True ของทุกคน ระบบจะเริ่มใส่ชื่อจริงลง transcript
    # ให้อัตโนมัติที่ความมั่นใจต่ำแบบเงียบ ๆ -- ตรงข้ามกับที่ค่าสองเกณฑ์นี้มีไว้กัน
    # ค่าที่พิมพ์กลับกันใน .env ต้องไม่ทำให้เปิดโปรแกรมไม่ได้ -- แบบเดียวกับ UI_PORT
    # และ _read_float จึงตกกลับไปที่ default ทั้งคู่ ไม่ใช่แค่ตัวเดียว เพราะ default
    # คู่นี้ถูกวัดมาคู่กัน ผสมค่าที่ผู้ใช้ตั้งเข้ากับ default ตัวเดียวอาจได้คู่ที่ยังกลับกันอยู่
    if speaker_match_high < speaker_match_low:
        logger.warning(
            "SPEAKER_MATCH_HIGH (%s) ต่ำกว่า SPEAKER_MATCH_LOW (%s) ซึ่งกลับด้านกัน "
            "ใช้ค่าเริ่มต้นทั้งคู่แทน (HIGH=%s, LOW=%s) / "
            "SPEAKER_MATCH_HIGH (%s) is lower than SPEAKER_MATCH_LOW (%s), which is "
            "inverted -- falling back to both defaults (HIGH=%s, LOW=%s)",
            speaker_match_high,
            speaker_match_low,
            DEFAULT_SPEAKER_MATCH_HIGH,
            DEFAULT_SPEAKER_MATCH_LOW,
            speaker_match_high,
            speaker_match_low,
            DEFAULT_SPEAKER_MATCH_HIGH,
            DEFAULT_SPEAKER_MATCH_LOW,
        )
        speaker_match_high = DEFAULT_SPEAKER_MATCH_HIGH
        speaker_match_low = DEFAULT_SPEAKER_MATCH_LOW

    return Config(
        base_dir=base_dir,
        inbox_dir=base_dir / "inbox",
        failed_dir=base_dir / "failed",
        meetings_dir=base_dir / "meetings",
        hf_token=hf_token,
        claude_model=claude_model,
        whisper_model=whisper_model,
        ui_port=ui_port,
        ui_lang=ui_lang,
        diarization_model=diarization_model,
        embedding_model=embedding_model,
        speaker_match_high=speaker_match_high,
        speaker_match_low=speaker_match_low,
        meeting_profile=meeting_profile,
        carryover_enabled=carryover_enabled,
        chunk_overlap_tokens=chunk_overlap_tokens,
        merge_speaker_turns=merge_speaker_turns,
        whisper_batched=whisper_batched,
        whisper_hotwords=whisper_hotwords,
        whisper_condition_on_previous_text=whisper_condition_on_previous_text,
        asr_engine=asr_engine,
    )
