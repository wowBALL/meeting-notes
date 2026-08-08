"""service ควบคุมการอัดที่หน้าเว็บคุยด้วย -- bind 127.0.0.1 เท่านั้น

ตัวอัดรันเป็น thread ในตัว service ไม่ใช่ process ลูก เพราะ record_streams_to_session
รับ threading.Event เป็นสัญญาณหยุดอยู่แล้ว ปุ่มปิดห้องจึงเป็น stop_event.set() ตรง ๆ
ไม่ต้องปลอมสัญญาณ Ctrl+C ข้าม process บน Windows

ผลพลอยได้ที่ตั้งใจ: ปิดหน้าเว็บระหว่างประชุมแล้วยังอัดต่อ เพราะการอัดไม่ได้อยู่
ในหน้าเว็บ เปิดใหม่เมื่อไหร่ก็เห็นสถานะเดิม
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src import activity, enroll, pending, speakers
from src.companion import Companion
from src.logsetup import UI_LOG, configure_logging
from src.messages import render
from src.record import run_recording
from src.storage import rename_speaker_in_transcript, safe_meeting_dir

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WORKER_PROBE_CACHE_SECONDS = 10
ACTIVITY_LIMIT = 200

# ทะเบียนเสียงถูกอ่าน-แก้-เขียนทับใน confirm, การตัดคิวของ skip, และ delete โดยไม่มี
# ตัวกัน ทั้งที่ service รันด้วย threaded=True -- สอง request ที่แก้พร้อมกันชนะกันได้
# ทำให้ฝ่ายที่แพ้เสียชื่อที่เพิ่งพิมพ์ไปทั้งที่ได้รับ ok กลับไปแล้ว ล็อกเดียวพอเพราะ
# ผู้ใช้มีคนเดียวและเว็บนี้ bind 127.0.0.1 เท่านั้น -- ไม่ต้องถึงขั้นล็อกระดับไฟล์
_registry_lock = threading.Lock()

# วิดเจ็ต (COWORK Desktop ตั้งแต่ v1.8.7) เปิด service นี้ด้วย pythonw.exe เพื่อไม่ให้มี
# หน้าต่างดำค้างอยู่ตลอด -- pythonw เป็น GUI subsystem จึงไม่มี console เลย ผลข้างเคียงคือ
# ลูกที่เป็น console subsystem ทุกตัวต้องถูกสร้าง console ใหม่ให้ ซึ่งบน Windows 11 จะถูก
# ส่งต่อให้ Windows Terminal -- probe ทุก 10 วินาทีจึงเด้งหน้าต่างใหม่ทุก 10 วินาที
# หน้าต่างดำที่ค้างตัวเดียวกลายเป็นหน้าต่างที่กระพริบแทน
#
# CREATE_NO_WINDOW ใช้ได้ตรงนี้เพราะลูกตัวนี้ไม่ได้ detached -- ที่แพ้ให้ defterm ของ
# Windows 11 คือกรณีที่รวมกับ DETACHED_PROCESS (เคสของฝั่งวิดเจ็ตที่ต้องแก้ด้วย pythonw.exe)
# getattr ไว้เพราะค่านี้มีเฉพาะบน Windows -- เทสต์รันบนแพลตฟอร์มอื่นได้เหมือนเดิม
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# คำสั่งเดียวกับที่ start-meeting.bat:36 ใช้ตรวจว่า watcher รันอยู่หรือไม่
_WORKER_PROBE_COMMAND = [
    "powershell",
    "-NoProfile",
    "-Command",
    "if (Get-CimInstance Win32_Process | Where-Object { "
    "$_.Name -match 'python' -and $_.CommandLine -like '*src.main*' }) "
    "{ exit 0 } exit 1",
]


def probe_worker() -> bool:
    """watcher รันอยู่หรือไม่

    เลือกเช็ค process แทนการให้ watcher เขียน heartbeat เพราะไม่ต้องแตะ
    src/watcher.py ซึ่งต้องทำงานเหมือนเดิมสำหรับทางเข้าเดิม ถ้าภายหลังอยากได้
    heartbeat ที่ถูกกว่านี้ เปลี่ยนได้โดยไม่กระทบสัญญาของ endpoint
    """
    try:
        return (
            subprocess.run(
                _WORKER_PROBE_COMMAND,
                capture_output=True,
                timeout=10,
                creationflags=_NO_WINDOW,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


# ขั้นที่งานถือการ์ดจออยู่จริง
GPU_STAGES = ("queued", "transcribe_started", "diarize_started")

# ขั้นที่แปลว่างานปล่อยการ์ดจอแล้ว -- summarize_started อยู่ในนี้ไม่ใช่เพราะงานจบ
# แต่เพราะขั้นสรุปยิงไป LiteLLM ไม่ใช่การ์ดจอเครื่องนี้ ถ้าไม่นับมัน ค่าล่าสุดของงาน
# จะค้างอยู่ที่ diarize_started ตลอดช่วงสรุป แล้วกั้นเกินจริงเป็นนาที ๆ
GPU_RELEASE_CODES = ("summarize_started", "meeting_done", "job_failed")


def gpu_is_busy(entries, worker_running: bool) -> bool:
    """การ์ดจอไม่ว่าง = watcher กำลังรัน และมีงานที่ขั้นล่าสุดยังถือการ์ดจออยู่

    worker_running ขาดไม่ได้: watcher ที่ตายกลางงานทิ้งงานค้างในคิวไว้ตลอดกาล
    ถ้าตัดเงื่อนไขนี้ออก companion จะเปิดไม่ได้อีกเลยโดยไม่มีใครรู้สาเหตุ

    อ่าน "ขั้นล่าสุดของแต่ละงาน" ไม่ใช่ "เคยเห็นขั้นนี้ไหม" -- activity ที่ส่งมาเป็นแค่
    ส่วนท้ายของไฟล์ การนับสะสมให้คำตอบผิดทันทีที่ไฟล์ถูกตัด (กฎเดียวกับ progressOf
    ใน meetingrun.js ฝั่ง COWORK Desktop)
    """
    if not worker_running:
        return False
    latest: dict[str, str] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        job = entry.get("job")
        code = entry.get("code")
        if not isinstance(job, str) or not isinstance(code, str):
            continue
        if code in GPU_STAGES or code in GPU_RELEASE_CODES:
            latest[job] = code
    return any(code in GPU_STAGES for code in latest.values())


class RecorderState:
    """สถานะของการอัดที่ service เป็นเจ้าของ

    มีแค่สามค่าโดยตั้งใจ เพราะ service เป็นเจ้าของแค่การอัด ความคืบหน้าหลังส่ง
    เข้า inbox/ เป็นของ watcher ซึ่งมาทาง activity log -- ให้ service เดาแทน
    watcher คือการสร้างสถานะที่โกหกได้
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | recording | stopping
        self.room = None
        self.model = None
        self.profile = None
        self.asr_engine = None
        self.started_at = None
        self.stop_event = None
        self.thread = None
        self.warnings = []
        self.last_result = None
        self.devices = {}
        # เดียวกับ stop_event: Event เดียวที่ recorder thread เช็คสด ๆ ทุกบล็อกเสียง
        # ไม่ใช่คำสั่งที่ต้องส่งข้าม process -- คงอยู่ข้าม recording แต่ต้อง clear()
        # ทุกครั้งที่เปิดห้องใหม่ ไม่งั้นห้องถัดไปจะเริ่มโดยไมค์ปิดอยู่แล้วอย่างเงียบ ๆ
        self.mic_muted = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = int(time.monotonic() - self.started_at) if self.started_at else 0
            return {
                "recorder": self.status,
                "room": self.room,
                "model": self.model,
                "elapsed_seconds": elapsed,
                "warnings": list(self.warnings),
                "last_result": self.last_result,
                "devices": dict(self.devices),
                "mic_muted": self.mic_muted.is_set(),
            }


def _public_sample(sample) -> dict | None:
    """ตัวอย่างเสียงหนึ่งช่วง -- ทุกใบประกาศชนิดไว้แล้วบังคับตามนั้น

    text เป็นสตริงที่ผู้ใช้อ่าน (app.js pendingHtml: esc(sample.text)) ส่วน start/end เป็น
    วินาทีตัวเลขเดี่ยวที่ app.js playSample() เอาไปตั้ง audioEl.currentTime ตรง ๆ

    ใช้ None ไม่ใช่การทิ้งคีย์: audioEl.currentTime = undefined โยน TypeError ("non-finite")
    ทั้งการ์ด ส่วน = null กลายเป็น 0 ซึ่งเป็นค่าที่ปุ่มเล่นเสียงรับได้ -- ทั้งไฟล์นี้เลือก
    None เป็นค่าปริยายด้วยเหตุผลเดียวกัน: หน้าเว็บถูกเขียนบนสัญญาที่ว่าคีย์เหล่านี้มีเสมอ
    (esc(x), x || 0, x || "") การหายไปของคีย์เป็นรูปร่างที่ยังไม่เคยมีใครทดสอบ
    """
    if not isinstance(sample, dict):
        return None
    return {
        "text": speakers.as_str(sample.get("text")),
        "start": speakers.as_number(sample.get("start")),
        "end": speakers.as_number(sample.get("end")),
    }


def _public_guess(guess) -> dict | None:
    if not isinstance(guess, dict):
        return None
    # ทั้งคู่เป็นสตริงที่ app.js เอาไปใส่ esc() ตรง ๆ (pendingHtml) -- evidence คือเหตุผล
    # ที่โมเดลเดาชื่อนี้ ซึ่งเป็นข้อความให้คนอ่าน ไม่ใช่โครงสร้างข้อมูล
    return {
        "name": speakers.as_str(guess.get("name")),
        "evidence": speakers.as_str(guess.get("evidence")),
    }


def _public_suggested(suggested) -> dict | None:
    if not isinstance(suggested, dict):
        return None
    # web/app.js อ่านแค่ suggested.name (ดู pendingHtml) -- speaker_id/score ที่
    # pending.build_pending_speakers ติดมาด้วยเป็นของฝั่งเซิร์ฟเวอร์ล้วน (ใช้ match_known
    # ตอนสร้างคิวเท่านั้น) จึงไม่มีเหตุผลต้องออกไปเลย ต่างจาก _public_guess ตรงนี้ที่ guess
    # ยังมี evidence ให้ผู้ใช้อ่านจริง
    return {"name": speakers.as_str(suggested.get("name"))}


def _public_speaker(speaker: dict) -> dict:
    """ผู้พูดหนึ่งคนในรูปที่ส่งออกหน้าเว็บได้

    allowlist ของ *ชื่อคีย์* อย่างเดียวไม่พอ (finding 1 ของรีวิวรอบที่สี่): เวอร์ชันก่อนหน้า
    เลือกคีย์ guess/samples/suggested ถูกแล้ว แต่คืน "ค่า" ของสามคีย์นั้นตรง ๆ โดยไม่กรอง
    อะไรเลย -- ไฟล์คิวแก้มือได้ตามดีไซน์ของโปรเจกต์นี้ (ดู pending.py) ใครใส่เวกเตอร์ไว้ใต้
    ชื่อคีย์ที่ไม่มีคำว่า "embedding" เลย (เช่น "voiceprint") ใน guess/samples[]/suggested
    จะหลุดออก endpoint ไปเงียบ ๆ เหมือนที่ระดับการประชุมเคยหลุดมาก่อนหน้านี้ (ดู
    _public_pending_meeting) ตอนนี้ทั้งสามจุดเป็น allowlist ของค่าจริง ๆ ผ่าน
    _public_guess/_public_sample/_public_suggested ข้างบน ตรงกับที่ web/app.js อ่านจริง
    (ดู pendingHtml/speakerAt ใน app.js): label, guess{name,evidence},
    samples[]{text,start,end}, suggested{name}, speaking_seconds -- คีย์อื่นทั้งหมด
    (diarization_id, model, embedding_model, embedding_seconds, segment_count, suggested.
    speaker_id/score, embedding หรือเวกเตอร์ใต้ชื่อไหนก็ตาม) เป็นของฝั่งเซิร์ฟเวอร์ล้วน
    ไม่มีเหตุผลต้องออกไปเลย ใช้ .get() และเช็ค isinstance ทุกชั้นเพราะไฟล์คิวที่แก้มือหรือ
    มาจากเวอร์ชันเก่ากว่านี้อาจไม่มีคีย์พวกนี้ครบ หรือมีแต่ผิดชนิด -- หน้าเว็บรับค่า None/[]
    ของทุกคีย์เหล่านี้ได้อยู่แล้ว (guess ? ... : "", samples || [], suggested ? ... : "")

    finding 1 ของรีวิวรอบที่ห้า: allowlist ข้างบนยังอยู่ต่อทั้งหมด แต่ผลลัพธ์ผ่าน
    speakers.drop_numeric_vectors อีกชั้น เพราะ allowlist กรอง "ชื่อคีย์" แล้วคืน "ค่า" ของ
    คีย์นั้นดิบ ๆ -- เวกเตอร์ที่ถูกวางเป็นค่าของ label / speaking_seconds / guess.evidence /
    samples[].start / suggested.name (ห้าจุดที่ทำซ้ำได้จริงบน endpoint ที่รันอยู่) หลุดออกไป
    ได้ทั้งที่ทุกคีย์อยู่ใน allowlist ถูกต้องแล้ว การแจกแจงชื่อคีย์เพิ่มอีกชั้นกันกรณีนี้ไม่ได้
    ไม่ว่าจะไล่ไปกี่ชั้น (ดู docstring ของ drop_numeric_vectors)

    รีวิวรอบที่หก: การกรองด้วยรูปทรงของรอบที่ห้าก็แพ้เหมือนกัน เพราะ "รูปทรง" เป็นแกนที่ห้า
    ที่คนแก้ไฟล์เลือกได้ ไม่ใช่ค่าคงที่ของข้อมูล (หกรูปแบบที่ทำซ้ำได้จริง ดู docstring ของ
    drop_numeric_vectors) ตอนนี้ทุกใบที่ออกจากฟังก์ชันนี้ *ประกาศชนิด* ไว้แล้วบังคับตามนั้น:
    label เป็น str, speaking_seconds เป็นตัวเลขเดี่ยว, ที่เหลืออยู่ใน _public_guess/
    _public_sample/_public_suggested -- ค่าที่ไม่ใช่ชนิดที่ประกาศไว้กลายเป็น None ทั้งหมด
    ตรงกับที่ app.js รับได้อยู่แล้ว (esc(speaker.label), speaker.speaking_seconds || 0)
    allowlist ชื่อคีย์และ drop_numeric_vectors ยังอยู่ครบทั้งคู่ในฐานะชั้นรอง ไม่ใช่ด่านหลัก
    """
    samples = speaker.get("samples")
    return speakers.drop_numeric_vectors(
        {
            "label": speakers.as_str(speaker.get("label")),
            "guess": _public_guess(speaker.get("guess")),
            "samples": [
                public_sample
                for public_sample in (
                    _public_sample(item)
                    for item in (samples if isinstance(samples, list) else [])
                )
                if public_sample is not None
            ],
            "suggested": _public_suggested(speaker.get("suggested")),
            "speaking_seconds": speakers.as_number(speaker.get("speaking_seconds")),
        }
    )


def _public_pending_meeting(meeting: dict) -> dict:
    """การประชุมหนึ่งรายการในคิว ในรูปที่ส่งออกหน้าเว็บได้ (finding ที่สามของรีวิวรอบนี้)

    allowlist ไม่ใช่ {**meeting, ...}: จุดที่แก้ไปแล้วสองจุด (_public_speaker ข้างบน กับ
    list_entries ใน enroll.py) เป็น allowlist แล้ว แต่ list_pending_speakers ยังสเปรดคีย์
    ระดับการประชุมทุกตัวตรง ๆ ผ่าน {**meeting, "speakers": ...} อยู่ดี -- ชั้นนี้อยู่เหนือ
    _public_speaker หนึ่งชั้น (มันกรอง speaker แต่ละคนแล้ว แต่ dict การประชุมที่ห่อ list
    นั้นไว้ไม่ถูกกรองเลย) _read_pending_file คืน JSON ดิบของไฟล์คิวทั้งก้อนไม่กรองคีย์อะไร
    เลย และไฟล์คิวแก้มือได้ตามดีไซน์ของโปรเจกต์นี้ -- ใครใส่ "embedding"/"voiceprint" ไว้
    ระดับบนสุดของ record (นอก speakers[]) จะหลุดออก endpoint ไปเงียบ ๆ

    รายการข้างล่างคือทุกคีย์ระดับการประชุมที่ web/app.js อ่านจริง (ดู pendingHtml/speakerAt
    ใน app.js: meeting.meeting_dir, meeting.speakers) คีย์อื่น (audio_file, created) เป็น
    ของฝั่งเซิร์ฟเวอร์ล้วน ไม่มีเหตุผลต้องออกไปเลย

    finding 1 ของรีวิวรอบที่ห้า: allowlist ชื่อคีย์ยังอยู่ต่อ (มันกันฟิลด์ที่ UI ไม่ได้ใช้)
    แต่ตอนนี้ผลลัพธ์ผ่าน speakers.drop_numeric_vectors อีกชั้น -- เวกเตอร์ที่วางเป็น *ค่า*
    ของ meeting_dir เอง (ซึ่งอยู่ใน allowlist) ไม่มีทางถูกกันด้วยการแจกแจงชื่อคีย์เพิ่มอีกกี่
    ชั้นก็ตาม ดู docstring ของ drop_numeric_vectors สำหรับประวัติสี่รอบที่ผ่านมา

    รีวิวรอบที่หก: meeting_dir ประกาศเป็น str แล้วบังคับตามนั้น -- app.js ใช้มันสามที่และ
    เป็นสตริงทั้งสามที่ (speakerKey() ต่อสตริง, esc(meeting.meeting_dir), และ
    encodeURIComponent(found.meeting) ตอนยิง /api/speakers/audio/) ไม่มีทางที่รูปทรงไหน
    ของเวกเตอร์จะเป็น str ได้ จึงไม่เหลือรูปทรงให้เลือกอีก
    """
    queued = meeting.get("speakers")
    return speakers.drop_numeric_vectors(
        {
            "meeting_dir": speakers.as_str(meeting.get("meeting_dir")),
            "speakers": [
                _public_speaker(s)
                for s in (queued if isinstance(queued, list) else [])
                if isinstance(s, dict)
            ],
        }
    )


# ความยาวสูงสุดของค่า param หนึ่งตัวที่ยอมให้ไปโผล่ในข้อความที่ render ออกมา (และใน
# params.path ที่ส่งออก)
#
# ทำไมต้องมีเพดาน: params เป็นฟิลด์ปลายเปิดฟิลด์เดียวที่เหลืออยู่ -- render() เอาค่าใน
# params ไปเติมลงเทมเพลต ("เสร็จแล้ว: {path}") ผลลัพธ์จึงเป็น str จริง ๆ ตามชนิดที่ text
# ประกาศไว้ การประกาศชนิดของใบจึงจับไม่ได้ตามนิยาม คนที่แก้ state/activity.jsonl ด้วยมือ
# วางสตริงที่เป็นเวกเตอร์ serialize แล้วไว้ใต้ {path} ได้ตรง ๆ
#
# ทำไม 512: param จริงในโปรเจกต์นี้มีแค่ชื่องาน (~20 ตัวอักษร) พาธไฟล์ (MAX_PATH ของ
# Windows คือ 260) ตัวนับเล็ก ๆ และข้อความ error จาก ffmpeg/OSError -- 512 กว้างพอสำหรับ
# ทุกตัวโดยยังไม่ต้องตัดอะไรที่มีประโยชน์ทิ้ง ส่วนเวกเตอร์ 256 มิติที่ serialize แล้วยาว
# เกินสองพันตัวอักษร (วัดแล้วที่ความละเอียดของ float จริง: 5,324) จึงถูกตัดจนกู้คืนไม่ได้
# เสมอ -- ตัดได้ดีที่สุดราว 9% ของเวกเตอร์เท่านั้น เพดานนี้เป็นชั้นที่สอง
# เท่านั้น -- ชั้นแรกคือ _render_param ที่ไม่ยอมให้ค่าที่ไม่ใช่ scalar ถูกเติมลงเทมเพลตเลย
# (str([0.11, 0.12]) คือการขนเวกเตอร์ออกไปในรูปสตริง จึงห้ามเด็ดขาด)
PARAM_VALUE_MAX_CHARS = 512


def _render_param(value) -> str | None:
    """ค่า param หนึ่งตัวในรูปสตริงที่เอาไปเติมเทมเพลตได้ -- None แปลว่าไม่เอาไปเติมเลย

    รับเฉพาะ scalar: สตริง ตัวเลขเดี่ยว หรือ bool ไม่มี param จริงตัวไหนในโปรเจกต์นี้ที่
    เป็น list หรือ dict (ดู activity.append/on_event ทุกจุดเรียก) ค่าที่ไม่ใช่ scalar จึง
    ถูกทิ้ง ไม่ใช่ str() ทับ -- render() รับมือกับ param ที่หายไปได้อยู่แล้วโดยไม่ raise
    (คืนเทมเพลตดิบกลับมาเมื่อ format() โยน KeyError ดู src/messages.py)
    """
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, str):
        text = value
    elif speakers.as_number(value) is not None:
        text = str(speakers.as_number(value))
    else:
        return None
    return text[:PARAM_VALUE_MAX_CHARS]


def _public_activity(entry: dict, lang: str) -> dict:
    """หนึ่งบรรทัดของ state/activity.jsonl ในรูปที่ส่งออกหน้าเว็บได้

    finding 2 ของรีวิวรอบที่ห้า -- รอบที่สี่ "ตัด job/params ทิ้ง" เพื่อกันเวกเตอร์ที่อาจแอบ
    อยู่ใน params ของบรรทัดที่ถูกแก้มือ โดยอ้าง grep ของ renderLog ตัวเดียวว่าไม่มีใครอ่าน
    สองคีย์นี้ grep นั้นไม่ครบ และการตัดทิ้งเป็น regression จริงที่ผู้ใช้เจอ ไม่ใช่รูรั่วเฉย ๆ:

      web/app.js jobProgress()                     อ่าน e.job  (แถบความคืบหน้าหลังปิดห้อง)
      web/app.js poll()                            อ่าน e.job  (สัญญาณ speakers_pending)
      COWORK Desktop/meetingrun.js progressOf()    อ่าน e.job
      COWORK Desktop/meetingrun.js finishedMeetingId()  อ่าน e.job และ e.params.path

    ผลจริงเมื่อ job หายไป: หน้าจอค้างที่ "กำลังประมวลผล" ขั้นที่ 1 ตลอดกาล viewDone ไปไม่ถึง
    และเพราะ viewProcessing ไม่ได้ render pendingHtml() คิวตั้งชื่อผู้พูด -- เหตุผลทั้งหมดที่
    ฟีเจอร์นี้มีอยู่ -- จึงเข้าไม่ถึงจากหน้าเว็บเลย ฝั่งวิดเจ็ตแถบความคืบหน้าไม่ขยับ และปุ่ม
    "เปิดโฟลเดอร์ประชุม" คืน null (tests/meetingrun.test.js ยึดฟิลด์พวกนี้ไว้เป็นสัญญา)

    ทั้งคู่จึงกลับมา แล้วให้ speakers.drop_numeric_vectors เป็นตัวทำให้ปลอดภัยแทนการตัดทิ้ง
    -- อนึ่ง job เป็นสตริงชื่องาน การตัดมันทิ้งจึงไม่เคยกันเวกเตอร์อะไรได้ตั้งแต่แรก

    กรองก่อน render ไม่ใช่หลัง: render() เอาค่าใน params ไปเติมลงเทมเพลตข้อความ ("เสร็จแล้ว:
    {path}") ถ้าส่ง params ดิบเข้าไป เวกเตอร์ที่วางไว้ใต้ {path} จะออกไปเป็น *สตริง* ในคีย์
    text ซึ่งการ์ดรูปทรงจับไม่ได้เพราะไม่ใช่ array อีกต่อไป

    coercion ที่เหลือเป็นของถูก ๆ ที่ป้องกัน 500 ตรง ๆ: code ที่ไม่ใช่สตริงทำให้
    catalog.get(code) โยน TypeError (unhashable) และ params ที่ไม่ใช่ dict ทำให้
    format(**params) โยน TypeError ทั้งสองตัวหลุดออกไปเป็น 500 ทั้งหน้าจากบรรทัดเดียวที่ถูก
    แก้มือ ทั้งที่ฟังก์ชันนี้อ่านไฟล์ที่ "แก้มือได้ตามดีไซน์" อยู่แล้ว

    รีวิวรอบที่หก -- สองการรั่วคนละทาง ต้องปิดคนละแบบ:

    (1) params ที่ส่งออก: ตรวจครบทั้งสามฝั่งที่บริโภค /api/state แล้ว (web/app.js,
        web/enroll.js, D:\\COWORK\\COWORK Desktop\\meetingrun.js) ไม่มีใครอ่านอะไรใน params
        นอกจาก params.path ซึ่งเป็นสตริง (meetingrun.js finishedMeetingId: e.params &&
        e.params.path แล้ว String(last.params.path).split(...)) การส่ง params ทั้งก้อนออกไป
        จึงเป็นการเปิดฟิลด์ปลายเปิดไว้เปล่า ๆ -- ตอนนี้ออกไปแค่ path ตัวเดียวและเฉพาะเมื่อ
        มันเป็นสตริงจริง params ที่ไม่มี path กลายเป็น {} ซึ่ง meetingrun.js กรองทิ้งเองอยู่แล้ว

    (2) text: render() เอาค่าใน params ไปเติมลงเทมเพลต ผลลัพธ์จึงเป็น str ตามชนิดที่ text
        ประกาศไว้เป๊ะ ๆ การประกาศชนิดของใบจับทางนี้ไม่ได้ตามนิยาม ต้องกรองที่ *ขาเข้า* ของ
        render แทน: _render_param ทิ้งทุกค่าที่ไม่ใช่ scalar และตัดความยาวที่
        PARAM_VALUE_MAX_CHARS (ดูเหตุผลของตัวเลขที่นั่น)
    """
    safe = speakers.drop_numeric_vectors(entry)
    code = speakers.as_str(safe.get("code")) or ""
    raw_params = safe.get("params")
    raw_params = raw_params if isinstance(raw_params, dict) else {}
    render_params = {}
    for key, value in raw_params.items():
        # คีย์ที่ไม่ใช่สตริงทำให้ format(**params) โยน TypeError -- และไม่มีเทมเพลตไหน
        # อ้างถึงมันได้อยู่แล้ว
        if not isinstance(key, str):
            continue
        text = _render_param(value)
        if text is not None:
            render_params[key] = text
    path = _render_param(speakers.as_str(raw_params.get("path")))
    return {
        "ts": speakers.as_str(safe.get("ts")),
        "job": speakers.as_str(safe.get("job")),
        "code": code,
        "level": speakers.as_str(safe.get("level")),
        "params": {} if path is None else {"path": path},
        "text": render(code, render_params, lang),
    }


def _speaker_summary(speaker: dict) -> dict:
    """คนหนึ่งคนในทะเบียน ในรูปสรุปที่ /api/speakers และ /api/enroll ใช้ร่วมกัน (finding C
    ของรีวิวรอบสุดท้าย -- คนละรอบกับ finding 1-6 ที่แก้ไปก่อนหน้านี้)

    ไม่ใช้ _public_speaker ตรง ๆ เพราะคนละ shape กันคนละเรื่อง (finding 3 ของรีวิวรอบที่ห้า
    -- ข้อความเดิมตรงนี้บอกว่า _public_speaker "เก็บทุกคีย์ยกเว้น embedding" ซึ่งเลิกจริง
    มาตั้งแต่รีวิวรอบที่สองแล้ว: มันเป็น allowlist ของ label/guess/samples/suggested/
    speaking_seconds มาสองรอบ ไม่ใช่ denylist ของ embedding อีกต่อไป) ที่ใช้แทนกันไม่ได้คือ
    รูปของข้อมูล: ผู้พูดที่รอตั้งชื่อมีตัวอย่างเสียงเป็น samples[]{text,start,end} ที่ผู้ใช้
    ต้องอ่าน ส่วน entry ในทะเบียนมี samples[] เป็นเวกเตอร์เสียงล้วน (samples[].embedding)
    ซึ่งไม่มีอะไรให้ผู้ใช้อ่านเลยและเป็นข้อมูล biometric ทั้งก้อน -- ที่นี่จึงเหลือแค่จำนวน

    ไม่มี drop_numeric_vectors ห่อไว้ (เอาออกในรีวิวรอบที่หก) เพราะมันไม่เคยกันอะไรได้ตรงนี้
    เลยและการปล่อยไว้ทำให้รายงานอ่านเหมือนมีการป้องกันที่ไม่มีจริง: ทั้งสามค่าถูกกำหนดชนิด
    ตั้งแต่ต้นทาง -- load_registry ปล่อยผ่านเฉพาะ entry ที่ id/name เป็น str และ samples
    เป็น list (isinstance ครบทั้งสามคีย์) ส่วน sample_count คือ len() ซึ่งเป็น int เสมอ
    นี่คือแพตเทิร์นเดียวกับที่ทำให้ /api/speakers เป็น endpoint เดียวที่ไม่เคยถูกเจาะเลย
    ตลอดหกรอบ -- ประกาศชนิดที่ขอบแล้วบังคับตามนั้น ไม่ใช่ไล่เดาว่าค่าไหนหน้าตาเหมือนเวกเตอร์
    """
    return {
        "id": speaker["id"],
        "name": speaker["name"],
        "sample_count": len(speaker.get("samples", [])),
    }


def create_app(
    config,
    recorder=run_recording,
    worker_probe=probe_worker,
    companion_factory=Companion,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    state = RecorderState()
    activity.trim(config.base_dir)
    probe_cache = {"value": False, "at": 0.0}

    def worker_ready() -> bool:
        # การเช็คเป็นการ spawn powershell หนึ่งตัว หน้าเว็บ poll ทุกวินาที
        # จึง cache ไว้ ไม่งั้นจะ spawn 60 ตัวต่อนาทีเพื่อตอบคำถามเดิม
        now = time.monotonic()
        if now - probe_cache["at"] > WORKER_PROBE_CACHE_SECONDS:
            probe_cache["value"] = worker_probe()
            probe_cache["at"] = now
        return probe_cache["value"]

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/state")
    def get_state():
        # ภาษามาจากหน้าเว็บ แล้ว service เป็นคนแปลงรหัสเป็นคำพูด -- ถ้าให้ JS
        # มี catalog ของตัวเองจะกลายเป็นสองชุดที่ต้องแก้พร้อมกันเสมอ
        lang = request.args.get("lang") or config.ui_lang
        body = state.snapshot()
        body["worker_ready"] = worker_ready()
        body["lang"] = lang
        body["warnings"] = [
            {**w, "text": render(w["code"], w.get("params"), lang)}
            for w in body["warnings"]
        ]
        # allowlist ไม่ใช่ {**e, ...}: entry มาจาก state/activity.jsonl ตรง ๆ (activity.tail)
        # ซึ่งแก้มือได้ตามดีไซน์เดียวกับไฟล์คิว (ดู activity.append) -- ฟิลด์ที่ออกไปกับ
        # เหตุผลที่ job/params ต้องอยู่ต่อ ดูที่ _public_activity ด้านบน
        body["activity"] = [
            _public_activity(e, lang)
            for e in activity.tail(config.base_dir, ACTIVITY_LIMIT)
        ]
        return jsonify(body)

    @app.post("/api/session")
    def open_room():
        payload = request.get_json(silent=True) or {}
        with state.lock:
            if state.status != "idle":
                return jsonify({"error": "already_recording"}), 409
            state.status = "recording"
            state.room = (payload.get("name") or "").strip() or None
            state.model = payload.get("model")
            # ไม่ validate ที่นี่ เหมือนที่ไม่ validate ชื่อโมเดล -- ฝั่งสรุปเป็นคน
            # ตัดสินใจกับค่าที่ไม่รู้จัก (เตือนแล้วใช้ dev) การปฏิเสธที่นี่จะทำให้
            # กดเปิดห้องไม่ได้เพราะค่าที่แก้ทีหลังได้ ทั้งที่ประชุมกำลังจะเริ่ม
            state.profile = payload.get("profile")
            # ไม่ validate เหมือนกัน แบบเดียวกับ model/profile -- ฝั่ง pipeline เป็นคน
            # ตัดสินใจกับค่าที่ไม่รู้จัก (ใช้ whisper ต่อ) การปฏิเสธที่นี่จะทำให้กดเปิด
            # ห้องไม่ได้เพราะค่าที่แก้ทีหลังได้ ทั้งที่ประชุมกำลังจะเริ่ม
            state.asr_engine = payload.get("asr_engine")
            state.started_at = time.monotonic()
            state.stop_event = threading.Event()
            state.warnings = []
            state.last_result = None
            state.devices = {}
            state.mic_muted.clear()
            stop_event = state.stop_event
            mic_muted = state.mic_muted
            room, model, profile, asr_engine = (
                state.room,
                state.model,
                state.profile,
                state.asr_engine,
            )

        # companion ผูกกับ "ครั้งที่อัด" ไม่ใช่กับ service -- ห้องใหม่ได้ตัวใหม่เสมอ
        # เป็นตัวแปรท้องถิ่นไม่ใช่ state เพราะไม่มีใครนอก thread นี้ต้องเห็นมัน และการ
        # เก็บลง state จะเปิดโอกาสให้ห้องถัดไปมองเห็นตัวที่ตายไปแล้ว
        #
        # ไม่ตั้งค่าไว้ = ไม่สร้างอะไรเลย ไม่ใช่สร้างตัวเปล่าที่สั่งอะไรก็ไม่เกิดผล --
        # เครื่องที่ไม่ใช้ฟีเจอร์นี้ต้องเดินเส้นทางเดิมเป๊ะ ไม่ใช่เส้นทางใหม่ที่บังเอิญ
        # ไม่ทำอะไร
        companion = None
        if config.companion_command:
            try:
                companion = companion_factory(config.companion_command, config.base_dir)
            except Exception:
                logger.exception("สร้างตัวคุมโปรเซสข้างเคียงไม่สำเร็จ -- ประชุมเดินต่อ")
        companion_stopped = threading.Event()

        def stop_companion():
            # ถูกเรียกได้จากสองทาง (on_event ของ recorder thread และ finally ของ
            # thread เดียวกัน) -- Event กันไม่ให้สั่งซ้ำ
            if companion is None or companion_stopped.is_set():
                return
            companion_stopped.set()
            companion.stop()

        def on_event(code, params=None, level="info"):
            # ไมค์/ลำโพงที่ถูกดักฟังจริงต้องเห็นได้ตลอดการอัด ไม่ใช่ไปขุดใน log --
            # การอัดจากอุปกรณ์ผิดตัวคือสาเหตุอันดับหนึ่งของเคส "ไม่มีเสียง"
            if code == "devices_selected":
                with state.lock:
                    state.devices = dict(params or {})
            if level in ("warn", "error"):
                with state.lock:
                    state.warnings.append({"code": code, "params": params or {}})
            # ปิดตรงนี้ ไม่ใช่ตอนตัวอัดคืนค่า: ช่วง encode คือหน้าต่างเวลาที่กว้างพอ
            # ให้ทรัพยากรระดับ process ถูกคืนทันก่อนขั้นถัดไปจะขอใช้ (วัดจากประชุม
            # จริง 62-69 วินาที) ส่วนช่วงจาก encode_done ถึงงานถัดไปแคบเกินไป (~2 วิ)
            if code == "encode_started":
                stop_companion()
            activity.append(config.base_dir, room or "unnamed", code, level, params)

        def work():
            # ตัวอัดที่ระเบิดต้องไม่ทิ้งหน้าจอค้างที่ "กำลังอัด" ตลอดไป -- สถานะ
            # ต้องกลับไป idle ไม่ว่าจะจบทางไหน
            #
            # start ที่นี่ไม่ใช่ใน request: ผู้ใช้ไม่ควรต้องรอโปรเซสของเสริมเปิดก่อน
            # ถึงจะได้ 201 กลับไป
            if companion is not None:
                companion.start({"MEETING_ROOM": room or ""})
            try:
                result = recorder(
                    room,
                    model,
                    config,
                    stop_event,
                    on_event,
                    mic_muted=mic_muted,
                    profile=profile,
                    asr_engine=asr_engine,
                )
            except Exception:
                logger.exception("ตัวอัดล้มระหว่างทำงาน")
                result = None
            finally:
                # ตาข่ายกันตกสำหรับเส้นทางที่ไม่เคยยิง encode_started เลย
                # (ตัวอัดพังกลางทาง / session ถูกทิ้งเพราะไม่มีเสียงจริง)
                stop_companion()
            with state.lock:
                state.status = "idle"
                state.started_at = None
                state.last_result = str(result) if result else None
                state.stop_event = None

        thread = threading.Thread(target=work, daemon=True)
        with state.lock:
            state.thread = thread
        thread.start()
        return jsonify({"ok": True}), 201

    @app.post("/api/session/stop")
    def stop_room():
        with state.lock:
            if state.status != "recording" or state.stop_event is None:
                return jsonify({"error": "not_recording"}), 409
            state.status = "stopping"
            state.stop_event.set()
        return jsonify({"ok": True}), 202

    @app.post("/api/session/mic")
    def set_mic_muted():
        # ปิดได้เฉพาะตอนกำลังอัดจริง: ตอน idle ไม่มี mic_muted.clear() ของห้องถัดไป
        # มาล้างค่านี้ทิ้ง ถ้ายอมให้ตั้งตอน idle ได้ ค่าจะรั่วข้ามไปห้องหน้าโดยไม่ตั้งใจ
        payload = request.get_json(silent=True) or {}
        muted = bool(payload.get("muted"))
        with state.lock:
            if state.status != "recording":
                return jsonify({"error": "not_recording"}), 409
            if muted:
                state.mic_muted.set()
            else:
                state.mic_muted.clear()
            room = state.room
        activity.append(
            config.base_dir, room or "unnamed", "mic_muted" if muted else "mic_unmuted"
        )
        return jsonify({"ok": True, "muted": muted}), 200

    @app.get("/api/speakers/pending")
    def list_pending_speakers():
        meetings = [
            _public_pending_meeting(meeting)
            for meeting in pending.load_all_pending(config.base_dir)
        ]
        return jsonify({"meetings": meetings})

    @app.get("/api/speakers")
    def list_speakers():
        return jsonify(
            {
                "speakers": [
                    _speaker_summary(speaker)
                    for speaker in speakers.load_registry(config.base_dir)
                ]
            }
        )

    @app.patch("/api/speakers/<speaker_id>")
    def rename_speaker(speaker_id):
        """แก้ชื่อคนในทะเบียน -- ไม่แตะตัวอย่างเสียงเลย

        มีไว้เพราะชื่อครั้งแรกมาจากชื่อไฟล์ที่วางใน enroll/ ซึ่งมักติดส่วนเกินมาด้วย
        ก่อนหน้านี้ทางแก้เดียวคือลบทิ้งแล้วอัดใหม่ ซึ่งทำลายตัวอย่างเสียงที่สะสมไว้
        ทั้งหมดเพียงเพราะสะกดชื่อผิด

        อยู่ในล็อกเดียวกับ confirm/delete: อ่าน -> แก้ -> เขียนทับ ถ้าไม่คุมช่วงนี้
        การแก้ชื่อที่ชนกับการ enroll พร้อมกันจะทำให้ฝ่ายหนึ่งหายไปทั้งที่ได้ ok กลับไป
        """
        payload = request.get_json(silent=True) or {}
        name = payload.get("name")
        if not isinstance(name, str):
            return jsonify({"error": "bad_name"}), 400
        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
            try:
                updated = speakers.rename_speaker(registry, speaker_id, name)
            except speakers.DuplicateNameError:
                # ต้องมาก่อน ValueError เพราะสืบทอดมาจากมัน -- สลับลำดับเมื่อไหร่
                # ผู้ใช้จะได้ "ชื่อว่าง" ทั้งที่พิมพ์ชื่อที่ซ้ำกับคนอื่น
                return jsonify({"error": "duplicate_name"}), 409
            except ValueError:
                return jsonify({"error": "bad_name"}), 400
            if updated is None:
                return jsonify({"error": "not_found"}), 404
            speakers.save_registry(config.base_dir, updated)
            renamed = next(s for s in updated if s["id"] == speaker_id)
        return jsonify({"ok": True, "speaker": _speaker_summary(renamed)})

    @app.delete("/api/speakers/<speaker_id>")
    def delete_speaker(speaker_id):
        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
            remaining = speakers.remove_speaker(registry, speaker_id)
            if len(remaining) == len(registry):
                return jsonify({"error": "not_found"}), 404
            speakers.save_registry(config.base_dir, remaining)
        return jsonify({"ok": True})

    @app.post("/api/speakers/confirm")
    def confirm_speaker():
        """ตั้งชื่อผู้พูดหนึ่งคน หรือข้ามไป

        ลำดับสำคัญ: อ่านข้อมูลมาตรวจให้ผ่านและบันทึกทะเบียนให้สำเร็จก่อน ค่อยตัดคนนั้น
        ออกจากคิว -- ตัดออกก่อนแล้วบันทึกพลาดคือการทำให้ผู้ใช้เสียทั้งชื่อที่เพิ่งพิมพ์
        และรายการที่จะกลับมาพิมพ์ใหม่
        """
        payload = request.get_json(silent=True) or {}
        meeting = payload.get("meeting")
        label = payload.get("label")
        if not isinstance(meeting, str) or not isinstance(label, str) or not label:
            return jsonify({"error": "bad_request"}), 400

        entry = pending.find_pending(config.base_dir, meeting, label)
        if entry is None:
            return jsonify({"error": "not_found"}), 404

        if payload.get("skip") is True:
            # ทางนี้ไม่ได้ทำอะไรอย่างอื่นเลย ถ้าตัดคิวไม่สำเร็จก็แปลว่าไม่มีอะไรเกิดขึ้น
            # จริง ๆ การตอบ ok จึงเป็นการโกหก
            with _registry_lock:
                dequeued = pending.resolve_pending(config.base_dir, meeting, label)
            if not dequeued:
                return jsonify({"error": "resolve_failed"}), 500
            return jsonify({"ok": True, "renamed": False, "name": None})

        # find_pending คืนสิ่งที่อ่านจากไฟล์คิวตรง ๆ โดยไม่ตรวจอะไรเลย (ดู docstring
        # ของมัน) -- ไฟล์คิวที่ถูกแก้มือหรือมาจากเวอร์ชันเก่ากว่านี้อาจไม่มีคีย์
        # embedding เลย หรือมีแต่เป็นเวกเตอร์ศูนย์ (pyannote pad เข้ามาเมื่อ label เกิน
        # จำนวน centroid) ปล่อยให้หลุดไปถึง save_registry จะได้ KeyError (500 ที่ไม่มี
        # ใครอธิบาย) หรือแย่กว่านั้นคือเก็บเวกเตอร์ศูนย์ลงทะเบียนถาวรซึ่ง "เหมือน" กับ
        # เวกเตอร์ศูนย์อื่นทุกตัว ต้องเช็คให้ผ่านก่อนแตะทะเบียนเลย
        if not speakers.is_usable_embedding(entry.get("embedding")):
            return jsonify({"error": "bad_embedding"}), 400

        # ป้ายพื้นที่เวกเตอร์เป็นเงื่อนไขก่อนแตะทะเบียน ไม่ใช่หลังจากนั้น: sample ที่ไม่มีป้าย
        # จะถูก match_known ข้ามตลอดกาล ผู้ใช้จะเห็นเป็น "กดยืนยันแล้วแต่ระบบไม่จำ" ซึ่งไม่มี
        # อะไรอธิบายได้เลย -- ปฏิเสธตรงนี้พร้อมเหตุผลเสียหายน้อยกว่ามาก ต้องเป็นรหัสของตัวเอง
        # ไม่ใช่ bad_embedding ข้างบน: เวกเตอร์ตัวนี้ใช้ได้ (ผ่านด่านนั้นมาแล้ว) ปัญหาคือไม่รู้ว่า
        # มันอยู่พื้นที่ไหน ซึ่งเป็นคนละเหตุผลกันโดยสิ้นเชิงและต้องบอกผู้ใช้คนละเรื่อง
        if speakers.sample_embedding_model(entry) is None:
            return jsonify({"error": "missing_embedding_model"}), 400

        name = payload.get("name")
        speaker_id = payload.get("speaker_id")
        # ล็อกคุมช่วง อ่านทะเบียน -> แก้ -> เขียนทับ เท่านั้น: service รันด้วย
        # threaded=True และสอง request ที่แก้พร้อมกันแบบไม่มีล็อกจะชนะกันได้ ทำให้ฝ่าย
        # แพ้เสียชื่อที่เพิ่งพิมพ์ไปทั้งที่ได้ ok กลับไปแล้ว
        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
            if isinstance(speaker_id, str) and speaker_id:
                existing = next((s for s in registry if s["id"] == speaker_id), None)
                if existing is None:
                    return jsonify({"error": "unknown_speaker"}), 404
                name = existing["name"]
            if not isinstance(name, str) or not speakers.clean_name(name):
                return jsonify({"error": "bad_name"}), 400
            cleaned = speakers.clean_name(name)

            speakers.save_registry(
                config.base_dir,
                speakers.add_sample(
                    registry,
                    cleaned,
                    {
                        "embedding": entry["embedding"],
                        # ป้ายจากคิว ไม่ใช่ config.embedding_model ตอนนี้ -- คิวอยู่ข้ามวันได้
                        # ผู้ใช้สลับ EMBEDDING_MODEL ระหว่างนั้นได้ ป้ายที่ติดตอนสร้างคิวคือ
                        # ป้ายเดียวที่บอกความจริงเรื่องพื้นที่เวกเตอร์ (ผ่านด่านข้างบนมาแล้ว
                        # ว่าไม่ใช่ None)
                        "embedding_model": speakers.sample_embedding_model(entry),
                        "embedding_seconds": entry.get("embedding_seconds"),
                        "segment_count": entry.get("segment_count"),
                        # โมเดลแยกผู้พูด (diarization) ที่ติดมากับคิว -- เป็น provenance
                        # เสริมเท่านั้นตอนนี้ (ดู speakers.add_sample) ไม่ใช่ป้ายที่
                        # match_known ใช้กรองอีกต่อไป นั่นคือ embedding_model ด้านบน
                        "model": entry.get("model"),
                    },
                    source=meeting,
                ),
            )

        # การแก้ไฟล์เก่าเป็นของแถม ทะเบียนคือของจริงเพราะมันไปออกดอกที่การประชุม
        # ครั้งหน้า -- โฟลเดอร์ที่ผู้ใช้ย้ายไปแล้วจึงต้องไม่ทำให้คำขอนี้ล้มเหลว การรีไรต์
        # transcript ไม่ต้องอยู่ในล็อก: มันแตะไฟล์คนละไฟล์กับทะเบียน/คิว ไม่มีอะไรแข่งกัน
        meeting_dir = safe_meeting_dir(config.meetings_dir, meeting)
        renamed = (
            rename_speaker_in_transcript(meeting_dir, label, cleaned)
            if meeting_dir is not None
            else False
        )
        # ทะเบียนถูกเซฟไปแล้วตรงนี้ = เสียงถูกจำแล้วจริง การตัดคิวไม่สำเร็จจึงต้องไม่
        # กลายเป็น 500 ที่บอกผู้ใช้ว่าล้มเหลว เพราะเขาจะพิมพ์ใหม่ และถ้าพิมพ์ชื่อไม่
        # เหมือนเดิมจะได้คนซ้ำในทะเบียนสำหรับเสียงเดียวกัน ปล่อยให้รายการค้างไว้แล้ว
        # โผล่ในหน้าเว็บรอบหน้าดีกว่า -- ผู้ใช้กดข้ามได้ และเสียงก็จำไปแล้ว
        with _registry_lock:
            dequeued = pending.resolve_pending(config.base_dir, meeting, label)
        if not dequeued:
            logger.warning(
                "จำเสียง %s สำเร็จแล้วแต่ตัดออกจากคิวไม่ได้ (%s/%s)", cleaned, meeting, label
            )
        return jsonify({"ok": True, "renamed": renamed, "name": cleaned})

    @app.get("/api/speakers/audio/<path:meeting>")
    def speaker_audio(meeting):
        """ไฟล์เสียงของการประชุมที่ยังมีคนรอตั้งชื่อ ให้กดฟังเพื่อนึกออกว่าใครพูด

        ชื่อไฟล์อ่านจากรายการในคิว ไม่ใช่จากการไล่ดูว่ามีไฟล์อะไรในโฟลเดอร์ -- ผลพลอยได้
        คือ endpoint นี้เสิร์ฟได้เฉพาะการประชุมที่มีงานค้างจริง ไม่ได้กลายเป็นช่องอ่าน
        ไฟล์ทั่ว meetings/
        """
        # อ่านเฉพาะไฟล์ของการประชุมนี้ ไม่ใช่ load_all_pending: ตัวเล่นเสียงยิง Range
        # request หลายครั้งตอน seek และ load_all_pending จะ glob + parse ไฟล์คิว
        # "ทุกไฟล์" (พร้อมเวกเตอร์เสียงข้างใน) ใหม่ทุกครั้งเพื่อหาสตริงเดียว
        record = pending.load_pending(config.base_dir, meeting)
        if record is None:
            return jsonify({"error": "not_found"}), 404
        audio_file = record.get("audio_file")
        if not isinstance(audio_file, str) or audio_file != Path(audio_file).name:
            return jsonify({"error": "not_found"}), 404
        # หมายเหตุที่คนอ่านต้องรู้: วันนี้ safe_meeting_dir ปฏิเสธอะไรไม่ได้เลยที่นี่
        # เพราะ load_pending ข้างบนผ่าน _is_safe_name มาแล้ว ซึ่งบังคับให้ meeting เป็น
        # ชื่อไฟล์เปล่า ๆ (วัดแล้ว: ทุกชื่อที่ผ่าน _is_safe_name ผ่านตัวนี้หมด) เก็บไว้
        # เป็นชั้นที่สองโดยตั้งใจ เพราะสองด่านนี้อยู่คนละไฟล์และถูกแก้แยกกันได้ -- แต่
        # อย่าเข้าใจผิดว่านี่คือด่านหลัก ถ้าจะแก้ _is_safe_name ให้หลวมลง ต้องกลับมา
        # ดูตรงนี้ว่ายังกันได้จริงไหม
        directory = safe_meeting_dir(config.meetings_dir, meeting)
        if directory is None or not (directory / audio_file).is_file():
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(directory, audio_file, conditional=True)

    @app.get("/enroll")
    def enroll_page():
        # หน้าแยกโดยเจตนา ไม่มีลิงก์จากหน้าจอ idle -- หน้าจอเดิมไม่ถูกแตะเลยแม้แต่
        # บรรทัดเดียว ต้องมาก่อน route catch-all ข้างล่างไม่งั้นถูกกลืน
        return send_from_directory(WEB_DIR, "enroll.html")

    @app.get("/api/enroll")
    def list_enroll():
        registry = speakers.load_registry(config.base_dir)
        entries = enroll.list_entries(config.base_dir)
        # เทียบกับคนที่มีอยู่แล้วในทะเบียนสำหรับไฟล์ที่ลงทะเบียนได้ (finding B ของรีวิว
        # รอบสุดท้าย: สเปกต้องการคะแนนความคล้ายแต่ไม่เคยถูกสร้างจริง) ทำที่นี่บนเซิร์ฟเวอร์
        # เท่านั้น -- entries ที่ enroll.list_entries คืนมาตัด embedding ออกไปแล้ว จึงต้อง
        # ไปอ่าน result.json ดิบอีกรอบเพื่อเอาเวกเตอร์มาเทียบ แล้วส่งกลับไปแค่ชื่อกับคะแนน
        # ที่ปัดแล้ว -- เวกเตอร์ตัวจริงต้องไม่ออกจากฟังก์ชันนี้เลย ใช้ match_known/
        # cosine_similarity ของ src/speakers.py ตัวเดิม ไม่เขียนตัวเทียบความเหมือนซ้ำ
        # และใช้เกณฑ์ config.speaker_match_high/low ตัวเดิม ไม่ตั้งเกณฑ์ใหม่
        for entry in entries:
            if entry.get("status") != "ok":
                continue
            result = enroll.read_result(config.base_dir, entry["audio_file"])
            embedding = result.get("embedding") if result else None
            if not speakers.is_usable_embedding(embedding):
                continue
            # ผลที่ไม่มีป้ายพื้นที่เวกเตอร์ (มาจากก่อนฟีเจอร์นี้ หรือถูกแก้มือ) ต้องถูกข้าม
            # ไปเงียบ ๆ ตรงนี้ ไม่ใช่ raise เหมือน confirm_speaker/confirm_enroll -- endpoint
            # นี้แค่โชว์คะแนนความคล้ายให้ดูเฉย ๆ ไม่ได้เขียนอะไรลงทะเบียนเลย ไฟล์ที่ยืนยันพื้นที่
            # ไม่ได้ก็แค่ไม่มีคะแนนให้โชว์ ไม่ใช่เหตุผลที่ทำให้ทั้งหน้าพัง
            stamp = speakers.sample_embedding_model(result)
            if stamp is None:
                continue
            matches = speakers.match_known(
                {entry["audio_file"]: embedding},
                registry,
                config.speaker_match_high,
                config.speaker_match_low,
                # เวกเตอร์นี้มาจาก result.json ที่วิเคราะห์ไว้ตอนไหนก็ได้ -- เทียบกับ
                # ทะเบียนในพื้นที่ของ *มัน* ไม่ใช่ของโมเดลที่ตั้งอยู่ตอนนี้ ป้ายมาจาก
                # ผลลัพธ์เอง (เช็คผ่านด่านข้างบนแล้วว่าไม่ใช่ None) ไม่ใช่ config.embedding_model
                embedding_model=stamp,
            )
            match = matches.get(entry["audio_file"])
            if match is not None:
                # ประกาศชนิดเหมือนทุกใบอื่น แม้ค่าทั้งสามจะมาจาก Match ที่ประกอบขึ้นบน
                # เซิร์ฟเวอร์เอง (name มาจากทะเบียนซึ่ง load_registry การันตีว่าเป็น str
                # แล้ว) -- score ต่างจากใบอื่นตรงที่ทิ้งทั้ง match ไม่ใช่ปล่อยเป็น None
                # เพราะ enroll.js เรียก file.match.score.toFixed(2) โดยไม่มีการ์ด ทั้ง
                # null และ undefined จะทำให้ทั้งหน้าพัง ส่วน "ไม่มี match" เป็นสถานะที่
                # หน้านั้นรองรับอยู่แล้ว (if (file.match))
                score = speakers.as_number(match.score)
                name = speakers.as_str(match.name)
                if score is not None and name is not None:
                    entry["match"] = {
                        "name": name,
                        "score": round(score, 2),
                        "confident": speakers.as_bool(match.confident),
                    }
        return jsonify(
            {
                "files": entries,
                # หน้านี้สั่งงานให้ watcher ทำ ถ้า watcher ไม่ได้รัน ไฟล์จะค้างที่
                # "กำลังวิเคราะห์" ตลอดไปโดยไม่มีอะไรบอกผู้ใช้ว่าทำไม
                "worker": worker_ready(),
                # ค่าเดียวกับที่ enroll.analyze ใช้ตัดสินสถานะ too_short ส่งมาให้หน้าเว็บ
                # แสดงในข้อความ แทนที่จะฝังตัวเลขซ้ำไว้เป็นสตริงคงที่อีกชุดหนึ่งในสอง
                # ภาษา (finding 5 ของรีวิวรอบสุดท้าย) -- ค่าคงที่มีแหล่งเดียวคือ
                # src/speakers.py
                "min_speaking_seconds": speakers.MIN_SPEAKING_SECONDS,
                "speakers": [_speaker_summary(speaker) for speaker in registry],
            }
        )

    @app.post("/api/enroll/analyze")
    def analyze_enroll():
        """สั่งวิเคราะห์ -- เขียนแค่ใบสั่งงาน งานจริงทำที่ watcher

        ตอบ 202 ไม่ใช่ 200 เพราะยังไม่มีอะไรเสร็จ หน้าเว็บต้อง poll ต่อ
        """
        payload = request.get_json(silent=True) or {}
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            return jsonify({"error": "bad_request"}), 400
        if not all(enroll.is_safe_filename(name) for name in files):
            return jsonify({"error": "bad_request"}), 400
        queued = [
            name for name in files if enroll.write_request(config.base_dir, name)
        ]
        if not queued:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"ok": True, "queued": queued}), 202

    @app.post("/api/enroll/confirm")
    def confirm_enroll():
        """ยืนยันชื่อแล้วเก็บเสียงเข้าทะเบียน

        ลำดับเดียวกับ /api/speakers/confirm: ตรวจให้ผ่านและบันทึกทะเบียนให้สำเร็จก่อน
        แล้วค่อยย้ายไฟล์ -- ย้ายก่อนแล้วบันทึกพลาดคือผู้ใช้เสียทั้งชื่อที่พิมพ์และไฟล์
        ที่จะลองใหม่
        """
        payload = request.get_json(silent=True) or {}
        audio_file = payload.get("audio_file")
        if not enroll.is_safe_filename(audio_file):
            return jsonify({"error": "bad_request"}), 400

        result = enroll.read_result(config.base_dir, audio_file)
        if result is None:
            return jsonify({"error": "not_found"}), 404
        if result.get("status") != "ok":
            return jsonify({"error": "not_enrollable"}), 400
        # ไฟล์ผลถูกแก้มือได้และมาจากเวอร์ชันเก่ากว่านี้ได้ -- เวกเตอร์ศูนย์ที่หลุดเข้า
        # ทะเบียนจะ "เหมือน" กับเวกเตอร์ศูนย์อื่นทุกตัว ต้องเช็คก่อนแตะทะเบียนเลย
        if not speakers.is_usable_embedding(result.get("embedding")):
            return jsonify({"error": "bad_embedding"}), 400

        # เหมือนกับ confirm_speaker ทุกประการ (ดูคอมเมนต์ที่นั่น): ผลที่ไม่มีป้ายพื้นที่
        # เวกเตอร์เลยต้องถูกปฏิเสธก่อนแตะทะเบียน ไม่ใช่หลังจากนั้น -- ไม่งั้นจะได้ sample ที่
        # match_known ข้ามตลอดกาลอย่างเงียบ ๆ ต้องเป็นรหัสของตัวเอง ไม่ใช่ bad_embedding
        # (เวกเตอร์ใช้ได้ ปัญหาคือไม่รู้ว่ามันอยู่พื้นที่ไหน)
        if speakers.sample_embedding_model(result) is None:
            return jsonify({"error": "missing_embedding_model"}), 400

        name = payload.get("name")
        if not isinstance(name, str) or not speakers.clean_name(name):
            return jsonify({"error": "bad_name"}), 400
        cleaned = speakers.clean_name(name)

        with _registry_lock:
            registry = speakers.load_registry(config.base_dir)
            try:
                speakers.save_registry(
                    config.base_dir,
                    speakers.add_sample(
                        registry,
                        cleaned,
                        {
                            "embedding": result["embedding"],
                            # ป้ายพื้นที่เวกเตอร์จริงที่ enroll.analyze บันทึกไว้ (ผ่านด่าน
                            # ข้างบนแล้วว่าไม่ใช่ None) ไม่ใช่ค่าใน config ตอนกดยืนยัน --
                            # ผลค้างข้ามการสลับ EMBEDDING_MODEL ได้
                            "embedding_model": speakers.sample_embedding_model(result),
                            "embedding_seconds": result.get("embedding_seconds"),
                            "segment_count": result.get("segment_count"),
                            # โมเดลแยกผู้พูดที่วิเคราะห์ไฟล์นี้จริง ๆ -- provenance เสริม
                            # เท่านั้น ไม่ใช่ป้ายที่ match_known ใช้กรองอีกต่อไป
                            "model": result.get("model"),
                        },
                        source=f"enroll:{audio_file}",
                    ),
                )
            except OSError as e:
                # ทะเบียนยังไม่ถูกแก้ ไฟล์ต้องอยู่ที่เดิมให้ลองใหม่ได้
                logger.warning("บันทึกทะเบียนจากการลงทะเบียนเสียงไม่ได้: %s", e)
                return jsonify({"error": "save_failed"}), 500

        # ทะเบียนถูกเซฟไปแล้ว = เสียงถูกจำแล้วจริง การย้ายไฟล์ไม่สำเร็จจึงต้องไม่กลายเป็น
        # 500 ที่บอกผู้ใช้ว่าล้มเหลว เพราะเขาจะกดใหม่แล้วได้ตัวอย่างซ้ำในทะเบียน -- ดัก
        # Exception กว้าง ๆ ไม่ใช่แค่ OSError (Minor E): shutil.move ลอง os.rename ก่อนแล้ว
        # ถอยไป copy2+remove เมื่อ os.rename พัง เส้นทางไหนก็โยน exception ที่ไม่ใช่ OSError
        # ได้เสมอ (เช่น shutil.Error ตอนมีไฟล์ชื่อชนกันโผล่ใน done/ ระหว่างเช็คกับตอนย้าย
        # จริง) -- คอมเมนต์ข้างบนยืนยันเองอยู่แล้วว่าไม่มีอะไรหลังบันทึกทะเบียนสำเร็จจะทำให้
        # request นี้ล้มเหลวได้อีก ดักแค่ OSError จึงขัดกับเจตนาที่เขียนไว้เอง
        try:
            enroll.archive(config.base_dir, audio_file)
        except Exception:
            # finding 6: ดักกว้างจาก OSError เป็น Exception โดยตั้งใจ (ดูคอมเมนต์ด้านบน
            # เรื่อง shutil.Error) -- สำหรับ exception ที่ไม่คาดคิดจริง ๆ ซึ่งเป็นเหตุผลที่
            # ขยายมาดักกว้างขนาดนี้ traceback คือส่วนที่มีประโยชน์ที่สุดตอนสืบสาเหตุ ใช้
            # logger.exception แทน logger.warning เพื่อให้ traceback ไม่หายไป
            logger.exception("ย้าย %s เข้า done/ ไม่ได้ แต่เสียงถูกจำแล้ว", audio_file)
            # archive() ปกติเรียก clear() เองหลังย้ายไฟล์สำเร็จ แต่พังก่อนถึงตรงนั้น --
            # ถ้าปล่อย result.json เดิมไว้ การ์ดนี้จะยังโชว์สถานะ "พร้อมบันทึก" ในรอบหน้า
            # กดซ้ำได้ตัวอย่างซ้ำเข้าทะเบียนคนเดิม (finding 2 ของรีวิวรอบสุดท้าย) ล้าง
            # แบบ best-effort พอ -- ล้างไม่สำเร็จก็ไม่ใช่เรื่องที่ทำให้ request นี้ล้มเหลว
            # เพราะทะเบียนถูกบันทึกไปแล้วจริง
            enroll.clear(config.base_dir, audio_file)
            return jsonify(
                {"ok": True, "name": cleaned, "warning": "archive_failed"}
            )
        return jsonify({"ok": True, "name": cleaned})

    @app.delete("/api/enroll/<path:audio_file>")
    def dismiss_enroll(audio_file):
        """เอาไฟล์ออกจากรายการโดยไม่ลงทะเบียน -- ย้ายเข้า done/ ไม่ลบทิ้ง

        Minor D: ปุ่มนี้โชว์ได้แม้การ์ดยังอยู่ในคิว (กำลังวิเคราะห์อยู่) แล้ว -- ไฟล์เสียง
        อาจยังถูก ffmpeg เปิดค้างอยู่ระหว่างแปลงเป็น wav ตอนนั้น shutil.move ใน
        enroll.archive raise PermissionError ได้บน Windows เดิมไม่มี try ล้อมเลย exception
        จึงหลุดเป็น 500 ที่ไม่มีอะไรบอกผู้ใช้ -- ต้องจับไว้แล้วคืนรูปแบบความล้มเหลวที่
        หน้าเว็บ (errAction) render เป็น notice ได้อยู่แล้ว ไฟล์ต้องอยู่ที่เดิมให้ลองใหม่ได้
        เพราะไม่รู้ว่า archive คืบหน้าไปแค่ไหนก่อนพัง
        """
        if not enroll.is_safe_filename(audio_file):
            return jsonify({"error": "bad_request"}), 400
        try:
            archived = enroll.archive(config.base_dir, audio_file)
        except Exception:
            # finding 6: เช่นเดียวกับ confirm_enroll ด้านบน -- ดักกว้างจาก OSError เป็น
            # Exception โดยตั้งใจ (ดู docstring ของฟังก์ชันนี้ เรื่อง PermissionError จาก
            # shutil.move) ใช้ logger.exception เพื่อไม่ให้ traceback ของ exception ที่ไม่
            # คาดคิดจริง ๆ หายไป
            logger.exception("เอา %s ออกจากรายการไม่ได้ (archive ล้มเหลว)", audio_file)
            return jsonify({"error": "archive_failed"}), 500
        if archived is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"ok": True})

    @app.get("/<path:filename>")
    def static_file(filename):
        return send_from_directory(WEB_DIR, filename)

    return app


def main() -> None:
    from src.config import load_config

    config = load_config()
    # ไฟล์แยกจาก watcher.log โดยเจตนา -- คนละ process กัน ดูเหตุผลใน src/logsetup.py
    # (RotatingFileHandler หมุนไฟล์ด้วยการ rename ซึ่งชนกันบน Windows)
    log_file = configure_logging(config.base_dir, UI_LOG)
    if log_file is not None:
        logging.info("Writing logs to %s", log_file)
    app = create_app(config)
    # 127.0.0.1 เท่านั้น -- ขอบเขตความปลอดภัยของ service นี้คือการไม่รับจากนอกเครื่อง
    app.run(host="127.0.0.1", port=config.ui_port, threaded=True)


if __name__ == "__main__":
    main()
