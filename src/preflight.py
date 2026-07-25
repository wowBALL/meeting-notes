"""ตรวจความพร้อมก่อนเริ่มอัดประชุม

ประชุมจริงย้อนกลับไม่ได้ -- อัดพลาดคือเสียเลย โมดูลนี้ตอบคำถามที่เคยทำให้ผลลัพธ์
ออกมาว่างเปล่าโดยไม่มีใครรู้ตัวจนสายเกินไป:

1. ไมค์ส่งสัญญาณเข้ามาจริงไหม (ปลั๊กหลวม / volume ต่ำ / เสียบผิดช่อง)
2. เสียงคู่สนทนาไหลผ่าน default output ตัวที่ recorder ดักฟังอยู่จริงไหม
   (เครื่องที่มีลำโพงหลายตัวสลับ default ไปมา แอปประชุมอาจส่งเสียงออกอีกตัว
   ทำให้ได้ยินแต่ฝั่งเราเอง)
3. key ที่จะใช้สรุปยังเรียกได้จริงไหม -- รู้ตอนนี้ ดีกว่ารู้ตอนประชุมเลิกแล้ว

ข้อ 1-2 เป็น "ไม่ผ่าน" ได้ เพราะเสียงที่ไม่ได้อัดหายถาวร ข้อ 3 เป็นได้แค่ "เตือน"
เพราะ transcript ยังได้ครบอยู่ดี เอาไปให้ Claude สรุปทีหลังได้
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.config import DEFAULT_CLAUDE_MODEL
from src.record import get_wasapi_loopback_device, get_wasapi_mic_device

# ระดับ peak ของเสียงพูดปกติอยู่ราว -20 ถึง -6 dBFS
MIC_GOOD_DBFS = -30.0
# ต่ำกว่านี้ถือว่าไม่มีสัญญาณ (noise floor ของช่องว่างเปล่าวัดได้ราว -50 dB)
MIC_WEAK_DBFS = -45.0
# มีเสียงดิจิทัลไหลผ่านจริงหรือเป็นแค่ความเงียบ
LOOPBACK_SILENT_DBFS = -50.0

MEASURE_SECONDS = 8

API_CHECK_NAME = "Claude API key"
# ตัดสั้นกว่า default ของ SDK (10 นาที) มาก -- นี่คือการตรวจก่อนอัด ไม่ใช่การเรียกจริง
# เน็ตอืดจนเกินนี้ให้รายงานเป็นคำเตือนแล้วเดินหน้าต่อ ดีกว่าค้างคาหน้าจอก่อนประชุม
API_PROBE_TIMEOUT_SECONDS = 15.0


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


def peak_dbfs(peak: float) -> float:
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def _fmt(db: float) -> str:
    return "เงียบสนิท" if db == -math.inf else f"peak {db:.1f} dB"


def evaluate_mic(db: float) -> CheckResult:
    if db >= MIC_GOOD_DBFS:
        return CheckResult("ไมค์", "ok", f"{_fmt(db)} -- ระดับเสียงพูดปกติ")
    if db >= MIC_WEAK_DBFS:
        return CheckResult(
            "ไมค์",
            "warn",
            f"{_fmt(db)} -- เบากว่าปกติมาก ลองขยับไมค์เข้าใกล้ / ดัน volume "
            "หรือเปิด Microphone Boost (mmsys.cpl)",
        )
    return CheckResult(
        "ไมค์",
        "fail",
        f"{_fmt(db)} -- แทบไม่มีสัญญาณ เช็คว่าเสียบช่องไมค์ (สีชมพู) แน่นดี "
        "และไม่ได้ mute อยู่",
    )


def evaluate_loopback(db: float, device_name: str) -> CheckResult:
    if db >= LOOPBACK_SILENT_DBFS:
        return CheckResult(
            "ลำโพง (คู่สนทนา)", "ok", f"{_fmt(db)} จาก {device_name} -- เสียงไหลผ่านจริง"
        )
    # แยกไม่ออกว่า "ไม่มีอะไรเล่นอยู่" หรือ "แอปประชุมส่งเสียงออกอุปกรณ์อื่น"
    # กรณีหลังคือกรณีที่ทำให้เสียเสียงฝั่งคู่สนทนาไปทั้งประชุมโดยไม่รู้ตัว
    return CheckResult(
        "ลำโพง (คู่สนทนา)",
        "warn",
        f"ไม่มีเสียงผ่าน {device_name} -- ถ้าตอนนี้เปิดเสียงอยู่ แปลว่าแอปส่งเสียง "
        "ออกอุปกรณ์อื่น ให้ตั้ง output ของแอปประชุมให้ตรงกับ default ของ Windows",
    )


def read_api_settings(base_dir: Path | None = None) -> tuple[str, str]:
    """(API key, โมเดลที่ตั้งไว้) อ่านจาก .env ตรง ๆ ไม่ผ่าน load_config

    load_config โยน KeyError เมื่อไม่มี ANTHROPIC_API_KEY -- ซึ่งเป็นหนึ่งในกรณีที่
    หน้าที่ของโมดูลนี้คือรายงานให้อ่านรู้เรื่อง ไม่ใช่พังคาหน้าจอด้วย traceback
    """
    load_dotenv((base_dir or Path.cwd()) / ".env")
    return (
        os.environ.get("ANTHROPIC_API_KEY", ""),
        os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
    )


def probe_claude(api_key: str, model: str) -> None:
    """ยิงคำขอที่เล็กที่สุดเท่าที่ยิงได้ ผ่าน = ไม่โยนอะไรออกมา

    ต้องเรียกจริงเพราะ Anthropic ไม่มี endpoint ให้ถามสถานะ key หรือยอดเครดิตคงเหลือ
    ทั้งสองอย่างอ่านได้จาก error ที่ตอบกลับมาเท่านั้น max_tokens=1 ทำให้ค่าใช้จ่ายต่อ
    การตรวจหนึ่งครั้งอยู่ในหลักเศษสตางค์
    """
    from anthropic import Anthropic

    client = Anthropic(
        api_key=api_key, timeout=API_PROBE_TIMEOUT_SECONDS, max_retries=0
    )
    client.messages.create(
        model=model,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )


def classify_probe_error(error: Exception, model: str) -> CheckResult:
    status_code = getattr(error, "status_code", None)
    message = str(error)
    if status_code == 429:
        # ผ่าน auth และมีเครดิตแล้วเท่านั้นถึงจะโดนจำกัดอัตรา -- ไม่ใช่ปัญหาของ key
        return CheckResult(
            API_CHECK_NAME,
            "ok",
            "key ใช้ได้ (ตอนนี้ชนลิมิตอัตราการเรียก แต่ตอนสรุปมี retry รออยู่แล้ว)",
        )
    if status_code == 401:
        return CheckResult(
            API_CHECK_NAME,
            "warn",
            "key ใช้ไม่ได้ (401) -- หมดอายุ ถูกเพิกถอน หรือพิมพ์ผิด "
            "ออก key ใหม่แล้วแก้ ANTHROPIC_API_KEY ใน .env",
        )
    if status_code in (400, 403) and "credit balance" in message.lower():
        return CheckResult(
            API_CHECK_NAME,
            "warn",
            "เครดิตไม่พอ -- เติมที่หน้า Plans & Billing ใน Anthropic Console "
            "ไม่งั้นประชุมนี้จะได้แค่ transcript",
        )
    if status_code == 403:
        return CheckResult(
            API_CHECK_NAME,
            "warn",
            f"key ไม่มีสิทธิ์เรียก {model} (403) -- ตรวจสิทธิ์ของ key หรือแก้ CLAUDE_MODEL",
        )
    return CheckResult(API_CHECK_NAME, "warn", f"ตรวจไม่สำเร็จ: {message}")


def check_api_key(api_key: str, model: str, probe=None) -> CheckResult:
    """สถานะของ key ที่จะใช้สรุป -- คืน "ok" หรือ "warn" เท่านั้น ไม่มี "fail"

    เพราะ start-meeting.bat ถามยกเลิกการอัดเมื่อเจอ "ไม่ผ่าน" และ key เสียไม่ใช่เหตุ
    ให้ไม่อัด: transcript ยังได้ครบ เอาไปให้ Claude สรุปทีหลังได้ ส่วนประชุมที่ไม่ได้อัด
    นั้นหายถาวร
    """
    if not api_key.strip():
        return CheckResult(
            API_CHECK_NAME,
            "warn",
            "ไม่ได้ตั้งค่า ANTHROPIC_API_KEY ใน .env -- อัดและถอดเสียงได้ปกติ "
            "แต่จะไม่ได้สรุปอัตโนมัติ",
        )
    try:
        (probe or probe_claude)(api_key, model)
    except Exception as e:
        return classify_probe_error(e, model)
    # ไม่ต้องบอกชื่อโมเดล (บรรทัด "กำลังตรวจ..." บอกไปแล้ว) และไม่ต้องอธิบายว่าทำไม
    # ถึงไม่มีตัวเลขเครดิต -- คนอ่านบรรทัดนี้ตอนกำลังจะเข้าประชุม ผ่านคือผ่าน จบ
    return CheckResult(API_CHECK_NAME, "ok", "ใช้งานได้ปกติ")


def format_report(results: list[CheckResult]) -> str:
    marks = {"ok": "[ ผ่าน ]", "warn": "[ เตือน ]", "fail": "[ ไม่ผ่าน ]"}
    lines = [f"{marks[r.status]} {r.name}: {r.detail}" for r in results]
    if any(r.status == "fail" for r in results):
        lines.append("สรุป: ยังไม่ควรเริ่มอัด -- แก้ข้อที่ไม่ผ่านก่อน")
    elif any(r.status == "warn" for r in results):
        lines.append("สรุป: พร้อมอัด แต่มีข้อเตือน อ่านให้ครบก่อนเริ่ม")
    else:
        lines.append("สรุป: พร้อมอัดประชุมได้เลย")
    return "\n".join(lines)


def pyaudio_instance():
    import pyaudiowpatch as pyaudio

    return pyaudio.PyAudio()


def measure_peaks(audio, mic_device: dict, loopback_device: dict, seconds: int):
    """เปิดทั้งสองสตรีมพร้อมกันแล้วคืนค่า peak (linear) ของแต่ละตัว

    วัดพร้อมกันโดยตั้งใจ -- เป็นสภาพเดียวกับตอนอัดจริงเป๊ะ ถ้าอุปกรณ์สองตัวอยู่
    ร่วมกันไม่ได้ จะได้รู้ตั้งแต่ตอนตรวจ ไม่ใช่กลางประชุม
    """
    import pyaudiowpatch as pyaudio

    peaks = {"mic": 0.0, "loopback": 0.0}

    def make_callback(key: str):
        def callback(in_data, frame_count, time_info, status):
            block = np.frombuffer(in_data, dtype=np.float32)
            if len(block):
                peaks[key] = max(peaks[key], float(np.abs(block).max()))
            return (None, pyaudio.paContinue)

        return callback

    streams = []
    for key, device in (("mic", mic_device), ("loopback", loopback_device)):
        streams.append(
            audio.open(
                format=pyaudio.paFloat32,
                channels=int(device["maxInputChannels"]),
                rate=int(device.get("defaultSampleRate", 48000)),
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=4096,
                stream_callback=make_callback(key),
            )
        )
    try:
        time.sleep(seconds)
    finally:
        for stream in streams:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
    return peaks["mic"], peaks["loopback"]


def run_preflight(seconds: int = MEASURE_SECONDS) -> list[CheckResult]:
    audio = pyaudio_instance()
    try:
        try:
            mic_device = get_wasapi_mic_device(audio)
            loopback_device = get_wasapi_loopback_device(audio)
        except Exception as e:
            return [
                CheckResult("ไมค์", "fail", str(e)),
                CheckResult("ลำโพง (คู่สนทนา)", "fail", str(e)),
            ]

        mic_peak, loopback_peak = measure_peaks(
            audio, mic_device, loopback_device, seconds
        )
    finally:
        try:
            audio.terminate()
        except Exception:
            pass

    return [
        evaluate_mic(peak_dbfs(mic_peak)),
        evaluate_loopback(peak_dbfs(loopback_peak), loopback_device["name"]),
    ]


def main() -> int:
    api_key, claude_model = read_api_settings()
    print(f"กำลังตรวจ key ที่จะใช้สรุป ({claude_model}) ...")
    api_result = check_api_key(api_key, claude_model)

    print(f"กำลังตรวจเสียง {MEASURE_SECONDS} วินาที -- พูดใส่ไมค์ และเปิดเสียงอะไรก็ได้ออกลำโพง")
    results = [*run_preflight(), api_result]
    print()
    print(format_report(results))
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
