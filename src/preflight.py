"""ตรวจความพร้อมของเสียงก่อนเริ่มอัดประชุม

ประชุมจริงย้อนกลับไม่ได้ -- อัดพลาดคือเสียเลย โมดูลนี้ตอบคำถามสองข้อที่เคยทำให้
transcript ออกมาว่างเปล่าโดยไม่มีใครรู้ตัวจนสายเกินไป:

1. ไมค์ส่งสัญญาณเข้ามาจริงไหม (ปลั๊กหลวม / volume ต่ำ / เสียบผิดช่อง)
2. เสียงคู่สนทนาไหลผ่าน default output ตัวที่ recorder ดักฟังอยู่จริงไหม
   (เครื่องที่มีลำโพงหลายตัวสลับ default ไปมา แอปประชุมอาจส่งเสียงออกอีกตัว
   ทำให้ได้ยินแต่ฝั่งเราเอง)
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from src.record import get_wasapi_loopback_device, get_wasapi_mic_device

# ระดับ peak ของเสียงพูดปกติอยู่ราว -20 ถึง -6 dBFS
MIC_GOOD_DBFS = -30.0
# ต่ำกว่านี้ถือว่าไม่มีสัญญาณ (noise floor ของช่องว่างเปล่าวัดได้ราว -50 dB)
MIC_WEAK_DBFS = -45.0
# มีเสียงดิจิทัลไหลผ่านจริงหรือเป็นแค่ความเงียบ
LOOPBACK_SILENT_DBFS = -50.0

MEASURE_SECONDS = 8


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
    print(f"กำลังตรวจเสียง {MEASURE_SECONDS} วินาที -- พูดใส่ไมค์ และเปิดเสียงอะไรก็ได้ออกลำโพง")
    results = run_preflight()
    print()
    print(format_report(results))
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
