"""ตรวจความพร้อมก่อนเริ่มอัดประชุม

ประชุมจริงย้อนกลับไม่ได้ -- อัดพลาดคือเสียเลย โมดูลนี้ตอบคำถามที่เคยทำให้ผลลัพธ์
ออกมาว่างเปล่าโดยไม่มีใครรู้ตัวจนสายเกินไป:

1. ไมค์ส่งสัญญาณเข้ามาจริงไหม (ปลั๊กหลวม / volume ต่ำ / เสียบผิดช่อง)
2. เสียงคู่สนทนาไหลผ่าน default output ตัวที่ recorder ดักฟังอยู่จริงไหม
   (เครื่องที่มีลำโพงหลายตัวสลับ default ไปมา แอปประชุมอาจส่งเสียงออกอีกตัว
   ทำให้ได้ยินแต่ฝั่งเราเอง)
3. ไมค์กับลำโพงตั้ง sample rate ตรงกันไหม -- ไม่ตรง = ตัวอัดปฏิเสธการทำงาน
   แต่จะบอกก็ต่อเมื่อหน้าจอขึ้นคำว่าเริ่มอัดไปแล้ว
4. โมเดลที่ตั้งไว้จะใช้สรุปได้จริงไหม -- อยู่ใน registry ของ resolve() หรือเปล่า

ข้อ 1-3 เป็น "ไม่ผ่าน" ได้ เพราะเสียงที่ไม่ได้อัดหายถาวร ข้อ 4 เป็นได้แค่ "เตือน"
เพราะ transcript ยังได้ครบอยู่ดี เอาไปสรุปทีหลังได้

เดิมข้อ 4 ยิงคำขอจริงไปยัง API เพื่อตรวจว่า key ใช้ได้ด้วย แต่ถูกถอดออกแล้ว: มันคือ
คำขอที่มีค่าใช้จ่ายจริงทุกครั้งก่อนอัดประชุม ต้องมี timeout ของตัวเอง และถึงจะรู้ว่า
key ใช้ไม่ได้ก็ยังอัดต่อได้อยู่ดี (transcript ไม่ได้หายไปไหน สรุปทีหลังด้วยมือได้)
ความคุ้มค่าจึงไม่พอเทียบกับต้นทุนที่ต้องจ่ายทุกครั้ง
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.config import DEFAULT_SUMMARY_MODEL, DEFAULT_UI_LANG
from src.job import NO_SUMMARY_MODEL
from src.llm import PROVIDERS, UnknownModelError, resolve
from src.messages import render
from src.record import (
    get_common_samplerate,
    get_wasapi_loopback_device,
    get_wasapi_mic_device,
)

# ระดับ peak ของเสียงพูดปกติอยู่ราว -20 ถึง -6 dBFS
MIC_GOOD_DBFS = -30.0
# ต่ำกว่านี้ถือว่าไม่มีสัญญาณ (noise floor ของช่องว่างเปล่าวัดได้ราว -50 dB)
MIC_WEAK_DBFS = -45.0
# มีเสียงดิจิทัลไหลผ่านจริงหรือเป็นแค่ความเงียบ
LOOPBACK_SILENT_DBFS = -50.0

MEASURE_SECONDS = 8

# ภาษาตั้งต้นของโมดูลนี้ -- ผู้เรียกส่ง lang เข้ามาทับได้ทุกฟังก์ชัน ค่านี้มีไว้ให้
# ผู้เรียกเดิม (และเทสต์ชุดเดิม) ได้ข้อความชุดเดียวกับก่อนมี catalog
TH = DEFAULT_UI_LANG

SAMPLERATE_CHECK_NAME = render("check_samplerate", {}, TH)


@dataclass
class CheckResult:
    """ผลตรวจหนึ่งข้อ

    `name` กับ `detail` เป็นข้อความที่ render แล้วในภาษาที่ขอมา ส่วน `name_code`
    `code` `params` คือของจริงที่เอาไป render ใหม่เป็นภาษาอื่นได้ -- เก็บทั้งสอง
    อย่างเพื่อให้ผู้เรียกเดิม (และเทสต์ชุดเดิม) ยังอ่าน .detail ได้ตรง ๆ เหมือนเดิม
    """

    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    name_code: str = ""
    code: str = ""
    params: dict = field(default_factory=dict)


def _result(
    name_code: str, status: str, code: str, params: dict | None = None, lang: str = TH
) -> CheckResult:
    params = params or {}
    return CheckResult(
        name=render(name_code, {}, lang),
        status=status,
        detail=render(code, params, lang),
        name_code=name_code,
        code=code,
        params=params,
    )


def localized(result: CheckResult, lang: str) -> CheckResult:
    """ผลตรวจเดิมในอีกภาษาหนึ่ง -- ผลที่ไม่มีรหัสกำกับคืนกลับไปตามเดิม"""
    if not result.code:
        return result
    return _result(result.name_code, result.status, result.code, result.params, lang)


def peak_dbfs(peak: float) -> float:
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def _fmt(db: float, lang: str = TH) -> str:
    if db == -math.inf:
        return render("peak_silent", {}, lang)
    return render("peak_level", {"db": f"{db:.1f}"}, lang)


def evaluate_mic(db: float, lang: str = TH) -> CheckResult:
    level = {"level": _fmt(db, lang)}
    if db >= MIC_GOOD_DBFS:
        return _result("check_mic", "ok", "mic_ok", level, lang)
    if db >= MIC_WEAK_DBFS:
        return _result("check_mic", "warn", "mic_weak", level, lang)
    return _result("check_mic", "fail", "mic_none", level, lang)


def evaluate_loopback(db: float, device_name: str, lang: str = TH) -> CheckResult:
    params = {"level": _fmt(db, lang), "device": device_name}
    if db >= LOOPBACK_SILENT_DBFS:
        return _result("check_loopback", "ok", "loopback_ok", params, lang)
    # แยกไม่ออกว่า "ไม่มีอะไรเล่นอยู่" หรือ "แอปประชุมส่งเสียงออกอุปกรณ์อื่น"
    # กรณีหลังคือกรณีที่ทำให้เสียเสียงฝั่งคู่สนทนาไปทั้งประชุมโดยไม่รู้ตัว
    return _result("check_loopback", "warn", "loopback_silent", params, lang)


def evaluate_samplerate(
    mic_device: dict, loopback_device: dict, lang: str = TH
) -> CheckResult:
    """ไมค์กับลำโพงคุยกันที่ sample rate เดียวกันไหม

    ถามผ่าน get_common_samplerate ตัวเดียวกับที่ record.py เรียกตอนเริ่มอัด แทนที่จะ
    เขียนเงื่อนไขเทียบเอง -- คำตอบของ preflight จึงไม่มีทางขัดกับสิ่งที่ตัวอัดยอมรับจริง
    แม้เงื่อนไขนั้นจะเปลี่ยนไปในอนาคต

    เป็น "fail" ได้ ต่างจาก check_summary_model ที่ได้แค่ "warn": ค่า sample rate ไม่
    ตรงกันแปลว่าอัดไม่ได้เลย ส่วนโมเดลที่ใช้ไม่ได้แปลว่าได้ transcript แต่ไม่ได้สรุป
    """
    try:
        rate = get_common_samplerate(mic_device, loopback_device)
    except RuntimeError as e:
        # ข้อความของ error นี้มาจาก record.py ซึ่งยังเป็นไทยอยู่ -- ส่งผ่านตรง ๆ
        # ดีกว่าแปลครึ่งเดียวแล้วได้ประโยคที่ปนสองภาษา
        return _result(
            "check_samplerate", "fail", "passthrough", {"error": str(e)}, lang
        )
    return _result("check_samplerate", "ok", "samplerate_ok", {"rate": rate}, lang)


def read_summary_model(base_dir: Path | None = None) -> str:
    """โมเดลที่ตั้งไว้ใช้สรุป อ่านจาก .env ตรง ๆ

    ไม่ผ่าน load_config เพราะหน้าที่ของโมดูลนี้คือรายงานปัญหาการตั้งค่าให้อ่านรู้เรื่อง
    ไม่ใช่พังคาหน้าจอด้วย traceback

    เดิมฟังก์ชันนี้ชื่อ read_summary_settings และคืน key คู่กับ model ด้วย เพราะตอนนั้น
    ยังมี check_api_key ยิง API จริงเพื่อตรวจ key พอ probe ถูกถอดออก ไม่มีใครในโมดูลนี้
    ต้องใช้ key อีกแล้ว จึงคืนแค่โมเดลและเปลี่ยนชื่อให้ตรงกับหน้าที่ที่เหลือ
    """
    load_dotenv((base_dir or Path.cwd()) / ".env")
    return os.environ.get("CLAUDE_MODEL", DEFAULT_SUMMARY_MODEL)


def check_summary_model(model: str, lang: str = TH) -> CheckResult:
    """model ที่จะใช้สรุปอยู่ใน registry ของ resolve() ไหม -- คืน "ok" หรือ "warn" เท่านั้น
    ไม่มี "fail"

    ก่อนหน้านี้ CLAUDE_MODEL ถูกส่งตรงเข้า Anthropic API โดยไม่ผ่านที่นี่เลย id อะไรที่ API
    รับก็ใช้ได้ พอ summarize_transcript หันมาเรียกผ่าน resolve() แล้ว id ที่ไม่อยู่ใน
    PROVIDERS จะโยน UnknownModelError กลางท่อ -- หลังถอดเสียงเสร็จไปแล้ว ซึ่งเป็นขั้นที่
    แพงที่สุด ต้องเช็คแยกจาก resolve() ตรง ๆ เท่านั้นถึงจะจับกรณีนี้ได้ก่อนอัด

    NO_SUMMARY_MODEL (transcript-only) ไม่อยู่ใน PROVIDERS เลย -- resolve() ไม่รู้จักมัน
    โดยเจตนา (ดู src/job.py) เพราะมันแปลว่า "อย่าเรียกโมเดลไหนเลย" ไม่ใช่ id ของโมเดล
    จริง ๆ ตัวหนึ่ง ต้องเช็คแยกก่อน resolve() ไม่งั้นจะโดนตีความว่าเป็นโมเดลที่พิมพ์ผิด
    ทั้งที่ผู้ใช้ตั้งค่าตามเอกสารทุกตัวอักษร (README.md, .env.example)

    เป็น "warn" ไม่ใช่ "fail": resolve ไม่ผ่านไม่ได้ทำให้อัดหรือถอดเสียงไม่ได้ -- transcript
    ยังออกมาครบ เอาไปสรุปด้วยมือทีหลังได้ ต่างจากอัดไม่ได้เลยซึ่งหายถาวร
    """
    if model == NO_SUMMARY_MODEL:
        return _result("check_model", "ok", "model_transcript_only", {"model": model}, lang)
    try:
        resolve(model)
    except UnknownModelError:
        # NO_SUMMARY_MODEL ต้องอยู่ในรายชื่อนี้ด้วย ไม่งั้นข้อความจะบอกไม่ครบว่า
        # transcript-only ก็เป็นค่าที่ใช้ได้จริง (แค่ไม่ได้อยู่ใน PROVIDERS)
        known = ", ".join(sorted((*PROVIDERS, NO_SUMMARY_MODEL)))
        return _result(
            "check_model", "warn", "model_unresolvable", {"model": model, "known": known}, lang
        )
    return _result("check_model", "ok", "model_ok", {}, lang)


def format_report(results: list[CheckResult], lang: str = TH) -> str:
    marks = {status: render(f"mark_{status}", {}, lang) for status in ("ok", "warn", "fail")}
    shown = [localized(r, lang) for r in results]
    lines = [f"{marks[r.status]} {r.name}: {r.detail}" for r in shown]
    if any(r.status == "fail" for r in shown):
        lines.append(render("report_fail", {}, lang))
    elif any(r.status == "warn" for r in shown):
        lines.append(render("report_warn", {}, lang))
    else:
        lines.append(render("report_ok", {}, lang))
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


def run_preflight(seconds: int = MEASURE_SECONDS, lang: str = TH) -> list[CheckResult]:
    audio = pyaudio_instance()
    try:
        try:
            mic_device = get_wasapi_mic_device(audio)
            loopback_device = get_wasapi_loopback_device(audio)
        except Exception as e:
            failure = {"error": str(e)}
            return [
                _result("check_mic", "fail", "passthrough", failure, lang),
                _result("check_loopback", "fail", "passthrough", failure, lang),
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
        evaluate_mic(peak_dbfs(mic_peak), lang),
        evaluate_loopback(peak_dbfs(loopback_peak), loopback_device["name"], lang),
        evaluate_samplerate(mic_device, loopback_device, lang),
    ]


def main() -> int:
    # ต้องเรียกก่อนอ่าน UI_LANG: read_summary_model เป็นตัวที่ load_dotenv เข้ามา
    # ถ้าสลับลำดับ ค่า UI_LANG ที่ตั้งใน .env จะไม่มีผลเลย
    model = read_summary_model()
    lang = os.environ.get("UI_LANG", TH)
    print(render("preflight_checking_model", {"model": model}, lang))
    model_result = check_summary_model(model, lang=lang)

    print(render("preflight_checking_audio", {"seconds": MEASURE_SECONDS}, lang))
    results = [*run_preflight(lang=lang), model_result]
    print()
    print(format_report(results, lang))
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
