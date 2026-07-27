"""ตรวจความพร้อมก่อนเริ่มอัดประชุม

ประชุมจริงย้อนกลับไม่ได้ -- อัดพลาดคือเสียเลย โมดูลนี้ตอบคำถามที่เคยทำให้ผลลัพธ์
ออกมาว่างเปล่าโดยไม่มีใครรู้ตัวจนสายเกินไป:

1. ไมค์ส่งสัญญาณเข้ามาจริงไหม (ปลั๊กหลวม / volume ต่ำ / เสียบผิดช่อง)
2. เสียงคู่สนทนาไหลผ่าน default output ตัวที่ recorder ดักฟังอยู่จริงไหม
   (เครื่องที่มีลำโพงหลายตัวสลับ default ไปมา แอปประชุมอาจส่งเสียงออกอีกตัว
   ทำให้ได้ยินแต่ฝั่งเราเอง)
3. ไมค์กับลำโพงตั้ง sample rate ตรงกันไหม -- ไม่ตรง = ตัวอัดปฏิเสธการทำงาน
   แต่จะบอกก็ต่อเมื่อหน้าจอขึ้นคำว่าเริ่มอัดไปแล้ว
4. key ที่จะใช้สรุปยังเรียกได้จริงไหม -- รู้ตอนนี้ ดีกว่ารู้ตอนประชุมเลิกแล้ว

ข้อ 1-3 เป็น "ไม่ผ่าน" ได้ เพราะเสียงที่ไม่ได้อัดหายถาวร ข้อ 4 เป็นได้แค่ "เตือน"
เพราะ transcript ยังได้ครบอยู่ดี เอาไปให้ Claude สรุปทีหลังได้
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.config import DEFAULT_SUMMARY_MODEL, DEFAULT_UI_LANG
from src.llm import PROVIDERS, MissingApiKeyError, UnknownModelError, resolve
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
API_CHECK_NAME = render("check_api", {}, TH)


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

    เป็น "fail" ได้ ต่างจากผลตรวจ API key: ค่าไม่ตรงกันแปลว่าอัดไม่ได้เลย ไม่ใช่แค่
    ได้ผลลัพธ์ไม่ครบ
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


def read_summary_settings(base_dir: Path | None = None) -> tuple[str, str]:
    """(key ที่ provider ของโมเดลนี้ต้องใช้, โมเดลที่ตั้งไว้) อ่านจาก .env ตรง ๆ

    ไม่ผ่าน load_config เพราะหน้าที่ของโมดูลนี้คือรายงานปัญหาการตั้งค่าให้อ่านรู้เรื่อง
    ไม่ใช่พังคาหน้าจอด้วย traceback

    ถาม resolve() ว่า provider ของโมเดลนี้อ่าน key จาก env var ชื่ออะไร แทนที่จะเดาเอง
    ว่า "ไม่ใช่ GLM-5.2 ก็ต้องเป็น Anthropic" -- ประโยคแบบนั้นคือความรู้เฉพาะ provider ที่
    ควรอยู่ใน registry ของ src/llm.py ที่เดียว เพิ่ม provider ตัวที่สี่แล้วลืมแก้ที่นี่
    จะเงียบ ๆ ไปเช็ค key ผิดตัวโดยไม่มีใครรู้

    โมเดลที่ไม่อยู่ใน registry ไม่มี provider ให้ถาม -- คืน key ว่างไว้ check_summary_model
    เป็นคนรายงานปัญหานั้นแทน
    """
    load_dotenv((base_dir or Path.cwd()) / ".env")
    model = os.environ.get("CLAUDE_MODEL", DEFAULT_SUMMARY_MODEL)
    try:
        key_env = resolve(model).key_env
    except UnknownModelError:
        return ("", model)
    return (os.environ.get(key_env, ""), model)


def probe_summary_model(model: str) -> None:
    """ยิงคำขอที่เล็กที่สุดเท่าที่ยิงได้ไปยัง provider ของโมเดลนี้ ผ่าน = ไม่โยนอะไรออกมา

    ต้องเรียกจริงเพราะไม่มี endpoint ให้ถามสถานะ key หรือยอดเครดิตคงเหลือ ทั้งสอง
    อย่างอ่านได้จาก error ที่ตอบกลับมาเท่านั้น max_tokens=1 ทำให้ค่าใช้จ่ายต่อการตรวจ
    หนึ่งครั้งอยู่ในหลักเศษสตางค์ ยิงผ่าน resolve(model).complete() ตัวเดียวกับที่
    summarize.py เรียกจริง -- provider ไหนถูกเลือกก็ถูกตรวจ ไม่ใช่ Anthropic เสมอ

    คำตอบว่างเปล่าถือว่าผ่าน -- reasoning model ที่ max_tokens=1 ใช้โควตาหมดก่อนเขียน
    อะไรได้ ซึ่งพิสูจน์แล้วว่าคำขอไปถึงโมเดลจริงและ key ผ่าน auth แล้ว สิ่งที่ตรวจคือ
    key ใช้ได้และโมเดลมีอยู่ ไม่ใช่คุณภาพของคำตอบ
    """
    try:
        resolve(model).complete("hi", "hi", 1)
    except MissingApiKeyError:
        # สืบทอดจาก RuntimeError เหมือนกัน แต่ความหมายตรงข้าม: ยังไปไม่ถึงโมเดลเลย
        # ต้องจับก่อน except RuntimeError ข้างล่าง ไม่งั้นจะถูกกลืนเป็น "ผ่าน"
        raise
    except RuntimeError as e:
        if hasattr(e, "status_code"):
            # HttpStatusError -- คำตอบเรื่อง auth/โควตา ต้องให้ classify_probe_error อ่าน
            raise
        # "returned no text" -- ไปถึงโมเดลแล้ว ถือว่าผ่าน
        return


def classify_probe_error(error: Exception, model: str, lang: str = TH) -> CheckResult:
    if isinstance(error, MissingApiKeyError):
        return _result(
            "check_api", "warn", "api_no_key_for", {"error": str(error)}, lang
        )
    status_code = getattr(error, "status_code", None)
    message = str(error)
    if status_code == 429:
        # ผ่าน auth และมีเครดิตแล้วเท่านั้นถึงจะโดนจำกัดอัตรา -- ไม่ใช่ปัญหาของ key
        return _result("check_api", "ok", "api_rate_limited", {}, lang)
    if status_code == 401:
        return _result("check_api", "warn", "api_unauthorized", {}, lang)
    if status_code in (400, 403) and "credit balance" in message.lower():
        return _result("check_api", "warn", "api_no_credit", {}, lang)
    if status_code == 403:
        return _result(
            "check_api", "warn", "api_model_forbidden", {"model": model}, lang
        )
    return _result("check_api", "warn", "api_probe_failed", {"error": message}, lang)


def check_summary_model(model: str, lang: str = TH) -> CheckResult:
    """model ที่จะใช้สรุปอยู่ใน registry ของ resolve() ไหม -- คืน "ok" หรือ "warn" เท่านั้น
    ไม่มี "fail"

    ก่อนหน้านี้ CLAUDE_MODEL ถูกส่งตรงเข้า Anthropic API โดยไม่ผ่านที่นี่เลย id อะไรที่ API
    รับก็ใช้ได้ พอ summarize_transcript หันมาเรียกผ่าน resolve() แล้ว id ที่ไม่อยู่ใน
    PROVIDERS จะโยน UnknownModelError กลางท่อ -- หลังถอดเสียงเสร็จไปแล้ว ซึ่งเป็นขั้นที่
    แพงที่สุด check_api_key ข้างล่างยิงไปถาม API จริงจึงตรวจไม่เจอกรณีนี้ (API รู้จักโมเดล
    id นั้นได้โดยไม่ต้องอยู่ใน registry ของเรา) ต้องเช็คแยกจาก resolve() ตรง ๆ เท่านั้น

    เป็น "warn" ไม่ใช่ "fail" ด้วยเหตุผลเดียวกับ check_api_key: resolve ไม่ผ่านไม่ได้ทำให้
    อัดหรือถอดเสียงไม่ได้ -- transcript ยังออกมาครบ เอาไปสรุปด้วยมือทีหลังได้ ต่างจากอัดไม่ได้
    เลยซึ่งหายถาวร
    """
    try:
        resolve(model)
    except UnknownModelError:
        known = ", ".join(sorted(PROVIDERS))
        return _result(
            "check_model", "warn", "model_unresolvable", {"model": model, "known": known}, lang
        )
    return _result("check_model", "ok", "model_ok", {}, lang)


def check_api_key(api_key: str, model: str, probe=None, lang: str = TH) -> CheckResult:
    """สถานะของ key ที่จะใช้สรุป -- คืน "ok" หรือ "warn" เท่านั้น ไม่มี "fail"

    เพราะ start-meeting.bat ถามยกเลิกการอัดเมื่อเจอ "ไม่ผ่าน" และ key เสียไม่ใช่เหตุ
    ให้ไม่อัด: transcript ยังได้ครบ เอาไปให้ Claude สรุปทีหลังได้ ส่วนประชุมที่ไม่ได้อัด
    นั้นหายถาวร

    เมื่อไม่มี `probe` ระบุมา (ทางเดินจริงตอน main() เรียก) จะยิงผ่าน probe_summary_model
    ซึ่งถาม resolve(model) เอง -- จึงตรวจ provider ที่ model จะใช้จริง ไม่ใช่ Anthropic เสมอ
    """
    if not api_key.strip():
        try:
            env_var = resolve(model).key_env
        except UnknownModelError:
            # ไม่มี provider ให้บอกชื่อ env var -- เกิดขึ้นได้เฉพาะตอนถูกเรียกตรงโดยไม่ผ่าน
            # check_summary_readiness (ซึ่งกันกรณีนี้ไว้แล้ว) ปล่อยว่างดีกว่าเดา
            env_var = ""
        return _result("check_api", "warn", "api_no_key", {"env_var": env_var}, lang)
    try:
        (probe or (lambda _api_key, m: probe_summary_model(m)))(api_key, model)
    except Exception as e:
        return classify_probe_error(e, model, lang)
    # ไม่ต้องบอกชื่อโมเดล (บรรทัด "กำลังตรวจ..." บอกไปแล้ว) และไม่ต้องอธิบายว่าทำไม
    # ถึงไม่มีตัวเลขเครดิต -- คนอ่านบรรทัดนี้ตอนกำลังจะเข้าประชุม ผ่านคือผ่าน จบ
    return _result("check_api", "ok", "api_ok", {}, lang)


def check_summary_readiness(api_key: str, model: str, lang: str = TH) -> list[CheckResult]:
    """ผลตรวจทั้งสองข้อของสาย "จะสรุปได้ไหม": โมเดลอยู่ใน registry ไหม แล้วถ้าอยู่ key
    ใช้ได้ไหม

    โมเดลไม่อยู่ใน registry ทำให้ทั้งสองข้อพังพร้อมกันด้วยสาเหตุเดียว -- ไม่มี provider
    ให้ยิงคำขอตรวจ key เลย จึงคืนแค่ผลตรวจโมเดลข้อเดียว ไม่ใช่สองคำเตือนสำหรับสาเหตุ
    เดียวกัน (ซึ่งจะสอนให้คนอ่านรายงานแบบข้ามผ่าน)
    """
    model_result = check_summary_model(model, lang=lang)
    if model_result.status != "ok":
        return [model_result]
    return [model_result, check_api_key(api_key, model, lang=lang)]


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
    # ต้องเรียกก่อนอ่าน UI_LANG: read_summary_settings เป็นตัวที่ load_dotenv เข้ามา
    # ถ้าสลับลำดับ ค่า UI_LANG ที่ตั้งใน .env จะไม่มีผลเลย
    api_key, model = read_summary_settings()
    lang = os.environ.get("UI_LANG", TH)
    print(render("preflight_checking_key", {"model": model}, lang))
    # ข้อเดียวถ้าโมเดลไม่อยู่ใน registry (ไม่มี provider ให้ตรวจ key เลย) สองข้อถ้า
    # โมเดลใช้ได้ -- ดู check_summary_readiness
    key_results = check_summary_readiness(api_key, model, lang=lang)

    print(render("preflight_checking_audio", {"seconds": MEASURE_SECONDS}, lang))
    results = [*run_preflight(lang=lang), *key_results]
    print()
    print(format_report(results, lang))
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
