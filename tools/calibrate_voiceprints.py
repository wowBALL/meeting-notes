"""วัดช่องว่างระหว่างคะแนนของ "คนเดียวกัน" กับ "คนละคน" ในพื้นที่ voiceprint

SPEAKER_MATCH_HIGH/LOW ใน config.py เป็นตัวเลขที่ตัดสินว่าจะใส่ชื่อใครลง transcript.md
โดยไม่ถามใคร -- ค่าคู่นั้นต้องมาจากการวัด ไม่ใช่การเดา สคริปต์นี้คือเครื่องมือที่ใช้วัด
และเป็นสิ่งเดียวที่ทำให้ตัวเลขนั้นตรวจสอบซ้ำได้เมื่อ EMBEDDING_MODEL หรือวิธีคัดช่วงเสียง
เปลี่ยนไป

รับไฟล์เฉลย (ใครเป็นป้ายไหนในไฟล์ไหน) แล้วรายงานสองกอง: คะแนนของคู่ที่เฉลยบอกว่าเป็นคน
เดียวกัน กับคู่ที่เฉลยบอกว่าเป็นคนละคน ถ้าสองกองแยกกันจะเสนอเกณฑ์ที่อยู่ในช่องว่างนั้น
ถ้าซ้อนกันจะไม่เสนอเลขอะไรเลยและคืน exit code 1 -- ดูเหตุผลใน suggest_thresholds

ไฟล์เฉลยไม่เข้า repo (มีชื่อคนจริง) ดู tools/README.md สำหรับรูปแบบและวิธีรัน
"""

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import tempfile
from pathlib import Path

if os.name == "nt":
    # เหตุผลเต็มอยู่ใน src/main.py: ตั้งแต่ python 3.8 PATH ไม่ถูกใช้ตอน ctypes/torch
    # แก้ชื่อ DLL ที่มันพึ่งพา ทำสำเนามาที่นี่เพราะสคริปต์นี้เป็นทางเข้าที่ไม่ผ่าน
    # src/main.py แต่โหลดโมเดลชุดเดียวกันบนเครื่องเดียวกัน -- และความพังของ DLL
    # resolution เป็นแบบเงียบ (การแยกผู้พูดตายโดยไม่มีใครเห็น) ซึ่งเป็นอาการที่แพงที่สุด
    # ที่จะมาเจอตอนรันจริงหลังรอ diarize ไปแล้วเจ็ดนาที
    for _dir in os.environ.get("PATH", "").split(os.pathsep):
        if _dir and os.path.isdir(_dir):
            try:
                os.add_dll_directory(_dir)
            except OSError:
                pass

import soundfile as sf

from src.audio_convert import convert_to_wav
from src.config import load_config
from src.diarize import diarize_audio, load_diarization_pipeline
from src.speakers import cosine_similarity
from src.voiceprint import clean_intervals, extract_voiceprints, load_embedder
from src.waveform import load_waveform

logger = logging.getLogger(__name__)

# ความยาวสูงสุดของคลิปที่ตัดให้ฟังยืนยันตัวตน -- 12 วินาทีพอให้จำเสียงคนที่รู้จักได้
# แน่นอน และสั้นพอที่จะฟังทั้งการประชุมที่มี 7 ป้ายจบในสองนาที ซึ่งเป็นเงื่อนไขจริงที่
# ทำให้มีคนยอมนั่งเขียนไฟล์เฉลย
CLIP_SECONDS = 12.0


def _cosine(a, b) -> float:
    """คะแนนความเหมือนของสองเวกเตอร์ -- ตัวเดียวกับที่ระบบใช้ตัดสินใจจริง

    เรียก speakers.cosine_similarity ไม่คำนวณเอง: เลขที่สคริปต์นี้รายงานจะถูกเอาไปตั้ง
    เป็น SPEAKER_MATCH_HIGH/LOW ซึ่ง match_known เอาไปเทียบกับคะแนนจาก
    cosine_similarity -- ถ้าที่นี่คำนวณด้วยสูตรของตัวเอง เกณฑ์ที่วัดได้จะเป็นเกณฑ์ของ
    สูตรที่ไม่มีใครใช้ และความต่างจะไม่โผล่ให้เห็นจนกว่าจะมีคนถูกใส่ชื่อผิด
    """
    return cosine_similarity(list(a), list(b))


def score_pairs(voiceprints: dict, truth: dict) -> tuple[list[float], list[float]]:
    """คะแนนของคู่ แยกเป็นกองคนเดียวกันกับกองคนละคน -- แต่ละกองรับคู่คนละชุดโดยเจตนา

    **กองคนเดียวกัน: คู่ข้ามไฟล์เท่านั้น** คู่คนเดียวกันในไฟล์เดียวกันใช้ไมค์ codec และ
    สภาพห้องเดียวกัน คะแนนสูงกว่าความจริงราว 0.05-0.15 (วัด 2026-07-30: กอง split-half
    ต่ำสุด 0.8042 ขณะที่คู่ข้ามการบันทึกของคนเดียวกันได้ 0.7493) เอามาเป็น *พื้น* ของกองนี้
    จะได้เกณฑ์เข้มเกินจริงแล้วปฏิเสธคนที่ถูกต้อง

    **กองคนละคน: ทุกคู่ รวมคู่ในไฟล์เดียวกัน** ความไม่สมมาตรนี้คือหัวใจ ไม่ใช่ความหลุด:
    แต่ละกองต้องรับค่ามาจากฝั่งที่อนุรักษ์นิยมที่สุดของตัวเอง และคู่คนละคนในประชุมเดียวกัน
    คือรูปทรงของการใส่ชื่อผิดที่เกิดขึ้นจริง -- คนหนึ่งลงทะเบียนไว้ อีกคนนั่งอยู่ในห้องเดียวกัน
    แล้วได้ชื่อเขาไปโดยไม่มีใครยืนยัน คู่พวกนี้ทำคะแนนสูงกว่าคู่คนละคนข้ามการบันทึกมาก
    (วัด 2026-07-30: 0.6313 เทียบกับ 0.4310) เวอร์ชันแรกของฟังก์ชันนี้ตัดมันทิ้งด้วย จึงเสนอ
    HIGH ที่ 0.59 -- ต่ำกว่าคะแนนที่คนละคนทำได้จริงถึง 0.04 คือเลขที่พาไปอยู่ในโซนใส่ชื่อผิด
    พอดี ทั้งที่ดูเหมือนคำนวณมาอย่างดี

    label ที่ไม่มีเฉลยถูกข้ามทั้งคู่ ไม่ถือว่า "เป็นคนอื่น" -- pyannote แยกคนเดียวเป็นสอง
    ป้ายได้ และการเดาจะดันเพดานกองคนละคนขึ้นเอง
    """
    resolved = {}
    for (path, label), vector in voiceprints.items():
        person = truth.get((path, label)) or truth.get((path, "*"))
        if person is not None:
            resolved[(path, label)] = (person, vector)

    same: list[float] = []
    different: list[float] = []
    keys = sorted(resolved)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            left_person, left_vector = resolved[left]
            right_person, right_vector = resolved[right]
            if left_person == right_person and left[0] == right[0]:
                continue
            score = _cosine(left_vector, right_vector)
            (same if left_person == right_person else different).append(score)
    return same, different


def suggest_thresholds(same: list[float], different: list[float]) -> dict:
    """เกณฑ์ที่หลักฐานรองรับ หรือรายงานว่าไม่มี

    HIGH ต้องอยู่ระหว่างเพดานของกองคนละคนกับพื้นของกองคนเดียวกัน -- เลือกกลางช่องว่างเพราะ
    ทั้งสองฝั่งคือความเสียหายจริง: สูงเกินไป = ไม่ใส่ชื่อคนที่จำได้ (แก้ได้ด้วยการคลิก) ต่ำ
    เกินไป = ใส่ชื่อผิดคนลงสรุปที่ระบุผู้รับผิดชอบ (แก้ไม่ได้ถ้าไม่มีใครสังเกต) ค่ากลางไม่ได้
    บอกว่าสองฝั่งเท่ากัน มันบอกว่าเรายังไม่มีข้อมูลพอที่จะเอียงไปทางไหน

    LOW อยู่เหนือเพดานของกองคนละคน เพื่อไม่ให้คนที่ไม่รู้จักไปโผล่ในโซน "เสนอให้ยืนยัน"

    ซ้อนกันแล้วคืน overlap=True และ high=None -- ไม่เลือกเลขให้ การเลือกเลขที่ "ดูดีที่สุด"
    ตอนที่สองกองซ้อนกันคือการซ่อนปัญหาไว้ใต้ค่าคอนฟิก
    """
    if not same or not different:
        return {"low": None, "high": None, "overlap": True}
    ceiling = max(different)
    floor = min(same)
    if floor <= ceiling:
        return {"low": None, "high": None, "overlap": True}
    return {
        "low": round(ceiling + (floor - ceiling) * 0.1, 2),
        "high": round((ceiling + floor) / 2, 2),
        "overlap": False,
    }


def load_truth(path: Path) -> tuple[list[str], dict]:
    """ไฟล์เฉลย -> (รายชื่อไฟล์เสียงตามลำดับในไฟล์, {(audio, label): person})

    คีย์ใช้สตริง `audio` ตามที่เขียนไว้ในไฟล์เฉลยดิบ ๆ ไม่ normalize เป็น path สัมบูรณ์ --
    คีย์ชุดเดียวกันนี้ถูกใช้เป็นชื่อในรายงานที่มนุษย์อ่าน สิ่งที่ผู้ใช้พิมพ์ลงไปคือสิ่งที่
    เขาจะเห็นกลับมา และการเทียบว่า "คู่นี้ข้ามไฟล์ไหม" ก็คือการเทียบสตริงนี้ตรง ๆ

    รายการที่ผิดรูปถูกข้ามพร้อมคำเตือน ไม่ raise -- ไฟล์เฉลยเขียนด้วยมือ และการพิมพ์ผิด
    บรรทัดเดียวไม่ควรทำให้ไฟล์ที่เหลือ (ซึ่ง diarize ไปแล้วเจ็ดนาทีต่อไฟล์) ไม่ได้วัดเลย
    """
    entries = json.loads(path.read_text(encoding="utf-8"))
    audio_files: list[str] = []
    truth: dict[tuple[str, str], str] = {}
    for entry in entries:
        audio = entry.get("audio") if isinstance(entry, dict) else None
        labels = entry.get("truth") if isinstance(entry, dict) else None
        if not isinstance(audio, str) or not isinstance(labels, dict):
            logger.warning("ข้ามรายการในไฟล์เฉลยที่ไม่มี audio/truth ที่ใช้ได้: %r", entry)
            continue
        if audio in audio_files:
            # ป้าย "*" ของรายการหลังจะทับของรายการแรกเงียบ ๆ และคู่ในไฟล์เดียวกันไม่ถูกนับ
            # อยู่แล้ว -- รายการซ้ำจึงไม่เคยเพิ่มข้อมูล มีแต่จะทำให้เฉลยที่อ่านไม่ตรงกับที่ใช้
            logger.warning("ข้ามรายการซ้ำของไฟล์ %s ในไฟล์เฉลย", audio)
            continue
        audio_files.append(audio)
        for label, person in labels.items():
            truth[(audio, label)] = person
    return audio_files, truth


def _cache_path(cache_dir: Path, audio_path: Path, model: str) -> Path:
    """ที่อยู่ของ turns ที่แคชไว้ของไฟล์นี้ ภายใต้โมเดลแยกผู้พูดตัวนี้

    ใส่ทั้งไบต์ของไฟล์ (ขนาด+mtime) และชื่อโมเดลลงในกุญแจ: ไฟล์เดิมที่ diarize ด้วยคนละ
    โมเดลต้องเป็นคนละรายการ ไม่ใช่รายการเดียวที่ทับกันไปมาตามว่ารันอะไรล่าสุด
    """
    stat = audio_path.stat()
    fingerprint = f"{audio_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{model}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.turns.json"


def _read_cached_turns(cache_dir: Path, audio_path: Path, model: str) -> list[dict] | None:
    """turns ที่แคชไว้ หรือ None เมื่อไม่มี/เชื่อไม่ได้

    ตรวจชื่อโมเดลใน payload อีกครั้งทั้งที่มันอยู่ในกุญแจแล้ว: กุญแจกันแค่การชนกันของ
    ไฟล์แคชสองใบ ส่วนไฟล์ที่ถูกก็อปข้ามเครื่อง แก้ด้วยมือ หรือเหลือค้างจากตอนที่กุญแจ
    ยังคิดคนละแบบ ยังนอนอยู่ใต้ชื่อที่ "ถูกต้อง" ได้อยู่ดี -- turns จากอีกโมเดลหนึ่งแบ่ง
    ช่วงเสียงคนละแบบ เวกเตอร์ที่ได้จึงมาจากเสียงคนละท่อน แล้วเกณฑ์ที่วัดได้จะบรรยาย
    คอนฟิกผสมที่ไม่มีใครรันจริง ทั้งที่ทุกอย่างดูสำเร็จดี
    """
    path = _cache_path(cache_dir, audio_path, model)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("model") != model:
        logger.warning("ข้ามแคชของ %s ที่มาจากโมเดลอื่น (%r)", audio_path.name, payload.get("model"))
        return None
    turns = payload.get("turns")
    if not isinstance(turns, list):
        return None
    return turns


def _write_cached_turns(cache_dir: Path, audio_path: Path, model: str, turns: list[dict]) -> None:
    """เก็บ turns ไว้ใช้รอบหน้า -- เขียนไม่สำเร็จก็แค่ช้า ไม่ทำให้รอบนี้พัง"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"audio": audio_path.name, "model": model, "turns": turns}
    try:
        _cache_path(cache_dir, audio_path, model).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("เขียนแคช turns ของ %s ไม่สำเร็จ: %s", audio_path.name, e)


def _safe_stem(name: str) -> str:
    """ชื่อไฟล์เสียงต้นทาง -> ส่วนที่เอาไปตั้งชื่อไฟล์คลิปบน Windows ได้"""
    return "".join("_" if character in '<>:"/\\|?*' else character for character in name).strip()


def _write_clips(clips_dir: Path, index: int, audio: str, wav_path: Path, turns: list[dict]) -> None:
    """ท่อนที่ยาวที่สุดของแต่ละป้าย ตัดที่ CLIP_SECONDS -- ของที่ทำให้เขียนไฟล์เฉลยได้

    ตัดจากช่วงที่ผ่าน clean_intervals แล้ว ไม่ใช่ turns ดิบ: ท่อนที่มีคนพูดทับอยู่ทำให้
    คนฟังตัดสินใจไม่ได้ว่า SPEAKER_xx คือเสียงไหนในสองเสียงที่ได้ยิน แล้วเฉลยที่เขียนจาก
    การเดานั้นจะย้อนกลับมาเป็นคู่ที่ติดป้ายผิดในกองคะแนน ซึ่งแย่กว่าไม่มีเฉลยเลย

    เอาท่อนที่ยาวที่สุดเพราะเป็นท่อนที่มีโอกาสมีประโยคเต็มมากที่สุด และเป็นท่อนแรกที่
    select_intervals เลือกเข้าเวกเตอร์อยู่แล้ว -- สิ่งที่ได้ยินจึงเป็นเสียงที่ถูกวัดจริง
    """
    clean = clean_intervals(turns)
    if not clean:
        return
    clips_dir.mkdir(parents=True, exist_ok=True)
    waveform = load_waveform(wav_path)
    samples = waveform["waveform"]
    rate = waveform["sample_rate"]
    stem = _safe_stem(Path(audio).stem)
    for label, intervals in sorted(clean.items()):
        start, end = max(intervals, key=lambda span: span[1] - span[0])
        end = min(end, start + CLIP_SECONDS)
        first = max(0, int(start * rate))
        last = min(samples.shape[1], int(end * rate))
        if last <= first:
            continue
        path = clips_dir / f"{index:02d}-{stem}__{label}.wav"
        sf.write(str(path), samples[:, first:last].T.numpy(), rate)
        print(f"  คลิป {path.name}  ({end - start:.1f} วิ จาก {start:.1f})")


def _describe(name: str, scores: list[float]) -> str:
    if not scores:
        return f"{name}: ไม่มีคู่เลย"
    return (
        f"{name}: n={len(scores)}  ต่ำสุด={min(scores):.4f}  "
        f"มัธยฐาน={statistics.median(scores):.4f}  สูงสุด={max(scores):.4f}"
    )


def _report(voiceprints: dict, truth: dict, same: list[float], different: list[float]) -> dict:
    """พิมพ์ทุกอย่างที่ต้องใช้ตัดสินใจ แล้วคืนผลของ suggest_thresholds

    พิมพ์คะแนนรายคู่ทุกคู่ ไม่ใช่แค่สรุปกอง: ตอนที่สองกองซ้อนกัน สิ่งที่ต้องรู้คือ *คู่ไหน*
    ที่ทำให้ซ้อน (เกือบทุกครั้งคือเฉลยที่ผิดหนึ่งบรรทัด หรือป้ายที่ pyannote ปนคนสองคนไว้)
    ตัวเลขสรุปกองบอกแค่ว่ามีปัญหา ไม่ได้บอกว่าจะไปแก้ที่ไหน

    ไม่จัดตารางให้คอลัมน์ตรงกันเมื่อมีข้อความไทยปน -- สระบนล่างของไทยกว้างศูนย์และ
    เทอร์มินัลนับความกว้างไม่ตรงกัน ตารางที่ "ตรง" บนเครื่องหนึ่งจะเบี้ยวบนอีกเครื่อง
    ตัวเลขขึ้นก่อนแล้วข้อความตามหลังจึงอ่านได้เสมอ
    """
    print()
    print("=== ป้ายที่วัดได้ ===")
    for (audio, label), voiceprint in sorted(voiceprints.items()):
        person = truth.get((audio, label)) or truth.get((audio, "*"))
        who = person if person is not None else "(ไม่มีเฉลย -- ไม่ถูกนับเป็นคู่ใด ๆ)"
        print(
            f"  {audio} / {label}: {voiceprint.seconds:.1f} วิ "
            f"{voiceprint.segment_count} ท่อน -> {who}"
        )

    result = suggest_thresholds(same, different)

    print()
    print("=== กองคะแนน (คนเดียวกัน: ข้ามไฟล์เท่านั้น / คนละคน: ทุกคู่) ===")
    print("  " + _describe("คนเดียวกัน", same))
    print("  " + _describe("คนละคน", different))
    if same and different:
        print(f"  ช่องว่าง: {min(same) - max(different):+.4f} (พื้นของกองคนเดียวกัน ลบ เพดานของกองคนละคน)")

    print()
    print("=== เกณฑ์ที่เสนอ ===")
    if result["overlap"]:
        print("  สองกองซ้อนกัน (หรือมีกองที่ว่าง) -- ไม่มีเกณฑ์คู่ไหนที่หลักฐานรองรับ")
        print("  ไม่เสนอเลขให้โดยเจตนา: การเลือกเลขที่ดูดีที่สุดตอนนี้คือการซ่อนปัญหาไว้ใต้ค่าคอนฟิก")
    else:
        print(f"  SPEAKER_MATCH_HIGH = {result['high']}")
        print(f"  SPEAKER_MATCH_LOW  = {result['low']}")
    return result


def _print_pairs(voiceprints: dict, truth: dict) -> None:
    """คะแนนรายคู่ที่ถูกนับ เรียงจากมากไปน้อย พร้อมชื่อคนตามเฉลย

    ต้องแสดงคู่ชุดเดียวกับที่ score_pairs นับเป๊ะ ๆ ไม่ใช่ชุดที่อ่านง่ายกว่า -- คู่ที่ดัน
    เพดานกองคนละคนขึ้นมักเป็นคู่ในประชุมเดียวกัน ถ้าตารางนี้ไม่แสดงมัน คนอ่านจะเห็นเกณฑ์ที่
    เข้มโดยไม่มีอะไรอธิบายว่าทำไม แล้วสรุปว่าเครื่องมือคิดผิด
    """
    resolved = {}
    for key, voiceprint in voiceprints.items():
        person = truth.get(key) or truth.get((key[0], "*"))
        if person is not None:
            resolved[key] = (person, voiceprint.embedding)

    rows = []
    keys = sorted(resolved)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            left_person, left_vector = resolved[left]
            right_person, right_vector = resolved[right]
            if left_person == right_person and left[0] == right[0]:
                continue
            rows.append(
                (
                    _cosine(left_vector, right_vector),
                    left_person == right_person,
                    left,
                    right,
                    left_person,
                    right_person,
                )
            )
    if not rows:
        return
    print()
    print("=== คะแนนรายคู่ที่ถูกนับ ===")
    for score, is_same, left, right, left_person, right_person in sorted(rows, reverse=True):
        verdict = "คนเดียวกัน" if is_same else "คนละคน  "
        print(
            f"  {score:.4f}  {verdict}  {left_person}:{left[0]}/{left[1]}"
            f"  <->  {right_person}:{right[0]}/{right[1]}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tools.calibrate_voiceprints")
    parser.add_argument("truth", help="ไฟล์เฉลย (ดู tools/README.md)")
    parser.add_argument(
        "--cache",
        default=None,
        help="โฟลเดอร์เก็บผล diarization ไว้ใช้ซ้ำ -- ควรใส่เสมอ ไฟล์ประชุมหนึ่งไฟล์ใช้เวลาหลายนาที",
    )
    parser.add_argument(
        "--extract-clips",
        default=None,
        help="โฟลเดอร์สำหรับเขียนคลิปเสียงของแต่ละป้ายไว้ฟังยืนยันว่า SPEAKER_xx คือใคร",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """วัดจริงกับไฟล์เสียงในไฟล์เฉลย คืน 0 เมื่อสองกองแยกกัน 1 เมื่อซ้อนกัน

    exit code ไม่ใช่ของประดับ: มันคือสิ่งที่ทำให้ "เกณฑ์ยังแยกสองกองได้อยู่ไหม" ถูกถาม
    ซ้ำได้โดยไม่ต้องมีคนอ่านผลลัพธ์ ถ้าวันหนึ่งมีคนสลับ EMBEDDING_MODEL แล้วสองกองซ้อนกัน
    สิ่งที่ต้องเกิดคือความล้มเหลวที่เห็นได้ ไม่ใช่ตารางสวย ๆ ที่ไม่มีใครอ่านบรรทัดสุดท้าย

    โหลด pipeline แยกผู้พูดแบบขี้เกียจ (เฉพาะตอนแคชไม่โดน): การรันซ้ำเพื่อปรับวิธีคิด
    คะแนนคือกรณีใช้งานหลักของสคริปต์นี้ และการรอโหลดโมเดล 10-20 วินาทีที่ไม่ได้ใช้เลย
    ทุกครั้งจะกัดกินเหตุผลทั้งหมดที่แคชมีอยู่
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = load_config()
    audio_files, truth = load_truth(Path(args.truth))
    cache_dir = Path(args.cache) if args.cache else None
    clips_dir = Path(args.extract_clips) if args.extract_clips else None

    print(f"โมเดลแยกผู้พูด: {config.diarization_model}")
    print(f"โมเดล voiceprint: {config.embedding_model}")
    embedder = load_embedder(config.hf_token, config.embedding_model)
    pipeline = None

    voiceprints: dict[tuple[str, str], object] = {}
    for index, audio in enumerate(audio_files):
        path = Path(audio)
        if not path.is_absolute():
            path = config.base_dir / path
        if not path.is_file():
            logger.warning("ข้ามไฟล์ที่ไม่มีอยู่จริง: %s", path)
            continue
        print()
        print(f"[{index + 1}/{len(audio_files)}] {audio}")
        # แปลงเป็น wav 16k โมโนก่อนเสมอ แบบเดียวกับ pipeline.process_file และ
        # enroll.analyze -- เวกเตอร์ที่วัดที่นี่ต้องมาจากการปรับสภาพเสียงชุดเดียวกับที่
        # ระบบจริงใช้ ไม่งั้นเกณฑ์ที่ได้เป็นเกณฑ์ของเสียงที่ไม่มีใครป้อนเข้าระบบ
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / f"{index:02d}.wav"
            convert_to_wav(path, wav_path)

            turns = None
            if cache_dir is not None:
                turns = _read_cached_turns(cache_dir, path, config.diarization_model)
            if turns is not None:
                print(f"  ใช้ turns จากแคช ({len(turns)} ท่อน)")
            else:
                if pipeline is None:
                    print("  กำลังโหลด pipeline แยกผู้พูด...")
                    pipeline = load_diarization_pipeline(
                        config.hf_token, config.diarization_model
                    )
                print("  กำลังแยกผู้พูด (ไฟล์ประชุมเต็มใช้เวลาหลายนาที)...")
                turns = diarize_audio(wav_path, "", pipeline).turns
                if cache_dir is not None:
                    _write_cached_turns(cache_dir, path, config.diarization_model, turns)

            for label, voiceprint in extract_voiceprints(wav_path, turns, embedder).items():
                voiceprints[(audio, label)] = voiceprint

            if clips_dir is not None:
                _write_clips(clips_dir, index, audio, wav_path, turns)

    vectors = {key: voiceprint.embedding for key, voiceprint in voiceprints.items()}
    same, different = score_pairs(vectors, truth)
    result = _report(voiceprints, truth, same, different)
    _print_pairs(voiceprints, truth)
    return 1 if result["overlap"] else 0


if __name__ == "__main__":
    sys.exit(main())
