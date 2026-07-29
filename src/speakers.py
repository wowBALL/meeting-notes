"""ทะเบียนเสียงถาวร -- ใครมีน้ำเสียงแบบไหน เพื่อจำข้ามการประชุม

ไฟล์นี้ถูกเขียนโดย UI service เท่านั้น และเฉพาะเมื่อมีคนกดยืนยันชื่อ ส่วน watcher
อ่านอย่างเดียวโดยเจตนา: การจับคู่ที่ผิดจะฝังตัวอย่างเสียงผิดคนลงโปรไฟล์ถาวรโดยไม่มี
มนุษย์เห็นเลยสักครั้ง

เก็บเป็น JSON ธรรมดาแทน format ที่กระชับกว่า เพราะทะเบียนของคนกลุ่มเดียวใหญ่ไม่กี่
สิบกิโลไบต์ และการเปิดดู/แก้/ลบด้วยมือได้เมื่อมีอะไรผิดสำคัญกว่าขนาดไฟล์
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import LEGACY_DIARIZATION_MODEL
from src.storage import replace_with_retry

logger = logging.getLogger(__name__)

REGISTRY_DIRNAME = "speakers"
REGISTRY_FILENAME = "registry.json"
REGISTRY_VERSION = 1

# เก็บหลายตัวอย่างต่อคน เพราะเสียงคนเดียวกันผ่านไมค์ตรงหน้ากับผ่าน codec ของแอป
# ประชุมให้เวกเตอร์ต่างกันจริง -- ตัวอย่างเดียวจะจำได้เฉพาะทางที่เคยได้ยินมา
MAX_SAMPLES_PER_SPEAKER = 10

# ชื่อถูกเขียนลง transcript.md ในรูป "**<ชื่อ>** [00:00]:" ชื่อที่ยาวเกินหรือมี
# markdown ปนจะทำให้บรรทัดนั้นเสียรูปทั้งไฟล์
MAX_NAME_LENGTH = 60

# ผู้พูดที่พูดรวมกันน้อยกว่านี้ไม่ถูกเสนอให้ตั้งชื่อและไม่ถูกเก็บเข้าทะเบียน --
# เวกเตอร์จากเสียงไม่กี่วินาทีเชื่อถือไม่ได้พอที่จะเอาไปเทียบข้ามการประชุม
MIN_SPEAKING_SECONDS = 10.0


def registry_path(base_dir: Path) -> Path:
    return Path(base_dir) / REGISTRY_DIRNAME / REGISTRY_FILENAME


def load_registry(base_dir: Path) -> list[dict]:
    """คนในทะเบียนทั้งหมด ไฟล์หาย/พัง/รูปร่างผิด = ทะเบียนว่าง ไม่ raise

    ทะเบียนที่อ่านไม่ออกต้องไม่ทำให้การประชุมที่อัดซ้ำไม่ได้พังตาม -- อย่างแย่ที่สุด
    คือกลับไปมีป้าย "ผู้พูด N" เหมือนก่อนมีฟีเจอร์นี้
    """
    path = registry_path(base_dir)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        logger.warning("อ่านทะเบียนเสียงไม่ได้ (%s) ใช้ทะเบียนว่างแทน: %s", path, e)
        return []
    if not isinstance(parsed, dict):
        return []
    speakers = parsed.get("speakers")
    if not isinstance(speakers, list):
        return []
    return [
        entry
        for entry in speakers
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("samples"), list)
    ]


def save_registry(base_dir: Path, speakers: list[dict]) -> None:
    """เขียนทะเบียนแบบ atomic

    เขียนไฟล์ชั่วคราวแล้วค่อย replace เพราะ watcher เป็นคนละ process และอ่านไฟล์นี้
    ได้ทุกเมื่อ -- การเขียนทับตรง ๆ เปิดช่องให้มันอ่านไฟล์ที่เขียนค้างครึ่งทาง
    """
    path = registry_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(
            {"version": REGISTRY_VERSION, "speakers": speakers},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    replace_with_retry(temp, path)


def clean_name(name: str) -> str:
    """ชื่อที่เขียนลง transcript.md ได้โดยไม่ทำให้ markdown เสียรูป"""
    collapsed = " ".join(str(name).split())
    for character in ("*", "[", "]", "`"):
        collapsed = collapsed.replace(character, "")
    return collapsed.strip()[:MAX_NAME_LENGTH]


def is_usable_embedding(vector) -> bool:
    """เวกเตอร์ที่เอาไปเทียบหรือเก็บเข้าทะเบียนได้จริง

    pyannote pad ศูนย์เข้ามาเมื่อจำนวน label มากกว่าจำนวน centroid (ดู
    pyannote/audio/pipelines/speaker_diarization.py ของไลบรารี บรรทัด ~765) เวกเตอร์
    ศูนย์ล้วนไม่มีทิศทาง cosine จึงไม่นิยาม และถ้าปล่อยเข้าทะเบียนมันจะ "เหมือน" กับ
    เวกเตอร์ศูนย์อื่นทุกตัว
    """
    if not isinstance(vector, list) or not vector:
        return False
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
        return False
    return math.sqrt(sum(float(value) ** 2 for value in vector)) > 0.0


def sample_model(sample: dict) -> str:
    """โมเดลที่สร้างเวกเตอร์ของตัวอย่างนี้

    ตัวอย่างที่ไม่มีคีย์ "model" คือตัวอย่างที่ถูกเก็บก่อนที่ทะเบียนจะเริ่มติดป้าย ซึ่ง
    ตอนนั้นโปรเจกต์นี้มีโมเดลเดียวคือ 3.1 -- เดาไม่ได้ก็จริงในทางทฤษฎี แต่ในทางประวัติ
    ของ repo นี้มันเป็นข้อเท็จจริง การถือว่า "ไม่รู้" แล้วทิ้งไปจะลบความจำของผู้ใช้เดิม
    ที่ยังใช้ 3.1 อยู่ทิ้งทั้งหมดโดยไม่มีเหตุผล
    """
    model = sample.get("model")
    return model if isinstance(model, str) and model else LEGACY_DIARIZATION_MODEL


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """ความเหมือนเชิงทิศทาง 1.0 = ทิศเดียวกัน, 0.0 = ตั้งฉาก, -1.0 = ตรงข้าม

    คืน 0.0 แทนการ raise เมื่อเทียบไม่ได้ (ยาวไม่เท่ากัน หรือมีเวกเตอร์ศูนย์) เพราะ
    ผู้เรียกทุกคนแปลค่าต่ำว่า "ไม่ใช่คนเดียวกัน" อยู่แล้ว ซึ่งเป็นคำตอบที่ถูกต้อง
    """
    if len(a) != len(b):
        return 0.0
    norm_a = math.sqrt(sum(float(value) ** 2 for value in a))
    norm_b = math.sqrt(sum(float(value) ** 2 for value in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class Match:
    """ผู้พูดในไฟล์หนึ่งตรงกับใครในทะเบียน และมั่นใจพอจะใส่ชื่อให้เลยหรือไม่

    `confident` แยกออกมาเป็น field แทนที่จะให้ผู้เรียกไปเทียบ score กับเกณฑ์เอง
    เพราะกฎ "เท่าไหร่ถึงจะใส่ชื่ออัตโนมัติ" ต้องอยู่ที่เดียว -- ผู้เรียกที่เทียบเองผิด
    เกณฑ์แปลว่าใส่ชื่อผิดคนลง transcript
    """

    speaker_id: str
    name: str
    score: float
    confident: bool


def match_known(
    embeddings: dict[str, list[float]],
    speakers: list[dict],
    high: float,
    low: float,
    model: str,
) -> dict[str, Match]:
    """จับคู่ผู้พูดในไฟล์นี้กับคนในทะเบียน คีย์เป็น label ของ pyannote

    label ที่ไม่ถึงเกณฑ์ล่างจะไม่มีใน dict เลย (ไม่ใช่ค่า None) -- ผู้เรียกจึงเขียน
    `label in matches` ได้ตรงไปตรงมา

    `model` คือโมเดลที่สร้าง `embeddings` ชุดที่ส่งเข้ามา ตัวอย่างในทะเบียนที่มาจาก
    โมเดลอื่นถูกข้ามทิ้ง ไม่ใช่เอามาเทียบแล้วให้คะแนนต่ำ -- เวกเตอร์ข้ามพื้นที่ให้เลข
    ที่ไม่มีความหมาย ซึ่ง "บังเอิญสูง" ได้พอ ๆ กับ "บังเอิญต่ำ" พารามิเตอร์นี้จึงบังคับ
    (ไม่มี default) โดยเจตนา: ผู้เรียกที่ลืมส่งต้องพังตอนเขียนโค้ด ไม่ใช่ตอนที่ชื่อผิด
    คนไปโผล่ใน transcript แล้ว
    """
    matches: dict[str, Match] = {}
    skipped_models: set[str] = set()
    for label, embedding in embeddings.items():
        if not is_usable_embedding(embedding):
            continue
        best: Match | None = None
        for speaker in speakers:
            for sample in speaker.get("samples", []):
                if sample_model(sample) != model:
                    skipped_models.add(sample_model(sample))
                    continue
                vector = sample.get("embedding")
                if not is_usable_embedding(vector):
                    continue
                score = cosine_similarity(embedding, vector)
                if best is None or score > best.score:
                    best = Match(
                        speaker_id=speaker["id"],
                        name=speaker["name"],
                        score=score,
                        confident=score >= high,
                    )
        if best is not None and best.score >= low:
            matches[label] = best
    if skipped_models:
        # ผู้ใช้ที่เพิ่งสลับ DIARIZATION_MODEL จะเห็นคนที่เคยจำได้กลายเป็น "ผู้พูด N"
        # เฉย ๆ -- ต้องมีบรรทัดเดียวใน log ที่อธิบายว่าเพราะอะไร ไม่งั้นดูเหมือนทะเบียนพัง
        logger.info(
            "ข้ามตัวอย่างเสียง %d โมเดลในทะเบียนที่ไม่ใช่ %s (%s) -- คนที่ลงทะเบียนไว้ด้วย"
            "โมเดลอื่นจะยังไม่ถูกจำจนกว่าจะ enroll ใหม่ หรือสลับ DIARIZATION_MODEL กลับ",
            len(skipped_models),
            model,
            ", ".join(sorted(skipped_models)),
        )
    return matches


def add_sample(
    speakers: list[dict],
    name: str,
    embedding: list[float],
    source: str,
    model: str,
    today: date | None = None,
) -> list[dict]:
    """ทะเบียนชุดใหม่ที่มีตัวอย่างเสียงนี้เพิ่มเข้าไป

    ชื่อซ้ำ = คนเดิม ไม่ใช่คนใหม่ -- ผู้ใช้ที่พิมพ์ชื่อเดิมในการประชุมที่สองกำลังบอกว่า
    "คนนี้แหละ" การสร้าง entry ที่สองจะทำให้โปรไฟล์แตกเป็นสองก้อนที่ต่างก็อ่อนแอ

    คืนรายการชุดใหม่แทนการแก้ของเดิมในที่ ผู้เรียกจึงยังถือของเดิมไว้ได้ถ้าการเขียน
    ไฟล์ล้มเหลว

    `model` ต้องเป็นโมเดลที่สร้าง `embedding` ตัวนี้จริง ๆ ไม่ใช่โมเดลที่ตั้งอยู่ใน
    config ตอนกดยืนยัน -- สองอย่างนี้ต่างกันได้เมื่อผู้ใช้สลับโมเดลหลังประชุมเสร็จแต่
    ก่อนกดตั้งชื่อ ป้ายที่ผิดแปลว่าเวกเตอร์จะถูกเอาไปเทียบข้ามพื้นที่ในภายหลัง (ดู
    match_known) ซึ่งเป็นสิ่งเดียวที่ป้ายนี้มีไว้กัน
    """
    cleaned = clean_name(name)
    if not cleaned:
        raise ValueError("ชื่อผู้พูดว่างเปล่าหลังตัดอักขระที่ใช้ไม่ได้ออก")
    sample = {
        "embedding": [float(value) for value in embedding],
        "source": source,
        "model": model,
        "added": (today or date.today()).isoformat(),
    }
    updated = [dict(speaker, samples=list(speaker.get("samples", []))) for speaker in speakers]
    for speaker in updated:
        if speaker.get("name") == cleaned:
            speaker["samples"] = (speaker["samples"] + [sample])[-MAX_SAMPLES_PER_SPEAKER:]
            return updated
    updated.append({"id": uuid.uuid4().hex, "name": cleaned, "samples": [sample]})
    return updated


def remove_speaker(speakers: list[dict], speaker_id: str) -> list[dict]:
    return [speaker for speaker in speakers if speaker.get("id") != speaker_id]
