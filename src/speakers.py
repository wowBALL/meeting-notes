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
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.storage import replace_with_retry

logger = logging.getLogger(__name__)

REGISTRY_DIRNAME = "speakers"
REGISTRY_FILENAME = "registry.json"
REGISTRY_VERSION = 2

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

    ที่เหลือคือขนาดที่ float เก็บค่าที่ต้องใช้เทียบไว้ไม่ได้ เกณฑ์จึงเป็น "ผลรวมกำลังสอง
    เป็น float ปกติ" ไม่ใช่ "norm > 0.0" ซึ่งกันทั้งสองปลายพร้อมเวกเตอร์ศูนย์ในตัว:

    * inf/nan -- norm ของ inf เป็น inf ซึ่ง > 0.0 จริง แต่ cosine ที่ได้เป็น nan และ nan
      แพ้ `score > best.score` ใน match_known ทุกครั้ง ตัวอย่างพิษตัวเดียวจึงล็อก best
      ไว้แล้วกลบตัวอย่างจริงของคนคนนั้นทั้งหมดเงียบ ๆ = อาการ "ลงทะเบียนแล้วระบบก็ยังจำ
      ไม่ได้" ที่การ์ดนี้มีไว้กันตั้งแต่แรก
    * ใหญ่เกิน -- 1e308 ก็พอให้ผลรวมกำลังสองล้นเป็น inf แปลว่าเทียบกับใครไม่ได้เลย
    * เล็กเกิน -- กำลังสองตกไปอยู่ช่วง subnormal เหลือ precision ไม่กี่บิต norm ที่ได้จึง
      ผิดและ cosine ออกนอก [-1, 1] ได้จริง (วัด 2026-07-30: คู่สเกล 1e-162 ให้ -2.0)
      ปลายนี้ร้ายที่สุดเพราะไม่พังให้เห็น -- ค่าบวกทำนองเดียวกัน >= high คือใส่ชื่อผิดคน
      ลง transcript.md ให้เองโดยไม่มีใครกดยืนยัน

    ทางเข้ามีจริงทุกแบบ: registry.json กับไฟล์คิวถูกแก้มือได้ตามเจตนาของโปรเจกต์ (ดู
    save_registry) json.loads รับ `Infinity`/`NaN` เปล่า ๆ และคืน int ที่ยาวเท่าไหร่ก็ได้
    -- จึงเช็คแต่ละช่องด้วยการแปลงเป็น float จริง ไม่ใช่ดูแค่ชนิด เพราะปลายทางทุกทางเรียก
    float(value) ตรง ๆ (add_sample ตอนเก็บ, cosine_similarity ตอนเทียบ) การ์ดที่ผ่านทั้งที่
    ปลายทางแปลงไม่ได้คือการเลื่อนความพังไปที่ที่ไม่มีใครดักไว้

    เกณฑ์นี้ไม่แตะเวกเตอร์จริง: ของจาก pyannote มี norm ราว 3.3 (ดู enroll.py) ห่างจากทั้ง
    สองปลายร้อยกว่า order of magnitude และเมื่อผลรวมกำลังสองที่นี่เป็นค่าปกติ กำลังสองของ
    ทุกช่องกับ dot product ที่ cosine_similarity คิดต่อก็คำนวณได้เสมอ
    """
    if not isinstance(vector, list) or not vector:
        return False
    numbers: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            number = float(value)
        except OverflowError:
            return False
        if not math.isfinite(number):
            return False
        numbers.append(number)
    # คูณเอง ไม่ใช้ `** 2`: float ที่ล้นจากการคูณได้ inf ซึ่งเช็คต่อได้ ส่วน `**` โยน
    # OverflowError ออกมาจากการ์ด และมันไม่ใช่ ValueError ที่ผู้เรียกดักไว้ (ดู
    # session_service ที่แปลงเป็น 400 bad_embedding) จึงกลายเป็น 500 ที่ไม่มีใครอธิบาย
    #
    # sys.float_info.min คือ float ปกติที่เล็กสุด ค่าที่ต่ำกว่านี้ยังเก็บได้แต่เสีย
    # precision ไปตามความเล็ก -- เทียบตรง ๆ แบบนี้จึงกันเวกเตอร์ศูนย์ล้วนไปด้วยในตัว
    total = sum(number * number for number in numbers)
    return math.isfinite(total) and total >= sys.float_info.min


def sample_embedding_model(sample: dict) -> str | None:
    """โมเดลที่สร้างเวกเตอร์ของตัวอย่างนี้ -- None เมื่อไม่รู้

    ไม่มี fallback โดยเจตนา ต่างจาก sample_model() เดิมที่เดาเป็น 3.1: ตอนนั้นการเดา
    เป็นข้อเท็จจริงของประวัติ repo (มีโมเดลเดียว) แต่ตัวอย่างที่ไม่มีป้ายนี้มาจาก centroid
    ของ diarization pipeline ซึ่งไม่ใช่พื้นที่นี้แน่นอนไม่ว่ากรณีใด การเดาว่า "น่าจะตรง"
    แล้วเอาไปหา cosine จะได้ตัวเลขที่ไม่มีความหมาย ซึ่ง "บังเอิญสูง" ได้พอ ๆ กับ
    "บังเอิญต่ำ" -- และค่าที่บังเอิญสูงคือชื่อผิดคนใน transcript ที่ไม่มีใครสังเกต
    """
    model = sample.get("embedding_model")
    if not isinstance(model, str):
        return None
    return model.strip() or None


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
    *,
    embedding_model: str,
) -> dict[str, Match]:
    """จับคู่ผู้พูดในไฟล์นี้กับคนในทะเบียน คีย์เป็น label ของ pyannote

    label ที่ไม่ถึงเกณฑ์ล่างจะไม่มีใน dict เลย (ไม่ใช่ค่า None) -- ผู้เรียกจึงเขียน
    `label in matches` ได้ตรงไปตรงมา

    `embedding_model` คือโมเดลที่สร้าง `embeddings` ชุดที่ส่งเข้ามา ตัวอย่างในทะเบียนที่มา
    จากโมเดลอื่น -- รวมถึงตัวอย่างที่ไม่มีป้ายเลย (มาจาก centroid ของ diarization pipeline
    ยุคก่อนฟีเจอร์นี้) -- ถูกข้ามทิ้ง ไม่ใช่เอามาเทียบแล้วให้คะแนนต่ำ: เวกเตอร์ข้ามพื้นที่
    ให้เลขที่ไม่มีความหมาย ซึ่ง "บังเอิญสูง" ได้พอ ๆ กับ "บังเอิญต่ำ"

    พารามิเตอร์นี้เป็น keyword-only โดยเจตนา: หัวฟังก์ชันเดิมมี `str` ติดกันหลายตัว ซึ่ง
    เป็นรูปทรงที่สลับตำแหน่งกันได้เงียบ ๆ การสลับ embedding_model กับพารามิเตอร์อื่นทำให้
    ทั้งทะเบียนถูกเทียบข้ามพื้นที่โดยไม่มีอะไรพังตอนเขียนโค้ด -- ผู้เรียกที่ลืมส่งหรือส่ง
    ผิดตำแหน่งต้องพังตอนนั้น ไม่ใช่ตอนที่ชื่อผิดคนไปโผล่ใน transcript แล้ว

    ต้องเป็น str ที่ไม่ว่างเปล่าหลัง strip เท่านั้น -- ผู้เรียกจริงสองราย (Task 11, 12)
    อ่านค่านี้มาจาก payload ที่เก็บไว้ก่อนฟีเจอร์นี้ ผ่าน sample_embedding_model() ซึ่งคืน
    None ให้ทุก sample ที่ไม่มีป้าย ถ้าปล่อยให้ None ไหลเข้ามาถึงตรงนี้ `stamp != None`
    จะเป็นเท็จพอดีกับ sample ที่ไม่มีป้ายเช่นกัน (stamp เป็น None ด้วย) -- ทุก legacy sample
    จะกลายเป็นจับคู่ได้หมด ซึ่งเป็นช่องโหว่เดียวกับที่ฟีเจอร์นี้ทั้งฟีเจอร์มีไว้ปิด จุดนี้เป็น
    fail-closed component ที่ถูกออกแบบไว้ที่เดียว การเช็คจึงอยู่ตรงนี้ ไม่ใช่ให้ผู้เรียก
    แต่ละรายเช็คเอง ค่าที่รับมาถูก strip ก่อนเทียบเช่นเดียวกับค่าที่เก็บไว้ (ผ่าน
    sample_embedding_model) -- ไม่งั้นตัวอย่างที่ถูกต้องแต่ผู้เรียกส่งมาพร้อมช่องว่างรอบ ๆ
    จะไม่ถูกจับคู่กับใครในทะเบียนเลยทั้งชุด
    """
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model ต้องเป็น string ที่ไม่ว่างเปล่า")
    wanted_model = embedding_model.strip()

    matches: dict[str, Match] = {}
    skipped_models: set[str] = set()
    for label, embedding in embeddings.items():
        if not is_usable_embedding(embedding):
            continue
        best: Match | None = None
        for speaker in speakers:
            for sample in speaker.get("samples", []):
                stamp = sample_embedding_model(sample)
                if stamp != wanted_model:
                    skipped_models.add(stamp or "(ไม่มีป้าย)")
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
        # ผู้ใช้ที่เพิ่งสลับโมเดล embedding จะเห็นคนที่เคยจำได้กลายเป็น "ผู้พูด N" เฉย ๆ
        # -- ต้องมีบรรทัดเดียวใน log ที่อธิบายว่าเพราะอะไร ไม่งั้นดูเหมือนทะเบียนพัง
        logger.info(
            "ข้ามตัวอย่างเสียงจาก %d พื้นที่เวกเตอร์ที่ไม่ใช่ %s (%s) -- คนที่ลงทะเบียนไว้"
            "ก่อนหน้านี้จะยังไม่ถูกจำจนกว่าจะ enroll ใหม่ (ตัวอย่างเดิมไม่หาย)",
            len(skipped_models),
            wanted_model,
            ", ".join(sorted(skipped_models)),
        )
    return matches


_OPTIONAL_SAMPLE_KEYS = ("embedding_seconds", "segment_count", "model")


def add_sample(
    speakers: list[dict],
    name: str,
    sample: dict,
    source: str,
    today: date | None = None,
) -> list[dict]:
    """ทะเบียนชุดใหม่ที่มีตัวอย่างเสียงนี้เพิ่มเข้าไป

    รับ `sample` เป็น dict ก้อนเดียวแทนพารามิเตอร์เรียงกัน: เดิมเป็น
    (embedding, source, model) ซึ่งเป็น str ติดกันสองตัว การสลับสองตัวนั้นจะติดป้ายพื้นที่
    เวกเตอร์ผิดโดยไม่มีอะไรพัง -- ซึ่งเป็นอันตรายเดียวที่ป้ายนี้มีไว้กัน

    `embedding_model` บังคับ: sample ที่ไม่มีป้ายจะถูก match_known ข้ามตลอดกาล การเขียนมัน
    ลงไปเงียบ ๆ คือการสร้างความจำที่ไม่มีวันถูกใช้ ซึ่งผู้ใช้จะเห็นเป็น "ลงทะเบียนแล้วแต่
    ระบบไม่จำ" โดยไม่มีอะไรอธิบาย ตรวจผ่าน sample_embedding_model() ตัวเดียวกับที่
    match_known ใช้อ่าน -- ไม่ใช่เช็ค truthiness ของตัวเอง เพราะ `if not sample.get(key)`
    จับได้แค่ key หายกับ "" ส่วนค่าอย่าง "   ", 42, True, ["x"] ผ่าน truthiness สบาย ๆ
    แต่ sample_embedding_model คืน None ให้ทุกตัว ผลคือ sample ถูกเขียนลงทะเบียนสำเร็จ
    แต่ match_known ข้ามมันตลอดกาลอย่างเงียบ ๆ -- ตัวเขียนกับตัวอ่านต้องเห็นตรงกันว่าอะไร
    ใช้ได้ จึงเรียกตัวอ่านตัวเดียวกันแทนที่จะมีกฎสองชุด ค่าที่เก็บเป็นค่าที่ strip แล้ว
    ไม่ใช่ค่าดิบจาก payload

    `embedding` ต้องผ่าน is_usable_embedding เช่นกัน -- เวกเตอร์ศูนย์ล้วน (cosine ไม่นิยาม)
    หรือ dict ที่ [float(v) for v in sample["embedding"]] จะแปลงเงียบ ๆ เป็นลิสต์ของ key
    (เช่น {"1": 2} กลายเป็น [1.0]) ต้องถูกปฏิเสธตั้งแต่ตรงนี้ ไม่ใช่กลายเป็น sample
    ที่เก็บสำเร็จแต่ใช้เทียบไม่ได้จริง

    `model` (โมเดลแยกผู้พูด) ไม่บังคับและ *ไม่มีใครใช้ตัดสินใจอะไรแล้ว* -- เก็บไว้เพราะมัน
    กำหนดขอบท่อน ซึ่งกำหนดว่าเสียงช่วงไหนเข้าไปอยู่ในเวกเตอร์ เป็น provenance ที่มีค่าตอน
    ต้องสืบว่าทำไม sample ตัวหนึ่งแปลก ไม่ใช่ป้ายที่กันการเทียบข้ามพื้นที่อีกต่อไป

    ชื่อซ้ำ = คนเดิม ไม่ใช่คนใหม่ (เหมือนเดิม) และคืนรายการชุดใหม่แทนการแก้ของเดิมในที่
    """
    cleaned = clean_name(name)
    if not cleaned:
        raise ValueError("ชื่อผู้พูดว่างเปล่าหลังตัดอักขระที่ใช้ไม่ได้ออก")
    stamp = sample_embedding_model(sample)
    if stamp is None:
        raise ValueError("ตัวอย่างเสียงไม่มี embedding_model ที่ใช้ได้")
    if not is_usable_embedding(sample.get("embedding")):
        raise ValueError("ตัวอย่างเสียงไม่มี embedding ที่ใช้ได้")

    stored = {
        "embedding": [float(value) for value in sample["embedding"]],
        "embedding_model": stamp,
        "source": source,
        "added": (today or date.today()).isoformat(),
    }
    for key in _OPTIONAL_SAMPLE_KEYS:
        if sample.get(key) is not None:
            stored[key] = sample[key]

    updated = [dict(speaker, samples=list(speaker.get("samples", []))) for speaker in speakers]
    for speaker in updated:
        if speaker.get("name") == cleaned:
            speaker["samples"] = (speaker["samples"] + [stored])[-MAX_SAMPLES_PER_SPEAKER:]
            return updated
    updated.append({"id": uuid.uuid4().hex, "name": cleaned, "samples": [stored]})
    return updated


class DuplicateNameError(ValueError):
    """เปลี่ยนชื่อไปชนกับคนที่มีอยู่แล้วในทะเบียน

    สืบทอดจาก ValueError เพื่อให้ผู้เรียกที่ดักแบบกว้าง ๆ ยังทำงานเหมือนเดิม แต่แยก
    ชนิดไว้เพราะปลายทาง (HTTP) ต้องบอกผู้ใช้คนละเรื่องกับ "ชื่อว่าง"
    """


def rename_speaker(speakers: list[dict], speaker_id: str, name: str) -> list[dict] | None:
    """ทะเบียนชุดใหม่ที่คนคนนี้เปลี่ยนชื่อแล้ว คืน None เมื่อไม่มี id นี้ในทะเบียน

    ชื่อผ่าน clean_name เหมือนตอน add_sample -- ชื่อที่แก้ทีหลังต้องปลอดภัยกับ
    transcript.md เท่ากับชื่อที่ตั้งครั้งแรก ไม่ใช่ช่องทางอ้อมให้ markdown เสียรูป

    ชื่อซ้ำกับคนอื่นถูกปฏิเสธ ไม่ใช่ยุบรวมให้เอง ทั้งที่ add_sample ถือว่า "ชื่อซ้ำ =
    คนเดิม": ตรงนั้นผู้ใช้กำลังบอกว่าเสียงนี้เป็นของคนที่มีอยู่แล้ว แต่ตรงนี้เขากำลัง
    แก้ตัวสะกด การยุบสองคนเข้าด้วยกันเพราะพิมพ์ชื่อผิดจะเอาตัวอย่างเสียงของคนละคนมา
    กองรวมกันถาวรโดยกู้ไม่ได้ -- ปฏิเสธแล้วให้เขาตัดสินใจเองเสียหายน้อยกว่ามาก

    คืนรายการชุดใหม่แทนการแก้ของเดิมในที่ แบบเดียวกับ add_sample/remove_speaker
    """
    cleaned = clean_name(name)
    if not cleaned:
        raise ValueError("ชื่อผู้พูดว่างเปล่าหลังตัดอักขระที่ใช้ไม่ได้ออก")
    if not any(speaker.get("id") == speaker_id for speaker in speakers):
        return None
    if any(
        speaker.get("name") == cleaned and speaker.get("id") != speaker_id
        for speaker in speakers
    ):
        raise DuplicateNameError(f'มี "{cleaned}" อยู่ในทะเบียนแล้ว')
    return [
        dict(speaker, name=cleaned) if speaker.get("id") == speaker_id else speaker
        for speaker in speakers
    ]


def remove_speaker(speakers: list[dict], speaker_id: str) -> list[dict]:
    return [speaker for speaker in speakers if speaker.get("id") != speaker_id]
