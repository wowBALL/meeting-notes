"""voiceprint สำหรับจำเสียงข้ามการประชุม -- แยกจากโมเดลแยกผู้พูดโดยสิ้นเชิง

ทำไมไม่ใช้ centroid ที่ pyannote คำนวณไว้แล้ว: 3.1 กับ community-1 ใช้โมเดล embedding
คนละตัว (ตรวจจาก config.yaml ของทั้งคู่: 3.1 ชี้ไป wespeaker-voxceleb-resnet34-LM ส่วน
community-1 bundle ของตัวเองมา 26MB) เวกเตอร์จึงอยู่คนละพื้นที่ และการสลับ
DIARIZATION_MODEL ทำให้ทุกคนในทะเบียนต้องลงทะเบียนใหม่

โมดูลนี้โหลดโมเดล embedding ตัวเดียวที่ตรึงไว้ด้วย EMBEDDING_MODEL แล้วสร้างเวกเตอร์เอง
จากช่วงเสียงที่คัดแล้ว -- ตัวโมเดลเป็น speaker verification จริง ๆ (VoxCeleb EER ~0.9%)
ของที่เป็นผลพลอยได้ในวิธีเดิมคือการเฉลี่ยเป็น centroid ก้อนเดียวโดยไม่คัดช่วงเสียง ไม่ใช่
ตัวโมเดล
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_EMBEDDING_MODEL
from src.gpu import cuda_device
from src.speakers import is_usable_embedding
from src.waveform import load_waveform

logger = logging.getLogger(__name__)

# ท่อนที่สั้นกว่านี้ให้เวกเตอร์ที่เชื่อไม่ได้ -- วัดจริง 2026-07-30 บน Meet1900: ท่อน 1.2
# วินาทีเทียบกับท่อน 4 วินาทีของคนเดียวกันได้ 0.628 ขณะที่ 4 วิเทียบ 4 วิ ได้ 0.682
MIN_SEGMENT_SECONDS = 1.5

# รวมเสียงสะอาดได้เท่านี้แล้วพอ -- งานวิจัย enrollment ที่ robust ต่อ noise (arXiv
# 2506.19875) รายงานว่าต้องการราว 20 วินาทีขึ้นไป เก็บเกินไปก็แค่จ่ายเวลา GPU เพิ่ม
TARGET_SECONDS = 20.0

# เพดานจำนวนท่อนต่อคน กันงาน GPU บานปลายในประชุมที่คนพูดสลับกันถี่มาก -- ถ้าท่อนยาวพอ
# TARGET_SECONDS จะมาถึงก่อนเพดานนี้เสมอ
MAX_SEGMENTS_PER_SPEAKER = 12


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """รวมช่วงที่ทับหรือติดกันเข้าด้วยกัน คืนช่วงที่เรียงแล้วและไม่ทับกันเอง"""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(
    keep: list[tuple[float, float]], remove: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """ช่วงใน `keep` ที่ไม่ถูก `remove` ทับ -- `remove` ต้องผ่าน _merge มาแล้ว"""
    result: list[tuple[float, float]] = []
    for start, end in keep:
        pieces = [(start, end)]
        for remove_start, remove_end in remove:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if remove_end <= piece_start or remove_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if remove_start > piece_start:
                    next_pieces.append((piece_start, remove_start))
                if remove_end < piece_end:
                    next_pieces.append((remove_end, piece_end))
            pieces = next_pieces
        result.extend(pieces)
    return result


def clean_intervals(turns: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """ช่วงเสียงของแต่ละ label ที่ไม่มีใครพูดทับ คีย์ด้วย label

    ลบด้วย coverage ของ *label อื่น* ไม่ใช่นับจำนวน turn ที่ทับกัน -- pyannote ปล่อย track
    ซ้อนกันภายใน label เดียวกันได้จริง (วัดกับคลิปพูดคนเดียวล้วน 2026-07-29: ป้ายเศษยาว
    0.5 วินาทีซ้อนกลางช่วง 12.8-16.5s ที่ป้ายเดิมกำลังพูดอยู่) ถ้านับ track จะกลายเป็นว่า
    คนคนหนึ่งพูดทับตัวเองแล้วเสียงสะอาดของเขาหายไปทั้งหมด

    label ที่ไม่มีช่วงสะอาดเหลือเลยจะ *ไม่มีคีย์* ในผลลัพธ์ ไม่ใช่คีย์ที่ชี้ไปลิสต์ว่าง --
    ผู้เรียกฝั่ง enroll นับจำนวนคีย์เพื่อตัดสินว่า "ไฟล์นี้มีคนพูดคนเดียวจริงไหม" คีย์ว่าง
    จะทำให้มันนับคนที่ไม่มีเสียงสะอาดเลยเป็นคน แล้วปฏิเสธไฟล์ที่ถูกต้อง

    pyannote ตั้ง embedding_exclude_overlap ให้ centroid ของมันอยู่แล้ว แต่เราเลิกใช้
    centroid นั้น จึงต้องทำขั้นนี้เอง ข้อมูลที่ต้องใช้อยู่ใน turns ครบ

    จงใจไม่ใช้ exclusive_speaker_diarization ของ community-1 ที่ให้ช่วง single-speaker มา
    ตรง ๆ: มันมีแค่ใน community-1 การพึ่งมันจะผูก voiceprint กลับไปกับโมเดลแยกผู้พูดอีกครั้ง
    ซึ่งเป็นสิ่งเดียวที่โมดูลนี้มีไว้กำจัด
    """
    coverage: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        start = float(turn["start"])
        end = float(turn["end"])
        if end <= start:
            continue
        coverage.setdefault(turn["speaker"], []).append((start, end))
    coverage = {label: _merge(spans) for label, spans in coverage.items()}

    clean: dict[str, list[tuple[float, float]]] = {}
    for label, spans in coverage.items():
        others = _merge(
            [span for other, other_spans in coverage.items() if other != label for span in other_spans]
        )
        remaining = [(start, end) for start, end in _subtract(spans, others) if end > start]
        if remaining:
            clean[label] = sorted(remaining)
    return clean


def select_intervals(
    clean: dict[str, list[tuple[float, float]]],
    min_seconds: float = MIN_SEGMENT_SECONDS,
    target_seconds: float = TARGET_SECONDS,
    max_segments: int = MAX_SEGMENTS_PER_SPEAKER,
) -> dict[str, list[tuple[float, float]]]:
    """ท่อนที่จะเอาไปเข้าโมเดล เรียงยาวไปสั้น -- label ที่ไม่มีท่อนผ่านเกณฑ์ไม่มีคีย์

    เอาท่อนยาวก่อนเพราะเวกเตอร์จากท่อนยาวนิ่งกว่า และหยุดเมื่อครบ `target_seconds`
    เพราะเสียงที่เกินจากนั้นแทบไม่ขยับค่าเฉลี่ยแล้ว แต่ยังจ่ายเวลา GPU เต็มราคา

    `(-ความยาว, start)` เป็นกุญแจการเรียง ไม่ใช่ `-ความยาว` เดี่ยว ๆ: ท่อนที่ยาวเท่ากัน
    ต้องมีลำดับที่แน่นอน ไม่งั้นรันสองครั้งกับไฟล์เดียวกันได้ voiceprint คนละตัว แล้วการ
    calibrate เกณฑ์ (ซึ่งวัดคะแนนระหว่าง voiceprint) ก็ไม่มีความหมาย

    เช็คเงื่อนไขหยุด *ก่อน* เก็บท่อน ไม่ใช่หลัง -- เก็บก่อนแล้วเช็คจะทำให้ได้เกิน
    max_segments มาหนึ่งท่อนเสมอ
    """
    selected: dict[str, list[tuple[float, float]]] = {}
    for label, intervals in clean.items():
        long_enough = [
            (start, end) for start, end in intervals if end - start >= min_seconds
        ]
        long_enough.sort(key=lambda span: (-(span[1] - span[0]), span[0]))
        chosen: list[tuple[float, float]] = []
        total = 0.0
        for start, end in long_enough:
            if total >= target_seconds or len(chosen) >= max_segments:
                break
            chosen.append((start, end))
            total += end - start
        if chosen:
            selected[label] = chosen
    return selected


def average_unit_vectors(vectors: list) -> list[float] | None:
    """ทิศทางเฉลี่ยของหลายเวกเตอร์ คืน None เมื่อหาไม่ได้

    normalize แต่ละตัวก่อนเฉลี่ย แล้ว normalize ผลลัพธ์อีกครั้ง -- cosine สนใจแค่ทิศทาง
    ท่อนที่เสียงดังกว่า (norm ใหญ่กว่า) จึงต้องไม่ได้สิทธิ์โหวตมากกว่า ส่วนน้ำหนักตาม
    ความยาวเกิดขึ้นเองแล้วจากการที่ select_intervals คัดท่อนยาวเข้ามาก่อน

    ใช้ speakers.is_usable_embedding ตัวเดิมตัดสินว่าเวกเตอร์ไหนใช้ได้ ไม่เขียนกฎชุดที่สอง
    -- กฎ "เวกเตอร์ศูนย์เทียบไม่ได้" ต้องมีคำจำกัดความเดียวในโปรเจกต์นี้ ไม่งั้นของที่ผ่าน
    ด่านหนึ่งจะไปตกอีกด่านในเส้นทางเดียวกัน

    ตัดเวกเตอร์ที่ยาวไม่เท่าเสียงส่วนใหญ่ทิ้ง: ความยาวต่างกันแปลว่ามาจากคนละโมเดล การ
    เฉลี่ยรวมกันจะได้ตัวเลขที่ไม่ได้อยู่ในพื้นที่ไหนเลย (แบบเดียวกับที่ cosine_similarity
    คืน 0.0 เมื่อความยาวไม่ตรง แทนที่จะ raise)

    ความยาวที่เลือกคือความยาวที่พบมากที่สุด และเมื่อเสมอกันจะใช้ตัวที่เวกเตอร์แรกของมัน
    โผล่ก่อนในอินพุต -- "เสียงข้างมาก" ไม่ได้มีอยู่จริงเสมอไปเมื่อเสมอกัน (เช่น 3 ท่อนจาก
    embedding รุ่นเก่ากับ 3 ท่อนจากรุ่นใหม่พอดีหลังสลับ EMBEDDING_MODEL) กติกาข้อนี้จึงต้อง
    ระบุไว้ตรง ๆ ไม่ใช่ปล่อยให้เป็นผลพลอยได้ของ hash
    """
    usable = [
        [float(value) for value in vector]
        for vector in vectors
        if is_usable_embedding(vector)
    ]
    if not usable:
        return None

    # เลือกความยาวที่พบมากที่สุด และเมื่อเสมอกันให้ใช้ตัวที่โผล่ก่อนในอินพุต -- เดิมโค้ด
    # ตรงนี้เป็น max() บน set ของความยาว ซึ่งตอนเสมอกันคืนตัวที่ set iterate เจอก่อน
    # ผลคือ 256 กับ 512 (ทั้งคู่ ≡ 0 mod 8 จึงชนกันใน hash table) สลับผู้ชนะตามลำดับของ
    # อินพุต -- voiceprint ของคนคนหนึ่งจะมีมิติต่างกันแล้วแต่ว่าเวกเตอร์ตัวไหนมาก่อน
    # ซึ่งเป็นอันตรายตัวเดียวกับที่ select_intervals ใส่ start ในกุญแจการเรียงเพื่อกัน
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for index, vector in enumerate(usable):
        size = len(vector)
        counts[size] = counts.get(size, 0) + 1
        first_seen.setdefault(size, index)
    dimension = min(counts, key=lambda size: (-counts[size], first_seen[size]))

    usable = [vector for vector in usable if len(vector) == dimension]

    total = [0.0] * dimension
    for vector in usable:
        norm = math.sqrt(sum(value * value for value in vector))
        for index, value in enumerate(vector):
            total[index] += value / norm
    mean_norm = math.sqrt(sum(value * value for value in total))
    if mean_norm == 0.0:
        return None
    return [value / mean_norm for value in total]


@dataclass(frozen=True)
class Voiceprint:
    """เวกเตอร์เสียงหนึ่งตัวของคนหนึ่งคน พร้อมหลักฐานว่ามันมาจากเสียงเท่าไหร่

    `seconds` กับ `segment_count` ไม่ใช่ของประดับ: ทั้งคู่ถูกเขียนลงทะเบียนถาวร และเป็น
    สิ่งเดียวที่ตอบได้ทีหลังว่า sample ที่จับคู่พลาดบ่อยตัวหนึ่งมาจากเสียง 3 วินาทีหรือ 20
    """

    embedding: list[float]
    seconds: float
    segment_count: int


def extract_voiceprints(audio_path, turns: list[dict], embed) -> dict[str, Voiceprint]:
    """voiceprint ต่อผู้พูดหนึ่งคน คีย์ด้วย label -- ไม่ raise ไม่ว่าเกิดอะไรขึ้น

    ผู้เรียกคือ pipeline.process_file ซึ่งกำลังบันทึกการประชุมที่อัดซ้ำไม่ได้ กฎเดิมของ
    repo นี้คือความล้มเหลวของ "การจำเสียง" ต้องไม่ทำลาย "การแยกผู้พูด" -- ฟังก์ชันนี้จึง
    กลืนทุกอย่างไว้ที่ตัวเอง เหมือนที่ diarize._speaker_embeddings เคยทำก่อนถูกลบ

    การกลืนความล้มเหลวมีสองชั้น แยกกันโดยเจตนา:
    - ระดับท่อน: `embed` คืน None ให้ท่อนไหนก็ตัดแค่ท่อนนั้นทิ้ง (ผ่าน is_usable_embedding)
    - ระดับคน: try ต่อคนด้านล่างครอบ *ทุกอย่างเกี่ยวกับคนคนนั้น* -- ไม่ใช่แค่ตัวเรียก
      embed(...) แต่รวมการประมวลผลค่าที่ embed คืนมาด้วย (นับความยาว, zip กับช่วงที่ขอ,
      กรองด้วย is_usable_embedding, เฉลี่ยด้วย average_unit_vectors, สร้าง Voiceprint แล้ว
      เขียนลง prints) ไม่ว่าจะพังตรงไหนในบล็อกนี้ก็ตัดแค่คนนั้นทิ้ง ไม่ใช่ทั้งฟังก์ชันทิ้ง
      ทุกคนเป็น {} เพราะคนหนึ่งพัง -- เคยแก้ผิดมาสองรอบแล้ว: รอบแรก try ครอบทั้งลูปทำให้คน
      หนึ่งพังแล้วทุกคนหาย รอบสอง try ครอบแค่บรรทัด embed(...) เอง แต่ len(vectors)/zip/
      is_usable_embedding/average_unit_vectors ที่กินค่าคืนของ embed ยังอยู่นอก try ทำให้
      embed ที่คืนค่าผิดสัญญา (generator/None/int ที่ไม่มี __len__) ยัง TypeError หลุดไปโดน
      except ชั้นนอกแล้วทุกคนหายเหมือนเดิม
    ท่อนเดียวที่พังต้องไม่ทำให้คนทั้งคนหาย และคนคนเดียวที่พังต้องไม่ทำให้คนอื่นในที่ประชุม
    เดียวกันหายไปด้วย -- กฎเดียวกัน ขยับขึ้นมาอีกชั้น

    try ชั้นนอก (ครอบ load_waveform/select_intervals) มีไว้สำหรับปัญหาระดับทั้งไฟล์เท่านั้น
    ไฟล์อ่านไม่ได้หรือ turns ผิดรูปทำให้ทุกคนเสีย voiceprint พร้อมกันอยู่แล้วโดยธรรมชาติ
    ของมัน จึงไม่ต้องแยกราย

    clamp ช่วงให้อยู่ในความยาวไฟล์ก่อนส่งเข้าโมเดลเสมอ: ขอบท่อนของ pyannote เกินความยาว
    ไฟล์ได้จากการ pad ของ segmentation และ Inference.crop โยน ValueError ทันทีเมื่อเจอ
    (วัดจริง 2026-07-30: end time 25.053s บนไฟล์ยาว 20.053s) ถ้าไม่ clamp ผู้พูดคนสุดท้าย
    ของทุกไฟล์มีโอกาสเสีย voiceprint ด้วยเหตุผลที่ไม่เกี่ยวกับเสียงของเขาเลย

    นับ `seconds`/`segment_count` จากท่อนที่ extract *สำเร็จ* เท่านั้น ไม่ใช่ท่อนที่ขอไป --
    ตัวเลขที่นับท่อนที่พังด้วยจะโกหกตลอดอายุของ sample นั้นในทะเบียน
    """
    try:
        waveform = load_waveform(Path(audio_path))
        duration = waveform["waveform"].shape[1] / waveform["sample_rate"]
        selected = select_intervals(clean_intervals(turns))

        prints: dict[str, Voiceprint] = {}
        for label, intervals in selected.items():
            clamped = []
            for start, end in intervals:
                start = max(0.0, start)
                end = min(end, duration)
                if end - start >= MIN_SEGMENT_SECONDS:
                    clamped.append((start, end))
            if not clamped:
                continue
            # try ต่อคน ครอบทุกอย่างของคนนี้ ตั้งแต่เรียก embed ไปจนถึงประมวลผลค่าที่มันคืน
            # และเขียนลง prints -- ไม่ใช่แค่ตัวเรียก embed(...) เอง เดิมเคยครอบแค่บรรทัดเรียก
            # แล้วปล่อยให้ len(vectors)/zip/is_usable_embedding/average_unit_vectors ที่กิน
            # ค่าคืนของ embed อยู่นอก try -- ถ้า embed ทำผิดสัญญา (คืน generator/None/int ที่
            # ไม่มี __len__ เช่น) TypeError จาก len() จะหลุดออกจาก try นี้ไปโดน except ชั้น
            # นอกที่ทิ้งทุกคนเป็น {} เหมือนบั๊กราวด์ 1 ทุกประการ แค่ทางเข้าใหม่ -- ขอบเขต
            # ของ try ต้องเป็น "ทุกอย่างเกี่ยวกับคนคนนี้" ไม่ใช่ "การเรียกฟังก์ชันเดียว" คนที่
            # คำนวณสำเร็จไปแล้วก่อนหน้าต้องไม่หายไปด้วยไม่ว่าคนหลังจะพังตรงจุดไหนในบล็อกนี้
            try:
                vectors = embed(waveform, clamped)
                if len(vectors) != len(clamped):
                    # สัญญาของ embed คือคืนลิสต์ยาวเท่าอินพุต -- ผิดสัญญาแล้ว zip จะตัดท้าย
                    # ทิ้งเงียบ ๆ (ไม่ได้ผูกผิดช่วง แต่หายไปโดยไม่มีร่องรอย) ต้องมีบรรทัดเดียว
                    # ใน log
                    logger.warning(
                        "embed คืนเวกเตอร์ %d ตัวสำหรับ %d ช่วงของ %s -- ส่วนเกินถูกตัดทิ้ง",
                        len(vectors),
                        len(clamped),
                        label,
                    )
                kept = [
                    (span, vector)
                    for span, vector in zip(clamped, vectors)
                    if is_usable_embedding(vector)
                ]
                embedding = average_unit_vectors([vector for _, vector in kept])
                if embedding is None:
                    continue
                prints[label] = Voiceprint(
                    embedding=embedding,
                    seconds=round(sum(end - start for (start, end), _ in kept), 1),
                    segment_count=len(kept),
                )
            except Exception as e:
                logger.warning(
                    "คำนวณเวกเตอร์ของ %s ไม่สำเร็จ ข้ามคนนี้ไป (คนอื่นไม่กระทบ): %s", label, e
                )
                continue
        return prints
    except Exception as e:
        # กว้างโดยตั้งใจ: soundfile/torch/pyannote โยนอะไรออกมาก็ได้จาก load_waveform หรือ
        # select_intervals -- ปัญหาระดับไฟล์ทั้งไฟล์ ไม่ใช่รายคน exception ที่หลุดออกไปจะไป
        # โดน except กว้างของ pipeline ซึ่งทิ้ง speaker_turns ทั้งชุด
        logger.warning("สร้าง voiceprint ไม่ได้ ไปต่อโดยไม่จำเสียง: %s", e)
        return {}


class PyannoteEmbedder:
    """เวกเตอร์ต่อหนึ่งช่วงเสียง จากโมเดล embedding ตัวเดียวที่ตรึงไว้

    ถือชื่อ checkpoint ติดตัวไว้เพราะปลายทางต้องติดป้ายลงทะเบียนด้วยชื่อของตัวที่ *คำนวณ
    จริง* ไม่ใช่ค่าใน config ณ ตอนที่ผู้ใช้กดยืนยัน -- คิวรอตั้งชื่ออยู่ข้ามวันได้ และผู้ใช้
    แก้ .env ระหว่างนั้นได้ (เหตุผลเดียวกับ enroll.analyze(..., model=...))

    ป้อน waveform dict เข้า crop ไม่ใช่ path: pyannote 4.x อ่านไฟล์ผ่าน torchcodec เท่านั้น
    ซึ่งโหลดไม่ขึ้นบนเครื่องนี้ (ดู src/waveform.py)
    """

    def __init__(self, inference, checkpoint: str, segment_factory):
        self._inference = inference
        self._segment = segment_factory
        self.checkpoint = checkpoint

    def __call__(self, waveform: dict, intervals: list[tuple[float, float]]) -> list:
        """เวกเตอร์เรียงตาม `intervals` -- ท่อนที่คำนวณไม่ได้เป็น None ไม่ใช่ช่องที่หายไป

        คืนลิสต์ยาวเท่าอินพุตเสมอ เพราะผู้เรียก (extract_voiceprints) zip มันกับช่วงเพื่อ
        นับว่าใช้เสียงไปกี่วินาที ลิสต์ที่สั้นกว่าจะทำให้การนับเลื่อนไปทั้งชุดเงียบ ๆ
        """
        vectors = []
        for start, end in intervals:
            try:
                raw = self._inference.crop(waveform, self._segment(start, end))
                vectors.append([float(value) for value in list(raw)])
            except Exception as e:
                logger.warning(
                    "คำนวณเวกเตอร์ของช่วง %.2f-%.2f ไม่ได้ ข้ามท่อนนี้: %s", start, end, e
                )
                vectors.append(None)
        return vectors


def load_embedder(
    hf_token: str, checkpoint: str = DEFAULT_EMBEDDING_MODEL
) -> PyannoteEmbedder:
    """โหลดโมเดล embedding ตัวเดียวไว้ใช้ซ้ำทั้ง process

    `window="whole"` แปลว่า "เอาเสียงทั้งท่อนที่ให้มาเป็นหนึ่งหน่วย" ซึ่งเป็นสิ่งที่เราต้องการ
    ตรง ๆ -- ค่า default `"sliding"` จะคืนลำดับเวกเตอร์ตามหน้าต่างเลื่อน แล้วเราต้องมาเฉลี่ย
    ซ้อนกับการเฉลี่ยของตัวเองอีกชั้นโดยไม่ได้อะไรเพิ่ม

    ไม่มีเทสต์ที่รันโมเดลจริง แบบเดียวกับ load_diarization_pipeline -- แต่การต่อสาย
    (checkpoint, window, device, การกลืน exception ต่อท่อน) มีเทสต์ครบ
    """
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    model = Model.from_pretrained(checkpoint, token=hf_token)
    inference = Inference(model, window="whole", device=cuda_device())
    return PyannoteEmbedder(inference, checkpoint, Segment)
