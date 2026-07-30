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

from src.speakers import is_usable_embedding

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
    """
    usable = [
        [float(value) for value in vector]
        for vector in vectors
        if is_usable_embedding(vector)
    ]
    if not usable:
        return None
    dimension = max(
        {len(vector) for vector in usable},
        key=lambda size: sum(1 for vector in usable if len(vector) == size),
    )
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
