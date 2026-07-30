import math

import numpy as np
import pytest
import soundfile as sf

from src.voiceprint import (
    MAX_SEGMENTS_PER_SPEAKER,
    MIN_SEGMENT_SECONDS,
    TARGET_SECONDS,
    Voiceprint,
    average_unit_vectors,
    clean_intervals,
    extract_voiceprints,
    select_intervals,
)


def _turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_clean_intervals_keeps_a_turn_nobody_talks_over():
    turns = [_turn(0.0, 5.0, "A"), _turn(10.0, 15.0, "B")]

    assert clean_intervals(turns) == {"A": [(0.0, 5.0)], "B": [(10.0, 15.0)]}


def test_clean_intervals_removes_the_part_two_people_share():
    # A พูด 0-10 B แทรก 4-6 -- ช่วง 4-6 เป็นเสียงผสม เอาไปทำ voiceprint = พิษ
    turns = [_turn(0.0, 10.0, "A"), _turn(4.0, 6.0, "B")]

    assert clean_intervals(turns) == {"A": [(0.0, 4.0), (6.0, 10.0)]}


def test_clean_intervals_drops_a_label_that_is_entirely_overlapped():
    # B พูดทับ A ทั้งช่วงของตัวเอง -- B ไม่มีเสียงสะอาดเลย จึงต้องไม่มีคีย์ ไม่ใช่ลิสต์ว่าง
    turns = [_turn(0.0, 10.0, "A"), _turn(3.0, 7.0, "B")]

    result = clean_intervals(turns)

    assert "B" not in result
    assert result == {"A": [(0.0, 3.0), (7.0, 10.0)]}


def test_clean_intervals_drops_everyone_the_long_talker_covers():
    # A พูดคลุม 0-12 ซึ่งกินทั้งช่วงของ B และของ C -- ทั้งคู่ไม่มีเสียงสะอาดเลยแม้จะพูด
    # คนละเวลากัน เหลือแค่หัวกับท้ายของ A ที่ไม่มีใครทับ
    #
    # เทสต์นี้เคยเขียนผิดตอนร่างแผน (คาดว่า C จะเหลือ (5.0, 9.0) เพราะลบแค่ B ออกจาก C)
    # จับได้จากการรันโค้ดในแผนกับเทสต์ในแผนก่อนลงมือ -- กฎคือลบ coverage ของ *ทุก* label
    # อื่นรวมกัน ไม่ใช่ลบทีละคู่
    turns = [_turn(0.0, 12.0, "A"), _turn(2.0, 5.0, "B"), _turn(4.0, 9.0, "C")]

    assert clean_intervals(turns) == {"A": [(0.0, 2.0), (9.0, 12.0)]}


def test_clean_intervals_keeps_clean_speech_for_all_three_when_turns_chain():
    # สามคนพูดต่อกันแบบคาบเกี่ยวทีละคู่: A 0-6, B 4-10, C 12-16 -- ทุกคนเหลือเสียงสะอาด
    turns = [_turn(0.0, 6.0, "A"), _turn(4.0, 10.0, "B"), _turn(12.0, 16.0, "C")]

    assert clean_intervals(turns) == {
        "A": [(0.0, 4.0)],
        "B": [(6.0, 10.0)],
        "C": [(12.0, 16.0)],
    }


def test_clean_intervals_does_not_treat_a_labels_own_overlap_as_overlap():
    # pyannote ปล่อย track ซ้อนกันใน label เดียวกันได้จริง (วัดกับคลิปพูดคนเดียว:
    # ป้ายเศษ 0.5 วิ ซ้อนกลางช่วง 12.8-16.5s ของป้ายเดิม) ถ้านับ track จะกลายเป็นว่า
    # คนคนหนึ่งพูดทับตัวเองแล้วเสียงสะอาดของเขาหายไปหมด
    turns = [_turn(0.0, 10.0, "A"), _turn(4.0, 6.0, "A")]

    assert clean_intervals(turns) == {"A": [(0.0, 10.0)]}


def test_clean_intervals_merges_touching_turns_of_the_same_label():
    turns = [_turn(0.0, 5.0, "A"), _turn(5.0, 9.0, "A")]

    assert clean_intervals(turns) == {"A": [(0.0, 9.0)]}


def test_clean_intervals_trims_only_the_overlapping_edge():
    turns = [_turn(0.0, 6.0, "A"), _turn(5.0, 11.0, "B")]

    assert clean_intervals(turns) == {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]}


def test_clean_intervals_returns_nothing_for_no_turns():
    assert clean_intervals([]) == {}


def test_clean_intervals_ignores_a_zero_length_turn():
    # merge/subtract ที่ปล่อยช่วงยาว 0 ผ่านจะสร้างคีย์ที่ select_intervals ต้องกรองทิ้ง
    # ทีหลัง -- กรองที่ต้นทางแทน ผลลัพธ์จะไม่มีช่วงที่ไม่มีเสียงอยู่เลย
    turns = [_turn(3.0, 3.0, "A"), _turn(0.0, 2.0, "B")]

    assert clean_intervals(turns) == {"B": [(0.0, 2.0)]}


def test_select_intervals_drops_segments_shorter_than_the_minimum():
    # ท่อน 1.2 วิของคนเดียวกันได้คะแนน 0.628 เทียบกับ 0.682 ของท่อน 4 วิ (วัด 2026-07-30)
    clean = {"A": [(0.0, 1.4), (10.0, 13.0)]}

    assert select_intervals(clean) == {"A": [(10.0, 13.0)]}


def test_select_intervals_drops_a_label_with_no_long_enough_segment():
    clean = {"A": [(0.0, 1.0), (2.0, 3.2)], "B": [(5.0, 9.0)]}

    result = select_intervals(clean)

    assert "A" not in result
    assert result == {"B": [(5.0, 9.0)]}


def test_select_intervals_takes_the_longest_segments_first():
    clean = {"A": [(0.0, 2.0), (10.0, 20.0), (30.0, 35.0)]}

    assert select_intervals(clean) == {"A": [(10.0, 20.0), (30.0, 35.0), (0.0, 2.0)]}


def test_select_intervals_stops_once_it_has_enough_seconds():
    # 3 ท่อน x 8 วิ = 24 วิ ผ่าน TARGET_SECONDS (20) ที่ท่อนที่สาม ท่อนที่สี่ไม่ถูกเก็บ
    clean = {"A": [(0.0, 8.0), (10.0, 18.0), (20.0, 28.0), (30.0, 38.0)]}

    result = select_intervals(clean)

    assert result["A"] == [(0.0, 8.0), (10.0, 18.0), (20.0, 28.0)]
    assert sum(end - start for start, end in result["A"]) >= TARGET_SECONDS


def test_select_intervals_respects_the_segment_cap_when_turns_are_short():
    # ท่อนสั้น ๆ จำนวนมาก: เพดานจำนวนท่อนต้องหยุดก่อนที่จะครบ 20 วิ
    clean = {"A": [(i * 10.0, i * 10.0 + 1.6) for i in range(30)]}

    result = select_intervals(clean)

    assert len(result["A"]) == MAX_SEGMENTS_PER_SPEAKER
    assert sum(end - start for start, end in result["A"]) < TARGET_SECONDS


def test_select_intervals_is_deterministic_when_lengths_tie():
    # ผลที่เรียงไม่นิ่งแปลว่ารันสองครั้งได้ voiceprint คนละตัวจากไฟล์เดียวกัน ซึ่งทำให้
    # การ calibrate เกณฑ์ไม่มีความหมายเลย -- ตัดสินด้วย start เมื่อความยาวเท่ากัน
    clean = {"A": [(30.0, 33.0), (10.0, 13.0), (20.0, 23.0)]}

    assert select_intervals(clean)["A"] == [(10.0, 13.0), (20.0, 23.0), (30.0, 33.0)]


def test_select_intervals_accepts_a_segment_exactly_at_the_minimum():
    clean = {"A": [(0.0, MIN_SEGMENT_SECONDS)]}

    assert select_intervals(clean) == {"A": [(0.0, MIN_SEGMENT_SECONDS)]}


def test_select_intervals_returns_nothing_for_an_empty_input():
    assert select_intervals({}) == {}


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def test_average_unit_vectors_returns_a_unit_vector():
    result = average_unit_vectors([[3.0, 0.0], [0.0, 4.0]])

    assert _norm(result) == pytest.approx(1.0)


def test_average_unit_vectors_normalises_before_averaging_not_after():
    # ท่อนที่ norm ใหญ่กว่า (เสียงดังกว่า) ต้องไม่ได้สิทธิ์โหวตมากกว่า -- cosine สนใจแค่
    # ทิศทาง น้ำหนักตามความยาวเกิดขึ้นเองแล้วจากการที่ท่อนยาวถูกคัดเข้ามาก่อน
    quiet = [[1.0, 0.0], [0.0, 1.0]]
    one_is_loud = [[100.0, 0.0], [0.0, 1.0]]

    assert average_unit_vectors(quiet) == pytest.approx(average_unit_vectors(one_is_loud))


def test_average_unit_vectors_of_one_vector_is_that_direction():
    assert average_unit_vectors([[0.0, 5.0]]) == pytest.approx([0.0, 1.0])


def test_average_unit_vectors_skips_a_zero_vector():
    # เวกเตอร์ศูนย์ไม่มีทิศทาง normalize ไม่ได้ -- ต้องข้าม ไม่ใช่ทำให้ทั้งชุดเสีย
    assert average_unit_vectors([[0.0, 0.0], [0.0, 3.0]]) == pytest.approx([0.0, 1.0])


def test_average_unit_vectors_returns_none_for_no_usable_vectors():
    assert average_unit_vectors([]) is None
    assert average_unit_vectors([[0.0, 0.0]]) is None
    assert average_unit_vectors([None, "ไม่ใช่เวกเตอร์"]) is None


def test_average_unit_vectors_returns_none_when_directions_cancel_out():
    # เกิดได้ในทางทฤษฎีเท่านั้น แต่ค่าเฉลี่ยที่ norm เป็น 0 normalize ไม่ได้ และถ้าปล่อย
    # ผ่านไปมันจะกลายเป็นเวกเตอร์ที่ "เหมือน" กับเวกเตอร์ศูนย์อื่นทุกตัวในทะเบียน
    assert average_unit_vectors([[1.0, 0.0], [-1.0, 0.0]]) is None


def test_average_unit_vectors_ignores_vectors_of_a_different_length():
    # ความยาวไม่เท่ากันแปลว่ามาจากคนละโมเดล เอามาเฉลี่ยรวมกันไม่ได้เลย
    assert average_unit_vectors([[0.0, 3.0], [1.0, 2.0, 3.0]]) == pytest.approx([0.0, 1.0])


def test_average_unit_vectors_breaks_a_length_tie_by_first_appearance_not_hash_order():
    # 3 ท่อนจาก embedding รุ่นเก่า (256 มิติ) เจอ 3 ท่อนจากรุ่นใหม่ (512 มิติ) พอดี --
    # เกิดได้จริงหลังสลับ EMBEDDING_MODEL แล้วทะเบียนมีทั้งสองรุ่นปนกันชั่วคราว "เสียงข้าง
    # มาก" ไม่มีอยู่จริงตอนเสมอกันแบบนี้ กติกาที่ประกาศไว้ (เอกสารของฟังก์ชันนี้) คือใช้ตัว
    # ที่เวกเตอร์แรกของมันโผล่ก่อนในอินพุต
    #
    # โค้ดเดิมเลือกด้วย max() บน set ของความยาว ซึ่งพอเสมอกันจะคืนตัวที่ set เจอก่อนตามผัง
    # hash table ไม่ใช่ตามลำดับอินพุต -- 256 กับ 512 ที่ไม่มีท่อนแปลกปนเลยบังเอิญให้ผลตรง
    # กับลำดับอินพุตพอดี (ชนกันใน hash table แบบที่ insertion order ยังกำหนดผลได้) จึงต้อง
    # ผสมเวกเตอร์ความยาวอื่นที่ใช้ได้ (แต่ไม่ใช่ผู้ชนะ) ปนเข้าไปด้วยเพื่อบังคับให้ hash table
    # ขยาย/จัดเรียงใหม่จนลำดับ iterate ไม่ตรงกับลำดับอินพุตอีกต่อไป -- ค่าความยาว 1, 6, 8
    # นี้ทดสอบแล้วว่าทำให้โค้ดเดิมเลือก "ผิด" กติกาของตัวเอง (ได้ 512 ตอนที่ 256 มาก่อน และ
    # ได้ 256 ตอนที่ 512 มาก่อน คือสลับกันหมด) พิสูจน์ว่าพึ่ง set iteration order ไม่ได้เลย
    old_model = [[1.0] * 256] * 3
    new_model = [[1.0] * 512] * 3
    other_stray_lengths = [[1.0] * 1, [1.0] * 6, [1.0] * 8]

    old_first = other_stray_lengths + old_model + new_model
    new_first = other_stray_lengths + new_model + old_model

    assert len(average_unit_vectors(old_first)) == 256
    assert len(average_unit_vectors(new_first)) == 512


def _write_wav(path, seconds, sample_rate=16000):
    frames = int(seconds * sample_rate)
    sf.write(path, np.sin(np.linspace(0.0, 400.0, frames, dtype="float32")), sample_rate)
    return path


class _RecordingEmbedder:
    """embed ปลอมที่จดว่าถูกถามด้วยช่วงไหน แล้วคืนเวกเตอร์ที่เดาผลได้"""

    def __init__(self, vector_for=None, fail_on=None, raises=False):
        self.calls = []
        self._vector_for = vector_for or (lambda start, end: [end - start, 1.0])
        self._fail_on = fail_on or set()
        self._raises = raises

    def __call__(self, waveform, intervals):
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append(list(intervals))
        return [
            None if (start, end) in self._fail_on else self._vector_for(start, end)
            for start, end in intervals
        ]


def test_extract_voiceprints_returns_one_print_per_speaker(tmp_path):
    audio = _write_wav(tmp_path / "a.wav", 40.0)
    turns = [_turn(0.0, 12.0, "A"), _turn(14.0, 24.0, "B")]

    result = extract_voiceprints(audio, turns, _RecordingEmbedder())

    assert sorted(result) == ["A", "B"]
    assert isinstance(result["A"], Voiceprint)
    assert result["A"].seconds == pytest.approx(12.0)
    assert result["A"].segment_count == 1


def test_extract_voiceprints_clamps_intervals_to_the_end_of_the_audio(tmp_path):
    # pyannote คืนขอบท่อนที่เกินความยาวไฟล์ได้ และ Inference.crop โยน ValueError ทันที
    # (วัดจริง: "end time (t=25.053s) greater than waveform file duration (20.053s)")
    audio = _write_wav(tmp_path / "a.wav", 20.0)
    turns = [_turn(10.0, 25.0, "A")]
    embedder = _RecordingEmbedder()

    result = extract_voiceprints(audio, turns, embedder)

    assert embedder.calls == [[(10.0, 20.0)]]
    assert result["A"].seconds == pytest.approx(10.0)


def test_extract_voiceprints_drops_an_interval_clamping_made_too_short(tmp_path):
    # ท่อนที่เหลือ 0.5 วิหลัง clamp สั้นกว่าเกณฑ์ -- ต้องไม่ถูกส่งไปให้โมเดล
    audio = _write_wav(tmp_path / "a.wav", 20.0)
    turns = [_turn(2.0, 8.0, "A"), _turn(19.5, 24.0, "A")]
    embedder = _RecordingEmbedder()

    extract_voiceprints(audio, turns, embedder)

    assert embedder.calls == [[(2.0, 8.0)]]


def test_extract_voiceprints_has_no_key_for_a_speaker_with_only_short_turns(tmp_path):
    audio = _write_wav(tmp_path / "a.wav", 40.0)
    turns = [_turn(0.0, 12.0, "A"), _turn(20.0, 21.0, "B")]

    result = extract_voiceprints(audio, turns, _RecordingEmbedder())

    assert "B" not in result


def test_extract_voiceprints_survives_one_failed_segment(tmp_path):
    # ท่อนเดียวที่พังต้องไม่ทำให้คนทั้งคนหาย -- กฎ "ความล้มเหลวของการจำเสียงต้องไม่ทำลาย
    # ของที่ใหญ่กว่า" ขยายลงมาถึงระดับท่อน
    audio = _write_wav(tmp_path / "a.wav", 60.0)
    turns = [_turn(0.0, 10.0, "A"), _turn(20.0, 28.0, "A")]
    embedder = _RecordingEmbedder(fail_on={(0.0, 10.0)})

    result = extract_voiceprints(audio, turns, embedder)

    assert result["A"].segment_count == 1
    assert result["A"].seconds == pytest.approx(8.0)


def test_extract_voiceprints_never_raises_when_the_embedder_explodes(tmp_path):
    # ผู้เรียกคือ pipeline.process_file ซึ่งกำลังบันทึกประชุมที่อัดซ้ำไม่ได้
    audio = _write_wav(tmp_path / "a.wav", 40.0)
    turns = [_turn(0.0, 12.0, "A")]

    assert extract_voiceprints(audio, turns, _RecordingEmbedder(raises=True)) == {}


def test_extract_voiceprints_survives_one_failed_speaker(tmp_path):
    # คนหนึ่งพังไม่ใช่เหตุที่คนอื่นต้องเสีย voiceprint ไปด้วย -- ก่อนแก้ ทั้งลูปต่อผู้พูดอยู่
    # ใต้ try/except เดียว ถ้า B พังหลังจาก A คำนวณสำเร็จไปแล้ว ผลลัพธ์ทั้งชุดจะถูกทิ้งเป็น
    # {} ทั้งที่เสียงของ A ไม่มีอะไรผิดเลย -- นี่คือกฎเดียวกับที่
    # test_extract_voiceprints_survives_one_failed_segment ทดสอบไว้ระดับท่อน ขยับขึ้นมา
    # ระดับคน
    audio = _write_wav(tmp_path / "a.wav", 60.0)
    turns = [_turn(0.0, 12.0, "A"), _turn(20.0, 32.0, "B")]

    def embed(waveform, intervals):
        if intervals[0][0] == 20.0:  # ท่อนของ B เท่านั้นที่พัง
            raise RuntimeError("boom")
        return [[end - start, 1.0] for start, end in intervals]

    result = extract_voiceprints(audio, turns, embed)

    assert "A" in result
    assert isinstance(result["A"], Voiceprint)
    assert "B" not in result


def test_extract_voiceprints_never_raises_when_the_audio_cannot_be_read(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"fake audio data")

    assert extract_voiceprints(broken, [_turn(0.0, 12.0, "A")], _RecordingEmbedder()) == {}


def test_extract_voiceprints_returns_nothing_for_no_turns(tmp_path):
    audio = _write_wav(tmp_path / "a.wav", 10.0)

    assert extract_voiceprints(audio, [], _RecordingEmbedder()) == {}


def test_extract_voiceprints_counts_only_the_segments_it_actually_used(tmp_path):
    # seconds/segment_count ถูกเขียนลงทะเบียนถาวร ถ้ามันนับท่อนที่ extract ไม่สำเร็จด้วย
    # การสืบย้อนว่า "sample นี้มาจากเสียงกี่วินาที" จะโกหกตลอดอายุของ sample นั้น
    audio = _write_wav(tmp_path / "a.wav", 60.0)
    turns = [_turn(0.0, 10.0, "A"), _turn(15.0, 21.0, "A"), _turn(30.0, 34.0, "A")]
    embedder = _RecordingEmbedder(fail_on={(15.0, 21.0)})

    result = extract_voiceprints(audio, turns, embedder)

    assert result["A"].segment_count == 2
    assert result["A"].seconds == pytest.approx(14.0)
