from src.voiceprint import (
    MAX_SEGMENTS_PER_SPEAKER,
    MIN_SEGMENT_SECONDS,
    TARGET_SECONDS,
    clean_intervals,
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
