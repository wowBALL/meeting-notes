import pytest

from tools.calibrate_voiceprints import score_pairs, suggest_thresholds


def _vp(*values):
    return list(values)


def test_score_pairs_counts_a_cross_file_pair_of_the_same_person_as_same():
    voiceprints = {
        ("a.ogg", "SPEAKER_00"): _vp(1.0, 0.0),
        ("b.ogg", "SPEAKER_03"): _vp(1.0, 0.0),
    }
    truth = {("a.ogg", "SPEAKER_00"): "satit", ("b.ogg", "SPEAKER_03"): "satit"}

    same, different = score_pairs(voiceprints, truth)

    assert same == pytest.approx([1.0])
    assert different == []


def test_score_pairs_ignores_pairs_inside_one_file():
    # คู่ในไฟล์เดียวกันวัดความง่ายของการแยกผู้พูดในไฟล์นั้น ไม่ใช่ความยากของการจำเสียงข้าม
    # การประชุม ซึ่งเป็นงานจริง -- และคะแนนของมันสูงกว่าความจริงราว 0.05-0.15 (วัด 2026-07-30)
    voiceprints = {
        ("a.ogg", "SPEAKER_00"): _vp(1.0, 0.0),
        ("a.ogg", "SPEAKER_01"): _vp(0.0, 1.0),
    }
    truth = {("a.ogg", "SPEAKER_00"): "satit", ("a.ogg", "SPEAKER_01"): "ton"}

    assert score_pairs(voiceprints, truth) == ([], [])


def test_score_pairs_skips_a_label_with_no_truth_entry():
    # ป้ายที่ไม่มีเฉลยอาจเป็นคนเดียวกับป้ายอื่นที่ pyannote แยกออกมา การเดาว่าเป็นคนอื่นจะดัน
    # เพดานกองคนละคนขึ้นเองแล้วทำให้เกณฑ์ที่คำนวณได้เข้มเกินจริง
    voiceprints = {
        ("a.ogg", "SPEAKER_00"): _vp(1.0, 0.0),
        ("b.ogg", "SPEAKER_01"): _vp(0.0, 1.0),
    }
    truth = {("a.ogg", "SPEAKER_00"): "satit"}

    assert score_pairs(voiceprints, truth) == ([], [])


def test_score_pairs_counts_a_cross_file_pair_of_different_people_as_different():
    voiceprints = {
        ("a.ogg", "SPEAKER_00"): _vp(1.0, 0.0),
        ("b.ogg", "SPEAKER_01"): _vp(0.0, 1.0),
    }
    truth = {("a.ogg", "SPEAKER_00"): "satit", ("b.ogg", "SPEAKER_01"): "ton"}

    same, different = score_pairs(voiceprints, truth)

    assert same == []
    assert different == pytest.approx([0.0])


def test_score_pairs_expands_the_wildcard_to_every_label_in_that_file():
    voiceprints = {
        ("clip.ogg", "SPEAKER_00"): _vp(1.0, 0.0),
        ("clip.ogg", "SPEAKER_01"): _vp(1.0, 0.0),
        ("m.ogg", "SPEAKER_05"): _vp(1.0, 0.0),
    }
    truth = {("clip.ogg", "*"): "satit", ("m.ogg", "SPEAKER_05"): "satit"}

    same, different = score_pairs(voiceprints, truth)

    assert len(same) == 2      # ทั้งสองป้ายของ clip เทียบกับป้ายในประชุม
    assert different == []


def test_suggest_thresholds_puts_high_between_the_two_groups():
    result = suggest_thresholds(same=[0.75, 0.80], different=[0.43, 0.63])

    assert result["overlap"] is False
    assert 0.63 < result["high"] <= 0.75
    assert 0.43 < result["low"] <= result["high"]


def test_suggest_thresholds_reports_an_overlap_instead_of_picking_a_number():
    # สองกองซ้อนกันแปลว่าไม่มีเกณฑ์คู่ไหนปลอดภัย ต้องแก้ที่อื่น (เพิ่ม sample, narrow
    # gallery) การเลือกเลขที่ "ดูดีที่สุด" คือการซ่อนปัญหาไว้ใต้ค่าคอนฟิก
    result = suggest_thresholds(same=[0.55, 0.80], different=[0.40, 0.62])

    assert result["overlap"] is True


def test_suggest_thresholds_handles_having_no_pairs_at_all():
    result = suggest_thresholds(same=[], different=[])

    assert result["overlap"] is True
    assert result["high"] is None
