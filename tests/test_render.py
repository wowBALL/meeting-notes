from src.render import (
    build_speaker_labels,
    format_timestamp,
    participants_line,
    render_transcript_markdown,
    replace_participants_line,
    speaker_stats,
    speaker_table,
)


def test_format_timestamp_formats_minutes_and_seconds():
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(5) == "00:05"
    assert format_timestamp(0) == "00:00"


def test_render_transcript_markdown_labels_speakers_in_order_of_appearance():
    merged = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01", "text": "ครับผม"},
        {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00", "text": "ต่อนะครับ"},
    ]

    result = render_transcript_markdown(merged)

    assert result == (
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:00]: สวัสดีครับ\n\n"
        "**ผู้พูด 2** [00:02]: ครับผม\n\n"
        "**ผู้พูด 1** [00:05]: ต่อนะครับ"
    )


def test_render_transcript_markdown_prepends_note_when_diarization_failed():
    merged = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_UNKNOWN", "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_UNKNOWN", "text": "ครับผม"},
    ]

    result = render_transcript_markdown(merged, diarization_failed=True)

    assert result == (
        "# Transcript\n\n"
        '> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ ข้อความทั้งหมดจึงแสดงเป็น "ผู้พูด 1" เพียงคนเดียว\n\n'
        "**ผู้พูด 1** [00:00]: สวัสดีครับ\n\n"
        "**ผู้พูด 1** [00:02]: ครับผม"
    )


def test_render_transcript_markdown_no_note_when_diarization_succeeded():
    merged = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "สวัสดีครับ"},
    ]

    result = render_transcript_markdown(merged, diarization_failed=False)

    assert "ไม่สามารถแยกผู้พูดได้อัตโนมัติ" not in result
    assert result == "# Transcript\n\n**ผู้พูด 1** [00:00]: สวัสดีครับ"


def test_build_speaker_labels_numbers_unknown_speakers_in_order_of_appearance():
    merged = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "ก"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "ข"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "ค"},
    ]

    assert build_speaker_labels(merged) == {
        "SPEAKER_01": "ผู้พูด 1",
        "SPEAKER_00": "ผู้พูด 2",
    }


def test_build_speaker_labels_uses_real_names_and_does_not_spend_numbers_on_them():
    # SPEAKER_00 รู้จักแล้ว จึงต้องไม่กินเลข 1 ไป -- ไม่งั้นคนแรกที่ยังไม่รู้จักจะ
    # กลายเป็น "ผู้พูด 2" ทั้งที่เป็นคนเดียวที่ไม่มีชื่อ
    merged = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "ก"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "ข"},
    ]

    labels = build_speaker_labels(merged, {"SPEAKER_00": "สมหญิง็ม"})

    assert labels == {"SPEAKER_00": "สมหญิง็ม", "SPEAKER_01": "ผู้พูด 1"}


def test_build_speaker_labels_ignores_names_for_speakers_not_in_the_transcript():
    merged = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "ก"}]

    assert build_speaker_labels(merged, {"SPEAKER_09": "พี่บี"}) == {"SPEAKER_00": "ผู้พูด 1"}


def test_render_transcript_markdown_writes_known_names_in_place_of_the_label():
    merged = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01", "text": "ครับผม"},
    ]

    result = render_transcript_markdown(merged, speaker_names={"SPEAKER_00": "สมหญิง็ม"})

    assert result == (
        "# Transcript\n\n"
        "**สมหญิง็ม** [00:00]: สวัสดีครับ\n\n"
        "**ผู้พูด 1** [00:02]: ครับผม"
    )


def test_render_transcript_markdown_without_names_is_unchanged():
    merged = [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "สวัสดีครับ"}]

    assert render_transcript_markdown(merged, speaker_names=None) == render_transcript_markdown(merged)


_TRANSCRIPT = """# Transcript

**Wachirakorn (สอง)** [00:10]: เดี๋ยวผมแชร์จอนะครับ ขอเปิด Odoo ก่อน

**ผู้พูด 1** [00:20]: ครับ

**Wachirakorn (สอง)** [00:30]: อันนี้คือ payslip ที่ engine คำนวณออกมา

**ผู้พูด 1** [00:40]: พี่บอลว่าไงบ้าง

**ผู้พูด 2** [08:00]: ผมเพิ่งเข้ามาครับ ขอโทษที่สายนะครับทุกคน
"""


def test_speaker_stats_counts_characters_not_lines():
    """ความยาวบรรทัดขึ้นกับวิธีที่ whisper ตัด segment -- ตัวเลขที่ใช้จัดอันดับผู้พูด
    จึงต้องเป็นอักขระ ไม่ใช่จำนวนบรรทัด"""
    stats = {entry["speaker"]: entry for entry in speaker_stats(_TRANSCRIPT)}

    assert stats["Wachirakorn (สอง)"]["lines"] == 2
    assert stats["Wachirakorn (สอง)"]["turns"] == 2
    assert stats["ผู้พูด 1"]["lines"] == 2
    # ผู้พูด 2 พูดบรรทัดเดียวแต่ยาวกว่าสองบรรทัดของผู้พูด 1 รวมกัน
    assert stats["ผู้พูด 2"]["chars"] > stats["ผู้พูด 1"]["chars"]
    assert [entry["speaker"] for entry in speaker_stats(_TRANSCRIPT)][0] == "Wachirakorn (สอง)"


def test_speaker_stats_separates_voiceprint_names_from_placeholders():
    """ชื่อที่มาจากลายเสียงคือหลักฐาน ป้าย "ผู้พูด N" คือช่องว่าง -- สรุปต้องบอกคนอ่าน
    ได้ว่ามีชื่อจริงกี่คน ไม่ใช่ปล่อยให้เข้าใจว่ารู้จักทุกคน"""
    stats = {entry["speaker"]: entry for entry in speaker_stats(_TRANSCRIPT)}

    assert stats["Wachirakorn (สอง)"]["identified"] is True
    assert stats["ผู้พูด 1"]["identified"] is False
    assert stats["ผู้พูด 2"]["identified"] is False


def test_participants_line_counts_instead_of_guessing():
    line = participants_line(_TRANSCRIPT)

    assert line.startswith("ผู้เข้าร่วม: 3 เสียง (ระบุตัวได้จากลายเสียง 1 คน)")
    assert "Wachirakorn (สอง)" in line
    assert "ผู้พูด 1" in line
    # "พี่บอล" ถูกพูดถึงในบทสนทนา แต่ไม่ใช่ป้ายผู้พูด จึงต้องไม่โผล่ในรายชื่อ
    assert "พี่บอล" not in line


def test_participants_line_marks_whoever_was_not_there_from_the_start():
    """คนที่เข้ามากลางทางไม่ได้อยู่ตอนที่ครึ่งแรกตกลงอะไรกัน"""
    line = participants_line(_TRANSCRIPT)

    assert "ผู้พูด 2 " in line and "(พูดครั้งแรก 08:00)" in line
    assert "Wachirakorn (สอง)" in line.split("(พูดครั้งแรก")[0]


def test_replace_participants_line_removes_whatever_the_model_wrote():
    """prompt เป็นคำขอ ไม่ใช่การบังคับ และประชุมเก่าก็มีบรรทัดนั้นติดมาด้วย"""
    summary = "ผู้เข้าร่วม: สมชาย, สมหญิง, พี่บอล\n\n## หัวข้อที่คุยกัน\n- เรื่องที่คุยกัน"

    result = replace_participants_line(summary, _TRANSCRIPT)

    assert result.count("ผู้เข้าร่วม:") == 1
    assert "สมชาย" not in result
    assert result.startswith("ผู้เข้าร่วม: 3 เสียง")
    assert "## หัวข้อที่คุยกัน" in result


def test_replace_participants_line_leaves_the_summary_alone_without_a_transcript():
    """transcript ว่าง (หรือ parse ไม่ออก) ต้องไม่ทำให้สรุปที่จ่ายเงินไปแล้วเสียหาย"""
    summary = "## หัวข้อที่คุยกัน\n- เรื่องที่คุยกัน"

    assert replace_participants_line(summary, "") == summary


def test_speaker_table_keeps_the_diagnostic_numbers():
    table = speaker_table(_TRANSCRIPT)

    assert table.startswith("## ผู้พูดใน transcript")
    assert "| ผู้พูด 2 |" in table
    assert "08:00" in table
