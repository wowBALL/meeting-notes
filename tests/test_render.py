from src.render import build_speaker_labels, format_timestamp, render_transcript_markdown


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
