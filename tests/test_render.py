from src.render import format_timestamp, render_transcript_markdown


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
