from src.chunk import (
    estimate_tokens,
    parse_timestamp,
    parse_transcript_segments,
    split_into_chunks,
)

TRANSCRIPT = """# Transcript

**ผู้พูด 1** [00:00]: สวัสดีครับ

**ผู้พูด 2** [00:05]: สวัสดีครับ วันนี้คุยเรื่องอะไร

**ผู้พูด 1** [01:30]: เรื่องระบบใหม่"""


def _segment(start_seconds: int, text: str) -> dict:
    return {"raw": f"**ผู้พูด 1** [00:00]: {text}", "start_seconds": start_seconds}


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 11) == 10
    assert estimate_tokens("a" * 12) == 11  # rounds up


def test_parse_timestamp_converts_minutes_and_seconds():
    assert parse_timestamp("00:00") == 0
    assert parse_timestamp("01:30") == 90


def test_parse_timestamp_handles_minutes_beyond_an_hour():
    # format_timestamp does not wrap at 60 minutes
    assert parse_timestamp("185:30") == 11130


def test_parse_transcript_segments_extracts_blocks_and_times():
    segments = parse_transcript_segments(TRANSCRIPT)

    assert [s["start_seconds"] for s in segments] == [0, 5, 90]
    assert segments[0]["raw"] == "**ผู้พูด 1** [00:00]: สวัสดีครับ"


def test_parse_transcript_segments_skips_heading_and_diarization_note():
    markdown = (
        "# Transcript\n\n"
        '> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ ข้อความทั้งหมดจึงแสดงเป็น "ผู้พูด 1" เพียงคนเดียว\n\n'
        "**ผู้พูด 1** [00:10]: เนื้อหา"
    )

    segments = parse_transcript_segments(markdown)

    assert len(segments) == 1
    assert segments[0]["start_seconds"] == 10


def test_split_into_chunks_returns_empty_for_no_segments():
    assert split_into_chunks([], max_tokens=100, overlap_tokens=10) == []


def test_split_into_chunks_keeps_short_transcript_in_one_chunk():
    segments = parse_transcript_segments(TRANSCRIPT)

    chunks = split_into_chunks(segments, max_tokens=10_000, overlap_tokens=100)

    assert len(chunks) == 1
    assert chunks[0]["start_seconds"] == 0
    assert chunks[0]["end_seconds"] == 90
    assert "สวัสดีครับ" in chunks[0]["text"]


def test_split_into_chunks_splits_when_budget_exceeded():
    segments = [_segment(i * 10, "x" * 100) for i in range(6)]
    per_segment = estimate_tokens(segments[0]["raw"])

    chunks = split_into_chunks(
        segments, max_tokens=per_segment * 2, overlap_tokens=0
    )

    assert len(chunks) == 3
    assert [c["start_seconds"] for c in chunks] == [0, 20, 40]


def test_split_into_chunks_never_splits_a_segment():
    segments = [_segment(i * 10, "x" * 100) for i in range(5)]

    chunks = split_into_chunks(segments, max_tokens=1, overlap_tokens=0)

    # every segment survives intact somewhere
    joined = "\n\n".join(c["text"] for c in chunks)
    for segment in segments:
        assert segment["raw"] in joined


def test_split_into_chunks_oversized_single_segment_gets_its_own_chunk():
    segments = [_segment(0, "x" * 5000), _segment(10, "y" * 10)]

    chunks = split_into_chunks(segments, max_tokens=100, overlap_tokens=0)

    assert segments[0]["raw"] in chunks[0]["text"]
    assert chunks[0]["text"].count("**ผู้พูด") == 1


def test_split_into_chunks_carries_overlap_into_the_next_chunk():
    segments = [_segment(i * 10, "x" * 100) for i in range(6)]
    per_segment = estimate_tokens(segments[0]["raw"])

    chunks = split_into_chunks(
        segments, max_tokens=per_segment * 3, overlap_tokens=per_segment
    )

    # last segment of chunk 0 reappears at the head of chunk 1
    assert segments[2]["raw"] in chunks[0]["text"]
    assert chunks[1]["text"].startswith(segments[2]["raw"])


def test_split_into_chunks_overlap_never_stalls_progress():
    segments = [_segment(i * 10, "x" * 100) for i in range(8)]
    per_segment = estimate_tokens(segments[0]["raw"])

    # overlap budget larger than a whole chunk must still terminate and cover all
    chunks = split_into_chunks(
        segments, max_tokens=per_segment * 2, overlap_tokens=per_segment * 99
    )

    assert chunks[-1]["end_seconds"] == 70
