import math

import pytest

from src.chunk import (
    CHARS_PER_TOKEN,
    MERGE_MAX_TOKENS,
    SEGMENT_PATTERN,
    estimate_tokens,
    merge_speaker_turns,
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


def test_parse_transcript_segments_keeps_a_blank_line_inside_a_segment():
    markdown = (
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:00]: บรรทัดแรก\n\nบรรทัดที่สองของคนเดิม\n\n"
        "**ผู้พูด 2** [00:20]: ต่อไป"
    )

    segments = parse_transcript_segments(markdown)

    assert len(segments) == 2
    assert "บรรทัดที่สองของคนเดิม" in segments[0]["raw"]
    assert segments[0]["start_seconds"] == 0
    assert segments[1]["start_seconds"] == 20


def test_parse_transcript_segments_loses_nothing_after_the_first_segment():
    markdown = (
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:00]: หนึ่ง\n\nสอง\n\nสาม\n\n"
        "**ผู้พูด 2** [00:20]: สี่"
    )

    joined = "\n\n".join(s["raw"] for s in parse_transcript_segments(markdown))

    for word in ("หนึ่ง", "สอง", "สาม", "สี่"):
        assert word in joined


def test_merge_speaker_turns_joins_adjacent_blocks_of_one_speaker():
    markdown = (
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:11]: ครับ มีแค่ของ Payment\n\n"
        "**ผู้พูด 1** [00:14]: ที่เป็น Result"
    )

    merged = merge_speaker_turns(markdown)

    assert merged == (
        "# Transcript\n\n**ผู้พูด 1** [00:11]: ครับ มีแค่ของ Payment ที่เป็น Result"
    )


def test_merge_speaker_turns_stops_at_another_speaker():
    # A A B A -- the trailing A starts a new block, it does not rejoin the first
    markdown = (
        "**ผู้พูด 1** [00:00]: หนึ่ง\n\n"
        "**ผู้พูด 1** [00:05]: สอง\n\n"
        "**ผู้พูด 2** [00:10]: สาม\n\n"
        "**ผู้พูด 1** [00:15]: สี่"
    )

    blocks = merge_speaker_turns(markdown).split("\n\n")

    assert blocks == [
        "**ผู้พูด 1** [00:00]: หนึ่ง สอง",
        "**ผู้พูด 2** [00:10]: สาม",
        "**ผู้พูด 1** [00:15]: สี่",
    ]


def test_merge_speaker_turns_splits_when_the_cap_is_reached():
    markdown = "\n\n".join(
        f"**ผู้พูด 1** [{i:02d}:00]: " + "ก" * 100 for i in range(10)
    )

    blocks = merge_speaker_turns(markdown, max_tokens=200).split("\n\n")

    assert len(blocks) > 1
    for block in blocks:
        # every block keeps a header of its own, carrying the real timestamp of
        # the utterance that starts it -- a split block is still a valid segment
        assert SEGMENT_PATTERN.match(block)
        assert estimate_tokens(block) <= 200
    assert blocks[0].startswith("**ผู้พูด 1** [00:00]:")
    assert blocks[1].startswith("**ผู้พูด 1** [0")
    assert not blocks[1].startswith("**ผู้พูด 1** [00:00]:")


def test_merge_speaker_turns_keeps_the_heading_and_diarization_note():
    markdown = (
        "# Transcript\n\n"
        '> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ ข้อความทั้งหมดจึงแสดงเป็น "ผู้พูด 1" เพียงคนเดียว\n\n'
        "**ผู้พูด 1** [00:00]: หนึ่ง\n\n"
        "**ผู้พูด 1** [00:05]: สอง"
    )

    merged = merge_speaker_turns(markdown)

    assert merged.startswith("# Transcript\n\n> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ")
    assert "หนึ่ง สอง" in merged


def test_merge_speaker_turns_keeps_a_blank_line_inside_an_utterance():
    markdown = (
        "**ผู้พูด 1** [00:00]: บรรทัดแรก\n\nบรรทัดที่สองของคนเดิม\n\n"
        "**ผู้พูด 1** [00:20]: ประโยคถัดไป\n\n"
        "**ผู้พูด 1** [00:25]: และอีกประโยค"
    )

    merged = merge_speaker_turns(markdown)

    # the stray paragraph stays attached to the block it belongs to and in order
    assert merged.index("บรรทัดแรก") < merged.index("บรรทัดที่สองของคนเดิม")
    assert merged.index("บรรทัดที่สองของคนเดิม") < merged.index("ประโยคถัดไป")
    # it also breaks the run: the next utterance starts its own block...
    assert "\n\n**ผู้พูด 1** [00:20]: ประโยคถัดไป และอีกประโยค" in merged


def test_merge_speaker_turns_merges_real_names_too():
    markdown = (
        "**สมหญิง** [00:00]: หนึ่ง\n\n**สมหญิง** [00:05]: สอง"
    )

    assert merge_speaker_turns(markdown) == "**สมหญิง** [00:00]: หนึ่ง สอง"


def test_merge_speaker_turns_returns_the_input_untouched_when_nothing_merges():
    assert merge_speaker_turns(TRANSCRIPT) == TRANSCRIPT
    assert merge_speaker_turns("") == ""
    assert merge_speaker_turns("# Transcript\n\nของเดิม") == "# Transcript\n\nของเดิม"


def test_merge_speaker_turns_output_still_parses_as_segments():
    markdown = (
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:00]: หนึ่ง\n\n"
        "**ผู้พูด 1** [00:05]: สอง\n\n"
        "**ผู้พูด 2** [01:30]: สาม"
    )

    segments = parse_transcript_segments(merge_speaker_turns(markdown))

    assert [s["start_seconds"] for s in segments] == [0, 90]
    assert segments[0]["raw"] == "**ผู้พูด 1** [00:00]: หนึ่ง สอง"


def test_merge_cap_leaves_room_for_the_overlap_replay():
    # A merged block larger than the overlap budget can never be replayed by
    # _overlap_tail, so the boundary between two chunks would silently lose its
    # context. Two blocks must fit, hence the factor of two.
    from src.config import DEFAULT_CHUNK_OVERLAP_TOKENS

    assert MERGE_MAX_TOKENS * 2 <= DEFAULT_CHUNK_OVERLAP_TOKENS


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

    # overlap budget just under max_tokens (as large as the guard in
    # split_into_chunks allows) must still terminate and cover all segments
    chunks = split_into_chunks(
        segments, max_tokens=per_segment * 2, overlap_tokens=per_segment * 2 - 1
    )

    assert chunks[-1]["end_seconds"] == 70


def test_split_into_chunks_raises_when_overlap_reaches_max_tokens():
    segments = [_segment(i * 10, "x" * 100) for i in range(3)]
    per_segment = estimate_tokens(segments[0]["raw"])

    with pytest.raises(ValueError):
        split_into_chunks(segments, max_tokens=per_segment * 2, overlap_tokens=per_segment * 2)


def test_split_into_chunks_exceeds_budget_only_by_joiner_overhead():
    segments = [_segment(i * 10, "x" * 100) for i in range(200)]
    max_tokens = 2000

    chunks = split_into_chunks(segments, max_tokens=max_tokens, overlap_tokens=0)

    for chunk in chunks:
        segment_count = chunk["text"].count("\n\n") + 1
        joiner_allowance = math.ceil(2 * (segment_count - 1) / CHARS_PER_TOKEN)
        assert estimate_tokens(chunk["text"]) <= max_tokens + joiner_allowance
