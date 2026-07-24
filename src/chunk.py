import logging
import math
import re

logger = logging.getLogger(__name__)

# Measured on this project's real Thai transcripts: 6,728 characters produced
# 5,902 tokens (~1.14 chars/token). 1.1 is deliberately conservative so the
# estimate runs slightly high and chunks stay under budget. Kept local so this
# module never needs a tokenizer or a network call.
CHARS_PER_TOKEN = 1.1

# Matches a block produced by render_transcript_markdown:
#   **ผู้พูด 1** [00:00]: text
# Blocks that don't match and come *before* the first segment (the "# Transcript"
# heading, the diarization-failed note) are skipped, which is how they stay out
# of every chunk. Anything after the first segment is meeting content and is
# re-attached to the segment it belongs to -- see parse_transcript_segments.
SEGMENT_PATTERN = re.compile(r"\*\*(?P<speaker>[^*]+)\*\*\s*\[(?P<timestamp>\d+:\d{2})\]:")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def parse_timestamp(timestamp: str) -> int:
    minutes, seconds = timestamp.split(":")
    return int(minutes) * 60 + int(seconds)


def parse_transcript_segments(markdown: str) -> list[dict]:
    segments: list[dict] = []
    skipped_before_first_segment = 0
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        match = SEGMENT_PATTERN.match(block)
        if match:
            segments.append(
                {"raw": block, "start_seconds": parse_timestamp(match.group("timestamp"))}
            )
        elif segments:
            # An utterance containing a blank line arrives here as two blocks.
            # Appending to the segment it belongs to -- rather than dropping it --
            # makes losing meeting content structurally impossible.
            segments[-1]["raw"] += "\n\n" + block
        else:
            skipped_before_first_segment += 1
    if skipped_before_first_segment:
        logger.debug(
            "Skipped %d transcript block(s) before the first segment",
            skipped_before_first_segment,
        )
    return segments


def _build_chunk(segments: list[dict]) -> dict:
    # end_seconds is when the last utterance in the chunk *started* -- the
    # transcript carries no end times, and this is close enough to label a range.
    return {
        "text": "\n\n".join(segment["raw"] for segment in segments),
        "start_seconds": segments[0]["start_seconds"],
        "end_seconds": segments[-1]["start_seconds"],
    }


def _overlap_tail(segments: list[dict], overlap_tokens: int) -> list[dict]:
    # Trailing segments of a finished chunk, replayed at the head of the next one
    # so content straddling the boundary keeps its context. Skipping index 0
    # guarantees we never return the whole chunk, so the next chunk always makes
    # forward progress no matter how large the overlap budget is.
    tail: list[dict] = []
    total = 0
    for segment in reversed(segments[1:]):
        segment_tokens = estimate_tokens(segment["raw"])
        if total + segment_tokens > overlap_tokens:
            break
        tail.insert(0, segment)
        total += segment_tokens
    return tail


def split_into_chunks(
    segments: list[dict], max_tokens: int, overlap_tokens: int
) -> list[dict]:
    if not segments:
        return []

    if overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than max_tokens "
            f"({max_tokens}); otherwise each chunk retains nearly all of the "
            "previous one, causing an API-call explosion with no error."
        )

    chunks: list[dict] = []
    current: list[dict] = []
    current_tokens = 0

    for segment in segments:
        segment_tokens = estimate_tokens(segment["raw"])
        if current and current_tokens + segment_tokens > max_tokens:
            chunks.append(_build_chunk(current))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(s["raw"]) for s in current)
        current.append(segment)
        current_tokens += segment_tokens

    chunks.append(_build_chunk(current))
    return chunks
