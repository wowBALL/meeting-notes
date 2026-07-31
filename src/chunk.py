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


# เพดานความยาวของบล็อกที่รวมแล้ว วัดจริงบน transcript 84 นาที (1,618 บล็อก):
# เพดาน 600 ลดขนาดได้ 43.4% ส่วนแบบไม่จำกัดเลยได้ 44.2% -- ต่างกันไม่ถึงหนึ่งจุด
# เพราะบทสนทนาส่วนใหญ่สลับผู้พูดเร็วอยู่แล้ว มีแค่ช่วงที่คนหนึ่งพูดยาวคนเดียวที่โดนตัด
#
# แต่แบบไม่จำกัดผลิตบล็อกขนาด 3,797 token ซึ่งใหญ่เกินงบ overlap (1,500) จน
# _overlap_tail เล่นซ้ำมันไม่ได้เลยแม้แต่ก้อนเดียว -- รอยต่อ chunk จะขาดบริบทโดยไม่มี
# อะไรฟ้อง และก้อนที่โตกว่านี้ก็มีทางชนเพดาน max_tokens ที่ split_into_chunks
# ปล่อยผ่าน (ดู `if current and` ด้านล่าง) เพดานนี้จึงกันไว้ที่ต้นทาง
MERGE_MAX_TOKENS = 600


def merge_speaker_turns(markdown: str, max_tokens: int = MERGE_MAX_TOKENS) -> str:
    """รวมบล็อกของผู้พูดคนเดียวกันที่อยู่ติดกัน ให้เหลือบล็อกเดียว

    "ติดกัน" คือไม่มีผู้พูดคนอื่นคั่นเท่านั้น A A B A จึงได้ AA / B / A ไม่ใช่ AAA / B
    หัวบล็อกที่เก็บไว้คือของประโยคแรก ส่วน timestamp ของประโยคที่ถูกรวมเข้ามาถูกทิ้ง --
    prompt ทั้งสามไม่ได้สั่งให้โมเดลอ้าง timestamp ในคำตอบ และช่วงเวลาที่โผล่ในสรุปมาจาก
    start_seconds ของบล็อกหัว/ท้าย chunk ซึ่งยังอยู่ครบ เก็บ timestamp ข้างในไว้ด้วยวัดแล้ว
    ว่าลดขนาดได้แค่ 30.6% แทนที่จะเป็น 43.4%

    *** ราคาที่ต้องรู้: การรวมบล็อกทำให้ diarization ที่ผิดกลืนหายไป ***
    ถ้าระบบแยกผู้พูดพลาดแล้วป้ายคนสองคนเป็นคนเดียวกัน ตอนยังไม่รวมยังพอเห็นเป็นสองบล็อก
    ที่น้ำเสียงไม่เข้ากัน (เจอจริงที่ [35:56] ของประชุม 07-31: "ขอบคุณค่ะ" กับ "ผมรู้สึกว่า"
    อยู่ติดกันใต้ป้ายเดียวกัน) พอรวมแล้วมันกลายเป็นประโยคเดียวที่อ่านเหมือนคนคนเดียวพูดยาว
    เพดาน max_tokens ช่วยจำกัดความเสียหายไว้ ไม่ได้แก้

    คืนค่าเดิมทั้งก้อนถ้าไม่มีอะไรถูกรวมเลย -- ปิดสวิตช์แล้วข้อความที่ส่งให้โมเดลต้องเท่า
    ของเดิมเป๊ะ และ transcript ที่ไม่มีบล็อกผู้พูดติดกันเลยก็ไม่ควรถูกจัดรูปใหม่โดยเปล่าประโยชน์
    """
    blocks: list[str] = []
    open_speaker: str | None = None
    segments_in = 0
    segments_out = 0

    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        match = SEGMENT_PATTERN.match(block)
        if not match:
            if segments_out:
                # ประโยคที่มีบรรทัดว่างในตัวเอง -- ต่อกลับเข้าบล็อกที่มันเป็นของ แบบเดียว
                # กับ parse_transcript_segments แล้วตัดสายการรวม: ประโยคถัดไปของคนเดิม
                # ต้องขึ้นบล็อกใหม่ ไม่ใช่ไปต่อท้ายย่อหน้าที่คนละที่กับหัวบล็อก
                blocks[-1] += "\n\n" + block
                open_speaker = None
            else:
                # "# Transcript" กับหมายเหตุตอน diarization ล้ม -- path ยิงรอบเดียวส่งทั้ง
                # ไฟล์รวมหัวเรื่อง (ต่างจาก chunker ที่ตัดทิ้ง) ที่นี่จึงต้องเก็บไว้
                blocks.append(block)
            continue

        segments_in += 1
        speaker = match.group("speaker")
        body = block[match.end() :].strip()
        if (
            speaker == open_speaker
            and estimate_tokens(blocks[-1]) + estimate_tokens(body) + 1 <= max_tokens
        ):
            blocks[-1] = f"{blocks[-1]} {body}".rstrip()
        else:
            blocks.append(block)
            segments_out += 1
            open_speaker = speaker

    if segments_out == segments_in:
        return markdown

    merged = "\n\n".join(blocks)
    logger.info(
        "Merged %d speaker blocks into %d (%.0f%% fewer characters)",
        segments_in,
        segments_out,
        100 * (1 - len(merged) / len(markdown)) if markdown else 0,
    )
    return merged


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
