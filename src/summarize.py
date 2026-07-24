import logging
import re
from concurrent.futures import ThreadPoolExecutor

from src.chunk import estimate_tokens, parse_transcript_segments, split_into_chunks
from src.render import format_timestamp
from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)

SINGLE_CALL_THRESHOLD_TOKENS = 20_000
CHUNK_MAX_TOKENS = 15_000
CHUNK_OVERLAP_TOKENS = 1_500
MAP_MAX_OUTPUT_TOKENS = 4096
REDUCE_MAX_OUTPUT_TOKENS = 8192
# Chunks are independent, so the map stage is bound by its slowest call rather
# than by their sum. 4 keeps a 13-chunk (5-hour) meeting well inside the API's
# rate limits while cutting the map stage to roughly a quarter of its wall clock.
MAP_MAX_CONCURRENCY = 4

# Substituted for a chunk whose summary failed every retry, so the surrounding
# chunks' summaries still reach the reader and the gap is visible in the timeline.
CHUNK_FAILURE_PLACEHOLDER = "> ⚠️ สรุปช่วงนี้ล้มเหลว (เรียก Claude ไม่สำเร็จหลังลองใหม่ครบทุกครั้ง)"

# ^ requires whitespace so "#1 ..." in prose is not mangled
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+", re.MULTILINE)


def _demote_headings(markdown: str, floor: int = 4) -> str:
    """A chunk summary is nested under a ### timeline entry; any ##-level heading
    the model returns would outrank it and break the document outline."""
    return _HEADING_RE.sub(lambda m: "#" * max(floor, len(m.group(1))) + " ", markdown)


SUMMARY_SYSTEM_PROMPT = """คุณเป็นผู้ช่วยสรุปการประชุม อ่าน transcript ที่ให้มาแล้วสรุปเป็นภาษาไทยในรูปแบบ Markdown ประกอบด้วย:

## ประเด็นสำคัญ
(สรุปหัวข้อและประเด็นหลักที่พูดคุยกัน เป็น bullet point)

## Action Items
(รายการสิ่งที่ต้องทำ พร้อมระบุผู้รับผิดชอบถ้าอ้างอิงได้จากบทสนทนา ถ้าไม่ระบุชัดเจนให้เขียนว่า "ไม่ระบุผู้รับผิดชอบ")

ถ้าจากบริบทการสนทนาพอเดาชื่อจริงของผู้พูดแต่ละคนได้ (เช่นมีการเอ่ยชื่อกัน) ให้ใช้ชื่อจริงแทน label "ผู้พูด N" ในสรุป ถ้าเดาไม่ได้ให้คงป้าย "ผู้พูด N" ไว้"""

CHUNK_SYSTEM_PROMPT = """คุณเป็นผู้ช่วยสรุปการประชุม ข้อความที่ให้มาคือ transcript "เพียงบางช่วง" ของการประชุมที่ยาวกว่านี้ ไม่ใช่ทั้งการประชุม

สรุปเฉพาะเนื้อหาในช่วงนี้เป็นภาษาไทยแบบ Markdown เป็น bullet point เก็บรายละเอียดให้ครบ ทั้งประเด็นที่คุยกัน ข้อสรุป และสิ่งที่ต้องทำพร้อมผู้รับผิดชอบถ้าระบุได้

ห้ามเดาเนื้อหาช่วงอื่นที่ไม่ได้ให้มา และไม่ต้องเขียนคำนำหรือคำลงท้าย ใช้ bullet point เท่านั้น ห้ามใส่ markdown heading (เช่น ## หรือ ###)"""

REDUCE_SYSTEM_PROMPT = """ข้อความที่ให้มาคือสรุปย่อยของการประชุมเดียวกัน เรียงตามช่วงเวลา

รวมทั้งหมดเป็นสรุปฉบับเดียวเป็นภาษาไทยแบบ Markdown ประกอบด้วย:

## ประเด็นสำคัญ
(รวมประเด็นจากทุกช่วง จัดกลุ่มตามหัวข้อไม่ใช่ตามเวลา ยุบเรื่องที่ซ้ำกันเข้าด้วยกัน)

## Action Items
(รวมสิ่งที่ต้องทำจากทุกช่วง พร้อมผู้รับผิดชอบถ้าอ้างอิงได้ ถ้าไม่ระบุชัดเจนให้เขียนว่า "ไม่ระบุผู้รับผิดชอบ")

ถ้าพอเดาชื่อจริงของผู้พูดได้จากบริบท ให้ใช้ชื่อจริงแทน label "ผู้พูด N" เก็บเนื้อหาสำคัญให้ครบ อย่าตัดทิ้งเพียงเพราะอยากให้สั้น"""


def _summarize(client, model: str, system: str, content: str, max_tokens: int) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    stop_reason = getattr(response, "stop_reason", None)
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(
            f"Claude returned no text block (stop_reason={stop_reason!r}); "
            "nothing to use as a summary"
        )
    if stop_reason == "max_tokens":
        # The summary stops mid-sentence but still reads as if it were complete,
        # so the only way anyone learns about it is this log line.
        logger.warning(
            "Summary hit the max_tokens cap (%d) and is truncated; "
            "stop_reason=max_tokens, %d characters returned",
            max_tokens,
            len(text),
        )
    return text


def _time_range(chunk: dict) -> str:
    return f"{format_timestamp(chunk['start_seconds'])}–{format_timestamp(chunk['end_seconds'])}"


def _summarize_chunk(
    client, model: str, chunk: dict, index: int, total: int
) -> str | Exception:
    """The chunk's summary, or the exception that ended it after every retry.
    Returning the failure instead of raising keeps one dead chunk from throwing
    away the summaries of every other chunk in the meeting."""
    try:
        return _demote_headings(
            retry_with_backoff(
                lambda: _summarize(
                    client, model, CHUNK_SYSTEM_PROMPT, chunk["text"], MAP_MAX_OUTPUT_TOKENS
                )
            )
        )
    except Exception as e:
        logger.error(
            "Chunk %d/%d [%s] failed after every retry, using a placeholder: %s",
            index + 1,
            total,
            _time_range(chunk),
            e,
        )
        return e


def summarize_transcript(
    transcript_markdown: str,
    model: str = "claude-opus-4-8",
    api_key: str | None = None,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    # Every API call below is retried here, inside summarize_transcript. Callers
    # must not add a retry of their own: with per-chunk retries in place, an outer
    # retry re-runs the entire map-reduce because of a single dead chunk.
    def single_call() -> str:
        # Reused deliberately: this short path is not a map call, but it shares
        # the map call's output budget so tuning one doesn't silently change the other.
        return retry_with_backoff(
            lambda: _summarize(
                client,
                model,
                SUMMARY_SYSTEM_PROMPT,
                transcript_markdown,
                MAP_MAX_OUTPUT_TOKENS,
            )
        )

    if estimate_tokens(transcript_markdown) <= SINGLE_CALL_THRESHOLD_TOKENS:
        return single_call()

    segments = parse_transcript_segments(transcript_markdown)
    chunks = split_into_chunks(segments, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)

    if not chunks:
        return single_call()

    # Chunks are independent, so summarize them concurrently. executor.map yields
    # results in submission order, so the timeline stays in transcript order no
    # matter which chunk finishes first.
    with ThreadPoolExecutor(max_workers=min(MAP_MAX_CONCURRENCY, len(chunks))) as pool:
        chunk_summaries = list(
            pool.map(
                lambda item: _summarize_chunk(client, model, item[1], item[0], len(chunks)),
                enumerate(chunks),
            )
        )

    succeeded = [
        (chunk, summary)
        for chunk, summary in zip(chunks, chunk_summaries)
        if not isinstance(summary, Exception)
    ]
    if not succeeded:
        first_error = chunk_summaries[0]
        raise RuntimeError(
            f"Every one of the {len(chunks)} chunk summaries failed: {first_error}"
        ) from first_error

    # The reduce stage sees real summaries only -- a placeholder here would invite
    # the model to write about the outage instead of about the meeting.
    combined = "\n\n".join(
        f"## ช่วง [{_time_range(chunk)}]\n\n{summary}" for chunk, summary in succeeded
    )
    overall = retry_with_backoff(
        lambda: _summarize(
            client, model, REDUCE_SYSTEM_PROMPT, combined, REDUCE_MAX_OUTPUT_TOKENS
        )
    )

    timeline = "\n\n".join(
        f"### [{_time_range(chunk)}]\n\n"
        f"{CHUNK_FAILURE_PLACEHOLDER if isinstance(summary, Exception) else summary}"
        for chunk, summary in zip(chunks, chunk_summaries)
    )
    return f"{overall}\n\n---\n\n## ไทม์ไลน์ตามช่วง\n\n{timeline}"
