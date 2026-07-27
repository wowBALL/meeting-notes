import logging
import re
from concurrent.futures import ThreadPoolExecutor

from src.chunk import estimate_tokens, parse_transcript_segments, split_into_chunks
from src.config import DEFAULT_SUMMARY_MODEL
from src.llm import MissingApiKeyError, Provider, UnusableAnswerError, resolve
from src.render import format_timestamp
from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)

SINGLE_CALL_THRESHOLD_TOKENS = 20_000
CHUNK_MAX_TOKENS = 15_000
CHUNK_OVERLAP_TOKENS = 1_500
# Chunks are independent, so the map stage is bound by its slowest call rather
# than by their sum. 4 keeps a 13-chunk (5-hour) meeting well inside the API's
# rate limits while cutting the map stage to roughly a quarter of its wall clock.
#
# The real worst-case bound this trades against: up to 2 provider calls per
# _summarize (the first call plus the doubled-budget retry when truncated) ×
# up to 3 _summarize attempts per retry_with_backoff = up to 6 provider calls
# per chunk (same bound applies to the single reduce call). A 13-chunk meeting
# is therefore up to 6 * 14 = 84 calls at worst. Against a fully stalled
# endpoint at LLM_TIMEOUT_SECONDS = 900 with only MAP_MAX_CONCURRENCY = 4
# chunks in flight at once, that worst case is hours, not minutes -- stated
# here plainly; no deadline/cutoff mechanism is added, that is a separate
# decision for whoever owns this trade-off to make.
MAP_MAX_CONCURRENCY = 4

# Substituted for a chunk whose summary failed every retry, so the surrounding
# chunks' summaries still reach the reader and the gap is visible in the timeline.
CHUNK_FAILURE_PLACEHOLDER = "> ⚠️ สรุปช่วงนี้ล้มเหลว (เรียกโมเดลไม่สำเร็จหลังลองใหม่ครบทุกครั้ง)"

# ใช้แทนบทสรุปรวมเมื่อ reduce ล้มเหลว สรุปรายช่วงด้านล่างเรียกสำเร็จและจ่ายเงินไปแล้ว
# การโยน exception ทิ้งทั้งหมดจึงแพงกว่าการส่งงานที่ยังไม่ได้ยุบรวมให้คนอ่าน
REDUCE_FAILURE_NOTICE = (
    "> ⚠️ รวมเป็นสรุปฉบับเดียวไม่สำเร็จ (เรียกโมเดลไม่ผ่านหลังลองใหม่ครบทุกครั้ง)\n"
    ">\n"
    "> ด้านล่างคือสรุปรายช่วงที่ทำสำเร็จแล้ว ยังไม่ได้ยุบรวมและยังไม่ได้แยก Action Items"
)

# ต่อท้ายสรุปที่ยังไม่จบหลังลองใหม่ด้วย budget สองเท่าแล้ว -- ต้องอยู่ในไฟล์ที่คนอ่าน
# ไม่ใช่แค่ใน log เพราะสรุปที่ขาดกลางประโยคยังอ่านเหมือนสรุปที่สมบูรณ์
TRUNCATION_NOTICE = (
    "\n\n> ⚠️ สรุปส่วนนี้ถูกตัดกลางทาง "
    "(โมเดลใช้โควตาคำตอบหมดทั้งสองครั้ง) เนื้อหาช่วงท้ายขาดไป"
)

# ^ requires whitespace so "#1 ..." in prose is not mangled
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+", re.MULTILINE)


def _demote_headings(markdown: str, floor: int = 4) -> str:
    """Push every heading down to at least `floor`, leaving deeper ones alone.
    A chunk summary is nested under a ### timeline entry, so any ##-level heading
    the model returns would outrank it; the merged summary sits above the ##
    timeline section, so an H1 there would swallow the timeline into its own."""
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


# ชุดเดียวกับที่ Anthropic SDK ถือว่าลองใหม่ได้: ต่อไม่ติด, 408, 409, 429 และ 5xx
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def is_retryable(error: Exception) -> bool:
    """เรียกซ้ำแล้วมีโอกาสได้คำตอบต่างจากเดิมไหม

    UnusableAnswerError กับ MissingApiKeyError คือคำตอบที่แน่นอนแล้ว -- คำขอเดิมยิงซ้ำ
    ก็ได้ผลเดิม การเรียกครั้งแรกที่ raise แบบนี้ไม่มีทางไปถึง path เพิ่ม budget เป็น
    สองเท่าใน _summarize ด้านล่างจริง (path นั้นทำงานเฉพาะตอนได้
    Completion(truncated=True) กลับมาเท่านั้น) แต่ path เพิ่ม budget เองก็เรียก
    provider.complete() อีกครั้งหนึ่ง และ raise ด้วยเหตุผลเดียวกันนี้ได้เหมือนกัน --
    _summarize ดักไว้เองแล้ว (try/except รอบ call ที่สอง) และคืนข้อความจาก call แรก
    พร้อม TRUNCATION_NOTICE แทนการโยนทิ้ง จึงไม่มีทางที่ error สองชนิดนี้จะหลุดออกมา
    ให้ is_retryable เห็นจาก call ที่สองเลย ฟังก์ชันนี้จึงยังแยกมันออกได้ถูกต้องโดย
    ไม่ต้องสนใจว่ามันมาจาก call ไหน

    4xx ที่เหลือเป็นคำตอบเรื่องคำขอหรือบัญชี -- key หมดอายุ เครดิตไม่พอ โมเดลผิด คำขอ
    ใหญ่เกิน ยิงซ้ำอีกกี่ครั้งก็ได้คำตอบเดิม เสียแค่เวลารอกับโควตา

    ไม่มี status_code แปลว่าไปไม่ถึง API (เน็ตหลุด หมดเวลา) ซึ่งเป็นอาการที่หายเองได้
    """
    if isinstance(error, (UnusableAnswerError, MissingApiKeyError)):
        return False
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return True
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def _summarize(provider: Provider, system: str, content: str, max_tokens: int) -> str:
    """ข้อความจากโมเดล ถูกตัดแล้วลองใหม่หนึ่งครั้งด้วย budget สองเท่า

    ลองใหม่ครั้งเดียวเพราะการเรียกซ้ำแพงและช้า (หนึ่ง call ใช้เวลาได้เป็นนาทีและมี
    ค่าใช้จ่ายจริง) และการเพิ่มเป็นสองเท่าครอบกรณีที่วัดเจอทั้งหมดแล้ว ถ้ายังไม่จบอีก
    ก็คืนของที่ได้พร้อมหมายเหตุ ดีกว่าทิ้งงานที่จ่ายเงินไปแล้วทั้งก้อน

    call ที่สอง (budget สองเท่า) ก็ raise ได้เหมือน call แรก -- ไม่ใช่แค่ในทางทฤษฎี:
    ขอ 24576 × 2 = 49152 token เกินเพดานต่อคำขอของบาง proxy (400), ส่ง reduce input
    ก้อนใหญ่ซ้ำ (413), หรือที่พบได้บ่อยที่สุดคือ reasoning model ใช้ budget ที่เพิ่มมา
    หมดไปกับ reasoning จน content ว่างเปล่า (UnusableAnswerError) -- ทั้งหมดนี้ไม่
    retryable (ดู is_retryable) จึงต้องดักไว้ที่นี่ ไม่ใช่ปล่อยให้ propagate ออกไป
    เพราะ completion.text จาก call แรกอยู่ในมือแล้วตอนนั้น การโยน exception ทิ้งทั้ง
    หมดจะเสียข้อความที่ได้มาแล้วไปเฉยๆ ทั้งที่ผลลัพธ์ควรแย่ที่สุดเท่ากับตอน retry
    สำเร็จแต่ยังไม่จบ (ข้อความบางส่วน + TRUNCATION_NOTICE) ไม่ใช่แย่กว่านั้น
    """
    completion = provider.complete(system, content, max_tokens)
    if not completion.truncated:
        return completion.text

    logger.warning(
        "%s truncated the answer at the %d-token cap (%d characters); "
        "retrying once at %d",
        provider.model_id,
        max_tokens,
        len(completion.text),
        max_tokens * 2,
    )
    try:
        retried = provider.complete(system, content, max_tokens * 2)
    except Exception as e:
        logger.error(
            "%s raised on the doubled-budget retry (%d tokens): %s -- keeping the "
            "first call's partial text instead of losing it",
            provider.model_id,
            max_tokens * 2,
            e,
        )
        return completion.text + TRUNCATION_NOTICE
    if not retried.truncated:
        return retried.text

    logger.error(
        "%s truncated the answer again at %d tokens; keeping the partial text "
        "and marking it in the summary",
        provider.model_id,
        max_tokens * 2,
    )
    return retried.text + TRUNCATION_NOTICE


def _time_range(chunk: dict) -> str:
    return f"{format_timestamp(chunk['start_seconds'])}–{format_timestamp(chunk['end_seconds'])}"


def _summarize_chunk(
    provider: Provider, chunk: dict, index: int, total: int
) -> str | Exception:
    """The chunk's summary, or the exception that ended it after every retry.
    Returning the failure instead of raising keeps one dead chunk from throwing
    away the summaries of every other chunk in the meeting."""
    try:
        return _demote_headings(
            retry_with_backoff(
                lambda: _summarize(
                    provider, CHUNK_SYSTEM_PROMPT, chunk["text"], provider.map_max_tokens
                ),
                should_retry=is_retryable,
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
    model: str = DEFAULT_SUMMARY_MODEL,
) -> str:
    provider = resolve(model)

    # Every API call below is retried here, inside summarize_transcript. Callers
    # must not add a retry of their own: with per-chunk retries in place, an outer
    # retry re-runs the entire map-reduce because of a single dead chunk.
    def single_call() -> str:
        # Reused deliberately: this short path is not a map call, but it shares
        # the map call's output budget so tuning one doesn't silently change the other.
        return retry_with_backoff(
            lambda: _summarize(
                provider,
                SUMMARY_SYSTEM_PROMPT,
                transcript_markdown,
                provider.map_max_tokens,
            ),
            should_retry=is_retryable,
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
                lambda item: _summarize_chunk(provider, item[1], item[0], len(chunks)),
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
    # the model to write about the outage instead of about the meeting. Same
    # reasoning for TRUNCATION_NOTICE: it's operational metadata about the call,
    # not something a meeting participant said, and the reduce prompt only asks
    # the model to merge/dedupe/group -- it would happily fold a stray blockquote
    # into the summary prose. The notice still reaches the reader via `timeline`
    # below, which reproduces each chunk's summary (untouched) verbatim.
    combined = "\n\n".join(
        f"## ช่วง [{_time_range(chunk)}]\n\n{summary.replace(TRUNCATION_NOTICE, '')}"
        for chunk, summary in succeeded
    )
    # floor=2: the model likes to open with an H1 title, which would rank above
    # "## ไทม์ไลน์ตามช่วง" and nest the whole timeline inside the merged summary.
    # Its own ## and ### structure is left untouched.
    try:
        overall = _demote_headings(
            retry_with_backoff(
                lambda: _summarize(
                    provider,
                    REDUCE_SYSTEM_PROMPT,
                    combined,
                    provider.reduce_max_tokens,
                ),
                should_retry=is_retryable,
            ),
            floor=2,
        )
    except Exception as e:
        # Every chunk summary below this line already succeeded and was already
        # billed. Letting the reduce failure propagate would throw all of them
        # away and send the recording to failed/, so the map stage would have to
        # be paid for a second time. The timeline alone is worth reading.
        logger.error(
            "Reduce stage failed after every retry, returning the %d chunk "
            "summaries without a merged summary: %s",
            len(succeeded),
            e,
        )
        overall = REDUCE_FAILURE_NOTICE

    timeline = "\n\n".join(
        f"### [{_time_range(chunk)}]\n\n"
        f"{CHUNK_FAILURE_PLACEHOLDER if isinstance(summary, Exception) else summary}"
        for chunk, summary in zip(chunks, chunk_summaries)
    )
    return f"{overall}\n\n---\n\n## ไทม์ไลน์ตามช่วง\n\n{timeline}"
