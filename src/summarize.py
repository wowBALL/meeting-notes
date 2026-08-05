import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from src.chunk import (
    MERGE_MAX_TOKENS,
    estimate_tokens,
    merge_speaker_turns,
    parse_transcript_segments,
    split_into_chunks,
)
from src.config import (
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_SUMMARY_MODEL,
)
from src.llm import (
    LLM_TIMEOUT_SECONDS,
    SINGLE_CALL_THRESHOLD_TOKENS,
    MissingSettingError,
    Provider,
    UnusableAnswerError,
    check_reachable,
    resolve,
)
from src.prompts import DEFAULT_PROFILE, FALLBACKS, render
from src.render import format_timestamp
from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# SINGLE_CALL_THRESHOLD_TOKENS ย้ายไปอยู่ src/llm.py แล้ว (เป็นคุณสมบัติของ provider ที่
# ตอนนี้แต่ละตัว override ได้ ดู Provider.single_call_threshold_tokens) import กลับมาไว้
# ในชื่อเดิมที่นี่เพราะเทสต์เก่าอ้างถึง src.summarize.SINGLE_CALL_THRESHOLD_TOKENS อยู่ --
# ตัวที่ใช้จริงตอนตัดสินใจ chunk คือ provider.single_call_threshold_tokens ด้านล่าง
# ไม่ใช่ค่าคงที่ตัวนี้ตรงๆ (ค่านี้เป็นแค่ค่า default ของ provider ส่วนใหญ่)
#
# ค่าจริงอยู่ใน config.py (ที่นั่นตรวจ .env ให้ด้วย) ชื่อสองตัวนี้คงไว้เพราะเป็นค่า
# เริ่มต้นที่ใช้เมื่อผู้เรียกไม่ได้ส่ง overlap มา และมีเทสต์อ้างถึงอยู่
CHUNK_MAX_TOKENS = DEFAULT_CHUNK_MAX_TOKENS
CHUNK_OVERLAP_TOKENS = DEFAULT_CHUNK_OVERLAP_TOKENS
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


# prompt จริงอยู่ในไฟล์ prompts/*.md แล้ว -- ชื่อสามตัวนี้คงไว้เพราะเป็น prompt สำรอง
# ที่ใช้เมื่อไฟล์หาย และมีโค้ด/เทสต์อ้างถึงอยู่ จูนถ้อยคำให้ไปแก้ไฟล์ ไม่ใช่ที่นี่
SUMMARY_SYSTEM_PROMPT = FALLBACKS["single"]
CHUNK_SYSTEM_PROMPT = FALLBACKS["map"]
REDUCE_SYSTEM_PROMPT = FALLBACKS["reduce"]


# ชุดเดียวกับที่ Anthropic SDK ถือว่าลองใหม่ได้: ต่อไม่ติด, 408, 409, 429 และ 5xx
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def is_retryable(error: Exception) -> bool:
    """เรียกซ้ำแล้วมีโอกาสได้คำตอบต่างจากเดิมไหม

    UnusableAnswerError กับ MissingSettingError คือคำตอบที่แน่นอนแล้ว -- คำขอเดิมยิงซ้ำ
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
    if isinstance(error, (UnusableAnswerError, MissingSettingError)):
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


def check_model_reachable(model: str) -> None:
    """ยิงคำขอเล็ก ๆ ไปที่โมเดลที่จะใช้สรุป raise ถ้าไปไม่ถึง

    ผู้เรียก (pipeline) คุยกับโมดูลนี้อยู่แล้ว การให้มันต้อง import resolve จาก llm
    เองเพิ่มอีกทางแปลว่ามีสองทางที่ pipeline รู้จักชั้น provider

    หมายเหตุที่ต้องเคารพ: src/preflight.py:17-20 บันทึกไว้ว่าการยิง API จริงเคยมีแล้ว
    **ถูกถอดออกโดยตั้งใจ** สิ่งที่ต่างออกไปรอบนี้คือตำแหน่งกับวัตถุประสงค์ ของเดิมยิงตอน
    ก่อนเริ่มอัดเสียงเพื่อตรวจว่าตั้งค่าถูกไหม ซึ่งเป็นคำถามที่ตอบได้โดยไม่ต้องยิงเน็ต
    ส่วนอันนี้ยิงตอนก่อนเริ่ม map เพื่อกันงานที่ไม่มีคนเฝ้าซึ่งกินเวลาได้เป็นชั่วโมง
    (5 chunks x 3 retries x 900 วินาที) ไม่ให้เริ่มทั้งที่ปลายทางไปไม่ถึงตั้งแต่แรก --
    2026-07-31 เสียไปหนึ่งชั่วโมงเต็มกับกรณีแบบนี้พอดี
    """
    started = time.monotonic()
    check_reachable(resolve(model))
    # บันทึกตอนผ่านด้วย ไม่ใช่แค่ตอนล้ม: คนที่ไล่ log ย้อนหลังต้องแยก "เช็คแล้วผ่าน"
    # ออกจาก "ไม่เคยเช็ค" ได้ ไม่งั้นความเงียบตีความได้สองแบบเหมือนเดิม
    logger.info(
        "%s answered a probe request in %.1fs, starting the real work",
        model,
        time.monotonic() - started,
    )


def _time_range(chunk: dict) -> str:
    return f"{format_timestamp(chunk['start_seconds'])}–{format_timestamp(chunk['end_seconds'])}"


def _summarize_chunk(
    provider: Provider, system: str, chunk: dict, index: int, total: int
) -> str | Exception:
    """The chunk's summary, or the exception that ended it after every retry.
    Returning the failure instead of raising keeps one dead chunk from throwing
    away the summaries of every other chunk in the meeting.

    The start/finish lines exist because this stage used to log nothing at all
    until a chunk died. A stalled endpoint produced up to 45 minutes of complete
    silence per chunk, which is indistinguishable from a hung process -- on
    2026-07-31 that cost an hour before anyone could tell something was wrong."""
    label = f"Chunk {index + 1}/{total} [{_time_range(chunk)}]"
    logger.info("%s: starting", label)
    started = time.monotonic()
    try:
        summary = _demote_headings(
            retry_with_backoff(
                lambda: _summarize(
                    provider, system, chunk["text"], provider.map_max_tokens
                ),
                should_retry=is_retryable,
                label=label,
            )
        )
    except Exception as e:
        logger.error(
            "%s: failed after every retry in %.1fs, using a placeholder: %s",
            label,
            time.monotonic() - started,
            e,
        )
        return e
    logger.info(
        "%s: done in %.1fs (%d chars)", label, time.monotonic() - started, len(summary)
    )
    return summary


def summarize_transcript(
    transcript_markdown: str,
    model: str = DEFAULT_SUMMARY_MODEL,
    glossary_text: str = "",
    profile: str = DEFAULT_PROFILE,
    carryover_text: str = "",
    chunk_overlap_tokens: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    merge_turns: bool = True,
) -> str:
    """`glossary_text` มาจาก glossary.format_for_prompt() -- ว่างได้ แปลว่าไม่มีตาราง
    `profile` เลือกไฟล์ prompts/profiles/<profile>.md ที่จะแทรกเข้า {profile_rules}

    ตัวแทนที่คำแบบเป๊ะ (apply_exact) ไม่ได้อยู่ในนี้โดยเจตนา มันทำที่ pipeline ก่อน
    เรียกฟังก์ชันนี้ เพราะฟังก์ชันนี้คืน str เปล่า ๆ และถูก mock ไว้หลายสิบจุดในเทสต์
    การให้มันคืนจำนวนที่แก้ด้วยจะเปลี่ยน return type ไปทั้งหมดโดยไม่ได้อะไรเพิ่ม

    prompt ทั้งสามถูก render ที่นี่ครั้งเดียว ไม่ใช่ต่อ chunk: การอ่านไฟล์ซ้ำทุก chunk
    เปิดช่องให้แก้ไฟล์กลางประชุมแล้วได้สรุปที่ครึ่งหนึ่งใช้กฎเก่าครึ่งหนึ่งใช้กฎใหม่
    """
    provider = resolve(model)
    overlap = (
        CHUNK_OVERLAP_TOKENS if chunk_overlap_tokens is None else chunk_overlap_tokens
    )
    if merge_turns:
        # ต้องรวมก่อนเช็ค provider.single_call_threshold_tokens ไม่ใช่หลัง: ประชุมที่เดิม
        # ต้องหั่นเป็นสอง chunk อาจเหลือต่ำกว่าเพดานแล้วยิงรอบเดียวจบ -- จากสองคำขอเหลือหนึ่ง
        # ซึ่งคือทั้งหมดที่ฟีเจอร์นี้มีไว้ทำ (ลดจำนวนงานที่เราใส่เข้าคิวของ endpoint)
        #
        # เพดานผูกกับงบ overlap เสมอ ไม่ใช่ค่าลอย ๆ: บล็อกที่ใหญ่กว่างบ overlap ถูก
        # _overlap_tail เล่นซ้ำไม่ได้เลย รอยต่อ chunk จะขาดบริบทแบบเงียบ ๆ หารสองเพื่อ
        # ให้ยังเล่นซ้ำได้อย่างน้อยสองบล็อก (ค่า default 1500//2=750 > 600 ⇒ ปกติเพดาน
        # คือ 600 ตามที่วัดมา วาล์วนี้ทำงานเฉพาะตอนมีคนไปลด CHUNK_OVERLAP_TOKENS)
        transcript_markdown = merge_speaker_turns(
            transcript_markdown, max_tokens=min(MERGE_MAX_TOKENS, max(1, overlap // 2))
        )
    # carryover ไม่เข้าขั้น map: map สรุปทีละช่วงของประชุมนี้ เรื่องค้างของประชุมก่อน
    # ไม่เกี่ยวกับมัน และ map ถูกเรียกต่อ chunk -- ยัดเข้าไปคือจ่ายค่า token ซ้ำทุก chunk
    single_system = render(
        "single", profile=profile, glossary_text=glossary_text, carryover_text=carryover_text
    )
    chunk_system = render("map", profile=profile, glossary_text=glossary_text)
    reduce_system = render(
        "reduce", profile=profile, glossary_text=glossary_text, carryover_text=carryover_text
    )

    # Every API call below is retried here, inside summarize_transcript. Callers
    # must not add a retry of their own: with per-chunk retries in place, an outer
    # retry re-runs the entire map-reduce because of a single dead chunk.
    def single_call() -> str:
        # Reused deliberately: this short path is not a map call, but it shares
        # the map call's output budget so tuning one doesn't silently change the other.
        return retry_with_backoff(
            lambda: _summarize(
                provider,
                single_system,
                transcript_markdown,
                provider.map_max_tokens,
            ),
            should_retry=is_retryable,
            label="Single-call summary",
        )

    if estimate_tokens(transcript_markdown) <= provider.single_call_threshold_tokens:
        logger.info(
            "Summarizing with %s in a single call (~%d tokens)",
            provider.model_id,
            estimate_tokens(transcript_markdown),
        )
        return single_call()

    segments = parse_transcript_segments(transcript_markdown)
    chunks = split_into_chunks(segments, CHUNK_MAX_TOKENS, overlap)

    if not chunks:
        return single_call()

    # Chunks are independent, so summarize them concurrently. executor.map yields
    # results in submission order, so the timeline stays in transcript order no
    # matter which chunk finishes first.
    workers = min(MAP_MAX_CONCURRENCY, len(chunks))
    # Whoever reads this log during an incident needs to know how much work is in
    # flight before the per-chunk lines start arriving -- without it, "Chunk 5/6"
    # appearing first out of the concurrent pool reads like chunks went missing.
    logger.info(
        "Summarizing with %s: %d chunks, %d at a time (timeout %ds per call)",
        provider.model_id,
        len(chunks),
        workers,
        LLM_TIMEOUT_SECONDS,
    )
    map_started = time.monotonic()

    # Reported on completion rather than on start, and counted rather than indexed,
    # because chunks finish out of order: "3/6 done" is true regardless of which
    # three they were, while "chunk 5 started" tells a watching user nothing about
    # how far along the meeting is.
    progress_lock = threading.Lock()
    completed = 0

    def run_chunk(item: tuple[int, dict]) -> str | Exception:
        nonlocal completed
        index, chunk = item
        result = _summarize_chunk(provider, chunk_system, chunk, index, len(chunks))
        if on_progress is not None:
            with progress_lock:
                completed += 1
                done = completed
            try:
                on_progress(done, len(chunks))
            except Exception:
                # A progress report that fails must never cost a chunk summary that
                # already succeeded and was already paid for -- same reasoning as
                # activity.append() swallowing OSError.
                logger.exception("Progress callback failed, continuing anyway")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunk_summaries = list(pool.map(run_chunk, enumerate(chunks)))

    succeeded = [
        (chunk, summary)
        for chunk, summary in zip(chunks, chunk_summaries)
        if not isinstance(summary, Exception)
    ]
    logger.info(
        "Map stage finished in %.1fs: %d/%d chunks succeeded",
        time.monotonic() - map_started,
        len(succeeded),
        len(chunks),
    )
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
    logger.info(
        "Reduce stage: merging %d chunk summaries (~%d tokens)",
        len(succeeded),
        estimate_tokens(combined),
    )
    reduce_started = time.monotonic()
    # floor=2: the model likes to open with an H1 title, which would rank above
    # "## ไทม์ไลน์ตามช่วง" and nest the whole timeline inside the merged summary.
    # Its own ## and ### structure is left untouched.
    try:
        overall = _demote_headings(
            retry_with_backoff(
                lambda: _summarize(
                    provider,
                    reduce_system,
                    combined,
                    provider.reduce_max_tokens,
                ),
                should_retry=is_retryable,
                label="Reduce stage",
            ),
            floor=2,
        )
    except Exception as e:
        # Every chunk summary below this line already succeeded and was already
        # billed. Letting the reduce failure propagate would throw all of them
        # away and send the recording to failed/, so the map stage would have to
        # be paid for a second time. The timeline alone is worth reading.
        logger.error(
            "Reduce stage failed after every retry in %.1fs, returning the %d chunk "
            "summaries without a merged summary: %s",
            time.monotonic() - reduce_started,
            len(succeeded),
            e,
        )
        overall = REDUCE_FAILURE_NOTICE
    else:
        logger.info("Reduce stage: done in %.1fs", time.monotonic() - reduce_started)

    timeline = "\n\n".join(
        f"### [{_time_range(chunk)}]\n\n"
        f"{CHUNK_FAILURE_PLACEHOLDER if isinstance(summary, Exception) else summary}"
        for chunk, summary in zip(chunks, chunk_summaries)
    )
    return f"{overall}\n\n---\n\n## ไทม์ไลน์ตามช่วง\n\n{timeline}"
