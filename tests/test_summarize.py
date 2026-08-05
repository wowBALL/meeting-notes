import contextlib
import inspect
import logging
import socket
import threading
import time
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

import src.summarize as summarize_module
from src.chunk import estimate_tokens, parse_transcript_segments, split_into_chunks
from src.llm import (
    CLAUDE_MAP_MAX_TOKENS,
    CLAUDE_REDUCE_MAX_TOKENS,
    HttpStatusError,
    MissingSettingError,
    Provider,
    UnusableAnswerError,
    _anthropic_completer,
)
from src.summarize import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    REDUCE_FAILURE_NOTICE,
    SINGLE_CALL_THRESHOLD_TOKENS,
    is_retryable,
    summarize_transcript,
)


class FakeAPIError(Exception):
    """รูปร่างเดียวกับ anthropic.APIStatusError: exception ที่พก .status_code มาด้วย"""

    def __init__(self, status_code: int):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


# prompt จริงมาจากไฟล์ prompts/*.md แล้ว จึงเทียบเท่ากันเป๊ะกับค่าคงที่ในโค้ดไม่ได้อีก
# (ค่าคงที่เหลือไว้เป็น prompt สำรองตอนไฟล์หายเท่านั้น) เทียบด้วยประโยคที่มีอยู่ทั้งใน
# ไฟล์และใน prompt สำรอง เพื่อให้ตัวแยกขั้นตอนนี้ยังถูกต้องทั้งสองทาง
def _is_map(system: str) -> bool:
    return "เพียงบางช่วง" in system


def _is_reduce(system: str) -> bool:
    return "รวมทั้งหมดเป็นสรุปฉบับเดียว" in system


def _is_single(system: str) -> bool:
    """นิยามด้วยการตัดออก ไม่ใช่ด้วยวลีของตัวเอง -- ถ้อยคำใน single.md เปลี่ยนได้
    ตอนจูน แต่ "ไม่ใช่ map และไม่ใช่ reduce" เป็นจริงตลอด
    (test_the_stage_markers_stay_mutually_exclusive ล็อกสมบัติข้อนี้ไว้)"""
    return not _is_map(system) and not _is_reduce(system)


def test_a_tuned_chunk_overlap_actually_changes_the_chunking():
    """ค่า overlap ที่ตั้งใน .env ต้องมีผลจริง ไม่ใช่รับมาแล้ววางเฉยๆ
    overlap มากขึ้น = เนื้อหาถูกเล่นซ้ำมากขึ้น = จำนวน chunk มากขึ้นบน transcript เดิม"""
    transcript = _long_transcript(200)

    def chunk_count(overlap):
        client = _prompt_aware_client()
        with _patch_anthropic(client):
            summarize_transcript(
                transcript, model="claude-opus-5", chunk_overlap_tokens=overlap
            )
        return sum(
            1
            for call in client.messages.create.call_args_list
            if _is_map(call.kwargs["system"])
        )

    no_overlap = chunk_count(0)
    heavy_overlap = chunk_count(7_500)

    assert no_overlap >= 2, "sanity: transcript ต้องยาวพอให้หั่นหลาย chunk"
    assert heavy_overlap > no_overlap


def test_the_default_overlap_is_used_when_none_is_given():
    transcript = _long_transcript(200)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    default_calls = sum(
        1 for c in client.messages.create.call_args_list if _is_map(c.kwargs["system"])
    )
    assert default_calls == _expected_chunk_count(transcript)


def test_the_stage_markers_stay_mutually_exclusive():
    """ตัวแยกขั้นตอนด้านบนคือฐานของเทสต์อีกสิบกว่าตัวในไฟล์นี้ ถ้ามันแยกผิด
    fake client จะตอบผิดสาขาแล้วเทสต์พวกนั้นล้มแบบชี้สาเหตุไม่ได้
    ตรวจทั้ง prompt จากไฟล์จริงและ prompt สำรองที่ฝังในโค้ด"""
    from src.prompts import FALLBACKS, render

    for name in ("map", "reduce", "single"):
        for source, system in (
            ("file", render(name)),
            ("fallback", FALLBACKS[name]),
        ):
            flags = {
                "map": _is_map(system),
                "reduce": _is_reduce(system),
                "single": _is_single(system),
            }
            matched = [stage for stage, hit in flags.items() if hit]
            assert matched == [name], (
                f"{name}.md ({source}) ถูกแยกเป็น {matched} ควรเป็น ['{name}']"
            )


def _response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


@contextlib.contextmanager
def _patch_anthropic(client):
    """patch ทั้ง SDK และ env var เพราะ llm._require_setting อ่าน ANTHROPIC_API_KEY จริง

    เดิมโค้ดรับ api_key มาเป็น argument จึงไม่ต้องมี env ตอนเทส ตอนนี้ key เป็นเรื่อง
    ของ provider ซึ่งอ่านจาก environment -- ใส่ให้ที่นี่ที่เดียวแทนการเติม 24 ที่
    """
    with (
        patch("anthropic.Anthropic", return_value=client),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        yield


def _single_response_client(text: str, stop_reason: str = "end_turn"):
    client = MagicMock()
    client.messages.create.return_value = _response(text, stop_reason)
    return client


def _truncating_client(texts_and_reasons):
    """ตอบตามลำดับที่กำหนด เพื่อ assert ได้ว่าการลองใหม่เกิดขึ้นกี่ครั้งและด้วย budget เท่าไร"""
    client = MagicMock()
    client.messages.create.side_effect = [
        _response(text, reason) for text, reason in texts_and_reasons
    ]
    return client


def _prompt_aware_client():
    """Answers map calls with a numbered chunk summary and the reduce call with a
    fixed marker, keyed off the system prompt. Keeps assertions independent of how
    many chunks the splitter happens to produce."""
    state = {"map_calls": 0}
    lock = threading.Lock()

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            return _response("สรุปรวมทั้งประชุม")
        with lock:
            index = state["map_calls"]
            state["map_calls"] += 1
        return _response(f"สรุปช่วง {index}")

    client = MagicMock()
    client.messages.create.side_effect = create
    return client


def _long_transcript(segment_count: int) -> str:
    # ผู้พูดสลับกันทุกบล็อกเหมือนบทสนทนาจริง เพื่อให้ merge_speaker_turns ไม่มีอะไรให้รวม
    # -- ถ้าให้เป็นคนเดียวทั้งไฟล์ เทสต์หลายสิบตัวที่ใช้ fixture นี้จะขึ้นกับเพดานการรวม
    # โดยบังเอิญ (ตอนนี้บล็อกละ ~387 token ซึ่งเกินครึ่งของเพดาน 600 พอดีจนไม่รวม)
    blocks = [
        f"**ผู้พูด {i % 2 + 1}** [{i:02d}:00]: " + ("ก" * 400)
        for i in range(segment_count)
    ]
    return "# Transcript\n\n" + "\n\n".join(blocks)


def _chunk_texts(transcript: str) -> list[str]:
    """The exact strings the map stage will send, so a test can key its fake
    client off *which* chunk it is answering rather than off call ordering --
    which the concurrent map stage no longer fixes."""
    segments = parse_transcript_segments(transcript)
    chunks = split_into_chunks(segments, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
    return [chunk["text"] for chunk in chunks]


def _expected_chunk_count(transcript: str) -> int:
    return len(_chunk_texts(transcript))


def test_map_budget_comes_from_the_provider_not_a_module_constant():
    """ค่า budget ต้องเดินทางมาจาก provider ที่ resolve() คืนมาโดยตรง ไม่ใช่จากค่าคงที่
    ในโมดูลนี้หรือใน llm.py -- แพตช์ resolve() ให้คืน budget ที่ไม่ตรงกับค่าคงที่ตัวไหน
    ในโค้ดเลย (ไม่ใช่ CLAUDE_MAP_MAX_TOKENS หรือ CLAUDE_REDUCE_MAX_TOKENS) ถ้า max_tokens
    ที่ไปถึง wire ยังตรงกับค่านี้ แปลว่ามันเดินทางมาจาก provider จริง ไม่ใช่ค่าตายตัวที่
    บังเอิญเท่ากัน (ก่อนหน้านี้เทสต์เทียบกับ CLAUDE_MAP_MAX_TOKENS ซึ่งบังเอิญเท่ากับค่า
    module constant เดิมที่ถูกลบไปแล้ว จึงผ่านได้แม้ก่อน refactor และจะผ่านต่อไปแม้มีคน
    เพิ่ม module constant กลับมาใหม่)"""
    distinct_map_budget = 1234
    assert distinct_map_budget not in (CLAUDE_MAP_MAX_TOKENS, CLAUDE_REDUCE_MAX_TOKENS)
    fake_provider = Provider(
        model_id="claude-opus-5",
        map_max_tokens=distinct_map_budget,
        reduce_max_tokens=distinct_map_budget * 2,
        complete=_anthropic_completer("claude-opus-5", "ANTHROPIC_API_KEY"),
    )
    client = _single_response_client("สรุป")

    with _patch_anthropic(client), patch.object(
        summarize_module, "resolve", return_value=fake_provider
    ):
        summarize_transcript("transcript", model="claude-opus-5")

    assert client.messages.create.call_args.kwargs["max_tokens"] == distinct_map_budget


def test_summarize_transcript_no_longer_accepts_an_api_key():
    """key เป็นเรื่องของ provider ไม่ใช่ของผู้เรียก การส่ง key ของ Anthropic เข้า
    เส้นทางที่อาจไปจบที่ provider อื่นเป็นสิ่งที่ต้องเป็นไปไม่ได้"""
    assert "api_key" not in inspect.signature(summarize_transcript).parameters


def test_short_transcript_returns_the_single_response_verbatim():
    client = _single_response_client("## ประเด็นสำคัญ\n- ทดสอบระบบ")

    with _patch_anthropic(client):
        result = summarize_transcript(
            "# Transcript\n\n**ผู้พูด 1** [00:00]: สั้นมาก", model="claude-opus-5"
        )

    assert result == "## ประเด็นสำคัญ\n- ทดสอบระบบ"
    assert client.messages.create.call_count == 1
    assert "ไทม์ไลน์" not in result


def _one_speaker_transcript(segment_count: int, chars: int = 100) -> str:
    """ผู้พูดคนเดียวรวดทั้งไฟล์ -- กรณีที่ merge_speaker_turns มีของให้รวมจริง"""
    blocks = [
        f"**ผู้พูด 1** [{i:02d}:00]: " + ("ก" * chars) for i in range(segment_count)
    ]
    return "# Transcript\n\n" + "\n\n".join(blocks)


def test_merging_runs_before_the_single_call_threshold():
    """ประชุมที่ดิบแล้วต้องหั่น chunk แต่รวมบล็อกแล้วเหลือรอบเดียว ต้องยิงครั้งเดียว

    นี่คือเหตุผลที่การรวมต้องเกิดก่อนบรรทัด SINGLE_CALL_THRESHOLD_TOKENS ไม่ใช่หลัง:
    จำนวนคำขอที่ลดลงคือทั้งหมดที่ฟีเจอร์นี้มีไว้ทำ ถ้าสลับลำดับ เทสต์นี้จะเห็นสองคำขอ
    """
    # บล็อกละ ~46 token: รวมกันได้เต็มเพดาน 600 ทำให้ทั้งไฟล์หดลงต่ำกว่าเกณฑ์ยิงรอบเดียว
    transcript = _one_speaker_transcript(700, chars=25)
    assert estimate_tokens(transcript) > SINGLE_CALL_THRESHOLD_TOKENS
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    assert client.messages.create.call_count == 1
    sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert estimate_tokens(sent) <= SINGLE_CALL_THRESHOLD_TOKENS
    assert "# Transcript" in sent


def test_merging_off_sends_the_transcript_untouched():
    transcript = _one_speaker_transcript(3)
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5", merge_turns=False)

    assert client.messages.create.call_args.kwargs["messages"][0]["content"] == (
        transcript
    )


def test_merging_on_joins_adjacent_turns_before_sending():
    transcript = _one_speaker_transcript(3)
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert sent != transcript
    assert sent.count("**ผู้พูด 1**") == 1
    assert sent.count("ก") == transcript.count("ก")


def test_merging_cap_shrinks_with_a_small_overlap_budget():
    """เพดานการรวมต้องไม่โตกว่างบ overlap ไม่งั้นบล็อกที่รวมแล้วถูกเล่นซ้ำไม่ได้เลย"""
    transcript = _one_speaker_transcript(3, chars=200)
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript(
            transcript, model="claude-opus-5", chunk_overlap_tokens=200
        )

    sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
    # เพดานกลายเป็น 100 (=200//2) บล็อกละ ~200 token จึงรวมไม่ได้เลยแม้จะเป็นคนเดียวกัน
    assert sent == transcript


def test_short_transcript_uses_given_model_and_original_prompt():
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript("transcript", model="claude-sonnet-5")

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert "transcript" in kwargs["messages"][0]["content"]
    # prompt ย้ายไปอยู่ prompts/single.md แล้ว ไม่ใช่ค่าคงที่ในโค้ด -- ค่าคงที่เหลือ
    # ไว้เป็นตัวสำรองเมื่อไฟล์หายเท่านั้น จึงเทียบว่าเนื้อหาหลักยังอยู่ ไม่เทียบเท่ากันเป๊ะ
    assert "## ตกลงแล้ว" in kwargs["system"]
    assert "## Action items" in kwargs["system"]
    assert kwargs["max_tokens"] == CLAUDE_MAP_MAX_TOKENS


def test_the_system_prompt_comes_from_the_prompt_files_not_the_constants():
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript("transcript", model="claude-sonnet-5")

    system = client.messages.create.call_args.kwargs["system"]
    assert system != summarize_module.SUMMARY_SYSTEM_PROMPT
    # ประโยคนี้อยู่ใน prompts/single.md เท่านั้น ไม่ได้อยู่ใน prompt สำรองที่ฝังในโค้ด
    assert "faster-whisper" in system
    assert "ในห้องมีแต่ทีม dev" in system, "profile dev ต้องถูกแทรกเข้ามาโดยปริยาย"


def test_the_cross_profile_adds_its_own_rules_to_the_prompt():
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript("transcript", model="claude-sonnet-5", profile="cross")

    system = client.messages.create.call_args.kwargs["system"]
    assert "ทำได้" in system and "ไม่ใช่การรับปาก" in system
    assert "ในห้องมีแต่ทีม dev" not in system


def test_an_unknown_profile_still_produces_a_summary(caplog):
    """profile ที่พิมพ์ผิดต้องไม่ทำให้ประชุมนั้นไม่ได้สรุปเลย -- transcript
    ถอดเสียงเสร็จและเสียเวลา GPU ไปแล้ว"""
    client = _single_response_client("สรุป")

    with _patch_anthropic(client), caplog.at_level(logging.WARNING):
        result = summarize_transcript(
            "transcript", model="claude-sonnet-5", profile="พิมพ์ผิด"
        )

    assert result == "สรุป"
    assert "พิมพ์ผิด" in caplog.text
    assert "ในห้องมีแต่ทีม dev" in client.messages.create.call_args.kwargs["system"]


def test_the_glossary_lands_in_the_prompt_where_the_placeholder_was():
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript(
            "transcript",
            model="claude-sonnet-5",
            glossary_text="## คำศัพท์ในประชุมนี้\n- Electron ← อิเล็กตรอน",
        )

    system = client.messages.create.call_args.kwargs["system"]
    assert "Electron ← อิเล็กตรอน" in system
    assert "{glossary}" not in system


def test_a_missing_prompts_directory_falls_back_and_still_summarizes(tmp_path, caplog):
    client = _single_response_client("สรุป")

    with (
        _patch_anthropic(client),
        patch("src.prompts.DEFAULT_PROMPTS_DIR", tmp_path / "ไม่มีอยู่"),
        caplog.at_level(logging.WARNING),
    ):
        result = summarize_transcript("transcript", model="claude-sonnet-5")

    assert result == "สรุป"
    assert client.messages.create.call_args.kwargs["system"] == (
        summarize_module.SUMMARY_SYSTEM_PROMPT
    )


def test_response_without_a_text_block_raises_a_diagnosable_error():
    block = MagicMock()
    block.type = "thinking"
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = response

    with (
        _patch_anthropic(client),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="no text block"),
    ):
        summarize_transcript("สั้น", model="claude-opus-5")


def test_a_truncated_answer_is_logged_as_a_warning(caplog):
    """log ยังต้องมี เพราะการลองใหม่คือค่าใช้จ่ายที่ควรมองเห็นย้อนหลังได้"""
    client = _truncating_client([("ขาด", "max_tokens"), ("จบ", "end_turn")])

    with _patch_anthropic(client), caplog.at_level(logging.WARNING):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert result == "จบ"
    assert any("truncated" in record.getMessage().lower() for record in caplog.records)


def test_a_truncated_answer_is_retried_once_with_double_the_budget():
    from src.llm import CLAUDE_MAP_MAX_TOKENS

    client = _truncating_client(
        [("ขาดกลางประโยค", "max_tokens"), ("เต็มแล้ว", "end_turn")]
    )

    with _patch_anthropic(client):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert result == "เต็มแล้ว"
    assert client.messages.create.call_count == 2
    budgets = [c.kwargs["max_tokens"] for c in client.messages.create.call_args_list]
    assert budgets == [CLAUDE_MAP_MAX_TOKENS, CLAUDE_MAP_MAX_TOKENS * 2]


def test_still_truncated_after_the_retry_keeps_the_text_and_appends_a_notice():
    """ลองใหม่แค่ครั้งเดียว: การเรียกซ้ำแพงและช้า (GLM หนึ่ง call ถึง 155 วินาที)
    ข้อความที่ได้มาแล้วมีค่ากว่าการทิ้ง แต่คนอ่านต้องรู้ว่ามันไม่จบ"""
    client = _truncating_client(
        [("ท่อนแรก", "max_tokens"), ("ท่อนแรกที่ยาวขึ้นแต่ยังไม่จบ", "max_tokens")]
    )

    with _patch_anthropic(client):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert client.messages.create.call_count == 2
    assert "ท่อนแรกที่ยาวขึ้นแต่ยังไม่จบ" in result
    assert summarize_module.TRUNCATION_NOTICE in result


def _client_whose_second_call_raises(first_text, first_reason, error):
    client = MagicMock()
    client.messages.create.side_effect = [_response(first_text, first_reason), error]
    return client


def test_a_truncated_answer_whose_doubled_budget_retry_raises_keeps_the_first_calls_text():
    """ถ้า call ที่สอง (budget สองเท่า) เจอ UnusableAnswerError -- เช่น reasoning model
    ได้ budget เพิ่มแล้วยิ่งใช้ไปกับ reasoning จน content ว่างเปล่า -- ต้องไม่โยนทิ้ง
    ข้อความที่ call แรกได้มาแล้วไปเฉยๆ ต้องคืนข้อความนั้นพร้อม TRUNCATION_NOTICE
    เหมือนกรณีที่ retry สำเร็จแต่ยังขาดอยู่ดี ไม่ใช่แย่ลงกว่าเดิม"""
    client = _client_whose_second_call_raises(
        "ท่อนแรกที่ได้มาแล้ว", "max_tokens", UnusableAnswerError("no text")
    )

    with _patch_anthropic(client):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert client.messages.create.call_count == 2
    assert "ท่อนแรกที่ได้มาแล้ว" in result
    assert summarize_module.TRUNCATION_NOTICE in result


def test_a_truncated_answer_whose_doubled_budget_retry_raises_400_keeps_the_first_calls_text():
    """เหมือนเทสต์ข้างบนแต่เป็น HttpStatusError(400) -- proxy บางตัวปฏิเสธ max_tokens
    ที่ใหญ่เกินเพดานต่อคำขอ (24576 × 2 = 49152) ด้วย 400 ซึ่งไม่ retryable เหมือนกัน"""
    client = _client_whose_second_call_raises(
        "ท่อนแรกที่ได้มาแล้ว", "max_tokens", HttpStatusError(400, "max_tokens too large")
    )

    with _patch_anthropic(client):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert client.messages.create.call_count == 2
    assert "ท่อนแรกที่ได้มาแล้ว" in result
    assert summarize_module.TRUNCATION_NOTICE in result


def test_a_doubled_budget_retry_that_raises_inside_a_chunk_keeps_the_first_calls_text():
    """เหมือนสองเทสต์ข้างบนแต่เกิดใน chunk หนึ่งของ map stage -- chunk นั้นต้องยังได้
    ข้อความบางส่วนไปต่อในไทม์ไลน์ ไม่ใช่กลายเป็น CHUNK_FAILURE_PLACEHOLDER ซึ่งเป็น
    ผลลัพธ์ที่แย่กว่าของเดิมที่ branch นี้ตั้งใจแก้"""
    transcript = _long_transcript(100)
    target_chunk = _chunk_texts(transcript)[0]
    lock = threading.Lock()
    calls_for_target = {"n": 0}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            return _response("สรุปรวมทั้งประชุม")
        if kwargs["messages"][0]["content"] == target_chunk:
            with lock:
                calls_for_target["n"] += 1
                n = calls_for_target["n"]
            if n == 1:
                return _response("ท่อนแรกของช่วงนี้", "max_tokens")
            raise UnusableAnswerError("GLM-5.2 returned no text (finish_reason='length')")
        return _response("สรุปช่วงปกติ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert calls_for_target["n"] == 2
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in result
    assert "ท่อนแรกของช่วงนี้" in result
    timeline_start = result.index("## ไทม์ไลน์ตามช่วง")
    assert summarize_module.TRUNCATION_NOTICE in result[timeline_start:]


def test_a_doubled_budget_retry_that_raises_400_inside_a_chunk_keeps_the_first_calls_text():
    transcript = _long_transcript(100)
    target_chunk = _chunk_texts(transcript)[0]
    lock = threading.Lock()
    calls_for_target = {"n": 0}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            return _response("สรุปรวมทั้งประชุม")
        if kwargs["messages"][0]["content"] == target_chunk:
            with lock:
                calls_for_target["n"] += 1
                n = calls_for_target["n"]
            if n == 1:
                return _response("ท่อนแรกของช่วงนี้", "max_tokens")
            raise HttpStatusError(400, "max_tokens too large")
        return _response("สรุปช่วงปกติ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert calls_for_target["n"] == 2
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in result
    assert "ท่อนแรกของช่วงนี้" in result
    timeline_start = result.index("## ไทม์ไลน์ตามช่วง")
    assert summarize_module.TRUNCATION_NOTICE in result[timeline_start:]


def test_an_answer_that_fits_is_not_retried():
    client = _single_response_client("พอดี")

    with _patch_anthropic(client):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    assert result == "พอดี"
    assert client.messages.create.call_count == 1
    assert summarize_module.TRUNCATION_NOTICE not in result


def test_long_transcript_makes_one_call_per_chunk_plus_one_reduce():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    assert client.messages.create.call_count == _expected_chunk_count(transcript) + 1


def test_long_transcript_output_has_merged_summary_then_timeline():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert result.startswith("สรุปรวมทั้งประชุม")
    assert "## ไทม์ไลน์ตามช่วง" in result
    assert "สรุปช่วง 0" in result
    assert "### [00:00–" in result


def test_long_transcript_chunk_calls_use_the_chunk_prompt():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    first_kwargs = client.messages.create.call_args_list[0].kwargs
    assert _is_map(first_kwargs["system"])
    assert first_kwargs["max_tokens"] == CLAUDE_MAP_MAX_TOKENS


def test_long_transcript_reduce_call_uses_reduce_prompt_and_larger_cap():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, model="claude-opus-5")

    reduce_kwargs = client.messages.create.call_args_list[-1].kwargs
    assert _is_reduce(reduce_kwargs["system"])
    assert reduce_kwargs["max_tokens"] == CLAUDE_REDUCE_MAX_TOKENS


def test_a_failing_chunk_is_retried_individually():
    transcript = _long_transcript(100)
    expected_chunks = _expected_chunk_count(transcript)
    first_chunk = _chunk_texts(transcript)[0]
    lock = threading.Lock()
    calls = {"total": 0, "first_chunk": 0}

    def flaky(**kwargs):
        with lock:
            calls["total"] += 1
            first_attempt_at_first_chunk = False
            if kwargs["messages"][0]["content"] == first_chunk:
                calls["first_chunk"] += 1
                first_attempt_at_first_chunk = calls["first_chunk"] == 1
        if first_attempt_at_first_chunk:
            raise RuntimeError("transient api error")
        return _response("ok")

    client = MagicMock()
    client.messages.create.side_effect = flaky

    with _patch_anthropic(client), patch("time.sleep"):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert "ok" in result
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in result
    # every chunk + one reduce + exactly one extra attempt for the retried chunk;
    # a restart-everything retry would cost far more calls than this
    assert calls["total"] == expected_chunks + 1 + 1


def test_short_path_is_retried_inside_summarize():
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient api error")
        return _response("สรุปสั้น")

    client = MagicMock()
    client.messages.create.side_effect = flaky

    with _patch_anthropic(client), patch("time.sleep"):
        result = summarize_transcript("สั้น", model="claude-opus-5")

    # the pipeline no longer wraps this call, so the retry has to live here
    assert result == "สรุปสั้น"
    assert calls["n"] == 2


def test_a_permanently_failing_chunk_becomes_a_placeholder():
    transcript = _long_transcript(100)
    dead_chunk = _chunk_texts(transcript)[0]
    reduce_input_holder = {}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            reduce_input_holder["content"] = kwargs["messages"][0]["content"]
            return _response("สรุปรวมทั้งประชุม")
        if kwargs["messages"][0]["content"] == dead_chunk:
            raise RuntimeError("permanent api error")
        return _response("สรุปช่วงที่สำเร็จ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client), patch("time.sleep"):
        result = summarize_transcript(transcript, model="claude-opus-5")

    # the healthy chunks still reach the user instead of the whole meeting failing
    assert result.startswith("สรุปรวมทั้งประชุม")
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER in result
    assert "สรุปช่วงที่สำเร็จ" in result
    # the reduce stage is fed real summaries only -- a placeholder there would
    # invite the model to describe the outage instead of the meeting
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in reduce_input_holder["content"]


def test_every_chunk_failing_still_raises():
    transcript = _long_transcript(100)

    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("permanent api error")

    with (
        _patch_anthropic(client),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="permanent api error"),
    ):
        summarize_transcript(transcript, model="claude-opus-5")


def test_map_calls_run_concurrently():
    transcript = _long_transcript(100)
    assert _expected_chunk_count(transcript) >= 2
    assert summarize_module.MAP_MAX_CONCURRENCY >= 2

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def create(**kwargs):
        if not _is_map(kwargs["system"]):
            return _response("สรุปรวมทั้งประชุม")
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        # held long enough that a sequential map stage could never overlap two
        time.sleep(0.05)
        with lock:
            state["in_flight"] -= 1
        return _response("ok")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert state["peak"] >= 2
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in result


def test_chunk_summaries_keep_transcript_order_under_concurrency():
    transcript = _long_transcript(100)
    chunk_texts = _chunk_texts(transcript)
    index_of = {text: i for i, text in enumerate(chunk_texts)}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            return _response("สรุปรวมทั้งประชุม")
        index = index_of[kwargs["messages"][0]["content"]]
        # earlier chunks answer last, so completion order is the reverse of
        # transcript order
        time.sleep(0.02 * (len(chunk_texts) - index))
        return _response(f"สรุปช่วง {index}")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    positions = [result.index(f"สรุปช่วง {i}") for i in range(len(chunk_texts))]
    assert positions == sorted(positions)


def test_long_transcript_demotes_heading_levels_in_chunk_summaries():
    transcript = _long_transcript(100)
    chunk_summary_with_headings = (
        "## หัวข้อระดับสอง\n- อยากจะเก็บ bullet\n"
        "### หัวข้อระดับสาม\n- bullet อีกอัน\n"
        "##### หัวข้อระดับห้า\n- bullet สุดท้าย"
    )
    reduce_input_holder = {}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            reduce_input_holder["content"] = kwargs["messages"][0]["content"]
            return _response("สรุปรวมทั้งประชุม")
        return _response(chunk_summary_with_headings)

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    # ## and ### are demoted to the floor (####); ##### is left alone
    assert "#### หัวข้อระดับสอง" in result
    assert "#### หัวข้อระดับสาม" in result
    assert "##### หัวข้อระดับห้า" in result
    assert "\n## หัวข้อระดับสอง" not in result
    assert "\n### หัวข้อระดับสาม" not in result

    # the reduce ("combined") input sees the same demoted text, not the raw headings
    combined = reduce_input_holder["content"]
    assert "#### หัวข้อระดับสอง" in combined
    assert "\n## หัวข้อระดับสอง" not in combined


def test_reduce_summary_headings_cannot_outrank_the_timeline_section():
    transcript = _long_transcript(100)
    reduce_output = (
        "# สรุปการประชุม: เรื่องที่คุยกัน\n\n"
        "## ประเด็นสำคัญ\n\n"
        "### หัวข้อย่อย\n- ข้อหนึ่ง\n\n"
        "## Action Items\n- ทำต่อ"
    )

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            return _response(reduce_output)
        return _response("- สรุปช่วง")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    # an H1 from the model would swallow "## ไทม์ไลน์ตามช่วง" into its own section
    assert result.startswith("## สรุปการประชุม: เรื่องที่คุยกัน")
    assert "# สรุปการประชุม" not in result.replace("## สรุปการประชุม", "")
    # levels at or below the document's own H2 are left exactly as the model wrote them
    assert "## ประเด็นสำคัญ" in result
    assert "### หัวข้อย่อย" in result
    assert "## Action Items" in result
    assert "## ไทม์ไลน์ตามช่วง" in result


def test_a_doubly_truncated_chunk_reaches_the_timeline_but_not_the_reduce_input():
    """chunk หนึ่งขาดทั้งสองครั้ง -- หมายเหตุต้องโผล่ในไทม์ไลน์ให้คนอ่านเห็น
    แต่ต้องไม่หลุดเข้าไปในข้อความที่ส่งให้ reduce call (นั่นคือสิ่งที่ finding
    ตัวนี้จับได้: reduce เห็นสรุปจริงเท่านั้น ไม่ใช่หมายเหตุเรื่อง call ที่ขาด)"""
    transcript = _long_transcript(100)
    assert _expected_chunk_count(transcript) >= 2
    target_chunk = _chunk_texts(transcript)[0]
    reduce_input_holder = {}
    lock = threading.Lock()
    calls_for_target = {"n": 0}

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            reduce_input_holder["content"] = kwargs["messages"][0]["content"]
            return _response("สรุปรวมทั้งประชุม")
        if kwargs["messages"][0]["content"] == target_chunk:
            with lock:
                calls_for_target["n"] += 1
                n = calls_for_target["n"]
            return _response(f"ท่อนที่ {n} ของช่วงนี้", "max_tokens")
        return _response("สรุปช่วงปกติ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    # the retry-once-at-double-budget escalation actually ran for this chunk
    assert calls_for_target["n"] == 2

    assert summarize_module.TRUNCATION_NOTICE not in reduce_input_holder["content"]

    timeline_start = result.index("## ไทม์ไลน์ตามช่วง")
    assert summarize_module.TRUNCATION_NOTICE in result[timeline_start:]


def test_reduce_truncated_twice_notice_appears_above_the_timeline_separator():
    """reduce เองก็ขาดทั้งสองครั้งได้เหมือนกัน -- หมายเหตุต้องอยู่ในสรุปรวม
    (เหนือ --- ที่คั่นก่อนไทม์ไลน์) ไม่ใช่แค่ในสรุปรายช่วง"""
    transcript = _long_transcript(100)
    reduce_calls = {"n": 0}
    lock = threading.Lock()

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            with lock:
                reduce_calls["n"] += 1
            return _response("สรุปที่ยังไม่จบ", "max_tokens")
        return _response("สรุปช่วงปกติ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    # the retry-once-at-double-budget escalation ran for the reduce call too
    assert reduce_calls["n"] == 2

    before_separator, _, after_separator = result.partition("\n\n---\n\n")
    assert summarize_module.TRUNCATION_NOTICE in before_separator
    assert "## ไทม์ไลน์ตามช่วง" in after_separator


def test_over_threshold_transcript_with_no_parseable_segments_falls_back_to_single_call():
    # Exceeds SINGLE_CALL_THRESHOLD_TOKENS but has no "**speaker** [MM:SS]:" blocks,
    # so parse_transcript_segments returns [] and chunks is [].
    transcript = "# Transcript\n\n" + ("ไม่มีรูปแบบ segment ที่แยกได้ " * 3000)
    client = _single_response_client("สรุปแบบเดียว")

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert result == "สรุปแบบเดียว"
    assert client.messages.create.call_count == 1
    kwargs = client.messages.create.call_args.kwargs
    assert _is_single(kwargs["system"])


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 529])
def test_is_retryable_accepts_failures_that_can_clear_on_their_own(status_code):
    assert is_retryable(FakeAPIError(status_code)) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413])
def test_is_retryable_rejects_failures_that_answer_the_same_every_time(status_code):
    assert is_retryable(FakeAPIError(status_code)) is False


def test_is_retryable_accepts_an_error_that_never_reached_the_api():
    # ไม่มี status_code = ต่อไม่ติด / หมดเวลา ซึ่งเป็นอาการที่หายเองได้
    assert is_retryable(OSError("connection reset by peer")) is True


@pytest.mark.parametrize(
    "error",
    [
        URLError("nodename nor servname provided, or not known"),
        TimeoutError("timed out"),
        URLError(socket.timeout()),
    ],
    ids=["urlerror_dns", "bare_timeout", "urlerror_wrapping_socket_timeout"],
)
def test_is_retryable_accepts_timeout_and_dns_failures(error):
    # จุดที่แพงที่สุดถ้าเดาผิด: ที่ LLM_TIMEOUT_SECONDS = 900 การตัดสินผิดที่นี่คือ
    # 900 วินาที คูณด้วยจำนวนรอบ retry ต่อ chunk -- ปลายทางที่ต่อไม่ติดหรือหมดเวลา
    # ต้องลองใหม่ได้เสมอ ไม่มีข้อยกเว้น
    assert is_retryable(error) is True


def test_is_retryable_by_exception_shape():
    """สี่รูปแบบที่ is_retryable ต้องแยกออกจากกัน:

    UnusableAnswerError และ MissingSettingError เป็นคำตอบที่แน่นอนแล้ว -- ยิงซ้ำก็ได้
    ผลเดิม HttpStatusError(429) ยังลองใหม่ได้เหมือน 4xx/5xx ทั่วไป ส่วน RuntimeError
    ธรรมดา (ไม่มี status_code, ไม่ใช่สองชนิดข้างต้น) ยังต้องลองใหม่ได้เหมือนเดิม --
    นี่คือกรณีที่กฎเดิม (ทุก RuntimeError ที่ไม่มี status_code ไม่ retryable) จับพลาด
    """
    assert is_retryable(UnusableAnswerError("no text")) is False
    assert is_retryable(MissingSettingError("no key")) is False
    assert is_retryable(HttpStatusError(429, "slow down")) is True
    assert is_retryable(RuntimeError("transient")) is True


def test_single_call_stops_after_one_attempt_when_the_key_is_rejected():
    client = MagicMock()
    client.messages.create.side_effect = FakeAPIError(401)

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        with pytest.raises(FakeAPIError):
            summarize_transcript("**ผู้พูด 1** [00:00]: สวัสดี", model="claude-opus-5")

    assert client.messages.create.call_count == 1
    mock_sleep.assert_not_called()


def test_map_stage_sends_one_request_per_chunk_when_credit_runs_out():
    # เดิมเครดิตหมดในประชุมยาวทำให้ยิงไป 3 เท่าของจำนวน chunk ทั้งที่รู้คำตอบตั้งแต่ครั้งแรก
    transcript = _long_transcript(100)
    client = MagicMock()
    client.messages.create.side_effect = FakeAPIError(400)

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Every one of the"):
            summarize_transcript(transcript, model="claude-opus-5")

    assert client.messages.create.call_count == _expected_chunk_count(transcript)
    mock_sleep.assert_not_called()


def _reduce_fails_client(status_code: int = 401):
    """map สำเร็จทุก chunk แต่ reduce โดนปฏิเสธ -- สภาพตอนเครดิตหมดกลางคัน"""

    def create(**kwargs):
        if _is_reduce(kwargs["system"]):
            raise FakeAPIError(status_code)
        return _response("สรุปช่วงนี้")

    client = MagicMock()
    client.messages.create.side_effect = create
    return client


def test_reduce_stage_stops_after_one_attempt_when_the_key_is_rejected():
    # reduce เป็นคนละ call site กับ map จึงต้องมีเทสของตัวเอง
    transcript = _long_transcript(100)
    client = _reduce_fails_client()

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        summarize_transcript(transcript, model="claude-opus-5")

    assert client.messages.create.call_count == _expected_chunk_count(transcript) + 1
    mock_sleep.assert_not_called()


def test_reduce_failure_keeps_the_chunk_summaries_that_were_already_paid_for():
    # เดิม reduce พังแล้วโยน exception ออกไปเลย ทิ้งสรุปย่อยทุกช่วงที่เรียกสำเร็จ
    # (และจ่ายเงินไปแล้ว) ทั้งหมด
    transcript = _long_transcript(100)
    client = _reduce_fails_client()

    with _patch_anthropic(client), patch("time.sleep"):
        result = summarize_transcript(transcript, model="claude-opus-5")

    assert REDUCE_FAILURE_NOTICE in result
    assert "## ไทม์ไลน์ตามช่วง" in result
    assert result.count("สรุปช่วงนี้") == _expected_chunk_count(transcript)


def test_reduce_failure_still_raises_when_no_chunk_ever_succeeded():
    # ไม่มีอะไรให้เก็บ -- ปล่อยให้ pipeline ย้ายไฟล์ไป failed/ ตามเดิม
    transcript = _long_transcript(100)
    client = MagicMock()
    client.messages.create.side_effect = FakeAPIError(401)

    with _patch_anthropic(client), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="Every one of the"):
            summarize_transcript(transcript, model="claude-opus-5")


def test_a_provider_that_returns_no_text_is_not_retried():
    calls = {"n": 0}

    def complete(system, content, max_tokens):
        calls["n"] += 1
        raise UnusableAnswerError(
            "GLM-5.2 returned no text (finish_reason='length')"
        )

    provider = MagicMock()
    provider.model_id = "GLM-5.2"
    provider.map_max_tokens = 100
    provider.reduce_max_tokens = 200
    provider.single_call_threshold_tokens = 20_000
    provider.complete = complete

    with (
        patch("src.summarize.resolve", return_value=provider),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="no text"),
    ):
        summarize_transcript("สั้น", model="GLM-5.2")

    assert calls["n"] == 1


def test_the_map_stage_announces_how_much_work_is_in_flight(caplog):
    """ก่อนหน้านี้ขั้นนี้ไม่ log อะไรเลยจนกว่า chunk จะตาย -- endpoint ที่ค้างจึงให้ความ
    เงียบสนิทได้ถึง 45 นาทีต่อ chunk ซึ่งแยกไม่ออกจาก process ที่แขวนไปแล้ว
    (2026-07-31: เสียไปหนึ่งชั่วโมงกว่าจะรู้ว่ามีอะไรผิดปกติ)

    คนอ่าน log ต้องรู้จำนวน chunk ก่อนบรรทัดรายก้อนจะทยอยมา ไม่งั้น "Chunk 5/6" ที่โผล่
    มาก่อนเพราะ pool ทำงานขนานกัน จะอ่านเหมือนว่า chunk อื่นหายไป"""
    transcript = _long_transcript(120)
    expected = _expected_chunk_count(transcript)
    assert expected >= 2, "sanity: transcript ต้องยาวพอให้เข้าเส้นทาง map-reduce"
    client = _prompt_aware_client()

    with _patch_anthropic(client), caplog.at_level(logging.INFO):
        summarize_transcript(transcript, model="claude-opus-5")

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        f"{expected} chunks" in m and "at a time" in m for m in messages
    ), messages
    # ทุก chunk ต้องมีทั้งบรรทัดเริ่มและบรรทัดจบ พร้อมเวลาที่ใช้
    for i in range(1, expected + 1):
        assert any(f"Chunk {i}/{expected}" in m and "starting" in m for m in messages)
        assert any(f"Chunk {i}/{expected}" in m and "done in" in m for m in messages)
    assert any("Map stage finished" in m for m in messages)
    assert any("Reduce stage: merging" in m for m in messages)
    assert any("Reduce stage: done" in m for m in messages)


def test_on_progress_reports_one_completion_per_chunk():
    """แถบใน UI เคยค้างที่ "กำลังสรุป" ตั้งแต่นาทีแรกจนจบ แยกไม่ออกว่ายังทำงานอยู่
    หรือแขวนไปแล้ว -- รายงานตอน "เสร็จ" ไม่ใช่ตอน "เริ่ม" และนับจำนวนแทนการอ้าง index
    เพราะ chunk จบไม่เรียงลำดับ: "3/6 เสร็จแล้ว" จริงเสมอไม่ว่าจะเป็นสามก้อนไหน"""
    transcript = _long_transcript(120)
    expected = _expected_chunk_count(transcript)
    assert expected >= 2, "sanity: transcript ต้องยาวพอให้เข้าเส้นทาง map-reduce"
    client = _prompt_aware_client()
    seen = []

    with _patch_anthropic(client):
        summarize_transcript(
            transcript, model="claude-opus-5", on_progress=lambda d, t: seen.append((d, t))
        )

    assert len(seen) == expected
    assert [d for d, _ in seen] == list(range(1, expected + 1))
    assert {t for _, t in seen} == {expected}


def test_a_failing_progress_callback_does_not_lose_the_summary(caplog):
    """รายงานความคืบหน้าที่พังต้องไม่ทำให้สรุปที่จ่ายเงินไปแล้วหายไป -- เหตุผลเดียวกับ
    ที่ activity.append() กลืน OSError ทิ้ง"""
    transcript = _long_transcript(120)
    client = _prompt_aware_client()

    def explode(done, total):
        raise RuntimeError("activity feed is on fire")

    with _patch_anthropic(client), caplog.at_level(logging.ERROR):
        result = summarize_transcript(
            transcript, model="claude-opus-5", on_progress=explode
        )

    assert "สรุปรวมทั้งประชุม" in result
    assert "Progress callback failed" in caplog.text


def test_on_progress_is_optional():
    """ผู้เรียกเดิมทุกจุดต้องยังทำงานได้โดยไม่ต้องแก้"""
    transcript = _long_transcript(120)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        assert summarize_transcript(transcript, model="claude-opus-5")
