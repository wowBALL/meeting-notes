import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import src.summarize as summarize_module
from src.chunk import parse_transcript_segments, split_into_chunks
from src.summarize import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    SUMMARY_SYSTEM_PROMPT,
    is_retryable,
    summarize_transcript,
)


class FakeAPIError(Exception):
    """รูปร่างเดียวกับ anthropic.APIStatusError: exception ที่พก .status_code มาด้วย"""

    def __init__(self, status_code: int):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def _response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


def _patch_anthropic(client):
    return patch("anthropic.Anthropic", return_value=client)


def _single_response_client(text: str, stop_reason: str = "end_turn"):
    client = MagicMock()
    client.messages.create.return_value = _response(text, stop_reason)
    return client


def _prompt_aware_client():
    """Answers map calls with a numbered chunk summary and the reduce call with a
    fixed marker, keyed off the system prompt. Keeps assertions independent of how
    many chunks the splitter happens to produce."""
    state = {"map_calls": 0}
    lock = threading.Lock()

    def create(**kwargs):
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            return _response("สรุปรวมทั้งประชุม")
        with lock:
            index = state["map_calls"]
            state["map_calls"] += 1
        return _response(f"สรุปช่วง {index}")

    client = MagicMock()
    client.messages.create.side_effect = create
    return client


def _long_transcript(segment_count: int) -> str:
    blocks = [
        f"**ผู้พูด 1** [{i:02d}:00]: " + ("ก" * 400) for i in range(segment_count)
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


def test_short_transcript_returns_the_single_response_verbatim():
    client = _single_response_client("## ประเด็นสำคัญ\n- ทดสอบระบบ")

    with _patch_anthropic(client):
        result = summarize_transcript(
            "# Transcript\n\n**ผู้พูด 1** [00:00]: สั้นมาก", api_key="test-key"
        )

    assert result == "## ประเด็นสำคัญ\n- ทดสอบระบบ"
    assert client.messages.create.call_count == 1
    assert "ไทม์ไลน์" not in result


def test_short_transcript_uses_given_model_and_original_prompt():
    client = _single_response_client("สรุป")

    with _patch_anthropic(client):
        summarize_transcript("transcript", model="claude-sonnet-5", api_key="test-key")

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert "transcript" in kwargs["messages"][0]["content"]
    assert kwargs["system"] == summarize_module.SUMMARY_SYSTEM_PROMPT
    assert kwargs["max_tokens"] == summarize_module.MAP_MAX_OUTPUT_TOKENS


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
        summarize_transcript("สั้น", api_key="test-key")


def test_truncated_summary_is_logged_as_a_warning(caplog):
    client = _single_response_client("สรุปที่ถูกตัด", stop_reason="max_tokens")

    with _patch_anthropic(client), caplog.at_level(logging.WARNING):
        result = summarize_transcript("สั้น", api_key="test-key")

    assert result == "สรุปที่ถูกตัด"
    assert any("max_tokens" in record.getMessage() for record in caplog.records)


def test_long_transcript_makes_one_call_per_chunk_plus_one_reduce():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, api_key="test-key")

    assert client.messages.create.call_count == _expected_chunk_count(transcript) + 1


def test_long_transcript_output_has_merged_summary_then_timeline():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, api_key="test-key")

    assert result.startswith("สรุปรวมทั้งประชุม")
    assert "## ไทม์ไลน์ตามช่วง" in result
    assert "สรุปช่วง 0" in result
    assert "### [00:00–" in result


def test_long_transcript_chunk_calls_use_the_chunk_prompt():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, api_key="test-key")

    first_kwargs = client.messages.create.call_args_list[0].kwargs
    assert first_kwargs["system"] == summarize_module.CHUNK_SYSTEM_PROMPT
    assert first_kwargs["max_tokens"] == summarize_module.MAP_MAX_OUTPUT_TOKENS


def test_long_transcript_reduce_call_uses_reduce_prompt_and_larger_cap():
    transcript = _long_transcript(100)
    client = _prompt_aware_client()

    with _patch_anthropic(client):
        summarize_transcript(transcript, api_key="test-key")

    reduce_kwargs = client.messages.create.call_args_list[-1].kwargs
    assert reduce_kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT
    assert reduce_kwargs["max_tokens"] == summarize_module.REDUCE_MAX_OUTPUT_TOKENS


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
        result = summarize_transcript(transcript, api_key="test-key")

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
        result = summarize_transcript("สั้น", api_key="test-key")

    # the pipeline no longer wraps this call, so the retry has to live here
    assert result == "สรุปสั้น"
    assert calls["n"] == 2


def test_a_permanently_failing_chunk_becomes_a_placeholder():
    transcript = _long_transcript(100)
    dead_chunk = _chunk_texts(transcript)[0]
    reduce_input_holder = {}

    def create(**kwargs):
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            reduce_input_holder["content"] = kwargs["messages"][0]["content"]
            return _response("สรุปรวมทั้งประชุม")
        if kwargs["messages"][0]["content"] == dead_chunk:
            raise RuntimeError("permanent api error")
        return _response("สรุปช่วงที่สำเร็จ")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client), patch("time.sleep"):
        result = summarize_transcript(transcript, api_key="test-key")

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
        summarize_transcript(transcript, api_key="test-key")


def test_map_calls_run_concurrently():
    transcript = _long_transcript(100)
    assert _expected_chunk_count(transcript) >= 2
    assert summarize_module.MAP_MAX_CONCURRENCY >= 2

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def create(**kwargs):
        if kwargs["system"] != summarize_module.CHUNK_SYSTEM_PROMPT:
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
        result = summarize_transcript(transcript, api_key="test-key")

    assert state["peak"] >= 2
    assert summarize_module.CHUNK_FAILURE_PLACEHOLDER not in result


def test_chunk_summaries_keep_transcript_order_under_concurrency():
    transcript = _long_transcript(100)
    chunk_texts = _chunk_texts(transcript)
    index_of = {text: i for i, text in enumerate(chunk_texts)}

    def create(**kwargs):
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            return _response("สรุปรวมทั้งประชุม")
        index = index_of[kwargs["messages"][0]["content"]]
        # earlier chunks answer last, so completion order is the reverse of
        # transcript order
        time.sleep(0.02 * (len(chunk_texts) - index))
        return _response(f"สรุปช่วง {index}")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, api_key="test-key")

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
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            reduce_input_holder["content"] = kwargs["messages"][0]["content"]
            return _response("สรุปรวมทั้งประชุม")
        return _response(chunk_summary_with_headings)

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, api_key="test-key")

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
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            return _response(reduce_output)
        return _response("- สรุปช่วง")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, api_key="test-key")

    # an H1 from the model would swallow "## ไทม์ไลน์ตามช่วง" into its own section
    assert result.startswith("## สรุปการประชุม: เรื่องที่คุยกัน")
    assert "# สรุปการประชุม" not in result.replace("## สรุปการประชุม", "")
    # levels at or below the document's own H2 are left exactly as the model wrote them
    assert "## ประเด็นสำคัญ" in result
    assert "### หัวข้อย่อย" in result
    assert "## Action Items" in result
    assert "## ไทม์ไลน์ตามช่วง" in result


def test_over_threshold_transcript_with_no_parseable_segments_falls_back_to_single_call():
    # Exceeds SINGLE_CALL_THRESHOLD_TOKENS but has no "**speaker** [MM:SS]:" blocks,
    # so parse_transcript_segments returns [] and chunks is [].
    transcript = "# Transcript\n\n" + ("ไม่มีรูปแบบ segment ที่แยกได้ " * 3000)
    client = _single_response_client("สรุปแบบเดียว")

    with _patch_anthropic(client):
        result = summarize_transcript(transcript, api_key="test-key")

    assert result == "สรุปแบบเดียว"
    assert client.messages.create.call_count == 1
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == SUMMARY_SYSTEM_PROMPT


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 529])
def test_is_retryable_accepts_failures_that_can_clear_on_their_own(status_code):
    assert is_retryable(FakeAPIError(status_code)) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413])
def test_is_retryable_rejects_failures_that_answer_the_same_every_time(status_code):
    assert is_retryable(FakeAPIError(status_code)) is False


def test_is_retryable_accepts_an_error_that_never_reached_the_api():
    # ไม่มี status_code = ต่อไม่ติด / หมดเวลา ซึ่งเป็นอาการที่หายเองได้
    assert is_retryable(OSError("connection reset by peer")) is True


def test_single_call_stops_after_one_attempt_when_the_key_is_rejected():
    client = MagicMock()
    client.messages.create.side_effect = FakeAPIError(401)

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        with pytest.raises(FakeAPIError):
            summarize_transcript("**ผู้พูด 1** [00:00]: สวัสดี", api_key="test-key")

    assert client.messages.create.call_count == 1
    mock_sleep.assert_not_called()


def test_map_stage_sends_one_request_per_chunk_when_credit_runs_out():
    # เดิมเครดิตหมดในประชุมยาวทำให้ยิงไป 3 เท่าของจำนวน chunk ทั้งที่รู้คำตอบตั้งแต่ครั้งแรก
    transcript = _long_transcript(100)
    client = MagicMock()
    client.messages.create.side_effect = FakeAPIError(400)

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Every one of the"):
            summarize_transcript(transcript, api_key="test-key")

    assert client.messages.create.call_count == _expected_chunk_count(transcript)
    mock_sleep.assert_not_called()


def test_reduce_stage_stops_after_one_attempt_when_the_key_is_rejected():
    # chunk ผ่านหมดแต่ reduce โดนปฏิเสธ -- เป็นคนละ call site กับ map จึงต้องมีเทสของตัวเอง
    transcript = _long_transcript(100)

    def create(**kwargs):
        if kwargs["system"] == summarize_module.REDUCE_SYSTEM_PROMPT:
            raise FakeAPIError(401)
        return _response("สรุปช่วงนี้")

    client = MagicMock()
    client.messages.create.side_effect = create

    with _patch_anthropic(client), patch("time.sleep") as mock_sleep:
        with pytest.raises(FakeAPIError):
            summarize_transcript(transcript, api_key="test-key")

    assert client.messages.create.call_count == _expected_chunk_count(transcript) + 1
    mock_sleep.assert_not_called()
