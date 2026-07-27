import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from src.llm import (
    CLAUDE_MAP_MAX_TOKENS,
    CLAUDE_REDUCE_MAX_TOKENS,
    DEFAULT_LLM_BASE_URL,
    GLM_MAP_MAX_TOKENS,
    GLM_REDUCE_MAX_TOKENS,
    Completion,
    HttpStatusError,
    MissingApiKeyError,
    UnknownModelError,
    UnusableAnswerError,
    resolve,
)


def _anthropic_response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


def test_resolve_returns_claude_budgets():
    provider = resolve("claude-opus-5")

    assert provider.model_id == "claude-opus-5"
    assert provider.map_max_tokens == CLAUDE_MAP_MAX_TOKENS
    assert provider.reduce_max_tokens == CLAUDE_REDUCE_MAX_TOKENS


def test_resolve_rejects_an_unknown_model_by_name():
    with pytest.raises(UnknownModelError, match="ไม่มี-โมเดล-นี้"):
        resolve("ไม่มี-โมเดล-นี้")


def test_claude_completer_sends_the_prompt_and_returns_the_text():
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response("สรุป")

    with (
        patch("anthropic.Anthropic", return_value=client),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        result = resolve("claude-sonnet-5").complete("ระบบ", "เนื้อหา", 1234)

    assert result == Completion(text="สรุป", truncated=False)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["system"] == "ระบบ"
    assert kwargs["max_tokens"] == 1234
    assert kwargs["messages"] == [{"role": "user", "content": "เนื้อหา"}]


def test_claude_completer_flags_a_truncated_answer():
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response("ขาด", "max_tokens")

    with (
        patch("anthropic.Anthropic", return_value=client),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        result = resolve("claude-opus-5").complete("ระบบ", "เนื้อหา", 10)

    assert result.truncated is True
    assert result.text == "ขาด"


def test_claude_completer_raises_when_there_is_no_text_block():
    block = MagicMock()
    block.type = "thinking"
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = response

    with (
        patch("anthropic.Anthropic", return_value=client),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        pytest.raises(RuntimeError, match="no text block"),
    ):
        resolve("claude-opus-5").complete("ระบบ", "เนื้อหา", 10)


def test_claude_completer_names_the_env_var_when_the_key_is_missing():
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "   "}),
        pytest.raises(MissingApiKeyError, match="ANTHROPIC_API_KEY"),
    ):
        resolve("claude-opus-5").complete("ระบบ", "เนื้อหา", 10)


def _llm_payload(content: str, finish_reason: str = "stop"):
    return {
        "choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _patch_urlopen(payload):
    """คืน (context manager, ตัวเก็บ request) เพื่อ assert ว่าส่งอะไรออกไปจริง"""
    captured = {}

    def urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    return patch("urllib.request.urlopen", side_effect=urlopen), captured


def test_resolve_returns_the_glm_budgets():
    provider = resolve("GLM-5.2")

    assert provider.model_id == "GLM-5.2"
    assert provider.map_max_tokens == GLM_MAP_MAX_TOKENS
    assert provider.reduce_max_tokens == GLM_REDUCE_MAX_TOKENS
    assert GLM_MAP_MAX_TOKENS > CLAUDE_MAP_MAX_TOKENS


def test_glm_sends_thai_as_utf8_not_escape_sequences():
    """จุดที่พังจริงตอนทดลอง: ถ้า json.dumps escape เป็น \\uXXXX ปลายทางจะได้ ????"""
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        result = resolve("GLM-5.2").complete("ระบบ", "ผู้พูด 1 พูดว่าสวัสดี", 999)

    assert result == Completion(text="สรุป", truncated=False)
    request = captured["request"]
    body = request.data.decode("utf-8")
    assert "ผู้พูด 1 พูดว่าสวัสดี" in body
    assert "\\u" not in body
    assert request.headers["Content-type"] == "application/json; charset=utf-8"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.full_url == f"{DEFAULT_LLM_BASE_URL}/chat/completions"

    sent = json.loads(body)
    assert sent["model"] == "GLM-5.2"
    assert sent["max_tokens"] == 999
    assert sent["messages"] == [
        {"role": "system", "content": "ระบบ"},
        {"role": "user", "content": "ผู้พูด 1 พูดว่าสวัสดี"},
    ]


def test_glm_finish_reason_length_means_truncated():
    patcher, _ = _patch_urlopen(_llm_payload("ขาด", "length"))

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        result = resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert result.truncated is True
    assert result.text == "ขาด"


def test_glm_empty_content_is_a_failure_not_an_empty_summary():
    """GLM เป็น reasoning model: reasoning กิน max_tokens หมดได้ โดย content เป็น
    สตริงว่าง ถ้าปล่อยผ่านจะได้ summary.md เปล่าที่ไม่มีอะไรเตือน

    ต้องเป็น UnusableAnswerError โดยเฉพาะ (ไม่ใช่แค่ RuntimeError เฉยๆ) เพราะ
    is_retryable ใน summarize.py แยกชนิดนี้ออกมาไม่ให้ลองใหม่"""
    patcher, _ = _patch_urlopen(_llm_payload("", "length"))

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError, match="no text"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_whitespace_only_content_is_also_a_failure():
    patcher, _ = _patch_urlopen(_llm_payload("   \n  "))

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError, match="no text"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_http_error_carries_a_status_code_for_is_retryable():
    """is_retryable ใน summarize.py อ่าน .status_code -- exception ฝั่งนี้ต้องมีให้"""
    error = HTTPError(
        "https://example/v1/chat/completions", 429, "Too Many", {}, BytesIO(b"slow down")
    )

    with (
        patch("urllib.request.urlopen", side_effect=error),
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(HttpStatusError) as caught,
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert caught.value.status_code == 429


def test_glm_names_its_own_env_var_when_the_key_is_missing():
    with (
        patch.dict("os.environ", {"LLM_API_KEY": ""}),
        pytest.raises(MissingApiKeyError, match="LLM_API_KEY"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_honours_a_custom_base_url():
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict(
        "os.environ",
        {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://other.example/v1"},
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert (
        captured["request"].full_url == "https://other.example/v1/chat/completions"
    )
