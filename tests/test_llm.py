import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from src.llm import (
    CLAUDE_MAP_MAX_TOKENS,
    CLAUDE_REDUCE_MAX_TOKENS,
    DEFAULT_LLM_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    GLM_MAP_MAX_TOKENS,
    GLM_REDUCE_MAX_TOKENS,
    Completion,
    HttpStatusError,
    MissingApiKeyError,
    UnknownModelError,
    UnusableAnswerError,
    UpstreamBodyError,
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


def _patch_urlopen_raw(raw_body: str):
    """เหมือน _patch_urlopen แต่ส่ง string ดิบๆ ตรงๆ ไม่ผ่าน json.dumps -- ใช้จำลอง
    body ที่ไม่ใช่ JSON เลย (หน้า error HTML, สตริงว่าง, JSON ที่ถูกตัดกลางคัน)"""
    captured = {}

    def urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(raw_body.encode("utf-8"))

    return patch("urllib.request.urlopen", side_effect=urlopen), captured


def _patch_urlopen_bytes(raw_body: bytes):
    """เหมือน _patch_urlopen_raw แต่รับ bytes ดิบๆ โดยไม่ encode -- ใช้จำลอง body ที่
    มีไบต์ที่ไม่ใช่ UTF-8 เป็นต้น"""
    captured = {}

    def urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(raw_body)

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
    assert captured["timeout"] == LLM_TIMEOUT_SECONDS

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


def test_glm_200_error_envelope_surfaces_the_proxys_own_message():
    """LiteLLM ตอบ 200 พร้อม error envelope แทนคำตอบได้ (เช่น เกิน budget ของ key)
    payload["choices"] จะ KeyError ถ้าไม่ระวัง -- ต้องจับก่อนแล้วโชว์ข้อความของ proxy
    เอง ไม่ใช่แค่ 'choices'"""
    patcher, _ = _patch_urlopen({"error": {"message": "budget exceeded for key", "type": "budget"}})

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError, match="budget exceeded for key"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_empty_choices_list_does_not_raise_indexerror():
    patcher, _ = _patch_urlopen({"choices": []})

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_choice_with_no_message_key_does_not_raise_keyerror():
    patcher, _ = _patch_urlopen({"choices": [{"finish_reason": "stop"}]})

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_content_as_parts_list_does_not_raise_attributeerror():
    """OpenAI content parts: content เป็น list ของ {"type": "text", "text": ...} ได้
    ค่านี้ truthy อยู่แล้วเลยผ่าน `or ""` ไป แล้วไป .strip() แตกด้วย AttributeError"""
    patcher, _ = _patch_urlopen(
        {"choices": [{"finish_reason": "stop", "message": {"content": [{"type": "text", "text": "สรุป"}]}}]}
    )

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        pytest.raises(UnusableAnswerError),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


@pytest.mark.parametrize(
    "payload",
    [None, [1, 2], "oops", 42],
    ids=["null", "list", "string", "number"],
)
def test_glm_non_object_top_level_is_not_retried(payload):
    """top-level ของ body เป็น JSON ที่ถูกต้อง แต่ไม่ใช่ object (null/list/string/
    number) เดิม .get() แตกเป็น AttributeError ดิบๆ ตอนนี้ต้องเป็น UnusableAnswerError
    ที่คงที่แน่นอน -- ยิงคำขอเดิมซ้ำก็ได้รูปร่างเดิม จึงต้องไม่ retryable"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen(payload)

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        with pytest.raises(UnusableAnswerError) as caught:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert is_retryable(caught.value) is False


@pytest.mark.parametrize(
    "raw_body,marker",
    [
        ("<html><body>502 Bad Gateway</body></html>", "502 Bad Gateway"),
        ("", None),
        ('{"choices": [', '{"choices": ['),
    ],
    ids=["html_error_page", "empty_body", "truncated_json"],
)
def test_glm_non_json_body_is_retried_with_body_preserved(raw_body, marker):
    """body ที่ไม่ใช่ JSON เลย (gateway hiccup กลางทาง) มักหายเองรอบถัดไป -- ต้องยัง
    retryable เหมือนเดิม แต่ข้อความ exception ต้องโชว์ raw body แทน JSONDecodeError
    ดิบๆ ที่บอกอะไรไม่ได้ ('Expecting value: line 1 column 1')"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen_raw(raw_body)

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        with pytest.raises(UpstreamBodyError) as caught:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert not isinstance(caught.value, (UnusableAnswerError, MissingApiKeyError))
    assert is_retryable(caught.value) is True
    if marker:
        assert marker in str(caught.value)


def test_glm_non_utf8_body_is_retried():
    """body ที่มีไบต์ที่ไม่ใช่ UTF-8 นั้นมักเป็นอาการชั่วคราวของ gateway corruption
    กลางทาง (เหมือนกับ HTML error page หรือ truncated stream) -- ต้อง retryable
    เหมือนเดิม แต่ข้อความต้องบอกว่า UTF-8 decode ล้มเหลว"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen_bytes(b"\xff\xfe not utf-8")

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        with pytest.raises(UpstreamBodyError) as caught:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert is_retryable(caught.value) is True
    assert "UTF-8" in str(caught.value)


def test_glm_malformed_response_is_not_retried():
    """เรียกซ้ำคำขอเดิมก็ได้ error envelope เดิม -- ต้องไม่ใช่ประเภทที่ is_retryable
    เดาว่า retryable (ดู src/summarize.py::is_retryable)"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen({"error": {"message": "budget exceeded"}})

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
        try:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)
            pytest.fail("expected an exception")
        except Exception as e:
            assert is_retryable(e) is False


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
