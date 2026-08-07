import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from src.llm import (
    CLAUDE_MAP_MAX_TOKENS,
    CLAUDE_REDUCE_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    PROBE_TIMEOUT_SECONDS,
    GLM_MAP_MAX_TOKENS,
    GLM_REDUCE_MAX_TOKENS,
    QWEN_MAP_MAX_TOKENS,
    QWEN_REDUCE_MAX_TOKENS,
    Completion,
    check_reachable,
    HttpStatusError,
    MissingSettingError,
    UnknownModelError,
    UnusableAnswerError,
    UpstreamBodyError,
    resolve,
)


# .invalid เป็น TLD ที่สงวนไว้ไม่มีทาง resolve ได้ -- ถ้าเทสหลุดไปยิงจริงก็ไปไม่ถึงไหน
# และที่อยู่ของ endpoint จริงไม่ต้องอยู่ในไฟล์นี้เลย
TEST_BASE_URL = "https://llm.test.invalid/v1"


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
        pytest.raises(MissingSettingError, match="ANTHROPIC_API_KEY"),
    ):
        resolve("claude-opus-5").complete("ระบบ", "เนื้อหา", 10)


def test_claude_completer_uses_the_sdks_own_timeout_but_disables_its_retries():
    # complete() มีแค่สามพารามิเตอร์ (ไม่มี timeout override แล้ว) -- ต้องไม่มีใครถูกจำกัด
    # เวลาแบบเงียบๆ ปล่อยให้ SDK ใช้ timeout default ของมันเองเสมอ แต่ max_retries ต้อง
    # เป็น 0 เป๊ะ: SDK default (2) จะซ้อนกับ retry_with_backoff และ escalation budget
    # สองเท่าใน summarize.py ทำให้ worst case ต่อ chunk คูณกันเกินจริง (ดู
    # src/pipeline.py) retry ทั้งหมดต้องอยู่ที่ is_retryable ที่เดียว
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response("สรุป")

    with (
        patch("anthropic.Anthropic", return_value=client) as anthropic_cls,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        resolve("claude-opus-5").complete("ระบบ", "เนื้อหา", 10)

    kwargs = anthropic_cls.call_args.kwargs
    assert "timeout" not in kwargs
    assert kwargs["max_retries"] == 0


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


def _patch_urlopen_bytes(raw_body: bytes):
    """คืน (context manager, ตัวเก็บ request) เพื่อ assert ว่าส่งอะไรออกไปจริง

    รับ bytes เพราะเป็นหน่วยที่ตรงกับสิ่งที่ response ส่งกลับจริง และเพราะ body ที่ไบต์
    ไม่ใช่ UTF-8 เขียนเป็น str ไม่ได้ อีกสองตัวข้างล่างเป็นเพียงทางสะดวกที่แปลงมาลงตัวนี้
    """
    captured = {}

    def urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(raw_body)

    return patch("urllib.request.urlopen", side_effect=urlopen), captured


def _patch_urlopen_raw(raw_body: str):
    """ส่ง string ดิบๆ ตรงๆ ไม่ผ่าน json.dumps -- ใช้จำลอง body ที่ไม่ใช่ JSON เลย
    (หน้า error HTML, สตริงว่าง, JSON ที่ถูกตัดกลางคัน)"""
    return _patch_urlopen_bytes(raw_body.encode("utf-8"))


def _patch_urlopen(payload):
    """ส่ง payload ที่ผ่าน json.dumps -- ใช้กับ response ที่มีรูปร่างถูกต้อง"""
    return _patch_urlopen_bytes(json.dumps(payload).encode("utf-8"))


def test_resolve_returns_the_glm_budgets():
    provider = resolve("GLM-5.2")

    assert provider.model_id == "GLM-5.2"
    assert provider.map_max_tokens == GLM_MAP_MAX_TOKENS
    assert provider.reduce_max_tokens == GLM_REDUCE_MAX_TOKENS
    assert GLM_MAP_MAX_TOKENS > CLAUDE_MAP_MAX_TOKENS


def test_resolve_returns_the_qwen_budgets():
    """Qwen ใช้ endpoint เดียวกับ GLM แต่ไม่ใช่ reasoning model

    งบจึงต้องเป็นชุดของ Claude ไม่ใช่ของ GLM ที่เผื่อไว้ให้ reasoning กินไปสี่เท่า
    -- วัดจริงแล้วมันใช้ 602 token บน chunk ที่ใหญ่ที่สุดของประชุม 84 นาที
    """
    provider = resolve("Qwen/Qwen3.6-35B-A3B")

    assert provider.model_id == "Qwen/Qwen3.6-35B-A3B"
    assert provider.map_max_tokens == QWEN_MAP_MAX_TOKENS
    assert provider.reduce_max_tokens == QWEN_REDUCE_MAX_TOKENS
    assert QWEN_MAP_MAX_TOKENS < GLM_MAP_MAX_TOKENS


def test_gemma4_is_barred_from_the_map_reduce_path():
    """เพดาน 150,000 token ย้ายประชุมขนาดปกติออกจากทาง map-reduce ได้ แต่ไม่ได้ปิดทางนั้น
    ทิ้ง -- ประชุมที่เกินเพดานยังตกลงไปเจอ reduce ที่ยุบเนื้อหาเงียบๆ ได้อยู่ดี ธงนี้คือ
    ตัวปิด ส่วนเพดานคือตัวเลี่ยง ต้องมีทั้งคู่ (ดูคอมเมนต์ที่ GEMMA_SINGLE_CALL_THRESHOLD_TOKENS)
    """
    assert resolve("litellm/gemma4").can_map_reduce is False


def test_every_other_model_keeps_the_map_reduce_path():
    """ข้อห้ามต้องผูกกับโมเดลที่วัดแล้วว่าพังเท่านั้น ไม่ใช่กลายเป็นค่าเริ่มต้นของทุกตัว
    -- Qwen สรุปครบทั้ง 3 chunks ในการวัดเดียวกันที่ gemma4 ตก
    """
    for model_id in ("GLM-5.2", "Qwen/Qwen3.6-35B-A3B", "claude-opus-5", "claude-sonnet-5"):
        assert resolve(model_id).can_map_reduce is True, model_id


def test_qwen_uses_the_same_endpoint_settings_as_glm():
    """key และ base URL ตัวเดียวกัน -- ตั้ง .env เพิ่มไม่ต้องทำอะไรเลยเมื่อสลับมาใช้ตัวนี้"""
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict(
        "os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}
    ):
        result = resolve("Qwen/Qwen3.6-35B-A3B").complete("ระบบ", "เนื้อหา", 999)

    assert result == Completion(text="สรุป", truncated=False)
    request = captured["request"]
    assert request.full_url == f"{TEST_BASE_URL}/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    # ชื่อรุ่นต้องออกไปเป๊ะ ๆ ทั้ง "Qwen/" และตัวพิมพ์ -- proxy ปฏิเสธชื่อที่ไม่ตรง
    assert json.loads(request.data.decode("utf-8"))["model"] == "Qwen/Qwen3.6-35B-A3B"


def test_qwen_without_the_llm_key_names_the_setting_to_fix():
    with patch.dict("os.environ", {"LLM_API_KEY": "", "LLM_BASE_URL": TEST_BASE_URL}):
        with pytest.raises(MissingSettingError, match="LLM_API_KEY"):
            resolve("Qwen/Qwen3.6-35B-A3B").complete("ระบบ", "เนื้อหา", 999)


def test_glm_sends_thai_as_utf8_not_escape_sequences():
    """จุดที่พังจริงตอนทดลอง: ถ้า json.dumps escape เป็น \\uXXXX ปลายทางจะได้ ????"""
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
        result = resolve("GLM-5.2").complete("ระบบ", "ผู้พูด 1 พูดว่าสวัสดี", 999)

    assert result == Completion(text="สรุป", truncated=False)
    request = captured["request"]
    body = request.data.decode("utf-8")
    assert "ผู้พูด 1 พูดว่าสวัสดี" in body
    assert "\\u" not in body
    assert request.headers["Content-type"] == "application/json; charset=utf-8"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.full_url == f"{TEST_BASE_URL}/chat/completions"
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

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
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
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
        pytest.raises(UnusableAnswerError, match="no text"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_whitespace_only_content_is_also_a_failure():
    patcher, _ = _patch_urlopen(_llm_payload("   \n  "))

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
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
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
        pytest.raises(UnusableAnswerError, match="budget exceeded for key"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_empty_choices_list_does_not_raise_indexerror():
    patcher, _ = _patch_urlopen({"choices": []})

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
        pytest.raises(UnusableAnswerError),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_choice_with_no_message_key_does_not_raise_keyerror():
    patcher, _ = _patch_urlopen({"choices": [{"finish_reason": "stop"}]})

    with (
        patcher,
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
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
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
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

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
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

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
        with pytest.raises(UpstreamBodyError) as caught:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert not isinstance(caught.value, (UnusableAnswerError, MissingSettingError))
    assert is_retryable(caught.value) is True
    if marker:
        assert marker in str(caught.value)


def test_glm_non_utf8_body_is_retried():
    """body ที่มีไบต์ที่ไม่ใช่ UTF-8 นั้นมักเป็นอาการชั่วคราวของ gateway corruption
    กลางทาง (เหมือนกับ HTML error page หรือ truncated stream) -- ต้อง retryable
    เหมือนเดิม แต่ข้อความต้องบอกว่า UTF-8 decode ล้มเหลว"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen_bytes(b"\xff\xfe not utf-8")

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
        with pytest.raises(UpstreamBodyError) as caught:
            resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert is_retryable(caught.value) is True
    assert "UTF-8" in str(caught.value)


def test_glm_malformed_response_is_not_retried():
    """เรียกซ้ำคำขอเดิมก็ได้ error envelope เดิม -- ต้องไม่ใช่ประเภทที่ is_retryable
    เดาว่า retryable (ดู src/summarize.py::is_retryable)"""
    from src.summarize import is_retryable

    patcher, _ = _patch_urlopen({"error": {"message": "budget exceeded"}})

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
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
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}),
        pytest.raises(HttpStatusError) as caught,
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert caught.value.status_code == 429


def test_glm_names_its_own_env_var_when_the_key_is_missing():
    with (
        patch.dict("os.environ", {"LLM_API_KEY": ""}),
        pytest.raises(MissingSettingError, match="LLM_API_KEY"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_completer_always_uses_llm_timeout_seconds():
    # complete() มีแค่สามพารามิเตอร์ (ไม่มี timeout override แล้ว) -- ต้องได้
    # LLM_TIMEOUT_SECONDS เสมอ
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 999)

    assert captured["timeout"] == LLM_TIMEOUT_SECONDS


def test_glm_names_the_base_url_env_var_when_it_is_not_set():
    """ไม่มี default ของ base URL ในโค้ดแล้ว (ที่อยู่ endpoint ภายในไม่ควรอยู่ใน repo
    สาธารณะ) -- ถ้าไม่ตั้ง LLM_BASE_URL ต้องบอกชื่อ env var ตรงๆ ไม่ใช่ไปยิง URL เปล่า
    แล้วได้ error ที่อ่านไม่ออก"""
    with (
        patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": ""}),
        pytest.raises(MissingSettingError, match="LLM_BASE_URL"),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)


def test_glm_uses_the_configured_base_url():
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict(
        "os.environ",
        {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://other.example/v1"},
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert (
        captured["request"].full_url == "https://other.example/v1/chat/completions"
    )


def test_glm_strips_a_trailing_slash_from_the_base_url():
    # README บอกห้ามใส่ "/" ปิดท้ายไว้แค่ในเนื้อความ ไม่มีอะไรบังคับจริง -- ถ้าใส่มา
    # ผลลัพธ์เดิมคือ ".../v1//chat/completions" ซึ่งเป็น 404 ทึบ ไม่ retryable แล้ว
    # ทุก chunk กลายเป็น placeholder ทั้งประชุม ต้องกันไว้ที่โค้ด ไม่ใช่แค่บอกในเอกสาร
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with patcher, patch.dict(
        "os.environ",
        {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://other.example/v1/"},
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 10)

    assert (
        captured["request"].full_url == "https://other.example/v1/chat/completions"
    )


def test_check_reachable_treats_an_empty_reasoning_answer_as_reachable():
    """*** กับดักที่วัดมาจากของจริง 2026-07-31 ***
    GLM-5.2 เป็น reasoning model ที่ max_tokens คุมผลรวมของ reasoning + คำตอบ ที่ budget
    ระดับ PROBE_MAX_TOKENS มันใช้หมดไปกับ reasoning แล้วคืน content="" เป็นเรื่องปกติ
    (ยิงจริงด้วย max_tokens=32 ได้ content ว่างคู่กับ reasoning_content ยาวห้าย่อหน้า)

    ถ้านับกรณีนี้เป็น "ไปไม่ถึง" ทุกประชุมจะถูกปฏิเสธตั้งแต่ยังไม่เริ่ม ทั้งที่ endpoint
    ปกติดีทุกอย่าง -- แย่กว่าปัญหาที่ฟังก์ชันนี้ถูกสร้างมาแก้เสียอีก"""
    patcher, _ = _patch_urlopen(_llm_payload("", "length"))

    with (
        patcher,
        patch.dict(
            "os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}
        ),
    ):
        check_reachable(resolve("GLM-5.2"))  # ต้องไม่ raise


def test_check_reachable_raises_when_the_endpoint_never_answers():
    def urlopen(request, timeout=None):
        raise TimeoutError("the read operation timed out")

    with (
        patch("urllib.request.urlopen", side_effect=urlopen),
        patch.dict(
            "os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}
        ),
    ):
        with pytest.raises(TimeoutError):
            check_reachable(resolve("GLM-5.2"))


def test_check_reachable_uses_the_short_timeout_not_the_summarizing_one():
    """ทั้งหมดของเรื่องนี้คือการไม่รอ 900 วินาทีเพื่อรู้ว่าปลายทางไปไม่ถึง"""
    patcher, captured = _patch_urlopen(_llm_payload("OK"))

    with (
        patcher,
        patch.dict(
            "os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}
        ),
    ):
        check_reachable(resolve("GLM-5.2"))

    assert captured["timeout"] == PROBE_TIMEOUT_SECONDS
    assert PROBE_TIMEOUT_SECONDS < LLM_TIMEOUT_SECONDS


def test_check_reachable_reports_a_missing_setting_instead_of_swallowing_it():
    """ตั้งค่าไม่ครบคือความล้มเหลวจริงที่ควรหยุดตั้งแต่ตรงนี้ ไม่ใช่ไปตายทีละ chunk"""
    with patch.dict("os.environ", {"LLM_API_KEY": "", "LLM_BASE_URL": ""}, clear=False):
        with pytest.raises(MissingSettingError):
            check_reachable(resolve("GLM-5.2"))


def test_a_normal_completion_still_leaves_the_sdk_timeout_alone():
    """timeout ที่ไม่ได้ส่งมา = ใช้ค่า default เดิมของแต่ละ provider ไม่ใช่ค่าที่
    check_reachable ต้องการ -- การเพิ่มพารามิเตอร์ต้องไม่เปลี่ยนพฤติกรรมของผู้เรียกเดิม"""
    patcher, captured = _patch_urlopen(_llm_payload("สรุป"))

    with (
        patcher,
        patch.dict(
            "os.environ", {"LLM_API_KEY": "test-key", "LLM_BASE_URL": TEST_BASE_URL}
        ),
    ):
        resolve("GLM-5.2").complete("ระบบ", "เนื้อหา", 100)

    assert captured["timeout"] == LLM_TIMEOUT_SECONDS
