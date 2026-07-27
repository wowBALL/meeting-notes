from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    CLAUDE_MAP_MAX_TOKENS,
    CLAUDE_REDUCE_MAX_TOKENS,
    Completion,
    MissingApiKeyError,
    UnknownModelError,
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
