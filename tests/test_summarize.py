from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.summarize import summarize_transcript


def test_summarize_transcript_returns_text_from_response():
    mock_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="## ประเด็นสำคัญ\n- ทดสอบระบบ")]
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = summarize_transcript("# Transcript\n\nสวัสดีครับ", api_key="test-key")

    assert result == "## ประเด็นสำคัญ\n- ทดสอบระบบ"


def test_summarize_transcript_uses_given_model():
    mock_response = SimpleNamespace(content=[SimpleNamespace(type="text", text="สรุป")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        summarize_transcript("transcript", model="claude-sonnet-5", api_key="test-key")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert "transcript" in call_kwargs["messages"][0]["content"]
