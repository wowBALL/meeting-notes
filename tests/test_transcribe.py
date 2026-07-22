from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.transcribe import transcribe_audio


def test_transcribe_audio_extracts_segments(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    mock_response = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=2.5, text="สวัสดีครับ"),
            SimpleNamespace(start=2.5, end=5.0, text="วันนี้เรามาคุยกัน"),
        ]
    )
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("src.transcribe.OpenAI", return_value=mock_client):
        result = transcribe_audio(audio_path, api_key="test-key")

    assert result == [
        {"start": 0.0, "end": 2.5, "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "text": "วันนี้เรามาคุยกัน"},
    ]


def test_transcribe_audio_calls_whisper_with_thai_language(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    mock_response = SimpleNamespace(segments=[])
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("src.transcribe.OpenAI", return_value=mock_client):
        transcribe_audio(audio_path, api_key="test-key")

    call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["language"] == "th"
    assert call_kwargs["response_format"] == "verbose_json"
