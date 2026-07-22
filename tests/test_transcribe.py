import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from src.transcribe import transcribe_audio


def _fake_pydub(mock_audio):
    """Build a fake `pydub` module so tests never require real pydub/ffmpeg.

    Mirrors how heavy deps are kept out of the test env: production code lazily
    does `from pydub import AudioSegment`, and we inject a stand-in module.
    """
    module = ModuleType("pydub")
    module.AudioSegment = MagicMock()
    module.AudioSegment.from_file.return_value = mock_audio
    return module


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


def test_transcribe_audio_small_file_uses_single_call_without_pydub(tmp_path):
    audio_path = tmp_path / "small.mp3"
    audio_path.write_bytes(b"tiny audio well under the limit")

    mock_response = SimpleNamespace(
        segments=[SimpleNamespace(start=0.0, end=1.0, text="สั้นๆ")]
    )
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    fake_pydub = _fake_pydub(MagicMock())
    with (
        patch("src.transcribe.OpenAI", return_value=mock_client),
        patch.dict(sys.modules, {"pydub": fake_pydub}),
    ):
        result = transcribe_audio(audio_path, api_key="test-key")

    # Small file must go through the single-call path; pydub is never touched.
    fake_pydub.AudioSegment.from_file.assert_not_called()
    assert mock_client.audio.transcriptions.create.call_count == 1
    assert result == [{"start": 0.0, "end": 1.0, "text": "สั้นๆ"}]


def test_transcribe_audio_chunks_large_file_and_offsets_timestamps(tmp_path):
    audio_path = tmp_path / "long.mp3"
    # 25MB > the 24MB threshold, so the chunking path is taken.
    audio_path.write_bytes(b"\0" * (25 * 1024 * 1024))

    # duration 600000ms + 25MB => chunk_ms = int(0.96 * 600000) = 576000 =>
    # two chunks: [0, 576000) and [576000, 600000), second offset by 576.0s.
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 600000

    # A real chunk file must exist on disk for the production code to open and
    # transcribe it; the fake export writes one (and the code deletes it after).
    def fake_export(out_path, format):
        Path(out_path).write_bytes(b"chunk-bytes")

    mock_chunk = MagicMock()
    mock_chunk.export.side_effect = fake_export
    mock_audio.__getitem__.return_value = mock_chunk

    resp1 = SimpleNamespace(
        segments=[SimpleNamespace(start=0.0, end=10.0, text="chunk-a")]
    )
    resp2 = SimpleNamespace(
        segments=[SimpleNamespace(start=0.0, end=5.0, text="chunk-b")]
    )
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.side_effect = [resp1, resp2]

    fake_pydub = _fake_pydub(mock_audio)
    with (
        patch("src.transcribe.OpenAI", return_value=mock_client),
        patch.dict(sys.modules, {"pydub": fake_pydub}),
    ):
        result = transcribe_audio(audio_path, api_key="test-key")

    assert mock_client.audio.transcriptions.create.call_count == 2
    assert result == [
        {"start": 0.0, "end": 10.0, "text": "chunk-a"},
        {"start": 576.0, "end": 581.0, "text": "chunk-b"},
    ]
    # Temp chunk files must be cleaned up.
    assert list(tmp_path.glob("*.chunk*")) == []
