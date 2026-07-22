import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import src.transcribe as transcribe_module
from src.transcribe import load_whisper_model, transcribe_audio


def test_transcribe_audio_extracts_segments_from_injected_model(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "สวัสดีครับ"},
            {"start": 2.5, "end": 5.0, "text": "วันนี้เรามาคุยกัน"},
        ]
    }

    result = transcribe_audio(audio_path, model=mock_model)

    assert result == [
        {"start": 0.0, "end": 2.5, "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "text": "วันนี้เรามาคุยกัน"},
    ]


def test_transcribe_audio_calls_model_with_thai_language(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": []}

    transcribe_audio(audio_path, model=mock_model)

    call_args = mock_model.transcribe.call_args
    assert call_args.args[0] == str(audio_path)
    assert call_args.kwargs["language"] == "th"


def test_transcribe_audio_loads_model_by_size_when_none_given(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": []}

    with patch(
        "src.transcribe.load_whisper_model", return_value=mock_model
    ) as mock_load:
        transcribe_audio(audio_path, model_size="medium")

    mock_load.assert_called_once_with("medium")
    mock_model.transcribe.assert_called_once()


def test_load_whisper_model_loads_and_caches_by_size(monkeypatch):
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    fake_whisper = ModuleType("whisper")
    fake_whisper.load_model = MagicMock(side_effect=lambda size: f"model-{size}")

    with patch.dict(sys.modules, {"whisper": fake_whisper}):
        model1 = load_whisper_model("small")
        model2 = load_whisper_model("small")

    assert model1 == "model-small"
    assert model2 == "model-small"
    fake_whisper.load_model.assert_called_once_with("small")
