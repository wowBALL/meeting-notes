from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}


def load_whisper_model(model_size: str) -> Any:
    if model_size not in _MODEL_CACHE:
        import whisper

        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def transcribe_audio(audio_path: Path, model_size: str = "small", model: Any = None) -> list[dict]:
    if model is None:
        model = load_whisper_model(model_size)
    result = model.transcribe(str(audio_path), language="th")
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result["segments"]
    ]
