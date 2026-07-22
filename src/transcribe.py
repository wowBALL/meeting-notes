from pathlib import Path

from openai import OpenAI


def transcribe_audio(audio_path: Path, api_key: str | None = None) -> list[dict]:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="th",
            response_format="verbose_json",
        )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text} for seg in response.segments
    ]
