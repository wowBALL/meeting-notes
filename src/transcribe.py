from pathlib import Path

from openai import OpenAI

# OpenAI Whisper rejects uploads over 25MB. Stay under that with headroom.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def transcribe_audio(audio_path: Path, api_key: str | None = None) -> list[dict]:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    file_size = audio_path.stat().st_size
    if file_size <= MAX_UPLOAD_BYTES:
        return _transcribe_file(client, audio_path)
    return _transcribe_in_chunks(client, audio_path, file_size)


def _transcribe_file(client: OpenAI, audio_path: Path) -> list[dict]:
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


def _transcribe_in_chunks(
    client: OpenAI, audio_path: Path, file_size: int
) -> list[dict]:
    # Lazy import so the common (small-file) path never pays the pydub/ffmpeg cost.
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    duration_ms = len(audio)
    # Size chunks by the file's actual average bitrate so each stays under the
    # byte threshold regardless of encoding.
    bytes_per_ms = file_size / duration_ms
    chunk_ms = max(1, int(MAX_UPLOAD_BYTES / bytes_per_ms))
    export_format = audio_path.suffix.lstrip(".") or "mp3"

    segments: list[dict] = []
    chunk_index = 0
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk_path = audio_path.with_suffix(audio_path.suffix + f".chunk{chunk_index}")
        try:
            audio[start_ms:end_ms].export(str(chunk_path), format=export_format)
            offset_seconds = start_ms / 1000.0
            for seg in _transcribe_file(client, chunk_path):
                segments.append(
                    {
                        "start": seg["start"] + offset_seconds,
                        "end": seg["end"] + offset_seconds,
                        "text": seg["text"],
                    }
                )
        finally:
            chunk_path.unlink(missing_ok=True)
        chunk_index += 1
        start_ms = end_ms
    return segments
