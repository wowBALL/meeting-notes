def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


DIARIZATION_FAILED_NOTE = (
    '> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ ข้อความทั้งหมดจึงแสดงเป็น "ผู้พูด 1" เพียงคนเดียว'
)


def render_transcript_markdown(
    merged_segments: list[dict], diarization_failed: bool = False
) -> str:
    lines = ["# Transcript"]
    if diarization_failed:
        lines.append(DIARIZATION_FAILED_NOTE)
    speaker_display: dict[str, str] = {}
    counter = 1
    for seg in merged_segments:
        speaker_key = seg["speaker"]
        if speaker_key not in speaker_display:
            speaker_display[speaker_key] = f"ผู้พูด {counter}"
            counter += 1
        timestamp = format_timestamp(seg["start"])
        lines.append(f"**{speaker_display[speaker_key]}** [{timestamp}]: {seg['text']}")
    return "\n\n".join(lines)
