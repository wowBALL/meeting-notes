def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


DIARIZATION_FAILED_NOTE = (
    '> ⚠️ ไม่สามารถแยกผู้พูดได้อัตโนมัติ ข้อความทั้งหมดจึงแสดงเป็น "ผู้พูด 1" เพียงคนเดียว'
)


def build_speaker_labels(
    merged_segments: list[dict], speaker_names: dict[str, str] | None = None
) -> dict[str, str]:
    """ป้ายที่จะถูกเขียนลงไฟล์ ต่อผู้พูดหนึ่งคน

    แยกออกมาจาก render เพราะขั้นตอนยืนยันชื่อทีหลังต้องรู้ว่าไฟล์ถูกเขียนด้วยป้ายอะไร
    ไปบ้าง จึงจะแทนที่สตริงใน transcript.md ได้ถูก -- กฎการตั้งป้ายจึงต้องอยู่ที่เดียว
    ไม่ใช่สองสูตรที่ต้องเหมือนกันเอง

    ตัวนับนับเฉพาะคนที่ยังไม่มีชื่อ: คนที่รู้จักแล้วกินเลขไปเปล่า ๆ จะทำให้ผู้อ่านเห็น
    "ผู้พูด 2" ทั้งที่มีคนนิรนามอยู่คนเดียวในห้อง
    """
    names = speaker_names or {}
    labels: dict[str, str] = {}
    counter = 1
    for segment in merged_segments:
        key = segment["speaker"]
        if key in labels:
            continue
        name = names.get(key)
        if name:
            labels[key] = name
        else:
            labels[key] = f"ผู้พูด {counter}"
            counter += 1
    return labels


def render_transcript_markdown(
    merged_segments: list[dict],
    diarization_failed: bool = False,
    speaker_names: dict[str, str] | None = None,
) -> str:
    lines = ["# Transcript"]
    if diarization_failed:
        lines.append(DIARIZATION_FAILED_NOTE)
    labels = build_speaker_labels(merged_segments, speaker_names)
    for seg in merged_segments:
        timestamp = format_timestamp(seg["start"])
        lines.append(f"**{labels[seg['speaker']]}** [{timestamp}]: {seg['text']}")
    return "\n\n".join(lines)
