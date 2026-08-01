import re

from src.chunk import SEGMENT_PATTERN, parse_transcript_segments


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


# ป้ายที่ build_speaker_labels ตั้งให้คนที่ยังไม่รู้ว่าเป็นใคร -- ทุกอย่างที่ไม่ตรงรูปนี้
# คือชื่อที่มาจากการจับลายเสียง ซึ่งเป็นหลักฐาน ไม่ใช่การเดา
_UNIDENTIFIED = re.compile(r"^ผู้พูด \d+$")

PARTICIPANTS_PREFIX = "ผู้เข้าร่วม:"

# เกินสัดส่วนนี้ของความยาวประชุมกว่าจะพูดประโยคแรก = ไม่ได้อยู่ตั้งแต่ต้น ให้เขียนกำกับไว้
#
# ไม่ใช่รายละเอียดจุกจิก: คนที่เข้ามากลางทางไม่ได้อยู่ตอนที่ครึ่งแรกตกลงอะไรกัน คนอ่าน
# สรุปที่เห็นชื่อเขาอยู่ในรายชื่อเฉย ๆ จะอ่าน "ตกลงแล้ว" ของครึ่งแรกผิดไปทั้งหมด
# (วัดจริง: ประชุม 164 นาที มีผู้พูดคนหนึ่งพูดประโยคแรกนาทีที่ 76 ทั้งที่สรุปทุกฉบับ
# ก่อนหน้าเขียนเหมือนเขาอยู่ตลอดทั้งประชุม)
LATE_ARRIVAL_SHARE = 0.25


def speaker_stats(transcript_markdown: str) -> list[dict]:
    """สถิติต่อป้ายผู้พูดหนึ่งป้าย เรียงจากคนที่พูดมากที่สุด

    ใช้ parser ตัวเดียวกับที่ขั้นสรุปใช้โดยเจตนา ตัวเลขที่ได้จึงเป็นตัวเลขของ transcript
    ชุดเดียวกับที่โมเดลเห็น ไม่ใช่ของไฟล์ที่ parse ด้วยกฎอีกชุดหนึ่ง

    วัดด้วย "จำนวนอักขระที่พูด" ไม่ใช่จำนวนบรรทัด เพราะความยาวบรรทัดขึ้นกับวิธีที่
    whisper ตัด segment (ดู DEFAULT_WHISPER_HOTWORDS ใน config.py -- การเปลี่ยนค่านั้น
    ทำให้บรรทัดยาวขึ้นเท่าตัวโดยที่ไม่มีใครพูดมากขึ้นเลย) อักขระไม่ขยับตามการตัด
    """
    stats: dict[str, dict] = {}
    previous: str | None = None
    for segment in parse_transcript_segments(transcript_markdown):
        match = SEGMENT_PATTERN.match(segment["raw"])
        if match is None:
            continue
        speaker = match.group("speaker").strip()
        start = segment["start_seconds"]
        entry = stats.setdefault(
            speaker,
            {
                "speaker": speaker,
                "lines": 0,
                "turns": 0,
                "chars": 0,
                "first_seconds": start,
                "last_seconds": start,
                "identified": not _UNIDENTIFIED.match(speaker),
            },
        )
        entry["lines"] += 1
        entry["chars"] += len(segment["raw"][match.end() :].strip())
        entry["first_seconds"] = min(entry["first_seconds"], start)
        entry["last_seconds"] = max(entry["last_seconds"], start)
        # ผลัดพูด: บล็อกติดกันของคนเดียวนับครั้งเดียว บอกว่าใครถูกดึงเข้าวงบ่อย
        # ซึ่งเป็นคนละคำถามกับใครพูดยาว
        if speaker != previous:
            entry["turns"] += 1
            previous = speaker
    total = sum(entry["chars"] for entry in stats.values())
    for entry in stats.values():
        entry["share"] = entry["chars"] / total if total else 0.0
    return sorted(stats.values(), key=lambda e: (-e["chars"], e["speaker"]))


def participants_line(transcript_markdown: str) -> str:
    """บรรทัด "ผู้เข้าร่วม:" ที่นับจาก transcript คืนสตริงว่างเมื่อไม่มี segment เลย

    เขียนด้วยโค้ดแทนที่จะให้โมเดลเขียน เพราะรายชื่อผู้พูดเป็นข้อเท็จจริงที่นับได้ ไม่ใช่
    สิ่งที่ต้องอนุมาน -- ขั้น reduce ไม่เห็น transcript อีกแล้ว (มันเห็นแค่สรุปรายช่วง
    ดู summarize.summarize_transcript) มันจึงตรวจสอบชื่อไม่ได้และรวมป้ายเดียวกันที่
    คนละ chunk เรียกคนละชื่อไม่ได้ ผลที่วัดได้จริงกับ Meet22: ชื่อ 15 ชื่อจากผู้พูด 9
    ป้าย โดย 5 ชื่อในนั้นไม่มีอยู่ใน transcript เลยแม้แต่ตัวอักษรเดียว
    """
    stats = speaker_stats(transcript_markdown)
    if not stats:
        return ""
    known = sum(1 for entry in stats if entry["identified"])
    started = min(entry["first_seconds"] for entry in stats)
    ended = max(entry["last_seconds"] for entry in stats)
    span = ended - started
    parts = []
    for entry in stats:
        part = f"{entry['speaker']} {entry['share'] * 100:.1f}%"
        if span and (entry["first_seconds"] - started) / span > LATE_ARRIVAL_SHARE:
            part += f" (พูดครั้งแรก {format_timestamp(entry['first_seconds'])})"
        parts.append(part)
    return (
        f"{PARTICIPANTS_PREFIX} {len(stats)} เสียง "
        f"(ระบุตัวได้จากลายเสียง {known} คน) — " + " · ".join(parts)
    )


def speaker_table(transcript_markdown: str) -> str:
    """ตารางเต็มสำหรับ summary.meta.md -- ตัวเลขพวกนี้ใช้ตัดสินคุณภาพการแยกเสียง
    (ป้ายที่มีไม่กี่บรรทัด หรือบรรทัดยาวผิดปกติจนน่าจะกินเสียงคนอื่นเข้ามา) ซึ่งเป็น
    เรื่องของระบบ ไม่ใช่เรื่องที่คนในห้องคุยกัน จึงไม่ควรอยู่ใน summary.md"""
    stats = speaker_stats(transcript_markdown)
    if not stats:
        return ""
    rows = [
        "## ผู้พูดใน transcript",
        "",
        "| ป้าย | บรรทัด | ผลัด | อักขระ | สัดส่วน | ครั้งแรก | ครั้งสุดท้าย |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in stats:
        rows.append(
            f"| {entry['speaker']} | {entry['lines']:,} | {entry['turns']:,} | "
            f"{entry['chars']:,} | {entry['share'] * 100:.1f}% | "
            f"{format_timestamp(entry['first_seconds'])} | "
            f"{format_timestamp(entry['last_seconds'])} |"
        )
    return "\n".join(rows)


def replace_participants_line(summary_markdown: str, transcript_markdown: str) -> str:
    """เอาบรรทัดผู้เข้าร่วมที่นับเองไปแทนของที่โมเดลเขียน

    ยังลบของโมเดลอยู่แม้จะถอดบรรทัดนั้นออกจาก prompt แล้ว: prompt เป็นคำขอ ไม่ใช่
    การบังคับ และประชุมเก่าที่ถูกสรุปด้วย prompt รุ่นก่อนหน้านี้ก็มีบรรทัดนั้นอยู่
    ปล่อยไว้จะได้สองบรรทัดที่ขัดกันเองในไฟล์เดียว
    """
    line = participants_line(transcript_markdown)
    kept = [
        raw
        for raw in summary_markdown.splitlines()
        if not raw.strip().startswith(PARTICIPANTS_PREFIX)
    ]
    body = "\n".join(kept).strip()
    if not line:
        return body
    return f"{line}\n\n{body}" if body else line
