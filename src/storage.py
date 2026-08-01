import re
import shutil
import time
from datetime import date, datetime, timedelta
from itertools import count
from pathlib import Path

from src.job import move_job

# Windows จับไฟล์ที่เพิ่งเขียนเสร็จค้างได้ราวหนึ่งวินาที (ตัวสแกนไวรัส/indexer)
# วัดมาแล้วในโปรเจกต์นี้กับการลบไฟล์ .wav -- การเขียนครั้งเดียวแล้วยอมแพ้แปลว่า
# ผู้ใช้เสียชื่อที่เพิ่งตั้งไปเฉย ๆ
_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY_SECONDS = 0.2


def replace_with_retry(temp: Path, target: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            temp.replace(target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY_SECONDS)


# Stems produced by record.build_output_filename:
#   named:   "<topic>-HH-MM-SS"
#   unnamed: "YYYY-MM-DD_HH-MM-SS"
# A ':' is illegal in a Windows path, so the requested HH:MM separator between
# hour and minute is written as HH-MM.
_UNNAMED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-\d{2}$")
_NAMED_RE = re.compile(r"^(?P<topic>.+)-(\d{2})-(\d{2})-\d{2}$")


def recording_day(stem: str, finished_at: datetime) -> date:
    """วันที่ประชุมนี้ "ถูกอัด" ไม่ใช่วันที่มันถูกประมวลผล

    stem ของประชุมที่ตั้งชื่อมีแค่ HH-MM-SS ไม่มีวันที่ (ดูรูปแบบด้านบน) ของเดิมจึงเติม
    วันที่ของวันที่รันเข้าไป ซึ่งตรงก็ต่อเมื่อถอดเสียงวันเดียวกับที่อัด -- ไฟล์ที่ค้างข้ามคืน
    ได้โฟลเดอร์ที่ระบุวันผิดไปเลย (Meet22 อัด 2026-07-31 ได้ชื่อ 2026-08-01)

    `finished_at` คือตอนที่ไฟล์เสียงถูกเขียนเสร็จ ซึ่งก็คือตอนที่ประชุม *จบ* -- การอัดจบ
    ก่อนเริ่มไม่ได้ ดังนั้นถ้า HH-MM ในชื่อไฟล์ตกหลังเวลานั้น แปลว่ามันเป็นของเมื่อวาน
    (ประชุมที่คร่อมเที่ยงคืน) กติกาข้อนี้ทำให้ทั้งสองทิศทางถูก ไม่ใช่แค่ทิศที่เพิ่งเจอ
    """
    unnamed = _UNNAMED_RE.match(stem)
    if unnamed:
        # stem รู้วันของตัวเองอยู่แล้ว ไม่ต้องเดาจากไฟล์
        return date.fromisoformat(unnamed.group(1))
    named = _NAMED_RE.match(stem)
    if not named:
        # ไฟล์ที่ผู้ใช้หย่อนเข้ามาเอง ไม่มี HH-MM ให้เทียบ เหลือแค่วันของไฟล์
        return finished_at.date()
    started = finished_at.replace(
        hour=int(named.group(2)), minute=int(named.group(3)), second=0, microsecond=0
    )
    if started > finished_at:
        started -= timedelta(days=1)
    return started.date()


def meeting_folder_name(stem: str, today: date) -> str:
    """Build 'YYYY-MM-DD_HH-MM-<topic>' from a recording's file stem."""
    unnamed = _UNNAMED_RE.match(stem)
    if unnamed:
        day, hh, mm = unnamed.groups()
        return f"{day}_{hh}-{mm}"
    named = _NAMED_RE.match(stem)
    if named:
        return f"{today.isoformat()}_{named.group(2)}-{named.group(3)}-{named.group('topic')}"
    # A file the user dropped into inbox/ themselves carries no recorder
    # timestamp to parse, so keep the whole name and just date-stamp it.
    return f"{today.isoformat()}_{stem}"


def create_meeting_folder(
    audio_path: Path, meetings_dir: Path, today: date | None = None
) -> Path:
    """Make a folder of this recording's own, never one another recording owns.

    The name carries no seconds, so two recordings of the same meeting started
    within one minute ask for the same folder -- and the second one's
    transcript.md would land on top of the first one's. Adding the seconds back
    is not the fix: the COWORK Desktop widget parses "<date>_HH-MM-<topic>" and
    would read a third number as the start of the meeting title. Only the actual
    collision gets a suffix, which that parser still reads as a title ("test-2").
    """
    today = today or date.today()
    # วันของไฟล์เสียงเอง ไม่ใช่วันที่รัน -- `today` เหลือไว้เป็นทางถอยเมื่อ stat ไม่ได้
    # (ไฟล์หายไประหว่างทาง/สิทธิ์ไม่พอ) ซึ่งเสียแค่ความแม่นของวันที่ ไม่ควรล้มทั้งงาน
    try:
        finished_at = datetime.fromtimestamp(audio_path.stat().st_mtime)
    except OSError:
        day = today
    else:
        day = recording_day(audio_path.stem, finished_at)
    base = meeting_folder_name(audio_path.stem, day)
    for attempt in count(1):
        candidate = meetings_dir / (base if attempt == 1 else f"{base}-{attempt}")
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            continue
        return candidate
    raise AssertionError("unreachable")  # pragma: no cover


# Saved separately because the pipeline writes them at different moments: the
# transcript goes to disk before summarizing, so a summarization failure can
# never discard a transcript that cost a full GPU pass over the recording.
def save_transcript(meeting_dir: Path, transcript_markdown: str) -> Path:
    path = meeting_dir / "transcript.md"
    path.write_text(transcript_markdown, encoding="utf-8")
    return path


def _busiest_first(counts: dict[str, int]) -> list[tuple[str, int]]:
    """คำที่เจอบ่อยสุดขึ้นก่อน ชื่อเรียงตัวอักษรเมื่อจำนวนเท่ากัน (ลำดับต้องนิ่ง
    ไม่ใช่ลำดับของ dict ไม่งั้นเทียบท้ายไฟล์ข้ามประชุมไม่ได้)"""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


# หัวข้อที่โมเดลเขียนไว้ใน summary แต่ไม่ใช่เนื้อหาการประชุม -- มันคือรายงานคุณภาพของ
# การถอดเสียง (คำไหนน่าจะเพี้ยน ช่วงไหนควรย้อนไปฟังเอง) ซึ่งเป็นเรื่องของระบบ ไม่ใช่
# เรื่องที่คนในห้องคุยกัน คนที่เปิด summary.md เพื่อส่งต่อให้หัวหน้าไม่ควรต้องเลื่อนผ่าน
# สองหัวข้อนี้ทุกครั้ง -- เหตุผลเดียวกับที่ footer ถูกแยกออกมาก่อนหน้านี้
#
# ยังให้โมเดลเขียนต่อไปเหมือนเดิม (prompt ไม่เปลี่ยน) แค่ย้ายปลายทางตอนบันทึก:
# ตัดขั้นตอนที่โมเดลเขียนออกไปเลยแปลว่าไม่มีใครรู้ว่าตรงไหนฟังไม่ชัด ซึ่งแย่กว่ามาก
#
# ต้องตรงกับหัวข้อใน prompts/single.md และ prompts/reduce.md เป๊ะ ๆ (ยึด prefix
# เพราะอันแรกมีวงเล็บต่อท้ายที่โมเดลเขียนไม่เหมือนกันทุกครั้ง)
TRANSCRIPT_QUALITY_HEADINGS = (
    "## คำที่น่าจะถอดเพี้ยน",
    "## จุดที่ควรตรวจเอง",
)


def _split_out_quality_sections(summary_markdown: str) -> tuple[str, list[str]]:
    """(สรุปที่เหลือ, หัวข้อคุณภาพที่ถูกดึงออกมา)

    เดินทีละบรรทัดแทน regex ด้วยเหตุผลเดียวกับ carryover._section_body: summary.md
    บน Windows เป็น CRLF และการยึด `$` ทำให้ match พลาดแบบเงียบ ๆ

    หัวข้อที่ไม่มีอยู่ = ข้ามไปเฉย ๆ ไม่ใช่ error: โมเดลอาจไม่เขียนมาให้ครบทุกครั้ง
    และประชุมที่ถูกสรุปด้วย prompt รุ่นเก่ากว่านี้ก็ไม่มีหัวข้อพวกนี้เลย
    """
    lines = summary_markdown.splitlines()
    kept: list[str] = []
    pulled: list[str] = []
    current: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            # ปิดหัวข้อคุณภาพที่กำลังเก็บอยู่ (ถ้ามี) ก่อนตัดสินใจเรื่องหัวข้อใหม่
            current = None
            if any(stripped.startswith(h) for h in TRANSCRIPT_QUALITY_HEADINGS):
                current = []
                pulled.append(current)
        if current is not None:
            current.append(line)
        else:
            kept.append(line)
    blocks = ["\n".join(block).strip() for block in pulled]
    return "\n".join(kept).strip(), [b for b in blocks if b]


def save_summary(
    meeting_dir: Path,
    summary_markdown: str,
    model: str,
    glossary_counts: dict[str, int] | None = None,
    fuzzy_seen: dict[str, int] | None = None,
    profile: str | None = None,
) -> Path:
    # `model` is required, not optional: the point of choosing a model per meeting
    # is being able to judge afterwards whether the pricier one was worth it, and
    # summary.meta.md with no attribution cannot be judged at all.
    #
    # อีกสองค่าเป็น optional และจะไม่เขียนบรรทัดอะไรเลยเมื่อว่าง -- คนที่ยังไม่มี
    # glossary.md ต้องได้ไฟล์หน้าตาเดิมเป๊ะ ไม่ใช่บรรทัดเปล่าที่อ่านไม่ได้ความ
    #
    # เมตาดาต้า (โมเดล, ประเภทประชุม, คำที่แก้ตาม glossary) แยกไปอยู่ summary.meta.md
    # คนละไฟล์กับ summary.md โดยเจตนา: summary.md คือของที่ส่งต่อให้คนอ่าน (หัวหน้า,
    # ทีม) ไม่ควรมีรายละเอียดเชิงเทคนิคของระบบปนอยู่ท้ายไฟล์ -- ดู carryover._profile_of
    # ที่อ่านค่า "ประเภทประชุม:" จากไฟล์นี้ (ตกกลับไปอ่านจาก summary.md เองถ้าเป็น
    # ประชุมเก่าก่อนมีไฟล์นี้)
    path = meeting_dir / "summary.md"
    body, quality_sections = _split_out_quality_sections(summary_markdown)
    path.write_text(f"{body}\n", encoding="utf-8")

    footer = [f"สรุปด้วย {model}"]
    if profile:
        # ต้องเห็นย้อนหลังได้ว่าสรุปนี้ใช้กฎชุดไหน -- เผลอกด dev ในประชุมข้ามฝ่ายแล้ว
        # สรุปจะดูปกติทุกอย่าง ยกเว้นว่ามันไม่ได้แยก "ทำได้" ออกจาก "จะทำ" ให้
        footer.append(f"ประเภทประชุม: {profile}")
    if glossary_counts:
        # "แก้ไปแล้ว" -- คำที่โค้ดแทนที่จริง รายคำเพื่อให้เห็นว่าคำไหนแทนที่ผิดที่
        # (จำนวนเฟ้อผิดปกติ) จนควรย้ายจาก exact ไป fuzzy
        corrected = ", ".join(
            f"{term} {count} จุด" for term, count in _busiest_first(glossary_counts)
        )
        footer.append(f"แก้คำตาม glossary: {corrected}")
    if fuzzy_seen:
        # "เจอ แต่ไม่ได้แก้" -- ชั้น fuzzy โมเดลเป็นคนตีความ บรรทัดนี้เป็นหลักฐาน
        # ชิ้นเดียวที่บอกได้ว่าคำใน fuzzy คำไหนตายแล้วควรลบ ต้องแยกจากบรรทัดบนเพราะ
        # ความหมายต่างกัน
        seen = ", ".join(
            f"{term} {count} ครั้ง" for term, count in _busiest_first(fuzzy_seen)
        )
        footer.append(f"คำ fuzzy ที่เจอในห้อง: {seen}")
    meta_path = meeting_dir / "summary.meta.md"
    meta = "\n".join(footer)
    if quality_sections:
        # บรรทัดสรุปสั้น ๆ อยู่บน หัวข้อยาวอยู่ล่าง คนเปิดไฟล์นี้ส่วนใหญ่มาดูว่าใช้โมเดล
        # อะไร ไม่ใช่มาอ่านรายงานคุณภาพ -- สิ่งที่ถูกถามบ่อยกว่าควรอยู่บรรทัดแรก
        meta += "\n\n" + "\n\n".join(quality_sections)
    meta_path.write_text(meta + "\n", encoding="utf-8")

    return path


def archive_audio(meeting_dir: Path, audio_path: Path) -> Path:
    destination = meeting_dir / audio_path.name
    shutil.move(str(audio_path), str(destination))
    return destination


def move_to_failed(audio_path: Path, failed_dir: Path, error_message: str) -> Path:
    failed_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_dir / audio_path.name
    shutil.move(str(audio_path), str(destination))
    error_log = failed_dir / f"{audio_path.stem}.error.log"
    error_log.write_text(error_message, encoding="utf-8")
    # The job file follows the recording so a later retry summarizes with the
    # model the user actually picked. Handled here rather than at each of
    # process_file's six failure branches, where a seventh would eventually
    # forget it.
    move_job(audio_path, failed_dir)
    return destination


def safe_meeting_dir(meetings_dir: Path, name: str) -> Path | None:
    """โฟลเดอร์การประชุมที่ชื่อมาจากฝั่ง client -- None เมื่อชื่อพาออกนอก meetings/

    การต่อสตริงตรง ๆ ไม่พอ: ".." หรือ path สัมบูรณ์พาไปอ่านไฟล์ที่ไหนก็ได้ในเครื่อง
    service bind 127.0.0.1 อยู่แล้ว แต่หน้าเว็บใด ๆ ที่ผู้ใช้เปิดอยู่ยิงมาที่นี่ได้
    """
    if not name:
        return None
    root = Path(meetings_dir).resolve()
    candidate = (root / name).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def rename_speaker_in_transcript(
    meeting_dir: Path, old_label: str, new_name: str
) -> bool:
    """แทนที่ป้ายผู้พูดใน transcript.md เฉพาะที่หัวบรรทัด คืน True เมื่อแก้จริง

    ยึดหัวบรรทัดแทนการ replace ทั้งไฟล์ เพราะสิ่งที่คนพูดอาจมีสตริง "ผู้พูด 2" อยู่
    กลางประโยคได้ตามปกติ

    ห้ามใช้ `$` เป็น anchor กับไฟล์ของโปรเจกต์นี้: ไฟล์ .md บนดิสก์เป็น CRLF และ
    `$` จะไปชนกับ \\r ที่มองไม่เห็นแล้วไม่ match อะไรเลยแบบเงียบ ๆ -- `^` ปลอดภัย
    เพราะ read_text แปลง newline กลับเป็น \\n ให้ก่อนแล้ว

    summary.md ไม่ถูกแตะโดยเจตนา: ข้อความในนั้นถูกโมเดลเรียบเรียงใหม่แล้ว การแทนที่
    สตริงจะได้ประโยคที่อ่านไม่รู้เรื่อง และการประชุมครั้งถัดไปก็สรุปด้วยชื่อจริงเองอยู่แล้ว
    """
    path = Path(meeting_dir) / "transcript.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    pattern = re.compile(rf"^\*\*{re.escape(old_label)}\*\* \[", re.MULTILINE)
    # ต้องแทนที่ด้วย "ฟังก์ชัน" ไม่ใช่สตริง: อาร์กิวเมนต์ตัวที่สองของ subn ถูกตีความเป็น
    # template ที่มี \1, \g<0>, \0 เป็นความหมายพิเศษ ชื่อที่ผู้ใช้พิมพ์เดินทางมาจาก
    # HTTP request และ clean_name ไม่ได้กรอง backslash ออก -- ชื่ออย่าง "\1" จะทำให้
    # ฟังก์ชันนี้ raise ทั้งที่สัญญาว่าคืน bool ส่วน "\g<0>" จะยัดข้อความที่ match ได้
    # กลับเข้าไฟล์แทนชื่อ ทำให้ transcript เสียรูปแบบเงียบ ๆ ฟังก์ชันไม่ตีความอะไรเลย
    updated, replaced = pattern.subn(lambda _match: f"**{new_name}** [", text)
    if not replaced:
        return False
    # เขียนผ่านไฟล์ชั่วคราวแล้วค่อยสลับ: write_text เปิดโหมด "w" ซึ่งตัดไฟล์ทิ้งก่อน
    # เขียน ถ้าล้มกลางทาง transcript ที่จ่ายไปด้วย GPU หนึ่งรอบเต็มจะหายไปเลย ทั้งที่
    # ผู้เรียกได้ False กลับไปแล้วแปลว่า "ไม่มีอะไรเปลี่ยน"
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(updated, encoding="utf-8")
        replace_with_retry(temp, path)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        return False
    return True
