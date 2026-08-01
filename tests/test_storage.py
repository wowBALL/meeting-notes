import os
from datetime import date, datetime
from unittest.mock import patch

from src.job import JOB_SUFFIX, read_model, write_job
from src.storage import (
    archive_audio,
    create_meeting_folder,
    meeting_folder_name,
    move_to_failed,
    recording_day,
    save_summary,
    save_transcript,
)

TODAY = date(2026, 7, 22)


def test_folder_name_for_a_named_recording_is_date_time_topic():
    # recorder stem: "<topic>-HH-MM-SS" -> "YYYY-MM-DD_HH-MM-<topic>"
    # (':' is illegal in a Windows path, so HH:MM is written HH-MM)
    assert meeting_folder_name("Meet1900-19-01-45", TODAY) == "2026-07-22_19-01-Meet1900"


def test_folder_name_keeps_a_topic_that_contains_dashes():
    assert (
        meeting_folder_name("Q3-2026-Review-09-30-00", TODAY)
        == "2026-07-22_09-30-Q3-2026-Review"
    )


def test_folder_name_for_an_unnamed_recording_has_no_topic():
    # unnamed recorder stem: "YYYY-MM-DD_HH-MM-SS" -> use its own date, drop the topic
    assert meeting_folder_name("2026-07-24_19-01-45", TODAY) == "2026-07-24_19-01"


def test_folder_name_for_a_user_dropped_file_keeps_the_whole_name():
    # no recorder timestamp to parse; just date-stamp whatever was dropped in
    assert meeting_folder_name("weekly-standup", TODAY) == "2026-07-22_weekly-standup"


def test_recording_day_is_the_day_it_was_recorded_not_the_day_it_was_processed():
    # Meet22 was recorded on 07-31 but only reached the watcher on 08-01, and the
    # folder claimed 08-01. A named stem carries HH-MM-SS and no date, so the day
    # has to come from the file itself, not from whenever the run happens to be.
    finished_at = datetime(2026, 7, 31, 22, 46, 0)

    assert recording_day("Meet22-19-59-59", finished_at) == date(2026, 7, 31)


def test_recording_day_rolls_back_a_day_when_the_meeting_crossed_midnight():
    # started 23:30, ffmpeg finished writing at 00:15 the next day. Taking the
    # file's own day would date the folder a day AFTER the meeting -- the same
    # class of error, just in the other direction. A recording cannot end before
    # it starts, so an HH-MM later than the write time belongs to the day before.
    finished_at = datetime(2026, 8, 1, 0, 15, 0)

    assert recording_day("Retro-23-30-00", finished_at) == date(2026, 7, 31)


def test_recording_day_prefers_the_date_an_unnamed_stem_already_carries():
    # an unnamed stem is "YYYY-MM-DD_HH-MM-SS" -- it knows its own day, so no
    # guessing from the file is needed or wanted
    finished_at = datetime(2026, 8, 1, 9, 0, 0)

    assert recording_day("2026-07-24_19-01-45", finished_at) == date(2026, 7, 24)


def test_create_meeting_folder_dates_the_folder_from_the_audio_not_the_clock(tmp_path):
    # the Meet22 bug end to end: recorded 07-31, processed 08-01, folder said 08-01
    meetings_dir = tmp_path / "meetings"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    audio_path = inbox / "Meet22-19-59-59.ogg"
    audio_path.write_bytes(b"")
    finished_at = datetime(2026, 7, 31, 22, 46, 0).timestamp()
    os.utime(audio_path, (finished_at, finished_at))

    result = create_meeting_folder(audio_path, meetings_dir, today=date(2026, 8, 1))

    assert result == meetings_dir / "2026-07-31_19-59-Meet22"


def test_create_meeting_folder_builds_the_new_format_and_makes_the_dir(tmp_path):
    meetings_dir = tmp_path / "meetings"
    audio_path = tmp_path / "inbox" / "Meet1900-19-01-45.ogg"

    result = create_meeting_folder(audio_path, meetings_dir, today=TODAY)

    assert result == meetings_dir / "2026-07-22_19-01-Meet1900"
    assert result.is_dir()


def test_create_meeting_folder_disambiguates_a_second_recording_in_the_same_minute(tmp_path):
    # the folder name drops the seconds, so two recordings of the same meeting
    # started within one minute ask for the same folder
    meetings_dir = tmp_path / "meetings"
    inbox = tmp_path / "inbox"

    first = create_meeting_folder(inbox / "test-12-07-11.ogg", meetings_dir, today=TODAY)
    second = create_meeting_folder(inbox / "test-12-07-44.ogg", meetings_dir, today=TODAY)

    assert first == meetings_dir / "2026-07-22_12-07-test"
    assert second == meetings_dir / "2026-07-22_12-07-test-2"
    assert second.is_dir()


def test_create_meeting_folder_keeps_counting_up_past_the_second_collision(tmp_path):
    meetings_dir = tmp_path / "meetings"
    inbox = tmp_path / "inbox"
    create_meeting_folder(inbox / "test-12-07-11.ogg", meetings_dir, today=TODAY)
    create_meeting_folder(inbox / "test-12-07-44.ogg", meetings_dir, today=TODAY)

    third = create_meeting_folder(inbox / "test-12-07-52.ogg", meetings_dir, today=TODAY)

    assert third == meetings_dir / "2026-07-22_12-07-test-3"


def test_create_meeting_folder_disambiguates_an_unnamed_recording_too(tmp_path):
    meetings_dir = tmp_path / "meetings"
    inbox = tmp_path / "inbox"
    create_meeting_folder(inbox / "2026-07-24_19-01-45.ogg", meetings_dir, today=TODAY)

    second = create_meeting_folder(inbox / "2026-07-24_19-01-58.ogg", meetings_dir, today=TODAY)

    assert second == meetings_dir / "2026-07-24_19-01-2"


def test_a_second_recording_never_overwrites_the_first_ones_transcript(tmp_path):
    # the bug this guards: both recordings landed in one folder and the second
    # transcript.md replaced the first, while both .ogg files survived
    meetings_dir = tmp_path / "meetings"
    inbox = tmp_path / "inbox"
    first = create_meeting_folder(inbox / "test-12-07-11.ogg", meetings_dir, today=TODAY)
    save_transcript(first, "# First recording")

    second = create_meeting_folder(inbox / "test-12-07-44.ogg", meetings_dir, today=TODAY)
    save_transcript(second, "# Second recording")

    assert (first / "transcript.md").read_text(encoding="utf-8") == "# First recording"
    assert (second / "transcript.md").read_text(encoding="utf-8") == "# Second recording"


def test_save_transcript_writes_the_transcript_file(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_transcript(meeting_dir, "# Transcript")

    assert path == meeting_dir / "transcript.md"
    assert path.read_text(encoding="utf-8") == "# Transcript"


def test_save_summary_writes_the_summary_file_without_metadata(tmp_path):
    """summary.md คือของที่ส่งต่อให้คนอ่าน -- ต้องไม่มีอะไรของระบบปนอยู่ท้ายไฟล์"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, "# Summary", "claude-sonnet-5")

    assert path == meeting_dir / "summary.md"
    assert path.read_text(encoding="utf-8") == "# Summary\n"


def test_save_summary_does_not_stack_blank_lines(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, "# Summary\n\n\n", "claude-opus-5")

    assert path.read_text(encoding="utf-8") == "# Summary\n"


def test_save_summary_writes_the_model_to_a_separate_meta_file(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(meeting_dir, "# Summary", "claude-sonnet-5")

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert meta == "สรุปด้วย claude-sonnet-5\n"


_WITH_QUALITY = """## หัวข้อที่คุยกัน
- เรื่องที่คุยกัน

## ต้องคุยต่อครั้งหน้า
- เรื่องค้าง

## คำที่น่าจะถอดเพี้ยน (ยังไม่อยู่ใน glossary)
- Payload → เดาว่าคือ Payroll (ได้ยิน 6 ครั้ง)

## จุดที่ควรตรวจเอง
- 08:00-16:20: เสียงทับกัน ควรฟังเอง
"""


def test_the_transcript_quality_sections_leave_the_summary(tmp_path):
    """สองหัวข้อนี้เป็นรายงานคุณภาพการถอดเสียง ไม่ใช่เรื่องที่คนในห้องคุยกัน
    คนที่เปิด summary.md เพื่อส่งต่อให้หัวหน้าไม่ควรต้องเลื่อนผ่านมันทุกครั้ง"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, _WITH_QUALITY, "GLM-5.2")

    summary = path.read_text(encoding="utf-8")
    assert "## คำที่น่าจะถอดเพี้ยน" not in summary
    assert "## จุดที่ควรตรวจเอง" not in summary
    assert "Payload" not in summary
    assert "เสียงทับกัน" not in summary
    # เนื้อหาการประชุมต้องอยู่ครบเหมือนเดิม
    assert "## หัวข้อที่คุยกัน" in summary
    assert "- เรื่องที่คุยกัน" in summary


def test_the_transcript_quality_sections_land_in_the_meta_file(tmp_path):
    """ย้ายที่เก็บ ไม่ใช่ทิ้ง -- ถ้าหายไปเลยจะไม่มีใครรู้ว่าตรงไหนฟังไม่ชัด"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(meeting_dir, _WITH_QUALITY, "GLM-5.2")

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert meta.startswith("สรุปด้วย GLM-5.2")
    assert "## คำที่น่าจะถอดเพี้ยน (ยังไม่อยู่ใน glossary)" in meta
    assert "- Payload → เดาว่าคือ Payroll (ได้ยิน 6 ครั้ง)" in meta
    assert "## จุดที่ควรตรวจเอง" in meta
    assert "- 08:00-16:20: เสียงทับกัน ควรฟังเอง" in meta


def test_carryover_still_finds_its_section_after_the_split(tmp_path):
    """"ต้องคุยต่อครั้งหน้า" ต้องอยู่ใน summary.md ต่อไป -- carryover อ่านจากไฟล์นั้น
    ถ้าเผลอย้ายไปด้วย ความต่อเนื่องข้ามประชุมจะขาดเงียบ ๆ"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, _WITH_QUALITY, "GLM-5.2", profile="dev")

    summary = path.read_text(encoding="utf-8")
    assert "## ต้องคุยต่อครั้งหน้า" in summary
    assert "- เรื่องค้าง" in summary


_WITH_TIMELINE_TAGS = """## หัวข้อที่คุยกัน
- เรื่องที่คุยกัน

## ไทม์ไลน์ตามช่วง

### [00:00–20:00]

[หัวข้อ] อัปเดตงาน Payroll
[ตกลงแล้ว] ใช้ Final Payslip เป็น source of truth — สรุปโดย: สอง
[คำเพี้ยน?] PlayLight → เดาว่าคือ Playwright
[ไม่มั่นใจ] ช่วง 12:30 เสียงทับกัน

### [20:00–40:00]

[หัวข้อ] เรื่อง BMAD
[คำเพี้ยน?] Bmat → เดาว่าคือ BMAD
"""


def test_the_map_stage_quality_tags_leave_the_summary(tmp_path):
    """หัวข้อไทม์ไลน์วางสรุปรายช่วงตามที่โมเดลเขียนทุกตัวอักษร ป้ายคุณภาพจึงหลุดมาถึง
    summary.md ทางนี้ ทั้งที่ฝั่งที่ถูกยุบรวมแล้วถูกย้ายออกไปตั้งแต่ก่อนหน้านี้"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, _WITH_TIMELINE_TAGS, "GLM-5.2")

    summary = path.read_text(encoding="utf-8")
    assert "[คำเพี้ยน?]" not in summary
    assert "[ไม่มั่นใจ]" not in summary
    # ป้ายที่เป็นเนื้อหาประชุมต้องอยู่ครบ ไม่ใช่โดนกวาดไปด้วย
    assert "[หัวข้อ] อัปเดตงาน Payroll" in summary
    assert "[ตกลงแล้ว] ใช้ Final Payslip เป็น source of truth — สรุปโดย: สอง" in summary
    # หัวข้อช่วงต้องยังอยู่ในไทม์ไลน์ แม้บรรทัดใต้มันจะถูกดึงออกไปบางส่วน
    assert "### [00:00–20:00]" in summary
    assert "### [20:00–40:00]" in summary


def test_the_map_stage_quality_tags_keep_their_time_range_in_the_meta_file(tmp_path):
    """คำที่ถอดเพี้ยนมีค่าตอนย้อนกลับไปฟัง -- ดึงมากองรวมกันโดยทิ้งช่วงเวลาไป แปลว่า
    ต้องไล่หาเองว่าคำนั้นอยู่ตรงไหนของไฟล์เสียงยาวสองชั่วโมงครึ่ง"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(meeting_dir, _WITH_TIMELINE_TAGS, "GLM-5.2")

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "## ป้ายคุณภาพที่ค้างอยู่ในไทม์ไลน์รายช่วง" in meta
    assert "[คำเพี้ยน?] PlayLight → เดาว่าคือ Playwright" in meta
    assert "[ไม่มั่นใจ] ช่วง 12:30 เสียงทับกัน" in meta
    assert "[คำเพี้ยน?] Bmat → เดาว่าคือ BMAD" in meta
    # แต่ละคำต้องอยู่ใต้ช่วงของตัวเอง ไม่ใช่กองรวมกันหมด
    first = meta.index("### [00:00–20:00]")
    second = meta.index("### [20:00–40:00]")
    assert first < meta.index("PlayLight") < second < meta.index("Bmat")


def test_a_summary_without_quality_sections_is_unchanged(tmp_path):
    """สรุปจาก prompt รุ่นเก่าที่ไม่มีสองหัวข้อนี้ ต้องได้ไฟล์หน้าตาเดิมเป๊ะ"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    path = save_summary(meeting_dir, "## ประเด็นสำคัญ\n- ทดสอบ", "GLM-5.2")

    assert path.read_text(encoding="utf-8") == "## ประเด็นสำคัญ\n- ทดสอบ\n"
    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert meta == "สรุปด้วย GLM-5.2\n"


def test_save_summary_records_glossary_corrections_per_term(tmp_path):
    """รายคำ ไม่ใช่ยอดรวม -- ยอดรวมบอกไม่ได้ว่าคำไหนแทนที่ผิดที่จนควรย้ายไป fuzzy"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(
        meeting_dir,
        "# Summary",
        "GLM-5.2",
        glossary_counts={"PostgreSQL": 3, "Railway": 1},
    )

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "แก้คำตาม glossary: PostgreSQL 3 จุด, Railway 1 จุด" in meta


def test_save_summary_keeps_fuzzy_sightings_on_their_own_line(tmp_path):
    """คนละความหมายกับบรรทัดบน: บรรทัดนั้น "แก้ไปแล้ว" บรรทัดนี้ "เจอ แต่ไม่ได้แก้"
    และบรรทัดนี้คือตัวเดียวที่บอกได้ว่าคำใน fuzzy คำไหนตายแล้วควรลบ"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(
        meeting_dir,
        "# Summary",
        "GLM-5.2",
        glossary_counts={"PostgreSQL": 1},
        fuzzy_seen={"Electron": 2},
    )

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "แก้คำตาม glossary: PostgreSQL 1 จุด" in meta
    assert "คำ fuzzy ที่เจอในห้อง: Electron 2 ครั้ง" in meta


def test_save_summary_meta_is_untouched_when_the_glossary_did_nothing(tmp_path):
    """คนที่ยังไม่มี glossary.md ต้องได้ไฟล์หน้าตาเดิมเป๊ะ ไม่มีบรรทัดเปล่าโผล่มา"""
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)

    save_summary(
        meeting_dir, "# Summary", "GLM-5.2", glossary_counts={}, fuzzy_seen={}
    )

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert meta == "สรุปด้วย GLM-5.2\n"


def test_archive_audio_moves_the_recording_into_the_meeting_folder(tmp_path):
    meeting_dir = tmp_path / "meetings" / "2026-07-22-weekly-standup"
    meeting_dir.mkdir(parents=True)
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = archive_audio(meeting_dir, audio_path)

    assert destination == meeting_dir / "weekly-standup.mp3"
    assert destination.exists()
    assert not audio_path.exists()


def test_move_to_failed_moves_file_and_writes_error_log(tmp_path):
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = move_to_failed(audio_path, failed_dir, "Transcription failed: network error")

    assert destination == failed_dir / "broken.mp3"
    assert destination.exists()
    assert not audio_path.exists()
    error_log = failed_dir / "broken.error.log"
    assert error_log.read_text(encoding="utf-8") == "Transcription failed: network error"


def test_move_to_failed_takes_the_job_file_along(tmp_path):
    # the next attempt must summarize with the model the user actually picked
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(inbox_dir, "broken", "claude-sonnet-5")

    move_to_failed(audio_path, failed_dir, "Summarization failed: boom")

    assert not (inbox_dir / f"broken{JOB_SUFFIX}").exists()
    assert read_model(failed_dir / "broken.mp3") == "claude-sonnet-5"


def test_move_to_failed_works_when_there_is_no_job_file(tmp_path):
    failed_dir = tmp_path / "failed"
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    audio_path = inbox_dir / "dropped.mp3"
    audio_path.write_bytes(b"fake audio")

    destination = move_to_failed(audio_path, failed_dir, "Transcription failed: boom")

    assert destination.exists()


from src.storage import rename_speaker_in_transcript, safe_meeting_dir


def test_safe_meeting_dir_accepts_a_direct_child(tmp_path):
    meetings = tmp_path / "meetings"
    (meetings / "2026-07-28_10-30-standup").mkdir(parents=True)

    result = safe_meeting_dir(meetings, "2026-07-28_10-30-standup")

    assert result == (meetings / "2026-07-28_10-30-standup").resolve()


def test_safe_meeting_dir_rejects_anything_that_escapes(tmp_path):
    meetings = tmp_path / "meetings"
    meetings.mkdir()

    assert safe_meeting_dir(meetings, "..") is None
    assert safe_meeting_dir(meetings, "../../Windows") is None
    assert safe_meeting_dir(meetings, "") is None


def test_rename_speaker_in_transcript_replaces_only_the_line_headings(tmp_path):
    meeting_dir = tmp_path / "m1"
    meeting_dir.mkdir()
    (meeting_dir / "transcript.md").write_text(
        "# Transcript\n\n"
        "**ผู้พูด 2** [00:00]: เมื่อกี้ **ผู้พูด 2** พูดว่าอะไรนะ\n\n"
        "**ผู้พูด 1** [00:05]: ไม่รู้ครับ\n",
        encoding="utf-8",
    )

    assert rename_speaker_in_transcript(meeting_dir, "ผู้พูด 2", "สมหญิง็ม") is True

    text = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert text.startswith("# Transcript\n\n**สมหญิง็ม** [00:00]:")
    # ข้อความที่คนพูดต้องไม่ถูกแตะ แม้จะมีสตริงเดียวกันอยู่กลางบรรทัด
    assert "เมื่อกี้ **ผู้พูด 2** พูดว่าอะไรนะ" in text
    assert "**ผู้พูด 1** [00:05]" in text


def test_rename_speaker_in_transcript_reports_false_when_the_label_is_absent(tmp_path):
    meeting_dir = tmp_path / "m1"
    meeting_dir.mkdir()
    (meeting_dir / "transcript.md").write_text(
        "# Transcript\n\n**ผู้พูด 1** [00:00]: ครับ\n", encoding="utf-8"
    )

    assert rename_speaker_in_transcript(meeting_dir, "ผู้พูด 9", "สมหญิง็ม") is False


def test_rename_speaker_in_transcript_reports_false_when_the_file_is_gone(tmp_path):
    # meetings/ เป็นโฟลเดอร์ของผู้ใช้ เขาย้าย/ลบได้ตลอด -- การตั้งชื่อต้องไม่ล้มตาม
    assert rename_speaker_in_transcript(tmp_path / "ไม่มีจริง", "ผู้พูด 1", "สมหญิง็ม") is False


def test_rename_speaker_in_transcript_survives_the_crlf_files_this_project_writes(tmp_path):
    # write_text แปลง \n เป็น \r\n บน Windows ไฟล์จริงจึงเป็น CRLF ทุกไฟล์
    meeting_dir = tmp_path / "m1"
    meeting_dir.mkdir()
    (meeting_dir / "transcript.md").write_bytes(
        "# Transcript\r\n\r\n**ผู้พูด 2** [00:00]: ครับ\r\n".encode("utf-8")
    )

    assert rename_speaker_in_transcript(meeting_dir, "ผู้พูด 2", "สมหญิง็ม") is True

    # ต้องอ่านเป็น bytes: read_text แปลง \r\n ให้เป็น \n ตั้งแต่ขาเข้า เทสที่อ่านด้วย
    # read_text จึงผ่านเหมือนกันหมดไม่ว่าไฟล์บนดิสก์จะเป็น CRLF หรือ LF -- พิสูจน์
    # อะไรเกี่ยวกับ CRLF ไม่ได้เลย ทั้งที่ชื่อเทสบอกว่าพิสูจน์
    raw = (meeting_dir / "transcript.md").read_bytes()
    assert "**สมหญิง็ม** [00:00]: ครับ".encode("utf-8") in raw
    assert raw.count(b"\r\n") > 0
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_rename_speaker_in_transcript_treats_a_name_with_backslashes_literally(tmp_path):
    # ชื่อเดินทางมาจาก HTTP request และ clean_name ไม่ได้กรอง backslash ออก ถ้าเอาไป
    # ต่อเป็น replacement template ตรง ๆ ชื่ออย่าง "\1" จะทำให้ raise และ "\g<0>" จะยัด
    # ข้อความที่ match ได้กลับเข้าไฟล์แทนชื่อ
    meeting_dir = tmp_path / "m1"
    meeting_dir.mkdir()
    transcript = meeting_dir / "transcript.md"
    original = "# Transcript\n\n**ผู้พูด 1** [00:00]: ครับ\n"

    for hostile in ("\\1", "\\g<0>", "\\g<name>", "back\\slash", "\\0", "\\\\"):
        transcript.write_text(original, encoding="utf-8")

        assert rename_speaker_in_transcript(meeting_dir, "ผู้พูด 1", hostile) is True

        text = transcript.read_text(encoding="utf-8")
        assert f"**{hostile}** [00:00]: ครับ" in text
        assert "\x00" not in text


def test_rename_speaker_in_transcript_retries_when_windows_holds_the_old_file(tmp_path):
    # WinError 32: ตัวสแกนไวรัส/indexer จับไฟล์ที่เพิ่งปิดไปค้างได้ราวหนึ่งวินาที
    # การเขียนครั้งเดียวแล้วยอมแพ้คือวิธีทำให้ผู้ใช้เสียชื่อที่เพิ่งตั้งไปเฉย ๆ
    meeting_dir = tmp_path / "m1"
    meeting_dir.mkdir()
    transcript = meeting_dir / "transcript.md"
    transcript.write_text(
        "# Transcript\n\n**ผู้พูด 1** [00:00]: ครับ\n", encoding="utf-8"
    )
    attempts = {"count": 0}
    real_replace = type(transcript).replace

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("WinError 32")
        return real_replace(self, target)

    with patch("pathlib.Path.replace", flaky_replace), patch("time.sleep"):
        assert rename_speaker_in_transcript(meeting_dir, "ผู้พูด 1", "สมหญิง็ม") is True

    assert attempts["count"] == 3
    text = transcript.read_text(encoding="utf-8")
    assert "**สมหญิง็ม** [00:00]: ครับ" in text
