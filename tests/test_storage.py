from datetime import date
from unittest.mock import patch

from src.job import JOB_SUFFIX, read_model, write_job
from src.storage import (
    archive_audio,
    create_meeting_folder,
    meeting_folder_name,
    move_to_failed,
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
