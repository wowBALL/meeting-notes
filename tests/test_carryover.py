import logging

from src.carryover import format_for_prompt, previous_open_items

OPEN_ITEMS_HEADING = "## ต้องคุยต่อครั้งหน้า"


def _summary(profile: str, open_items: str = "- เรื่องค้าง A\n- เรื่องค้าง B") -> str:
    return (
        "## หัวข้อที่คุยกัน\n- อะไรบ้าง\n\n"
        "## ตกลงแล้ว\n- ตกลงเรื่องนี้\n\n"
        f"{OPEN_ITEMS_HEADING}\n{open_items}\n\n"
        "## คำที่น่าจะถอดเพี้ยน (ยังไม่อยู่ใน glossary)\n- ไม่มี\n"
        "\n---\n"
        f"สรุปด้วย GLM-5.2\nประเภทประชุม: {profile}\n"
    )


def _meeting(meetings_dir, name, profile, open_items="- เรื่องค้าง A\n- เรื่องค้าง B"):
    d = meetings_dir / name
    d.mkdir(parents=True)
    (d / "summary.md").write_text(_summary(profile, open_items), encoding="utf-8")
    return d


def test_finds_the_open_items_of_the_latest_same_profile_meeting(tmp_path):
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", "- เก่าสุด")
    _meeting(meetings, "2026-07-27_09-00-standup", "dev", "- ล่าสุด")

    assert previous_open_items(meetings, "dev") == "- ล่าสุด"


def test_a_different_profile_is_skipped(tmp_path):
    """เรื่องค้างของประชุมข้ามฝ่ายห้ามไปโผล่ในสรุป dev ล้วน มันคุยกันคนละวง"""
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", "- ของ dev")
    _meeting(meetings, "2026-07-27_09-00-crossteam", "cross", "- ของ cross")

    assert previous_open_items(meetings, "dev") == "- ของ dev"
    assert previous_open_items(meetings, "cross") == "- ของ cross"


def test_the_meeting_being_summarized_is_excluded(tmp_path):
    """path ลองใหม่ (ลาก .job.json กลับ inbox) เข้ามาที่โฟลเดอร์เดิมที่มี summary.md
    จากรอบก่อนอยู่แล้ว ถ้าไม่กันไว้ ประชุมจะอ่านเรื่องค้างของตัวเองมาเป็น carryover"""
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", "- ของประชุมก่อน")
    current = _meeting(meetings, "2026-07-27_09-00-standup", "dev", "- ของตัวเอง")

    assert previous_open_items(meetings, "dev", exclude_dir=current) == "- ของประชุมก่อน"


def test_no_meetings_directory_is_not_an_error(tmp_path):
    assert previous_open_items(tmp_path / "ไม่มีอยู่", "dev") == ""


def test_no_matching_profile_returns_nothing(tmp_path):
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-crossteam", "cross")

    assert previous_open_items(meetings, "dev") == ""


def test_a_summary_written_before_profiles_existed_is_skipped(tmp_path):
    """สรุปเก่าไม่มีบรรทัด "ประเภทประชุม:" จับคู่ profile ไม่ได้ จึงข้ามไป
    ดีกว่าเดาว่าเป็น dev แล้วลากเรื่องค้างผิดวงเข้ามา"""
    meetings = tmp_path / "meetings"
    old = meetings / "2026-07-20_09-00-standup"
    old.mkdir(parents=True)
    (old / "summary.md").write_text(
        f"{OPEN_ITEMS_HEADING}\n- เรื่องค้างเก่า\n\n---\nสรุปด้วย GLM-5.2\n",
        encoding="utf-8",
    )

    assert previous_open_items(meetings, "dev") == ""


def test_a_parenthetical_no_items_note_is_not_carried_over(tmp_path):
    """เจอจากสรุปประชุมจริง (2026-07-30): แม้ prompt สั่งให้เว้นว่าง โมเดลก็เขียน
    "(ไม่พบประเด็นที่เห็นไม่ตรงกันชัดเจน)" ใส่หัวข้อที่ไม่มีเนื้อหา ถ้าปล่อยไว้
    ประชุมครั้งถัดไปจะได้ "(ไม่มีเรื่องค้าง)" มาเป็นเรื่องค้างแล้วเขียนความคืบหน้าของมัน

    เรื่องค้างคือ bullet เท่านั้น (`- ...`) ตามที่ template ใน prompt กำหนด
    บรรทัดที่ไม่ใช่ bullet จึงไม่ใช่รายการ ไม่ต้องยกไป
    """
    meetings = tmp_path / "meetings"
    for name, body in (
        ("2026-07-20_09-00-a", "(ไม่มีเรื่องค้าง)"),
        ("2026-07-21_09-00-b", "ไม่มีเรื่องที่ต้องคุยต่อ"),
        ("2026-07-22_09-00-c", "(ไม่พบ)\n(ไม่มีอะไรค้าง)"),
    ):
        d = meetings / name
        d.mkdir(parents=True)
        (d / "summary.md").write_text(_summary("dev", body), encoding="utf-8")
        assert previous_open_items(meetings, "dev") == "", f"{name} ไม่ควรถูกยกไป"


def test_a_real_bullet_next_to_a_note_still_carries_only_the_bullet(tmp_path):
    meetings = tmp_path / "meetings"
    _meeting(
        meetings,
        "2026-07-20_09-00-standup",
        "dev",
        "(ยังไม่ได้สะสาง)\n- เรื่องค้างจริง A\n- เรื่องค้างจริง B",
    )

    assert previous_open_items(meetings, "dev") == "- เรื่องค้างจริง A\n- เรื่องค้างจริง B"


def test_an_empty_open_items_section_returns_nothing(tmp_path):
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", open_items="")

    assert previous_open_items(meetings, "dev") == ""


def test_a_summary_without_the_open_items_heading_returns_nothing(tmp_path):
    meetings = tmp_path / "meetings"
    d = meetings / "2026-07-20_09-00-standup"
    d.mkdir(parents=True)
    (d / "summary.md").write_text(
        "## ตกลงแล้ว\n- x\n\n---\nสรุปด้วย GLM-5.2\nประเภทประชุม: dev\n",
        encoding="utf-8",
    )

    assert previous_open_items(meetings, "dev") == ""


def test_a_crlf_summary_file_parses(tmp_path):
    """summary.md ที่ Python เขียนบน Windows เป็น CRLF (write_text แปลง \\n เป็น
    os.linesep) ถ้าตัวอ่านยึด \\n ตรงๆ หรือใช้ regex ที่ยึด $ มันจะไม่ match อะไรเลย
    เขียนด้วย write_bytes เพื่อบังคับ CRLF จริง ไม่ให้ write_text มาแปลงให้"""
    meetings = tmp_path / "meetings"
    d = meetings / "2026-07-20_09-00-standup"
    d.mkdir(parents=True)
    (d / "summary.md").write_bytes(
        _summary("dev", "- เรื่องค้าง CRLF").replace("\n", "\r\n").encode("utf-8")
    )

    assert previous_open_items(meetings, "dev") == "- เรื่องค้าง CRLF"


def test_an_unreadable_summary_is_skipped_without_crashing(tmp_path, caplog):
    meetings = tmp_path / "meetings"
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", "- ที่อ่านได้")
    broken = meetings / "2026-07-27_09-00-standup"
    broken.mkdir(parents=True)
    (broken / "summary.md").write_bytes(b"\xff\xfe\x00 invalid utf-8")

    with caplog.at_level(logging.WARNING):
        result = previous_open_items(meetings, "dev")

    assert result == "- ที่อ่านได้"


def test_a_file_where_a_meeting_folder_should_be_is_ignored(tmp_path):
    meetings = tmp_path / "meetings"
    meetings.mkdir()
    (meetings / "notes.txt").write_text("ไม่ใช่โฟลเดอร์ประชุม", encoding="utf-8")
    _meeting(meetings, "2026-07-20_09-00-standup", "dev", "- ของจริง")

    assert previous_open_items(meetings, "dev") == "- ของจริง"


def test_format_for_prompt_is_empty_when_there_is_nothing_to_carry(tmp_path):
    assert format_for_prompt("") == ""


def test_format_for_prompt_asks_for_the_progress_section(tmp_path):
    text = format_for_prompt("- เรื่องค้าง A")

    assert "- เรื่องค้าง A" in text
    assert "## คืบหน้าจากครั้งก่อน" in text
    # ห้ามให้โมเดลแต่งความคืบหน้าที่ไม่มีใครพูดถึง -- carryover ผูกสรุปเป็นลูกโซ่
    # ถ้ามันเดาได้ ความผิดจะถูกส่งต่อไปครั้งถัดไปเรื่อยๆ
    assert "ยังไม่ได้แตะ" in text
