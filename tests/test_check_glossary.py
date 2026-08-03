from src.glossary import DuplicateKey, Glossary
from tools.check_glossary import (
    count_in_transcripts,
    find_collisions,
    format_duplicate_keys,
    main,
    sample_context,
)


def _levels(findings: list[tuple[str, str]]) -> list[str]:
    return [level for level, _ in findings]


def test_a_wrong_form_inside_a_correct_term_is_the_heavy_finding():
    """ถ้าคำผิดอยู่ข้างในคำถูก ทุกครั้งที่มีคนพูดคำถูกออกมาถูกต้องแล้ว มันจะโดนแก้ให้
    เพี้ยน -- ตรงข้ามกับหน้าที่ของตารางนี้ทั้งหมด (เคส Control Pane/Control Panel)"""
    glossary = Glossary(exact={"Control Plane": ["Pane"], "Pane Layout": []})

    findings = find_collisions(glossary)

    assert "หนัก" in _levels(findings)
    assert any("Pane Layout" in message for _, message in findings)


def test_a_clean_table_produces_no_heavy_finding():
    glossary = Glossary(exact={"Kubernetes": ["คัตเตอร์ที่ยาวพอ"], "Xero": ["ซีโร่"]})

    assert "หนัก" not in _levels(find_collisions(glossary))


def test_short_wrong_forms_are_flagged_because_thai_has_no_word_breaks():
    glossary = Glossary(exact={"Bill": ["Bin"]})

    findings = find_collisions(glossary)

    assert _levels(findings) == ["เบา"]
    assert any('"Bin"' in message for _, message in findings)


def test_a_form_pointing_at_itself_is_flagged():
    glossary = Glossary(exact={"TANDA": ["TANDA"]})

    assert any("ชี้ไปหาตัวมันเอง" in message for _, message in find_collisions(glossary))


def test_fuzzy_forms_are_not_flagged_because_the_model_reads_the_context():
    """คำที่มีความหมายของตัวเองปลอดภัยเมื่ออยู่ใน fuzzy -- นั่นคือเหตุผลทั้งหมดที่ชั้นนั้น
    มีอยู่ การเตือนเรื่องมันคือการเตือนผิดที่ และจะกลบเสียงเตือนที่ควรฟัง"""
    glossary = Glossary(fuzzy={"Load Test": ["Node Test"], "Role": ["Low"]})

    assert find_collisions(glossary) == []


def test_sample_context_shows_a_collision_find_collisions_cannot_see(tmp_path):
    """find_collisions เช็คได้แค่คำผิดชนกับ "คำถูก" ที่ประกาศไว้ในตารางเอง แต่คำผิดสั้นๆ
    ไปกินกลางคำไทยทั่วไปที่ไม่ได้อยู่ในตารางเลยก็ได้ (เคสจริง: "ครูป" ของ Kubernetes
    ไปกินกลาง "ครูประกาศ" ซึ่งเป็นชื่อคน ไม่ใช่คำถูกของใครในตารางนี้สักคำ) sample_context
    คือทางเดียวที่จับเคสนี้ได้ -- ให้คนเห็นบริบทจริงแล้วตัดสินเอง"""
    meeting = tmp_path / "2026-08-03_16-01-PCI"
    meeting.mkdir()
    (meeting / "transcript.md").write_text(
        "**satit (บอล)** [22:19]: ให้ครูประกาศช่วยดูให้ว่าทั้งหมดมันโอเคไหม",
        encoding="utf-8",
    )
    glossary = Glossary(exact={"Kubernetes": ["ครูป"]})

    context = sample_context(glossary, tmp_path)

    assert "ครูป" in context
    assert any("ครูประกาศ" in snippet for snippet in context["ครูป"])


def test_sample_context_only_covers_exact_and_aliases(tmp_path):
    """fuzzy/project-names ให้โมเดลตัดสินจากบริบทเองอยู่แล้ว (ดู _replacing_layers) --
    โชว์บริบทของชั้นนั้นเป็นเสียงเตือนผิดที่ ไม่ควรอยู่ในผลลัพธ์เลย"""
    meeting = tmp_path / "2026-01-01_09-00-a"
    meeting.mkdir()
    (meeting / "transcript.md").write_text(
        "**ผู้พูด 1** [00:00]: โปรแกรมนี้ใช้ GORM", encoding="utf-8"
    )
    glossary = Glossary(fuzzy={"GORM": ["กรม"]})

    assert sample_context(glossary, tmp_path) == {}


def test_sample_context_caps_examples_per_wrong_form(tmp_path):
    meeting = tmp_path / "2026-01-01_09-00-a"
    meeting.mkdir()
    (meeting / "transcript.md").write_text(
        "**ผู้พูด 1** [00:00]: Depth หนึ่ง Depth สอง Depth สาม", encoding="utf-8"
    )
    glossary = Glossary(exact={"Dev": ["Depth"]})

    context = sample_context(glossary, tmp_path)

    assert len(context["Depth"]) == 2


def test_count_in_transcripts_measures_against_the_raw_files(tmp_path):
    """transcript.md เป็นของดิบ ตัวกรองไม่แตะมัน ตัวเลขนี้จึงเป็นสิ่งที่ whisper ถอด
    ออกมาจริง ไม่ใช่ผลหลังแก้ -- header ของ glossary.md ขอให้ปรับตามของจริงที่เจอ"""
    for name, text in (
        ("2026-01-01_09-00-a", "**ผู้พูด 1** [00:00]: ขึ้น Depth ก่อน แล้ว Depth อีกที"),
        ("2026-01-02_09-00-b", "**ผู้พูด 1** [00:00]: Depth กับ Staging"),
    ):
        meeting = tmp_path / name
        meeting.mkdir()
        (meeting / "transcript.md").write_text(text, encoding="utf-8")
    glossary = Glossary(exact={"Dev": ["Depth", "Death"]})

    tally = count_in_transcripts(glossary, tmp_path)

    assert tally["Depth"] == (3, 2)
    # คำที่ไม่เคยโผล่ต้องอยู่ในผลด้วย เป็นศูนย์ -- ไม่งั้นหาของที่ควรลบไม่เจอ
    assert tally["Death"] == (0, 0)


# --- เขียนซ้ำในไฟล์ (คีย์ซ้ำ) --------------------------------------------------
#
# src/glossary.py รวมฟอร์มให้อัตโนมัติแล้วตั้งแต่แก้บั๊ก assignment->merge จึงไม่ใช่
# "หนัก" (ข้อมูลไม่หาย) แต่ยังควรรายงานไว้ให้คนไปรวมบรรทัดในไฟล์เพื่อความสะอาด


def test_format_duplicate_keys_lists_every_line_the_term_was_declared_on():
    dup = DuplicateKey(section="exact", term="JWKS", lines=(66, 115))

    messages = format_duplicate_keys([dup])

    assert len(messages) == 1
    assert "JWKS" in messages[0]
    assert "66" in messages[0]
    assert "115" in messages[0]


def test_format_duplicate_keys_returns_one_message_per_duplicate():
    dups = [
        DuplicateKey(section="exact", term="JWKS", lines=(66, 115)),
        DuplicateKey(section="exact", term="Approve", lines=(108, 129, 171)),
    ]

    messages = format_duplicate_keys(dups)

    assert len(messages) == 2


def test_no_duplicates_produces_no_messages():
    assert format_duplicate_keys([]) == []


def _write_glossary(tmp_path, text):
    glossary_path = tmp_path / "glossary.md"
    teams_path = tmp_path / "teams.md"
    glossary_path.write_text(text, encoding="utf-8")
    teams_path.write_text("", encoding="utf-8")
    return glossary_path, teams_path


def test_duplicate_keys_are_reported_but_do_not_force_a_nonzero_exit_code(
    tmp_path, capsys
):
    """คีย์ซ้ำไม่ทำให้ข้อมูลหายอีกต่อไปหลังแก้ parser (รวมอัตโนมัติ) -- จึงเป็นแค่
    เรื่องความสะอาดของไฟล์ ไม่ใช่บั๊กที่ควรทำให้ exit code ไม่เป็นศูนย์

    ฟอร์มยาว >= 4 อักขระโดยตั้งใจ -- ฟอร์มสั้นจะโดน MIN_SAFE_LENGTH ของ find_collisions
    แล้ว "JWKS" จะโผล่ในบล็อก "ควรดู" ไปด้วย ทำให้ assert "JWKS" in out ผ่านได้แม้
    บล็อกคีย์ซ้ำจะถูกตัดออกไปเงียบๆ (พิสูจน์แล้วด้วย mutation -- ปิดบล็อกคีย์ซ้ำทิ้ง
    ทั้งก้อนแล้วเทสรุ่นแรกที่ใช้ฟอร์มสั้นยังผ่านอยู่) จึง assert หัวข้อ/สัญลักษณ์ของ
    บล็อกนี้โดยตรง ไม่ใช่แค่ชื่อคำถูกซึ่งโผล่ได้จากหลายจุด
    """
    glossary_path, teams_path = _write_glossary(
        tmp_path, "## exact\nJWKS: alpha\nJWKS: bravo\n"
    )

    code = main(
        [
            "--glossary",
            str(glossary_path),
            "--teams",
            str(teams_path),
            "--meetings",
            str(tmp_path / "no-such-meetings-dir"),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "เขียนซ้ำในไฟล์ (1):" in out
    assert "↻" in out
    assert "JWKS" in out


def test_a_heavy_collision_alongside_a_duplicate_key_still_exits_nonzero(
    tmp_path, capsys
):
    """คีย์ซ้ำไม่ควรไปกลบหรือลด severity ของ finding อื่นที่ยังร้ายแรงอยู่จริง"""
    glossary_path, teams_path = _write_glossary(
        tmp_path,
        "## exact\nControl Plane: Pane\nPane Layout: PaneLayoutTypo\n"
        "JWKS: alpha\nJWKS: bravo\n",
    )

    code = main(
        [
            "--glossary",
            str(glossary_path),
            "--teams",
            str(teams_path),
            "--meetings",
            str(tmp_path / "no-such-meetings-dir"),
        ]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "Pane" in out
    assert "เขียนซ้ำในไฟล์ (1):" in out


def test_a_clean_file_with_no_duplicates_prints_no_duplicate_section(
    tmp_path, capsys
):
    glossary_path, teams_path = _write_glossary(tmp_path, "## exact\nRailway: เรลเวย์\n")

    main(
        [
            "--glossary",
            str(glossary_path),
            "--teams",
            str(teams_path),
            "--meetings",
            str(tmp_path / "no-such-meetings-dir"),
        ]
    )

    out = capsys.readouterr().out
    assert "เขียนซ้ำ" not in out
