import logging

from src.chunk import parse_transcript_segments
from src.glossary import load

GLOSSARY_SAMPLE = """# Glossary

## exact
# แทนที่ตรงๆ ในโค้ด -- บรรทัดนี้เป็น comment ต้องถูกข้าม
PostgreSQL: โพสเกรส, พอสเกรส
Railway: เรลเวย์

## fuzzy
Electron: อิเล็กตรอน

## project-names
DungImage: ดังอิมเมจ, เดกอิมเมจ

## aliases
ระบบชำระเงิน: หน้าจ่ายเงิน, checkout

## ambiguous
เสร็จ | Business = demo ให้ลูกค้าได้ | dev = merge แล้ว ยังไม่ deploy
"""

TEAMS_SAMPLE = """# ฝ่ายของผู้เข้าร่วม
# comment

Business: สมชาย, สุดา
dev: บอล
"""


def _write(tmp_path, glossary=GLOSSARY_SAMPLE, teams=TEAMS_SAMPLE):
    glossary_path = tmp_path / "glossary.md"
    teams_path = tmp_path / "teams.md"
    glossary_path.write_text(glossary, encoding="utf-8")
    teams_path.write_text(teams, encoding="utf-8")
    return glossary_path, teams_path


TRANSCRIPT = """# Transcript

**ผู้พูด 1** [00:00]: ตอนนี้เราใช้โพสเกรสอยู่

**ผู้พูด 2** [00:12]: พอสเกรสเวอร์ชันไหนครับ
"""


def _loaded(tmp_path, glossary=GLOSSARY_SAMPLE, teams=TEAMS_SAMPLE):
    return load(*_write(tmp_path, glossary, teams))


def test_apply_exact_replaces_every_wrong_form_and_counts_per_term(tmp_path):
    glossary = _loaded(tmp_path)

    corrected, counts = glossary.apply_exact(TRANSCRIPT)

    assert "โพสเกรส" not in corrected
    assert "พอสเกรส" not in corrected
    assert corrected.count("PostgreSQL") == 2
    assert counts == {"PostgreSQL": 2}


def test_apply_exact_leaves_text_untouched_when_nothing_matches(tmp_path):
    glossary = _loaded(tmp_path)
    text = "**ผู้พูด 1** [00:00]: ไม่มีศัพท์ในตารางเลย"

    corrected, counts = glossary.apply_exact(text)

    assert corrected == text
    assert counts == {}


def test_aliases_collapse_both_sides_to_the_central_name(tmp_path):
    glossary = _loaded(tmp_path)
    text = (
        "**ผู้พูด 1** [00:00]: หน้าจ่ายเงินพังอยู่\n\n"
        "**ผู้พูด 2** [00:20]: checkout ฝั่ง dev ก็เห็นเหมือนกัน"
    )

    corrected, counts = glossary.apply_exact(text)

    assert "หน้าจ่ายเงิน" not in corrected
    assert "checkout" not in corrected
    assert corrected.count("ระบบชำระเงิน") == 2
    assert counts == {"ระบบชำระเงิน": 2}


def test_apply_exact_never_rewrites_a_speaker_label(tmp_path):
    """ป้ายผู้พูดเป็นงานของ speakers/registry.json ตัวกรองศัพท์ห้ามแตะ
    ต่อให้คำผิดในตารางไปตรงกับชื่อผู้พูดเป๊ะๆ ก็ต้องแก้แค่ในเนื้อความ"""
    glossary = _loaded(tmp_path, glossary="## exact\nBall: บอล\n")
    text = "**บอล** [00:00]: บอลจะทำให้เสร็จวันนี้"

    corrected, counts = glossary.apply_exact(text)

    assert corrected == "**บอล** [00:00]: Ballจะทำให้เสร็จวันนี้"
    assert counts == {"Ball": 1}


def test_apply_exact_keeps_every_segment_parseable(tmp_path):
    """invariant กันเคสที่พังเงียบที่สุด: ถ้าหัว segment เสีย
    parse_transcript_segments คืน list ว่าง แล้ว summarize ตกไปสรุปแบบ single call
    ทั้งไฟล์ ได้สรุปหน้าตาปกติ ไม่มี error แต่เสียไทม์ไลน์ทั้งหมด

    glossary นี้พยายามกลืนหัว segment ทั้งชิ้น (`** [00:00]:` -> `X`)
    ซึ่งเป็นรูปเดียวที่ทำให้ block ไม่ match SEGMENT_PATTERN ได้จริง
    วัดจากการรัน mutation แล้ว: มันต้องถอด "ทั้งสองด่าน" (ด่านคัดคำที่มี markup
    ตอน parse และสาขาหัว segment ใน pattern) เทสต์นี้จึงจะแดง ถอดด่านเดียวยังเขียว
    -- เก็บไว้เป็นตาข่ายชั้นสุดท้าย ไม่ใช่เทสต์ที่พิสูจน์ด่านใดด่านหนึ่ง
    (ด่านหัว segment มีเทสต์ของตัวเองที่ test_apply_exact_never_rewrites_a_speaker_label)
    """
    glossary = _loaded(tmp_path, glossary="## exact\nX: ** [00:00]:\n")
    before = len(parse_transcript_segments(TRANSCRIPT))
    assert before == 2, "sanity: fixture ต้องมี 2 segment จริง"

    corrected, _ = glossary.apply_exact(TRANSCRIPT)

    assert len(parse_transcript_segments(corrected)) == before


def test_longer_wrong_form_wins_over_a_shorter_one_it_contains(tmp_path):
    # คำสั้นเขียนไว้ "ก่อน" คำยาวโดยเจตนา ถ้าโค้ดไม่เรียงยาวก่อนสั้นเอง
    # มันจะ match "โพสเกรส" แล้วเหลือเศษ "เอสคิวแอล" ค้างอยู่
    glossary = _loaded(
        tmp_path, glossary="## exact\nPostgreSQL: โพสเกรส, โพสเกรสเอสคิวแอล\n"
    )

    corrected, counts = glossary.apply_exact("**ผู้พูด 1** [00:00]: โพสเกรสเอสคิวแอลล่ม")

    assert corrected == "**ผู้พูด 1** [00:00]: PostgreSQLล่ม"
    assert counts == {"PostgreSQL": 1}


def test_replacement_runs_once_and_never_cascades(tmp_path):
    """ตารางที่คำถูกของบรรทัดหนึ่งเป็นคำผิดของอีกบรรทัด ต้องไม่ไล่ต่อเป็นทอดๆ
    "เรลเว" -> "เรลเวย์" แล้วหยุด ไม่ใช่ไล่ต่อไปเป็น "Railway" """
    glossary = _loaded(tmp_path, glossary="## exact\nRailway: เรลเวย์\nเรลเวย์: เรลเว\n")

    corrected, _ = glossary.apply_exact("**ผู้พูด 1** [00:00]: เรลเว")

    assert corrected == "**ผู้พูด 1** [00:00]: เรลเวย์"


def test_backslash_in_a_correct_form_is_inserted_literally(tmp_path):
    """re.sub อ่านสตริงแทนที่เป็น template: \\1 จะ raise, \\g<0> จะแทนคำเดิมกลับ
    แบบเงียบๆ เคยเกิดจริงใน rename_speaker_in_transcript ของ repo นี้"""
    glossary = _loaded(
        tmp_path, glossary="## exact\na\\1b: ผิดหนึ่ง\nx\\g<0>y: ผิดสอง\n"
    )

    corrected, counts = glossary.apply_exact(
        "**ผู้พูด 1** [00:00]: ผิดหนึ่ง และ ผิดสอง"
    )

    assert corrected == "**ผู้พูด 1** [00:00]: a\\1b และ x\\g<0>y"
    assert counts == {"a\\1b": 1, "x\\g<0>y": 1}


def test_apply_exact_is_idempotent(tmp_path):
    glossary = _loaded(tmp_path)

    once, first_counts = glossary.apply_exact(TRANSCRIPT)
    twice, second_counts = glossary.apply_exact(once)

    assert twice == once
    assert first_counts == {"PostgreSQL": 2}
    assert second_counts == {}


def test_terms_containing_transcript_markup_are_skipped_with_a_warning(
    tmp_path, caplog
):
    """`*` `[` `]` คืออักขระที่ประกอบ **ผู้พูด N** [mm:ss]: ถ้าปล่อยให้คำพวกนี้
    ถูกแทนที่ลงไป transcript จะ parse ไม่ออก"""
    with caplog.at_level(logging.WARNING):
        glossary = _loaded(
            tmp_path,
            glossary="## exact\n**หนา**: ผิดหนึ่ง\nดี[1]: ผิดสอง\nRailway: เรลเวย์\n",
        )

    assert glossary.exact == {"Railway": ["เรลเวย์"]}
    assert "**หนา**" in caplog.text
    assert "ดี[1]" in caplog.text


def test_count_only_counts_without_changing_the_text(tmp_path):
    """ชั้น fuzzy โมเดลเป็นคนตีความ ไม่ได้ถูกแทนที่ในโค้ด ถ้าไม่มีตัวเลขนี้จะไม่มีทาง
    รู้ว่าคำไหนตายแล้วควรตัดออกจาก prompt (fuzzy กิน token ทุกครั้งที่สรุป)"""
    glossary = _loaded(tmp_path)
    text = (
        "**ผู้พูด 1** [00:00]: อิเล็กตรอนโหลดช้า\n\n"
        "**ผู้พูด 2** [00:30]: ดังอิมเมจกับอิเล็กตรอนคนละตัวนะ"
    )

    seen = glossary.count_only(text)

    assert seen == {"Electron": 2, "DungImage": 1}


def test_count_only_returns_a_copy_of_the_text_unchanged(tmp_path):
    glossary = _loaded(tmp_path)
    text = "**ผู้พูด 1** [00:00]: อิเล็กตรอนโหลดช้า"

    glossary.count_only(text)

    assert text == "**ผู้พูด 1** [00:00]: อิเล็กตรอนโหลดช้า"


def test_count_only_ignores_matches_inside_a_speaker_label(tmp_path):
    glossary = _loaded(tmp_path, glossary="## fuzzy\nElectron: บอล\n")
    text = "**บอล** [00:00]: บอลพูดเอง"

    assert glossary.count_only(text) == {"Electron": 1}


def test_count_only_ignores_exact_and_aliases(tmp_path):
    """สองชั้นนั้นถูกแทนที่ไปแล้วใน apply_exact การนับซ้ำจะทำให้ตัวเลขสองบรรทัด
    ท้าย summary.md หมายถึงเรื่องเดียวกันทั้งที่ตั้งใจให้ต่างกัน"""
    glossary = _loaded(tmp_path)
    text = "**ผู้พูด 1** [00:00]: โพสเกรสกับหน้าจ่ายเงิน"

    assert glossary.count_only(text) == {}


def test_format_for_prompt_omits_cross_team_tables_by_default(tmp_path):
    glossary = _loaded(tmp_path)

    prompt = glossary.format_for_prompt()

    assert "Electron" in prompt and "อิเล็กตรอน" in prompt
    assert "DungImage" in prompt
    assert "เสร็จ" not in prompt, "ตาราง ambiguous ไม่ควรเข้าประชุม dev ล้วน"
    assert "สมชาย" not in prompt, "teams ไม่ควรเข้าประชุม dev ล้วน"


def test_format_for_prompt_includes_cross_team_tables_when_asked(tmp_path):
    glossary = _loaded(tmp_path)

    prompt = glossary.format_for_prompt(include_cross_team_context=True)

    assert "เสร็จ" in prompt
    assert "merge แล้ว ยังไม่ deploy" in prompt
    assert "สมชาย" in prompt and "บอล" in prompt


def test_format_for_prompt_never_leaks_already_replaced_layers(tmp_path):
    """exact กับ aliases ถูกแทนที่ในโค้ดไปแล้ว ใส่ซ้ำใน prompt = เปลือง token เปล่า"""
    glossary = _loaded(tmp_path)

    prompt = glossary.format_for_prompt(include_cross_team_context=True)

    assert "PostgreSQL" not in prompt
    assert "โพสเกรส" not in prompt
    assert "ระบบชำระเงิน" not in prompt


def test_cross_context_without_teams_warns_that_ambiguous_cannot_work(tmp_path, caplog):
    """ตาราง ambiguous บอกโมเดลให้ "ตีความตามฝ่ายของผู้พูด" ถ้าไม่มี teams.md
    โมเดลไม่รู้ว่าใครอยู่ฝ่ายไหน ตารางนั้นแทบไม่ช่วยอะไร -- และกฎใน cross.md
    ยังสั่งให้ไปดูตารางฝ่ายที่ไม่มีอยู่ ต้องเตือนตอนที่มันเกิด ไม่ใช่ปล่อยเงียบ"""
    glossary = _loaded(tmp_path, teams="# ไม่มีใครเลย\n")
    assert glossary.ambiguous, "sanity: fixture ต้องมีตาราง ambiguous"
    assert glossary.teams == {}

    with caplog.at_level(logging.WARNING):
        prompt = glossary.format_for_prompt(include_cross_team_context=True)

    assert "teams.md" in caplog.text
    # ตาราง ambiguous ยังใส่ให้ตามที่ขอ -- เตือน ไม่ใช่เงียบ ไม่ใช่ตัดของทิ้ง
    assert "เสร็จ" in prompt


def test_no_teams_warning_when_cross_context_is_not_requested(tmp_path, caplog):
    glossary = _loaded(tmp_path, teams="# ไม่มีใครเลย\n")

    with caplog.at_level(logging.WARNING):
        glossary.format_for_prompt()

    assert "teams.md" not in caplog.text


def test_format_for_prompt_is_empty_when_nothing_is_configured(tmp_path):
    glossary = load(tmp_path / "nope.md", tmp_path / "also-nope.md")

    assert glossary.format_for_prompt(include_cross_team_context=True) == ""


def test_inline_comments_are_stripped_from_terms(tmp_path):
    """format ที่ออกแบบไว้ใช้ comment ต่อท้ายบรรทัดจริง เช่น
    `Electron: อิเล็กตรอน  # เฉพาะตอนคุยเรื่อง desktop app`
    ถ้าไม่ตัดออก คำผิดจะกลายเป็น "อิเล็กตรอน  # เฉพาะ..." ซึ่งไม่ match อะไรเลย
    และข้อความ comment จะไหลเข้าไปอยู่ใน prompt ด้วย"""
    glossary = _loaded(
        tmp_path,
        glossary=(
            "## exact\nRailway: เรลเวย์  # ชื่อ PaaS ไม่ใช่รถไฟ\n\n"
            "## fuzzy\nElectron: อิเล็กตรอน  # เฉพาะบริบท desktop app\n\n"
            "## ambiguous\nเสร็จ | Business = demo ได้ | dev = merge แล้ว  # กับดักคลาสสิก\n"
        ),
    )

    assert glossary.exact == {"Railway": ["เรลเวย์"]}
    assert glossary.fuzzy == {"Electron": ["อิเล็กตรอน"]}
    assert glossary.ambiguous[0]["meanings"]["dev"] == "merge แล้ว"
    assert "กับดักคลาสสิก" not in glossary.format_for_prompt(
        include_cross_team_context=True
    )


def test_a_hash_inside_a_term_is_not_treated_as_a_comment(tmp_path):
    """`C#` กับ `F#` เป็นชื่อภาษาจริง จะตัดที่ # เฉยๆ ไม่ได้
    ต้องตัดเฉพาะ # ที่มีช่องว่างนำหน้า ซึ่งเป็นรูปที่ comment จริงใช้"""
    glossary = _loaded(tmp_path, glossary="## exact\nC#: ซีชาร์ป, ซี ชาร์ป\n")

    assert glossary.exact == {"C#": ["ซีชาร์ป", "ซี ชาร์ป"]}

    corrected, counts = glossary.apply_exact("**ผู้พูด 1** [00:00]: เขียนด้วยซีชาร์ป")

    assert corrected == "**ผู้พูด 1** [00:00]: เขียนด้วยC#"
    assert counts == {"C#": 1}


def test_load_parses_every_section(tmp_path):
    glossary_path, teams_path = _write(tmp_path)

    result = load(glossary_path, teams_path)

    assert result.exact == {
        "PostgreSQL": ["โพสเกรส", "พอสเกรส"],
        "Railway": ["เรลเวย์"],
    }
    assert result.fuzzy == {"Electron": ["อิเล็กตรอน"]}
    assert result.project_names == {"DungImage": ["ดังอิมเมจ", "เดกอิมเมจ"]}
    assert result.aliases == {"ระบบชำระเงิน": ["หน้าจ่ายเงิน", "checkout"]}
    assert result.ambiguous == [
        {
            "term": "เสร็จ",
            "meanings": {
                "Business": "demo ให้ลูกค้าได้",
                "dev": "merge แล้ว ยังไม่ deploy",
            },
        }
    ]


def test_load_parses_teams(tmp_path):
    glossary_path, teams_path = _write(tmp_path)

    result = load(glossary_path, teams_path)

    assert result.teams == {"Business": ["สมชาย", "สุดา"], "dev": ["บอล"]}


def test_missing_files_load_as_empty_without_raising(tmp_path):
    result = load(tmp_path / "nope.md", tmp_path / "also-nope.md")

    assert result.exact == {}
    assert result.fuzzy == {}
    assert result.aliases == {}
    assert result.ambiguous == []
    assert result.teams == {}


def test_crlf_file_leaves_no_carriage_return_in_any_term(tmp_path):
    """glossary.md เขียนบน Windows เป็น CRLF ถ้า \\r ติดมากับคำถูก มันจะถูกยัดกลาง
    transcript เขียนด้วย write_bytes ตรงๆ เพราะ write_text จะแปลง newline ให้เอง"""
    glossary_path = tmp_path / "glossary.md"
    teams_path = tmp_path / "teams.md"
    glossary_path.write_bytes(GLOSSARY_SAMPLE.replace("\n", "\r\n").encode("utf-8"))
    teams_path.write_bytes(TEAMS_SAMPLE.replace("\n", "\r\n").encode("utf-8"))

    result = load(glossary_path, teams_path)

    every_string = [
        *result.exact,
        *result.fuzzy,
        *result.project_names,
        *result.aliases,
        *result.teams,
        *[w for forms in result.exact.values() for w in forms],
        *[w for forms in result.aliases.values() for w in forms],
        *[n for names in result.teams.values() for n in names],
        *[entry["term"] for entry in result.ambiguous],
        *[m for entry in result.ambiguous for m in entry["meanings"].values()],
    ]
    assert every_string, "sanity: the sample must actually have parsed"
    for value in every_string:
        assert "\r" not in value, f"carriage return survived in {value!r}"
