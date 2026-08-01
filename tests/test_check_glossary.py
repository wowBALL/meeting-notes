from src.glossary import Glossary
from tools.check_glossary import count_in_transcripts, find_collisions


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
