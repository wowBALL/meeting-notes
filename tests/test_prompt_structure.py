"""เทสต์ "เนื้อหา" ของไฟล์ prompt ไม่ใช่พฤติกรรมของโมเดล

ข้อจำกัดที่ต้องพูดตรงๆ: เทสต์พวกนี้ยืนยันแค่ว่าคำสั่งยังอยู่ในไฟล์ ไม่ได้พิสูจน์ว่า
โมเดลทำตาม การพิสูจน์อย่างหลังต้องเอาสรุปจากประชุมจริงมาอ่านเทียบ ทำอัตโนมัติไม่ได้

แล้วมันมีค่าอะไร: กฎในไฟล์พวกนี้แก้ได้โดยไม่ต้องรันเทสต์ (นั่นคือจุดประสงค์ของข้อ 1)
กฎที่สำคัญที่สุดจึงหายไปได้ง่ายที่สุดตอนมีคนไป "จัดระเบียบถ้อยคำ" ทีหลัง
เทสต์พวกนี้คือสัญญาว่ากฎไหนห้ามหลุด -- ถ้าจะเรียบเรียงใหม่ ให้แก้เทสต์ด้วยโดยเจตนา
ไม่ใช่ให้มันหลุดไปเงียบๆ
"""

from src.prompts import DEFAULT_PROMPTS_DIR, render

# หัวข้อที่สรุปฉบับเต็มต้องมีครบ -- reduce (ประชุมยาว) กับ single (ประชุมสั้น) ต้อง
# ให้ผลลัพธ์หน้าตาเดียวกัน ไม่ใช่คนละแบบตามความยาวของประชุม
REQUIRED_SECTIONS = (
    "## หัวข้อที่คุยกัน",
    "## ตกลงแล้ว",
    "## เสนอไว้ ยังไม่สรุป",
    "## Action items",
    "## ความเห็นที่ยังไม่ตรงกัน",
    "## ต้องคุยต่อครั้งหน้า",
    "## คำที่น่าจะถอดเพี้ยน",
)

FULL_SUMMARY_PROMPTS = ("reduce", "single")


def _raw(name: str) -> str:
    return (DEFAULT_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def test_both_full_summary_prompts_ask_for_every_section():
    for name in FULL_SUMMARY_PROMPTS:
        text = _raw(name)
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{name}.md ไม่มีหัวข้อ {section}"


def test_reduce_and_single_ask_for_the_same_sections():
    """กันสิ่งที่แผนกลัวที่สุด: prompt สองไฟล์ที่ต้องให้ผลเหมือนกันแล้ว drift ออกจากกัน
    ประชุมสั้นกับประชุมยาวต้องได้สรุปหน้าตาเดียวกัน"""
    def sections(name):
        return {
            line.strip()
            for line in _raw(name).splitlines()
            if line.strip().startswith("## ")
        }

    assert sections("reduce") == sections("single")


def test_the_no_promotion_rule_is_in_both_full_summary_prompts():
    """failure mode ที่อันตรายที่สุด: มีคนเสนอ "น่าจะย้ายไป X" อีกคนตอบ "อืม"
    แล้วสรุปเขียนว่า "ทีมตัดสินใจย้ายไป X" คนที่ไม่ได้เข้าประชุมเข้าใจผิดทั้งทีม"""
    for name in FULL_SUMMARY_PROMPTS:
        text = _raw(name)
        assert "ห้ามยกระดับ" in text, f"{name}.md ไม่มีกฎห้ามยกระดับข้อเสนอเป็นข้อสรุป"


def test_both_full_summary_prompts_forbid_guessing_dates_and_speakers():
    for name in FULL_SUMMARY_PROMPTS:
        text = _raw(name)
        assert "ห้ามเดาวันที่" in text, f"{name}.md ไม่ห้ามเดากำหนดเวลา"
        assert "ไม่ชัด" in text, f"{name}.md ไม่ได้บอกให้เขียน 'ไม่ชัด' แทนการเดาผู้พูด"


def test_no_prompt_asks_the_model_for_the_meeting_name_or_date():
    """โมเดลเห็นแค่ transcript มันไม่มีทางรู้วันที่หรือชื่อประชุม -- ข้อมูลนั้นอยู่ใน
    ชื่อโฟลเดอร์ meetings/ อยู่แล้ว การขอให้มันเขียนหัวเรื่องคือการเชิญให้เดาวันที่
    ซึ่งขัดกับกฎห้ามเดาวันที่ในไฟล์เดียวกัน"""
    for name in ("map", "reduce", "single"):
        text = _raw(name)
        assert "# สรุปประชุม" not in text, f"{name}.md ขอหัวเรื่องที่โมเดลต้องเดาวันที่"


def test_the_map_prompt_still_forbids_headings():
    """สรุปรายช่วงถูกวางไว้ใต้หัวข้อ ### [ช่วงเวลา] ในไทม์ไลน์ ถ้ามันคืน ## มา
    หัวข้อนั้นจะไปแย่งระดับกับหัวข้อของสรุปรวม"""
    text = _raw("map")
    assert "ห้ามใส่ markdown heading" in text
    for section in REQUIRED_SECTIONS:
        assert section not in text, f"map.md ไม่ควรขอหัวข้อ {section} (มันถูกซ้อนอยู่)"


def test_the_map_prompt_carries_the_agreed_versus_proposed_distinction():
    """ขั้น reduce เห็นแค่สรุปย่อย ไม่เห็น transcript ดิบ ถ้า map ไม่ติดป้ายว่า
    อันไหนตกลงแล้วอันไหนแค่เสนอ reduce จะไม่มีข้อมูลพอจะแยก แล้วมันจะเดา"""
    text = _raw("map")
    assert "ตกลงแล้ว" in text
    assert "เสนอไว้" in text


def test_the_dev_profile_adds_the_blocker_section_only():
    dev = render("map", profile="dev")
    cross = render("map", profile="cross")

    assert "## ติดขัดตรงไหน" in dev
    assert "## ติดขัดตรงไหน" not in cross


def test_the_cross_profile_adds_the_customer_facing_section_only():
    dev = render("map", profile="dev")
    cross = render("map", profile="cross")

    assert "## สิ่งที่ฝ่าย Business สื่อกับลูกค้าได้" in cross
    assert "## สิ่งที่ฝ่าย Business สื่อกับลูกค้าได้" not in dev


def test_the_customer_facing_section_may_only_draw_from_agreed_items():
    """หัวข้อนี้คือของที่จะถูกเอาไปพูดกับลูกค้าจริง การดึงจาก "เสนอไว้" มาใส่
    คือกลไกเดียวกับที่ทำให้ Business ไปสัญญาลูกค้าเรื่องที่ dev ไม่รู้เรื่อง"""
    cross = (DEFAULT_PROMPTS_DIR / "profiles" / "cross.md").read_text(encoding="utf-8")

    assert 'เฉพาะที่อยู่ในหมวด "ตกลงแล้ว"' in cross
    assert 'ห้ามดึงจาก "เสนอไว้"' in cross


def test_the_action_item_table_columns_are_spelled_out():
    for name in FULL_SUMMARY_PROMPTS:
        text = _raw(name)
        for column in ("งาน", "ผู้รับผิดชอบ", "กำหนด", "ความชัดเจน"):
            assert column in text, f"{name}.md ไม่มีคอลัมน์ {column}"
        assert "คาดเดา" in text, f"{name}.md ไม่มีกฎ mark ความชัดเจนเป็น 'คาดเดา'"


def test_backchannel_filtering_is_asked_for_at_the_map_stage():
    """คำรับสั้นๆ แทรกอยู่ใน transcript ดิบ ซึ่งมีแต่ขั้น map ที่เห็น"""
    assert "ครับ" in _raw("map")


def test_reduce_still_asks_to_merge_topics_split_across_chunks():
    text = _raw("reduce")
    assert "หัวข้อเดียวกัน" in text
    assert "เกิดทีหลัง" in text
