"""ตรวจ glossary.md หาคำผิดที่จะไปกินคำอื่น และวัดว่าคำไหนยังมีชีวิตอยู่จริง

`## exact` แทนที่แบบ substring ล้วนและแทนที่ *ก่อน* โมเดลเห็น transcript แปลว่าเมื่อมัน
แก้ผิด ไม่มีใครในสายงานที่เหลือรู้ได้เลยว่าเดิมพูดว่าอะไร -- ทั้งโมเดลและคนอ่านสรุป
ความผิดพลาดชั้นนี้จึงเงียบสนิทโดยธรรมชาติ ไม่มีอะไรฟ้อง

สองเคสที่เจอมาแล้วกับประชุมจริง (2026-08-01, Meet22) และเป็นเหตุผลที่สคริปต์นี้มีอยู่:

  * `Engage` ถูกใส่เป็นคำผิดของ Ingress -- คนพูดว่า "มันจะ Engage ให้ได้ไหม" ถามว่า
    เครื่องมือจะจัดไฟล์ให้เองได้ไหม โมเดลได้รับ "มันจะ Ingress ให้ได้ไหม"
  * ถอด `Control Panel` ออกจากคำผิดของ Control Plane แล้วยังไม่พอ เพราะ `Control Pane`
    ที่เหลือไว้กิน "Control Panel" ไปครึ่งเดียวจนได้ "Control Planel"

ทั้งสองเคสถูกเจอโดยบังเอิญระหว่างไล่เรื่องอื่น สคริปต์นี้ทำให้ไม่ต้องรอความบังเอิญ

รัน: python -m tools.check_glossary
"""

import argparse
import sys
from pathlib import Path

from src.glossary import DuplicateKey, load

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ต่ำกว่านี้ในชั้นที่แทนที่จริง = เกือบแน่ว่าจะไปโดนกลางคำอื่น ตัวเลขนี้มาจากกฎที่เขียนไว้
# ใน header ของ glossary.md เอง ("คำผิดสั้นๆ ต่ำกว่า ~4 อักขระ จะไปโดนกลางคำอื่น")
# สคริปต์นี้แค่บังคับใช้กฎที่ไฟล์ประกาศไว้แล้ว ไม่ได้ตั้งกฎใหม่
#
# เป็นแค่ heuristic โดยประมาณ ไม่ใช่การรับประกัน: "ครูป" ยาวพอดี 4 อักขระ ผ่านเช็คนี้
# ไปได้ แต่ยังชนจริงกับ "ครูประกาศ" (ชื่อคน ไม่ใช่คำถูกในตารางเลย เจอใน PCI 2026-08-03)
# -- find_collisions() เช็คได้แค่ชนกับ "คำถูก" ที่รู้จักในตาราง ไม่เช็คคำไทยทั่วไปที่ไม่ได้
# อยู่ในตารางเลย ภาษาไทยไม่มีพจนานุกรมคำต้องห้ามให้เทียบ จึงไม่มีทางเช็คอัตโนมัติให้ครบ
# ทางเดียวที่จับพวกนี้ได้คือให้คนเห็นบริบทจริงแล้วตัดสินเอง (ดู sample_context ด้านล่าง)
MIN_SAFE_LENGTH = 4

# ความกว้างของบริบทที่โชว์รอบคำผิดแต่ละครั้ง (อักขระ) -- พอให้อ่านออกว่าเป็นคำถูกจริง
# หรือไปกินคำอื่นเงียบๆ โดยไม่ต้องเปิดไฟล์ transcript เต็มไฟล์
CONTEXT_CHARS = 25
MAX_EXAMPLES_PER_WRONG = 2


def _replacing_layers(glossary) -> dict[str, dict[str, list[str]]]:
    """ชั้นที่ apply_exact แทนที่จริง -- fuzzy/project-names ไม่อยู่ในนี้โดยเจตนา

    สองชั้นนั้นโมเดลเป็นคนตีความตามบริบท คำที่มีความหมายของตัวเองจึงปลอดภัยเมื่ออยู่
    ที่นั่น (นั่นคือเหตุผลทั้งหมดที่ชั้นนั้นมีอยู่) การเตือนเรื่องมันจึงเป็นการเตือนผิดที่
    """
    return {"exact": glossary.exact, "aliases": glossary.aliases}


def find_collisions(glossary) -> list[tuple[str, str]]:
    """(ระดับความรุนแรง, ข้อความ) ของคำผิดที่จะไปกินคำอื่น

    ตรวจสองแบบ:

    1. คำผิดที่เป็น substring ของ "คำถูก" ตัวใดตัวหนึ่ง -- ร้ายแรงที่สุด เพราะแปลว่า
       ทุกครั้งที่มีคนพูดคำถูกนั้นออกมาถูกต้องแล้ว มันจะถูกแก้ให้เพี้ยน ตรงข้ามกับ
       หน้าที่ของตารางนี้ทั้งหมด
    2. คำผิดที่เป็น substring ของคำผิดอีกตัวที่ชี้ไปคนละคำถูก -- _compile เรียงยาว
       ก่อนสั้นอยู่แล้ว ตัวยาวจึงชนะเสมอและผลลัพธ์ไม่กำกวม แต่ต้องรายงานเพราะมันแปลว่า
       ตัวสั้นจะไม่มีวันทำงานในบริบทนั้น ซึ่งมักเป็นสัญญาณว่าใส่ซ้ำโดยไม่ตั้งใจ
    """
    layers = _replacing_layers(glossary)
    forms: list[tuple[str, str, str]] = []  # (คำผิด, คำถูก, ชั้น)
    for layer, mapping in layers.items():
        for correct, wrongs in mapping.items():
            for wrong in wrongs:
                forms.append((wrong, correct, layer))

    # คำถูกจากทุกชั้น รวม fuzzy ด้วย: คำถูกใน fuzzy ก็เป็นคำที่คนพูดออกมาถูกได้เหมือนกัน
    # และถ้าคำผิดของ exact ไปกินมัน ความเสียหายก็เท่ากัน
    all_correct = set()
    for mapping in (
        glossary.exact,
        glossary.aliases,
        glossary.fuzzy,
        glossary.project_names,
    ):
        all_correct.update(mapping)

    findings: list[tuple[str, str]] = []
    for wrong, correct, layer in forms:
        for target in all_correct:
            if wrong != target and wrong in target:
                findings.append(
                    (
                        "หนัก",
                        f'[{layer}] "{wrong}" (→ {correct}) อยู่ข้างใน "{target}" '
                        f'-- ใครพูด "{target}" ถูกอยู่แล้วจะโดนแก้ให้เพี้ยน',
                    )
                )
        for other_wrong, other_correct, other_layer in forms:
            if wrong == other_wrong or correct == other_correct:
                continue
            if wrong in other_wrong:
                findings.append(
                    (
                        "เบา",
                        f'[{layer}] "{wrong}" (→ {correct}) อยู่ข้างใน '
                        f'"{other_wrong}" (→ {other_correct}) ของชั้น {other_layer} '
                        f"-- ตัวยาวชนะเสมอ ตัวสั้นจะไม่ทำงานในบริบทนั้น",
                    )
                )

    for wrong, correct, layer in forms:
        if len(wrong.strip()) < MIN_SAFE_LENGTH:
            findings.append(
                (
                    "เบา",
                    f'[{layer}] "{wrong}" (→ {correct}) สั้นกว่า {MIN_SAFE_LENGTH} '
                    f"อักขระ -- ภาษาไทยไม่มีช่องว่างระหว่างคำ มันจะไปโดนกลางคำอื่น",
                )
            )
        if wrong == correct:
            findings.append(
                ("เบา", f'[{layer}] "{wrong}" ชี้ไปหาตัวมันเอง -- ลบทิ้งได้')
            )

    return findings


def format_duplicate_keys(duplicates: list[DuplicateKey]) -> list[str]:
    """ข้อความอ่านง่ายจาก glossary.duplicate_keys -- แค่ชี้ตำแหน่ง ไม่ใช่บั๊กร้ายแรง

    src/glossary.py รวมฟอร์มจากทุกบรรทัดที่ประกาศคำถูกตัวเดียวกันซ้ำให้อัตโนมัติ
    แล้ว (ดู _parse_glossary_file) ข้อมูลจึงไม่หายอีกต่อไปแม้ไม่รัน check ตัวนี้เลย
    ฟังก์ชันนี้จึงแค่บอกว่าควรไปรวมบรรทัดในไฟล์ให้เหลือบรรทัดเดียวเพื่อความสะอาด
    ไม่ใช่คำเตือนที่ควรบล็อกอะไร
    """
    messages = []
    for dup in duplicates:
        line_list = ", ".join(str(n) for n in dup.lines)
        messages.append(
            f'[{dup.section}] "{dup.term}" ประกาศซ้ำที่บรรทัด {line_list} -- '
            "โหลดแล้วรวมฟอร์มให้อัตโนมัติ แต่ควรรวมเป็นบรรทัดเดียวในไฟล์เพื่อความสะอาด"
        )
    return messages


def count_in_transcripts(glossary, meetings_dir: Path) -> dict[str, tuple[int, int]]:
    """คำผิดแต่ละตัวโผล่ในประชุมจริงกี่ครั้ง ในกี่ประชุม

    transcript.md เป็นของดิบ ตัวกรองศัพท์ไม่แตะมัน (ดู pipeline._summarize_and_save)
    ตัวเลขนี้จึงเป็นสิ่งที่ whisper ถอดออกมาจริง ๆ ไม่ใช่ผลหลังแก้

    header ของ glossary.md เขียนไว้เองว่าคำผิดในไฟล์เป็นการ "เดา" ว่า whisper จะเพี้ยน
    แบบไหน และให้ปรับตามของจริงที่เจอ -- ฟังก์ชันนี้คือของจริงที่ว่านั้น
    """
    transcripts = sorted(meetings_dir.glob("*/transcript.md"))
    tally: dict[str, tuple[int, int]] = {}
    for layer_mapping in (
        glossary.exact,
        glossary.aliases,
        glossary.fuzzy,
        glossary.project_names,
    ):
        for wrongs in layer_mapping.values():
            for wrong in wrongs:
                tally.setdefault(wrong, (0, 0))
    for path in transcripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        for wrong in tally:
            hits = text.count(wrong)
            if hits:
                total, meetings = tally[wrong]
                tally[wrong] = (total + hits, meetings + 1)
    return tally


def sample_context(glossary, meetings_dir: Path) -> dict[str, list[str]]:
    """ตัวอย่างข้อความรอบคำผิดชั้น exact/aliases ที่เจอจริงใน transcript

    เฉพาะชั้น exact/aliases เท่านั้น (ดู _replacing_layers) -- fuzzy/project-names
    ให้โมเดลตัดสินจากบริบทเองอยู่แล้ว โชว์บริบทของชั้นนั้นจะเป็นเสียงเตือนผิดที่
    เหมือนที่ find_collisions ตั้งใจไม่ทำกับมัน

    เจตนาคือปิดช่องที่ find_collisions ปิดไม่ได้: มันเช็คได้แค่คำผิดชนกับ "คำถูก" ที่
    ประกาศไว้ในตารางเอง แต่คำผิดสั้นๆ ไปกินกลางคำไทยทั่วไปที่ไม่ได้อยู่ในตารางเลยก็ได้
    (เคสจริง: "ครูป" ของ Kubernetes ไปกินกลาง "ครูประกาศ" ซึ่งเป็นชื่อคน ไม่ใช่คำถูก
    ของใครในตารางนี้สักคำ -- find_collisions ผ่านเคสนี้เพราะไม่มีอะไรให้เทียบชน)
    ไม่มีพจนานุกรมคำไทยให้เทียบอัตโนมัติได้ครบ ทางเดียวที่เหลือคือให้คนเห็นบริบทจริง
    แล้วตัดสินเอง
    """
    layers = _replacing_layers(glossary)
    wrongs: set[str] = set()
    for mapping in layers.values():
        for forms in mapping.values():
            wrongs.update(forms)

    transcripts = sorted(meetings_dir.glob("*/transcript.md"))
    snippets: dict[str, list[str]] = {}
    for path in transcripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        for wrong in wrongs:
            if len(snippets.get(wrong, [])) >= MAX_EXAMPLES_PER_WRONG:
                continue
            start = 0
            while len(snippets.get(wrong, [])) < MAX_EXAMPLES_PER_WRONG:
                idx = text.find(wrong, start)
                if idx == -1:
                    break
                lo = max(0, idx - CONTEXT_CHARS)
                hi = min(len(text), idx + len(wrong) + CONTEXT_CHARS)
                snippet = text[lo:hi].replace("\n", " ")
                snippets.setdefault(wrong, []).append(snippet)
                start = idx + len(wrong)
    return snippets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tools.check_glossary")
    parser.add_argument(
        "--glossary", default=str(PROJECT_ROOT / "glossary.md"), help="ไฟล์ glossary"
    )
    parser.add_argument(
        "--teams", default=str(PROJECT_ROOT / "teams.md"), help="ไฟล์ teams"
    )
    parser.add_argument(
        "--meetings",
        default=str(PROJECT_ROOT / "meetings"),
        help="โฟลเดอร์ประชุม ใช้นับว่าคำผิดตัวไหนเคยโผล่จริง",
    )
    parser.add_argument(
        "--show-dead",
        action="store_true",
        help="แสดงคำผิดที่ไม่เคยโผล่ในประชุมไหนเลย (ค่าเริ่มต้นแสดงแค่จำนวน)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    glossary = load(Path(args.glossary), Path(args.teams))

    findings = find_collisions(glossary)
    heavy = [message for level, message in findings if level == "หนัก"]
    light = [message for level, message in findings if level == "เบา"]

    if heavy:
        print(f"คำผิดที่จะไปกินคำถูก ({len(heavy)}):")
        for message in heavy:
            print(f"  ✗ {message}")
        print()
    if light:
        print(f"ควรดู ({len(light)}):")
        for message in light:
            print(f"  ! {message}")
        print()
    if not findings:
        print("ไม่เจอคำผิดที่กินคำอื่น\n")

    # ไม่นับรวมกับ heavy/light: src/glossary.py รวมฟอร์มให้อัตโนมัติแล้ว ข้อมูลไม่หาย
    # จึงไม่ควรทำให้ exit code ไม่เป็นศูนย์ -- แค่ชี้ตำแหน่งให้คนไปรวมบรรทัดเอง
    duplicate_messages = format_duplicate_keys(glossary.duplicate_keys)
    if duplicate_messages:
        print(f"เขียนซ้ำในไฟล์ ({len(duplicate_messages)}):")
        for message in duplicate_messages:
            print(f"  ↻ {message}")
        print()

    meetings_dir = Path(args.meetings)
    if meetings_dir.is_dir():
        tally = count_in_transcripts(glossary, meetings_dir)
        transcripts = sorted(meetings_dir.glob("*/transcript.md"))
        alive = {w: v for w, v in tally.items() if v[0]}
        dead = sorted(w for w, v in tally.items() if not v[0])
        print(f"วัดกับ transcript จริง {len(transcripts)} ประชุม:")
        print(f"  เคยโผล่จริง {len(alive)} คำ / ไม่เคยโผล่เลย {len(dead)} คำ")

        # เฉพาะ exact/aliases เท่านั้น: ชั้นที่แทนที่แบบเงียบๆ ก่อนโมเดลเห็น จึงเป็นชั้น
        # เดียวที่ต้องให้คนตรวจบริบทเอง (ดู docstring ของ sample_context)
        risky_wrongs: set[str] = set()
        for mapping in _replacing_layers(glossary).values():
            for forms in mapping.values():
                risky_wrongs.update(forms)
        context = sample_context(glossary, meetings_dir) if risky_wrongs else {}

        for wrong, (hits, meetings) in sorted(
            alive.items(), key=lambda kv: -kv[1][0]
        )[:15]:
            print(f'    {hits:>4} ครั้ง / {meetings} ประชุม  "{wrong}"')
            if wrong in risky_wrongs:
                for snippet in context.get(wrong, []):
                    print(f'          เช่น: ...{snippet}...')
        if args.show_dead and dead:
            print("\n  ไม่เคยโผล่เลย -- ในชั้น fuzzy พวกนี้กิน token ทุกครั้งที่สรุป:")
            for wrong in dead:
                print(f'    "{wrong}"')
        elif dead:
            print("    (ดูรายการที่ไม่เคยโผล่ด้วย --show-dead)")
    else:
        print(f"ข้ามการวัด: ไม่มีโฟลเดอร์ {meetings_dir}")

    # exit 1 เฉพาะเคสที่ทำให้สรุปผิดแน่ ๆ -- ของที่แค่ "ควรดู" ไม่ควรทำให้คนเลิกรัน
    return 1 if heavy else 0


if __name__ == "__main__":
    sys.exit(main())
