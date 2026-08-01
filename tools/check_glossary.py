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

from src.glossary import load

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ต่ำกว่านี้ในชั้นที่แทนที่จริง = เกือบแน่ว่าจะไปโดนกลางคำอื่น ตัวเลขนี้มาจากกฎที่เขียนไว้
# ใน header ของ glossary.md เอง ("คำผิดสั้นๆ ต่ำกว่า ~4 อักขระ จะไปโดนกลางคำอื่น")
# สคริปต์นี้แค่บังคับใช้กฎที่ไฟล์ประกาศไว้แล้ว ไม่ได้ตั้งกฎใหม่
MIN_SAFE_LENGTH = 4


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

    meetings_dir = Path(args.meetings)
    if meetings_dir.is_dir():
        tally = count_in_transcripts(glossary, meetings_dir)
        transcripts = sorted(meetings_dir.glob("*/transcript.md"))
        alive = {w: v for w, v in tally.items() if v[0]}
        dead = sorted(w for w, v in tally.items() if not v[0])
        print(f"วัดกับ transcript จริง {len(transcripts)} ประชุม:")
        print(f"  เคยโผล่จริง {len(alive)} คำ / ไม่เคยโผล่เลย {len(dead)} คำ")
        for wrong, (hits, meetings) in sorted(
            alive.items(), key=lambda kv: -kv[1][0]
        )[:15]:
            print(f'    {hits:>4} ครั้ง / {meetings} ประชุม  "{wrong}"')
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
