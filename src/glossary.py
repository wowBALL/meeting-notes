"""ตัวกรองศัพท์: แก้คำที่ faster-whisper ถอดเพี้ยน ก่อนส่ง transcript ให้ LLM สรุป

สองชั้นตามที่ออกแบบไว้:
  1. แทนที่ตรงๆ ในโค้ด (section `exact` กับ `aliases`) -- ฟรี ชัวร์ ไม่กิน token
  2. ยัดตารางเข้า prompt ให้โมเดลตีความ (`fuzzy`, `project-names`, `ambiguous`)

ไฟล์ทั้งสองเป็น optional: ไม่มี/ว่าง/format เพี้ยน = ทำงานต่อได้โดยไม่กรองอะไร
transcript ที่ถอดด้วย GPU มาแล้วสำคัญกว่าความสวยของสรุป
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.chunk import SEGMENT_PATTERN

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^##\s+(?P<name>[\w-]+)\s*$")

# `## exact` ต้องถูกตรวจก่อนกฎ comment เพราะมันก็ขึ้นต้นด้วย #
_MAPPING_SECTIONS = ("exact", "fuzzy", "project-names", "aliases")


def _wrong_to_correct(mapping: dict[str, list[str]]) -> dict[str, str]:
    return {form: correct for correct, forms in mapping.items() for form in forms}


_HEADER_GROUP = "segment_header"


def _compile(lookup: dict[str, str]) -> re.Pattern | None:
    """regex ตัวเดียวที่ครอบทุกคำผิด -- None เมื่อไม่มีคำเลย

    สองอย่างที่รวมไว้ใน pattern เดียวโดยเจตนา:

    1. เรียงคำผิดยาวก่อนสั้น เพราะ alternation ของ `re` หยุดที่สาขาแรกที่ match ได้
       ถ้าไม่เรียง คำผิดที่สั้นกว่าจะกินคำที่ยาวกว่าไปครึ่งเดียวแล้วเหลือเศษ
    2. เอา SEGMENT_PATTERN มาเป็น "สาขาแรก" เพื่อให้หัว segment
       (`**ผู้พูด 1** [00:00]:`) ถูก match กินไปก่อน แล้วคืนของเดิมใน _fix
       ป้ายผู้พูดจึงไม่ถูกแตะ ทั้งที่ยังแก้เนื้อความในบรรทัดเดียวกันได้ปกติ

    ยิงรอบเดียวจึงกันการไล่แก้เป็นทอดๆ ไปด้วย (ตารางที่มี A->B และ B->C
    ต้องได้ B ไม่ใช่ C) และทำให้ทุกอย่างข้างบนเป็นคุณสมบัติของ pattern เดียว
    ไม่ใช่ลำดับการเรียกที่เผลอสลับกันได้
    """
    if not lookup:
        return None
    ordered = sorted(lookup, key=len, reverse=True)
    wrong_forms = "|".join(re.escape(form) for form in ordered)
    return re.compile(f"(?P<{_HEADER_GROUP}>{SEGMENT_PATTERN.pattern})|{wrong_forms}")


def _substitute(text: str, mapping: dict[str, list[str]], counts: dict[str, int]) -> str:
    lookup = _wrong_to_correct(mapping)
    pattern = _compile(lookup)
    if pattern is None:
        return text

    def _fix(match: re.Match) -> str:
        # ต้องเป็นฟังก์ชัน ไม่ใช่สตริง: re.sub อ่านสตริงแทนที่เป็น template
        # คำถูกที่มี \ หรือ \g<0> จะ raise หรือแทนคำเดิมกลับเงียบๆ
        if match.group(_HEADER_GROUP) is not None:
            return match.group(0)
        correct = lookup[match.group(0)]
        counts[correct] = counts.get(correct, 0) + 1
        return correct

    return pattern.sub(_fix, text)


@dataclass
class Glossary:
    exact: dict[str, list[str]] = field(default_factory=dict)
    fuzzy: dict[str, list[str]] = field(default_factory=dict)
    project_names: dict[str, list[str]] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    ambiguous: list[dict] = field(default_factory=list)
    teams: dict[str, list[str]] = field(default_factory=dict)

    def apply_exact(self, text: str) -> tuple[str, dict[str, int]]:
        """(ข้อความที่แก้แล้ว, จำนวนที่แก้รายคำถูก)

        `exact` ให้จบก่อนแล้วค่อย `aliases`: ชื่อกลางใน aliases เป็นศัพท์อังกฤษที่
        `exact` อาจเพิ่งแก้มา ถ้ายุบชื่อก่อน ตัวที่ยังเพี้ยนจะรอด aliases ไปเงียบๆ

        คำที่แก้ 0 จุดไม่ปรากฏใน counts -- ไม่มีชื่อ = ไม่เจอ
        """
        counts: dict[str, int] = {}
        corrected = _substitute(text, self.exact, counts)
        corrected = _substitute(corrected, self.aliases, counts)
        return corrected, counts

    def count_only(self, text: str) -> dict[str, int]:
        """คำผิดของชั้น fuzzy/project-names โผล่กี่ครั้ง -- ไม่แทนที่อะไรเลย

        สองชั้นนี้โมเดลเป็นคนตีความ ไม่ได้ถูกแก้ในโค้ด จึงไม่มีตัวเลขจาก apply_exact
        ถ้าไม่นับแยกไว้ ตารางในส่วนนี้จะโตทางเดียวไม่มีวันหด เพราะไม่มีหลักฐานว่า
        คำไหนเลิกใช้ไปแล้ว ทั้งที่มันกิน token ทุกครั้งที่สรุป

        ใช้ pattern ชุดเดียวกับ apply_exact จึงข้ามหัว segment เหมือนกันโดยอัตโนมัติ
        """
        merged: dict[str, list[str]] = {}
        for mapping in (self.fuzzy, self.project_names):
            for correct, forms in mapping.items():
                merged.setdefault(correct, []).extend(forms)
        counts: dict[str, int] = {}
        _substitute(text, merged, counts)
        return counts

    def format_for_prompt(self, include_cross_team_context: bool = False) -> str:
        """ตารางศัพท์สำหรับยัดเข้า prompt -- สตริงว่างเมื่อไม่มีอะไรจะบอก

        ไม่ใส่ `exact` กับ `aliases` เพราะถูกแทนที่ในโค้ดไปแล้ว การใส่ซ้ำคือเปลือง
        token เปล่าๆ

        `ambiguous` กับ `teams` เข้าเฉพาะประชุมข้ามฝ่าย: ประชุม dev ล้วนไม่ต้องใช้
        และถ้าใส่ไป โมเดลจะไป qualify คำพูดปกติเกินจำเป็นจนสรุปอ่านแล้วอ้อมค้อม
        """
        blocks: list[str] = []
        if self.fuzzy:
            lines = "\n".join(
                f"- {correct} ← {', '.join(forms)}" for correct, forms in self.fuzzy.items()
            )
            blocks.append(
                "### คำที่อาจถอดเสียงเพี้ยน (ออกเสียงใกล้เคียง = คำนี้ "
                "ไม่แน่ใจให้คงคำเดิม อย่าเดา)\n" + lines
            )
        if self.project_names:
            lines = "\n".join(
                f"- {correct} ← {', '.join(forms)}"
                for correct, forms in self.project_names.items()
            )
            blocks.append("### ชื่อโปรเจกต์ภายใน (คงรูปตามนี้เสมอ)\n" + lines)
        if include_cross_team_context and self.ambiguous and not self.teams:
            # ตาราง ambiguous สั่งให้โมเดล "ตีความตามฝ่ายของผู้พูด" และ cross.md ยัง
            # สั่งให้ไปดูตารางฝ่ายด้วย -- ไม่มี teams.md แปลว่าทั้งสองอย่างชี้ไปที่
            # ข้อมูลที่ไม่มีอยู่ ยังใส่ตารางให้ตามที่ขอ (ดีกว่าไม่มีอะไรเลย) แต่ต้อง
            # เตือนตรงจุดที่มันเกิด ไม่ใช่ปล่อยให้สรุปออกมาเพี้ยนแล้วหาสาเหตุไม่เจอ
            logger.warning(
                "ประชุมข้ามฝ่ายแต่ไม่มี teams.md -- โมเดลจะไม่รู้ว่าใครอยู่ฝ่ายไหน "
                "ตารางคำกำกวม (%d คำ) จึงช่วยได้น้อยมาก คัดลอกจาก teams.example.md "
                "แล้วใส่ชื่อจริง / cross-team meeting without teams.md: the ambiguous "
                "table cannot be applied per-speaker",
                len(self.ambiguous),
            )
        if include_cross_team_context and self.ambiguous:
            lines = "\n".join(
                f"- {entry['term']}: "
                + " | ".join(
                    f"{team} = {meaning}" for team, meaning in entry["meanings"].items()
                )
                for entry in self.ambiguous
            )
            blocks.append(
                "### คำกำกวม ให้ตีความตามฝ่ายของผู้พูด แล้วเขียนคำที่ชัดเจนแทนในสรุป\n"
                + lines
            )
        if include_cross_team_context and self.teams:
            lines = "\n".join(
                f"- {team}: {', '.join(names)}" for team, names in self.teams.items()
            )
            blocks.append("### ฝ่ายของผู้เข้าร่วม\n" + lines)

        if not blocks:
            return ""
        return "## คำศัพท์ในประชุมนี้\n\n" + "\n\n".join(blocks)


def _read_lines(path: Path | None) -> list[str]:
    """บรรทัดในไฟล์ -- ลิสต์ว่างเมื่อไม่มีไฟล์หรืออ่านไม่ได้

    splitlines() ตัด \\r\\n ให้เอง แต่ยัง .strip() ทุก token ทีหลังอีกชั้น
    เพราะไฟล์อาจมี \\r โดดๆ หรือช่องว่างท้ายบรรทัดปนมา
    """
    if path is None:
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        logger.debug("No glossary file at %s, continuing without it", path)
        return []
    except OSError as e:
        logger.warning("Could not read %s (%s), continuing without it", path, e)
        return []


# ต้องมีช่องว่างนำหน้า `#` จึงนับเป็น comment -- `C#` กับ `F#` เป็นชื่อภาษาจริง
# ตัดที่ `#` เฉยๆ จะทำให้คำพวกนั้นใช้ไม่ได้
_INLINE_COMMENT_RE = re.compile(r"\s+#.*$")


def _strip_inline_comment(line: str) -> str:
    return _INLINE_COMMENT_RE.sub("", line).strip()


def _split_forms(raw: str) -> list[str]:
    return [form.strip() for form in raw.split(",") if form.strip()]


def _has_markup(term: str) -> bool:
    return any(char in term for char in "*[]")


def _parse_ambiguous(line: str) -> dict | None:
    """`คำ | Business = ... | dev = ...` -> dict, None เมื่อบรรทัดไม่ครบรูป"""
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 2 or not parts[0]:
        return None
    meanings = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        team, meaning = part.split("=", 1)
        if team.strip() and meaning.strip():
            meanings[team.strip()] = meaning.strip()
    if not meanings:
        return None
    return {"term": parts[0], "meanings": meanings}


def _parse_glossary_file(path: Path | None) -> Glossary:
    glossary = Glossary()
    buckets = {
        "exact": glossary.exact,
        "fuzzy": glossary.fuzzy,
        "project-names": glossary.project_names,
        "aliases": glossary.aliases,
    }
    section = None

    for raw_line in _read_lines(path):
        line = raw_line.strip()
        if not line:
            continue
        heading = _SECTION_RE.match(line)
        if heading:
            section = heading.group("name")
            continue
        if line.startswith("#"):
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        if section in _MAPPING_SECTIONS:
            if ":" not in line:
                logger.warning("Skipping malformed %s line: %r", section, line)
                continue
            correct, forms = line.split(":", 1)
            correct = correct.strip()
            parsed = _split_forms(forms)
            if not correct or not parsed:
                logger.warning("Skipping malformed %s line: %r", section, line)
                continue
            unsafe = [t for t in (correct, *parsed) if _has_markup(t)]
            if unsafe:
                # `*` `[` `]` เป็นอักขระที่ประกอบหัว segment ของ transcript
                # ปล่อยให้คำพวกนี้ถูกแทนที่ลงไป = transcript parse ไม่ออก
                logger.warning(
                    "Skipping %s entry %r: %s contains transcript markup (* [ ])",
                    section,
                    correct,
                    ", ".join(repr(t) for t in unsafe),
                )
                continue
            buckets[section][correct] = parsed
        elif section == "ambiguous":
            entry = _parse_ambiguous(line)
            if entry is None:
                logger.warning("Skipping malformed ambiguous line: %r", line)
                continue
            glossary.ambiguous.append(entry)

    return glossary


def _parse_teams_file(path: Path | None) -> dict[str, list[str]]:
    teams: dict[str, list[str]] = {}
    for raw_line in _read_lines(path):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        if ":" not in line:
            logger.warning("Skipping malformed teams line: %r", line)
            continue
        team, names = line.split(":", 1)
        parsed = _split_forms(names)
        if team.strip() and parsed:
            teams[team.strip()] = parsed
    return teams


def load(glossary_path: Path | None, teams_path: Path | None) -> Glossary:
    glossary = _parse_glossary_file(glossary_path)
    glossary.teams = _parse_teams_file(teams_path)
    return glossary
