import logging
import tempfile
from pathlib import Path
from typing import Any

from src import activity
from src.audio_convert import convert_to_wav
from src.config import Config
from src.diarize import diarize_audio
from src.job import (
    NO_SUMMARY_MODEL,
    discard_job,
    read_model,
    read_profile,
    read_transcript,
    record_transcript,
)
from src.merge import merge_transcript_and_speakers
from src.pending import build_pending_speakers, write_pending
from src.render import (
    build_speaker_labels,
    render_transcript_markdown,
    replace_participants_line,
    speaker_table,
)
from src.retry import retry_with_backoff
from src.speaker_guess import guess_speaker_names
from src.speakers import Match, load_registry, match_known
from src.storage import (
    archive_audio,
    create_meeting_folder,
    move_to_failed,
    save_summary,
    save_transcript,
)
from src import carryover
from src.glossary import load as load_glossary
from src.prompts import CROSS_TEAM_PROFILE
from src.summarize import check_model_reachable, summarize_transcript
from src.transcribe import transcribe_audio
from src.voiceprint import Voiceprint, extract_voiceprints, load_embedder

logger = logging.getLogger(__name__)


def _reuse_saved_transcript(audio_path: Path) -> tuple[Path, Path, str] | None:
    """(meeting dir, transcript path, transcript) left by an earlier run, or None.

    A recording only comes back through here after a run that got as far as saving
    the transcript and then failed -- almost always at the summary. Transcribing it
    again would spend another full GPU pass to reproduce a file that is already on
    disk, byte for byte. Anything unexpected returns None and the caller does the
    whole pipeline, which is exactly the old behaviour.
    """
    transcript_path = read_transcript(audio_path)
    if transcript_path is None:
        return None
    try:
        transcript_markdown = transcript_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Cannot read the saved transcript %s, transcribing again: %s",
            transcript_path,
            e,
        )
        return None
    if not transcript_markdown.strip():
        logger.warning(
            "The saved transcript %s is empty, transcribing again", transcript_path
        )
        return None
    logger.info("Reusing the transcript from an earlier run: %s", transcript_path)
    return transcript_path.parent, transcript_path, transcript_markdown


def _match_known_speakers(
    voiceprints: dict[str, Voiceprint], config: Config, job: str, embedding_model: str
) -> dict[str, Match]:
    """ผู้พูดในไฟล์นี้ที่ตรงกับคนในทะเบียนเสียง

    ล้มเหลวแล้วคืน dict ว่าง ไม่ปล่อย exception ขึ้นไป: ผลลัพธ์ที่แย่ที่สุดของฟีเจอร์นี้
    ต้องเท่ากับสภาพก่อนมีมัน (ป้าย "ผู้พูด N") ไม่ใช่การประชุมที่หายไปทั้งครั้ง

    `embedding_model` มาจาก embedder ที่คำนวณเวกเตอร์ชุดนี้จริง ไม่ใช่ config.embedding_model
    -- watcher ถือโมเดลค้างในหน่วยความจำข้าม .env ที่ผู้ใช้แก้ระหว่างนั้นได้ ป้ายที่ผิดแปลว่า
    เทียบข้ามพื้นที่เวกเตอร์โดยไม่มีอะไรเตือน (ดู speakers.match_known)

    embedding_model ว่างเปล่าพร้อมกับ voiceprints ไม่ว่างเป็นไปไม่ได้ -- ไม่ใช่เพราะ
    embedder ตัวจริงมีอยู่แล้วรับประกัน .checkpoint เสมอ (ไม่มีอะไรบังคับแบบนั้น
    embedder เป็นแค่ Any ที่สนองสัญญา embed() เท่านั้น) กรณีนี้เคยหลุดผ่านมาได้จริงสองทาง:
    ทาง checkpoint ว่างเปล่า/ไม่ใช่สตริง (_record_pending_speakers เขียน embedding_model=""
    ลงคิวรอตั้งชื่อ ซึ่ง speakers.add_sample ปฏิเสธถาวรทีหลัง -- Minor 1 รีวิวรอบสอง) และทาง
    except Exception ตัวนอกสุดของบล็อกที่คำนวณ voiceprints ซึ่งเดิมจับได้แค่ AttributeError
    จาก getattr(embedder, "checkpoint", "") แต่ปล่อย exception ชนิดอื่นให้หลุดผ่านไปโดยไม่
    ล้าง voiceprints/embedding_model เลย เช่น .checkpoint เป็น property ที่ raise
    RuntimeError เวลาอ่าน (รีวิวรอบสี่) ทั้งสองทางตอนนี้เป็นไปไม่ได้จริง เพราะจุดที่คำนวณ
    voiceprints (ท่อหลักใน process_file) ล้าง voiceprints ทิ้งพร้อมกับ embedding_model เป็น
    "" ทุกครั้งที่มีอะไรผิดพลาดระหว่างคำนวณหรือตรวจสอบ checkpoint -- ทั้งกรณี checkpoint
    ที่ใช้ไม่ได้โดยเฉพาะ (ว่างเปล่า/ไม่ใช่สตริง) และกรณี exception อื่นใดก็ตามที่หลุดออกมา
    จาก try ทั้งก้อน ไม่ใช่เฉพาะ AttributeError เช็ค `if not voiceprints` ด้านล่างจึงยังกัน
    พารามิเตอร์นี้ว่างเปล่าไปในตัวโดยไม่ต้องเช็คซ้ำ แต่กันได้เพราะมีการล้างไว้ต้นทางครอบ
    ทุกเส้นทางความล้มเหลว ไม่ใช่กันได้ฟรี ๆ
    """
    if not voiceprints:
        return {}
    try:
        matches = match_known(
            {label: voiceprint.embedding for label, voiceprint in voiceprints.items()},
            load_registry(config.base_dir),
            high=config.speaker_match_high,
            low=config.speaker_match_low,
            embedding_model=embedding_model,
        )
    except Exception as e:
        logger.warning("จับคู่เสียงกับทะเบียนไม่สำเร็จ ไปต่อโดยไม่ใส่ชื่อ: %s", e)
        activity.append(config.base_dir, job, "speakers_failed", "warn", {"error": str(e)})
        return {}
    recognized = sum(1 for match in matches.values() if match.confident)
    if recognized:
        activity.append(
            config.base_dir, job, "speakers_matched", params={"count": recognized}
        )
    return matches


def _record_pending_speakers(
    config: Config,
    job: str,
    meeting_dir: Path,
    audio_file: str,
    transcript_markdown: str,
    claude_model: str,
    merged: list[dict],
    labels: dict[str, str],
    voiceprints: dict[str, Voiceprint],
    embedding_model: str,
    matches: dict[str, Match],
) -> None:
    """งานหลังบ้านที่รันหลังการประชุมเสร็จสมบูรณ์แล้ว

    ทุกอย่างในนี้เป็นของแถม: ไฟล์ที่ผู้ใช้รออยู่ถูกเซฟไปหมดแล้วก่อนถูกเรียก และ
    เหตุการณ์ meeting_done ถูกบันทึกไปแล้ว จึงห้ามมีอะไรในนี้ raise ออกไป

    การเรียกโมเดลเพื่อเดาชื่ออยู่ตรงนี้ ไม่ใช่ก่อนหน้า เพราะมันคือ call เดียวที่เพิ่ม
    เข้ามาในท่อทั้งหมด -- วางไว้ก่อน save จะทำให้เวลาที่ผู้ใช้รอ transcript ยาวขึ้น
    เพื่อสิ่งที่เขาจะมาดูทีหลังเมื่อไหร่ก็ได้

    `embedding_model` ติดไปกับทุกคนในคิว (ดู pending.build_pending_speakers) เพราะคิวนี้
    อยู่ข้ามวันได้และผู้ใช้แก้ .env ก่อนกลับมากดตั้งชื่อได้ -- ต้องเป็นค่าจาก embedder
    ตัวจริงที่คำนวณ voiceprints ชุดนี้ ไม่ใช่ config.embedding_model ณ ตอนนี้
    """
    try:
        candidates = build_pending_speakers(
            merged,
            labels,
            voiceprints,
            config.diarization_model,
            embedding_model,
            matches=matches,
        )
        if not candidates:
            return
        try:
            guesses = guess_speaker_names(
                transcript_markdown,
                [candidate["label"] for candidate in candidates],
                model=claude_model,
            )
        except Exception as e:
            logger.warning("เดาชื่อผู้พูดไม่สำเร็จ ไปต่อโดยไม่มีคำใบ้: %s", e)
            guesses = {}
        for candidate in candidates:
            candidate["guess"] = guesses.get(candidate["label"])
        write_pending(config.base_dir, meeting_dir.name, audio_file, candidates)
        activity.append(
            config.base_dir, job, "speakers_pending", params={"count": len(candidates)}
        )
    except Exception as e:
        logger.warning("บันทึกรายการผู้พูดที่รอตั้งชื่อไม่สำเร็จ: %s", e)
        activity.append(config.base_dir, job, "speakers_failed", "warn", {"error": str(e)})


def process_file(
    audio_path: Path,
    config: Config,
    diarization_pipeline: Any = None,
    whisper_model: Any = None,
    embedder: Any = None,
) -> Path:
    # The recorder wrote this next to the audio; the watcher's own config was read
    # once at startup and cannot know what this meeting asked for.
    claude_model = read_model(audio_path) or config.claude_model
    # กลไกเดียวกันเป๊ะ: ค่าที่เลือกไว้ตอนอัดชนะค่าใน .env เพราะคิวอยู่ข้ามวันได้
    # และผู้ใช้แก้ .env ระหว่างนั้นได้ ประชุมที่ค้างอยู่ต้องยังใช้ค่าที่เลือกไว้จริง
    # ไม่มีค่า = ไฟล์ที่ลากใส่ inbox/ เอง หรือ .job.json ที่เขียนก่อนมีฟีเจอร์นี้
    profile = read_profile(audio_path) or config.meeting_profile

    # ทุก activity.append ที่นี่ไม่มีทาง raise (ดู src/activity.py) จึงไม่ห่อ try --
    # ห่อแล้วจะบังคับให้คนอ่านคิดว่ามันอาจ raise ได้ ซึ่งไม่จริง
    job = audio_path.stem
    activity.append(config.base_dir, job, "queued")

    reused = _reuse_saved_transcript(audio_path)
    if reused is not None:
        meeting_dir, transcript_path, transcript_markdown = reused
        return _finish_meeting(
            audio_path,
            config,
            claude_model,
            meeting_dir,
            transcript_path,
            transcript_markdown,
            profile,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        try:
            convert_to_wav(audio_path, wav_path)
        except Exception as e:
            activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
            move_to_failed(audio_path, config.failed_dir, f"Audio conversion failed: {e}")
            raise

        # โหลด glossary รอบของตัวเองตรงนี้ ไม่ใช้ตัวที่ _finish_meeting โหลด สองเหตุผล:
        # (1) ตัวนั้นอยู่ในบล็อกที่ถูกข้ามทั้งก้อนเมื่อ claude_model เป็น transcript-only
        #     ซึ่งเป็นโหมดที่ transcript คือผลลัพธ์สุดท้ายและต้องการ hotwords มากที่สุด
        # (2) _finish_meeting ถูกเรียกจาก path ลองใหม่ที่ไม่ถอดเสียงเลยด้วย ย้ายการโหลด
        #     ออกมาข้างนอกจึงทำให้ path นั้นโหลดของที่ไม่ได้ใช้
        # ไฟล์เล็กและอ่านครั้งเดียวต่อประชุม ราคาของการอ่านซ้ำจึงถูกกว่าการผูกสองขั้นนี้
        # เข้าด้วยกัน -- ขั้นถอดเสียงกับขั้นสรุปใช้ตารางเดียวกันคนละวัตถุประสงค์
        hotwords = ""
        if config.whisper_hotwords:
            hotwords = load_glossary(
                config.base_dir / "glossary.md", config.base_dir / "teams.md"
            ).hotwords_text()

        activity.append(config.base_dir, job, "transcribe_started")
        try:
            whisper_segments = retry_with_backoff(
                lambda: transcribe_audio(
                    wav_path,
                    model_size=config.whisper_model,
                    model=whisper_model,
                    batched=config.whisper_batched,
                    # สตริงว่าง (ไม่มี glossary.md) ต้องกลายเป็น None ไม่ใช่ "" --
                    # faster-whisper เปิดช่อง sot_prev ให้ทันทีที่ hotwords truthy
                    # ค่าว่างจึงเป็นการจ่าย token ทิ้งเปล่า ๆ ทุกหน้าต่าง
                    hotwords=hotwords or None,
                    condition_on_previous_text=(
                        config.whisper_condition_on_previous_text
                    ),
                )
            )
        except Exception as e:
            activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
            move_to_failed(audio_path, config.failed_dir, f"Transcription failed: {e}")
            raise

        activity.append(config.base_dir, job, "diarize_started")
        diarization_failed = False
        voiceprints: dict[str, Voiceprint] = {}
        embedding_model = ""
        try:
            diarization = diarize_audio(
                wav_path, hf_token=config.hf_token, pipeline=diarization_pipeline
            )
            speaker_turns = diarization.turns
        except Exception as e:
            logger.warning("Diarization failed, continuing without speaker labels: %s", e)
            activity.append(
                config.base_dir, job, "diarize_failed", "warn", {"error": str(e)}
            )
            speaker_turns = []
            diarization_failed = True

        # อยู่นอก try ของ diarization โดยเจตนา และอยู่ใน with ของ wav_path เพราะต้องอ่าน
        # ไฟล์เดียวกัน -- ความล้มเหลวของการจำเสียงต้องไม่ทำให้ speaker_turns ที่ได้มาแล้ว
        # หายไป (กฎเดิมของ repo: "การจำเสียง" ล้มเหลวได้ แต่ต้องไม่ทำลาย "การแยกผู้พูด"
        # ของประชุมที่อัดซ้ำไม่ได้ -- การป้องกันนี้เคยอยู่ใน diarize._speaker_embeddings
        # ก่อนโมดูลนั้นถูกลบทิ้งไปตอนย้าย voiceprint ออกมาเป็นโมเดล speaker verification
        # แยกต่างหาก ดู src/voiceprint.py)
        #
        # extract_voiceprints ไม่ raise อยู่แล้วโดยสัญญาของมันเอง (ดู docstring ของมัน)
        # แต่ยังกัน try ไว้อีกชั้นเผื่อสัญญานั้นเปลี่ยนในอนาคต หรือ load_embedder เองพัง
        # (โหลดโมเดลไม่ได้/hf_token ผิด) ซึ่งไม่มีสัญญาแบบเดียวกันเลย -- ทั้งสองอย่างต้อง
        # ไม่ทำให้ speaker_turns ที่ได้มาแล้วหายไปด้วย
        if speaker_turns:
            try:
                if embedder is None:
                    # โหลดครั้งเดียวเมื่อไม่มีใครส่งมาให้ ไม่ใช่โหลดใหม่ทุกไฟล์ -- ทางปกติ
                    # (main.py) โหลดไว้ก่อนแล้วส่งเข้ามา เส้นทางนี้มีไว้สำหรับผู้เรียกที่
                    # ไม่ได้ถือโมเดลไว้เอง (เทสต์/สคริปต์แยก)
                    embedder = load_embedder(config.hf_token, config.embedding_model)
                voiceprints = extract_voiceprints(wav_path, speaker_turns, embedder)
                # getattr, not embedder.checkpoint: docstring ของ process_file เองผูก
                # พารามิเตอร์ embedder ไว้แค่สัญญา embed(waveform, intervals) -> list
                # ของ extract_voiceprints เท่านั้น ไม่ได้บังคับว่าต้องมี .checkpoint
                # ผู้เรียกที่ทำตามสัญญานั้นแต่ไม่มี attribute นี้ต้องไม่พังตรงนี้ -- ทั้ง
                # บล็อกนี้มีไว้เพื่อให้ voiceprint ที่ล้มเหลวลดขั้นเหลือ "ผู้พูด N" แทนที่จะ
                # ทิ้งรอบถอดเสียงที่เสร็จไปแล้วทั้งหมด
                embedding_model = getattr(embedder, "checkpoint", "")
                if not isinstance(embedding_model, str) or not embedding_model.strip():
                    # รีวิวรอบสาม: เช็คของรอบสอง (`if not embedding_model or not
                    # embedding_model.strip()`) เรียก .strip() โดยไม่เช็ค type ก่อน --
                    # embedder.checkpoint ที่ truthy แต่ไม่ใช่สตริง (เช่น int, list) ทำให้
                    # .strip() ยิง AttributeError หลุดไปถึง except Exception ตัวนอกสุดของ
                    # บล็อกนี้ ซึ่งเขียน log voiceprint_failed เหมือนกันแต่ "ไม่" ล้าง
                    # voiceprints -- checkpoint ที่ไม่ใช่สตริงจึงเดินทางต่อไปถึง
                    # _record_pending_speakers ได้เหมือนกับที่ Minor 1 ของรีวิวรอบสองเคยแก้
                    # ไปเฉพาะกรณีสตริงว่างเปล่า เช็คด้วย isinstance ก่อนเรียก .strip() เสมอ
                    # จึงครอบทั้งสองกรณี (ว่างเปล่า และไม่ใช่สตริงเลย) ในเงื่อนไขเดียว
                    #
                    # ล้าง voiceprints/embedding_model ก่อน log โดยเจตนา (ไม่ใช่หลัง เหมือน
                    # โค้ดรอบก่อน): activity.append กลืนเฉพาะ OSError ไว้ข้างใน (ดู
                    # src/activity.py) exception ชนิดอื่นที่หลุดออกมาจาก logger.warning หรือ
                    # activity.append เองต้องไม่ทิ้ง voiceprints ที่มี checkpoint เสียไว้ค้าง
                    # อยู่ -- invariant "checkpoint ใช้ไม่ได้ = voiceprints ว่างเปล่าเสมอ"
                    # ต้องเป็นจริงไม่ว่า logging จะพังหรือไม่ก็ตาม
                    voiceprints = {}
                    embedding_model = ""
                    logger.warning(
                        "embedder คืน checkpoint ที่ใช้ไม่ได้ (ว่างเปล่าหรือไม่ใช่สตริง) "
                        "ถือว่า voiceprint ล้มเหลว ไปต่อโดยไม่จำเสียง"
                    )
                    activity.append(
                        config.base_dir,
                        job,
                        "voiceprint_failed",
                        "warn",
                        {"error": "embedder คืน checkpoint ที่ใช้ไม่ได้"},
                    )
            except Exception as e:
                voiceprints = {}
                embedding_model = ""
                logger.warning("สร้าง voiceprint ไม่สำเร็จ ไปต่อโดยไม่จำเสียง: %s", e)
                activity.append(
                    config.base_dir, job, "voiceprint_failed", "warn", {"error": str(e)}
                )

    matches = _match_known_speakers(voiceprints, config, job, embedding_model)

    try:
        merged = merge_transcript_and_speakers(whisper_segments, speaker_turns)
        speaker_names = {
            label: match.name for label, match in matches.items() if match.confident
        }
        speaker_labels = build_speaker_labels(merged, speaker_names)
        transcript_markdown = render_transcript_markdown(
            merged, diarization_failed=diarization_failed, speaker_names=speaker_names
        )
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Rendering failed: {e}")
        raise

    # Written before summarizing: the transcript is the expensive artifact of
    # this pipeline (a full GPU pass over the recording), and summarization is
    # the step most likely to fail. Persisting it first means a failed summary
    # costs one summarizer call to redo, not another transcription.
    try:
        meeting_dir = create_meeting_folder(audio_path, config.meetings_dir)
        transcript_path = save_transcript(meeting_dir, transcript_markdown)
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    # Saving it is not enough to reuse it: a retry has to be able to find it. The
    # folder name cannot be recomputed for a named recording, whose date comes
    # from the day the folder was made rather than from the file, so the path is
    # written down next to the audio and travels with it into failed/.
    record_transcript(audio_path, transcript_path)

    # อ่านชื่อไฟล์ไว้ตรงนี้เพื่อความชัดเจน -- _finish_meeting ย้ายไฟล์บนดิสก์ แต่ไม่ได้
    # แก้ออบเจกต์ Path ตัวนี้ ค่าจึงเท่ากันไม่ว่าจะอ่านก่อนหรือหลัง สลับบรรทัดได้ ไม่พัง
    audio_file = audio_path.name
    meeting_dir = _finish_meeting(
        audio_path,
        config,
        claude_model,
        meeting_dir,
        transcript_path,
        transcript_markdown,
        profile,
    )
    _record_pending_speakers(
        config,
        job,
        meeting_dir,
        audio_file,
        transcript_markdown,
        claude_model,
        merged,
        speaker_labels,
        voiceprints,
        embedding_model,
        matches,
    )
    return meeting_dir


def _finish_meeting(
    audio_path: Path,
    config: Config,
    claude_model: str,
    meeting_dir: Path,
    transcript_path: Path,
    transcript_markdown: str,
    profile: str,
) -> Path:
    """The half of the pipeline that runs whether or not the transcript is new."""
    # No retry here on purpose: summarize_transcript retries every API call it
    # makes, per chunk. Wrapping it again would re-run a whole map-reduce
    # (8 chunks + 1 reduce, up to 54 calls at worst -- see MAP_MAX_CONCURRENCY's
    # comment in src/summarize.py for how that bound is derived) because of one
    # permanently failing chunk.
    job = audio_path.stem
    summary_markdown = None
    glossary_counts: dict[str, int] = {}
    fuzzy_seen: dict[str, int] = {}
    if claude_model != NO_SUMMARY_MODEL:
        # ตัวกรองศัพท์เสียบที่นี่ ไม่ใช่ใน summarize_transcript สองเหตุผล:
        # (1) ต้องแก้ก่อนหั่น chunk -- split_into_chunks เล่นซ้ำ segment ท้าย chunk
        #     ที่หัว chunk ถัดไป (overlap) ถ้าแทนที่ทีละ chunk คำในโซนนั้นจะถูกนับสองรอบ
        #     เรียกครั้งเดียวที่นี่จึงครอบทั้ง path ที่หั่น chunk และ path ที่ยิงรอบเดียว
        # (2) summarize_transcript คืน str เปล่า ๆ และถูก mock ไว้หลายสิบจุดในเทสต์
        #     ให้มันคืนจำนวนที่แก้ด้วยจะเปลี่ยน return type ไปทั้งหมดโดยไม่ได้อะไรเพิ่ม
        #
        # transcript_markdown ที่ถูกแก้เป็นแค่ตัวแปรในหน่วยความจำ -- transcript.md บน
        # ดิสก์ถูกเขียนไปก่อนถึงบรรทัดนี้แล้วและยังเป็นของดิบ ถ้า glossary แก้ผิด
        # คนอ่านย้อนดูได้ว่าเดิมพูดว่าอะไร และ path ลองใหม่ก็อ่านของดิบตัวเดิมเสมอ
        glossary = load_glossary(
            config.base_dir / "glossary.md", config.base_dir / "teams.md"
        )
        corrected_markdown, glossary_counts = glossary.apply_exact(transcript_markdown)
        fuzzy_seen = glossary.count_only(corrected_markdown)
        # profile ต้องไปสองที่พร้อมกัน ไม่ใช่ที่เดียว: ไฟล์ prompts/profiles/cross.md
        # สั่งโมเดลให้ไปดูตารางคำกำกวมกับตารางฝ่ายของผู้เข้าร่วม ถ้าส่ง profile ให้
        # summarize_transcript แต่ไม่เปิด cross ให้ format_for_prompt ด้วย กฎนั้นจะ
        # ชี้ไปที่ตารางที่ไม่ได้ถูกใส่เข้ามา -- สรุปเพี้ยนโดยไม่มีอะไรฟ้อง
        glossary_text = glossary.format_for_prompt(
            include_cross_team_context=(profile == CROSS_TEAM_PROFILE)
        )
        # exclude_dir=meeting_dir สำคัญ ไม่ใช่ทางเลือก: path ลองใหม่ (ลาก .job.json
        # กลับ inbox) เข้ามาที่โฟลเดอร์เดิมซึ่งอาจมี summary.md จากรอบก่อนอยู่แล้ว
        # ไม่กันไว้ ประชุมจะยกเรื่องค้างของตัวเองมาเป็น "คืบหน้าจากครั้งก่อน" วนไปเรื่อยๆ
        carryover_text = ""
        if config.carryover_enabled:
            carryover_text = carryover.format_for_prompt(
                carryover.previous_open_items(
                    config.meetings_dir, profile, exclude_dir=meeting_dir
                )
            )
        activity.append(
            config.base_dir,
            job,
            "summarize_started",
            params={"model": claude_model, "profile": profile},
        )
        # summarize.py ไม่รู้จัก activity feed และไม่ควรรู้ -- ส่ง callback เข้าไปแทน
        # การ import src.activity เข้าไปในนั้น รูปแบบเดียวกับ on_event ที่ record.py
        # ใช้อยู่แล้ว เหตุผลที่ต้องมี: ก่อนหน้านี้แถบใน UI ค้างที่ขั้น "กำลังสรุป"
        # ตั้งแต่นาทีแรกจนจบ ทำให้แยกไม่ออกว่ากำลังทำงานอยู่หรือแขวนไปแล้ว
        def report_progress(done: int, total: int) -> None:
            activity.append(
                config.base_dir,
                job,
                "summarize_progress",
                params={"done": done, "total": total},
            )

        try:
            # ยิงคำขอเล็ก ๆ ก่อนลงทุนกับงานจริง: ถ้าปลายทางไปไม่ถึง เราจะรู้ใน 30 วินาที
            # แทนที่จะเป็นหนึ่งชั่วโมง transcript ถูกเซฟไปแล้วก่อนถึงบรรทัดนี้ การล้ม
            # ตรงนี้จึงไม่ทำให้งานถอดเสียงหาย และกู้ได้ตามขั้นตอนใน README
            check_model_reachable(claude_model)
            summary_markdown = summarize_transcript(
                corrected_markdown,
                model=claude_model,
                glossary_text=glossary_text,
                profile=profile,
                carryover_text=carryover_text,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
                on_progress=report_progress,
                merge_turns=config.merge_speaker_turns,
            )
        except Exception as e:
            activity.append(
                config.base_dir, job, "job_failed", "error", {"error": str(e)}
            )
            move_to_failed(
                audio_path,
                config.failed_dir,
                f"Summarization failed: {e}\nTranscript was saved to {transcript_path}",
            )
            raise

    try:
        # save_summary stamps the model name into summary.meta.md, so calling it
        # in transcript-only mode would write "สรุปด้วย transcript-only" for a
        # summary nobody asked for. Everything else below still runs: the meeting
        # is finished, just without that one file.
        if summary_markdown is not None:
            # transcript_markdown ตัวดิบ ไม่ใช่ corrected_markdown: บรรทัดผู้เข้าร่วมนับ
            # จากป้ายผู้พูด ซึ่ง glossary ไม่ได้แตะอยู่แล้ว และของดิบคือสิ่งที่ตรงกับ
            # transcript.md บนดิสก์ที่คนจะเปิดไปตรวจต่อ
            save_summary(
                meeting_dir,
                replace_participants_line(summary_markdown, transcript_markdown),
                claude_model,
                glossary_counts=glossary_counts,
                fuzzy_seen=fuzzy_seen,
                profile=profile,
                speaker_table=speaker_table(transcript_markdown),
            )
        archive_audio(meeting_dir, audio_path)
        discard_job(audio_path)
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    activity.append(
        config.base_dir, job, "meeting_done", params={"path": str(meeting_dir)}
    )
    return meeting_dir
