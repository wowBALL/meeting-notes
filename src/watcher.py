import logging
import time
from pathlib import Path
from typing import Any

from src import enroll
from src.config import Config
from src.pipeline import process_file
from src.voiceprint import load_embedder

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg"}


def scan_inbox(inbox_dir: Path) -> list[Path]:
    if not inbox_dir.exists():
        return []
    return sorted(
        p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def is_file_stable(path: Path, check_interval: float = 1.0) -> bool:
    size_before = path.stat().st_size
    time.sleep(check_interval)
    size_after = path.stat().st_size
    return size_before == size_after and size_before > 0


def process_enroll_requests(
    config: Config, diarization_pipeline: Any = None, embedder: Any = None
) -> None:
    """วิเคราะห์ไฟล์ลงทะเบียนเสียงที่มีใบสั่งงานค้างอยู่

    ใช้ embedder ตัวเดียวกับที่ประมวลผลประชุมโดยเจตนา: เวกเตอร์ที่จะเอาไปเทียบกัน (ตอนกด
    ยืนยันชื่อ กับตอน match_known เทียบกับทะเบียน) ต้องมาจากโมเดล speaker verification
    เดียวกัน ไม่งั้น cosine similarity ระหว่างสองฝั่งไม่มีความหมายเลย -- diarization_pipeline
    ไม่ใช่ตัวที่ต้องเหมือนกันอีกต่อไป มันแค่กำหนดขอบท่อน (ว่าเสียงช่วงไหนเป็นของใคร) ซึ่งไม่ได้
    เข้าไปอยู่ในเวกเตอร์เอง (ดู src/voiceprint.py: voiceprint แยกจากโมเดลแยกผู้พูดโดยสิ้นเชิง)

    โหลด embedder เองครั้งเดียวก่อนวนลูปเมื่อไม่มีใครส่งมาให้ ไม่ใช่โหลดใหม่ทุกไฟล์ในคิว --
    แบบเดียวกับ pipeline.process_file (ดูเหตุผลเต็มที่นั่น) ทางปกติ (watch_loop ที่เรียกจาก
    main.py) จะส่ง embedder ที่โหลดไว้แล้วมาให้เสมอ เส้นทางนี้มีไว้สำหรับผู้เรียกที่ไม่ได้
    ถือโมเดลไว้เอง (เทสต์/สคริปต์แยก)

    ไม่เคยเขียน registry.json -- เขียนแค่ผลวิเคราะห์ไว้ให้คนมากดยืนยันที่หน้าเว็บ
    """
    try:
        # pending_requests เองก็ raise ได้ (อ่านโฟลเดอร์ไม่ได้ ฯลฯ) และแต่เดิมมันอยู่ใน
        # for header ตรง ๆ นอก try ใด ๆ เลย -- ถ้าพังตรงนี้ exception จะทะลุออกไปถึง
        # watch_loop ซึ่งไม่ได้ห่อการเรียกฟังก์ชันนี้ไว้ ฆ่า watcher ทั้งตัวพร้อมงาน
        # ประมวลผล inbox/ ที่ยังไม่ได้ทำในรอบถัดไปไปด้วย (finding 3 ของรีวิวรอบสุดท้าย)
        pending = enroll.pending_requests(config.base_dir)
    except Exception:
        logger.exception("Failed to list pending enrollment requests")
        return
    if not pending:
        # ต้องเช็คก่อนโหลด embedder สำรอง ไม่ใช่หลัง: คิวว่างคือกรณีปกติของแทบทุก poll
        # (ผู้ใช้ enroll กันไม่บ่อย) โหลดก่อนเช็คแปลว่า hf_token/EMBEDDING_MODEL ที่ผิด
        # ยิง traceback เต็มลง log ทุก poll แม้ไม่มีงานให้ทำเลยสักไฟล์ -- ย้ายมาไว้หลัง
        # เช็คนี้แล้วมันจะยิงเฉพาะตอนมีงานค้างจริง ๆ เท่านั้น
        return
    if embedder is None:
        # โหลดไม่สำเร็จ (hf_token ผิด/ไม่มีเน็ต) ต้องไม่ทำให้ watch_loop ทั้งตัวตาย
        # เหมือนกับเหตุผลข้างบนเรื่อง pending_requests -- watch_loop ไม่ได้ห่อการเรียก
        # ฟังก์ชันนี้ไว้เลย ข้ามรอบ poll นี้ไปทั้งหมด (ใบสั่งงานยังอยู่ครบ ไม่มีอะไรหาย)
        # ดีกว่าปล่อยให้ exception ทะลุออกไปฆ่า watcher พร้อมงานประมวลผล inbox/ ที่ยังไม่ได้
        # ทำในรอบถัดไปไปด้วย โหลดครั้งเดียวตรงนี้ ไม่ใช่ในลูปข้างล่าง -- คิวมีได้หลายไฟล์
        # ต่อรอบ poll เดียว
        try:
            embedder = load_embedder(config.hf_token, config.embedding_model)
        except Exception:
            logger.exception(
                "โหลด embedder ไม่สำเร็จ -- ข้ามการวิเคราะห์ไฟล์ลงทะเบียนรอบ poll นี้ "
                "(ใบสั่งงานยังอยู่ครบ รอบหน้าจะลองใหม่เอง)"
            )
            return
    for audio_file in pending:
        audio_path = enroll.enroll_dir(config.base_dir) / audio_file
        # CRITICAL A: ต้อง stat ไฟล์ *ก่อน* ส่งเข้า analyze() เสมอ ไม่ใช่ตอนเขียนผลทีหลัง
        # -- diarization ใช้เวลานาน (วินาทีถึงนาที) ถ้าผู้ใช้แทนที่ไฟล์ระหว่างนั้น write_result
        # ที่ stat() ตอนเขียนผลอย่างเดียวจะเห็นไฟล์ใหม่ "ตรง" กับตัวเองเสมอ แล้วผูก embedding
        # ของไบต์ชุดเก่าเข้ากับไฟล์ใหม่โดยไม่มีทางจับได้เลย ต้องส่ง stat ที่ถ่ายไว้ก่อนนี้
        # (pre_analysis_stat) ให้ write_result เทียบกับ stat ตอนเขียนผลด้วยตัวเอง
        try:
            pre_stat = audio_path.stat()
            pre_analysis_stat: tuple[int, float] | None = (
                pre_stat.st_size,
                pre_stat.st_mtime,
            )
        except OSError as e:
            # finding 1 (blocks merge): เดิมตรงนี้ตั้ง pre_analysis_stat = None แล้ว "เดินหน้า
            # วิเคราะห์ต่อ" -- write_result.เดิมเช็คแค่ elif pre_analysis_stat is not None and
            # ... ทำให้ None ข้ามการเปรียบเทียบไปเงียบ ๆ เท่ากับปิดการป้องกันทั้งหมดที่เพิ่ง
            # เพิ่มเข้ามา (CRITICAL A) พอดีตอนที่ต้องการมันที่สุด: PermissionError ชั่วคราว
            # (สแกนไวรัส/ตัวซิงก์ไฟล์) มักคลายล็อกเองไม่กี่ร้อย ms ให้หลัง ffmpeg/analyze()
            # จึงอ่านไฟล์สำเร็จตามปกติ แต่ diarization กินเวลานาน (วินาทีถึงนาที) พอถึงตอน
            # write_result stat ไฟล์ใหม่อาจถูกผู้ใช้แทนที่ไปแล้ว (อัดคนละคนทับชื่อเดิม) และ
            # เพราะไม่มี pre_analysis_stat มาเทียบ ผล "ok" จะถูกเขียนพร้อม stat ของไฟล์ใหม่
            # ผูกเข้ากับ embedding ของไฟล์เก่า -- ต้องไม่วิเคราะห์ไฟล์ที่ผูกกับไบต์อะไรไม่ได้
            # ตั้งแต่ต้นแทน ข้ามไปเลย (ไม่เรียก analyze/write_result) ใบสั่งงานยังอยู่ครบ
            # (pending_requests ยังเห็นว่า "สั่งแล้วแต่ยังไม่มีผล") รอบ poll หน้าจะลองใหม่เอง
            # เมื่อล็อกคลาย -- ผลพลอยได้คือไม่เสีย GPU diarization เต็มรอบไปกับผลที่ต้องทิ้ง
            logger.warning(
                "stat ไฟล์ %s ก่อนวิเคราะห์ไม่สำเร็จชั่วคราว (%s) -- ข้ามรอบ poll นี้ ไม่วิเคราะห์"
                "ไฟล์ที่ผูกกับไบต์อะไรไม่ได้ รอบหน้าจะลองใหม่เองเมื่อล็อกคลาย",
                audio_file,
                e,
            )
            continue
        try:
            analyzed = enroll.analyze(
                audio_path,
                pipeline=diarization_pipeline,
                model=config.diarization_model,
                embedder=embedder,
            )
        except Exception:
            # enroll.analyze ไม่ raise อยู่แล้วโดยสัญญาของมัน แต่กันไว้อีกชั้น: ไฟล์เดียว
            # ที่พังต้องไม่ทำให้ไฟล์อื่นในคิวไม่ถูกทำ
            logger.exception("Failed to analyze enrollment clip %s", audio_file)
            continue
        try:
            written = enroll.write_result(
                config.base_dir,
                audio_file,
                analyzed,
                pre_analysis_stat=pre_analysis_stat,
            )
            # finding 4 ของรีวิวรอบนี้: write_result คืน None เมื่อผลถูกทิ้งเพราะผูกกับ
            # ไบต์ไหนไม่ได้ (ไฟล์เสียงถูกแทนที่ระหว่างวิเคราะห์ ฯลฯ) -- เดิม log บรรทัด
            # ข้างล่างนี้ยิง INFO "-> ok" แบบไม่มีเงื่อนไขเสมอ แม้ตอนที่ผลถูกทิ้งไปแล้ว
            # ผู้คุมระบบจะเห็น WARNING เรื่องผูกไม่ได้ตามด้วย INFO อ้างว่าสำเร็จทันที ต้อง
            # แยก log ตามค่าที่คืนจริง ไม่ใช่ตาม status ที่ analyze() รายงานมาเฉย ๆ
            if written is None:
                logger.warning(
                    "Discarded the analysis result for enrollment clip %s "
                    "(could not verify it against the audio on disk -- see the "
                    "warning above); the card goes back to idle instead of ok",
                    audio_file,
                )
            else:
                logger.info(
                    "Analyzed enrollment clip %s -> %s", audio_file, analyzed.get("status")
                )
        except Exception:
            # เขียนผลไม่สำเร็จ (ดิสก์เต็ม/ไฟล์ถูกล็อกจน replace_with_retry หมดความ
            # พยายาม) -- ถ้าปล่อยใบสั่งงานค้างไว้เฉย ๆ pending_requests จะยังเห็นว่า
            # "สั่งแล้วแต่ยังไม่มีผล" แล้วสั่งวิเคราะห์ซ้ำ (ถอดเสียงเต็มรอบบน GPU) ทุก
            # poll ไปเรื่อย ๆ ไม่มีที่สิ้นสุด ในขณะที่หน้าเว็บค้างที่ "กำลังวิเคราะห์"
            # ตลอดกาล (finding 3) -- ลองเขียนผลล้มเหลวแบบย่อแทน ผู้ใช้จะได้เห็นเหตุผล
            # และกดเอาไฟล์ออกจากรายการได้ แทนที่จะเห็นหน้าจอค้าง
            logger.exception(
                "Failed to write the result for enrollment clip %s", audio_file
            )
            try:
                enroll.write_result(
                    config.base_dir,
                    audio_file,
                    {
                        "status": "rejected",
                        "reason": "analysis_failed",
                        "suggested_name": enroll.suggested_name_from(audio_file),
                    },
                )
            except Exception:
                # เขียนแม้แต่ผลล้มเหลวแบบย่อก็ยังไม่ได้ (ดิสก์เต็มจริง ๆ) -- ทางเดียวที่
                # เหลือคือตัดใบสั่งงานทิ้งไม่ให้ค้างวนซ้ำไม่รู้จบ แม้ผู้ใช้จะไม่เห็นเหตุผล
                # ที่ชัดเจนก็ตาม ดีกว่าไฟล์นี้กิน GPU ทุก poll ตลอดไป
                logger.exception(
                    "Failed to write even a minimal failure result for %s; "
                    "dropping the stuck request",
                    audio_file,
                )
                try:
                    enroll.clear(config.base_dir, audio_file)
                except Exception:
                    logger.exception(
                        "Failed to clear the stuck request for enrollment clip %s",
                        audio_file,
                    )


def watch_loop(
    config: Config,
    poll_interval: float = 5.0,
    single_pass: bool = False,
    diarization_pipeline: Any = None,
    whisper_model: Any = None,
    embedder: Any = None,
) -> None:
    while True:
        for audio_path in scan_inbox(config.inbox_dir):
            if is_file_stable(audio_path, check_interval=0.5):
                try:
                    meeting_dir = process_file(
                        audio_path,
                        config,
                        diarization_pipeline=diarization_pipeline,
                        whisper_model=whisper_model,
                        embedder=embedder,
                    )
                    logger.info("Processed %s -> %s", audio_path.name, meeting_dir)
                except Exception:
                    logger.exception("Failed to process %s", audio_path.name)
        # หลัง inbox เสมอ: การประชุมที่อัดซ้ำไม่ได้ต้องได้ GPU ก่อนงานลงทะเบียนเสียง
        # ซึ่งผู้ใช้สั่งใหม่ได้ตลอด try/except ของตัวเองอยู่ข้างในแล้ว
        process_enroll_requests(
            config, diarization_pipeline=diarization_pipeline, embedder=embedder
        )
        if single_pass:
            return
        time.sleep(poll_interval)
