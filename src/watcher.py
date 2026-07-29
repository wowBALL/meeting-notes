import logging
import time
from pathlib import Path
from typing import Any

from src import enroll
from src.config import Config
from src.pipeline import process_file

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


def process_enroll_requests(config: Config, diarization_pipeline: Any = None) -> None:
    """วิเคราะห์ไฟล์ลงทะเบียนเสียงที่มีใบสั่งงานค้างอยู่

    ใช้ pipeline ตัวเดียวกับที่ถอดเทปประชุมโดยเจตนา: เวกเตอร์ที่จะเอาไปเทียบกันต้องมา
    จากโมเดลเดียวกัน ไม่งั้น cosine similarity ระหว่างสองฝั่งไม่มีความหมายเลย

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
        except OSError:
            pre_analysis_stat = None
        try:
            analyzed = enroll.analyze(audio_path, pipeline=diarization_pipeline)
        except Exception:
            # enroll.analyze ไม่ raise อยู่แล้วโดยสัญญาของมัน แต่กันไว้อีกชั้น: ไฟล์เดียว
            # ที่พังต้องไม่ทำให้ไฟล์อื่นในคิวไม่ถูกทำ
            logger.exception("Failed to analyze enrollment clip %s", audio_file)
            continue
        try:
            enroll.write_result(
                config.base_dir,
                audio_file,
                analyzed,
                pre_analysis_stat=pre_analysis_stat,
            )
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
                    )
                    logger.info("Processed %s -> %s", audio_path.name, meeting_dir)
                except Exception:
                    logger.exception("Failed to process %s", audio_path.name)
        # หลัง inbox เสมอ: การประชุมที่อัดซ้ำไม่ได้ต้องได้ GPU ก่อนงานลงทะเบียนเสียง
        # ซึ่งผู้ใช้สั่งใหม่ได้ตลอด try/except ของตัวเองอยู่ข้างในแล้ว
        process_enroll_requests(config, diarization_pipeline=diarization_pipeline)
        if single_pass:
            return
        time.sleep(poll_interval)
