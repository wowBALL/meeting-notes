"""ตั้งค่า logging ให้เขียนลงไฟล์จริง ไม่ใช่แค่ console ที่ปิดแล้วหายไปพร้อมกัน

เหตุผลที่ไฟล์นี้ต้องมี -- 2026-07-31 ขั้นสรุปของประชุมหนึ่งค้างไปราวหนึ่งชั่วโมงแล้ว
ล้มด้วย "The read operation timed out" ครบทั้ง 5 chunk พอจะย้อนไปหาสาเหตุกลับพบว่า
**ไม่มีหลักฐานเหลืออยู่เลย**: watcher ถูกเปิดด้วย `cmd /k` ที่ไม่ redirect อะไรทั้งสิ้น
(start-meeting.bat, start-ui.bat) log ทั้งหมดจึงอยู่ใน scrollback ของ console window
ที่ปิดไปแล้ว ไฟล์ watcher_err.log ที่ root เป็นของตกค้างจากการ redirect ด้วยมือครั้งเดียว
เมื่อ 2026-07-22 ไม่มีโค้ดไหนเขียนมันเลย

ทดสอบภายหลังแล้วว่า endpoint รับ workload จริงได้สบาย (chunk เดียว 88.9 วิ, ยิงพร้อมกัน
4 ก้อน 191 วิ เทียบกับ timeout 900 วิ/call) สาเหตุที่แท้จริงของเหตุการณ์นั้นจึงยังไม่รู้
และจะไม่มีวันรู้ -- นั่นคือปัญหาที่ไฟล์นี้แก้ ไม่ใช่การเดาว่าครั้งนั้นเกิดอะไร

*** encoding="utf-8" ไม่ใช่ของแถม แต่เป็นเงื่อนไขว่าไฟล์นี้จะทำงานได้จริงหรือไม่ ***
ข้อความ log ของโปรเจกต์นี้เป็นภาษาไทยเกือบทั้งหมด ส่วนค่า default ของ Windows คือ
cp1252 ซึ่ง encode อักษรไทยไม่ได้ ("'charmap' codec can't encode characters" -- เจอจริง
ระหว่างสืบสวนเหตุการณ์ข้างบน) handler ที่ไม่ระบุ encoding จะโยน UnicodeEncodeError
ตอนเขียนบรรทัดภาษาไทยบรรทัดแรก แปลว่ากลไกที่สร้างมาเพื่อจับ error กลายเป็นตัวสร้าง
error เสียเอง ในจังหวะที่เราต้องการมันที่สุดพอดี
"""

import logging
import logging.handlers
from pathlib import Path

# ใช้ค่าเดียวกับ activity.jsonl โดยตั้งใจ: log กับ activity feed เป็นบันทึกของเหตุการณ์
# เดียวกันคนละมุม การให้มันอยู่คนละโฟลเดอร์แปลว่าเวลาสืบสวนต้องเปิดสองที่
from src.activity import ACTIVITY_DIRNAME

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# 5MB × 4 ไฟล์ = 20MB เพดาน -- ประชุมหนึ่งครั้งเขียนไม่กี่สิบบรรทัด เพดานนี้จึงเก็บ
# ย้อนหลังได้เป็นเดือน ซึ่งเป็นช่วงเวลาที่ใช้จริงตอนไล่หาว่า "มันเริ่มพังตั้งแต่เมื่อไหร่"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

WATCHER_LOG = "watcher.log"
UI_LOG = "ui.log"


def log_path(base_dir: Path, filename: str) -> Path:
    return Path(base_dir) / ACTIVITY_DIRNAME / filename


def configure_logging(
    base_dir: Path, filename: str, level: int = logging.INFO
) -> Path | None:
    """ตั้ง root logger ให้เขียนทั้ง console และไฟล์ คืน path ของไฟล์ (None ถ้าเปิดไม่ได้)

    filename ต้องแยกกันต่อ process (watcher.log / ui.log) -- RotatingFileHandler
    หมุนไฟล์ด้วยการ rename ซึ่งบน Windows จะล้มถ้าอีก process ถือ handle ของไฟล์
    เดียวกันค้างไว้ สอง process เขียนไฟล์เดียวกันจึงพังตอนหมุน ไม่ใช่ตอนเขียน --
    อาการที่โผล่มาเป็นครั้งคราวหลังใช้ไปหลายเดือนแล้วหาต้นเหตุยาก

    เปิดไฟล์ไม่ได้ต้องไม่ทำให้โปรแกรมไม่เริ่ม: log ลงไฟล์เป็นเครื่องมือช่วยวินิจฉัย
    ไม่ใช่ความสามารถหลัก ดิสก์เต็มหรือโฟลเดอร์ read-only ต้องยังอัดประชุมได้ตามปกติ
    เหตุผลเดียวกับที่ activity.append() กลืน OSError ทิ้ง
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    path: Path | None = None
    failure: OSError | None = None
    try:
        candidate = log_path(base_dir, filename)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                candidate,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
        )
        path = candidate
    except OSError as e:
        failure = e

    # force=True: basicConfig ไม่ทำอะไรเลยถ้า root logger มี handler อยู่แล้ว ซึ่ง
    # เกิดได้เมื่อ import chain ไปแตะ library ที่ตั้ง handler ไว้ก่อน -- โดยไม่มี force
    # ความล้มเหลวจะเงียบสนิท (ไม่มีไฟล์ ไม่มี error) ซึ่งเป็นอาการเดียวกับที่ไฟล์นี้
    # ถูกสร้างมาเพื่อกำจัด
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)

    if failure is not None:
        logging.getLogger(__name__).warning(
            "เขียน log ลงไฟล์ไม่ได้ (%s) -- log จะอยู่ที่ console อย่างเดียว "
            "และจะหายไปเมื่อปิดหน้าต่าง",
            failure,
        )
    return path
