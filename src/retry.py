import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    should_retry: Callable[[Exception], bool] | None = None,
    label: str | None = None,
) -> T:
    """เรียก func ใหม่แบบถอยห่างขึ้นเรื่อย ๆ จนสำเร็จหรือครบจำนวนครั้ง

    ส่ง should_retry มาได้ถ้าผู้เรียกแยกออกว่าความล้มเหลวแบบไหนลองใหม่แล้วไม่มีวันหาย
    -- คืน False เมื่อไหร่ ข้อผิดพลาดนั้นจะถูกโยนต่อทันทีโดยไม่รอและไม่ลองซ้ำ
    ค่าเริ่มต้นคือลองใหม่ทุกกรณี ซึ่งเหมาะกับงานที่ไม่มีทางรู้ล่วงหน้า เช่นการถอดเสียง

    label คือชื่อที่จะโผล่ใน log ของทุกรอบที่ล้ม -- ค่า default เป็น None แปลว่าเงียบ
    สนิทเหมือนเดิม เพราะฟังก์ชันนี้ถูกใช้จากหลายที่ (การถอดเสียงด้วย) การเปิด log ให้
    ทุกผู้เรียกพร้อมกันคือการยัด log ที่ไม่มีใครขอเข้าไปในเส้นทางที่ไม่ได้มีปัญหา

    ทำไมต้องมี: 2026-07-31 ขั้นสรุปเงียบไปราวหนึ่งชั่วโมงก่อนจะยอมแพ้ ซึ่งคือ 3 รอบ
    × 900 วินาทีต่อ chunk ที่ผ่านไปโดยไม่มีบรรทัดเดียวบอกว่ากำลังรออะไรอยู่ ไฟล์นี้
    เคยไม่มี logging เลยแม้แต่บรรทัดเดียว
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if should_retry is not None and not should_retry(e):
                if label is not None:
                    logger.warning(
                        "%s: attempt %d/%d failed with a permanent error, not retrying (%s: %s)",
                        label,
                        attempt + 1,
                        max_retries,
                        type(e).__name__,
                        e,
                    )
                raise
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                if label is not None:
                    logger.warning(
                        "%s: attempt %d/%d failed (%s: %s) -- waiting %.1fs before retrying",
                        label,
                        attempt + 1,
                        max_retries,
                        type(e).__name__,
                        e,
                        delay,
                    )
                time.sleep(delay)
            elif label is not None:
                logger.warning(
                    "%s: attempt %d/%d failed (%s: %s) -- no attempts left",
                    label,
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    e,
                )
    assert last_exception is not None
    raise last_exception
