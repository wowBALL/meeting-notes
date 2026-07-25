import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """เรียก func ใหม่แบบถอยห่างขึ้นเรื่อย ๆ จนสำเร็จหรือครบจำนวนครั้ง

    ส่ง should_retry มาได้ถ้าผู้เรียกแยกออกว่าความล้มเหลวแบบไหนลองใหม่แล้วไม่มีวันหาย
    -- คืน False เมื่อไหร่ ข้อผิดพลาดนั้นจะถูกโยนต่อทันทีโดยไม่รอและไม่ลองซ้ำ
    ค่าเริ่มต้นคือลองใหม่ทุกกรณี ซึ่งเหมาะกับงานที่ไม่มีทางรู้ล่วงหน้า เช่นการถอดเสียง
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if should_retry is not None and not should_retry(e):
                raise
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_exception is not None
    raise last_exception
