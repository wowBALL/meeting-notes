import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T], max_retries: int = 3, base_delay: float = 1.0
) -> T:
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_exception is not None
    raise last_exception
