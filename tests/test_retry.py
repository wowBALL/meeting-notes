import logging
from unittest.mock import patch

import pytest

from src.retry import retry_with_backoff


def test_retry_with_backoff_returns_result_on_eventual_success():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("temporary failure")
        return "success"

    with patch("time.sleep"):
        result = retry_with_backoff(flaky, max_retries=3)

    assert result == "success"
    assert calls["count"] == 3


def test_retry_with_backoff_raises_after_max_retries():
    def always_fails():
        raise ValueError("permanent failure")

    with patch("time.sleep"):
        with pytest.raises(ValueError, match="permanent failure"):
            retry_with_backoff(always_fails, max_retries=2)


def test_retry_with_backoff_gives_up_immediately_when_should_retry_says_no():
    calls = {"count": 0}

    def always_fails():
        calls["count"] += 1
        raise ValueError("no point trying this again")

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="no point trying this again"):
            retry_with_backoff(always_fails, max_retries=3, should_retry=lambda e: False)

    assert calls["count"] == 1
    mock_sleep.assert_not_called()


def test_retry_with_backoff_still_retries_when_should_retry_says_yes():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("temporary failure")
        return "success"

    with patch("time.sleep"):
        result = retry_with_backoff(flaky, max_retries=3, should_retry=lambda e: True)

    assert result == "success"
    assert calls["count"] == 3


def test_retry_with_backoff_does_not_retry_on_first_success():
    calls = {"count": 0}

    def succeeds_immediately():
        calls["count"] += 1
        return "ok"

    with patch("time.sleep") as mock_sleep:
        result = retry_with_backoff(succeeds_immediately, max_retries=3)

    assert result == "ok"
    assert calls["count"] == 1
    mock_sleep.assert_not_called()


def test_no_label_keeps_the_old_silence(caplog):
    """ค่า default ต้องเงียบเหมือนเดิม -- ฟังก์ชันนี้ถูกเรียกจากการถอดเสียงด้วย การเปิด
    log ให้ทุกผู้เรียกพร้อมกันคือการยัด log ที่ไม่มีใครขอเข้าไปในเส้นทางที่ไม่มีปัญหา"""

    def flaky():
        raise TimeoutError("the read operation timed out")

    with caplog.at_level(logging.WARNING), patch("time.sleep"):
        with pytest.raises(TimeoutError):
            retry_with_backoff(flaky, max_retries=2)

    assert caplog.text == ""


def test_every_retry_attempt_is_logged_when_labelled(caplog):
    """อาการของ 2026-07-31: 3 รอบ x 900 วินาทีต่อ chunk ผ่านไปโดยไม่มีบรรทัดเดียว
    บอกว่ากำลังรออะไรอยู่ ไฟล์นี้เคยไม่มี logging เลยแม้แต่บรรทัดเดียว"""

    def always_times_out():
        raise TimeoutError("the read operation timed out")

    with caplog.at_level(logging.WARNING), patch("time.sleep"):
        with pytest.raises(TimeoutError):
            retry_with_backoff(always_times_out, max_retries=3, label="Chunk 1/5")

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 3, messages
    assert all("Chunk 1/5" in m for m in messages)
    assert all("TimeoutError" in m for m in messages)
    assert "waiting" in messages[0]
    # รอบสุดท้ายต้องบอกว่าหมดสิทธิ์แล้ว ไม่ใช่บอกว่ากำลังจะรอต่ออีก
    assert "no attempts left" in messages[-1]


def test_a_permanent_failure_says_it_is_not_retrying(caplog):
    """should_retry คืน False = ยิงซ้ำก็ได้ผลเดิม คนอ่าน log ต้องแยกกรณีนี้ออกจาก
    "ลองครบทุกรอบแล้วยังไม่ผ่าน" ไม่งั้นจะไล่หาสาเหตุผิดทาง"""

    def permanent():
        raise ValueError("budget exceeded for key")

    with caplog.at_level(logging.WARNING), patch("time.sleep"):
        with pytest.raises(ValueError):
            retry_with_backoff(
                permanent, should_retry=lambda e: False, label="Reduce stage"
            )

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1, messages
    assert "not retrying" in messages[0]
    assert "Reduce stage" in messages[0]


def test_success_on_the_first_try_logs_nothing_even_when_labelled(caplog):
    with caplog.at_level(logging.WARNING):
        assert retry_with_backoff(lambda: "fine", label="Chunk 1/1") == "fine"
    assert caplog.text == ""
