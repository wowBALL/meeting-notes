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
