from datetime import UTC, datetime, timedelta

import pytest

from cartiq.clock import FixedClock, SystemClock


def test_fixed_clock_only_moves_when_advanced() -> None:
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))

    assert clock.now() == clock.now() == datetime(2026, 1, 1, tzinfo=UTC)
    clock.advance(timedelta(minutes=5))
    assert clock.now() == datetime(2026, 1, 1, 0, 5, tzinfo=UTC)


def test_fixed_clock_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(datetime(2026, 1, 1))  # noqa: DTZ001


def test_system_clock_is_utc_aware() -> None:
    assert SystemClock().now().tzinfo is UTC
