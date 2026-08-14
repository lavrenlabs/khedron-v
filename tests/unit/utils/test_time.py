from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khedron.utils.time import from_iso8601, now_utc, to_iso8601


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_now_utc_returns_timezone_aware_utc_datetime() -> None:
    value = now_utc()
    _check(value.tzinfo == UTC, "now_utc should return UTC")


def test_to_iso8601_serializes_aware_datetime() -> None:
    value = datetime(2026, 5, 3, 12, 30, tzinfo=UTC)
    _check(to_iso8601(value) == "2026-05-03T12:30:00+00:00", "timestamp mismatch")


def test_to_iso8601_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        to_iso8601(datetime(2026, 5, 3, 12, 30))


def test_from_iso8601_parses_aware_datetime() -> None:
    parsed = from_iso8601("2026-05-03T12:30:00+00:00")
    _check(parsed.tzinfo is not None, "parsed timestamp should be timezone-aware")


def test_from_iso8601_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        from_iso8601("2026-05-03T12:30:00")
