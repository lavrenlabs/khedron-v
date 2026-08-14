from __future__ import annotations

from datetime import UTC, datetime


def now_utc() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_iso8601(value: datetime) -> str:
    """Serialize a timezone-aware datetime to ISO 8601."""
    if value.tzinfo is None:
        raise ValueError("Cannot serialize naive datetime; must be timezone-aware")
    return value.isoformat()


def from_iso8601(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into a timezone-aware datetime."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp '{value}' is naive; must include timezone")
    return parsed
