from __future__ import annotations

from ulid import ULID


def generate_ulid() -> str:
    """Generate a new ULID as a 26-character string."""
    return str(ULID())


def is_valid_ulid(value: str) -> bool:
    """Return whether a string is a valid ULID."""
    try:
        ULID.from_str(value)
    except (TypeError, ValueError):
        return False
    return True
