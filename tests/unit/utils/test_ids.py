from __future__ import annotations

from typing import cast

from khedron.utils.ids import generate_ulid, is_valid_ulid


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_generate_ulid_has_expected_length() -> None:
    _check(len(generate_ulid()) == 26, "ULID length should be 26")


def test_generate_ulid_returns_valid_ulid() -> None:
    _check(is_valid_ulid(generate_ulid()), "generated ULID should validate")


def test_generate_ulid_returns_unique_values() -> None:
    _check(generate_ulid() != generate_ulid(), "generated ULIDs should be unique")


def test_is_valid_ulid_rejects_short_string() -> None:
    _check(not is_valid_ulid("not-a-ulid"), "short string should not validate")


def test_is_valid_ulid_rejects_non_string_value() -> None:
    _check(not is_valid_ulid(cast(str, 123)), "non-string value should not validate")
