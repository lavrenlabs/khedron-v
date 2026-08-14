from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol, cast


class SupportsMetadata(Protocol):
    metadata: Mapping[str, Any]


def matches_filter(memory: SupportsMetadata, filter_dict: Mapping[str, Any]) -> bool:
    """Return whether a memory-like object matches all metadata filters."""
    for key, expected in filter_dict.items():
        actual = memory.metadata.get(key)
        if not _value_matches(actual, expected):
            return False
    return True


def _value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, list):
        return actual in expected
    if isinstance(expected, dict):
        return _match_operators(actual, cast(Mapping[str, Any], expected))
    return actual == expected


def _match_operators(actual: Any, operators: Mapping[str, Any]) -> bool:
    for operator, expected in operators.items():
        if operator == "$regex":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            if re.search(expected, actual) is None:
                return False
        elif operator in {"$gt", "$gte", "$lt", "$lte"}:
            if actual is None or not _compare(operator, actual, expected):
                return False
        else:
            return False
    return True


def _compare(operator: str, actual: Any, expected: Any) -> bool:
    try:
        if operator == "$gt":
            return actual > expected
        if operator == "$gte":
            return actual >= expected
        if operator == "$lt":
            return actual < expected
        if operator == "$lte":
            return actual <= expected
    except TypeError:
        return False
    return False
