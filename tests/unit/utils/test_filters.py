from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from khedron.utils.filters import matches_filter


@dataclass(frozen=True)
class FakeMemory:
    metadata: dict[str, Any]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_matches_exact_metadata_value() -> None:
    memory = FakeMemory(metadata={"speaker": "alice"})
    _check(matches_filter(memory, {"speaker": "alice"}), "exact metadata should match")


def test_missing_metadata_matches_none_filter() -> None:
    memory = FakeMemory(metadata={})
    _check(matches_filter(memory, {"speaker": None}), "missing metadata should match None")


def test_matches_list_membership() -> None:
    memory = FakeMemory(metadata={"speaker": "alice"})
    _check(matches_filter(memory, {"speaker": ["alice", "bob"]}), "list membership should match")


def test_matches_range_operators() -> None:
    memory = FakeMemory(metadata={"session_number": 5})
    filter_dict = {"session_number": {"$gte": 3, "$lt": 10}}
    _check(matches_filter(memory, filter_dict), "range operators should match")


def test_rejects_failed_range_operator() -> None:
    memory = FakeMemory(metadata={"session_number": 5})
    _check(not matches_filter(memory, {"session_number": {"$gt": 5}}), "range should fail")


def test_matches_regex_operator() -> None:
    memory = FakeMemory(metadata={"topic": "birthday party"})
    _check(matches_filter(memory, {"topic": {"$regex": "birth.*party"}}), "regex should match")


def test_requires_all_filters_to_match() -> None:
    memory = FakeMemory(metadata={"speaker": "alice", "session_number": 5})
    filter_dict = {"speaker": "alice", "session_number": {"$lte": 4}}
    _check(not matches_filter(memory, filter_dict), "all filters should be required")
