from __future__ import annotations

from collections.abc import Iterator

import pytest

from khedron.judges.registry import (
    JUDGE_REGISTRY,
    get_judge_class,
    register_judge,
)


@pytest.fixture(autouse=True)
def isolate_judge_registry() -> Iterator[None]:
    snapshot = JUDGE_REGISTRY.copy()
    JUDGE_REGISTRY.clear()
    try:
        yield
    finally:
        JUDGE_REGISTRY.clear()
        JUDGE_REGISTRY.update(snapshot)


def test_register_judge_decorator_resolves_class() -> None:
    @register_judge("test_judge")
    class TestJudge:
        pass

    if get_judge_class("test_judge") is not TestJudge:
        raise AssertionError(JUDGE_REGISTRY)


def test_register_judge_decorator_returns_original_class() -> None:
    class TestJudge:
        pass

    decorated = register_judge("test_judge")(TestJudge)

    if decorated is not TestJudge:
        raise AssertionError(decorated)


def test_unknown_judge_lookup_lists_sorted_available_names() -> None:
    class AlphaJudge:
        pass

    class BravoJudge:
        pass

    class ZuluJudge:
        pass

    register_judge("zulu")(ZuluJudge)
    register_judge("alpha")(AlphaJudge)
    register_judge("bravo")(BravoJudge)

    with pytest.raises(KeyError) as exc_info:
        get_judge_class("missing")

    message = str(exc_info.value)
    if "missing" not in message:
        raise AssertionError(message)
    if "Available judges: alpha, bravo, zulu" not in message:
        raise AssertionError(message)


def test_duplicate_same_judge_class_registration_is_idempotent() -> None:
    class TestJudge:
        pass

    first = register_judge("test_judge")(TestJudge)
    second = register_judge("test_judge")(TestJudge)

    if first is not TestJudge or second is not TestJudge:
        raise AssertionError((first, second))
    if get_judge_class("test_judge") is not TestJudge:
        raise AssertionError(JUDGE_REGISTRY)
    if len(JUDGE_REGISTRY) != 1:
        raise AssertionError(JUDGE_REGISTRY)


def test_duplicate_different_judge_class_registration_is_rejected() -> None:
    class FirstJudge:
        pass

    class SecondJudge:
        pass

    register_judge("test_judge")(FirstJudge)

    with pytest.raises(ValueError) as exc_info:
        register_judge("test_judge")(SecondJudge)

    message = str(exc_info.value)
    if "test_judge" not in message:
        raise AssertionError(message)
    if "FirstJudge" not in message:
        raise AssertionError(message)


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_empty_judge_names_are_rejected(name: str) -> None:
    class TestJudge:
        pass

    with pytest.raises(ValueError, match="Judge name"):
        register_judge(name)(TestJudge)

    with pytest.raises(ValueError, match="Judge name"):
        get_judge_class(name)
