from __future__ import annotations

from collections.abc import Iterator

import pytest

from khedron.providers.registry import (
    PROVIDER_REGISTRY,
    get_provider_class,
    register_provider,
)


@pytest.fixture(autouse=True)
def isolate_provider_registry() -> Iterator[None]:
    snapshot = PROVIDER_REGISTRY.copy()
    PROVIDER_REGISTRY.clear()
    try:
        yield
    finally:
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(snapshot)


def test_register_provider_decorator_resolves_class() -> None:
    @register_provider("test_provider")
    class TestProvider:
        pass

    if get_provider_class("test_provider") is not TestProvider:
        raise AssertionError(PROVIDER_REGISTRY)


def test_register_provider_decorator_returns_original_class() -> None:
    class TestProvider:
        pass

    decorated = register_provider("test_provider")(TestProvider)

    if decorated is not TestProvider:
        raise AssertionError(decorated)


def test_unknown_provider_lookup_lists_sorted_available_names() -> None:
    class AlphaProvider:
        pass

    class BravoProvider:
        pass

    class ZuluProvider:
        pass

    register_provider("zulu")(ZuluProvider)
    register_provider("alpha")(AlphaProvider)
    register_provider("bravo")(BravoProvider)

    with pytest.raises(KeyError) as exc_info:
        get_provider_class("missing")

    message = str(exc_info.value)
    if "missing" not in message:
        raise AssertionError(message)
    if "Available providers: alpha, bravo, zulu" not in message:
        raise AssertionError(message)


def test_duplicate_same_provider_class_registration_is_idempotent() -> None:
    class TestProvider:
        pass

    first = register_provider("test_provider")(TestProvider)
    second = register_provider("test_provider")(TestProvider)

    if first is not TestProvider or second is not TestProvider:
        raise AssertionError((first, second))
    if get_provider_class("test_provider") is not TestProvider:
        raise AssertionError(PROVIDER_REGISTRY)
    if len(PROVIDER_REGISTRY) != 1:
        raise AssertionError(PROVIDER_REGISTRY)


def test_duplicate_different_provider_class_registration_is_rejected() -> None:
    class FirstProvider:
        pass

    class SecondProvider:
        pass

    register_provider("test_provider")(FirstProvider)

    with pytest.raises(ValueError) as exc_info:
        register_provider("test_provider")(SecondProvider)

    message = str(exc_info.value)
    if "test_provider" not in message:
        raise AssertionError(message)
    if "FirstProvider" not in message:
        raise AssertionError(message)


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_empty_provider_names_are_rejected(name: str) -> None:
    class TestProvider:
        pass

    with pytest.raises(ValueError, match="Provider name"):
        register_provider(name)(TestProvider)

    with pytest.raises(ValueError, match="Provider name"):
        get_provider_class(name)
