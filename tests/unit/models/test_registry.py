from __future__ import annotations

from collections.abc import Iterator

import pytest

from khedron.models.registry import (
    MODEL_REGISTRY,
    get_model_class,
    register_model,
)


@pytest.fixture(autouse=True)
def isolate_model_registry() -> Iterator[None]:
    snapshot = MODEL_REGISTRY.copy()
    MODEL_REGISTRY.clear()
    try:
        yield
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(snapshot)


def test_register_model_decorator_resolves_class() -> None:
    @register_model("test_model")
    class TestModel:
        pass

    if get_model_class("test_model") is not TestModel:
        raise AssertionError(MODEL_REGISTRY)


def test_register_model_decorator_returns_original_class() -> None:
    class TestModel:
        pass

    decorated = register_model("test_model")(TestModel)

    if decorated is not TestModel:
        raise AssertionError(decorated)


def test_unknown_model_lookup_lists_sorted_available_names() -> None:
    class AlphaModel:
        pass

    class BravoModel:
        pass

    class ZuluModel:
        pass

    register_model("zulu")(ZuluModel)
    register_model("alpha")(AlphaModel)
    register_model("bravo")(BravoModel)

    with pytest.raises(KeyError) as exc_info:
        get_model_class("missing")

    message = str(exc_info.value)
    if "missing" not in message:
        raise AssertionError(message)
    if "Available models: alpha, bravo, zulu" not in message:
        raise AssertionError(message)


def test_duplicate_same_model_class_registration_is_idempotent() -> None:
    class TestModel:
        pass

    first = register_model("test_model")(TestModel)
    second = register_model("test_model")(TestModel)

    if first is not TestModel or second is not TestModel:
        raise AssertionError((first, second))
    if get_model_class("test_model") is not TestModel:
        raise AssertionError(MODEL_REGISTRY)
    if len(MODEL_REGISTRY) != 1:
        raise AssertionError(MODEL_REGISTRY)


def test_duplicate_different_model_class_registration_is_rejected() -> None:
    class FirstModel:
        pass

    class SecondModel:
        pass

    register_model("test_model")(FirstModel)

    with pytest.raises(ValueError) as exc_info:
        register_model("test_model")(SecondModel)

    message = str(exc_info.value)
    if "test_model" not in message:
        raise AssertionError(message)
    if "FirstModel" not in message:
        raise AssertionError(message)


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_empty_model_names_are_rejected(name: str) -> None:
    class TestModel:
        pass

    with pytest.raises(ValueError, match="Model name"):
        register_model(name)(TestModel)

    with pytest.raises(ValueError, match="Model name"):
        get_model_class(name)
