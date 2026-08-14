from __future__ import annotations

from collections.abc import Iterator

import pytest

from khedron.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    get_benchmark_class,
    register_benchmark,
)


@pytest.fixture(autouse=True)
def isolate_benchmark_registry() -> Iterator[None]:
    snapshot = BENCHMARK_REGISTRY.copy()
    BENCHMARK_REGISTRY.clear()
    try:
        yield
    finally:
        BENCHMARK_REGISTRY.clear()
        BENCHMARK_REGISTRY.update(snapshot)


def test_register_benchmark_decorator_resolves_class() -> None:
    @register_benchmark("test_benchmark")
    class TestBenchmark:
        pass

    if get_benchmark_class("test_benchmark") is not TestBenchmark:
        raise AssertionError(BENCHMARK_REGISTRY)


def test_register_benchmark_decorator_returns_original_class() -> None:
    class TestBenchmark:
        pass

    decorated = register_benchmark("test_benchmark")(TestBenchmark)

    if decorated is not TestBenchmark:
        raise AssertionError(decorated)


def test_unknown_benchmark_lookup_lists_sorted_available_names() -> None:
    class AlphaBenchmark:
        pass

    class BravoBenchmark:
        pass

    class ZuluBenchmark:
        pass

    register_benchmark("zulu")(ZuluBenchmark)
    register_benchmark("alpha")(AlphaBenchmark)
    register_benchmark("bravo")(BravoBenchmark)

    with pytest.raises(KeyError) as exc_info:
        get_benchmark_class("missing")

    message = str(exc_info.value)
    if "missing" not in message:
        raise AssertionError(message)
    if "Available benchmarks: alpha, bravo, zulu" not in message:
        raise AssertionError(message)


def test_duplicate_same_benchmark_class_registration_is_idempotent() -> None:
    class TestBenchmark:
        pass

    first = register_benchmark("test_benchmark")(TestBenchmark)
    second = register_benchmark("test_benchmark")(TestBenchmark)

    if first is not TestBenchmark or second is not TestBenchmark:
        raise AssertionError((first, second))
    if get_benchmark_class("test_benchmark") is not TestBenchmark:
        raise AssertionError(BENCHMARK_REGISTRY)
    if len(BENCHMARK_REGISTRY) != 1:
        raise AssertionError(BENCHMARK_REGISTRY)


def test_duplicate_different_benchmark_class_registration_is_rejected() -> None:
    class FirstBenchmark:
        pass

    class SecondBenchmark:
        pass

    register_benchmark("test_benchmark")(FirstBenchmark)

    with pytest.raises(ValueError) as exc_info:
        register_benchmark("test_benchmark")(SecondBenchmark)

    message = str(exc_info.value)
    if "test_benchmark" not in message:
        raise AssertionError(message)
    if "FirstBenchmark" not in message:
        raise AssertionError(message)


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_empty_benchmark_names_are_rejected(name: str) -> None:
    class TestBenchmark:
        pass

    with pytest.raises(ValueError, match="Benchmark name"):
        register_benchmark(name)(TestBenchmark)

    with pytest.raises(ValueError, match="Benchmark name"):
        get_benchmark_class(name)
