from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

TBenchmark = TypeVar("TBenchmark", bound=object)

BENCHMARK_REGISTRY: dict[str, type[object]] = {}


def _validate_benchmark_name(name: str) -> str:
    if not name.strip():
        raise ValueError("Benchmark name must not be empty or whitespace.")
    return name


def _format_available_benchmarks() -> str:
    if not BENCHMARK_REGISTRY:
        return "none"
    return ", ".join(sorted(BENCHMARK_REGISTRY))


def register_benchmark(name: str) -> Callable[[type[TBenchmark]], type[TBenchmark]]:
    """Decorator to register a benchmark class with a stable name."""
    benchmark_name = _validate_benchmark_name(name)

    def decorator(cls: type[TBenchmark]) -> type[TBenchmark]:
        existing = BENCHMARK_REGISTRY.get(benchmark_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Benchmark name '{benchmark_name}' already registered to {existing!r}"
            )
        BENCHMARK_REGISTRY[benchmark_name] = cls
        return cls

    return decorator


def get_benchmark_class(name: str) -> type[object]:
    """Resolve a benchmark class by name."""
    benchmark_name = _validate_benchmark_name(name)
    try:
        return BENCHMARK_REGISTRY[benchmark_name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available benchmarks: {_format_available_benchmarks()}"
        ) from None
