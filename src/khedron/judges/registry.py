from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

TJudge = TypeVar("TJudge", bound=object)

JUDGE_REGISTRY: dict[str, type[object]] = {}


def _validate_judge_name(name: str) -> str:
    if not name.strip():
        raise ValueError("Judge name must not be empty or whitespace.")
    return name


def _format_available_judges() -> str:
    if not JUDGE_REGISTRY:
        return "none"
    return ", ".join(sorted(JUDGE_REGISTRY))


def register_judge(name: str) -> Callable[[type[TJudge]], type[TJudge]]:
    """Decorator to register a judge class with a stable name."""
    judge_name = _validate_judge_name(name)

    def decorator(cls: type[TJudge]) -> type[TJudge]:
        existing = JUDGE_REGISTRY.get(judge_name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Judge name '{judge_name}' already registered to {existing!r}")
        JUDGE_REGISTRY[judge_name] = cls
        return cls

    return decorator


def get_judge_class(name: str) -> type[object]:
    """Resolve a judge class by name."""
    judge_name = _validate_judge_name(name)
    try:
        return JUDGE_REGISTRY[judge_name]
    except KeyError:
        raise KeyError(
            f"Unknown judge '{judge_name}'. Available judges: {_format_available_judges()}"
        ) from None
