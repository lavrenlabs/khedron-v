from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

TModel = TypeVar("TModel", bound=object)

MODEL_REGISTRY: dict[str, type[object]] = {}


def _validate_model_name(name: str) -> str:
    if not name.strip():
        raise ValueError("Model name must not be empty or whitespace.")
    return name


def _format_available_models() -> str:
    if not MODEL_REGISTRY:
        return "none"
    return ", ".join(sorted(MODEL_REGISTRY))


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Decorator to register a model class with a stable name."""
    model_name = _validate_model_name(name)

    def decorator(cls: type[TModel]) -> type[TModel]:
        existing = MODEL_REGISTRY.get(model_name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Model name '{model_name}' already registered to {existing!r}")
        MODEL_REGISTRY[model_name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[object]:
    """Resolve a model class by name."""
    model_name = _validate_model_name(name)
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise KeyError(
            f"Unknown model '{model_name}'. Available models: {_format_available_models()}"
        ) from None
