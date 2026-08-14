from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

TProvider = TypeVar("TProvider", bound=object)

PROVIDER_REGISTRY: dict[str, type[object]] = {}


def _validate_provider_name(name: str) -> str:
    if not name.strip():
        raise ValueError("Provider name must not be empty or whitespace.")
    return name


def _format_available_providers() -> str:
    if not PROVIDER_REGISTRY:
        return "none"
    return ", ".join(sorted(PROVIDER_REGISTRY))


def register_provider(name: str) -> Callable[[type[TProvider]], type[TProvider]]:
    """Decorator to register a provider class with a stable name."""
    provider_name = _validate_provider_name(name)

    def decorator(cls: type[TProvider]) -> type[TProvider]:
        existing = PROVIDER_REGISTRY.get(provider_name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Provider name '{provider_name}' already registered to {existing!r}")
        PROVIDER_REGISTRY[provider_name] = cls
        return cls

    return decorator


def get_provider_class(name: str) -> type[object]:
    """Resolve a provider class by name."""
    provider_name = _validate_provider_name(name)
    try:
        return PROVIDER_REGISTRY[provider_name]
    except KeyError:
        raise KeyError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {_format_available_providers()}"
        ) from None
