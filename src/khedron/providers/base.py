from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from khedron.types import Memory

__all__ = ["MemoryProvider", "ProviderHealthStatus"]


class ProviderHealthStatus(BaseModel):
    """Health check result for a memory provider."""

    model_config = ConfigDict(frozen=True)

    healthy: bool
    version: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MemoryProvider(ABC):
    """Abstract base for all memory providers."""

    #: Whether `search` honours the `top_k` it is given.
    #:
    #: A methodology profile can pin a retrieval budget, but the budget is meaningless for a
    #: provider that returns everything -- `FullContextProvider` discards it by design.
    #: A run must record that as not-applicable rather than claim a budget it never applied, and
    #: validation must refuse rather than shrug, so the capability is declared here instead of
    #: being inferred from the provider's name.
    honours_top_k: bool = True

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Stable identifier for this provider type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_version(self) -> str:
        """Version of the underlying provider system."""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider resources."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> ProviderHealthStatus:
        """Verify that the provider is operational."""
        raise NotImplementedError

    @abstractmethod
    async def reset(self) -> None:
        """Clear all stored memories."""
        raise NotImplementedError

    @abstractmethod
    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one memory and return the provider's memory identifier."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        """Return memories relevant to the query."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        raise NotImplementedError
