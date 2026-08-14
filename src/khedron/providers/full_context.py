from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from khedron.providers.base import MemoryProvider, ProviderHealthStatus
from khedron.providers.registry import register_provider
from khedron.types import Memory
from khedron.utils.filters import SupportsMetadata, matches_filter
from khedron.utils.ids import generate_ulid

__all__ = ["FullContextProvider", "FullContextProviderConfig"]


class FullContextProviderConfig(BaseModel):
    """Configuration for the deterministic full-context baseline provider."""

    model_config = ConfigDict(frozen=True)

    max_memories_per_search: int | None = Field(default=None, gt=0)


@register_provider("full_context")
class FullContextProvider(MemoryProvider):
    """No-memory baseline that returns stored memories in insertion order."""

    # Returns every stored memory and discards `top_k` by design: it is the no-memory
    # baseline standing in for passing the whole history to the model, so honouring a retrieval
    # budget would turn it into a retrieval-limited provider and destroy the baseline.
    honours_top_k = False

    def __init__(self, config: FullContextProviderConfig) -> None:
        self._config = config
        self._memories: list[Memory] = []

    @property
    def provider_type(self) -> str:
        return "full_context"

    @property
    def provider_version(self) -> str:
        return "1.0"

    async def initialize(self) -> None:
        return None

    async def health_check(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(healthy=True, version=self.provider_version)

    async def reset(self) -> None:
        self._memories.clear()

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_id = generate_ulid()
        self._memories.append(
            Memory(
                memory_id=memory_id,
                content=content,
                metadata=dict(metadata) if metadata is not None else {},
                timestamp=datetime.now(UTC),
            )
        )
        return memory_id

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        # Full-context mode intentionally ignores query relevance and top_k.
        del query, top_k

        memories = list(self._memories)
        if metadata_filter is not None:
            memories = [
                memory
                for memory in memories
                if matches_filter(cast(SupportsMetadata, memory), metadata_filter)
            ]

        if self._config.max_memories_per_search is not None:
            memories = memories[: self._config.max_memories_per_search]
        return memories

    async def close(self) -> None:
        return None
