from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError

from khedron.providers.base import MemoryProvider, ProviderHealthStatus
from khedron.types import Memory

NOW = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)


class CompleteProvider(MemoryProvider):
    @property
    def provider_type(self) -> str:
        return "synthetic_provider"

    @property
    def provider_version(self) -> str:
        return "1.0"

    async def initialize(self) -> None:
        return None

    async def health_check(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(healthy=True, version=self.provider_version)

    async def reset(self) -> None:
        return None

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if content == "":
            raise AssertionError(content)
        if metadata is not None and not isinstance(metadata, dict):
            raise AssertionError(metadata)
        return "mem-1"

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        if query == "":
            raise AssertionError(query)
        if top_k != 10:
            raise AssertionError(top_k)
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise AssertionError(metadata_filter)
        return [
            Memory(
                memory_id="mem-1",
                content="Alice moved to Rome.",
                metadata={"speaker": "Alice"},
                score=0.9,
                timestamp=NOW,
            )
        ]

    async def close(self) -> None:
        return None


def test_provider_health_status_defaults_and_is_frozen() -> None:
    status = ProviderHealthStatus(healthy=True)

    if status.healthy is not True:
        raise AssertionError(status)
    if status.version is not None:
        raise AssertionError(status)
    if status.details != {}:
        raise AssertionError(status)

    with pytest.raises(ValidationError):
        status.healthy = False


def test_provider_health_status_details_default_isolated() -> None:
    first = ProviderHealthStatus(healthy=True)
    second = ProviderHealthStatus(healthy=True)
    first.details["latency_ms"] = 1.0

    if second.details:
        raise AssertionError(second)


def test_memory_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        MemoryProvider()


def test_incomplete_memory_provider_subclass_is_abstract() -> None:
    class IncompleteProvider(MemoryProvider):
        @property
        def provider_type(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteProvider()


@pytest.mark.asyncio
async def test_complete_memory_provider_subclass_is_awaitable() -> None:
    provider = CompleteProvider()

    if provider.provider_type != "synthetic_provider":
        raise AssertionError(provider.provider_type)
    if provider.provider_version != "1.0":
        raise AssertionError(provider.provider_version)

    await provider.initialize()
    health = await provider.health_check()
    await provider.reset()
    memory_id = await provider.add("Alice moved to Rome.", {"speaker": "Alice"})
    memories = await provider.search("Where did Alice move?")
    await provider.close()

    if health != ProviderHealthStatus(healthy=True, version="1.0"):
        raise AssertionError(health)
    if memory_id != "mem-1":
        raise AssertionError(memory_id)
    if len(memories) != 1 or memories[0].memory_id != "mem-1":
        raise AssertionError(memories)


def test_memory_provider_async_contract_and_annotations() -> None:
    expected_abstract_methods = {
        "add",
        "close",
        "health_check",
        "initialize",
        "provider_type",
        "provider_version",
        "reset",
        "search",
    }
    if MemoryProvider.__abstractmethods__ != expected_abstract_methods:
        raise AssertionError(MemoryProvider.__abstractmethods__)

    for method_name in ("initialize", "health_check", "reset", "add", "search", "close"):
        if not inspect.iscoroutinefunction(getattr(MemoryProvider, method_name)):
            raise AssertionError(method_name)

    add_signature = inspect.signature(MemoryProvider.add)
    search_signature = inspect.signature(MemoryProvider.search)
    if add_signature.parameters["metadata"].default is not None:
        raise AssertionError(add_signature)
    if search_signature.parameters["top_k"].default != 10:
        raise AssertionError(search_signature)
    if search_signature.parameters["metadata_filter"].default is not None:
        raise AssertionError(search_signature)

    if get_type_hints(MemoryProvider.health_check)["return"] is not ProviderHealthStatus:
        raise AssertionError(get_type_hints(MemoryProvider.health_check))
    if get_type_hints(MemoryProvider.search)["return"] != list[Memory]:
        raise AssertionError(get_type_hints(MemoryProvider.search))
