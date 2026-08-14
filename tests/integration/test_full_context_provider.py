from __future__ import annotations

import pytest

from khedron.providers.full_context import FullContextProvider, FullContextProviderConfig
from khedron.providers.registry import get_provider_class
from khedron.utils.ids import is_valid_ulid


def make_provider(
    max_memories_per_search: int | None = None,
) -> FullContextProvider:
    return FullContextProvider(
        FullContextProviderConfig(max_memories_per_search=max_memories_per_search)
    )


@pytest.mark.asyncio
async def test_provider_resolves_via_registry() -> None:
    provider_class = get_provider_class("full_context")

    if provider_class is not FullContextProvider:
        raise AssertionError(provider_class)


@pytest.mark.asyncio
async def test_search_returns_all_memories_by_default_ignoring_top_k() -> None:
    provider = make_provider()
    expected_contents = [f"memory-{index}" for index in range(100)]

    for content in expected_contents:
        await provider.add(content, {"session_id": "s1"})

    results = await provider.search("memory", top_k=10)

    if [memory.content for memory in results] != expected_contents:
        raise AssertionError(results)


@pytest.mark.asyncio
async def test_search_respects_configured_cap_after_filtering() -> None:
    provider = make_provider(max_memories_per_search=3)

    for index in range(8):
        group = "target" if index % 2 == 0 else "other"
        await provider.add(f"memory-{index}", {"group": group})

    results = await provider.search("memory", top_k=1, metadata_filter={"group": "target"})

    if [memory.content for memory in results] != ["memory-0", "memory-2", "memory-4"]:
        raise AssertionError(results)


@pytest.mark.asyncio
async def test_metadata_filters_are_applied() -> None:
    provider = make_provider()
    await provider.add("Alice moved to Rome.", {"speaker": "Alice", "session_id": "s1"})
    await provider.add("Bob moved to Paris.", {"speaker": "Bob", "session_id": "s1"})
    await provider.add("Alice likes espresso.", {"speaker": "Alice", "session_id": "s2"})

    results = await provider.search("moved", metadata_filter={"speaker": "Alice"})

    if [memory.content for memory in results] != ["Alice moved to Rome.", "Alice likes espresso."]:
        raise AssertionError(results)


@pytest.mark.asyncio
async def test_reset_clears_stored_memories() -> None:
    provider = make_provider()
    await provider.add("Alice moved to Rome.")

    await provider.reset()

    results = await provider.search("Alice")
    if results != []:
        raise AssertionError(results)


@pytest.mark.asyncio
async def test_health_check_reports_version() -> None:
    provider = make_provider()

    health = await provider.health_check()

    if health.healthy is not True:
        raise AssertionError(health)
    if health.version != "1.0":
        raise AssertionError(health)
    if provider.provider_type != "full_context" or provider.provider_version != "1.0":
        raise AssertionError((provider.provider_type, provider.provider_version))


@pytest.mark.asyncio
async def test_initialize_and_close_are_idempotent_and_preserve_memories() -> None:
    provider = make_provider()
    await provider.initialize()
    await provider.initialize()
    await provider.add("Alice moved to Rome.")

    await provider.close()
    await provider.close()

    results = await provider.search("Alice", top_k=1)
    if [memory.content for memory in results] != ["Alice moved to Rome."]:
        raise AssertionError(results)


@pytest.mark.asyncio
async def test_stored_memory_ids_are_unique_valid_ulids() -> None:
    provider = make_provider()

    memory_ids = [await provider.add(f"memory-{index}") for index in range(25)]

    if len(set(memory_ids)) != len(memory_ids):
        raise AssertionError(memory_ids)
    invalid_ids = [memory_id for memory_id in memory_ids if not is_valid_ulid(memory_id)]
    if invalid_ids:
        raise AssertionError(invalid_ids)

    results = await provider.search("memory")
    if [memory.memory_id for memory in results] != memory_ids:
        raise AssertionError(results)
