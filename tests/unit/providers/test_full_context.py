from __future__ import annotations

import pytest
from pydantic import ValidationError

from khedron.providers.full_context import FullContextProviderConfig


def test_full_context_config_defaults_to_unbounded() -> None:
    config = FullContextProviderConfig()

    if config.max_memories_per_search is not None:
        raise AssertionError(config)


@pytest.mark.parametrize("max_memories_per_search", [0, -1])
def test_full_context_config_rejects_non_positive_caps(
    max_memories_per_search: int,
) -> None:
    with pytest.raises(ValidationError):
        FullContextProviderConfig(max_memories_per_search=max_memories_per_search)


def test_full_context_config_is_frozen() -> None:
    config = FullContextProviderConfig(max_memories_per_search=1)

    with pytest.raises(ValidationError):
        config.max_memories_per_search = 2
