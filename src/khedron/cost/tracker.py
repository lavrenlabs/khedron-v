from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from khedron.types import APICallRecord

__all__ = ["CostTracker"]


class CostTracker:
    """Tracks API costs across a run.

    A run's cost is the cost of every call made under its run_id, including the calls an earlier
    attempt made before it was interrupted. `inherited` carries those back in, so the invariant a
    resumed run has to satisfy -- reported cost equals the sum of its own api_calls.jsonl -- holds
    by construction rather than by each consumer remembering to add the inherited amount.

    Taking the records rather than a scalar keeps the breakdowns honest too: seeding only a total
    would leave `cost_by_phase` and `cost_by_model` describing the second attempt while
    `total_cost_usd` described both, and the two would silently disagree.
    """

    def __init__(self, *, inherited: Sequence[APICallRecord] = ()) -> None:
        self._records: list[APICallRecord] = list(inherited)

    def record(self, api_call: APICallRecord) -> None:
        """Record an API call that has also been persisted to api_calls.jsonl."""

        self._records.append(api_call)

    def total_cost_usd(self) -> float:
        """Return the total cost for recorded API calls."""

        return sum(record.cost_usd for record in self._records)

    def cost_by_phase(self) -> dict[str, float]:
        """Return recorded costs grouped by pipeline phase."""

        result: defaultdict[str, float] = defaultdict(float)
        for record in self._records:
            result[record.phase] += record.cost_usd
        return dict(result)

    def cost_by_model(self) -> dict[str, float]:
        """Return recorded costs grouped by model ID."""

        result: defaultdict[str, float] = defaultdict(float)
        for record in self._records:
            result[record.model_id] += record.cost_usd
        return dict(result)
