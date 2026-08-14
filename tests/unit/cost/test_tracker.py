from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khedron.cost.tracker import CostTracker
from khedron.types import APICallRecord

NOW = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)


def api_call(
    api_call_id: str,
    *,
    phase: str = "generate",
    model_id: str = "gpt-4o-mini-2024-07-18",
    cost_usd: float = 0.01,
) -> APICallRecord:
    return APICallRecord(
        api_call_id=api_call_id,
        run_id="run-1",
        question_id="q-1",
        timestamp=NOW,
        phase=phase,
        vendor="openai",
        model_id=model_id,
        input_tokens=100,
        output_tokens=20,
        latency_ms=250.0,
        cost_usd=cost_usd,
        status="success",
        attempt_number=1,
    )


def test_empty_tracker_returns_zero_and_empty_breakdowns() -> None:
    tracker = CostTracker()

    if tracker.total_cost_usd() != 0.0:
        raise AssertionError(tracker.total_cost_usd())
    if tracker.cost_by_phase() != {}:
        raise AssertionError(tracker.cost_by_phase())
    if tracker.cost_by_model() != {}:
        raise AssertionError(tracker.cost_by_model())


def test_total_cost_aggregation() -> None:
    tracker = CostTracker()
    tracker.record(api_call("api-1", cost_usd=0.01))
    tracker.record(api_call("api-2", cost_usd=0.02))

    if tracker.total_cost_usd() != pytest.approx(0.03):
        raise AssertionError(tracker.total_cost_usd())


def test_cost_by_phase_aggregation() -> None:
    tracker = CostTracker()
    tracker.record(api_call("api-1", phase="generate", cost_usd=0.01))
    tracker.record(api_call("api-2", phase="judge", cost_usd=0.02))
    tracker.record(api_call("api-3", phase="generate", cost_usd=0.03))

    if tracker.cost_by_phase() != pytest.approx({"generate": 0.04, "judge": 0.02}):
        raise AssertionError(tracker.cost_by_phase())


def test_cost_by_model_aggregation() -> None:
    tracker = CostTracker()
    tracker.record(api_call("api-1", model_id="gpt-4o-mini-2024-07-18", cost_usd=0.01))
    tracker.record(api_call("api-2", model_id="claude-sonnet-4-5", cost_usd=0.02))
    tracker.record(api_call("api-3", model_id="gpt-4o-mini-2024-07-18", cost_usd=0.03))

    expected = {"gpt-4o-mini-2024-07-18": 0.04, "claude-sonnet-4-5": 0.02}
    if tracker.cost_by_model() != pytest.approx(expected):
        raise AssertionError(tracker.cost_by_model())


def test_record_keeps_canonical_api_call_values_unmutated() -> None:
    tracker = CostTracker()
    record = api_call("api-1", phase="judge", model_id="claude-sonnet-4-5", cost_usd=0.25)
    before = record.model_dump()

    tracker.record(record)

    if record.model_dump() != before:
        raise AssertionError(record)
    if tracker.total_cost_usd() != pytest.approx(0.25):
        raise AssertionError(tracker.total_cost_usd())
    if tracker.cost_by_phase() != {"judge": 0.25}:
        raise AssertionError(tracker.cost_by_phase())
    if tracker.cost_by_model() != {"claude-sonnet-4-5": 0.25}:
        raise AssertionError(tracker.cost_by_model())
