from __future__ import annotations

# ruff: noqa: S101
import asyncio
from datetime import UTC, datetime

import pytest

from khedron.budget import BUDGET_ENV_VAR as _BUDGET_ENV_VAR
from khedron.budget import resolve_budget_cap_usd as _resolve_budget_cap_usd
from khedron.cost.tracker import CostTracker
from khedron.errors import ConfigurationError, CostExceededError
from khedron.runner import _ensure_within_budget, _gather_or_cancel_siblings, _RunSuiteResult
from khedron.types import APICallRecord


@pytest.fixture(autouse=True)
def clean_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_BUDGET_ENV_VAR, raising=False)


def _api_call_record(api_call_id: str, cost_usd: float) -> APICallRecord:
    return APICallRecord(
        api_call_id=api_call_id,
        run_id="run-1",
        question_id=None,
        timestamp=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        phase="generate",
        vendor="fake",
        model_id="fake-model",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        cost_usd=cost_usd,
        status="success",
        attempt_number=1,
    )


def test_resolve_budget_cap_defaults_to_config_when_env_unset() -> None:
    assert _resolve_budget_cap_usd(None) is None
    assert _resolve_budget_cap_usd(15.0) == 15.0


def test_resolve_budget_cap_env_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_BUDGET_ENV_VAR, "2.5")

    assert _resolve_budget_cap_usd(None) == 2.5
    assert _resolve_budget_cap_usd(100.0) == 2.5


def test_resolve_budget_cap_env_accepts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_BUDGET_ENV_VAR, "0")

    assert _resolve_budget_cap_usd(15.0) == 0.0


@pytest.mark.parametrize("raw_value", ["abc", "", " ", "-1", "-0.01", "nan", "inf", "1.0.0"])
def test_resolve_budget_cap_rejects_invalid_env_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv(_BUDGET_ENV_VAR, raw_value)

    with pytest.raises(ConfigurationError) as exc_info:
        _resolve_budget_cap_usd(None)

    assert _BUDGET_ENV_VAR in str(exc_info.value)


def test_ensure_within_budget_no_cap_never_raises() -> None:
    _ensure_within_budget(
        cap_usd=None, observed_cost_usd=1_000_000.0, worst_case_cost_usd=2_000_000.0
    )


def test_ensure_within_budget_under_or_at_cap_is_silent() -> None:
    _ensure_within_budget(cap_usd=1.0, observed_cost_usd=0.5, worst_case_cost_usd=0.5)
    _ensure_within_budget(cap_usd=1.0, observed_cost_usd=1.0, worst_case_cost_usd=1.0)


def test_ensure_within_budget_enforces_worst_case_not_observed() -> None:
    # The guarantee under test: observed confirmed spend is under the cap, but the worst case
    # (with an ambiguous billed-but-failed hold) is over it, so the run must halt.
    with pytest.raises(CostExceededError) as exc_info:
        _ensure_within_budget(cap_usd=1.0, observed_cost_usd=0.7, worst_case_cost_usd=1.5)

    error = exc_info.value
    assert error.context["budget_cap_usd"] == 1.0
    # total_cost_usd keeps its present meaning (confirmed spend == observed), option (a).
    assert error.context["total_cost_usd"] == 0.7
    assert error.context["observed_cost_usd"] == 0.7
    assert error.context["worst_case_cost_usd"] == 1.5
    # Both figures are named, distinctly, in the message.
    assert "$1.500000" in str(error)
    assert "$0.700000" in str(error)
    assert "$1.000000" in str(error)


def test_ensure_within_budget_observed_over_cap_still_raises() -> None:
    # When there is no ambiguous residual, worst_case == observed and the check is the old halt.
    with pytest.raises(CostExceededError) as exc_info:
        _ensure_within_budget(cap_usd=1.0, observed_cost_usd=1.5, worst_case_cost_usd=1.5)

    assert exc_info.value.context["total_cost_usd"] == 1.5


def test_cap_trigger_with_fake_tracker_exceeding_cap() -> None:
    tracker = CostTracker()
    tracker.record(_api_call_record("call-1", 0.6))
    tracker.record(_api_call_record("call-2", 0.5))

    spent = tracker.total_cost_usd()
    with pytest.raises(CostExceededError):
        _ensure_within_budget(cap_usd=1.0, observed_cost_usd=spent, worst_case_cost_usd=spent)


def test_tracker_under_cap_does_not_trigger() -> None:
    tracker = CostTracker()
    tracker.record(_api_call_record("call-1", 0.2))
    tracker.record(_api_call_record("call-2", 0.3))

    spent = tracker.total_cost_usd()
    _ensure_within_budget(cap_usd=1.0, observed_cost_usd=spent, worst_case_cost_usd=spent)


def test_suite_result_worst_case_defaults_to_observed_on_the_happy_path() -> None:
    # No ambiguous residual: worst_case == total, so the CLI shows nothing extra (byte-for-byte as
    # before worst-case reporting was introduced).
    result = _RunSuiteResult(suite_id="s", experiment_results=[], total_cost_usd=1.25)
    assert result.worst_case_cost_usd == 1.25
    assert result.worst_case_cost_usd == result.total_cost_usd


def test_suite_result_keeps_a_worst_case_above_observed() -> None:
    # A nonzero residual is preserved and is strictly greater than confirmed spend, so the CLI
    # surfaces it additively.
    result = _RunSuiteResult(
        suite_id="s", experiment_results=[], total_cost_usd=1.0, worst_case_cost_usd=1.4
    )
    assert result.total_cost_usd == 1.0
    assert result.worst_case_cost_usd == 1.4


@pytest.mark.asyncio
async def test_a_budget_refusal_cancels_sibling_questions_before_they_dispatch() -> None:
    # The clean-halt requirement: when one concurrent question refuses on budget, the siblings must
    # be cancelled before they can dispatch more paid work into the cap's remaining headroom. A bare
    # asyncio.gather would leave them running; _gather_or_cancel_siblings must not.
    dispatched: list[str] = []
    sibling_running = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def refusing() -> str:
        raise CostExceededError("worst-case over cap", budget_cap_usd=1.0)

    async def sibling() -> str:
        sibling_running.set()
        try:
            await asyncio.sleep(5)  # stands in for reaching dispatch
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        dispatched.append("sibling")  # only reached if never cancelled
        return "sibling"

    refuse_task = asyncio.create_task(refusing())
    sibling_task = asyncio.create_task(sibling())
    await sibling_running.wait()  # the sibling is actively running when the refusal lands

    with pytest.raises(CostExceededError):
        await _gather_or_cancel_siblings([refuse_task, sibling_task])

    assert sibling_cancelled.is_set()  # it was cancelled, not left running
    assert dispatched == []  # and never got to "dispatch" more paid work


@pytest.mark.asyncio
async def test_gather_returns_results_in_order_when_all_succeed() -> None:
    async def value(n: int) -> int:
        await asyncio.sleep(0)
        return n

    tasks = [asyncio.create_task(value(n)) for n in range(4)]
    assert await _gather_or_cancel_siblings(tasks) == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_and_drains_every_task() -> None:
    # A cancellation of the whole conversation must leave no question task running
    # detached -- the finally cancels every task and drains it to completion, even when the cancel
    # lands while tasks are still in flight.
    cancelled: list[int] = []
    running = [asyncio.Event() for _ in range(3)]

    async def worker(index: int) -> int:
        running[index].set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        return index

    tasks = [asyncio.create_task(worker(index)) for index in range(3)]
    operation = asyncio.create_task(_gather_or_cancel_siblings(tasks))
    for event in running:
        await event.wait()  # every worker is in flight when the cancellation lands

    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert sorted(cancelled) == [0, 1, 2]  # every worker was cancelled
    assert all(task.done() for task in tasks)  # and drained to completion -- none orphaned


@pytest.mark.asyncio
async def test_cancellation_during_the_drain_still_finishes_draining() -> None:
    # A specific race: a cancellation that arrives WHILE the drain is in progress must
    # not abandon it. A slow task parks in its CancelledError handler so the drain is demonstrably
    # in flight; the operation is then cancelled mid-drain. The guarantee under test is that the
    # operation still ends by raising CancelledError with every task drained to completion (none
    # orphaned) -- not that any particular task runs to its normal end.
    release = asyncio.Event()
    draining = asyncio.Event()

    async def failing() -> int:
        raise CostExceededError("halt", budget_cap_usd=1.0)

    async def slow_to_cancel() -> int:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # The first cancel was just consumed by this except, so awaiting here parks the task and
            # holds the drain open until the test releases it (or a further cancel lands).
            draining.set()
            await release.wait()
            raise
        return 0

    tasks = [asyncio.create_task(failing()), asyncio.create_task(slow_to_cancel())]
    operation = asyncio.create_task(_gather_or_cancel_siblings(tasks))
    await draining.wait()  # the failure fired; the drain is parked on the slow task

    operation.cancel()  # cancellation lands DURING the drain
    await asyncio.sleep(0)
    release.set()  # unblock the park in case this asyncio version does not re-cancel it

    with pytest.raises(asyncio.CancelledError):
        await operation
    assert all(task.done() for task in tasks)  # the drain finished; no task left orphaned
