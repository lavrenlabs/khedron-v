from __future__ import annotations

# ruff: noqa: S101
from datetime import date

import pytest

from khedron.budget import (
    BudgetLedger,
    BudgetReservationError,
    upper_bound_cost_usd,
)
from khedron.cost.pricing import PricingEntry, PricingTable
from khedron.errors import CostExceededError, ModelError
from khedron.pacing import (
    TIMEOUT_MAX_RETRIES,
    DispatchOutcome,
    RetryDisposition,
    paced_generate,
)
from khedron.types import APICallResult

# --------------------------------------------------------------------------------------------------
# Shared offline fixtures: no network, no provider, no clock. A dispatch is scripted, extract is the
# identity (the "response" a scripted dispatch returns is already an APICallResult), and classify is
# chosen per test. estimate_cost returns a fixed dollar figure so each test sets the reservation.
# --------------------------------------------------------------------------------------------------

_MAX_COST = 0.30


def _result(*, cost_usd: float, input_tokens: int = 10, output_tokens: int = 5) -> APICallResult:
    return APICallResult(
        output="ok",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=1.0,
        cost_usd=cost_usd,
        model_id="m",
    )


async def _noop_sleep(_delay: float) -> None:
    return None


def _identity(response: object) -> APICallResult:
    assert isinstance(response, APICallResult)
    return response


def _terminal(exc: Exception) -> DispatchOutcome:
    return DispatchOutcome(disposition=RetryDisposition.TERMINAL, make_error=lambda _n: exc)


def _retryable(exc: Exception) -> DispatchOutcome:
    return DispatchOutcome(disposition=RetryDisposition.RETRYABLE, make_error=lambda _n: exc)


def _rate_limited(exc: Exception) -> DispatchOutcome:
    return DispatchOutcome(disposition=RetryDisposition.RATE_LIMITED, make_error=lambda _n: exc)


def _timeout(exc: Exception) -> DispatchOutcome:
    return DispatchOutcome(
        disposition=RetryDisposition.RETRYABLE,
        make_error=lambda _n: exc,
        retry_cap=TIMEOUT_MAX_RETRIES,
    )


class _ScriptedDispatch:
    """A dispatch that plays a queued list of outcomes and records whether it was ever awaited.

    Each behavior is ``("ok", result)`` to return a response, ``("raise", exc)`` to raise one, or
    ``("cancel",)`` to raise ``asyncio.CancelledError``. ``awaited`` proves whether the call was
    reached at all -- the assertion that matters for a pre-dispatch refusal.
    """

    def __init__(self, behaviors: list[tuple]) -> None:
        self._behaviors = list(behaviors)
        self.calls = 0
        self.awaited = False

    async def __call__(self) -> object:
        self.awaited = True
        self.calls += 1
        kind, *rest = self._behaviors.pop(0)
        if kind == "ok":
            return rest[0]
        if kind == "cancel":
            import asyncio

            raise asyncio.CancelledError
        raise rest[0]


async def _run(
    *,
    budget: BudgetLedger | None,
    dispatch: _ScriptedDispatch,
    classify=lambda exc: _terminal(exc),
    extract=_identity,
    estimate_cost=lambda: _MAX_COST,
    max_retries: int = 0,
    limiter=None,
) -> APICallResult:
    return await paced_generate(
        limiter=limiter,
        vendor="v",
        model_id="m",
        scope=None,
        estimate=lambda: 0,
        max_retries=max_retries,
        dispatch=dispatch,
        extract=extract,
        classify=classify,
        backoff_sleep=_noop_sleep,
        budget=budget,
        estimate_cost=estimate_cost,
    )


# --------------------------------------------------------------------------------------------------
# BudgetLedger + CostReservation lifecycle (direct unit tests)
# --------------------------------------------------------------------------------------------------


def test_settle_moves_a_hold_from_outstanding_to_observed() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 10.0)
    reservation = ledger.reserve(0.40)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.40)
    assert ledger.observed_cost_usd() == pytest.approx(0.0)

    reservation.settle(0.10)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
    assert ledger.observed_cost_usd() == pytest.approx(0.10)
    assert ledger.worst_case_cost_usd() == pytest.approx(0.10)


def test_kept_hold_contributes_its_reserved_maximum_to_worst_case() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 10.0)
    ledger.reserve(0.50).keep()
    # A kept hold is worst-case spend, not confirmed spend.
    assert ledger.observed_cost_usd() == pytest.approx(0.0)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.50)
    assert ledger.worst_case_cost_usd() == pytest.approx(0.50)


def test_released_hold_contributes_nothing() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 10.0)
    ledger.reserve(0.50).release()
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(0.0)


def test_a_reservation_disposes_exactly_once() -> None:
    # settle wins; a later release/keep is a no-op, so a hold is never double-freed or resurrected.
    ledger = BudgetLedger(cap_provider=lambda: 10.0)
    reservation = ledger.reserve(0.50)
    reservation.settle(0.20)
    reservation.release()
    reservation.keep()
    assert ledger.observed_cost_usd() == pytest.approx(0.20)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)


def test_reserve_refuses_when_worst_case_would_breach_the_cap() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 1.0, prior_spend_usd=0.9)
    with pytest.raises(BudgetReservationError) as exc_info:
        ledger.reserve(0.3)
    error = exc_info.value
    assert isinstance(error, CostExceededError)  # so the run/suite halt path treats it identically
    assert error.context["budget_cap_usd"] == pytest.approx(1.0)
    assert error.context["observed_cost_usd"] == pytest.approx(0.9)
    assert error.context["worst_case_cost_usd"] == pytest.approx(1.2)
    assert error.context["attempted_cost_usd"] == pytest.approx(0.3)
    # A refused reservation changes nothing: it was never granted.
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
    assert ledger.observed_cost_usd() == pytest.approx(0.9)


def test_reserve_at_exactly_the_cap_is_allowed() -> None:
    # The cap is a strict-exceed threshold, so a reservation landing exactly on it fits.
    ledger = BudgetLedger(cap_provider=lambda: 1.0, prior_spend_usd=0.7)
    ledger.reserve(0.3)
    assert ledger.worst_case_cost_usd() == pytest.approx(1.0)


def test_no_cap_never_refuses_but_still_tracks_worst_case() -> None:
    ledger = BudgetLedger(cap_provider=lambda: None)
    ledger.reserve(1_000.0).keep()
    assert ledger.worst_case_cost_usd() == pytest.approx(1_000.0)


def test_the_cap_is_read_live_not_snapshotted() -> None:
    # recover-blocked re-resolves the cap mid-invocation; the ledger must honour the new value.
    cap = {"value": 1.0}
    ledger = BudgetLedger(cap_provider=lambda: cap["value"])
    ledger.reserve(0.5).settle(0.5)  # observed 0.5, under cap
    cap["value"] = 0.6
    with pytest.raises(BudgetReservationError):
        ledger.reserve(0.2)  # 0.5 + 0.2 = 0.7 > the new 0.6 cap


@pytest.mark.parametrize("bad", [-0.01, float("inf"), float("nan")])
def test_reserve_and_settle_reject_a_negative_or_non_finite_amount(bad: float) -> None:
    ledger = BudgetLedger(cap_provider=lambda: 10.0)
    with pytest.raises(ValueError):
        ledger.reserve(bad)
    reservation = ledger.reserve(0.5)
    with pytest.raises(ValueError):
        reservation.settle(bad)


def test_prior_spend_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        BudgetLedger(cap_provider=lambda: 1.0, prior_spend_usd=-1.0)


def test_seed_prior_spend_makes_the_cap_account_for_already_spent_money() -> None:
    # The recover-blocked / resume path: a run that already spent money before this session must not
    # get a fresh cap on top of it. Seeding folds that spend into the committed baseline, so only
    # the remaining headroom is reservable. Without the seed this reservation would be admitted.
    ledger = BudgetLedger(cap_provider=lambda: 1.0)
    ledger.seed_prior_spend(0.8)
    assert ledger.observed_cost_usd() == pytest.approx(0.8)
    ledger.reserve(0.2).keep()  # exactly fills the remaining headroom
    with pytest.raises(BudgetReservationError):
        ledger.reserve(0.01)


def test_seed_prior_spend_of_zero_is_a_no_op_for_a_fresh_run() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 1.0)
    ledger.seed_prior_spend(0.0)
    assert ledger.observed_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------------
# upper_bound_cost_usd: a true ceiling, fail-closed on an unpriceable model
# --------------------------------------------------------------------------------------------------


def _pricing(*, per_million: float = 1_000_000.0, pattern: str = "m") -> PricingTable:
    # per_million == 1e6 makes cost == token_count, so the arithmetic is easy to assert.
    return PricingTable(
        last_updated=date(2026, 1, 1),
        prices=[
            PricingEntry(
                model_pattern=pattern,
                input_per_million_usd=per_million,
                output_per_million_usd=per_million,
            )
        ],
    )


def test_upper_bound_is_byte_length_plus_wrap_plus_output() -> None:
    # 4 ASCII bytes + 32 wrap overhead = 36 input tokens; 10 output tokens. per_million == 1e6 so
    # the dollar figure equals the token count: 36 + 10 = 46.
    cost = upper_bound_cost_usd(
        prompt="abcd", max_output_tokens=10, model_id="m", pricing=_pricing()
    )
    assert cost == pytest.approx(46.0)


def test_upper_bound_uses_utf8_bytes_not_characters() -> None:
    # A 2-byte character must count as (at least) 2, never 1 -- the property that makes it a bound.
    one_char = upper_bound_cost_usd(
        prompt="é", max_output_tokens=0, model_id="m", pricing=_pricing()
    )
    assert one_char == pytest.approx(2.0 + 32.0)


def test_upper_bound_refuses_an_unpriceable_model() -> None:
    with pytest.raises(ValueError):
        upper_bound_cost_usd(
            prompt="abcd",
            max_output_tokens=10,
            model_id="not-in-table",
            pricing=_pricing(pattern="other"),
        )


# --------------------------------------------------------------------------------------------------
# paced_generate with an injected budget: the reservation lifecycle over the real pacer
# --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserve_before_dispatch_then_settle_on_success() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 10.0)

    async def dispatch() -> object:
        # By the time the paid call is dispatched, its worst-case cost is already held: this is what
        # "reserve before dispatch" means, asserted from inside the dispatch itself.
        assert ledger.outstanding_cost_usd() == pytest.approx(_MAX_COST)
        return _result(cost_usd=0.10)

    result = await paced_generate(
        limiter=None,
        vendor="v",
        model_id="m",
        scope=None,
        estimate=lambda: 0,
        max_retries=0,
        dispatch=dispatch,
        extract=_identity,
        classify=lambda exc: _terminal(exc),
        backoff_sleep=_noop_sleep,
        budget=ledger,
        estimate_cost=lambda: _MAX_COST,
    )
    assert result.cost_usd == pytest.approx(0.10)
    # settle replaced the reserved maximum with the real cost; no residual remains.
    assert ledger.observed_cost_usd() == pytest.approx(0.10)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(ledger.observed_cost_usd())


@pytest.mark.asyncio
async def test_refuse_before_dispatch_when_over_cap_bills_nothing() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 1.0, prior_spend_usd=0.9)
    dispatch = _ScriptedDispatch([("ok", _result(cost_usd=0.10))])
    with pytest.raises(BudgetReservationError):
        await _run(budget=ledger, dispatch=dispatch)
    # The bound: dispatch was never awaited, nothing settled, spend unchanged.
    assert dispatch.awaited is False
    assert ledger.observed_cost_usd() == pytest.approx(0.9)
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_keep_on_ambiguous_retryable_holds_one_reservation_per_attempt() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 100.0)
    server_error = ModelError("server blip")
    dispatch = _ScriptedDispatch([("raise", server_error)] * 3)
    with pytest.raises(ModelError):
        await _run(
            budget=ledger,
            dispatch=dispatch,
            classify=lambda exc: _retryable(exc),
            max_retries=2,
        )
    # Three physical attempts, each holding its own kept reservation.
    assert dispatch.calls == 3
    assert ledger.observed_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(3 * _MAX_COST)


@pytest.mark.asyncio
async def test_keep_on_timeout_is_capped_independently_of_max_retries() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 100.0)
    timeout = ModelError("timeout")
    # max_retries is high (8), but the timeout class caps its own retries at TIMEOUT_MAX_RETRIES.
    dispatch = _ScriptedDispatch([("raise", timeout)] * (TIMEOUT_MAX_RETRIES + 1))
    with pytest.raises(ModelError):
        await _run(
            budget=ledger,
            dispatch=dispatch,
            classify=lambda exc: _timeout(exc),
            max_retries=8,
        )
    assert dispatch.calls == TIMEOUT_MAX_RETRIES + 1
    assert ledger.worst_case_cost_usd() == pytest.approx((TIMEOUT_MAX_RETRIES + 1) * _MAX_COST)


@pytest.mark.asyncio
async def test_release_on_rate_limit_does_not_contribute_to_worst_case() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 100.0)
    rate_limited = ModelError("429")
    dispatch = _ScriptedDispatch([("raise", rate_limited)] * 2)
    with pytest.raises(ModelError):
        await _run(
            budget=ledger,
            dispatch=dispatch,
            classify=lambda exc: _rate_limited(exc),
            max_retries=1,
        )
    # A provider-confirmed 429 means the request was not admitted: every hold is released.
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_keep_on_extract_failure_over_a_charged_response() -> None:
    ledger = BudgetLedger(cap_provider=lambda: 100.0)

    def failing_extract(_response: object) -> APICallResult:
        raise ModelError("could not parse a response the provider already charged for")

    dispatch = _ScriptedDispatch([("ok", _result(cost_usd=0.10))])
    with pytest.raises(ModelError):
        await _run(budget=ledger, dispatch=dispatch, extract=failing_extract)
    # A response came back (so it was billed) but extract failed: the hold is kept, at its maximum.
    assert ledger.observed_cost_usd() == pytest.approx(0.0)
    assert ledger.worst_case_cost_usd() == pytest.approx(_MAX_COST)


@pytest.mark.asyncio
async def test_keep_on_cancel_and_propagate() -> None:
    import asyncio

    ledger = BudgetLedger(cap_provider=lambda: 100.0)
    dispatch = _ScriptedDispatch([("cancel",)])
    with pytest.raises(asyncio.CancelledError):
        await _run(budget=ledger, dispatch=dispatch)
    # A cancelled attempt may be in flight and billable, so its reserved max stays in worst_case.
    assert ledger.worst_case_cost_usd() == pytest.approx(_MAX_COST)
    assert ledger.observed_cost_usd() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_budget_without_a_limiter_still_reserves_and_refuses() -> None:
    # The money path is gated on `budget`, not `limiter`: a cap with no rate limits still bounds it.
    ledger = BudgetLedger(cap_provider=lambda: 1.0, prior_spend_usd=0.95)
    dispatch = _ScriptedDispatch([("ok", _result(cost_usd=0.10))])
    with pytest.raises(BudgetReservationError):
        await _run(budget=ledger, dispatch=dispatch, limiter=None)
    assert dispatch.awaited is False


@pytest.mark.asyncio
async def test_no_budget_does_no_money_work() -> None:
    # The mirror: with no budget the pacer never prices a call, so an uncapped run is exactly as
    # before -- estimate_cost is never even evaluated.
    def forbidden_estimate_cost() -> float:
        raise AssertionError("estimate_cost must not be called when budget is None")

    dispatch = _ScriptedDispatch([("ok", _result(cost_usd=0.10))])
    result = await _run(budget=None, dispatch=dispatch, estimate_cost=forbidden_estimate_cost)
    assert result.cost_usd == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_an_unpriceable_model_refuses_before_dispatch() -> None:
    # Decision 3: estimate_cost raises for a model with no usable price, before any reservation or
    # dispatch, so an unpriceable call is refused rather than billed then discovered post-hoc.
    ledger = BudgetLedger(cap_provider=lambda: 100.0)

    def unpriceable() -> float:
        raise ModelError("pricing is not configured for this model")

    dispatch = _ScriptedDispatch([("ok", _result(cost_usd=0.10))])
    with pytest.raises(ModelError):
        await _run(budget=ledger, dispatch=dispatch, estimate_cost=unpriceable)
    assert dispatch.awaited is False
    assert ledger.outstanding_cost_usd() == pytest.approx(0.0)
