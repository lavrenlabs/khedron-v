from __future__ import annotations

import asyncio

import pytest

from khedron.ratelimit import (
    ImpossibleReservationError,
    RateLimit,
    RateLimiter,
    full_jitter_backoff,
)


class FakeClock:
    """A monotonic clock whose sleep advances virtual time instead of waiting.

    Advancing on sleep is what makes the reserve loop testable: a blocked reservation re-checks the
    window after the clock has moved past the entries that were in its way, with no real time
    passing and no reliance on task scheduling for its final state.
    """

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)


# --- The pure decision logic: asserted with plain numbers, no async ---


def test_a_call_over_the_token_budget_waits_until_the_window_rolls() -> None:
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)
    bucket = limiter._path("m", None)[0]

    bucket.add(now=1000.0, entry_id=0, tokens=100)
    # Full: a further 1 token cannot be admitted, and the wait is exactly the time until the entry
    # admitted at t=1000 leaves the 60s window.
    wait = bucket.try_admit(now=1000.0, tokens=1)
    if wait != pytest.approx(60.0):
        raise AssertionError(wait)
    # 30s later, still full, and the wait has shrunk by exactly the elapsed time.
    if bucket.try_admit(now=1030.0, tokens=1) != pytest.approx(30.0):
        raise AssertionError(bucket.try_admit(now=1030.0, tokens=1))
    # Once the entry expires the same call is admitted (None == admit).
    if bucket.try_admit(now=1060.001, tokens=1) is not None:
        raise AssertionError("a call after the window rolled should be admitted")


def test_requests_per_minute_blocks_small_calls_that_fit_the_token_budget() -> None:
    limiter = RateLimiter(
        {("model", "m"): RateLimit(tokens_per_minute=10_000, requests_per_minute=2)}
    )
    bucket = limiter._path("m", None)[0]

    bucket.add(now=1000.0, entry_id=0, tokens=1)
    bucket.add(now=1001.0, entry_id=1, tokens=1)
    # Tokens are nowhere near the budget, but the third request exceeds RPM=2 and must wait for the
    # oldest request to leave the window. This is the failure the token-only design missed.
    wait = bucket.try_admit(now=1002.0, tokens=1)
    if wait != pytest.approx(58.0):
        raise AssertionError(wait)


def test_settling_to_real_usage_frees_the_over_reserved_capacity() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]

    bucket.add(now=1000.0, entry_id=0, tokens=90)  # reserved input + max_output
    if bucket.try_admit(now=1000.0, tokens=50) != pytest.approx(60.0):
        raise AssertionError("a 90-token hold should block a 50-token call against a 100 budget")
    bucket.settle(entry_id=0, tokens=20)  # the answer came back short
    if bucket.try_admit(now=1000.0, tokens=50) is not None:
        raise AssertionError("settling the hold to 20 should admit the 50-token call")


def test_release_frees_the_hold_entirely() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.add(now=1000.0, entry_id=0, tokens=100)
    bucket.release(entry_id=0)
    if bucket.tokens_in_window(now=1000.0) != 0:
        raise AssertionError("release should remove the hold")


def test_an_unreleased_hold_stays_until_its_window_expires() -> None:
    # The post-dispatch safety rule at the ledger level: a hold nobody releases keeps occupying the
    # window until it expires, so an ambiguous dispatch outcome never lets the pacer over-admit.
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.add(now=1000.0, entry_id=0, tokens=100)
    if bucket.tokens_in_window(now=1030.0) != 100:
        raise AssertionError("an unreleased hold must persist within its window")
    if bucket.tokens_in_window(now=1060.001) != 0:
        raise AssertionError("the hold must expire with its window and no sooner")


def test_a_cooldown_blocks_the_whole_bucket_until_it_ends() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.cool_down(now=1000.0, retry_after=3.0)
    if bucket.try_admit(now=1000.0, tokens=1) != pytest.approx(3.0):
        raise AssertionError("a cooldown must block even a tiny call")
    if bucket.try_admit(now=1003.001, tokens=1) is not None:
        raise AssertionError("the cooldown must clear when it ends")


def test_header_reconciliation_tightens_but_never_loosens() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.add(now=1000.0, entry_id=0, tokens=40)  # ledger predicts 60 remaining

    # Server says only 10 remain (another process spent 50 we cannot see): a deficit of 50 is held
    # until the reported reset, so a 40-token call that the ledger alone would admit now waits.
    bucket.reconcile(now=1000.0, server_remaining_tokens=10, reset_at=1005.0)
    if bucket.try_admit(now=1000.0, tokens=40) is None:
        raise AssertionError("a server-reported shortfall should tighten admission")

    # After the reported reset the deficit is gone.
    if bucket.try_admit(now=1005.001, tokens=40) is not None:
        raise AssertionError("the deficit must clear at the reported reset")

    # A server 'remaining' larger than the ledger predicts is not authority to admit more.
    bucket2 = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})._path("m", None)[0]
    bucket2.add(now=1000.0, entry_id=0, tokens=90)  # ledger predicts 10 remaining
    bucket2.reconcile(now=1000.0, server_remaining_tokens=80, reset_at=1005.0)
    if bucket2.try_admit(now=1000.0, tokens=50) is None:
        raise AssertionError("a generous 'remaining' must not loosen the local ledger")


def test_full_jitter_backoff_is_bounded_floored_and_actually_zero_reaching() -> None:
    # Full jitter, not bounded-multiplicative: samples must be able to reach near zero, and must
    # never exceed the cap for the attempt. Deterministic via an injected rng.
    lo = full_jitter_backoff(0, rng=lambda a, b: a)
    hi = full_jitter_backoff(0, rng=lambda a, b: b)
    if lo != pytest.approx(0.0):
        raise AssertionError(f"full jitter must reach 0, got {lo}")
    if hi != pytest.approx(1.0):
        raise AssertionError(f"attempt 0 ceiling should be base=1.0, got {hi}")
    # Attempt 3 ceiling is base*2**3 = 8.
    if full_jitter_backoff(3, rng=lambda a, b: b) != pytest.approx(8.0):
        raise AssertionError(full_jitter_backoff(3, rng=lambda a, b: b))
    # Cap bounds a large attempt.
    if full_jitter_backoff(20, rng=lambda a, b: b) != pytest.approx(32.0):
        raise AssertionError("the cap must bound a large attempt")
    # Retry-After is a floor even when the jitter samples below it.
    if full_jitter_backoff(0, retry_after=5.0, rng=lambda a, b: a) != pytest.approx(5.0):
        raise AssertionError("Retry-After must floor the backoff")


# --- The async reserve loop, driven by the fake clock ---


def test_reserve_blocks_then_admits_once_the_window_rolls() -> None:
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        first = await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)
        first.keep()  # dispatched, outcome ambiguous: the hold stands
        # The window is full; this reserve must wait ~60s (virtual) and then succeed.
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)

    asyncio.run(scenario())
    if not clock.sleeps or sum(clock.sleeps) < 60.0:
        raise AssertionError(f"the second reserve should have waited ~60s, slept {clock.sleeps}")


def test_an_impossible_reservation_fails_fast_instead_of_hanging() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})

    async def scenario() -> None:
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=101)

    with pytest.raises(ImpossibleReservationError):
        asyncio.run(scenario())


def test_the_generator_and_judge_share_one_scope_bucket() -> None:
    # Two different models under one org/project scope must see each other's holds: the answer model
    # filling the shared bucket makes the judge wait, which is the whole reason a limiter is shared.
    clock = FakeClock()
    limiter = RateLimiter({("scope", "openai-org"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        answer = await limiter.reserve(
            model_id="gpt-answer", scope_id="openai-org", estimated_tokens=100
        )
        answer.keep()
        # A different model, same scope: it must wait for the shared bucket to roll.
        await limiter.reserve(model_id="claude-judge", scope_id="openai-org", estimated_tokens=100)

    asyncio.run(scenario())
    if sum(clock.sleeps) < 60.0:
        raise AssertionError(f"the judge should have waited on the shared bucket, {clock.sleeps}")


def test_a_retry_after_cooldown_is_shared_across_the_path() -> None:
    clock = FakeClock()
    limiter = RateLimiter(
        {("scope", "openai-org"): RateLimit(tokens_per_minute=10_000)}, clock=clock
    )
    limiter.note_rate_limited(model_id="gpt-answer", scope_id="openai-org", retry_after=4.0)

    async def scenario() -> None:
        # A call on a different model of the same scope still waits the cooldown the first raised.
        await limiter.reserve(model_id="claude-judge", scope_id="openai-org", estimated_tokens=1)

    asyncio.run(scenario())
    if sum(clock.sleeps) < 4.0:
        raise AssertionError(f"the cooldown must apply across the shared path, {clock.sleeps}")


def test_an_unconfigured_call_is_unconstrained() -> None:
    # No limit configured for this model: the pacer must impose nothing and never block, so a run
    # without configured limits behaves exactly as before the pacer existed.
    clock = FakeClock()
    limiter = RateLimiter({}, clock=clock)

    async def scenario() -> None:
        reservation = await limiter.reserve(
            model_id="unlimited", scope_id=None, estimated_tokens=1_000_000
        )
        reservation.settle(500)  # harmless on an empty path

    asyncio.run(scenario())
    if clock.sleeps:
        raise AssertionError(f"an unconfigured call must not wait, slept {clock.sleeps}")


# --- The public Reservation API: settle/release exercised through reserve(), not _Bucket ---


def test_reserve_then_settle_frees_capacity_through_the_public_api() -> None:
    # Binds Reservation.settle: the earlier settle test acted on _Bucket directly, so mutating the
    # public wrapper broke nothing. This goes reserve -> settle -> reserve.
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        held = await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)
        held.settle(10)  # the answer came back short
        # 90 more fits only because settle freed 90; without it this would wait a full window.
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=90)

    asyncio.run(scenario())
    if sum(clock.sleeps) > 0.0:
        raise AssertionError(
            f"settle through the public API should have freed room, {clock.sleeps}"
        )


def test_reserve_then_release_frees_the_bucket_through_the_public_api() -> None:
    # Binds Reservation.release: reserve -> release (pre-dispatch failure) -> reserve.
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        held = await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)
        held.release()  # request never dispatched
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)

    asyncio.run(scenario())
    if sum(clock.sleeps) > 0.0:
        raise AssertionError(
            f"release through the public API should have freed the bucket, {clock.sleeps}"
        )


# --- reconcile: the bug, and the fail-safe max rule ---


def test_a_stricter_second_header_grows_the_deficit_it_must_not_shrink() -> None:
    # The blocking defect: a second, stricter header recomputed against a window that already held
    # the first deficit, shrinking it. remaining=0 must leave zero headroom, not 40.
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.add(now=1000.0, entry_id=0, tokens=40)
    bucket.reconcile(now=1000.0, server_remaining_tokens=10, reset_at=1005.0)  # deficit 50
    bucket.reconcile(now=1001.0, server_remaining_tokens=0, reset_at=1006.0)  # server: nothing left
    if bucket.tokens_in_window(now=1001.0) != 100:
        raise AssertionError(
            f"window should be full at the budget, got {bucket.tokens_in_window(1001.0)}"
        )
    if bucket.try_admit(now=1001.0, tokens=1) is None:
        raise AssertionError("a stricter header must not leave room the server said is gone")


def test_a_looser_second_header_still_below_prediction_does_not_loosen() -> None:
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]
    bucket.add(now=1000.0, entry_id=0, tokens=40)
    bucket.reconcile(now=1000.0, server_remaining_tokens=10, reset_at=1005.0)  # deficit 50
    # A later, more generous header (30 remain) still below the local prediction (60): the fail-safe
    # max rule keeps the deficit at 50, it does not loosen to 30.
    bucket.reconcile(now=1001.0, server_remaining_tokens=30, reset_at=1006.0)
    if bucket.try_admit(now=1001.0, tokens=11) is None:
        raise AssertionError("the deficit must not loosen below the max seen before the reset")
    if bucket.try_admit(now=1001.0, tokens=10) is not None:
        raise AssertionError("exactly the max-deficit headroom (10) should remain")


def test_a_reserve_blocked_only_by_a_deficit_waits_for_the_reset_not_the_floor() -> None:
    # A deficit with no live entries must wait for the reported reset, not spin on the 1ms floor.
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)
    limiter.note_observed_remaining(
        model_id="m", scope_id=None, remaining_tokens=0, reset_at_monotonic=clock.t + 5.0
    )

    async def scenario() -> None:
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)

    asyncio.run(scenario())
    if sum(clock.sleeps) < 5.0:
        raise AssertionError(f"a deficit-only block must wait ~5s for the reset, {clock.sleeps}")


def test_two_concurrent_reserves_serialize_and_the_second_waits_a_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        await asyncio.gather(
            limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100),
            limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100),
        )

    asyncio.run(scenario())
    # Only one 100-token hold fits the 100 budget, so the second must wait a full window.
    if sum(clock.sleeps) < 60.0:
        raise AssertionError(
            f"two concurrent full reserves must not both admit at once, {clock.sleeps}"
        )


def test_negative_limits_and_estimates_are_rejected() -> None:
    with pytest.raises(ValueError):
        RateLimit(tokens_per_minute=-1)
    with pytest.raises(ValueError):
        RateLimit(requests_per_minute=0)

    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})

    async def scenario() -> None:
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=-5)

    with pytest.raises(ValueError):
        asyncio.run(scenario())


# --- Malformed public inputs must be rejected, not silently admitted (found in review) ---


def test_a_nan_estimate_is_rejected_and_never_admitted() -> None:
    # float("nan") passes every < and > check, so without an explicit type/finiteness guard a NaN
    # reservation would clear capacity and admit unbounded traffic. It must raise, and leave the
    # bucket empty so a later real call still sees the full budget.
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    bucket = limiter._path("m", None)[0]

    async def scenario() -> None:
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=float("nan"))  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(scenario())
    if bucket.tokens_in_window(now=1000.0) != 0:
        raise AssertionError("a rejected NaN reservation must not have been recorded")


def test_settling_a_negative_usage_is_rejected_and_keeps_the_hold() -> None:
    # Normalising a negative observed usage to 0 would free a live hold through the public API. It
    # must raise, and the original 100-token hold must remain so the bucket stays full.
    clock = FakeClock()
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)}, clock=clock)

    async def scenario() -> None:
        held = await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)
        with pytest.raises((TypeError, ValueError)):
            held.settle(-1)
        # The hold stands: a second full reserve must still wait a whole window.
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=100)

    asyncio.run(scenario())
    if sum(clock.sleeps) < 60.0:
        raise AssertionError(f"a rejected settle must leave the hold in place, {clock.sleeps}")


def test_a_boolean_is_not_a_valid_count() -> None:
    # bool is an int subclass, so True/False would slip through a bare isinstance(int) check and be
    # counted as 1/0 tokens. Both the limit and the estimate reject it.
    with pytest.raises(TypeError):
        RateLimit(tokens_per_minute=True)  # type: ignore[arg-type]
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})

    async def scenario() -> None:
        await limiter.reserve(model_id="m", scope_id=None, estimated_tokens=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        asyncio.run(scenario())


def test_a_non_finite_cooldown_or_reset_is_rejected() -> None:
    # A NaN or infinite cooldown/reset would never clear, locking the bucket forever.
    limiter = RateLimiter({("model", "m"): RateLimit(tokens_per_minute=100)})
    with pytest.raises(ValueError):
        limiter.note_rate_limited(model_id="m", scope_id=None, retry_after=float("inf"))
    with pytest.raises(ValueError):
        limiter.note_observed_remaining(
            model_id="m", scope_id=None, remaining_tokens=0, reset_at_monotonic=float("nan")
        )


def test_full_jitter_backoff_rejects_a_non_finite_or_bad_attempt() -> None:
    # A NaN Retry-After would make max(nan, x) return nan and destroy the floor the backoff
    # guarantees; a negative or non-int attempt index has no well-defined ceiling. Both raise.
    with pytest.raises(ValueError):
        full_jitter_backoff(0, retry_after=float("nan"))
    with pytest.raises(ValueError):
        full_jitter_backoff(0, retry_after=float("inf"))
    with pytest.raises(ValueError):
        full_jitter_backoff(0, retry_after=-1.0)
    with pytest.raises(ValueError):
        full_jitter_backoff(-1)
    with pytest.raises(TypeError):
        full_jitter_backoff(True)  # type: ignore[arg-type]


def test_full_jitter_backoff_clamps_a_huge_attempt_without_computing_a_giant_power() -> None:
    # A valid but enormous attempt index must return the cap immediately, not raise 2 to a billion
    # first. If the exponent were not clamped before the power, this call would hang or OOM.
    if full_jitter_backoff(10**9, rng=lambda _a, b: b) != pytest.approx(32.0):
        raise AssertionError("a huge attempt index must return the cap")


def test_pacer_name_qualifies_by_vendor_and_rejects_the_control_char() -> None:
    from khedron.ratelimit import pacer_name

    a = pacer_name("openai", "gpt-4o-mini")
    b = pacer_name("anthropic", "gpt-4o-mini")
    if a == b:
        raise AssertionError("the same model name under two vendors must qualify to distinct keys")
    with pytest.raises(ValueError):
        pacer_name("open\x1fai", "m")  # a vendor smuggling the qualifier must be rejected
