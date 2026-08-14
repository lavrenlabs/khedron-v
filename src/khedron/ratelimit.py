"""A request pacer that keeps a run under an account's published rate limits.

This exists because a baseline run lost a question to an OpenAI 429: at ~37k input
tokens per question and two concurrent questions, a single account ran over its 200,000 tokens-per-
minute ceiling, and a lost question is scored as wrong and cannot be recovered by resume. The pacer
is admission control, not billing: it holds capacity before a call and reconciles to the real usage
the call reports, so the account's own limits — not luck — decide the request rate.

The decision logic (`_Bucket.try_admit` and the
window/cooldown accounting) is deliberately synchronous and pure so it can be asserted directly with
plain numbers; only the wait loop in `RateLimiter.reserve` is async. That split is intentional — the
correctness of this module lives in the pure functions, and a test that cannot fail is the failure
mode this project has paid for more than once.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "ImpossibleReservationError",
    "RateLimit",
    "RateLimiter",
    "Reservation",
    "SystemClock",
    "full_jitter_backoff",
    "pacer_name",
]

# The unit separator, chosen to qualify a bucket name by vendor: it cannot occur in a vendor or
# model id, so two vendors that happen to share a scope or model name never collide on one bucket.
_QUALIFIER = "\x1f"


def pacer_name(vendor: str, name: str) -> str:
    """Qualify a bucket name by vendor so a shared scope never mixes across vendors.

    The core keys buckets by an opaque `(kind, name)`; the wiring qualifies `name` here rather than
    extending the core, and both the config-to-limits builder and the adapters call this one
    function so the two sides cannot drift on how a bucket is named.
    """
    if _QUALIFIER in vendor or _QUALIFIER in name:
        raise ValueError("vendor and name must not contain the bucket-qualifier control character")
    return f"{vendor}{_QUALIFIER}{name}"


# The window every published TPM/RPM limit is expressed over. A rolling 60-second window rather than
# a fixed calendar minute, because the account's limiter is itself rolling: two bursts 40 seconds
# apart share a window even though they might straddle a wall-clock minute boundary.
_WINDOW_SECONDS = 60.0

# The exponential-backoff base, replacing the fixed 1,2,4,8 the adapters used. The cap bounds the
# worst-case wait so a run does not stall for minutes on a single unlucky sequence.
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 32.0
# The exponent at which the backoff already reaches its cap (base * 2**e >= cap). Clamping the
# exponent to this before raising 2 to it is what stops a valid-but-huge attempt index from
# computing an astronomically large integer for CPU/memory before min() would discard it anyway.
_MAX_BACKOFF_EXPONENT = max(0, math.ceil(math.log2(_BACKOFF_CAP_SECONDS / _BACKOFF_BASE_SECONDS)))


def _require_count(value: object, name: str, *, minimum: int) -> int:
    """Reject anything but a real non-boolean int at or above `minimum`.

    Takes `object`, not `int`, on purpose: the guard exists precisely for callers that violate the
    declared type at runtime, and typing the parameter `int` would make the check statically
    redundant. A pacer that admits a malformed count can be evaded — `float("nan")` passes every
    ``<`` and ``>`` check silently (all comparisons with NaN are False), so a NaN reservation clears
    capacity and admits unbounded traffic, and a negative observed usage would free a live hold. The
    boolean exclusion matters because `bool` is an `int` subclass and `True`/`False` are not counts.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_finite(value: object, name: str) -> float:
    """Reject a non-finite duration: a NaN or infinite cooldown/reset would never clear."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value}")
    return float(value)


class Clock(Protocol):
    """The two time operations the pacer needs, injected so tests are deterministic.

    `monotonic` drives the rolling window and must never go backwards; `sleep` is how the pacer
    waits for capacity. A fake implementation of both is what lets the wait loop be tested without
    real time passing.
    """

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The production clock: a monotonic source for the window and asyncio.sleep for waiting."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(frozen=True)
class RateLimit:
    """A declared quota for one bucket: tokens and requests per rolling minute.

    Declared, never inferred from a tier — the account's real limits are configuration, because
    guessing them is how a run either throttles itself needlessly or not at all. Either bound may be
    None, meaning that dimension is unconstrained for this bucket.
    """

    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None

    def __post_init__(self) -> None:
        # A negative, zero, or non-integer limit is a misconfiguration, not a valid "very tight"
        # quota: zero rejects every call, negative silently widens capacity, and a NaN would defeat
        # every comparison. Rejected at construction so it surfaces where the limit is declared, not
        # on the first paid call.
        if self.tokens_per_minute is not None:
            _require_count(self.tokens_per_minute, "tokens_per_minute", minimum=1)
        if self.requests_per_minute is not None:
            _require_count(self.requests_per_minute, "requests_per_minute", minimum=1)


@dataclass
class _Entry:
    """One reserved request in a bucket's window: when it was admitted and what it holds.

    `tokens` starts at the reservation estimate and is replaced by the observed usage on settle, so
    a max-output hold does not keep charging the window after a short answer comes back.
    """

    entry_id: int
    admitted_at: float
    tokens: int


def _empty_entries() -> dict[int, _Entry]:
    """Typed default for a bucket's entry map, so pyright infers dict[int, _Entry], not dict[?]."""
    return {}


@dataclass
class _Bucket:
    """One quota group's rolling-window ledger, plus any active cooldown.

    A bucket is the unit a limit applies to: a per-(vendor, model) bucket, and above it a
    per-(vendor, scope) bucket for an org/project-shared limit. A call fits every bucket on it.
    """

    limit: RateLimit
    entries: dict[int, _Entry] = field(default_factory=_empty_entries)
    cooldown_until: float = 0.0
    # A server-reported shortfall: the account has less headroom than our ledger predicts (another
    # process is spending on the same key). Held as a phantom reservation that expires at the reset
    # the server named, so it tightens admission without ever loosening it.
    reconcile_deficit: int = 0
    reconcile_until: float = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - _WINDOW_SECONDS
        expired = [
            entry_id for entry_id, entry in self.entries.items() if entry.admitted_at <= cutoff
        ]
        for entry_id in expired:
            del self.entries[entry_id]
        if now >= self.reconcile_until:
            self.reconcile_deficit = 0

    def _held_tokens(self, now: float) -> int:
        """Tokens held by live reservations alone, without any reconcile deficit.

        Separate from `tokens_in_window` because `reconcile` must compute a shortfall against local
        consumption only: measuring against a window that already includes the previous deficit is
        exactly the bug that let a second, stricter header shrink the deficit it should have grown.
        """
        self._prune(now)
        return sum(entry.tokens for entry in self.entries.values())

    def tokens_in_window(self, now: float) -> int:
        held = self._held_tokens(now)
        return held + (self.reconcile_deficit if now < self.reconcile_until else 0)

    def fits_capacity(self, tokens: int) -> bool:
        """Whether a call of this size could ever fit in an empty window.

        Separate from admission because a call larger than the whole bucket must fail fast, not wait
        forever: no amount of expiry frees room for a reservation that exceeds the capacity itself.
        """
        tpm = self.limit.tokens_per_minute
        rpm = self.limit.requests_per_minute
        if tpm is not None and tokens > tpm:
            return False
        if rpm is not None and rpm < 1:
            return False
        return True

    def try_admit(self, now: float, tokens: int) -> float | None:
        """Return None if a call of `tokens` fits now, else the seconds to wait before retrying.

        Pure and synchronous: given the window state at `now`, it decides admit-or-wait with no side
        effects, so every branch is assertable with plain numbers. The wait it returns is the
        shortest time after which the binding constraint could clear — a cooldown's end, or the
        expiry of the oldest entries that together free enough room.
        """
        self._prune(now)
        if now < self.cooldown_until:
            return self.cooldown_until - now

        waits: list[float] = []
        tpm = self.limit.tokens_per_minute
        if tpm is not None:
            used = self.tokens_in_window(now)
            if used + tokens > tpm:
                waits.append(self._wait_for_token_room(now, freed_needed=used + tokens - tpm))

        rpm = self.limit.requests_per_minute
        if rpm is not None and len(self.entries) + 1 > rpm:
            waits.append(self._wait_for_one_request_slot(now))

        if not waits:
            return None
        # The binding constraint is the one that takes longest to clear; waiting less would fail the
        # same check again. Never return 0 as a "wait" — that would spin.
        return max(max(waits), _MIN_WAIT_SECONDS)

    def _wait_for_token_room(self, now: float, *, freed_needed: int) -> float:
        """Seconds until enough of the oldest entries expire to free `freed_needed` tokens."""
        ordered = sorted(self.entries.values(), key=lambda entry: entry.admitted_at)
        freed = 0
        for entry in ordered:
            freed += entry.tokens
            if freed >= freed_needed:
                return max(0.0, entry.admitted_at + _WINDOW_SECONDS - now)
        # The live entries alone cannot free it, but a reconcile deficit is also occupying room and
        # expires at its reset; wait for that. If neither suffices the call cannot fit and
        # fits_capacity would already have rejected it, so this is a lower bound.
        if now < self.reconcile_until:
            return max(0.0, self.reconcile_until - now)
        return _MIN_WAIT_SECONDS

    def _wait_for_one_request_slot(self, now: float) -> float:
        ordered = sorted(self.entries.values(), key=lambda entry: entry.admitted_at)
        oldest = ordered[0]
        return max(0.0, oldest.admitted_at + _WINDOW_SECONDS - now)

    def add(self, now: float, entry_id: int, tokens: int) -> None:
        self.entries[entry_id] = _Entry(entry_id=entry_id, admitted_at=now, tokens=tokens)

    def settle(self, entry_id: int, tokens: int) -> None:
        # `tokens` is validated non-negative by the public Reservation.settle before it arrives, so
        # this assigns rather than clamps: a clamp would hide a bad value instead of rejecting it.
        entry = self.entries.get(entry_id)
        if entry is not None:
            entry.tokens = tokens

    def release(self, entry_id: int) -> None:
        self.entries.pop(entry_id, None)

    def cool_down(self, now: float, retry_after: float) -> None:
        self.cooldown_until = max(self.cooldown_until, now + max(0.0, retry_after))

    def reconcile(self, now: float, server_remaining_tokens: int, reset_at: float) -> None:
        """Tighten the window when the server reports less headroom than the ledger predicts.

        Never loosens: a bare `remaining` is not authority to admit more, only to admit less. If the
        server says fewer tokens remain than our own accounting implies, the difference is spend we
        cannot see, held as a deficit until the server's own reset.
        """
        tpm = self.limit.tokens_per_minute
        if tpm is None:
            return
        # Against local consumption only. Using tokens_in_window here would fold the prior deficit
        # into the baseline, so a later stricter header would compute a smaller shortfall and could
        # loosen the very limit it was reporting.
        predicted_remaining = tpm - self._held_tokens(now)
        shortfall = predicted_remaining - server_remaining_tokens
        if shortfall <= 0 or reset_at <= now:
            return
        # Keep the largest shortfall seen before the reset. reconcile is best-effort and fail-safe:
        # it errs toward under-admission, and holding the max removes any dependence on the order in
        # which concurrent responses' headers happen to arrive.
        active_deficit = self.reconcile_deficit if now < self.reconcile_until else 0
        self.reconcile_deficit = max(active_deficit, shortfall)
        self.reconcile_until = max(self.reconcile_until, reset_at) if active_deficit else reset_at


# A wait must be strictly positive or the reserve loop spins; this floor is the smallest step it
# takes when a constraint is technically clearing "now" but the same check would still fail.
_MIN_WAIT_SECONDS = 0.001


class Reservation:
    """A held capacity grant for one call, and the only thing that may release it.

    The lifecycle is the safety-critical part (agreed in review): after a request is dispatched its
    outcome may be unknown — a timeout or dropped connection does not tell us whether the provider
    accepted and charged it. Releasing the hold then would let the pacer over-admit at exactly the
    moment it exists to prevent a 429. So `release` is valid ONLY before dispatch or when the
    provider confirms non-admission (a 429); after dispatch with an unknown outcome the caller keeps
    the hold and lets it expire with its window. The default is to keep: nothing is released unless
    a method is called that says it is safe.
    """

    def __init__(self, buckets: list[_Bucket], entry_id: int) -> None:
        self._buckets = buckets
        self._entry_id = entry_id
        self._closed = False

    def settle(self, observed_tokens: int) -> None:
        """Replace the estimate with the real usage the successful call reported.

        Rejects a negative or non-integer usage rather than clamping it: normalising `-1` to `0`
        would, through this public method, silently free a live hold and admit the next call — the
        exact over-admission the pacer exists to prevent.
        """
        _require_count(observed_tokens, "observed_tokens", minimum=0)
        if self._closed:
            return
        for bucket in self._buckets:
            bucket.settle(self._entry_id, observed_tokens)
        self._closed = True

    def release(self) -> None:
        """Free the hold. Only for pre-dispatch failure or provider-confirmed non-admission."""
        if self._closed:
            return
        for bucket in self._buckets:
            bucket.release(self._entry_id)
        self._closed = True

    def keep(self) -> None:
        """Explicitly keep the hold to expiry — the safe default after an ambiguous dispatch.

        A no-op against the ledger (the entry already stands), named so a caller can state the
        intent and so tests assert the post-dispatch path was taken, not a release forgotten.
        """
        self._closed = True


class RateLimiter:
    """One pacer shared by every paid model call in a run — the answer model and the judge alike.

    Shared deliberately: the answer model and the judge may bill the same account bucket, so each
    must see the other's holds and cooldowns or the pacer would admit twice the traffic the account
    allows. It is constructed once at the run boundary and injected into both, including the judge
    that `rejudge` builds outside the runner.
    """

    def __init__(
        self,
        limits: dict[tuple[str, str], RateLimit] | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        # Keyed by (kind, name): ("model", model_id) for a per-model bucket, ("scope", scope_id) for
        # the org/project-shared bucket that sits above it. Absent keys are unconstrained.
        self._limits = dict(limits or {})
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._clock = clock if clock is not None else SystemClock()
        self._next_entry_id = 0
        self._lock = asyncio.Lock()

    def _bucket(self, key: tuple[str, str]) -> _Bucket | None:
        limit = self._limits.get(key)
        if limit is None:
            return None
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(limit=limit)
            self._buckets[key] = bucket
        return bucket

    def _path(self, model_id: str, scope_id: str | None) -> list[_Bucket]:
        keys: list[tuple[str, str]] = [("model", model_id)]
        if scope_id is not None:
            keys.append(("scope", scope_id))
        return [bucket for key in keys if (bucket := self._bucket(key)) is not None]

    async def reserve(
        self, *, model_id: str, scope_id: str | None, estimated_tokens: int
    ) -> Reservation:
        """Block until a call of `estimated_tokens` fits every bucket on its path, then hold it.

        Fails fast rather than waiting forever when the reservation is larger than a bucket's whole
        capacity: that is a configuration error (max_output plus input exceeds the declared TPM),
        and hanging the run on it would hide the misconfiguration behind an apparent stall.
        """
        _require_count(estimated_tokens, "estimated_tokens", minimum=0)

        buckets = self._path(model_id, scope_id)
        if not buckets:
            # Nothing is configured for this call, so the pacer imposes no constraint. Returning a
            # reservation with an empty path keeps the caller's settle/release calls harmless.
            return Reservation([], entry_id=self._reserve_id())

        for bucket in buckets:
            if not bucket.fits_capacity(estimated_tokens):
                raise ImpossibleReservationError(
                    f"a single call reserving {estimated_tokens} tokens can never fit "
                    f"a bucket limited to {bucket.limit.tokens_per_minute} tokens per minute"
                )

        while True:
            async with self._lock:
                now = self._clock.monotonic()
                wait = max((bucket.try_admit(now, estimated_tokens) or 0.0) for bucket in buckets)
                if wait <= 0.0:
                    entry_id = self._reserve_id()
                    for bucket in buckets:
                        bucket.add(now, entry_id, estimated_tokens)
                    return Reservation(buckets, entry_id)
            # Slept outside the lock so other callers can settle and free room while we wait.
            await self._clock.sleep(wait)

    def note_rate_limited(self, *, model_id: str, scope_id: str | None, retry_after: float) -> None:
        """Apply a provider's Retry-After as a cooldown on the whole path, honoured as a minimum.

        On the bucket, not just the failing call, so concurrent callers do not stampede the reset
        the server asked us to wait for.
        """
        _require_finite(retry_after, "retry_after")
        now = self._clock.monotonic()
        for bucket in self._path(model_id, scope_id):
            bucket.cool_down(now, retry_after)

    def note_observed_remaining(
        self,
        *,
        model_id: str,
        scope_id: str | None,
        remaining_tokens: int,
        reset_at_monotonic: float,
    ) -> None:
        """Best-effort reconciliation from rate-limit headers; tightens the ledger, never loosens.

        Off entirely when an adapter cannot obtain headers — correctness rests on TPM/RPM admission
        and settle-to-real-usage, and this only helps when another process is spending on the same
        key. It never fabricates headroom.
        """
        _require_count(remaining_tokens, "remaining_tokens", minimum=0)
        _require_finite(reset_at_monotonic, "reset_at_monotonic")
        now = self._clock.monotonic()
        for bucket in self._path(model_id, scope_id):
            bucket.reconcile(now, remaining_tokens, reset_at_monotonic)

    def _reserve_id(self) -> int:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        return entry_id


class ImpossibleReservationError(RuntimeError):
    """A single call reserves more than a bucket can ever hold — a limit misconfiguration."""


def full_jitter_backoff(
    attempt_index: int,
    *,
    retry_after: float = 0.0,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Exponential backoff with true full jitter, floored at any server Retry-After.

    Full jitter (AWS, "Exponential Backoff and Jitter"): sleep uniformly in [0, cap], not a bounded
    multiple of the cap — the earlier `uniform(0.5,1.0)*cap` form was bounded multiplicative jitter,
    which keeps every client waiting most of the cap and defeats the decorrelation jitter is for.
    The Retry-After floor is honoured because the server's stated wait is a minimum, not a hint.

    Both inputs are validated: a NaN `retry_after` would make `max(nan, x)` return `nan` and destroy
    the floor this function exists to guarantee, turning an adapter's retry into an invalid or
    immediate wait. `attempt_index` must be a non-boolean int >= 0 so the ceiling is well defined.
    """
    _require_count(attempt_index, "attempt_index", minimum=0)
    _require_finite(retry_after, "retry_after")
    # Clamp the exponent before raising, not the result after: 2**attempt_index for a huge index
    # builds an enormous integer first, and min() discarding it afterwards is too late.
    exponent = min(attempt_index, _MAX_BACKOFF_EXPONENT)
    ceiling = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**exponent))
    return max(retry_after, rng(0.0, ceiling))
