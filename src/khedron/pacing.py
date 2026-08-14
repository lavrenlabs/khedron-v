"""Config-to-limiter wiring and estimation for the request pacer.

The pure decision core lives in `khedron.ratelimit`. This module is the boundary between it and the
rest of Khedron: it declares the `rate_limits` configuration shape, builds a `RateLimiter` from it
with vendor-qualified bucket names, estimates the tokens a call should reserve, and reads a server's
Retry-After. Kept separate from the adapters so this logic is pure and offline-testable, and from
the core so the core stays free of config and provider concerns.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import tiktoken
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from khedron.errors import ConfigurationError, KhedronError
from khedron.ratelimit import (
    RateLimit,
    RateLimiter,
    Reservation,
    full_jitter_backoff,
    pacer_name,
)
from khedron.utils.time import now_utc

if TYPE_CHECKING:
    # Annotation-only: importing these at runtime would close a cycle (types imports config, config
    # imports this module for its rate_limits field, and budget imports cost.pricing which pulls in
    # types). paced_generate only annotates with BudgetReserver/CostReservation and reads attributes
    # off an APICallResult, never constructing one, so the runtime import is unnecessary.
    from khedron.budget import BudgetReserver, CostReservation
    from khedron.types import APICallResult

__all__ = [
    "TIMEOUT_MAX_RETRIES",
    "DispatchOutcome",
    "RateLimitEntry",
    "RateLimitsConfig",
    "RetryDisposition",
    "build_rate_limiter",
    "estimate_reservation_tokens",
    "load_rate_limits_config",
    "paced_generate",
    "retry_after_seconds",
]

# Framing tokens a chat request adds around the prompt (role markers, message envelope). A generous
# fixed margin so the reservation is never below the wire size by this known, bounded amount.
_WRAP_OVERHEAD_TOKENS = 32

# The exact fixed exponential backoff the adapters used before the pacer (1, 2, 4, ...). Kept for
# the no-limiter path so an unpaced run's retry timing is unchanged; the exponent is clamped only to
# avoid a pathological power for a huge attempt index, which does not affect realistic retry counts.
_UNPACED_BACKOFF_BASE_SECONDS = 1.0
_UNPACED_BACKOFF_MAX_EXPONENT = 16


def _unpaced_backoff_seconds(attempt_index: int) -> float:
    return _UNPACED_BACKOFF_BASE_SECONDS * (2 ** min(attempt_index, _UNPACED_BACKOFF_MAX_EXPONENT))


class RateLimitEntry(BaseModel):
    """One declared quota: a per-model limit, or a per-vendor scope limit when `model` is unset.

    Declared, not inferred from a tier. A negative or zero limit is rejected downstream by
    `RateLimit`; here the vendor and model are just carried through to the bucket name.
    """

    # extra="forbid" so the strict loader keeps its promise: an unknown or misspelled key is an
    # error, not a silently dropped field that leaves a limit the operator meant to set unset.
    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    tokens_per_minute: int | None = Field(default=None)
    requests_per_minute: int | None = Field(default=None)


class RateLimitsConfig(BaseModel):
    """The `rate_limits` block: an optional shared-scope id and the per-bucket limits.

    Absent or empty means the pacer is off — the builder returns None and adapters skip it entirely,
    so a run that configures nothing behaves exactly as before the pacer existed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str | None = Field(default=None, min_length=1)
    limits: tuple[RateLimitEntry, ...] = ()


def load_rate_limits_config(path: Path) -> RateLimitsConfig:
    """Load a standalone `rate_limits` YAML block for a path (e.g. rejudge) with no suite config.

    Deliberately a dedicated, strict loader rather than reusing the full suite loader: rejudge names
    a source run, not a suite, so requiring a whole suite YAML just to pace it would be a trap. The
    file is exactly the `rate_limits` mapping (`scope`, `limits`), and anything else is rejected.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Could not read the rate-limits file.", path=str(path)) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            "A rate-limits file must be a mapping of scope and limits.", path=str(path)
        )
    try:
        return RateLimitsConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            "Invalid rate-limits file.", path=str(path), error=str(exc)
        ) from exc


def build_rate_limiter(config: RateLimitsConfig | None) -> RateLimiter | None:
    """Build one shared limiter from config, or None when nothing is configured.

    Returns None rather than an empty limiter so the no-config path is "never enter the pacer",
    not "enter a pacer that admits everything" — identical behaviour to today, guaranteed by the
    adapters not calling reserve at all.

    Bucket names are vendor-qualified through `pacer_name`, so the same scope id under two vendors
    resolves to two distinct buckets and their traffic is never conflated. A scope-level entry (no
    model) with no configured `scope` is a contradiction and is rejected here, not silently dropped.
    """
    if config is None or not config.limits:
        return None

    limits: dict[tuple[str, str], RateLimit] = {}
    for entry in config.limits:
        if entry.tokens_per_minute is None and entry.requests_per_minute is None:
            raise ConfigurationError(
                "A rate-limit entry declares neither a token nor a request limit, so it constrains "
                "nothing; remove it or give it a limit.",
                vendor=entry.vendor,
                model=entry.model,
            )
        limit = RateLimit(
            tokens_per_minute=entry.tokens_per_minute,
            requests_per_minute=entry.requests_per_minute,
        )
        try:
            if entry.model is not None:
                key = ("model", pacer_name(entry.vendor, entry.model))
            elif config.scope is None:
                raise ConfigurationError(
                    "A rate-limit entry without a model is a shared-scope limit, but no scope is "
                    "configured to attach it to.",
                    vendor=entry.vendor,
                )
            else:
                key = ("scope", pacer_name(entry.vendor, config.scope))
        except ValueError as exc:
            # pacer_name rejects a control character in a name; surface it as a configuration error
            # rather than a bare ValueError, since it originates in the config the operator wrote.
            raise ConfigurationError(
                "A rate-limit name contains an invalid character.",
                vendor=entry.vendor,
                model=entry.model,
            ) from exc
        if key in limits:
            raise ConfigurationError(
                "Two rate-limit entries resolve to the same bucket.",
                vendor=entry.vendor,
                model=entry.model,
            )
        limits[key] = limit
    return RateLimiter(limits)


def estimate_reservation_tokens(
    *, vendor: str, model_id: str, prompt: str, max_output_tokens: int
) -> int:
    """A local estimate of the tokens a call should hold, corrected to real usage on settle.

    No network: the Anthropic and Google token counters are remote calls, and adding an unbounded
    network round-trip before every rate-limited call would defeat the purpose. So:

    - **anthropic / google**: the UTF-8 byte length of the prompt. A token is never fewer than one
      byte for a byte-fallback BPE tokeniser, so this is a genuine upper bound and the pacer cannot
      over-admit these — deliberately conservative rather than a chars/4 guess that CJK or emoji
      would blow past. Their prompts are small (a judge grades one answer), so over-reserving is
      cheap.
    - **openai**: `tiktoken` with the model's own encoding. This is an *estimate*, not a hard upper
      bound — it does not count the Responses envelope, and a future model could tokenise unusually
      — but the OpenAI arm's prompts are ~37k tokens, and reserving their byte length instead would
      throttle the run roughly four-fold. `settle` corrects it to real usage within the window, and
      any 429 that slips through is absorbed by the Retry-After cooldown and backoff. The line is
      drawn explicitly rather than hidden: bytes guarantee, tiktoken estimates.

    Every path adds `max_output_tokens` (the capacity the answer may consume) and a fixed wrapping
    overhead for the message framing.

    The byte figure bounds the raw prompt text plus that fixed framing margin — nothing more. If a
    call later carries tool schemas, images, or variable framing, the bound no longer covers the
    whole request and this must be revisited where those are added.
    """
    if vendor == "openai":
        body = len(_openai_encoding(model_id).encode(prompt))
    else:
        body = len(prompt.encode("utf-8"))
    return body + max(0, max_output_tokens) + _WRAP_OVERHEAD_TOKENS


def _openai_encoding(model_id: str) -> tiktoken.Encoding:
    """The model's own tiktoken encoding, falling back to o200k_base for an unknown model id.

    `encoding_for_model` maps gpt-4o* to o200k_base and gpt-4* to cl100k_base; a bare or future id
    raises KeyError, and o200k_base is the current default rather than the older cl100k_base the
    generic token estimator uses.
    """
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def retry_after_seconds(exc: BaseException, *, now: datetime | None = None) -> float:
    """Read a provider's Retry-After from an exception, in either valid HTTP form, else 0.0.

    HTTP allows Retry-After as delta-seconds (`Retry-After: 3`) or an HTTP date
    (`Retry-After: Wed, 21 Oct 2026 07:28:00 GMT`). Both are honoured; anything absent, malformed,
    or in the past yields 0.0, so a missing or garbled header never produces a negative or
    non-finite wait that would defeat the backoff floor it feeds.
    """
    raw = _retry_after_header(exc)
    if raw is None:
        return 0.0

    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if math.isfinite(seconds) and seconds >= 0 else 0.0

    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return 0.0
    reference = now if now is not None else now_utc()
    if when.tzinfo is None:
        # An HTTP date is always GMT; parsedate_to_datetime may return it naive.
        when = when.replace(tzinfo=reference.tzinfo)
    delta = (when - reference).total_seconds()
    return max(0.0, delta) if math.isfinite(delta) else 0.0


def _retry_after_header(exc: BaseException) -> str | None:
    """Pull the raw `retry-after` header off an SDK exception, tolerant of its shape.

    The OpenAI and Anthropic SDKs hang the HTTP response off the exception as `.response`, with a
    mapping-like `.headers`. Read defensively: a header that is not there, or an exception that does
    not carry a response at all, is not an error — it just means no server-stated wait. Both header
    spellings are tried because a case-insensitive SDK headers object accepts either while a plain
    dict (a test double, or an atypical exception) is case-sensitive. The whole extraction is
    wrapped so a property or `.get` that itself raises yields "no wait" rather than propagating —
    `Exception`, not `BaseException`, so a cancellation still passes through untouched.
    """
    try:
        response = getattr(exc, "response", None)
        headers: Any = getattr(response, "headers", None)
        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        for spelling in ("retry-after", "Retry-After"):
            value = getter(spelling)
            if isinstance(value, str):
                return value
        return None
    except Exception:
        return None


class RetryDisposition(Enum):
    """How a failed dispatch should be treated for both the reservation and the retry.

    The three cases the pacer must tell apart, because they dispose of the held reservation
    differently: a confirmed 429 means the provider did not admit the request (release the hold), an
    ambiguous transient failure may have been admitted and charged (keep the hold), and a terminal
    failure stops the retry loop (keep the hold; a request may still have been sent).
    """

    RATE_LIMITED = "rate_limited"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"


# A timeout is transient, so it is retryable -- but a timed-out request is the disposition most
# likely to have completed and been billed server-side, so every retry risks a duplicate charge.
# The timeout class therefore carries its own hard cap, independent of the suite's `max_retries`
# (canonical configs set 8): worst-case duplicate billing for a timed-out logical call is bounded
# to TIMEOUT_MAX_RETRIES extra attempts, whatever the config sets. This cap is latched
# independently of the general retry budget, by design.
TIMEOUT_MAX_RETRIES: Final = 2


@dataclass(frozen=True)
class DispatchOutcome:
    """An adapter's classification of a dispatch exception, and the error to surface if not retried.

    `make_error` builds the public error to raise, given the number of attempts made, so the adapter
    keeps ownership of its error types and messages while the pacer owns the reserve/retry cycle.

    `retry_cap` lets a disposition bound its own retries below the call's `max_retries` -- the
    timeout class sets it to TIMEOUT_MAX_RETRIES to cap duplicate-billing exposure regardless of
    config. None means `max_retries` governs unchanged.
    """

    disposition: RetryDisposition
    make_error: Callable[[int], Exception]
    retry_after: float = 0.0
    retry_cap: int | None = None


async def paced_generate(
    *,
    limiter: RateLimiter | None,
    vendor: str,
    model_id: str,
    scope: str | None,
    estimate: Callable[[], int],
    max_retries: int,
    dispatch: Callable[[], Awaitable[Any]],
    extract: Callable[[Any], APICallResult],
    classify: Callable[[Exception], DispatchOutcome],
    backoff_sleep: Callable[[float], Awaitable[None]],
    budget: BudgetReserver | None = None,
    estimate_cost: Callable[[], float] | None = None,
) -> APICallResult:
    """Run one logical model call through the pacer: reserve, dispatch, dispose, retry.

    A reservation is taken **per physical attempt**, never reused across two dispatches: each retry
    is a distinct request that may consume the account's quota, so counting them separately is what
    keeps the window honest. The disposition rules, agreed in review, are:

    - a confirmed 429 releases the hold (the provider did not admit it) and records the cooldown;
    - any other post-dispatch outcome -- ambiguous transient failure, a terminal error, a
      cancellation, or a failure extracting the result from a response the provider already returned
      and charged -- keeps the hold until its window expires, because the request may have been
      billed;
    - only a fully successful call settles the hold to the real usage.

    `dispatch` performs the raw API call; `extract` turns its response into an APICallResult and may
    itself fail after a real, charged response; `classify` maps a dispatch exception to a
    DispatchOutcome. The pacer imposes nothing when `limiter` is None -- and crucially does no work
    for it either: `estimate` (which may load a tokeniser) and `pacer_name` are only touched when a
    limiter is present, so the unpaced path is byte-for-byte the behaviour before the pacer existed.
    """
    model_key: str | None = None
    scope_key: str | None = None
    estimated_tokens = 0
    if limiter is not None:
        model_key = pacer_name(vendor, model_id)
        scope_key = pacer_name(vendor, scope) if scope is not None else None
        estimated_tokens = estimate()

    # The retry ceiling starts at max_retries and only ratchets DOWN: a capped disposition (the
    # timeout class, via retry_cap=TIMEOUT_MAX_RETRIES) lowers it for the rest of this call and no
    # later failure may raise it. So once a request has timed out (possibly already billed), total
    # dispatches stay bounded whatever follows; a never-timing-out call uses the full max_retries.
    retry_ceiling = max_retries
    for attempt in range(max_retries + 1):
        # Reserve money BEFORE token capacity: a refusal or an unpriceable model must not first wait
        # out the rate limiter, and if the token reservation itself fails the money hold is freed
        # below. estimate_cost() raises for a model with no usable price (fail-closed, Decision 3);
        # reserve() raises BudgetReservationError when worst-case spend would breach the cap (the
        # bound, Decision 1.4). Both raise before dispatch, so nothing is billed. The
        # money path is gated on `budget`, independently of `limiter`, so a capped run with no rate
        # limits still refuses on cost and a paced run with no cap does no money work.
        cost_reservation: CostReservation | None = None
        if budget is not None:
            assert estimate_cost is not None  # noqa: S101  # paired with budget by the caller
            cost_reservation = budget.reserve(estimate_cost())
        reservation: Reservation | None = None
        if limiter is not None:
            # Set together with the limiter above; narrows the Optional for the type checker.
            assert model_key is not None  # noqa: S101
            try:
                reservation = await limiter.reserve(
                    model_id=model_key, scope_id=scope_key, estimated_tokens=estimated_tokens
                )
            except BaseException:
                # The token grant was never taken, so this is a pre-dispatch failure: release the
                # money hold rather than leave a phantom maximum standing against the cap.
                if cost_reservation is not None:
                    cost_reservation.release()
                raise
        try:
            response = await dispatch()
        except asyncio.CancelledError:
            # Not an Exception subclass in 3.11, so it must be caught before the general handler.
            # The request may be in flight; keep both holds and let the cancellation propagate.
            if reservation is not None:
                reservation.keep()
            if cost_reservation is not None:
                cost_reservation.keep()
            raise
        except Exception as exc:
            outcome = classify(exc)
            if outcome.disposition is RetryDisposition.RATE_LIMITED:
                if reservation is not None:
                    reservation.release()
                if cost_reservation is not None:
                    cost_reservation.release()
                if limiter is not None:
                    assert model_key is not None  # noqa: S101
                    limiter.note_rate_limited(
                        model_id=model_key, scope_id=scope_key, retry_after=outcome.retry_after
                    )
            elif reservation is not None or cost_reservation is not None:
                if reservation is not None:
                    reservation.keep()
                if cost_reservation is not None:
                    cost_reservation.keep()

            # Ratchet the ceiling down when this disposition caps itself (the timeout class does):
            # min() with the running value latches it, so a later uncapped failure cannot undo a
            # timeout's cap -- so a mixed sequence (not just all-timeouts) is bounded too.
            if outcome.retry_cap is not None:
                retry_ceiling = min(retry_ceiling, outcome.retry_cap)
            if outcome.disposition is not RetryDisposition.TERMINAL and attempt < retry_ceiling:
                # Full jitter only when the pacer is active -- it is part of the pacer's improvement
                # and must not change an unpaced run's timing. Without a limiter, keep the exact
                # fixed exponential backoff the adapters used before, so "behaviour untouched" is
                # literal for a run that configured no rate limits.
                if limiter is not None:
                    delay = full_jitter_backoff(attempt, retry_after=outcome.retry_after)
                else:
                    delay = _unpaced_backoff_seconds(attempt)
                await backoff_sleep(delay)
                continue
            surfaced_error = outcome.make_error(attempt + 1)
            if isinstance(surfaced_error, KhedronError):
                surfaced_error.context["retryable"] = (
                    outcome.disposition is not RetryDisposition.TERMINAL
                )
            raise surfaced_error from exc
        else:
            try:
                result = extract(response)
            except Exception:
                # A response came back -- so it was admitted and charged -- but turning it into a
                # result failed. Keep both holds; releasing would let the next call over-admit and
                # would drop a possibly-billed cost from worst-case spend.
                if reservation is not None:
                    reservation.keep()
                if cost_reservation is not None:
                    cost_reservation.keep()
                raise
            if reservation is not None:
                reservation.settle(result.input_tokens + result.output_tokens)
            if cost_reservation is not None:
                # The successful call's real billed cost replaces its reserved maximum -- the same
                # cost_usd the pipeline records as an APICallRecord just after this returns.
                cost_reservation.settle(result.cost_usd)
            return result

    raise RuntimeError("paced_generate exhausted its retry loop without returning or raising")
