from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Final, Protocol

from khedron.cost.pricing import PricingTable, compute_cost_usd
from khedron.errors import ConfigurationError, CostExceededError

__all__ = [
    "BUDGET_ENV_VAR",
    "BudgetLedger",
    "BudgetReservationError",
    "BudgetReserver",
    "CostReservation",
    "resolve_budget_cap_usd",
    "upper_bound_cost_usd",
]

BUDGET_ENV_VAR: Final = "KHEDRON_MAX_BUDGET_USD"

# The message framing a chat request wraps the prompt in (role markers, envelope). Mirrors
# pacing._WRAP_OVERHEAD_TOKENS: a fixed, generous margin added to the byte upper bound so the
# reservation is never below the wire size by this known, bounded amount.
_WRAP_OVERHEAD_TOKENS: Final = 32


class BudgetReservationError(CostExceededError):
    """Raised BEFORE dispatch when reserving an attempt's worst-case cost would breach the cap.

    A subclass of ``CostExceededError`` so the run/suite halt path (which already re-raises
    ``CostExceededError`` rather than treating it as a run failure) stops cleanly with the cap and
    the two named figures. The defining property is the timing: it is raised before the paid call is
    ever sent, so a refused call spends nothing -- the pre-dispatch refusal, not the post-hoc
    boundary check, is what bounds worst-case spend.
    """


def upper_bound_cost_usd(
    *, prompt: str, max_output_tokens: int, model_id: str, pricing: PricingTable
) -> float:
    """A hard upper bound on one physical attempt's cost, computed before dispatch.

    A monetary reservation must be a TRUE ceiling: a figure that could under-reserve is not a bound.
    So the input side is bounded by the UTF-8 byte length of the prompt plus a fixed framing margin
    -- a token is never fewer than one byte for a byte-fallback BPE tokeniser, so this never
    under-counts for any vendor. It deliberately over-reserves openai relative to its tiktoken token
    count (bytes >= tokens); over-reserving refuses a call that would have fit (recoverable: raise
    the cap and rerun), whereas under-reserving breaches the cap with real money, which is not.
    The output side is pinned to the caller's ``max_output_tokens``
    ceiling, the most the answer can consume.

    Raises ``ValueError`` (propagated from :func:`compute_cost_usd`) when the model has no usable
    price. The caller turns that into a fail-closed refusal before dispatch: an unpriceable model
    cannot be bounded, so it is refused rather than billed then discovered.
    """
    input_upper_bound = len(prompt.encode("utf-8")) + _WRAP_OVERHEAD_TOKENS
    return compute_cost_usd(model_id, input_upper_bound, max(0, max_output_tokens), pricing)


class CostReservation:
    """A held dollar grant for one physical attempt, and the only thing that may dispose of it.

    The dollar-denominated sibling of the pacer's token ``Reservation`` (``ratelimit.Reservation``),
    with the identical safety-critical lifecycle: after a request is dispatched its outcome may be
    unknown -- a timeout or dropped connection does not tell us whether the provider accepted and
    charged it -- so ``release`` is valid ONLY before dispatch or on a provider-confirmed
    non-admission (a 429). After any ambiguous/terminal post-dispatch outcome the caller keeps the
    hold, because the request may have been billed. The default is to keep: nothing frees unless a
    method is called that proves it is safe.

    A settled reservation contributes its *observed* cost to the ledger's committed total; a
    kept-but-unsettled reservation contributes its *reserved maximum* to worst-case spend.
    """

    def __init__(
        self,
        *,
        reserved_max_usd: float,
        on_settle: Callable[[float], None],
        on_release: Callable[[], None],
    ) -> None:
        # Disposition is driven through callbacks the ledger binds to this hold's entry, so the
        # ledger's book-keeping methods stay private to it -- the reservation cannot reach into any
        # other hold, only settle or release its own.
        self._reserved_max_usd = reserved_max_usd
        self._on_settle = on_settle
        self._on_release = on_release
        self._closed = False

    @property
    def reserved_max_usd(self) -> float:
        """The worst-case dollar figure this attempt is holding against the cap."""
        return self._reserved_max_usd

    def settle(self, observed_cost_usd: float) -> None:
        """Replace the reserved maximum with the successful call's real, billed cost.

        Rejects a negative or non-finite cost rather than clamping it: normalising it would, through
        this public method, silently understate committed spend and under-enforce the cap.
        """
        _require_amount(observed_cost_usd, "observed_cost_usd")
        if self._closed:
            return
        self._on_settle(observed_cost_usd)
        self._closed = True

    def release(self) -> None:
        """Free the hold. Only for a pre-dispatch failure or a provider-confirmed non-admission."""
        if self._closed:
            return
        self._on_release()
        self._closed = True

    def keep(self) -> None:
        """Explicitly keep the reserved maximum against worst-case spend -- the safe default.

        Named so a caller can state the intent and so tests can assert the post-dispatch path was
        taken, not a release forgotten. The entry already stands in the ledger; this only closes the
        reservation so a later settle/release cannot move it.
        """
        self._closed = True


class BudgetReserver(Protocol):
    """The narrow money authority the pacer depends on, so ``paced_generate`` never imports pricing.

    Injected exactly as the rate limiter is: the pacer calls ``reserve`` per
    physical attempt and disposes the returned reservation on the same branches as the token
    hold. Kept a Protocol so the pacer stays cost-agnostic and tests can pass a fake.
    """

    def reserve(self, max_cost_usd: float) -> CostReservation:
        """Reserve an attempt's worst-case cost, or raise ``BudgetReservationError`` if over cap."""
        ...


class BudgetLedger:
    """One suite-shared dollar ledger that bounds worst-case spend by refusing before dispatch.

    The monetary analogue of the token ``RateLimiter``, and injected the same way (one instance per
    run/suite, handed to the answer model and every judge's wrapped model). It holds two tallies:

    - ``_committed_usd`` -- the confirmed cost of settled attempts, seeded with any prior spend so a
      resumed run's cap accounts for what earlier attempts already spent. This is the ledger's own
      current view of observed spend, used for the pre-dispatch check because the run's
      ``CostTracker`` is stale at ``settle`` time (the pipeline records an observed call *after*
      ``paced_generate`` returns);
    - ``_outstanding`` -- the reserved maxima of every live-or-kept reservation. A kept
      (ambiguous billed-but-failed) hold stays here until the run ends, so its possible spend counts
      against both the cap and ``worst_case_cost_usd``; a released (not-billed) hold leaves.

    ``worst_case_cost_usd`` is ``_committed_usd + sum(_outstanding)``. The cap is enforced against
    that pessimistic figure: ``reserve`` refuses when ``committed + outstanding + this_attempt_max``
    would exceed the cap, so no further paid call is dispatched once worst-case would breach it.
    One ledger is shared across every run of a suite, so kept holds and committed spend accumulate
    cumulatively -- exactly matching the cap the runner enforces (``suite_spend_usd`` across runs).

    The cap is read through ``cap_provider`` on every reservation, not snapshotted, so a caller that
    re-resolves the cap mid-invocation (``recover-blocked``'s ``--max-cost-usd``) is
    honoured without rebuilding the ledger. ``None`` from the provider means no cap: holds are
    still taken (so ``worst_case_cost_usd`` stays honest) but can never refuse.

    Concurrency: ``reserve`` and the ``_settle``/``_release``/``_keep`` callbacks are sync and
    never await, so under the single-threaded asyncio event loop each runs atomically -- the loop
    cannot interleave two of them. No lock is needed (and none is taken), mirroring why the token
    ``Reservation``'s settle/release/keep are lock-free; the token *limiter* locks only because
    its ``reserve`` awaits while waiting for capacity, which the money reservation never does (a cap
    does not clear over time -- a call either fits or is refused now).
    """

    def __init__(
        self,
        *,
        cap_provider: Callable[[], float | None],
        prior_spend_usd: float = 0.0,
    ) -> None:
        _require_amount(prior_spend_usd, "prior_spend_usd")
        self._cap_provider = cap_provider
        self._committed_usd = prior_spend_usd
        self._outstanding: dict[int, float] = {}
        self._next_entry_id = 0

    def reserve(self, max_cost_usd: float) -> CostReservation:
        """Hold ``max_cost_usd`` against the cap, or refuse before dispatch if it would breach it.

        The refusal is the bound: raised before the caller dispatches, so a
        refused attempt is never billed. When no cap is configured the hold is still taken (so
        ``worst_case_cost_usd`` stays honest) but can never refuse.
        """
        _require_amount(max_cost_usd, "max_cost_usd")
        cap_usd = self._cap_provider()
        prospective_worst_case = self._committed_usd + self._outstanding_total() + max_cost_usd
        if cap_usd is not None and prospective_worst_case > cap_usd:
            raise BudgetReservationError(
                f"Reserving this call's worst-case cost would raise worst-case spend to "
                f"${prospective_worst_case:.6f}, over the configured budget cap "
                f"${cap_usd:.6f}; refusing before dispatch so it is never billed.",
                budget_cap_usd=cap_usd,
                observed_cost_usd=self._committed_usd,
                worst_case_cost_usd=prospective_worst_case,
                attempted_cost_usd=max_cost_usd,
            )
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        self._outstanding[entry_id] = max_cost_usd
        # keep() needs no callback: a kept hold simply stays in _outstanding until the run ends.
        return CostReservation(
            reserved_max_usd=max_cost_usd,
            on_settle=lambda observed: self._settle(entry_id, observed),
            on_release=lambda: self._release(entry_id),
        )

    def _settle(self, entry_id: int, observed_cost_usd: float) -> None:
        """Move a reservation from held maximum to confirmed committed spend (its real cost)."""
        if self._outstanding.pop(entry_id, None) is not None:
            self._committed_usd += observed_cost_usd

    def _release(self, entry_id: int) -> None:
        """Drop a provably-not-billed reservation; it contributes nothing to worst-case spend."""
        self._outstanding.pop(entry_id, None)

    def seed_prior_spend(self, prior_spend_usd: float) -> None:
        """Fold spend that predates this session into committed, so the cap accounts for it.

        Called once as each run's cost tracker is built, with that run's inherited (already-billed)
        api_calls total: a resumed or force-recovered run must not get a fresh cap on top of what it
        already spent. A fresh run inherits nothing, so this is a no-op for it. Kept separate from
        construction so the one shared ledger is seeded uniformly for every run -- normal, resumed,
        and force-recovered -- from the same authoritative figure (the tracker's initial total).
        """
        _require_amount(prior_spend_usd, "prior_spend_usd")
        self._committed_usd += prior_spend_usd

    def _outstanding_total(self) -> float:
        return sum(self._outstanding.values())

    def outstanding_cost_usd(self) -> float:
        """The reserved maxima of every ambiguous billed-but-failed hold not yet disposed."""
        return self._outstanding_total()

    def observed_cost_usd(self) -> float:
        """The ledger's current view of confirmed spend (settled attempts plus prior spend)."""
        return self._committed_usd

    def worst_case_cost_usd(self) -> float:
        """Confirmed spend plus every outstanding reserved max -- the figure the cap enforces."""
        return self._committed_usd + self._outstanding_total()


def _require_amount(value: float, name: str) -> None:
    """Reject a negative or non-finite dollar amount rather than clamping it.

    Clamping a bad figure to 0 through a public method would silently understate spend and
    under-enforce the cap -- the exact fail-open the ledger exists to prevent. A non-number reaches
    ``math.isfinite`` and raises there, so a wrong type still fails loudly rather than corrupting
    the ledger.
    """
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number of US dollars.")


def resolve_budget_cap_usd(config_cap_usd: float | None) -> float | None:
    """Resolve the effective budget cap, letting the environment override the config.

    Why: operators need a hard spend ceiling they can impose per invocation
    (``KHEDRON_MAX_BUDGET_USD``) without editing committed suite configuration; when the
    variable is unset, the suite's ``max_cost_usd`` applies. Invalid values fail closed with
    ``ConfigurationError`` instead of silently running uncapped, because an unnoticed typo
    must never disable a budget guard.

    Shared rather than private to the runner because preflight has to report the cap that
    will actually apply. Reading ``config.max_cost_usd`` directly made it claim spend was
    unbounded when the environment had set a cap, and claim the config's cap applied when
    the environment had lowered it.
    """
    raw_value = os.environ.get(BUDGET_ENV_VAR)
    if raw_value is None:
        return config_cap_usd
    try:
        cap_usd = float(raw_value.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid {BUDGET_ENV_VAR} value; expected a non-negative number of US dollars.",
            observed_value=raw_value,
        ) from exc
    if not math.isfinite(cap_usd) or cap_usd < 0.0:
        raise ConfigurationError(
            f"Invalid {BUDGET_ENV_VAR} value; expected a finite, non-negative number "
            "of US dollars.",
            observed_value=raw_value,
        )
    return cap_usd
