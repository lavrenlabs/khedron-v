from __future__ import annotations

from dataclasses import dataclass, field

from khedron.build import UNIDENTIFIED_BUILD_SUFFIX
from khedron.eligibility import Eligibility
from khedron.methodology import get_runtime_profile
from khedron.methodology.archived import archived_profile
from khedron.types import RunStartedEvent

__all__ = ["Disposition", "resolve_disposition"]


@dataclass(frozen=True)
class Disposition:
    """Whether a run's scores may be published, and every reason they may not."""

    publishable: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if self.publishable:
            return "publishable"
        return "withdrawn: " + "; ".join(self.reasons)


def resolve_disposition(
    run_started: RunStartedEvent, eligibility: Eligibility | None = None
) -> Disposition:
    """Decide from the run's own record whether its scores may be published.

    One resolver, called by every consumer, rather than a check per consumer. The previous design
    listed a check per consumer and that is how they drift apart and how one gets forgotten -- the
    same defect class as the redaction denylist this project already replaced with an allowlist.

    It reads the run's own record rather than a register of decisions, so a run made before any of
    these rules existed still resolves correctly without anyone writing an assessment for it. The
    day-1 results are the case in point: executed under a profile that pinned no model, no
    retrieval budget and no category set, they resolve as withdrawn because the profile they name
    is superseded, not because someone remembered to withdraw them.

    Fail-closed. A run whose disposition cannot be established is not publishable: an unknown
    profile, an unidentifiable build, or a record missing the fields these rules read all resolve
    to withdrawn. The cost of a false withdrawal is a number nobody publishes; the cost of a false
    publication is the thing this whole project exists to prevent.
    """
    reasons: list[str] = []

    archived = archived_profile(run_started.methodology_profile)
    if archived is not None:
        # A named outcome, not a shrug. The tombstone exists so a removed profile's runs stay
        # readable with a specific disposition; falling through to "cannot be resolved" would make
        # every one of them indistinguishable from a typo, which is the state the archive was built
        # to avoid.
        return Disposition(
            publishable=False,
            reasons=(
                f"it was measured under {archived.name!r}, archived on {archived.archived_on}: "
                f"{archived.reason}",
            ),
        )

    try:
        profile = get_runtime_profile(run_started.methodology_profile)
    except Exception:
        return Disposition(
            publishable=False,
            reasons=(
                f"its methodology profile {run_started.methodology_profile!r} cannot be resolved, "
                "so what it measured cannot be established",
            ),
        )

    if profile.superseded_by is not None:
        reasons.append(
            f"it was measured under {profile.name!r}, superseded by {profile.superseded_by!r}"
        )
    if profile.status != "ready":
        reasons.append(f"its profile status is {profile.status!r}, not 'ready'")

    unpinned = [
        name
        for name in ("answer_model_id", "judge_model_id", "evaluation_categories")
        if getattr(profile, name, None) is None
    ]
    if unpinned:
        reasons.append(
            "its profile pins neither " + nor_join(unpinned) + ", so the run cannot state what it "
            "measured"
        )

    version = run_started.framework_version
    if version.endswith(".dirty"):
        reasons.append(
            f"it was produced from an uncommitted working tree ({version}), so the code that "
            "produced it is in no commit"
        )
    elif version.endswith(f"+{UNIDENTIFIED_BUILD_SUFFIX}"):
        reasons.append(f"it records no identifiable build ({version})")

    reasons.extend(_corpus_reasons(run_started))
    if eligibility is not None and not eligibility.eligible:
        reasons.extend(eligibility.reasons)

    if run_started.runtime_environment.get("answers_regenerated") is False:
        reasons.append(
            "its answers were produced by another run and only re-scored here, so it is a "
            "re-grading of that run rather than an independent measurement"
        )

    return Disposition(publishable=not reasons, reasons=tuple(reasons))


def _corpus_reasons(run_started: RunStartedEvent) -> list[str]:
    """Refuse anything but a demonstrated full-corpus run.

    Written the other way round first, and that was the defect: the check fired only when both
    counts were present integers and one was smaller. A run whose coverage was absent, non-numeric
    or zero therefore produced no reason at all and resolved as publishable -- in a resolver whose
    own docstring promises it fails closed. Unknown coverage is exactly the case the promise was
    about.
    """
    environment = run_started.runtime_environment
    evaluated = environment.get("corpus_conversations_evaluated")
    available = environment.get("corpus_conversations_available")
    if (
        not isinstance(evaluated, int)
        or not isinstance(available, int)
        or isinstance(evaluated, bool)
        or isinstance(available, bool)
    ):
        return [
            "its corpus coverage is not recorded, so whether it measured the benchmark or a "
            "fragment of it cannot be established"
        ]
    if available <= 0:
        return [f"it records a corpus of {available} conversations, which cannot be measured"]
    if evaluated < available:
        return [
            f"it covered {evaluated} of {available} conversations, so its scores describe the "
            "selected subset rather than the benchmark"
        ]
    return []


def nor_join(names: list[str]) -> str:
    """Join field names for a refusal message: 'a', 'a nor b', 'a, b nor c'."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " nor " + names[-1]
