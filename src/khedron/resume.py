from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from khedron.benchmarks.locomo import EXPECTED_DATASET_SHA256
from khedron.build import is_unidentifiable_build, resolve_framework_version
from khedron.config import ExperimentSuiteConfig
from khedron.errors import ConfigurationError
from khedron.methodology import (
    METHODOLOGY_FINGERPRINT_KEY,
    get_runtime_profile,
    methodology_fingerprint,
)
from khedron.persistence.repository import RunRepository
from khedron.persistence.stage_reader import RunStageStream
from khedron.types import APICallRecord, ConversationProcessedEvent, ErrorRecord, RunStartedEvent
from khedron.utils.ids import generate_ulid
from khedron.utils.redaction import redact_secrets

if TYPE_CHECKING:
    from khedron.runner import RecoverBlockedResult

__all__ = [
    "RecoveryWorkItem",
    "ResumeState",
    "acquire_run_lock",
    "load_forced_recovery_state",
    "load_resume_state",
    "recover_blocked_run",
    "revalidate_forced_recovery",
    "suite_config_digest",
]


@dataclass(frozen=True)
class RecoveryWorkItem:
    """One unresolved stage selected from a strict JSONL snapshot for re-dispatch.

    The automatic resume path selects only ``retryable is True`` errors and leaves the
    forced-recovery fields at their defaults. The forced path selects only
    ``retryable is not True`` errors -- the disjoint complement -- and marks the item ``forced``,
    carrying the operator ``reason`` and a pre-minted ``forced_recovery_id`` for the audit record.
    """

    question_id: str
    stage: Literal["retrieve", "generate", "judge"]
    error_id: str
    forced: bool = False
    reason: str | None = None
    forced_recovery_id: str | None = None


@dataclass(frozen=True)
class ResumeState:
    """What a resumed run may skip, and where its event stream continues from."""

    run_id: str
    experiment_name: str
    run_number: int
    seed: int
    completed_conversations: frozenset[str]
    completed_questions: frozenset[str]
    # The interrupted attempt's own API calls, carried rather than summed here. They seed the
    # resumed run's cost tracker, which is what makes its reported cost the cost of the whole run
    # instead of the cost of the part that happened to execute last.
    prior_api_calls: tuple[APICallRecord, ...]
    next_sequence_number: int
    recovery_work: tuple[RecoveryWorkItem, ...] = ()
    blocked_questions: frozenset[str] = frozenset()

    @property
    def prior_spend_usd(self) -> float:
        """What this run has already spent, under its own run_id.

        Derived rather than stored: as a separate field it could disagree with the records it was
        meant to summarise, and a cost that disagrees with its own evidence is the defect this
        property exists to make impossible.
        """
        return sum(call.cost_usd for call in self.prior_api_calls)


def suite_config_digest(config: ExperimentSuiteConfig) -> str:
    """Hash the suite configuration as it would be recorded, credentials removed.

    Redacted before hashing for the same reason the artifacts are: a digest taken over a live API
    key changes when that key rotates, which would refuse a legitimate resume for a reason with
    nothing to do with the measurement.
    """
    payload = json.dumps(redact_secrets(config.model_dump(mode="json")), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_resume_state(
    *,
    run_id: str,
    config: ExperimentSuiteConfig,
    repository: RunRepository,
) -> ResumeState:
    """Establish that a run may be resumed under this configuration, and what it may skip.

    Every check here exists to stop a resumed run splicing two different measurements into one
    artifact. A run is a claim about what was measured; continuing it under a changed profile, a
    changed corpus or a different build would make that claim false while leaving every score and
    interval looking untouched -- the failure mode this methodology version exists to remove.

    Refusing is always the safe outcome. The alternative to a resumed run is a fresh one: it costs
    money and hours, but it cannot be wrong.
    """
    started = repository.get_run_started_event(run_id)
    experiment_name = _matching_experiment(config, started, run_id)
    _require_resumable(started=started, config=config, run_id=run_id)

    processed: set[str] = set()
    next_sequence = 0
    for event in repository.list_run_events(run_id):
        next_sequence = max(next_sequence, event.sequence_number + 1)
        if isinstance(event, ConversationProcessedEvent):
            processed.add(event.conversation_id)

    # Every question the run has a record for, whether or not it reached a verdict. A question
    # left half-written by a hard kill cannot be retried: the index is unique on
    # (run_id, question_id), so re-asking it collides rather than overwriting. Skipping it keeps
    # the record it has -- it scores as unanswered, which is what it was -- and at most one
    # question per interruption ends that way, far below the rate that excludes a run.
    #
    # Filtering on `verdict is not None` here was the defect: it left half-written questions out of
    # the skip set, so the resume re-asked them and the insert failed. The comment above the fix
    # already described the intended behaviour; the code did not implement it.
    snapshot = repository.get_strict_stage_stream(run_id)
    if snapshot.plan is None:
        raise ConfigurationError(
            "Refusing to resume a legacy stream without a planned-question manifest.", run_id=run_id
        )
    answered = {judgment.question_id for judgment in snapshot.judgments}
    # The automatic path takes only retryable errors (`forced=False`). Selection lives in one shared
    # helper so the forced path is provably "the same selection with one clause dropped".
    eligible_errors = _select_eligible_errors(snapshot, config, forced=False)
    recovery_work = tuple(
        RecoveryWorkItem(
            cast(str, error.question_id),
            cast(Literal["retrieve", "generate", "judge"], error.phase),
            error.error_id,
        )
        for _, error in sorted(eligible_errors.items())
    )
    recovery_question_ids = {item.question_id for item in recovery_work}
    reconciled_pending_question_ids = {
        pending.attempt.question_id
        for pending in snapshot.recovery_pending
        if pending.target is not None
    }
    blocked_questions = (
        {evaluation.question_id for evaluation in snapshot.evaluations}
        - answered
        - recovery_question_ids
        - reconciled_pending_question_ids
    )
    prior_api_calls = tuple(repository.get_api_calls_for_run(run_id))

    return ResumeState(
        run_id=run_id,
        experiment_name=experiment_name,
        run_number=started.run_number,
        # Read back rather than recomputed: the seed is part of what the interrupted run *was*, and
        # re-deriving it would silently change the run's identity if the derivation ever changed.
        seed=started.seed,
        completed_conversations=frozenset(processed),
        completed_questions=frozenset(answered),
        prior_api_calls=prior_api_calls,
        next_sequence_number=next_sequence,
        recovery_work=recovery_work,
        blocked_questions=frozenset(blocked_questions),
    )


def load_forced_recovery_state(
    *,
    run_id: str,
    question_ids: Sequence[str],
    reason: str,
    config: ExperimentSuiteConfig,
    repository: RunRepository,
) -> ResumeState:
    """Establish that named, blocked, non-retryable questions may be operator-forced.

    A sibling of ``load_resume_state`` that returns an ordinary ``ResumeState`` so the whole
    execution path is inherited unchanged; it differs only in seeding ``recovery_work`` from named,
    blocked, *non-retryable* errors instead of the automatic retryable set. Every resume guard is
    reused verbatim (``_matching_experiment``, ``_require_resumable``), so the same-build and
    identifiable-build refusals apply with no relaxation -- there is deliberately no
    framework-drift or unidentifiable-build override here. The whole command is refused fail-closed,
    before any lock or spend, if any named question fails the eligibility predicate; the *binding*
    eligibility and build-identity decision is re-taken under the run lock, which is what makes it
    atomic against a concurrent resume.
    """
    if not reason or not reason.strip():
        raise ConfigurationError(
            "recover-blocked requires a non-empty operator reason.", run_id=run_id
        )
    if not question_ids:
        raise ConfigurationError(
            "recover-blocked requires at least one question to recover.", run_id=run_id
        )

    started = repository.get_run_started_event(run_id)
    experiment_name = _matching_experiment(config, started, run_id)
    _require_resumable(started=started, config=config, run_id=run_id)

    processed: set[str] = set()
    next_sequence = 0
    for event in repository.list_run_events(run_id):
        next_sequence = max(next_sequence, event.sequence_number + 1)
        if isinstance(event, ConversationProcessedEvent):
            processed.add(event.conversation_id)

    snapshot = repository.get_strict_stage_stream(run_id)
    if snapshot.plan is None:
        raise ConfigurationError(
            "Refusing to force-recover a legacy stream without a planned-question manifest.",
            run_id=run_id,
        )

    forced_work = _select_forced_recovery_work(
        run_id=run_id,
        question_ids=question_ids,
        reason=reason,
        config=config,
        snapshot=snapshot,
    )

    answered = {judgment.question_id for judgment in snapshot.judgments}
    forced_question_ids = {item.question_id for item in forced_work}
    reconciled_pending_question_ids = {
        pending.attempt.question_id
        for pending in snapshot.recovery_pending
        if pending.target is not None
    }
    # Every named forced question is in recovery_work, so it is excluded from blocked_questions and
    # dispatched. Every OTHER still-blocked question -- including any retryable one, which this
    # command deliberately does not touch (that is plain resume's job) -- stays blocked and skipped,
    # so it is never re-asked into a colliding (run_id, question_id) write.
    blocked_questions = (
        {evaluation.question_id for evaluation in snapshot.evaluations}
        - answered
        - forced_question_ids
        - reconciled_pending_question_ids
    )
    prior_api_calls = tuple(repository.get_api_calls_for_run(run_id))

    return ResumeState(
        run_id=run_id,
        experiment_name=experiment_name,
        run_number=started.run_number,
        seed=started.seed,
        completed_conversations=frozenset(processed),
        completed_questions=frozenset(answered),
        prior_api_calls=prior_api_calls,
        next_sequence_number=next_sequence,
        recovery_work=forced_work,
        blocked_questions=frozenset(blocked_questions),
    )


async def recover_blocked_run(
    *,
    run_id: str,
    question_ids: Sequence[str],
    reason: str,
    config: ExperimentSuiteConfig,
    repository: RunRepository,
    max_cost_usd: float | None = None,
) -> RecoverBlockedResult:
    """Force-recover named, blocked, non-retryable questions of an existing run.

    Builds the forced ``ResumeState`` (which re-applies every resume guard, fail-closed and free,
    before any lock or spend) and delegates to the existing execution engine through a thin
    ``Runner`` method. A sibling of ``rejudge_run``, so the CLI wiring is identical.
    """
    forced_state = load_forced_recovery_state(
        run_id=run_id,
        question_ids=question_ids,
        reason=reason,
        config=config,
        repository=repository,
    )
    # Lazy import: runner imports resume, so importing Runner at module scope would be a cycle.
    from khedron.runner import Runner

    runner = Runner(config, repository)  # resume_run_id stays None -> self._resume is None
    return await runner.recover_blocked(forced_state, max_cost_usd=max_cost_usd)


def revalidate_forced_recovery(
    *,
    run_id: str,
    question_ids: Sequence[str],
    reason: str,
    config: ExperimentSuiteConfig,
    repository: RunRepository,
) -> None:
    """Re-take the binding forced-recovery decision under the run lock.

    Called by the runner AFTER ``acquire_run_lock`` and BEFORE any dispatch. Re-runs the FULL
    ``_require_resumable`` -- both Wall A (recorded == current build) and Wall B (current build
    identifiable), re-resolving the current build fresh -- so a build changed between load and lock,
    including one clean commit to another (``commitA`` -> ``commitB``) that the identifiability
    predicate alone would miss, is caught here. It then re-runs the full 4.1 eligibility predicate
    for every named question against a freshly re-read strict snapshot, so a question resolved or
    auto-recovered by a competing path between load and lock is refused. Raises fail-closed on any
    failure; because it fires before any write, a lost race costs an aborted command, never a
    duplicate ``(run_id, question_id)`` attempt.
    """
    started = repository.get_run_started_event(run_id)
    _require_resumable(started=started, config=config, run_id=run_id)
    snapshot = repository.get_strict_stage_stream(run_id)
    if snapshot.plan is None:
        raise ConfigurationError(
            "Refusing to force-recover a legacy stream without a planned-question manifest.",
            run_id=run_id,
        )
    _select_forced_recovery_work(
        run_id=run_id,
        question_ids=question_ids,
        reason=reason,
        config=config,
        snapshot=snapshot,
    )


def _select_eligible_errors(
    snapshot: RunStageStream,
    config: ExperimentSuiteConfig,
    *,
    forced: bool,
) -> dict[tuple[str, str], ErrorRecord]:
    """Select at most one dispatchable error per (question, stage) from a strict snapshot.

    The automatic and forced paths are the SAME selection differing in exactly one clause:
    automatic takes ``retryable is True`` errors, forced takes
    ``retryable is not True`` errors -- the disjoint complement, so no error is ever eligible for
    both. Every other clause is shared: skip already-resolved errors, skip reconciled-pending
    errors, and honour the per-(question, stage) ``max_recovery_attempts`` bound so a repeatedly
    failing recovery -- forced or not -- stays bounded across invocations. A stage can emit another
    error while recovering, so only the latest ``(timestamp, error_id)`` per (question, stage) is
    kept; dispatching two would collide on the unique (run_id, question_id) question index.
    """
    resolved_error_ids = {resolution.error_id for resolution in snapshot.resolutions}
    reconciled_pending_error_ids = {
        pending.error.error_id
        for pending in snapshot.recovery_pending
        if pending.target is not None
    }
    attempt_counts: dict[tuple[str, str], int] = {}
    for attempt in snapshot.attempts:
        key = (attempt.question_id, attempt.stage)
        attempt_counts[key] = attempt_counts.get(key, 0) + 1
    eligible_errors: dict[tuple[str, str], ErrorRecord] = {}
    for error in snapshot.errors:
        if error.question_id is None or error.phase not in {"retrieve", "generate", "judge"}:
            continue
        # The one clause that separates the two selection sets, and nothing else.
        if forced:
            if error.retryable is True:
                continue
        elif error.retryable is not True:
            continue
        if (
            error.error_id in resolved_error_ids
            or error.error_id in reconciled_pending_error_ids
            or attempt_counts.get((error.question_id, error.phase), 0)
            >= config.max_recovery_attempts
        ):
            continue
        key = (error.question_id, error.phase)
        previous = eligible_errors.get(key)
        if previous is None or (error.timestamp, error.error_id) > (
            previous.timestamp,
            previous.error_id,
        ):
            eligible_errors[key] = error
    return eligible_errors


def _select_forced_recovery_work(
    *,
    run_id: str,
    question_ids: Sequence[str],
    reason: str,
    config: ExperimentSuiteConfig,
    snapshot: RunStageStream,
) -> tuple[RecoveryWorkItem, ...]:
    """Enforce the forced-recovery eligibility predicate per named question.

    Fail-closed and atomic: one ineligible named question refuses the whole command, so nothing is
    selected and no side effect follows. Exactly one ``RecoveryWorkItem`` per question,
    mandatory, not cosmetic: the executor keys recovery by question id, so two items sharing a
    question would silently drop one. When a question carries forcible errors in more than one
    stage they collapse to the single latest ``(timestamp, error_id)`` across its stages.
    """
    plan = snapshot.plan
    planned: set[str] = set(plan.question_ids) if plan is not None else set()
    answered = {judgment.question_id for judgment in snapshot.judgments}
    reconciled_pending_question_ids = {
        pending.attempt.question_id
        for pending in snapshot.recovery_pending
        if pending.target is not None
    }
    # A retryable error is the automatic path's job (Decision 4.1 step 4): forcing it is a misuse.
    auto_recoverable_question_ids = {
        question_id
        for (question_id, _stage) in _select_eligible_errors(snapshot, config, forced=False)
    }
    forcible = _select_eligible_errors(snapshot, config, forced=True)
    forcible_by_question: dict[str, ErrorRecord] = {}
    for (question_id, _stage), error in forcible.items():
        previous = forcible_by_question.get(question_id)
        if previous is None or (error.timestamp, error.error_id) > (
            previous.timestamp,
            previous.error_id,
        ):
            forcible_by_question[question_id] = error

    seen: set[str] = set()
    items: list[RecoveryWorkItem] = []
    for question_id in question_ids:
        if question_id in seen:
            continue
        seen.add(question_id)
        if question_id not in planned:
            raise ConfigurationError(
                "Refusing forced recovery: the question is not in this run's planned-question "
                "manifest, so it was never part of the measurement.",
                run_id=run_id,
                question_id=question_id,
            )
        if question_id in answered:
            raise ConfigurationError(
                "Refusing forced recovery: the question already has a judgment (any verdict, "
                "including UNKNOWN, disqualifies it). Re-grading a recorded verdict is `rejudge`, "
                "not recover-blocked.",
                run_id=run_id,
                question_id=question_id,
            )
        if question_id in auto_recoverable_question_ids:
            raise ConfigurationError(
                "Refusing forced recovery: the question has a retryable error the automatic resume "
                "path already handles. Use `run --resume-run-id`, not recover-blocked.",
                run_id=run_id,
                question_id=question_id,
            )
        if question_id in reconciled_pending_question_ids:
            raise ConfigurationError(
                "Refusing forced recovery: the question is mid-reconciliation from a recovery "
                "crash window, which the resume path completes automatically.",
                run_id=run_id,
                question_id=question_id,
            )
        error = forcible_by_question.get(question_id)
        if error is None:
            raise ConfigurationError(
                "Refusing forced recovery: the question has no unresolved, non-retryable, "
                "verdict-less question-stage error to force -- its errors are resolved, retryable, "
                "over the recovery-attempt budget, or absent.",
                run_id=run_id,
                question_id=question_id,
            )
        items.append(
            RecoveryWorkItem(
                question_id=question_id,
                stage=cast(Literal["retrieve", "generate", "judge"], error.phase),
                error_id=error.error_id,
                forced=True,
                reason=reason,
                forced_recovery_id=generate_ulid(),
            )
        )
    return tuple(items)


def _matching_experiment(
    config: ExperimentSuiteConfig,
    started: RunStartedEvent,
    run_id: str,
) -> str:
    for experiment in config.experiments:
        if experiment.name == started.experiment_name:
            return experiment.name
    raise ConfigurationError(
        "The configuration contains no experiment matching the run being resumed.",
        run_id=run_id,
        recorded_experiment=started.experiment_name,
        configured_experiments=[experiment.name for experiment in config.experiments],
    )


def _require_resumable(
    *,
    started: RunStartedEvent,
    config: ExperimentSuiteConfig,
    run_id: str,
) -> None:
    profile = get_runtime_profile(config.methodology_profile)
    framework_version = resolve_framework_version()
    checks: tuple[tuple[str, object, object], ...] = (
        ("methodology profile", started.methodology_profile, profile.name),
        (
            "methodology fingerprint",
            started.runtime_environment.get(METHODOLOGY_FINGERPRINT_KEY),
            methodology_fingerprint(profile),
        ),
        ("methodology version", started.methodology_version, config.methodology_version),
        (
            "configuration digest",
            started.runtime_environment.get("config_digest"),
            suite_config_digest(config),
        ),
        ("dataset checksum", started.benchmark_checksum, _configured_dataset_checksum(config)),
        ("framework version", started.framework_version, framework_version),
    )
    for name, recorded, current in checks:
        if current is None:
            continue
        # A recorded None means the run predates the field. Refusing on that would make every older
        # run unresumable for a reason unrelated to whether continuing is safe, so it is skipped --
        # and not counted as a match either.
        if recorded is None:
            continue
        if recorded != current:
            raise ConfigurationError(
                f"Refusing to resume: the run was produced under a different {name}, so continuing "
                "it would splice two measurements into one artifact.",
                run_id=run_id,
                field=name,
                recorded=recorded,
                current=current,
            )

    if is_unidentifiable_build(framework_version):
        raise ConfigurationError(
            "Refusing to resume from an unidentifiable build: the code that produced the earlier "
            "questions cannot be established, so the resumed run could not state what measured it.",
            run_id=run_id,
            framework_version=framework_version,
        )


def _configured_dataset_checksum(config: ExperimentSuiteConfig) -> str | None:
    """The dataset digest this configuration would load, or None when it cannot be established.

    Compared against the digest the interrupted run recorded, so a corpus edited between the two
    halves of a run is refused rather than averaged. None when the suite's experiments name
    different datasets: there is then no single value to compare, and the per-run benchmark
    checksum recorded at load time remains the binding check.
    """
    # Only LoCoMo has a digest this function knows how to derive. Applying its default to any
    # other benchmark type would compare a value the run never recorded and refuse every resume
    # for a reason that has nothing to do with the corpus.
    checksums = {
        str(experiment.benchmark.config.get("expected_dataset_checksum", EXPECTED_DATASET_SHA256))
        for experiment in config.experiments
        if experiment.benchmark.type == "locomo"
    }
    if len(checksums) != 1 or len(config.experiments) != len(
        [e for e in config.experiments if e.benchmark.type == "locomo"]
    ):
        return None
    return checksums.pop()


def acquire_run_lock(results_dir: Path, run_id: str) -> Path:
    """Claim exclusive write access to a run directory, or refuse.

    The reference implementation has no protection here at all: two processes handed the same run
    id interleave their writes silently. Two writers on one append-only stream produce an artifact
    that is internally inconsistent in a way no later reader can detect, which is a worse outcome
    than either process refusing to start.
    """
    lock_path = results_dir / "runs" / run_id / "resume.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ConfigurationError(
            "Another process holds the resume lock for this run. Two writers on one append-only "
            "stream produce an artifact no reader can untangle. Remove the lock only once you are "
            "certain no other run is in flight.",
            run_id=run_id,
            lock_path=str(lock_path),
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "run_id": run_id}))
    return lock_path
