"""Fail-closed eligibility derived from a single strict JSONL snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from khedron.persistence.repository import RunRepository
from khedron.persistence.stage_reader import RunStageStream
from khedron.types import RunCompletedEvent, RunFailedEvent, RunResumedEvent, RunStartedEvent

__all__ = [
    "Eligibility",
    "assess_run_eligibility",
    "assess_stream_eligibility",
    "reduce_lifecycle_status",
]


@dataclass(frozen=True)
class Eligibility:
    """One source-of-truth answer for aggregate, publication, and comparison admission."""

    eligible: bool
    planned_question_ids: frozenset[str]
    judgment_question_ids: frozenset[str]
    lifecycle_status: Literal["running", "completed", "partial", "failed"]
    recovered_error_count: int
    # How many of the recovered errors were reached by operator-forced recovery. Kept
    # SEPARATE from recovered_error_count so a reader sees both the total recovered and how many of
    # those a human forced through the automatic non-retryable gate. Disclosed unconditionally; the
    # publish-time gate reads this value, but disclosure does not depend on the gate.
    forced_recovery_count: int = 0
    reasons: tuple[str, ...] = ()


def assess_run_eligibility(repository: RunRepository, run_id: str) -> Eligibility:
    """Read exactly one strict JSONL snapshot; SQLite is deliberately never consulted."""
    return assess_stream_eligibility(repository.get_strict_stage_stream(run_id))


def assess_stream_eligibility(stream: RunStageStream) -> Eligibility:
    """Purely reduce an already captured strict stream."""
    status = reduce_lifecycle_status(stream)
    planned: frozenset[str] = (
        frozenset(stream.plan.question_ids) if stream.plan is not None else frozenset()
    )
    judgments = frozenset(item.question_id for item in stream.judgments)
    resolved = {item.error_id for item in stream.resolutions}
    unresolved = [
        item
        for item in stream.errors
        if item.question_id is not None
        and item.phase in {"retrieve", "generate", "judge"}
        and not item.recovered
        and item.error_id not in resolved
    ]
    reasons: list[str] = []
    if stream.plan is None:
        reasons.append("the run has no planned-question manifest (legacy run)")
    elif planned != judgments:
        reasons.append("planned question IDs do not equal durable judgment question IDs")
    if unresolved:
        reasons.append("it has unresolved question-stage errors")
    if status != "completed":
        reasons.append(f"its lifecycle status is {status!r}, not 'completed'")
    # A forced recovery counts only once it has a consummated attempt: an
    # authorization whose process died before the attempt is an admissible crash window, never a
    # reached verdict, so it must not inflate the disclosed count.
    attempt_ids = {attempt.attempt_id for attempt in stream.attempts}
    forced_recovery_count = sum(
        1 for forced in stream.forced_recoveries if forced.attempt_id in attempt_ids
    )
    return Eligibility(
        eligible=not reasons,
        planned_question_ids=planned,
        judgment_question_ids=judgments,
        lifecycle_status=status,
        recovered_error_count=len(resolved),
        forced_recovery_count=forced_recovery_count,
        reasons=tuple(reasons),
    )


def reduce_lifecycle_status(
    stream: RunStageStream,
) -> Literal["running", "completed", "partial", "failed"]:
    """Reduce lifecycle status from the supplied snapshot without another repository read."""
    status: Literal["running", "completed", "partial", "failed"] = "running"
    started = False
    terminal = False
    for event in stream.lifecycle:
        if isinstance(event, RunStartedEvent):
            started, terminal, status = True, False, "running"
        elif isinstance(event, RunResumedEvent) and started:
            terminal, status = False, "running"
        elif started and not terminal and isinstance(event, RunCompletedEvent):
            terminal, status = True, event.status
        elif started and not terminal and isinstance(event, RunFailedEvent):
            terminal, status = True, "failed"
    return status
