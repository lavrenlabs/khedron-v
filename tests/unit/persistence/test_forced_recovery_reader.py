"""Strict-reader validation of the forced-recovery audit trail."""

from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from khedron.persistence.stage_reader import (
    HistoricalRunClassification,
    StrictRunReader,
    StrictStreamReason,
)
from khedron.types import (
    ErrorRecord,
    ErrorResolutionRecord,
    ForcedRecoveryEvent,
    Judgment,
    JudgmentVerdict,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunStartedEvent,
    question_plan_fingerprint,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
RUN_ID = "run-forced"


def at(seconds: float) -> datetime:
    return NOW + timedelta(seconds=seconds)


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records), encoding="utf-8"
    )


@dataclass
class ForcedFixture:
    """The individual records of one complete judge-stage forced recovery, mutable per test."""

    error: ErrorRecord
    forced: ForcedRecoveryEvent
    attempt: RecoveryAttemptRecord
    judgment: Judgment | None
    resolution: ErrorResolutionRecord | None
    run_id: str = RUN_ID

    def write(self, root: Path) -> Path:
        run_dir = root / self.run_id
        question_ids = ("q-1",)
        write_jsonl(
            run_dir / "lifecycle.jsonl",
            [
                RunStartedEvent(
                    event_id="start-1",
                    timestamp=NOW,
                    run_id=self.run_id,
                    sequence_number=0,
                    suite_id="suite-1",
                    experiment_id="experiment-1",
                    experiment_name="test",
                    run_number=0,
                    provider_type="full_context",
                    provider_version="1",
                    benchmark_type="locomo",
                    benchmark_version="1",
                    benchmark_checksum="sha256:test",
                    answer_model_id="model",
                    answer_model_vendor="openai",
                    judge_model_id="judge",
                    judge_model_vendor="openai",
                    config={},
                    methodology_version="1",
                    methodology_profile="test",
                    framework_version="1",
                    seed=1,
                    runtime_environment={},
                )
            ],
        )
        write_jsonl(
            run_dir / "question_plan.jsonl",
            [
                QuestionPlanRecord(
                    run_id=self.run_id,
                    benchmark_id="locomo",
                    categories=("single_hop",),
                    corpus_checksum="sha256:test",
                    question_ids=question_ids,
                    fingerprint=question_plan_fingerprint(
                        benchmark_id="locomo",
                        categories=("single_hop",),
                        corpus_checksum="sha256:test",
                        question_ids=question_ids,
                    ),
                    timestamp=NOW,
                )
            ],
        )
        write_jsonl(
            run_dir / "question_evaluations.jsonl",
            [
                QuestionEvaluationRecord(
                    question_evaluation_id="evaluation-1",
                    run_id=self.run_id,
                    question_id="q-1",
                    conversation_id="conversation-1",
                    category=QuestionCategory.SINGLE_HOP,
                    question_text="Question",
                    expected_answer="Answer",
                    is_audited_error=False,
                    timestamp=NOW,
                )
            ],
        )
        write_jsonl(
            run_dir / "retrievals.jsonl",
            [
                RetrievalRecord(
                    retrieval_id="retrieval-1",
                    question_evaluation_id="evaluation-1",
                    run_id=self.run_id,
                    question_id="q-1",
                    timestamp=at(1),
                    query="Question",
                    top_k=1,
                    n_returned=0,
                    memories=[],
                    retrieval_latency_ms=1,
                )
            ],
        )
        write_jsonl(
            run_dir / "responses.jsonl",
            [
                Response(
                    response_id="response-1",
                    run_id=self.run_id,
                    question_id="q-1",
                    retrieval_id="retrieval-1",
                    timestamp=at(2),
                    model_id="model",
                    prompt="prompt",
                    answer_text="answer",
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1,
                    cost_usd=0,
                )
            ],
        )
        write_jsonl(run_dir / "errors.jsonl", [self.error])
        write_jsonl(run_dir / "recovery_attempts.jsonl", [self.attempt])
        write_jsonl(run_dir / "forced_recoveries.jsonl", [self.forced])
        if self.judgment is not None:
            write_jsonl(run_dir / "judgments.jsonl", [self.judgment])
        if self.resolution is not None:
            write_jsonl(run_dir / "error_resolutions.jsonl", [self.resolution])
        return run_dir


def make_fixture() -> ForcedFixture:
    """A complete, valid judge-stage forced recovery (error<=forced<=attempt<=target<=res)."""
    error = ErrorRecord(
        error_id="error-1",
        run_id=RUN_ID,
        timestamp=at(3),
        phase="judge",
        question_id="q-1",
        error_type="ModelAuthenticationError",
        error_message="safe test error",
        stack_trace=None,
        context={},
        recovered=False,
        retryable=False,
    )
    forced = ForcedRecoveryEvent(
        forced_recovery_id="forced-1",
        run_id=RUN_ID,
        question_id="q-1",
        error_id="error-1",
        attempt_id="attempt-1",
        stage="judge",
        reason="key reissued out of band",
        timestamp=at(4),
    )
    attempt = RecoveryAttemptRecord(
        attempt_id="attempt-1",
        run_id=RUN_ID,
        question_id="q-1",
        error_id="error-1",
        stage="judge",
        timestamp=at(5),
    )
    judgment = Judgment(
        judgment_id="judgment-1",
        run_id=RUN_ID,
        response_id="response-1",
        question_id="q-1",
        timestamp=at(6),
        judge_model_id="judge",
        prompt="judge prompt",
        raw_judge_output="{}",
        parsed_verdict=JudgmentVerdict.CORRECT,
        parsed_score=1.0,
        parsed_reasoning="ok",
        parse_was_successful=True,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        cost_usd=0,
        recovery_attempt_id="attempt-1",
    )
    resolution = ErrorResolutionRecord(
        resolution_id="resolution-1",
        run_id=RUN_ID,
        question_id="q-1",
        error_id="error-1",
        recovery_attempt_id="attempt-1",
        resolved_by_stage="judge",
        resolved_by_stage_record_id="judgment-1",
        timestamp=at(7),
    )
    return ForcedFixture(error, forced, attempt, judgment, resolution)


def reason_codes(run_dir: Path, run_id: str = RUN_ID) -> set[StrictStreamReason]:
    return set(StrictRunReader().finding(run_dir, run_id).reason_codes)


# --- Test 15: round-trip -----------------------------------------------------------------------


def test_forced_recovery_round_trips_through_the_reader(tmp_path: Path) -> None:
    run_dir = make_fixture().write(tmp_path)

    stream = StrictRunReader().read(run_dir, RUN_ID)

    assert [forced.forced_recovery_id for forced in stream.forced_recoveries] == ["forced-1"]


# --- Test 16: misfiled detection ---------------------------------------------------------------


def test_forced_record_in_errors_file_is_file_record_type(tmp_path: Path) -> None:
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    # A ForcedRecoveryEvent line dropped into errors.jsonl alongside the real error.
    (run_dir / "errors.jsonl").write_text(
        f"{fixture.error.model_dump_json()}\n{fixture.forced.model_dump_json()}\n",
        encoding="utf-8",
    )

    assert StrictStreamReason.FILE_RECORD_TYPE in reason_codes(run_dir)


def test_attempt_record_in_forced_file_is_file_record_type(tmp_path: Path) -> None:
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    # A RecoveryAttemptRecord line dropped into forced_recoveries.jsonl.
    (run_dir / "forced_recoveries.jsonl").write_text(
        f"{fixture.attempt.model_dump_json()}\n", encoding="utf-8"
    )

    assert StrictStreamReason.FILE_RECORD_TYPE in reason_codes(run_dir)


# --- Test 17: id/run/attempt uniqueness --------------------------------------------------------


def test_forced_run_id_mismatch_is_run_mismatch(tmp_path: Path) -> None:
    fixture = make_fixture()
    fixture.forced = fixture.forced.model_copy(update={"run_id": "some-other-run"})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.RUN_MISMATCH in reason_codes(run_dir)


def test_duplicate_forced_recovery_id_is_duplicate_record_id(tmp_path: Path) -> None:
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    (run_dir / "forced_recoveries.jsonl").write_text(
        f"{fixture.forced.model_dump_json()}\n{fixture.forced.model_dump_json()}\n",
        encoding="utf-8",
    )

    assert StrictStreamReason.DUPLICATE_RECORD_ID in reason_codes(run_dir)


def test_two_forced_records_on_one_attempt_is_duplicate_authorization(tmp_path: Path) -> None:
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    second = fixture.forced.model_copy(update={"forced_recovery_id": "forced-2"})
    (run_dir / "forced_recoveries.jsonl").write_text(
        f"{fixture.forced.model_dump_json()}\n{second.model_dump_json()}\n",
        encoding="utf-8",
    )

    assert StrictStreamReason.DUPLICATE_FORCED_AUTHORIZATION in reason_codes(run_dir)


# --- Test 18: reference integrity --------------------------------------------------------------


def test_forced_missing_error_is_forced_error_reference(tmp_path: Path) -> None:
    fixture = make_fixture()
    fixture.forced = fixture.forced.model_copy(update={"error_id": "no-such-error"})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.FORCED_ERROR_REFERENCE in reason_codes(run_dir)


def test_forced_question_disagrees_with_error_is_forced_error_reference(tmp_path: Path) -> None:
    fixture = make_fixture()
    fixture.forced = fixture.forced.model_copy(update={"question_id": "q-other"})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.FORCED_ERROR_REFERENCE in reason_codes(run_dir)


def test_forced_on_retryable_error_is_forced_on_retryable(tmp_path: Path) -> None:
    fixture = make_fixture()
    # The one thing forced recovery must never authorize: a retryable error is the automatic path's.
    fixture.error = fixture.error.model_copy(update={"retryable": True})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.FORCED_ON_RETRYABLE in reason_codes(run_dir)


def test_forced_attempt_reference_disagreement_is_forced_attempt_reference(tmp_path: Path) -> None:
    # The forced record references the error correctly (judge/q-1), but the attempt it names is for
    # a different stage, so the authorization does not describe the attempt it claims to authorize.
    fixture = make_fixture()
    fixture.attempt = fixture.attempt.model_copy(update={"stage": "generate"})
    fixture.judgment = None
    fixture.resolution = None
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.FORCED_ATTEMPT_REFERENCE in reason_codes(run_dir)


# --- Test 19: temporal order -------------------------------------------------------------------


def test_forced_before_error_is_temporal_order(tmp_path: Path) -> None:
    fixture = make_fixture()
    # forced.timestamp < error.timestamp violates error <= forced.
    fixture.forced = fixture.forced.model_copy(update={"timestamp": at(2.5)})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.TEMPORAL_ORDER in reason_codes(run_dir)


def test_attempt_before_forced_is_temporal_order(tmp_path: Path) -> None:
    fixture = make_fixture()
    # attempt.timestamp < forced.timestamp violates forced <= attempt -- exactly the inversion the
    # old attempt-first write order produced, which the current stamp order fixes.
    fixture.attempt = fixture.attempt.model_copy(update={"timestamp": at(3.5)})
    run_dir = fixture.write(tmp_path)

    assert StrictStreamReason.TEMPORAL_ORDER in reason_codes(run_dir)


# --- Test 20: crash window admissible ----------------------------------------------------------


def test_forced_authorization_without_attempt_is_admissible(tmp_path: Path) -> None:
    # Authorization persisted, process died before the attempt: a benign crash window, never a
    # diagnostic. The forced record names an attempt id that no attempt record exists for.
    fixture = make_fixture()
    fixture.forced = fixture.forced.model_copy(update={"attempt_id": "attempt-never-written"})
    # Remove the attempt/judgment/resolution so nothing references the missing attempt.
    fixture.attempt = fixture.attempt.model_copy(
        update={"attempt_id": "attempt-unrelated", "error_id": "error-unrelated"}
    )
    fixture.judgment = None
    fixture.resolution = None
    run_dir = fixture.write(tmp_path)
    # Give the unrelated attempt a real (retryable) error so it does not itself become a diagnostic.
    unrelated_error = fixture.error.model_copy(
        update={"error_id": "error-unrelated", "retryable": True, "timestamp": at(3)}
    )
    write_jsonl(run_dir / "errors.jsonl", [fixture.error, unrelated_error])

    codes = reason_codes(run_dir)
    assert StrictStreamReason.FORCED_ATTEMPT_REFERENCE not in codes
    assert StrictStreamReason.FORCED_ERROR_REFERENCE not in codes


# --- Test 21: inverse invariant (advisory-only) ------------------------------------------------


def test_unauthorized_non_retryable_recovery_is_flagged_by_finding(tmp_path: Path) -> None:
    # A RecoveryAttemptRecord against a non-retryable error with NO authorizing ForcedRecoveryEvent
    # is corrupt -- the forgery the inverse invariant refuses. Advisory-only: finding() flags it.
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    (run_dir / "forced_recoveries.jsonl").unlink()

    finding = StrictRunReader().finding(run_dir, RUN_ID)
    assert finding.classification is HistoricalRunClassification.CORRUPT
    assert StrictStreamReason.UNAUTHORIZED_RECOVERY in finding.reason_codes


def test_inverse_invariant_is_advisory_only_read_does_not_raise(tmp_path: Path) -> None:
    # R2: the inverse invariant is wired into finding() ONLY, never the raising read() path, until
    # it is verified against real historical fixtures. read() must NOT raise on the forged stream.
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    (run_dir / "forced_recoveries.jsonl").unlink()

    stream = StrictRunReader().read(run_dir, RUN_ID)  # must not raise
    assert stream.attempts  # the unauthorized attempt is still present, just not raised on


# --- Test 22: backward compatibility -----------------------------------------------------------


def test_run_without_forced_recoveries_file_reads_clean(tmp_path: Path) -> None:
    fixture = make_fixture()
    run_dir = fixture.write(tmp_path)
    (run_dir / "forced_recoveries.jsonl").unlink()
    # Make the sole attempt reference a retryable error so no inverse-invariant concern arises.
    retryable_error = fixture.error.model_copy(update={"retryable": True})
    write_jsonl(run_dir / "errors.jsonl", [retryable_error])

    finding = StrictRunReader().finding(run_dir, RUN_ID)
    assert finding.classification is HistoricalRunClassification.VALID
    assert not finding.reason_codes


def test_historical_retryable_only_attempts_are_not_flagged_by_inverse_invariant(
    tmp_path: Path,
) -> None:
    # The historical shape (no forced_recoveries.jsonl; every attempt against a RETRYABLE error, the
    # only kind the runner ever dispatched before this feature) must NOT be flagged. This is the
    # backward-compat guarantee R2 requires before the check could ever move into read().
    fixture = make_fixture()
    fixture.error = fixture.error.model_copy(update={"retryable": True})
    run_dir = fixture.write(tmp_path)
    (run_dir / "forced_recoveries.jsonl").unlink()

    finding = StrictRunReader().finding(run_dir, RUN_ID)
    assert StrictStreamReason.UNAUTHORIZED_RECOVERY not in finding.reason_codes
