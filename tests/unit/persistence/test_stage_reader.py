"""Causal-resolution validation for the strict stage reader."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from khedron.eligibility import assess_stream_eligibility
from khedron.errors import PersistenceError
from khedron.persistence.stage_reader import (
    HistoricalRunClassification,
    StrictRunReader,
    StrictStreamReason,
    preflight_historical_runs,
)
from khedron.types import (
    ConversationIngestionRecord,
    ErrorRecord,
    ErrorResolutionRecord,
    Judgment,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
    question_plan_fingerprint,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)

EXPECTED_STRICT_STREAM_REASONS = {
    "MISSING_LIFECYCLE",
    "LIFECYCLE_PARSE",
    "LIFECYCLE_RUN_MISMATCH",
    "JSON_PARSE",
    "RECORD_VALIDATION",
    "FILE_RECORD_TYPE",
    "EMPTY_MANIFEST",
    "MULTIPLE_MANIFESTS",
    "MANIFEST_RUN_MISMATCH",
    "DUPLICATE_MANIFEST_QUESTION_ID",
    "STAGE_OUTSIDE_MANIFEST",
    "NEW_RECORD_WITHOUT_MANIFEST",
    "DUPLICATE_RECORD_ID",
    "DUPLICATE_STAGE",
    "ORPHAN_STAGE",
    "RUN_MISMATCH",
    "QUESTION_MISMATCH",
    "BROKEN_PARENT_REFERENCE",
    "ATTEMPT_ERROR_REFERENCE",
    "ATTEMPT_STAGE_MISMATCH",
    "RESOLUTION_ERROR_REFERENCE",
    "DUPLICATE_RESOLUTION_ERROR",
    "RESOLUTION_ATTEMPT_REFERENCE",
    "RESOLUTION_STAGE_REFERENCE",
    "RECOVERY_ATTEMPT_MISMATCH",
    "TEMPORAL_ORDER",
    "INGESTION_PROVENANCE",
    "UNREFERENCED_ATTEMPT",
    "FORCED_ERROR_REFERENCE",
    "FORCED_ATTEMPT_REFERENCE",
    "FORCED_ON_RETRYABLE",
    "DUPLICATE_FORCED_AUTHORIZATION",
    "UNAUTHORIZED_RECOVERY",
}


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{record.model_dump_json()}\n" for record in records)
    path.write_text(content, encoding="utf-8")


def write_causal_stream(root: Path) -> Path:
    """Write one complete, causally linked generate-recovery stream."""
    run_dir = root / "run-1"
    write_jsonl(
        run_dir / "lifecycle.jsonl",
        [
            RunStartedEvent(
                event_id="start-1",
                timestamp=NOW,
                run_id="run-1",
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
    question_ids = ("q-1",)
    write_jsonl(
        run_dir / "question_plan.jsonl",
        [
            QuestionPlanRecord(
                run_id="run-1",
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
                run_id="run-1",
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
                run_id="run-1",
                question_id="q-1",
                timestamp=NOW,
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
                run_id="run-1",
                question_id="q-1",
                retrieval_id="retrieval-1",
                timestamp=NOW + timedelta(seconds=2),
                model_id="model",
                prompt="prompt",
                answer_text="answer",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                cost_usd=0,
                recovery_attempt_id="attempt-1",
            )
        ],
    )
    write_jsonl(
        run_dir / "errors.jsonl",
        [
            ErrorRecord(
                error_id="error-1",
                run_id="run-1",
                timestamp=NOW,
                phase="generate",
                question_id="q-1",
                error_type="Timeout",
                error_message="safe test error",
                stack_trace=None,
                context={},
                recovered=False,
                retryable=True,
            )
        ],
    )
    write_jsonl(
        run_dir / "recovery_attempts.jsonl",
        [
            RecoveryAttemptRecord(
                attempt_id="attempt-1",
                run_id="run-1",
                question_id="q-1",
                error_id="error-1",
                stage="generate",
                timestamp=NOW + timedelta(seconds=1),
            )
        ],
    )
    write_jsonl(
        run_dir / "error_resolutions.jsonl",
        [
            ErrorResolutionRecord(
                resolution_id="resolution-1",
                run_id="run-1",
                question_id="q-1",
                error_id="error-1",
                recovery_attempt_id="attempt-1",
                resolved_by_stage="generate",
                resolved_by_stage_record_id="response-1",
                timestamp=NOW + timedelta(seconds=3),
            )
        ],
    )
    return run_dir


def test_reader_accepts_complete_causal_resolution(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)

    stream = StrictRunReader().read(run_dir, "run-1")

    if stream.resolutions[0].resolution_id != "resolution-1":
        raise AssertionError(stream.resolutions)


def test_eligibility_treats_a_causally_resolved_error_as_resolved(tmp_path: Path) -> None:
    stream = StrictRunReader().read(write_causal_stream(tmp_path), "run-1")

    eligibility = assess_stream_eligibility(stream)

    if eligibility.recovered_error_count != 1:
        raise AssertionError(eligibility)
    if "it has unresolved question-stage errors" in eligibility.reasons:
        raise AssertionError(eligibility)
    # The helper intentionally has no terminal lifecycle event or judgment.  Those independent
    # conditions still fail closed, proving the reducer does not let recovery launder a partial run.
    if eligibility.eligible or eligibility.lifecycle_status != "running":
        raise AssertionError(eligibility)


def test_eligibility_marks_a_legacy_stream_ineligible_not_corrupt(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    for name in (
        "question_plan.jsonl",
        "errors.jsonl",
        "recovery_attempts.jsonl",
        "error_resolutions.jsonl",
    ):
        (run_dir / name).unlink()

    eligibility = assess_stream_eligibility(StrictRunReader().read(run_dir, "run-1"))

    expected = (
        "the run has no planned-question manifest (legacy run)",
        "its lifecycle status is 'running', not 'completed'",
    )
    if eligibility.eligible or eligibility.reasons != expected:
        raise AssertionError(eligibility)


def test_eligibility_never_downgrades_a_corrupt_stream(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    (run_dir / "judgments.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(PersistenceError, match="Strict run stream is corrupt"):
        StrictRunReader().read(run_dir, "run-1")


def test_pure_eligibility_accepts_a_complete_snapshot_without_sqlite(tmp_path: Path) -> None:
    """The reducer accepts only the supplied stream; no repository or SQLite exists here."""
    stream = StrictRunReader().read(write_causal_stream(tmp_path), "run-1")
    completed = RunCompletedEvent.model_construct(status="completed")
    judgment = Judgment.model_construct(question_id="q-1")

    eligibility = assess_stream_eligibility(
        replace(stream, lifecycle=(*stream.lifecycle, completed), judgments=(judgment,))
    )

    if not eligibility.eligible:
        raise AssertionError(eligibility)


def test_eligibility_rejects_a_completed_snapshot_missing_a_judgment(tmp_path: Path) -> None:
    stream = StrictRunReader().read(write_causal_stream(tmp_path), "run-1")
    completed = RunCompletedEvent.model_construct(status="completed")

    eligibility = assess_stream_eligibility(
        replace(stream, lifecycle=(*stream.lifecycle, completed))
    )

    if (
        eligibility.eligible
        or "planned question IDs do not equal durable judgment question IDs"
        not in eligibility.reasons
    ):
        raise AssertionError(eligibility)


@pytest.mark.parametrize(
    "error",
    [
        ErrorRecord.model_construct(
            question_id="q-1", phase="retrieve", recovered=True, error_id="allowed-empty"
        ),
        ErrorRecord.model_construct(
            question_id=None, phase="ingest", recovered=False, error_id="ingestion"
        ),
    ],
)
def test_eligibility_ignores_allowed_empty_and_questionless_ingestion(
    tmp_path: Path, error: ErrorRecord
) -> None:
    stream = StrictRunReader().read(write_causal_stream(tmp_path), "run-1")
    complete = replace(
        stream,
        lifecycle=(*stream.lifecycle, RunCompletedEvent.model_construct(status="completed")),
        judgments=(Judgment.model_construct(question_id="q-1"),),
        errors=(error,),
        resolutions=(),
    )
    if not assess_stream_eligibility(complete).eligible:
        raise AssertionError(assess_stream_eligibility(complete))


def test_reader_exposes_a_complete_unresolved_crash_window_as_pending(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    write_jsonl(run_dir / "error_resolutions.jsonl", [])

    stream = StrictRunReader().read(run_dir, "run-1")

    if len(stream.recovery_pending) != 1:
        raise AssertionError(stream.recovery_pending)
    if stream.recovery_pending[0].attempt.attempt_id != "attempt-1":
        raise AssertionError(stream.recovery_pending)
    if stream.recovery_pending[0].target.recovery_attempt_id != "attempt-1":
        raise AssertionError(stream.recovery_pending)


def test_reader_accepts_a_durable_recovery_dispatch_before_the_recovered_stage(
    tmp_path: Path,
) -> None:
    run_dir = write_causal_stream(tmp_path)
    write_jsonl(run_dir / "error_resolutions.jsonl", [])
    (run_dir / "responses.jsonl").unlink()

    stream = StrictRunReader().read(run_dir, "run-1")

    if len(stream.recovery_pending) != 1 or stream.recovery_pending[0].target is not None:
        raise AssertionError(stream.recovery_pending)


def test_reader_rejects_ambiguous_targetless_recovery_attempts(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    write_jsonl(run_dir / "error_resolutions.jsonl", [])
    (run_dir / "responses.jsonl").unlink()
    first = RecoveryAttemptRecord.model_validate_json(
        (run_dir / "recovery_attempts.jsonl").read_text(encoding="utf-8").strip()
    )
    second = first.model_copy(update={"attempt_id": "attempt-2"})
    write_jsonl(run_dir / "recovery_attempts.jsonl", [first, second])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if finding.reason_codes != (StrictStreamReason.UNREFERENCED_ATTEMPT,):
        raise AssertionError(finding)


@pytest.mark.parametrize(
    ("file_name", "record_index", "updates"),
    [
        ("errors.jsonl", 0, {"question_id": "q-other"}),
        ("errors.jsonl", 0, {"phase": "judge"}),
        ("recovery_attempts.jsonl", 0, {"error_id": "other-error"}),
        ("recovery_attempts.jsonl", 0, {"question_id": "q-other"}),
        ("recovery_attempts.jsonl", 0, {"stage": "judge"}),
        ("error_resolutions.jsonl", 0, {"error_id": "other-error"}),
        ("error_resolutions.jsonl", 0, {"question_id": "q-other"}),
        ("error_resolutions.jsonl", 0, {"resolved_by_stage": "retrieve"}),
        ("error_resolutions.jsonl", 0, {"resolved_by_stage_record_id": "retrieval-1"}),
    ],
    ids=[
        "error-question",
        "error-stage",
        "attempt-error",
        "attempt-question",
        "attempt-stage",
        "resolution-error",
        "resolution-question",
        "resolution-stage",
        "resolution-target-stage",
    ],
)
def test_reader_rejects_causal_link_mismatches(
    tmp_path: Path,
    file_name: str,
    record_index: int,
    updates: dict[str, str],
) -> None:
    run_dir = write_causal_stream(tmp_path)
    model_by_file: dict[str, type[BaseModel]] = {
        "errors.jsonl": ErrorRecord,
        "recovery_attempts.jsonl": RecoveryAttemptRecord,
        "error_resolutions.jsonl": ErrorResolutionRecord,
    }
    model = model_by_file[file_name]
    contents = (run_dir / file_name).read_text().splitlines()
    records = [model.model_validate_json(line) for line in contents]
    records[record_index] = records[record_index].model_copy(update=updates)
    write_jsonl(run_dir / file_name, records)

    with pytest.raises(PersistenceError):
        StrictRunReader().read(run_dir, "run-1")


@pytest.mark.parametrize(
    ("file_name", "timestamp"),
    [
        ("recovery_attempts.jsonl", NOW - timedelta(seconds=1)),
        ("responses.jsonl", NOW + timedelta(milliseconds=500)),
        ("error_resolutions.jsonl", NOW + timedelta(milliseconds=1500)),
    ],
    ids=["attempt-before-error", "stage-before-attempt", "resolution-before-stage"],
)
def test_reader_rejects_non_causal_temporal_order(
    tmp_path: Path, file_name: str, timestamp: datetime
) -> None:
    run_dir = write_causal_stream(tmp_path)
    model_by_file: dict[str, type[BaseModel]] = {
        "recovery_attempts.jsonl": RecoveryAttemptRecord,
        "responses.jsonl": Response,
        "error_resolutions.jsonl": ErrorResolutionRecord,
    }
    model = model_by_file[file_name]
    path = run_dir / file_name
    records = [model.model_validate_json(line) for line in path.read_text().splitlines()]
    records[0] = records[0].model_copy(update={"timestamp": timestamp})
    write_jsonl(path, records)

    with pytest.raises(PersistenceError):
        StrictRunReader().read(run_dir, "run-1")


def test_preflight_distinguishes_missing_and_empty_manifest(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path / "runs")
    (run_dir / "question_plan.jsonl").unlink()
    (run_dir / "recovery_attempts.jsonl").unlink()
    (run_dir / "error_resolutions.jsonl").unlink()
    legacy = preflight_historical_runs(tmp_path)[0]
    if legacy.classification is not HistoricalRunClassification.LEGACY_INELIGIBLE:
        raise AssertionError(legacy)

    write_jsonl(run_dir / "question_plan.jsonl", [])
    corrupt = preflight_historical_runs(tmp_path)[0]
    if corrupt.reason_codes != (StrictStreamReason.EMPTY_MANIFEST,):
        raise AssertionError(corrupt)


def test_retrieve_recovery_requires_matching_ingestion_snapshot(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    retrieval = RetrievalRecord.model_validate_json(
        (run_dir / "retrievals.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"recovery_attempt_id": "attempt-1", "ingestion_attempt_id": "missing"})
    write_jsonl(run_dir / "retrievals.jsonl", [retrieval])
    error = ErrorRecord.model_validate_json(
        (run_dir / "errors.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"phase": "retrieve"})
    write_jsonl(run_dir / "errors.jsonl", [error])
    attempt = RecoveryAttemptRecord.model_validate_json(
        (run_dir / "recovery_attempts.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"stage": "retrieve"})
    write_jsonl(run_dir / "recovery_attempts.jsonl", [attempt])
    resolution = ErrorResolutionRecord.model_validate_json(
        (run_dir / "error_resolutions.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(
        update={"resolved_by_stage": "retrieve", "resolved_by_stage_record_id": "retrieval-1"}
    )
    write_jsonl(run_dir / "error_resolutions.jsonl", [resolution])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if StrictStreamReason.INGESTION_PROVENANCE not in finding.reason_codes:
        raise AssertionError(finding)


def test_preflight_reports_exact_ordered_independent_root_causes(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path / "runs")
    (run_dir / "question_plan.jsonl").write_text("\n", encoding="utf-8")
    (run_dir / "recovery_attempts.jsonl").unlink()
    (run_dir / "error_resolutions.jsonl").unlink()
    (run_dir / "responses.jsonl").write_text("{not-json}\n", encoding="utf-8")

    finding = preflight_historical_runs(tmp_path)[0]

    if finding.reason_codes != (
        StrictStreamReason.EMPTY_MANIFEST,
        StrictStreamReason.JSON_PARSE,
    ):
        raise AssertionError(finding)


def test_invalid_exact_parent_suppresses_only_its_child_orphan(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    raw_retrieval = (run_dir / "retrievals.jsonl").read_text(encoding="utf-8")
    (run_dir / "retrievals.jsonl").write_text(
        raw_retrieval.replace('"top_k":1', '"top_k":"bad"'), encoding="utf-8"
    )

    finding = StrictRunReader().finding(run_dir, "run-1")

    if StrictStreamReason.RECORD_VALIDATION not in finding.reason_codes:
        raise AssertionError(finding)
    if StrictStreamReason.ORPHAN_STAGE in finding.reason_codes:
        raise AssertionError(finding)


def test_reader_rejects_second_evaluation_for_one_question(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    evaluation = QuestionEvaluationRecord.model_validate_json(
        (run_dir / "question_evaluations.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"question_evaluation_id": "evaluation-2"})
    write_jsonl(
        run_dir / "question_evaluations.jsonl",
        [
            QuestionEvaluationRecord.model_validate_json(
                (run_dir / "question_evaluations.jsonl").read_text(encoding="utf-8").strip()
            ),
            evaluation,
        ],
    )

    finding = StrictRunReader().finding(run_dir, "run-1")

    if finding.reason_codes != (StrictStreamReason.DUPLICATE_STAGE,):
        raise AssertionError(finding)
    diagnostic = finding.diagnostics[0]
    if diagnostic.line_number != 2 or diagnostic.detail != "q-1":
        raise AssertionError(diagnostic)


def test_foreign_parent_suppresses_only_its_derived_orphan(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    retrieval = RetrievalRecord.model_validate_json(
        (run_dir / "retrievals.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"run_id": "foreign"})
    write_jsonl(run_dir / "retrievals.jsonl", [retrieval])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if StrictStreamReason.RUN_MISMATCH not in finding.reason_codes:
        raise AssertionError(finding)
    if StrictStreamReason.ORPHAN_STAGE in finding.reason_codes:
        raise AssertionError(finding)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", StrictStreamReason.MISSING_LIFECYCLE),
        ("empty", StrictStreamReason.LIFECYCLE_PARSE),
        ("malformed", StrictStreamReason.LIFECYCLE_PARSE),
        ("foreign", StrictStreamReason.LIFECYCLE_RUN_MISMATCH),
    ],
)
def test_lifecycle_contract_diagnostics(
    tmp_path: Path, mutation: str, expected: StrictStreamReason
) -> None:
    run_dir = write_causal_stream(tmp_path)
    lifecycle = run_dir / "lifecycle.jsonl"
    if mutation == "missing":
        lifecycle.unlink()
    elif mutation == "empty":
        lifecycle.write_text("\n", encoding="utf-8")
    elif mutation == "malformed":
        lifecycle.write_text("{broken}\n", encoding="utf-8")
    else:
        event = RunStartedEvent.model_validate_json(lifecycle.read_text(encoding="utf-8"))
        write_jsonl(lifecycle, [event.model_copy(update={"run_id": "foreign"})])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if expected not in finding.reason_codes:
        raise AssertionError(finding)


def test_multiple_manifest_and_new_records_without_manifest_are_exact(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    plan = QuestionPlanRecord.model_validate_json(
        (run_dir / "question_plan.jsonl").read_text(encoding="utf-8").strip()
    )
    write_jsonl(run_dir / "question_plan.jsonl", [plan, plan])
    multiple = StrictRunReader().finding(run_dir, "run-1")
    if StrictStreamReason.MULTIPLE_MANIFESTS not in multiple.reason_codes:
        raise AssertionError(multiple)

    (run_dir / "question_plan.jsonl").unlink()
    without_manifest = StrictRunReader().finding(run_dir, "run-1")
    if StrictStreamReason.NEW_RECORD_WITHOUT_MANIFEST not in without_manifest.reason_codes:
        raise AssertionError(without_manifest)


def test_complete_unresolved_attempt_is_classified_as_recovery_pending(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    (run_dir / "error_resolutions.jsonl").unlink()

    stream = StrictRunReader().read(run_dir, "run-1")

    if len(stream.recovery_pending) != 1:
        raise AssertionError(stream.recovery_pending)


def test_invalid_unreferenced_attempt_reports_causal_reason_before_unreferenced(
    tmp_path: Path,
) -> None:
    run_dir = write_causal_stream(tmp_path)
    (run_dir / "error_resolutions.jsonl").unlink()
    attempt = RecoveryAttemptRecord.model_validate_json(
        (run_dir / "recovery_attempts.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"error_id": "missing-error"})
    write_jsonl(run_dir / "recovery_attempts.jsonl", [attempt])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if finding.reason_codes != (
        StrictStreamReason.ATTEMPT_ERROR_REFERENCE,
        StrictStreamReason.UNREFERENCED_ATTEMPT,
    ):
        raise AssertionError(finding)


def test_null_recovery_target_is_rejected(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    response = Response.model_validate_json(
        (run_dir / "responses.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"recovery_attempt_id": None})
    write_jsonl(run_dir / "responses.jsonl", [response])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if finding.reason_codes != (StrictStreamReason.RECOVERY_ATTEMPT_MISMATCH,):
        raise AssertionError(finding)


@pytest.mark.parametrize(
    ("ingestion_attempt_id", "expected"),
    [(None, StrictStreamReason.INGESTION_PROVENANCE), ("ingestion-1", None)],
    ids=["null", "matching"],
)
def test_pending_retrieve_recovery_requires_same_run_ingestion_provenance(
    tmp_path: Path, ingestion_attempt_id: str | None, expected: StrictStreamReason | None
) -> None:
    run_dir = write_causal_stream(tmp_path)
    retrieval = RetrievalRecord.model_validate_json(
        (run_dir / "retrievals.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(
        update={
            "recovery_attempt_id": "attempt-1",
            "ingestion_attempt_id": ingestion_attempt_id,
            "timestamp": NOW + timedelta(seconds=2),
        }
    )
    write_jsonl(run_dir / "retrievals.jsonl", [retrieval])
    error = ErrorRecord.model_validate_json(
        (run_dir / "errors.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"phase": "retrieve"})
    write_jsonl(run_dir / "errors.jsonl", [error])
    attempt = RecoveryAttemptRecord.model_validate_json(
        (run_dir / "recovery_attempts.jsonl").read_text(encoding="utf-8").strip()
    ).model_copy(update={"stage": "retrieve"})
    write_jsonl(run_dir / "recovery_attempts.jsonl", [attempt])
    # The strict reader must verify provenance before exposing this crash window to the runner,
    # not only when a later resolution happens to reference it.
    write_jsonl(run_dir / "error_resolutions.jsonl", [])
    if ingestion_attempt_id is not None:
        write_jsonl(
            run_dir / "conversations.jsonl",
            [
                ConversationIngestionRecord(
                    run_id="run-1",
                    conversation_id="conversation-1",
                    started_at=NOW,
                    finished_at=NOW,
                    n_sessions=1,
                    n_turns=1,
                    n_turns_succeeded=1,
                    n_turns_failed=0,
                    total_latency_ms=1,
                    avg_latency_per_turn_ms=1,
                    ingestion_attempt_id=ingestion_attempt_id,
                )
            ],
        )

    finding = StrictRunReader().finding(run_dir, "run-1")

    if expected is None and finding.reason_codes:
        raise AssertionError(finding)
    if expected is None:
        stream = StrictRunReader().read(run_dir, "run-1")
        if len(stream.recovery_pending) != 1:
            raise AssertionError(stream.recovery_pending)
    if expected is not None and finding.reason_codes != (expected,):
        raise AssertionError(finding)


def test_duplicate_ingestion_attempt_id_is_rejected(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    ingestion = ConversationIngestionRecord(
        run_id="run-1",
        conversation_id="conversation-1",
        started_at=NOW,
        finished_at=NOW,
        n_sessions=1,
        n_turns=1,
        n_turns_succeeded=1,
        n_turns_failed=0,
        total_latency_ms=1,
        avg_latency_per_turn_ms=1,
        ingestion_attempt_id="duplicate-ingestion",
    )
    write_jsonl(run_dir / "conversations.jsonl", [ingestion, ingestion])

    finding = StrictRunReader().finding(run_dir, "run-1")

    if finding.reason_codes != (StrictStreamReason.DUPLICATE_RECORD_ID,):
        raise AssertionError(finding)


def test_duplicate_ingestion_diagnostics_are_per_id_and_source_ordered(tmp_path: Path) -> None:
    run_dir = write_causal_stream(tmp_path)
    base = ConversationIngestionRecord(
        run_id="run-1",
        conversation_id="conversation-1",
        started_at=NOW,
        finished_at=NOW,
        n_sessions=1,
        n_turns=1,
        n_turns_succeeded=1,
        n_turns_failed=0,
        total_latency_ms=1,
        avg_latency_per_turn_ms=1,
        ingestion_attempt_id="ingestion-a",
    )
    write_jsonl(
        run_dir / "conversations.jsonl",
        [
            base,
            base.model_copy(update={"ingestion_attempt_id": "ingestion-b"}),
            base,
            base.model_copy(update={"ingestion_attempt_id": "ingestion-b"}),
        ],
    )

    finding = StrictRunReader().finding(run_dir, "run-1")

    expected = (
        ("conversations.jsonl", 3, StrictStreamReason.DUPLICATE_RECORD_ID, "ingestion-a"),
        ("conversations.jsonl", 4, StrictStreamReason.DUPLICATE_RECORD_ID, "ingestion-b"),
    )
    actual = tuple(
        (item.file_name, item.line_number, item.code, item.detail) for item in finding.diagnostics
    )
    if actual != expected:
        raise AssertionError(actual)
    if finding.reason_codes != (StrictStreamReason.DUPLICATE_RECORD_ID,):
        raise AssertionError(finding)


def test_strict_stream_reason_enum_is_complete_and_closed() -> None:
    if {reason.value for reason in StrictStreamReason} != EXPECTED_STRICT_STREAM_REASONS:
        raise AssertionError(tuple(StrictStreamReason))


@pytest.mark.parametrize(
    ("source_file", "target_file"),
    [("responses.jsonl", "retrievals.jsonl"), ("retrievals.jsonl", "responses.jsonl")],
)
def test_complete_known_record_in_wrong_file_is_file_record_type(
    tmp_path: Path, source_file: str, target_file: str
) -> None:
    run_dir = write_causal_stream(tmp_path)
    source = (run_dir / source_file).read_text(encoding="utf-8")
    (run_dir / target_file).write_text(source, encoding="utf-8")

    finding = StrictRunReader().finding(run_dir, "run-1")

    if StrictStreamReason.FILE_RECORD_TYPE not in finding.reason_codes:
        raise AssertionError(finding)


@pytest.mark.parametrize("target_file", ["retrievals.jsonl", "responses.jsonl"])
def test_shape_invalid_wrong_file_remains_record_validation(
    tmp_path: Path, target_file: str
) -> None:
    run_dir = write_causal_stream(tmp_path)
    (run_dir / target_file).write_text('{"record_type":"response"}\n', encoding="utf-8")

    finding = StrictRunReader().finding(run_dir, "run-1")

    if StrictStreamReason.RECORD_VALIDATION not in finding.reason_codes:
        raise AssertionError(finding)
    if StrictStreamReason.FILE_RECORD_TYPE in finding.reason_codes:
        raise AssertionError(finding)
