"""Fail-closed, immutable JSONL snapshots for run-stage indexing."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from khedron.errors import PersistenceError
from khedron.types import (
    ConversationIngestionRecord,
    ErrorRecord,
    ErrorResolutionRecord,
    FailurePattern,
    ForcedRecoveryEvent,
    Judgment,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunLifecycleEvent,
)


class HistoricalRunClassification(StrEnum):
    VALID = "valid"
    LEGACY_INELIGIBLE = "legacy-ineligible"
    CORRUPT = "corrupt"


class StrictStreamReason(StrEnum):
    MISSING_LIFECYCLE = "MISSING_LIFECYCLE"
    LIFECYCLE_PARSE = "LIFECYCLE_PARSE"
    LIFECYCLE_RUN_MISMATCH = "LIFECYCLE_RUN_MISMATCH"
    JSON_PARSE = "JSON_PARSE"
    RECORD_VALIDATION = "RECORD_VALIDATION"
    FILE_RECORD_TYPE = "FILE_RECORD_TYPE"
    EMPTY_MANIFEST = "EMPTY_MANIFEST"
    MULTIPLE_MANIFESTS = "MULTIPLE_MANIFESTS"
    MANIFEST_RUN_MISMATCH = "MANIFEST_RUN_MISMATCH"
    DUPLICATE_MANIFEST_QUESTION_ID = "DUPLICATE_MANIFEST_QUESTION_ID"
    STAGE_OUTSIDE_MANIFEST = "STAGE_OUTSIDE_MANIFEST"
    NEW_RECORD_WITHOUT_MANIFEST = "NEW_RECORD_WITHOUT_MANIFEST"
    DUPLICATE_RECORD_ID = "DUPLICATE_RECORD_ID"
    DUPLICATE_STAGE = "DUPLICATE_STAGE"
    ORPHAN_STAGE = "ORPHAN_STAGE"
    RUN_MISMATCH = "RUN_MISMATCH"
    QUESTION_MISMATCH = "QUESTION_MISMATCH"
    BROKEN_PARENT_REFERENCE = "BROKEN_PARENT_REFERENCE"
    ATTEMPT_ERROR_REFERENCE = "ATTEMPT_ERROR_REFERENCE"
    ATTEMPT_STAGE_MISMATCH = "ATTEMPT_STAGE_MISMATCH"
    RESOLUTION_ERROR_REFERENCE = "RESOLUTION_ERROR_REFERENCE"
    DUPLICATE_RESOLUTION_ERROR = "DUPLICATE_RESOLUTION_ERROR"
    RESOLUTION_ATTEMPT_REFERENCE = "RESOLUTION_ATTEMPT_REFERENCE"
    RESOLUTION_STAGE_REFERENCE = "RESOLUTION_STAGE_REFERENCE"
    RECOVERY_ATTEMPT_MISMATCH = "RECOVERY_ATTEMPT_MISMATCH"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    INGESTION_PROVENANCE = "INGESTION_PROVENANCE"
    UNREFERENCED_ATTEMPT = "UNREFERENCED_ATTEMPT"
    # Forced-recovery audit (ForcedRecoveryEvent, forced_recoveries.jsonl).
    FORCED_ERROR_REFERENCE = "FORCED_ERROR_REFERENCE"
    FORCED_ATTEMPT_REFERENCE = "FORCED_ATTEMPT_REFERENCE"
    FORCED_ON_RETRYABLE = "FORCED_ON_RETRYABLE"
    DUPLICATE_FORCED_AUTHORIZATION = "DUPLICATE_FORCED_AUTHORIZATION"
    UNAUTHORIZED_RECOVERY = "UNAUTHORIZED_RECOVERY"


@dataclass(frozen=True)
class StrictStreamDiagnostic:
    code: StrictStreamReason
    file_name: str
    line_number: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class HistoricalRunFinding:
    run_id: str
    classification: HistoricalRunClassification
    reason_codes: tuple[StrictStreamReason, ...] = ()
    diagnostics: tuple[StrictStreamDiagnostic, ...] = ()


@dataclass(frozen=True)
class RecoveryPending:
    """One validated crash window, before or after the recovered stage write."""

    attempt: RecoveryAttemptRecord
    error: ErrorRecord
    target: RetrievalRecord | Response | Judgment | None


@dataclass(frozen=True)
class RunStageStream:
    run_id: str
    lifecycle: tuple[RunLifecycleEvent, ...]
    plan: QuestionPlanRecord | None
    evaluations: tuple[QuestionEvaluationRecord, ...]
    retrievals: tuple[RetrievalRecord, ...]
    responses: tuple[Response, ...]
    judgments: tuple[Judgment, ...]
    errors: tuple[ErrorRecord, ...]
    attempts: tuple[RecoveryAttemptRecord, ...]
    resolutions: tuple[ErrorResolutionRecord, ...]
    ingestions: tuple[ConversationIngestionRecord, ...]
    failure_patterns: tuple[FailurePattern, ...]
    forced_recoveries: tuple[ForcedRecoveryEvent, ...] = ()
    recovery_pending: tuple[RecoveryPending, ...] = ()


_EVENTS: TypeAdapter[RunLifecycleEvent] = cast(
    TypeAdapter[RunLifecycleEvent], TypeAdapter(RunLifecycleEvent)
)


class StrictRunReader:
    """Read each run-local JSONL source once and reject structural corruption."""

    def read(self, run_dir: Path, run_id: str) -> RunStageStream:
        # advisory=False: the raising path deliberately omits the forced-recovery inverse
        # invariant (UNAUTHORIZED_RECOVERY) until it has been verified against real historical
        # fixtures. It therefore reports as a finding but never raises yet.
        stream, diagnostics = self._capture(run_dir, run_id, advisory=False)
        if diagnostics:
            normalized = _normalize(diagnostics)
            raise PersistenceError(
                "Strict run stream is corrupt", run_id=run_id, diagnostics=normalized
            )
        return stream

    def finding(self, run_dir: Path, run_id: str) -> HistoricalRunFinding:
        # advisory=True: the advisory classification path DOES apply the inverse invariant, so a
        # forged non-retryable recovery without an authorizing ForcedRecoveryEvent is surfaced
        # here before it is ever promoted into read()'s raising path.
        stream, diagnostics = self._capture(run_dir, run_id, advisory=True)
        normalized = _normalize(diagnostics)
        if normalized:
            return HistoricalRunFinding(
                run_id, HistoricalRunClassification.CORRUPT, _reason_codes(normalized), normalized
            )
        classification = (
            HistoricalRunClassification.VALID
            if stream.plan is not None
            else HistoricalRunClassification.LEGACY_INELIGIBLE
        )
        return HistoricalRunFinding(run_id, classification)

    def _capture(
        self, run_dir: Path, run_id: str, *, advisory: bool = False
    ) -> tuple[RunStageStream, list[StrictStreamDiagnostic]]:
        diagnostics: list[StrictStreamDiagnostic] = []
        self._record_lines: dict[tuple[str, str], int] = {}
        self._record_object_lines: dict[int, int] = {}
        self._invalid_parent_ids: dict[str, set[str]] = {
            "question_evaluations.jsonl": set(),
            "retrievals.jsonl": set(),
            "responses.jsonl": set(),
        }
        lifecycle = self._read(
            run_dir / "lifecycle.jsonl", _EVENTS, run_id, diagnostics, required=True, lifecycle=True
        )
        plan_rows = self._read(
            run_dir / "question_plan.jsonl",
            TypeAdapter(QuestionPlanRecord),
            run_id,
            diagnostics,
            manifest=True,
        )
        evaluations = self._read(
            run_dir / "question_evaluations.jsonl",
            TypeAdapter(QuestionEvaluationRecord),
            run_id,
            diagnostics,
        )
        retrievals = self._read(
            run_dir / "retrievals.jsonl", TypeAdapter(RetrievalRecord), run_id, diagnostics
        )
        responses = self._read(
            run_dir / "responses.jsonl", TypeAdapter(Response), run_id, diagnostics
        )
        judgments = self._read(
            run_dir / "judgments.jsonl", TypeAdapter(Judgment), run_id, diagnostics
        )
        errors = self._read(run_dir / "errors.jsonl", TypeAdapter(ErrorRecord), run_id, diagnostics)
        attempts = self._read(
            run_dir / "recovery_attempts.jsonl",
            TypeAdapter(RecoveryAttemptRecord),
            run_id,
            diagnostics,
        )
        resolutions = self._read(
            run_dir / "error_resolutions.jsonl",
            TypeAdapter(ErrorResolutionRecord),
            run_id,
            diagnostics,
        )
        ingestions = self._read(
            run_dir / "conversations.jsonl",
            TypeAdapter(ConversationIngestionRecord),
            run_id,
            diagnostics,
        )
        failure_patterns = self._read(
            run_dir / "failure_patterns.jsonl",
            TypeAdapter(FailurePattern),
            run_id,
            diagnostics,
        )
        forced_recoveries = self._read(
            run_dir / "forced_recoveries.jsonl",
            TypeAdapter(ForcedRecoveryEvent),
            run_id,
            diagnostics,
        )
        if (run_dir / "question_plan.jsonl").exists() and not plan_rows:
            diagnostics.append(_diag(StrictStreamReason.EMPTY_MANIFEST, "question_plan.jsonl"))
        if len(plan_rows) > 1:
            diagnostics.append(_diag(StrictStreamReason.MULTIPLE_MANIFESTS, "question_plan.jsonl"))
        plan = cast(QuestionPlanRecord | None, plan_rows[0] if len(plan_rows) == 1 else None)
        if plan is None and (attempts or resolutions or forced_recoveries):
            diagnostics.append(
                _diag(
                    StrictStreamReason.NEW_RECORD_WITHOUT_MANIFEST,
                    "recovery_attempts.jsonl"
                    if attempts
                    else "error_resolutions.jsonl"
                    if resolutions
                    else "forced_recoveries.jsonl",
                )
            )
        stream = RunStageStream(
            run_id,
            tuple(cast(list[RunLifecycleEvent], lifecycle)),
            plan,
            tuple(cast(list[QuestionEvaluationRecord], evaluations)),
            tuple(cast(list[RetrievalRecord], retrievals)),
            tuple(cast(list[Response], responses)),
            tuple(cast(list[Judgment], judgments)),
            tuple(cast(list[ErrorRecord], errors)),
            tuple(cast(list[RecoveryAttemptRecord], attempts)),
            tuple(cast(list[ErrorResolutionRecord], resolutions)),
            tuple(cast(list[ConversationIngestionRecord], ingestions)),
            tuple(cast(list[FailurePattern], failure_patterns)),
            forced_recoveries=tuple(cast(list[ForcedRecoveryEvent], forced_recoveries)),
        )
        stream = replace(
            stream, recovery_pending=self._validate(stream, diagnostics, advisory=advisory)
        )
        return stream, diagnostics

    def _read(
        self,
        path: Path,
        adapter: TypeAdapter[Any],
        run_id: str,
        diagnostics: list[StrictStreamDiagnostic],
        *,
        required: bool = False,
        lifecycle: bool = False,
        manifest: bool = False,
    ) -> list[Any]:
        name = path.name
        if not path.exists():
            if required:
                diagnostics.append(_diag(StrictStreamReason.MISSING_LIFECYCLE, name))
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            diagnostics.append(_diag(StrictStreamReason.RECORD_VALIDATION, name))
            return []
        records: list[Any] = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.LIFECYCLE_PARSE
                        if lifecycle
                        else StrictStreamReason.JSON_PARSE,
                        name,
                        number,
                    )
                )
                continue
            try:
                record = adapter.validate_json(line)
            except ValidationError:
                if isinstance(raw, dict):
                    self._remember_invalid_parent(name, cast(dict[str, Any], raw))
                diagnostics.append(
                    _diag(
                        StrictStreamReason.LIFECYCLE_PARSE
                        if lifecycle
                        else (
                            StrictStreamReason.FILE_RECORD_TYPE
                            if _matches_known_wrong_file_model(line, name)
                            else StrictStreamReason.RECORD_VALIDATION
                        ),
                        name,
                        number,
                    )
                )
                continue
            actual_run = getattr(record, "run_id", None)
            if actual_run != run_id:
                self._remember_invalid_record(name, record)
                code = (
                    StrictStreamReason.LIFECYCLE_RUN_MISMATCH
                    if lifecycle
                    else (
                        StrictStreamReason.MANIFEST_RUN_MISMATCH
                        if manifest
                        else StrictStreamReason.RUN_MISMATCH
                    )
                )
                diagnostics.append(
                    _diag(code, name, number, f"expected={run_id};actual={actual_run}")
                )
                continue
            self._remember_record_line(name, record, number)
            records.append(record)
        if required and not records and not any(d.file_name == name for d in diagnostics):
            diagnostics.append(_diag(StrictStreamReason.LIFECYCLE_PARSE, name))
        return records

    def _remember_invalid_parent(self, file_name: str, raw: dict[str, Any]) -> None:
        field = {
            "question_evaluations.jsonl": "question_evaluation_id",
            "retrievals.jsonl": "retrieval_id",
            "responses.jsonl": "response_id",
        }.get(file_name)
        if field is not None and isinstance(raw.get(field), str):
            self._invalid_parent_ids[file_name].add(cast(str, raw[field]))

    def _remember_invalid_record(self, file_name: str, record: Any) -> None:
        field = {
            "question_evaluations.jsonl": "question_evaluation_id",
            "retrievals.jsonl": "retrieval_id",
            "responses.jsonl": "response_id",
        }.get(file_name)
        identifier = getattr(record, field, None) if field is not None else None
        if isinstance(identifier, str):
            self._invalid_parent_ids[file_name].add(identifier)

    def _remember_record_line(self, file_name: str, record: Any, line: int) -> None:
        self._record_object_lines[id(record)] = line
        identifier = _record_identifier(record)
        if identifier is not None:
            self._record_lines[(file_name, identifier)] = line

    def _validate(
        self,
        stream: RunStageStream,
        diagnostics: list[StrictStreamDiagnostic],
        *,
        advisory: bool = False,
    ) -> tuple[RecoveryPending, ...]:
        if stream.plan and len(set(stream.plan.question_ids)) != len(stream.plan.question_ids):
            diagnostics.append(
                _diag(StrictStreamReason.DUPLICATE_MANIFEST_QUESTION_ID, "question_plan.jsonl")
            )
        self._duplicates(stream, diagnostics)
        evaluations = _first_by_id(stream.evaluations, lambda value: value.question_evaluation_id)
        retrievals = _first_by_id(stream.retrievals, lambda value: value.retrieval_id)
        responses = _first_by_id(stream.responses, lambda value: value.response_id)
        judgments = _first_by_id(stream.judgments, lambda value: value.judgment_id)
        errors = _first_by_id(stream.errors, lambda value: value.error_id)
        attempts = _first_by_id(stream.attempts, lambda value: value.attempt_id)
        ingestions = {
            x.ingestion_attempt_id: x
            for x in stream.ingestions
            if x.ingestion_attempt_id is not None
        }
        seen_ingestion_ids: set[str] = set()
        for record in stream.ingestions:
            ingestion_id = record.ingestion_attempt_id
            if ingestion_id is None:
                continue
            if ingestion_id in seen_ingestion_ids:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.DUPLICATE_RECORD_ID,
                        "conversations.jsonl",
                        self._record_object_lines.get(id(record)),
                        ingestion_id,
                    )
                )
            seen_ingestion_ids.add(ingestion_id)
        pattern_ids = [record.pattern_id for record in stream.failure_patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            diagnostics.append(
                _diag(StrictStreamReason.DUPLICATE_RECORD_ID, "failure_patterns.jsonl")
            )
        for attempt in stream.attempts:
            error = errors.get(attempt.error_id)
            if error is None:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.ATTEMPT_ERROR_REFERENCE,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
                continue
            if error.question_id != attempt.question_id or error.phase != attempt.stage:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.ATTEMPT_STAGE_MISMATCH,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
        planned = set(stream.plan.question_ids) if stream.plan else None
        for name, records in (
            ("question_evaluations.jsonl", stream.evaluations),
            ("retrievals.jsonl", stream.retrievals),
            ("responses.jsonl", stream.responses),
            ("judgments.jsonl", stream.judgments),
            ("errors.jsonl", stream.errors),
        ):
            for record in records:
                question_id = getattr(record, "question_id", None)
                if planned is not None and question_id is not None and question_id not in planned:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.STAGE_OUTSIDE_MANIFEST, name, detail=str(question_id)
                        )
                    )
        for record in stream.retrievals:
            parent = evaluations.get(record.question_evaluation_id)
            if parent is None:
                if (
                    record.question_evaluation_id
                    not in self._invalid_parent_ids["question_evaluations.jsonl"]
                ):
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.BROKEN_PARENT_REFERENCE,
                            "retrievals.jsonl",
                            detail=record.retrieval_id,
                        )
                    )
            elif parent.question_id != record.question_id:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.QUESTION_MISMATCH,
                        "retrievals.jsonl",
                        detail=record.retrieval_id,
                    )
                )
        for record in stream.responses:
            parent = retrievals.get(record.retrieval_id)
            if parent is None:
                if record.retrieval_id not in self._invalid_parent_ids["retrievals.jsonl"]:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.ORPHAN_STAGE,
                            "responses.jsonl",
                            detail=record.response_id,
                        )
                    )
            elif parent.question_id != record.question_id:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.QUESTION_MISMATCH,
                        "responses.jsonl",
                        detail=record.response_id,
                    )
                )
        for record in stream.judgments:
            parent = responses.get(record.response_id)
            if parent is None:
                if record.response_id not in self._invalid_parent_ids["responses.jsonl"]:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.ORPHAN_STAGE,
                            "judgments.jsonl",
                            detail=record.judgment_id,
                        )
                    )
            elif parent.question_id != record.question_id:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.QUESTION_MISMATCH,
                        "judgments.jsonl",
                        detail=record.judgment_id,
                    )
                )
        resolved_attempts: set[str] = set()
        resolved_errors: set[str] = set()
        stages: dict[str, RetrievalRecord | Response | Judgment] = {
            **retrievals,
            **responses,
            **judgments,
        }
        for resolution in stream.resolutions:
            error = errors.get(resolution.error_id)
            attempt = attempts.get(resolution.recovery_attempt_id)
            target = stages.get(resolution.resolved_by_stage_record_id)
            if error is None:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.RESOLUTION_ERROR_REFERENCE,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
                continue
            if attempt is None:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.RESOLUTION_ATTEMPT_REFERENCE,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
                continue
            if target is None:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.RESOLUTION_STAGE_REFERENCE,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
                continue
            if resolution.error_id in resolved_errors:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.DUPLICATE_RESOLUTION_ERROR,
                        "error_resolutions.jsonl",
                        detail=resolution.error_id,
                    )
                )
            resolved_errors.add(resolution.error_id)
            resolved_attempts.add(attempt.attempt_id)
            if attempt.error_id != error.error_id:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.ATTEMPT_ERROR_REFERENCE,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
            if any(
                (
                    error.question_id != resolution.question_id,
                    attempt.question_id != resolution.question_id,
                    error.question_id != attempt.question_id,
                    error.phase != attempt.stage,
                    attempt.stage != resolution.resolved_by_stage,
                )
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.ATTEMPT_STAGE_MISMATCH,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
            stage_name = (
                "retrieve"
                if isinstance(target, RetrievalRecord)
                else "generate"
                if isinstance(target, Response)
                else "judge"
            )
            if (
                target.question_id != resolution.question_id
                or stage_name != resolution.resolved_by_stage
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.RESOLUTION_STAGE_REFERENCE,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
            if getattr(target, "recovery_attempt_id", None) != attempt.attempt_id:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.RECOVERY_ATTEMPT_MISMATCH,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
            if not (
                error.timestamp <= attempt.timestamp <= target.timestamp <= resolution.timestamp
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.TEMPORAL_ORDER,
                        "error_resolutions.jsonl",
                        detail=resolution.resolution_id,
                    )
                )
            if isinstance(target, RetrievalRecord):
                evaluation = evaluations.get(target.question_evaluation_id)
                ingestion = (
                    ingestions.get(target.ingestion_attempt_id)
                    if target.ingestion_attempt_id is not None
                    else None
                )
                if (
                    evaluation is None
                    or ingestion is None
                    or ingestion.run_id != stream.run_id
                    or ingestion.conversation_id != evaluation.conversation_id
                ):
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.INGESTION_PROVENANCE,
                            "conversations.jsonl",
                            detail=resolution.resolution_id,
                        )
                    )
        self._validate_forced_recoveries(stream, diagnostics, errors, attempts, advisory=advisory)
        pending_candidates: dict[
            tuple[str, str, str],
            list[
                tuple[
                    RecoveryAttemptRecord, ErrorRecord, RetrievalRecord | Response | Judgment | None
                ]
            ],
        ] = {}
        for attempt in stream.attempts:
            if attempt.attempt_id in resolved_attempts:
                continue
            error = errors.get(attempt.error_id)
            target = next(
                (
                    stage
                    for stage in stages.values()
                    if getattr(stage, "recovery_attempt_id", None) == attempt.attempt_id
                ),
                None,
            )
            stage_name = (
                "retrieve"
                if isinstance(target, RetrievalRecord)
                else "generate"
                if isinstance(target, Response)
                else "judge"
                if isinstance(target, Judgment)
                else None
            )
            key = (attempt.error_id, attempt.question_id, attempt.stage)
            if (
                error is None
                or error.question_id != attempt.question_id
                or error.phase != attempt.stage
                or error.timestamp >= attempt.timestamp
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.UNREFERENCED_ATTEMPT,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
                continue
            if target is None:
                pending_candidates.setdefault(key, []).append((attempt, error, None))
                continue
            if (
                stage_name != attempt.stage
                or target.question_id != attempt.question_id
                or target.timestamp <= attempt.timestamp
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.UNREFERENCED_ATTEMPT,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )
                continue
            if isinstance(target, RetrievalRecord):
                evaluation = evaluations.get(target.question_evaluation_id)
                ingestion = (
                    ingestions.get(target.ingestion_attempt_id)
                    if target.ingestion_attempt_id is not None
                    else None
                )
                if (
                    evaluation is None
                    or ingestion is None
                    or ingestion.run_id != stream.run_id
                    or ingestion.conversation_id != evaluation.conversation_id
                ):
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.INGESTION_PROVENANCE,
                            "conversations.jsonl",
                            detail=attempt.attempt_id,
                        )
                    )
                    continue
            pending_candidates.setdefault(key, []).append((attempt, error, target))
        pending: list[RecoveryPending] = []
        for candidates in pending_candidates.values():
            ordered = sorted(
                candidates, key=lambda value: (value[0].timestamp, value[0].attempt_id)
            )
            if any(
                later[0].timestamp <= earlier[0].timestamp for earlier, later in pairwise(ordered)
            ) or any(target is not None for _, _, target in ordered[:-1]):
                for attempt, _, _ in ordered:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.UNREFERENCED_ATTEMPT,
                            "recovery_attempts.jsonl",
                            detail=attempt.attempt_id,
                        )
                    )
                continue
            attempt, error, target = ordered[-1]
            pending.append(RecoveryPending(attempt, error, target))
        return tuple(
            sorted(
                pending,
                key=lambda value: (
                    value.attempt.question_id,
                    value.attempt.stage,
                    value.attempt.timestamp,
                    value.attempt.attempt_id,
                ),
            )
        )

    def _validate_forced_recoveries(
        self,
        stream: RunStageStream,
        diagnostics: list[StrictStreamDiagnostic],
        errors: dict[str, ErrorRecord],
        attempts: dict[str, RecoveryAttemptRecord],
        *,
        advisory: bool,
    ) -> None:
        """Validate the forced-recovery audit trail, forward and (advisory) inverse.

        Forward (always enforced -- historical runs carry no forced_recoveries.jsonl, so this
        flags nothing pre-feature): every ForcedRecoveryEvent must reference a real non-retryable
        error of its question/stage, be stamped no earlier than that error, and, when its attempt
        exists, reference that attempt's error/question/stage and be stamped no later than it --
        the monotone chain ``error <= forced <= attempt`` the runner's write order produces. An
        absent attempt is the benign persisted-authorization/crashed crash window and is
        admissible, mirroring the existing attempt-written/target-absent tolerance.

        Inverse (ADVISORY ONLY): a RecoveryAttemptRecord against a non-retryable error can only
        have come from a forced recovery (the automatic path dispatches only ``retryable is
        True``), so it must have an authorizing ForcedRecoveryEvent; its absence is
        UNAUTHORIZED_RECOVERY. This closes the forgery gap where the existing attempt loop never
        checks ``retryable``. It is wired into the advisory finding() path ONLY, never the raising
        read() path, until it has been verified against real historical fixtures. PROMOTE TO
        read() (drop the ``advisory`` gate here) only once a fixture sweep confirms no real
        historical run carries an attempt against a non-retryable error.
        """
        seen_attempt_authorizations: set[str] = set()
        for forced in stream.forced_recoveries:
            error = errors.get(forced.error_id)
            if (
                error is None
                or error.question_id != forced.question_id
                or error.phase != forced.stage
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.FORCED_ERROR_REFERENCE,
                        "forced_recoveries.jsonl",
                        detail=forced.forced_recovery_id,
                    )
                )
            else:
                if error.retryable is True:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.FORCED_ON_RETRYABLE,
                            "forced_recoveries.jsonl",
                            detail=forced.forced_recovery_id,
                        )
                    )
                if error.timestamp > forced.timestamp:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.TEMPORAL_ORDER,
                            "forced_recoveries.jsonl",
                            detail=forced.forced_recovery_id,
                        )
                    )
            attempt = attempts.get(forced.attempt_id)
            if attempt is not None:
                if (
                    attempt.error_id != forced.error_id
                    or attempt.question_id != forced.question_id
                    or attempt.stage != forced.stage
                ):
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.FORCED_ATTEMPT_REFERENCE,
                            "forced_recoveries.jsonl",
                            detail=forced.forced_recovery_id,
                        )
                    )
                if forced.timestamp > attempt.timestamp:
                    diagnostics.append(
                        _diag(
                            StrictStreamReason.TEMPORAL_ORDER,
                            "forced_recoveries.jsonl",
                            detail=forced.forced_recovery_id,
                        )
                    )
            # Absent attempt: admissible crash window (authorization persisted, process died before
            # the attempt). Never counted as a consummated forced recovery.
            if forced.attempt_id in seen_attempt_authorizations:
                diagnostics.append(
                    _diag(
                        StrictStreamReason.DUPLICATE_FORCED_AUTHORIZATION,
                        "forced_recoveries.jsonl",
                        detail=forced.attempt_id,
                    )
                )
            seen_attempt_authorizations.add(forced.attempt_id)

        if not advisory:
            return
        forced_by_attempt = {forced.attempt_id: forced for forced in stream.forced_recoveries}
        for attempt in stream.attempts:
            error = errors.get(attempt.error_id)
            if error is None or error.retryable is True:
                # Retryable (or danglingly-referenced) attempts belong to the automatic path and
                # need no authorization; only a non-retryable attempt requires an operator override.
                continue
            forced = forced_by_attempt.get(attempt.attempt_id)
            if (
                forced is None
                or forced.run_id != attempt.run_id
                or forced.question_id != attempt.question_id
                or forced.error_id != attempt.error_id
                or forced.stage != attempt.stage
            ):
                diagnostics.append(
                    _diag(
                        StrictStreamReason.UNAUTHORIZED_RECOVERY,
                        "recovery_attempts.jsonl",
                        detail=attempt.attempt_id,
                    )
                )

    def _duplicates(
        self, stream: RunStageStream, diagnostics: list[StrictStreamDiagnostic]
    ) -> None:
        groups: tuple[tuple[str, tuple[Any, ...], str], ...] = (
            ("retrievals.jsonl", stream.retrievals, "retrieval_id"),
            ("responses.jsonl", stream.responses, "response_id"),
            ("judgments.jsonl", stream.judgments, "judgment_id"),
            ("errors.jsonl", stream.errors, "error_id"),
            ("recovery_attempts.jsonl", stream.attempts, "attempt_id"),
            ("error_resolutions.jsonl", stream.resolutions, "resolution_id"),
            ("forced_recoveries.jsonl", stream.forced_recoveries, "forced_recovery_id"),
        )
        for name, records, field in groups:
            self._append_duplicate_diagnostics(name, records, field, diagnostics)
        self._append_duplicate_diagnostics(
            "question_evaluations.jsonl",
            stream.evaluations,
            "question_evaluation_id",
            diagnostics,
            code=StrictStreamReason.DUPLICATE_STAGE,
        )
        self._append_duplicate_values(
            "question_evaluations.jsonl",
            stream.evaluations,
            "question_id",
            diagnostics,
        )
        for name, records in (
            ("retrievals.jsonl", stream.retrievals),
            ("responses.jsonl", stream.responses),
            ("judgments.jsonl", stream.judgments),
        ):
            self._append_duplicate_values(name, records, "question_id", diagnostics)

    def _append_duplicate_diagnostics(
        self,
        file_name: str,
        records: tuple[Any, ...],
        field: str,
        diagnostics: list[StrictStreamDiagnostic],
        *,
        code: StrictStreamReason = StrictStreamReason.DUPLICATE_RECORD_ID,
    ) -> None:
        self._append_duplicate_values(file_name, records, field, diagnostics, code=code)

    def _append_duplicate_values(
        self,
        file_name: str,
        records: tuple[Any, ...],
        field: str,
        diagnostics: list[StrictStreamDiagnostic],
        *,
        code: StrictStreamReason = StrictStreamReason.DUPLICATE_STAGE,
    ) -> None:
        seen: set[str] = set()
        for record in records:
            value = cast(str, getattr(record, field))
            if value in seen:
                identifier = _record_identifier(record) or value
                diagnostics.append(
                    _diag(code, file_name, self._record_lines.get((file_name, identifier)), value)
                )
            seen.add(value)


def preflight_historical_runs(results_dir: Path) -> tuple[HistoricalRunFinding, ...]:
    """Classify every immediate run directory without opening SQLite."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return ()
    try:
        entries = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    except OSError as exc:
        raise PersistenceError("Run directory cannot be read", path=str(runs_dir)) from exc
    reader = StrictRunReader()
    return tuple(reader.finding(path, path.name) for path in entries)


def _diag(
    code: StrictStreamReason, file_name: str, line_number: int | None = None, detail: str = ""
) -> StrictStreamDiagnostic:
    return StrictStreamDiagnostic(code, file_name, line_number, detail)


def _record_identifier(record: Any) -> str | None:
    for field in (
        "question_evaluation_id",
        "retrieval_id",
        "response_id",
        "judgment_id",
        "error_id",
        "attempt_id",
        "resolution_id",
        "forced_recovery_id",
        "pattern_id",
        "ingestion_attempt_id",
    ):
        value = getattr(record, field, None)
        if isinstance(value, str):
            return value
    return None


_FILE_MODELS: dict[str, tuple[type[BaseModel], ...]] = {
    "question_plan.jsonl": (QuestionPlanRecord,),
    "question_evaluations.jsonl": (QuestionEvaluationRecord,),
    "retrievals.jsonl": (RetrievalRecord,),
    "responses.jsonl": (Response,),
    "judgments.jsonl": (Judgment,),
    "errors.jsonl": (ErrorRecord,),
    "recovery_attempts.jsonl": (RecoveryAttemptRecord,),
    "error_resolutions.jsonl": (ErrorResolutionRecord,),
    "forced_recoveries.jsonl": (ForcedRecoveryEvent,),
    "conversations.jsonl": (ConversationIngestionRecord,),
    "failure_patterns.jsonl": (FailurePattern,),
}


def _matches_known_wrong_file_model(line: str, file_name: str) -> bool:
    """Identify only a complete persisted record placed in the wrong exact file."""
    for candidate_file, models in _FILE_MODELS.items():
        if candidate_file == file_name:
            continue
        for model in models:
            try:
                model.model_validate_json(line)
            except ValidationError:
                continue
            return True
    return False


def _first_by_id(records: tuple[Any, ...], get_id: Callable[[Any], str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        result.setdefault(get_id(record), record)
    return result


def _normalize(diagnostics: list[StrictStreamDiagnostic]) -> tuple[StrictStreamDiagnostic, ...]:
    return tuple(
        sorted(
            set(diagnostics),
            key=lambda d: (d.file_name, d.line_number or 0, d.code.value, d.detail),
        )
    )


def _reason_codes(
    diagnostics: tuple[StrictStreamDiagnostic, ...],
) -> tuple[StrictStreamReason, ...]:
    return tuple(dict.fromkeys(diagnostic.code for diagnostic in diagnostics))
