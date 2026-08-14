from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, cast

import structlog
from pydantic import BaseModel, TypeAdapter

from khedron.errors import PersistenceError
from khedron.persistence.jsonl_writer import JsonlWriter
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.persistence.stage_reader import RunStageStream, StrictRunReader
from khedron.types import (
    AggregateScore,
    APICallRecord,
    ConversationIngestionRecord,
    ConversationProcessedEvent,
    ErrorRecord,
    ErrorResolutionRecord,
    ExperimentCompletedEvent,
    ExperimentFailedEvent,
    ExperimentResult,
    ExperimentStartedEvent,
    FailurePattern,
    ForcedRecoveryEvent,
    Judgment,
    JudgmentVerdict,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    QuestionRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunFilters,
    RunLifecycleEvent,
    RunResumedEvent,
    RunStartedEvent,
    RunStatus,
    RunSummary,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteLifecycleEvent,
    SuiteStartedEvent,
    SuiteStatus,
)

__all__ = ["RunRepository"]

ModelT = TypeVar("ModelT", bound=BaseModel)
RunState: TypeAlias = Literal["running", "completed", "failed", "partial"]
SuiteState: TypeAlias = Literal["running", "completed", "failed", "partial"]
RunErrorPhase: TypeAlias = Literal[
    "initialization",
    "ingest",
    "retrieve",
    "generate",
    "judge",
    "analysis",
]
QuestionErrorPhase: TypeAlias = Literal["retrieve", "generate", "judge"]

_RUN_EVENT_ADAPTER: TypeAdapter[RunLifecycleEvent] = TypeAdapter(RunLifecycleEvent)
_SUITE_EVENT_ADAPTER: TypeAdapter[SuiteLifecycleEvent] = TypeAdapter(SuiteLifecycleEvent)
_LOGGER = structlog.get_logger(__name__)


class _RepositorySqliteIndexer(SqliteIndexer):
    """SqliteIndexer variant that honors RunRepository's explicit results root."""

    def __init__(self, db_path: Path, results_root: Path) -> None:
        super().__init__(db_path)
        self._results_root = results_root

    def index_suite(self, suite_id: str) -> None:
        self._index_suite(suite_id)


class RunRepository:
    """Persistence boundary over JSONL source files and the SQLite query layer."""

    def __init__(self, results_dir: Path, sqlite_path: Path) -> None:
        self._results_dir = Path(results_dir)
        self._sqlite_path = Path(sqlite_path)
        self._sqlite_indexer = _RepositorySqliteIndexer(
            db_path=self._sqlite_path,
            results_root=self._results_dir,
        )
        self._writers: dict[Path, JsonlWriter] = {}
        self._log = _LOGGER.bind(sqlite_path=str(self._sqlite_path))
        # index_run does a full read-JSONL then delete+reinsert of the run's rows. The
        # runner evaluates questions concurrently and re-indexes after each one, so two
        # index passes for the same run could overlap and let an older JSONL snapshot
        # commit after a newer one, dropping rows and leaving SQLite an unfaithful index
        # of the JSONL source of truth. Serialize all indexing through one lock.
        self._index_lock = asyncio.Lock()

    async def append_suite_event(self, event: SuiteLifecycleEvent) -> None:
        await self._write_jsonl(
            self._suite_file(event.suite_id, "lifecycle.jsonl"),
            event,
        )
        await self._index_suite_best_effort(event.suite_id)

    async def append_run_event(self, event: RunLifecycleEvent) -> None:
        await self._write_jsonl(
            self._run_file(event.run_id, "lifecycle.jsonl"),
            event,
        )
        await self._index_run_best_effort(event.run_id)

    async def append_experiment_result(self, result: ExperimentResult) -> None:
        await self._write_jsonl(
            self._suite_file(result.suite_id, "experiments.jsonl"),
            result,
        )
        await self._index_experiment_result_best_effort(result)

    async def append_question_evaluation(self, record: QuestionEvaluationRecord) -> None:
        await self._append_run_record(record.run_id, "question_evaluations.jsonl", record)

    async def append_question_plan(self, record: QuestionPlanRecord) -> None:
        """Persist the immutable planned-question set before any question dispatch."""
        await self._append_run_record(record.run_id, "question_plan.jsonl", record)

    async def append_conversation_ingestion(self, record: ConversationIngestionRecord) -> None:
        await self._write_jsonl(self._run_file(record.run_id, "conversations.jsonl"), record)

    async def append_retrieval(self, retrieval: RetrievalRecord) -> None:
        await self._append_run_record(retrieval.run_id, "retrievals.jsonl", retrieval)

    async def append_response(self, response: Response) -> None:
        await self._append_run_record(response.run_id, "responses.jsonl", response)

    async def append_judgment(self, judgment: Judgment) -> None:
        await self._append_run_record(judgment.run_id, "judgments.jsonl", judgment)

    async def append_api_call(self, record: APICallRecord) -> None:
        await self._write_jsonl(self._run_file(record.run_id, "api_calls.jsonl"), record)

    async def append_error(self, error: ErrorRecord) -> None:
        await self._append_run_record(error.run_id, "errors.jsonl", error)

    async def append_recovery_attempt(self, record: RecoveryAttemptRecord) -> None:
        """Persist a recovery budget unit before its external stage dispatch."""
        await self._append_run_record(record.run_id, "recovery_attempts.jsonl", record)

    async def append_error_resolution(self, record: ErrorResolutionRecord) -> None:
        """Persist the causal evidence that one question-stage error was repaired."""
        await self._append_run_record(record.run_id, "error_resolutions.jsonl", record)

    async def append_forced_recovery(self, record: ForcedRecoveryEvent) -> None:
        """Persist an operator's authorization to force-recover a non-retryable error.

        Mirrors ``append_recovery_attempt``. Written to its own run-local JSONL source
        (``forced_recoveries.jsonl``); ``index_run`` is a no-op for it, since this design ships
        migration-free with no SQLite projection.
        """
        await self._append_run_record(record.run_id, "forced_recoveries.jsonl", record)

    def get_strict_stage_stream(self, run_id: str) -> RunStageStream:
        """Return the immutable JSONL snapshot used for recovery decisions, never SQLite."""
        return StrictRunReader().read(self._run_file(run_id, "lifecycle.jsonl").parent, run_id)

    def get_run_status(self, run_id: str) -> RunStatus:
        events = sorted(self._read_run_events(run_id), key=lambda event: event.sequence_number)
        return _reduce_run_status(run_id, events)

    def get_suite_status(self, suite_id: str) -> SuiteStatus:
        events = sorted(self._read_suite_events(suite_id), key=lambda event: event.sequence_number)
        return _reduce_suite_status(suite_id, events)

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        for result in self.list_experiment_results():
            if result.experiment_id == experiment_id:
                return result
        raise PersistenceError("Experiment result does not exist", experiment_id=experiment_id)

    def get_experiment_result_for_run(self, run_id: str) -> ExperimentResult | None:
        started = self.get_run_started_event(run_id)
        for result in self.list_experiment_results(started.suite_id):
            if result.experiment_id == started.experiment_id:
                return result
        return None

    def list_experiment_results(self, suite_id: str | None = None) -> list[ExperimentResult]:
        if suite_id is not None:
            results = self._read_records(
                self._suite_file(suite_id, "experiments.jsonl"),
                ExperimentResult,
            )
        else:
            results = [
                result
                for suite_dir in self._iter_suite_dirs()
                for result in self._read_records(suite_dir / "experiments.jsonl", ExperimentResult)
            ]
        return sorted(
            results,
            key=lambda result: (result.suite_id, result.experiment_name, result.experiment_id),
        )

    def get_run_started_event(self, run_id: str) -> RunStartedEvent:
        for event in reversed(self.list_run_events(run_id)):
            if isinstance(event, RunStartedEvent):
                return event
        raise PersistenceError("Run lifecycle has no run_started event", run_id=run_id)

    def list_run_events(self, run_id: str) -> list[RunLifecycleEvent]:
        return sorted(self._read_run_events(run_id), key=lambda event: event.sequence_number)

    def list_suite_events(self, suite_id: str) -> list[SuiteLifecycleEvent]:
        return sorted(self._read_suite_events(suite_id), key=lambda event: event.sequence_number)

    def get_conversation_ingestions_for_run(
        self,
        run_id: str,
    ) -> list[ConversationIngestionRecord]:
        return self._read_records(
            self._run_file(run_id, "conversations.jsonl"),
            ConversationIngestionRecord,
        )

    def get_evaluated_question_ids(self, run_id: str) -> set[str]:
        """Every question the run has written an evaluation record for.

        Read from the append-only stream rather than from `question_results`, which materialises
        only questions that reached a verdict. The distinction is the whole point here: a question
        killed between its evaluation and its judgment is absent from the projection and present in
        the file, and it is precisely that question a resumed run must not ask again -- the index
        is unique on (run_id, question_id), so re-asking collides rather than overwriting.
        """
        return {
            record.question_id
            for record in self._read_records(
                self._run_file(run_id, "question_evaluations.jsonl"),
                QuestionEvaluationRecord,
            )
        }

    def get_responses_for_run(self, run_id: str) -> list[Response]:
        """Every answer the run generated, in the order it produced them."""
        return self._read_records(self._run_file(run_id, "responses.jsonl"), Response)

    def get_retrievals_for_run(self, run_id: str) -> list[RetrievalRecord]:
        """Every retrieval the run performed, in the order it performed them."""
        return self._read_records(self._run_file(run_id, "retrievals.jsonl"), RetrievalRecord)

    def get_api_calls_for_run(self, run_id: str) -> list[APICallRecord]:
        return self._read_records(self._run_file(run_id, "api_calls.jsonl"), APICallRecord)

    def get_errors_for_run(self, run_id: str) -> list[ErrorRecord]:
        return self._read_records(self._run_file(run_id, "errors.jsonl"), ErrorRecord)

    def get_failure_patterns_for_run(self, run_id: str) -> list[FailurePattern]:
        return self._read_records(
            self._run_file(run_id, "failure_patterns.jsonl"),
            FailurePattern,
        )

    def list_runs(self, filters: RunFilters | None = None) -> list[RunSummary]:
        query = "SELECT * FROM runs"
        clauses: list[str] = []
        params: list[object] = []
        run_filters = filters or RunFilters()
        for field_name in (
            "suite_id",
            "experiment_id",
            "experiment_name",
            "status",
            "provider_type",
            "benchmark_type",
            "answer_model_id",
            "judge_model_id",
        ):
            value = getattr(run_filters, field_name)
            if value is not None:
                clauses.append(f"{field_name} = ?")
                params.append(value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC"
        return [_run_summary_from_row(row) for row in self._fetch_all(query, tuple(params))]

    def get_question_record(self, run_id: str, question_id: str) -> QuestionRecord:
        row = self._fetch_one_required(
            """
            SELECT *
            FROM question_results
            WHERE run_id = ? AND question_id = ?
            """,
            (run_id, question_id),
            "Question record does not exist",
            run_id=run_id,
            question_id=question_id,
        )
        return self._hydrate_question_record(_question_record_from_row(row))

    def get_questions_for_run(self, run_id: str) -> list[QuestionRecord]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM question_results
            WHERE run_id = ?
            ORDER BY question_id
            """,
            (run_id,),
        )
        return [_question_record_from_row(row) for row in rows]

    def get_retrieval(self, retrieval_id: str) -> RetrievalRecord:
        run_id = self._lookup_run_id(
            "SELECT run_id FROM retrievals WHERE retrieval_id = ?",
            (retrieval_id,),
        )
        record = self._find_retrieval(retrieval_id, run_id)
        if record is None and run_id is not None:
            record = self._find_retrieval(retrieval_id, None)
        if record is None:
            raise PersistenceError("Retrieval record does not exist", retrieval_id=retrieval_id)
        return record

    def get_response(self, response_id: str) -> Response:
        run_id = self._lookup_run_id(
            "SELECT run_id FROM question_results WHERE response_id = ?",
            (response_id,),
        )
        record = self._find_response(response_id, run_id)
        if record is None and run_id is not None:
            record = self._find_response(response_id, None)
        if record is None:
            raise PersistenceError("Response record does not exist", response_id=response_id)
        return record

    def get_judgment(self, judgment_id: str) -> Judgment:
        run_id = self._lookup_run_id(
            "SELECT run_id FROM question_results WHERE judgment_id = ?",
            (judgment_id,),
        )
        record = self._find_judgment(judgment_id, run_id)
        if record is None and run_id is not None:
            record = self._find_judgment(judgment_id, None)
        if record is None:
            raise PersistenceError("Judgment record does not exist", judgment_id=judgment_id)
        return record

    def get_failed_questions(
        self,
        run_id: str,
        category: str | None = None,
    ) -> list[QuestionRecord]:
        if category is None:
            rows = self._fetch_all(
                """
                SELECT *
                FROM question_results
                WHERE run_id = ? AND verdict != ?
                ORDER BY question_id
                """,
                (run_id, JudgmentVerdict.CORRECT.value),
            )
        else:
            rows = self._fetch_all(
                """
                SELECT *
                FROM question_results
                WHERE run_id = ? AND verdict != ? AND category = ?
                ORDER BY question_id
                """,
                (run_id, JudgmentVerdict.CORRECT.value, category),
            )
        return [_question_record_from_row(row) for row in rows]

    async def rebuild_sqlite(self) -> None:
        await asyncio.to_thread(self._rebuild_sqlite)

    async def _append_run_record(
        self,
        run_id: str,
        file_name: str,
        record: BaseModel,
    ) -> None:
        await self._write_jsonl(self._run_file(run_id, file_name), record)
        await self._index_run_best_effort(run_id)

    async def _write_jsonl(self, path: Path, record: BaseModel) -> None:
        await self._writer_for(path).write(record)

    def _writer_for(self, path: Path) -> JsonlWriter:
        writer = self._writers.get(path)
        if writer is None:
            writer = JsonlWriter(path, buffer_size=1)
            self._writers[path] = writer
        return writer

    async def _index_run_best_effort(self, run_id: str) -> None:
        try:
            async with self._index_lock:
                await asyncio.to_thread(self._sqlite_indexer.index_run, run_id)
        except Exception as exc:
            try:
                self._log.warning(
                    "sqlite_indexing_failed",
                    operation="index_run",
                    run_id=run_id,
                    error=str(exc),
                    exc_info=True,
                )
            except Exception:  # noqa: S110 - deliberately silent; see below
                # A best-effort path that can kill the process is not best-effort. Rendering this
                # warning to a cp1252 console raised UnicodeEncodeError on content the corpus
                # legitimately contains, and that exception propagated out of an indexing failure
                # the caller was designed to survive -- ending a run over an unprintable character.
                # The CLI now forces UTF-8, but the library must hold on its own: any process using
                # it, test or script, meets the same console.
                pass

    async def _index_suite_best_effort(self, suite_id: str) -> None:
        try:
            async with self._index_lock:
                await asyncio.to_thread(self._sqlite_indexer.index_suite, suite_id)
        except Exception as exc:
            self._log.warning(
                "sqlite_indexing_failed",
                operation="index_suite",
                suite_id=suite_id,
                error=str(exc),
                exc_info=True,
            )

    async def _index_experiment_result_best_effort(self, result: ExperimentResult) -> None:
        try:
            async with self._index_lock:
                await asyncio.to_thread(self._index_experiment_result, result)
        except Exception as exc:
            self._log.warning(
                "sqlite_indexing_failed",
                operation="index_experiment_result",
                suite_id=result.suite_id,
                experiment_id=result.experiment_id,
                error=str(exc),
                exc_info=True,
            )

    def _index_experiment_result(self, result: ExperimentResult) -> None:
        self._sqlite_indexer.initialize()
        standard = result.aggregate_overall_standard
        audited = result.aggregate_overall_audited
        standard_score = _aggregate_score(standard)
        audited_score = _aggregate_score(audited)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO experiment_results (
                        experiment_id, suite_id, experiment_name, n_runs,
                        overall_standard_pooled_score, overall_standard_ci_low,
                        overall_standard_ci_high, overall_standard_mean,
                        overall_standard_stddev, overall_audited_pooled_score,
                        overall_audited_ci_low, overall_audited_ci_high,
                        overall_audited_mean, overall_audited_stddev,
                        total_cost_usd, config_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(experiment_id) DO UPDATE SET
                        suite_id = excluded.suite_id,
                        experiment_name = excluded.experiment_name,
                        n_runs = excluded.n_runs,
                        overall_standard_pooled_score = excluded.overall_standard_pooled_score,
                        overall_standard_ci_low = excluded.overall_standard_ci_low,
                        overall_standard_ci_high = excluded.overall_standard_ci_high,
                        overall_standard_mean = excluded.overall_standard_mean,
                        overall_standard_stddev = excluded.overall_standard_stddev,
                        overall_audited_pooled_score = excluded.overall_audited_pooled_score,
                        overall_audited_ci_low = excluded.overall_audited_ci_low,
                        overall_audited_ci_high = excluded.overall_audited_ci_high,
                        overall_audited_mean = excluded.overall_audited_mean,
                        overall_audited_stddev = excluded.overall_audited_stddev,
                        total_cost_usd = excluded.total_cost_usd,
                        config_json = excluded.config_json
                    """,
                    (
                        result.experiment_id,
                        result.suite_id,
                        result.experiment_name,
                        result.n_runs,
                        _score_point(standard_score),
                        _score_low(standard_score),
                        _score_high(standard_score),
                        standard.mean if standard is not None else None,
                        standard.stddev if standard is not None else None,
                        _score_point(audited_score),
                        _score_low(audited_score),
                        _score_high(audited_score),
                        audited.mean if audited is not None else None,
                        audited.stddev if audited is not None else None,
                        result.total_cost_usd,
                        result.config.model_dump_json(),
                    ),
                )

    def _rebuild_sqlite(self) -> None:
        self._sqlite_indexer.rebuild_all()

    def _hydrate_question_record(self, record: QuestionRecord) -> QuestionRecord:
        updates: dict[str, object] = {}
        if record.retrieval_id is not None:
            retrieval = self.get_retrieval(record.retrieval_id)
            updates["retrieval_timestamp"] = retrieval.timestamp
            updates["retrieval_latency_ms"] = retrieval.retrieval_latency_ms
            updates["n_memories_retrieved"] = retrieval.n_returned
            updates["retrieved_memory_ids"] = [memory.memory_id for memory in retrieval.memories]
        if record.response_id is not None:
            response = self.get_response(record.response_id)
            updates["generation_timestamp"] = response.timestamp
            updates["generation_latency_ms"] = response.latency_ms
            updates["generation_cost_usd"] = response.cost_usd
        if record.judgment_id is not None:
            judgment = self.get_judgment(record.judgment_id)
            updates["judgment_timestamp"] = judgment.timestamp
            updates["judgment_latency_ms"] = judgment.latency_ms
            updates["judgment_cost_usd"] = judgment.cost_usd
        if not updates:
            return record
        return record.model_copy(update=updates)

    def _find_retrieval(
        self,
        retrieval_id: str,
        run_id: str | None,
    ) -> RetrievalRecord | None:
        return self._find_run_record(
            "retrievals.jsonl",
            RetrievalRecord,
            lambda record: record.retrieval_id == retrieval_id,
            run_id,
        )

    def _find_response(self, response_id: str, run_id: str | None) -> Response | None:
        return self._find_run_record(
            "responses.jsonl",
            Response,
            lambda record: record.response_id == response_id,
            run_id,
        )

    def _find_judgment(self, judgment_id: str, run_id: str | None) -> Judgment | None:
        return self._find_run_record(
            "judgments.jsonl",
            Judgment,
            lambda record: record.judgment_id == judgment_id,
            run_id,
        )

    def _find_run_record(
        self,
        file_name: str,
        model_cls: type[ModelT],
        predicate: Callable[[ModelT], bool],
        run_id: str | None,
    ) -> ModelT | None:
        if run_id is not None:
            return self._find_record_in_path(
                self._run_file(run_id, file_name),
                model_cls,
                predicate,
            )
        for run_dir in self._iter_run_dirs():
            record = self._find_record_in_path(run_dir / file_name, model_cls, predicate)
            if record is not None:
                return record
        return None

    def _find_record_in_path(
        self,
        path: Path,
        model_cls: type[ModelT],
        predicate: Callable[[ModelT], bool],
    ) -> ModelT | None:
        for record in self._read_records(path, model_cls):
            if predicate(record):
                return record
        return None

    def _read_run_events(self, run_id: str) -> list[RunLifecycleEvent]:
        path = self._run_file(run_id, "lifecycle.jsonl")
        if not path.exists():
            raise PersistenceError("Run lifecycle file does not exist", path=str(path))
        return _read_union_jsonl(path, _RUN_EVENT_ADAPTER)

    def _read_suite_events(self, suite_id: str) -> list[SuiteLifecycleEvent]:
        path = self._suite_file(suite_id, "lifecycle.jsonl")
        if not path.exists():
            raise PersistenceError("Suite lifecycle file does not exist", path=str(path))
        return _read_union_jsonl(path, _SUITE_EVENT_ADAPTER)

    def _read_records(self, path: Path, model_cls: type[ModelT]) -> list[ModelT]:
        if not path.exists():
            return []
        records: list[ModelT] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    records.append(model_cls.model_validate_json(stripped))
        return records

    def _fetch_one_required(
        self,
        query: str,
        params: tuple[object, ...],
        missing_message: str,
        **context: object,
    ) -> sqlite3.Row:
        rows = self._fetch_all(query, params)
        if not rows:
            raise PersistenceError(missing_message, **context)
        return rows[0]

    def _fetch_all(self, query: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        if not self._sqlite_path.exists():
            raise PersistenceError(
                "SQLite database does not exist",
                sqlite_path=str(self._sqlite_path),
            )
        try:
            with closing(self._connect()) as connection:
                return list(connection.execute(query, params).fetchall())
        except sqlite3.Error as exc:
            raise PersistenceError(
                "SQLite read failed",
                sqlite_path=str(self._sqlite_path),
                error=str(exc),
            ) from exc

    def _lookup_run_id(self, query: str, params: tuple[object, ...]) -> str | None:
        if not self._sqlite_path.exists():
            return None
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(query, params).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        value = cast(object, row["run_id"])
        return value if isinstance(value, str) else None

    def _connect(self) -> sqlite3.Connection:
        # The 30s busy timeout makes the per-question read (get_question_record) wait
        # for the indexer's writes instead of failing with "database is locked". WAL is
        # deliberately NOT set here: this connection also serves dashboard and report
        # reads, whose contract forbids mutating persistence, and it may open a
        # read-only or pre-existing DB. The indexer (the creating writer) enables WAL on
        # the file, and reads through this connection then benefit from it for free.
        connection = sqlite3.connect(self._sqlite_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _iter_run_dirs(self) -> Iterable[Path]:
        runs_dir = self._results_dir / "runs"
        if not runs_dir.exists():
            return ()
        return tuple(sorted(path for path in runs_dir.iterdir() if path.is_dir()))

    def _iter_suite_dirs(self) -> Iterable[Path]:
        suites_dir = self._results_dir / "suites"
        if not suites_dir.exists():
            return ()
        return tuple(sorted(path for path in suites_dir.iterdir() if path.is_dir()))

    def _suite_file(self, suite_id: str, file_name: str) -> Path:
        return self._results_dir / "suites" / suite_id / file_name

    def _run_file(self, run_id: str, file_name: str) -> Path:
        return self._results_dir / "runs" / run_id / file_name


def _reduce_run_status(run_id: str, events: Iterable[RunLifecycleEvent]) -> RunStatus:
    started: RunStartedEvent | None = None
    status: RunState = "running"
    finished_at = None
    n_conversations_processed = 0
    n_questions_attempted = 0
    n_questions_succeeded = 0
    n_questions_errored = 0
    overall_score_standard: ScoreWithCI | None = None
    overall_score_audited: ScoreWithCI | None = None
    by_category_standard: dict[str, ScoreWithCI] = {}
    by_category_audited: dict[str, ScoreWithCI] = {}
    total_cost_usd = 0.0
    error_message: str | None = None
    error_phase: RunErrorPhase | None = None
    terminal_seen = False

    for event in events:
        if isinstance(event, RunStartedEvent):
            started = event
            status = "running"
            finished_at = None
            n_conversations_processed = 0
            n_questions_attempted = 0
            n_questions_succeeded = 0
            n_questions_errored = 0
            overall_score_standard = None
            overall_score_audited = None
            by_category_standard = {}
            by_category_audited = {}
            total_cost_usd = 0.0
            error_message = None
            error_phase = None
            terminal_seen = False
            continue
        if isinstance(event, RunResumedEvent):
            # A resume supersedes the prior terminal without erasing the work it recorded. Return to
            # running and clear the terminal fields, but keep the historical counters and cost: the
            # resumed run continues the interrupted one, and its ConversationProcessedEvents and
            # final RunCompletedEvent build on what already happened. Handled before the
            # terminal_seen guard below, which would otherwise skip it -- and every later event --
            # so a resumed run's completion stayed invisible and its status stuck at 'failed'.
            if started is None:
                continue
            status = "running"
            finished_at = None
            error_message = None
            error_phase = None
            terminal_seen = False
            continue
        if started is None or terminal_seen:
            continue
        if isinstance(event, ConversationProcessedEvent):
            n_conversations_processed += 1
            n_questions_attempted += event.n_questions_evaluated
            n_questions_succeeded += event.n_questions_correct
            n_questions_errored += event.n_questions_errored
            total_cost_usd += event.cost_usd
        elif isinstance(event, RunCompletedEvent):
            status = event.status
            finished_at = event.timestamp
            n_questions_attempted = event.n_questions_attempted
            n_questions_succeeded = event.n_questions_succeeded
            n_questions_errored = event.n_questions_errored
            overall_score_standard = event.overall_score_standard
            overall_score_audited = event.overall_score_audited
            by_category_standard = dict(event.by_category_standard)
            by_category_audited = dict(event.by_category_audited)
            total_cost_usd = event.total_cost_usd
            terminal_seen = True
        else:
            status = "failed"
            finished_at = event.timestamp
            n_questions_attempted = event.n_questions_completed_before_failure
            total_cost_usd = event.partial_cost_usd
            error_message = event.error_message
            error_phase = event.error_phase
            terminal_seen = True

    if started is None:
        raise PersistenceError("Run lifecycle has no run_started event", run_id=run_id)
    return RunStatus(
        run_id=started.run_id,
        suite_id=started.suite_id,
        experiment_id=started.experiment_id,
        experiment_name=started.experiment_name,
        run_number=started.run_number,
        status=status,
        started_at=started.timestamp,
        finished_at=finished_at,
        provider_type=started.provider_type,
        provider_version=started.provider_version,
        answer_model_id=started.answer_model_id,
        judge_model_id=started.judge_model_id,
        methodology_version=started.methodology_version,
        methodology_profile=started.methodology_profile,
        framework_version=started.framework_version,
        n_conversations_processed=n_conversations_processed,
        n_questions_attempted=n_questions_attempted,
        n_questions_succeeded=n_questions_succeeded,
        n_questions_errored=n_questions_errored,
        overall_score_standard=overall_score_standard,
        overall_score_audited=overall_score_audited,
        by_category_standard=by_category_standard,
        by_category_audited=by_category_audited,
        total_cost_usd=total_cost_usd,
        error_message=error_message,
        error_phase=error_phase,
    )


def _reduce_suite_status(suite_id: str, events: Iterable[SuiteLifecycleEvent]) -> SuiteStatus:
    started: SuiteStartedEvent | None = None
    status: SuiteState = "running"
    finished_at = None
    started_experiments: set[str] = set()
    completed_experiments: set[str] = set()
    failed_experiments: set[str] = set()
    n_experiments_completed = 0
    n_experiments_failed = 0
    total_cost_usd = 0.0
    last_event_at = None
    last_event_type = ""
    terminal_seen = False

    for event in events:
        last_event_at = event.timestamp
        last_event_type = event.event_type
        if isinstance(event, SuiteStartedEvent):
            started = event
            status = "running"
            finished_at = None
            started_experiments = set()
            completed_experiments = set()
            failed_experiments = set()
            n_experiments_completed = 0
            n_experiments_failed = 0
            total_cost_usd = 0.0
            terminal_seen = False
            continue
        if started is None or terminal_seen:
            continue
        if isinstance(event, ExperimentStartedEvent):
            started_experiments.add(event.experiment_id)
        elif isinstance(event, ExperimentCompletedEvent):
            completed_experiments.add(event.experiment_id)
            n_experiments_completed = len(completed_experiments)
            total_cost_usd += event.cost_usd
        elif isinstance(event, ExperimentFailedEvent):
            failed_experiments.add(event.experiment_id)
            n_experiments_failed = len(failed_experiments)
        elif isinstance(event, SuiteCompletedEvent):
            status = "partial" if event.n_experiments_failed > 0 else "completed"
            finished_at = event.timestamp
            n_experiments_completed = event.n_experiments_completed
            n_experiments_failed = event.n_experiments_failed
            total_cost_usd = event.total_cost_usd
            terminal_seen = True
        else:
            status = "failed"
            finished_at = event.timestamp
            n_experiments_completed = event.n_experiments_completed_before_failure
            terminal_seen = True

    if started is None or last_event_at is None:
        raise PersistenceError("Suite lifecycle has no suite_started event", suite_id=suite_id)

    in_progress = 0
    if not terminal_seen:
        known_finished = completed_experiments | failed_experiments
        in_progress = max(0, len(started_experiments - known_finished))

    return SuiteStatus(
        suite_id=started.suite_id,
        status=status,
        started_at=started.timestamp,
        finished_at=finished_at,
        config_yaml_content=started.config_yaml_content,
        methodology_version=started.methodology_version,
        methodology_profile=started.methodology_profile,
        framework_version=started.framework_version,
        n_experiments_planned=started.n_experiments_planned,
        n_experiments_completed=n_experiments_completed,
        n_experiments_failed=n_experiments_failed,
        n_experiments_in_progress=in_progress,
        total_cost_usd=total_cost_usd,
        last_event_at=last_event_at,
        last_event_type=last_event_type,
    )


def _question_record_from_row(row: sqlite3.Row) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=_required_str(row, "question_evaluation_id"),
        run_id=_required_str(row, "run_id"),
        question_id=_required_str(row, "question_id"),
        conversation_id=_required_str(row, "conversation_id"),
        category=QuestionCategory(_required_str(row, "category")),
        question_text=_required_str(row, "question_text"),
        expected_answer=_required_str(row, "expected_answer"),
        is_audited_error=_required_bool(row, "is_audited_error"),
        retrieval_id=_optional_str(row, "retrieval_id"),
        retrieval_timestamp=None,
        retrieval_latency_ms=_optional_float(row, "retrieval_latency_ms"),
        n_memories_retrieved=_optional_int(row, "n_memories_retrieved"),
        retrieved_memory_ids=[],
        response_id=_optional_str(row, "response_id"),
        generation_timestamp=None,
        generation_latency_ms=_optional_float(row, "generation_latency_ms"),
        generated_answer=_optional_str(row, "generated_answer"),
        generation_input_tokens=_optional_int(row, "generation_input_tokens"),
        generation_output_tokens=_optional_int(row, "generation_output_tokens"),
        generation_cost_usd=None,
        judgment_id=_optional_str(row, "judgment_id"),
        judgment_timestamp=None,
        judgment_latency_ms=_optional_float(row, "judgment_latency_ms"),
        verdict=JudgmentVerdict(_required_str(row, "verdict")),
        score=_required_float(row, "score"),
        judgment_reasoning=_optional_str(row, "judgment_reasoning"),
        judgment_input_tokens=_optional_int(row, "judgment_input_tokens"),
        judgment_output_tokens=_optional_int(row, "judgment_output_tokens"),
        judgment_cost_usd=None,
        total_latency_ms=_optional_float(row, "total_latency_ms"),
        total_cost_usd=_required_float(row, "total_cost_usd"),
        error_message=_optional_str(row, "error_message"),
        error_phase=_optional_question_error_phase(row, "error_phase"),
    )


def _run_summary_from_row(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        run_id=_required_str(row, "run_id"),
        suite_id=_required_str(row, "suite_id"),
        experiment_id=_required_str(row, "experiment_id"),
        experiment_name=_required_str(row, "experiment_name"),
        run_number=_required_int(row, "run_number"),
        status=_required_run_state(row, "status"),
        started_at=_required_datetime(row, "started_at"),
        finished_at=_optional_datetime(row, "finished_at"),
        provider_type=_required_str(row, "provider_type"),
        provider_version=_optional_str(row, "provider_version"),
        benchmark_type=_required_str(row, "benchmark_type"),
        answer_model_id=_required_str(row, "answer_model_id"),
        judge_model_id=_required_str(row, "judge_model_id"),
        n_questions_attempted=_optional_int_or_zero(row, "n_questions_attempted"),
        n_questions_succeeded=_optional_int_or_zero(row, "n_questions_succeeded"),
        n_questions_errored=_optional_int_or_zero(row, "n_questions_errored"),
        overall_score_standard=_optional_float(row, "overall_score_standard"),
        overall_score_audited=_optional_float(row, "overall_score_audited"),
        total_cost_usd=_optional_float_or_zero(row, "total_cost_usd"),
        methodology_version=_required_str(row, "methodology_version"),
        methodology_profile=_required_str(row, "methodology_profile"),
        framework_version=_required_str(row, "framework_version"),
        error_message=_optional_str(row, "error_message"),
        error_phase=_optional_run_error_phase(row, "error_phase"),
    )


def _read_union_jsonl(path: Path, adapter: TypeAdapter[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(adapter.validate_json(stripped))
    return records


def _aggregate_score(aggregate: AggregateScore | None) -> ScoreWithCI | None:
    return aggregate.pooled_score if aggregate is not None else None


def _score_point(score: ScoreWithCI | None) -> float | None:
    return score.point_estimate if score is not None else None


def _score_low(score: ScoreWithCI | None) -> float | None:
    return score.ci_95_low if score is not None else None


def _score_high(score: ScoreWithCI | None) -> float | None:
    return score.ci_95_high if score is not None else None


def _required_str(row: sqlite3.Row, key: str) -> str:
    value = cast(object, row[key])
    if isinstance(value, str):
        return value
    raise PersistenceError("SQLite row field is not a string", field=key, value=value)


def _optional_str(row: sqlite3.Row, key: str) -> str | None:
    value = cast(object, row[key])
    if value is None or isinstance(value, str):
        return value
    raise PersistenceError("SQLite row field is not an optional string", field=key, value=value)


def _required_int(row: sqlite3.Row, key: str) -> int:
    value = cast(object, row[key])
    if isinstance(value, int):
        return value
    raise PersistenceError("SQLite row field is not an integer", field=key, value=value)


def _optional_int(row: sqlite3.Row, key: str) -> int | None:
    value = cast(object, row[key])
    if value is None or isinstance(value, int):
        return value
    raise PersistenceError("SQLite row field is not an optional integer", field=key, value=value)


def _optional_int_or_zero(row: sqlite3.Row, key: str) -> int:
    value = _optional_int(row, key)
    return 0 if value is None else value


def _required_float(row: sqlite3.Row, key: str) -> float:
    value = cast(object, row[key])
    if isinstance(value, int | float):
        return float(value)
    raise PersistenceError("SQLite row field is not numeric", field=key, value=value)


def _optional_float(row: sqlite3.Row, key: str) -> float | None:
    value = cast(object, row[key])
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise PersistenceError("SQLite row field is not optional numeric", field=key, value=value)


def _optional_float_or_zero(row: sqlite3.Row, key: str) -> float:
    value = _optional_float(row, key)
    return 0.0 if value is None else value


def _required_bool(row: sqlite3.Row, key: str) -> bool:
    value = _required_int(row, key)
    if value in {0, 1}:
        return bool(value)
    raise PersistenceError("SQLite row field is not boolean-like", field=key, value=value)


def _required_datetime(row: sqlite3.Row, key: str) -> datetime:
    value = _required_str(row, key)
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise PersistenceError(
            "SQLite row field is not an ISO datetime",
            field=key,
            value=value,
        ) from exc


def _optional_datetime(row: sqlite3.Row, key: str) -> datetime | None:
    value = _optional_str(row, key)
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise PersistenceError(
            "SQLite row field is not an optional ISO datetime",
            field=key,
            value=value,
        ) from exc


def _required_run_state(row: sqlite3.Row, key: str) -> RunState:
    value = _required_str(row, key)
    if value in {"running", "completed", "failed", "partial"}:
        return cast(RunState, value)
    raise PersistenceError("SQLite row field is not a run state", field=key, value=value)


def _optional_run_error_phase(row: sqlite3.Row, key: str) -> RunErrorPhase | None:
    value = _optional_str(row, key)
    if value is None:
        return None
    if value in {"initialization", "ingest", "retrieve", "generate", "judge", "analysis"}:
        return cast(RunErrorPhase, value)
    raise PersistenceError("SQLite row field is not a run error phase", field=key, value=value)


def _optional_question_error_phase(
    row: sqlite3.Row,
    key: str,
) -> QuestionErrorPhase | None:
    value = _optional_str(row, key)
    if value is None:
        return None
    if value in {"retrieve", "generate", "judge"}:
        return cast(QuestionErrorPhase, value)
    raise PersistenceError("SQLite row field is not a question error phase", field=key, value=value)
