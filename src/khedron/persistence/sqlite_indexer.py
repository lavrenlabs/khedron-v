from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, TypeAdapter

from khedron.build import resolve_framework_version
from khedron.errors import PersistenceError
from khedron.persistence.schemas import SCHEMA_DDL, SCHEMA_VERSION, V1_SCHEMA_DDL
from khedron.persistence.stage_reader import RunStageStream, StrictRunReader
from khedron.types import (
    AggregateScore,
    ErrorRecord,
    ExperimentResult,
    FailurePattern,
    Judgment,
    JudgmentVerdict,
    QuestionEvaluationRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunLifecycleEvent,
    RunResumedEvent,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteFailedEvent,
    SuiteLifecycleEvent,
    SuiteStartedEvent,
)
from khedron.utils.time import now_utc, to_iso8601

__all__ = ["SqliteIndexer"]

_RUN_EVENT_ADAPTER: TypeAdapter[RunLifecycleEvent] = TypeAdapter(RunLifecycleEvent)
_SUITE_EVENT_ADAPTER: TypeAdapter[SuiteLifecycleEvent] = TypeAdapter(SuiteLifecycleEvent)

_DROP_SCHEMA_DDL = """
DROP TABLE IF EXISTS error_resolutions;
DROP TABLE IF EXISTS recovery_attempts;
DROP TABLE IF EXISTS question_plan;
DROP TABLE IF EXISTS error_log;
DROP TABLE IF EXISTS failure_patterns;
DROP TABLE IF EXISTS scores_by_category;
DROP TABLE IF EXISTS question_results;
DROP TABLE IF EXISTS retrievals;
DROP TABLE IF EXISTS question_evaluations;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS run_lifecycle_events;
DROP TABLE IF EXISTS suite_lifecycle_events;
DROP TABLE IF EXISTS suites;
DROP TABLE IF EXISTS schema_meta;
"""

_DELETE_RUN_QUERIES = (
    "DELETE FROM error_resolutions WHERE run_id = ?",
    "DELETE FROM recovery_attempts WHERE run_id = ?",
    "DELETE FROM question_plan WHERE run_id = ?",
    "DELETE FROM error_log WHERE run_id = ?",
    "DELETE FROM failure_patterns WHERE run_id = ?",
    "DELETE FROM scores_by_category WHERE run_id = ?",
    "DELETE FROM question_results WHERE run_id = ?",
    "DELETE FROM retrievals WHERE run_id = ?",
    "DELETE FROM question_evaluations WHERE run_id = ?",
    "DELETE FROM run_lifecycle_events WHERE run_id = ?",
    "DELETE FROM runs WHERE run_id = ?",
)

ModelT = TypeVar("ModelT", bound=BaseModel)
RunStatusValue = Literal["running", "completed", "failed", "partial"]
SuiteStatusValue = Literal["running", "completed", "failed", "partial"]
QuestionErrorPhase = Literal["retrieve", "generate", "judge"]
ScoreMode = Literal["standard", "audited"]

_V1_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "schema_meta": ("version", "applied_at", "framework_version"),
    "suite_lifecycle_events": (
        "event_id",
        "suite_id",
        "sequence_number",
        "timestamp",
        "event_type",
        "event_data",
    ),
    "run_lifecycle_events": (
        "event_id",
        "run_id",
        "sequence_number",
        "timestamp",
        "event_type",
        "event_data",
    ),
    "suites": (
        "suite_id",
        "started_at",
        "finished_at",
        "status",
        "total_cost_usd",
        "n_experiments_planned",
        "n_experiments_completed",
        "n_experiments_failed",
        "methodology_version",
        "methodology_profile",
        "framework_version",
        "config_yaml",
    ),
    "experiment_results": (
        "experiment_id",
        "suite_id",
        "experiment_name",
        "n_runs",
        "overall_standard_pooled_score",
        "overall_standard_ci_low",
        "overall_standard_ci_high",
        "overall_standard_mean",
        "overall_standard_stddev",
        "overall_audited_pooled_score",
        "overall_audited_ci_low",
        "overall_audited_ci_high",
        "overall_audited_mean",
        "overall_audited_stddev",
        "total_cost_usd",
        "config_json",
    ),
    "runs": (
        "run_id",
        "experiment_id",
        "suite_id",
        "experiment_name",
        "run_number",
        "provider_type",
        "provider_version",
        "benchmark_type",
        "benchmark_version",
        "benchmark_checksum",
        "answer_model_vendor",
        "answer_model_id",
        "judge_model_vendor",
        "judge_model_id",
        "started_at",
        "finished_at",
        "status",
        "n_questions_attempted",
        "n_questions_succeeded",
        "n_questions_errored",
        "overall_score_standard",
        "overall_score_standard_ci_low",
        "overall_score_standard_ci_high",
        "overall_score_audited",
        "overall_score_audited_ci_low",
        "overall_score_audited_ci_high",
        "total_cost_usd",
        "methodology_version",
        "methodology_profile",
        "framework_version",
        "runtime_environment",
        "seed",
        "error_message",
        "error_phase",
    ),
    "question_evaluations": (
        "question_evaluation_id",
        "run_id",
        "question_id",
        "conversation_id",
        "category",
        "question_text",
        "expected_answer",
        "is_audited_error",
        "timestamp",
    ),
    "retrievals": (
        "retrieval_id",
        "question_evaluation_id",
        "run_id",
        "question_id",
        "timestamp",
        "query",
        "top_k",
        "n_returned",
        "retrieval_latency_ms",
    ),
    "question_results": (
        "question_id",
        "run_id",
        "question_evaluation_id",
        "conversation_id",
        "category",
        "question_text",
        "expected_answer",
        "generated_answer",
        "retrieval_id",
        "response_id",
        "judgment_id",
        "verdict",
        "score",
        "judgment_reasoning",
        "is_audited_error",
        "n_memories_retrieved",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "judgment_latency_ms",
        "total_latency_ms",
        "generation_input_tokens",
        "generation_output_tokens",
        "judgment_input_tokens",
        "judgment_output_tokens",
        "total_cost_usd",
        "error_message",
        "error_phase",
    ),
    "scores_by_category": (
        "run_id",
        "category",
        "mode",
        "n_total",
        "n_correct",
        "n_errors",
        "n_partial",
        "n_unknown",
        "point_estimate",
        "ci_95_low",
        "ci_95_high",
    ),
    "failure_patterns": (
        "pattern_id",
        "run_id",
        "pattern_name",
        "description",
        "suggested_remedy",
        "n_affected_questions",
        "confidence",
        "affected_question_ids",
    ),
    "error_log": (
        "error_id",
        "run_id",
        "timestamp",
        "phase",
        "question_id",
        "error_type",
        "error_message",
        "recovered",
    ),
}

_V2_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    **_V1_TABLE_COLUMNS,
    "retrievals": (*_V1_TABLE_COLUMNS["retrievals"], "recovery_attempt_id", "ingestion_attempt_id"),
    "error_log": (*_V1_TABLE_COLUMNS["error_log"], "retryable"),
    "question_plan": (
        "run_id",
        "benchmark_id",
        "categories",
        "corpus_checksum",
        "question_ids",
        "fingerprint",
        "timestamp",
    ),
    "recovery_attempts": ("attempt_id", "run_id", "question_id", "error_id", "stage", "timestamp"),
    "error_resolutions": (
        "resolution_id",
        "run_id",
        "question_id",
        "error_id",
        "recovery_attempt_id",
        "resolved_by_stage",
        "resolved_by_stage_record_id",
        "timestamp",
    ),
}


def _schema_signature(connection: sqlite3.Connection) -> tuple[object, ...]:
    """Return the complete SQLite contract relevant to safe migration decisions."""
    tables = connection.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    table_signature: list[object] = []
    index_signature: list[tuple[str, int, str, tuple[str, ...]]] = []
    for table in tables:
        table_name = str(table["name"])
        columns = tuple(
            (
                str(row["name"]),
                str(row["type"]),
                int(row["notnull"]),
                row["dflt_value"],
                int(row["pk"]),
            )
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        )
        foreign_keys = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
        )
        table_sql = " ".join(str(table["sql"]).replace("IF NOT EXISTS ", "").split())
        table_signature.append((table_name, table_sql, columns, foreign_keys))
        for index in connection.execute(f"PRAGMA index_list({table_name})").fetchall():
            index_name = str(index["name"])
            index_columns = tuple(
                str(row["name"])
                for row in connection.execute(f"PRAGMA index_info({index_name})").fetchall()
            )
            index_signature.append(
                (index_name, int(index["unique"]), str(index["origin"]), index_columns)
            )
    return tuple(table_signature), tuple(sorted(index_signature))


def _canonical_schema_signature(ddl: str) -> tuple[object, ...]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(ddl)
        return _schema_signature(connection)
    finally:
        connection.close()


_V1_SCHEMA_SIGNATURE = _canonical_schema_signature(V1_SCHEMA_DDL)
_V2_SCHEMA_SIGNATURE = _canonical_schema_signature(SCHEMA_DDL)


def _empty_score_map() -> dict[str, ScoreWithCI]:
    return {}


@dataclass(frozen=True)
class SchemaMeta:
    applied_at: str
    framework_version: str


@dataclass
class RunProjection:
    run_id: str
    experiment_id: str
    suite_id: str
    experiment_name: str
    run_number: int
    provider_type: str
    provider_version: str
    benchmark_type: str
    benchmark_version: str
    benchmark_checksum: str
    answer_model_vendor: str
    answer_model_id: str
    judge_model_vendor: str
    judge_model_id: str
    started_at: str
    methodology_version: str
    methodology_profile: str
    framework_version: str
    runtime_environment: str
    seed: int
    finished_at: str | None = None
    status: RunStatusValue = "running"
    n_questions_attempted: int = 0
    n_questions_succeeded: int = 0
    n_questions_errored: int = 0
    overall_score_standard: ScoreWithCI | None = None
    overall_score_audited: ScoreWithCI | None = None
    by_category_standard: dict[str, ScoreWithCI] = field(default_factory=_empty_score_map)
    by_category_audited: dict[str, ScoreWithCI] = field(default_factory=_empty_score_map)
    total_cost_usd: float = 0.0
    error_message: str | None = None
    error_phase: str | None = None


@dataclass
class SuiteProjection:
    suite_id: str
    started_at: str
    n_experiments_planned: int
    methodology_version: str
    methodology_profile: str
    framework_version: str
    config_yaml: str
    finished_at: str | None = None
    status: SuiteStatusValue = "running"
    total_cost_usd: float = 0.0
    n_experiments_completed: int = 0
    n_experiments_failed: int = 0


@dataclass(frozen=True)
class SuiteIndexData:
    events: tuple[SuiteLifecycleEvent, ...]
    projection: SuiteProjection


@dataclass(frozen=True)
class IndexingSnapshot:
    """All JSONL inputs for a run captured before SQLite is allowed to mutate."""

    stream: RunStageStream
    run_events: tuple[RunLifecycleEvent, ...]
    projection: RunProjection
    failure_patterns: tuple[FailurePattern, ...]
    suite_data: SuiteIndexData | None


@dataclass(frozen=True)
class RebuildSnapshot:
    """Complete JSONL input captured before a rebuild creates its temporary database."""

    runs: tuple[IndexingSnapshot, ...]
    suites: tuple[SuiteIndexData, ...]
    experiment_results: tuple[ExperimentResult, ...]


class SqliteIndexer:
    """Build and maintain the SQLite query layer from JSONL source-of-truth files."""

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._results_root = db_path.parent

    def initialize(self) -> None:
        """Create v2, or atomically migrate a structurally valid v1 projection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not enable WAL until compatibility has been validated: it is a persistent
        # file mutation and malformed databases must remain untouched.
        with closing(self._connect()) as connection:
            try:
                tables = self._user_table_names(connection)
            except sqlite3.DatabaseError as exc:
                raise PersistenceError(
                    "SQLite database has an unreadable schema definition",
                    db_path=str(self._db_path),
                ) from exc
            if not tables:
                with connection:
                    connection.executescript(SCHEMA_DDL)
                    self._replace_schema_meta(connection, None)
                return

            meta = self._read_raw_schema_meta(connection)
            if meta is None:
                raise PersistenceError(
                    "SQLite database has missing or malformed schema metadata",
                    db_path=str(self._db_path),
                )
            version, previous_meta = meta
            if version == 1:
                if not self._has_schema_signature(connection, _V1_SCHEMA_SIGNATURE):
                    raise PersistenceError(
                        "SQLite v1 database has an unreadable projection structure",
                        db_path=str(self._db_path),
                    )
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    self._migrate_v1_to_v2(connection)
                    if not self._has_schema_signature(connection, _V2_SCHEMA_SIGNATURE):
                        raise PersistenceError(
                            "SQLite v1-to-v2 migration produced an unreadable projection structure",
                            db_path=str(self._db_path),
                        )
                    self._replace_schema_meta(connection, previous_meta)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                return
            if version != SCHEMA_VERSION or not self._has_schema_signature(
                connection, _V2_SCHEMA_SIGNATURE
            ):
                raise PersistenceError(
                    "SQLite database has an unsupported or structurally unreadable schema",
                    db_path=str(self._db_path),
                    schema_version=version,
                )

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        """Apply the v1-to-v2 DDL inside the caller's transaction."""
        connection.execute("ALTER TABLE error_log ADD COLUMN retryable INTEGER")
        connection.execute("ALTER TABLE retrievals ADD COLUMN recovery_attempt_id TEXT")
        connection.execute("ALTER TABLE retrievals ADD COLUMN ingestion_attempt_id TEXT")
        connection.execute(
            """
            CREATE TABLE question_plan (
                run_id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, categories TEXT NOT NULL,
                corpus_checksum TEXT NOT NULL, question_ids TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                timestamp TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE recovery_attempts (
                attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, question_id TEXT NOT NULL,
                error_id TEXT NOT NULL, stage TEXT NOT NULL, timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE error_resolutions (
                resolution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, question_id TEXT NOT NULL,
                error_id TEXT NOT NULL UNIQUE, recovery_attempt_id TEXT NOT NULL,
                resolved_by_stage TEXT NOT NULL, resolved_by_stage_record_id TEXT NOT NULL,
                timestamp TEXT NOT NULL, FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )

    def get_schema_version(self) -> int:
        """Return v2 after creating or migrating the CLI/indexer projection path."""
        self.initialize()
        return self.read_schema_version()

    def read_schema_version(self) -> int:
        """Read a validated v2 schema version without creating, migrating, or writing."""
        if not self._db_path.exists():
            raise PersistenceError("SQLite database does not exist", db_path=str(self._db_path))
        uri = f"file:{self._db_path.resolve().as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                raw_meta = self._read_raw_schema_meta(connection)
                if raw_meta is None:
                    raise PersistenceError(
                        "SQLite database has missing or malformed schema metadata",
                        db_path=str(self._db_path),
                    )
                version, _ = raw_meta
                if version != SCHEMA_VERSION or not self._has_schema_signature(
                    connection, _V2_SCHEMA_SIGNATURE
                ):
                    raise PersistenceError(
                        "SQLite database has an unsupported or structurally unreadable schema",
                        db_path=str(self._db_path),
                        schema_version=version,
                    )
        except sqlite3.Error as error:
            raise PersistenceError(
                "SQLite schema version could not be read", db_path=str(self._db_path)
            ) from error
        return SCHEMA_VERSION

    def index_run(self, run_id: str) -> None:
        """Index one run directory into SQLite."""
        snapshot = self._capture_snapshot(run_id)
        self.initialize()
        with closing(self._connect(write=True)) as connection:
            with connection:
                self._project_snapshot(connection, snapshot)

    def _capture_snapshot(self, run_id: str) -> IndexingSnapshot:
        """Capture the one immutable input used by all run projection paths."""
        run_dir = self._run_dir(run_id)
        stream = StrictRunReader().read(run_dir, run_id)
        run_events = tuple(sorted(stream.lifecycle, key=_sequence))
        projection = self._reduce_run_events(run_id, run_events)
        suite_data = self._load_suite_index_data(projection.suite_id)
        return IndexingSnapshot(stream, run_events, projection, stream.failure_patterns, suite_data)

    def _project_snapshot(self, connection: sqlite3.Connection, snapshot: IndexingSnapshot) -> None:
        """Project a captured snapshot without reopening any JSONL source."""
        if snapshot.suite_data is not None:
            self._replace_suite(
                connection, snapshot.suite_data.events, snapshot.suite_data.projection
            )
        stream = snapshot.stream
        self._delete_run_rows(connection, stream.run_id)
        self._insert_run_lifecycle_events(connection, snapshot.run_events)
        self._insert_run(connection, snapshot.projection)
        self._insert_question_evaluations(connection, stream.evaluations)
        self._insert_retrievals(connection, stream.retrievals)
        self._insert_question_results(
            connection,
            stream.evaluations,
            stream.retrievals,
            stream.responses,
            stream.judgments,
            stream.errors,
        )
        self._insert_scores_by_category(connection, snapshot.projection)
        self._insert_failure_patterns(connection, snapshot.failure_patterns)
        self._insert_errors(connection, stream.errors)
        self._insert_strict_records(connection, stream)

    def rebuild_all(self) -> None:
        """Rebuild SQLite from JSONL with a two-phase promotion contract.

        Pre-promotion the old bundle is preserved. Post-promotion the new bundle is
        authoritative: backup-unlink failure is non-fatal-but-surfaced, the backup remains
        recoverable, and no stale active sidecar is restored. No rollback is attempted.
        """
        runs_dir = self._results_root / "runs"
        snapshots = tuple(
            self._capture_snapshot(run_dir.name)
            for run_dir in sorted(runs_dir.iterdir() if runs_dir.exists() else ())
            if run_dir.is_dir()
        )
        # Parse suite inputs before creating the temporary database.  They are also captured in
        # every run snapshot, but unreferenced suites must be preserved by a full rebuild.
        suites_dir = self._results_root / "suites"
        suite_snapshots = tuple(
            data
            for data in (
                self._load_suite_index_data(path.name)
                for path in sorted(suites_dir.iterdir() if suites_dir.exists() else ())
                if path.is_dir()
            )
            if data is not None
        )
        experiment_results = tuple(
            result
            for suite_dir in sorted(suites_dir.iterdir() if suites_dir.exists() else ())
            if suite_dir.is_dir()
            for result in self._read_records(suite_dir / "experiments.jsonl", ExperimentResult)
        )
        rebuild_snapshot = RebuildSnapshot(snapshots, suite_snapshots, experiment_results)
        previous_meta = self._read_schema_meta_from_disk()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".khedron-rebuild-", suffix=".db", dir=self._db_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        temporary_indexer = SqliteIndexer(temporary_path)
        temporary_indexer._results_root = self._results_root
        sidecar_backups: list[tuple[Path, Path]] = []
        promoted = False
        try:
            temporary_indexer.initialize()
            with closing(temporary_indexer._connect(write=True)) as connection:
                with connection:
                    connection.executescript(_DROP_SCHEMA_DDL)
                    connection.executescript(SCHEMA_DDL)
                    temporary_indexer._replace_schema_meta(connection, previous_meta)
            with closing(temporary_indexer._connect(write=True)) as connection:
                with connection:
                    for suite_data in rebuild_snapshot.suites:
                        temporary_indexer._replace_suite(
                            connection, suite_data.events, suite_data.projection
                        )
                    for result in rebuild_snapshot.experiment_results:
                        temporary_indexer._insert_experiment_result(connection, result)
                    for snapshot in rebuild_snapshot.runs:
                        temporary_indexer._project_snapshot(connection, snapshot)
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # Rename, rather than delete, old sidecars before replacing the main file.  A failure
            # before promotion can restore the exact old bundle; a cleanup failure after promotion
            # cannot leave a stale file at the SQLite sidecar name.
            for sidecar in (
                self._db_path.with_name(f"{self._db_path.name}-wal"),
                self._db_path.with_name(f"{self._db_path.name}-shm"),
            ):
                if sidecar.exists():
                    backup = sidecar.with_name(f".{sidecar.name}.pre-rebuild")
                    if backup.exists():
                        backup.unlink()
                    os.replace(sidecar, backup)
                    sidecar_backups.append((sidecar, backup))
            os.replace(temporary_path, self._db_path)
            promoted = True
            try:
                for _, backup in sidecar_backups:
                    backup.unlink()
            except OSError as exc:
                raise PersistenceError(
                    "SQLite rebuild promoted but could not remove retired sidecar backup",
                    db_path=str(self._db_path),
                ) from exc
            sidecar_backups.clear()
        finally:
            if not promoted:
                for sidecar, backup in reversed(sidecar_backups):
                    if backup.exists():
                        os.replace(backup, sidecar)
            if temporary_path.exists():
                temporary_path.unlink()
        """
        with closing(self._connect(write=True)) as connection:
            with connection:
                connection.executescript(_DROP_SCHEMA_DDL)
                connection.executescript(SCHEMA_DDL)
                self._replace_schema_meta(connection, previous_meta)

        suites_dir = self._results_root / "suites"
        if suites_dir.exists():
            for suite_dir in sorted(path for path in suites_dir.iterdir() if path.is_dir()):
                if (suite_dir / "lifecycle.jsonl").exists():
                    self._index_suite(suite_dir.name)

        runs_dir = self._results_root / "runs"
        if runs_dir.exists():
            for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
                if (run_dir / "lifecycle.jsonl").exists():
                self.index_run(run_dir.name)
        """

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        # The 30s busy timeout lets a read wait for a concurrent write instead of failing
        # with "database is locked". WAL is enabled only on write connections (the DB
        # create/index paths): journal mode is a persistent file property, so once a
        # writer sets it every reader benefits for free, while read-only inspections
        # (get_schema_version, the `db version` CLI) never mutate a possibly read-only DB.
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        if write:
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _run_dir(self, run_id: str) -> Path:
        return self._results_root / "runs" / run_id

    def _suite_dir(self, suite_id: str) -> Path:
        return self._results_root / "suites" / suite_id

    def _read_schema_meta_from_disk(self) -> SchemaMeta | None:
        if not self._db_path.exists():
            return None
        with closing(self._connect()) as connection:
            return self._read_schema_meta(connection)

    def _read_raw_schema_meta(
        self, connection: sqlite3.Connection
    ) -> tuple[int, SchemaMeta] | None:
        try:
            rows = connection.execute(
                """
                SELECT version, applied_at, framework_version
                FROM schema_meta
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if len(rows) != 1:
            return None
        row = rows[0]

        version_value = cast(object, row["version"])
        applied_at = cast(object, row["applied_at"])
        framework_version = cast(object, row["framework_version"])
        if (
            not isinstance(version_value, int)
            or not isinstance(applied_at, str)
            or not isinstance(framework_version, str)
        ):
            return None
        return version_value, SchemaMeta(applied_at=applied_at, framework_version=framework_version)

    def _read_schema_meta(self, connection: sqlite3.Connection) -> SchemaMeta | None:
        raw_meta = self._read_raw_schema_meta(connection)
        if raw_meta is None:
            return None
        version, meta = raw_meta
        return meta if version == SCHEMA_VERSION else None

    def _user_table_names(self, connection: sqlite3.Connection) -> set[str]:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {str(row["name"]) for row in rows}

    def _has_schema_signature(
        self, connection: sqlite3.Connection, expected: tuple[object, ...]
    ) -> bool:
        return _schema_signature(connection) == expected

    def _replace_schema_meta(
        self,
        connection: sqlite3.Connection,
        previous_meta: SchemaMeta | None,
    ) -> None:
        applied_at = (
            previous_meta.applied_at if previous_meta is not None else to_iso8601(now_utc())
        )
        framework_version = (
            previous_meta.framework_version
            if previous_meta is not None
            else resolve_framework_version()
        )
        connection.execute("DELETE FROM schema_meta")
        connection.execute(
            """
            INSERT INTO schema_meta (version, applied_at, framework_version)
            VALUES (?, ?, ?)
            """,
            (SCHEMA_VERSION, applied_at, framework_version),
        )

    def _index_suite(self, suite_id: str) -> None:
        self.initialize()
        suite_data = self._load_suite_index_data(suite_id)
        if suite_data is None:
            return
        with closing(self._connect(write=True)) as connection:
            with connection:
                self._replace_suite(connection, suite_data.events, suite_data.projection)

    def _load_suite_index_data(self, suite_id: str) -> SuiteIndexData | None:
        lifecycle_path = self._suite_dir(suite_id) / "lifecycle.jsonl"
        if not lifecycle_path.exists():
            return None
        suite_events = tuple(sorted(self._read_suite_events(lifecycle_path), key=_sequence))
        return SuiteIndexData(
            events=suite_events,
            projection=self._reduce_suite_events(suite_id, suite_events),
        )

    def _replace_suite(
        self,
        connection: sqlite3.Connection,
        events: Iterable[SuiteLifecycleEvent],
        projection: SuiteProjection,
    ) -> None:
        connection.execute(
            "DELETE FROM suite_lifecycle_events WHERE suite_id = ?",
            (projection.suite_id,),
        )
        connection.executemany(
            """
            INSERT INTO suite_lifecycle_events (
                event_id, suite_id, sequence_number, timestamp, event_type, event_data
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event.event_id,
                    event.suite_id,
                    event.sequence_number,
                    to_iso8601(event.timestamp),
                    event.event_type,
                    event.model_dump_json(),
                )
                for event in events
            ],
        )
        connection.execute(
            """
            INSERT INTO suites (
                suite_id, started_at, finished_at, status, total_cost_usd,
                n_experiments_planned, n_experiments_completed, n_experiments_failed,
                methodology_version, methodology_profile, framework_version, config_yaml
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(suite_id) DO UPDATE SET
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                status = excluded.status,
                total_cost_usd = excluded.total_cost_usd,
                n_experiments_planned = excluded.n_experiments_planned,
                n_experiments_completed = excluded.n_experiments_completed,
                n_experiments_failed = excluded.n_experiments_failed,
                methodology_version = excluded.methodology_version,
                methodology_profile = excluded.methodology_profile,
                framework_version = excluded.framework_version,
                config_yaml = excluded.config_yaml
            """,
            (
                projection.suite_id,
                projection.started_at,
                projection.finished_at,
                projection.status,
                projection.total_cost_usd,
                projection.n_experiments_planned,
                projection.n_experiments_completed,
                projection.n_experiments_failed,
                projection.methodology_version,
                projection.methodology_profile,
                projection.framework_version,
                projection.config_yaml,
            ),
        )

    def _delete_run_rows(self, connection: sqlite3.Connection, run_id: str) -> None:
        for query in _DELETE_RUN_QUERIES:
            connection.execute(query, (run_id,))

    def _insert_experiment_result(
        self, connection: sqlite3.Connection, result: ExperimentResult
    ) -> None:
        """Project one pre-captured experiment aggregate during an atomic rebuild."""
        standard = result.aggregate_overall_standard
        audited = result.aggregate_overall_audited
        connection.execute(
            """
            INSERT INTO experiment_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.experiment_id,
                result.suite_id,
                result.experiment_name,
                result.n_runs,
                _aggregate_point(standard),
                _aggregate_low(standard),
                _aggregate_high(standard),
                standard.mean if standard is not None else None,
                standard.stddev if standard is not None else None,
                _aggregate_point(audited),
                _aggregate_low(audited),
                _aggregate_high(audited),
                audited.mean if audited is not None else None,
                audited.stddev if audited is not None else None,
                result.total_cost_usd,
                result.config.model_dump_json(),
            ),
        )

    def _insert_run_lifecycle_events(
        self,
        connection: sqlite3.Connection,
        events: Iterable[RunLifecycleEvent],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO run_lifecycle_events (
                event_id, run_id, sequence_number, timestamp, event_type, event_data
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event.event_id,
                    event.run_id,
                    event.sequence_number,
                    to_iso8601(event.timestamp),
                    event.event_type,
                    event.model_dump_json(),
                )
                for event in events
            ],
        )

    def _insert_run(self, connection: sqlite3.Connection, projection: RunProjection) -> None:
        standard_score = projection.overall_score_standard
        audited_score = projection.overall_score_audited
        connection.execute(
            """
            INSERT INTO runs (
                run_id, experiment_id, suite_id, experiment_name, run_number,
                provider_type, provider_version, benchmark_type, benchmark_version,
                benchmark_checksum, answer_model_vendor, answer_model_id,
                judge_model_vendor, judge_model_id, started_at, finished_at, status,
                n_questions_attempted, n_questions_succeeded, n_questions_errored,
                overall_score_standard, overall_score_standard_ci_low,
                overall_score_standard_ci_high, overall_score_audited,
                overall_score_audited_ci_low, overall_score_audited_ci_high,
                total_cost_usd, methodology_version, methodology_profile,
                framework_version, runtime_environment, seed, error_message, error_phase
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                projection.run_id,
                projection.experiment_id,
                projection.suite_id,
                projection.experiment_name,
                projection.run_number,
                projection.provider_type,
                projection.provider_version,
                projection.benchmark_type,
                projection.benchmark_version,
                projection.benchmark_checksum,
                projection.answer_model_vendor,
                projection.answer_model_id,
                projection.judge_model_vendor,
                projection.judge_model_id,
                projection.started_at,
                projection.finished_at,
                projection.status,
                projection.n_questions_attempted,
                projection.n_questions_succeeded,
                projection.n_questions_errored,
                _score_point(standard_score),
                _score_low(standard_score),
                _score_high(standard_score),
                _score_point(audited_score),
                _score_low(audited_score),
                _score_high(audited_score),
                projection.total_cost_usd,
                projection.methodology_version,
                projection.methodology_profile,
                projection.framework_version,
                projection.runtime_environment,
                projection.seed,
                projection.error_message,
                projection.error_phase,
            ),
        )

    def _insert_question_evaluations(
        self,
        connection: sqlite3.Connection,
        records: Iterable[QuestionEvaluationRecord],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO question_evaluations (
                question_evaluation_id, run_id, question_id, conversation_id, category,
                question_text, expected_answer, is_audited_error, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.question_evaluation_id,
                    record.run_id,
                    record.question_id,
                    record.conversation_id,
                    record.category.value,
                    record.question_text,
                    record.expected_answer,
                    _sqlite_bool(record.is_audited_error),
                    to_iso8601(record.timestamp),
                )
                for record in records
            ],
        )

    def _insert_retrievals(
        self,
        connection: sqlite3.Connection,
        records: Iterable[RetrievalRecord],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO retrievals (
                retrieval_id, question_evaluation_id, run_id, question_id, timestamp,
                query, top_k, n_returned, retrieval_latency_ms, recovery_attempt_id,
                ingestion_attempt_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.retrieval_id,
                    record.question_evaluation_id,
                    record.run_id,
                    record.question_id,
                    to_iso8601(record.timestamp),
                    record.query,
                    record.top_k,
                    record.n_returned,
                    record.retrieval_latency_ms,
                    record.recovery_attempt_id,
                    record.ingestion_attempt_id,
                )
                for record in records
            ],
        )

    def _insert_question_results(
        self,
        connection: sqlite3.Connection,
        question_evaluations: Iterable[QuestionEvaluationRecord],
        retrievals: Iterable[RetrievalRecord],
        responses: Iterable[Response],
        judgments: Iterable[Judgment],
        errors: Iterable[ErrorRecord],
    ) -> None:
        retrieval_by_evaluation_id = {
            record.question_evaluation_id: record for record in retrievals
        }
        response_by_retrieval_id = {record.retrieval_id: record for record in responses}
        judgment_by_response_id = {record.response_id: record for record in judgments}
        error_messages = _question_error_messages(errors)

        rows: list[tuple[object, ...]] = []
        for question in question_evaluations:
            retrieval_record = retrieval_by_evaluation_id.get(question.question_evaluation_id)
            response_record = (
                response_by_retrieval_id.get(retrieval_record.retrieval_id)
                if retrieval_record is not None
                else None
            )
            judgment_record = (
                judgment_by_response_id.get(response_record.response_id)
                if response_record is not None
                else None
            )
            result = _project_question_result(
                question,
                retrieval_record,
                response_record,
                judgment_record,
                error_messages,
            )
            rows.append(result)

        connection.executemany(
            """
            INSERT INTO question_results (
                question_id, run_id, question_evaluation_id, conversation_id, category,
                question_text, expected_answer, generated_answer, retrieval_id, response_id,
                judgment_id, verdict, score, judgment_reasoning, is_audited_error,
                n_memories_retrieved, retrieval_latency_ms, generation_latency_ms,
                judgment_latency_ms, total_latency_ms, generation_input_tokens,
                generation_output_tokens, judgment_input_tokens, judgment_output_tokens,
                total_cost_usd, error_message, error_phase
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_scores_by_category(
        self,
        connection: sqlite3.Connection,
        projection: RunProjection,
    ) -> None:
        for category, score in projection.by_category_standard.items():
            self._insert_category_score(connection, projection.run_id, category, "standard", score)
        for category, score in projection.by_category_audited.items():
            self._insert_category_score(connection, projection.run_id, category, "audited", score)

    def _insert_category_score(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        category: str,
        mode: ScoreMode,
        score: ScoreWithCI,
    ) -> None:
        connection.execute(
            """
            INSERT INTO scores_by_category (
                run_id, category, mode, n_total, n_correct, n_errors, n_partial,
                n_unknown, point_estimate, ci_95_low, ci_95_high
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                category,
                mode,
                score.n_total,
                score.n_correct,
                score.n_errors,
                score.n_partial,
                score.n_unknown,
                score.point_estimate,
                score.ci_95_low,
                score.ci_95_high,
            ),
        )

    def _insert_failure_patterns(
        self,
        connection: sqlite3.Connection,
        records: Iterable[FailurePattern],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO failure_patterns (
                pattern_id, run_id, pattern_name, description, suggested_remedy,
                n_affected_questions, confidence, affected_question_ids
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.pattern_id,
                    record.run_id,
                    record.pattern_name,
                    record.description,
                    record.suggested_remedy,
                    record.n_affected_questions,
                    record.confidence,
                    _json_dumps(record.affected_question_ids),
                )
                for record in records
            ],
        )

    def _insert_errors(
        self,
        connection: sqlite3.Connection,
        records: Iterable[ErrorRecord],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO error_log (
                error_id, run_id, timestamp, phase, question_id, error_type,
                error_message, recovered, retryable
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.error_id,
                    record.run_id,
                    to_iso8601(record.timestamp),
                    record.phase,
                    record.question_id,
                    record.error_type,
                    record.error_message,
                    _sqlite_bool(record.recovered),
                    _sqlite_bool(record.retryable) if record.retryable is not None else None,
                )
                for record in records
            ],
        )

    def _insert_strict_records(
        self, connection: sqlite3.Connection, stream: RunStageStream
    ) -> None:
        """Project B.1 durable records after the shared reader has validated them."""
        strict_stream = stream
        if strict_stream.plan is not None:
            plan = strict_stream.plan
            connection.execute(
                "INSERT INTO question_plan VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    plan.run_id,
                    plan.benchmark_id,
                    json.dumps(plan.categories),
                    plan.corpus_checksum,
                    json.dumps(plan.question_ids),
                    plan.fingerprint,
                    to_iso8601(plan.timestamp),
                ),
            )
        connection.executemany(
            "INSERT INTO recovery_attempts VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r.attempt_id,
                    r.run_id,
                    r.question_id,
                    r.error_id,
                    r.stage,
                    to_iso8601(r.timestamp),
                )
                for r in strict_stream.attempts
            ],
        )
        connection.executemany(
            "INSERT INTO error_resolutions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.resolution_id,
                    r.run_id,
                    r.question_id,
                    r.error_id,
                    r.recovery_attempt_id,
                    r.resolved_by_stage,
                    r.resolved_by_stage_record_id,
                    to_iso8601(r.timestamp),
                )
                for r in strict_stream.resolutions
            ],
        )

    def _read_run_events(self, path: Path) -> list[RunLifecycleEvent]:
        if not path.exists():
            raise PersistenceError("Run lifecycle file does not exist", path=str(path))
        events: list[RunLifecycleEvent] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    events.append(_RUN_EVENT_ADAPTER.validate_json(stripped))
        return events

    def _read_suite_events(self, path: Path) -> list[SuiteLifecycleEvent]:
        if not path.exists():
            raise PersistenceError("Suite lifecycle file does not exist", path=str(path))
        events: list[SuiteLifecycleEvent] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    events.append(_SUITE_EVENT_ADAPTER.validate_json(stripped))
        return events

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

    def _reduce_run_events(
        self,
        run_id: str,
        events: Iterable[RunLifecycleEvent],
    ) -> RunProjection:
        projection: RunProjection | None = None
        terminal_seen = False
        for event in events:
            if isinstance(event, RunStartedEvent):
                projection = _run_projection_from_started_event(event)
                terminal_seen = False
                continue
            if isinstance(event, RunResumedEvent):
                # A resume supersedes the prior terminal without erasing the recorded work: back to
                # running and the terminal fields cleared, historical counters and cost kept.
                # Handled before the terminal_seen guard so the SQLite projection matches the
                # append-only reducer -- otherwise a resumed run stayed 'failed' in the query layer.
                if projection is None:
                    continue
                projection.status = "running"
                projection.finished_at = None
                projection.error_message = None
                projection.error_phase = None
                terminal_seen = False
                continue
            if projection is None or terminal_seen:
                continue
            if event.event_type == "conversation_processed":
                projection.n_questions_attempted += event.n_questions_evaluated
                projection.n_questions_succeeded += event.n_questions_correct
                projection.n_questions_errored += event.n_questions_errored
                projection.total_cost_usd += event.cost_usd
            elif isinstance(event, RunCompletedEvent):
                projection.status = event.status
                projection.finished_at = to_iso8601(event.timestamp)
                projection.n_questions_attempted = event.n_questions_attempted
                projection.n_questions_succeeded = event.n_questions_succeeded
                projection.n_questions_errored = event.n_questions_errored
                projection.overall_score_standard = event.overall_score_standard
                projection.overall_score_audited = event.overall_score_audited
                projection.by_category_standard = dict(event.by_category_standard)
                projection.by_category_audited = dict(event.by_category_audited)
                projection.total_cost_usd = event.total_cost_usd
                terminal_seen = True
            else:
                projection.status = "failed"
                projection.finished_at = to_iso8601(event.timestamp)
                projection.n_questions_attempted = event.n_questions_completed_before_failure
                projection.total_cost_usd = event.partial_cost_usd
                projection.error_message = event.error_message
                projection.error_phase = event.error_phase
                terminal_seen = True

        if projection is None:
            raise PersistenceError("Run lifecycle has no run_started event", run_id=run_id)
        return projection

    def _reduce_suite_events(
        self,
        suite_id: str,
        events: Iterable[SuiteLifecycleEvent],
    ) -> SuiteProjection:
        projection: SuiteProjection | None = None
        terminal_seen = False
        for event in events:
            if isinstance(event, SuiteStartedEvent):
                projection = SuiteProjection(
                    suite_id=event.suite_id,
                    started_at=to_iso8601(event.timestamp),
                    n_experiments_planned=event.n_experiments_planned,
                    methodology_version=event.methodology_version,
                    methodology_profile=event.methodology_profile,
                    framework_version=event.framework_version,
                    config_yaml=event.config_yaml_content,
                )
                terminal_seen = False
                continue
            if projection is None or terminal_seen:
                continue
            if event.event_type == "experiment_completed":
                projection.n_experiments_completed += 1
                projection.total_cost_usd += event.cost_usd
            elif event.event_type == "experiment_failed":
                projection.n_experiments_failed += 1
            elif isinstance(event, SuiteCompletedEvent):
                projection.status = "partial" if event.n_experiments_failed > 0 else "completed"
                projection.finished_at = to_iso8601(event.timestamp)
                projection.total_cost_usd = event.total_cost_usd
                projection.n_experiments_completed = event.n_experiments_completed
                projection.n_experiments_failed = event.n_experiments_failed
                terminal_seen = True
            elif isinstance(event, SuiteFailedEvent):
                projection.status = "failed"
                projection.finished_at = to_iso8601(event.timestamp)
                projection.n_experiments_completed = event.n_experiments_completed_before_failure
                terminal_seen = True

        if projection is None:
            raise PersistenceError("Suite lifecycle has no suite_started event", suite_id=suite_id)
        return projection


def _run_projection_from_started_event(event: RunStartedEvent) -> RunProjection:
    return RunProjection(
        run_id=event.run_id,
        experiment_id=event.experiment_id,
        suite_id=event.suite_id,
        experiment_name=event.experiment_name,
        run_number=event.run_number,
        provider_type=event.provider_type,
        provider_version=event.provider_version,
        benchmark_type=event.benchmark_type,
        benchmark_version=event.benchmark_version,
        benchmark_checksum=event.benchmark_checksum,
        answer_model_vendor=event.answer_model_vendor,
        answer_model_id=event.answer_model_id,
        judge_model_vendor=event.judge_model_vendor,
        judge_model_id=event.judge_model_id,
        started_at=to_iso8601(event.timestamp),
        methodology_version=event.methodology_version,
        methodology_profile=event.methodology_profile,
        framework_version=event.framework_version,
        runtime_environment=_json_dumps(event.runtime_environment),
        seed=event.seed,
    )


def _project_question_result(
    question: QuestionEvaluationRecord,
    retrieval_record: RetrievalRecord | None,
    response_record: Response | None,
    judgment_record: Judgment | None,
    error_messages: dict[tuple[str, QuestionErrorPhase], str],
) -> tuple[object, ...]:
    if judgment_record is not None:
        verdict = judgment_record.parsed_verdict.value
        question_score = judgment_record.parsed_score
        judgment_reasoning: str | None = judgment_record.parsed_reasoning
        judgment_id: str | None = judgment_record.judgment_id
        error_phase: QuestionErrorPhase | None = None
        error_message: str | None = None
    else:
        verdict = JudgmentVerdict.ERROR.value
        question_score = 0.0
        judgment_reasoning = None
        judgment_id = None
        error_phase = _missing_phase(retrieval_record, response_record)
        error_message = error_messages.get((question.question_id, error_phase))

    total_latency_ms = _sum_optional(
        (
            retrieval_record.retrieval_latency_ms if retrieval_record is not None else None,
            response_record.latency_ms if response_record is not None else None,
            judgment_record.latency_ms if judgment_record is not None else None,
        )
    )
    total_cost_usd = (response_record.cost_usd if response_record is not None else 0.0) + (
        judgment_record.cost_usd if judgment_record is not None else 0.0
    )

    return (
        question.question_id,
        question.run_id,
        question.question_evaluation_id,
        question.conversation_id,
        question.category.value,
        question.question_text,
        question.expected_answer,
        response_record.answer_text if response_record is not None else None,
        retrieval_record.retrieval_id if retrieval_record is not None else None,
        response_record.response_id if response_record is not None else None,
        judgment_id,
        verdict,
        question_score,
        judgment_reasoning,
        _sqlite_bool(question.is_audited_error),
        retrieval_record.n_returned if retrieval_record is not None else None,
        retrieval_record.retrieval_latency_ms if retrieval_record is not None else None,
        response_record.latency_ms if response_record is not None else None,
        judgment_record.latency_ms if judgment_record is not None else None,
        total_latency_ms,
        response_record.input_tokens if response_record is not None else None,
        response_record.output_tokens if response_record is not None else None,
        judgment_record.input_tokens if judgment_record is not None else None,
        judgment_record.output_tokens if judgment_record is not None else None,
        total_cost_usd,
        error_message,
        error_phase,
    )


def _missing_phase(
    retrieval_record: RetrievalRecord | None,
    response_record: Response | None,
) -> QuestionErrorPhase:
    if retrieval_record is None:
        return "retrieve"
    if response_record is None:
        return "generate"
    return "judge"


def _question_error_messages(
    errors: Iterable[ErrorRecord],
) -> dict[tuple[str, QuestionErrorPhase], str]:
    messages: dict[tuple[str, QuestionErrorPhase], str] = {}
    for error in errors:
        phase = _question_error_phase(error.phase)
        if error.question_id is None or phase is None:
            continue
        messages.setdefault((error.question_id, phase), error.error_message)
    return messages


def _question_error_phase(value: str) -> QuestionErrorPhase | None:
    if value in {"retrieve", "generate", "judge"}:
        return cast(QuestionErrorPhase, value)
    return None


def _sum_optional(values: Iterable[float | None]) -> float | None:
    total = 0.0
    found_value = False
    for value in values:
        if value is not None:
            total += value
            found_value = True
    return total if found_value else None


def _score_point(score: ScoreWithCI | None) -> float | None:
    return score.point_estimate if score is not None else None


def _score_low(score: ScoreWithCI | None) -> float | None:
    return score.ci_95_low if score is not None else None


def _score_high(score: ScoreWithCI | None) -> float | None:
    return score.ci_95_high if score is not None else None


def _aggregate_point(score: AggregateScore | None) -> float | None:
    return score.pooled_score.point_estimate if score is not None else None


def _aggregate_low(score: AggregateScore | None) -> float | None:
    return score.pooled_score.ci_95_low if score is not None else None


def _aggregate_high(score: AggregateScore | None) -> float | None:
    return score.pooled_score.ci_95_high if score is not None else None


def _sqlite_bool(value: bool) -> int:
    return 1 if value else 0


def _sequence(event: RunLifecycleEvent | SuiteLifecycleEvent) -> int:
    return event.sequence_number


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
