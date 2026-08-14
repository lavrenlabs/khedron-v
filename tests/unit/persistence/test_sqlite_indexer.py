from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from khedron.errors import PersistenceError
from khedron.persistence import sqlite_indexer as sqlite_indexer_module
from khedron.persistence.schemas import SCHEMA_VERSION, V1_SCHEMA_DDL
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.persistence.stage_reader import HistoricalRunClassification, preflight_historical_runs
from khedron.types import (
    APICallRecord,
    ConversationProcessedEvent,
    ErrorRecord,
    ErrorResolutionRecord,
    FailurePattern,
    Judgment,
    JudgmentVerdict,
    Memory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteStartedEvent,
    question_plan_fingerprint,
)
from khedron.utils.stats import wilson_score_interval

NOW = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)

SNAPSHOT_QUERIES = (
    "SELECT * FROM schema_meta ORDER BY version",
    "SELECT * FROM suite_lifecycle_events ORDER BY suite_id, sequence_number",
    "SELECT * FROM run_lifecycle_events ORDER BY run_id, sequence_number",
    "SELECT * FROM suites ORDER BY suite_id",
    "SELECT * FROM runs ORDER BY run_id",
    "SELECT * FROM question_evaluations ORDER BY run_id, question_id",
    "SELECT * FROM retrievals ORDER BY run_id, question_id",
    "SELECT * FROM question_results ORDER BY run_id, question_id",
    "SELECT * FROM scores_by_category ORDER BY run_id, category, mode",
    "SELECT * FROM failure_patterns ORDER BY run_id, pattern_id",
    "SELECT * FROM error_log ORDER BY run_id, error_id",
)


def create_v1_database(db_path: Path) -> None:
    """Build the complete historical v1 projection from its canonical migration DDL."""
    with closing(connect(db_path)) as connection:
        connection.executescript(V1_SCHEMA_DDL)
        connection.execute(
            "INSERT INTO schema_meta VALUES (?, ?, ?)", (1, "2026-01-01T00:00:00Z", "0.1.0")
        )
        connection.execute(
            "INSERT INTO retrievals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ret-legacy", "qe-legacy", "run-legacy", "q-legacy", "now", "q", 1, 0, 1.0),
        )
        connection.execute(
            "INSERT INTO error_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("err-legacy", "run-legacy", "now", "generate", "q-legacy", "Timeout", "boom", 0),
        )
        connection.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(record.model_dump_json())
            file.write("\n")


def scalar_int(db_path: Path, query: str, params: tuple[object, ...] = ()) -> int:
    with closing(connect(db_path)) as connection:
        row = connection.execute(query, params).fetchone()
    if row is None:
        raise AssertionError(query)
    value = row[0]
    if not isinstance(value, int):
        raise AssertionError(value)
    return value


def fetch_one(
    db_path: Path,
    query: str,
    params: tuple[object, ...] = (),
) -> sqlite3.Row:
    with closing(connect(db_path)) as connection:
        row = connection.execute(query, params).fetchone()
    if row is None:
        raise AssertionError(query)
    return row


def capture_sqlite_state(db_path: Path) -> dict[str, list[tuple[object, ...]]]:
    state: dict[str, list[tuple[object, ...]]] = {}
    with closing(connect(db_path)) as connection:
        for query in SNAPSHOT_QUERIES:
            rows = connection.execute(query).fetchall()
            state[query] = [tuple(row) for row in rows]
    return state


def score(n_total: int = 1, n_correct: int = 1) -> ScoreWithCI:
    low, high = wilson_score_interval(n_correct, n_total)
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_total - n_correct,
        point_estimate=n_correct / n_total,
        ci_95_low=low,
        ci_95_high=high,
    )


def suite_started_event(suite_id: str = "suite-1") -> SuiteStartedEvent:
    return SuiteStartedEvent(
        event_id=f"{suite_id}-started",
        timestamp=NOW,
        suite_id=suite_id,
        sequence_number=0,
        config_yaml_path="experiments/synthetic.yaml",
        config_yaml_content="experiments: []",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        runtime_environment={"python": "3.11", "os": "Windows"},
    )


def write_suite(results_root: Path, suite_id: str = "suite-1", completed: bool = True) -> None:
    records: list[BaseModel] = [suite_started_event(suite_id)]
    if completed:
        records.append(
            SuiteCompletedEvent(
                event_id=f"{suite_id}-completed",
                timestamp=NOW,
                suite_id=suite_id,
                sequence_number=1,
                total_cost_usd=0.03,
                n_experiments_completed=1,
                n_experiments_failed=0,
            )
        )
    write_jsonl(results_root / "suites" / suite_id / "lifecycle.jsonl", records)


def run_started_event(run_id: str = "run-1", suite_id: str = "suite-1") -> RunStartedEvent:
    return RunStartedEvent(
        event_id=f"{run_id}-started",
        timestamp=NOW,
        run_id=run_id,
        sequence_number=0,
        suite_id=suite_id,
        experiment_id="experiment-1",
        experiment_name="Synthetic experiment",
        run_number=0,
        provider_type="full_context",
        provider_version="0.1.0",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="sha256:synthetic",
        answer_model_id="gpt-4o-mini-2024-07-18",
        answer_model_vendor="openai",
        judge_model_id="gpt-4o-2024-08-06",
        judge_model_vendor="openai",
        config={"provider": {"type": "full_context"}},
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=123,
        runtime_environment={"python": "3.11", "os": "Windows"},
    )


def question_evaluation(
    run_id: str = "run-1",
    question_id: str = "q-1",
    question_evaluation_id: str = "qe-1",
) -> QuestionEvaluationRecord:
    return QuestionEvaluationRecord(
        question_evaluation_id=question_evaluation_id,
        run_id=run_id,
        question_id=question_id,
        conversation_id="conversation-1",
        category="single_hop",
        question_text=f"What is the answer for {question_id}?",
        expected_answer="Rome",
        is_audited_error=False,
        timestamp=NOW,
    )


def retrieval(
    run_id: str = "run-1",
    question_id: str = "q-1",
    question_evaluation_id: str = "qe-1",
    retrieval_id: str = "ret-1",
) -> RetrievalRecord:
    return RetrievalRecord(
        retrieval_id=retrieval_id,
        question_evaluation_id=question_evaluation_id,
        run_id=run_id,
        question_id=question_id,
        timestamp=NOW,
        query="Where did Alice move?",
        top_k=10,
        n_returned=1,
        memories=[
            Memory(
                memory_id="memory-1",
                content="Alice moved to Rome.",
                metadata={"speaker": "Alice"},
                score=0.9,
                timestamp=NOW,
            )
        ],
        retrieval_latency_ms=12.5,
    )


def response(
    run_id: str = "run-1",
    question_id: str = "q-1",
    retrieval_id: str = "ret-1",
    response_id: str = "resp-1",
) -> Response:
    return Response(
        response_id=response_id,
        run_id=run_id,
        question_id=question_id,
        retrieval_id=retrieval_id,
        timestamp=NOW,
        model_id="gpt-4o-mini-2024-07-18",
        prompt="Memories: Alice moved to Rome.",
        answer_text="Rome",
        input_tokens=32,
        output_tokens=3,
        latency_ms=200.0,
        cost_usd=0.01,
        raw_api_response={"id": "api-response-1"},
    )


def judgment(
    run_id: str = "run-1",
    question_id: str = "q-1",
    response_id: str = "resp-1",
    judgment_id: str = "judgment-1",
) -> Judgment:
    return Judgment(
        judgment_id=judgment_id,
        run_id=run_id,
        response_id=response_id,
        question_id=question_id,
        timestamp=NOW,
        judge_model_id="gpt-4o-2024-08-06",
        prompt="Judge this answer.",
        raw_judge_output='{"verdict": "correct"}',
        parsed_verdict=JudgmentVerdict.CORRECT,
        parsed_score=1.0,
        parsed_reasoning="The answer matches.",
        parse_was_successful=True,
        input_tokens=50,
        output_tokens=10,
        latency_ms=180.0,
        cost_usd=0.02,
    )


def write_happy_path_run(results_root: Path, run_id: str = "run-1") -> None:
    suite_id = "suite-1"
    write_suite(results_root, suite_id)
    write_jsonl(
        results_root / "runs" / run_id / "lifecycle.jsonl",
        [
            run_started_event(run_id, suite_id),
            ConversationProcessedEvent(
                event_id=f"{run_id}-conversation-processed",
                timestamp=NOW,
                run_id=run_id,
                sequence_number=1,
                conversation_id="conversation-1",
                n_questions_evaluated=1,
                n_questions_correct=1,
                n_questions_errored=0,
                cost_usd=0.03,
            ),
            RunCompletedEvent(
                event_id=f"{run_id}-completed",
                timestamp=NOW,
                run_id=run_id,
                sequence_number=2,
                status="completed",
                n_questions_attempted=1,
                n_questions_succeeded=1,
                n_questions_errored=0,
                overall_score_standard=score(),
                overall_score_audited=score(),
                by_category_standard={"single_hop": score()},
                by_category_audited={"single_hop": score()},
                total_cost_usd=0.03,
            ),
        ],
    )
    write_jsonl(
        results_root / "runs" / run_id / "question_evaluations.jsonl",
        [question_evaluation(run_id)],
    )
    write_jsonl(results_root / "runs" / run_id / "retrievals.jsonl", [retrieval(run_id)])
    write_jsonl(results_root / "runs" / run_id / "responses.jsonl", [response(run_id)])
    write_jsonl(results_root / "runs" / run_id / "judgments.jsonl", [judgment(run_id)])
    write_jsonl(
        results_root / "runs" / run_id / "failure_patterns.jsonl",
        [
            FailurePattern(
                pattern_id="pattern-1",
                run_id=run_id,
                pattern_name="missing_memory_failure",
                description="No relevant memory was retrieved.",
                suggested_remedy="Improve retrieval recall.",
                n_affected_questions=1,
                affected_question_ids=["q-2"],
                confidence=0.9,
            )
        ],
    )
    write_jsonl(
        results_root / "runs" / run_id / "errors.jsonl",
        [
            ErrorRecord(
                error_id="error-1",
                run_id=run_id,
                timestamp=NOW,
                phase="generate",
                question_id="q-2",
                error_type="ModelTimeoutError",
                error_message="Timed out.",
                stack_trace=None,
                context={"attempt": 1},
                recovered=True,
            )
        ],
    )
    write_jsonl(
        results_root / "runs" / run_id / "api_calls.jsonl",
        [
            APICallRecord(
                api_call_id="api-1",
                run_id=run_id,
                question_id="q-1",
                timestamp=NOW,
                phase="generate",
                vendor="openai",
                model_id="gpt-4o-mini-2024-07-18",
                input_tokens=32,
                output_tokens=3,
                latency_ms=200.0,
                cost_usd=0.01,
                status="success",
                attempt_number=1,
            )
        ],
    )


def write_partial_projection_run(results_root: Path, run_id: str = "run-partial") -> None:
    suite_id = "suite-partial"
    write_suite(results_root, suite_id, completed=False)
    write_jsonl(
        results_root / "runs" / run_id / "lifecycle.jsonl",
        [run_started_event(run_id, suite_id)],
    )
    write_jsonl(
        results_root / "runs" / run_id / "question_evaluations.jsonl",
        [
            question_evaluation(run_id, "q-retrieve", "qe-retrieve"),
            question_evaluation(run_id, "q-generate", "qe-generate"),
            question_evaluation(run_id, "q-judge", "qe-judge"),
        ],
    )
    write_jsonl(
        results_root / "runs" / run_id / "retrievals.jsonl",
        [
            retrieval(run_id, "q-generate", "qe-generate", "ret-generate"),
            retrieval(run_id, "q-judge", "qe-judge", "ret-judge"),
        ],
    )
    write_jsonl(
        results_root / "runs" / run_id / "responses.jsonl",
        [response(run_id, "q-judge", "ret-judge", "resp-judge")],
    )
    write_jsonl(
        results_root / "runs" / run_id / "errors.jsonl",
        [
            ErrorRecord(
                error_id="error-retrieve",
                run_id=run_id,
                timestamp=NOW,
                phase="retrieve",
                question_id="q-retrieve",
                error_type="ProviderTimeoutError",
                error_message="Retrieval failed.",
                stack_trace=None,
                context={},
                recovered=False,
            ),
            ErrorRecord(
                error_id="error-generate",
                run_id=run_id,
                timestamp=NOW,
                phase="generate",
                question_id="q-generate",
                error_type="ModelTimeoutError",
                error_message="Generation failed.",
                stack_trace=None,
                context={},
                recovered=False,
            ),
            ErrorRecord(
                error_id="error-judge",
                run_id=run_id,
                timestamp=NOW,
                phase="judge",
                question_id="q-judge",
                error_type="JudgeMalformedResponseError",
                error_message="Judgment failed.",
                stack_trace=None,
                context={},
                recovered=False,
            ),
        ],
    )


def test_initialize_creates_tables_and_records_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "results" / "benchmark.db"
    indexer = SqliteIndexer(db_path)

    indexer.initialize()

    if not db_path.exists():
        raise AssertionError(db_path)
    if indexer.get_schema_version() != SCHEMA_VERSION:
        raise AssertionError(indexer.get_schema_version())
    if scalar_int(db_path, "SELECT COUNT(*) FROM schema_meta") != 1:
        raise AssertionError("schema_meta row was not created")
    if scalar_int(db_path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'runs'") != 1:
        raise AssertionError("runs table was not created")


def test_initialize_migrates_v1_in_place_and_preserves_metadata_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    create_v1_database(db_path)

    SqliteIndexer(db_path).initialize()

    meta = fetch_one(db_path, "SELECT * FROM schema_meta")
    if tuple(meta) != (SCHEMA_VERSION, "2026-01-01T00:00:00Z", "0.1.0"):
        raise AssertionError(tuple(meta))
    retrieval = fetch_one(
        db_path,
        "SELECT recovery_attempt_id, ingestion_attempt_id FROM retrievals WHERE retrieval_id = ?",
        ("ret-legacy",),
    )
    if tuple(retrieval) != (None, None):
        raise AssertionError(tuple(retrieval))
    retryable = fetch_one(
        db_path, "SELECT retryable FROM error_log WHERE error_id = ?", ("err-legacy",)
    )[0]
    if retryable is not None:
        raise AssertionError(retryable)
    for table in ("question_plan", "recovery_attempts", "error_resolutions"):
        if scalar_int(db_path, "SELECT COUNT(*) FROM sqlite_master WHERE name = ?", (table,)) != 1:
            raise AssertionError(table)


def test_get_schema_version_migrates_a_complete_v1_projection(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    create_v1_database(db_path)

    if SqliteIndexer(db_path).get_schema_version() != SCHEMA_VERSION:
        raise AssertionError("CLI/indexer version route did not migrate v1")


def test_initialize_rejects_malformed_v1_structure_without_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    create_v1_database(db_path)
    with closing(connect(db_path)) as connection:
        connection.execute("ALTER TABLE error_log DROP COLUMN recovered")
        connection.commit()
    before = db_path.read_bytes()

    with pytest.raises(PersistenceError, match="v1 database has an unreadable"):
        SqliteIndexer(db_path).initialize()

    if db_path.read_bytes() != before:
        raise AssertionError("malformed v1 database was mutated")


def test_initialize_rejects_nonempty_database_without_metadata_without_mutation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "unknown.db"
    with closing(connect(db_path)) as connection:
        connection.execute("CREATE TABLE user_data (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO user_data VALUES ('preserve')")
        connection.commit()
    before = db_path.read_bytes()

    try:
        SqliteIndexer(db_path).initialize()
    except PersistenceError:
        pass
    else:
        raise AssertionError("expected malformed database to fail closed")

    if db_path.read_bytes() != before:
        raise AssertionError("malformed database was mutated")


def test_initialize_rolls_back_a_failed_v1_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "legacy.db"
    create_v1_database(db_path)
    indexer = SqliteIndexer(db_path)

    def fail_after_first_change(connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE error_log ADD COLUMN retryable INTEGER")
        raise sqlite3.OperationalError("forced migration failure")

    monkeypatch.setattr(indexer, "_migrate_v1_to_v2", fail_after_first_change)
    with pytest.raises(sqlite3.OperationalError, match="forced migration failure"):
        indexer.initialize()

    if scalar_int(db_path, "SELECT version FROM schema_meta") != 1:
        raise AssertionError("migration updated metadata before committing")
    with closing(connect(db_path)) as connection:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(error_log)").fetchall()
        }
    if "retryable" in columns:
        raise AssertionError("failed migration retained an added column")


@pytest.mark.parametrize("version", [3, 99])
def test_public_version_routes_reject_future_metadata_without_mutation(
    tmp_path: Path,
    version: int,
) -> None:
    db_path = tmp_path / "future.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    with closing(connect(db_path)) as connection:
        connection.execute("UPDATE schema_meta SET version = ?", (version,))
        connection.commit()
    before = db_path.read_bytes()

    with pytest.raises(PersistenceError):
        indexer.read_schema_version()
    with pytest.raises(PersistenceError):
        indexer.get_schema_version()

    if db_path.read_bytes() != before:
        raise AssertionError("rejected future schema was mutated")


def test_public_version_routes_reject_multiple_metadata_without_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "multiple-meta.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    with closing(connect(db_path)) as connection:
        connection.execute(
            "INSERT INTO schema_meta VALUES (?, ?, ?)", (SCHEMA_VERSION, "later", "x")
        )
        connection.commit()
    before = db_path.read_bytes()

    with pytest.raises(PersistenceError):
        indexer.read_schema_version()
    with pytest.raises(PersistenceError):
        indexer.get_schema_version()

    if db_path.read_bytes() != before:
        raise AssertionError("rejected multiple metadata rows were mutated")


def test_historical_v1_ddl_is_a_frozen_migration_contract() -> None:
    """A v2 edit must not silently redefine the on-disk v1 migration input."""
    actual_digest = sha256(V1_SCHEMA_DDL.encode("utf-8")).hexdigest()
    expected_digest = "21f9d2ac1b14bac2a75cb39eb83f6bd3732e28130c541cf04736152865e377fd"
    if actual_digest != expected_digest:
        raise AssertionError("the historical v1 migration DDL changed")


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("benchmark_checksum TEXT NOT NULL", "benchmark_checksum BLOB NOT NULL"),
        ("benchmark_checksum TEXT NOT NULL", "benchmark_checksum TEXT"),
        ("run_id TEXT PRIMARY KEY", "run_id TEXT"),
        ("error_id TEXT PRIMARY KEY", "error_id TEXT NOT NULL UNIQUE"),
        (
            "FOREIGN KEY (run_id) REFERENCES runs(run_id)",
            "FOREIGN KEY (run_id) REFERENCES suites(suite_id)",
        ),
    ],
    ids=["declared-type", "not-null", "primary-key", "unique", "foreign-key"],
)
def test_schema_signature_rejects_table_contract_mutations_without_mutation(
    tmp_path: Path,
    needle: str,
    replacement: str,
) -> None:
    db_path = tmp_path / "mutated.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    with closing(connect(db_path)) as connection:
        # SQLite's supported ALTER TABLE cannot modify these declarations. Editing the stored
        # schema text models a corrupted database definition directly; the validator must reject
        # it before any repair/migration writes are attempted.
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = replace(sql, ?, ?) "
            "WHERE type = 'table' AND instr(sql, ?) > 0",
            (needle, replacement, needle),
        )
        connection.execute("PRAGMA writable_schema = OFF")
        connection.commit()
    before = db_path.read_bytes()

    with pytest.raises(PersistenceError):
        indexer.read_schema_version()
    with pytest.raises(PersistenceError):
        indexer.initialize()

    if db_path.read_bytes() != before:
        raise AssertionError("schema-signature rejection mutated the database")


@pytest.mark.parametrize(
    "index_name",
    ["idx_runs_started_at", "sqlite_autoindex_error_log_1"],
    ids=["explicit-index", "unique-constraint-index"],
)
def test_schema_signature_rejects_index_mutations_without_mutation(
    tmp_path: Path,
    index_name: str,
) -> None:
    db_path = tmp_path / "mutated-index.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    with closing(connect(db_path)) as connection:
        if index_name.startswith("sqlite_autoindex"):
            # SQLite forbids dropping an autoindex created by a UNIQUE/PK constraint. Removing
            # the owning constraint from its stored declaration changes the generated-index
            # contract without relying on unsupported DDL.
            connection.execute("PRAGMA writable_schema = ON")
            connection.execute(
                "UPDATE sqlite_master SET sql = replace(sql, 'error_id TEXT PRIMARY KEY', "
                "'error_id TEXT NOT NULL') WHERE name = 'error_log'"
            )
            connection.execute("PRAGMA writable_schema = OFF")
        else:
            connection.execute(f"DROP INDEX {index_name}")
        connection.commit()
    before = db_path.read_bytes()

    with pytest.raises(PersistenceError):
        indexer.read_schema_version()
    with pytest.raises(PersistenceError):
        indexer.initialize()

    if db_path.read_bytes() != before:
        raise AssertionError("schema-signature rejection mutated the database")


def test_error_projection_preserves_retryability_tristate(tmp_path: Path) -> None:
    db_path = tmp_path / "errors.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    records = [
        ErrorRecord(
            error_id=f"error-{value}",
            run_id="run-1",
            timestamp=NOW,
            phase="generate",
            question_id="q-1",
            error_type="TestError",
            error_message="test",
            stack_trace=None,
            context={},
            recovered=False,
            retryable=value,
        )
        for value in (True, False, None)
    ]
    with closing(connect(db_path)) as connection:
        indexer._insert_errors(connection, records)
        connection.commit()
        values = [
            row[0]
            for row in connection.execute("SELECT retryable FROM error_log ORDER BY error_id")
        ]
    if values != [0, None, 1]:
        raise AssertionError(values)


def test_indexer_write_connection_enables_wal_read_connection_does_not(tmp_path: Path) -> None:
    # WAL is enabled only on write connections (create/index paths). Journal mode is a
    # persistent file property, so reads benefit without a read connection ever switching
    # the mode on a possibly read-only DB (the get_schema_version / `db version` path).
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    write_indexer = SqliteIndexer(results_dir / "benchmark.db")
    with closing(write_indexer._connect(write=True)) as connection:
        write_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        write_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    if str(write_mode).lower() != "wal":
        raise AssertionError(write_mode)
    if write_timeout != 30000:
        raise AssertionError(write_timeout)

    # A read connection carries the busy timeout but never issues PRAGMA journal_mode=WAL,
    # so a fresh DB opened for reading stays in its default (non-WAL) journal mode.
    read_indexer = SqliteIndexer(results_dir / "fresh.db")
    with closing(read_indexer._connect()) as connection:
        read_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        read_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    if str(read_mode).lower() == "wal":
        raise AssertionError(read_mode)
    if read_timeout != 30000:
        raise AssertionError(read_timeout)


def test_index_run_writes_happy_path_rows_and_ignores_api_calls(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)

    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")

    run_row = fetch_one(db_path, "SELECT * FROM runs WHERE run_id = ?", ("run-1",))
    if run_row["status"] != "completed":
        raise AssertionError(run_row["status"])
    if run_row["overall_score_standard"] != 1.0:
        raise AssertionError(run_row["overall_score_standard"])
    if scalar_int(db_path, "SELECT COUNT(*) FROM run_lifecycle_events WHERE run_id = 'run-1'") != 3:
        raise AssertionError("run lifecycle events were not indexed")
    if scalar_int(db_path, "SELECT COUNT(*) FROM question_evaluations WHERE run_id = 'run-1'") != 1:
        raise AssertionError("question evaluations were not indexed")
    if scalar_int(db_path, "SELECT COUNT(*) FROM retrievals WHERE run_id = 'run-1'") != 1:
        raise AssertionError("retrievals were not indexed")
    if scalar_int(db_path, "SELECT COUNT(*) FROM scores_by_category WHERE run_id = 'run-1'") != 2:
        raise AssertionError("category scores were not indexed")
    if scalar_int(db_path, "SELECT COUNT(*) FROM failure_patterns WHERE run_id = 'run-1'") != 1:
        raise AssertionError("failure patterns were not indexed")
    if scalar_int(db_path, "SELECT COUNT(*) FROM error_log WHERE run_id = 'run-1'") != 1:
        raise AssertionError("errors were not indexed")

    question_row = fetch_one(
        db_path,
        "SELECT * FROM question_results WHERE run_id = ? AND question_id = ?",
        ("run-1", "q-1"),
    )
    if question_row["verdict"] != "correct":
        raise AssertionError(question_row["verdict"])
    if question_row["score"] != 1.0:
        raise AssertionError(question_row["score"])
    if question_row["error_phase"] is not None:
        raise AssertionError(question_row["error_phase"])
    if question_row["total_cost_usd"] != 0.03:
        raise AssertionError(question_row["total_cost_usd"])
    if scalar_int(db_path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'api_calls'") != 0:
        raise AssertionError("api_calls.jsonl should not create a SQLite table")


def test_retrieval_recovery_and_ingestion_ids_round_trip_to_sqlite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    path = results_root / "runs" / "run-1" / "retrievals.jsonl"
    record = RetrievalRecord.model_validate_json(
        path.read_text(encoding="utf-8").strip()
    ).model_copy(update={"recovery_attempt_id": "attempt-r", "ingestion_attempt_id": "ingestion-r"})
    write_jsonl(path, [record])

    SqliteIndexer(db_path).index_run("run-1")

    row = fetch_one(
        db_path,
        "SELECT recovery_attempt_id, ingestion_attempt_id FROM retrievals WHERE retrieval_id = ?",
        ("ret-1",),
    )
    if tuple(row) != ("attempt-r", "ingestion-r"):
        raise AssertionError(dict(row))


@pytest.mark.parametrize(
    ("recovery_attempt_id", "ingestion_attempt_id", "expected"),
    [
        ("attempt-only", None, ("attempt-only", None)),
        (None, "ingestion-only", (None, "ingestion-only")),
    ],
)
def test_retrieval_id_columns_preserve_mixed_null_states(
    tmp_path: Path,
    recovery_attempt_id: str | None,
    ingestion_attempt_id: str | None,
    expected: tuple[str | None, str | None],
) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    path = results_root / "runs" / "run-1" / "retrievals.jsonl"
    retrieval_record = RetrievalRecord.model_validate_json(
        path.read_text(encoding="utf-8").strip()
    ).model_copy(
        update={
            "recovery_attempt_id": recovery_attempt_id,
            "ingestion_attempt_id": ingestion_attempt_id,
        }
    )
    write_jsonl(path, [retrieval_record])
    SqliteIndexer(db_path).index_run("run-1")

    row = fetch_one(
        db_path,
        "SELECT recovery_attempt_id, ingestion_attempt_id FROM retrievals WHERE retrieval_id = ?",
        ("ret-1",),
    )
    if tuple(row) != expected:
        raise AssertionError(dict(row))


def test_partial_question_projection_marks_failed_phase(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_partial_projection_run(results_root)

    SqliteIndexer(db_path).index_run("run-partial")

    with closing(connect(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT question_id, retrieval_id, response_id, judgment_id, verdict, score,
                   error_phase, error_message
            FROM question_results
            WHERE run_id = ?
            ORDER BY question_id
            """,
            ("run-partial",),
        ).fetchall()
    by_question = {str(row["question_id"]): row for row in rows}

    retrieve_row = by_question["q-retrieve"]
    if retrieve_row["retrieval_id"] is not None or retrieve_row["error_phase"] != "retrieve":
        raise AssertionError(dict(retrieve_row))
    if retrieve_row["error_message"] != "Retrieval failed.":
        raise AssertionError(retrieve_row["error_message"])

    generate_row = by_question["q-generate"]
    if generate_row["response_id"] is not None or generate_row["error_phase"] != "generate":
        raise AssertionError(dict(generate_row))
    if generate_row["error_message"] != "Generation failed.":
        raise AssertionError(generate_row["error_message"])

    judge_row = by_question["q-judge"]
    if judge_row["judgment_id"] is not None or judge_row["error_phase"] != "judge":
        raise AssertionError(dict(judge_row))
    if judge_row["error_message"] != "Judgment failed.":
        raise AssertionError(judge_row["error_message"])

    for row in rows:
        if row["verdict"] != "error" or row["score"] != 0.0:
            raise AssertionError(dict(row))


def test_index_run_is_idempotent_for_same_run(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)

    indexer.index_run("run-1")
    first_state = capture_sqlite_state(db_path)
    indexer.index_run("run-1")
    second_state = capture_sqlite_state(db_path)

    if second_state != first_state:
        raise AssertionError(second_state)


def test_reindex_is_idempotent_for_new_projection_tables(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    ids = ("q-1",)
    plan = QuestionPlanRecord(
        run_id="run-1",
        benchmark_id="locomo",
        categories=("single_hop",),
        corpus_checksum="sha256:synthetic",
        question_ids=ids,
        fingerprint=question_plan_fingerprint(
            benchmark_id="locomo",
            categories=("single_hop",),
            corpus_checksum="sha256:synthetic",
            question_ids=ids,
        ),
        timestamp=NOW,
    )
    write_jsonl(results_root / "runs" / "run-1" / "question_plan.jsonl", [plan])
    recovered_response = response().model_copy(update={"recovery_attempt_id": "attempt-1"})
    write_jsonl(results_root / "runs" / "run-1" / "responses.jsonl", [recovered_response])
    error = ErrorRecord(
        error_id="error-1",
        run_id="run-1",
        timestamp=NOW,
        phase="generate",
        question_id="q-1",
        error_type="Test",
        error_message="retry",
        stack_trace=None,
        context={},
        recovered=False,
        retryable=True,
    )
    write_jsonl(results_root / "runs" / "run-1" / "errors.jsonl", [error])
    attempt = RecoveryAttemptRecord(
        attempt_id="attempt-1",
        run_id="run-1",
        question_id="q-1",
        error_id="error-1",
        stage="generate",
        timestamp=NOW,
    )
    write_jsonl(results_root / "runs" / "run-1" / "recovery_attempts.jsonl", [attempt])
    resolution = ErrorResolutionRecord(
        resolution_id="resolution-1",
        run_id="run-1",
        question_id="q-1",
        error_id="error-1",
        recovery_attempt_id="attempt-1",
        resolved_by_stage="generate",
        resolved_by_stage_record_id="resp-1",
        timestamp=NOW,
    )
    write_jsonl(results_root / "runs" / "run-1" / "error_resolutions.jsonl", [resolution])
    with closing(connect(db_path)) as connection:
        connection.execute("SELECT 1")
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    indexer.index_run("run-1")
    if (
        scalar_int(db_path, "SELECT COUNT(*) FROM question_plan") != 1
        or scalar_int(db_path, "SELECT COUNT(*) FROM recovery_attempts") != 1
        or scalar_int(db_path, "SELECT COUNT(*) FROM error_resolutions") != 1
    ):
        raise AssertionError("new-table reindex was not idempotent")


def test_corrupt_strict_stream_does_not_create_sqlite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    duplicate = response("run-1")
    write_jsonl(results_root / "runs" / "run-1" / "responses.jsonl", [duplicate, duplicate])

    with pytest.raises(PersistenceError):
        SqliteIndexer(db_path).index_run("run-1")

    if db_path.exists() or db_path.with_name("benchmark.db-wal").exists():
        raise AssertionError("corrupt JSONL created a SQLite projection")


def test_duplicate_failure_pattern_is_rejected_before_sqlite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    pattern_path = results_root / "runs" / "run-1" / "failure_patterns.jsonl"
    pattern = FailurePattern.model_validate_json(pattern_path.read_text(encoding="utf-8").strip())
    write_jsonl(pattern_path, [pattern, pattern])

    with pytest.raises(PersistenceError):
        SqliteIndexer(db_path).index_run("run-1")

    if db_path.exists():
        raise AssertionError("duplicate failure pattern created SQLite")


@pytest.mark.parametrize(
    "file_name, records, missing_parent_file",
    [
        ("responses.jsonl", [response()], "retrievals.jsonl"),
        ("judgments.jsonl", [judgment()], "responses.jsonl"),
        ("retrievals.jsonl", [retrieval(question_evaluation_id="missing")], None),
    ],
    ids=["orphan-generate", "orphan-judge", "broken-retrieval-parent"],
)
def test_strict_reader_rejects_orphan_and_broken_stage_links(
    tmp_path: Path, file_name: str, records: list[BaseModel], missing_parent_file: str | None
) -> None:
    results_root = tmp_path / "results"
    write_happy_path_run(results_root)
    write_jsonl(results_root / "runs" / "run-1" / file_name, records)
    if missing_parent_file is not None:
        write_jsonl(results_root / "runs" / "run-1" / missing_parent_file, [])
    with pytest.raises(PersistenceError):
        SqliteIndexer(results_root / "benchmark.db").index_run("run-1")


@pytest.mark.parametrize(
    "remove_files",
    [
        ("retrievals.jsonl", "responses.jsonl", "judgments.jsonl"),
        ("responses.jsonl", "judgments.jsonl"),
        ("judgments.jsonl",),
        (),
    ],
    ids=["evaluation", "evaluation-retrieve", "evaluation-retrieve-generate", "full"],
)
def test_strict_reader_accepts_all_valid_stage_prefixes(
    tmp_path: Path, remove_files: tuple[str, ...]
) -> None:
    results_root = tmp_path / "results"
    write_happy_path_run(results_root)
    for file_name in remove_files:
        write_jsonl(results_root / "runs" / "run-1" / file_name, [])
    SqliteIndexer(results_root / "benchmark.db").index_run("run-1")


def test_historical_preflight_classifies_legacy_and_corrupt_once_per_run(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    write_happy_path_run(results_root, "legacy")
    write_happy_path_run(results_root, "corrupt")
    duplicate = response("corrupt")
    write_jsonl(results_root / "runs" / "corrupt" / "responses.jsonl", [duplicate, duplicate])

    findings = preflight_historical_runs(results_root)
    if [(item.run_id, item.classification) for item in findings] != [
        ("corrupt", HistoricalRunClassification.CORRUPT),
        ("legacy", HistoricalRunClassification.LEGACY_INELIGIBLE),
    ]:
        raise AssertionError(findings)


def test_duplicate_manifest_question_id_is_rejected_before_sqlite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    ids = ("q-1", "q-1")
    write_jsonl(
        results_root / "runs" / "run-1" / "question_plan.jsonl",
        [
            QuestionPlanRecord(
                run_id="run-1",
                benchmark_id="locomo",
                categories=("single_hop",),
                corpus_checksum="sha256:synthetic",
                question_ids=ids,
                fingerprint=question_plan_fingerprint(
                    benchmark_id="locomo",
                    categories=("single_hop",),
                    corpus_checksum="sha256:synthetic",
                    question_ids=ids,
                ),
                timestamp=NOW,
            )
        ],
    )
    with pytest.raises(PersistenceError):
        SqliteIndexer(db_path).index_run("run-1")
    if db_path.exists():
        raise AssertionError("invalid manifest created SQLite")


def test_rebuild_all_is_idempotent(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    write_partial_projection_run(results_root)
    indexer = SqliteIndexer(db_path)

    indexer.rebuild_all()
    first_state = capture_sqlite_state(db_path)
    indexer.rebuild_all()
    second_state = capture_sqlite_state(db_path)

    if second_state != first_state:
        raise AssertionError(second_state)


def test_corrupt_rebuild_preserves_existing_projection_bytes(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    before = db_path.read_bytes()
    duplicate = response("run-1")
    write_jsonl(results_root / "runs" / "run-1" / "responses.jsonl", [duplicate, duplicate])

    with pytest.raises(PersistenceError):
        indexer.rebuild_all()
    if db_path.read_bytes() != before:
        raise AssertionError("corrupt rebuild changed the existing SQLite projection")
    wal_path = db_path.with_name("benchmark.db-wal")
    if wal_path.exists() and wal_path.read_bytes() != b"":
        raise AssertionError("corrupt rebuild left a WAL sidecar mutation")


def test_index_run_projects_captured_snapshot_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    original_capture = indexer._capture_snapshot

    def capture_then_mutate(run_id: str) -> object:
        snapshot = original_capture(run_id)
        write_jsonl(results_root / "runs" / run_id / "responses.jsonl", [])
        write_jsonl(results_root / "runs" / run_id / "errors.jsonl", [])
        write_jsonl(results_root / "runs" / run_id / "failure_patterns.jsonl", [])
        (results_root / "runs" / run_id / "lifecycle.jsonl").write_text(
            "{mutated-run-lifecycle}\n", encoding="utf-8"
        )
        (results_root / "suites" / "suite-1" / "lifecycle.jsonl").write_text(
            "{mutated-suite-lifecycle}\n", encoding="utf-8"
        )
        return snapshot

    monkeypatch.setattr(indexer, "_capture_snapshot", capture_then_mutate)
    indexer.index_run("run-1")

    if (
        scalar_int(db_path, "SELECT COUNT(*) FROM question_results WHERE response_id = 'resp-1'")
        != 1
    ):
        raise AssertionError(
            "projection re-read mutated responses instead of its captured snapshot"
        )
    if scalar_int(db_path, "SELECT COUNT(*) FROM error_log WHERE error_id = 'error-1'") != 1:
        raise AssertionError("projection re-read mutated errors instead of its captured snapshot")
    if (
        scalar_int(db_path, "SELECT COUNT(*) FROM failure_patterns WHERE pattern_id = 'pattern-1'")
        != 1
    ):
        raise AssertionError("projection re-read failure patterns instead of its captured snapshot")


def test_rebuild_replaces_target_and_removes_stale_sidecars(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    SqliteIndexer(db_path).index_run("run-1")
    wal_path = db_path.with_name("benchmark.db-wal")
    shm_path = db_path.with_name("benchmark.db-shm")
    wal_path.write_bytes(b"stale-wal")
    shm_path.write_bytes(b"stale-shm")
    SqliteIndexer(db_path).rebuild_all()

    with closing(connect(db_path)) as connection:
        if connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] != 1:
            raise AssertionError("promoted database is not openable")
    if wal_path.exists() or shm_path.exists():
        raise AssertionError("successful promotion retained stale WAL/SHM sidecars")


def test_rebuild_fails_closed_when_post_promotion_sidecar_backup_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    wal_path = db_path.with_name("benchmark.db-wal")
    shm_path = db_path.with_name("benchmark.db-shm")
    original_read_meta = indexer._read_schema_meta_from_disk

    def read_meta_then_add_sidecars() -> object:
        meta = original_read_meta()
        wal_path.write_bytes(b"stale-wal")
        shm_path.write_bytes(b"stale-shm")
        return meta

    original_unlink = Path.unlink

    def fail_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == ".benchmark.db-wal.pre-rebuild":
            raise OSError("injected backup cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    monkeypatch.setattr(indexer, "_read_schema_meta_from_disk", read_meta_then_add_sidecars)

    with pytest.raises(PersistenceError, match="could not remove retired sidecar backup"):
        indexer.rebuild_all()

    if wal_path.exists() or shm_path.exists():
        raise AssertionError(
            "active WAL/SHM names retained stale sidecar data after failed cleanup"
        )
    backup = db_path.with_name(".benchmark.db-wal.pre-rebuild")
    if not backup.exists() or backup.read_bytes() != b"stale-wal":
        raise AssertionError("recoverable WAL backup was not retained")
    with closing(connect(db_path)) as connection:
        if connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] != 1:
            raise AssertionError("promoted database is not readable after backup cleanup failure")


def test_rebuild_restores_original_sidecars_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    wal_path = db_path.with_name("benchmark.db-wal")
    shm_path = db_path.with_name("benchmark.db-shm")
    original_read_meta = indexer._read_schema_meta_from_disk
    original_replace = sqlite_indexer_module.os.replace

    def read_meta_then_add_sidecars() -> object:
        meta = original_read_meta()
        wal_path.write_bytes(b"old-wal")
        shm_path.write_bytes(b"old-shm")
        return meta

    def fail_main_promotion(source: Path, target: Path) -> None:
        if source.name.startswith(".khedron-rebuild-"):
            raise OSError("injected pre-promotion failure")
        original_replace(source, target)

    monkeypatch.setattr(indexer, "_read_schema_meta_from_disk", read_meta_then_add_sidecars)
    monkeypatch.setattr(sqlite_indexer_module.os, "replace", fail_main_promotion)

    with pytest.raises(OSError, match="injected pre-promotion failure"):
        indexer.rebuild_all()

    if wal_path.read_bytes() != b"old-wal" or shm_path.read_bytes() != b"old-shm":
        raise AssertionError("unpromoted rebuild did not restore original sidecars")


def test_rebuild_projects_captured_suite_and_experiment_inputs_after_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    experiments = results_root / "suites" / "suite-1" / "experiments.jsonl"
    experiments.write_text("", encoding="utf-8")
    indexer = SqliteIndexer(db_path)
    original_read_records = indexer._read_records

    def read_then_mutate(path: Path, model: type[BaseModel]) -> list[BaseModel]:
        records = original_read_records(path, model)
        if path.name == "experiments.jsonl":
            experiments.write_text("{late-experiment-mutation}\n", encoding="utf-8")
            (results_root / "suites" / "suite-1" / "lifecycle.jsonl").write_text(
                "{late-suite-mutation}\n", encoding="utf-8"
            )
        return records

    monkeypatch.setattr(indexer, "_read_records", read_then_mutate)
    indexer.rebuild_all()

    with closing(connect(db_path)) as connection:
        if connection.execute("SELECT COUNT(*) FROM suites").fetchone()[0] != 1:
            raise AssertionError("rebuild re-read mutated suite input")


def test_corrupt_suite_rebuild_preserves_database_bundle_and_source(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    before = db_path.read_bytes()
    lifecycle = results_root / "suites" / "suite-1" / "lifecycle.jsonl"
    lifecycle.write_text("{bad-suite}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        indexer.rebuild_all()

    if db_path.read_bytes() != before or lifecycle.read_text(encoding="utf-8") != "{bad-suite}\n":
        raise AssertionError("corrupt suite rebuild changed target bundle or source artifact")


def test_corrupt_experiments_rebuild_preserves_database_and_source(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_happy_path_run(results_root)
    indexer = SqliteIndexer(db_path)
    indexer.index_run("run-1")
    before = db_path.read_bytes()
    experiments = results_root / "suites" / "suite-1" / "experiments.jsonl"
    experiments.write_text("{bad-experiment}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        indexer.rebuild_all()

    if (
        db_path.read_bytes() != before
        or experiments.read_text(encoding="utf-8") != "{bad-experiment}\n"
    ):
        raise AssertionError("corrupt experiments rebuild changed target database or source")


def test_missing_terminal_event_remains_running(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_suite(results_root, "suite-running", completed=False)
    write_jsonl(
        results_root / "runs" / "run-running" / "lifecycle.jsonl",
        [
            run_started_event("run-running", "suite-running"),
            ConversationProcessedEvent(
                event_id="run-running-conversation-processed",
                timestamp=NOW,
                run_id="run-running",
                sequence_number=1,
                conversation_id="conversation-1",
                n_questions_evaluated=2,
                n_questions_correct=1,
                n_questions_errored=1,
                cost_usd=0.12,
            ),
        ],
    )

    SqliteIndexer(db_path).index_run("run-running")

    row = fetch_one(db_path, "SELECT * FROM runs WHERE run_id = ?", ("run-running",))
    if row["status"] != "running":
        raise AssertionError(row["status"])
    if row["finished_at"] is not None:
        raise AssertionError(row["finished_at"])
    if row["n_questions_attempted"] != 2:
        raise AssertionError(row["n_questions_attempted"])
    if row["total_cost_usd"] != 0.12:
        raise AssertionError(row["total_cost_usd"])


def test_index_run_enforces_documented_foreign_keys(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    db_path = results_root / "benchmark.db"
    write_jsonl(
        results_root / "runs" / "run-orphan" / "lifecycle.jsonl",
        [run_started_event("run-orphan", "missing-suite")],
    )

    try:
        SqliteIndexer(db_path).index_run("run-orphan")
    except sqlite3.IntegrityError:
        return
    except PersistenceError as exc:
        raise AssertionError(str(exc)) from exc

    raise AssertionError("index_run accepted a run with no referenced suite row")
