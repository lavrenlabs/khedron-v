from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from typing import TypeVar

from khedron.persistence.schemas import SCHEMA_DDL, SCHEMA_VERSION

T = TypeVar("T")


EXPECTED_TABLES = frozenset(
    {
        "schema_meta",
        "suite_lifecycle_events",
        "run_lifecycle_events",
        "suites",
        "experiment_results",
        "runs",
        "question_evaluations",
        "retrievals",
        "question_results",
        "scores_by_category",
        "failure_patterns",
        "error_log",
        "question_plan",
        "recovery_attempts",
        "error_resolutions",
    }
)

EXPECTED_INDEXES = frozenset(
    {
        "idx_suite_events_suite",
        "idx_suite_events_type",
        "idx_run_events_run",
        "idx_run_events_type",
        "idx_suites_started_at",
        "idx_experiment_results_suite",
        "idx_experiment_results_name",
        "idx_runs_experiment",
        "idx_runs_started_at",
        "idx_runs_provider",
        "idx_runs_models",
        "idx_question_evaluations_run",
        "idx_question_evaluations_category",
        "idx_question_evaluations_audited",
        "idx_retrievals_run",
        "idx_retrievals_question",
        "idx_retrievals_question_eval",
        "idx_qr_verdict",
        "idx_qr_category",
        "idx_qr_run_category",
        "idx_qr_run_verdict",
        "idx_qr_audited",
        "idx_qr_retrieval",
        "idx_patterns_run",
        "idx_patterns_name",
    }
)

REPRESENTATIVE_COLUMNS = {
    "runs": frozenset({"methodology_profile"}),
    "retrievals": frozenset({"recovery_attempt_id", "ingestion_attempt_id"}),
    "error_log": frozenset({"retryable"}),
    "scores_by_category": frozenset({"n_partial", "n_unknown"}),
    "experiment_results": frozenset({"config_json"}),
    "suite_lifecycle_events": frozenset({"event_data"}),
    "run_lifecycle_events": frozenset({"event_data"}),
}

REPRESENTATIVE_FOREIGN_KEYS = {
    "experiment_results": frozenset({("suite_id", "suites", "suite_id")}),
    "runs": frozenset({("suite_id", "suites", "suite_id")}),
    "question_evaluations": frozenset({("run_id", "runs", "run_id")}),
    "retrievals": frozenset(
        {
            ("run_id", "runs", "run_id"),
            (
                "question_evaluation_id",
                "question_evaluations",
                "question_evaluation_id",
            ),
        }
    ),
    "question_results": frozenset(
        {
            ("run_id", "runs", "run_id"),
            (
                "question_evaluation_id",
                "question_evaluations",
                "question_evaluation_id",
            ),
            ("retrieval_id", "retrievals", "retrieval_id"),
        }
    ),
    "scores_by_category": frozenset({("run_id", "runs", "run_id")}),
    "failure_patterns": frozenset({("run_id", "runs", "run_id")}),
    "error_log": frozenset({("run_id", "runs", "run_id")}),
}


def create_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def apply_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_DDL)


def object_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = ? AND name NOT LIKE 'sqlite_%'
        """,
        (object_type,),
    ).fetchall()
    return {str(row["name"]) for row in rows}


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM pragma_table_info(?)",
        (table_name,),
    ).fetchall()
    return {str(row["name"]) for row in rows}


def foreign_keys(connection: sqlite3.Connection, table_name: str) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT "from" AS source_column, "table" AS target_table, "to" AS target_column
        FROM pragma_foreign_key_list(?)
        """,
        (table_name,),
    ).fetchall()
    return {
        (
            str(row["source_column"]),
            str(row["target_table"]),
            str(row["target_column"]),
        )
        for row in rows
    }


def schema_snapshot(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = connection.execute(
        """
        SELECT type, name
        FROM sqlite_master
        WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return {(str(row["type"]), str(row["name"])) for row in rows}


def require_subset(actual: set[T], expected: Iterable[T]) -> None:
    expected_set = set(expected)
    missing = expected_set - actual
    if missing:
        raise AssertionError(sorted(missing))


def test_schema_version_is_v2() -> None:
    if SCHEMA_VERSION != 2:
        raise AssertionError(SCHEMA_VERSION)


def test_schema_ddl_executes_on_in_memory_sqlite() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)


def test_schema_creates_all_expected_tables() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)

        require_subset(object_names(connection, "table"), EXPECTED_TABLES)


def test_schema_creates_all_expected_indexes() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)

        require_subset(object_names(connection, "index"), EXPECTED_INDEXES)


def test_schema_includes_representative_contract_columns() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)

        for table_name, expected_columns in REPRESENTATIVE_COLUMNS.items():
            require_subset(table_columns(connection, table_name), expected_columns)


def test_schema_includes_representative_foreign_keys() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)

        for table_name, expected_foreign_keys in REPRESENTATIVE_FOREIGN_KEYS.items():
            require_subset(foreign_keys(connection, table_name), expected_foreign_keys)


def test_schema_ddl_is_idempotent() -> None:
    with closing(create_connection()) as connection:
        apply_schema(connection)
        first_snapshot = schema_snapshot(connection)

        apply_schema(connection)
        second_snapshot = schema_snapshot(connection)

    if second_snapshot != first_snapshot:
        raise AssertionError(second_snapshot)
