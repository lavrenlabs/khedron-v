from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from khedron.config import ExperimentConfig
from khedron.errors import PersistenceError
from khedron.persistence.repository import RunRepository
from khedron.types import (
    AggregateScore,
    APICallRecord,
    ConversationIngestionRecord,
    ConversationProcessedEvent,
    ErrorRecord,
    ExperimentCompletedEvent,
    ExperimentResult,
    ExperimentStartedEvent,
    FailurePattern,
    Judgment,
    JudgmentVerdict,
    Memory,
    QuestionEvaluationRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunFailedEvent,
    RunFilters,
    RunResumedEvent,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteStartedEvent,
)
from khedron.utils.stats import wilson_score_interval

NOW = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)


class CapturingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.warnings.append((event, kwargs))


def expect(condition: bool, message: object) -> None:
    if not condition:
        raise AssertionError(message)


def score(n_total: int = 2, n_correct: int = 1) -> ScoreWithCI:
    low, high = wilson_score_interval(n_correct, n_total)
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_total - n_correct,
        point_estimate=n_correct / n_total,
        ci_95_low=low,
        ci_95_high=high,
    )


def experiment_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "Synthetic",
            "provider": {"type": "full_context"},
            "benchmark": {"type": "locomo"},
            "answer_model": {"type": "openai", "model": "gpt-4o-mini-2024-07-18"},
            "judge": {"type": "anthropic", "model": "claude-sonnet-4-5"},
        }
    )


def suite_started() -> SuiteStartedEvent:
    return SuiteStartedEvent(
        event_id="suite-1-started",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=0,
        config_yaml_path="experiments/synthetic.yaml",
        config_yaml_content="experiments: []",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        runtime_environment={"python": "3.11"},
    )


def experiment_started() -> ExperimentStartedEvent:
    return ExperimentStartedEvent(
        event_id="suite-1-exp-1-started",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=1,
        experiment_id="experiment-1",
        experiment_name="Synthetic experiment",
        n_runs_planned=1,
    )


def experiment_completed() -> ExperimentCompletedEvent:
    return ExperimentCompletedEvent(
        event_id="suite-1-exp-1-completed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=2,
        experiment_id="experiment-1",
        overall_standard_pooled_score=score(),
        overall_standard_mean=0.5,
        overall_standard_stddev=0.0,
        overall_audited_pooled_score=score(),
        overall_audited_mean=0.5,
        overall_audited_stddev=0.0,
        cost_usd=0.03,
    )


def suite_completed() -> SuiteCompletedEvent:
    return SuiteCompletedEvent(
        event_id="suite-1-completed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=3,
        total_cost_usd=0.03,
        n_experiments_completed=1,
        n_experiments_failed=0,
    )


def run_started() -> RunStartedEvent:
    return RunStartedEvent(
        event_id="run-1-started",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=0,
        suite_id="suite-1",
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
        runtime_environment={"python": "3.11"},
    )


def conversation_processed() -> ConversationProcessedEvent:
    return ConversationProcessedEvent(
        event_id="run-1-conversation-processed",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=1,
        conversation_id="conversation-1",
        n_questions_evaluated=2,
        n_questions_correct=1,
        n_questions_errored=1,
        cost_usd=0.03,
    )


def run_completed() -> RunCompletedEvent:
    return RunCompletedEvent(
        event_id="run-1-completed",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=2,
        status="partial",
        n_questions_attempted=2,
        n_questions_succeeded=1,
        n_questions_errored=1,
        overall_score_standard=score(),
        overall_score_audited=score(),
        by_category_standard={"single_hop": score(1, 1), "multi_hop": score(1, 0)},
        by_category_audited={"single_hop": score(1, 1), "multi_hop": score(1, 0)},
        total_cost_usd=0.03,
    )


def question_evaluation(
    question_id: str,
    question_evaluation_id: str,
    category: str,
) -> QuestionEvaluationRecord:
    return QuestionEvaluationRecord(
        question_evaluation_id=question_evaluation_id,
        run_id="run-1",
        question_id=question_id,
        conversation_id="conversation-1",
        category=category,
        question_text=f"What is the answer for {question_id}?",
        expected_answer="Rome",
        is_audited_error=False,
        timestamp=NOW,
    )


def retrieval() -> RetrievalRecord:
    return RetrievalRecord(
        retrieval_id="retrieval-1",
        question_evaluation_id="qe-pass",
        run_id="run-1",
        question_id="q-pass",
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


def response() -> Response:
    return Response(
        response_id="response-1",
        run_id="run-1",
        question_id="q-pass",
        retrieval_id="retrieval-1",
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


def judgment() -> Judgment:
    return Judgment(
        judgment_id="judgment-1",
        run_id="run-1",
        response_id="response-1",
        question_id="q-pass",
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


def error_record() -> ErrorRecord:
    return ErrorRecord(
        error_id="error-1",
        run_id="run-1",
        timestamp=NOW,
        phase="retrieve",
        question_id="q-fail",
        error_type="ProviderTimeoutError",
        error_message="Retrieval failed.",
        stack_trace=None,
        context={"attempt": 1},
        recovered=False,
    )


def api_call() -> APICallRecord:
    return APICallRecord(
        api_call_id="api-1",
        run_id="run-1",
        question_id="q-pass",
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


def conversation_ingestion() -> ConversationIngestionRecord:
    return ConversationIngestionRecord(
        run_id="run-1",
        conversation_id="conversation-1",
        started_at=NOW,
        finished_at=NOW,
        n_sessions=1,
        n_turns=2,
        n_turns_succeeded=2,
        n_turns_failed=0,
        total_latency_ms=25.0,
        avg_latency_per_turn_ms=12.5,
        error_summary=[],
    )


def failure_pattern() -> FailurePattern:
    return FailurePattern(
        pattern_id="pattern-1",
        run_id="run-1",
        pattern_name="missing_memory_failure",
        description="The answer failed when no relevant memory was retrieved.",
        suggested_remedy="Improve retrieval recall for the affected question set.",
        n_affected_questions=1,
        affected_question_ids=["q-fail"],
        confidence=0.8,
    )


def experiment_result(
    experiment_id: str = "experiment-1",
    suite_id: str = "suite-1",
    experiment_name: str = "Synthetic experiment",
    run_ids: list[str] | None = None,
) -> ExperimentResult:
    aggregate = AggregateScore(
        experiment_id=experiment_id,
        category="overall",
        mode="audited",
        n_runs=1,
        pooled_score=score(),
        individual_run_scores=[0.5],
        mean=0.5,
        stddev=0.0,
        min=0.5,
        max=0.5,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        suite_id=suite_id,
        experiment_name=experiment_name,
        n_runs=1,
        run_ids=run_ids or ["run-1"],
        aggregate_overall_standard=aggregate,
        aggregate_overall_audited=aggregate,
        aggregate_by_category_standard={"overall": aggregate},
        aggregate_by_category_audited={"overall": aggregate},
        total_cost_usd=0.03,
        config=experiment_config(),
    )


async def append_synthetic_suite_and_run(repository: RunRepository) -> None:
    await repository.append_suite_event(suite_started())
    await repository.append_suite_event(experiment_started())
    await repository.append_run_event(run_started())
    await repository.append_question_evaluation(
        question_evaluation("q-pass", "qe-pass", "single_hop")
    )
    await repository.append_retrieval(retrieval())
    await repository.append_response(response())
    await repository.append_judgment(judgment())
    await repository.append_question_evaluation(
        question_evaluation("q-fail", "qe-fail", "multi_hop")
    )
    await repository.append_error(error_record())
    await repository.append_run_event(conversation_processed())
    await repository.append_run_event(run_completed())
    await repository.append_suite_event(experiment_completed())
    await repository.append_experiment_result(experiment_result())
    await repository.append_suite_event(suite_completed())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_model_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )


def table_count(sqlite_path: Path, table_name: str) -> int:
    with closing(sqlite3.connect(sqlite_path)) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    if row is None:
        raise AssertionError(table_name)
    value = row[0]
    if not isinstance(value, int):
        raise AssertionError(value)
    return value


def experiment_result_count(sqlite_path: Path) -> int:
    with closing(sqlite3.connect(sqlite_path)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM experiment_results").fetchone()
    if row is None:
        raise AssertionError("experiment_results")
    value = row[0]
    if not isinstance(value, int):
        raise AssertionError(value)
    return value


@pytest.mark.asyncio
async def test_repository_append_and_read_cycle(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)

    await append_synthetic_suite_and_run(repository)
    await repository.append_api_call(api_call())
    await repository.append_conversation_ingestion(conversation_ingestion())
    write_model_jsonl(
        results_dir / "runs" / "run-1" / "failure_patterns.jsonl",
        [failure_pattern()],
    )

    suite_lifecycle = results_dir / "suites" / "suite-1" / "lifecycle.jsonl"
    suite_experiments = results_dir / "suites" / "suite-1" / "experiments.jsonl"
    run_lifecycle = results_dir / "runs" / "run-1" / "lifecycle.jsonl"
    run_errors = results_dir / "runs" / "run-1" / "errors.jsonl"
    run_api_calls = results_dir / "runs" / "run-1" / "api_calls.jsonl"
    run_failure_patterns = results_dir / "runs" / "run-1" / "failure_patterns.jsonl"

    expect(len(read_jsonl(suite_lifecycle)) == 4, "suite jsonl")
    expect(len(read_jsonl(suite_experiments)) == 1, "experiments jsonl")
    expect(len(read_jsonl(run_lifecycle)) == 3, "run jsonl")
    expect(len(read_jsonl(run_errors)) == 1, "errors jsonl")
    expect(len(read_jsonl(run_api_calls)) == 1, "api calls jsonl")
    expect(len(read_jsonl(run_failure_patterns)) == 1, "failure patterns jsonl")

    suite_events = repository.list_suite_events("suite-1")
    expect(
        [event.sequence_number for event in suite_events] == [0, 1, 2, 3],
        suite_events,
    )
    run_events = repository.list_run_events("run-1")
    expect([event.sequence_number for event in run_events] == [0, 1, 2], run_events)
    expect(repository.get_run_started_event("run-1") == run_started(), "run started event")

    suite_status = repository.get_suite_status("suite-1")
    expect(suite_status.status == "completed", suite_status)
    expect(suite_status.n_experiments_completed == 1, suite_status)

    run_status = repository.get_run_status("run-1")
    expect(run_status.status == "partial", run_status)
    expect(run_status.n_conversations_processed == 1, run_status)
    expect(run_status.n_questions_attempted == 2, run_status)
    expect(run_status.total_cost_usd == 0.03, run_status)

    runs = repository.list_runs()
    expect(len(runs) == 1, runs)
    expect(runs[0].run_id == "run-1", runs[0])
    expect(runs[0].status == "partial", runs[0])
    expect(runs[0].experiment_name == "Synthetic experiment", runs[0])
    expect(runs[0].total_cost_usd == 0.03, runs[0])

    filtered_runs = repository.list_runs(
        RunFilters(
            experiment_name="Synthetic experiment",
            status="partial",
            provider_type="full_context",
        )
    )
    expect([run.run_id for run in filtered_runs] == ["run-1"], filtered_runs)
    expect(repository.list_runs(RunFilters(experiment_name="Other")) == [], "run filters")

    question = repository.get_question_record("run-1", "q-pass")
    expect(question.retrieved_memory_ids == ["memory-1"], question)
    expect(question.retrieval_timestamp == NOW, question)
    expect(question.generation_timestamp == NOW, question)
    expect(question.judgment_timestamp == NOW, question)
    expect(question.generation_cost_usd == 0.01, question)
    expect(question.judgment_cost_usd == 0.02, question)

    questions = repository.get_questions_for_run("run-1")
    expect(
        [question_record.question_id for question_record in questions] == ["q-fail", "q-pass"],
        questions,
    )
    expect(questions[1].retrieval_id == "retrieval-1", questions[1])

    failed = repository.get_failed_questions("run-1")
    expect([question_record.question_id for question_record in failed] == ["q-fail"], failed)
    failed_multi_hop = repository.get_failed_questions("run-1", category="multi_hop")
    expect(
        [question_record.question_id for question_record in failed_multi_hop] == ["q-fail"],
        failed_multi_hop,
    )
    failed_single_hop = repository.get_failed_questions("run-1", category="single_hop")
    expect(failed_single_hop == [], failed_single_hop)

    expect(repository.get_retrieval("retrieval-1") == retrieval(), "retrieval source record")
    expect(repository.get_response("response-1") == response(), "response source record")
    expect(repository.get_judgment("judgment-1") == judgment(), "judgment source record")
    expect(
        repository.get_conversation_ingestions_for_run("run-1") == [conversation_ingestion()],
        "conversation ingestion source records",
    )
    expect(repository.get_api_calls_for_run("run-1") == [api_call()], "api call source records")
    expect(repository.get_errors_for_run("run-1") == [error_record()], "error source records")
    expect(
        repository.get_failure_patterns_for_run("run-1") == [failure_pattern()],
        "failure pattern source records",
    )
    expect(table_count(sqlite_path, "api_calls") == 0, "api_calls table should not exist")
    expect(experiment_result_count(sqlite_path) == 1, "experiment result projection")


@pytest.mark.asyncio
async def test_repository_lifecycle_event_reads_sort_by_sequence_and_latest_start(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)
    restarted = run_started().model_copy(
        update={
            "event_id": "run-1-restarted",
            "sequence_number": 5,
            "experiment_id": "experiment-restarted",
            "experiment_name": "Restarted experiment",
        }
    )

    write_model_jsonl(
        results_dir / "runs" / "run-1" / "lifecycle.jsonl",
        [
            run_completed().model_copy(update={"sequence_number": 6}),
            conversation_processed(),
            restarted,
            run_started(),
        ],
    )
    write_model_jsonl(
        results_dir / "suites" / "suite-1" / "lifecycle.jsonl",
        [
            suite_completed(),
            experiment_started(),
            suite_started(),
            experiment_completed(),
        ],
    )

    run_events = repository.list_run_events("run-1")
    expect([event.sequence_number for event in run_events] == [0, 1, 5, 6], run_events)
    expect(repository.get_run_started_event("run-1") == restarted, "latest run start")

    suite_events = repository.list_suite_events("suite-1")
    expect(
        [event.sequence_number for event in suite_events] == [0, 1, 2, 3],
        suite_events,
    )


@pytest.mark.asyncio
async def test_repository_experiment_result_read_apis(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)
    await append_synthetic_suite_and_run(repository)
    alpha_result = experiment_result(
        experiment_id="experiment-alpha",
        experiment_name="Alpha experiment",
        run_ids=["run-alpha"],
    )
    await repository.append_experiment_result(alpha_result)

    expect(
        repository.get_experiment_result("experiment-1") == experiment_result(),
        "experiment result by id",
    )
    expect(
        repository.get_experiment_result_for_run("run-1") == experiment_result(),
        "experiment result by run id",
    )
    expect(
        [result.experiment_id for result in repository.list_experiment_results("suite-1")]
        == ["experiment-alpha", "experiment-1"],
        "suite experiment result sort",
    )
    expect(
        [result.experiment_id for result in repository.list_experiment_results()]
        == ["experiment-alpha", "experiment-1"],
        "all experiment result sort",
    )
    with pytest.raises(PersistenceError, match="Experiment result does not exist"):
        repository.get_experiment_result("missing-experiment")


@pytest.mark.asyncio
async def test_repository_experiment_result_for_run_returns_none_until_aggregate_exists(
    tmp_path: Path,
) -> None:
    repository = RunRepository(results_dir=tmp_path / "results", sqlite_path=tmp_path / "db.sqlite")
    await repository.append_run_event(run_started())

    expect(repository.get_experiment_result_for_run("run-1") is None, "missing aggregate result")


@pytest.mark.asyncio
async def test_repository_optional_jsonl_reads_return_records_or_empty_lists(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)

    expect(repository.get_conversation_ingestions_for_run("missing-run") == [], "missing conv")
    expect(repository.get_api_calls_for_run("missing-run") == [], "missing api")
    expect(repository.get_errors_for_run("missing-run") == [], "missing errors")
    expect(repository.get_failure_patterns_for_run("missing-run") == [], "missing patterns")

    await repository.append_conversation_ingestion(conversation_ingestion())
    await repository.append_api_call(api_call())
    await repository.append_error(error_record())
    write_model_jsonl(
        results_dir / "runs" / "run-1" / "failure_patterns.jsonl",
        [failure_pattern()],
    )

    expect(
        repository.get_conversation_ingestions_for_run("run-1") == [conversation_ingestion()],
        "conversation ingestion records",
    )
    expect(repository.get_api_calls_for_run("run-1") == [api_call()], "api call records")
    expect(repository.get_errors_for_run("run-1") == [error_record()], "error records")
    expect(repository.get_failure_patterns_for_run("run-1") == [failure_pattern()], "patterns")


@pytest.mark.asyncio
async def test_sqlite_indexing_failure_preserves_jsonl_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)
    logger = CapturingLogger()
    repository._log = logger

    def fail_index_run(run_id: str) -> None:
        raise sqlite3.OperationalError(f"boom for {run_id}")

    monkeypatch.setattr(repository._sqlite_indexer, "index_run", fail_index_run)

    await repository.append_run_event(run_started())

    run_events = read_jsonl(results_dir / "runs" / "run-1" / "lifecycle.jsonl")
    expect(run_events[0]["event_type"] == "run_started", run_events)
    expect(len(logger.warnings) == 1, logger.warnings)
    event, fields = logger.warnings[0]
    expect(event == "sqlite_indexing_failed", event)
    expect(fields["operation"] == "index_run", fields)
    expect(fields["run_id"] == "run-1", fields)


@pytest.mark.asyncio
async def test_append_api_call_only_writes_jsonl(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)

    await repository.append_api_call(api_call())

    api_calls = read_jsonl(results_dir / "runs" / "run-1" / "api_calls.jsonl")
    expect(api_calls[0]["record_type"] == "api_call", api_calls)
    expect(not sqlite_path.exists(), "api call append should not initialize SQLite")


@pytest.mark.asyncio
async def test_append_conversation_ingestion_only_writes_jsonl(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)

    await repository.append_conversation_ingestion(conversation_ingestion())

    records = read_jsonl(results_dir / "runs" / "run-1" / "conversations.jsonl")
    expect(records[0]["record_type"] == "conversation_ingestion", records)
    expect(records[0]["conversation_id"] == "conversation-1", records)
    expect(not sqlite_path.exists(), "conversation ingestion append should not initialize SQLite")


@pytest.mark.asyncio
async def test_rebuild_sqlite_restores_projection_after_delete(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)
    await append_synthetic_suite_and_run(repository)

    sqlite_path.unlink()
    await repository.rebuild_sqlite()

    question = repository.get_question_record("run-1", "q-pass")
    expect(question.retrieved_memory_ids == ["memory-1"], question)
    expect(repository.get_run_status("run-1").status == "partial", "run status")
    expect(experiment_result_count(sqlite_path) == 1, "experiment result projection")


@pytest.mark.asyncio
async def test_rebuild_sqlite_projects_experiment_snapshot_despite_late_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)
    await append_synthetic_suite_and_run(repository)
    experiments_path = results_dir / "suites" / "suite-1" / "experiments.jsonl"
    original_read_records = repository._sqlite_indexer._read_records

    def read_then_mutate(path: Path, model: type[BaseModel]) -> list[BaseModel]:
        records = original_read_records(path, model)
        if path == experiments_path:
            write_model_jsonl(path, [])
        return records

    monkeypatch.setattr(repository._sqlite_indexer, "_read_records", read_then_mutate)
    sqlite_path.unlink()

    await repository.rebuild_sqlite()

    expect(experiment_result_count(sqlite_path) == 1, "captured experiment result projection")


def test_repository_read_connection_has_busy_timeout_without_forcing_wal(tmp_path: Path) -> None:
    # RunRepository connections also serve dashboard/report reads, whose contract forbids
    # mutating persistence. The connection must carry the 30s busy timeout (so a read
    # waits for the indexer's writes instead of failing) but must NOT switch the journal
    # mode to WAL, which would write -wal/-shm files and can fail on a read-only DB. The
    # indexer (the creating writer) is what enables WAL on the file.
    repository = RunRepository(
        results_dir=tmp_path / "results", sqlite_path=tmp_path / "benchmark.db"
    )
    with closing(repository._connect()) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    expect(str(journal_mode).lower() != "wal", journal_mode)
    expect(busy_timeout == 30000, busy_timeout)


@pytest.mark.asyncio
async def test_concurrent_run_indexing_is_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runner re-indexes a run after each concurrently-evaluated question. index_run
    # reads the JSONL snapshot then delete+reinserts; if two passes overlap, an older
    # snapshot can commit after a newer one and drop rows. The repository must serialize
    # indexing so at most one pass runs at a time.
    repository = RunRepository(
        results_dir=tmp_path / "results", sqlite_path=tmp_path / "benchmark.db"
    )

    active = 0
    max_active = 0

    def fake_index_run(run_id: str) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.02)
        active -= 1

    monkeypatch.setattr(repository._sqlite_indexer, "index_run", fake_index_run)

    await asyncio.gather(*(repository._index_run_best_effort("run-1") for _ in range(5)))

    expect(max_active == 1, max_active)


def _run_completed_at(sequence_number: int) -> RunCompletedEvent:
    event = run_completed()
    return event.model_copy(update={"sequence_number": sequence_number})


@pytest.mark.asyncio
async def test_a_resumed_run_supersedes_its_prior_failure_in_both_reducers(
    tmp_path: Path,
) -> None:
    # B0: the reducers ignored run_resumed. After a run_failed, terminal_seen skipped the resume and
    # the later run_completed, so a resumed run stayed 'failed' in both the append-only status and
    # the SQLite projection. Both must now show the resumed completion; the failure stays in the
    # stream as history but no longer decides the status.
    results_dir = tmp_path / "results"
    sqlite_path = tmp_path / "benchmark.db"
    repository = RunRepository(results_dir=results_dir, sqlite_path=sqlite_path)

    # The suite/experiment must exist first: the SQLite runs row has a foreign key to its suite.
    await repository.append_suite_event(suite_started())
    await repository.append_suite_event(experiment_started())
    await repository.append_run_event(run_started())
    await repository.append_run_event(conversation_processed())
    await repository.append_run_event(
        RunFailedEvent(
            event_id="run-1-failed",
            timestamp=NOW,
            run_id="run-1",
            sequence_number=2,
            error_message="connection dropped",
            error_phase="generate",
            n_questions_completed_before_failure=1,
            partial_cost_usd=0.03,
        )
    )
    # After the failure alone, the status is 'failed'.
    expect(repository.get_run_status("run-1").status == "failed", "failed before resume")

    await repository.append_run_event(
        RunResumedEvent(
            event_id="run-1-resumed",
            timestamp=NOW,
            run_id="run-1",
            sequence_number=3,
            n_conversations_inherited=1,
            n_questions_inherited=1,
            inherited_cost_usd=0.03,
            framework_version="0.1.0",
        )
    )
    # Immediately after the resume, before any new work: the run is running again with the terminal
    # fields cleared, but the cost the interrupted attempt already recorded is kept, not reset.
    resumed = repository.get_run_status("run-1")
    expect(resumed.status == "running", ("running immediately after resume", resumed.status))
    expect(resumed.finished_at is None, ("finished_at cleared on resume", resumed.finished_at))
    expect(resumed.error_phase is None, ("error_phase cleared on resume", resumed.error_phase))
    expect(
        resumed.total_cost_usd == 0.03, ("inherited cost kept on resume", resumed.total_cost_usd)
    )

    await repository.append_run_event(_run_completed_at(4))

    # Append-only reducer: the resumed completion decides the status, not the superseded failure.
    status = repository.get_run_status("run-1")
    expect(status.status == "partial", ("append-only status after resume", status.status))
    expect(status.error_message is None, ("terminal error cleared on resume", status.error_message))

    # SQLite reducer (built by index_run on append): same answer, or the query layer would disagree
    # with the append-only truth.
    with closing(sqlite3.connect(sqlite_path)) as connection:
        row = connection.execute(
            "SELECT status, error_message FROM runs WHERE run_id = ?", ("run-1",)
        ).fetchone()
    expect(row[0] == "partial", ("sqlite status after resume", row[0]))
    expect(row[1] is None, ("sqlite error cleared on resume", row[1]))
