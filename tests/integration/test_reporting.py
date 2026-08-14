from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from pathlib import Path

import pytest

from khedron.analysis.types import ComparisonReport, QuestionDifference, ScoreDelta
from khedron.config import ExperimentConfig
from khedron.persistence.repository import RunRepository
from khedron.reporting import ReportContextBuilder, ReportGenerator
from khedron.types import (
    AggregateScore,
    APICallRecord,
    ConversationProcessedEvent,
    ErrorRecord,
    ExperimentCompletedEvent,
    ExperimentResult,
    ExperimentStartedEvent,
    Judgment,
    JudgmentVerdict,
    Memory,
    QuestionEvaluationRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteStartedEvent,
)
from khedron.utils.stats import wilson_score_interval

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_repository_backed_context_renders_all_report_types(tmp_path: Path) -> None:
    repository = RunRepository(
        results_dir=tmp_path / "results",
        sqlite_path=tmp_path / "benchmark.db",
    )
    await append_synthetic_suite_and_run(repository)
    await repository.append_api_call(api_call())

    context = ReportContextBuilder().build("run-1", repository)
    output_dir = tmp_path / "reports"
    generator = ReportGenerator()

    generator.generate_executive_summary(context, output_dir / "executive.md")
    generator.generate_technical_deep_dive(context, output_dir / "technical.md")
    generator.generate_failure_analysis_report(context, output_dir / "failures.md")
    generator.generate_comparison_report(comparison_report(), output_dir / "comparison.md")

    executive = (output_dir / "executive.md").read_text(encoding="utf-8")
    technical = (output_dir / "technical.md").read_text(encoding="utf-8")
    failures = (output_dir / "failures.md").read_text(encoding="utf-8")
    comparison = (output_dir / "comparison.md").read_text(encoding="utf-8")

    assert "# Executive Summary - Synthetic experiment" in executive
    assert "| Failed questions | 1 |" in executive
    assert "| generate | $0.01 |" in executive
    assert "# Technical Deep Dive - Synthetic experiment" in technical
    assert "| Experiment ID | experiment-1 |" in technical
    assert "| Top-K retrieval | 10 |" in technical
    assert "# Failure Analysis - Synthetic experiment" in failures
    assert "| q-fail | multi_hop | error | retrieve |" in failures
    assert "# Comparison Report" in comparison
    assert all(path.read_text(encoding="utf-8").endswith("\n") for path in output_dir.glob("*.md"))


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
            "name": "Synthetic experiment",
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


def experiment_result() -> ExperimentResult:
    aggregate = AggregateScore(
        experiment_id="experiment-1",
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
        experiment_id="experiment-1",
        suite_id="suite-1",
        experiment_name="Synthetic experiment",
        n_runs=1,
        run_ids=["run-1"],
        aggregate_overall_standard=aggregate,
        aggregate_overall_audited=aggregate,
        aggregate_by_category_standard={"overall": aggregate},
        aggregate_by_category_audited={"overall": aggregate},
        total_cost_usd=0.03,
        config=experiment_config(),
    )


def comparison_report() -> ComparisonReport:
    return ComparisonReport(
        run_ids=["baseline-run", "candidate-run"],
        mode="audited",
        compatible=True,
        compatibility_warnings=[],
        score_deltas=[
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=score(),
                candidate_score=score(n_correct=2),
                point_delta=0.5,
                ci_overlaps=True,
                statistically_significant=False,
            )
        ],
        differing_questions=[
            QuestionDifference(
                question_id="q-1",
                category="single_hop",
                verdicts_by_run_id={
                    "baseline-run": JudgmentVerdict.INCORRECT,
                    "candidate-run": JudgmentVerdict.CORRECT,
                },
                scores_by_run_id={"baseline-run": 0.0, "candidate-run": 1.0},
            )
        ],
    )
