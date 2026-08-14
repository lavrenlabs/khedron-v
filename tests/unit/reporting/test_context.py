from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from khedron.analysis.types import (
    CategoryFailureBreakdown,
    DetectedPattern,
    FailedQuestionSummary,
    FailureAnalysisReport,
    TraceabilityIndexEntry,
)
from khedron.reporting.context import CostSummary, MethodologyDisclosure, ReportContext
from khedron.types import (
    AggregateScore,
    ExperimentResult,
    JudgmentVerdict,
    QuestionCategory,
    QuestionRecord,
    RunStartedEvent,
    RunStatus,
    ScoreWithCI,
    SuiteStatus,
)

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


def experiment_config() -> dict[str, Any]:
    return {
        "name": "Synthetic",
        "provider": {"type": "full_context"},
        "benchmark": {"type": "locomo"},
        "answer_model": {"type": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "judge": {"type": "anthropic", "model": "claude-sonnet-4-5"},
    }


def score() -> ScoreWithCI:
    return ScoreWithCI(
        n_total=10,
        n_correct=8,
        n_errors=1,
        n_partial=1,
        n_unknown=0,
        point_estimate=0.8,
        ci_95_low=0.49,
        ci_95_high=0.94,
    )


def question_record() -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id="qe-1",
        run_id="run-1",
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.TEMPORAL,
        question_text="When did Alice move?",
        expected_answer="May 2025",
        is_audited_error=False,
        retrieval_id="ret-1",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=["mem-1"],
        response_id="resp-1",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="June 2025",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id="judge-1",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=JudgmentVerdict.INCORRECT,
        score=0.0,
        judgment_reasoning="The generated month does not match.",
        judgment_input_tokens=50,
        judgment_output_tokens=10,
        judgment_cost_usd=0.02,
        total_latency_ms=392.5,
        total_cost_usd=0.03,
        error_message=None,
        error_phase=None,
    )


def failure_analysis_report() -> FailureAnalysisReport:
    return FailureAnalysisReport(
        run_id="run-1",
        category_breakdown=[
            CategoryFailureBreakdown(
                category=QuestionCategory.TEMPORAL,
                n_total=1,
                n_failed=1,
                failure_rate=1.0,
            )
        ],
        failed_questions=[
            FailedQuestionSummary(
                question_id="q-1",
                category=QuestionCategory.TEMPORAL,
                verdict=JudgmentVerdict.INCORRECT,
                question_text="When did Alice move?",
                expected_answer="May 2025",
                generated_answer="June 2025",
                error_phase=None,
                error_message=None,
            )
        ],
        patterns=[
            DetectedPattern(
                pattern_name="temporal_arithmetic_failure",
                description="Temporal answer was incorrect.",
                suggested_remedy="Improve temporal reasoning prompts.",
                affected_question_ids=["q-1"],
                n_affected_questions=1,
                confidence=0.75,
            )
        ],
        traceability_index=[
            TraceabilityIndexEntry(
                question_id="q-1",
                retrieval_id="ret-1",
                response_id="resp-1",
                judgment_id="judge-1",
            )
        ],
    )


def run_started_event() -> RunStartedEvent:
    return RunStartedEvent(
        event_id="event-run-started-1",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=0,
        suite_id="suite-1",
        experiment_id="exp-1",
        experiment_name="Synthetic",
        run_number=0,
        provider_type="full_context",
        provider_version="0.1.0",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="checksum-1",
        answer_model_id="gpt-4o-mini-2024-07-18",
        answer_model_vendor="openai",
        judge_model_id="claude-sonnet-4-5",
        judge_model_vendor="anthropic",
        config={"seed": 123},
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=123,
        runtime_environment={"python": "3.11"},
    )


def run_status() -> RunStatus:
    score_value = score()
    return RunStatus(
        run_id="run-1",
        suite_id="suite-1",
        experiment_id="exp-1",
        experiment_name="Synthetic",
        run_number=0,
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        provider_type="full_context",
        provider_version="0.1.0",
        answer_model_id="gpt-4o-mini-2024-07-18",
        judge_model_id="claude-sonnet-4-5",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_conversations_processed=1,
        n_questions_attempted=1,
        n_questions_succeeded=0,
        n_questions_errored=0,
        overall_score_standard=score_value,
        overall_score_audited=score_value,
        by_category_standard={"temporal": score_value},
        by_category_audited={"temporal": score_value},
        total_cost_usd=0.03,
        error_message=None,
        error_phase=None,
    )


def suite_status() -> SuiteStatus:
    return SuiteStatus(
        suite_id="suite-1",
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        config_yaml_content="experiments: []",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        n_experiments_completed=1,
        n_experiments_failed=0,
        n_experiments_in_progress=0,
        total_cost_usd=0.03,
        last_event_at=NOW,
        last_event_type="suite_completed",
    )


def experiment_result() -> ExperimentResult:
    score_value = score()
    aggregate = AggregateScore(
        experiment_id="exp-1",
        category="overall",
        mode="audited",
        n_runs=1,
        pooled_score=score_value,
        individual_run_scores=[0.8],
        mean=0.8,
        stddev=0.0,
        min=0.8,
        max=0.8,
    )
    return ExperimentResult(
        experiment_id="exp-1",
        suite_id="suite-1",
        experiment_name="Synthetic",
        n_runs=1,
        run_ids=["run-1"],
        aggregate_overall_standard=aggregate,
        aggregate_overall_audited=aggregate,
        aggregate_by_category_standard={"temporal": aggregate},
        aggregate_by_category_audited={"temporal": aggregate},
        total_cost_usd=0.03,
        config=experiment_config(),
    )


def methodology_disclosure() -> MethodologyDisclosure:
    return MethodologyDisclosure(
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        scoring_mode="audited",
        confidence_interval="Wilson 95%",
        same_vendor_warning=False,
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="checksum-1",
    )


def cost_summary() -> CostSummary:
    return CostSummary(
        total_cost_usd=0.03,
        cost_by_phase={"generate": 0.01, "judge": 0.02},
        cost_by_model={"gpt-4o-mini-2024-07-18": 0.01, "claude-sonnet-4-5": 0.02},
    )


def report_context() -> ReportContext:
    question = question_record()
    return ReportContext(
        run_status=run_status(),
        run_started_event=run_started_event(),
        suite_status=suite_status(),
        experiment_result=experiment_result(),
        questions=[question],
        failed_questions=[question],
        failure_analysis=failure_analysis_report(),
        methodology=methodology_disclosure(),
        cost_summary=cost_summary(),
    )


def model_instances() -> list[BaseModel]:
    context = report_context()
    return [
        context.methodology,
        context.cost_summary,
        context,
    ]


@pytest.mark.parametrize("instance", model_instances())
def test_reporting_contracts_round_trip_through_model_dump(instance: BaseModel) -> None:
    clone = type(instance).model_validate(instance.model_dump())

    if clone != instance:
        raise AssertionError((clone, instance))


def test_reporting_contracts_are_frozen() -> None:
    methodology = methodology_disclosure()

    with pytest.raises(ValidationError):
        methodology.scoring_mode = "standard"


@pytest.mark.parametrize(
    ("model_class", "data"),
    [
        (
            MethodologyDisclosure,
            {
                "methodology_version": "1.0",
                "methodology_profile": "canonical-v1",
                "scoring_mode": "both",
                "confidence_interval": "Wilson 95%",
                "same_vendor_warning": False,
                "benchmark_type": "locomo",
                "benchmark_version": "1.0",
                "benchmark_checksum": "checksum-1",
            },
        ),
        (
            CostSummary,
            {
                "total_cost_usd": -0.01,
                "cost_by_phase": {},
                "cost_by_model": {},
            },
        ),
        (
            CostSummary,
            {
                "total_cost_usd": 0.0,
                "cost_by_phase": {"generate": -0.01},
                "cost_by_model": {},
            },
        ),
        (
            CostSummary,
            {
                "total_cost_usd": 0.0,
                "cost_by_phase": {},
                "cost_by_model": {"model": -0.01},
            },
        ),
    ],
)
def test_reporting_contract_validation_failures(
    model_class: type[BaseModel],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model_class.model_validate(data)


def test_report_context_uses_synthetic_canonical_types() -> None:
    context = report_context()

    if not isinstance(context.run_status, RunStatus):
        raise AssertionError(context.run_status)
    if not isinstance(context.run_started_event, RunStartedEvent):
        raise AssertionError(context.run_started_event)
    if not isinstance(context.suite_status, SuiteStatus):
        raise AssertionError(context.suite_status)
    if not isinstance(context.experiment_result, ExperimentResult):
        raise AssertionError(context.experiment_result)
    if not isinstance(context.questions[0], QuestionRecord):
        raise AssertionError(context.questions[0])
    if context.failed_questions != context.questions:
        raise AssertionError(context.failed_questions)
    if context.failure_analysis.run_id != context.run_status.run_id:
        raise AssertionError(context.failure_analysis)
