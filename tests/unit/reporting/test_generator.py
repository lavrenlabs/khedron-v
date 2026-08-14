from __future__ import annotations

from datetime import UTC, datetime

# ruff: noqa: S101
from pathlib import Path

import pytest
from jinja2 import UndefinedError

from khedron.analysis.types import (
    CategoryFailureBreakdown,
    ComparisonReport,
    FailedQuestionSummary,
    FailureAnalysisReport,
    QuestionDifference,
    ScoreDelta,
    TraceabilityIndexEntry,
)
from khedron.reporting import ReportGenerator
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


def test_generator_writes_each_report_type_and_creates_parent_directories(
    tmp_path: Path,
) -> None:
    generator = ReportGenerator()
    context = report_context()
    comparison = comparison_report()
    outputs = {
        "executive": tmp_path / "nested" / "executive.md",
        "technical": tmp_path / "nested" / "technical.md",
        "failures": tmp_path / "nested" / "failures.md",
        "comparison": tmp_path / "nested" / "comparison.md",
    }

    generator.generate_executive_summary(context, outputs["executive"])
    generator.generate_technical_deep_dive(context, outputs["technical"])
    generator.generate_failure_analysis_report(context, outputs["failures"])
    generator.generate_comparison_report(comparison, outputs["comparison"])

    assert (
        outputs["executive"]
        .read_text(encoding="utf-8")
        .startswith("# Executive Summary - Synthetic Memory Run")
    )
    assert (
        outputs["technical"]
        .read_text(encoding="utf-8")
        .startswith("# Technical Deep Dive - Synthetic Memory Run")
    )
    assert (
        outputs["failures"]
        .read_text(encoding="utf-8")
        .startswith("# Failure Analysis - Synthetic Memory Run")
    )
    assert outputs["comparison"].read_text(encoding="utf-8").startswith("# Comparison Report")


def test_generator_output_has_exactly_one_trailing_newline(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "executive.md"

    ReportGenerator().generate_executive_summary(report_context(), output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_generator_renders_comparison_report_in_memory() -> None:
    rendered = ReportGenerator().render_comparison_report(comparison_report())

    assert rendered.startswith("# Comparison Report")
    assert "baseline-run" in rendered
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_generator_does_not_inject_generated_at_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "executive.md"

    ReportGenerator().generate_executive_summary(report_context(), output_path)

    assert "Generated at:" not in output_path.read_text(encoding="utf-8")


def test_generator_strict_undefined_raises_for_missing_template_variables(
    tmp_path: Path,
) -> None:
    generator = ReportGenerator()

    with pytest.raises(UndefinedError):
        generator._render_to_path("executive_summary.md.j2", tmp_path / "broken.md", {})


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
        methodology=MethodologyDisclosure(
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            scoring_mode="audited",
            confidence_interval="Wilson 95%",
            same_vendor_warning=False,
            benchmark_type="locomo",
            benchmark_version="1.0",
            benchmark_checksum="sha256:synthetic",
        ),
        cost_summary=CostSummary(
            total_cost_usd=0.03,
            cost_by_phase={"generate": 0.01, "judge": 0.02},
            cost_by_model={"answer-model": 0.01, "judge-model": 0.02},
        ),
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
                category=QuestionCategory.SINGLE_HOP,
                verdicts_by_run_id={
                    "baseline-run": JudgmentVerdict.INCORRECT,
                    "candidate-run": JudgmentVerdict.CORRECT,
                },
                scores_by_run_id={"baseline-run": 0.0, "candidate-run": 1.0},
            )
        ],
    )


def score(n_correct: int = 1) -> ScoreWithCI:
    return ScoreWithCI(
        n_total=2,
        n_correct=n_correct,
        n_errors=2 - n_correct,
        point_estimate=n_correct / 2,
        ci_95_low=0.09,
        ci_95_high=0.91,
    )


def run_status() -> RunStatus:
    score_value = score()
    return RunStatus(
        run_id="run-1",
        suite_id="suite-1",
        experiment_id="experiment-1",
        experiment_name="Synthetic Memory Run",
        run_number=0,
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        provider_type="full_context",
        provider_version="0.1.0",
        answer_model_id="answer-model",
        judge_model_id="judge-model",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_conversations_processed=1,
        n_questions_attempted=1,
        n_questions_succeeded=0,
        n_questions_errored=0,
        overall_score_standard=score_value,
        overall_score_audited=score_value,
        by_category_standard={"single_hop": score_value},
        by_category_audited={"single_hop": score_value},
        total_cost_usd=0.03,
        error_message=None,
        error_phase=None,
    )


def run_started_event() -> RunStartedEvent:
    return RunStartedEvent(
        event_id="run-1-started",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=0,
        suite_id="suite-1",
        experiment_id="experiment-1",
        experiment_name="Synthetic Memory Run",
        run_number=0,
        provider_type="full_context",
        provider_version="0.1.0",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="sha256:synthetic",
        answer_model_id="answer-model",
        answer_model_vendor="openai",
        judge_model_id="judge-model",
        judge_model_vendor="anthropic",
        config={},
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=123,
        runtime_environment={"python": "3.11"},
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
    return ExperimentResult.model_validate(
        {
            "experiment_id": "experiment-1",
            "suite_id": "suite-1",
            "experiment_name": "Synthetic Memory Run",
            "n_runs": 1,
            "run_ids": ["run-1"],
            "aggregate_overall_standard": aggregate,
            "aggregate_overall_audited": aggregate,
            "aggregate_by_category_standard": {"overall": aggregate},
            "aggregate_by_category_audited": {"overall": aggregate},
            "total_cost_usd": 0.03,
            "config": {
                "name": "Synthetic Memory Run",
                "provider": {"type": "full_context"},
                "benchmark": {"type": "locomo"},
                "answer_model": {"type": "openai", "model": "answer-model"},
                "judge": {"type": "anthropic", "model": "judge-model"},
                "top_k_retrieval": 10,
                "max_concurrent_questions": 4,
            },
        }
    )


def question_record() -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id="qe-1",
        run_id="run-1",
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Where did Alice move?",
        expected_answer="Rome",
        is_audited_error=False,
        retrieval_id="ret-1",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=["mem-1"],
        response_id="resp-1",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="Paris",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id="judge-1",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=JudgmentVerdict.INCORRECT,
        score=0.0,
        judgment_reasoning="The answer does not match.",
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
                category=QuestionCategory.SINGLE_HOP,
                n_total=1,
                n_failed=1,
                failure_rate=1.0,
            )
        ],
        failed_questions=[
            FailedQuestionSummary(
                question_id="q-1",
                category=QuestionCategory.SINGLE_HOP,
                verdict=JudgmentVerdict.INCORRECT,
                question_text="Where did Alice move?",
                expected_answer="Rome",
                generated_answer="Paris",
                error_phase=None,
                error_message=None,
            )
        ],
        patterns=[],
        traceability_index=[
            TraceabilityIndexEntry(
                question_id="q-1",
                retrieval_id="ret-1",
                response_id="resp-1",
                judgment_id="judge-1",
            )
        ],
    )
