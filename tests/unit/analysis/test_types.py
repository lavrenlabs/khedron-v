from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import khedron.analysis.types as analysis_types
from khedron.analysis.types import (
    CategoryFailureBreakdown,
    ComparisonReport,
    DetectedPattern,
    FailedQuestionSummary,
    FailureAnalysisReport,
    QuestionDifference,
    ScoreDelta,
    TraceabilityIndexEntry,
)
from khedron.types import FailurePattern, JudgmentVerdict, QuestionCategory, ScoreWithCI


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


def detected_pattern() -> DetectedPattern:
    return DetectedPattern(
        pattern_name="missing_memory_failure",
        description="No memory was retrieved before answering.",
        suggested_remedy="Increase retrieval recall.",
        affected_question_ids=["q-1"],
        n_affected_questions=1,
        confidence=0.9,
    )


def failure_analysis_report() -> FailureAnalysisReport:
    return FailureAnalysisReport(
        run_id="run-1",
        category_breakdown=[
            CategoryFailureBreakdown(
                category=QuestionCategory.TEMPORAL,
                n_total=2,
                n_failed=1,
                failure_rate=0.5,
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
        patterns=[detected_pattern()],
        traceability_index=[
            TraceabilityIndexEntry(
                question_id="q-1",
                retrieval_id="ret-1",
                response_id="resp-1",
                judgment_id="judge-1",
            )
        ],
    )


def comparison_report() -> ComparisonReport:
    score_value = score()
    return ComparisonReport(
        run_ids=["baseline-run", "candidate-run"],
        mode="audited",
        compatible=False,
        compatibility_warnings=["benchmark checksum differs"],
        score_deltas=[
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=score_value,
                candidate_score=score_value,
                point_delta=0.0,
                ci_overlaps=True,
                statistically_significant=False,
            )
        ],
        differing_questions=[
            QuestionDifference(
                question_id="q-1",
                category=QuestionCategory.TEMPORAL,
                verdicts_by_run_id={
                    "baseline-run": JudgmentVerdict.CORRECT,
                    "candidate-run": JudgmentVerdict.INCORRECT,
                },
                scores_by_run_id={"baseline-run": 1.0, "candidate-run": 0.0},
            )
        ],
    )


def model_instances() -> list[BaseModel]:
    report = failure_analysis_report()
    comparison = comparison_report()
    return [
        report.category_breakdown[0],
        report.failed_questions[0],
        report.patterns[0],
        report.traceability_index[0],
        report,
        comparison.score_deltas[0],
        comparison.differing_questions[0],
        comparison,
    ]


@pytest.mark.parametrize("instance", model_instances())
def test_analysis_contracts_round_trip_through_model_dump(instance: BaseModel) -> None:
    clone = type(instance).model_validate(instance.model_dump())

    if clone != instance:
        raise AssertionError((clone, instance))


def test_analysis_contracts_are_frozen() -> None:
    pattern = detected_pattern()

    with pytest.raises(ValidationError):
        pattern.pattern_name = "changed"


def test_comparison_report_default_warnings_are_independent() -> None:
    first_report = ComparisonReport(
        run_ids=["run-1", "run-2"],
        mode="standard",
        compatible=True,
        score_deltas=[],
        differing_questions=[],
    )
    second_report = ComparisonReport(
        run_ids=["run-3", "run-4"],
        mode="standard",
        compatible=True,
        score_deltas=[],
        differing_questions=[],
    )

    first_report.compatibility_warnings.append("methodology profile differs")

    if second_report.compatibility_warnings:
        raise AssertionError(second_report.compatibility_warnings)


@pytest.mark.parametrize(
    ("model_class", "data"),
    [
        (
            CategoryFailureBreakdown,
            {
                "category": QuestionCategory.TEMPORAL,
                "n_total": -1,
                "n_failed": 0,
                "failure_rate": 0.0,
            },
        ),
        (
            CategoryFailureBreakdown,
            {
                "category": QuestionCategory.TEMPORAL,
                "n_total": 1,
                "n_failed": 0,
                "failure_rate": 1.1,
            },
        ),
        (
            DetectedPattern,
            {
                "pattern_name": "pattern",
                "description": "description",
                "suggested_remedy": "remedy",
                "affected_question_ids": ["q-1"],
                "n_affected_questions": -1,
                "confidence": 0.5,
            },
        ),
        (
            DetectedPattern,
            {
                "pattern_name": "pattern",
                "description": "description",
                "suggested_remedy": "remedy",
                "affected_question_ids": ["q-1"],
                "n_affected_questions": 1,
                "confidence": 1.1,
            },
        ),
        (
            ComparisonReport,
            {
                "run_ids": ["run-1", "run-2"],
                "mode": "both",
                "compatible": True,
                "score_deltas": [],
                "differing_questions": [],
            },
        ),
        (
            QuestionDifference,
            {
                "question_id": "q-1",
                "category": QuestionCategory.TEMPORAL,
                "verdicts_by_run_id": {"run-1": JudgmentVerdict.CORRECT},
                "scores_by_run_id": {"run-1": 1.2},
            },
        ),
    ],
)
def test_analysis_contract_validation_failures(
    model_class: type[BaseModel],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model_class.model_validate(data)


def test_detected_pattern_does_not_collide_with_persisted_failure_pattern() -> None:
    if hasattr(analysis_types, "FailurePattern"):
        raise AssertionError("analysis.types must not export a runtime FailurePattern")
    if "FailurePattern" in analysis_types.__all__:
        raise AssertionError(analysis_types.__all__)
    if DetectedPattern.__name__ == FailurePattern.__name__:
        raise AssertionError((DetectedPattern, FailurePattern))
