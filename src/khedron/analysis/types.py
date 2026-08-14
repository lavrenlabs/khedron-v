from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from khedron.types import JudgmentVerdict, QuestionCategory, ScoreWithCI

__all__ = [
    "CategoryFailureBreakdown",
    "ComparisonReport",
    "DetectedPattern",
    "FailedQuestionSummary",
    "FailureAnalysisReport",
    "QuestionDifference",
    "ScoreDelta",
    "TraceabilityIndexEntry",
]

_Confidence: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
_NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]
_NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
_ScoreValue: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
_ScoringMode: TypeAlias = Literal["standard", "audited"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class CategoryFailureBreakdown(_FrozenModel):
    """Failure counts for one question category."""

    category: QuestionCategory
    n_total: _NonNegativeInt
    n_failed: _NonNegativeInt
    failure_rate: _ScoreValue


class FailedQuestionSummary(_FrozenModel):
    """Report-friendly summary for one failed question."""

    question_id: str
    category: QuestionCategory
    verdict: JudgmentVerdict
    question_text: str
    expected_answer: str
    generated_answer: str | None
    error_phase: str | None
    error_message: str | None


class TraceabilityIndexEntry(_FrozenModel):
    """Persisted record IDs needed to trace one question evaluation."""

    question_id: str
    retrieval_id: str | None
    response_id: str | None
    judgment_id: str | None


class DetectedPattern(_FrozenModel):
    """Runtime output from failure-pattern detection."""

    pattern_name: str
    description: str
    suggested_remedy: str
    affected_question_ids: list[str]
    n_affected_questions: _NonNegativeInt
    confidence: _Confidence


class FailureAnalysisReport(_FrozenModel):
    """Four-level failure analysis for one run."""

    run_id: str
    category_breakdown: list[CategoryFailureBreakdown]
    failed_questions: list[FailedQuestionSummary]
    patterns: list[DetectedPattern]
    traceability_index: list[TraceabilityIndexEntry]


class ScoreDelta(_FrozenModel):
    """Score comparison for one category or the overall score."""

    category: QuestionCategory | Literal["overall"]
    mode: _ScoringMode
    baseline_score: ScoreWithCI | None
    candidate_score: ScoreWithCI | None
    point_delta: float | None
    ci_overlaps: bool | None
    statistically_significant: bool | None


class QuestionDifference(_FrozenModel):
    """Per-question verdict and score differences keyed by run ID."""

    question_id: str
    category: QuestionCategory
    verdicts_by_run_id: dict[str, JudgmentVerdict]
    scores_by_run_id: dict[str, _ScoreValue]


class ComparisonReport(_FrozenModel):
    """Runtime comparison output for two or more runs."""

    run_ids: list[str]
    mode: _ScoringMode
    compatible: bool
    compatibility_warnings: list[str] = Field(default_factory=list)
    score_deltas: list[ScoreDelta]
    differing_questions: list[QuestionDifference]
