from __future__ import annotations

from collections.abc import Sequence

from khedron.analysis.pattern_detector import PatternDetector
from khedron.analysis.types import (
    CategoryFailureBreakdown,
    FailedQuestionSummary,
    FailureAnalysisReport,
    TraceabilityIndexEntry,
)
from khedron.errors import PersistenceError
from khedron.persistence.repository import RunRepository
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord

__all__ = ["FailureAnalyzer"]


class FailureAnalyzer:
    """Produce four-level failure analysis for one persisted run."""

    def __init__(self, pattern_detector: PatternDetector | None = None) -> None:
        self._pattern_detector = pattern_detector or PatternDetector()

    def analyze(self, run_id: str, repository: RunRepository) -> FailureAnalysisReport:
        """Build category, question, pattern, and traceability failure analysis."""
        question_records = repository.get_questions_for_run(run_id)
        failed_records = self._hydrate_failed_records(run_id, repository, question_records)
        failed_records = sorted(failed_records, key=lambda record: record.question_id)

        return FailureAnalysisReport(
            run_id=run_id,
            category_breakdown=_build_category_breakdown(question_records),
            failed_questions=[_failed_question_summary(record) for record in failed_records],
            patterns=self._pattern_detector.detect(failed_records),
            traceability_index=[_traceability_entry(record) for record in failed_records],
        )

    def _hydrate_failed_records(
        self,
        run_id: str,
        repository: RunRepository,
        question_records: Sequence[QuestionRecord],
    ) -> list[QuestionRecord]:
        failed_records: list[QuestionRecord] = []
        for record in question_records:
            if record.verdict is JudgmentVerdict.CORRECT:
                continue
            try:
                failed_records.append(repository.get_question_record(run_id, record.question_id))
            except PersistenceError:
                failed_records.append(record)
        return failed_records


def _build_category_breakdown(
    question_records: Sequence[QuestionRecord],
) -> list[CategoryFailureBreakdown]:
    by_category: dict[QuestionCategory, list[QuestionRecord]] = {}
    for record in question_records:
        by_category.setdefault(record.category, []).append(record)

    breakdown: list[CategoryFailureBreakdown] = []
    for category in sorted(by_category, key=lambda item: item.value):
        records = by_category[category]
        n_total = len(records)
        n_failed = sum(1 for record in records if record.verdict is not JudgmentVerdict.CORRECT)
        breakdown.append(
            CategoryFailureBreakdown(
                category=category,
                n_total=n_total,
                n_failed=n_failed,
                failure_rate=n_failed / n_total,
            )
        )
    return breakdown


def _failed_question_summary(record: QuestionRecord) -> FailedQuestionSummary:
    return FailedQuestionSummary(
        question_id=record.question_id,
        category=record.category,
        verdict=record.verdict,
        question_text=record.question_text,
        expected_answer=record.expected_answer,
        generated_answer=record.generated_answer,
        error_phase=record.error_phase,
        error_message=record.error_message,
    )


def _traceability_entry(record: QuestionRecord) -> TraceabilityIndexEntry:
    return TraceabilityIndexEntry(
        question_id=record.question_id,
        retrieval_id=record.retrieval_id,
        response_id=record.response_id,
        judgment_id=record.judgment_id,
    )
