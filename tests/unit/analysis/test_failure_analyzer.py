from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Sequence
from datetime import UTC, datetime

from khedron.analysis import FailureAnalyzer, PatternDetector
from khedron.analysis.types import DetectedPattern
from khedron.errors import PersistenceError
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


def test_category_breakdown_uses_all_questions_and_computes_failure_rates() -> None:
    repository = _FakeRepository(
        [
            question_record("q-single-correct"),
            question_record(
                "q-temporal-failed",
                category=QuestionCategory.TEMPORAL,
                verdict=JudgmentVerdict.INCORRECT,
            ),
            question_record("q-temporal-correct", category=QuestionCategory.TEMPORAL),
            question_record(
                "q-multi-failed",
                category=QuestionCategory.MULTI_HOP,
                verdict=JudgmentVerdict.ERROR,
            ),
        ]
    )

    report = FailureAnalyzer(pattern_detector=_RecordingDetector()).analyze("run-1", repository)

    by_category = {item.category: item for item in report.category_breakdown}
    assert [item.category for item in report.category_breakdown] == [
        QuestionCategory.MULTI_HOP,
        QuestionCategory.SINGLE_HOP,
        QuestionCategory.TEMPORAL,
    ]
    assert by_category[QuestionCategory.SINGLE_HOP].n_total == 1
    assert by_category[QuestionCategory.SINGLE_HOP].n_failed == 0
    assert by_category[QuestionCategory.SINGLE_HOP].failure_rate == 0.0
    assert by_category[QuestionCategory.TEMPORAL].n_total == 2
    assert by_category[QuestionCategory.TEMPORAL].n_failed == 1
    assert by_category[QuestionCategory.TEMPORAL].failure_rate == 0.5
    assert by_category[QuestionCategory.MULTI_HOP].n_total == 1
    assert by_category[QuestionCategory.MULTI_HOP].n_failed == 1
    assert by_category[QuestionCategory.MULTI_HOP].failure_rate == 1.0


def test_failed_question_summaries_include_hydrated_answer_and_error_details() -> None:
    base_record = question_record(
        "q-failed",
        verdict=JudgmentVerdict.ERROR,
        generated_answer=None,
        error_phase=None,
        error_message=None,
    )
    hydrated_record = base_record.model_copy(
        update={
            "generated_answer": "Hydrated generated answer",
            "error_phase": "judge",
            "error_message": "Judge output was malformed",
        }
    )
    repository = _FakeRepository([base_record], hydrated_by_id={"q-failed": hydrated_record})

    report = FailureAnalyzer(pattern_detector=_RecordingDetector()).analyze("run-1", repository)

    assert report.failed_questions[0].question_id == "q-failed"
    assert report.failed_questions[0].generated_answer == "Hydrated generated answer"
    assert report.failed_questions[0].error_phase == "judge"
    assert report.failed_questions[0].error_message == "Judge output was malformed"


def test_traceability_entries_include_hydrated_record_ids() -> None:
    base_record = question_record(
        "q-failed",
        verdict=JudgmentVerdict.INCORRECT,
        retrieval_id=None,
        response_id=None,
        judgment_id=None,
    )
    hydrated_record = base_record.model_copy(
        update={
            "retrieval_id": "ret-hydrated",
            "response_id": "resp-hydrated",
            "judgment_id": "judge-hydrated",
        }
    )
    repository = _FakeRepository([base_record], hydrated_by_id={"q-failed": hydrated_record})

    report = FailureAnalyzer(pattern_detector=_RecordingDetector()).analyze("run-1", repository)

    assert report.traceability_index[0].retrieval_id == "ret-hydrated"
    assert report.traceability_index[0].response_id == "resp-hydrated"
    assert report.traceability_index[0].judgment_id == "judge-hydrated"


def test_analyzer_delegates_to_supplied_detector_with_hydrated_failed_records() -> None:
    base_record = question_record(
        "q-failed",
        category=QuestionCategory.TEMPORAL,
        verdict=JudgmentVerdict.INCORRECT,
        question_text="What happened?",
    )
    hydrated_record = base_record.model_copy(
        update={"question_text": "What happened two years later?"}
    )
    pattern = DetectedPattern(
        pattern_name="custom_pattern",
        description="custom description",
        suggested_remedy="custom remedy",
        affected_question_ids=["q-failed"],
        n_affected_questions=1,
        confidence=0.6,
    )
    detector = _RecordingDetector(patterns=[pattern])
    repository = _FakeRepository([base_record], hydrated_by_id={"q-failed": hydrated_record})

    report = FailureAnalyzer(pattern_detector=detector).analyze("run-1", repository)

    assert detector.seen_question_ids == ["q-failed"]
    assert detector.seen_question_texts == ["What happened two years later?"]
    assert report.patterns == [pattern]


def test_hydration_fallback_uses_run_list_record_when_lookup_fails() -> None:
    base_record = question_record(
        "q-failed",
        verdict=JudgmentVerdict.INCORRECT,
        generated_answer="List generated answer",
        error_phase="generate",
        error_message="Generation failed after retries",
    )
    repository = _FakeRepository([base_record], missing_hydration_ids={"q-failed"})

    report = FailureAnalyzer(pattern_detector=_RecordingDetector()).analyze("run-1", repository)

    assert report.failed_questions[0].generated_answer == "List generated answer"
    assert report.failed_questions[0].error_phase == "generate"
    assert report.failed_questions[0].error_message == "Generation failed after retries"


def test_all_correct_run_has_breakdowns_but_no_failure_lists() -> None:
    detector = _RecordingDetector()
    repository = _FakeRepository(
        [
            question_record("q-single"),
            question_record("q-temporal", category=QuestionCategory.TEMPORAL),
        ]
    )

    report = FailureAnalyzer(pattern_detector=detector).analyze("run-1", repository)

    assert [item.category for item in report.category_breakdown] == [
        QuestionCategory.SINGLE_HOP,
        QuestionCategory.TEMPORAL,
    ]
    assert all(item.n_failed == 0 for item in report.category_breakdown)
    assert report.failed_questions == []
    assert report.patterns == []
    assert report.traceability_index == []
    assert detector.seen_question_ids == []


def test_empty_run_has_empty_breakdowns_and_failure_lists() -> None:
    report = FailureAnalyzer(pattern_detector=_RecordingDetector()).analyze(
        "run-1",
        _FakeRepository([]),
    )

    assert report.category_breakdown == []
    assert report.failed_questions == []
    assert report.patterns == []
    assert report.traceability_index == []


class _FakeRepository:
    def __init__(
        self,
        records: Sequence[QuestionRecord],
        *,
        hydrated_by_id: dict[str, QuestionRecord] | None = None,
        missing_hydration_ids: set[str] | None = None,
    ) -> None:
        self._records = list(records)
        self._hydrated_by_id = hydrated_by_id or {}
        self._missing_hydration_ids = missing_hydration_ids or set()

    def get_questions_for_run(self, run_id: str) -> list[QuestionRecord]:
        return list(self._records)

    def get_question_record(self, run_id: str, question_id: str) -> QuestionRecord:
        if question_id in self._missing_hydration_ids:
            raise PersistenceError(
                "Question record does not exist",
                run_id=run_id,
                question_id=question_id,
            )
        return self._hydrated_by_id.get(
            question_id,
            next(record for record in self._records if record.question_id == question_id),
        )


class _RecordingDetector(PatternDetector):
    def __init__(self, patterns: Sequence[DetectedPattern] | None = None) -> None:
        self.seen_question_ids: list[str] = []
        self.seen_question_texts: list[str] = []
        self._patterns = list(patterns or [])

    def detect(self, failed_records: Sequence[QuestionRecord]) -> list[DetectedPattern]:
        self.seen_question_ids = [record.question_id for record in failed_records]
        self.seen_question_texts = [record.question_text for record in failed_records]
        return list(self._patterns)


def question_record(
    question_id: str,
    *,
    category: QuestionCategory = QuestionCategory.SINGLE_HOP,
    verdict: JudgmentVerdict = JudgmentVerdict.CORRECT,
    question_text: str | None = None,
    expected_answer: str = "Expected answer",
    generated_answer: str | None = None,
    retrieval_id: str | None = "",
    response_id: str | None = "",
    judgment_id: str | None = "",
    error_phase: str | None = None,
    error_message: str | None = None,
) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=f"qe-{question_id}",
        run_id="run-1",
        question_id=question_id,
        conversation_id="conv-1",
        category=category,
        question_text=question_text or f"Question {question_id}?",
        expected_answer=expected_answer,
        is_audited_error=False,
        retrieval_id=retrieval_id if retrieval_id != "" else f"ret-{question_id}",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=[f"mem-{question_id}"],
        response_id=response_id if response_id != "" else f"resp-{question_id}",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer=(
            generated_answer
            if generated_answer is not None
            else "Expected answer"
            if verdict is JudgmentVerdict.CORRECT
            else "Different"
        ),
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id=judgment_id if judgment_id != "" else f"judge-{question_id}",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=verdict,
        score=1.0 if verdict is JudgmentVerdict.CORRECT else 0.0,
        judgment_reasoning="Synthetic judgment.",
        judgment_input_tokens=50,
        judgment_output_tokens=10,
        judgment_cost_usd=0.02,
        total_latency_ms=392.5,
        total_cost_usd=0.03,
        error_message=error_message,
        error_phase=error_phase,
    )
