from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Sequence
from datetime import UTC, datetime

from khedron.analysis import (
    MissingMemoryFailure,
    MultiHopReasoningFailure,
    PatternDetector,
    PatternRule,
    SpeakerAttributionFailure,
    TemporalArithmeticFailure,
)
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


def test_temporal_arithmetic_failure_matches_failed_temporal_keyword_records() -> None:
    records = [
        question_record(
            "q-match",
            category=QuestionCategory.TEMPORAL,
            verdict=JudgmentVerdict.INCORRECT,
            question_text="How many years after Alice moved did Bob visit?",
        ),
        question_record(
            "q-nearby",
            category=QuestionCategory.TEMPORAL,
            verdict=JudgmentVerdict.INCORRECT,
            question_text="What gift did Alice buy?",
        ),
        question_record(
            "q-correct",
            category=QuestionCategory.TEMPORAL,
            verdict=JudgmentVerdict.CORRECT,
            question_text="What year did Alice move?",
        ),
    ]

    assert TemporalArithmeticFailure().match(records) == ["q-match"]


def test_missing_memory_failure_matches_zero_retrievals_but_not_unknown() -> None:
    records = [
        question_record(
            "q-match",
            verdict=JudgmentVerdict.ERROR,
            n_memories_retrieved=0,
        ),
        question_record(
            "q-unknown",
            verdict=JudgmentVerdict.INCORRECT,
            n_memories_retrieved=None,
        ),
        question_record(
            "q-correct",
            verdict=JudgmentVerdict.CORRECT,
            n_memories_retrieved=0,
        ),
    ]

    assert MissingMemoryFailure().match(records) == ["q-match"]


def test_speaker_attribution_failure_matches_failed_attribution_questions() -> None:
    records = [
        question_record(
            "q-match",
            verdict=JudgmentVerdict.PARTIAL,
            question_text="Who said they wanted to visit in July?",
        ),
        question_record(
            "q-nearby",
            verdict=JudgmentVerdict.INCORRECT,
            question_text="What was the destination in July?",
        ),
        question_record(
            "q-correct",
            verdict=JudgmentVerdict.CORRECT,
            question_text="Who said they wanted to visit in July?",
        ),
    ]

    assert SpeakerAttributionFailure().match(records) == ["q-match"]


def test_multi_hop_reasoning_failure_matches_failed_multi_hop_with_retrievals() -> None:
    records = [
        question_record(
            "q-match",
            category=QuestionCategory.MULTI_HOP,
            verdict=JudgmentVerdict.UNKNOWN,
            n_memories_retrieved=2,
        ),
        question_record(
            "q-no-memory",
            category=QuestionCategory.MULTI_HOP,
            verdict=JudgmentVerdict.INCORRECT,
            n_memories_retrieved=0,
        ),
        question_record(
            "q-correct",
            category=QuestionCategory.MULTI_HOP,
            verdict=JudgmentVerdict.CORRECT,
            n_memories_retrieved=2,
        ),
        question_record(
            "q-single-hop",
            category=QuestionCategory.SINGLE_HOP,
            verdict=JudgmentVerdict.INCORRECT,
            n_memories_retrieved=2,
        ),
    ]

    assert MultiHopReasoningFailure().match(records) == ["q-match"]


def test_detector_preserves_rule_order_and_sorts_unique_affected_ids() -> None:
    detector = PatternDetector(
        rules=[
            _StaticRule("second_pattern", ["q-b", "q-a", "q-b"]),
            _StaticRule("first_pattern", ["q-c"]),
        ]
    )

    patterns = detector.detect(
        [
            question_record("q-a", verdict=JudgmentVerdict.INCORRECT),
            question_record("q-b", verdict=JudgmentVerdict.INCORRECT),
            question_record("q-c", verdict=JudgmentVerdict.INCORRECT),
        ]
    )

    assert [pattern.pattern_name for pattern in patterns] == [
        "second_pattern",
        "first_pattern",
    ]
    assert patterns[0].affected_question_ids == ["q-a", "q-b"]
    assert patterns[0].n_affected_questions == 2


def test_detector_does_not_emit_ids_absent_from_failed_records() -> None:
    detector = PatternDetector(rules=[_StaticRule("custom_pattern", ["q-present", "q-absent"])])

    patterns = detector.detect(
        [
            question_record("q-present", verdict=JudgmentVerdict.INCORRECT),
            question_record("q-correct", verdict=JudgmentVerdict.CORRECT),
        ]
    )

    assert len(patterns) == 1
    assert patterns[0].affected_question_ids == ["q-present"]


def test_detector_returns_empty_list_when_no_rules_match() -> None:
    detector = PatternDetector(rules=[_StaticRule("empty_pattern", [])])

    assert detector.detect([question_record("q-a", verdict=JudgmentVerdict.INCORRECT)]) == []


class _StaticRule(PatternRule):
    def __init__(self, name: str, question_ids: Sequence[str]) -> None:
        self._name = name
        self._question_ids = list(question_ids)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} description"

    @property
    def suggested_remedy(self) -> str:
        return f"{self._name} remedy"

    @property
    def confidence(self) -> float:
        return 0.5

    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        return list(self._question_ids)


def question_record(
    question_id: str,
    *,
    category: QuestionCategory = QuestionCategory.SINGLE_HOP,
    verdict: JudgmentVerdict = JudgmentVerdict.CORRECT,
    question_text: str | None = None,
    expected_answer: str = "Expected answer",
    n_memories_retrieved: int | None = 1,
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
        retrieval_id=f"ret-{question_id}",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=n_memories_retrieved,
        retrieved_memory_ids=[f"mem-{question_id}"] if n_memories_retrieved else [],
        response_id=f"resp-{question_id}",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="Expected answer" if verdict is JudgmentVerdict.CORRECT else "Different",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id=f"judge-{question_id}",
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
        error_message=None,
        error_phase=None,
    )
