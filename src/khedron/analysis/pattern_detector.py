from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from khedron.analysis.types import DetectedPattern
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord

__all__ = [
    "MissingMemoryFailure",
    "MultiHopReasoningFailure",
    "PatternDetector",
    "PatternRule",
    "SpeakerAttributionFailure",
    "TemporalArithmeticFailure",
]

_TEMPORAL_KEYWORDS = (
    "date",
    "time",
    "day",
    "week",
    "month",
    "year",
    "before",
    "after",
    "earlier",
    "later",
    "how long",
    "duration",
    "age",
    "timeline",
)
_SPEAKER_ATTRIBUTION_KEYWORDS = (
    "who said",
    "speaker",
    "said",
    "told",
    "mentioned",
    "asked",
    "replied",
    "responded",
    "according to",
)


class PatternRule(ABC):
    """Heuristic rule that identifies one runtime failure pattern."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable runtime pattern name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Report-facing explanation of the pattern."""
        raise NotImplementedError

    @property
    @abstractmethod
    def suggested_remedy(self) -> str:
        """Report-facing remediation suggestion."""
        raise NotImplementedError

    @property
    @abstractmethod
    def confidence(self) -> float:
        """Rule-level confidence, not a statistical probability."""
        raise NotImplementedError

    @abstractmethod
    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        """Return matching failed question IDs."""
        raise NotImplementedError


class TemporalArithmeticFailure(PatternRule):
    """Detect failed temporal questions that appear to require time reasoning."""

    @property
    def name(self) -> str:
        return "temporal_arithmetic_failure"

    @property
    def description(self) -> str:
        return "Failed temporal questions whose text or expected answer references dates or time."

    @property
    def suggested_remedy(self) -> str:
        return "Review temporal parsing, timeline construction, and date arithmetic prompts."

    @property
    def confidence(self) -> float:
        return 0.75

    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        return [
            record.question_id
            for record in failed_records
            if _is_failed(record)
            and record.category is QuestionCategory.TEMPORAL
            and _contains_keyword(
                f"{record.question_text} {record.expected_answer}",
                _TEMPORAL_KEYWORDS,
            )
        ]


class MissingMemoryFailure(PatternRule):
    """Detect failed questions where retrieval returned no memories."""

    @property
    def name(self) -> str:
        return "missing_memory_failure"

    @property
    def description(self) -> str:
        return "Failed questions where no memories were retrieved before answer generation."

    @property
    def suggested_remedy(self) -> str:
        return (
            "Increase retrieval recall or inspect provider ingestion coverage for these questions."
        )

    @property
    def confidence(self) -> float:
        return 0.9

    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        return [
            record.question_id
            for record in failed_records
            if _is_failed(record) and record.n_memories_retrieved == 0
        ]


class SpeakerAttributionFailure(PatternRule):
    """Detect failed questions that ask about who said or mentioned something."""

    @property
    def name(self) -> str:
        return "speaker_attribution_failure"

    @property
    def description(self) -> str:
        return "Failed questions whose text asks for speaker attribution."

    @property
    def suggested_remedy(self) -> str:
        return "Inspect speaker metadata preservation and attribution cues in retrieval prompts."

    @property
    def confidence(self) -> float:
        return 0.8

    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        return [
            record.question_id
            for record in failed_records
            if _is_failed(record)
            and _contains_keyword(record.question_text, _SPEAKER_ATTRIBUTION_KEYWORDS)
        ]


class MultiHopReasoningFailure(PatternRule):
    """Detect failed multi-hop questions with at least one retrieved memory."""

    @property
    def name(self) -> str:
        return "multi_hop_reasoning_failure"

    @property
    def description(self) -> str:
        return "Failed multi-hop questions where at least one memory was retrieved."

    @property
    def suggested_remedy(self) -> str:
        return (
            "Review multi-hop synthesis prompts and whether retrieved memories "
            "are presented clearly."
        )

    @property
    def confidence(self) -> float:
        return 0.7

    def match(self, failed_records: Sequence[QuestionRecord]) -> list[str]:
        return [
            record.question_id
            for record in failed_records
            if _is_failed(record)
            and record.category is QuestionCategory.MULTI_HOP
            and record.n_memories_retrieved is not None
            and record.n_memories_retrieved > 0
        ]


class PatternDetector:
    """Run heuristic pattern rules over failed question records."""

    def __init__(self, rules: Sequence[PatternRule] | None = None) -> None:
        self._rules: tuple[PatternRule, ...] = (
            tuple(rules) if rules is not None else _default_rules()
        )

    def detect(self, failed_records: Sequence[QuestionRecord]) -> list[DetectedPattern]:
        """Return detected patterns in configured rule order."""
        failed_question_ids = {
            record.question_id for record in failed_records if _is_failed(record)
        }
        detected: list[DetectedPattern] = []
        for rule in self._rules:
            affected_question_ids = sorted(set(rule.match(failed_records)) & failed_question_ids)
            if not affected_question_ids:
                continue
            detected.append(
                DetectedPattern(
                    pattern_name=rule.name,
                    description=rule.description,
                    suggested_remedy=rule.suggested_remedy,
                    affected_question_ids=affected_question_ids,
                    n_affected_questions=len(affected_question_ids),
                    confidence=rule.confidence,
                )
            )
        return detected


def _default_rules() -> tuple[PatternRule, ...]:
    return (
        TemporalArithmeticFailure(),
        MissingMemoryFailure(),
        SpeakerAttributionFailure(),
        MultiHopReasoningFailure(),
    )


def _is_failed(record: QuestionRecord) -> bool:
    return record.verdict is not JudgmentVerdict.CORRECT


def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    haystack = text.casefold()
    return any(keyword in haystack for keyword in keywords)
