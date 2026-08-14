from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import ValidationError

from khedron.analysis import Scorer
from khedron.errors import ConfigurationError
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord
from khedron.utils.stats import wilson_score_interval

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


def test_compute_score_counts_mixed_verdicts_and_wilson_interval() -> None:
    records = [
        question_record("q-correct-1", verdict=JudgmentVerdict.CORRECT),
        question_record("q-correct-2", verdict=JudgmentVerdict.CORRECT),
        question_record("q-incorrect", verdict=JudgmentVerdict.INCORRECT),
        question_record("q-error", verdict=JudgmentVerdict.ERROR),
        question_record("q-partial", verdict=JudgmentVerdict.PARTIAL),
        question_record("q-unknown", verdict=JudgmentVerdict.UNKNOWN),
    ]

    score = Scorer().compute_score(records)

    assert score is not None
    assert score.n_total == 6
    assert score.n_correct == 2
    assert score.n_errors == 1
    assert score.n_partial == 1
    assert score.n_unknown == 1
    assert score.point_estimate == pytest.approx(2 / 6)
    expected_low, expected_high = wilson_score_interval(2, 6)
    assert score.ci_95_low == expected_low
    assert score.ci_95_high == expected_high


def test_scorer_uses_canonical_wilson_interval_for_eighty_two_of_one_hundred() -> None:
    records = [
        question_record(f"q-correct-{index}", verdict=JudgmentVerdict.CORRECT)
        for index in range(82)
    ]
    records.extend(
        question_record(f"q-incorrect-{index}", verdict=JudgmentVerdict.INCORRECT)
        for index in range(18)
    )

    scores = Scorer().score_run(records, mode="standard")

    assert scores.overall_standard is not None
    assert round(scores.overall_standard.ci_95_low, 4) == 0.7333
    assert round(scores.overall_standard.ci_95_high, 4) == 0.8830


def test_audited_mode_excludes_record_flagged_audit_errors() -> None:
    scores = Scorer().score_run(
        [
            question_record("q-good", verdict=JudgmentVerdict.CORRECT),
            question_record(
                "q-audited-wrong",
                verdict=JudgmentVerdict.INCORRECT,
                is_audited_error=True,
            ),
            question_record(
                "q-audited-correct",
                verdict=JudgmentVerdict.CORRECT,
                is_audited_error=True,
            ),
        ],
        mode="both",
    )

    assert scores.overall_standard is not None
    assert scores.overall_audited is not None
    assert scores.overall_standard.n_total == 3
    assert scores.overall_standard.n_correct == 2
    assert scores.overall_audited.n_total == 1
    assert scores.overall_audited.n_correct == 1


@pytest.mark.parametrize(
    ("mode", "has_standard", "has_audited"),
    [
        ("standard", True, False),
        ("audited", False, True),
        ("both", True, True),
    ],
)
def test_score_run_mode_populates_only_requested_fields(
    mode: Literal["standard", "audited", "both"],
    has_standard: bool,
    has_audited: bool,
) -> None:
    scores = Scorer().score_run([question_record("q-1")], mode=mode)

    assert (scores.overall_standard is not None) is has_standard
    assert bool(scores.by_category_standard) is has_standard
    assert (scores.overall_audited is not None) is has_audited
    assert bool(scores.by_category_audited) is has_audited


def test_all_audited_input_returns_empty_audited_scores() -> None:
    scores = Scorer().score_run(
        [
            question_record(
                "q-audited-1",
                verdict=JudgmentVerdict.CORRECT,
                is_audited_error=True,
            ),
            question_record(
                "q-audited-2",
                verdict=JudgmentVerdict.INCORRECT,
                is_audited_error=True,
            ),
        ]
    )

    assert scores.overall_standard is not None
    assert scores.overall_audited is None
    assert scores.by_category_audited == {}


def test_category_maps_include_only_categories_remaining_after_filtering() -> None:
    scores = Scorer().score_run(
        [
            question_record(
                "q-temporal-audited",
                category=QuestionCategory.TEMPORAL,
                is_audited_error=True,
            ),
            question_record(
                "q-single-hop",
                category=QuestionCategory.SINGLE_HOP,
                verdict=JudgmentVerdict.INCORRECT,
            ),
        ],
        mode="audited",
    )

    assert scores.by_category_standard == {}
    assert set(scores.by_category_audited) == {QuestionCategory.SINGLE_HOP}


def test_conflicting_audit_errors_raise_configuration_error_with_context() -> None:
    records = [
        question_record("q-flagged", is_audited_error=True),
        question_record("q-unflagged", is_audited_error=False),
    ]

    with pytest.raises(ConfigurationError) as exc_info:
        Scorer().score_run(records, audit_errors={"q-unflagged"})

    message = str(exc_info.value)
    assert "extra_audit_errors" in message
    assert "missing_audit_errors" in message
    assert "q-unflagged" in message
    assert "q-flagged" in message


def test_audit_errors_for_absent_questions_do_not_fail_partial_runs() -> None:
    scores = Scorer().score_run(
        [question_record("q-present", is_audited_error=False)],
        audit_errors={"q-absent"},
    )

    assert scores.overall_standard is not None
    assert scores.overall_standard.n_total == 1


def test_returned_run_scores_are_frozen() -> None:
    scores = Scorer().score_run([question_record("q-1")])

    with pytest.raises(ValidationError):
        scores.overall_standard = None


def question_record(
    question_id: str,
    *,
    category: QuestionCategory = QuestionCategory.SINGLE_HOP,
    verdict: JudgmentVerdict = JudgmentVerdict.CORRECT,
    is_audited_error: bool = False,
) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=f"qe-{question_id}",
        run_id="run-1",
        question_id=question_id,
        conversation_id="conv-1",
        category=category,
        question_text=f"Question {question_id}?",
        expected_answer="Expected",
        is_audited_error=is_audited_error,
        retrieval_id="ret-1",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=["mem-1"],
        response_id="resp-1",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="Expected" if verdict is JudgmentVerdict.CORRECT else "Different",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id="judgment-1",
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


def test_scored_categories_narrow_the_headline_and_keep_every_diagnostic() -> None:
    # The whole point of the split. A test asserting only `overall_*` would pass with the excluded
    # category silently dropped from the run, which is the opposite of the intent: adversarial must
    # still be measured and reported, just not counted in the number that gets quoted.
    records = [
        question_record("s1", verdict=JudgmentVerdict.CORRECT),
        question_record("s2", verdict=JudgmentVerdict.INCORRECT),
        question_record(
            "a1", category=QuestionCategory.ADVERSARIAL, verdict=JudgmentVerdict.CORRECT
        ),
        question_record(
            "a2", category=QuestionCategory.ADVERSARIAL, verdict=JudgmentVerdict.CORRECT
        ),
    ]

    scores = Scorer().score_run(records, scored_categories=["single_hop"])

    assert scores.overall_standard is not None
    # 1 of 2 scored questions, not 3 of 4: the two adversarial correct answers stay out.
    assert scores.overall_standard.n_total == 2
    assert scores.overall_standard.n_correct == 1
    # And the excluded category is still reported, because the split changes the headline, not
    # what is measured.
    assert QuestionCategory.ADVERSARIAL in scores.by_category_standard
    assert scores.by_category_standard[QuestionCategory.ADVERSARIAL].n_total == 2


def test_no_scored_categories_preserves_pre_split_behaviour() -> None:
    # Guards every profile written before the split, and every existing expectation in this file.
    records = [
        question_record("s1", verdict=JudgmentVerdict.CORRECT),
        question_record(
            "a1", category=QuestionCategory.ADVERSARIAL, verdict=JudgmentVerdict.CORRECT
        ),
    ]

    scores = Scorer().score_run(records)

    assert scores.overall_standard is not None
    assert scores.overall_standard.n_total == 2
    assert scores.overall_standard.n_correct == 2


def test_the_day_one_headline_figures_reproduce_through_the_split() -> None:
    # The strongest available check: data whose answers are already known. The day-1 run scored
    # 536/1986 = 26.99% over all five categories and 118/1540 = 7.66% over the four scored ones.
    # Synthesised at those proportions rather than replayed, because results/ is gitignored -- so
    # this asserts the arithmetic of the split, not the artifacts themselves.
    scored = [
        question_record(
            f"s{index}",
            verdict=JudgmentVerdict.CORRECT if index < 118 else JudgmentVerdict.INCORRECT,
        )
        for index in range(1540)
    ]
    adversarial = [
        question_record(
            f"a{index}",
            category=QuestionCategory.ADVERSARIAL,
            verdict=JudgmentVerdict.CORRECT if index < 418 else JudgmentVerdict.INCORRECT,
        )
        for index in range(446)
    ]
    records = scored + adversarial

    everything = Scorer().score_run(records)
    subset = Scorer().score_run(records, scored_categories=["single_hop"])

    assert everything.overall_standard is not None
    assert subset.overall_standard is not None
    assert (everything.overall_standard.n_correct, everything.overall_standard.n_total) == (
        536,
        1986,
    )
    assert round(everything.overall_standard.point_estimate * 100, 2) == 26.99
    assert (subset.overall_standard.n_correct, subset.overall_standard.n_total) == (118, 1540)
    assert round(subset.overall_standard.point_estimate * 100, 2) == 7.66
