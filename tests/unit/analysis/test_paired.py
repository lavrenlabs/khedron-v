from __future__ import annotations

from datetime import UTC, datetime

from khedron.analysis.paired import compare_paired
from khedron.types import JudgmentVerdict, QuestionCategory, QuestionRecord

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def _record(question_id: str, verdict: JudgmentVerdict | None) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=f"qe-{question_id}",
        run_id="run",
        question_id=question_id,
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="q",
        expected_answer="Expected",
        is_audited_error=False,
        retrieval_id=f"ret-{question_id}",
        retrieval_timestamp=NOW,
        response_id=f"resp-{question_id}",
        generation_timestamp=NOW,
        generated_answer="Expected" if verdict is JudgmentVerdict.CORRECT else "Other",
        judgment_id=f"judge-{question_id}",
        judgment_timestamp=NOW,
        verdict=verdict,
        score=1.0 if verdict is JudgmentVerdict.CORRECT else 0.0,
        judgment_reasoning="Synthetic.",
        total_cost_usd=0.0,
        error_message=None,
        error_phase=None,
    )


def _runs(
    pattern: list[tuple[JudgmentVerdict | None, JudgmentVerdict | None]],
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    baseline = [_record(f"q{index}", left) for index, (left, _) in enumerate(pattern)]
    candidate = [_record(f"q{index}", right) for index, (_, right) in enumerate(pattern)]
    return baseline, candidate


CORRECT = JudgmentVerdict.CORRECT
WRONG = JudgmentVerdict.INCORRECT
ERROR = JudgmentVerdict.ERROR


def test_only_the_disagreements_carry_signal() -> None:
    # The point of pairing: questions both runs answered the same way say nothing about which is
    # better, and in an unpaired interval they dominate. Two comparisons with identical discordant
    # counts must give the identical p-value however many questions both runs agreed on.
    few_agreements, _ = _runs([(CORRECT, WRONG)] * 8 + [(WRONG, CORRECT)] * 1)
    _, few_candidate = _runs([(CORRECT, WRONG)] * 8 + [(WRONG, CORRECT)] * 1)
    many = [(CORRECT, WRONG)] * 8 + [(WRONG, CORRECT)] * 1 + [(CORRECT, CORRECT)] * 500
    many_baseline, many_candidate = _runs(many)

    sparse = compare_paired(few_agreements, few_candidate)
    padded = compare_paired(many_baseline, many_candidate)

    if sparse.p_value != padded.p_value:
        raise AssertionError((sparse.p_value, padded.p_value))
    if padded.n_paired != 509 or padded.n_discordant != 9:
        raise AssertionError(padded)


def test_a_lopsided_disagreement_is_significant_and_a_balanced_one_is_not() -> None:
    lopsided_baseline, lopsided_candidate = _runs(
        [(WRONG, CORRECT)] * 15 + [(CORRECT, WRONG)] * 2 + [(CORRECT, CORRECT)] * 50
    )
    balanced_baseline, balanced_candidate = _runs(
        [(WRONG, CORRECT)] * 9 + [(CORRECT, WRONG)] * 8 + [(CORRECT, CORRECT)] * 50
    )

    lopsided = compare_paired(lopsided_baseline, lopsided_candidate)
    balanced = compare_paired(balanced_baseline, balanced_candidate)

    if not lopsided.is_significant:
        raise AssertionError(lopsided)
    if balanced.is_significant:
        raise AssertionError(balanced)
    # And the direction is readable: the candidate won the disputed questions.
    if lopsided.net_delta <= 0:
        raise AssertionError(lopsided.net_delta)


def test_perfect_agreement_reports_no_evidence_rather_than_equivalence() -> None:
    # Two runs that never disagree have not demonstrated equivalence; the test simply cannot
    # distinguish them. Reporting p = 1.0 says that. Reporting significance either way would not.
    baseline, candidate = _runs([(CORRECT, CORRECT)] * 40 + [(WRONG, WRONG)] * 10)

    result = compare_paired(baseline, candidate)

    if result.p_value != 1.0 or result.is_significant:
        raise AssertionError(result)
    if result.n_discordant != 0 or result.net_delta != 0.0:
        raise AssertionError(result)


def test_an_errored_question_is_excluded_rather_than_scored_wrong() -> None:
    # An infrastructure failure in one run says nothing about the other's grading. Counting it as
    # a disagreement would manufacture signal out of a timeout.
    baseline, candidate = _runs([(CORRECT, ERROR), (ERROR, CORRECT), (CORRECT, WRONG)])

    result = compare_paired(baseline, candidate)

    if result.n_paired != 1:
        raise AssertionError(result)
    if result.n_only_baseline_correct != 1 or result.n_only_candidate_correct != 0:
        raise AssertionError(result)


def test_the_exact_p_value_matches_the_binomial_it_claims_to_be() -> None:
    # Checked against the closed form rather than against itself: with 5 discordant pairs split
    # 5-0, the two-sided exact p is 2 * (1/2)^5 = 0.0625.
    baseline, candidate = _runs([(WRONG, CORRECT)] * 5 + [(CORRECT, CORRECT)] * 20)

    result = compare_paired(baseline, candidate)

    if abs(result.p_value - 0.0625) > 1e-12:
        raise AssertionError(result.p_value)
    # 0.0625 > 0.05: five out of five is not enough to reject at the 5% level, however lopsided.
    if result.is_significant:
        raise AssertionError(result)
