from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

from khedron.types import JudgmentVerdict, QuestionRecord

__all__ = ["PairedComparison", "compare_paired"]


@dataclass(frozen=True)
class PairedComparison:
    """A same-question comparison of two runs, and what the disagreement supports.

    The two runs must have asked the same questions. When they also share their answers -- as a
    rejudged run does with its source -- the only thing that varied is the judge, and this is a
    direct measurement of grading disagreement rather than an inference from two point estimates.
    """

    n_paired: int
    n_both_correct: int
    n_both_wrong: int
    # The discordant cells, named for what they mean rather than b and c: these are the only
    # observations a paired test uses, and the ones a reader has to be able to check.
    n_only_baseline_correct: int
    n_only_candidate_correct: int
    p_value: float
    # Questions one run has and the other does not. A paired test over the intersection is only
    # honest if the intersection is the whole set: a partial run drops precisely the questions it
    # failed on, which biases the surviving sample in its own favour. Reported rather than assumed
    # away.
    n_baseline_only: int = 0
    n_candidate_only: int = 0

    @property
    def question_sets_match(self) -> bool:
        return self.n_baseline_only == 0 and self.n_candidate_only == 0

    @property
    def n_discordant(self) -> int:
        return self.n_only_baseline_correct + self.n_only_candidate_correct

    @property
    def net_delta(self) -> float:
        """Candidate minus baseline accuracy over the paired set, as a proportion."""
        if self.n_paired == 0:
            return 0.0
        return (self.n_only_candidate_correct - self.n_only_baseline_correct) / self.n_paired

    @property
    def is_significant(self) -> bool:
        """Whether the disagreement is beyond what chance explains at the 5% level.

        Not a claim that the difference matters: a significant 1-point difference on a benchmark
        that cannot resolve 4 points is still below the instrument. Significance and materiality
        are separate questions and reports must state both.
        """
        return self.p_value < 0.05


def compare_paired(
    baseline: Sequence[QuestionRecord],
    candidate: Sequence[QuestionRecord],
) -> PairedComparison:
    """Compare two runs question by question with an exact McNemar test.

    Why paired rather than two independent intervals. Comparing 62.1% [54.3, 69.2] against
    61.8% [54.0, 68.9] discards the fact that both numbers come from the *same questions*: the
    questions neither run got right carry no information about which is better, and they dominate
    the interval. A paired test looks only at the questions where the two disagree, which is where
    the entire signal is, and is correspondingly more sensitive.

    Exact rather than the chi-square approximation, because the comparisons this exists for can
    have few discordant pairs -- a rejudged run agrees with its source on most questions by
    construction -- and the approximation is unreliable there.

    Errored questions are excluded rather than scored as wrong. An infrastructure failure in one
    run says nothing about the other's grading or answering, and counting it as a disagreement
    would manufacture signal out of a timeout.
    """
    baseline_by_id = {record.question_id: record for record in baseline}
    candidate_by_id = {record.question_id: record for record in candidate}

    both_correct = both_wrong = only_baseline = only_candidate = 0
    for question_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        left = baseline_by_id[question_id]
        right = candidate_by_id[question_id]
        if left.verdict is JudgmentVerdict.ERROR or right.verdict is JudgmentVerdict.ERROR:
            continue
        left_correct = left.verdict is JudgmentVerdict.CORRECT
        right_correct = right.verdict is JudgmentVerdict.CORRECT
        if left_correct and right_correct:
            both_correct += 1
        elif not left_correct and not right_correct:
            both_wrong += 1
        elif left_correct:
            only_baseline += 1
        else:
            only_candidate += 1

    return PairedComparison(
        n_baseline_only=len(set(baseline_by_id) - set(candidate_by_id)),
        n_candidate_only=len(set(candidate_by_id) - set(baseline_by_id)),
        n_paired=both_correct + both_wrong + only_baseline + only_candidate,
        n_both_correct=both_correct,
        n_both_wrong=both_wrong,
        n_only_baseline_correct=only_baseline,
        n_only_candidate_correct=only_candidate,
        p_value=_exact_mcnemar_p_value(only_baseline, only_candidate),
    )


def _exact_mcnemar_p_value(only_baseline: int, only_candidate: int) -> float:
    """Two-sided exact McNemar p-value: a sign test over the discordant pairs.

    Under the null the two runs are equally likely to be the one that got a disputed question
    right, so the discordant pairs are binomial(n, 0.5). Summing both tails of that distribution
    at or beyond the observed split is the exact test; no normal approximation is involved, which
    matters because these comparisons often have few discordant pairs.

    No discordant pairs means no evidence either way, reported as p = 1.0 rather than as agreement:
    two runs that never disagree have not demonstrated equivalence, only that this test cannot
    distinguish them.
    """
    discordant = only_baseline + only_candidate
    if discordant == 0:
        return 1.0
    smaller = min(only_baseline, only_candidate)
    tail = sum(comb(discordant, index) for index in range(smaller + 1))
    p_value = 2.0 * tail / (2.0**discordant)
    return min(1.0, p_value)
