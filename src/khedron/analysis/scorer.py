from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from khedron.errors import ConfigurationError
from khedron.types import (
    JudgmentVerdict,
    QuestionCategory,
    QuestionRecord,
    RunScores,
    ScoreWithCI,
)
from khedron.utils.stats import wilson_score_interval

__all__ = ["Scorer"]

ScoringMode = Literal["standard", "audited", "both"]


class Scorer:
    """Compute immutable run scores for standard and audited modes."""

    def score_run(
        self,
        question_records: Sequence[QuestionRecord],
        mode: ScoringMode = "both",
        audit_errors: set[str] | None = None,
        scored_categories: Sequence[str] | None = None,
    ) -> RunScores:
        """Compute standard and/or audited scores for a run.

        ``scored_categories`` restricts what enters ``overall_*`` without restricting what is
        reported per category. The two are separate because a category can be worth measuring and
        worth keeping out of the headline: LoCoMo's adversarial questions are answered correctly by
        refusing, so a system that retrieves nothing scores 93.7% there and the headline stops
        describing memory. Excluding them from the *run* instead would discard the diagnostic.

        ``None`` scores everything, which is what every profile written before this split did.

        The set is passed in rather than read from a profile: the scorer has no business knowing
        that methodology profiles exist, and threading the value keeps it testable without one.
        """
        if mode not in {"standard", "audited", "both"}:
            raise ConfigurationError("Unsupported scoring mode.", mode=mode)

        self._validate_audit_errors(question_records, audit_errors)

        standard_records = list(question_records)
        audited_records = [record for record in question_records if not record.is_audited_error]
        # `by_category_*` is computed from the unrestricted lists below; only the overall figures
        # narrow, so the diagnostics survive the split intact.
        scored_standard = _in_scored_categories(standard_records, scored_categories)
        scored_audited = _in_scored_categories(audited_records, scored_categories)

        overall_standard: ScoreWithCI | None = None
        overall_audited: ScoreWithCI | None = None
        by_category_standard: dict[QuestionCategory, ScoreWithCI] = {}
        by_category_audited: dict[QuestionCategory, ScoreWithCI] = {}
        if mode in {"standard", "both"}:
            overall_standard = self.compute_score(scored_standard)
            by_category_standard = self._score_by_category(standard_records)
        if mode in {"audited", "both"}:
            overall_audited = self.compute_score(scored_audited)
            by_category_audited = self._score_by_category(audited_records)

        return RunScores(
            overall_standard=overall_standard,
            overall_audited=overall_audited,
            by_category_standard=by_category_standard,
            by_category_audited=by_category_audited,
        )

    def compute_score(self, records: Sequence[QuestionRecord]) -> ScoreWithCI | None:
        """Compute one binary Wilson score.

        Only CORRECT counts as success. ERROR, PARTIAL, UNKNOWN, and INCORRECT
        count as non-successes; ERROR, PARTIAL, and UNKNOWN also get dedicated
        counters.
        """
        if not records:
            return None

        n_total = len(records)
        n_correct = sum(1 for record in records if record.verdict is JudgmentVerdict.CORRECT)
        n_errors = sum(1 for record in records if record.verdict is JudgmentVerdict.ERROR)
        n_partial = sum(1 for record in records if record.verdict is JudgmentVerdict.PARTIAL)
        n_unknown = sum(1 for record in records if record.verdict is JudgmentVerdict.UNKNOWN)
        ci_low, ci_high = wilson_score_interval(n_correct, n_total)

        return ScoreWithCI(
            n_total=n_total,
            n_correct=n_correct,
            n_errors=n_errors,
            n_partial=n_partial,
            n_unknown=n_unknown,
            point_estimate=n_correct / n_total,
            ci_95_low=ci_low,
            ci_95_high=ci_high,
        )

    def _score_by_category(
        self,
        records: Sequence[QuestionRecord],
    ) -> dict[QuestionCategory, ScoreWithCI]:
        by_category: dict[QuestionCategory, list[QuestionRecord]] = {}
        for record in records:
            by_category.setdefault(record.category, []).append(record)

        category_scores: dict[QuestionCategory, ScoreWithCI] = {}
        for category in sorted(by_category, key=lambda item: item.value):
            score = self.compute_score(by_category[category])
            if score is not None:
                category_scores[category] = score
        return category_scores

    def _validate_audit_errors(
        self,
        question_records: Sequence[QuestionRecord],
        audit_errors: set[str] | None,
    ) -> None:
        if audit_errors is None:
            return

        present_question_ids = {record.question_id for record in question_records}
        record_audited_ids = {
            record.question_id for record in question_records if record.is_audited_error
        }
        supplied_audited_ids = audit_errors & present_question_ids

        extra_audit_errors = supplied_audited_ids - record_audited_ids
        missing_audit_errors = record_audited_ids - supplied_audited_ids
        if extra_audit_errors or missing_audit_errors:
            raise ConfigurationError(
                "Audit error set conflicts with question record audit flags.",
                extra_audit_errors=sorted(extra_audit_errors),
                missing_audit_errors=sorted(missing_audit_errors),
            )


def _in_scored_categories(
    records: Sequence[QuestionRecord],
    scored_categories: Sequence[str] | None,
) -> list[QuestionRecord]:
    if scored_categories is None:
        return list(records)
    wanted = set(scored_categories)
    return [record for record in records if record.category.value in wanted]
