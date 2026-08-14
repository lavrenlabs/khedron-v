from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

from khedron.analysis.types import ComparisonReport, QuestionDifference, ScoreDelta
from khedron.build import UNIDENTIFIED_BUILD_SUFFIX
from khedron.eligibility import Eligibility, assess_run_eligibility
from khedron.errors import ConfigurationError
from khedron.persistence.repository import RunRepository
from khedron.types import (
    QuestionCategory,
    QuestionRecord,
    RunStartedEvent,
    RunStatus,
    ScoreWithCI,
)

__all__ = ["RunComparator"]

ComparisonMode: TypeAlias = Literal["standard", "audited"]


class RunComparator:
    """Compare one baseline run against one candidate run."""

    def compare(
        self,
        run_ids: Sequence[str],
        repository: RunRepository,
        mode: ComparisonMode = "audited",
        candidate_repository: RunRepository | None = None,
    ) -> ComparisonReport:
        """Build a two-run comparison report from public repository reads.

        `candidate_repository` exists because the comparison this project was built to make spans
        two results roots. A memory provider and its retrieval-free baseline are separate suites
        with separate `output_dir`s -- deliberately, so their runs do not share one index -- and
        with a single repository the central comparison of the whole measurement simply could not
        be executed. It defaults to `repository`, so same-root comparisons are unchanged.
        """
        if mode not in {"standard", "audited"}:
            raise ConfigurationError("Unsupported comparison mode.", mode=mode)
        if len(run_ids) != 2:
            raise ConfigurationError(
                "Phase 6 comparison requires exactly two run IDs.",
                n_run_ids=len(run_ids),
                run_ids=list(run_ids),
            )

        baseline_run_id, candidate_run_id = run_ids
        candidate_source = candidate_repository if candidate_repository is not None else repository
        if hasattr(repository, "get_strict_stage_stream"):
            baseline_eligibility = _require_eligible(repository, baseline_run_id)
            candidate_eligibility = _require_eligible(candidate_source, candidate_run_id)
            if (
                baseline_eligibility.planned_question_ids
                != candidate_eligibility.planned_question_ids
            ):
                raise ConfigurationError(
                    "Runs have different planned question ID sets and cannot be compared.",
                    baseline_run_id=baseline_run_id,
                    candidate_run_id=candidate_run_id,
                )
        baseline_status = repository.get_run_status(baseline_run_id)
        candidate_status = candidate_source.get_run_status(candidate_run_id)
        baseline_started = repository.get_run_started_event(baseline_run_id)
        candidate_started = candidate_source.get_run_started_event(candidate_run_id)
        baseline_questions = repository.get_questions_for_run(baseline_run_id)
        candidate_questions = candidate_source.get_questions_for_run(candidate_run_id)

        baseline_question_ids = {record.question_id for record in baseline_questions}
        candidate_question_ids = {record.question_id for record in candidate_questions}
        compatibility_warnings = _compatibility_warnings(
            baseline_status=baseline_status,
            candidate_status=candidate_status,
            baseline_started=baseline_started,
            candidate_started=candidate_started,
            baseline_question_ids=baseline_question_ids,
            candidate_question_ids=candidate_question_ids,
        )

        return ComparisonReport(
            run_ids=list(run_ids),
            mode=mode,
            compatible=not compatibility_warnings,
            compatibility_warnings=compatibility_warnings,
            score_deltas=_score_deltas(
                baseline_status=baseline_status,
                candidate_status=candidate_status,
                mode=mode,
            ),
            differing_questions=_question_differences(
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
                baseline_questions=baseline_questions,
                candidate_questions=candidate_questions,
            ),
        )


def _require_eligible(repository: RunRepository, run_id: str) -> Eligibility:
    eligibility = assess_run_eligibility(repository, run_id)
    if not eligibility.eligible:
        raise ConfigurationError(
            "Run is not eligible for comparison.", run_id=run_id, reasons=list(eligibility.reasons)
        )
    return eligibility


def _compatibility_warnings(
    *,
    baseline_status: RunStatus,
    candidate_status: RunStatus,
    baseline_started: RunStartedEvent,
    candidate_started: RunStartedEvent,
    baseline_question_ids: set[str],
    candidate_question_ids: set[str],
) -> list[str]:
    warnings: list[str] = []
    _append_mismatch_warning(
        warnings,
        "Methodology version",
        baseline_status.methodology_version,
        candidate_status.methodology_version,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    _append_mismatch_warning(
        warnings,
        "Methodology profile",
        baseline_status.methodology_profile,
        candidate_status.methodology_profile,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    _append_mismatch_warning(
        warnings,
        "Benchmark type",
        baseline_started.benchmark_type,
        candidate_started.benchmark_type,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    _append_mismatch_warning(
        warnings,
        "Benchmark version",
        baseline_started.benchmark_version,
        candidate_started.benchmark_version,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    _append_mismatch_warning(
        warnings,
        "Benchmark checksum",
        baseline_started.benchmark_checksum,
        candidate_started.benchmark_checksum,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    _append_mismatch_warning(
        warnings,
        "Framework version",
        baseline_started.framework_version,
        candidate_started.framework_version,
        baseline_status.run_id,
        candidate_status.run_id,
    )
    warnings.extend(_partial_corpus_warnings(baseline_started, baseline_status.run_id))
    warnings.extend(_partial_corpus_warnings(candidate_started, candidate_status.run_id))
    warnings.extend(
        _unverifiable_build_warnings(baseline_started.framework_version, baseline_status.run_id)
    )
    warnings.extend(
        _unverifiable_build_warnings(candidate_started.framework_version, candidate_status.run_id)
    )

    if baseline_question_ids != candidate_question_ids:
        baseline_only = sorted(baseline_question_ids - candidate_question_ids)
        candidate_only = sorted(candidate_question_ids - baseline_question_ids)
        warnings.append(
            "Question ID sets differ: "
            f"only in baseline run {baseline_status.run_id!r}: {baseline_only}; "
            f"only in candidate run {candidate_status.run_id!r}: {candidate_only}."
        )

    return warnings


def _partial_corpus_warnings(run_started: RunStartedEvent, run_id: str) -> list[str]:
    """Warn when a compared run measured only part of the corpus.

    The run report now says so above its own headline, but a comparison is a separate artifact and
    was the easier one to misread: two partial-corpus runs produce a delta table with confidence
    intervals and a significance column, all arithmetically correct over the questions asked and
    none of it a statement about the benchmark. Two partial runs can be compared with each other
    -- that is a legitimate use -- so this warns rather than refuses.

    Silent on runs recorded before coverage was captured: absent is unknown, and inventing a
    warning for every historical run would train the reader to skim past the real one.
    """
    evaluated = run_started.runtime_environment.get("corpus_conversations_evaluated")
    available = run_started.runtime_environment.get("corpus_conversations_available")
    if not isinstance(evaluated, int) or not isinstance(available, int):
        return []
    if isinstance(evaluated, bool) or isinstance(available, bool) or evaluated >= available:
        return []
    return [
        f"Run {run_id!r} covered {evaluated} of {available} conversations, so its scores describe "
        "the selected subset. Deltas below compare those subsets and are not a measurement of the "
        "benchmark."
    ]


def _unverifiable_build_warnings(framework_version: str, run_id: str) -> list[str]:
    """Warn when a version string identifies no build, even if both runs carry the same one.

    Equality is not enough here, which is why this is separate from the mismatch check. Two runs
    labelled `0.0.0+abc1234.dirty` share a string, not a codebase: a dirty tree is uncommitted, so
    the label says nothing about what actually executed and the two could differ arbitrarily.
    `+unidentified` is worse -- git could not answer at all. Reporting either pair as
    "direct comparison is valid" would assert something no evidence supports.
    """
    if framework_version.endswith(".dirty"):
        return [
            f"Run {run_id!r} was produced from an uncommitted working tree "
            f"({framework_version}), so its code is in no commit and two runs sharing this "
            "label need not share a codebase."
        ]
    if framework_version.endswith(f"+{UNIDENTIFIED_BUILD_SUFFIX}"):
        return [
            f"Run {run_id!r} records no identifiable build ({framework_version}), so the code "
            "that produced it cannot be established."
        ]
    return []


def _append_mismatch_warning(
    warnings: list[str],
    label: str,
    baseline_value: str,
    candidate_value: str,
    baseline_run_id: str,
    candidate_run_id: str,
) -> None:
    if baseline_value == candidate_value:
        return
    warnings.append(
        f"{label} differs: baseline run {baseline_run_id!r} has {baseline_value!r}; "
        f"candidate run {candidate_run_id!r} has {candidate_value!r}."
    )


def _score_deltas(
    *,
    baseline_status: RunStatus,
    candidate_status: RunStatus,
    mode: ComparisonMode,
) -> list[ScoreDelta]:
    baseline_overall, baseline_by_category = _select_scores(baseline_status, mode)
    candidate_overall, candidate_by_category = _select_scores(candidate_status, mode)

    deltas = [
        _score_delta(
            category="overall",
            mode=mode,
            baseline_score=baseline_overall,
            candidate_score=candidate_overall,
        )
    ]
    categories = sorted(
        set(baseline_by_category) | set(candidate_by_category),
        key=lambda category: category.value,
    )
    deltas.extend(
        _score_delta(
            category=category,
            mode=mode,
            baseline_score=baseline_by_category.get(category),
            candidate_score=candidate_by_category.get(category),
        )
        for category in categories
    )
    return deltas


def _select_scores(
    status: RunStatus,
    mode: ComparisonMode,
) -> tuple[ScoreWithCI | None, dict[QuestionCategory, ScoreWithCI]]:
    if mode == "standard":
        return status.overall_score_standard, _category_scores(
            status.by_category_standard,
            status.run_id,
        )
    return status.overall_score_audited, _category_scores(
        status.by_category_audited,
        status.run_id,
    )


def _category_scores(
    raw_scores: dict[str, ScoreWithCI],
    run_id: str,
) -> dict[QuestionCategory, ScoreWithCI]:
    scores: dict[QuestionCategory, ScoreWithCI] = {}
    for raw_category, score in raw_scores.items():
        try:
            category = QuestionCategory(raw_category)
        except ValueError as exc:
            raise ConfigurationError(
                "Run status contains an unsupported question category.",
                run_id=run_id,
                category=raw_category,
            ) from exc
        scores[category] = score
    return scores


def _score_delta(
    *,
    category: QuestionCategory | Literal["overall"],
    mode: ComparisonMode,
    baseline_score: ScoreWithCI | None,
    candidate_score: ScoreWithCI | None,
) -> ScoreDelta:
    point_delta: float | None = None
    ci_overlaps: bool | None = None
    statistically_significant: bool | None = None
    if baseline_score is not None and candidate_score is not None:
        point_delta = candidate_score.point_estimate - baseline_score.point_estimate
        ci_overlaps = _ci_overlaps(baseline_score, candidate_score)
        statistically_significant = not ci_overlaps

    return ScoreDelta(
        category=category,
        mode=mode,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        point_delta=point_delta,
        ci_overlaps=ci_overlaps,
        statistically_significant=statistically_significant,
    )


def _ci_overlaps(baseline_score: ScoreWithCI, candidate_score: ScoreWithCI) -> bool:
    return max(baseline_score.ci_95_low, candidate_score.ci_95_low) <= min(
        baseline_score.ci_95_high,
        candidate_score.ci_95_high,
    )


def _question_differences(
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_questions: Sequence[QuestionRecord],
    candidate_questions: Sequence[QuestionRecord],
) -> list[QuestionDifference]:
    baseline_by_id = {record.question_id: record for record in baseline_questions}
    candidate_by_id = {record.question_id: record for record in candidate_questions}
    shared_question_ids = sorted(set(baseline_by_id) & set(candidate_by_id))

    differences: list[QuestionDifference] = []
    for question_id in shared_question_ids:
        baseline_record = baseline_by_id[question_id]
        candidate_record = candidate_by_id[question_id]
        if (
            baseline_record.verdict == candidate_record.verdict
            and baseline_record.score == candidate_record.score
        ):
            continue
        differences.append(
            QuestionDifference(
                question_id=question_id,
                category=baseline_record.category,
                verdicts_by_run_id={
                    baseline_run_id: baseline_record.verdict,
                    candidate_run_id: candidate_record.verdict,
                },
                scores_by_run_id={
                    baseline_run_id: baseline_record.score,
                    candidate_run_id: candidate_record.score,
                },
            )
        )
    return differences
