from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final, cast

import pytest

from khedron.analysis import RunComparator
from khedron.errors import ConfigurationError
from khedron.types import (
    JudgmentVerdict,
    QuestionCategory,
    QuestionRecord,
    RunStartedEvent,
    RunStatus,
    ScoreWithCI,
)

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
BASELINE_RUN_ID = "baseline-run"
CANDIDATE_RUN_ID = "candidate-run"
_MISSING: Final = object()


def test_compare_compatible_runs_returns_no_warnings() -> None:
    report = RunComparator().compare(
        [BASELINE_RUN_ID, CANDIDATE_RUN_ID],
        fake_repository(),
    )

    assert report.run_ids == [BASELINE_RUN_ID, CANDIDATE_RUN_ID]
    assert report.mode == "audited"
    assert report.compatible is True
    assert report.compatibility_warnings == []


def test_standard_mode_computes_overall_and_sorted_category_deltas() -> None:
    baseline_temporal = score(0.4, low=0.25, high=0.55)
    candidate_temporal = score(0.6, low=0.45, high=0.75)
    repository = fake_repository(
        baseline_status=run_status(
            BASELINE_RUN_ID,
            overall_standard=score(0.5, low=0.35, high=0.65),
            by_category_standard={
                QuestionCategory.SINGLE_HOP: score(0.8),
                QuestionCategory.TEMPORAL: baseline_temporal,
            },
        ),
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            overall_standard=score(0.7, low=0.55, high=0.85),
            by_category_standard={
                QuestionCategory.MULTI_HOP: score(0.9),
                QuestionCategory.TEMPORAL: candidate_temporal,
            },
        ),
    )

    report = RunComparator().compare(
        [BASELINE_RUN_ID, CANDIDATE_RUN_ID],
        repository,
        mode="standard",
    )

    assert [delta.category for delta in report.score_deltas] == [
        "overall",
        QuestionCategory.MULTI_HOP,
        QuestionCategory.SINGLE_HOP,
        QuestionCategory.TEMPORAL,
    ]
    assert report.score_deltas[0].point_delta == pytest.approx(0.2)
    assert report.score_deltas[1].baseline_score is None
    assert report.score_deltas[1].candidate_score is not None
    assert report.score_deltas[1].point_delta is None
    assert report.score_deltas[2].baseline_score is not None
    assert report.score_deltas[2].candidate_score is None
    assert report.score_deltas[2].statistically_significant is None
    assert report.score_deltas[3].baseline_score == baseline_temporal
    assert report.score_deltas[3].candidate_score == candidate_temporal
    assert report.score_deltas[3].point_delta == pytest.approx(0.2)


def test_audited_mode_uses_audited_scores_and_category_maps() -> None:
    baseline_audited = score(0.65)
    candidate_audited = score(0.75)
    baseline_category = score(0.5)
    candidate_category = score(0.7)
    repository = fake_repository(
        baseline_status=run_status(
            BASELINE_RUN_ID,
            overall_standard=score(0.1),
            overall_audited=baseline_audited,
            by_category_standard={QuestionCategory.TEMPORAL: score(0.1)},
            by_category_audited={QuestionCategory.TEMPORAL: baseline_category},
        ),
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            overall_standard=score(0.2),
            overall_audited=candidate_audited,
            by_category_standard={QuestionCategory.TEMPORAL: score(0.2)},
            by_category_audited={QuestionCategory.TEMPORAL: candidate_category},
        ),
    )

    report = RunComparator().compare(
        [BASELINE_RUN_ID, CANDIDATE_RUN_ID],
        repository,
        mode="audited",
    )

    assert report.score_deltas[0].baseline_score == baseline_audited
    assert report.score_deltas[0].candidate_score == candidate_audited
    assert report.score_deltas[1].category is QuestionCategory.TEMPORAL
    assert report.score_deltas[1].baseline_score == baseline_category
    assert report.score_deltas[1].candidate_score == candidate_category


def test_ci_overlap_and_non_overlap_significance_are_flagged() -> None:
    repository = fake_repository(
        baseline_status=run_status(
            BASELINE_RUN_ID,
            overall_audited=score(0.3, low=0.2, high=0.4),
            by_category_audited={
                QuestionCategory.TEMPORAL: score(0.4, low=0.2, high=0.5),
            },
        ),
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            overall_audited=score(0.6, low=0.5, high=0.7),
            by_category_audited={
                QuestionCategory.TEMPORAL: score(0.7, low=0.5, high=0.8),
            },
        ),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.score_deltas[0].ci_overlaps is False
    assert report.score_deltas[0].statistically_significant is True
    assert report.score_deltas[1].ci_overlaps is True
    assert report.score_deltas[1].statistically_significant is False


def test_missing_baseline_or_candidate_scores_have_unknown_deltas_and_significance() -> None:
    repository = fake_repository(
        baseline_status=run_status(
            BASELINE_RUN_ID,
            overall_audited=None,
            by_category_audited={QuestionCategory.SINGLE_HOP: score(0.8)},
        ),
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            overall_audited=score(0.7),
            by_category_audited={QuestionCategory.TEMPORAL: score(0.6)},
        ),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.score_deltas[0].baseline_score is None
    assert report.score_deltas[0].candidate_score is not None
    assert report.score_deltas[0].point_delta is None
    assert report.score_deltas[0].ci_overlaps is None
    assert report.score_deltas[0].statistically_significant is None
    assert all(delta.point_delta is None for delta in report.score_deltas[1:])
    assert all(delta.ci_overlaps is None for delta in report.score_deltas[1:])
    assert all(delta.statistically_significant is None for delta in report.score_deltas[1:])


def test_question_verdict_and_score_differences_are_reported_for_shared_questions() -> None:
    repository = fake_repository(
        baseline_questions=[
            question_record(BASELINE_RUN_ID, "q-score", verdict=JudgmentVerdict.PARTIAL, value=0.5),
            question_record(
                BASELINE_RUN_ID,
                "q-same",
                verdict=JudgmentVerdict.INCORRECT,
                value=0.0,
            ),
            question_record(BASELINE_RUN_ID, "q-verdict", verdict=JudgmentVerdict.CORRECT),
            question_record(BASELINE_RUN_ID, "q-baseline-only"),
        ],
        candidate_questions=[
            question_record(
                CANDIDATE_RUN_ID,
                "q-score",
                verdict=JudgmentVerdict.PARTIAL,
                value=0.25,
            ),
            question_record(
                CANDIDATE_RUN_ID,
                "q-same",
                verdict=JudgmentVerdict.INCORRECT,
                value=0.0,
            ),
            question_record(
                CANDIDATE_RUN_ID,
                "q-verdict",
                verdict=JudgmentVerdict.INCORRECT,
                value=0.0,
            ),
            question_record(CANDIDATE_RUN_ID, "q-candidate-only"),
        ],
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert [difference.question_id for difference in report.differing_questions] == [
        "q-score",
        "q-verdict",
    ]
    score_difference = report.differing_questions[0]
    assert score_difference.verdicts_by_run_id == {
        BASELINE_RUN_ID: JudgmentVerdict.PARTIAL,
        CANDIDATE_RUN_ID: JudgmentVerdict.PARTIAL,
    }
    assert score_difference.scores_by_run_id == {
        BASELINE_RUN_ID: 0.5,
        CANDIDATE_RUN_ID: 0.25,
    }
    verdict_difference = report.differing_questions[1]
    assert verdict_difference.verdicts_by_run_id == {
        BASELINE_RUN_ID: JudgmentVerdict.CORRECT,
        CANDIDATE_RUN_ID: JudgmentVerdict.INCORRECT,
    }


def test_methodology_and_profile_mismatches_produce_compatibility_warnings() -> None:
    repository = fake_repository(
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            methodology_version="methodology-v2",
            methodology_profile="audit-strict",
        ),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    assert len(report.compatibility_warnings) == 2
    assert "Methodology version differs" in report.compatibility_warnings[0]
    assert "Methodology profile differs" in report.compatibility_warnings[1]


def test_benchmark_metadata_mismatches_produce_compatibility_warnings() -> None:
    repository = fake_repository(
        candidate_started=run_started(
            CANDIDATE_RUN_ID,
            benchmark_type="other-benchmark",
            benchmark_version="2026-05",
            benchmark_checksum="sha256:different",
        ),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    assert len(report.compatibility_warnings) == 3
    assert "Benchmark type differs" in report.compatibility_warnings[0]
    assert "Benchmark version differs" in report.compatibility_warnings[1]
    assert "Benchmark checksum differs" in report.compatibility_warnings[2]


def test_question_set_mismatch_warns_but_differences_use_intersection() -> None:
    repository = fake_repository(
        baseline_questions=[
            question_record(BASELINE_RUN_ID, "q-shared", verdict=JudgmentVerdict.CORRECT),
            question_record(BASELINE_RUN_ID, "q-baseline-only"),
        ],
        candidate_questions=[
            question_record(
                CANDIDATE_RUN_ID,
                "q-shared",
                verdict=JudgmentVerdict.INCORRECT,
                value=0.0,
            ),
            question_record(CANDIDATE_RUN_ID, "q-candidate-only"),
        ],
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    assert len(report.compatibility_warnings) == 1
    assert "Question ID sets differ" in report.compatibility_warnings[0]
    assert "q-baseline-only" in report.compatibility_warnings[0]
    assert "q-candidate-only" in report.compatibility_warnings[0]
    assert [difference.question_id for difference in report.differing_questions] == ["q-shared"]


def test_provider_model_and_judge_differences_do_not_warn_by_themselves() -> None:
    repository = fake_repository(
        baseline_status=run_status(
            BASELINE_RUN_ID,
            provider_type="full_context",
            provider_version="1.0",
            answer_model_id="baseline-answer",
            judge_model_id="baseline-judge",
        ),
        candidate_status=run_status(
            CANDIDATE_RUN_ID,
            provider_type="full_context",
            provider_version="2.0",
            answer_model_id="candidate-answer",
            judge_model_id="candidate-judge",
        ),
        baseline_started=run_started(
            BASELINE_RUN_ID,
            provider_type="full_context",
            provider_version="1.0",
            answer_model_id="baseline-answer",
            answer_model_vendor="openai",
            judge_model_id="baseline-judge",
            judge_model_vendor="openai",
        ),
        candidate_started=run_started(
            CANDIDATE_RUN_ID,
            provider_type="full_context",
            provider_version="2.0",
            answer_model_id="candidate-answer",
            answer_model_vendor="anthropic",
            judge_model_id="candidate-judge",
            judge_model_vendor="google",
        ),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is True
    assert report.compatibility_warnings == []


@pytest.mark.parametrize(
    "run_ids",
    [
        [],
        [BASELINE_RUN_ID],
        [BASELINE_RUN_ID, CANDIDATE_RUN_ID, "third-run"],
    ],
)
def test_non_two_run_inputs_raise_configuration_error(run_ids: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        RunComparator().compare(run_ids, fake_repository())


def test_invalid_mode_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        RunComparator().compare(
            [BASELINE_RUN_ID, CANDIDATE_RUN_ID],
            fake_repository(),
            mode="both",
        )


class _FakeRepository:
    def __init__(
        self,
        *,
        statuses: dict[str, RunStatus],
        started_events: dict[str, RunStartedEvent],
        questions: dict[str, Sequence[QuestionRecord]],
    ) -> None:
        self._statuses = statuses
        self._started_events = started_events
        self._questions = questions

    def get_run_status(self, run_id: str) -> RunStatus:
        return self._statuses[run_id]

    def get_run_started_event(self, run_id: str) -> RunStartedEvent:
        return self._started_events[run_id]

    def get_questions_for_run(self, run_id: str) -> list[QuestionRecord]:
        return list(self._questions[run_id])


def fake_repository(
    *,
    baseline_status: RunStatus | None = None,
    candidate_status: RunStatus | None = None,
    baseline_started: RunStartedEvent | None = None,
    candidate_started: RunStartedEvent | None = None,
    baseline_questions: Sequence[QuestionRecord] | None = None,
    candidate_questions: Sequence[QuestionRecord] | None = None,
) -> _FakeRepository:
    return _FakeRepository(
        statuses={
            BASELINE_RUN_ID: baseline_status or run_status(BASELINE_RUN_ID),
            CANDIDATE_RUN_ID: candidate_status or run_status(CANDIDATE_RUN_ID),
        },
        started_events={
            BASELINE_RUN_ID: baseline_started or run_started(BASELINE_RUN_ID),
            CANDIDATE_RUN_ID: candidate_started or run_started(CANDIDATE_RUN_ID),
        },
        questions={
            BASELINE_RUN_ID: list(
                baseline_questions
                or [
                    question_record(BASELINE_RUN_ID, "q-1"),
                    question_record(BASELINE_RUN_ID, "q-2", category=QuestionCategory.TEMPORAL),
                ]
            ),
            CANDIDATE_RUN_ID: list(
                candidate_questions
                or [
                    question_record(CANDIDATE_RUN_ID, "q-1"),
                    question_record(CANDIDATE_RUN_ID, "q-2", category=QuestionCategory.TEMPORAL),
                ]
            ),
        },
    )


def score(
    point_estimate: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> ScoreWithCI:
    n_correct = round(point_estimate * 100)
    return ScoreWithCI(
        n_total=100,
        n_correct=n_correct,
        n_errors=100 - n_correct,
        point_estimate=point_estimate,
        ci_95_low=max(0.0, point_estimate - 0.1) if low is None else low,
        ci_95_high=min(1.0, point_estimate + 0.1) if high is None else high,
    )


def _category_map(scores: dict[QuestionCategory, ScoreWithCI]) -> dict[str, ScoreWithCI]:
    return {category.value: score_value for category, score_value in scores.items()}


def run_status(
    run_id: str,
    *,
    overall_standard: ScoreWithCI | None | object = _MISSING,
    overall_audited: ScoreWithCI | None | object = _MISSING,
    by_category_standard: dict[QuestionCategory, ScoreWithCI] | None = None,
    by_category_audited: dict[QuestionCategory, ScoreWithCI] | None = None,
    methodology_version: str = "methodology-v1",
    methodology_profile: str = "published",
    provider_type: str = "full_context",
    provider_version: str = "1.0",
    answer_model_id: str = "answer-model",
    judge_model_id: str = "judge-model",
) -> RunStatus:
    return RunStatus(
        run_id=run_id,
        suite_id="suite-1",
        experiment_id=f"experiment-{run_id}",
        experiment_name=f"Experiment {run_id}",
        run_number=0,
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        provider_type=provider_type,
        provider_version=provider_version,
        answer_model_id=answer_model_id,
        judge_model_id=judge_model_id,
        methodology_version=methodology_version,
        methodology_profile=methodology_profile,
        framework_version="0.0.0",
        n_conversations_processed=1,
        n_questions_attempted=2,
        n_questions_succeeded=1,
        n_questions_errored=0,
        overall_score_standard=_score_or_default(overall_standard, score(0.5)),
        overall_score_audited=_score_or_default(overall_audited, score(0.6)),
        by_category_standard=_category_map(
            by_category_standard
            if by_category_standard is not None
            else {QuestionCategory.SINGLE_HOP: score(0.5)}
        ),
        by_category_audited=_category_map(
            by_category_audited
            if by_category_audited is not None
            else {QuestionCategory.SINGLE_HOP: score(0.6)}
        ),
        total_cost_usd=0.0,
        error_message=None,
        error_phase=None,
    )


def _score_or_default(
    value: ScoreWithCI | None | object,
    default: ScoreWithCI,
) -> ScoreWithCI | None:
    if value is _MISSING:
        return default
    return cast(ScoreWithCI | None, value)


def run_started(
    run_id: str,
    *,
    provider_type: str = "full_context",
    provider_version: str = "1.0",
    benchmark_type: str = "locomo",
    benchmark_version: str = "2026-05-06",
    benchmark_checksum: str = "sha256:locomo",
    answer_model_id: str = "answer-model",
    answer_model_vendor: str = "openai",
    judge_model_id: str = "judge-model",
    judge_model_vendor: str = "openai",
    framework_version: str = "0.1.0+aaaaaaa",
    runtime_environment: dict[str, object] | None = None,
) -> RunStartedEvent:
    return RunStartedEvent(
        event_id=f"event-{run_id}-started",
        timestamp=NOW,
        suite_id="suite-1",
        run_id=run_id,
        sequence_number=0,
        experiment_id=f"experiment-{run_id}",
        experiment_name=f"Experiment {run_id}",
        run_number=0,
        provider_type=provider_type,
        provider_version=provider_version,
        benchmark_type=benchmark_type,
        benchmark_version=benchmark_version,
        benchmark_checksum=benchmark_checksum,
        answer_model_id=answer_model_id,
        answer_model_vendor=answer_model_vendor,
        judge_model_id=judge_model_id,
        judge_model_vendor=judge_model_vendor,
        config={},
        methodology_version="methodology-v1",
        methodology_profile="published",
        framework_version=framework_version,
        seed=123,
        runtime_environment=dict(runtime_environment or {}),
    )


def question_record(
    run_id: str,
    question_id: str,
    *,
    category: QuestionCategory = QuestionCategory.SINGLE_HOP,
    verdict: JudgmentVerdict = JudgmentVerdict.CORRECT,
    value: float | None = None,
) -> QuestionRecord:
    record_score = (
        value if value is not None else 1.0 if verdict is JudgmentVerdict.CORRECT else 0.0
    )
    return QuestionRecord(
        question_evaluation_id=f"qe-{run_id}-{question_id}",
        run_id=run_id,
        question_id=question_id,
        conversation_id="conv-1",
        category=category,
        question_text=f"Question {question_id}?",
        expected_answer="Expected",
        is_audited_error=False,
        retrieval_id=f"ret-{run_id}-{question_id}",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=[f"mem-{question_id}"],
        response_id=f"resp-{run_id}-{question_id}",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="Expected" if verdict is JudgmentVerdict.CORRECT else "Different",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id=f"judge-{run_id}-{question_id}",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=verdict,
        score=record_score,
        judgment_reasoning="Synthetic judgment.",
        judgment_input_tokens=50,
        judgment_output_tokens=10,
        judgment_cost_usd=0.02,
        total_latency_ms=392.5,
        total_cost_usd=0.03,
        error_message=None,
        error_phase=None,
    )


def test_a_differing_framework_version_makes_runs_incompatible() -> None:
    # Framework version is the first thing to check when two runs disagree, yet the comparator
    # did not compare it -- so a report could state "direct comparison is valid" for runs
    # produced by different code. Since `compatible` is derived from the warning list, the
    # absence of the check was an affirmative false claim.
    repository = fake_repository(
        candidate_started=run_started(CANDIDATE_RUN_ID, framework_version="0.1.0+bbbbbbb"),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    assert any("Framework version differs" in warning for warning in report.compatibility_warnings)


def test_two_dirty_builds_sharing_a_label_are_still_not_comparable() -> None:
    # The subtle case, and the reason this is a separate rule from the mismatch check: equality
    # of a dirty version string proves nothing. Both runs say `+aaaaaaa.dirty`, but a dirty tree
    # is uncommitted, so the label describes no specific code and the two could differ freely.
    dirty = "0.1.0+aaaaaaa.dirty"
    repository = fake_repository(
        baseline_started=run_started(BASELINE_RUN_ID, framework_version=dirty),
        candidate_started=run_started(CANDIDATE_RUN_ID, framework_version=dirty),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    warnings = report.compatibility_warnings
    if not any("uncommitted working tree" in warning for warning in warnings):
        raise AssertionError(warnings)
    # One per run: both are unverifiable, and naming only one would hide the other.
    assert sum("uncommitted working tree" in warning for warning in warnings) == 2
    # And not as a mismatch, because the strings are equal -- the rules must not conflate.
    assert not any("Framework version differs" in warning for warning in warnings)


def test_an_unidentified_build_is_not_comparable() -> None:
    repository = fake_repository(
        candidate_started=run_started(CANDIDATE_RUN_ID, framework_version="0.1.0+unidentified"),
    )

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is False
    assert any("no identifiable build" in warning for warning in report.compatibility_warnings)


def test_identical_clean_builds_raise_no_build_warning() -> None:
    # Guards against over-triggering: the rules above must not make every comparison
    # incompatible, which would be as useless as never checking.
    repository = fake_repository()

    report = RunComparator().compare([BASELINE_RUN_ID, CANDIDATE_RUN_ID], repository)

    assert report.compatible is True
    assert report.compatibility_warnings == []


def test_comparing_a_partial_corpus_run_warns_that_the_delta_is_not_a_measurement() -> None:
    # The run report says so above its own headline, but a comparison is a separate artifact and
    # the easier one to misread: a delta table with intervals and a significance column, all
    # arithmetically correct over the questions asked and none of it about the benchmark. Two
    # partial runs may legitimately be compared -- that is exactly when an early comparison is
    # most useful -- so this warns rather than refuses.
    from khedron.analysis.comparator import _partial_corpus_warnings

    partial = run_started(
        "run-a",
        runtime_environment={
            "corpus_conversations_evaluated": 1,
            "corpus_conversations_available": 10,
        },
    )
    whole = run_started(
        "run-b",
        runtime_environment={
            "corpus_conversations_evaluated": 10,
            "corpus_conversations_available": 10,
        },
    )
    unrecorded = run_started("run-c")

    warnings = _partial_corpus_warnings(partial, "run-a")
    if len(warnings) != 1 or "1 of 10 conversations" not in warnings[0]:
        raise AssertionError(warnings)
    if "not a measurement" not in warnings[0]:
        raise AssertionError(warnings[0])
    # A complete run and a run predating the field must both stay silent: a warning on every
    # historical comparison trains the reader to skim past the real one.
    if _partial_corpus_warnings(whole, "run-b"):
        raise AssertionError(_partial_corpus_warnings(whole, "run-b"))
    if _partial_corpus_warnings(unrecorded, "run-c"):
        raise AssertionError(_partial_corpus_warnings(unrecorded, "run-c"))


def test_a_candidate_in_a_second_results_root_can_be_compared() -> None:
    # The comparison this project exists to make spans two results roots: a provider suite and its
    # retrieval-free baseline are separate suites with separate output directories, deliberately,
    # so their runs do not share one index. With a single repository that comparison could not be
    # executed at all -- discovered only when attempted for real, rather than caught by any test.
    class _SplitRepository:
        def __init__(self, run_id: str) -> None:
            self._run_id = run_id

        def get_run_status(self, run_id: str) -> object:
            if run_id != self._run_id:
                raise FileNotFoundError(f"{run_id} does not live in this results root")
            return run_status(run_id)

        def get_run_started_event(self, run_id: str) -> object:
            if run_id != self._run_id:
                raise FileNotFoundError(f"{run_id} does not live in this results root")
            return run_started(run_id)

        def get_questions_for_run(self, run_id: str) -> list[object]:
            if run_id != self._run_id:
                raise FileNotFoundError(f"{run_id} does not live in this results root")
            return [question_record(run_id, "q1")]

    baseline = cast(Any, _SplitRepository("run-a"))
    candidate = cast(Any, _SplitRepository("run-b"))

    report = RunComparator().compare(["run-a", "run-b"], baseline, candidate_repository=candidate)

    if report.run_ids != ["run-a", "run-b"]:
        raise AssertionError(report.run_ids)
    # And the single-root form still reads both runs from the one repository it was given.
    with pytest.raises(FileNotFoundError):
        RunComparator().compare(["run-a", "run-b"], baseline)
