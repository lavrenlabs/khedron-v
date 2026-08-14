from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from khedron.analysis import RunComparator
from khedron.cli import compare_paired_command
from khedron.errors import ConfigurationError
from khedron.persistence.repository import RunRepository
from khedron.types import (
    Judgment,
    JudgmentVerdict,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreWithCI,
    question_plan_fingerprint,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path: Path) -> RunRepository:
    return RunRepository(tmp_path / "results", tmp_path / "benchmark.db")


@pytest.mark.asyncio
async def test_standard_comparator_refuses_eligible_runs_with_different_manifests(
    repository: RunRepository,
) -> None:
    await _append_completed_run(repository, "baseline", ("q-baseline",))
    await _append_completed_run(repository, "candidate", ("q-candidate",))

    with pytest.raises(ConfigurationError, match="different planned question ID sets"):
        RunComparator().compare(["baseline", "candidate"], repository, mode="standard")


@pytest.mark.asyncio
async def test_paired_cli_refuses_different_manifests_before_running_mcnemar(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _append_completed_run(repository, "baseline", ("q-baseline",))
    await _append_completed_run(repository, "candidate", ("q-candidate",))
    called = False

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("compare_paired must not run for different manifests")

    monkeypatch.setattr("khedron.cli.compare_paired", must_not_run)

    with pytest.raises(typer.Exit):
        compare_paired_command(
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            results_dir=repository._results_dir,
        )

    assert called is False
    assert "different planned question ID sets" in capsys.readouterr().err


async def _append_completed_run(
    repository: RunRepository, run_id: str, question_ids: tuple[str, ...]
) -> None:
    await repository.append_run_event(
        RunStartedEvent(
            event_id=f"{run_id}-started",
            timestamp=NOW,
            run_id=run_id,
            sequence_number=0,
            suite_id="suite",
            experiment_id="experiment",
            experiment_name="manifest guard",
            run_number=0,
            provider_type="offline",
            provider_version="1",
            benchmark_type="synthetic",
            benchmark_version="1",
            benchmark_checksum="sha256:synthetic",
            answer_model_id="offline-answer",
            answer_model_vendor="offline",
            judge_model_id="offline-judge",
            judge_model_vendor="offline",
            config={},
            methodology_version="v1",
            methodology_profile="canonical-v2",
            framework_version="0.0.0+test",
            seed=1,
            runtime_environment={},
        )
    )
    await repository.append_question_plan(
        QuestionPlanRecord(
            run_id=run_id,
            benchmark_id="synthetic",
            categories=("single_hop",),
            corpus_checksum="sha256:synthetic",
            question_ids=question_ids,
            fingerprint=question_plan_fingerprint(
                benchmark_id="synthetic",
                categories=("single_hop",),
                corpus_checksum="sha256:synthetic",
                question_ids=question_ids,
            ),
            timestamp=NOW,
        )
    )
    for question_id in question_ids:
        evaluation_id = f"{run_id}-{question_id}-evaluation"
        retrieval_id = f"{run_id}-{question_id}-retrieval"
        response_id = f"{run_id}-{question_id}-response"
        await repository.append_question_evaluation(
            QuestionEvaluationRecord(
                question_evaluation_id=evaluation_id,
                run_id=run_id,
                question_id=question_id,
                conversation_id="conversation",
                category=QuestionCategory.SINGLE_HOP,
                question_text="Question?",
                expected_answer="Answer",
                is_audited_error=False,
                timestamp=NOW,
            )
        )
        await repository.append_retrieval(
            RetrievalRecord(
                retrieval_id=retrieval_id,
                question_evaluation_id=evaluation_id,
                run_id=run_id,
                question_id=question_id,
                timestamp=NOW,
                query="Question?",
                top_k=0,
                n_returned=0,
                memories=[],
                retrieval_latency_ms=0.0,
            )
        )
        await repository.append_response(
            Response(
                response_id=response_id,
                run_id=run_id,
                question_id=question_id,
                retrieval_id=retrieval_id,
                timestamp=NOW,
                model_id="offline-answer",
                prompt="Question?",
                answer_text="Answer",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                cost_usd=0.0,
            )
        )
        await repository.append_judgment(
            Judgment(
                judgment_id=f"{run_id}-{question_id}-judgment",
                run_id=run_id,
                response_id=response_id,
                question_id=question_id,
                timestamp=NOW,
                judge_model_id="offline-judge",
                prompt="Question?",
                raw_judge_output="correct",
                parsed_verdict=JudgmentVerdict.CORRECT,
                parsed_score=1.0,
                parsed_reasoning="Matches.",
                parse_was_successful=True,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                cost_usd=0.0,
            )
        )
    score = ScoreWithCI(
        n_total=len(question_ids),
        n_correct=len(question_ids),
        n_errors=0,
        point_estimate=1.0,
        ci_95_low=0.2,
        ci_95_high=1.0,
    )
    await repository.append_run_event(
        RunCompletedEvent(
            event_id=f"{run_id}-completed",
            timestamp=NOW,
            run_id=run_id,
            sequence_number=1,
            status="completed",
            n_questions_attempted=len(question_ids),
            n_questions_succeeded=len(question_ids),
            n_questions_errored=0,
            overall_score_standard=score,
            overall_score_audited=score,
            by_category_standard={"single_hop": score},
            by_category_audited={"single_hop": score},
            total_cost_usd=0.0,
        )
    )
