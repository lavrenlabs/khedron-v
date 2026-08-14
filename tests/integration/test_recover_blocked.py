"""Operator-forced recovery: selection, guards, engine reuse, disclosure, TOCTOU.

Offline by default: every provider/model/judge is a fake, fixtures are deterministic JSONL run
directories read back through the repository and StrictRunReader.
"""

from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer

from khedron.benchmarks.registry import BENCHMARK_REGISTRY
from khedron.cli import _require_publishable, inspect_run
from khedron.config import ExperimentSuiteConfig
from khedron.eligibility import assess_run_eligibility
from khedron.errors import ConfigurationError
from khedron.judges.registry import JUDGE_REGISTRY
from khedron.models.registry import MODEL_REGISTRY
from khedron.persistence.repository import RunRepository
from khedron.persistence.stage_reader import StrictRunReader
from khedron.providers.registry import PROVIDER_REGISTRY
from khedron.resume import load_forced_recovery_state, recover_blocked_run
from khedron.runner import Runner
from khedron.types import ErrorRecord, JudgmentVerdict, RecoveryAttemptRecord
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc
from tests.integration.test_runner import (
    FakeAnswerModel,
    FakeBenchmark,
    FakeJudge,
    FakeProvider,
    _isolated_registries,
    _question_spec,
    _read_jsonl,
    _suite_config,
)

REASON = "auth key reissued out of band"


@pytest.fixture(autouse=True)
def _skip_methodology_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("khedron.runner.validate_suite_methodology_profile", lambda config: None)


@pytest.fixture(autouse=True)
def _clean_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KHEDRON_MAX_BUDGET_USD", raising=False)


@pytest.fixture(autouse=True)
def _identifiable_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both the runner and resume resolvers must read a clean, identical build so _require_resumable
    # passes; individual tests re-patch these to exercise the build-identity walls.
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")


@pytest.fixture(autouse=True)
def _fake_registries() -> Iterator[None]:
    with _isolated_registries():
        FakeAnswerModel.reset_observations()
        FakeJudge.reset_observations()
        FakeProvider.reset_observations()
        PROVIDER_REGISTRY["runner_fake"] = FakeProvider
        BENCHMARK_REGISTRY["runner_fake"] = FakeBenchmark
        MODEL_REGISTRY["runner_fake"] = FakeAnswerModel
        JUDGE_REGISTRY["runner_fake"] = FakeJudge
        yield


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def repository(results_dir: Path) -> RunRepository:
    # sqlite under results_dir so the CLI's default (--results-dir/benchmark.db) resolves it.
    return RunRepository(results_dir=results_dir, sqlite_path=results_dir / "benchmark.db")


def _suite(repository: RunRepository, questions: list[dict[str, Any]], **kwargs: Any) -> Any:
    # output_dir pinned to the repository's results root so the recovery lock lands beside the run
    # data (the recovery engine locks at config.output_dir), and so the config digest is stable
    # across the original run and the recovery.
    suite = _suite_config(questions, **kwargs)
    return suite.model_copy(update={"output_dir": str(repository._results_dir)})


async def _run_fresh(repository: RunRepository, suite: ExperimentSuiteConfig) -> str:
    await Runner(suite, repository).run()
    return repository.list_runs()[0].run_id


def _run_dir(repository: RunRepository, run_id: str) -> Path:
    return repository._results_dir / "runs" / run_id


def _blocked_at(
    repository: RunRepository,
    stage: str,
    *,
    fail_once: bool = False,
    question_id: str = "q-blocked",
    **suite_kwargs: Any,
) -> Any:
    """A suite whose one question fails `stage` durably with retryable=False.

    With fail_once the stage succeeds on the recovery attempt; without it the stage always fails,
    which is the shape a failed forced recovery needs.
    """
    payload: dict[str, Any] = {"retryable": False}
    if fail_once:
        payload["fail_once"] = True
    config_key = {
        "retrieve": "provider_config",
        "generate": "answer_model_config",
        "judge": "judge_config",
    }[stage]
    return _suite(
        repository,
        [_question_spec(question_id, fail_stage=stage)],
        **{config_key: payload},
        **suite_kwargs,
    )


# =================================================================================================
# Selection / eligibility (load_forced_recovery_state)
# =================================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict",
    [JudgmentVerdict.CORRECT, JudgmentVerdict.PARTIAL, JudgmentVerdict.UNKNOWN],
)
async def test_1_refuses_a_question_that_has_any_judgment(
    repository: RunRepository, verdict: JudgmentVerdict
) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)
    # A judge-blocked question has a stored response; inject a judgment (any verdict) onto it.
    from khedron.types import Judgment

    response = repository.get_responses_for_run(run_id)[0]
    await repository.append_judgment(
        Judgment(
            judgment_id=generate_ulid(),
            run_id=run_id,
            response_id=response.response_id,
            question_id="q-blocked",
            timestamp=now_utc(),
            judge_model_id="fake-judge-model",
            prompt="p",
            raw_judge_output="{}",
            parsed_verdict=verdict,
            parsed_score=0.0,
            parsed_reasoning="r",
            parse_was_successful=True,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost_usd=0.0,
        )
    )

    with pytest.raises(ConfigurationError, match="already has a judgment"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_2_refuses_a_question_not_in_the_plan(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)

    with pytest.raises(ConfigurationError, match="not in this run's planned-question manifest"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["never-planned"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_3_refuses_a_question_with_no_forcible_error(repository: RunRepository) -> None:
    # Only a retryable error: belongs to the automatic path, so recover-blocked refuses it.
    suite = _suite(
        repository,
        [_question_spec("q-retryable", fail_stage="generate")],
        answer_model_config={"retryable": True},
    )
    run_id = await _run_fresh(repository, suite)
    with pytest.raises(ConfigurationError, match="retryable error the automatic resume"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-retryable"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_3_refuses_a_planned_question_with_no_records_at_all(
    repository: RunRepository,
) -> None:
    # A question in the plan that the run never reached: no error to force.
    suite = _suite(repository, [_question_spec("q-a"), _question_spec("q-b")])
    run_id = await _run_fresh(repository, suite)
    run_dir = _run_dir(repository, run_id)
    # Drop every stage record for q-b, leaving it planned but record-less (a hard-kill shape).
    for name in (
        "question_evaluations.jsonl",
        "retrievals.jsonl",
        "responses.jsonl",
        "judgments.jsonl",
    ):
        rows = [row for row in _read_jsonl(run_dir / name) if row["question_id"] != "q-b"]
        (run_dir / name).write_text(
            "".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    await repository.rebuild_sqlite()

    with pytest.raises(ConfigurationError, match="no unresolved, non-retryable"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-b"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_5_selects_the_stage_of_the_blocking_error(repository: RunRepository) -> None:
    for stage in ("retrieve", "generate", "judge"):
        with _isolated_registries():
            FakeAnswerModel.reset_observations()
            FakeJudge.reset_observations()
            FakeProvider.reset_observations()
            PROVIDER_REGISTRY["runner_fake"] = FakeProvider
            BENCHMARK_REGISTRY["runner_fake"] = FakeBenchmark
            MODEL_REGISTRY["runner_fake"] = FakeAnswerModel
            JUDGE_REGISTRY["runner_fake"] = FakeJudge
            repo = RunRepository(
                results_dir=repository._results_dir / stage,
                sqlite_path=repository._results_dir / stage / "benchmark.db",
            )
            suite = _blocked_at(repo, stage, question_id="q-blocked")
            run_id = await _run_fresh(repo, suite)
            state = load_forced_recovery_state(
                run_id=run_id,
                question_ids=["q-blocked"],
                reason=REASON,
                config=suite,
                repository=repo,
            )
            assert len(state.recovery_work) == 1
            item = state.recovery_work[0]
            assert item.stage == stage
            assert item.forced is True
            assert item.reason == REASON
            assert item.forced_recovery_id is not None


@pytest.mark.asyncio
async def test_6_collapses_multiple_stage_errors_to_one_item(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "generate")
    run_id = await _run_fresh(repository, suite)
    # Inject a SECOND, later, non-retryable error for the same question in a different stage.
    later_error = ErrorRecord(
        error_id=generate_ulid(),
        run_id=run_id,
        timestamp=now_utc(),
        phase="judge",
        question_id="q-blocked",
        error_type="ModelBillingError",
        error_message="second blocked stage",
        stack_trace=None,
        context={},
        recovered=False,
        retryable=False,
    )
    await repository.append_error(later_error)

    state = load_forced_recovery_state(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )
    assert len(state.recovery_work) == 1
    # The later (judge) error wins the per-question collapse.
    assert state.recovery_work[0].stage == "judge"
    assert state.recovery_work[0].error_id == later_error.error_id


@pytest.mark.asyncio
async def test_7_respects_the_max_recovery_attempts_budget(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "generate", max_recovery_attempts=1)
    run_id = await _run_fresh(repository, suite)
    error = repository.get_errors_for_run(run_id)[0]
    # One attempt already recorded exhausts the max_recovery_attempts=1 budget for (q, generate).
    await repository.append_recovery_attempt(
        RecoveryAttemptRecord(
            attempt_id=generate_ulid(),
            run_id=run_id,
            question_id="q-blocked",
            error_id=error.error_id,
            stage="generate",
            timestamp=now_utc(),
        )
    )

    with pytest.raises(ConfigurationError, match="over the recovery-attempt budget"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_8_one_invalid_question_refuses_the_whole_command(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)

    with pytest.raises(ConfigurationError):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked", "not-planned"],
            reason=REASON,
            config=suite,
            repository=repository,
        )
    # Nothing selected, nothing written: the loader has no side effects.
    assert not (_run_dir(repository, run_id) / "forced_recoveries.jsonl").exists()


@pytest.mark.asyncio
async def test_1b_refuses_empty_reason(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)
    with pytest.raises(ConfigurationError, match="non-empty operator reason"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason="   ",
            config=suite,
            repository=repository,
        )


# =================================================================================================
# Resume-guard reuse (fail-closed)
# =================================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty", ["0.0.0+abc1234.dirty", "0.0.0+unidentified"])
async def test_9_10_refuses_an_unidentifiable_current_build(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch, dirty: str
) -> None:
    # Record the run under the SAME unidentifiable string so Wall A (same build) passes and Wall B
    # (identifiable current build) is the wall that fires -- the case tests 9/10 are about.
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: dirty)
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: dirty)
    suite = _blocked_at(repository, "judge")
    await Runner(suite, repository, allow_unidentifiable_build=True).run()
    run_id = repository.list_runs()[0].run_id

    with pytest.raises(ConfigurationError, match="unidentifiable"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
async def test_11_refuses_a_framework_version_mismatch(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)
    # A different clean commit than the one recorded: no flag exists to override this.
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+different")

    with pytest.raises(ConfigurationError) as exc_info:
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )
    assert exc_info.value.context.get("field") == "framework version"


@pytest.mark.asyncio
async def test_12_refuses_a_config_digest_mismatch(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)
    edited = _blocked_at(repository, "judge", answer_max_output_tokens=2048)

    with pytest.raises(ConfigurationError) as exc_info:
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=edited,
            repository=repository,
        )
    assert exc_info.value.context.get("field") == "configuration digest"


@pytest.mark.asyncio
async def test_13_refuses_a_legacy_stream_with_no_plan(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge")
    run_id = await _run_fresh(repository, suite)
    (_run_dir(repository, run_id) / "question_plan.jsonl").unlink()

    with pytest.raises(ConfigurationError, match="legacy stream"):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "message"),
    [("0.0.0+5d989e9.dirty", "unidentifiable"), ("0.0.0+clean", "different")],
)
async def test_14_dirty_origin_run_is_inescapable(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch, current: str, message: str
) -> None:
    # A run recorded under a .dirty build is unrecoverable from any build: matching it needs a dirty
    # current tree (refused by Wall B), and a clean tree can never match the dirty string (Wall A).
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+5d989e9.dirty")
    suite = _suite(repository, [_question_spec("q-a")])
    await Runner(suite, repository, allow_unidentifiable_build=True).run()
    run_id = repository.list_runs()[0].run_id

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: current)
    with pytest.raises(ConfigurationError, match=message):
        load_forced_recovery_state(
            run_id=run_id,
            question_ids=["q-a"],
            reason=REASON,
            config=suite,
            repository=repository,
        )


# =================================================================================================
# Driver / engine reuse (integration, offline)
# =================================================================================================


@pytest.mark.asyncio
async def test_23_judge_stage_forced_recovery_end_to_end(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    prior_spend = sum(call.cost_usd for call in repository.get_api_calls_for_run(run_id))
    assert not assess_run_eligibility(repository, run_id).eligible  # blocked to start with

    result = await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    run_dir = _run_dir(repository, run_id)
    stream = StrictRunReader().read(run_dir, run_id)
    assert len(stream.forced_recoveries) == 1
    forced = stream.forced_recoveries[0]
    attempt = next(a for a in stream.attempts if a.attempt_id == forced.attempt_id)
    # Stamp order: the forced authorization is stamped no later than its attempt.
    assert forced.timestamp <= attempt.timestamp
    assert forced.reason == REASON
    assert len(stream.judgments) == 1
    assert len(stream.resolutions) == 1

    events = repository.list_run_events(run_id)
    event_types = [e.event_type for e in events]
    # A RunResumedEvent (not a second RunStartedEvent) precedes the fresh terminal RunCompleted.
    assert "run_resumed" in event_types
    assert events[-1].event_type == "run_completed"
    assert events[-1].status == "completed"
    resumed = next(e for e in events if e.event_type == "run_resumed")
    assert event_types.index("run_resumed") > event_types.index("run_started")
    assert resumed.inherited_cost_usd == pytest.approx(prior_spend)

    eligibility = assess_run_eligibility(repository, run_id)
    assert eligibility.eligible
    assert eligibility.forced_recovery_count == 1
    assert not (run_dir / "resume.lock").exists()
    assert result.recovered_question_ids == ("q-blocked",)
    assert result.forced_recovery_count == 1


@pytest.mark.asyncio
async def test_24_generate_stage_forced_recovery(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "generate", fail_once=True)
    run_id = await _run_fresh(repository, suite)

    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    stream = StrictRunReader().read(_run_dir(repository, run_id), run_id)
    assert len(stream.responses) == 1
    assert len(stream.judgments) == 1
    # No retrieve-stage re-ingestion: the generate recovery reuses the stored retrieval.
    assert len(stream.retrievals) == 1
    assert stream.retrievals[0].ingestion_attempt_id is None
    assert assess_run_eligibility(repository, run_id).eligible


@pytest.mark.asyncio
async def test_25_retrieve_stage_forced_recovery(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "retrieve", fail_once=True)
    run_id = await _run_fresh(repository, suite)

    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    run_dir = _run_dir(repository, run_id)
    stream = StrictRunReader().read(run_dir, run_id)
    assert len(stream.retrievals) == 1
    # A provider reset + re-ingestion occurred: the recovered retrieval carries its ingestion id,
    # and a second conversation ingestion record exists.
    assert stream.retrievals[0].ingestion_attempt_id is not None
    assert len(_read_jsonl(run_dir / "conversations.jsonl")) == 2
    assert len(stream.responses) == 1
    assert len(stream.judgments) == 1
    assert assess_run_eligibility(repository, run_id).eligible


@pytest.mark.asyncio
async def test_26_failed_forced_recovery_stays_ineligible(repository: RunRepository) -> None:
    # Judge always fails (no fail_once), so the forced recovery itself fails.
    suite = _blocked_at(repository, "judge", max_recovery_attempts=1)
    run_id = await _run_fresh(repository, suite)

    result = await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    run_dir = _run_dir(repository, run_id)
    stream = StrictRunReader().read(run_dir, run_id)
    # The override is recorded even though the recovery failed, and a fresh durable error exists.
    assert len(stream.forced_recoveries) == 1
    assert len(stream.errors) == 2
    assert len(stream.resolutions) == 0
    events = repository.list_run_events(run_id)
    assert events[-1].event_type == "run_completed"
    assert events[-1].status == "partial"
    assert not assess_run_eligibility(repository, run_id).eligible
    assert result.unrecovered_question_ids == ("q-blocked",)

    # A second invocation is bounded by max_recovery_attempts (=1, now exhausted).
    with pytest.raises(ConfigurationError, match="over the recovery-attempt budget"):
        await recover_blocked_run(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 1


@pytest.mark.asyncio
async def test_27_interruption_reopen_and_repair_keeps_monotone_chain(
    repository: RunRepository, results_dir: Path
) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)

    # Reopen the run directory from disk through a fresh repository (a cold, cross-process restart).
    reopened = RunRepository(results_dir=results_dir, sqlite_path=results_dir / "benchmark.db")
    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=reopened,
    )

    stream = StrictRunReader().read(_run_dir(repository, run_id), run_id)  # validates on reopen
    forced = stream.forced_recoveries[0]
    error = next(e for e in stream.errors if e.error_id == forced.error_id)
    attempt = next(a for a in stream.attempts if a.attempt_id == forced.attempt_id)
    resolution = next(r for r in stream.resolutions if r.error_id == forced.error_id)
    target = next(
        j for j in stream.judgments if j.judgment_id == resolution.resolved_by_stage_record_id
    )
    # error <= forced <= attempt <= target <= resolution
    assert (
        error.timestamp
        <= forced.timestamp
        <= attempt.timestamp
        <= target.timestamp
        <= resolution.timestamp
    )
    assert assess_run_eligibility(repository, run_id).eligible


@pytest.mark.asyncio
async def test_28_lock_contention_refuses(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    run_dir = _run_dir(repository, run_id)
    (run_dir / "resume.lock").write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="resume lock"):
        await recover_blocked_run(
            run_id=run_id,
            question_ids=["q-blocked"],
            reason=REASON,
            config=suite,
            repository=repository,
        )
    # The stage records are untouched: no authorization, no attempt.
    assert not (run_dir / "forced_recoveries.jsonl").exists()
    assert not (run_dir / "recovery_attempts.jsonl").exists()


# =================================================================================================
# Disclosure
# =================================================================================================


@pytest.mark.asyncio
async def test_29_inspect_run_discloses_forced_recoveries(
    repository: RunRepository, results_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    inspect_run(run_id=run_id, results_dir=results_dir)
    out = capsys.readouterr().out
    assert "Forced recoveries: 1" in out
    assert REASON in out


@pytest.mark.asyncio
async def test_29_inspect_run_omits_line_when_no_forced_recovery(
    repository: RunRepository, results_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(repository, [_question_spec("q-ok")])
    run_id = await _run_fresh(repository, suite)

    inspect_run(run_id=run_id, results_dir=results_dir)
    assert "Forced recoveries" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_30_publish_gate_refuses_without_the_flag_and_discloses(
    repository: RunRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    # force=True isolates the R5 forced-recovery gate from any other publishability concern.
    with pytest.raises(typer.Exit):
        _require_publishable(repository, run_id, force=True, acknowledge_forced_recoveries=False)
    err = capsys.readouterr().err
    assert "DISCLOSURE" in err
    assert REASON in err


@pytest.mark.asyncio
async def test_30_publish_gate_passes_with_the_flag_but_still_discloses(
    repository: RunRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    # Passes the forced-recovery gate; force=True clears any other disposition block.
    _require_publishable(repository, run_id, force=True, acknowledge_forced_recoveries=True)
    err = capsys.readouterr().err
    assert "DISCLOSURE" in err  # disclosure is unconditional, independent of the flag


# =================================================================================================
# Timestamp ordering & under-lock re-validation
# =================================================================================================


@pytest.mark.asyncio
async def test_31_forced_before_attempt_timestamp(repository: RunRepository) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    await recover_blocked_run(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )

    stream = StrictRunReader().read(_run_dir(repository, run_id), run_id)
    forced = stream.forced_recoveries[0]
    error = next(e for e in stream.errors if e.error_id == forced.error_id)
    attempt = next(a for a in stream.attempts if a.attempt_id == forced.attempt_id)
    assert error.timestamp <= forced.timestamp <= attempt.timestamp


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["0.0.0+abc1234.dirty", "0.0.0+commitB"])
async def test_32_build_change_between_load_and_lock_is_refused(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    # Original run under commitA (== testbuild). load succeeds; then the build changes before the
    # under-lock re-validation, which re-runs the FULL _require_resumable (Wall A + Wall B).
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    forced_state = load_forced_recovery_state(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )
    # The build changes after load, before dispatch. Wall B (dirty) or Wall A (commitA->commitB).
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: changed)
    runner = Runner(suite, repository)

    with pytest.raises(ConfigurationError):
        await runner.recover_blocked(forced_state)

    run_dir = _run_dir(repository, run_id)
    stream = StrictRunReader().read(run_dir, run_id)
    assert len(stream.forced_recoveries) == 0
    assert len(stream.attempts) == 0
    assert not any(
        e.event_type == "run_completed" and e.status == "completed"
        for e in repository.list_run_events(run_id)[1:]
    )
    assert not (run_dir / "resume.lock").exists()


@pytest.mark.asyncio
async def test_33_concurrent_resolution_between_load_and_lock_aborts(
    repository: RunRepository,
) -> None:
    suite = _blocked_at(repository, "judge", fail_once=True)
    run_id = await _run_fresh(repository, suite)
    forced_state = load_forced_recovery_state(
        run_id=run_id,
        question_ids=["q-blocked"],
        reason=REASON,
        config=suite,
        repository=repository,
    )
    # A competing path resolves the question between load and lock: inject a judgment so it is now
    # answered. The under-lock re-validation must find it no longer qualifies and abort.
    from khedron.types import Judgment

    response = repository.get_responses_for_run(run_id)[0]
    await repository.append_judgment(
        Judgment(
            judgment_id=generate_ulid(),
            run_id=run_id,
            response_id=response.response_id,
            question_id="q-blocked",
            timestamp=now_utc(),
            judge_model_id="fake-judge-model",
            prompt="p",
            raw_judge_output="{}",
            parsed_verdict=JudgmentVerdict.CORRECT,
            parsed_score=1.0,
            parsed_reasoning="r",
            parse_was_successful=True,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost_usd=0.0,
        )
    )
    runner = Runner(suite, repository)

    with pytest.raises(ConfigurationError):
        await runner.recover_blocked(forced_state)

    run_dir = _run_dir(repository, run_id)
    assert not (run_dir / "forced_recoveries.jsonl").exists()
    # No duplicate attempt was written for the now-resolved question.
    assert not (run_dir / "recovery_attempts.jsonl").exists()
    assert not (run_dir / "resume.lock").exists()
