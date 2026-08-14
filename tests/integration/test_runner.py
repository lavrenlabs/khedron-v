from __future__ import annotations

# ruff: noqa: S101
import asyncio
import collections
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from khedron.analysis import Scorer
from khedron.benchmarks.base import Benchmark
from khedron.benchmarks.registry import BENCHMARK_REGISTRY
from khedron.config import (
    AnswerModelConfig,
    ExperimentSuiteConfig,
    JudgeConfig,
    load_suite_config,
)
from khedron.errors import (
    ConfigurationError,
    CostExceededError,
    JudgeError,
    ModelError,
    PersistenceError,
    ProviderError,
)
from khedron.judges.base import Judge, JudgeResult
from khedron.judges.registry import JUDGE_REGISTRY
from khedron.models.base import AnswerModel
from khedron.models.registry import MODEL_REGISTRY
from khedron.persistence.repository import RunRepository
from khedron.providers.base import MemoryProvider, ProviderHealthStatus
from khedron.providers.registry import PROVIDER_REGISTRY
from khedron.runner import Runner
from khedron.types import (
    APICallResult,
    Conversation,
    Judgment,
    JudgmentVerdict,
    Memory,
    Question,
    QuestionCategory,
    RecoveryAttemptRecord,
    Response,
    Session,
    Turn,
)
from khedron.utils.ids import generate_ulid
from khedron.utils.stats import wilson_score_interval
from khedron.utils.time import now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def skip_methodology_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let runner tests use fake plugins without satisfying a real methodology contract.

    These tests exercise the runner: their provider, models and judge are `runner_fake`, which no
    contractual profile can pin, and several deliberately vary temperature or category to prove the
    runner passes them through. Methodology conformance is not what they are for and has its own
    tests in tests/unit/methodology, so the check is disabled here rather than weakened in
    production or worked around by loosening a real profile.
    """
    monkeypatch.setattr("khedron.runner.validate_suite_methodology_profile", lambda config: None)


@pytest.fixture(autouse=True)
def clean_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KHEDRON_MAX_BUDGET_USD", raising=False)


@pytest.fixture(autouse=True)
def _identifiable_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive runner tests as if from a clean, committed tree.

    run() now refuses an unidentifiable build at its execution boundary, and this suite runs from
    the live checkout (routinely dirty during development). Pinning the runner's resolver to a clean
    version lets the fresh-run tests exercise what they are for; the resume and build-identity tests
    re-patch the runner (and resume) resolver to their own value, overriding this default.
    """
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")


@pytest.fixture(autouse=True)
def fake_registries() -> Iterator[None]:
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
def repository(tmp_path: Path) -> RunRepository:
    return RunRepository(results_dir=tmp_path / "results", sqlite_path=tmp_path / "benchmark.db")


@pytest.mark.asyncio
async def test_runner_happy_path_persists_lifecycle_and_question_records(
    repository: RunRepository,
) -> None:
    result = await Runner(_suite_config([_question_spec("q1")]), repository).run()

    suite_status = repository.get_suite_status(result.suite_id)
    run = repository.list_runs()[0]
    question = repository.get_question_record(run.run_id, "q1")
    results_dir = repository._results_dir

    assert suite_status.status == "completed"
    assert run.status == "completed"
    assert run.n_questions_attempted == 1
    assert run.n_questions_succeeded == 1
    assert question.verdict is JudgmentVerdict.CORRECT
    assert question.generated_answer == "Blue"
    assert question.retrieved_memory_ids == ["memory-1"]

    suite_events = _read_jsonl(results_dir / "suites" / result.suite_id / "lifecycle.jsonl")
    run_events = _read_jsonl(results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    assert [event["event_type"] for event in suite_events] == [
        "suite_started",
        "experiment_started",
        "experiment_completed",
        "suite_completed",
    ]
    assert [event["sequence_number"] for event in suite_events] == [0, 1, 2, 3]
    assert [event["event_type"] for event in run_events] == [
        "run_started",
        "conversation_processed",
        "run_completed",
    ]
    assert [event["sequence_number"] for event in run_events] == [0, 1, 2]

    run_dir = results_dir / "runs" / run.run_id
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "retrievals.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "responses.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "judgments.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "api_calls.jsonl")) == 2
    assert len(_read_jsonl(run_dir / "conversations.jsonl")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty_version", ["0.0.0+abc1234.dirty", "0.0.0+unidentified"])
async def test_run_refuses_an_unidentifiable_build_at_the_execution_boundary(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch, dirty_version: str
) -> None:
    # The programmatic Runner(...).run() path -- not only the CLI preflight -- must refuse an
    # unidentifiable build, writing nothing and spending nothing. This closes the seam a script or a
    # future entry point would otherwise slip through: an unidentifiable build once burned real
    # money and hours on a 0.0.0+....dirty build that could never be published or resumed.
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: dirty_version)

    with pytest.raises(ConfigurationError) as exc_info:
        await Runner(_suite_config([_question_spec("q1")]), repository).run()

    assert "unidentifiable" in str(exc_info.value).lower()
    # Refused before the first durable write (SuiteStartedEvent): nothing exists -- not the run or
    # suite directories, and not even the SQLite index (created lazily on the first write).
    assert not (repository._results_dir / "runs").exists()
    assert not (repository._results_dir / "suites").exists()


@pytest.mark.asyncio
async def test_allow_unidentifiable_build_authorizes_a_fresh_dirty_run(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The override is a deliberate decision to spend on a knowingly un-publishable build. A fresh
    # run then proceeds and stamps the dirty version honestly -- and asserting the PERSISTED
    # framework_version equals the guarded value proves the guard checks exactly what gets stamped
    # (no check/stamp divergence).
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+abc1234.dirty")

    result = await Runner(
        _suite_config([_question_spec("q1")]), repository, allow_unidentifiable_build=True
    ).run()

    run = repository.list_runs()[0]
    run_events = _read_jsonl(repository._results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    started = next(event for event in run_events if event["event_type"] == "run_started")
    assert started["framework_version"] == "0.0.0+abc1234.dirty"
    assert result.suite_id


@pytest.mark.asyncio
async def test_override_does_not_authorize_resuming_onto_a_dirty_build(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    # allow_unidentifiable_build relaxes ONLY a fresh run's build-identity guard. Resuming across a
    # build change would splice two measurements, so resume's refusal is unconditional and must win
    # over the override -- otherwise the escape hatch for a knowingly un-publishable fresh run would
    # silently reopen the splice the resume guard exists to prevent.
    # Recorded == current == the same dirty build, so Wall A (different-version) passes and this
    # isolates Wall B (dirty build). Create the run itself with the override (a knowingly
    # un-publishable fresh dirty run); resuming it must still refuse, override or not.
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+abc1234.dirty")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+abc1234.dirty")

    suite = _suite_config([_question_spec(f"q{index}") for index in range(4)])
    await Runner(suite, repository, allow_unidentifiable_build=True).run()
    run_id = repository.list_runs()[0].run_id
    _make_interrupted_history(repository._results_dir / "runs" / run_id, {"q0", "q1"})

    # Even with the override set, the resume must refuse in the constructor
    # (load_resume_state -> _require_resumable): the dirty-build refusal is unconditional.
    with pytest.raises(ConfigurationError) as exc_info:
        Runner(suite, repository, resume_run_id=run_id, allow_unidentifiable_build=True)

    assert "unidentifiable" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question_spec", "expected_phase"),
    [
        (
            {
                "id": "q-retrieve",
                "question": "What color is Alice's notebook? fail-retrieve",
                "expected_answer": "Blue",
                "category": "single_hop",
            },
            "retrieve",
        ),
        (
            {
                "id": "q-generate",
                "question": "What color is Alice's notebook? fail-generate",
                "expected_answer": "Blue",
                "category": "single_hop",
            },
            "generate",
        ),
        (
            {
                "id": "q-judge",
                "question": "What color is Alice's notebook? fail-judge",
                "expected_answer": "Blue",
                "category": "single_hop",
            },
            "judge",
        ),
    ],
)
async def test_runner_per_question_stage_failure_materializes_partial_question_record(
    repository: RunRepository,
    question_spec: dict[str, Any],
    expected_phase: str,
) -> None:
    await Runner(_suite_config([_question_spec("q-ok"), question_spec]), repository).run()

    run = repository.list_runs()[0]
    failed_question = repository.get_question_record(run.run_id, str(question_spec["id"]))

    assert run.status == "partial"
    assert run.n_questions_attempted == 2
    assert run.n_questions_succeeded == 1
    assert run.n_questions_errored == 1
    assert failed_question.verdict is JudgmentVerdict.ERROR
    assert failed_question.error_phase == expected_phase


@pytest.mark.asyncio
async def test_runner_multi_run_aggregation_pools_counts_and_wilson_scores(
    repository: RunRepository,
) -> None:
    result = await Runner(
        _suite_config(
            [_question_spec("q1"), _question_spec("q2", answer="Wrong")],
            runs=2,
        ),
        repository,
    ).run()

    aggregate = result.experiment_results[0].aggregate_overall_standard
    assert aggregate is not None
    assert aggregate.pooled_score.n_total == 4
    assert aggregate.pooled_score.n_correct == 2
    assert aggregate.pooled_score.point_estimate == 0.5
    assert aggregate.individual_run_scores == [0.5, 0.5]
    assert aggregate.mean == 0.5
    assert aggregate.stddev == 0.0
    expected_low, expected_high = wilson_score_interval(2, 4)
    assert aggregate.pooled_score.ci_95_low == expected_low
    assert aggregate.pooled_score.ci_95_high == expected_high


@pytest.mark.asyncio
async def test_runner_audited_mode_excludes_audited_error_questions(
    repository: RunRepository,
) -> None:
    await Runner(
        _suite_config(
            [
                _question_spec("q-good", answer="Blue"),
                _question_spec("q-audit", answer="Wrong", audited=True),
            ]
        ),
        repository,
    ).run()

    run = repository.list_runs()[0]
    status = repository.get_run_status(run.run_id)
    expected_scores = Scorer().score_run(repository.get_questions_for_run(run.run_id))

    assert status.overall_score_standard is not None
    assert status.overall_score_audited is not None
    assert status.overall_score_standard == expected_scores.overall_standard
    assert status.overall_score_audited == expected_scores.overall_audited
    assert status.by_category_audited == {
        category.value: score for category, score in expected_scores.by_category_audited.items()
    }
    assert status.overall_score_standard.n_total == 2
    assert status.overall_score_standard.n_correct == 1
    assert status.overall_score_audited.n_total == 1
    assert status.overall_score_audited.n_correct == 1
    assert "single_hop" in status.by_category_audited


@pytest.mark.asyncio
async def test_runner_bounds_question_concurrency(repository: RunRepository) -> None:
    FakeProvider.reset_observations()
    FakeProvider.search_barrier = asyncio.Event()

    await Runner(
        _suite_config(
            [_question_spec(f"q{index}") for index in range(5)],
            provider_config={"search_barrier_concurrency": 2},
            max_concurrent_questions=2,
        ),
        repository,
    ).run()

    # The barrier proves the bound is reached -- the run would have timed out otherwise --
    # and this proves it is never exceeded, which is the semaphore's actual contract.
    assert FakeProvider.max_observed_searches == 2


@pytest.mark.asyncio
async def test_runner_excludes_runs_degraded_by_infrastructure_errors(
    repository: RunRepository,
) -> None:
    # Questions lost to infrastructure errors score as wrong, so a degraded run would
    # publish a number that understates the system under test. With 2 of 5 questions
    # errored (40%) against a 10% ceiling, the run must not reach the aggregate, and the
    # experiment must report "no publishable run" rather than a misleading score.
    result = await Runner(
        _suite_config(
            [
                _question_spec("q1"),
                _question_spec("q2"),
                _question_spec("q3"),
                _question_spec("q4", fail_stage="generate"),
                _question_spec("q5", fail_stage="judge"),
            ],
            max_run_error_rate=0.1,
        ),
        repository,
    ).run()

    suite_events = _read_jsonl(
        repository._results_dir / "suites" / result.suite_id / "lifecycle.jsonl"
    )
    failed_events = [event for event in suite_events if event["event_type"] == "experiment_failed"]
    failed_event = failed_events[0]

    assert result.experiment_results == []
    assert "failed eligibility" in failed_event["error_message"]
    assert "max_run_error_rate" not in failed_event["error_message"]
    assert not [event for event in suite_events if event["event_type"] == "experiment_completed"]
    assert len(failed_events) == 1
    # The run itself still completed and stays on disk as evidence: excluding it from the
    # aggregate must not erase the record a reader needs to recompute the error rate.
    run = repository.list_runs()[0]
    assert run.status == "partial"
    assert run.n_questions_errored == 2
    # Excluding a run from the aggregate must not erase the money it spent.
    assert result.total_cost_usd > 0.0


@pytest.mark.asyncio
async def test_runner_excludes_runs_with_lost_ingestion(repository: RunRepository) -> None:
    # A turn that never reaches the provider produces no ERROR verdict, so the
    # question-level guard cannot see it: a production run once published a score over a
    # conversation that had only been partly ingested. Losing any turn must make the run
    # unpublishable, and the run must still be persisted as evidence.
    result = await Runner(
        _suite_config(
            [_question_spec("q1"), _question_spec("q2")],
            provider_config={"fail_ingest_permanently": True},
            benchmark_config={"n_filler_turns": 39},
        ),
        repository,
    ).run()

    suite_events = _read_jsonl(
        repository._results_dir / "suites" / result.suite_id / "lifecycle.jsonl"
    )
    failed = [event for event in suite_events if event["event_type"] == "experiment_failed"]

    assert result.experiment_results == []
    assert len(failed) == 1
    # The terminal event must name the guard that actually excluded the run. Reporting
    # max_run_error_rate here would send a reader after a threshold this run never
    # breached -- its question error rate is zero.
    assert "max_ingestion_failure_rate" in failed[0]["error_message"]
    assert "max_run_error_rate" not in failed[0]["error_message"]
    # The run itself completed its questions -- the question-level guard would have
    # passed it. Only the ingestion guard catches this.
    run = repository.list_runs()[0]
    assert run.n_questions_errored == 0


def test_runner_warns_when_a_provider_embedder_escapes_the_budget_cap(
    repository: RunRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    # max_cost_usd bounds only what the cost tracker records. A provider embedder bills the
    # same account through a channel Khedron never sees, so the cap reads as a guarantee it
    # cannot give -- an early production measurement once ran that way silently.
    Runner(
        _suite_config(
            [_question_spec("q1")],
            provider_config={"embedder": "openai"},
            max_cost_usd=5.0,
        ),
        repository,
    )

    captured = capsys.readouterr()
    if "unmetered_spend_outside_budget_cap" not in captured.out + captured.err:
        raise AssertionError(captured.out + captured.err)


def test_runner_does_not_warn_without_an_external_embedder(
    repository: RunRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    Runner(_suite_config([_question_spec("q1")], max_cost_usd=5.0), repository)

    captured = capsys.readouterr()
    if "unmetered_spend_outside_budget_cap" in captured.out + captured.err:
        raise AssertionError(captured.out + captured.err)


@pytest.mark.asyncio
async def test_runner_closes_plugins_when_initialization_is_cancelled(
    repository: RunRepository,
) -> None:
    # Initialization runs before the run is announced so the real provider version can be
    # recorded, which means a provider subprocess may already exist. CancelledError derives
    # from BaseException, so an `except Exception` around initialization does not see it:
    # Ctrl-C at this moment used to leak that subprocess.
    FakeProvider.reset_observations()
    runner = Runner(
        _suite_config([_question_spec("q1")], provider_config={"hang_initialize": True}),
        repository,
    )
    FakeProvider.init_reached = asyncio.Event()
    task = asyncio.create_task(runner.run())
    # Wait for the provider to actually be inside initialize(). A fixed sleep silently
    # cancelled before that point, so the test proved nothing.
    await asyncio.wait_for(FakeProvider.init_reached.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (FakeProvider.init_entered, FakeProvider.closed) == (1, 1)
    # The suite already announced this experiment, so the run must be surfaceable as
    # cancelled work rather than existing as a run_id with no lifecycle stream at all.
    run = repository.list_runs()[0]
    run_events = _read_jsonl(repository._results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    assert [event["event_type"] for event in run_events] == ["run_started", "run_failed"]
    assert run_events[1]["error_phase"] == "initialization"
    # The suite owes a terminal event too. Catching only Exception at the suite level left
    # it permanently `running` -- a zombie whose work had actually finished cleanly.
    suite_dir = next((repository._results_dir / "suites").iterdir())
    suite_events = _read_jsonl(suite_dir / "lifecycle.jsonl")
    assert suite_events[-1]["event_type"] == "suite_failed"


@pytest.mark.asyncio
async def test_runner_terminates_a_run_cancelled_after_it_was_announced(
    repository: RunRepository,
) -> None:
    # The initialization case above is not the whole gap: once the run is announced, a
    # cancellation during ingestion, retrieval or analysis also derives from BaseException,
    # so `except Exception` around the run body never appended a terminal event and left the
    # run permanently `running` -- a zombie whose work had in fact stopped.
    FakeProvider.reset_observations()
    runner = Runner(
        _suite_config([_question_spec("q1")], provider_config={"hang_ingest": True}),
        repository,
    )
    FakeProvider.ingest_reached = asyncio.Event()
    task = asyncio.create_task(runner.run())
    # Wait until ingestion is genuinely in progress: cancelling on a fixed sleep could land
    # before the started event and would then re-prove the initialization case instead.
    await asyncio.wait_for(FakeProvider.ingest_reached.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    run = repository.list_runs()[0]
    run_events = _read_jsonl(repository._results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    assert [event["event_type"] for event in run_events] == ["run_started", "run_failed"]
    # The phase the cancellation actually interrupted, not a default: a reader has to be
    # able to tell an ingestion cancellation from an initialization one.
    assert run_events[1]["error_phase"] == "ingest"
    assert "cancelled" in run_events[1]["error_message"].lower()
    assert FakeProvider.closed == 1
    suite_dir = next((repository._results_dir / "suites").iterdir())
    suite_events = _read_jsonl(suite_dir / "lifecycle.jsonl")
    assert suite_events[-1]["event_type"] == "suite_failed"


@pytest.mark.asyncio
async def test_runner_does_not_duplicate_the_started_event_when_its_append_is_cancelled(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    # append_run_event writes the JSONL line and *then* awaits the SQLite index
    # (repository.py:105-111), so a cancellation between the two raises with the started event
    # already durable. Marking the run announced only after that call returned made the
    # terminal handler conclude nothing had been announced and emit a SECOND run_started on
    # the same sequence number -- a corrupt stream, worse than the zombie it prevents.
    original_index = repository._index_run_best_effort
    calls = {"n": 0}

    async def cancel_the_first_index(run_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError
        # Later appends index normally, so a duplicate would hit the UNIQUE constraint on
        # (run_id, sequence_number) and fail loudly rather than pass unnoticed.
        await original_index(run_id)

    monkeypatch.setattr(repository, "_index_run_best_effort", cancel_the_first_index)

    with pytest.raises(asyncio.CancelledError):
        await Runner(_suite_config([_question_spec("q1")]), repository).run()

    run_dir = next((repository._results_dir / "runs").iterdir())
    events = _read_jsonl(run_dir / "lifecycle.jsonl")
    assert [event["event_type"] for event in events] == ["run_started", "run_failed"]
    assert [event["sequence_number"] for event in events] == [0, 1]


@pytest.mark.asyncio
async def test_runner_writes_no_credential_into_any_artifact(
    repository: RunRepository, tmp_path: Path
) -> None:
    # The incident this guards against put a live key into ten files on disk. Unit-testing
    # the redaction proves the function, not that the three persistence paths call it, so
    # this searches every byte the run wrote -- JSONL streams and the SQLite index alike --
    # for the value itself. `some_unanticipated_field` is the point: it carries no
    # credential-looking name, so only the allowlist polarity can catch it.
    secret = "sk-proj-canary-must-never-be-persisted"  # noqa: S105
    await Runner(
        _suite_config(
            [_question_spec("q1")],
            provider_config={
                "openai_api_key": secret,
                "some_unanticipated_field": secret,
                "nested": {"deeper": secret},
            },
        ),
        repository,
    ).run()

    needle = secret.encode()
    # ASYNC240: the run has finished writing, so these reads block nothing.
    leaked = [
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")  # noqa: ASYNC240
        if path.is_file() and needle in path.read_bytes()
    ]
    assert leaked == [], f"credential persisted in: {leaked}"


@pytest.mark.asyncio
async def test_runner_writes_no_credential_into_a_recorded_error_context(
    repository: RunRepository, tmp_path: Path
) -> None:
    # The third error channel: ErrorRecord.context copied KhedronError.context structurally,
    # so a per-question failure persisted the raw values into errors.jsonl while the
    # error_message beside them was already clean. Neither earlier canary touched this file --
    # one covered a successful run, the other a run that died in initialization.
    secret = "sk-proj-canary-must-never-reach-a-context"  # noqa: S105
    await Runner(
        _suite_config(
            [_question_spec("q1", fail_stage="retrieve")],
            provider_config={"binary_path": secret},
        ),
        repository,
    ).run()

    run_dir = next((repository._results_dir / "runs").iterdir())
    errors = _read_jsonl(run_dir / "errors.jsonl")
    # Assert the channel was actually exercised, and specifically that the *field carrying the
    # canary* reached a persisted context. An earlier version asserted only that some context
    # was non-empty; it passed against the unfixed code, because the retrieval failure carried
    # a context that did not include this field at all.
    assert errors, "the run recorded no error, so this proves nothing about error records"
    carriers = [record for record in errors if "binary_path" in record["context"]]
    assert carriers, f"no persisted error context carried the field under test: {errors}"

    needle = secret.encode()
    leaked = [
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")  # noqa: ASYNC240
        if path.is_file() and needle in path.read_bytes()
    ]
    assert leaked == [], f"credential persisted in: {leaked}"


@pytest.mark.asyncio
async def test_runner_writes_no_credential_when_the_run_fails(
    repository: RunRepository, tmp_path: Path
) -> None:
    # The canary above only covers a run that succeeds, so it never touched the error path --
    # and the error path is its own leak channel: a provider reports a failure with the value
    # it was handed, `str(exc)` becomes RunFailedEvent.error_message, and SQLite projects it.
    # A plugin config block is exactly where a credential legitimately lives, so this is the
    # one place the value cannot simply be refused at the door.
    secret = "sk-proj-canary-must-never-reach-an-error"  # noqa: S105
    await Runner(
        _suite_config(
            [_question_spec("q1")],
            provider_config={"fail_initialize": True, "binary_path": secret},
        ),
        repository,
    ).run()

    needle = secret.encode()
    leaked = [
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")  # noqa: ASYNC240
        if path.is_file() and needle in path.read_bytes()
    ]
    assert leaked == [], f"credential persisted in: {leaked}"
    # The run really did fail, so the assertion above was made about a populated error path
    # rather than about an empty one.
    run_dir = next((repository._results_dir / "runs").iterdir())
    events = _read_jsonl(run_dir / "lifecycle.jsonl")
    assert events[-1]["event_type"] == "run_failed"
    assert events[-1]["error_message"]


@pytest.mark.asyncio
async def test_runner_records_the_provider_version_after_initialization(
    repository: RunRepository,
) -> None:
    # provider_version was read before initialize(), so every artifact recorded a
    # placeholder and named no reproducible provider build.
    await Runner(_suite_config([_question_spec("q1")]), repository).run()

    run = repository.list_runs()[0]
    assert run.provider_version == "test"


@pytest.mark.asyncio
async def test_runner_requires_the_minimum_run_count_to_publish(
    repository: RunRepository,
) -> None:
    # This gate is fidelity to the PLANNED run count, not a minimum of three: a three-run
    # experiment that loses one run to infrastructure errors must not silently publish the
    # weaker two-run design it did not plan. (A sample SD is defined for any n >= 2; the point
    # here is the configured design, not an impossibility at n = 2.)
    result = await Runner(
        _suite_config(
            [_question_spec("q1"), _question_spec("q2")],
            runs=3,
            # Degrade only the first run; the other two complete cleanly, leaving two
            # publishable runs where the configured design calls for three.
            provider_config={"fail_retrieve_on_run": 0},
            max_run_error_rate=0.1,
        ),
        repository,
    ).run()

    suite_events = _read_jsonl(
        repository._results_dir / "suites" / result.suite_id / "lifecycle.jsonl"
    )
    failed = [event for event in suite_events if event["event_type"] == "experiment_failed"]

    assert result.experiment_results == []
    assert len(failed) == 1
    assert "3 required" in failed[0]["error_message"]
    # Every executed run paid for model and judge calls, publishable or not.
    assert result.total_cost_usd > 0.0


@pytest.mark.asyncio
async def test_experiment_cost_matches_its_sample_while_suite_reports_actual_spend(
    repository: RunRepository,
) -> None:
    # Two accounting rules that must hold at once: an ExperimentResult's cost describes
    # the same runs as its n_runs/run_ids (so a report never pairs one sample's score
    # with another sample's spend), while the suite total reflects every dollar actually
    # spent, including on the run that was excluded.
    result = await Runner(
        _suite_config(
            [_question_spec("q1"), _question_spec("q2")],
            runs=4,
            provider_config={"fail_retrieve_on_run": 0},
            max_run_error_rate=0.1,
        ),
        repository,
    ).run()

    experiment = result.experiment_results[0]
    published_cost = sum(
        run.total_cost_usd for run in repository.list_runs() if run.run_id in experiment.run_ids
    )

    assert experiment.n_runs == 3
    assert experiment.total_cost_usd == pytest.approx(published_cost)
    # The excluded run still cost money, so the suite total must exceed the published one.
    assert result.total_cost_usd > experiment.total_cost_usd


@pytest.mark.asyncio
async def test_runner_excludes_partial_runs_even_below_the_error_threshold(
    repository: RunRepository,
) -> None:
    # This eligibility rule is stricter than the operational error-rate tolerance: an unresolved
    # question-stage error leaves no durable judgment and therefore cannot enter an aggregate.
    result = await Runner(
        _suite_config(
            [
                _question_spec("q1"),
                _question_spec("q2"),
                _question_spec("q3"),
                _question_spec("q4"),
                _question_spec("q5", fail_stage="judge"),
            ],
            max_run_error_rate=0.5,
        ),
        repository,
    ).run()

    assert result.experiment_results == []


@pytest.mark.asyncio
async def test_runner_initialization_failure_produces_run_failed_event(
    repository: RunRepository,
) -> None:
    result = await Runner(
        _suite_config([_question_spec("q1")], provider_config={"fail_initialize": True}),
        repository,
    ).run()

    run = repository.list_runs()[0]
    run_events = _read_jsonl(repository._results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    suite_events = _read_jsonl(
        repository._results_dir / "suites" / result.suite_id / "lifecycle.jsonl"
    )

    assert run.status == "failed"
    assert run.error_phase == "initialization"
    assert [event["event_type"] for event in run_events] == ["run_started", "run_failed"]
    assert run_events[1]["error_phase"] == "initialization"
    assert [event["event_type"] for event in suite_events] == [
        "suite_started",
        "experiment_started",
        "experiment_failed",
        "suite_completed",
    ]


@pytest.mark.asyncio
async def test_runner_budget_cap_halts_suite_and_persists_completed_work(
    repository: RunRepository,
) -> None:
    # Each fake question costs 0.03 USD (0.01 answer + 0.02 judge), so a 0.01
    # cap is exceeded at the first conversation boundary.
    with pytest.raises(CostExceededError) as exc_info:
        await Runner(
            _suite_config([_question_spec("q1")], max_cost_usd=0.01),
            repository,
        ).run()

    assert "$0.030000" in str(exc_info.value)
    assert "$0.010000" in str(exc_info.value)

    run = repository.list_runs()[0]
    run_events = _read_jsonl(repository._results_dir / "runs" / run.run_id / "lifecycle.jsonl")
    suite_events = _read_jsonl(
        repository._results_dir / "suites" / run.suite_id / "lifecycle.jsonl"
    )
    question = repository.get_question_record(run.run_id, "q1")

    assert run.status == "failed"
    assert [event["event_type"] for event in run_events] == [
        "run_started",
        "conversation_processed",
        "run_failed",
    ]
    assert "budget cap" in run_events[2]["error_message"]
    assert run_events[2]["partial_cost_usd"] == pytest.approx(0.03)
    assert [event["event_type"] for event in suite_events] == [
        "suite_started",
        "experiment_started",
        "suite_failed",
    ]
    assert question.verdict is JudgmentVerdict.CORRECT


@pytest.mark.asyncio
async def test_runner_budget_env_override_wins_over_config(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KHEDRON_MAX_BUDGET_USD", "0.01")

    with pytest.raises(CostExceededError):
        await Runner(
            _suite_config([_question_spec("q1")], max_cost_usd=100.0),
            repository,
        ).run()

    assert repository.list_runs()[0].status == "failed"


class _JudgmentPersistenceFailingRepository(RunRepository):
    """Fails judgment persistence after answer and judge costs are recorded.

    This produces a cost-bearing run failure that never passes a
    conversation-boundary budget check, exercising the run-boundary check.
    """

    async def append_judgment(self, judgment: Judgment) -> None:
        raise PersistenceError("simulated judgment persistence failure")


@pytest.mark.asyncio
async def test_runner_failed_run_spend_over_cap_stops_before_next_run(
    tmp_path: Path,
) -> None:
    repository = _JudgmentPersistenceFailingRepository(
        results_dir=tmp_path / "results",
        sqlite_path=tmp_path / "benchmark.db",
    )

    # The question records 0.03 USD (0.01 answer + 0.02 judge) before the
    # simulated persistence failure fails run 1 mid-conversation, so no
    # conversation-boundary check ever saw the spend. With a 0.02 cap, the
    # run-boundary check must halt the suite before run 2 starts paid work.
    with pytest.raises(CostExceededError) as exc_info:
        await Runner(
            _suite_config([_question_spec("q1")], runs=2, max_cost_usd=0.02),
            repository,
        ).run()

    assert exc_info.value.context["budget_cap_usd"] == pytest.approx(0.02)
    assert exc_info.value.context["total_cost_usd"] == pytest.approx(0.03)

    runs = repository.list_runs()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert len(FakeAnswerModel.generate_calls) == 1
    suite_events = _read_jsonl(
        repository._results_dir / "suites" / runs[0].suite_id / "lifecycle.jsonl"
    )
    assert [event["event_type"] for event in suite_events] == [
        "suite_started",
        "experiment_started",
        "suite_failed",
    ]


@pytest.mark.asyncio
async def test_runner_under_budget_cap_is_unaffected(repository: RunRepository) -> None:
    result = await Runner(
        _suite_config([_question_spec("q1")], max_cost_usd=5.0),
        repository,
    ).run()

    run = repository.list_runs()[0]
    assert repository.get_suite_status(result.suite_id).status == "completed"
    assert run.status == "completed"
    assert result.total_cost_usd == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_runner_same_vendor_fake_config_does_not_raise(repository: RunRepository) -> None:
    await Runner(
        _suite_config(
            [_question_spec("q1")],
            answer_model_type="runner_fake",
            judge_type="runner_fake",
        ),
        repository,
    ).run()

    assert repository.list_runs()[0].status == "completed"


@pytest.mark.asyncio
async def test_runner_passes_answer_model_generation_options(repository: RunRepository) -> None:
    await Runner(
        _suite_config(
            [_question_spec("q1")],
            answer_temperature=0.25,
            answer_max_output_tokens=77,
        ),
        repository,
    ).run()

    assert FakeAnswerModel.generate_calls == [(77, 0.25)]


class FakeProvider(MemoryProvider):
    active_searches: ClassVar[int] = 0
    max_observed_searches: ClassVar[int] = 0
    # The runner builds one provider per run, so counting instantiations gives each run
    # an index and lets a test degrade a specific run rather than all of them.
    instantiations: ClassVar[int] = 0
    closed: ClassVar[int] = 0
    init_entered: ClassVar[int] = 0
    init_reached: ClassVar[asyncio.Event | None] = None
    ingest_reached: ClassVar[asyncio.Event | None] = None
    search_barrier: ClassVar[asyncio.Event | None] = None
    # Every top_k the provider was actually handed. `search` used to `del top_k`, so nothing
    # proved the budget in the YAML reached the provider at all -- the one number canonical-v2
    # pins, and the difference between measuring 200 memories and measuring 10.
    observed_top_k: ClassVar[list[int]] = []
    honours_top_k: ClassVar[bool] = True
    # Persist the memory and then raise, the way a provider does when the write lands and the
    # reply is lost in transit. Nothing distinguishes this from a write that never happened, seen
    # from the caller -- which is exactly why the recovery path asks before it retries.
    persist_then_fail: ClassVar[bool] = False
    transient_search_failures: ClassVar[int] = 0
    last_instance: ClassVar[Any] = None

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._memories: list[Memory] = []
        self._run_index = FakeProvider.instantiations
        self._searches = 0
        self._failed_one_ingest = False
        self._permanently_failed_turn: str | None = None
        FakeProvider.instantiations += 1
        FakeProvider.last_instance = self

    @property
    def provider_type(self) -> str:
        return "runner_fake"

    @property
    def provider_version(self) -> str:
        return "test"

    async def initialize(self) -> None:
        if self._config.get("fail_initialize"):
            # Reports the value it was handed, the way a spawned provider does when a spawn
            # fails. That is what turns a provider's own error
            # into a persistence channel for whatever the config block contained.
            raise ProviderError(
                "fake provider initialization failed",
                binary_path=self._config.get("binary_path", "fake"),
            )
        if self._config.get("hang_initialize"):
            FakeProvider.init_entered += 1
            if FakeProvider.init_reached is not None:
                FakeProvider.init_reached.set()
            # Stands in for a real provider that has already spawned its subprocess and is
            # waiting on a handshake when the operator hits Ctrl-C.
            await asyncio.sleep(3600)

    async def health_check(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(healthy=True, version=self.provider_version)

    async def reset(self) -> None:
        self._memories.clear()

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        # Reproduces a known failure mode: a single turn times out on ingestion while every
        # other turn succeeds, so the run finishes with a memory store that silently does
        # not match the corpus.
        if self._config.get("fail_first_ingest") and not self._failed_one_ingest:
            self._failed_one_ingest = True
            raise ProviderError("fake ingestion timeout")
        # A turn the provider never accepts, however often it is asked. `fail_first_ingest` is the
        # transient case the recovery path is built to repair; this is the one that stays lost, and
        # it is what the exclusion guard exists for.
        if FakeProvider.persist_then_fail and not self._failed_one_ingest:
            self._failed_one_ingest = True
            self._store(content, metadata)
            raise ProviderError("fake timeout after the write landed")
        if self._config.get("fail_ingest_permanently"):
            self._permanently_failed_turn = self._permanently_failed_turn or content
            if content == self._permanently_failed_turn:
                raise ProviderError("fake ingestion timeout that never resolves")
        if self._config.get("hang_ingest"):
            # Stands in for Ctrl-C once the run is already announced and ingesting, which
            # is where a cancellation used to escape the terminal handler entirely.
            if FakeProvider.ingest_reached is not None:
                FakeProvider.ingest_reached.set()
            await asyncio.sleep(3600)
        return self._store(content, metadata)

    def _store(self, content: str, metadata: dict[str, Any] | None) -> str:
        memory_id = f"memory-{len(self._memories) + 1}"
        self._memories.append(
            Memory(
                memory_id=memory_id,
                content=content,
                metadata=dict(metadata) if metadata is not None else {},
                score=1.0,
                timestamp=NOW,
            )
        )
        return memory_id

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        FakeProvider.observed_top_k.append(top_k)
        del metadata_filter
        if "fail-retrieve" in query and (
            not self._config.get("fail_once") or not self.transient_search_failures
        ):
            # Reports the value it was handed, as the real providers do. Without this the
            # per-question error carries no operator value and a canary over errors.jsonl
            # would have nothing to find -- which is how the first version of that test
            # passed against the unfixed code.
            type(self).transient_search_failures += 1
            raise ProviderError(
                "fake retrieval failed",
                binary_path=self._config.get("binary_path", "fake"),
                retryable=self._config.get("retryable"),
            )
        # Degrade one run without zeroing it out: only its first retrieval fails, so the
        # run still errors above a low threshold while its remaining questions reach the
        # model and judge and therefore cost money, like a real rate-limited run.
        first_search = self._searches == 0
        self._searches += 1
        if first_search and self._config.get("fail_retrieve_on_run") == self._run_index:
            raise ProviderError("fake retrieval failed for this run")
        delay = float(self._config.get("search_delay_seconds", 0.0))
        expected_concurrency = self._config.get("search_barrier_concurrency")
        FakeProvider.active_searches += 1
        FakeProvider.max_observed_searches = max(
            FakeProvider.max_observed_searches,
            FakeProvider.active_searches,
        )
        try:
            if expected_concurrency is not None and FakeProvider.search_barrier is not None:
                # Hold each search until the expected number is genuinely in flight. Sleeping
                # a fixed delay instead made the observation depend on two searches happening
                # to overlap inside that window: the concurrency bound wraps the whole
                # question pipeline, so under load one search could finish before the next
                # began and the test failed while the bound was correct. Timing out here
                # fails in bounded time rather than hanging if the bound is too low.
                if FakeProvider.active_searches >= int(expected_concurrency):
                    FakeProvider.search_barrier.set()
                await asyncio.wait_for(FakeProvider.search_barrier.wait(), timeout=5)
            if delay:
                await asyncio.sleep(delay)
            return list(self._memories)
        finally:
            FakeProvider.active_searches -= 1

    async def close(self) -> None:
        FakeProvider.closed += 1

    @classmethod
    def reset_observations(cls) -> None:
        cls.active_searches = 0
        cls.max_observed_searches = 0
        cls.instantiations = 0
        cls.closed = 0
        cls.init_entered = 0
        cls.init_reached = None
        cls.ingest_reached = None
        cls.search_barrier = None
        cls.observed_top_k = []
        cls.honours_top_k = True
        cls.persist_then_fail = False
        cls.transient_search_failures = 0
        cls.last_instance = None

    @classmethod
    def last_instance_contents(cls) -> list[str]:
        return [memory.content for memory in cls.last_instance._memories]


class FakeBenchmark(Benchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._questions: list[Question] = []
        self._conversation = Conversation(
            conversation_id="conversation-1",
            speakers=["Alice", "Bob"],
            sessions=[
                Session(
                    session_id="session-1",
                    session_number=1,
                    timestamp=NOW,
                    # Filler turns are opt-in so the shared fixture stays minimal. A test
                    # that drops one turn needs enough of them to stay under the
                    # ingester's own 5% unreliability threshold, reproducing the real
                    # incident (1 turn of 588); with a single turn, one loss is 100% and
                    # takes a different code path entirely.
                    turns=[
                        Turn(
                            turn_id="turn-1",
                            speaker="Alice",
                            content="Alice bought a blue notebook.",
                        ),
                        *(
                            Turn(
                                turn_id=f"turn-filler-{index}",
                                speaker="Bob",
                                content=f"Filler turn {index}.",
                            )
                            for index in range(2, int(config.get("n_filler_turns", 0)) + 2)
                        ),
                    ],
                )
            ],
        )

    @property
    def benchmark_type(self) -> str:
        return "runner_fake"

    @property
    def benchmark_version(self) -> str:
        return "test"

    @property
    def dataset_checksum(self) -> str:
        return "sha256:runner-fake"

    async def load(self) -> None:
        self._questions = [
            Question(
                question_id=str(raw["id"]),
                conversation_id="conversation-1",
                category=QuestionCategory(str(raw.get("category", "single_hop"))),
                question_text=str(raw["question"]),
                expected_answer=str(raw.get("expected_answer", "Blue")),
                is_audited_error=bool(raw.get("audited", False)),
            )
            for raw in self._config["questions"]
        ]

    def get_conversations(self) -> list[Conversation]:
        return [self._conversation]

    def get_questions(
        self,
        conversation_id: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Question]:
        questions = list(self._questions)
        if conversation_id is not None:
            questions = [
                question for question in questions if question.conversation_id == conversation_id
            ]
        if categories is not None:
            category_values = set(categories)
            questions = [
                question for question in questions if question.category.value in category_values
            ]
        return questions

    def get_audit_errors(self) -> set[str]:
        return {question.question_id for question in self._questions if question.is_audited_error}


class FakeAnswerModel(AnswerModel):
    generate_calls: ClassVar[list[tuple[int, float]]] = []
    transient_failures: ClassVar[int] = 0

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_id = str(config["model_id"])
        self._config = config

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return "runner_fake"

    async def initialize(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        self.generate_calls.append((max_output_tokens, temperature))
        if "fail-generate" in prompt and (
            not self._config.get("fail_once") or not self.transient_failures
        ):
            type(self).transient_failures += 1
            raise ModelError("fake generation failed", retryable=self._config.get("retryable"))
        output = "Wrong" if "answer-wrong" in prompt else "Blue"
        if "fail-judge" in prompt:
            output = "fail-judge"
        return _api_result(output=output, model_id=self._model_id, cost=0.01)

    async def close(self) -> None:
        return None

    @classmethod
    def reset_observations(cls) -> None:
        cls.generate_calls = []
        cls.transient_failures = 0


class FakeJudge(Judge):
    transient_failures: ClassVar[int] = 0

    def __init__(self, config: dict[str, Any], *, temperature: float = 0.0) -> None:
        del temperature
        self._model_id = str(config["model_id"])
        self._config = config

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return "runner_fake"

    async def initialize(self) -> None:
        return None

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        del question, category
        if generated_answer == "fail-judge" and (
            not self._config.get("fail_once") or not self.transient_failures
        ):
            type(self).transient_failures += 1
            raise JudgeError("fake judge failed", retryable=self._config.get("retryable"))
        verdict = (
            JudgmentVerdict.CORRECT
            if generated_answer == expected_answer
            else JudgmentVerdict.INCORRECT
        )
        return JudgeResult(
            verdict=verdict,
            score=1.0 if verdict is JudgmentVerdict.CORRECT else 0.0,
            reasoning="fake reasoning",
            api_call=_api_result(
                output=json.dumps({"verdict": verdict.value}),
                model_id=self._model_id,
                cost=0.02,
                raw_response={"prompt": "fake judge prompt"},
            ),
        )

    async def close(self) -> None:
        return None

    @classmethod
    def reset_observations(cls) -> None:
        cls.transient_failures = 0


def _suite_config(
    questions: list[dict[str, Any]],
    *,
    runs: int = 1,
    provider_config: dict[str, Any] | None = None,
    benchmark_config: dict[str, Any] | None = None,
    max_concurrent_questions: int = 5,
    answer_model_type: str = "runner_fake",
    judge_type: str = "runner_fake",
    answer_temperature: float = 0.0,
    answer_max_output_tokens: int = 1024,
    answer_model_config: dict[str, Any] | None = None,
    judge_config: dict[str, Any] | None = None,
    max_cost_usd: float | None = None,
    max_run_error_rate: float = 1.0,
    max_recovery_attempts: int = 3,
    methodology_profile: str = "canonical-v1",
    methodology_version: str = "1.0",
    top_k_retrieval: int = 10,
    rate_limits: dict[str, Any] | None = None,
) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig.model_validate(
        {
            "runs": runs,
            "methodology_version": methodology_version,
            "methodology_profile": methodology_profile,
            "max_cost_usd": max_cost_usd,
            "max_run_error_rate": max_run_error_rate,
            "max_recovery_attempts": max_recovery_attempts,
            "rate_limits": rate_limits,
            "experiments": [
                {
                    "name": "Runner fake experiment",
                    "provider": {
                        "type": "runner_fake",
                        "config": provider_config or {},
                    },
                    "benchmark": {
                        "type": "runner_fake",
                        "config": {"questions": questions, **(benchmark_config or {})},
                    },
                    "answer_model": {
                        "type": answer_model_type,
                        "model": "fake-answer-model",
                        "temperature": answer_temperature,
                        "max_output_tokens": answer_max_output_tokens,
                        "config": answer_model_config or {},
                    },
                    "judge": {
                        "type": judge_type,
                        "model": "fake-judge-model",
                        "config": judge_config or {},
                    },
                    "top_k_retrieval": top_k_retrieval,
                    "max_concurrent_questions": max_concurrent_questions,
                }
            ],
        }
    )


def _question_spec(
    question_id: str,
    *,
    answer: str = "Blue",
    audited: bool = False,
    fail_stage: str | None = None,
    category: str = "single_hop",
) -> dict[str, Any]:
    markers = {
        "retrieve": " fail-retrieve",
        "generate": " fail-generate",
        "judge": " fail-judge",
    }
    wrong_marker = " answer-wrong" if answer != "Blue" else ""
    return {
        "id": question_id,
        "question": f"What color is Alice's notebook?{markers.get(fail_stage, '')}{wrong_marker}",
        "expected_answer": "Blue",
        "audited": audited,
        "category": category,
    }


def _api_result(
    *,
    output: str,
    model_id: str,
    cost: float,
    raw_response: dict[str, Any] | None = None,
) -> APICallResult:
    return APICallResult(
        output=output,
        input_tokens=10,
        output_tokens=2,
        latency_ms=5.0,
        cost_usd=cost,
        model_id=model_id,
        raw_response=raw_response or {},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _make_interrupted_history(run_dir: Path, completed_question_ids: set[str]) -> None:
    """Trim a completed fixture into the durable prefix a hard interruption would leave."""
    for name in (
        "question_evaluations.jsonl",
        "retrievals.jsonl",
        "responses.jsonl",
        "judgments.jsonl",
    ):
        path = run_dir / name
        rows = [row for row in _read_jsonl(path) if row["question_id"] in completed_question_ids]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lifecycle = run_dir / "lifecycle.jsonl"
    started = [row for row in _read_jsonl(lifecycle) if row["event_type"] == "run_started"]
    lifecycle.write_text("".join(json.dumps(row) + "\n" for row in started), encoding="utf-8")


@contextmanager
def _isolated_registries() -> Iterator[None]:
    provider_registry = dict(PROVIDER_REGISTRY)
    benchmark_registry = dict(BENCHMARK_REGISTRY)
    model_registry = dict(MODEL_REGISTRY)
    judge_registry = dict(JUDGE_REGISTRY)
    PROVIDER_REGISTRY.clear()
    BENCHMARK_REGISTRY.clear()
    MODEL_REGISTRY.clear()
    JUDGE_REGISTRY.clear()
    try:
        yield
    finally:
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(provider_registry)
        BENCHMARK_REGISTRY.clear()
        BENCHMARK_REGISTRY.update(benchmark_registry)
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(model_registry)
        JUDGE_REGISTRY.clear()
        JUDGE_REGISTRY.update(judge_registry)


@pytest.mark.asyncio
async def test_the_scored_subset_reaches_the_persisted_run_through_the_runner(
    repository: RunRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DoR review named this case specifically: a Scorer-level test passes even if the Runner never
    # passes `scored_categories` down, so unit-testing the arithmetic proves nothing about whether
    # production supplies the right input. This drives the real Runner and reads the persisted run.
    from khedron.methodology import profiles as profiles_module

    narrowed = replace(
        profiles_module.get_runtime_profile("canonical-v1"),
        evaluation_categories=("single_hop", "adversarial"),
        scored_categories=("single_hop",),
    )
    monkeypatch.setattr(profiles_module, "_CANONICAL_V1_PROFILE", narrowed)

    await Runner(
        _suite_config(
            [
                _question_spec("q-scored"),
                _question_spec("q-excluded", category="adversarial"),
            ]
        ),
        repository,
    ).run()

    run = repository.list_runs()[0]
    status = repository.get_run_status(run.run_id)

    assert status.overall_score_standard is not None
    # One scored question, not two: the adversarial answer is measured and kept out of the number.
    assert status.overall_score_standard.n_total == 1
    # And it is still reported per category, which is what makes it a diagnostic rather than a
    # discarded question.
    assert "adversarial" in status.by_category_standard


@pytest.mark.asyncio
async def test_the_configured_retrieval_budget_reaches_the_provider(
    repository: RunRepository,
) -> None:
    # canonical-v2's central change is top_k 10 -> 200, and the whole cost model and every claim
    # about retrieval depth rests on that number arriving at the provider. Nothing proved it did:
    # the fake discarded top_k, so a runner that dropped the value on the floor -- or passed the
    # profile's pin instead of the experiment's -- would have kept every test green while a
    # production measurement quietly used a tenth of what its artifacts claimed.
    await Runner(
        _suite_config([_question_spec("q1"), _question_spec("q2")], top_k_retrieval=200),
        repository,
    ).run()

    if FakeProvider.observed_top_k != [200, 200]:
        raise AssertionError(FakeProvider.observed_top_k)


@pytest.mark.asyncio
async def test_the_runner_refuses_a_provider_that_ignores_a_pinned_retrieval_budget(
    repository: RunRepository,
) -> None:
    # The refusal is reachable from preflight, but preflight is advisory: an operator can run the
    # suite without it. This asserts the runner itself fails closed, and that it does so before
    # anything is ingested or any model is called -- a refusal after the spend would be a report,
    # not a guard.
    FakeProvider.honours_top_k = False
    runner = Runner(_suite_config([_question_spec("q1")]), repository)
    runner._methodology_profile = replace(runner._methodology_profile, top_k_retrieval=200)

    with pytest.raises(ConfigurationError) as exc_info:
        await runner.run()

    context = exc_info.value.context
    if context.get("pinned_top_k") != 200:
        raise AssertionError(context)
    if context.get("provider_type") != "runner_fake":
        raise AssertionError(context)
    if FakeProvider.observed_top_k:
        raise AssertionError(f"refused after retrieving: {FakeProvider.observed_top_k}")


def test_full_context_under_canonical_v2_is_refused_by_the_real_pair() -> None:
    # The same guard over the objects that actually ship, rather than a fake standing in for them:
    # if `FullContextProvider.honours_top_k` were flipped, or canonical-v2 stopped pinning a
    # budget, the fake-based test above would still pass while the real baseline recorded a
    # retrieval budget of 200 it never applied.
    from khedron.methodology import get_runtime_profile
    from khedron.providers.full_context import FullContextProvider
    from khedron.runner import _require_retrieval_budget_is_applicable

    suite = load_suite_config(REPO_ROOT / "experiments" / "quickstart.yaml")
    experiment = suite.experiments[0]
    provider = FullContextProvider({})

    with pytest.raises(ConfigurationError):
        _require_retrieval_budget_is_applicable(
            get_runtime_profile("canonical-v2"), provider, experiment
        )
    # ...and stays runnable under the profile the baseline actually declares.
    _require_retrieval_budget_is_applicable(
        get_runtime_profile(suite.methodology_profile), provider, experiment
    )


@pytest.mark.asyncio
async def test_the_runner_refuses_a_corpus_missing_a_category_the_profile_evaluates(
    repository: RunRepository,
) -> None:
    # The real case: conv-30 is the only LoCoMo conversation with no
    # `open_domain` questions, so a corpus reduced to it alone would have exercised
    # neither that category's generation nor its judging while reporting a clean pass. Scores,
    # intervals and fingerprint would all have looked untouched.
    runner = Runner(
        _suite_config([_question_spec("q1", category="single_hop")]),
        repository,
    )
    runner._methodology_profile = replace(
        runner._methodology_profile,
        evaluation_categories=("single_hop", "open_domain"),
    )

    result = await runner.run()

    # The refusal lands as a failed run rather than an exception out of `run()`: it is raised
    # after the corpus is loaded, which is the earliest point the gap is knowable, and the runner
    # records per-run failures rather than propagating them. What matters is that it fails closed
    # -- no model is called, and no aggregate exists to be mistaken for a measurement.
    run = repository.list_runs()[0]
    if run.status != "failed":
        raise AssertionError(run.status)
    if "no questions for categories" not in (run.error_message or ""):
        raise AssertionError(run.error_message)
    if FakeAnswerModel.generate_calls:
        raise AssertionError(FakeAnswerModel.generate_calls)
    if result.experiment_results and result.experiment_results[0].aggregate_overall_standard:
        raise AssertionError("a refused run still produced an aggregate")


@pytest.mark.asyncio
async def test_a_corpus_covering_every_evaluated_category_runs(repository: RunRepository) -> None:
    runner = Runner(
        _suite_config(
            [
                _question_spec("q1", category="single_hop"),
                _question_spec("q2", category="open_domain"),
            ]
        ),
        repository,
    )
    runner._methodology_profile = replace(
        runner._methodology_profile,
        evaluation_categories=("single_hop", "open_domain"),
    )

    result = await runner.run()

    if repository.get_suite_status(result.suite_id).status != "completed":
        raise AssertionError(repository.get_suite_status(result.suite_id))


@pytest.mark.asyncio
async def test_a_transient_ingestion_failure_is_retried_rather_than_lost(
    repository: RunRepository,
) -> None:
    # A single turn once timed out on ingestion, and that one loss was enough to exclude an
    # entire run from the publishable aggregate. A transient failure must be
    # repaired rather than counted.
    result = await Runner(
        _suite_config([_question_spec("q1")], provider_config={"fail_first_ingest": True}),
        repository,
    ).run()

    run = repository.list_runs()[0]
    if run.status != "completed":
        raise AssertionError(run.status)
    if not result.experiment_results:
        raise AssertionError("a recovered ingestion still produced no aggregate")


@pytest.mark.asyncio
async def test_a_turn_the_provider_already_persisted_is_not_written_twice(
    repository: RunRepository,
) -> None:
    # The dangerous case, and the reason this is not a plain retry: the provider persists the
    # memory and the reply is lost on the way back. A blind retry would duplicate the turn, and a
    # duplicate silently changes what retrieval sees -- worse than the loss it repairs, because
    # nothing downstream counts it.
    FakeProvider.persist_then_fail = True
    result = await Runner(_suite_config([_question_spec("q1")]), repository).run()

    if not result.experiment_results:
        raise AssertionError("the run produced no aggregate")
    contents = FakeProvider.last_instance_contents()
    duplicates = [item for item, count in collections.Counter(contents).items() if count > 1]
    if duplicates:
        raise AssertionError(f"turn written twice: {duplicates}")
    run = repository.list_runs()[0]
    if run.status != "completed":
        raise AssertionError(run.status)


@pytest.mark.asyncio
async def test_a_run_beyond_the_ingestion_threshold_stops_before_paying_for_questions(
    repository: RunRepository,
) -> None:
    # A lost turn during ingestion can otherwise go unnoticed until every question and judgement
    # for that conversation has already been paid for, only to be discarded afterward by the
    # aggregate guard. That spend is avoidable: the run is already known to be unpublishable the
    # moment the turn is lost.
    result = await Runner(
        _suite_config(
            [_question_spec("q1"), _question_spec("q2")],
            provider_config={"fail_ingest_permanently": True},
            # 40 turns so one permanent loss is 2.5%: under the ingester's own 5% unreliability
            # threshold, which would abort for an entirely different reason and prove nothing,
            # and over the suite's publishable threshold, which is what this asserts.
            benchmark_config={"n_filler_turns": 39},
        ),
        repository,
    ).run()

    if FakeAnswerModel.generate_calls:
        raise AssertionError(
            f"questions were paid for after the run was doomed: {FakeAnswerModel.generate_calls}"
        )
    run = repository.list_runs()[0]
    if run.status != "failed":
        raise AssertionError(run.status)
    if result.experiment_results:
        raise AssertionError("an abandoned run still produced an aggregate")


@pytest.mark.asyncio
async def test_rejudging_reuses_the_stored_answers_byte_for_byte(
    repository: RunRepository,
) -> None:
    # The point of rejudging is that the graded text does not change: run the benchmark twice with
    # different judges and the answers are regenerated too, so the difference mixes the judges'
    # disagreement with the answer model's own variability. This asserts the property that makes
    # the judge comparison exact.
    from khedron.rejudge import rejudge_run

    await Runner(_suite_config([_question_spec("q1"), _question_spec("q2")]), repository).run()
    source_run_id = repository.list_runs()[0].run_id
    source_answers = {
        response.question_id: response.answer_text
        for response in repository.get_responses_for_run(source_run_id)
    }

    result = await rejudge_run(
        source_run_id=source_run_id,
        profile_name="canonical-v1",
        judge=FakeJudge({"model_id": "second-judge"}),
        repository=repository,
    )

    if result.run_id == source_run_id:
        raise AssertionError("rejudging must produce a new run, not mutate the source")
    derived_answers = {
        response.question_id: response.answer_text
        for response in repository.get_responses_for_run(result.run_id)
    }
    if derived_answers != source_answers:
        raise AssertionError(f"answers changed: {source_answers} -> {derived_answers}")
    if result.n_questions != 2:
        raise AssertionError(result.n_questions)

    started = repository.get_run_started_event(result.run_id)
    if started.runtime_environment.get("derived_from_run_id") != source_run_id:
        raise AssertionError(started.runtime_environment)
    if started.runtime_environment.get("answers_regenerated") is not False:
        raise AssertionError(started.runtime_environment)
    if started.judge_model_id != "second-judge":
        raise AssertionError(started.judge_model_id)
    # The source run is untouched: its own judge is still recorded against it.
    if repository.get_run_started_event(source_run_id).judge_model_id == "second-judge":
        raise AssertionError("rejudging overwrote the source run's judge")


@pytest.mark.asyncio
async def test_rejudging_refuses_a_profile_that_pins_a_different_answer_model(
    repository: RunRepository,
) -> None:
    # Rejudging holds the answers fixed and varies the judge. A target profile pinning a different
    # answer model would make the derived run claim answers it never produced, and the comparison
    # it exists to support would vary two things at once.
    from khedron.rejudge import rejudge_run

    await Runner(_suite_config([_question_spec("q1")]), repository).run()
    source_run_id = repository.list_runs()[0].run_id

    with pytest.raises(ConfigurationError) as exc_info:
        await rejudge_run(
            source_run_id=source_run_id,
            profile_name="canonical-v2",
            judge=FakeJudge({"model_id": "claude-haiku-4-5-20251001"}),
            repository=repository,
        )

    # The guard now reports every generation-affecting field that differs, not the answer model
    # alone: a target differing on the corpus policy or the generator prompt would have the derived
    # run declare a generation it never performed, which is the same defect one field wider.
    differing = exc_info.value.context.get("differing_fields") or []
    if "answer_model_id" not in differing:
        raise AssertionError(exc_info.value.context)


@pytest.mark.asyncio
async def test_resuming_skips_what_was_already_paid_for(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The property that makes resume worth having: a run interrupted after N questions must not
    # pay for those N again. Asserted against the answer model's own call log, because that is
    # what the money is spent on -- a resume that merely produced the right final score while
    # re-asking everything would cost exactly as much as starting over.
    from khedron.resume import load_resume_state

    # Pinned so the test does not depend on whether the developer's tree happens to be committed:
    # resume refuses an unidentifiable build, which is correct in production and irrelevant here.
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec(f"q{index}") for index in range(4)])
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    calls_for_full_run = len(FakeAnswerModel.generate_calls)

    # Build the durable state an interruption would have left: the immutable manifest and stages
    # for q0/q1 exist, but no terminal lifecycle event was written.  Do not forge a ResumeState on
    # top of a completed stream: that creates duplicate stages, which the strict reader must reject.
    _make_interrupted_history(repository._results_dir / "runs" / run_id, {"q0", "q1"})
    state = load_resume_state(run_id=run_id, config=suite, repository=repository)

    FakeAnswerModel.reset_observations()
    runner = Runner(suite, repository)
    runner._resume = state
    await runner.run()

    if calls_for_full_run != 4:
        raise AssertionError(calls_for_full_run)
    if len(FakeAnswerModel.generate_calls) != 2:
        raise AssertionError(
            f"resume re-asked questions it had already answered: "
            f"{len(FakeAnswerModel.generate_calls)} calls, expected 2"
        )

    events = repository.list_run_events(run_id)
    resumed = [event for event in events if event.event_type == "run_resumed"]
    if len(resumed) != 1:
        raise AssertionError([event.event_type for event in events])
    if resumed[0].n_questions_inherited != 2:
        raise AssertionError(resumed[0])
    # The original started event stays authoritative, and the stream still records the first pass.
    if repository.get_run_started_event(run_id).run_id != run_id:
        raise AssertionError("resume lost the run's identity")


def test_resuming_refuses_a_configuration_that_changed(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resumed run is a single artifact claiming a single measurement. Continuing it under an
    # edited configuration would make that claim false while every score and interval still looked
    # untouched -- which is the failure mode the whole methodology version exists to remove.
    from khedron.resume import load_resume_state, suite_config_digest

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec("q1")])
    import asyncio as _asyncio

    _asyncio.run(Runner(suite, repository).run())
    run_id = repository.list_runs()[0].run_id

    edited = _suite_config([_question_spec("q1")], answer_max_output_tokens=2048)
    if suite_config_digest(edited) == suite_config_digest(suite):
        raise AssertionError("the two configurations must differ for this test to mean anything")

    with pytest.raises(ConfigurationError) as exc_info:
        load_resume_state(run_id=run_id, config=edited, repository=repository)

    if exc_info.value.context.get("field") != "configuration digest":
        raise AssertionError(exc_info.value.context)


@pytest.mark.asyncio
async def test_a_resumed_run_scores_everything_it_asked_not_only_the_second_half(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The record accumulator started empty, so a run interrupted at
    # 90% would publish a score computed from its last 10% with nothing in the artifact saying so.
    from khedron.resume import load_resume_state

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    # Two right, two wrong: any subset scores differently from the whole, so the assertion binds.
    suite = _suite_config(
        [
            _question_spec("q0"),
            _question_spec("q1"),
            _question_spec("q2", answer="Wrong"),
            _question_spec("q3", answer="Wrong"),
        ]
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id

    state = load_resume_state(run_id=run_id, config=suite, repository=repository)
    runner = Runner(suite, repository)
    # Everything was answered, so the resumed pass asks nothing: the score must still cover all
    # four questions, taken from the records the interrupted run left behind.
    runner._resume = replace(state, completed_conversations=frozenset())
    result = await runner.run()

    scores = result.experiment_results[0].aggregate_overall_standard
    if scores is None:
        raise AssertionError("a resumed run produced no aggregate")
    # 2 of 4, not 0 of 2: the inherited correct answers count.
    if scores.pooled_score.n_total != 4:
        raise AssertionError(f"scored {scores.pooled_score.n_total} questions, expected all 4")
    if scores.pooled_score.n_correct != 2:
        raise AssertionError(scores.pooled_score.n_correct)


@pytest.mark.asyncio
async def test_a_resumed_run_reports_the_whole_runs_cost_not_the_second_attempts(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `resume.prior_spend_usd` was computed, written into the
    # run_resumed event, and then never used again: the run's reported cost came from a cost
    # tracker that started empty, so a run interrupted at 90% declared the cost of its last 10%.
    # Asserted as the invariant rather than against a literal -- a run's reported cost is the sum
    # of the API calls made under its run_id, and nothing else is a defensible definition.
    from khedron.resume import load_resume_state

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec(f"q{index}") for index in range(4)])
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    spend_before_resume = sum(call.cost_usd for call in repository.get_api_calls_for_run(run_id))
    if spend_before_resume <= 0.0:
        raise AssertionError("the fixture must bill something for this test to bind")

    _make_interrupted_history(repository._results_dir / "runs" / run_id, {"q0", "q1"})
    state = load_resume_state(run_id=run_id, config=suite, repository=repository)
    runner = Runner(suite, repository)
    # Two questions remain, so the second attempt's cost is strictly less than the run's.
    runner._resume = state
    await runner.run()

    billed = sum(call.cost_usd for call in repository.get_api_calls_for_run(run_id))
    if billed <= spend_before_resume:
        raise AssertionError("the resumed pass must have billed something further")
    # Read from the append-only stream rather than the SQLite projection: the projection is
    # refreshed best-effort, and this fixture -- which re-asks questions the first pass had in
    # fact already recorded -- collides on its unique index and leaves it stale.
    completed = [
        event for event in repository.list_run_events(run_id) if event.event_type == "run_completed"
    ]
    reported = completed[-1].total_cost_usd
    if reported != pytest.approx(billed):
        raise AssertionError(
            f"run reported ${reported:.4f} against ${billed:.4f} actually billed under its run_id"
        )


@pytest.mark.asyncio
async def test_the_budget_cap_counts_what_a_resumed_run_already_spent(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other half of the same defect, and the worse half: the cap was measured against the
    # understated figure too, so a resumed run could exceed it by whatever the interrupted attempt
    # had already cost. The cap is applied through the environment override here because putting it
    # in the suite config would change the configuration digest, and resume correctly refuses that.
    from khedron.resume import load_resume_state

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec(f"q{index}") for index in range(4)])
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    already_spent = sum(call.cost_usd for call in repository.get_api_calls_for_run(run_id))

    state = load_resume_state(run_id=run_id, config=suite, repository=repository)
    # Above what the second attempt alone costs, below what the run has already spent: the cap can
    # only fire if the inherited spend is counted. Set before the runner is built, because the cap
    # is resolved once at construction.
    monkeypatch.setenv("KHEDRON_MAX_BUDGET_USD", f"{already_spent * 0.75:.6f}")
    runner = Runner(suite, repository)
    runner._resume = replace(
        state,
        completed_conversations=frozenset(),
        completed_questions=frozenset({"q0", "q1"}),
    )

    with pytest.raises(CostExceededError):
        await runner.run()


def test_resume_refuses_a_question_stage_outside_the_durable_manifest(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unplanned stage makes the manifest-bearing stream corrupt. Resume must refuse it
    # before it can construct a provider or risk extending an unverifiable measurement.
    import asyncio as _asyncio

    from khedron.errors import PersistenceError
    from khedron.resume import load_resume_state
    from khedron.types import QuestionEvaluationRecord
    from khedron.utils.ids import generate_ulid
    from khedron.utils.time import now_utc

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec("q-done")])
    _asyncio.run(Runner(suite, repository).run())
    run_id = repository.list_runs()[0].run_id

    _asyncio.run(
        repository.append_question_evaluation(
            QuestionEvaluationRecord(
                question_evaluation_id=generate_ulid(),
                run_id=run_id,
                question_id="q-killed",
                conversation_id="conversation-1",
                category=QuestionCategory.SINGLE_HOP,
                question_text="interrupted",
                expected_answer="Blue",
                is_audited_error=False,
                timestamp=now_utc(),
            )
        )
    )
    if "q-killed" not in repository.get_evaluated_question_ids(run_id):
        raise AssertionError("the half-written record was not written as intended")

    with pytest.raises(PersistenceError):
        load_resume_state(run_id=run_id, config=suite, repository=repository)


@pytest.mark.asyncio
async def test_resume_recovers_only_a_retryable_generate_stage(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-recover", fail_stage="generate")],
        answer_model_config={"fail_once": True, "retryable": True},
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    assert FakeAnswerModel.transient_failures == 1

    await Runner(suite, repository, resume_run_id=run_id).run()
    lifecycle = repository.list_run_events(run_id)
    assert lifecycle[-1].event_type == "run_completed"

    run_dir = repository._results_dir / "runs" / run_id
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "retrievals.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "responses.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "judgments.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "error_resolutions.jsonl")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "provider_config", "judge_config"),
    [
        ("retrieve", {"fail_once": True, "retryable": True}, None),
        ("judge", None, {"fail_once": True, "retryable": True}),
    ],
)
async def test_resume_recovers_retryable_stage_without_duplicate_evaluation(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    provider_config: dict[str, Any] | None,
    judge_config: dict[str, Any] | None,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-recover", fail_stage=stage)],
        provider_config=provider_config,
        judge_config=judge_config,
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id

    await Runner(suite, repository, resume_run_id=run_id).run()

    run_dir = repository._results_dir / "runs" / run_id
    assert repository.list_run_events(run_id)[-1].event_type == "run_completed"
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "responses.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "judgments.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "error_resolutions.jsonl")) == 1
    retrievals = _read_jsonl(run_dir / "retrievals.jsonl")
    assert len(retrievals) == 1
    if stage == "retrieve":
        assert retrievals[0]["ingestion_attempt_id"] is not None
        assert len(_read_jsonl(run_dir / "conversations.jsonl")) == 2
    else:
        assert retrievals[0]["ingestion_attempt_id"] is None


@pytest.mark.asyncio
async def test_resume_reconciles_crash_after_target_idempotently(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-reconcile", fail_stage="generate")],
        answer_model_config={"retryable": True},
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    error = repository.get_errors_for_run(run_id)[0]
    retrieval = repository.get_retrievals_for_run(run_id)[0]
    attempt = RecoveryAttemptRecord(
        attempt_id=generate_ulid(),
        run_id=run_id,
        question_id="q-reconcile",
        error_id=error.error_id,
        stage="generate",
        timestamp=now_utc(),
    )
    await repository.append_recovery_attempt(attempt)
    await repository.append_response(
        Response(
            response_id=generate_ulid(),
            run_id=run_id,
            question_id="q-reconcile",
            retrieval_id=retrieval.retrieval_id,
            timestamp=now_utc(),
            model_id="fake-answer-model",
            prompt="crash-window response",
            answer_text="Blue",
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.0,
            cost_usd=0.0,
            recovery_attempt_id=attempt.attempt_id,
        )
    )

    await Runner(suite, repository, resume_run_id=run_id).run()
    run_dir = repository._results_dir / "runs" / run_id
    counts_after_first_resume = {
        name: len(_read_jsonl(run_dir / name))
        for name in (
            "question_evaluations.jsonl",
            "responses.jsonl",
            "judgments.jsonl",
            "error_resolutions.jsonl",
        )
    }
    assert counts_after_first_resume == {
        "question_evaluations.jsonl": 1,
        "responses.jsonl": 1,
        "judgments.jsonl": 1,
        "error_resolutions.jsonl": 1,
    }

    await Runner(suite, repository, resume_run_id=run_id).run()
    assert {
        name: len(_read_jsonl(run_dir / name)) for name in counts_after_first_resume
    } == counts_after_first_resume


@pytest.mark.asyncio
async def test_resume_stops_retryable_recovery_at_its_durable_attempt_budget(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-budget", fail_stage="generate")],
        answer_model_config={"retryable": True},
        max_recovery_attempts=1,
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    await Runner(suite, repository, resume_run_id=run_id).run()
    run_dir = repository._results_dir / "runs" / run_id
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1

    await Runner(suite, repository, resume_run_id=run_id).run()
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 1
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1


@pytest.mark.asyncio
async def test_resume_aggregates_multiple_recovery_attempts_before_budget_terminal(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-budget-two", fail_stage="generate")],
        answer_model_config={"retryable": True},
        max_recovery_attempts=2,
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    run_dir = repository._results_dir / "runs" / run_id

    # Each recovery fails durably, producing a later error. The next resume must select only that
    # latest error for the same question/stage, until the shared two-attempt budget is exhausted.
    await Runner(suite, repository, resume_run_id=run_id).run()
    await Runner(suite, repository, resume_run_id=run_id).run()
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 2
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1

    await Runner(suite, repository, resume_run_id=run_id).run()
    assert len(_read_jsonl(run_dir / "recovery_attempts.jsonl")) == 2
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [False, None], ids=["false", "legacy-none"])
async def test_resume_does_not_retry_non_retryable_or_legacy_unknown_errors(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
    retryable: bool | None,
) -> None:
    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    suite = _suite_config(
        [_question_spec("q-tristate", fail_stage="generate")],
        answer_model_config={"retryable": retryable},
    )
    await Runner(suite, repository).run()
    run_id = repository.list_runs()[0].run_id
    await Runner(suite, repository, resume_run_id=run_id).run()

    run_dir = repository._results_dir / "runs" / run_id
    assert not (run_dir / "recovery_attempts.jsonl").exists()
    assert len(_read_jsonl(run_dir / "question_evaluations.jsonl")) == 1


@pytest.mark.asyncio
async def test_a_rejudged_run_references_only_records_it_contains(
    repository: RunRepository,
) -> None:
    # The derived run copies the retrieval under a new identifier. Leaving the answer pointing at
    # the source run's makes it reference a record it does not contain -- the self-containment the
    # copying exists to provide, silently absent, and the index joins responses on that field.
    from khedron.rejudge import rejudge_run

    await Runner(_suite_config([_question_spec("q1"), _question_spec("q2")]), repository).run()
    source_run_id = repository.list_runs()[0].run_id

    result = await rejudge_run(
        source_run_id=source_run_id,
        profile_name="canonical-v1",
        judge=FakeJudge({"model_id": "second-judge"}),
        repository=repository,
    )

    retrieval_ids = {
        retrieval.retrieval_id for retrieval in repository.get_retrievals_for_run(result.run_id)
    }
    dangling = [
        response.response_id
        for response in repository.get_responses_for_run(result.run_id)
        if response.retrieval_id not in retrieval_ids
    ]
    if dangling:
        raise AssertionError(f"{len(dangling)} answers reference a retrieval from another run")
    source_retrievals = {
        retrieval.retrieval_id for retrieval in repository.get_retrievals_for_run(source_run_id)
    }
    if retrieval_ids & source_retrievals:
        raise AssertionError("the derived run reused the source run's retrieval identifiers")


@pytest.mark.asyncio
async def test_the_fingerprint_a_run_records_is_the_one_its_readers_look_for(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three spellings of one key: the runner wrote `methodology_profile_fingerprint`, resume read
    # `methodology_fingerprint`, and rejudge wrote that second name. Resume's guard therefore never
    # fired -- it skips a recorded None and always found one -- and a rejudged run's report
    # disclosed the fingerprint it inherited from its source instead of its own.
    from khedron.methodology import (
        METHODOLOGY_FINGERPRINT_KEY,
        get_runtime_profile,
        methodology_fingerprint,
    )
    from khedron.rejudge import rejudge_run
    from khedron.resume import load_resume_state

    monkeypatch.setattr("khedron.resume.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")

    suite = _suite_config([_question_spec("q1")])
    await Runner(suite, repository).run()
    source_run_id = repository.list_runs()[0].run_id
    source_environment = repository.get_run_started_event(source_run_id).runtime_environment

    if METHODOLOGY_FINGERPRINT_KEY not in source_environment:
        raise AssertionError(sorted(source_environment))

    result = await rejudge_run(
        source_run_id=source_run_id,
        profile_name="canonical-v1",
        judge=FakeJudge({"model_id": "second-judge"}),
        repository=repository,
    )
    derived = repository.get_run_started_event(result.run_id).runtime_environment
    expected = methodology_fingerprint(get_runtime_profile("canonical-v1"))
    if derived[METHODOLOGY_FINGERPRINT_KEY] != expected:
        raise AssertionError(
            "the derived run discloses a fingerprint that is not its own: "
            f"{derived[METHODOLOGY_FINGERPRINT_KEY]} != {expected}"
        )

    # And the resume guard now reads a key that is actually written, so it can compare rather than
    # skip. Loading state under a matching config must succeed and under a different profile must
    # not -- which is only possible if the recorded value is found at all.
    load_resume_state(run_id=source_run_id, config=suite, repository=repository)


@pytest.mark.asyncio
async def test_a_rejudged_answer_gets_its_own_identity(repository: RunRepository) -> None:
    # `response_id` is a global lookup key -- `get_response` resolves it across runs -- so reusing
    # the source's made one identifier match two rows and a lookup could hydrate the wrong run's
    # answer. And it must be minted once: the response persisted into the derived run and the one
    # the judge scores have to be the same record.
    from khedron.rejudge import rejudge_run

    await Runner(_suite_config([_question_spec("q1"), _question_spec("q2")]), repository).run()
    source_run_id = repository.list_runs()[0].run_id
    source_ids = {r.response_id for r in repository.get_responses_for_run(source_run_id)}

    result = await rejudge_run(
        source_run_id=source_run_id,
        profile_name="canonical-v1",
        judge=FakeJudge({"model_id": "second-judge"}),
        repository=repository,
    )

    derived = repository.get_responses_for_run(result.run_id)
    derived_ids = {r.response_id for r in derived}
    if derived_ids & source_ids:
        raise AssertionError("the derived run reused the source run's response identifiers")
    # Every judgment points at an answer this run actually holds.
    judged = {
        json.loads(line)["response_id"]
        for line in (repository._results_dir / "runs" / result.run_id / "judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    if not judged <= derived_ids:
        raise AssertionError("a judgment references an answer the derived run does not contain")


@pytest.mark.asyncio
async def test_rejudging_refuses_a_target_that_changes_what_the_corpus_contained(
    repository: RunRepository,
) -> None:
    # A case caught before it could be executed: re-scoring answers generated
    # without image descriptions, under a profile whose policy declares they were ingested. The
    # derived run stamps the target's fingerprint, so the artifact would assert a corpus it never
    # saw. The narrow answer-model check admitted it; the field set does not.
    from khedron.rejudge import rejudge_run

    await Runner(
        _suite_config(
            [
                # Every evaluated category, because the profile pins all five and the runner
                # refuses a corpus that cannot supply one of them.
                _question_spec("q1", category="single_hop"),
                _question_spec("q2", category="multi_hop"),
                _question_spec("q3", category="temporal"),
                _question_spec("q4", category="open_domain"),
                _question_spec("q5", category="adversarial"),
            ],
            methodology_profile="canonical-v3-generator-only",
        ),
        repository,
    ).run()
    source_run_id = repository.list_runs()[0].run_id

    with pytest.raises(ConfigurationError) as exc_info:
        await rejudge_run(
            source_run_id=source_run_id,
            profile_name="canonical-v3",
            judge=FakeJudge({"model_id": "claude-haiku-4-5-20251001"}),
            repository=repository,
        )

    differing = exc_info.value.context.get("differing_fields") or []
    if "image_description_policy" not in differing:
        raise AssertionError(exc_info.value.context)

    # The legitimate ablation runs the other way round -- rejudging a v3-corpus run under the v2
    # judge -- so this source, generated on the v2 corpus, has no valid v3 target at all. That the
    # guard refuses every one of them is the property under test.
    with pytest.raises(ConfigurationError) as second:
        await rejudge_run(
            source_run_id=source_run_id,
            profile_name="canonical-v3-prior-judge",
            judge=FakeJudge({"model_id": "claude-haiku-4-5-20251001"}),
            repository=repository,
        )
    if "image_description_policy" not in (second.value.context.get("differing_fields") or []):
        raise AssertionError(second.value.context)


def test_the_runner_threads_one_shared_limiter_into_answer_models_and_judges(
    repository: RunRepository,
) -> None:
    # The whole point of the wiring: the pacer configured on the suite must reach the paid calls.
    # One limiter, shared -- the answer model and the judge's wrapped model hold the same instance,
    # so a rolling-window limit is not split across them. Registers the real openai/anthropic
    # adapters (the isolated registry only has runner_fake, which bypasses the pacer).
    from khedron.judges.anthropic import AnthropicJudge
    from khedron.models.openai import OpenAIModel

    MODEL_REGISTRY["openai"] = OpenAIModel
    JUDGE_REGISTRY["anthropic"] = AnthropicJudge

    config = _suite_config(
        [_question_spec("q1")],
        rate_limits={
            "limits": [{"vendor": "openai", "model": "gpt-4o-mini", "tokens_per_minute": 100_000}]
        },
    )
    runner = Runner(config, repository)
    if runner._rate_limiter is None:
        raise AssertionError("a configured rate_limits block must build a limiter")

    answer = runner._instantiate_answer_model(AnswerModelConfig(type="openai", model="gpt-4o-mini"))
    if answer._rate_limiter is not runner._rate_limiter:  # type: ignore[attr-defined]
        raise AssertionError("the answer model did not receive the suite's shared limiter")

    judge = runner._instantiate_judge(
        JudgeConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    )
    if judge._model._rate_limiter is not runner._rate_limiter:  # type: ignore[attr-defined]
        raise AssertionError("the judge's wrapped model did not receive the shared limiter")


def test_the_runner_leaves_models_unpaced_when_no_rate_limits_are_configured(
    repository: RunRepository,
) -> None:
    from khedron.models.openai import OpenAIModel

    MODEL_REGISTRY["openai"] = OpenAIModel

    config = _suite_config([_question_spec("q1")])  # no rate_limits
    runner = Runner(config, repository)
    if runner._rate_limiter is not None:
        raise AssertionError("no rate_limits must leave the limiter None")

    answer = runner._instantiate_answer_model(AnswerModelConfig(type="openai", model="gpt-4o-mini"))
    if answer._rate_limiter is not None:  # type: ignore[attr-defined]
        raise AssertionError("without configured rate limits the answer model must be unpaced")


def test_rejudge_rate_limits_file_threads_a_limiter_into_the_judge(
    tmp_path: Path,
) -> None:
    # The rejudge path builds its judge outside the Runner, from an explicit --rate-limits file.
    # This exercises that chain: strict-load the file, build the limiter, and confirm build_judge
    # (the same function the CLI calls) threads it into the judge's wrapped model.
    from khedron.judges.anthropic import AnthropicJudge
    from khedron.methodology import get_runtime_profile
    from khedron.pacing import build_rate_limiter, load_rate_limits_config
    from khedron.runner import build_judge

    JUDGE_REGISTRY["anthropic"] = AnthropicJudge

    limits_path = tmp_path / "rate_limits.yaml"
    limits_path.write_text(
        "limits:\n  - vendor: anthropic\n    model: claude-haiku-4-5-20251001\n"
        "    tokens_per_minute: 100000\n",
        encoding="utf-8",
    )
    limits_config = load_rate_limits_config(limits_path)
    limiter = build_rate_limiter(limits_config)
    if limiter is None:
        raise AssertionError("the rate-limits file must build a limiter")

    profile = get_runtime_profile("canonical-v2")
    judge = build_judge(
        JudgeConfig(type="anthropic", model="claude-haiku-4-5-20251001"),
        profile,
        rate_limiter=limiter,
        rate_limit_scope=limits_config.scope,
    )
    if judge._model._rate_limiter is not limiter:  # type: ignore[attr-defined]
        raise AssertionError("rejudge's --rate-limits limiter did not reach the judge's model")
