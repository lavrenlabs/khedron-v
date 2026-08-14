from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal, TypeVar, cast, get_type_hints

import structlog
from pydantic import BaseModel, ValidationError

from khedron.analysis.scorer import Scorer
from khedron.benchmarks.base import Benchmark
from khedron.benchmarks.locomo import EXPECTED_DATASET_SHA256, LocomoBenchmark
from khedron.benchmarks.registry import get_benchmark_class
from khedron.budget import BudgetLedger, BudgetReserver, resolve_budget_cap_usd
from khedron.build import is_unidentifiable_build, resolve_framework_version
from khedron.config import (
    AnswerModelConfig,
    BenchmarkConfig,
    ExperimentConfig,
    ExperimentSuiteConfig,
    JudgeConfig,
    ProviderConfig,
    format_validation_errors,
)
from khedron.cost.tracker import CostTracker
from khedron.eligibility import assess_run_eligibility
from khedron.errors import (
    BenchmarkError,
    ConfigurationError,
    CostExceededError,
    JudgeError,
    ModelError,
    ProviderError,
)
from khedron.judges.anthropic import AnthropicJudge
from khedron.judges.base import Judge
from khedron.judges.google import GoogleJudge
from khedron.judges.openai import OpenAIJudge
from khedron.judges.registry import get_judge_class
from khedron.methodology import (
    METHODOLOGY_FINGERPRINT_KEY,
    MethodologyRuntimeProfile,
    get_runtime_profile,
    methodology_fingerprint,
    validate_suite_methodology_profile,
)
from khedron.models.anthropic import AnthropicModel, AnthropicModelConfig
from khedron.models.base import AnswerModel
from khedron.models.google import GoogleModel, GoogleModelConfig
from khedron.models.openai import OpenAIModel, OpenAIModelConfig
from khedron.models.registry import get_model_class
from khedron.pacing import build_rate_limiter
from khedron.persistence.repository import RunRepository
from khedron.persistence.stage_reader import RecoveryPending
from khedron.pipeline.generator import generate_answer
from khedron.pipeline.ingester import ingest_conversation
from khedron.pipeline.judger import judge_response
from khedron.pipeline.retriever import retrieve_for_question
from khedron.providers.base import MemoryProvider
from khedron.providers.registry import get_provider_class
from khedron.ratelimit import RateLimiter
from khedron.resume import (
    RecoveryWorkItem,
    ResumeState,
    acquire_run_lock,
    load_resume_state,
    revalidate_forced_recovery,
    suite_config_digest,
)
from khedron.types import (
    AggregateScore,
    Conversation,
    ConversationProcessedEvent,
    ErrorResolutionRecord,
    ExperimentCompletedEvent,
    ExperimentFailedEvent,
    ExperimentResult,
    ExperimentStartedEvent,
    ForcedRecoveryEvent,
    JudgmentVerdict,
    Question,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    QuestionRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunFailedEvent,
    RunResumedEvent,
    RunScores,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteFailedEvent,
    SuiteStartedEvent,
    question_plan_fingerprint,
)
from khedron.utils.ids import generate_ulid
from khedron.utils.redaction import REDACTED, looks_like_credential, redact_secrets
from khedron.utils.stats import wilson_score_interval
from khedron.utils.time import now_utc

__all__ = ["RecoverBlockedResult", "Runner"]

_LOGGER = structlog.get_logger(__name__)


class _ForcedRecoveryRevalidationAborted(ConfigurationError):
    """Signals that the under-lock forced-recovery re-validation refused before any write.

    Distinct from a general run failure so the terminal handler can re-raise it cleanly -- without
    announcing the run or recording a RunFailedEvent -- while the lock's ``finally`` still releases
    the lock. A ConfigurationError subclass, so the CLI still maps it to a user error: a lost race
    costs an aborted command, never a mutated run.
    """


@dataclass(frozen=True)
class RecoverBlockedResult:
    """Outcome of a ``recover-blocked`` invocation, for the CLI summary."""

    run_id: str
    recovered_question_ids: tuple[str, ...]
    unrecovered_question_ids: tuple[str, ...]
    forced_recovery_count: int
    incremental_cost_usd: float


_RUN_ERROR_PHASES = {"initialization", "ingest", "retrieve", "generate", "judge", "analysis"}
# Three runs are the documented minimum for a meaningful aggregate: two cannot compute a
# standard deviation. See the methodology documentation for the full rationale.
_MIN_RUNS_FOR_AGGREGATE = 3

RunErrorPhase = Literal["initialization", "ingest", "retrieve", "generate", "judge", "analysis"]
ScoreMode = Literal["standard", "audited"]


@dataclass
class _RunContext:
    run_id: str
    suite_id: str
    experiment_id: str
    experiment_name: str
    run_number: int
    seed: int
    cost_tracker: CostTracker


@dataclass(frozen=True)
class _RunResult:
    run_id: str
    suite_id: str
    experiment_id: str
    experiment_name: str
    run_number: int
    scores: RunScores
    question_records: list[QuestionRecord]
    cost_usd: float
    failed: bool = False
    error_message: str | None = None
    error_phase: RunErrorPhase | None = None
    n_turns_ingested: int = 0
    n_turns_failed_ingestion: int = 0

    @property
    def ingestion_failure_rate(self) -> float:
        """Fraction of conversation turns that never reached the memory provider."""
        attempted = self.n_turns_ingested
        if attempted == 0:
            return 0.0
        return self.n_turns_failed_ingestion / attempted

    @property
    def n_questions_attempted(self) -> int:
        return len(self.question_records)

    @property
    def n_questions_succeeded(self) -> int:
        return _count_correct(self.question_records)

    @property
    def n_questions_errored(self) -> int:
        return _count_errored(self.question_records)

    @property
    def error_rate(self) -> float:
        """Fraction of attempted questions lost to infrastructure errors."""
        attempted = self.n_questions_attempted
        if attempted == 0:
            return 0.0
        return self.n_questions_errored / attempted


@dataclass(frozen=True)
class _RunSuiteResult:
    suite_id: str
    experiment_results: list[ExperimentResult]
    total_cost_usd: float
    # `total_cost_usd` is confirmed, billed spend (== observed). `worst_case_cost_usd` adds the
    # reserved maxima of every ambiguous billed-but-failed hold that the post-hoc recorder never
    # saw. It equals `total_cost_usd` on the happy path (no residual) and is only worth surfacing
    # when it differs.
    worst_case_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        # A default of exactly total_cost_usd would need the value at declaration time; instead
        # normalise a not-yet-set (0.0) or below-observed residual up to the observed floor, so a
        # consumer can always treat worst_case as >= total without special-casing.
        if self.worst_case_cost_usd < self.total_cost_usd:
            object.__setattr__(self, "worst_case_cost_usd", self.total_cost_usd)


@dataclass(frozen=True)
class _LocomoConstructorArgs:
    dataset_path: Path | None = None
    audit_path: Path | None = None
    expected_dataset_checksum: str | None = None
    conversations: list[str] | None = None
    categories: list[str] | None = None


class Runner:
    """Orchestrates configured experiments and runs."""

    def __init__(
        self,
        config: ExperimentSuiteConfig,
        repository: RunRepository,
        *,
        resume_run_id: str | None = None,
        allow_unidentifiable_build: bool = False,
    ) -> None:
        validate_suite_methodology_profile(config)
        self.config = config
        self.repository = repository
        # Relaxes ONLY the build-identity guard in run(), and only on a fresh run: a deliberate
        # decision to spend on a knowingly un-publishable build. Not a general "skip guards" switch,
        # and resume's unconditional dirty-refusal (in the constructor, below) still wins over it.
        self._allow_unidentifiable_build = allow_unidentifiable_build
        self._budget_cap_usd = resolve_budget_cap_usd(config.max_cost_usd)
        self._framework_version = resolve_framework_version()
        self._methodology_profile = get_runtime_profile(config.methodology_profile)
        self._methodology_fingerprint = methodology_fingerprint(self._methodology_profile)
        self._generator_prompt_template = self._methodology_profile.generator_prompt_path.read_text(
            encoding="utf-8"
        )
        self._config_digest = suite_config_digest(config)
        # One request pacer for the whole suite invocation, not per run: a rolling-window limit
        # crosses the boundary between two sequential runs, so a per-run limiter would over-admit at
        # the next run's start. None when no rate_limits are configured -- the pacer stays inert.
        self._rate_limiter = build_rate_limiter(config.rate_limits)
        self._rate_limit_scope = (
            config.rate_limits.scope if config.rate_limits is not None else None
        )
        # Loaded here rather than at first use: every guard that could refuse the resume runs
        # before any provider is spawned or any model is reached, so a refusal costs nothing.
        self._resume = (
            load_resume_state(run_id=resume_run_id, config=config, repository=repository)
            if resume_run_id is not None
            else None
        )
        # One money ledger for the whole suite invocation, injected into every paid model exactly as
        # the rate limiter is. It bounds worst-case spend by refusing a reservation before dispatch
        # once `committed + outstanding + next_max` would breach the cap, so kept holds and settled
        # spend accumulate cumulatively across runs -- the same cumulative `suite_spend_usd`
        # enforces at the run boundary. Any prior-session spend is seeded per run as its tracker is
        # built (`seed_prior_spend`), for normal, resumed, and force-recovered runs; the cap is
        # read live through the provider so recover-blocked's re-resolved `--max-cost-usd` is
        # honoured without rebuilding the ledger.
        self._budget_ledger = BudgetLedger(cap_provider=lambda: self._budget_cap_usd)
        _warn_on_unmetered_spend(config, cap_usd=self._budget_cap_usd)

    async def run(self) -> _RunSuiteResult:
        # Fail-closed at the execution boundary: no entry point -- CLI, script, test, or a future
        # caller -- may start paid work on a build no commit can reproduce. This checks the exact
        # value stamped into every lifecycle event (self._framework_version, resolved once in
        # __init__), so "guard passed" is equivalent to "the stamped version is identifiable". A
        # dirty resume is already refused in the constructor (resume._require_resumable), before
        # this point, so `self._resume is None` honours the override only on a fresh run.
        # Raising here -- before the first durable write (SuiteStartedEvent) -- guarantees a refused
        # start writes nothing and spends nothing.
        if is_unidentifiable_build(self._framework_version) and not (
            self._allow_unidentifiable_build and self._resume is None
        ):
            raise ConfigurationError(
                "Refusing to start a run on an unidentifiable build: the code that produced it "
                "cannot be pinned to a commit, so the result could not state what measured it. "
                "Commit or stash every change and run from a clean tree, or pass "
                "allow_unidentifiable_build for a deliberate, knowingly un-publishable run.",
                framework_version=self._framework_version,
            )
        suite_id = generate_ulid()
        suite_sequence = 0
        experiment_results: list[ExperimentResult] = []
        n_experiments_failed = 0
        # Suite-level spend for budget enforcement. It intentionally includes the
        # cost of failed runs: the budget cap bounds real money spent, not just
        # the cost of runs that produced aggregable results.
        suite_spend_usd = 0.0

        await self.repository.append_suite_event(
            SuiteStartedEvent(
                event_id=generate_ulid(),
                timestamp=now_utc(),
                suite_id=suite_id,
                sequence_number=suite_sequence,
                config_yaml_path="",
                # Redacted: env-var references are resolved before the runner sees the
                # config, so dumping it verbatim writes live API keys into artifacts.
                config_yaml_content=json.dumps(redact_secrets(self.config.model_dump(mode="json"))),
                methodology_version=self.config.methodology_version,
                methodology_profile=self.config.methodology_profile,
                framework_version=self._framework_version,
                n_experiments_planned=len(self.config.experiments),
                runtime_environment={
                    **_capture_runtime_environment(),
                    **_methodology_runtime_environment(
                        self._methodology_profile,
                        self._methodology_fingerprint,
                        config_digest=self._config_digest,
                    ),
                },
            )
        )
        suite_sequence += 1

        try:
            for experiment in self.config.experiments:
                experiment_id = generate_ulid()
                await self.repository.append_suite_event(
                    ExperimentStartedEvent(
                        event_id=generate_ulid(),
                        timestamp=now_utc(),
                        suite_id=suite_id,
                        sequence_number=suite_sequence,
                        experiment_id=experiment_id,
                        experiment_name=experiment.name,
                        n_runs_planned=self.config.runs,
                    )
                )
                suite_sequence += 1

                run_results: list[_RunResult] = []
                for run_number in range(self.config.runs):
                    run_result = await self._execute_single_run(
                        resume=(
                            self._resume
                            if self._resume is not None
                            and self._resume.experiment_name == experiment.name
                            and self._resume.run_number == run_number
                            else None
                        ),
                        experiment=experiment,
                        run_number=run_number,
                        suite_id=suite_id,
                        experiment_id=experiment_id,
                        prior_spend_usd=suite_spend_usd,
                    )
                    suite_spend_usd += run_result.cost_usd
                    run_results.append(run_result)
                    # Re-check at the run boundary: a failed run's spend reaches
                    # the suite total without passing a conversation-boundary
                    # check, and no further paid work may start once the cap is
                    # breached. The failed run already has its own terminal
                    # RunFailedEvent; the caller persists SuiteFailedEvent.
                    _ensure_within_budget(
                        cap_usd=self._budget_cap_usd,
                        observed_cost_usd=suite_spend_usd,
                        worst_case_cost_usd=(
                            suite_spend_usd + self._budget_ledger.outstanding_cost_usd()
                        ),
                    )
                # A run that finished is not automatically publishable: questions lost to
                # infrastructure errors score as wrong, so a degraded run drags the
                # aggregate down and misrepresents the system under test. Exclude those
                # above the configured error rate rather than pooling them silently. The
                # evidence stays durable and recomputable: RunCompletedEvent already
                # records attempted and errored counts for every run.
                finished_runs = [run for run in run_results if not run.failed]
                completed_runs: list[_RunResult] = []
                eligibility_by_run: dict[str, tuple[bool, tuple[str, ...], int, int]] = {}
                for run in finished_runs:
                    eligibility = assess_run_eligibility(self.repository, run.run_id)
                    eligibility_by_run[run.run_id] = (
                        eligibility.eligible,
                        eligibility.reasons,
                        eligibility.recovered_error_count,
                        eligibility.forced_recovery_count,
                    )
                    if (
                        eligibility.eligible
                        and run.ingestion_failure_rate <= self.config.max_ingestion_failure_rate
                    ):
                        completed_runs.append(run)
                    elif not eligibility.eligible:
                        _LOGGER.warning(
                            "run_excluded_from_aggregate",
                            reason="eligibility check failed",
                            run_id=run.run_id,
                            reasons=eligibility.reasons,
                            recovered_error_count=eligibility.recovered_error_count,
                            forced_recovery_count=eligibility.forced_recovery_count,
                        )
                for run in finished_runs:
                    if run not in completed_runs:
                        above_ingestion = (
                            run.ingestion_failure_rate > self.config.max_ingestion_failure_rate
                        )
                        _LOGGER.warning(
                            "run_excluded_from_aggregate",
                            reason=(
                                "ingestion failure rate above max_ingestion_failure_rate"
                                if above_ingestion
                                else "eligibility check failed"
                            ),
                            run_id=run.run_id,
                            experiment_name=experiment.name,
                            error_rate=round(run.error_rate, 4),
                            max_run_error_rate=self.config.max_run_error_rate,
                            ingestion_failure_rate=round(run.ingestion_failure_rate, 6),
                            max_ingestion_failure_rate=self.config.max_ingestion_failure_rate,
                            n_questions_errored=run.n_questions_errored,
                            n_questions_attempted=run.n_questions_attempted,
                            n_turns_failed_ingestion=run.n_turns_failed_ingestion,
                            n_turns_ingested=run.n_turns_ingested,
                            eligibility_reasons=eligibility_by_run[run.run_id][1],
                            recovered_error_count=eligibility_by_run[run.run_id][2],
                            forced_recovery_count=eligibility_by_run[run.run_id][3],
                        )

                # A multi-run aggregate must not silently drop below its PLANNED run count:
                # publishing the surviving runs of a degraded experiment would report a
                # weaker design than the one configured. (A sample SD is defined
                # for any n >= 2; the guard here is fidelity to the planned N, and a single-pass
                # N=1 is admissible as a labelled non-aggregate rather than as an aggregate.)
                min_publishable_runs = min(self.config.runs, _MIN_RUNS_FOR_AGGREGATE)
                n_excluded = len(finished_runs) - len(completed_runs)

                if len(completed_runs) < min_publishable_runs:
                    n_experiments_failed += 1
                    first_failure = _first_failed_run(run_results)
                    # Runs excluded for their error rate are a different outcome from runs
                    # failing outright, and a mixed outcome must report both rather than
                    # letting the first failure hide the exclusions.
                    # Name the guard that actually excluded each run: a run lost to
                    # ingestion can have a zero question-error rate, so reporting
                    # max_run_error_rate for it sends the reader after the wrong threshold.
                    n_excluded_ingestion = sum(
                        1
                        for run in finished_runs
                        if run not in completed_runs
                        and run.ingestion_failure_rate > self.config.max_ingestion_failure_rate
                    )
                    n_excluded_eligibility = n_excluded - n_excluded_ingestion
                    breached: list[str] = []
                    if n_excluded_eligibility:
                        breached.append(f"{n_excluded_eligibility} run(s) failed eligibility")
                    if n_excluded_ingestion:
                        breached.append(
                            f"{n_excluded_ingestion} run(s) exceeded "
                            f"max_ingestion_failure_rate={self.config.max_ingestion_failure_rate}"
                        )
                    no_result_message = (
                        "Experiment produced no publishable aggregate: "
                        f"{len(completed_runs)} of {self.config.runs} run(s) usable, "
                        f"{min_publishable_runs} required; " + "; ".join(breached) + "."
                        if n_excluded
                        else "Experiment completed with no successful runs."
                    )
                    if n_excluded and first_failure is not None and first_failure.error_message:
                        no_result_message = (
                            f"{no_result_message} First run failure: {first_failure.error_message}"
                        )
                        first_failure = None
                    await self.repository.append_suite_event(
                        ExperimentFailedEvent(
                            event_id=generate_ulid(),
                            timestamp=now_utc(),
                            suite_id=suite_id,
                            sequence_number=suite_sequence,
                            experiment_id=experiment_id,
                            error_message=(
                                first_failure.error_message
                                if first_failure is not None
                                else no_result_message
                            )
                            or no_result_message,
                            error_phase=(
                                first_failure.error_phase
                                if first_failure is not None and first_failure.error_phase
                                else "analysis"
                            ),
                        )
                    )
                    suite_sequence += 1
                    continue

                experiment_result = self._aggregate_runs(
                    experiment=experiment,
                    suite_id=suite_id,
                    experiment_id=experiment_id,
                    runs=completed_runs,
                )
                experiment_results.append(experiment_result)
                await self.repository.append_experiment_result(experiment_result)
                await self.repository.append_suite_event(
                    ExperimentCompletedEvent(
                        event_id=generate_ulid(),
                        timestamp=now_utc(),
                        suite_id=suite_id,
                        sequence_number=suite_sequence,
                        experiment_id=experiment_id,
                        overall_standard_pooled_score=_pooled_score(
                            experiment_result.aggregate_overall_standard
                        ),
                        overall_standard_mean=_aggregate_mean(
                            experiment_result.aggregate_overall_standard
                        ),
                        overall_standard_stddev=_aggregate_stddev(
                            experiment_result.aggregate_overall_standard
                        ),
                        overall_audited_pooled_score=_pooled_score(
                            experiment_result.aggregate_overall_audited
                        ),
                        overall_audited_mean=_aggregate_mean(
                            experiment_result.aggregate_overall_audited
                        ),
                        overall_audited_stddev=_aggregate_stddev(
                            experiment_result.aggregate_overall_audited
                        ),
                        cost_usd=experiment_result.total_cost_usd,
                    )
                )
                suite_sequence += 1

            # Report the money actually spent, not the cost of the runs that happened to
            # be publishable: an experiment whose runs all failed or were excluded still
            # paid for model and judge calls, and would otherwise vanish from the total.
            total_cost_usd = suite_spend_usd
            await self.repository.append_suite_event(
                SuiteCompletedEvent(
                    event_id=generate_ulid(),
                    timestamp=now_utc(),
                    suite_id=suite_id,
                    sequence_number=suite_sequence,
                    total_cost_usd=total_cost_usd,
                    n_experiments_completed=len(experiment_results),
                    n_experiments_failed=n_experiments_failed,
                )
            )
            return _RunSuiteResult(
                suite_id=suite_id,
                experiment_results=experiment_results,
                total_cost_usd=total_cost_usd,
                # Confirmed spend plus any ambiguous billed-but-failed residual still outstanding at
                # the end of the suite. Equals total_cost_usd on a clean run (every hold settled).
                worst_case_cost_usd=(total_cost_usd + self._budget_ledger.outstanding_cost_usd()),
            )
        except BaseException as exc:
            # BaseException, not Exception: a cancelled run now records its own terminal
            # event, and catching only Exception here left the suite permanently `running`
            # -- a zombie whose work had in fact finished cleanly. Suites owe an initial and a
            # terminal event, and cancellation is a terminal outcome like any other.
            await self.repository.append_suite_event(
                SuiteFailedEvent(
                    event_id=generate_ulid(),
                    timestamp=now_utc(),
                    suite_id=suite_id,
                    sequence_number=suite_sequence,
                    error_message=str(exc) or repr(exc),
                    n_experiments_completed_before_failure=len(experiment_results),
                )
            )
            raise

    async def _execute_single_run(
        self,
        *,
        experiment: ExperimentConfig,
        run_number: int,
        suite_id: str,
        experiment_id: str,
        prior_spend_usd: float = 0.0,
        resume: ResumeState | None = None,
    ) -> _RunResult:
        run_id = resume.run_id if resume is not None else generate_ulid()
        run_sequence = resume.next_sequence_number if resume is not None else 0
        context = _RunContext(
            run_id=run_id,
            suite_id=suite_id,
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            run_number=run_number,
            # Read back on resume rather than re-derived: the seed is part of what the interrupted
            # run was, and the questions it already asked were asked under it.
            seed=resume.seed if resume is not None else self._seed_for(experiment.name, run_number),
            # Seeded with what this run already spent under this run_id. Left empty, a resumed run
            # reported the cost of its second attempt as the cost of the run -- and the budget cap
            # was measured against that same understated figure, so a resume could also overspend
            # the cap by whatever the first attempt had cost.
            cost_tracker=CostTracker(inherited=resume.prior_api_calls if resume else ()),
        )
        # Seed the shared money ledger with what this run already spent before this session (its
        # inherited api_calls), so the pre-dispatch cap check does not hand a resumed or
        # force-recovered run a fresh budget on top of prior spend. A fresh run inherits nothing, so
        # this adds 0. Done here, from the tracker's authoritative initial total, so every entry
        # point (run, resume, recover-blocked) seeds identically.
        self._budget_ledger.seed_prior_spend(context.cost_tracker.total_cost_usd())
        # Seeded with what the interrupted run already answered. Starting empty would score a
        # resumed run over the half it happened to execute, so a run interrupted at 90% would
        # publish a figure computed from its last 10% -- with nothing in the artifact to say so.
        question_records: list[QuestionRecord] = (
            list(self.repository.get_questions_for_run(resume.run_id)) if resume is not None else []
        )
        current_phase: RunErrorPhase = "initialization"
        run_started_announced = False
        n_turns_ingested = 0
        n_turns_failed_ingestion = 0

        lock_path: Path | None = None

        # Initialize before announcing the run so the recorded provider_version is the
        # real one. Reading it beforehand recorded a placeholder ("…-uninitialized") on
        # every run, leaving artifacts that name no reproducible provider build. A failure
        # here is re-raised after the started event so the started -> failed event order
        # and the "initialization" phase are preserved.
        # One cleanup path covers everything from the first initializer onward. Earlier
        # revisions bolted a separate teardown onto each way out of this region and each
        # time missed one: an `except Exception` around initialization does not catch
        # asyncio.CancelledError, so Ctrl-C after the provider subprocess had spawned
        # leaked it. A single try/finally cannot have that gap.
        provider = self._instantiate_provider(experiment.provider)
        _require_retrieval_budget_is_applicable(self._methodology_profile, provider, experiment)
        benchmark = self._instantiate_benchmark(experiment.benchmark)
        answer_model = self._instantiate_answer_model(experiment.answer_model)
        judge = self._instantiate_judge(experiment.judge)

        try:
            # Acquired inside the block that releases it, and after the instantiations above.
            # Taken before the try, anything they raised stranded a lock whose own refusal message
            # tells the operator to remove it only if certain no run is in flight. Taken after
            # them, a failure there happens before the lock exists, so there is nothing to strand;
            # everything from here is covered by the finally.
            #
            # The instantiations stay outside deliberately: the retrieval-budget refusal must reach
            # the caller as a refusal, not be recorded as a failed run.
            forced_items = [item for item in resume.recovery_work if item.forced] if resume else []
            if resume is not None:
                try:
                    lock_path = acquire_run_lock(Path(self.config.output_dir), resume.run_id)
                except ConfigurationError as exc:
                    # Lock contention on a forced recovery must refuse cleanly,
                    # never write a RunStarted/RunFailed into the very run whose concurrent writer
                    # the lock exists to keep out. Re-raise as a clean abort the terminal handler
                    # re-raises untouched; lock_path stays None, so the finally unlinks nothing.
                    if forced_items:
                        raise _ForcedRecoveryRevalidationAborted(str(exc)) from exc
                    raise

            # Under-lock re-validation for forced recovery. Fires only when
            # the resume state carries forced items, AFTER the lock and BEFORE the first
            # ForcedRecoveryEvent write or any dispatch, so eligibility and build identity are the
            # ones observed under exclusive access. A refusal here is a clean abort (no run
            # announced, no terminal event, no attempt) that the terminal handler re-raises; the
            # finally still releases the lock.
            if forced_items:
                try:
                    revalidate_forced_recovery(
                        run_id=cast(ResumeState, resume).run_id,
                        question_ids=[item.question_id for item in forced_items],
                        reason=next((item.reason for item in forced_items if item.reason), ""),
                        config=self.config,
                        repository=self.repository,
                    )
                except ConfigurationError as exc:
                    raise _ForcedRecoveryRevalidationAborted(str(exc)) from exc

            initialization_error: Exception | None = None
            try:
                await provider.initialize()
                await benchmark.load()
                _require_evaluated_categories_are_present(
                    self._methodology_profile, benchmark, experiment
                )
                await answer_model.initialize()
                await judge.initialize()
            except Exception as exc:
                # Held, not raised: the started event must be recorded first so the
                # started -> failed order and the "initialization" phase survive. A
                # cancellation is deliberately not caught here -- it falls through to the
                # single terminal handler below, which announces the run if it has not been
                # announced yet. Recording the terminal event in two places collided on
                # the run's sequence numbers.
                initialization_error = exc

            # Reserve the started event's sequence number and mark the run announced
            # *before* awaiting the append. `append_run_event` writes the JSONL line and then
            # awaits the SQLite index (`repository.py:105-111`), so a cancellation between
            # those two raises with the line already durable. Setting the flag afterwards let
            # the terminal handler conclude nothing had been announced and write a *second*
            # RunStartedEvent on the same sequence number -- a corrupt stream that breaks
            # index rebuild, which is worse than the zombie the flag exists to prevent.
            # Residual, accepted: a cancellation landing before the JSONL write leaves a
            # terminal event whose started twin is absent, and reconstruction skips it. That
            # is an append dying mid-flight, not a double write.
            started_sequence = run_sequence
            run_sequence += 1
            run_started_announced = True
            if resume is not None:
                # Not a second `run_started`: a run that was resumed is not a run that started
                # twice, and the artifact has to be able to tell a reader which it was. The
                # original started event stays authoritative for the run's identity -- its
                # models, profile and seed -- and the earlier terminal event stays in the stream
                # as the record of what actually happened.
                await self.repository.append_run_event(
                    RunResumedEvent(
                        event_id=generate_ulid(),
                        timestamp=now_utc(),
                        run_id=run_id,
                        sequence_number=started_sequence,
                        n_conversations_inherited=len(resume.completed_conversations),
                        n_questions_inherited=len(resume.completed_questions),
                        inherited_cost_usd=resume.prior_spend_usd,
                        framework_version=self._framework_version,
                    )
                )
            else:
                await self._append_run_started(
                    run_id=run_id,
                    run_sequence=started_sequence,
                    suite_id=suite_id,
                    experiment_id=experiment_id,
                    experiment=experiment,
                    run_number=run_number,
                    provider=provider,
                    benchmark=benchmark,
                    answer_model=answer_model,
                    judge=judge,
                    context=context,
                )

            if initialization_error is not None:
                raise initialization_error

            reconciled_pending: dict[str, RecoveryPending] = {}
            if resume is not None:
                reconciled_pending = {
                    pending.attempt.question_id: pending
                    for pending in await self._reconcile_recovery_pending(run_id)
                    if pending.target is not None
                }

            if resume is None:
                all_questions = tuple(benchmark.get_questions())
                question_ids = tuple(sorted(question.question_id for question in all_questions))
                categories = tuple(sorted({question.category.value for question in all_questions}))
                await self.repository.append_question_plan(
                    QuestionPlanRecord(
                        run_id=run_id,
                        benchmark_id=benchmark.benchmark_type,
                        categories=categories,
                        corpus_checksum=benchmark.dataset_checksum,
                        question_ids=question_ids,
                        fingerprint=question_plan_fingerprint(
                            benchmark_id=benchmark.benchmark_type,
                            categories=categories,
                            corpus_checksum=benchmark.dataset_checksum,
                            question_ids=question_ids,
                        ),
                        timestamp=now_utc(),
                    )
                )

            # Starts at the inherited spend, not at zero: these deltas are per-conversation costs,
            # and the first conversation of a resumed run did not cost everything the run had
            # spent before it started.
            previous_cost = context.cost_tracker.total_cost_usd()
            recovery_by_question = (
                {item.question_id: item for item in resume.recovery_work} if resume else {}
            )
            for conversation in benchmark.get_conversations():
                if (
                    resume is not None
                    and conversation.conversation_id in resume.completed_conversations
                    and not any(
                        question.question_id in recovery_by_question
                        or question.question_id in reconciled_pending
                        for question in benchmark.get_questions(
                            conversation_id=conversation.conversation_id
                        )
                    )
                ):
                    # Its questions are all answered, so its memories are dead weight: the runner
                    # resets the store at every conversation boundary anyway. Skipping the whole
                    # conversation is what keeps a resume to one re-ingestion, never seven.
                    _LOGGER.info(
                        "conversation_skipped_on_resume",
                        run_id=run_id,
                        conversation_id=conversation.conversation_id,
                    )
                    continue
                questions = benchmark.get_questions(conversation_id=conversation.conversation_id)
                question_ids = {question.question_id for question in questions}
                has_retrieve_recovery = any(
                    recovery_by_question.get(question.question_id, None) is not None
                    and recovery_by_question[question.question_id].stage == "retrieve"
                    for question in questions
                ) or any(
                    isinstance(pending.target, RetrievalRecord)
                    for pending in reconciled_pending.values()
                    if pending.attempt.question_id in question_ids
                )
                if not has_retrieve_recovery:
                    current_phase = "ingest"
                    await provider.reset()
                    ingestion = await ingest_conversation(
                        conversation, provider, self.repository, context
                    )
                    # Turns lost here mean the memory store no longer matches the corpus the
                    # methodology specifies. Previously these stats were discarded, so a run
                    # could publish a score over a conversation it had only partly ingested.
                    n_turns_ingested += ingestion.record.n_turns
                    n_turns_failed_ingestion += ingestion.n_turns_failed

                    # Stop as soon as the run can no longer be published, rather than at the end.
                    # Left unchecked, a single early ingestion loss lets a run keep paying for every
                    # remaining question and its judge before an aggregate guard discards the whole
                    # run anyway -- real money spent on a run already known to be unusable at the
                    # moment the turn was lost.
                    if _ingestion_is_beyond_repair(
                        n_turns_failed_ingestion, n_turns_ingested, experiment_or_suite=self.config
                    ):
                        raise ProviderError(
                            "Ingestion losses have already put this run beyond the publishable "
                            "threshold, so it is abandoned before any further questions are paid "
                            "for.",
                            run_id=run_id,
                            conversation_id=conversation.conversation_id,
                            n_turns_ingested=n_turns_ingested,
                            n_turns_failed_ingestion=n_turns_failed_ingestion,
                            max_ingestion_failure_rate=self.config.max_ingestion_failure_rate,
                        )

                current_phase = "retrieve"
                recovered_records: list[QuestionRecord] = []
                if resume is not None:
                    for question in questions:
                        pending = reconciled_pending.get(question.question_id)
                        if pending is not None:
                            recovered_records.append(
                                await self._continue_reconciled_recovery(
                                    question=question,
                                    pending=pending,
                                    answer_model=answer_model,
                                    judge=judge,
                                    experiment=experiment,
                                    context=context,
                                )
                            )
                            continue
                        work = recovery_by_question.get(question.question_id)
                        if work is None:
                            continue
                        if work.forced:
                            # A forced recovery that itself fails must not fail the whole run: the
                            # override was authorized for one blocked question, and a stage that
                            # raises again leaves that question errored (verdict ERROR) while the
                            # run re-emits status="partial" and stays ineligible -- the correct
                            # outcome. The stage error is already persisted
                            # durably by the pipeline before it raises, and the ForcedRecoveryEvent
                            # authorizing the attempt is still on disk. The automatic (non-forced)
                            # path keeps its existing propagate-and-fail behaviour; nothing else
                            # changes.
                            try:
                                recovered_records.append(
                                    await self._recover_question_stage(
                                        question=question,
                                        work=work,
                                        provider=provider,
                                        answer_model=answer_model,
                                        judge=judge,
                                        conversation=conversation,
                                        experiment=experiment,
                                        context=context,
                                    )
                                )
                            except (ProviderError, ModelError, JudgeError):
                                recovered_records.append(
                                    await asyncio.to_thread(
                                        self.repository.get_question_record,
                                        run_id,
                                        question.question_id,
                                    )
                                )
                            continue
                        recovered_records.append(
                            await self._recover_question_stage(
                                question=question,
                                work=work,
                                provider=provider,
                                answer_model=answer_model,
                                judge=judge,
                                conversation=conversation,
                                experiment=experiment,
                                context=context,
                            )
                        )
                if resume is not None:
                    # Skipped, never retried: the question index is unique on (run_id,
                    # question_id), so re-asking one would collide rather than overwrite.
                    questions = [
                        question
                        for question in questions
                        if question.question_id not in resume.completed_questions
                        and question.question_id not in recovery_by_question
                        and question.question_id not in reconciled_pending
                        and question.question_id not in resume.blocked_questions
                    ]
                conversation_records = await self._evaluate_questions_for_conversation(
                    questions=questions,
                    provider=provider,
                    answer_model=answer_model,
                    judge=judge,
                    experiment=experiment,
                    context=context,
                )
                # Replace, do not duplicate. `question_records` was seeded with every question the
                # interrupted run had a record for -- including the blocked ones now being
                # recovered, which the repository still projects with their pre-recovery ERROR
                # verdict. Appending the fresh recovered record without dropping that stale twin
                # double-counts the question: the terminal RunCompletedEvent would then read the
                # stale ERROR and report status="partial" even for a fully successful recovery,
                # leaving a repaired run ineligible (planned == judgments yet lifecycle_status !=
                # "completed"). Dropping the stale entry for every re-fetched question keeps one
                # record per question, so a forced (or organic) recovery that reaches a verdict
                # makes the run eligible -- the whole point of this design.
                refreshed_ids = {record.question_id for record in recovered_records}
                question_records = [
                    record for record in question_records if record.question_id not in refreshed_ids
                ]
                question_records.extend([*recovered_records, *conversation_records])

                current_cost = context.cost_tracker.total_cost_usd()
                event_cost = current_cost - previous_cost
                previous_cost = current_cost
                await self.repository.append_run_event(
                    ConversationProcessedEvent(
                        event_id=generate_ulid(),
                        timestamp=now_utc(),
                        run_id=run_id,
                        sequence_number=run_sequence,
                        conversation_id=conversation.conversation_id,
                        n_questions_evaluated=len(conversation_records),
                        n_questions_correct=_count_correct(conversation_records),
                        n_questions_errored=_count_errored(conversation_records),
                        cost_usd=event_cost,
                    )
                )
                run_sequence += 1

                # Budget enforcement happens at conversation boundaries so the
                # current conversation's records and lifecycle event are already
                # durable before the run is halted.
                _ensure_within_budget(
                    cap_usd=self._budget_cap_usd,
                    observed_cost_usd=prior_spend_usd + current_cost,
                    worst_case_cost_usd=(
                        prior_spend_usd + current_cost + self._budget_ledger.outstanding_cost_usd()
                    ),
                )

            current_phase = "analysis"
            scores = _compute_scores(
                question_records,
                scored_categories=self._methodology_profile.scored_categories,
            )
            status: Literal["completed", "partial"] = (
                "partial" if _count_errored(question_records) else "completed"
            )
            run_result = _RunResult(
                run_id=run_id,
                suite_id=suite_id,
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                run_number=run_number,
                scores=scores,
                question_records=question_records,
                cost_usd=context.cost_tracker.total_cost_usd(),
                n_turns_ingested=n_turns_ingested,
                n_turns_failed_ingestion=n_turns_failed_ingestion,
            )
            await self.repository.append_run_event(
                RunCompletedEvent(
                    event_id=generate_ulid(),
                    timestamp=now_utc(),
                    run_id=run_id,
                    sequence_number=run_sequence,
                    status=status,
                    n_questions_attempted=run_result.n_questions_attempted,
                    n_questions_succeeded=run_result.n_questions_succeeded,
                    n_questions_errored=run_result.n_questions_errored,
                    overall_score_standard=scores.overall_standard,
                    overall_score_audited=scores.overall_audited,
                    by_category_standard=_string_score_map(scores.by_category_standard),
                    by_category_audited=_string_score_map(scores.by_category_audited),
                    total_cost_usd=run_result.cost_usd,
                )
            )
            return run_result
        except _ForcedRecoveryRevalidationAborted:
            # A lost race between load and lock: the forced recovery no longer qualifies under the
            # lock. Refused before any write or announcement, so this is a clean abort -- no
            # RunStartedEvent, no RunFailedEvent, no dispatch -- re-raised to the caller. The
            # finally below still releases the lock.
            raise
        except BaseException as exc:
            # BaseException, not Exception: cancellation (Ctrl-C, task shutdown) is not an
            # Exception, so it escaped here and left the run announced but never terminated
            # -- a permanent `running` zombie. Every exit from this region is terminal; only
            # the disposition differs, so the terminal event is recorded first and the
            # disposition decided after.
            if not run_started_announced:
                # Cancelled before the run was announced. Terminating a run_id that has no
                # started event would leave it unsurfaceable -- the suite already announced
                # the experiment -- so announce it, then terminate it.
                await self._append_run_started(
                    run_id=run_id,
                    run_sequence=run_sequence,
                    suite_id=suite_id,
                    experiment_id=experiment_id,
                    experiment=experiment,
                    run_number=run_number,
                    provider=provider,
                    benchmark=benchmark,
                    answer_model=answer_model,
                    judge=judge,
                    context=context,
                )
                run_sequence += 1
            if current_phase in _RUN_ERROR_PHASES:
                error_phase = current_phase
            elif isinstance(exc, Exception):
                error_phase = _classify_run_error_phase(exc)
            else:
                error_phase = "initialization"
            try:
                await self.repository.append_run_event(
                    RunFailedEvent(
                        event_id=generate_ulid(),
                        timestamp=now_utc(),
                        run_id=run_id,
                        sequence_number=run_sequence,
                        # A cancellation's str() is empty, and naming the phase it interrupted
                        # is what makes the artifact readable afterwards.
                        error_message=(
                            str(exc)
                            if isinstance(exc, Exception)
                            else f"Run cancelled during {current_phase}: {exc!r}"
                        ),
                        error_phase=error_phase,
                        n_questions_completed_before_failure=len(question_records),
                        partial_cost_usd=context.cost_tracker.total_cost_usd(),
                    )
                )
            except Exception as append_error:
                # The run's own failure is the one worth diagnosing. Letting a persistence
                # error escape from here replaced it in the traceback, so the operator saw a
                # disk problem and never learned what the run had actually hit. Record both
                # and carry on with the original error's disposition; the stream is left
                # non-terminal, which is a persistence failure the repository surfaces.
                _LOGGER.error(
                    "run_terminal_event_not_persisted",
                    run_id=run_id,
                    original_error=str(exc) or repr(exc),
                    original_error_phase=error_phase,
                    persistence_error=str(append_error),
                )
            if isinstance(exc, CostExceededError) or not isinstance(exc, Exception):
                # A budget halt must stop the entire suite, not just this run.
                # Re-raise after the run's terminal event is durable; the caller
                # persists SuiteFailedEvent and surfaces the error to the CLI.
                # Cancellation propagates for the same reason: it is not this run's
                # failure to absorb and report as a result.
                raise
            return _RunResult(
                run_id=run_id,
                suite_id=suite_id,
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                run_number=run_number,
                scores=RunScores(),
                question_records=question_records,
                cost_usd=context.cost_tracker.total_cost_usd(),
                failed=True,
                error_message=str(exc),
                error_phase=error_phase,
            )
        finally:
            await _close_plugins(provider, answer_model, judge)
            if lock_path is not None:
                # Released whatever happened, including a crash: a lock left behind by a dead
                # process blocks the very resume it exists to make safe.
                lock_path.unlink(missing_ok=True)

    async def recover_blocked(
        self, forced_state: ResumeState, *, max_cost_usd: float | None = None
    ) -> RecoverBlockedResult:
        """Force-recover the named blocked questions of an existing run, reusing the resume engine.

        Resolves the run's identity from its RunStartedEvent, finds the matching ExperimentConfig,
        and delegates to the existing ``_execute_single_run`` with ``resume=forced_state`` and
        ``prior_spend_usd=0.0``. Build-identity protection comes from two
        structural gates -- the load-time ``_require_resumable`` already applied in the loader, and
        the under-lock re-validation inside ``_execute_single_run`` -- NOT from ``run()``'s inline
        guard, which this path deliberately never passes through.
        """
        run_id = forced_state.run_id
        started = self.repository.get_run_started_event(run_id)
        experiment = next(
            (item for item in self.config.experiments if item.name == started.experiment_name),
            None,
        )
        if experiment is None:
            raise ConfigurationError(
                "The configuration contains no experiment matching the run being recovered.",
                run_id=run_id,
                recorded_experiment=started.experiment_name,
            )
        forced_ids = {item.forced_recovery_id for item in forced_state.recovery_work if item.forced}
        question_ids = tuple(item.question_id for item in forced_state.recovery_work if item.forced)
        # --max-cost-usd is a post-hoc ceiling for this invocation: threaded into the same budget
        # cap _execute_single_run's conversation-boundary check already honours.
        if max_cost_usd is not None:
            self._budget_cap_usd = resolve_budget_cap_usd(max_cost_usd)

        result = await self._execute_single_run(
            experiment=experiment,
            run_number=forced_state.run_number,
            suite_id=started.suite_id,
            experiment_id=started.experiment_id,
            prior_spend_usd=0.0,
            resume=forced_state,
        )

        after = self.repository.get_strict_stage_stream(run_id)
        answered_after = {judgment.question_id for judgment in after.judgments}
        recovered = tuple(qid for qid in question_ids if qid in answered_after)
        unrecovered = tuple(qid for qid in question_ids if qid not in answered_after)
        forced_written = sum(
            1 for forced in after.forced_recoveries if forced.forced_recovery_id in forced_ids
        )
        return RecoverBlockedResult(
            run_id=run_id,
            recovered_question_ids=recovered,
            unrecovered_question_ids=unrecovered,
            forced_recovery_count=forced_written,
            incremental_cost_usd=result.cost_usd - forced_state.prior_spend_usd,
        )

    async def _reconcile_recovery_pending(self, run_id: str) -> tuple[RecoveryPending, ...]:
        """Close only crash-after-stage windows; a second pass observes no pending work."""
        snapshot = self.repository.get_strict_stage_stream(run_id)
        for pending in snapshot.recovery_pending:
            if pending.target is None:
                continue
            target = pending.target
            if isinstance(target, RetrievalRecord):
                target_id = target.retrieval_id
            elif isinstance(target, Response):
                target_id = target.response_id
            else:
                target_id = target.judgment_id
            await self.repository.append_error_resolution(
                ErrorResolutionRecord(
                    resolution_id=generate_ulid(),
                    run_id=run_id,
                    question_id=pending.attempt.question_id,
                    error_id=pending.error.error_id,
                    recovery_attempt_id=pending.attempt.attempt_id,
                    resolved_by_stage=pending.attempt.stage,
                    resolved_by_stage_record_id=target_id,
                    timestamp=now_utc(),
                )
            )
        return snapshot.recovery_pending

    async def _continue_reconciled_recovery(
        self,
        *,
        question: Question,
        pending: RecoveryPending,
        answer_model: AnswerModel,
        judge: Judge,
        experiment: ExperimentConfig,
        context: _RunContext,
    ) -> QuestionRecord:
        """Finish only the downstream work left after a crash-before-resolution window."""
        target = pending.target
        if isinstance(target, RetrievalRecord):
            response = await generate_answer(
                question,
                target,
                answer_model,
                self.repository,
                context,
                max_output_tokens=experiment.answer_model.max_output_tokens,
                temperature=experiment.answer_model.temperature,
                prompt_template=self._generator_prompt_template,
                answer_marker=self._methodology_profile.answer_marker,
            )
            await judge_response(response, question, judge, self.repository, context)
        elif isinstance(target, Response):
            await judge_response(target, question, judge, self.repository, context)
        return await asyncio.to_thread(
            self.repository.get_question_record, context.run_id, question.question_id
        )

    async def _recover_question_stage(
        self,
        *,
        question: Question,
        work: RecoveryWorkItem,
        provider: MemoryProvider,
        answer_model: AnswerModel,
        judge: Judge,
        conversation: Conversation,
        experiment: ExperimentConfig,
        context: _RunContext,
    ) -> QuestionRecord:
        """Repair one strict-snapshot-selected stage without creating another evaluation."""
        snapshot = self.repository.get_strict_stage_stream(context.run_id)
        evaluation = next(
            item for item in snapshot.evaluations if item.question_id == question.question_id
        )
        retrieval = next(
            (item for item in snapshot.retrievals if item.question_id == question.question_id), None
        )
        response = next(
            (item for item in snapshot.responses if item.question_id == question.question_id), None
        )
        # Mint the id ONLY here; the timestamp is stamped later, deliberately. A forced recovery
        # must persist its ForcedRecoveryEvent BEFORE the attempt's timestamp is taken, so that
        # forced.timestamp <= attempt.timestamp holds by construction. The forced authorization
        # still lands on disk before the attempt, keeping the inverse invariant satisfiable at
        # every intermediate index read, AND its now_utc() is taken first, keeping the timestamp
        # chain error <= forced <= attempt monotone -- which the strict reader's precedence check
        # enforces and would otherwise reject every legitimate forced pair written in the old
        # attempt-first order.
        attempt_id = generate_ulid()
        if work.forced:
            await self.repository.append_forced_recovery(
                ForcedRecoveryEvent(
                    forced_recovery_id=cast(str, work.forced_recovery_id),
                    run_id=context.run_id,
                    question_id=question.question_id,
                    error_id=work.error_id,
                    attempt_id=attempt_id,
                    stage=work.stage,
                    reason=cast(str, work.reason),
                    timestamp=now_utc(),  # t_forced
                )
            )
        attempt = RecoveryAttemptRecord(
            attempt_id=attempt_id,
            run_id=context.run_id,
            question_id=question.question_id,
            error_id=work.error_id,
            stage=work.stage,
            timestamp=now_utc(),  # t_attempt >= t_forced
        )
        await self.repository.append_recovery_attempt(attempt)
        if work.stage == "retrieve":
            await provider.reset()
            ingestion = await ingest_conversation(
                conversation,
                provider,
                self.repository,
                context,
                ingestion_attempt_id=generate_ulid(),
            )
            retrieval = await retrieve_for_question(
                question,
                evaluation,
                provider,
                experiment.top_k_retrieval,
                self.repository,
                context,
                recovery_attempt_id=attempt.attempt_id,
                ingestion_attempt_id=ingestion.record.ingestion_attempt_id,
            )
            target_id = retrieval.retrieval_id
        elif work.stage == "generate":
            if retrieval is None:
                raise ConfigurationError(
                    "Recovery generate stage has no durable retrieval.",
                    question_id=question.question_id,
                )
            response = await generate_answer(
                question,
                retrieval,
                answer_model,
                self.repository,
                context,
                max_output_tokens=experiment.answer_model.max_output_tokens,
                temperature=experiment.answer_model.temperature,
                prompt_template=self._generator_prompt_template,
                answer_marker=self._methodology_profile.answer_marker,
                recovery_attempt_id=attempt.attempt_id,
            )
            target_id = response.response_id
        else:
            if response is None:
                raise ConfigurationError(
                    "Recovery judge stage has no durable response.",
                    question_id=question.question_id,
                )
            judgment = await judge_response(
                response,
                question,
                judge,
                self.repository,
                context,
                recovery_attempt_id=attempt.attempt_id,
            )
            target_id = judgment.judgment_id
        await self.repository.append_error_resolution(
            ErrorResolutionRecord(
                resolution_id=generate_ulid(),
                run_id=context.run_id,
                question_id=question.question_id,
                error_id=work.error_id,
                recovery_attempt_id=attempt.attempt_id,
                resolved_by_stage=work.stage,
                resolved_by_stage_record_id=target_id,
                timestamp=now_utc(),
            )
        )
        if work.stage == "retrieve":
            if retrieval is None:
                raise AssertionError("retrieve recovery did not create retrieval")
            response = await generate_answer(
                question,
                retrieval,
                answer_model,
                self.repository,
                context,
                max_output_tokens=experiment.answer_model.max_output_tokens,
                temperature=experiment.answer_model.temperature,
                prompt_template=self._generator_prompt_template,
                answer_marker=self._methodology_profile.answer_marker,
            )
        if work.stage in {"retrieve", "generate"}:
            if response is None:
                raise AssertionError("recovery did not create response")
            await judge_response(response, question, judge, self.repository, context)
        return await asyncio.to_thread(
            self.repository.get_question_record, context.run_id, question.question_id
        )

    async def _append_run_started(
        self,
        *,
        run_id: str,
        run_sequence: int,
        suite_id: str,
        experiment_id: str,
        experiment: ExperimentConfig,
        run_number: int,
        provider: MemoryProvider,
        benchmark: Benchmark,
        answer_model: AnswerModel,
        judge: Judge,
        context: _RunContext,
    ) -> None:
        """Record the run's existence and the identity of everything it will use.

        Extracted because a cancelled initialization must emit this too: recording the run
        only on the success path would leave a cancelled run_id with no lifecycle stream.
        """
        await self.repository.append_run_event(
            RunStartedEvent(
                event_id=generate_ulid(),
                timestamp=now_utc(),
                run_id=run_id,
                sequence_number=run_sequence,
                suite_id=suite_id,
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                run_number=run_number,
                provider_type=provider.provider_type,
                provider_version=provider.provider_version,
                benchmark_type=benchmark.benchmark_type,
                benchmark_version=benchmark.benchmark_version,
                benchmark_checksum=benchmark.dataset_checksum,
                answer_model_id=answer_model.model_id,
                answer_model_vendor=answer_model.vendor,
                judge_model_id=judge.model_id,
                judge_model_vendor=judge.vendor,
                config=redact_secrets(experiment.model_dump(mode="json")),
                methodology_version=self.config.methodology_version,
                methodology_profile=self.config.methodology_profile,
                framework_version=self._framework_version,
                seed=context.seed,
                runtime_environment={
                    **_capture_runtime_environment(),
                    **_methodology_runtime_environment(
                        self._methodology_profile,
                        self._methodology_fingerprint,
                        config_digest=self._config_digest,
                        provider_honours_top_k=provider.honours_top_k,
                        corpus_conversations_evaluated=len(benchmark.get_conversations()),
                        corpus_conversations_available=benchmark.corpus_conversation_count,
                    ),
                },
            )
        )

    async def _evaluate_questions_for_conversation(
        self,
        *,
        questions: Sequence[Question],
        provider: MemoryProvider,
        answer_model: AnswerModel,
        judge: Judge,
        experiment: ExperimentConfig,
        context: _RunContext,
    ) -> list[QuestionRecord]:
        semaphore = asyncio.Semaphore(experiment.max_concurrent_questions)

        async def process(question: Question) -> QuestionRecord:
            async with semaphore:
                return await self._evaluate_question(
                    question=question,
                    provider=provider,
                    answer_model=answer_model,
                    judge=judge,
                    top_k=experiment.top_k_retrieval,
                    max_output_tokens=experiment.answer_model.max_output_tokens,
                    temperature=experiment.answer_model.temperature,
                    context=context,
                )

        # Not a bare asyncio.gather: a budget refusal in one question must cancel the sibling
        # questions before it propagates, or they would keep dispatching paid calls while the run is
        # already halting on cost.
        tasks = [asyncio.create_task(process(question)) for question in questions]
        return await _gather_or_cancel_siblings(tasks)

    async def _evaluate_question(
        self,
        *,
        question: Question,
        provider: MemoryProvider,
        answer_model: AnswerModel,
        judge: Judge,
        top_k: int,
        max_output_tokens: int,
        temperature: float,
        context: _RunContext,
    ) -> QuestionRecord:
        question_evaluation = QuestionEvaluationRecord(
            question_evaluation_id=generate_ulid(),
            run_id=context.run_id,
            question_id=question.question_id,
            conversation_id=question.conversation_id,
            category=question.category,
            question_text=question.question_text,
            expected_answer=question.expected_answer,
            is_audited_error=question.is_audited_error,
            timestamp=now_utc(),
        )
        await self.repository.append_question_evaluation(question_evaluation)

        try:
            retrieval = await retrieve_for_question(
                question,
                question_evaluation,
                provider,
                top_k,
                self.repository,
                context,
            )
            response = await generate_answer(
                question,
                retrieval,
                answer_model,
                self.repository,
                context,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                prompt_template=self._generator_prompt_template,
                answer_marker=self._methodology_profile.answer_marker,
            )
            await judge_response(response, question, judge, self.repository, context)
        except (ProviderError, ModelError, JudgeError):
            pass

        return await asyncio.to_thread(
            self.repository.get_question_record,
            context.run_id,
            question.question_id,
        )

    def _aggregate_runs(
        self,
        *,
        experiment: ExperimentConfig,
        suite_id: str,
        experiment_id: str,
        runs: Sequence[_RunResult],
    ) -> ExperimentResult:
        """Aggregate scores and cost over the published runs.

        Cost stays scoped to ``runs`` so the result is internally consistent: its cost,
        ``n_runs`` and ``run_ids`` all describe the same sample. Money spent on runs that
        failed or were excluded is not lost from reporting -- the suite total is taken
        from actual accumulated spend, which covers every run executed.
        """
        aggregate_overall_standard = _aggregate_score(
            experiment_id=experiment_id,
            category="overall",
            mode="standard",
            scores=[
                run.scores.overall_standard
                for run in runs
                if run.scores.overall_standard is not None
            ],
        )
        aggregate_overall_audited = _aggregate_score(
            experiment_id=experiment_id,
            category="overall",
            mode="audited",
            scores=[
                run.scores.overall_audited for run in runs if run.scores.overall_audited is not None
            ],
        )

        categories = sorted(
            {
                *[category for run in runs for category in run.scores.by_category_standard.keys()],
                *[category for run in runs for category in run.scores.by_category_audited.keys()],
            },
            key=lambda category: category.value,
        )
        aggregate_by_category_standard: dict[str, AggregateScore] = {}
        aggregate_by_category_audited: dict[str, AggregateScore] = {}
        for category in categories:
            standard = _aggregate_score(
                experiment_id=experiment_id,
                category=category,
                mode="standard",
                scores=[
                    run.scores.by_category_standard[category]
                    for run in runs
                    if category in run.scores.by_category_standard
                ],
            )
            if standard is not None:
                aggregate_by_category_standard[category.value] = standard
            audited = _aggregate_score(
                experiment_id=experiment_id,
                category=category,
                mode="audited",
                scores=[
                    run.scores.by_category_audited[category]
                    for run in runs
                    if category in run.scores.by_category_audited
                ],
            )
            if audited is not None:
                aggregate_by_category_audited[category.value] = audited

        return ExperimentResult(
            experiment_id=experiment_id,
            suite_id=suite_id,
            experiment_name=experiment.name,
            n_runs=len(runs),
            run_ids=[run.run_id for run in runs],
            aggregate_overall_standard=aggregate_overall_standard,
            aggregate_overall_audited=aggregate_overall_audited,
            aggregate_by_category_standard=aggregate_by_category_standard,
            aggregate_by_category_audited=aggregate_by_category_audited,
            total_cost_usd=sum(run.cost_usd for run in runs),
            config=ExperimentConfig.model_validate(
                redact_secrets(experiment.model_dump(mode="json"))
            ),
        )

    def _instantiate_provider(self, config: ProviderConfig) -> MemoryProvider:
        cls = get_provider_class(config.type)
        instance = cast(Any, cls)(_provider_constructor_config(cls, config))
        if not isinstance(instance, MemoryProvider):
            raise ConfigurationError(
                "Registered provider does not implement MemoryProvider.",
                provider_type=config.type,
            )
        return instance

    def _instantiate_benchmark(self, config: BenchmarkConfig) -> Benchmark:
        cls = get_benchmark_class(config.type)
        if cls is LocomoBenchmark:
            args = _locomo_constructor_args(config.config)
            instance = LocomoBenchmark(
                dataset_path=args.dataset_path,
                audit_path=args.audit_path,
                expected_dataset_checksum=(
                    args.expected_dataset_checksum
                    if args.expected_dataset_checksum is not None
                    else EXPECTED_DATASET_SHA256
                ),
                conversations=args.conversations,
                categories=args.categories,
                # From the profile, never from the experiment config: it decides what the corpus
                # contains, and a corpus a config could vary silently is the unpinned-parameter
                # defect this methodology version exists to remove.
                include_image_descriptions=(
                    self._methodology_profile.image_description_policy != "none"
                ),
            )
        else:
            instance = cast(Any, cls)(config.config)
        if not isinstance(instance, Benchmark):
            raise ConfigurationError(
                "Registered benchmark does not implement Benchmark.",
                benchmark_type=config.type,
            )
        return instance

    def _instantiate_answer_model(self, config: AnswerModelConfig) -> AnswerModel:
        cls = get_model_class(config.type)
        raw_config = {**config.config, "model_id": config.model}
        # The suite's one shared pacer reaches every paid call: the answer model here and, below,
        # the model each judge wraps. A custom adapter (the else branch) is left unpaced -- it does
        # not take the parameters, and the pacer is opt-in. Typed dict[str, Any] so unpacking it as
        # keyword arguments does not collapse the two value types into a union the checker rejects.
        paced: dict[str, Any] = {
            "rate_limiter": self._rate_limiter,
            "rate_limit_scope": self._rate_limit_scope,
            "budget": self._budget_ledger,
        }
        if cls is OpenAIModel:
            instance = OpenAIModel(OpenAIModelConfig.model_validate(raw_config), **paced)
        elif cls is AnthropicModel:
            instance = AnthropicModel(AnthropicModelConfig.model_validate(raw_config), **paced)
        elif cls is GoogleModel:
            instance = GoogleModel(GoogleModelConfig.model_validate(raw_config), **paced)
        else:
            instance = cast(Any, cls)(raw_config)
        if not isinstance(instance, AnswerModel):
            raise ConfigurationError(
                "Registered answer model does not implement AnswerModel.",
                model_type=config.type,
            )
        return instance

    def _instantiate_judge(self, config: JudgeConfig) -> Judge:
        return build_judge(
            config,
            self._methodology_profile,
            rate_limiter=self._rate_limiter,
            rate_limit_scope=self._rate_limit_scope,
            budget=self._budget_ledger,
        )

    def _seed_for(self, experiment_name: str, run_number: int) -> int:
        if self.config.seed_strategy == "random":
            return secrets.randbits(31)
        digest = hashlib.sha256(f"{experiment_name}:{run_number}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _provider_config_model(provider_cls: type[object]) -> type[BaseModel] | None:
    """Resolve the provider constructor's ``config`` annotation to a Pydantic model.

    Why: registered providers declare their configuration contract as the type
    annotation of their constructor's ``config`` parameter — either a Pydantic
    model (a memory provider, FullContextProvider) or a plain mapping. Reading
    the annotation keeps instantiation generic: new providers get validated
    construction without runner special-casing. Unresolvable annotations fall
    back to the raw-mapping behavior instead of failing instantiation.
    """

    try:
        hints = get_type_hints(provider_cls.__init__)
    except Exception:
        return None
    annotation = hints.get("config")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _provider_constructor_config(
    provider_cls: type[object],
    config: ProviderConfig,
) -> object:
    """Build the constructor argument a registered provider expects.

    Why: passing the raw experiment mapping into a provider that expects its
    Pydantic config model fails deep inside the provider (attribute errors at
    construction time). Validating here surfaces invalid experiment YAML as a
    ``ConfigurationError`` naming the provider type, before any process or
    network activity.
    """

    model_type = _provider_config_model(provider_cls)
    if model_type is None:
        return config.config
    try:
        return model_type.model_validate(config.config)
    except ValidationError as exc:
        raise ConfigurationError(
            "Invalid provider configuration.",
            provider_type=config.type,
            error=format_validation_errors(exc),
        ) from exc


def _warn_on_unmetered_spend(config: ExperimentSuiteConfig, *, cap_usd: float | None) -> None:
    """Warn when the suite will spend money the budget cap cannot see.

    Why: ``max_cost_usd`` bounds only what the cost tracker records — the answer model and
    the judge. A provider that calls an external embedder bills the same account through a
    channel Khedron never observes, so the cap silently understates real spend and reads as
    a guarantee it cannot give. The day-1 measurement ran this way with no warning at all.
    """
    for experiment in config.experiments:
        embedder = experiment.provider.config.get("embedder")
        if not isinstance(embedder, str) or embedder in ("", "none"):
            continue
        _LOGGER.warning(
            "unmetered_spend_outside_budget_cap",
            warning=(
                "Provider embedder calls are billed to your account but are not recorded "
                "by the cost tracker, so max_cost_usd does not bound them."
            ),
            experiment_name=experiment.name,
            provider_type=experiment.provider.type,
            # Bag value reaching a structured log; see preflight for the same reasoning.
            embedder=REDACTED if looks_like_credential(embedder) else embedder,
            max_cost_usd=cap_usd,
        )


_T = TypeVar("_T")


async def _gather_or_cancel_siblings(tasks: list[asyncio.Task[_T]]) -> list[_T]:
    """Like ``asyncio.gather``, but on the first task to raise it cancels and drains the rest.

    Why: ``asyncio.gather`` propagates the first exception without cancelling the other tasks, which
    keep running. For a conversation's concurrently-evaluated questions the only exception that
    escapes ``_evaluate_question`` is a pre-dispatch budget refusal (``CostExceededError``); every
    ProviderError/ModelError/JudgeError is caught and recorded there. When one escapes, the run is
    halting on cost, so the sibling questions must not keep dispatching more paid calls into the
    cap's remaining headroom while the run enters its terminal handler -- defeating the clean halt
    the pre-dispatch refusal exists to produce. Cancelled siblings hit ``paced_generate``'s
    ``CancelledError`` branch and KEEP their monetary holds, so ``worst_case`` stays honest and no
    task is orphaned. Results are returned in task order when every task succeeds.

    Cleanup lives in a ``finally`` so it runs on EVERY exit -- a task failed, or this coroutine was
    cancelled during the initial wait OR during the drain itself. It cancels every task and awaits
    them to completion; a cancellation arriving mid-drain is captured, re-cancels, and is re-raised
    only once every task has settled, so the drain always finishes and no task is left detached.
    """
    if not tasks:
        return []
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:
            # Report the first failure in task order (deterministic); .exception() is only safe on a
            # task that finished and was not cancelled.
            if task in done and not task.cancelled():
                failure = task.exception()
                if failure is not None:
                    raise failure
        return [task.result() for task in tasks]
    finally:
        deferred_cancel: asyncio.CancelledError | None = None
        for task in tasks:
            task.cancel()
        # Drain to completion, tolerating a cancellation that lands mid-drain: the tasks are already
        # cancelled, so the loop converges as soon as each finishes settling. A cancellation of THIS
        # coroutine during the drain is captured and re-raised only after every task is done, so the
        # drain is never abandoned half-way. A successful run enters here too, where every task is
        # already done and cancel()/gather are no-ops.
        while not all(task.done() for task in tasks):
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError as cancel:
                deferred_cancel = cancel
        if deferred_cancel is not None:
            raise deferred_cancel


def _ensure_within_budget(
    *, cap_usd: float | None, observed_cost_usd: float, worst_case_cost_usd: float
) -> None:
    """Raise ``CostExceededError`` when worst-case spend strictly exceeds the cap.

    Why: the pre-dispatch monetary reservation is now the primary bound; this boundary check is
    the backstop. It enforces the pessimistic ``worst_case`` figure
    -- confirmed spend plus the reserved maxima of every ambiguous billed-but-failed hold -- because
    that is the only figure accounting for spend the post-hoc recorder never saw (a billed attempt
    that failed client-side). Both figures are named in the error, distinctly, so the operator can
    tell confirmed spend from possible spend and decide whether to raise the cap or reduce scope.
    ``total_cost_usd`` keeps its present meaning (confirmed spend == observed) for every existing
    consumer of the error.
    """

    if cap_usd is None:
        return
    if worst_case_cost_usd > cap_usd:
        raise CostExceededError(
            f"Worst-case cost ${worst_case_cost_usd:.6f} (confirmed ${observed_cost_usd:.6f}) "
            f"exceeded the configured budget cap ${cap_usd:.6f}; halting after persisting "
            f"completed work.",
            budget_cap_usd=cap_usd,
            total_cost_usd=observed_cost_usd,
            observed_cost_usd=observed_cost_usd,
            worst_case_cost_usd=worst_case_cost_usd,
        )


def build_judge(
    config: JudgeConfig,
    profile: MethodologyRuntimeProfile,
    *,
    rate_limiter: RateLimiter | None = None,
    rate_limit_scope: str | None = None,
    budget: BudgetReserver | None = None,
) -> Judge:
    """Construct the judge a config names, reading its prompt from the given profile.

    Module-level rather than a Runner method because rejudging needs exactly this and has no
    Runner: a derived run reuses stored answers and only re-grades them. Duplicating the
    construction would let the two paths drift on the one thing that must not differ -- which
    prompt the judge is given.

    A judge is a model wrapped in a grading prompt (LLMJudge delegates every call to the model it
    holds), so pacing the judge means pacing that wrapped model. The limiter is injected into the
    model built here and passed to the judge, which is why rejudge -- which builds its judge through
    this same function -- gets a paced judge too. None leaves the judge unpaced, as before.
    """
    cls = get_judge_class(config.type)
    raw_config = {**config.config, "model_id": config.model}
    # `budget` is None for rejudge (which builds its judge here but has no runner ledger), so
    # rejudge's spend stays post-hoc as today; a suite run injects the ledger, so the judge's
    # calls are bounded before dispatch on the same money path as the answer model.
    paced: dict[str, Any] = {
        "rate_limiter": rate_limiter,
        "rate_limit_scope": rate_limit_scope,
        "budget": budget,
    }
    if cls is OpenAIJudge:
        model_config = OpenAIModelConfig.model_validate(raw_config)
        instance = OpenAIJudge(
            model_config,
            model=OpenAIModel(model_config, **paced),
            temperature=config.temperature,
            prompt_path=profile.judge_prompt_path,
        )
    elif cls is AnthropicJudge:
        anthropic_config = AnthropicModelConfig.model_validate(raw_config)
        instance = AnthropicJudge(
            anthropic_config,
            model=AnthropicModel(anthropic_config, **paced),
            temperature=config.temperature,
            prompt_path=profile.judge_prompt_path,
        )
    elif cls is GoogleJudge:
        google_config = GoogleModelConfig.model_validate(raw_config)
        instance = GoogleJudge(
            google_config,
            model=GoogleModel(google_config, **paced),
            temperature=config.temperature,
            prompt_path=profile.judge_prompt_path,
        )
    else:
        instance = cast(Any, cls)(raw_config, temperature=config.temperature)
    if not isinstance(instance, Judge):
        raise ConfigurationError(
            "Registered judge does not implement Judge.",
            judge_type=config.type,
        )
    return instance


def _ingestion_is_beyond_repair(
    n_turns_failed: int,
    n_turns_ingested: int,
    *,
    experiment_or_suite: ExperimentSuiteConfig,
) -> bool:
    """Whether the losses so far already exceed what the suite will publish.

    Judged on turns seen so far rather than on the whole corpus, which makes it conservative in
    the only direction that matters: later conversations can only add turns, so a rate above the
    threshold now may fall back under it by the end. Aborting early therefore discards a run that
    might have recovered.

    That trade is deliberate. The alternative -- the behaviour this replaces -- is paying for
    every remaining question and judgement of a run that is discarded anyway, when the loss can
    arrive as early as the first conversation. A run abandoned in the first minutes can
    be relaunched; the money cannot.
    """
    if n_turns_ingested <= 0:
        return False
    return n_turns_failed / n_turns_ingested > experiment_or_suite.max_ingestion_failure_rate


def _require_evaluated_categories_are_present(
    profile: MethodologyRuntimeProfile,
    benchmark: Benchmark,
    experiment: ExperimentConfig,
) -> None:
    """Refuse a run whose corpus yields no questions for a category the profile pins.

    A contractual profile pins `evaluation_categories` to state what the run asks. Nothing made
    the corpus honour that: restricting the suite to a subset of conversations could silently drop
    a category entirely, and the run would report four categories under a profile promising five
    while every score, interval and fingerprint looked untouched.

    A reduced conversation subset can silently miss a category: conv-30 is the only LoCoMo
    conversation with no `open_domain` questions, so evaluating only it would exercise neither
    that category's generation nor its judging -- the paths least like the others -- while
    reporting a clean pass.

    Failing closed, and here rather than in preflight, because preflight is advisory and reads only
    the config: which questions a corpus actually yields is knowable only once it is loaded. This
    runs after `benchmark.load()` and before any model is initialised, so the refusal costs
    nothing.
    """
    if profile.evaluation_categories is None:
        return
    present = {
        getattr(question.category, "value", str(question.category))
        for question in benchmark.get_questions()
    }
    missing = sorted(set(profile.evaluation_categories) - present)
    if not missing:
        return
    raise ConfigurationError(
        "The corpus yields no questions for categories the methodology profile evaluates, so the "
        "run would silently measure a narrower benchmark than the profile states. Widen the "
        "corpus selection or use a profile that does not evaluate them.",
        methodology_profile=profile.name,
        missing_categories=missing,
        evaluated_categories=list(profile.evaluation_categories),
        experiment_name=experiment.name,
    )


def _require_retrieval_budget_is_applicable(
    profile: MethodologyRuntimeProfile,
    provider: MemoryProvider,
    experiment: ExperimentConfig,
) -> None:
    """Refuse a run whose provider would silently ignore the profile's pinned retrieval budget.

    The contract pins `top_k_retrieval` because an unpinned budget is one of the things that made
    `canonical-v1` certify nothing. But the budget is meaningless for a provider that returns
    everything -- `FullContextProvider` discards it by design -- and a run that recorded a
    pinned 200 while the provider used all memories would be asserting something false about how it
    executed.

    Failing closed rather than warning, per contract section 3: "validation should fail, not
    shrug". A profile that does not pin a budget imposes nothing, so a full-context baseline stays
    runnable under one that leaves `top_k_retrieval` open.
    """
    if profile.top_k_retrieval is None:
        return
    if provider.honours_top_k:
        return
    raise ConfigurationError(
        "Provider does not honour a retrieval budget that the methodology profile pins, so a run "
        "would record a budget it never applied. Use a profile that leaves top_k_retrieval "
        "unpinned for this baseline, or a provider that retrieves a subset.",
        methodology_profile=profile.name,
        provider_type=experiment.provider.type,
        pinned_top_k=profile.top_k_retrieval,
        experiment_name=experiment.name,
    )


def _compute_scores(
    question_records: Sequence[QuestionRecord],
    *,
    scored_categories: Sequence[str] | None = None,
) -> RunScores:
    """Score a run, narrowing the headline to the profile's scored categories.

    The profile decides which categories enter `overall_*`; the scorer only applies it. Passing
    `None` scores everything, which is what every profile written before the split did.
    """
    return Scorer().score_run(question_records, scored_categories=scored_categories)


def _aggregate_score(
    *,
    experiment_id: str,
    category: QuestionCategory | Literal["overall"],
    mode: ScoreMode,
    scores: Sequence[ScoreWithCI],
) -> AggregateScore | None:
    if not scores:
        return None
    pooled = _pool_scores(scores)
    point_estimates = [score.point_estimate for score in scores]
    return AggregateScore(
        experiment_id=experiment_id,
        category=category,
        mode=mode,
        n_runs=len(scores),
        pooled_score=pooled,
        individual_run_scores=point_estimates,
        mean=mean(point_estimates),
        stddev=stdev(point_estimates) if len(point_estimates) > 1 else 0.0,
        min=min(point_estimates),
        max=max(point_estimates),
    )


def _pool_scores(scores: Sequence[ScoreWithCI]) -> ScoreWithCI:
    n_total = sum(score.n_total for score in scores)
    n_correct = sum(score.n_correct for score in scores)
    n_errors = sum(score.n_errors for score in scores)
    n_partial = sum(score.n_partial for score in scores)
    n_unknown = sum(score.n_unknown for score in scores)
    ci_low, ci_high = wilson_score_interval(n_correct, n_total)
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_errors,
        n_partial=n_partial,
        n_unknown=n_unknown,
        point_estimate=n_correct / n_total if n_total else 0.0,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
    )


def _count_correct(records: Iterable[QuestionRecord]) -> int:
    return sum(1 for record in records if record.verdict is JudgmentVerdict.CORRECT)


def _count_errored(records: Iterable[QuestionRecord]) -> int:
    return sum(1 for record in records if record.verdict is JudgmentVerdict.ERROR)


def _string_score_map(scores: dict[QuestionCategory, ScoreWithCI]) -> dict[str, ScoreWithCI]:
    return {category.value: score for category, score in scores.items()}


def _first_failed_run(runs: Sequence[_RunResult]) -> _RunResult | None:
    return next((run for run in runs if run.failed), None)


def _pooled_score(aggregate: AggregateScore | None) -> ScoreWithCI | None:
    return aggregate.pooled_score if aggregate is not None else None


def _aggregate_mean(aggregate: AggregateScore | None) -> float | None:
    return aggregate.mean if aggregate is not None else None


def _aggregate_stddev(aggregate: AggregateScore | None) -> float | None:
    return aggregate.stddev if aggregate is not None else None


def _classify_run_error_phase(exc: Exception) -> RunErrorPhase:
    if isinstance(exc, ProviderError):
        context_phase = exc.context.get("phase")
        if context_phase in _RUN_ERROR_PHASES:
            return cast(RunErrorPhase, context_phase)
        return "ingest"
    if isinstance(exc, BenchmarkError | ConfigurationError):
        return "initialization"
    if isinstance(exc, ModelError):
        return "generate"
    if isinstance(exc, JudgeError):
        return "judge"
    return "analysis"


async def _close_plugins(
    provider: MemoryProvider,
    answer_model: AnswerModel,
    judge: Judge,
) -> None:
    for plugin_name, close in (
        ("provider", provider.close),
        ("answer_model", answer_model.close),
        ("judge", judge.close),
    ):
        try:
            await close()
        except Exception as exc:
            _LOGGER.warning("plugin_close_failed", plugin=plugin_name, error=str(exc))


def _locomo_constructor_args(config: dict[str, Any]) -> _LocomoConstructorArgs:
    allowed = {
        "dataset_path",
        "audit_path",
        "expected_dataset_checksum",
        "conversations",
        "categories",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ConfigurationError(
            "Unsupported LoCoMo benchmark configuration keys.",
            keys=sorted(unknown),
        )

    return _LocomoConstructorArgs(
        dataset_path=(
            Path(_require_str_config(config["dataset_path"], "dataset_path"))
            if "dataset_path" in config
            else None
        ),
        audit_path=(
            Path(_require_str_config(config["audit_path"], "audit_path"))
            if "audit_path" in config
            else None
        ),
        expected_dataset_checksum=(
            _require_str_config(
                config["expected_dataset_checksum"],
                "expected_dataset_checksum",
            )
            if "expected_dataset_checksum" in config
            else None
        ),
        conversations=(
            _require_str_list_config(config["conversations"], "conversations")
            if "conversations" in config
            else None
        ),
        categories=(
            _require_str_list_config(config["categories"], "categories")
            if "categories" in config
            else None
        ),
    )


def _require_str_config(value: object, key: str) -> str:
    if isinstance(value, str):
        return value
    raise ConfigurationError("Configuration value must be a string.", key=key)


def _require_str_list_config(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigurationError(
            "Configuration value must be a list of strings.",
            key=key,
            observed_type=type(value).__name__,
        )

    raw_items = cast(list[object], value)
    items: list[str] = []
    for item_index, item in enumerate(raw_items):
        if not isinstance(item, str):
            raise ConfigurationError(
                "Configuration list item must be a string.",
                key=f"{key}[{item_index}]",
                observed_type=type(item).__name__,
            )
        items.append(item)
    return items


def _capture_runtime_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _methodology_runtime_environment(
    profile: MethodologyRuntimeProfile,
    fingerprint: str,
    *,
    config_digest: str | None = None,
    provider_honours_top_k: bool = True,
    corpus_conversations_evaluated: int | None = None,
    corpus_conversations_available: int | None = None,
) -> dict[str, Any]:
    return {
        "methodology_profile_status": profile.status,
        # Bound to the configuration that produced the run, so a resume under an edited config is
        # refused rather than splicing two measurements into one artifact.
        "config_digest": config_digest,
        # How much of the corpus the run covered. A suite may legitimately narrow itself to a few
        # conversations -- rehearsals and smoke tests exist for good reasons -- but the resulting
        # report is otherwise indistinguishable from one over the whole corpus: same headline
        # phrasing, same Wilson intervals, arithmetically valid over 158 questions and meaningless
        # as a statement about the system. Recorded here so the report can say which it is.
        "corpus_conversations_evaluated": corpus_conversations_evaluated,
        "corpus_conversations_available": corpus_conversations_available,
        # Whether the provider applied the retrieval budget at all. Recorded so a report can say
        # "not applicable" instead of printing a configured default the run never used.
        "provider_honours_top_k": provider_honours_top_k,
        # Which categories the run's overall_* actually covers. Without it the headline is a
        # number over an undeclared subset: 7.66% over 1,540 questions and 26.99% over 1,986 are
        # not comparable and nothing else distinguishes them. `None` means every evaluated
        # category, which is what profiles written before the split meant.
        "methodology_scored_categories": (
            list(profile.scored_categories) if profile.scored_categories is not None else None
        ),
        "methodology_evaluation_categories": (
            list(profile.evaluation_categories)
            if profile.evaluation_categories is not None
            else None
        ),
        "methodology_profile_reference": profile.reference_name,
        "methodology_profile_reference_url": profile.reference_url,
        "methodology_profile_reference_commit": profile.reference_commit,
        METHODOLOGY_FINGERPRINT_KEY: fingerprint,
        "methodology_generator_prompt": str(profile.generator_prompt_path),
        "methodology_judge_prompt": str(profile.judge_prompt_path),
        "methodology_top_k_cutoffs": list(profile.top_k_cutoffs),
    }
