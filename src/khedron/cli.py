from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from khedron.analysis import RunComparator
from khedron.analysis.paired import compare_paired
from khedron.config import ExperimentSuiteConfig, JudgeConfig, load_suite_config
from khedron.eligibility import Eligibility, assess_run_eligibility, assess_stream_eligibility
from khedron.errors import ConfigurationError, KhedronError
from khedron.methodology import get_runtime_profile
from khedron.pacing import build_rate_limiter, load_rate_limits_config
from khedron.persistence.repository import RunRepository
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.persistence.stage_reader import RunStageStream
from khedron.preflight import PreflightFinding, preflight_suite
from khedron.progress import read_progress
from khedron.publishability import resolve_disposition
from khedron.rejudge import rejudge_run
from khedron.reporting import ReportContextBuilder, ReportGenerator
from khedron.resume import recover_blocked_run
from khedron.runner import Runner, build_judge
from khedron.tls import configure_os_trust_store
from khedron.types import RunFilters, ScoreWithCI, SuiteStatus

USER_ERROR = 1
SYSTEM_ERROR = 2
DEFAULT_RESULTS_DIR = Path("results")

app = typer.Typer(help="Khedron command line interface.")
inspect_app = typer.Typer(help="Inspect persisted benchmark results.")
db_app = typer.Typer(help="SQLite query-layer maintenance commands.")
app.add_typer(inspect_app, name="inspect")
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    """Run once before any command: trust the OS certificate store for TLS.

    Keeps live API calls working on machines behind a TLS-inspection proxy or antivirus
    HTTPS scanner, whose root CA the OS trusts but certifi does not.
    """
    configure_os_trust_store()
    _force_utf8_console()


def _force_utf8_console() -> None:
    """Make stdout and stderr encode anything the corpus contains.

    Windows consoles default to cp1252, which cannot encode most of what a multilingual benchmark
    prints. That turned a *best-effort* indexing warning into a UnicodeEncodeError that killed the
    process mid-run: the fallback meant to keep a run alive was the thing that ended it. Replacing
    unencodable characters degrades a log line; refusing to print one ends a run that is already
    spending money.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


ConfigOption = Annotated[Path, typer.Option("--config", dir_okay=False)]
CandidateResultsDirOption = Annotated[
    Path | None,
    typer.Option(
        "--candidate-results-dir",
        help="Results root holding the candidate run, when it differs from --results-dir.",
    ),
]
ResultsDirOption = Annotated[Path, typer.Option("--results-dir")]
SqlitePathOption = Annotated[Path | None, typer.Option("--sqlite-path")]
OutputOption = Annotated[Path | None, typer.Option("--output", dir_okay=False)]
ReportTypeOption = Annotated[str | None, typer.Option("--type")]
RunIdOption = Annotated[str | None, typer.Option("--run-id")]
RunIdsOption = Annotated[list[str] | None, typer.Option("--run-ids")]
RebuildOption = Annotated[bool, typer.Option("--rebuild")]


@app.command()
def validate(config: ConfigOption) -> None:
    """Validate a suite configuration without executing it."""

    suite_config = _load_config_or_exit(config)
    typer.echo(
        "Configuration valid: "
        f"{len(suite_config.experiments)} experiment(s), {suite_config.runs} run(s)."
    )


@app.command()
def preflight(config: ConfigOption) -> None:
    """Report what would make a run fail or produce an unpublishable result."""

    suite_config = _load_config_or_exit(config)
    findings = _preflight_or_exit(suite_config)
    blockers = [finding for finding in findings if finding.severity == "blocker"]
    if blockers:
        _exit_user_error(f"Preflight found {len(blockers)} blocker(s); do not spend.")
    typer.echo(f"Preflight passed with {len(findings)} warning(s).")


@app.command()
def run(
    config: ConfigOption,
    resume_run_id: Annotated[str | None, typer.Option("--resume-run-id")] = None,
    allow_unidentifiable_build: Annotated[
        bool, typer.Option("--allow-unidentifiable-build")
    ] = False,
) -> None:
    """Execute a benchmark suite, optionally continuing an interrupted run.

    Resuming skips every question already judged and every conversation already processed, and
    re-ingests only the conversation that was in flight. It refuses if anything that defines the
    measurement has changed since -- profile, fingerprint, configuration, dataset or build --
    because continuing across such a change would splice two measurements into one artifact.

    `--allow-unidentifiable-build` overrides only the build-identity blocker, for a deliberate
    decision to spend on a run whose result is knowingly un-publishable (its provenance cannot be
    pinned to a commit). It never overrides any other blocker.
    """

    suite_config = _load_config_or_exit(config)
    # Refuse to spend on a configuration that cannot produce a usable result.
    findings = _preflight_or_exit(suite_config)
    blockers = [finding for finding in findings if finding.severity == "blocker"]
    if allow_unidentifiable_build:
        overridden = [finding for finding in blockers if finding.check == "build_identity"]
        blockers = [finding for finding in blockers if finding.check != "build_identity"]
        for finding in overridden:
            typer.echo(
                "WARNING: build-identity blocker overridden by explicit request; this run is "
                f"knowingly un-publishable. {finding.detail}",
                err=True,
            )
    if blockers:
        _exit_user_error(f"Preflight found {len(blockers)} blocker(s); refusing to run.")
    repository = _repository(Path(suite_config.output_dir), None)
    try:
        result = asyncio.run(
            Runner(
                suite_config,
                repository,
                resume_run_id=resume_run_id,
                allow_unidentifiable_build=allow_unidentifiable_build,
            ).run()
        )
    except (ConfigurationError, KeyError, ValueError) as exc:
        _exit_user_error(f"Run configuration error: {exc}")
    except KhedronError as exc:
        _exit_system_error(f"Run failed: {exc}")
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")

    typer.echo(f"Suite ID: {result.suite_id}")
    typer.echo(f"Experiments completed: {len(result.experiment_results)}")
    typer.echo(f"Total cost USD: {_money(result.total_cost_usd)}")
    # Only surfaced when it differs: confirmed spend is the headline figure; a nonzero ambiguous
    # residual means some billed-but-failed attempts may have cost more than the recorder
    # confirmed, and the operator should see that possible-spend ceiling.
    if result.worst_case_cost_usd > result.total_cost_usd:
        typer.echo(f"Worst-case cost USD: {_money(result.worst_case_cost_usd)}")
    # An experiment whose runs were all excluded by the publication guard produces no result
    # and does not stop the suite. Printing only the completed count made that read as a
    # clean run to anything reading exit codes -- money spent, nothing measured, exit 0.
    n_missing = len(suite_config.experiments) - len(result.experiment_results)
    if n_missing > 0:
        message = (
            f"Experiments with no publishable result: {n_missing} of "
            f"{len(suite_config.experiments)}. Inspect the suite lifecycle events for the "
            "guard that excluded each run."
        )
        if not result.experiment_results:
            _exit_system_error(message)
        typer.echo(message, err=True)


@app.command()
def progress(
    run_id: RunIdOption = None,
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    expected_cost_usd: Annotated[float | None, typer.Option("--expected-cost-usd")] = None,
) -> None:
    """Report a run in flight, and every reason to kill it now.

    Reads the append-only stream, so it works while the run is still writing. It aborts nothing:
    it reports, and the operator decides. Its checks are not generic health metrics -- each one
    corresponds to a defect this project has already paid for, the most expensive being answers
    truncated at a token ceiling and scored as though they were finished.
    """
    if run_id is None:
        _exit_user_error("Missing required option: --run-id")

    try:
        report = read_progress(results_dir, run_id, expected_cost_usd=expected_cost_usd)
    except Exception as exc:
        _exit_user_error(f"Unable to read progress for {run_id}: {exc}")

    typer.echo(f"Run {report.run_id} ({report.profile})")
    typer.echo(f"  answered:         {report.n_answered}")
    typer.echo(f"  errored:          {report.n_errored}")
    typer.echo(f"  judge unparsed:   {report.n_judge_parse_failures}")
    typer.echo(f"  refusals:         {report.n_refusals} ({report.refusal_rate * 100:.1f}%)")
    typer.echo(f"  cost so far:      {_money(report.cost_usd)}")
    ceiling = f" of {report.max_output_tokens}" if report.max_output_tokens else ""
    typer.echo(f"  output tokens p95: {report.p95_output_tokens}{ceiling}")
    for reason in report.warn:
        typer.echo(f"  WARNING: {reason}", err=True)
    for reason in report.stop:
        typer.echo(f"  STOP: {reason}", err=True)
    if report.stop:
        _exit_user_error(f"{len(report.stop)} reason(s) to kill this run.")


@app.command()
def estimate(config: ConfigOption) -> None:
    """Summarize a suite configuration without calling providers, models, or judges."""

    suite_config = _load_config_or_exit(config)
    typer.echo("Estimate:")
    typer.echo(f"Runs: {suite_config.runs}")
    typer.echo(f"Experiments: {len(suite_config.experiments)}")
    for experiment in suite_config.experiments:
        typer.echo(f"- {experiment.name}")


@inspect_app.command("suite")
def inspect_suite(
    suite_id: Annotated[str, typer.Option("--suite-id")],
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    field: Annotated[str | None, typer.Option("--field")] = None,
) -> None:
    """Show suite status and high-level lifecycle state."""

    repository = _repository(results_dir, None)
    try:
        status = repository.get_suite_status(suite_id)
        runs = repository.list_runs(RunFilters(suite_id=suite_id))
    except KhedronError as exc:
        _exit_user_error(f"Unable to inspect suite: {exc}")

    if field is not None:
        typer.echo(_suite_field(status, field))
        return

    typer.echo(f"Suite ID: {status.suite_id}")
    typer.echo(f"Status: {status.status}")
    typer.echo(f"Phase: {status.last_event_type}")
    typer.echo(f"Total cost USD: {_money(status.total_cost_usd)}")
    typer.echo(f"Experiments planned: {status.n_experiments_planned}")
    typer.echo(f"Experiments completed: {status.n_experiments_completed}")
    typer.echo(f"Experiments failed: {status.n_experiments_failed}")
    typer.echo(f"Experiments in progress: {status.n_experiments_in_progress}")
    typer.echo(f"Started at: {status.started_at.isoformat()}")
    typer.echo(f"Finished at: {_optional_datetime(status.finished_at)}")
    typer.echo(f"Last event at: {status.last_event_at.isoformat()}")
    typer.echo(f"Run IDs: {', '.join(run.run_id for run in runs) if runs else '(none)'}")


@inspect_app.command("run")
def inspect_run(
    run_id: Annotated[str, typer.Option("--run-id")],
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
) -> None:
    """Show run status, lifecycle counts, cost, models, and scores."""

    repository = _repository(results_dir, None)
    try:
        status = repository.get_run_status(run_id)
    except KhedronError as exc:
        _exit_user_error(f"Unable to inspect run: {exc}")

    typer.echo(f"Run ID: {status.run_id}")
    typer.echo(f"Suite ID: {status.suite_id}")
    typer.echo(f"Experiment: {status.experiment_name}")
    typer.echo(f"Status: {status.status}")
    typer.echo(f"Started at: {status.started_at.isoformat()}")
    typer.echo(f"Finished at: {_optional_datetime(status.finished_at)}")
    typer.echo(f"Provider: {status.provider_type} {status.provider_version}")
    typer.echo(f"Answer model: {status.answer_model_id}")
    typer.echo(f"Judge model: {status.judge_model_id}")
    typer.echo(f"Conversations processed: {status.n_conversations_processed}")
    typer.echo(f"Questions attempted: {status.n_questions_attempted}")
    typer.echo(f"Questions succeeded: {status.n_questions_succeeded}")
    typer.echo(f"Questions errored: {status.n_questions_errored}")
    typer.echo(f"Total cost USD: {_money(status.total_cost_usd)}")
    typer.echo(f"Standard score: {_score(status.overall_score_standard)}")
    typer.echo(f"Audited score: {_score(status.overall_score_audited)}")
    # Forced-recovery disclosure: print the count and reasons when any
    # verdict was reached by operator-forced recovery. Read from the strict stream, best-effort: a
    # run too corrupt to read has larger problems, and inspect must still show its status.
    try:
        stream = repository.get_strict_stage_stream(run_id)
        forced_recovery_count = assess_stream_eligibility(stream).forced_recovery_count
    except KhedronError:
        forced_recovery_count = 0
        stream = None
    if forced_recovery_count > 0 and stream is not None:
        reasons = "; ".join(_forced_recovery_reasons(stream))
        typer.echo(f"Forced recoveries: {forced_recovery_count} (reasons: {reasons})")
    if status.error_phase is not None:
        typer.echo(f"Error phase: {status.error_phase}")
    if status.error_message is not None:
        typer.echo(f"Error: {status.error_message}")


@inspect_app.command("question")
def inspect_question(
    run_id: Annotated[str, typer.Option("--run-id")],
    question_id: Annotated[str, typer.Option("--question-id")],
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
) -> None:
    """Show a hydrated question trace."""

    repository = _repository(results_dir, None)
    try:
        record = repository.get_question_record(run_id, question_id)
    except KhedronError as exc:
        _exit_user_error(f"Unable to inspect question: {exc}")

    typer.echo(f"Question ID: {record.question_id}")
    typer.echo(f"Run ID: {record.run_id}")
    typer.echo(f"Category: {record.category.value}")
    typer.echo(f"Verdict: {record.verdict.value}")
    typer.echo(f"Score: {record.score:.4f}")
    typer.echo(f"Question: {record.question_text}")
    typer.echo(f"Expected answer: {record.expected_answer}")
    typer.echo(f"Generated answer: {_optional_text(record.generated_answer)}")
    typer.echo(f"Retrieval ID: {_optional_text(record.retrieval_id)}")
    typer.echo(f"Response ID: {_optional_text(record.response_id)}")
    typer.echo(f"Judgment ID: {_optional_text(record.judgment_id)}")
    typer.echo(f"Memories retrieved: {_optional_int(record.n_memories_retrieved)}")
    typer.echo(f"Retrieved memory IDs: {_joined(record.retrieved_memory_ids)}")
    typer.echo(f"Judgment reasoning: {_optional_text(record.judgment_reasoning)}")
    if record.error_phase is not None:
        typer.echo(f"Error phase: {record.error_phase}")
    if record.error_message is not None:
        typer.echo(f"Error: {record.error_message}")


@db_app.command("rebuild")
def db_rebuild(
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    sqlite_path: SqlitePathOption = None,
) -> None:
    """Rebuild SQLite from JSONL source-of-truth files."""

    repository = _repository(results_dir, sqlite_path)
    try:
        asyncio.run(repository.rebuild_sqlite())
    except KhedronError as exc:
        _exit_user_error(f"Unable to rebuild SQLite: {exc}")
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")
    typer.echo(f"SQLite rebuilt: {_effective_sqlite_path(results_dir, sqlite_path)}")


@db_app.command("version")
def db_version(
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    sqlite_path: SqlitePathOption = None,
) -> None:
    """Print the current SQLite schema version."""

    try:
        version = SqliteIndexer(
            _effective_sqlite_path(results_dir, sqlite_path)
        ).get_schema_version()
    except KhedronError as exc:
        _exit_user_error(f"Unable to read SQLite schema version: {exc}")
    typer.echo(str(version))


@app.command()
def report(
    run_id: RunIdOption = None,
    report_type: ReportTypeOption = None,
    output: OutputOption = None,
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    sqlite_path: SqlitePathOption = None,
    rebuild: RebuildOption = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    acknowledge_forced_recoveries: Annotated[
        bool, typer.Option("--acknowledge-forced-recoveries")
    ] = False,
) -> None:
    """Generate a Markdown report for a persisted benchmark run.

    Refuses a run whose scores must not be published -- a superseded profile, an unidentifiable
    build, a partial corpus -- unless `--force` names the exception deliberately. A forced-recovered
    run is disclosed and refused unless `--acknowledge-forced-recoveries` names it.
    """

    if run_id is None:
        _exit_user_error("Missing required option: --run-id")
    if report_type is None:
        _exit_user_error("Missing required option: --type")
    if output is None:
        _exit_user_error("Missing required option: --output")
    if report_type not in {"executive", "technical", "failures"}:
        _exit_user_error(
            f"Invalid report type: {report_type}. Expected one of: executive, technical, failures."
        )

    repository = _repository(results_dir, sqlite_path)
    _require_publishable(
        repository,
        run_id,
        force=force,
        acknowledge_forced_recoveries=acknowledge_forced_recoveries,
    )
    repository = _repository(results_dir, sqlite_path)
    try:
        if rebuild:
            asyncio.run(repository.rebuild_sqlite())
        context = ReportContextBuilder().build(run_id, repository)
        generator = ReportGenerator()
        if report_type == "executive":
            generator.generate_executive_summary(context, output)
        elif report_type == "technical":
            generator.generate_technical_deep_dive(context, output)
        else:
            generator.generate_failure_analysis_report(context, output)
    except ConfigurationError as exc:
        _exit_user_error(f"Unable to generate report: {exc}")
    except KhedronError as exc:
        _exit_user_error(f"Unable to generate report: {exc}")
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")

    typer.echo(f"Report written: {output}")


@app.command("compare-paired")
def compare_paired_command(
    baseline_run_id: Annotated[str | None, typer.Option("--baseline-run-id")] = None,
    candidate_run_id: Annotated[str | None, typer.Option("--candidate-run-id")] = None,
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    candidate_results_dir: Annotated[Path | None, typer.Option("--candidate-results-dir")] = None,
    sqlite_path: SqlitePathOption = None,
) -> None:
    """Compare two runs question by question with an exact McNemar test.

    Use this when the two runs asked the same questions -- and especially when they share their
    answers, as a rejudged run does with its source. Two independent intervals throw away the
    pairing, and with it the only observations that carry any signal about which run is better.
    """
    if baseline_run_id is None:
        _exit_user_error("Missing required option: --baseline-run-id")
    if candidate_run_id is None:
        _exit_user_error("Missing required option: --candidate-run-id")

    baseline_repository = _repository(results_dir, sqlite_path)
    candidate_repository = (
        _repository(candidate_results_dir, None)
        if candidate_results_dir is not None
        else baseline_repository
    )
    try:
        baseline_eligibility = _require_eligible_for_comparison(
            baseline_repository, baseline_run_id
        )
        candidate_eligibility = _require_eligible_for_comparison(
            candidate_repository, candidate_run_id
        )
        if baseline_eligibility.planned_question_ids != candidate_eligibility.planned_question_ids:
            _exit_user_error("Refusing comparison: runs have different planned question ID sets.")
        result = compare_paired(
            baseline_repository.get_questions_for_run(baseline_run_id),
            candidate_repository.get_questions_for_run(candidate_run_id),
        )
    except KhedronError as exc:
        _exit_user_error(f"Unable to compare: {exc}")

    typer.echo(f"Paired questions: {result.n_paired}")
    typer.echo(f"  both correct:            {result.n_both_correct}")
    typer.echo(f"  both wrong:              {result.n_both_wrong}")
    typer.echo(f"  only baseline correct:   {result.n_only_baseline_correct}")
    typer.echo(f"  only candidate correct:  {result.n_only_candidate_correct}")
    if not result.question_sets_match:
        typer.echo(
            f"WARNING: the two runs do not ask the same questions "
            f"({result.n_baseline_only} only in baseline, {result.n_candidate_only} only in "
            "candidate). A paired test over the intersection is biased when the missing questions "
            "are the ones a run failed on.",
            err=True,
        )
    typer.echo(f"Discordant pairs: {result.n_discordant}")
    typer.echo(f"Net delta (candidate - baseline): {result.net_delta * 100:+.2f} pp")
    typer.echo(f"Exact McNemar p-value: {result.p_value:.6f}")
    typer.echo(
        "Significant at 5%: "
        + ("yes" if result.is_significant else "no")
        + ". Significance is not materiality: a difference below the benchmark's resolution is "
        "still below it."
    )


def _require_eligible_for_comparison(repository: RunRepository, run_id: str) -> Eligibility:
    eligibility = assess_run_eligibility(repository, run_id)
    if not eligibility.eligible:
        reasons = "".join(f"\n  - {reason}" for reason in eligibility.reasons)
        _exit_user_error(f"Refusing comparison for ineligible run {run_id}:{reasons}")
    return eligibility


def _require_publishable(
    repository: RunRepository,
    run_id: str,
    *,
    force: bool,
    acknowledge_forced_recoveries: bool = False,
) -> None:
    """Refuse to write an outward-facing artifact for a run whose scores must not be published.

    `report` and `compare` are the two commands that produce artifacts leaving this machine, so
    they are where the guard belongs. Overridable, because a withdrawn run still needs to be
    inspectable -- and the override is explicit and named, which is the difference between a
    deliberate act and an accident.

    A forced-recovered run is disclosed unconditionally and gated at publish time: its
    count and reasons are always printed, and it is refused unless `--acknowledge-forced-recoveries`
    names the operator consent, exactly as `--force` clears a withdrawn run. Disclosure does not
    depend on the flag; the flag only clears the publish block.
    """
    try:
        stream = repository.get_strict_stage_stream(run_id)
        eligibility = assess_stream_eligibility(stream)
        disposition = resolve_disposition(repository.get_run_started_event(run_id), eligibility)
    except KhedronError as exc:
        _exit_user_error(f"Unable to establish publishability for {run_id}: {exc}")

    if eligibility.forced_recovery_count > 0:
        reasons = "".join(f"\n  - {reason}" for reason in _forced_recovery_reasons(stream))
        typer.echo(
            f"DISCLOSURE: {run_id} reached {eligibility.forced_recovery_count} verdict(s) by "
            f"operator-forced recovery of a non-retryable error:{reasons}",
            err=True,
        )
        if not acknowledge_forced_recoveries:
            _exit_user_error(
                f"Refusing to publish {run_id}: it was forced-recovered, so it must not be "
                "published or compared as if fully organic. Pass "
                "--acknowledge-forced-recoveries to write it anyway; the disclosure above stands "
                "regardless."
            )

    if disposition.publishable:
        return
    reasons = "".join(f"\n  - {reason}" for reason in disposition.reasons)
    if force:
        typer.echo(
            f"WARNING: {run_id} is withdrawn/non-publishable and --force was given:{reasons}",
            err=True,
        )
        return
    _exit_user_error(
        f"Refusing to write a report for {run_id}, whose scores must not be published:{reasons}"
        "\nPass --force to write it anyway; the artifact will carry this warning."
    )


def _forced_recovery_reasons(stream: RunStageStream) -> list[str]:
    """The operator reasons of the run's consummated forced recoveries, for disclosure.

    Only forced recoveries whose attempt exists are counted (a persisted-then-crashed authorization
    is an admissible crash window, never a reached verdict), matching the forced_recovery_count that
    eligibility computes.
    """
    attempt_ids = {attempt.attempt_id for attempt in stream.attempts}
    return [
        forced.reason for forced in stream.forced_recoveries if forced.attempt_id in attempt_ids
    ]


@app.command()
def rejudge(
    run_id: RunIdOption = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    judge_type: Annotated[str | None, typer.Option("--judge-type")] = None,
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    sqlite_path: SqlitePathOption = None,
    max_cost_usd: Annotated[float | None, typer.Option("--max-cost-usd")] = None,
    rate_limits: Annotated[Path | None, typer.Option("--rate-limits")] = None,
) -> None:
    """Re-score an existing run's stored answers with the judge a profile pins.

    The answers are not regenerated, so every difference between the source run and the derived one
    is the judge and nothing else. That is what makes a grading difference measurable rather than
    arguable -- and it costs the judge calls alone, on a corpus whose answers already exist.
    """
    if run_id is None:
        _exit_user_error("Missing required option: --run-id")
    if profile is None:
        _exit_user_error("Missing required option: --profile")

    repository = _repository(results_dir, sqlite_path)
    try:
        runtime_profile = get_runtime_profile(profile)
        if runtime_profile.judge_type is None or runtime_profile.judge_model_id is None:
            _exit_user_error(
                f"Profile {profile!r} does not pin a judge, so it cannot name one to re-score with."
            )
        judge_config = JudgeConfig(
            type=judge_type or runtime_profile.judge_type,
            model=runtime_profile.judge_model_id,
            temperature=0.0,
        )
        # rejudge builds its judge outside the Runner, so the pacer must be threaded in here or the
        # judge calls would be unpaced. The limiter comes from an explicit --rate-limits file, not
        # from an invented default; absent it, the judge is unpaced as before.
        limits_config = load_rate_limits_config(rate_limits) if rate_limits is not None else None
        limiter = build_rate_limiter(limits_config)
        result = asyncio.run(
            rejudge_run(
                source_run_id=run_id,
                profile_name=profile,
                judge=build_judge(
                    judge_config,
                    runtime_profile,
                    rate_limiter=limiter,
                    rate_limit_scope=limits_config.scope if limits_config is not None else None,
                ),
                repository=repository,
                max_cost_usd=max_cost_usd,
            )
        )
    except ConfigurationError as exc:
        _exit_user_error(f"Unable to rejudge: {exc}")
    except KhedronError as exc:
        _exit_user_error(f"Unable to rejudge: {exc}")
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")

    typer.echo(f"Rejudged run ID: {result.run_id}")
    typer.echo(f"Derived from: {result.source_run_id}")
    typer.echo(f"Questions re-scored: {result.n_questions}")
    typer.echo(f"Total cost USD: {result.cost_usd:.6f}")


@app.command("recover-blocked")
def recover_blocked(
    config: ConfigOption,
    run_id: RunIdOption = None,
    question: Annotated[list[str] | None, typer.Option("--question")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    sqlite_path: SqlitePathOption = None,
    max_cost_usd: Annotated[float | None, typer.Option("--max-cost-usd")] = None,
) -> None:
    """Operator-force recovery of named, blocked, verdict-less questions of a run.

    Drives the residual `retryable=False` classes -- auth/billing failures and any TERMINAL
    disposition -- that automatic resume can never dispatch, and only when an operator supplies a
    reason recorded verbatim as an audit record. It refuses unless the current build both matches
    the original run and is identifiable (no framework-drift or unidentifiable-build override), and
    it refuses a question that already has a verdict (that is `rejudge`) or one the automatic resume
    path already handles. The whole command is atomic: one ineligible question refuses it entirely.

    `--config` must be the same suite YAML the original run used: the resume guards recompute the
    fingerprint, config digest and dataset checksum from it, so no trust is placed in a config
    reconstructed from the recorded events.
    """
    if run_id is None:
        _exit_user_error("Missing required option: --run-id")
    if not question:
        _exit_user_error("Missing required option: at least one --question")
    if reason is None or not reason.strip():
        _exit_user_error("Missing required option: --reason (a non-empty operator justification)")

    suite_config = _load_config_or_exit(config)
    # The run lives under the suite's output_dir and the recovery engine locks it there, so the
    # repository must point at the same directory (as `run`/resume do) or the lock and the data
    # would diverge. --results-dir is therefore deliberately not exposed here.
    repository = _repository(Path(suite_config.output_dir), sqlite_path)
    try:
        result = asyncio.run(
            recover_blocked_run(
                run_id=run_id,
                question_ids=question,
                reason=reason,
                config=suite_config,
                repository=repository,
                max_cost_usd=max_cost_usd,
            )
        )
    except ConfigurationError as exc:
        _exit_user_error(f"Unable to recover blocked questions: {exc}")
    except KhedronError as exc:
        _exit_user_error(f"Unable to recover blocked questions: {exc}")
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")

    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(
        "Questions recovered: "
        f"{', '.join(result.recovered_question_ids) if result.recovered_question_ids else '(none)'}"
    )
    typer.echo(
        "Questions not recovered: "
        + (
            ", ".join(result.unrecovered_question_ids)
            if result.unrecovered_question_ids
            else "(none)"
        )
    )
    typer.echo(f"Forced recoveries written: {result.forced_recovery_count}")
    typer.echo(f"Incremental cost USD: {_money(result.incremental_cost_usd)}")


@app.command()
def compare(
    run_ids: RunIdsOption = None,
    output: OutputOption = None,
    results_dir: ResultsDirOption = DEFAULT_RESULTS_DIR,
    candidate_results_dir: CandidateResultsDirOption = None,
    sqlite_path: SqlitePathOption = None,
    rebuild: RebuildOption = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    acknowledge_forced_recoveries: Annotated[
        bool, typer.Option("--acknowledge-forced-recoveries")
    ] = False,
) -> None:
    """Generate a Markdown comparison report for exactly two persisted runs.

    Refuses either run if its scores must not be published, unless `--force` names the exception:
    a comparison inherits the publishability of both sides, and a delta against a withdrawn run
    launders it into a number that reads as evidence. A forced-recovered side is disclosed and
    refused unless `--acknowledge-forced-recoveries` names it.

    The candidate may live in a second results root: a provider suite and its baseline are
    separate suites with separate output directories, and comparing them is the point of running
    the baseline at all.
    """

    if run_ids is None:
        _exit_user_error("Missing required option: --run-ids")
    if output is None:
        _exit_user_error("Missing required option: --output")

    repository = _repository(results_dir, sqlite_path)
    try:
        if rebuild:
            asyncio.run(repository.rebuild_sqlite())
        candidate_repository = (
            _repository(candidate_results_dir, None) if candidate_results_dir is not None else None
        )
        # Guarded on the count first. Indexing before it turned "you gave one run id" from a user
        # error into an IndexError reported as a system fault -- the guard breaking the diagnosis
        # of a mistake it has no opinion about.
        if len(run_ids) == 2:
            _require_publishable(
                repository,
                run_ids[0],
                force=False,
                acknowledge_forced_recoveries=acknowledge_forced_recoveries,
            )
            _require_publishable(
                candidate_repository or repository,
                run_ids[1],
                force=False,
                acknowledge_forced_recoveries=acknowledge_forced_recoveries,
            )
        comparison = RunComparator().compare(
            run_ids, repository, candidate_repository=candidate_repository
        )
        ReportGenerator().generate_comparison_report(comparison, output)
    except ConfigurationError as exc:
        _exit_user_error(f"Unable to generate comparison: {exc}")
    except KhedronError as exc:
        _exit_user_error(f"Unable to generate comparison: {exc}")
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_system_error(f"Unexpected system error: {exc}")

    typer.echo(f"Comparison report written: {output}")


def _load_config_or_exit(config_path: Path) -> ExperimentSuiteConfig:
    try:
        return load_suite_config(config_path)
    except ConfigurationError as exc:
        _exit_user_error(f"Configuration error: {exc}")


def _repository(results_dir: Path, sqlite_path: Path | None) -> RunRepository:
    return RunRepository(
        results_dir=results_dir,
        sqlite_path=_effective_sqlite_path(results_dir, sqlite_path),
    )


def _effective_sqlite_path(results_dir: Path, sqlite_path: Path | None) -> Path:
    return sqlite_path if sqlite_path is not None else results_dir / "benchmark.db"


def _suite_field(status: SuiteStatus, field: str) -> str:
    fields = {
        "suite_id": status.suite_id,
        "status": status.status,
        "phase": status.last_event_type,
        "total_cost_usd": _money(status.total_cost_usd),
        "n_experiments_planned": str(status.n_experiments_planned),
        "n_experiments_completed": str(status.n_experiments_completed),
        "n_experiments_failed": str(status.n_experiments_failed),
        "n_experiments_in_progress": str(status.n_experiments_in_progress),
        "started_at": status.started_at.isoformat(),
        "finished_at": _optional_datetime(status.finished_at),
        "last_event_at": status.last_event_at.isoformat(),
        "last_event_type": status.last_event_type,
    }
    try:
        return fields[field]
    except KeyError:
        _exit_user_error(f"Unknown suite field: {field}")


def _score(score: ScoreWithCI | None) -> str:
    if score is None:
        return "(none)"
    return (
        f"{score.point_estimate:.4f} "
        f"({score.n_correct}/{score.n_total}, 95% CI "
        f"{score.ci_95_low:.4f}-{score.ci_95_high:.4f})"
    )


def _optional_datetime(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "(none)"


def _optional_text(value: str | None) -> str:
    return value if value is not None else "(none)"


def _optional_int(value: int | None) -> str:
    return str(value) if value is not None else "(none)"


def _joined(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _money(value: float) -> str:
    return f"{value:.6f}"


def _preflight_or_exit(suite_config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    """Run preflight and print every finding, blockers and warnings alike.

    Warnings are printed rather than filtered out because an operator about to spend money
    needs to see that the budget is unbounded or that some spend is untracked, even though
    neither halts the run. ConfigurationError is caught here because preflight resolves the
    budget cap from the environment: an invalid KHEDRON_MAX_BUDGET_USD used to surface as an
    unhandled traceback rather than as a user error.
    """
    try:
        findings = preflight_suite(suite_config)
    except ConfigurationError as exc:
        _exit_user_error(f"Preflight configuration error: {exc}")
    for finding in findings:
        typer.echo(str(finding))
    return findings


def _exit_user_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(USER_ERROR)


def _exit_system_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(SYSTEM_ERROR)
