from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import streamlit as st

from khedron.analysis.comparator import ComparisonMode, RunComparator
from khedron.analysis.failure_analyzer import FailureAnalyzer
from khedron.analysis.types import (
    ComparisonReport,
    DetectedPattern,
    FailedQuestionSummary,
    FailureAnalysisReport,
    QuestionDifference,
    ScoreDelta,
    TraceabilityIndexEntry,
)
from khedron.errors import KhedronError
from khedron.persistence.repository import RunRepository
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.reporting.context import ReportContext
from khedron.reporting.generator import ReportContextBuilder, ReportGenerator
from khedron.types import (
    Judgment,
    QuestionRecord,
    Response,
    RetrievalRecord,
    RunStartedEvent,
    RunStatus,
    RunSummary,
    ScoreWithCI,
)

PAGE_NAMES: tuple[str, ...] = (
    "Runs Overview",
    "Run Detail",
    "Compare Runs",
    "Failure Explorer",
    "Settings",
)

ALL_METHODOLOGIES = "All"
SORT_OPTIONS: tuple[str, ...] = (
    "Date desc",
    "Date asc",
    "Audited score desc",
    "Audited score asc",
    "Cost desc",
    "Cost asc",
)
FAILURE_SORT_OPTIONS: tuple[str, ...] = (
    "Question ID asc",
    "Question ID desc",
    "Category asc",
    "Category desc",
    "Verdict asc",
    "Verdict desc",
    "Score asc",
    "Score desc",
    "Cost asc",
    "Cost desc",
)
AUDITED_FILTER_OPTIONS: tuple[str, ...] = ("All", "Audited errors only", "Exclude audited errors")
RETRIEVAL_QUALITY_OPTIONS: tuple[str, ...] = (
    "All",
    "No retrieved memories",
    "Some retrieved memories",
    "Unknown/not attempted",
)
SortOption = Literal[
    "Date desc",
    "Date asc",
    "Audited score desc",
    "Audited score asc",
    "Cost desc",
    "Cost asc",
]


def main() -> None:
    """Render the Khedron Streamlit dashboard."""

    st.set_page_config(page_title="Khedron", layout="wide")
    st.title("Khedron")
    st.caption("Read-only dashboard for persisted benchmark results.")

    sidebar_state = _render_sidebar()
    results_dir = sidebar_state.results_dir
    sqlite_path = sidebar_state.sqlite_path
    resolved_sqlite_path = effective_sqlite_path(results_dir, sqlite_path)

    repository = _cached_repository(str(results_dir), str(resolved_sqlite_path))

    if sidebar_state.page_name == "Runs Overview":
        _render_runs_overview(repository, results_dir)
    elif sidebar_state.page_name == "Run Detail":
        _render_run_detail(repository, results_dir)
    elif sidebar_state.page_name == "Compare Runs":
        _render_compare_runs(repository, results_dir)
    elif sidebar_state.page_name == "Failure Explorer":
        _render_failure_explorer(repository, results_dir)
    elif sidebar_state.page_name == "Settings":
        _render_settings(results_dir, resolved_sqlite_path)
    else:
        _render_placeholder(sidebar_state.page_name)


def create_repository(results_dir: Path, sqlite_path: Path | None = None) -> RunRepository:
    """Create the read repository used by dashboard pages."""

    return RunRepository(results_dir, effective_sqlite_path(results_dir, sqlite_path))


def effective_sqlite_path(results_dir: Path, sqlite_path: Path | None = None) -> Path:
    """Return the explicit SQLite path or the default under the results directory."""

    return sqlite_path if sqlite_path is not None else results_dir / "benchmark.db"


def format_score(score: ScoreWithCI | None) -> str:
    """Format a binary score with its Wilson confidence interval."""

    if score is None:
        return "No score"
    point = _format_percent(score.point_estimate)
    ci_low = _format_percent(score.ci_95_low)
    ci_high = _format_percent(score.ci_95_high)
    return f"{point} [{ci_low}, {ci_high}] Wilson 95% CI ({score.n_correct}/{score.n_total})"


def score_card_value(score: ScoreWithCI | None) -> str:
    """Format a score for the Run Detail metric cards."""

    if score is None:
        return "No score"
    point = _format_percent(score.point_estimate)
    ci_low = _format_percent(score.ci_95_low)
    ci_high = _format_percent(score.ci_95_high)
    return f"{point} (Wilson 95% CI: {ci_low} - {ci_high}; {score.n_correct}/{score.n_total})"


def format_money(value: float) -> str:
    """Format a USD value for dashboard display."""

    return f"${value:,.2f}"


def format_timestamp(value: datetime | None) -> str:
    """Format a timestamp as an explicit UTC value."""

    if value is None:
        return "(none)"
    if value.tzinfo is None:
        utc_value = value.replace(tzinfo=UTC)
    else:
        utc_value = value.astimezone(UTC)
    return utc_value.strftime("%Y-%m-%d %H:%M:%S UTC")


def duration_label(started_at: datetime, finished_at: datetime | None) -> str:
    """Return a readable duration label for a finished or unfinished run."""

    if finished_at is None:
        return "(unfinished)"
    elapsed = max((_utc_datetime(finished_at) - _utc_datetime(started_at)).total_seconds(), 0.0)
    total_seconds = round(elapsed)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def available_statuses(runs: list[RunSummary]) -> list[str]:
    """Return statuses in a stable, task-friendly display order."""

    preferred_order = ("completed", "failed", "partial", "running")
    present = {run.status for run in runs}
    ordered = [status for status in preferred_order if status in present]
    return ordered + sorted(present.difference(preferred_order))


def default_status_selection(statuses: list[str]) -> list[str]:
    """Default to completed runs when available, otherwise every visible status."""

    return ["completed"] if "completed" in statuses else list(statuses)


def available_providers(runs: list[RunSummary]) -> list[str]:
    """Return provider types present in the loaded run summaries."""

    return sorted({run.provider_type for run in runs})


def available_methodologies(runs: list[RunSummary]) -> list[str]:
    """Return methodology version filter options including the all-runs option."""

    versions = sorted({run.methodology_version for run in runs})
    return [ALL_METHODOLOGIES, *versions]


def default_methodology_version(runs: list[RunSummary]) -> str:
    """Return the common methodology, using recency as the tie-breaker."""

    if not runs:
        return ALL_METHODOLOGIES

    counts: dict[str, int] = {}
    latest_seen: dict[str, datetime] = {}
    for run in runs:
        counts[run.methodology_version] = counts.get(run.methodology_version, 0) + 1
        previous = latest_seen.get(run.methodology_version)
        if previous is None or _utc_datetime(run.started_at) > _utc_datetime(previous):
            latest_seen[run.methodology_version] = run.started_at

    return max(
        counts,
        key=lambda version: (counts[version], _utc_datetime(latest_seen[version])),
    )


def started_date_bounds(runs: list[RunSummary]) -> tuple[date, date] | None:
    """Return the full available started_at date span."""

    if not runs:
        return None
    started_dates = [_utc_datetime(run.started_at).date() for run in runs]
    return min(started_dates), max(started_dates)


def normalize_date_range(
    value: date | tuple[date, ...] | list[date],
    default_start: date,
    default_end: date,
) -> tuple[date, date]:
    """Normalize Streamlit date input values to an inclusive date range."""

    if isinstance(value, date):
        return value, value
    if len(value) >= 2:
        start, end = value[0], value[1]
        return (start, end) if start <= end else (end, start)
    return default_start, default_end


def filter_runs(
    runs: list[RunSummary],
    *,
    statuses: list[str],
    providers: list[str],
    methodology_version: str,
    started_date_range: tuple[date, date],
) -> list[RunSummary]:
    """Filter run summaries in memory for the Runs Overview page."""

    start_date, end_date = started_date_range
    selected_statuses = set(statuses)
    selected_providers = set(providers)

    return [
        run
        for run in runs
        if run.status in selected_statuses
        and run.provider_type in selected_providers
        and (
            methodology_version == ALL_METHODOLOGIES
            or run.methodology_version == methodology_version
        )
        and start_date <= _utc_datetime(run.started_at).date() <= end_date
    ]


def sort_runs(runs: list[RunSummary], sort_option: str) -> list[RunSummary]:
    """Sort run summaries using the supported Runs Overview sort options."""

    option = _coerce_sort_option(sort_option)
    if option == "Date desc":
        return sorted(runs, key=lambda run: _utc_datetime(run.started_at), reverse=True)
    if option == "Date asc":
        return sorted(runs, key=lambda run: _utc_datetime(run.started_at))
    if option == "Audited score desc":
        return sorted(
            runs,
            key=lambda run: _nullable_score_sort_key(run.overall_score_audited, descending=True),
        )
    if option == "Audited score asc":
        return sorted(
            runs,
            key=lambda run: _nullable_score_sort_key(run.overall_score_audited, descending=False),
        )
    if option == "Cost desc":
        return sorted(runs, key=lambda run: run.total_cost_usd, reverse=True)
    return sorted(runs, key=lambda run: run.total_cost_usd)


def dashboard_mean_audited_score(runs: list[RunSummary]) -> float | None:
    """Return the orientation-only mean audited score over visible scored rows."""

    scores = [run.overall_score_audited for run in runs if run.overall_score_audited is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def total_run_cost(runs: list[RunSummary]) -> float:
    """Return total visible run cost."""

    return sum(run.total_cost_usd for run in runs)


def most_recent_started_at(runs: list[RunSummary]) -> datetime | None:
    """Return the most recent visible run timestamp."""

    if not runs:
        return None
    return max((run.started_at for run in runs), key=_utc_datetime)


def run_table_rows(runs: list[RunSummary]) -> list[dict[str, str]]:
    """Format run summaries for the Runs Overview table."""

    return [
        {
            "Started": format_timestamp(run.started_at),
            "Experiment": run.experiment_name,
            "Run ID": run.run_id,
            "Status": run.status,
            "Provider": _provider_label(run),
            "Benchmark": run.benchmark_type,
            "Models": f"{run.answer_model_id} / {run.judge_model_id}",
            "Audited Score": _format_optional_percent(run.overall_score_audited),
            "Standard Score": _format_optional_percent(run.overall_score_standard),
            "Questions": _questions_label(run),
            "Cost": format_money(run.total_cost_usd),
        }
        for run in runs
    ]


def run_detail_options(runs: list[RunSummary]) -> dict[str, str]:
    """Return readable Run Detail selector labels mapped to run IDs."""

    ordered_runs = sorted(
        runs,
        key=lambda run: (
            -_utc_datetime(run.started_at).timestamp(),
            run.experiment_name.casefold(),
            run.run_id,
        ),
    )
    return {
        (
            f"{format_timestamp(run.started_at)} | {run.experiment_name} | "
            f"{run.run_id} | {run.status}"
        ): run.run_id
        for run in ordered_runs
    }


def compare_run_options(runs: list[RunSummary]) -> dict[str, str]:
    """Return readable Compare Runs selector labels mapped to run IDs."""

    return run_detail_options(runs)


def failure_run_options(runs: list[RunSummary]) -> dict[str, str]:
    """Return readable Failure Explorer selector labels mapped to run IDs."""

    return run_detail_options(runs)


def category_score_rows(scores: dict[str, ScoreWithCI]) -> list[dict[str, object]]:
    """Format per-category score objects for charting and display."""

    rows: list[dict[str, object]] = []
    for category, score in sorted(scores.items()):
        point = score.point_estimate * 100
        ci_low = score.ci_95_low * 100
        ci_high = score.ci_95_high * 100
        rows.append(
            {
                "Category": category,
                "Score": _format_percent(score.point_estimate),
                "Wilson 95% CI": f"[{_format_percent(score.ci_95_low)}, "
                f"{_format_percent(score.ci_95_high)}]",
                "Correct": f"{score.n_correct}/{score.n_total}",
                "point_percent": point,
                "ci_low_percent": ci_low,
                "ci_high_percent": ci_high,
                "error_plus": max(ci_high - point, 0.0),
                "error_minus": max(point - ci_low, 0.0),
            }
        )
    return rows


def cost_phase_rows(context: ReportContext) -> list[dict[str, str]]:
    """Format phase-level cost rows from a report context."""

    return _cost_breakdown_rows(
        context.cost_summary.cost_by_phase,
        total=context.cost_summary.total_cost_usd,
        preferred_order=(
            "ingestion",
            "ingest",
            "retrieval",
            "retrieve",
            "generation",
            "generate",
            "judgment",
            "judge",
            "other",
        ),
    )


def cost_model_rows(context: ReportContext) -> list[dict[str, str]]:
    """Format model-level cost rows from a report context."""

    return _cost_breakdown_rows(
        context.cost_summary.cost_by_model,
        total=context.cost_summary.total_cost_usd,
        preferred_order=(),
        preserve_names=True,
    )


def top_failed_question_rows(
    context: ReportContext,
    limit: int = 10,
) -> list[dict[str, str]]:
    """Format failed-question summaries with a stable maximum row count."""

    return [
        {
            "Question ID": question.question_id,
            "Category": _display_value(question.category),
            "Verdict": _display_value(question.verdict),
            "Question": _truncate(question.question_text, 96),
            "Error phase": question.error_phase or "(none)",
            "Error": _truncate(question.error_message or "", 96) or "(none)",
        }
        for question in context.failure_analysis.failed_questions[:limit]
    ]


def failure_category_rows(context: ReportContext) -> list[dict[str, str]]:
    """Format failure counts by question category."""

    return [
        {
            "Category": _display_value(row.category),
            "Failures": str(row.n_failed),
            "Total": str(row.n_total),
            "Failure rate": _format_percent(row.failure_rate),
        }
        for row in context.failure_analysis.category_breakdown
    ]


def top_pattern_rows(context: ReportContext, limit: int = 3) -> list[dict[str, str]]:
    """Format top detected failure patterns."""

    patterns = sorted(
        context.failure_analysis.patterns,
        key=lambda pattern: (-pattern.n_affected_questions, pattern.pattern_name),
    )
    return [
        {
            "Pattern": pattern.pattern_name,
            "Affected questions": str(pattern.n_affected_questions),
            "Confidence": _format_percent(pattern.confidence),
            "Suggested remedy": pattern.suggested_remedy,
        }
        for pattern in patterns[:limit]
    ]


def same_vendor_warning_text(context: ReportContext) -> str | None:
    """Return the same-vendor warning text when answer and judge vendors match."""

    if not context.methodology.same_vendor_warning:
        return None
    return (
        "Answer model and judge model use the same vendor. "
        "Interpret judged scores with this methodology disclosure."
    )


def score_delta_rows(deltas: list[ScoreDelta]) -> list[dict[str, str]]:
    """Format comparator score deltas for the Compare Runs table."""

    return [
        {
            "Metric": _comparison_category_label(delta.category),
            "Mode": delta.mode,
            "Baseline": _score_cell(delta.baseline_score),
            "Candidate": _score_cell(delta.candidate_score),
            "Delta": _delta_cell(delta.point_delta),
            "CI overlap": _ci_overlap_label(delta.ci_overlaps),
            "Significance": _significance_label(delta.statistically_significant),
        }
        for delta in deltas
    ]


def comparison_chart_rows(comparison: ComparisonReport) -> list[dict[str, object]]:
    """Prepare grouped per-category comparison chart rows with CI error values."""

    rows: list[dict[str, object]] = []
    baseline_run_id, candidate_run_id = comparison.run_ids
    for delta in comparison.score_deltas:
        if delta.category == "overall":
            continue
        category = _comparison_category_label(delta.category)
        for run_id, label, score in (
            (baseline_run_id, "Baseline", delta.baseline_score),
            (candidate_run_id, "Candidate", delta.candidate_score),
        ):
            if score is None:
                continue
            point = score.point_estimate * 100
            ci_low = score.ci_95_low * 100
            ci_high = score.ci_95_high * 100
            rows.append(
                {
                    "Category": category,
                    "Run": label,
                    "Run ID": run_id,
                    "point_percent": point,
                    "error_plus": max(ci_high - point, 0.0),
                    "error_minus": max(point - ci_low, 0.0),
                }
            )
    return rows


def question_difference_rows(
    differences: list[QuestionDifference],
    baseline_run_id: str,
    candidate_run_id: str,
) -> list[dict[str, str]]:
    """Format question-level comparison differences."""

    return [
        {
            "Question ID": difference.question_id,
            "Category": difference.category.value,
            "Baseline verdict": difference.verdicts_by_run_id[baseline_run_id].value,
            "Baseline score": _format_score_value(
                difference.scores_by_run_id[baseline_run_id],
            ),
            "Candidate verdict": difference.verdicts_by_run_id[candidate_run_id].value,
            "Candidate score": _format_score_value(
                difference.scores_by_run_id[candidate_run_id],
            ),
        }
        for difference in differences
    ]


def configuration_difference_rows(
    baseline_status: RunStatus,
    candidate_status: RunStatus,
    baseline_started: RunStartedEvent,
    candidate_started: RunStartedEvent,
) -> list[dict[str, str]]:
    """Format comparison-relevant configuration values from public repository reads."""

    return [
        _configuration_row(
            "Provider",
            _provider_config_label(baseline_started),
            _provider_config_label(candidate_started),
        ),
        _configuration_row(
            "Benchmark",
            _benchmark_config_label(baseline_started),
            _benchmark_config_label(candidate_started),
        ),
        _configuration_row(
            "Answer model",
            baseline_started.answer_model_id,
            candidate_started.answer_model_id,
        ),
        _configuration_row(
            "Judge model",
            baseline_started.judge_model_id,
            candidate_started.judge_model_id,
        ),
        _configuration_row(
            "Methodology version",
            baseline_status.methodology_version,
            candidate_status.methodology_version,
        ),
        _configuration_row(
            "Methodology profile",
            baseline_status.methodology_profile,
            candidate_status.methodology_profile,
        ),
        _configuration_row(
            "Benchmark checksum",
            baseline_started.benchmark_checksum,
            candidate_started.benchmark_checksum,
        ),
        # First on the documented list of "common surprises" when two runs disagree, and formerly
        # absent from this section -- so the documentation told an analyst to check a field the
        # page never showed. Now that the value identifies a commit rather than a placeholder, a
        # difference here is readable.
        _configuration_row(
            "Framework version",
            baseline_started.framework_version,
            candidate_started.framework_version,
        ),
    ]


def failure_pattern_names_by_question(
    patterns: list[DetectedPattern],
) -> dict[str, list[str]]:
    """Map question IDs to detected pattern names."""

    names_by_question: dict[str, list[str]] = {}
    for pattern in sorted(patterns, key=lambda item: item.pattern_name):
        for question_id in pattern.affected_question_ids:
            names_by_question.setdefault(question_id, []).append(pattern.pattern_name)
    return names_by_question


def failure_trace_ids_by_question(
    traceability_index: list[TraceabilityIndexEntry],
) -> dict[str, TraceabilityIndexEntry]:
    """Map question IDs to traceability entries."""

    return {entry.question_id: entry for entry in traceability_index}


def failure_rows(
    failures: list[FailedQuestionSummary],
    questions: list[QuestionRecord],
    patterns: list[DetectedPattern],
) -> list[dict[str, object]]:
    """Join failure summaries with question context for filtering and display."""

    questions_by_id = {question.question_id: question for question in questions}
    patterns_by_question = failure_pattern_names_by_question(patterns)
    rows: list[dict[str, object]] = []
    for failure in failures:
        question = questions_by_id.get(failure.question_id)
        rows.append(
            {
                "question_id": failure.question_id,
                "category": _display_value(failure.category),
                "verdict": _display_value(failure.verdict),
                "score": question.score if question is not None else None,
                "is_audited_error": question.is_audited_error if question is not None else False,
                "retrieved_memory_count": (
                    question.n_memories_retrieved if question is not None else None
                ),
                "retrieval_quality": _retrieval_quality_label(question),
                "pattern_names": patterns_by_question.get(failure.question_id, []),
                "question_text": failure.question_text,
                "expected_answer": failure.expected_answer,
                "generated_answer": failure.generated_answer,
                "error_phase": failure.error_phase,
                "error_message": failure.error_message,
                "total_cost_usd": question.total_cost_usd if question is not None else None,
            }
        )
    return rows


def filter_failure_rows(
    rows: list[dict[str, object]],
    *,
    categories: list[str],
    verdicts: list[str],
    patterns: list[str],
    audited_filter: str,
    retrieval_quality: str,
    search: str,
) -> list[dict[str, object]]:
    """Filter Failure Explorer rows in memory."""

    selected_categories = set(categories)
    selected_verdicts = set(verdicts)
    selected_patterns = set(patterns)
    needle = search.casefold().strip()

    filtered: list[dict[str, object]] = []
    for row in rows:
        if selected_categories and row["category"] not in selected_categories:
            continue
        if selected_verdicts and row["verdict"] not in selected_verdicts:
            continue
        row_patterns = set(_object_list(row["pattern_names"]))
        if selected_patterns and row_patterns.isdisjoint(selected_patterns):
            continue
        if audited_filter == "Audited errors only" and row["is_audited_error"] is not True:
            continue
        if audited_filter == "Exclude audited errors" and row["is_audited_error"] is True:
            continue
        if retrieval_quality != "All" and row["retrieval_quality"] != retrieval_quality:
            continue
        if needle and needle not in _failure_row_search_text(row):
            continue
        filtered.append(row)
    return filtered


def sort_failure_rows(
    rows: list[dict[str, object]],
    sort_option: str,
) -> list[dict[str, object]]:
    """Sort Failure Explorer rows with missing numeric values last."""

    reverse = sort_option.endswith("desc")
    field_by_option = {
        "Question ID asc": "question_id",
        "Question ID desc": "question_id",
        "Category asc": "category",
        "Category desc": "category",
        "Verdict asc": "verdict",
        "Verdict desc": "verdict",
        "Score asc": "score",
        "Score desc": "score",
        "Cost asc": "total_cost_usd",
        "Cost desc": "total_cost_usd",
    }
    field = field_by_option.get(sort_option, "question_id")
    if field in {"score", "total_cost_usd"}:
        return sorted(
            rows,
            key=lambda row: _nullable_number_sort_key(row[field], descending=reverse),
        )
    return sorted(rows, key=lambda row: str(row[field]).casefold(), reverse=reverse)


def visible_failure_table_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Format visible failure rows for Streamlit tables and CSV export."""

    return [
        {
            "Question ID": str(row["question_id"]),
            "Category": str(row["category"]),
            "Verdict": str(row["verdict"]),
            "Score": _format_nullable_score(row["score"]),
            "Audited error": "yes" if row["is_audited_error"] is True else "no",
            "Retrieved memories": _format_optional_int(row["retrieved_memory_count"]),
            "Patterns": ", ".join(_object_list(row["pattern_names"])) or "(none)",
            "Question": str(row["question_text"]),
            "Expected answer": str(row["expected_answer"]),
            "Generated answer": _display_optional_text(row["generated_answer"]),
            "Error phase": _display_optional_text(row["error_phase"]),
            "Error message": _display_optional_text(row["error_message"]),
            "Total cost": _format_optional_money(row["total_cost_usd"]),
        }
        for row in rows
    ]


def failure_csv(rows: list[dict[str, object]]) -> str:
    """Build an in-memory CSV payload for visible failures."""

    table_rows = visible_failure_table_rows(rows)
    if not table_rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(table_rows[0]))
    writer.writeheader()
    writer.writerows(table_rows)
    return buffer.getvalue()


def visible_question_ids(rows: list[dict[str, object]]) -> str:
    """Return a newline-delimited block of visible question IDs."""

    return "\n".join(str(row["question_id"]) for row in rows)


class _SidebarState:
    def __init__(
        self,
        *,
        results_dir: Path,
        sqlite_path: Path | None,
        page_name: str,
    ) -> None:
        self.results_dir = results_dir
        self.sqlite_path = sqlite_path
        self.page_name = page_name


@st.cache_resource(show_spinner=False)
def _cached_repository(results_dir: str, sqlite_path: str) -> RunRepository:
    return create_repository(Path(results_dir), Path(sqlite_path))


def _render_sidebar() -> _SidebarState:
    with st.sidebar:
        st.header("Navigation")
        results_dir_text = st.text_input("Results directory", value="results")
        sqlite_path_text = st.text_input("SQLite path (optional)", value="")

        if st.button("Refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        page_name = st.selectbox("Page", PAGE_NAMES)
        st.divider()
        st.caption("Dashboard reads persisted results only.")

    sqlite_path = Path(sqlite_path_text).expanduser() if sqlite_path_text.strip() else None
    return _SidebarState(
        results_dir=Path(results_dir_text).expanduser(),
        sqlite_path=sqlite_path,
        page_name=page_name,
    )


def _render_runs_overview(repository: RunRepository, results_dir: Path) -> None:
    st.header("Runs Overview")

    try:
        with st.spinner("Loading runs..."):
            runs = repository.list_runs()
    except KhedronError as exc:
        if _has_no_source_runs(results_dir):
            st.info("No runs found. Persisted benchmark runs will appear here.")
            return
        st.warning(f"Unable to load runs from the SQLite index: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error: {exc}")
        return

    if not runs:
        st.info("No runs found. Persisted benchmark runs will appear here.")
        return

    filtered_runs = _render_runs_overview_filters(runs)
    sorted_runs = sort_runs(filtered_runs, _selected_sort_option())
    _render_runs_overview_kpis(sorted_runs)

    st.write(f"Showing {len(sorted_runs)} of {len(runs)} runs")
    if not sorted_runs:
        st.info(
            "No runs match these filters. Try widening the date range, "
            "including more statuses, or selecting all methodologies."
        )
        return

    st.dataframe(run_table_rows(sorted_runs), hide_index=True, use_container_width=True)


def _render_run_detail(repository: RunRepository, results_dir: Path) -> None:
    st.header("Run Detail")

    try:
        with st.spinner("Loading runs..."):
            runs = repository.list_runs()
    except KhedronError as exc:
        if _has_no_source_runs(results_dir):
            st.info("No runs found. Persisted benchmark runs will appear here.")
            return
        st.warning(f"Unable to load runs from the SQLite index: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error: {exc}")
        return

    if not runs:
        st.info("No runs found. Persisted benchmark runs will appear here.")
        return

    options = run_detail_options(runs)
    labels = list(options)
    selected_run_id = st.session_state.get("selected_run_id")
    selected_index = 0
    if isinstance(selected_run_id, str) and selected_run_id in options.values():
        selected_index = list(options.values()).index(selected_run_id)
    selected_label = st.selectbox("Run", labels, index=selected_index)
    run_id = options[selected_label]
    st.session_state["selected_run_id"] = run_id

    try:
        with st.spinner("Loading run detail..."):
            context = ReportContextBuilder().build(run_id, repository)
    except KhedronError as exc:
        st.warning(f"Unable to build run detail for `{run_id}`: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error while loading `{run_id}`: {exc}")
        return

    _render_run_identity(context)
    _render_score_cards(context)
    _render_category_scores(context)
    _render_cost_breakdown(context)
    _render_failure_summary(context)
    _render_configuration(context)


def _render_compare_runs(repository: RunRepository, results_dir: Path) -> None:
    st.header("Compare Runs")

    try:
        with st.spinner("Loading runs..."):
            runs = repository.list_runs()
    except KhedronError as exc:
        if _has_no_source_runs(results_dir):
            st.info("No runs found. Persisted benchmark runs will appear here.")
            return
        st.warning(f"Unable to load runs from the SQLite index: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error: {exc}")
        return

    if len(runs) < 2:
        st.info("At least two persisted runs are required before comparisons are available.")
        return

    options = compare_run_options(runs)
    labels = list(options)
    baseline_label = st.selectbox("Baseline run", labels, index=0)
    candidate_label = st.selectbox("Candidate run", labels, index=1)
    mode: ComparisonMode = st.radio(
        "Comparison mode",
        ("audited", "standard"),
        horizontal=True,
        index=0,
    )
    baseline_run_id = options[baseline_label]
    candidate_run_id = options[candidate_label]

    if baseline_run_id == candidate_run_id:
        st.warning("Choose two distinct run IDs before comparing.")
        return

    try:
        with st.spinner("Building comparison..."):
            comparison = RunComparator().compare(
                [baseline_run_id, candidate_run_id],
                repository,
                mode,
            )
            baseline_status = repository.get_run_status(baseline_run_id)
            candidate_status = repository.get_run_status(candidate_run_id)
            baseline_started = repository.get_run_started_event(baseline_run_id)
            candidate_started = repository.get_run_started_event(candidate_run_id)
    except KhedronError as exc:
        st.warning(
            "Comparison withheld: the selected runs are ineligible or cannot be compared safely. "
            f"No score deltas or export were produced. Details: {exc}"
        )
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error while comparing runs: {exc}")
        return

    _render_comparison_compatibility(comparison)
    _render_comparison_scores(comparison)
    _render_comparison_chart(comparison)
    _render_question_differences(comparison)
    _render_configuration_differences(
        baseline_status,
        candidate_status,
        baseline_started,
        candidate_started,
    )
    _render_comparison_export(comparison, baseline_run_id, candidate_run_id)


def _render_failure_explorer(repository: RunRepository, results_dir: Path) -> None:
    st.header("Failure Explorer")

    try:
        with st.spinner("Loading runs..."):
            runs = repository.list_runs()
    except KhedronError as exc:
        if _has_no_source_runs(results_dir):
            st.info("No runs found. Persisted benchmark runs will appear here.")
            return
        st.warning(f"Unable to load runs from the SQLite index: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error: {exc}")
        return

    if not runs:
        st.info("No runs found. Persisted benchmark runs will appear here.")
        return

    options = failure_run_options(runs)
    labels = list(options)
    selected_run_id = st.session_state.get("selected_run_id")
    selected_index = 0
    if isinstance(selected_run_id, str) and selected_run_id in options.values():
        selected_index = list(options.values()).index(selected_run_id)
    selected_label = st.selectbox("Run", labels, index=selected_index, key="failure_run")
    run_id = options[selected_label]
    st.session_state["selected_run_id"] = run_id

    try:
        with st.spinner("Analyzing failures..."):
            questions = repository.get_questions_for_run(run_id)
            analysis = FailureAnalyzer().analyze(run_id, repository)
    except KhedronError as exc:
        st.warning(f"Unable to build failure analysis for `{run_id}`: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected dashboard read error while loading `{run_id}`: {exc}")
        return

    rows = failure_rows(analysis.failed_questions, questions, analysis.patterns)
    if not rows:
        st.success("No failed questions are recorded for this run.")
        return

    visible_rows = _render_failure_filters(rows, analysis.patterns)
    sorted_rows = sort_failure_rows(visible_rows, _selected_failure_sort_option())
    st.write(f"Showing {len(sorted_rows)} of {len(rows)} failed questions")
    _render_failure_bulk_actions(sorted_rows, run_id)

    if not sorted_rows:
        st.info("No failures match these filters. Clear filters or broaden the search.")
        return

    st.dataframe(
        visible_failure_table_rows(sorted_rows),
        hide_index=True,
        use_container_width=True,
    )
    _render_failure_expanders(sorted_rows, analysis, repository, run_id)


def _render_runs_overview_filters(runs: list[RunSummary]) -> list[RunSummary]:
    statuses = available_statuses(runs)
    providers = available_providers(runs)
    methodologies = available_methodologies(runs)
    methodology_default = default_methodology_version(runs)
    bounds = started_date_bounds(runs)
    if bounds is None:
        return []
    default_start, default_end = bounds

    with st.sidebar:
        st.divider()
        st.header("Filters")
        selected_statuses = st.multiselect(
            "Status",
            options=statuses,
            default=default_status_selection(statuses),
        )
        selected_providers = st.multiselect(
            "Provider",
            options=providers,
            default=providers,
        )
        selected_methodology = st.selectbox(
            "Methodology version",
            options=methodologies,
            index=methodologies.index(methodology_default),
        )
        selected_dates = st.date_input(
            "Started date range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
        )
        st.selectbox("Sort by", SORT_OPTIONS, key="runs_overview_sort")

    return filter_runs(
        runs,
        statuses=selected_statuses,
        providers=selected_providers,
        methodology_version=selected_methodology,
        started_date_range=normalize_date_range(selected_dates, default_start, default_end),
    )


def _render_failure_filters(
    rows: list[dict[str, object]],
    patterns: list[DetectedPattern],
) -> list[dict[str, object]]:
    category_options = sorted({str(row["category"]) for row in rows})
    verdict_options = sorted({str(row["verdict"]) for row in rows})
    pattern_options = sorted({pattern.pattern_name for pattern in patterns})

    with st.sidebar:
        st.divider()
        st.header("Failure Filters")
        selected_categories = st.multiselect(
            "Category",
            options=category_options,
            default=category_options,
            key="failure_categories",
        )
        selected_verdicts = st.multiselect(
            "Verdict",
            options=verdict_options,
            default=verdict_options,
            key="failure_verdicts",
        )
        selected_patterns = st.multiselect(
            "Pattern",
            options=pattern_options,
            default=[],
            key="failure_patterns",
        )
        audited_filter = st.radio(
            "Audited flag",
            AUDITED_FILTER_OPTIONS,
            horizontal=False,
            key="failure_audited_filter",
        )
        retrieval_quality = st.selectbox(
            "Retrieval quality",
            RETRIEVAL_QUALITY_OPTIONS,
            key="failure_retrieval_quality",
        )
        search = st.text_input("Search failures", key="failure_search")
        st.selectbox("Failure sort by", FAILURE_SORT_OPTIONS, key="failure_sort")

    return filter_failure_rows(
        rows,
        categories=selected_categories,
        verdicts=selected_verdicts,
        patterns=selected_patterns,
        audited_filter=audited_filter,
        retrieval_quality=retrieval_quality,
        search=search,
    )


def _selected_failure_sort_option() -> str:
    value = st.session_state.get("failure_sort", FAILURE_SORT_OPTIONS[0])
    if isinstance(value, str) and value in FAILURE_SORT_OPTIONS:
        return value
    return FAILURE_SORT_OPTIONS[0]


def _selected_sort_option() -> SortOption:
    value = st.session_state.get("runs_overview_sort", SORT_OPTIONS[0])
    return _coerce_sort_option(value)


def _render_runs_overview_kpis(runs: list[RunSummary]) -> None:
    total_runs, mean_score, cost, most_recent = st.columns(4)

    total_runs.metric("Total runs", str(len(runs)))
    mean_score.metric(
        "Dashboard mean audited score",
        _format_optional_percent(dashboard_mean_audited_score(runs)),
    )
    cost.metric("Total cost", format_money(total_run_cost(runs)))
    most_recent.metric("Most recent", format_timestamp(most_recent_started_at(runs)))


def _render_failure_bulk_actions(rows: list[dict[str, object]], run_id: str) -> None:
    st.subheader("Bulk Actions")
    csv_payload = failure_csv(rows)
    st.download_button(
        "Download visible failures as CSV",
        data=csv_payload,
        file_name=f"failures-{run_id}.csv",
        mime="text/csv",
        disabled=not rows,
    )
    st.text_area(
        "Visible question IDs",
        value=visible_question_ids(rows),
        height=120,
        help="Select and copy these IDs for follow-up analysis.",
    )


def _render_failure_expanders(
    rows: list[dict[str, object]],
    analysis: FailureAnalysisReport,
    repository: RunRepository,
    run_id: str,
) -> None:
    st.subheader("Failure Details")
    trace_by_question = failure_trace_ids_by_question(analysis.traceability_index)
    selected_detail = st.session_state.get("failure_detail_question_id")
    for row in rows:
        question_id = str(row["question_id"])
        label = (
            f"{question_id} | {row['category']} | {row['verdict']} | "
            f"score {_format_nullable_score(row['score'])}"
        )
        with st.expander(label):
            if st.button("Load trace", key=f"failure_load_trace_{question_id}"):
                st.session_state["failure_detail_question_id"] = question_id
                selected_detail = question_id
            if selected_detail == question_id:
                _render_hydrated_failure_detail(
                    run_id,
                    question_id,
                    trace_by_question.get(question_id),
                    repository,
                )
            else:
                st.info("Trace hydration is loaded on demand for the selected failure.")


def _render_hydrated_failure_detail(
    run_id: str,
    question_id: str,
    trace: TraceabilityIndexEntry | None,
    repository: RunRepository,
) -> None:
    question = _safe_get_question_record(repository, run_id, question_id)
    retrieval = _safe_get_retrieval(repository, trace.retrieval_id if trace is not None else None)
    response = _safe_get_response(repository, trace.response_id if trace is not None else None)
    judgment = _safe_get_judgment(repository, trace.judgment_id if trace is not None else None)

    if question is None:
        st.warning("Question record could not be hydrated from the repository.")
        return

    st.write("Question")
    st.table(
        [
            {"Field": "Question ID", "Value": question.question_id},
            {"Field": "Category", "Value": _display_value(question.category)},
            {"Field": "Question", "Value": question.question_text},
            {"Field": "Expected answer", "Value": question.expected_answer},
            {
                "Field": "Generated answer",
                "Value": _display_optional_text(question.generated_answer),
            },
            {"Field": "Verdict", "Value": _display_value(question.verdict)},
            {"Field": "Score", "Value": _format_score_value(question.score)},
            {"Field": "Audited error", "Value": "yes" if question.is_audited_error else "no"},
            {"Field": "Error phase", "Value": _display_optional_text(question.error_phase)},
            {"Field": "Error message", "Value": _display_optional_text(question.error_message)},
        ]
    )

    _render_latency_cost_fields(question)
    _render_retrieval_detail(question, trace, retrieval)
    _render_response_detail(trace, response)
    _render_judgment_detail(trace, judgment)


def _render_latency_cost_fields(question: QuestionRecord) -> None:
    st.write("Latency, Tokens, and Cost")
    st.table(
        [
            {
                "Field": "Retrieval latency",
                "Value": _format_optional_latency(question.retrieval_latency_ms),
            },
            {
                "Field": "Generation latency",
                "Value": _format_optional_latency(question.generation_latency_ms),
            },
            {
                "Field": "Judgment latency",
                "Value": _format_optional_latency(question.judgment_latency_ms),
            },
            {
                "Field": "Total latency",
                "Value": _format_optional_latency(question.total_latency_ms),
            },
            {
                "Field": "Generation tokens",
                "Value": _token_pair(
                    question.generation_input_tokens,
                    question.generation_output_tokens,
                ),
            },
            {
                "Field": "Judgment tokens",
                "Value": _token_pair(
                    question.judgment_input_tokens, question.judgment_output_tokens
                ),
            },
            {
                "Field": "Generation cost",
                "Value": _format_optional_money(question.generation_cost_usd),
            },
            {
                "Field": "Judgment cost",
                "Value": _format_optional_money(question.judgment_cost_usd),
            },
            {"Field": "Total cost", "Value": format_money(question.total_cost_usd)},
        ]
    )


def _render_retrieval_detail(
    question: QuestionRecord,
    trace: TraceabilityIndexEntry | None,
    retrieval: RetrievalRecord | None,
) -> None:
    st.write("Retrieved Memories")
    if trace is None or trace.retrieval_id is None:
        st.info("No retrieval ID is recorded for this question.")
        return
    if retrieval is None:
        st.warning(f"Retrieval `{trace.retrieval_id}` could not be hydrated.")
        return
    if not retrieval.memories:
        st.info("Retrieval completed but returned no memories.")
        return

    st.caption(
        f"Query: {retrieval.query} | top_k={retrieval.top_k} | returned={retrieval.n_returned} | "
        f"recorded count={_format_optional_int(question.n_memories_retrieved)}"
    )
    for memory in retrieval.memories:
        st.markdown(f"**{memory.memory_id}**")
        st.table(
            [
                {"Field": "Score", "Value": _format_optional_number(memory.score)},
                {"Field": "Timestamp", "Value": format_timestamp(memory.timestamp)},
                {"Field": "Metadata", "Value": _jsonish(memory.metadata)},
            ]
        )
        st.write(memory.content)


def _render_response_detail(
    trace: TraceabilityIndexEntry | None,
    response: Response | None,
) -> None:
    st.write("Generation")
    if trace is None or trace.response_id is None:
        st.info("No response ID is recorded for this question.")
        return
    if response is None:
        st.warning(f"Response `{trace.response_id}` could not be hydrated.")
        return

    st.caption(
        f"Model: {response.model_id} | latency={_format_optional_latency(response.latency_ms)} | "
        f"tokens={_token_pair(response.input_tokens, response.output_tokens)} | "
        f"cost={format_money(response.cost_usd)}"
    )
    st.text_area("Generation prompt", value=response.prompt, height=180)
    st.text_area("Model answer", value=response.answer_text, height=120)


def _render_judgment_detail(
    trace: TraceabilityIndexEntry | None,
    judgment: Judgment | None,
) -> None:
    st.write("Judgment")
    if trace is None or trace.judgment_id is None:
        st.info("No judgment ID is recorded for this question.")
        return
    if judgment is None:
        st.warning(f"Judgment `{trace.judgment_id}` could not be hydrated.")
        return

    score = _format_score_value(judgment.parsed_score)
    st.caption(
        f"Model: {judgment.judge_model_id} | parsed={judgment.parse_was_successful} | "
        f"verdict={judgment.parsed_verdict.value} | score={score} | "
        f"latency={_format_optional_latency(judgment.latency_ms)} | "
        f"tokens={_token_pair(judgment.input_tokens, judgment.output_tokens)} | "
        f"cost={format_money(judgment.cost_usd)}"
    )
    st.text_area("Judgment prompt", value=judgment.prompt, height=180)
    st.text_area("Raw judge output", value=judgment.raw_judge_output, height=120)
    st.text_area("Judgment reasoning", value=judgment.parsed_reasoning, height=120)


def _render_settings(results_dir: Path, sqlite_path: Path) -> None:
    st.header("Settings")
    schema_version = _read_schema_version(sqlite_path)

    st.subheader("Storage")
    st.write(f"Results dir: `{results_dir}`")
    st.write(f"SQLite path: `{sqlite_path}`")
    if schema_version is None:
        st.warning("SQLite schema version is not readable.")
    else:
        st.write(f"Schema version: `{schema_version}`")

    st.subheader("Pages")
    st.write(", ".join(PAGE_NAMES))


def _render_comparison_compatibility(comparison: ComparisonReport) -> None:
    st.subheader("Compatibility")
    if comparison.compatible:
        st.success("Compatible: direct comparison is valid.")
        return
    st.warning("Incompatible: review these warnings before interpreting deltas.")
    for warning in comparison.compatibility_warnings:
        st.write(f"- {warning}")


def _render_comparison_scores(comparison: ComparisonReport) -> None:
    st.subheader("Score Deltas")
    rows = score_delta_rows(comparison.score_deltas)
    if not rows:
        st.info("No score deltas are available for these runs.")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_comparison_chart(comparison: ComparisonReport) -> None:
    st.subheader("Per-Category Comparison")
    rows = comparison_chart_rows(comparison)
    if not rows:
        st.info("No per-category scores are available for charting.")
        return

    baseline_rows = [row for row in rows if row["Run"] == "Baseline"]
    candidate_rows = [row for row in rows if row["Run"] == "Candidate"]
    plotly_go = cast(Any, import_module("plotly.graph_objects"))
    fig = plotly_go.Figure()
    for label, group_rows in (("Baseline", baseline_rows), ("Candidate", candidate_rows)):
        add_bar = fig.add_bar
        add_bar(
            name=label,
            x=[row["point_percent"] for row in group_rows],
            y=[row["Category"] for row in group_rows],
            orientation="h",
            error_x={
                "type": "data",
                "array": [row["error_plus"] for row in group_rows],
                "arrayminus": [row["error_minus"] for row in group_rows],
                "visible": True,
            },
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        )
    update_layout = fig.update_layout
    update_layout(
        barmode="group",
        xaxis_title="Score",
        yaxis_title="Category",
        xaxis={"range": [0, 100], "ticksuffix": "%"},
        margin={"l": 120, "r": 24, "t": 8, "b": 48},
    )
    st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]


def _render_question_differences(comparison: ComparisonReport) -> None:
    st.subheader("Question Differences")
    baseline_run_id, candidate_run_id = comparison.run_ids
    rows = question_difference_rows(
        comparison.differing_questions,
        baseline_run_id,
        candidate_run_id,
    )
    if not rows:
        st.success("No question-level verdict or score differences were detected.")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_configuration_differences(
    baseline_status: RunStatus,
    candidate_status: RunStatus,
    baseline_started: RunStartedEvent,
    candidate_started: RunStartedEvent,
) -> None:
    st.subheader("Configuration Differences")
    st.dataframe(
        configuration_difference_rows(
            baseline_status,
            candidate_status,
            baseline_started,
            candidate_started,
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_comparison_export(
    comparison: ComparisonReport,
    baseline_run_id: str,
    candidate_run_id: str,
) -> None:
    st.subheader("Export")
    markdown = ReportGenerator().render_comparison_report(comparison)
    file_name = f"comparison-{baseline_run_id}-vs-{candidate_run_id}.md"
    st.download_button(
        "Export as Markdown",
        data=markdown,
        file_name=file_name,
        mime="text/markdown",
    )


def _render_run_identity(context: ReportContext) -> None:
    status = context.run_status
    st.subheader(status.experiment_name)
    st.table(
        [
            {"Field": "Run ID", "Value": status.run_id},
            {"Field": "Suite ID", "Value": status.suite_id},
            {"Field": "Experiment ID", "Value": status.experiment_id},
            {"Field": "Run number", "Value": str(status.run_number)},
            {"Field": "Status", "Value": status.status},
            {"Field": "Started", "Value": format_timestamp(status.started_at)},
            {"Field": "Finished", "Value": format_timestamp(status.finished_at)},
            {"Field": "Duration", "Value": duration_label(status.started_at, status.finished_at)},
        ]
    )


def _render_score_cards(context: ReportContext) -> None:
    st.subheader("Scores")
    # An "Overall" with no scope cannot be told apart from a figure over every category, and under
    # canonical-v2 the two differ by roughly 19 points. The reports carry this label; the dashboard
    # did not, which left the day-1 failure mode open in the view people actually look at.
    st.caption(f"Overall scores cover {context.methodology.scored_scope_label}.")
    audited, standard = st.columns(2)
    audited.metric("Overall audited", score_card_value(context.run_status.overall_score_audited))
    standard.metric("Overall standard", score_card_value(context.run_status.overall_score_standard))


def _render_category_scores(context: ReportContext) -> None:
    st.subheader("Per-Category Scores")
    score_mode = st.radio(
        "Score mode",
        ("Audited", "Standard"),
        horizontal=True,
        key="run_detail_score_mode",
    )
    scores = (
        context.run_status.by_category_audited
        if score_mode == "Audited"
        else context.run_status.by_category_standard
    )
    rows = category_score_rows(scores)
    if not rows:
        st.info(f"No {score_mode.lower()} per-category scores are available for this run.")
        return

    plotly_go = cast(Any, import_module("plotly.graph_objects"))
    fig = plotly_go.Figure(
        plotly_go.Bar(
            x=[row["point_percent"] for row in rows],
            y=[row["Category"] for row in rows],
            orientation="h",
            error_x={
                "type": "data",
                "array": [row["error_plus"] for row in rows],
                "arrayminus": [row["error_minus"] for row in rows],
                "visible": True,
            },
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    update_layout = fig.update_layout
    update_layout(
        xaxis_title="Score",
        yaxis_title="Category",
        xaxis={"range": [0, 100], "ticksuffix": "%"},
        margin={"l": 120, "r": 24, "t": 8, "b": 48},
    )
    st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]
    st.dataframe(
        [
            {
                "Category": row["Category"],
                "Score": row["Score"],
                "Wilson 95% CI": row["Wilson 95% CI"],
                "Correct": row["Correct"],
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_cost_breakdown(context: ReportContext) -> None:
    st.subheader("Cost Breakdown")
    st.metric("Total cost", format_money(context.cost_summary.total_cost_usd))

    phase_rows = cost_phase_rows(context)
    if phase_rows:
        st.write("Phase costs")
        st.dataframe(phase_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No phase-level API cost rows are available for this run.")

    model_rows = cost_model_rows(context)
    if model_rows:
        st.write("Model costs")
        st.dataframe(model_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No model-level API cost rows are available for this run.")


def _render_failure_summary(context: ReportContext) -> None:
    st.subheader("Failure Summary")
    total_questions = len(context.questions)
    failed_questions = len(context.failure_analysis.failed_questions)
    failure_rate = failed_questions / total_questions if total_questions else 0.0

    failures, rate, attempted = st.columns(3)
    failures.metric("Failures", str(failed_questions))
    rate.metric("Failure rate", _format_percent(failure_rate))
    attempted.metric("Questions", str(total_questions))

    category_rows = failure_category_rows(context)
    if category_rows:
        st.write("Category breakdown")
        st.dataframe(category_rows, hide_index=True, use_container_width=True)

    pattern_rows = top_pattern_rows(context)
    if pattern_rows:
        st.write("Top patterns")
        st.dataframe(pattern_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No failure patterns were detected for this run.")

    question_rows = top_failed_question_rows(context)
    if question_rows:
        st.write("Top failed questions")
        st.dataframe(question_rows, hide_index=True, use_container_width=True)
    else:
        st.success("No failed questions are recorded for this run.")

    st.button("Open Failure Explorer (later task)", disabled=True)
    st.caption("Failure Explorer drill-down is scheduled for Phase 7 Task 7.5.")


def _render_configuration(context: ReportContext) -> None:
    st.subheader("Configuration and Methodology")
    warning = same_vendor_warning_text(context)
    if warning is not None:
        st.warning(warning)

    started = context.run_started_event
    methodology = context.methodology
    st.table(
        [
            {"Field": "Framework version", "Value": started.framework_version},
            {"Field": "Provider", "Value": f"{started.provider_type} {started.provider_version}"},
            {
                "Field": "Benchmark",
                "Value": f"{methodology.benchmark_type} {methodology.benchmark_version}",
            },
            {"Field": "Answer model", "Value": started.answer_model_id},
            {"Field": "Answer vendor", "Value": started.answer_model_vendor},
            {"Field": "Judge model", "Value": started.judge_model_id},
            {"Field": "Judge vendor", "Value": started.judge_model_vendor},
            {"Field": "Methodology version", "Value": methodology.methodology_version},
            {"Field": "Methodology profile", "Value": methodology.methodology_profile},
            {"Field": "Scoring mode", "Value": methodology.scoring_mode},
            {"Field": "Confidence interval", "Value": methodology.confidence_interval},
            {"Field": "Seed", "Value": str(started.seed)},
            {
                "Field": "Same-vendor warning",
                "Value": "yes" if methodology.same_vendor_warning else "no",
            },
            {"Field": "Benchmark checksum", "Value": methodology.benchmark_checksum},
        ]
    )
    st.write("Experiment config")
    st.json(_experiment_config_payload(context), expanded=False)


def _render_placeholder(page_name: str) -> None:
    st.header(page_name)
    st.info(f"{page_name} will be completed by a later Phase 7 task.")


def _provider_label(run: RunSummary) -> str:
    if run.provider_version is None:
        return run.provider_type
    return f"{run.provider_type} {run.provider_version}"


def _questions_label(run: RunSummary) -> str:
    label = f"{run.n_questions_succeeded}/{run.n_questions_attempted} succeeded"
    if run.n_questions_errored == 0:
        return label
    return f"{label} ({run.n_questions_errored} errors)"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "(none)"
    return _format_percent(value)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _score_cell(score: ScoreWithCI | None) -> str:
    if score is None:
        return "n/a"
    return (
        f"{_format_percent(score.point_estimate)} "
        f"[{_format_percent(score.ci_95_low)}, {_format_percent(score.ci_95_high)}] "
        f"({score.n_correct}/{score.n_total})"
    )


def _delta_cell(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta * 100:+.1f} pp"


def _ci_overlap_label(ci_overlaps: bool | None) -> str:
    if ci_overlaps is None:
        return "Insufficient data"
    if ci_overlaps:
        return "Overlaps"
    return "Separated"


def _significance_label(significant: bool | None) -> str:
    if significant is None:
        return "Insufficient data"
    if significant:
        return "Significant"
    return "Not significant"


def _format_score_value(value: float) -> str:
    return f"{value:.1f}"


def _format_nullable_score(value: object) -> str:
    if isinstance(value, int | float):
        return _format_score_value(float(value))
    return "n/a"


def _format_optional_int(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return "(unknown)"


def _format_optional_money(value: object) -> str:
    if isinstance(value, int | float):
        return format_money(float(value))
    return "(unknown)"


def _format_optional_number(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "(none)"


def _format_optional_latency(value: object) -> str:
    if not isinstance(value, int | float):
        return "(unknown)"
    if value <= 100:
        return f"{float(value):.1f}ms"
    return f"{float(value):,.0f}ms"


def _display_optional_text(value: object) -> str:
    if value is None:
        return "(none)"
    text = str(value)
    return text if text else "(none)"


def _token_pair(input_tokens: int | None, output_tokens: int | None) -> str:
    if input_tokens is None and output_tokens is None:
        return "(unknown)"
    return f"{input_tokens or 0:,} in / {output_tokens or 0:,} out"


def _comparison_category_label(category: object) -> str:
    if category == "overall":
        return "overall"
    raw_value = getattr(category, "value", category)
    return str(raw_value)


def _configuration_row(field: str, baseline: str, candidate: str) -> dict[str, str]:
    return {
        "Field": field,
        "Baseline": baseline,
        "Candidate": candidate,
        "Changed": "yes" if baseline != candidate else "no",
    }


def _provider_config_label(started: RunStartedEvent) -> str:
    return f"{started.provider_type} {started.provider_version}".strip()


def _benchmark_config_label(started: RunStartedEvent) -> str:
    return f"{started.benchmark_type} {started.benchmark_version}".strip()


def _coerce_sort_option(value: object) -> SortOption:
    if isinstance(value, str) and value in SORT_OPTIONS:
        return cast(SortOption, value)
    return "Date desc"


def _nullable_score_sort_key(value: float | None, *, descending: bool) -> tuple[bool, float]:
    if value is None:
        return (True, 0.0)
    return (False, -value if descending else value)


def _nullable_number_sort_key(value: object, *, descending: bool) -> tuple[bool, float]:
    if not isinstance(value, int | float):
        return (True, 0.0)
    number = float(value)
    return (False, -number if descending else number)


def _retrieval_quality_label(question: QuestionRecord | None) -> str:
    if question is None or question.retrieval_id is None or question.n_memories_retrieved is None:
        return "Unknown/not attempted"
    if question.n_memories_retrieved == 0:
        return "No retrieved memories"
    return "Some retrieved memories"


def _failure_row_search_text(row: dict[str, object]) -> str:
    fields = (
        row["question_id"],
        row["question_text"],
        row["expected_answer"],
        row["generated_answer"],
        row["error_message"],
    )
    return "\n".join(_display_optional_text(value) for value in fields).casefold()


def _object_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in cast(list[object], value)]
    return []


def _jsonish(value: object) -> str:
    if value in ({}, [], None):
        return "(none)"
    return str(value)


def _safe_get_question_record(
    repository: RunRepository,
    run_id: str,
    question_id: str,
) -> QuestionRecord | None:
    try:
        return repository.get_question_record(run_id, question_id)
    except KhedronError:
        return None


def _safe_get_retrieval(
    repository: RunRepository,
    retrieval_id: str | None,
) -> RetrievalRecord | None:
    if retrieval_id is None:
        return None
    try:
        return repository.get_retrieval(retrieval_id)
    except KhedronError:
        return None


def _safe_get_response(repository: RunRepository, response_id: str | None) -> Response | None:
    if response_id is None:
        return None
    try:
        return repository.get_response(response_id)
    except KhedronError:
        return None


def _safe_get_judgment(repository: RunRepository, judgment_id: str | None) -> Judgment | None:
    if judgment_id is None:
        return None
    try:
        return repository.get_judgment(judgment_id)
    except KhedronError:
        return None


def _cost_breakdown_rows(
    values: dict[str, float],
    *,
    total: float,
    preferred_order: tuple[str, ...],
    preserve_names: bool = False,
) -> list[dict[str, str]]:
    order = {name: index for index, name in enumerate(preferred_order)}
    sorted_items = sorted(
        values.items(),
        key=lambda item: (order.get(item[0], len(order)), -item[1], item[0]),
    )
    rows: list[dict[str, str]] = []
    for name, cost in sorted_items:
        percentage = cost / total if total > 0 else 0.0
        rows.append(
            {
                "Name": name if preserve_names else _cost_label(name),
                "Cost": format_money(cost),
                "% of total": _format_percent(percentage),
            }
        )
    return rows


def _cost_label(value: str) -> str:
    labels = {
        "ingest": "Ingestion",
        "generate": "Generation",
        "generation": "Generation",
        "judge": "Judgment",
        "judgment": "Judgment",
        "retrieve": "Retrieval",
        "retrieval": "Retrieval",
    }
    return labels.get(value, value.replace("_", " ").title())


def _display_value(value: object) -> str:
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)].rstrip() + "..."


def _experiment_config_payload(context: ReportContext) -> dict[str, Any]:
    if context.experiment_result is not None:
        return context.experiment_result.config.model_dump(mode="json")
    return context.run_started_event.config


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _has_no_source_runs(results_dir: Path) -> bool:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return True
    return not any(path.is_dir() for path in runs_dir.iterdir())


def _read_schema_version(sqlite_path: Path) -> int | None:
    try:
        return SqliteIndexer(sqlite_path).read_schema_version()
    except KhedronError:
        return None


if __name__ == "__main__":
    main()
