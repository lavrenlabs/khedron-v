from __future__ import annotations

# ruff: noqa: S101
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from khedron.analysis.types import (
    CategoryFailureBreakdown,
    ComparisonReport,
    DetectedPattern,
    FailedQuestionSummary,
    FailureAnalysisReport,
    QuestionDifference,
    ScoreDelta,
)
from khedron.config import ExperimentConfig
from khedron.dashboard import app
from khedron.persistence.repository import RunRepository
from khedron.persistence.schemas import SCHEMA_VERSION
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.reporting.context import CostSummary, MethodologyDisclosure, ReportContext
from khedron.types import (
    AggregateScore,
    APICallRecord,
    ConversationProcessedEvent,
    ExperimentCompletedEvent,
    ExperimentResult,
    ExperimentStartedEvent,
    Judgment,
    JudgmentVerdict,
    Memory,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    QuestionRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
    RunStatus,
    RunSummary,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteStartedEvent,
    SuiteStatus,
    question_plan_fingerprint,
)
from khedron.utils.stats import wilson_score_interval


def test_effective_sqlite_path_defaults_under_results_dir() -> None:
    results_dir = Path("results")

    assert app.effective_sqlite_path(results_dir) == Path("results") / "benchmark.db"


def test_effective_sqlite_path_uses_override() -> None:
    results_dir = Path("results")
    sqlite_path = Path("custom") / "index.db"

    assert app.effective_sqlite_path(results_dir, sqlite_path) == sqlite_path


@pytest.mark.parametrize("version", [1, 3])
def test_dashboard_schema_version_read_never_migrates_or_mutates(
    tmp_path: Path, version: int
) -> None:
    sqlite_path = tmp_path / "benchmark.db"
    SqliteIndexer(sqlite_path).initialize()
    with closing(sqlite3.connect(sqlite_path)) as connection:
        connection.execute("UPDATE schema_meta SET version = ?", (version,))
        connection.commit()
    before = sqlite_path.read_bytes()

    assert app._read_schema_version(sqlite_path) is None
    assert sqlite_path.read_bytes() == before


def test_dashboard_schema_version_read_rejects_multiple_metadata_without_mutation(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "benchmark.db"
    SqliteIndexer(sqlite_path).initialize()
    with closing(sqlite3.connect(sqlite_path)) as connection:
        connection.execute(
            "INSERT INTO schema_meta VALUES (?, ?, ?)", (SCHEMA_VERSION, "later", "x")
        )
        connection.commit()
    before = sqlite_path.read_bytes()

    assert app._read_schema_version(sqlite_path) is None
    assert sqlite_path.read_bytes() == before


def test_create_repository_uses_expected_paths(monkeypatch) -> None:
    captured: dict[str, Path] = {}

    class FakeRepository:
        def __init__(self, results_dir: Path, sqlite_path: Path) -> None:
            captured["results_dir"] = results_dir
            captured["sqlite_path"] = sqlite_path

    monkeypatch.setattr(app, "RunRepository", FakeRepository)

    repository = app.create_repository(Path("results"))

    assert isinstance(repository, FakeRepository)
    assert captured == {
        "results_dir": Path("results"),
        "sqlite_path": Path("results") / "benchmark.db",
    }


def test_format_score_includes_point_estimate_and_wilson_ci() -> None:
    score = ScoreWithCI(
        n_total=100,
        n_correct=82,
        n_errors=18,
        point_estimate=0.82,
        ci_95_low=0.7333,
        ci_95_high=0.8830,
    )

    formatted = app.format_score(score)

    assert "82.0%" in formatted
    assert "[73.3%, 88.3%]" in formatted
    assert "Wilson 95% CI" in formatted
    assert "(82/100)" in formatted


def test_format_score_handles_missing_score() -> None:
    assert app.format_score(None) == "No score"


def test_format_money() -> None:
    assert app.format_money(211.5) == "$211.50"
    assert app.format_money(1234.567) == "$1,234.57"


def test_format_timestamp_as_utc() -> None:
    timestamp = datetime(2026, 4, 30, 10, 23, 45, tzinfo=timezone(timedelta(hours=-4)))

    assert app.format_timestamp(timestamp) == "2026-04-30 14:23:45 UTC"


def test_format_timestamp_handles_none() -> None:
    assert app.format_timestamp(None) == "(none)"


def test_format_timestamp_treats_naive_values_as_utc() -> None:
    timestamp = datetime(2026, 4, 30, 14, 23, 45)

    assert app.format_timestamp(timestamp) == "2026-04-30 14:23:45 UTC"


def test_default_status_selection_prefers_completed_runs() -> None:
    assert app.default_status_selection(["completed", "failed"]) == ["completed"]
    assert app.default_status_selection(["failed", "running"]) == ["failed", "running"]


def test_filter_runs_by_status_provider_methodology_and_started_date() -> None:
    runs = [
        _run(
            "run-1",
            status="completed",
            provider_type="full_context",
            methodology_version="v1.0",
            started_at=datetime(2026, 5, 1, tzinfo=UTC),
        ),
        _run(
            "run-2",
            status="failed",
            provider_type="full_context",
            methodology_version="v1.0",
            started_at=datetime(2026, 5, 2, tzinfo=UTC),
        ),
        _run(
            "run-3",
            status="completed",
            provider_type="full_context",
            methodology_version="v1.1",
            started_at=datetime(2026, 5, 3, tzinfo=UTC),
        ),
    ]

    filtered = app.filter_runs(
        runs,
        statuses=["completed"],
        providers=["full_context"],
        methodology_version="v1.0",
        started_date_range=(date(2026, 5, 1), date(2026, 5, 2)),
    )

    assert [run.run_id for run in filtered] == ["run-1"]


def test_filter_runs_all_methodologies_and_inclusive_date_range() -> None:
    runs = [
        _run("run-1", methodology_version="v1.0", started_at=datetime(2026, 5, 1)),
        _run("run-2", methodology_version="v1.1", started_at=datetime(2026, 5, 2)),
    ]

    filtered = app.filter_runs(
        runs,
        statuses=["completed"],
        providers=["full_context"],
        methodology_version=app.ALL_METHODOLOGIES,
        started_date_range=(date(2026, 5, 1), date(2026, 5, 2)),
    )

    assert [run.run_id for run in filtered] == ["run-1", "run-2"]


def test_default_methodology_version_uses_common_version_with_recency_tie_break() -> None:
    common_runs = [
        _run("run-1", methodology_version="v1.0", started_at=datetime(2026, 5, 1)),
        _run("run-2", methodology_version="v1.0", started_at=datetime(2026, 5, 2)),
        _run("run-3", methodology_version="v1.1", started_at=datetime(2026, 5, 3)),
    ]
    tied_runs = [
        _run("run-1", methodology_version="v1.0", started_at=datetime(2026, 5, 1)),
        _run("run-2", methodology_version="v1.1", started_at=datetime(2026, 5, 3)),
    ]

    assert app.default_methodology_version(common_runs) == "v1.0"
    assert app.default_methodology_version(tied_runs) == "v1.1"
    assert app.default_methodology_version([]) == app.ALL_METHODOLOGIES


def test_started_date_bounds_use_full_available_range() -> None:
    runs = [
        _run("run-1", started_at=datetime(2026, 5, 2, 14, tzinfo=UTC)),
        _run("run-2", started_at=datetime(2026, 4, 30, 23, tzinfo=UTC)),
    ]

    assert app.started_date_bounds(runs) == (date(2026, 4, 30), date(2026, 5, 2))
    assert app.started_date_bounds([]) is None


def test_normalize_date_range_handles_single_reversed_and_empty_values() -> None:
    default_start = date(2026, 5, 1)
    default_end = date(2026, 5, 7)

    assert app.normalize_date_range(date(2026, 5, 3), default_start, default_end) == (
        date(2026, 5, 3),
        date(2026, 5, 3),
    )
    assert app.normalize_date_range(
        (date(2026, 5, 6), date(2026, 5, 2)),
        default_start,
        default_end,
    ) == (date(2026, 5, 2), date(2026, 5, 6))
    assert app.normalize_date_range((), default_start, default_end) == (
        default_start,
        default_end,
    )


def test_sort_runs_supports_dates_scores_costs_and_none_scores_last() -> None:
    runs = [
        _run(
            "run-1",
            started_at=datetime(2026, 5, 1),
            overall_score_audited=0.8,
            total_cost_usd=20.0,
        ),
        _run(
            "run-2",
            started_at=datetime(2026, 5, 3),
            overall_score_audited=None,
            total_cost_usd=10.0,
        ),
        _run(
            "run-3",
            started_at=datetime(2026, 5, 2),
            overall_score_audited=0.9,
            total_cost_usd=30.0,
        ),
    ]

    assert [run.run_id for run in app.sort_runs(runs, "Date desc")] == [
        "run-2",
        "run-3",
        "run-1",
    ]
    assert [run.run_id for run in app.sort_runs(runs, "Date asc")] == [
        "run-1",
        "run-3",
        "run-2",
    ]
    assert [run.run_id for run in app.sort_runs(runs, "Audited score desc")] == [
        "run-3",
        "run-1",
        "run-2",
    ]
    assert [run.run_id for run in app.sort_runs(runs, "Audited score asc")] == [
        "run-1",
        "run-3",
        "run-2",
    ]
    assert [run.run_id for run in app.sort_runs(runs, "Cost desc")] == [
        "run-3",
        "run-1",
        "run-2",
    ]
    assert [run.run_id for run in app.sort_runs(runs, "Cost asc")] == [
        "run-2",
        "run-1",
        "run-3",
    ]


def test_kpi_helpers_calculate_visible_row_values() -> None:
    runs = [
        _run(
            "run-1",
            overall_score_audited=0.8,
            total_cost_usd=20.0,
            started_at=datetime(2026, 5, 1, tzinfo=UTC),
        ),
        _run(
            "run-2",
            overall_score_audited=None,
            total_cost_usd=5.5,
            started_at=datetime(2026, 5, 3, tzinfo=UTC),
        ),
        _run(
            "run-3",
            overall_score_audited=0.6,
            total_cost_usd=10.0,
            started_at=datetime(2026, 5, 2, tzinfo=UTC),
        ),
    ]

    assert app.dashboard_mean_audited_score(runs) == 0.7
    assert app.dashboard_mean_audited_score([_run("run-4", overall_score_audited=None)]) is None
    assert app.total_run_cost(runs) == 35.5
    assert app.most_recent_started_at(runs) == datetime(2026, 5, 3, tzinfo=UTC)
    assert app.most_recent_started_at([]) is None


def test_run_table_rows_include_required_columns_and_formatting() -> None:
    rows = app.run_table_rows(
        [
            _run(
                "run-1",
                provider_version="2.5.0",
                benchmark_type="locomo",
                overall_score_audited=0.821,
                overall_score_standard=None,
                n_questions_attempted=50,
                n_questions_succeeded=48,
                n_questions_errored=2,
                total_cost_usd=211.5,
            )
        ]
    )

    assert rows == [
        {
            "Started": "2026-05-01 12:00:00 UTC",
            "Experiment": "Experiment run-1",
            "Run ID": "run-1",
            "Status": "completed",
            "Provider": "full_context 2.5.0",
            "Benchmark": "locomo",
            "Models": "answer-model / judge-model",
            "Audited Score": "82.1%",
            "Standard Score": "(none)",
            "Questions": "48/50 succeeded (2 errors)",
            "Cost": "$211.50",
        }
    ]


def test_run_detail_options_are_readable_and_deterministically_ordered() -> None:
    runs = [
        _run("run-old", experiment_name="Beta", started_at=datetime(2026, 5, 1, tzinfo=UTC)),
        _run("run-new-b", experiment_name="Beta", started_at=datetime(2026, 5, 3, tzinfo=UTC)),
        _run("run-new-a", experiment_name="Alpha", started_at=datetime(2026, 5, 3, tzinfo=UTC)),
    ]

    options = app.run_detail_options(runs)

    assert list(options.values()) == ["run-new-a", "run-new-b", "run-old"]
    assert next(iter(options)) == "2026-05-03 00:00:00 UTC | Alpha | run-new-a | completed"


def test_compare_run_options_match_run_detail_ordering() -> None:
    runs = [
        _run("run-old", experiment_name="Beta", started_at=datetime(2026, 5, 1, tzinfo=UTC)),
        _run("run-new", experiment_name="Alpha", started_at=datetime(2026, 5, 3, tzinfo=UTC)),
    ]

    assert app.compare_run_options(runs) == app.run_detail_options(runs)


def test_score_card_value_formats_present_and_missing_scores() -> None:
    formatted = app.score_card_value(_score(n_total=100, n_correct=82))

    assert formatted == "82.0% (Wilson 95% CI: 73.3% - 88.3%; 82/100)"
    assert app.score_card_value(None) == "No score"


def test_category_score_rows_include_ci_and_error_fields() -> None:
    rows = app.category_score_rows(
        {
            "temporal": _score(n_total=10, n_correct=7, low=0.397, high=0.892),
            "single_hop": _score(n_total=10, n_correct=8, low=0.490, high=0.943),
        }
    )

    assert rows[0]["Category"] == "single_hop"
    assert rows[0]["Score"] == "80.0%"
    assert rows[0]["Wilson 95% CI"] == "[49.0%, 94.3%]"
    assert rows[0]["Correct"] == "8/10"
    assert rows[0]["point_percent"] == 80.0
    assert rows[0]["ci_low_percent"] == 49.0
    assert rows[0]["ci_high_percent"] == 94.3
    assert rows[0]["error_plus"] == 14.299999999999997
    assert rows[0]["error_minus"] == 31.0


def test_score_delta_rows_format_ci_overlap_and_significance_labels() -> None:
    rows = app.score_delta_rows(
        [
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=_score(n_total=10, n_correct=7, low=0.500, high=0.900),
                candidate_score=_score(n_total=10, n_correct=9, low=0.600, high=0.982),
                point_delta=0.2,
                ci_overlaps=True,
                statistically_significant=False,
            ),
            ScoreDelta(
                category=QuestionCategory.TEMPORAL,
                mode="audited",
                baseline_score=None,
                candidate_score=_score(n_total=10, n_correct=9),
                point_delta=None,
                ci_overlaps=None,
                statistically_significant=None,
            ),
        ]
    )

    assert rows == [
        {
            "Metric": "overall",
            "Mode": "audited",
            "Baseline": "70.0% [50.0%, 90.0%] (7/10)",
            "Candidate": "90.0% [60.0%, 98.2%] (9/10)",
            "Delta": "+20.0 pp",
            "CI overlap": "Overlaps",
            "Significance": "Not significant",
        },
        {
            "Metric": "temporal",
            "Mode": "audited",
            "Baseline": "n/a",
            "Candidate": "90.0% [73.3%, 88.3%] (9/10)",
            "Delta": "n/a",
            "CI overlap": "Insufficient data",
            "Significance": "Insufficient data",
        },
    ]


def test_comparison_chart_rows_exclude_overall_and_missing_scores() -> None:
    comparison = _comparison_report(
        score_deltas=[
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=_score(n_total=10, n_correct=8),
                candidate_score=_score(n_total=10, n_correct=9),
                point_delta=0.1,
                ci_overlaps=True,
                statistically_significant=False,
            ),
            ScoreDelta(
                category=QuestionCategory.TEMPORAL,
                mode="audited",
                baseline_score=_score(n_total=10, n_correct=7, low=0.397, high=0.892),
                candidate_score=None,
                point_delta=None,
                ci_overlaps=None,
                statistically_significant=None,
            ),
        ]
    )

    rows = app.comparison_chart_rows(comparison)

    assert rows == [
        {
            "Category": "temporal",
            "Run": "Baseline",
            "Run ID": "baseline-run",
            "point_percent": 70.0,
            "error_plus": 19.200000000000003,
            "error_minus": 30.299999999999997,
        }
    ]


def test_question_difference_rows_format_verdicts_and_scores_by_selected_runs() -> None:
    rows = app.question_difference_rows(
        [
            QuestionDifference(
                question_id="q-1",
                category=QuestionCategory.SINGLE_HOP,
                verdicts_by_run_id={
                    "baseline-run": JudgmentVerdict.INCORRECT,
                    "candidate-run": JudgmentVerdict.CORRECT,
                },
                scores_by_run_id={"baseline-run": 0.0, "candidate-run": 1.0},
            )
        ],
        "baseline-run",
        "candidate-run",
    )

    assert rows == [
        {
            "Question ID": "q-1",
            "Category": "single_hop",
            "Baseline verdict": "incorrect",
            "Baseline score": "0.0",
            "Candidate verdict": "correct",
            "Candidate score": "1.0",
        }
    ]


def test_configuration_differences_surface_a_framework_version_change() -> None:
    # A differing framework_version is the first "common surprise" an analyst should check when
    # two runs disagree, and this section did not carry the field at all -- the documented
    # workflow pointed an analyst at something the page never showed.
    baseline_context = _report_context()
    candidate_started = baseline_context.run_started_event.model_copy(
        update={"framework_version": "0.0.0+deadbee"},
    )

    rows = app.configuration_difference_rows(
        baseline_context.run_status,
        baseline_context.run_status,
        baseline_context.run_started_event,
        candidate_started,
    )

    framework_rows = [row for row in rows if row["Field"] == "Framework version"]
    if not framework_rows:
        raise AssertionError(f"the field the workflow tells an analyst to check is absent: {rows}")
    if framework_rows[0]["Changed"] != "yes":
        raise AssertionError(framework_rows[0])
    if framework_rows[0]["Candidate"] != "0.0.0+deadbee":
        raise AssertionError(framework_rows[0])


def test_configuration_difference_rows_use_public_run_status_and_started_event_fields() -> None:
    baseline_context = _report_context()
    candidate_status = baseline_context.run_status.model_copy(
        update={"methodology_profile": "canonical-v2"},
    )
    candidate_started = baseline_context.run_started_event.model_copy(
        update={
            "provider_type": "full_context",
            "provider_version": "0.1.0",
            "answer_model_id": "other-answer",
        },
    )

    rows = app.configuration_difference_rows(
        baseline_context.run_status,
        candidate_status,
        baseline_context.run_started_event,
        candidate_started,
    )

    assert rows[0] == {
        "Field": "Provider",
        "Baseline": "full_context 2.5.0",
        "Candidate": "full_context 0.1.0",
        "Changed": "yes",
    }
    assert rows[2] == {
        "Field": "Answer model",
        "Baseline": "answer-model",
        "Candidate": "other-answer",
        "Changed": "yes",
    }
    assert rows[3]["Changed"] == "no"
    assert rows[5] == {
        "Field": "Methodology profile",
        "Baseline": "canonical-v1",
        "Candidate": "canonical-v2",
        "Changed": "yes",
    }


def test_failure_rows_join_question_context_and_patterns() -> None:
    failed = FailedQuestionSummary(
        question_id="q-1",
        category=QuestionCategory.TEMPORAL,
        verdict=JudgmentVerdict.INCORRECT,
        question_text="When did Alice move?",
        expected_answer="April",
        generated_answer="May",
        error_phase="judge",
        error_message="low score",
    )
    question = _question_record("q-1", JudgmentVerdict.INCORRECT).model_copy(
        update={
            "is_audited_error": True,
            "retrieval_id": "ret-1",
            "n_memories_retrieved": 2,
            "total_cost_usd": 0.12,
        }
    )
    pattern = DetectedPattern(
        pattern_name="temporal_mismatch",
        description="Temporal answer mismatch.",
        suggested_remedy="Inspect chronology.",
        affected_question_ids=["q-1"],
        n_affected_questions=1,
        confidence=0.8,
    )

    rows = app.failure_rows([failed], [question], [pattern])

    assert rows[0]["question_id"] == "q-1"
    assert rows[0]["is_audited_error"] is True
    assert rows[0]["retrieved_memory_count"] == 2
    assert rows[0]["retrieval_quality"] == "Some retrieved memories"
    assert rows[0]["pattern_names"] == ["temporal_mismatch"]
    assert rows[0]["total_cost_usd"] == 0.12


def test_filter_failure_rows_supports_category_verdict_pattern_audit_retrieval_and_search() -> None:
    rows = [
        {
            "question_id": "q-1",
            "category": "temporal",
            "verdict": "incorrect",
            "score": 0.0,
            "is_audited_error": True,
            "retrieved_memory_count": 0,
            "retrieval_quality": "No retrieved memories",
            "pattern_names": ["temporal_mismatch"],
            "question_text": "When did Alice move?",
            "expected_answer": "April",
            "generated_answer": "May",
            "error_phase": None,
            "error_message": "Missing evidence",
            "total_cost_usd": 0.2,
        },
        {
            "question_id": "q-2",
            "category": "single_hop",
            "verdict": "partial",
            "score": 0.5,
            "is_audited_error": False,
            "retrieved_memory_count": None,
            "retrieval_quality": "Unknown/not attempted",
            "pattern_names": [],
            "question_text": "Who called Bob?",
            "expected_answer": "Alice",
            "generated_answer": "Alice and Carol",
            "error_phase": "generate",
            "error_message": None,
            "total_cost_usd": 0.1,
        },
    ]

    filtered = app.filter_failure_rows(
        rows,
        categories=["temporal"],
        verdicts=["incorrect"],
        patterns=["temporal_mismatch"],
        audited_filter="Audited errors only",
        retrieval_quality="No retrieved memories",
        search="MISSING",
    )

    assert [row["question_id"] for row in filtered] == ["q-1"]
    assert (
        app.filter_failure_rows(
            rows,
            categories=[],
            verdicts=[],
            patterns=[],
            audited_filter="Exclude audited errors",
            retrieval_quality="Unknown/not attempted",
            search="carol",
        )[0]["question_id"]
        == "q-2"
    )


def test_sort_failure_rows_supports_question_id_category_verdict_score_and_cost() -> None:
    rows = [
        {
            "question_id": "q-2",
            "category": "single_hop",
            "verdict": "partial",
            "score": None,
            "total_cost_usd": 0.1,
        },
        {
            "question_id": "q-1",
            "category": "temporal",
            "verdict": "incorrect",
            "score": 0.2,
            "total_cost_usd": 0.3,
        },
        {
            "question_id": "q-3",
            "category": "adversarial",
            "verdict": "error",
            "score": 0.8,
            "total_cost_usd": None,
        },
    ]

    assert [row["question_id"] for row in app.sort_failure_rows(rows, "Question ID asc")] == [
        "q-1",
        "q-2",
        "q-3",
    ]
    assert [row["question_id"] for row in app.sort_failure_rows(rows, "Category asc")] == [
        "q-3",
        "q-2",
        "q-1",
    ]
    assert [row["question_id"] for row in app.sort_failure_rows(rows, "Verdict desc")] == [
        "q-2",
        "q-1",
        "q-3",
    ]
    assert [row["question_id"] for row in app.sort_failure_rows(rows, "Score desc")] == [
        "q-3",
        "q-1",
        "q-2",
    ]
    assert [row["question_id"] for row in app.sort_failure_rows(rows, "Cost asc")] == [
        "q-2",
        "q-1",
        "q-3",
    ]


def test_failure_csv_and_visible_question_ids_are_in_memory_payloads() -> None:
    rows = [
        {
            "question_id": "q-1",
            "category": "temporal",
            "verdict": "incorrect",
            "score": 0.0,
            "is_audited_error": True,
            "retrieved_memory_count": 0,
            "pattern_names": ["temporal_mismatch"],
            "question_text": "Question?",
            "expected_answer": "Expected",
            "generated_answer": "Generated",
            "error_phase": "judge",
            "error_message": "Bad answer",
            "total_cost_usd": 0.25,
        }
    ]

    payload = app.failure_csv(rows)

    assert "Question ID,Category,Verdict" in payload
    assert "q-1,temporal,incorrect" in payload
    assert app.visible_question_ids(rows) == "q-1"
    assert app.failure_csv([]) == ""


def test_duration_label_formats_finished_and_unfinished_runs() -> None:
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

    assert app.duration_label(started, started + timedelta(seconds=45)) == "45s"
    assert app.duration_label(started, started + timedelta(minutes=3, seconds=5)) == "3m 5s"
    assert app.duration_label(started, started + timedelta(hours=2, minutes=1, seconds=9)) == (
        "2h 1m 9s"
    )
    assert app.duration_label(started, None) == "(unfinished)"


def test_cost_breakdown_rows_format_phase_and_model_costs() -> None:
    context = _report_context()

    assert app.cost_phase_rows(context) == [
        {"Name": "Generation", "Cost": "$0.03", "% of total": "33.3%"},
        {"Name": "Judgment", "Cost": "$0.06", "% of total": "66.7%"},
    ]
    assert app.cost_model_rows(context) == [
        {"Name": "judge-model", "Cost": "$0.06", "% of total": "66.7%"},
        {"Name": "answer-model", "Cost": "$0.03", "% of total": "33.3%"},
    ]


def test_top_failed_question_rows_limit_and_format_values() -> None:
    context = _report_context(n_failed=12)

    rows = app.top_failed_question_rows(context, limit=10)

    assert len(rows) == 10
    assert rows[0]["Question ID"] == "q-00"
    assert rows[0]["Category"] == "temporal"
    assert rows[0]["Verdict"] == "incorrect"
    assert rows[0]["Error phase"] == "(none)"
    assert rows[0]["Error"] == "(none)"


def test_same_vendor_warning_text_only_when_disclosed() -> None:
    assert app.same_vendor_warning_text(_report_context()) is None
    assert "same vendor" in app.same_vendor_warning_text(_report_context(same_vendor=True))


def test_app_smoke_renders_empty_results_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    dashboard_path = Path(__file__).parents[3] / "src" / "khedron" / "dashboard" / "app.py"

    page = AppTest.from_file(str(dashboard_path))
    page.run(timeout=10)

    assert not page.exception
    assert page.title[0].value == "Khedron"
    assert page.text_input[0].label == "Results directory"
    assert page.text_input[1].label == "SQLite path (optional)"
    assert page.selectbox[0].label == "Page"
    assert page.selectbox[0].options == list(app.PAGE_NAMES)


@pytest.mark.asyncio
async def test_app_smoke_renders_all_pages_with_synthetic_persisted_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repository = RunRepository(
        results_dir=tmp_path / "results",
        sqlite_path=tmp_path / "results" / "benchmark.db",
    )
    await _append_dashboard_smoke_results(repository)

    dashboard_path = Path(__file__).parents[3] / "src" / "khedron" / "dashboard" / "app.py"
    page = _run_dashboard_page(dashboard_path, "Runs Overview")

    _assert_page_ok(page)
    _assert_page_contains(page, "Runs Overview", "Synthetic dashboard experiment", "run-smoke-2")
    assert page.text_input[0].label == "Results directory"
    assert page.selectbox[0].label == "Page"
    assert page.multiselect[0].label == "Status"
    assert page.multiselect[1].label == "Provider"
    assert page.metric[0].label == "Total runs"
    assert page.metric[0].value == "2"

    page = _run_dashboard_page(dashboard_path, "Run Detail")

    _assert_page_ok(page)
    _assert_page_contains(
        page,
        "Run Detail",
        "Synthetic dashboard experiment",
        "Overall audited",
        "Per-Category Scores",
        "Cost Breakdown",
        "Failure Summary",
        "Configuration and Methodology",
    )
    assert "Run" in _element_labels(page.selectbox)
    assert "Score mode" in _element_labels(page.radio)

    page = _run_dashboard_page(dashboard_path, "Compare Runs")

    _assert_page_ok(page)
    _assert_page_contains(
        page,
        "Compare Runs",
        "Baseline run",
        "Candidate run",
        "Score Deltas",
        "Configuration Differences",
        "Export",
    )
    assert "Baseline run" in _element_labels(page.selectbox)
    assert "Candidate run" in _element_labels(page.selectbox)
    assert "Comparison mode" in _element_labels(page.radio)

    page = _run_dashboard_page(dashboard_path, "Failure Explorer")

    _assert_page_ok(page)
    _assert_page_contains(
        page,
        "Failure Explorer",
        "Showing 1 of 1 failed questions",
        "Bulk Actions",
        "Visible question IDs",
        "q-failed",
        "Generated wrong city",
    )
    assert "Run" in _element_labels(page.selectbox)
    assert "Category" in _element_labels(page.multiselect)
    assert "Verdict" in _element_labels(page.multiselect)
    load_trace = next(button for button in page.button if button.label == "Load trace")
    load_trace.click()
    page.run(timeout=15)
    _assert_page_ok(page)
    _assert_page_contains(
        page,
        "Retrieved Memories",
        "Alice moved to Rome in April 2026.",
        "Generation prompt",
        "Judgment reasoning",
        "The generated answer does not match the expected answer.",
    )

    page = _run_dashboard_page(dashboard_path, "Settings")

    _assert_page_ok(page)
    _assert_page_contains(page, "Settings", "Storage", "Schema version", "Runs Overview")


@pytest.mark.asyncio
async def test_compare_runs_withholds_deltas_for_legacy_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    repository = RunRepository(tmp_path / "results", tmp_path / "results" / "benchmark.db")
    await _append_dashboard_smoke_results(repository)
    for run_number in (1, 2):
        (tmp_path / "results" / "runs" / f"run-smoke-{run_number}" / "question_plan.jsonl").unlink()

    dashboard_path = Path(__file__).parents[3] / "src" / "khedron" / "dashboard" / "app.py"
    page = _run_dashboard_page(dashboard_path, "Compare Runs")

    _assert_page_ok(page)
    _assert_page_contains(page, "Comparison withheld", "No score deltas or export were produced")
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert "Score Deltas" not in rendered


def test_dashboard_runtime_source_stays_read_only() -> None:
    source = (Path(__file__).parents[3] / "src" / "khedron" / "dashboard" / "app.py").read_text(
        encoding="utf-8"
    )
    forbidden_patterns = (
        ".append_",
        ".rebuild_sqlite(",
        ".initialize(",
        ".index_run(",
        ".index_suite(",
        ".rebuild_all(",
        ".delete(",
        ".insert(",
        ".update(",
        ".create(",
        "JsonlWriter",
        "jsonl",
        ".open(",
        "open(",
        "write_text(",
        "read_text(",
        "unlink(",
    )

    found = [pattern for pattern in forbidden_patterns if pattern in source]

    assert found == []


async def _append_dashboard_smoke_results(repository: RunRepository) -> None:
    await repository.append_suite_event(_smoke_suite_started())
    await repository.append_suite_event(_smoke_experiment_started())

    for run_number in (1, 2):
        run_id = f"run-smoke-{run_number}"
        await repository.append_run_event(_smoke_run_started(run_id, run_number))
        await repository.append_question_plan(_smoke_question_plan(run_id))
        await repository.append_question_evaluation(
            _smoke_question_evaluation(
                run_id,
                "q-correct",
                "qe-correct",
                QuestionCategory.SINGLE_HOP,
            )
        )
        await repository.append_retrieval(_smoke_retrieval(run_id, "q-correct", "qe-correct"))
        await repository.append_response(
            _smoke_response(
                run_id,
                "q-correct",
                retrieval_id=f"{run_id}-retrieval-q-correct",
                answer_text="Rome",
            )
        )
        await repository.append_judgment(
            _smoke_judgment(
                run_id,
                "q-correct",
                response_id=f"{run_id}-response-q-correct",
                verdict=JudgmentVerdict.CORRECT,
                score=1.0,
                reasoning="The answer matches the expected city.",
            )
        )
        await repository.append_api_call(_smoke_api_call(run_id, "q-correct", "generate", 0.01))
        await repository.append_api_call(_smoke_api_call(run_id, "q-correct", "judge", 0.02))

        await repository.append_question_evaluation(
            _smoke_question_evaluation(run_id, "q-failed", "qe-failed", QuestionCategory.TEMPORAL)
        )
        await repository.append_retrieval(_smoke_retrieval(run_id, "q-failed", "qe-failed"))
        await repository.append_response(
            _smoke_response(
                run_id,
                "q-failed",
                retrieval_id=f"{run_id}-retrieval-q-failed",
                answer_text="Generated wrong city",
            )
        )
        await repository.append_judgment(
            _smoke_judgment(
                run_id,
                "q-failed",
                response_id=f"{run_id}-response-q-failed",
                verdict=JudgmentVerdict.INCORRECT,
                score=0.0,
                reasoning="The generated answer does not match the expected answer.",
            )
        )
        await repository.append_api_call(_smoke_api_call(run_id, "q-failed", "generate", 0.01))
        await repository.append_api_call(_smoke_api_call(run_id, "q-failed", "judge", 0.02))
        await repository.append_run_event(_smoke_conversation_processed(run_id, run_number))
        await repository.append_run_event(_smoke_run_completed(run_id, run_number))

    await repository.append_suite_event(_smoke_experiment_completed())
    await repository.append_experiment_result(_smoke_experiment_result())
    await repository.append_suite_event(_smoke_suite_completed())


def _smoke_question_plan(run_id: str) -> QuestionPlanRecord:
    question_ids = ("q-correct", "q-failed")
    return QuestionPlanRecord(
        run_id=run_id,
        benchmark_id="locomo",
        categories=("single_hop", "temporal"),
        corpus_checksum="sha256:synthetic-dashboard",
        question_ids=question_ids,
        fingerprint=question_plan_fingerprint(
            benchmark_id="locomo",
            categories=("single_hop", "temporal"),
            corpus_checksum="sha256:synthetic-dashboard",
            question_ids=question_ids,
        ),
        timestamp=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
    )


def _smoke_suite_started() -> SuiteStartedEvent:
    return SuiteStartedEvent(
        event_id="suite-smoke-started",
        timestamp=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
        suite_id="suite-smoke",
        sequence_number=0,
        config_yaml_path="experiments/synthetic-dashboard.yaml",
        config_yaml_content="experiments:\n  - name: Synthetic dashboard experiment\n",
        methodology_version="v1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        runtime_environment={"python": "3.11"},
    )


def _smoke_experiment_started() -> ExperimentStartedEvent:
    return ExperimentStartedEvent(
        event_id="suite-smoke-experiment-started",
        timestamp=datetime(2026, 5, 7, 12, 0, 1, tzinfo=UTC),
        suite_id="suite-smoke",
        sequence_number=1,
        experiment_id="experiment-smoke",
        experiment_name="Synthetic dashboard experiment",
        n_runs_planned=2,
    )


def _smoke_experiment_completed() -> ExperimentCompletedEvent:
    return ExperimentCompletedEvent(
        event_id="suite-smoke-experiment-completed",
        timestamp=datetime(2026, 5, 7, 12, 3, tzinfo=UTC),
        suite_id="suite-smoke",
        sequence_number=2,
        experiment_id="experiment-smoke",
        overall_standard_pooled_score=_smoke_score(4, 2),
        overall_standard_mean=0.5,
        overall_standard_stddev=0.0,
        overall_audited_pooled_score=_smoke_score(4, 2),
        overall_audited_mean=0.5,
        overall_audited_stddev=0.0,
        cost_usd=0.12,
    )


def _smoke_suite_completed() -> SuiteCompletedEvent:
    return SuiteCompletedEvent(
        event_id="suite-smoke-completed",
        timestamp=datetime(2026, 5, 7, 12, 3, 1, tzinfo=UTC),
        suite_id="suite-smoke",
        sequence_number=3,
        total_cost_usd=0.12,
        n_experiments_completed=1,
        n_experiments_failed=0,
    )


def _smoke_run_started(run_id: str, run_number: int) -> RunStartedEvent:
    return RunStartedEvent(
        event_id=f"{run_id}-started",
        timestamp=datetime(2026, 5, 7, 12, run_number, tzinfo=UTC),
        run_id=run_id,
        sequence_number=0,
        suite_id="suite-smoke",
        experiment_id="experiment-smoke",
        experiment_name="Synthetic dashboard experiment",
        run_number=run_number,
        provider_type="full_context",
        provider_version="0.1.0",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="sha256:dashboard-smoke",
        answer_model_id="fake-answer-model",
        answer_model_vendor="offline",
        judge_model_id="fake-judge-model",
        judge_model_vendor="offline-judge",
        config={"provider": {"type": "full_context"}, "top_k_retrieval": 2},
        methodology_version="v1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=100 + run_number,
        runtime_environment={"python": "3.11"},
    )


def _smoke_conversation_processed(
    run_id: str,
    sequence_number: int,
) -> ConversationProcessedEvent:
    return ConversationProcessedEvent(
        event_id=f"{run_id}-conversation-processed",
        timestamp=datetime(2026, 5, 7, 12, 2, sequence_number, tzinfo=UTC),
        run_id=run_id,
        sequence_number=1,
        conversation_id="conversation-smoke",
        n_questions_evaluated=2,
        n_questions_correct=1,
        n_questions_errored=0,
        cost_usd=0.06,
    )


def _smoke_run_completed(run_id: str, run_number: int) -> RunCompletedEvent:
    return RunCompletedEvent(
        event_id=f"{run_id}-completed",
        timestamp=datetime(2026, 5, 7, 12, 2, 30 + run_number, tzinfo=UTC),
        run_id=run_id,
        sequence_number=2,
        status="completed",
        n_questions_attempted=2,
        n_questions_succeeded=2,
        n_questions_errored=0,
        overall_score_standard=_smoke_score(2, 1),
        overall_score_audited=_smoke_score(2, 1),
        by_category_standard={
            "single_hop": _smoke_score(1, 1),
            "temporal": _smoke_score(1, 0),
        },
        by_category_audited={
            "single_hop": _smoke_score(1, 1),
            "temporal": _smoke_score(1, 0),
        },
        total_cost_usd=0.06,
    )


def _smoke_question_evaluation(
    run_id: str,
    question_id: str,
    suffix: str,
    category: QuestionCategory,
) -> QuestionEvaluationRecord:
    return QuestionEvaluationRecord(
        question_evaluation_id=f"{run_id}-{suffix}",
        run_id=run_id,
        question_id=question_id,
        conversation_id="conversation-smoke",
        category=category,
        question_text=(
            "Which city did Alice move to?"
            if question_id == "q-correct"
            else "When did Alice move to Rome?"
        ),
        expected_answer="Rome" if question_id == "q-correct" else "April 2026",
        is_audited_error=False,
        timestamp=datetime(2026, 5, 7, 12, 1, tzinfo=UTC),
    )


def _smoke_retrieval(run_id: str, question_id: str, suffix: str) -> RetrievalRecord:
    return RetrievalRecord(
        retrieval_id=f"{run_id}-retrieval-{question_id}",
        question_evaluation_id=f"{run_id}-{suffix}",
        run_id=run_id,
        question_id=question_id,
        timestamp=datetime(2026, 5, 7, 12, 1, 10, tzinfo=UTC),
        query=f"memory query for {question_id}",
        top_k=2,
        n_returned=1,
        memories=[
            Memory(
                memory_id=f"{run_id}-memory-{question_id}",
                content="Alice moved to Rome in April 2026.",
                metadata={"speaker": "Alice"},
                score=0.95,
                timestamp=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
            )
        ],
        retrieval_latency_ms=12.0,
    )


def _smoke_response(
    run_id: str,
    question_id: str,
    *,
    retrieval_id: str,
    answer_text: str,
) -> Response:
    return Response(
        response_id=f"{run_id}-response-{question_id}",
        run_id=run_id,
        question_id=question_id,
        retrieval_id=retrieval_id,
        timestamp=datetime(2026, 5, 7, 12, 1, 20, tzinfo=UTC),
        model_id="fake-answer-model",
        prompt=f"Use the synthetic memory to answer {question_id}.",
        answer_text=answer_text,
        input_tokens=32,
        output_tokens=4,
        latency_ms=120.0,
        cost_usd=0.01,
        raw_api_response={"id": f"{run_id}-{question_id}-response"},
    )


def _smoke_judgment(
    run_id: str,
    question_id: str,
    *,
    response_id: str,
    verdict: JudgmentVerdict,
    score: float,
    reasoning: str,
) -> Judgment:
    return Judgment(
        judgment_id=f"{run_id}-judgment-{question_id}",
        run_id=run_id,
        response_id=response_id,
        question_id=question_id,
        timestamp=datetime(2026, 5, 7, 12, 1, 30, tzinfo=UTC),
        judge_model_id="fake-judge-model",
        prompt=f"Judge answer for {question_id}.",
        raw_judge_output=f'{{"verdict": "{verdict.value}", "score": {score}}}',
        parsed_verdict=verdict,
        parsed_score=score,
        parsed_reasoning=reasoning,
        parse_was_successful=True,
        input_tokens=40,
        output_tokens=12,
        latency_ms=140.0,
        cost_usd=0.02,
    )


def _smoke_api_call(
    run_id: str,
    question_id: str,
    phase: str,
    cost_usd: float,
) -> APICallRecord:
    return APICallRecord(
        api_call_id=f"{run_id}-{question_id}-{phase}-api",
        run_id=run_id,
        question_id=question_id,
        timestamp=datetime(2026, 5, 7, 12, 1, 40, tzinfo=UTC),
        phase=phase,
        vendor="offline",
        model_id="fake-answer-model" if phase == "generate" else "fake-judge-model",
        input_tokens=32,
        output_tokens=4,
        latency_ms=120.0,
        cost_usd=cost_usd,
        status="success",
        attempt_number=1,
    )


def _smoke_experiment_result() -> ExperimentResult:
    aggregate = AggregateScore(
        experiment_id="experiment-smoke",
        category="overall",
        mode="audited",
        n_runs=2,
        pooled_score=_smoke_score(4, 2),
        individual_run_scores=[0.5, 0.5],
        mean=0.5,
        stddev=0.0,
        min=0.5,
        max=0.5,
    )
    return ExperimentResult(
        experiment_id="experiment-smoke",
        suite_id="suite-smoke",
        experiment_name="Synthetic dashboard experiment",
        n_runs=2,
        run_ids=["run-smoke-1", "run-smoke-2"],
        aggregate_overall_standard=aggregate.model_copy(update={"mode": "standard"}),
        aggregate_overall_audited=aggregate,
        aggregate_by_category_standard={
            "overall": aggregate.model_copy(update={"mode": "standard"})
        },
        aggregate_by_category_audited={"overall": aggregate},
        total_cost_usd=0.12,
        config=ExperimentConfig.model_validate(
            {
                "name": "Synthetic dashboard experiment",
                "provider": {"type": "full_context"},
                "benchmark": {"type": "locomo"},
                "answer_model": {"type": "fake", "model": "fake-answer-model"},
                "judge": {"type": "fake", "model": "fake-judge-model"},
            }
        ),
    )


def _smoke_score(n_total: int, n_correct: int) -> ScoreWithCI:
    low, high = wilson_score_interval(n_correct, n_total)
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_total - n_correct,
        point_estimate=n_correct / n_total,
        ci_95_low=low,
        ci_95_high=high,
    )


def _assert_page_ok(page: AppTest) -> None:
    assert not page.exception
    assert not page.error


def _run_dashboard_page(dashboard_path: Path, page_name: str) -> AppTest:
    page = AppTest.from_file(str(dashboard_path))
    page.run(timeout=15)
    if page_name != "Runs Overview":
        page_selector = next(element for element in page.selectbox if element.label == "Page")
        page_selector.set_value(page_name)
        page.run(timeout=15)
    return page


def _element_labels(elements: object) -> list[str]:
    return [str(element.label) for element in elements if hasattr(element, "label")]


def _assert_page_contains(page: AppTest, *expected_values: str) -> None:
    rendered_parts = [str(page)]
    for element_name in (
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "text",
        "info",
        "success",
        "warning",
        "metric",
        "selectbox",
        "multiselect",
        "radio",
        "text_input",
        "text_area",
        "button",
        "download_button",
        "dataframe",
        "table",
    ):
        for element in getattr(page, element_name, []):
            for attr in ("value", "label", "options"):
                if hasattr(element, attr):
                    value = getattr(element, attr)
                    if hasattr(value, "to_string"):
                        rendered_parts.append(value.to_string())
                    else:
                        rendered_parts.append(str(value))
    rendered = "\n".join(rendered_parts)
    missing = [value for value in expected_values if value not in rendered]

    assert missing == []


def _run(
    run_id: str,
    *,
    suite_id: str = "suite-1",
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    run_number: int = 1,
    status: str = "completed",
    started_at: datetime = datetime(2026, 5, 1, 12, tzinfo=UTC),
    finished_at: datetime | None = None,
    provider_type: str = "full_context",
    provider_version: str | None = None,
    benchmark_type: str = "locomo",
    answer_model_id: str = "answer-model",
    judge_model_id: str = "judge-model",
    n_questions_attempted: int = 10,
    n_questions_succeeded: int = 8,
    n_questions_errored: int = 0,
    overall_score_standard: float | None = 0.7,
    overall_score_audited: float | None = 0.8,
    total_cost_usd: float = 1.25,
    methodology_version: str = "v1.0",
    methodology_profile: str = "canonical-v1",
    framework_version: str = "0.1.0",
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        suite_id=suite_id,
        experiment_id=experiment_id or f"experiment-{run_id}",
        experiment_name=experiment_name or f"Experiment {run_id}",
        run_number=run_number,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        provider_type=provider_type,
        provider_version=provider_version,
        benchmark_type=benchmark_type,
        answer_model_id=answer_model_id,
        judge_model_id=judge_model_id,
        n_questions_attempted=n_questions_attempted,
        n_questions_succeeded=n_questions_succeeded,
        n_questions_errored=n_questions_errored,
        overall_score_standard=overall_score_standard,
        overall_score_audited=overall_score_audited,
        total_cost_usd=total_cost_usd,
        methodology_version=methodology_version,
        methodology_profile=methodology_profile,
        framework_version=framework_version,
        error_message=None,
        error_phase=None,
    )


def _score(
    *,
    n_total: int,
    n_correct: int,
    low: float = 0.7333,
    high: float = 0.8830,
) -> ScoreWithCI:
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_total - n_correct,
        point_estimate=n_correct / n_total,
        ci_95_low=low,
        ci_95_high=high,
    )


def _report_context(*, same_vendor: bool = False, n_failed: int = 2) -> ReportContext:
    started_at = datetime(2026, 5, 1, 12, tzinfo=UTC)
    failed_summaries = [
        FailedQuestionSummary(
            question_id=f"q-{index:02d}",
            category=QuestionCategory.TEMPORAL,
            verdict=JudgmentVerdict.INCORRECT,
            question_text=f"What happened in question {index}?",
            expected_answer="Expected answer",
            generated_answer="Wrong answer",
            error_phase=None,
            error_message=None,
        )
        for index in range(n_failed)
    ]
    question_records = [
        _question_record(summary.question_id, summary.verdict) for summary in failed_summaries
    ]
    question_records.append(_question_record("q-pass", JudgmentVerdict.CORRECT))

    return ReportContext(
        run_status=RunStatus(
            run_id="run-1",
            suite_id="suite-1",
            experiment_id="experiment-1",
            experiment_name="Experiment run-1",
            run_number=1,
            status="completed",
            started_at=started_at,
            finished_at=started_at + timedelta(minutes=1),
            provider_type="full_context",
            provider_version="2.5.0",
            answer_model_id="answer-model",
            judge_model_id="judge-model",
            methodology_version="v1.0",
            methodology_profile="canonical-v1",
            framework_version="0.1.0",
            n_conversations_processed=1,
            n_questions_attempted=len(question_records),
            n_questions_succeeded=1,
            n_questions_errored=0,
            overall_score_standard=_score(n_total=10, n_correct=7),
            overall_score_audited=_score(n_total=10, n_correct=8),
            by_category_standard={"temporal": _score(n_total=10, n_correct=7)},
            by_category_audited={"temporal": _score(n_total=10, n_correct=8)},
            total_cost_usd=0.09,
            error_message=None,
            error_phase=None,
        ),
        run_started_event=RunStartedEvent(
            event_id="run-started-1",
            timestamp=started_at,
            run_id="run-1",
            sequence_number=0,
            suite_id="suite-1",
            experiment_id="experiment-1",
            experiment_name="Experiment run-1",
            run_number=1,
            provider_type="full_context",
            provider_version="2.5.0",
            benchmark_type="locomo",
            benchmark_version="1.0",
            benchmark_checksum="sha256:synthetic",
            answer_model_id="answer-model",
            answer_model_vendor="openai",
            judge_model_id="judge-model",
            judge_model_vendor="openai" if same_vendor else "anthropic",
            config={"provider": {"type": "full_context"}},
            methodology_version="v1.0",
            methodology_profile="canonical-v1",
            framework_version="0.1.0",
            seed=123,
            runtime_environment={"python": "3.11"},
        ),
        suite_status=SuiteStatus(
            suite_id="suite-1",
            status="completed",
            started_at=started_at,
            finished_at=started_at + timedelta(minutes=1),
            config_yaml_content="experiments: []",
            methodology_version="v1.0",
            methodology_profile="canonical-v1",
            framework_version="0.1.0",
            n_experiments_planned=1,
            n_experiments_completed=1,
            n_experiments_failed=0,
            n_experiments_in_progress=0,
            total_cost_usd=0.09,
            last_event_at=started_at + timedelta(minutes=1),
            last_event_type="suite_completed",
        ),
        experiment_result=None,
        questions=question_records,
        failed_questions=[
            record for record in question_records if record.verdict != JudgmentVerdict.CORRECT
        ],
        failure_analysis=FailureAnalysisReport(
            run_id="run-1",
            category_breakdown=[
                CategoryFailureBreakdown(
                    category=QuestionCategory.TEMPORAL,
                    n_total=len(question_records),
                    n_failed=n_failed,
                    failure_rate=n_failed / len(question_records),
                )
            ],
            failed_questions=failed_summaries,
            patterns=[
                DetectedPattern(
                    pattern_name="temporal_arithmetic_failure",
                    description="Temporal answer mismatch.",
                    suggested_remedy="Inspect temporal reasoning.",
                    affected_question_ids=["q-00"],
                    n_affected_questions=1,
                    confidence=0.75,
                )
            ],
            traceability_index=[],
        ),
        methodology=MethodologyDisclosure(
            methodology_version="v1.0",
            methodology_profile="canonical-v1",
            scoring_mode="audited",
            confidence_interval="Wilson 95%",
            same_vendor_warning=same_vendor,
            benchmark_type="locomo",
            benchmark_version="1.0",
            benchmark_checksum="sha256:synthetic",
        ),
        cost_summary=CostSummary(
            total_cost_usd=0.09,
            cost_by_phase={"generate": 0.03, "judge": 0.06},
            cost_by_model={"answer-model": 0.03, "judge-model": 0.06},
        ),
    )


def _comparison_report(
    *,
    score_deltas: list[ScoreDelta] | None = None,
) -> ComparisonReport:
    return ComparisonReport(
        run_ids=["baseline-run", "candidate-run"],
        mode="audited",
        compatible=True,
        compatibility_warnings=[],
        score_deltas=score_deltas
        or [
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=_score(n_total=10, n_correct=8),
                candidate_score=_score(n_total=10, n_correct=9),
                point_delta=0.1,
                ci_overlaps=True,
                statistically_significant=False,
            )
        ],
        differing_questions=[],
    )


def _question_record(question_id: str, verdict: JudgmentVerdict) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=f"qe-{question_id}",
        run_id="run-1",
        question_id=question_id,
        conversation_id="conv-1",
        category=QuestionCategory.TEMPORAL,
        question_text=f"What happened in {question_id}?",
        expected_answer="Expected answer",
        is_audited_error=False,
        retrieval_id=None,
        retrieval_timestamp=None,
        retrieval_latency_ms=None,
        n_memories_retrieved=None,
        retrieved_memory_ids=[],
        response_id=None,
        generation_timestamp=None,
        generation_latency_ms=None,
        generated_answer=None,
        generation_input_tokens=None,
        generation_output_tokens=None,
        generation_cost_usd=None,
        judgment_id=None,
        judgment_timestamp=None,
        judgment_latency_ms=None,
        verdict=verdict,
        score=1.0 if verdict is JudgmentVerdict.CORRECT else 0.0,
        judgment_reasoning=None,
        judgment_input_tokens=None,
        judgment_output_tokens=None,
        judgment_cost_usd=None,
        total_latency_ms=None,
        total_cost_usd=0.0,
        error_message=None,
        error_phase=None,
    )
