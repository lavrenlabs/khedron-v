from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from khedron.analysis.types import (
    CategoryFailureBreakdown,
    ComparisonReport,
    DetectedPattern,
    FailedQuestionSummary,
    FailureAnalysisReport,
    QuestionDifference,
    ScoreDelta,
    TraceabilityIndexEntry,
)
from khedron.config import (
    AnswerModelConfig,
    BenchmarkConfig,
    ExperimentConfig,
    JudgeConfig,
    ProviderConfig,
)
from khedron.reporting.context import CostSummary, MethodologyDisclosure, ReportContext
from khedron.types import (
    AggregateScore,
    ExperimentResult,
    JudgmentVerdict,
    QuestionCategory,
    QuestionRecord,
    RunStartedEvent,
    RunStatus,
    ScoreWithCI,
    SuiteStatus,
)

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
GENERATED_AT = "2026-05-06T12:34:56Z"
GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden" / "reports"
TEMPLATE_DIR = files("khedron.reporting").joinpath("templates")
RUN_REPORT_TEMPLATES = [
    "executive_summary.md.j2",
    "technical_deep_dive.md.j2",
    "failure_analysis.md.j2",
]
FORBIDDEN_TEMPLATE_TERMS = [
    "RunRepository",
    "sqlite",
    "SELECT",
    "get_questions_for_run",
    "rebuild_sqlite",
]


def test_templates_render_exact_golden_markdown() -> None:
    context = report_context()
    comparison = comparison_report()
    cases = {
        "executive_summary.md.j2": ("executive_summary.md", {"context": context}),
        "technical_deep_dive.md.j2": ("technical_deep_dive.md", {"context": context}),
        "failure_analysis.md.j2": ("failure_analysis.md", {"context": context}),
        "_methodology_disclosure.md.j2": ("methodology_disclosure.md", {"context": context}),
        "comparison.md.j2": ("comparison.md", {"comparison": comparison}),
    }

    environment = template_environment()
    for template_name, (golden_name, variables) in cases.items():
        rendered = environment.get_template(template_name).render(
            generated_at=GENERATED_AT,
            **variables,
        )
        expected = (GOLDEN_DIR / golden_name).read_text(encoding="utf-8")

        assert rendered == expected


@pytest.mark.parametrize("template_name", RUN_REPORT_TEMPLATES)
def test_run_report_templates_include_methodology_disclosure(template_name: str) -> None:
    rendered = (
        template_environment()
        .get_template(template_name)
        .render(
            context=report_context(),
            generated_at=GENERATED_AT,
        )
    )

    assert "Methodology Disclosure" in rendered


@pytest.mark.parametrize(
    "template_name, input_name",
    [
        ("executive_summary.md.j2", "context"),
        ("technical_deep_dive.md.j2", "context"),
        ("comparison.md.j2", "comparison"),
    ],
)
def test_rendered_score_strings_include_confidence_intervals(
    template_name: str,
    input_name: str,
) -> None:
    variables: dict[str, object]
    if input_name == "context":
        variables = {"context": report_context()}
    else:
        variables = {"comparison": comparison_report()}

    rendered = (
        template_environment()
        .get_template(template_name)
        .render(
            generated_at=GENERATED_AT,
            **variables,
        )
    )

    assert "% [" in rendered
    assert "]" in rendered


def test_comparison_warnings_render_when_incompatible() -> None:
    rendered = (
        template_environment()
        .get_template("comparison.md.j2")
        .render(
            comparison=comparison_report(),
            generated_at=GENERATED_AT,
        )
    )

    assert "| Compatible | No |" in rendered
    assert "Methodology profile differs" in rendered
    assert "Question ID sets differ" in rendered


def test_comparison_without_question_differences_says_so() -> None:
    comparison = comparison_report().model_copy(update={"differing_questions": []})

    rendered = (
        template_environment()
        .get_template("comparison.md.j2")
        .render(
            comparison=comparison,
            generated_at=GENERATED_AT,
        )
    )

    assert "No question-level differences were detected." in rendered


def test_template_sources_do_not_contain_storage_terms() -> None:
    for template_path in Path(str(TEMPLATE_DIR)).glob("*.md.j2"):
        source = template_path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TEMPLATE_TERMS:
            assert forbidden not in source


def template_environment() -> Environment:
    return Environment(  # noqa: S701 - Markdown reports are not HTML-rendered here.
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
    )


def score(
    *,
    n_total: int,
    n_correct: int,
    low: float,
    high: float,
    n_partial: int = 0,
    n_unknown: int = 0,
) -> ScoreWithCI:
    n_errors = n_total - n_correct - n_partial - n_unknown
    return ScoreWithCI(
        n_total=n_total,
        n_correct=n_correct,
        n_errors=n_errors,
        n_partial=n_partial,
        n_unknown=n_unknown,
        point_estimate=n_correct / n_total,
        ci_95_low=low,
        ci_95_high=high,
    )


def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="Synthetic Memory Run",
        provider=ProviderConfig(type="full_context", config={"project": "benchmark"}),
        benchmark=BenchmarkConfig(type="locomo", audit_mode="both"),
        answer_model=AnswerModelConfig(type="anthropic", model="claude-sonnet-4-5"),
        judge=JudgeConfig(type="openai", model="gpt-4o-2024-08-06"),
        top_k_retrieval=10,
        max_concurrent_questions=4,
    )


def aggregate_score(
    *,
    category: QuestionCategory | str,
    mode: str,
    pooled_score: ScoreWithCI,
) -> AggregateScore:
    return AggregateScore(
        experiment_id="exp-1",
        category=category,
        mode=mode,
        n_runs=2,
        pooled_score=pooled_score,
        individual_run_scores=[pooled_score.point_estimate, pooled_score.point_estimate],
        mean=pooled_score.point_estimate,
        stddev=0.0,
        min=pooled_score.point_estimate,
        max=pooled_score.point_estimate,
    )


def question_record(
    question_id: str,
    *,
    category: QuestionCategory,
    verdict: JudgmentVerdict,
    score_value: float,
    generated_answer: str | None,
    error_phase: str | None = None,
    error_message: str | None = None,
) -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id=f"qe-{question_id}",
        run_id="run-1",
        question_id=question_id,
        conversation_id="conv-1",
        category=category,
        question_text=f"What is the answer for {question_id}?",
        expected_answer="Expected answer",
        is_audited_error=False,
        retrieval_id=f"ret-{question_id}",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=2,
        retrieved_memory_ids=[f"mem-{question_id}-1", f"mem-{question_id}-2"],
        response_id=f"resp-{question_id}",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer=generated_answer,
        generation_input_tokens=32,
        generation_output_tokens=5,
        generation_cost_usd=0.01,
        judgment_id=f"judge-{question_id}",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=verdict,
        score=score_value,
        judgment_reasoning="Synthetic judgment.",
        judgment_input_tokens=50,
        judgment_output_tokens=10,
        judgment_cost_usd=0.02,
        total_latency_ms=392.5,
        total_cost_usd=0.03,
        error_message=error_message,
        error_phase=error_phase,
    )


def failure_analysis_report() -> FailureAnalysisReport:
    return FailureAnalysisReport(
        run_id="run-1",
        category_breakdown=[
            CategoryFailureBreakdown(
                category=QuestionCategory.MULTI_HOP,
                n_total=1,
                n_failed=1,
                failure_rate=1.0,
            ),
            CategoryFailureBreakdown(
                category=QuestionCategory.TEMPORAL,
                n_total=2,
                n_failed=1,
                failure_rate=0.5,
            ),
        ],
        failed_questions=[
            FailedQuestionSummary(
                question_id="q-temporal",
                category=QuestionCategory.TEMPORAL,
                verdict=JudgmentVerdict.INCORRECT,
                question_text="What is the answer for q-temporal?",
                expected_answer="Expected answer",
                generated_answer="Wrong month",
                error_phase=None,
                error_message=None,
            ),
            FailedQuestionSummary(
                question_id="q-multihop",
                category=QuestionCategory.MULTI_HOP,
                verdict=JudgmentVerdict.ERROR,
                question_text="What is the answer for q-multihop?",
                expected_answer="Expected answer",
                generated_answer=None,
                error_phase="generate",
                error_message="Generation timed out",
            ),
        ],
        patterns=[
            DetectedPattern(
                pattern_name="temporal_arithmetic_failure",
                description="Temporal answer was inconsistent with the expected date.",
                suggested_remedy="Add explicit date reasoning to the answer prompt.",
                affected_question_ids=["q-temporal"],
                n_affected_questions=1,
                confidence=0.75,
            )
        ],
        traceability_index=[
            TraceabilityIndexEntry(
                question_id="q-temporal",
                retrieval_id="ret-q-temporal",
                response_id="resp-q-temporal",
                judgment_id="judge-q-temporal",
            ),
            TraceabilityIndexEntry(
                question_id="q-multihop",
                retrieval_id="ret-q-multihop",
                response_id="resp-q-multihop",
                judgment_id="judge-q-multihop",
            ),
        ],
    )


def report_context() -> ReportContext:
    audited_overall = score(n_total=15, n_correct=12, low=0.548, high=0.930)
    standard_overall = score(n_total=20, n_correct=14, low=0.481, high=0.855)
    audited_temporal = score(n_total=6, n_correct=4, low=0.300, high=0.903)
    audited_single = score(n_total=9, n_correct=8, low=0.565, high=0.980)
    standard_temporal = score(n_total=5, n_correct=3, low=0.231, high=0.882)
    standard_single = score(n_total=10, n_correct=8, low=0.490, high=0.943)
    config = experiment_config()
    failure_analysis = failure_analysis_report()
    q_temporal = question_record(
        "q-temporal",
        category=QuestionCategory.TEMPORAL,
        verdict=JudgmentVerdict.INCORRECT,
        score_value=0.0,
        generated_answer="Wrong month",
    )
    q_multihop = question_record(
        "q-multihop",
        category=QuestionCategory.MULTI_HOP,
        verdict=JudgmentVerdict.ERROR,
        score_value=0.0,
        generated_answer=None,
        error_phase="generate",
        error_message="Generation timed out",
    )
    q_single = question_record(
        "q-single",
        category=QuestionCategory.SINGLE_HOP,
        verdict=JudgmentVerdict.CORRECT,
        score_value=1.0,
        generated_answer="Expected answer",
    )

    return ReportContext(
        run_status=RunStatus(
            run_id="run-1",
            suite_id="suite-1",
            experiment_id="exp-1",
            experiment_name="Synthetic Memory Run",
            run_number=1,
            status="completed",
            started_at=NOW,
            finished_at=NOW,
            provider_type="full_context",
            provider_version="2.5.0",
            answer_model_id="claude-sonnet-4-5",
            judge_model_id="gpt-4o-2024-08-06",
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            framework_version="0.0.0",
            n_conversations_processed=1,
            n_questions_attempted=20,
            n_questions_succeeded=18,
            n_questions_errored=1,
            overall_score_standard=standard_overall,
            overall_score_audited=audited_overall,
            by_category_standard={
                "temporal": standard_temporal,
                "single_hop": standard_single,
            },
            by_category_audited={
                "temporal": audited_temporal,
                "single_hop": audited_single,
            },
            total_cost_usd=0.09,
            error_message=None,
            error_phase=None,
        ),
        run_started_event=RunStartedEvent(
            event_id="event-run-started-1",
            timestamp=NOW,
            run_id="run-1",
            sequence_number=0,
            suite_id="suite-1",
            experiment_id="exp-1",
            experiment_name="Synthetic Memory Run",
            run_number=1,
            provider_type="full_context",
            provider_version="2.5.0",
            benchmark_type="locomo",
            benchmark_version="1.0",
            benchmark_checksum="sha256:synthetic",
            answer_model_id="claude-sonnet-4-5",
            answer_model_vendor="anthropic",
            judge_model_id="gpt-4o-2024-08-06",
            judge_model_vendor="openai",
            config=config.model_dump(),
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            framework_version="0.0.0",
            seed=42,
            runtime_environment={"python": "3.11"},
        ),
        suite_status=SuiteStatus(
            suite_id="suite-1",
            status="completed",
            started_at=NOW,
            finished_at=NOW,
            config_yaml_content="experiments: []",
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            framework_version="0.0.0",
            n_experiments_planned=1,
            n_experiments_completed=1,
            n_experiments_failed=0,
            n_experiments_in_progress=0,
            total_cost_usd=0.09,
            last_event_at=NOW,
            last_event_type="suite_completed",
        ),
        experiment_result=ExperimentResult(
            experiment_id="exp-1",
            suite_id="suite-1",
            experiment_name="Synthetic Memory Run",
            n_runs=2,
            run_ids=["run-1", "run-2"],
            aggregate_overall_standard=aggregate_score(
                category="overall",
                mode="standard",
                pooled_score=standard_overall,
            ),
            aggregate_overall_audited=aggregate_score(
                category="overall",
                mode="audited",
                pooled_score=audited_overall,
            ),
            aggregate_by_category_standard={
                "temporal": aggregate_score(
                    category=QuestionCategory.TEMPORAL,
                    mode="standard",
                    pooled_score=standard_temporal,
                )
            },
            aggregate_by_category_audited={
                "temporal": aggregate_score(
                    category=QuestionCategory.TEMPORAL,
                    mode="audited",
                    pooled_score=audited_temporal,
                )
            },
            total_cost_usd=0.18,
            config=config,
        ),
        questions=[q_temporal, q_multihop, q_single],
        failed_questions=[q_temporal, q_multihop],
        failure_analysis=failure_analysis,
        methodology=MethodologyDisclosure(
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            scoring_mode="audited",
            confidence_interval="Wilson 95%",
            same_vendor_warning=False,
            benchmark_type="locomo",
            benchmark_version="1.0",
            benchmark_checksum="sha256:synthetic",
        ),
        cost_summary=CostSummary(
            total_cost_usd=0.09,
            cost_by_phase={"generation": 0.03, "judgment": 0.06},
            cost_by_model={"claude-sonnet-4-5": 0.03, "gpt-4o-2024-08-06": 0.06},
        ),
    )


def comparison_report() -> ComparisonReport:
    return ComparisonReport(
        run_ids=["baseline-run", "candidate-run"],
        mode="audited",
        compatible=False,
        compatibility_warnings=[
            "Methodology profile differs: baseline=canonical-v1 candidate=canonical-v2",
            "Question ID sets differ: q-baseline-only / q-candidate-only",
        ],
        score_deltas=[
            ScoreDelta(
                category="overall",
                mode="audited",
                baseline_score=score(n_total=20, n_correct=14, low=0.481, high=0.855),
                candidate_score=score(n_total=20, n_correct=16, low=0.584, high=0.919),
                point_delta=0.10,
                ci_overlaps=True,
                statistically_significant=False,
            ),
            ScoreDelta(
                category=QuestionCategory.TEMPORAL,
                mode="audited",
                baseline_score=score(n_total=10, n_correct=4, low=0.168, high=0.687),
                candidate_score=score(n_total=10, n_correct=7, low=0.397, high=0.892),
                point_delta=0.30,
                ci_overlaps=False,
                statistically_significant=True,
            ),
            ScoreDelta(
                category=QuestionCategory.MULTI_HOP,
                mode="audited",
                baseline_score=None,
                candidate_score=score(n_total=6, n_correct=3, low=0.188, high=0.812),
                point_delta=None,
                ci_overlaps=None,
                statistically_significant=None,
            ),
        ],
        differing_questions=[
            QuestionDifference(
                question_id="q-temporal",
                category=QuestionCategory.TEMPORAL,
                verdicts_by_run_id={
                    "baseline-run": JudgmentVerdict.CORRECT,
                    "candidate-run": JudgmentVerdict.INCORRECT,
                },
                scores_by_run_id={"baseline-run": 1.0, "candidate-run": 0.0},
            )
        ],
    )


def test_the_disclosure_says_not_applicable_when_the_provider_ignores_the_budget() -> None:
    # A full-context baseline applies no retrieval budget, but `ExperimentConfig` supplies a
    # default of 10 regardless -- so the report would have stated "Top-K retrieval 10" for a run
    # that never used one. An artifact asserting something false about its own execution is the
    # defect class this methodology version exists to remove.
    from khedron.reporting.context import MethodologyDisclosure

    applies = MethodologyDisclosure(
        methodology_version="2.0",
        methodology_profile="canonical-v2",
        scoring_mode="audited",
        confidence_interval="Wilson 95%",
        same_vendor_warning=False,
        benchmark_type="locomo",
        benchmark_version="v1",
        benchmark_checksum="sha256:x",
        retrieval_budget_applies=True,
    )
    ignores = applies.model_copy(update={"retrieval_budget_applies": False})

    if applies.retrieval_budget_label(200) != "200":
        raise AssertionError(applies.retrieval_budget_label(200))
    label = ignores.retrieval_budget_label(10)
    if "not applicable" not in label:
        raise AssertionError(label)
    # The number the run never used must not appear at all.
    if "10" in label:
        raise AssertionError(label)


def test_the_scope_label_names_the_answerable_subset() -> None:
    from khedron.reporting.context import MethodologyDisclosure

    base = MethodologyDisclosure(
        methodology_version="2.0",
        methodology_profile="canonical-v2",
        scoring_mode="audited",
        confidence_interval="Wilson 95%",
        same_vendor_warning=False,
        benchmark_type="locomo",
        benchmark_version="v1",
        benchmark_checksum="sha256:x",
        scored_categories=["single_hop", "multi_hop", "temporal", "open_domain"],
    )

    label = base.scored_scope_label
    if "answerable subset" not in label or "single_hop" not in label:
        raise AssertionError(label)
    if "adversarial" in label:
        raise AssertionError(f"the excluded category must not appear in the scope: {label}")
    # And the pre-split default still reads sensibly.
    if base.model_copy(update={"scored_categories": None}).scored_scope_label != (
        "all evaluated categories"
    ):
        raise AssertionError(base.scored_scope_label)


def _disclosure_with_corpus(evaluated: int | None, available: int | None) -> object:
    from khedron.reporting.context import MethodologyDisclosure

    return MethodologyDisclosure(
        methodology_version="2.0",
        methodology_profile="canonical-v2",
        scoring_mode="audited",
        confidence_interval="Wilson 95%",
        same_vendor_warning=False,
        benchmark_type="locomo",
        benchmark_version="v1",
        benchmark_checksum="sha256:x",
        corpus_conversations_evaluated=evaluated,
        corpus_conversations_available=available,
    )


def test_a_partial_corpus_is_disclosed_as_not_a_measurement() -> None:
    # A partial corpus run (e.g. three runs over one conversation) must not read like a full
    # measurement: nothing else in the report distinguishes it -- same headline phrasing, same
    # Wilson intervals, arithmetically valid over the sampled questions and meaningless as a
    # claim about the benchmark.
    partial = _disclosure_with_corpus(1, 10)

    if partial.corpus_is_complete is not False:
        raise AssertionError(partial.corpus_is_complete)
    label = partial.corpus_scope_label
    if "1 of 10" not in label or "not a measurement" not in label:
        raise AssertionError(label)


def test_a_complete_corpus_reads_as_complete_and_an_unrecorded_one_does_not() -> None:
    complete = _disclosure_with_corpus(10, 10)
    if complete.corpus_is_complete is not True:
        raise AssertionError(complete.corpus_is_complete)
    if complete.corpus_scope_label != "all 10 conversations":
        raise AssertionError(complete.corpus_scope_label)

    # A run recorded before the field existed is unknown, not complete. Defaulting the other way
    # would let every pre-existing artifact assert full coverage it never demonstrated.
    unknown = _disclosure_with_corpus(None, None)
    if unknown.corpus_is_complete is not None:
        raise AssertionError(unknown.corpus_is_complete)
    if unknown.corpus_scope_label != "not recorded":
        raise AssertionError(unknown.corpus_scope_label)


def test_the_executive_summary_banners_a_partial_corpus_above_the_score() -> None:
    context = report_context()
    partial = context.model_copy(
        update={"methodology": _disclosure_with_corpus(1, 10)},
    )

    rendered = (
        template_environment()
        .get_template("executive_summary.md.j2")
        .render(generated_at=GENERATED_AT, context=partial)
    )

    if "This run is not a measurement." not in rendered:
        raise AssertionError(rendered[:800])
    # The banner must precede the number it qualifies; a caveat below the headline is a footnote.
    if rendered.index("not a measurement") > rendered.index("achieved"):
        raise AssertionError("the disclaimer renders after the score it qualifies")
    # And a complete run carries no banner at all.
    whole = context.model_copy(update={"methodology": _disclosure_with_corpus(10, 10)})
    if "not a measurement" in template_environment().get_template("executive_summary.md.j2").render(
        generated_at=GENERATED_AT, context=whole
    ):
        raise AssertionError("a full-corpus run was labelled a non-measurement")


def test_a_single_run_report_discloses_the_retrieval_budget_it_used() -> None:
    # A single-run report has no aggregate, and the template printed "n/a" -- so a run that
    # retrieved 200 memories disclosed nothing about the one parameter canonical-v2 exists to pin.
    # Caught by reading a live single-run report rather than by any test.
    context = report_context()
    single = context.model_copy(update={"experiment_result": None})

    rendered = (
        template_environment()
        .get_template("_methodology_disclosure.md.j2")
        .render(generated_at=GENERATED_AT, context=single)
    )

    top_k = single.run_started_event.config.get("top_k_retrieval")
    if top_k is None:
        raise AssertionError("fixture must carry a top_k to make this meaningful")
    if f"| Top-K retrieval | {top_k} |" not in rendered:
        raise AssertionError([line for line in rendered.splitlines() if "Top-K" in line])


def test_lost_turns_are_named_and_located_in_the_report() -> None:
    # A non-zero ingestion tolerance is acceptable only on the condition that every report
    # states the rate actually observed and where it fell. A tolerated loss that goes
    # unreported is a silent one: the run answers the same questions and computes the same
    # intervals, having simply had less to retrieve from.
    from khedron.reporting.context import IngestionSummary

    lossy = IngestionSummary(
        n_turns=6750, n_turns_failed=8, failed_by_conversation={"conv-26": 5, "conv-44": 3}
    )
    label = lossy.label
    for expected in ("8 of 6750", "conv-26: 5", "conv-44: 3", "0.119%"):
        if expected not in label:
            raise AssertionError(f"{expected!r} missing from {label!r}")

    clean = IngestionSummary(n_turns=6750, n_turns_failed=0)
    if clean.label != "complete (6750 turns, none lost)":
        raise AssertionError(clean.label)
    # A run with no ingestion records at all must not read as a clean one.
    if IngestionSummary().label != "not recorded":
        raise AssertionError(IngestionSummary().label)
