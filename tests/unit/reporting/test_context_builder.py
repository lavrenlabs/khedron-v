from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from khedron.analysis import FailureAnalyzer
from khedron.analysis.types import (
    CategoryFailureBreakdown,
    FailedQuestionSummary,
    FailureAnalysisReport,
    TraceabilityIndexEntry,
)
from khedron.errors import ConfigurationError, PersistenceError
from khedron.reporting import ReportContextBuilder
from khedron.types import (
    AggregateScore,
    APICallRecord,
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
_MISSING = object()


def test_builder_assembles_report_context_from_repository_fields() -> None:
    failed = question_record().model_copy(update={"question_id": "q-failed"})
    correct = question_record().model_copy(
        update={
            "question_id": "q-correct",
            "verdict": JudgmentVerdict.CORRECT,
            "score": 1.0,
        }
    )
    hydrated_failed = failed.model_copy(update={"generated_answer": "Hydrated answer"})
    analysis = failure_analysis_report()
    repository = _FakeRepository(
        questions=[correct, failed],
        hydrated_by_id={"q-failed": hydrated_failed},
        api_calls=[
            api_call("api-generate", phase="generate", model_id="answer-model", cost=0.01),
            api_call("api-judge", phase="judge", model_id="judge-model", cost=0.02),
        ],
    )
    analyzer = _RecordingFailureAnalyzer(analysis)

    context = ReportContextBuilder(failure_analyzer=analyzer).build("run-1", repository)

    assert context.run_status == repository.run_status
    assert context.run_started_event == repository.run_started
    assert context.suite_status == repository.suite_status
    assert context.experiment_result == repository.experiment_result
    assert context.questions == [correct, failed]
    assert context.failed_questions == [hydrated_failed]
    assert context.failure_analysis == analysis
    assert context.methodology.methodology_version == repository.run_status.methodology_version
    assert context.methodology.methodology_profile == repository.run_status.methodology_profile
    assert context.methodology.scoring_mode == "audited"
    assert context.methodology.confidence_interval == "Wilson 95%"
    assert context.methodology.benchmark_type == repository.run_started.benchmark_type
    assert context.methodology.benchmark_version == repository.run_started.benchmark_version
    assert context.methodology.benchmark_checksum == repository.run_started.benchmark_checksum
    assert context.cost_summary.total_cost_usd == repository.run_status.total_cost_usd
    assert analyzer.calls == [("run-1", repository)]


def test_invalid_mode_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
            "run-1",
            _FakeRepository(),
            mode="both",
        )


def test_failed_question_hydration_falls_back_to_list_record_when_lookup_fails() -> None:
    failed = question_record().model_copy(update={"question_id": "q-failed"})
    repository = _FakeRepository(
        questions=[failed],
        missing_hydration_ids={"q-failed"},
    )

    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        repository,
    )

    assert context.failed_questions == [failed]


def test_same_vendor_warning_is_true_for_matching_answer_and_judge_vendors() -> None:
    repository = _FakeRepository(
        run_started=run_started_event().model_copy(update={"judge_model_vendor": "openai"})
    )

    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        repository,
    )

    assert context.methodology.same_vendor_warning is True


def test_methodology_runtime_reference_metadata_is_disclosed() -> None:
    repository = _FakeRepository(
        run_started=run_started_event().model_copy(
            update={
                "runtime_environment": {
                    "python": "3.11",
                    "methodology_profile_fingerprint": "a" * 64,
                    "methodology_profile_reference": "LoCoMo benchmark (Snap Research)",
                    "methodology_profile_reference_url": "https://github.com/snap-research/locomo",
                    "methodology_profile_reference_commit": (
                        "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
                    ),
                }
            }
        )
    )

    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        repository,
    )

    assert context.methodology.methodology_fingerprint == "a" * 64
    assert context.methodology.reference_name == "LoCoMo benchmark (Snap Research)"
    assert context.methodology.reference_url == "https://github.com/snap-research/locomo"
    assert context.methodology.reference_commit == "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"


def test_cost_summary_aggregates_api_calls_by_phase_and_model_id() -> None:
    repository = _FakeRepository(
        api_calls=[
            api_call("api-1", phase="generate", model_id="answer-model", cost=0.01),
            api_call("api-2", phase="generate", model_id="answer-model", cost=0.02),
            api_call("api-3", phase="judge", model_id="judge-model", cost=0.03),
        ]
    )

    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        repository,
    )

    assert context.cost_summary.total_cost_usd == repository.run_status.total_cost_usd
    assert context.cost_summary.cost_by_phase == {"generate": 0.03, "judge": 0.03}
    assert context.cost_summary.cost_by_model == {"answer-model": 0.03, "judge-model": 0.03}


def test_cost_breakdowns_are_empty_when_no_api_calls_exist() -> None:
    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        _FakeRepository(api_calls=[]),
    )

    assert context.cost_summary.cost_by_phase == {}
    assert context.cost_summary.cost_by_model == {}


def test_missing_experiment_result_is_preserved_as_none() -> None:
    context = ReportContextBuilder(failure_analyzer=_RecordingFailureAnalyzer()).build(
        "run-1",
        _FakeRepository(experiment_result_value=None),
    )

    assert context.experiment_result is None


class _RecordingFailureAnalyzer(FailureAnalyzer):
    def __init__(self, report: FailureAnalysisReport | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self._report = report or failure_analysis_report()

    def analyze(self, run_id: str, repository: object) -> FailureAnalysisReport:
        self.calls.append((run_id, repository))
        return self._report


class _FakeRepository:
    conversation_ingestions: ClassVar[list[Any]] = []

    def __init__(
        self,
        *,
        questions: Sequence[QuestionRecord] | None = None,
        hydrated_by_id: dict[str, QuestionRecord] | None = None,
        missing_hydration_ids: set[str] | None = None,
        api_calls: Sequence[APICallRecord] | None = None,
        run_started: RunStartedEvent | None = None,
        experiment_result_value: ExperimentResult | None | object = _MISSING,
    ) -> None:
        self.run_status = run_status()
        self.run_started = run_started or run_started_event()
        self.suite_status = suite_status()
        self.experiment_result = (
            experiment_result() if experiment_result_value is _MISSING else experiment_result_value
        )
        self.questions = list(questions or [question_record()])
        self.hydrated_by_id = hydrated_by_id or {}
        self.missing_hydration_ids = missing_hydration_ids or set()
        self.api_calls = list(api_calls or [])

    def get_run_status(self, run_id: str) -> object:
        assert run_id == "run-1"
        return self.run_status

    def get_run_started_event(self, run_id: str) -> object:
        assert run_id == "run-1"
        return self.run_started

    def get_suite_status(self, suite_id: str) -> object:
        assert suite_id == self.run_started.suite_id
        return self.suite_status

    def get_experiment_result_for_run(self, run_id: str) -> object | None:
        assert run_id == "run-1"
        return self.experiment_result

    def get_questions_for_run(self, run_id: str) -> list[QuestionRecord]:
        assert run_id == "run-1"
        return list(self.questions)

    def get_question_record(self, run_id: str, question_id: str) -> QuestionRecord:
        assert run_id == "run-1"
        if question_id in self.missing_hydration_ids:
            raise PersistenceError(
                "Question record does not exist",
                run_id=run_id,
                question_id=question_id,
            )
        return self.hydrated_by_id.get(
            question_id,
            next(record for record in self.questions if record.question_id == question_id),
        )

    def get_conversation_ingestions_for_run(self, run_id: str) -> list[Any]:
        del run_id
        return list(self.conversation_ingestions)

    def get_api_calls_for_run(self, run_id: str) -> list[APICallRecord]:
        assert run_id == "run-1"
        return list(self.api_calls)


def api_call(
    api_call_id: str,
    *,
    phase: str,
    model_id: str,
    cost: float,
) -> APICallRecord:
    return APICallRecord(
        api_call_id=api_call_id,
        run_id="run-1",
        question_id="q-1",
        timestamp=NOW,
        phase=phase,
        vendor="synthetic",
        model_id=model_id,
        input_tokens=10,
        output_tokens=2,
        latency_ms=5.0,
        cost_usd=cost,
        status="success",
        attempt_number=1,
    )


def score() -> ScoreWithCI:
    return ScoreWithCI(
        n_total=2,
        n_correct=1,
        n_errors=1,
        point_estimate=0.5,
        ci_95_low=0.09,
        ci_95_high=0.91,
    )


def run_status() -> RunStatus:
    score_value = score()
    return RunStatus(
        run_id="run-1",
        suite_id="suite-1",
        experiment_id="experiment-1",
        experiment_name="Synthetic experiment",
        run_number=0,
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        provider_type="full_context",
        provider_version="0.1.0",
        answer_model_id="answer-model",
        judge_model_id="judge-model",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_conversations_processed=1,
        n_questions_attempted=2,
        n_questions_succeeded=1,
        n_questions_errored=1,
        overall_score_standard=score_value,
        overall_score_audited=score_value,
        by_category_standard={"single_hop": score_value},
        by_category_audited={"single_hop": score_value},
        total_cost_usd=0.06,
        error_message=None,
        error_phase=None,
    )


def run_started_event() -> RunStartedEvent:
    return RunStartedEvent(
        event_id="run-1-started",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=0,
        suite_id="suite-1",
        experiment_id="experiment-1",
        experiment_name="Synthetic experiment",
        run_number=0,
        provider_type="full_context",
        provider_version="0.1.0",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="sha256:synthetic",
        answer_model_id="answer-model",
        answer_model_vendor="openai",
        judge_model_id="judge-model",
        judge_model_vendor="anthropic",
        config={},
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=123,
        runtime_environment={"python": "3.11"},
    )


def suite_status() -> SuiteStatus:
    return SuiteStatus(
        suite_id="suite-1",
        status="completed",
        started_at=NOW,
        finished_at=NOW,
        config_yaml_content="experiments: []",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        n_experiments_completed=1,
        n_experiments_failed=0,
        n_experiments_in_progress=0,
        total_cost_usd=0.06,
        last_event_at=NOW,
        last_event_type="suite_completed",
    )


def experiment_result() -> ExperimentResult:
    aggregate = AggregateScore(
        experiment_id="experiment-1",
        category="overall",
        mode="audited",
        n_runs=1,
        pooled_score=score(),
        individual_run_scores=[0.5],
        mean=0.5,
        stddev=0.0,
        min=0.5,
        max=0.5,
    )
    return ExperimentResult.model_validate(
        {
            "experiment_id": "experiment-1",
            "suite_id": "suite-1",
            "experiment_name": "Synthetic experiment",
            "n_runs": 1,
            "run_ids": ["run-1"],
            "aggregate_overall_standard": aggregate,
            "aggregate_overall_audited": aggregate,
            "aggregate_by_category_standard": {"overall": aggregate},
            "aggregate_by_category_audited": {"overall": aggregate},
            "total_cost_usd": 0.06,
            "config": {
                "name": "Synthetic experiment",
                "provider": {"type": "full_context"},
                "benchmark": {"type": "locomo"},
                "answer_model": {"type": "openai", "model": "answer-model"},
                "judge": {"type": "anthropic", "model": "judge-model"},
            },
        }
    )


def question_record() -> QuestionRecord:
    return QuestionRecord(
        question_evaluation_id="qe-1",
        run_id="run-1",
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Where did Alice move?",
        expected_answer="Rome",
        is_audited_error=False,
        retrieval_id="ret-1",
        retrieval_timestamp=NOW,
        retrieval_latency_ms=12.5,
        n_memories_retrieved=1,
        retrieved_memory_ids=["mem-1"],
        response_id="resp-1",
        generation_timestamp=NOW,
        generation_latency_ms=200.0,
        generated_answer="Paris",
        generation_input_tokens=32,
        generation_output_tokens=3,
        generation_cost_usd=0.01,
        judgment_id="judge-1",
        judgment_timestamp=NOW,
        judgment_latency_ms=180.0,
        verdict=JudgmentVerdict.INCORRECT,
        score=0.0,
        judgment_reasoning="The answer does not match.",
        judgment_input_tokens=50,
        judgment_output_tokens=10,
        judgment_cost_usd=0.02,
        total_latency_ms=392.5,
        total_cost_usd=0.03,
        error_message=None,
        error_phase=None,
    )


def failure_analysis_report() -> FailureAnalysisReport:
    return FailureAnalysisReport(
        run_id="run-1",
        category_breakdown=[
            CategoryFailureBreakdown(
                category=QuestionCategory.SINGLE_HOP,
                n_total=1,
                n_failed=1,
                failure_rate=1.0,
            )
        ],
        failed_questions=[
            FailedQuestionSummary(
                question_id="q-1",
                category=QuestionCategory.SINGLE_HOP,
                verdict=JudgmentVerdict.INCORRECT,
                question_text="Where did Alice move?",
                expected_answer="Rome",
                generated_answer="Paris",
                error_phase=None,
                error_message=None,
            )
        ],
        patterns=[],
        traceability_index=[
            TraceabilityIndexEntry(
                question_id="q-1",
                retrieval_id="ret-1",
                response_id="resp-1",
                judgment_id="judge-1",
            )
        ],
    )
