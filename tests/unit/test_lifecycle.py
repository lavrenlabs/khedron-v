from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from khedron.types import (
    ConversationProcessedEvent,
    ExperimentCompletedEvent,
    ExperimentFailedEvent,
    ExperimentStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunLifecycleEvent,
    RunStartedEvent,
    ScoreWithCI,
    SuiteCompletedEvent,
    SuiteFailedEvent,
    SuiteLifecycleEvent,
    SuiteStartedEvent,
)

NOW = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)


def score() -> ScoreWithCI:
    return ScoreWithCI(
        n_total=10,
        n_correct=8,
        n_errors=2,
        n_partial=1,
        n_unknown=0,
        point_estimate=0.8,
        ci_95_low=0.49,
        ci_95_high=0.94,
    )


def suite_started_event() -> SuiteStartedEvent:
    return SuiteStartedEvent(
        event_id="event-suite-started",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=0,
        config_yaml_path="experiments/quickstart.yaml",
        config_yaml_content="experiments: []",
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        n_experiments_planned=1,
        runtime_environment={"python": "3.11", "os": "Windows"},
    )


def experiment_started_event() -> ExperimentStartedEvent:
    return ExperimentStartedEvent(
        event_id="event-experiment-started",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=1,
        experiment_id="experiment-1",
        experiment_name="Synthetic experiment",
        n_runs_planned=1,
    )


def experiment_completed_event() -> ExperimentCompletedEvent:
    score_value = score()
    return ExperimentCompletedEvent(
        event_id="event-experiment-completed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=2,
        experiment_id="experiment-1",
        overall_standard_pooled_score=score_value,
        overall_standard_mean=0.8,
        overall_standard_stddev=0.0,
        overall_audited_pooled_score=score_value,
        overall_audited_mean=0.8,
        overall_audited_stddev=0.0,
        cost_usd=0.12,
    )


def experiment_failed_event() -> ExperimentFailedEvent:
    return ExperimentFailedEvent(
        event_id="event-experiment-failed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=2,
        experiment_id="experiment-1",
        error_message="Provider initialization failed.",
        error_phase="initialization",
    )


def suite_completed_event() -> SuiteCompletedEvent:
    return SuiteCompletedEvent(
        event_id="event-suite-completed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=3,
        total_cost_usd=0.12,
        n_experiments_completed=1,
        n_experiments_failed=0,
    )


def suite_failed_event() -> SuiteFailedEvent:
    return SuiteFailedEvent(
        event_id="event-suite-failed",
        timestamp=NOW,
        suite_id="suite-1",
        sequence_number=3,
        error_message="Suite halted.",
        n_experiments_completed_before_failure=0,
    )


def run_started_event() -> RunStartedEvent:
    return RunStartedEvent(
        event_id="event-run-started",
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
        answer_model_id="gpt-4o-mini-2024-07-18",
        answer_model_vendor="openai",
        judge_model_id="gpt-4o-2024-08-06",
        judge_model_vendor="openai",
        config={"provider": {"type": "full_context"}},
        methodology_version="1.0",
        methodology_profile="canonical-v1",
        framework_version="0.1.0",
        seed=123,
        runtime_environment={"python": "3.11", "os": "Windows"},
    )


def conversation_processed_event() -> ConversationProcessedEvent:
    return ConversationProcessedEvent(
        event_id="event-conversation-processed",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=1,
        conversation_id="conversation-1",
        n_questions_evaluated=10,
        n_questions_correct=8,
        n_questions_errored=2,
        cost_usd=0.05,
    )


def run_completed_event() -> RunCompletedEvent:
    score_value = score()
    return RunCompletedEvent(
        event_id="event-run-completed",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=2,
        status="partial",
        n_questions_attempted=10,
        n_questions_succeeded=8,
        n_questions_errored=2,
        overall_score_standard=score_value,
        overall_score_audited=score_value,
        by_category_standard={"single_hop": score_value},
        by_category_audited={"single_hop": score_value},
        total_cost_usd=0.05,
    )


def run_failed_event() -> RunFailedEvent:
    return RunFailedEvent(
        event_id="event-run-failed",
        timestamp=NOW,
        run_id="run-1",
        sequence_number=2,
        error_message="Quota exhausted.",
        error_phase="generate",
        n_questions_completed_before_failure=5,
        partial_cost_usd=0.04,
    )


def suite_events() -> list[BaseModel]:
    return [
        suite_started_event(),
        experiment_started_event(),
        experiment_completed_event(),
        experiment_failed_event(),
        suite_completed_event(),
        suite_failed_event(),
    ]


def run_events() -> list[BaseModel]:
    return [
        run_started_event(),
        conversation_processed_event(),
        run_completed_event(),
        run_failed_event(),
    ]


@pytest.mark.parametrize("event", suite_events())
def test_suite_lifecycle_events_round_trip_through_model_dump(event: BaseModel) -> None:
    clone = type(event).model_validate(event.model_dump())
    if clone != event:
        raise AssertionError((clone, event))


@pytest.mark.parametrize("event", run_events())
def test_run_lifecycle_events_round_trip_through_model_dump(event: BaseModel) -> None:
    clone = type(event).model_validate(event.model_dump())
    if clone != event:
        raise AssertionError((clone, event))


@pytest.mark.parametrize("event", [*suite_events(), *run_events()])
def test_lifecycle_events_round_trip_through_json(event: BaseModel) -> None:
    clone = type(event).model_validate_json(event.model_dump_json())
    if clone != event:
        raise AssertionError((clone, event))


def test_json_round_trip_preserves_datetimes_and_nested_scores() -> None:
    event = run_completed_event()
    clone = RunCompletedEvent.model_validate_json(event.model_dump_json())

    if clone.timestamp != NOW:
        raise AssertionError(clone.timestamp)
    if clone.overall_score_standard != score():
        raise AssertionError(clone.overall_score_standard)
    if clone.by_category_standard["single_hop"] != score():
        raise AssertionError(clone.by_category_standard)


@pytest.mark.parametrize("event", suite_events())
def test_suite_discriminated_union_selects_concrete_class(event: BaseModel) -> None:
    parsed = TypeAdapter(SuiteLifecycleEvent).validate_python(event.model_dump())
    if type(parsed) is not type(event):
        raise AssertionError((type(parsed), type(event)))


@pytest.mark.parametrize("event", run_events())
def test_run_discriminated_union_selects_concrete_class(event: BaseModel) -> None:
    parsed = TypeAdapter(RunLifecycleEvent).validate_python(event.model_dump())
    if type(parsed) is not type(event):
        raise AssertionError((type(parsed), type(event)))


def test_invalid_suite_event_type_fails_union_validation() -> None:
    data = suite_started_event().model_dump()
    data["event_type"] = "suite_begun"

    with pytest.raises(ValidationError):
        TypeAdapter(SuiteLifecycleEvent).validate_python(data)


def test_invalid_run_event_type_fails_union_validation() -> None:
    data = run_started_event().model_dump()
    data["event_type"] = "run_begun"

    with pytest.raises(ValidationError):
        TypeAdapter(RunLifecycleEvent).validate_python(data)


@pytest.mark.parametrize(
    ("event", "field_name"),
    [
        (suite_started_event(), "sequence_number"),
        (suite_started_event(), "n_experiments_planned"),
        (experiment_started_event(), "n_runs_planned"),
        (experiment_completed_event(), "cost_usd"),
        (suite_completed_event(), "total_cost_usd"),
        (suite_completed_event(), "n_experiments_completed"),
        (suite_failed_event(), "n_experiments_completed_before_failure"),
        (run_started_event(), "sequence_number"),
        (conversation_processed_event(), "n_questions_evaluated"),
        (conversation_processed_event(), "cost_usd"),
        (run_completed_event(), "n_questions_attempted"),
        (run_completed_event(), "total_cost_usd"),
        (run_failed_event(), "n_questions_completed_before_failure"),
        (run_failed_event(), "partial_cost_usd"),
    ],
)
def test_negative_sequence_count_and_cost_fields_fail(event: BaseModel, field_name: str) -> None:
    data = event.model_dump()
    data[field_name] = -1

    with pytest.raises(ValidationError):
        type(event).model_validate(data)


def test_frozen_lifecycle_model_mutation_raises_validation_error() -> None:
    event = run_started_event()

    with pytest.raises((TypeError, ValidationError)):
        event.experiment_name = "Changed"
