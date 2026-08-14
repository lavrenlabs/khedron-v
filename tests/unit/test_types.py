from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from khedron.config import ExperimentConfig
from khedron.types import (
    AggregateScore,
    APICallRecord,
    APICallResult,
    CategoryScore,
    Conversation,
    ConversationIngestionRecord,
    ErrorRecord,
    ErrorResolutionRecord,
    ExperimentResult,
    FailurePattern,
    Judgment,
    JudgmentVerdict,
    Memory,
    Question,
    QuestionCategory,
    QuestionEvaluationRecord,
    QuestionPlanRecord,
    QuestionRecord,
    RecoveryAttemptRecord,
    Response,
    RetrievalRecord,
    RunFilters,
    RunScores,
    RunStatus,
    RunSummary,
    ScoreWithCI,
    Session,
    SuiteStatus,
    Turn,
    question_plan_fingerprint,
)

NOW = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)


def score() -> ScoreWithCI:
    return ScoreWithCI(
        n_total=100,
        n_correct=82,
        n_errors=18,
        n_partial=1,
        n_unknown=2,
        point_estimate=0.82,
        ci_95_low=0.7333,
        ci_95_high=0.8830,
    )


def memory() -> Memory:
    return Memory(
        memory_id="mem-1",
        content="Alice moved to Rome.",
        metadata={"speaker": "Alice"},
        score=0.91,
        timestamp=NOW,
    )


def experiment_config() -> dict[str, Any]:
    return {
        "name": "Synthetic",
        "provider": {"type": "full_context"},
        "benchmark": {"type": "locomo"},
        "answer_model": {"type": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "judge": {"type": "anthropic", "model": "claude-sonnet-4-5"},
    }


def model_instances() -> list[BaseModel]:
    score_value = score()
    aggregate = AggregateScore(
        experiment_id="exp-1",
        category="overall",
        mode="audited",
        n_runs=3,
        pooled_score=score_value,
        individual_run_scores=[0.81, 0.82, 0.83],
        mean=0.82,
        stddev=0.01,
        min=0.81,
        max=0.83,
    )
    turn = Turn(turn_id="turn-1", speaker="Alice", content="I moved to Rome.")
    session = Session(
        session_id="session-1",
        session_number=0,
        timestamp=NOW,
        turns=[turn],
    )
    return [
        turn,
        session,
        Conversation(conversation_id="conv-1", speakers=["Alice", "Bob"], sessions=[session]),
        Question(
            question_id="q-1",
            conversation_id="conv-1",
            category=QuestionCategory.SINGLE_HOP,
            question_text="Where did Alice move?",
            expected_answer="Rome",
            evidence_dialog_ids=["turn-1"],
            metadata={"source": "synthetic"},
        ),
        memory(),
        QuestionEvaluationRecord(
            question_evaluation_id="qe-1",
            run_id="run-1",
            question_id="q-1",
            conversation_id="conv-1",
            category=QuestionCategory.SINGLE_HOP,
            question_text="Where did Alice move?",
            expected_answer="Rome",
            is_audited_error=False,
            timestamp=NOW,
        ),
        ConversationIngestionRecord(
            run_id="run-1",
            conversation_id="conv-1",
            started_at=NOW,
            finished_at=NOW,
            n_sessions=1,
            n_turns=1,
            n_turns_succeeded=1,
            n_turns_failed=0,
            total_latency_ms=10.0,
            avg_latency_per_turn_ms=10.0,
        ),
        RetrievalRecord(
            retrieval_id="ret-1",
            question_evaluation_id="qe-1",
            run_id="run-1",
            question_id="q-1",
            timestamp=NOW,
            query="Where did Alice move?",
            top_k=10,
            n_returned=1,
            memories=[memory()],
            retrieval_latency_ms=12.5,
        ),
        Response(
            response_id="resp-1",
            run_id="run-1",
            question_id="q-1",
            retrieval_id="ret-1",
            timestamp=NOW,
            model_id="gpt-4o-mini-2024-07-18",
            prompt="Memories: Alice moved to Rome.",
            answer_text="Rome",
            input_tokens=32,
            output_tokens=3,
            latency_ms=200.0,
            cost_usd=0.01,
            raw_api_response={"id": "api-response-1"},
        ),
        Judgment(
            judgment_id="judgment-1",
            run_id="run-1",
            response_id="resp-1",
            question_id="q-1",
            timestamp=NOW,
            judge_model_id="gpt-4o-2024-08-06",
            prompt="Judge this answer.",
            raw_judge_output='{"verdict": "correct"}',
            parsed_verdict=JudgmentVerdict.CORRECT,
            parsed_score=1.0,
            parsed_reasoning="The answer matches.",
            parse_was_successful=True,
            input_tokens=50,
            output_tokens=10,
            latency_ms=180.0,
            cost_usd=0.02,
        ),
        APICallResult(
            output="Rome",
            input_tokens=32,
            output_tokens=3,
            latency_ms=200.0,
            cost_usd=0.01,
            model_id="gpt-4o-mini-2024-07-18",
            raw_response={"id": "api-response-1"},
        ),
        APICallRecord(
            api_call_id="api-1",
            run_id="run-1",
            question_id="q-1",
            timestamp=NOW,
            phase="generate",
            vendor="openai",
            model_id="gpt-4o-mini-2024-07-18",
            input_tokens=32,
            output_tokens=3,
            latency_ms=200.0,
            cost_usd=0.01,
            status="success",
            attempt_number=1,
        ),
        score_value,
        CategoryScore(
            run_id="run-1",
            category=QuestionCategory.SINGLE_HOP,
            mode="audited",
            score=score_value,
        ),
        RunScores(
            overall_standard=score_value,
            overall_audited=score_value,
            by_category_standard={QuestionCategory.SINGLE_HOP: score_value},
            by_category_audited={QuestionCategory.SINGLE_HOP: score_value},
        ),
        SuiteStatus(
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
            total_cost_usd=0.03,
            last_event_at=NOW,
            last_event_type="suite_completed",
        ),
        RunStatus(
            run_id="run-1",
            suite_id="suite-1",
            experiment_id="exp-1",
            experiment_name="Synthetic",
            run_number=0,
            status="completed",
            started_at=NOW,
            finished_at=NOW,
            provider_type="full_context",
            provider_version="0.1.0",
            answer_model_id="gpt-4o-mini-2024-07-18",
            judge_model_id="gpt-4o-2024-08-06",
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            framework_version="0.1.0",
            n_conversations_processed=1,
            n_questions_attempted=1,
            n_questions_succeeded=1,
            n_questions_errored=0,
            overall_score_standard=score_value,
            overall_score_audited=score_value,
            by_category_standard={"single_hop": score_value},
            by_category_audited={"single_hop": score_value},
            total_cost_usd=0.03,
            error_message=None,
            error_phase=None,
        ),
        RunFilters(
            experiment_name="Synthetic",
            status="completed",
            provider_type="full_context",
        ),
        RunSummary(
            run_id="run-1",
            suite_id="suite-1",
            experiment_id="exp-1",
            experiment_name="Synthetic",
            run_number=0,
            status="completed",
            started_at=NOW,
            finished_at=NOW,
            provider_type="full_context",
            provider_version="0.1.0",
            benchmark_type="locomo",
            answer_model_id="gpt-4o-mini-2024-07-18",
            judge_model_id="gpt-4o-2024-08-06",
            n_questions_attempted=1,
            n_questions_succeeded=1,
            n_questions_errored=0,
            overall_score_standard=0.82,
            overall_score_audited=0.82,
            total_cost_usd=0.03,
            methodology_version="1.0",
            methodology_profile="canonical-v1",
            framework_version="0.1.0",
            error_message=None,
            error_phase=None,
        ),
        aggregate,
        ExperimentResult(
            experiment_id="exp-1",
            suite_id="suite-1",
            experiment_name="Synthetic",
            n_runs=3,
            run_ids=["run-1", "run-2", "run-3"],
            aggregate_overall_standard=aggregate,
            aggregate_overall_audited=aggregate,
            aggregate_by_category_standard={"single_hop": aggregate},
            aggregate_by_category_audited={"single_hop": aggregate},
            total_cost_usd=1.23,
            config=experiment_config(),
        ),
        FailurePattern(
            pattern_id="pattern-1",
            run_id="run-1",
            pattern_name="missing_memory_failure",
            description="No relevant memory was retrieved.",
            suggested_remedy="Improve retrieval recall.",
            n_affected_questions=1,
            affected_question_ids=["q-1"],
            confidence=0.9,
        ),
        ErrorRecord(
            error_id="error-1",
            run_id="run-1",
            timestamp=NOW,
            phase="generate",
            question_id="q-1",
            error_type="ModelTimeoutError",
            error_message="Timed out.",
            stack_trace=None,
            context={"attempt": 1},
            recovered=True,
        ),
        QuestionRecord(
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
            generated_answer="Rome",
            generation_input_tokens=32,
            generation_output_tokens=3,
            generation_cost_usd=0.01,
            judgment_id="judgment-1",
            judgment_timestamp=NOW,
            judgment_latency_ms=180.0,
            verdict=JudgmentVerdict.CORRECT,
            score=1.0,
            judgment_reasoning="The answer matches.",
            judgment_input_tokens=50,
            judgment_output_tokens=10,
            judgment_cost_usd=0.02,
            total_latency_ms=392.5,
            total_cost_usd=0.03,
            error_message=None,
            error_phase=None,
        ),
    ]


def test_question_category_values() -> None:
    if QuestionCategory.SINGLE_HOP.value != "single_hop":
        raise AssertionError(QuestionCategory.SINGLE_HOP)
    if QuestionCategory.MULTI_HOP.value != "multi_hop":
        raise AssertionError(QuestionCategory.MULTI_HOP)
    if QuestionCategory.TEMPORAL.value != "temporal":
        raise AssertionError(QuestionCategory.TEMPORAL)
    if QuestionCategory.OPEN_DOMAIN.value != "open_domain":
        raise AssertionError(QuestionCategory.OPEN_DOMAIN)
    if QuestionCategory.ADVERSARIAL.value != "adversarial":
        raise AssertionError(QuestionCategory.ADVERSARIAL)


def test_judgment_verdict_values() -> None:
    if JudgmentVerdict.CORRECT.value != "correct":
        raise AssertionError(JudgmentVerdict.CORRECT)
    if JudgmentVerdict.INCORRECT.value != "incorrect":
        raise AssertionError(JudgmentVerdict.INCORRECT)
    if JudgmentVerdict.PARTIAL.value != "partial":
        raise AssertionError(JudgmentVerdict.PARTIAL)
    if JudgmentVerdict.ERROR.value != "error":
        raise AssertionError(JudgmentVerdict.ERROR)
    if JudgmentVerdict.UNKNOWN.value != "unknown":
        raise AssertionError(JudgmentVerdict.UNKNOWN)


@pytest.mark.parametrize("instance", model_instances())
def test_models_round_trip_through_model_dump(instance: BaseModel) -> None:
    clone = type(instance).model_validate(instance.model_dump())
    if clone != instance:
        raise AssertionError((clone, instance))


@pytest.mark.parametrize("instance", model_instances())
def test_models_round_trip_through_json(instance: BaseModel) -> None:
    clone = type(instance).model_validate_json(instance.model_dump_json())
    if clone != instance:
        raise AssertionError((clone, instance))


def test_invalid_enum_value_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Question(
            question_id="q-1",
            conversation_id="conv-1",
            category="not-a-category",
            question_text="Question?",
            expected_answer="Answer",
        )


@pytest.mark.parametrize(
    ("model_class", "data"),
    [
        (
            Conversation,
            {
                "conversation_id": "conv-1",
                "speakers": ["Alice"],
                "sessions": [],
            },
        ),
        (Session, {"session_id": "s-1", "session_number": -1, "timestamp": NOW, "turns": []}),
        (
            ScoreWithCI,
            {
                "n_total": 1,
                "n_correct": 1,
                "n_errors": 0,
                "point_estimate": 1.2,
                "ci_95_low": 0.0,
                "ci_95_high": 1.0,
            },
        ),
        (
            Judgment,
            {
                "judgment_id": "j-1",
                "run_id": "run-1",
                "response_id": "resp-1",
                "question_id": "q-1",
                "timestamp": NOW,
                "judge_model_id": "judge",
                "prompt": "prompt",
                "raw_judge_output": "{}",
                "parsed_verdict": JudgmentVerdict.CORRECT,
                "parsed_score": 2.0,
                "parsed_reasoning": "reason",
                "parse_was_successful": True,
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 1.0,
                "cost_usd": 0.0,
            },
        ),
        (
            FailurePattern,
            {
                "pattern_id": "p-1",
                "run_id": "run-1",
                "pattern_name": "pattern",
                "description": "description",
                "suggested_remedy": "remedy",
                "n_affected_questions": 1,
                "affected_question_ids": ["q-1"],
                "confidence": 1.1,
            },
        ),
        (
            APICallRecord,
            {
                "api_call_id": "api-1",
                "run_id": "run-1",
                "question_id": None,
                "timestamp": NOW,
                "phase": "generate",
                "vendor": "openai",
                "model_id": "model",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 1.0,
                "cost_usd": 0.0,
                "status": "success",
                "attempt_number": 0,
            },
        ),
    ],
)
def test_validation_failures(model_class: type[BaseModel], data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model_class.model_validate(data)


def test_frozen_model_mutation_raises_validation_error() -> None:
    question = Question(
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Question?",
        expected_answer="Answer",
    )

    with pytest.raises(ValidationError):
        question.question_text = "Changed"


def test_mutable_defaults_are_isolated_between_instances() -> None:
    first_question = Question(
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Question?",
        expected_answer="Answer",
    )
    second_question = Question(
        question_id="q-2",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Question?",
        expected_answer="Answer",
    )
    first_question.evidence_dialog_ids.append("turn-1")
    first_question.metadata["source"] = "synthetic"

    if second_question.evidence_dialog_ids:
        raise AssertionError(second_question.evidence_dialog_ids)
    if second_question.metadata:
        raise AssertionError(second_question.metadata)

    first_scores = RunScores()
    second_scores = RunScores()
    first_scores.by_category_standard[QuestionCategory.SINGLE_HOP] = score()

    if second_scores.by_category_standard:
        raise AssertionError(second_scores.by_category_standard)


def test_experiment_result_config_is_concrete_experiment_config() -> None:
    aggregate = AggregateScore(
        experiment_id="exp-1",
        category="overall",
        mode="audited",
        n_runs=1,
        pooled_score=score(),
        individual_run_scores=[0.82],
        mean=0.82,
        stddev=0.0,
        min=0.82,
        max=0.82,
    )
    result = ExperimentResult(
        experiment_id="exp-1",
        suite_id="suite-1",
        experiment_name="Synthetic",
        n_runs=1,
        run_ids=["run-1"],
        aggregate_overall_standard=None,
        aggregate_overall_audited=aggregate,
        aggregate_by_category_standard={},
        aggregate_by_category_audited={},
        total_cost_usd=0.0,
        config=experiment_config(),
    )

    if not isinstance(result.config, ExperimentConfig):
        raise AssertionError(result.config)
    if result.config.provider.type != "full_context":
        raise AssertionError(result.config)


def test_question_plan_fingerprint_has_stable_adr_0005_wire_digest() -> None:
    fingerprint = question_plan_fingerprint(
        benchmark_id="bench-β",
        categories=["temporal", "single_hop"],
        corpus_checksum="SHA256:ABCD",
        question_ids=["\N{GREEK SMALL LETTER ALPHA}-q", "q-2"],
    )
    expected = "sha256:8f2db2c91112b1ff97062617c0dee7da2115a5bd79b717a2896effe1c9a785c3"
    if fingerprint != expected:
        raise AssertionError(fingerprint)

    plan = QuestionPlanRecord(
        run_id="run-1",
        benchmark_id="bench-β",
        categories=["temporal", "single_hop"],
        corpus_checksum="SHA256:ABCD",
        question_ids=["\N{GREEK SMALL LETTER ALPHA}-q", "q-2"],
        fingerprint=expected,
        timestamp=NOW,
    )
    if plan.fingerprint != expected:
        raise AssertionError(plan)
    if not isinstance(plan.categories, tuple) or not isinstance(plan.question_ids, tuple):
        raise AssertionError("manifest collections must be deeply immutable")
    with pytest.raises(AttributeError):
        plan.categories.append("open_domain")
    with pytest.raises(ValidationError):
        plan.question_ids += ("q-3",)
    if plan.fingerprint != expected:
        raise AssertionError("manifest mutation changed its verified fingerprint")
    with pytest.raises(ValidationError):
        QuestionPlanRecord.model_validate({**plan.model_dump(), "fingerprint": "sha256:wrong"})


def test_recovery_records_constrain_stage_and_legacy_fields_decode_as_unknown() -> None:
    attempt = RecoveryAttemptRecord(
        attempt_id="attempt-1",
        run_id="run-1",
        question_id="q-1",
        error_id="error-1",
        stage="generate",
        timestamp=NOW,
    )
    resolution = ErrorResolutionRecord(
        resolution_id="resolution-1",
        run_id="run-1",
        question_id="q-1",
        error_id="error-1",
        recovery_attempt_id=attempt.attempt_id,
        resolved_by_stage="generate",
        resolved_by_stage_record_id="response-1",
        timestamp=NOW,
    )
    if resolution.resolved_by_stage != "generate":
        raise AssertionError(resolution)
    with pytest.raises(ValidationError):
        RecoveryAttemptRecord.model_validate({**attempt.model_dump(), "stage": "invalid"})

    for record in model_instances():
        if isinstance(
            record,
            (ConversationIngestionRecord, RetrievalRecord, Response, Judgment, ErrorRecord),
        ):
            payload = record.model_dump()
            for field_name in ("ingestion_attempt_id", "recovery_attempt_id", "retryable"):
                payload.pop(field_name, None)
            decoded = type(record).model_validate(payload)
            if isinstance(decoded, ErrorRecord) and decoded.retryable is not None:
                raise AssertionError(decoded.retryable)
            if (
                isinstance(decoded, ConversationIngestionRecord)
                and decoded.ingestion_attempt_id is not None
            ):
                raise AssertionError(decoded.ingestion_attempt_id)
            if isinstance(decoded, RetrievalRecord) and (
                decoded.ingestion_attempt_id is not None or decoded.recovery_attempt_id is not None
            ):
                raise AssertionError(decoded)
            if (
                isinstance(decoded, (Response, Judgment))
                and decoded.recovery_attempt_id is not None
            ):
                raise AssertionError(decoded.recovery_attempt_id)
