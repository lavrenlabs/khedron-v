from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from khedron.config import ExperimentConfig

__all__ = [
    "APICallRecord",
    "APICallResult",
    "AggregateScore",
    "CategoryScore",
    "Conversation",
    "ConversationIngestionRecord",
    "ConversationProcessedEvent",
    "ErrorRecord",
    "ErrorResolutionRecord",
    "ExperimentCompletedEvent",
    "ExperimentFailedEvent",
    "ExperimentResult",
    "ExperimentStartedEvent",
    "FailurePattern",
    "ForcedRecoveryEvent",
    "Judgment",
    "JudgmentVerdict",
    "LifecycleEventBase",
    "Memory",
    "Question",
    "QuestionCategory",
    "QuestionEvaluationRecord",
    "QuestionPlanRecord",
    "QuestionRecord",
    "RecoveryAttemptRecord",
    "Response",
    "RetrievalRecord",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunFilters",
    "RunLifecycleEvent",
    "RunLifecycleEventBase",
    "RunResumedEvent",
    "RunScores",
    "RunStartedEvent",
    "RunStatus",
    "RunSummary",
    "ScoreWithCI",
    "Session",
    "SuiteCompletedEvent",
    "SuiteFailedEvent",
    "SuiteLifecycleEvent",
    "SuiteLifecycleEventBase",
    "SuiteStartedEvent",
    "SuiteStatus",
    "Turn",
    "question_plan_fingerprint",
]


def _empty_error_summary() -> list[dict[str, Any]]:
    return []


def _empty_category_score_map() -> dict[QuestionCategory, ScoreWithCI]:
    return {}


def question_plan_fingerprint(
    *,
    benchmark_id: str,
    categories: Sequence[str],
    corpus_checksum: str,
    question_ids: Sequence[str],
) -> str:
    """Return the interoperable fingerprint for a planned question set."""
    payload = json.dumps(
        {
            "benchmark_id": benchmark_id,
            "categories": sorted(categories),
            "corpus_checksum": corpus_checksum.lower(),
            "question_ids": sorted(question_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class QuestionCategory(StrEnum):
    """LoCoMo question categories."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    OPEN_DOMAIN = "open_domain"
    ADVERSARIAL = "adversarial"


class JudgmentVerdict(StrEnum):
    """Outcome of judging a generated answer."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    ERROR = "error"
    UNKNOWN = "unknown"


class Turn(BaseModel):
    """A single turn within a benchmark conversation session."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    speaker: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A single session within a benchmark conversation."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    session_number: int = Field(ge=0)
    timestamp: datetime
    turns: list[Turn]


class Conversation(BaseModel):
    """A single benchmark conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    speakers: list[str] = Field(min_length=2, max_length=2)
    sessions: list[Session]


class Question(BaseModel):
    """A single benchmark question to be answered."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    conversation_id: str
    category: QuestionCategory
    question_text: str
    expected_answer: str
    evidence_dialog_ids: list[str] = Field(default_factory=list)
    is_audited_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """A memory record returned by a memory provider."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = Field(default=None, ge=0.0)
    timestamp: datetime | None = None


class QuestionEvaluationRecord(BaseModel):
    """Canonical question metadata for a single run/question attempt."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["question_evaluation"] = "question_evaluation"
    question_evaluation_id: str
    run_id: str
    question_id: str
    conversation_id: str
    category: QuestionCategory
    question_text: str
    expected_answer: str
    is_audited_error: bool
    timestamp: datetime


class QuestionPlanRecord(BaseModel):
    """Immutable manifest proving the question set a run was planned to evaluate."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    record_type: Literal["question_plan"] = "question_plan"
    run_id: str
    benchmark_id: str
    categories: tuple[str, ...]
    corpus_checksum: str
    question_ids: tuple[str, ...]
    fingerprint: str
    timestamp: datetime

    @model_validator(mode="after")
    def _validate_fingerprint(self) -> QuestionPlanRecord:
        expected = question_plan_fingerprint(
            benchmark_id=self.benchmark_id,
            categories=self.categories,
            corpus_checksum=self.corpus_checksum,
            question_ids=self.question_ids,
        )
        if self.fingerprint != expected:
            raise ValueError("question-plan fingerprint does not match its manifest fields")
        return self


class ConversationIngestionRecord(BaseModel):
    """Records ingestion of one benchmark conversation into a provider."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["conversation_ingestion"] = "conversation_ingestion"
    run_id: str
    conversation_id: str
    started_at: datetime
    finished_at: datetime
    n_sessions: int = Field(ge=0)
    n_turns: int = Field(ge=0)
    n_turns_succeeded: int = Field(ge=0)
    n_turns_failed: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0.0)
    avg_latency_per_turn_ms: float = Field(ge=0.0)
    error_summary: list[dict[str, Any]] = Field(default_factory=_empty_error_summary)
    ingestion_attempt_id: str | None = None


class RetrievalRecord(BaseModel):
    """Memories retrieved for a question before answer generation."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["retrieval"] = "retrieval"
    retrieval_id: str
    question_evaluation_id: str
    run_id: str
    question_id: str
    timestamp: datetime
    query: str
    top_k: int = Field(ge=0)
    n_returned: int = Field(ge=0)
    memories: list[Memory]
    retrieval_latency_ms: float = Field(ge=0.0)
    recovery_attempt_id: str | None = None
    ingestion_attempt_id: str | None = None


class Response(BaseModel):
    """A single answer generated by an answer model."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["response"] = "response"
    response_id: str
    run_id: str
    question_id: str
    retrieval_id: str
    timestamp: datetime
    model_id: str
    prompt: str
    answer_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    raw_api_response: dict[str, Any] = Field(default_factory=dict)
    recovery_attempt_id: str | None = None


class Judgment(BaseModel):
    """A single judgment of a response."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["judgment"] = "judgment"
    judgment_id: str
    run_id: str
    response_id: str
    question_id: str
    timestamp: datetime
    judge_model_id: str
    prompt: str
    raw_judge_output: str
    parsed_verdict: JudgmentVerdict
    parsed_score: float = Field(ge=0.0, le=1.0)
    parsed_reasoning: str
    parse_was_successful: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    recovery_attempt_id: str | None = None


class APICallResult(BaseModel):
    """Result of an external LLM API call."""

    model_config = ConfigDict(frozen=True)

    output: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    #: Prompt tokens served from the provider's cache on a cache hit (subset of input_tokens).
    #: Observability only, defaulting to 0 for vendors/responses that do not report it; it is
    #: deliberately not used to discount cost (the pricing schema excludes cached pricing, so the
    #: recorded cost stays full-price and the budget cap conservative).
    cached_input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    model_id: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


class APICallRecord(BaseModel):
    """Audit record for one external API call."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    record_type: Literal["api_call"] = "api_call"
    api_call_id: str
    run_id: str
    question_id: str | None
    timestamp: datetime
    phase: Literal["ingest", "retrieve", "generate", "judge"]
    vendor: str
    model_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    #: Cached prompt tokens (subset of input_tokens); 0 when the vendor does not report a cache hit.
    cached_input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    status: Literal["success", "retry", "failure"]
    attempt_number: int = Field(ge=1)
    error_message: str | None = None


class ScoreWithCI(BaseModel):
    """Binary proportion score with Wilson 95% confidence interval."""

    model_config = ConfigDict(frozen=True)

    n_total: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    n_errors: int = Field(ge=0)
    n_partial: int = Field(default=0, ge=0)
    n_unknown: int = Field(default=0, ge=0)
    point_estimate: float = Field(ge=0.0, le=1.0)
    ci_95_low: float = Field(ge=0.0, le=1.0)
    ci_95_high: float = Field(ge=0.0, le=1.0)


class LifecycleEventBase(BaseModel):
    """Base for all lifecycle events."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    event_id: str
    timestamp: datetime


class SuiteLifecycleEventBase(LifecycleEventBase):
    """Base for suite lifecycle events."""

    model_config = ConfigDict(frozen=True)

    suite_id: str
    sequence_number: int = Field(ge=0)


class SuiteStartedEvent(SuiteLifecycleEventBase):
    """Marks the start of an experiment suite."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["suite_started"] = "suite_started"
    config_yaml_path: str
    config_yaml_content: str
    methodology_version: str
    methodology_profile: str
    framework_version: str
    n_experiments_planned: int = Field(ge=0)
    runtime_environment: dict[str, Any]


class ExperimentStartedEvent(SuiteLifecycleEventBase):
    """Marks the start of one experiment within a suite."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["experiment_started"] = "experiment_started"
    experiment_id: str
    experiment_name: str
    n_runs_planned: int = Field(ge=0)


class ExperimentCompletedEvent(SuiteLifecycleEventBase):
    """Marks successful or partial completion of one experiment aggregate."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["experiment_completed"] = "experiment_completed"
    experiment_id: str
    overall_standard_pooled_score: ScoreWithCI | None
    overall_standard_mean: float | None = Field(ge=0.0, le=1.0)
    overall_standard_stddev: float | None = Field(ge=0.0)
    overall_audited_pooled_score: ScoreWithCI | None
    overall_audited_mean: float | None = Field(ge=0.0, le=1.0)
    overall_audited_stddev: float | None = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)


class ExperimentFailedEvent(SuiteLifecycleEventBase):
    """Marks an experiment-level failure."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["experiment_failed"] = "experiment_failed"
    experiment_id: str
    error_message: str
    error_phase: Literal["initialization", "ingest", "retrieve", "generate", "judge", "analysis"]


class SuiteCompletedEvent(SuiteLifecycleEventBase):
    """Terminal event for a completed or partially completed suite."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["suite_completed"] = "suite_completed"
    total_cost_usd: float = Field(ge=0.0)
    n_experiments_completed: int = Field(ge=0)
    n_experiments_failed: int = Field(ge=0)


class SuiteFailedEvent(SuiteLifecycleEventBase):
    """Terminal event for a failed suite."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["suite_failed"] = "suite_failed"
    error_message: str
    n_experiments_completed_before_failure: int = Field(ge=0)


SuiteLifecycleEvent: TypeAlias = Annotated[
    Union[  # noqa: UP007 - handoff requires Annotated[Union[...], Field(...)].
        SuiteStartedEvent,
        ExperimentStartedEvent,
        ExperimentCompletedEvent,
        ExperimentFailedEvent,
        SuiteCompletedEvent,
        SuiteFailedEvent,
    ],
    Field(discriminator="event_type"),
]


class RunLifecycleEventBase(LifecycleEventBase):
    """Base for run lifecycle events."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    sequence_number: int = Field(ge=0)


class RunStartedEvent(RunLifecycleEventBase):
    """Marks the start of a single run."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["run_started"] = "run_started"
    suite_id: str
    experiment_id: str
    experiment_name: str
    run_number: int = Field(ge=0)
    provider_type: str
    provider_version: str
    benchmark_type: str
    benchmark_version: str
    benchmark_checksum: str
    answer_model_id: str
    answer_model_vendor: str
    judge_model_id: str
    judge_model_vendor: str
    config: dict[str, Any]
    methodology_version: str
    methodology_profile: str
    framework_version: str
    seed: int
    runtime_environment: dict[str, Any]


class ConversationProcessedEvent(RunLifecycleEventBase):
    """One emitted per conversation processed in the run."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["conversation_processed"] = "conversation_processed"
    conversation_id: str
    n_questions_evaluated: int = Field(ge=0)
    n_questions_correct: int = Field(ge=0)
    n_questions_errored: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class RunCompletedEvent(RunLifecycleEventBase):
    """Terminal event for a successful or partial run."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["run_completed"] = "run_completed"
    status: Literal["completed", "partial"]
    n_questions_attempted: int = Field(ge=0)
    n_questions_succeeded: int = Field(ge=0)
    n_questions_errored: int = Field(ge=0)
    overall_score_standard: ScoreWithCI | None
    overall_score_audited: ScoreWithCI | None
    by_category_standard: dict[str, ScoreWithCI]
    by_category_audited: dict[str, ScoreWithCI]
    total_cost_usd: float = Field(ge=0.0)


class RunFailedEvent(RunLifecycleEventBase):
    """Terminal event when a run cannot complete due to an unrecoverable error."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["run_failed"] = "run_failed"
    error_message: str
    error_phase: Literal["initialization", "ingest", "retrieve", "generate", "judge", "analysis"]
    n_questions_completed_before_failure: int = Field(ge=0)
    partial_cost_usd: float = Field(ge=0.0)


class RunResumedEvent(RunLifecycleEventBase):
    """Marks a run continuing after an interruption, and states what it inherited.

    A separate event rather than a second `run_started`, because the two say different things: a
    run that was resumed is not a run that started twice, and an artifact must be able to tell a
    reader which it was. It also supersedes the previous terminal event without erasing it -- the
    earlier failure remains in the stream, which is the record of what actually happened.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["run_resumed"] = "run_resumed"
    # What the resumed run did not repeat. Recorded so the artifact can be read without diffing
    # two halves of a lifecycle stream to work out which questions this process actually asked.
    n_conversations_inherited: int = Field(ge=0)
    n_questions_inherited: int = Field(ge=0)
    inherited_cost_usd: float = Field(ge=0.0)
    framework_version: str


RunLifecycleEvent: TypeAlias = Annotated[
    Union[  # noqa: UP007 - handoff requires Annotated[Union[...], Field(...)].
        RunStartedEvent,
        RunResumedEvent,
        ConversationProcessedEvent,
        RunCompletedEvent,
        RunFailedEvent,
    ],
    Field(discriminator="event_type"),
]


class CategoryScore(BaseModel):
    """Persisted score for one run, category, and scoring mode."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_id: str
    category: QuestionCategory
    mode: Literal["standard", "audited"]
    score: ScoreWithCI


class RunScores(BaseModel):
    """In-memory score bundle for one run."""

    model_config = ConfigDict(frozen=True)

    overall_standard: ScoreWithCI | None = None
    overall_audited: ScoreWithCI | None = None
    by_category_standard: dict[QuestionCategory, ScoreWithCI] = Field(
        default_factory=_empty_category_score_map
    )
    by_category_audited: dict[QuestionCategory, ScoreWithCI] = Field(
        default_factory=_empty_category_score_map
    )


class SuiteStatus(BaseModel):
    """Current state of a suite reduced from lifecycle events."""

    model_config = ConfigDict(frozen=True)

    suite_id: str
    status: Literal["running", "completed", "failed", "partial"]
    started_at: datetime
    finished_at: datetime | None
    config_yaml_content: str
    methodology_version: str
    methodology_profile: str
    framework_version: str
    n_experiments_planned: int = Field(ge=0)
    n_experiments_completed: int = Field(ge=0)
    n_experiments_failed: int = Field(ge=0)
    n_experiments_in_progress: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    last_event_at: datetime
    last_event_type: str


class RunStatus(BaseModel):
    """Current state of a run reduced from lifecycle events."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    suite_id: str
    experiment_id: str
    experiment_name: str
    run_number: int = Field(ge=0)
    status: Literal["running", "completed", "failed", "partial"]
    started_at: datetime
    finished_at: datetime | None
    provider_type: str
    provider_version: str
    answer_model_id: str
    judge_model_id: str
    methodology_version: str
    methodology_profile: str
    framework_version: str
    n_conversations_processed: int = Field(ge=0)
    n_questions_attempted: int = Field(ge=0)
    n_questions_succeeded: int = Field(ge=0)
    n_questions_errored: int = Field(ge=0)
    overall_score_standard: ScoreWithCI | None
    overall_score_audited: ScoreWithCI | None
    by_category_standard: dict[str, ScoreWithCI] = Field(default_factory=dict)
    by_category_audited: dict[str, ScoreWithCI] = Field(default_factory=dict)
    total_cost_usd: float = Field(ge=0.0)
    error_message: str | None
    error_phase: (
        Literal[
            "initialization",
            "ingest",
            "retrieve",
            "generate",
            "judge",
            "analysis",
        ]
        | None
    )


class RunFilters(BaseModel):
    """Supported filters for repository run listing."""

    model_config = ConfigDict(frozen=True)

    suite_id: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    status: Literal["running", "completed", "failed", "partial"] | None = None
    provider_type: str | None = None
    benchmark_type: str | None = None
    answer_model_id: str | None = None
    judge_model_id: str | None = None


class RunSummary(BaseModel):
    """Dashboard-friendly row from the SQLite runs projection."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    suite_id: str
    experiment_id: str
    experiment_name: str
    run_number: int = Field(ge=0)
    status: Literal["running", "completed", "failed", "partial"]
    started_at: datetime
    finished_at: datetime | None
    provider_type: str
    provider_version: str | None
    benchmark_type: str
    answer_model_id: str
    judge_model_id: str
    n_questions_attempted: int = Field(ge=0)
    n_questions_succeeded: int = Field(ge=0)
    n_questions_errored: int = Field(ge=0)
    overall_score_standard: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_score_audited: float | None = Field(default=None, ge=0.0, le=1.0)
    total_cost_usd: float = Field(ge=0.0)
    methodology_version: str
    methodology_profile: str
    framework_version: str
    error_message: str | None
    error_phase: (
        Literal[
            "initialization",
            "ingest",
            "retrieve",
            "generate",
            "judge",
            "analysis",
        ]
        | None
    )


class AggregateScore(BaseModel):
    """Aggregate score across multiple runs of the same experiment."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    experiment_id: str
    category: QuestionCategory | Literal["overall"]
    mode: Literal["standard", "audited"]
    n_runs: int = Field(ge=1)
    pooled_score: ScoreWithCI
    individual_run_scores: list[float]
    mean: float = Field(ge=0.0, le=1.0)
    stddev: float = Field(ge=0.0)
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)


class ExperimentResult(BaseModel):
    """Aggregate result of multiple runs for one experiment."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    experiment_id: str
    suite_id: str
    experiment_name: str
    n_runs: int = Field(ge=1)
    run_ids: list[str]
    aggregate_overall_standard: AggregateScore | None
    aggregate_overall_audited: AggregateScore | None
    aggregate_by_category_standard: dict[str, AggregateScore]
    aggregate_by_category_audited: dict[str, AggregateScore]
    total_cost_usd: float = Field(ge=0.0)
    config: ExperimentConfig


class FailurePattern(BaseModel):
    """A detected pattern in failed questions."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    pattern_id: str
    run_id: str
    pattern_name: str
    description: str
    suggested_remedy: str
    n_affected_questions: int = Field(ge=0)
    affected_question_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ErrorRecord(BaseModel):
    """A recoverable error encountered during a run."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    error_id: str
    run_id: str
    timestamp: datetime
    phase: Literal["initialization", "ingest", "retrieve", "generate", "judge", "analysis"]
    question_id: str | None
    error_type: str
    error_message: str
    stack_trace: str | None
    context: dict[str, Any]
    recovered: bool
    retryable: bool | None = None


class RecoveryAttemptRecord(BaseModel):
    """A recovery dispatch budget unit, persisted before the retried stage executes."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    record_type: Literal["recovery_attempt"] = "recovery_attempt"
    attempt_id: str
    run_id: str
    question_id: str
    error_id: str
    stage: Literal["retrieve", "generate", "judge"]
    timestamp: datetime


class ErrorResolutionRecord(BaseModel):
    """Append-only causal evidence that a question-stage error was repaired."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    record_type: Literal["error_resolution"] = "error_resolution"
    resolution_id: str
    run_id: str
    question_id: str
    error_id: str
    recovery_attempt_id: str
    resolved_by_stage: Literal["retrieve", "generate", "judge"]
    resolved_by_stage_record_id: str
    timestamp: datetime


class ForcedRecoveryEvent(BaseModel):
    """Operator authorization to recover a blocked, non-retryable question-stage error.

    An append-only causal record -- a sibling of ``RecoveryAttemptRecord`` /
    ``ErrorResolutionRecord``, never a member of the ``RunLifecycleEvent`` union. It records the
    one fact none of the recovery engine's existing records disclose: that a human deliberately
    dispatched an error the automatic classifier marked ``retryable=False``, which the automatic
    filter would otherwise never turn into work. Its presence is what makes an artifact unable to
    silently hide an operator-forced recovery, and its absence beside a non-retryable recovery
    attempt is what makes the audit trail forgery-resistant (the strict reader's inverse
    invariant).
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    record_type: Literal["forced_recovery"] = "forced_recovery"
    forced_recovery_id: str
    run_id: str
    question_id: str
    # The non-retryable error being overridden, and the RecoveryAttemptRecord this authorizes.
    error_id: str
    attempt_id: str
    stage: Literal["retrieve", "generate", "judge"]
    # Operator justification, recorded verbatim. Non-secret free text: the standard JsonlWriter
    # credential scrub still runs over it as a backstop, but that is not a licence to place a
    # credential here.
    reason: str = Field(min_length=1)
    timestamp: datetime


class QuestionRecord(BaseModel):
    """Composite question evaluation view assembled from persisted records."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    question_evaluation_id: str
    run_id: str
    question_id: str
    conversation_id: str
    category: QuestionCategory
    question_text: str
    expected_answer: str
    is_audited_error: bool
    retrieval_id: str | None
    retrieval_timestamp: datetime | None
    retrieval_latency_ms: float | None = Field(default=None, ge=0.0)
    n_memories_retrieved: int | None = Field(default=None, ge=0)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    response_id: str | None
    generation_timestamp: datetime | None
    generation_latency_ms: float | None = Field(default=None, ge=0.0)
    generated_answer: str | None
    generation_input_tokens: int | None = Field(default=None, ge=0)
    generation_output_tokens: int | None = Field(default=None, ge=0)
    generation_cost_usd: float | None = Field(default=None, ge=0.0)
    judgment_id: str | None
    judgment_timestamp: datetime | None
    judgment_latency_ms: float | None = Field(default=None, ge=0.0)
    verdict: JudgmentVerdict
    score: float = Field(ge=0.0, le=1.0)
    judgment_reasoning: str | None
    judgment_input_tokens: int | None = Field(default=None, ge=0)
    judgment_output_tokens: int | None = Field(default=None, ge=0)
    judgment_cost_usd: float | None = Field(default=None, ge=0.0)
    total_latency_ms: float | None = Field(default=None, ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    error_message: str | None
    error_phase: Literal["retrieve", "generate", "judge"] | None
