from __future__ import annotations

# ruff: noqa: S101
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from khedron.cost.tracker import CostTracker
from khedron.errors import JudgeError, ModelError, ProviderError
from khedron.judges.base import Judge, JudgeResult
from khedron.models.base import AnswerModel
from khedron.pipeline.generator import build_generator_prompt, generate_answer
from khedron.pipeline.ingester import ingest_conversation
from khedron.pipeline.judger import judge_response
from khedron.pipeline.retriever import retrieve_for_question
from khedron.providers.base import MemoryProvider, ProviderHealthStatus
from khedron.types import (
    APICallRecord,
    APICallResult,
    Conversation,
    ConversationIngestionRecord,
    ErrorRecord,
    Judgment,
    JudgmentVerdict,
    Memory,
    Question,
    QuestionCategory,
    QuestionEvaluationRecord,
    Response,
    RetrievalRecord,
    Session,
    Turn,
)


@dataclass
class FakeRunContext:
    run_id: str = "run_1"
    cost_tracker: CostTracker = field(default_factory=CostTracker)


class FakeRepository:
    def __init__(self) -> None:
        self.conversation_ingestions: list[ConversationIngestionRecord] = []
        self.retrievals: list[RetrievalRecord] = []
        self.responses: list[Response] = []
        self.judgments: list[Judgment] = []
        self.api_calls: list[APICallRecord] = []
        self.errors: list[ErrorRecord] = []

    async def append_conversation_ingestion(
        self,
        record: ConversationIngestionRecord,
    ) -> None:
        self.conversation_ingestions.append(record)

    async def append_retrieval(self, retrieval: RetrievalRecord) -> None:
        self.retrievals.append(retrieval)

    async def append_response(self, response: Response) -> None:
        self.responses.append(response)

    async def append_judgment(self, judgment: Judgment) -> None:
        self.judgments.append(judgment)

    async def append_api_call(self, record: APICallRecord) -> None:
        self.api_calls.append(record)

    async def append_error(self, error: ErrorRecord) -> None:
        self.errors.append(error)


class FakeProvider(MemoryProvider):
    def __init__(
        self,
        memories: list[Memory] | None = None,
        add_fail_turn_ids: set[str] | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self.add_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.search_calls: list[tuple[str, int, dict[str, Any] | None]] = []
        self._memories = memories or []
        self._add_fail_turn_ids = add_fail_turn_ids or set()
        self._search_error = search_error

    @property
    def provider_type(self) -> str:
        return "fake"

    @property
    def provider_version(self) -> str:
        return "test"

    async def initialize(self) -> None:
        return None

    async def health_check(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(healthy=True)

    async def reset(self) -> None:
        return None

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        self.add_calls.append((content, metadata))
        turn_id = metadata.get("turn_id") if metadata is not None else None
        if isinstance(turn_id, str) and turn_id in self._add_fail_turn_ids:
            raise ProviderError("fake add failed", turn_id=turn_id)
        return f"memory_{len(self.add_calls)}"

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        self.search_calls.append((query, top_k, metadata_filter))
        if self._search_error is not None:
            raise self._search_error
        return self._memories

    async def close(self) -> None:
        return None


class FakeAnswerModel(AnswerModel):
    def __init__(self, result: APICallResult | None = None, error: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.generate_calls: list[tuple[int, float]] = []
        self._result = result or _api_result(output="blue", model_id="answer-model", cost=0.02)
        self._error = error

    @property
    def model_id(self) -> str:
        return "answer-model"

    @property
    def vendor(self) -> str:
        return "fake-model-vendor"

    async def initialize(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        self.prompts.append(prompt)
        self.generate_calls.append((max_output_tokens, temperature))
        if self._error is not None:
            raise self._error
        return self._result

    async def close(self) -> None:
        return None


class FakeJudge(Judge):
    def __init__(self, result: JudgeResult | None = None, error: Exception | None = None) -> None:
        self.evaluate_calls: list[tuple[str, str, str, str | None]] = []
        self._result = result or JudgeResult(
            verdict=JudgmentVerdict.CORRECT,
            score=1.0,
            reasoning="Matches.",
            api_call=_api_result(output='{"verdict":"correct"}', model_id="judge-model", cost=0.03),
        )
        self._error = error

    @property
    def model_id(self) -> str:
        return "judge-model"

    @property
    def vendor(self) -> str:
        return "fake-judge-vendor"

    async def initialize(self) -> None:
        return None

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        self.evaluate_calls.append((question, expected_answer, generated_answer, category))
        if self._error is not None:
            raise self._error
        return self._result

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_ingester_happy_path_persists_metadata_counts_and_latency() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider()

    stats = await ingest_conversation(_conversation(n_turns=2), provider, repo, context)

    assert stats.n_turns_succeeded == 2
    assert stats.n_turns_failed == 0
    assert len(repo.conversation_ingestions) == 1
    record = repo.conversation_ingestions[0]
    assert record.run_id == "run_1"
    assert record.n_sessions == 1
    assert record.n_turns == 2
    assert record.total_latency_ms >= 0.0
    assert record.avg_latency_per_turn_ms >= 0.0
    assert provider.add_calls[0] == (
        "content 1",
        {
            "speaker": "Alice",
            "session_id": "session_1",
            "session_number": 1,
            "session_timestamp": "2026-01-01T00:00:00+00:00",
            "turn_id": "turn_1",
            "conversation_id": "conversation_1",
        },
    )
    assert repo.api_calls == []


@pytest.mark.asyncio
async def test_ingester_persists_recovery_provenance_id() -> None:
    repo = FakeRepository()
    stats = await ingest_conversation(
        _conversation(n_turns=1),
        FakeProvider(),
        repo,
        FakeRunContext(),
        ingestion_attempt_id="ingestion-attempt-1",
    )

    assert stats.record.ingestion_attempt_id == "ingestion-attempt-1"


@pytest.mark.asyncio
async def test_ingester_isolated_turn_failure_is_summarized() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider(add_fail_turn_ids={"turn_21"})

    stats = await ingest_conversation(_conversation(n_turns=21), provider, repo, context)

    assert stats.n_turns_succeeded == 20
    assert stats.n_turns_failed == 1
    assert repo.conversation_ingestions[0].error_summary == [
        {
            "session_id": "session_1",
            "session_number": 1,
            "turn_id": "turn_21",
            "error_type": "ProviderError",
            "error_message": "fake add failed (turn_id='turn_21')",
        }
    ]
    assert len(repo.errors) == 1
    assert repo.errors[0].phase == "ingest"
    assert repo.errors[0].question_id is None
    assert repo.errors[0].recovered is True


@pytest.mark.asyncio
async def test_retriever_persists_memory_ids_and_latency_without_provider_api_call() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider(memories=[_memory("m1"), _memory("m2")])
    question = _question()
    question_evaluation = _question_evaluation(question)

    retrieval = await retrieve_for_question(
        question, question_evaluation, provider, 10, repo, context
    )

    assert provider.search_calls == [(question.question_text, 10, None)]
    assert retrieval.query == question.question_text
    assert retrieval.n_returned == 2
    assert [memory.memory_id for memory in retrieval.memories] == ["m1", "m2"]
    assert retrieval.retrieval_latency_ms >= 0.0
    assert repo.retrievals == [retrieval]
    assert repo.api_calls == []


@pytest.mark.asyncio
async def test_retriever_persists_recovery_and_ingestion_provenance_ids() -> None:
    repo = FakeRepository()
    question = _question()
    retrieval = await retrieve_for_question(
        question,
        _question_evaluation(question),
        FakeProvider(memories=[_memory("m1")]),
        10,
        repo,
        FakeRunContext(),
        recovery_attempt_id="recovery-attempt-1",
        ingestion_attempt_id="ingestion-attempt-1",
    )

    assert retrieval.recovery_attempt_id == "recovery-attempt-1"
    assert retrieval.ingestion_attempt_id == "ingestion-attempt-1"


@pytest.mark.asyncio
async def test_retriever_records_an_empty_result_as_recovered_evidence() -> None:
    # A search that succeeds but returns nothing raises no exception, so it left no trace:
    # the model was handed "No memories retrieved", refused, and the run still reported a
    # clean bill of health. It must be countable without failing the question.
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider(memories=[])
    question = _question()
    question_evaluation = _question_evaluation(question)

    retrieval = await retrieve_for_question(
        question, question_evaluation, provider, 10, repo, context
    )

    assert retrieval.n_returned == 0
    assert repo.retrievals == [retrieval]
    assert len(repo.errors) == 1
    assert repo.errors[0].phase == "retrieve"
    assert repo.errors[0].question_id == question.question_id
    assert repo.errors[0].recovered is True


@pytest.mark.asyncio
async def test_retriever_does_not_flag_an_empty_result_when_zero_memories_were_requested() -> None:
    # top_k=0 is a legal configuration, and an empty result is then the correct outcome.
    # Logging a provider error per question would make errors.jsonl report failures for a
    # retrieval that behaved exactly as configured.
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider(memories=[])
    question = _question()
    question_evaluation = _question_evaluation(question)

    await retrieve_for_question(question, question_evaluation, provider, 0, repo, context)

    assert repo.errors == []


@pytest.mark.asyncio
async def test_retriever_provider_failure_appends_error_record() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    provider = FakeProvider(search_error=RuntimeError("search down"))
    question = _question()
    question_evaluation = _question_evaluation(question)

    with pytest.raises(ProviderError):
        await retrieve_for_question(question, question_evaluation, provider, 10, repo, context)

    assert len(repo.errors) == 1
    assert repo.errors[0].phase == "retrieve"
    assert repo.errors[0].question_id == question.question_id
    assert repo.errors[0].recovered is False
    assert repo.retrievals == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False, None])
async def test_retriever_preserves_retryability_from_provider_context(
    retryable: bool | None,
) -> None:
    repo = FakeRepository()
    question = _question()
    with pytest.raises(ProviderError):
        await retrieve_for_question(
            question,
            _question_evaluation(question),
            FakeProvider(search_error=ProviderError("down", retryable=retryable)),
            10,
            repo,
            FakeRunContext(),
        )
    assert repo.errors[0].retryable is retryable


@pytest.mark.asyncio
async def test_generator_persists_api_call_response_and_updates_cost_tracker() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    question = _question()
    retrieval = _retrieval(question, memories=[_memory("m1")])
    model = FakeAnswerModel()

    response = await generate_answer(
        question,
        retrieval,
        model,
        repo,
        context,
        max_output_tokens=77,
        temperature=0.2,
    )

    assert len(repo.api_calls) == 1
    assert repo.api_calls[0].phase == "generate"
    assert repo.api_calls[0].vendor == "fake-model-vendor"
    assert context.cost_tracker.total_cost_usd() == 0.02
    assert context.cost_tracker.cost_by_phase() == {"generate": 0.02}
    assert context.cost_tracker._records[0] is repo.api_calls[0]
    assert response.answer_text == "blue"
    assert response.prompt == repo.responses[0].prompt
    assert model.generate_calls == [(77, 0.2)]
    assert "[Memory 1] (Session 1, Speaker: Alice)" in response.prompt
    assert "Question: What color is Alice's notebook?" in response.prompt
    assert repo.responses == [response]


@pytest.mark.asyncio
async def test_generator_persists_recovery_attempt_id() -> None:
    repo = FakeRepository()
    question = _question()
    response = await generate_answer(
        question,
        _retrieval(question, memories=[_memory("m1")]),
        FakeAnswerModel(),
        repo,
        FakeRunContext(),
        recovery_attempt_id="recovery-attempt-1",
    )

    assert response.recovery_attempt_id == "recovery-attempt-1"


@pytest.mark.asyncio
async def test_generator_model_failure_appends_error_record_without_api_call() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    question = _question()
    retrieval = _retrieval(question, memories=[_memory("m1")])

    with pytest.raises(ModelError):
        await generate_answer(
            question,
            retrieval,
            FakeAnswerModel(error=RuntimeError("model down")),
            repo,
            context,
        )

    assert len(repo.errors) == 1
    assert repo.errors[0].phase == "generate"
    assert repo.errors[0].question_id == question.question_id
    assert repo.api_calls == []
    assert repo.responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False, None])
async def test_generator_preserves_retryability_from_model_context(retryable: bool | None) -> None:
    repo = FakeRepository()
    question = _question()
    with pytest.raises(ModelError):
        await generate_answer(
            question,
            _retrieval(question, memories=[_memory("m1")]),
            FakeAnswerModel(error=ModelError("down", retryable=retryable)),
            repo,
            FakeRunContext(),
        )
    assert repo.errors[0].retryable is retryable


@pytest.mark.asyncio
async def test_judger_persists_api_call_judgment_and_updates_cost_tracker() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    question = _question()
    response = _response(question)
    judge = FakeJudge()

    judgment = await judge_response(response, question, judge, repo, context)

    assert judge.evaluate_calls == [
        (
            question.question_text,
            question.expected_answer,
            response.answer_text,
            question.category.value,
        )
    ]
    assert len(repo.api_calls) == 1
    assert repo.api_calls[0].phase == "judge"
    assert repo.api_calls[0].vendor == "fake-judge-vendor"
    assert context.cost_tracker.total_cost_usd() == 0.03
    assert context.cost_tracker._records[0] is repo.api_calls[0]
    assert judgment.parsed_verdict is JudgmentVerdict.CORRECT
    assert repo.judgments == [judgment]


@pytest.mark.asyncio
async def test_judger_persists_recovery_attempt_id() -> None:
    repo = FakeRepository()
    question = _question()
    judgment = await judge_response(
        _response(question),
        question,
        FakeJudge(),
        repo,
        FakeRunContext(),
        recovery_attempt_id="recovery-attempt-1",
    )

    assert judgment.recovery_attempt_id == "recovery-attempt-1"


@pytest.mark.asyncio
async def test_judger_preserves_unknown_verdict_as_record() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    question = _question()
    response = _response(question)
    judge = FakeJudge(
        result=JudgeResult(
            verdict=JudgmentVerdict.UNKNOWN,
            score=0.0,
            reasoning="Ambiguous.",
            api_call=_api_result(
                output='{"verdict":"unknown"}',
                model_id="judge-model",
                cost=0.01,
                raw_response={"prompt": "judge prompt"},
            ),
        )
    )

    judgment = await judge_response(response, question, judge, repo, context)

    assert judgment.parsed_verdict is JudgmentVerdict.UNKNOWN
    assert judgment.parsed_score == 0.0
    assert judgment.prompt == "judge prompt"
    assert repo.errors == []


@pytest.mark.asyncio
async def test_judger_failure_appends_error_record_without_api_call() -> None:
    repo = FakeRepository()
    context = FakeRunContext()
    question = _question()
    response = _response(question)

    with pytest.raises(JudgeError):
        await judge_response(
            response,
            question,
            FakeJudge(error=RuntimeError("judge down")),
            repo,
            context,
        )

    assert len(repo.errors) == 1
    assert repo.errors[0].phase == "judge"
    assert repo.errors[0].question_id == question.question_id
    assert repo.api_calls == []
    assert repo.judgments == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False, None])
async def test_judger_preserves_retryability_from_judge_context(retryable: bool | None) -> None:
    repo = FakeRepository()
    question = _question()
    with pytest.raises(JudgeError):
        await judge_response(
            _response(question),
            question,
            FakeJudge(error=JudgeError("down", retryable=retryable)),
            repo,
            FakeRunContext(),
        )
    assert repo.errors[0].retryable is retryable


def _conversation(n_turns: int) -> Conversation:
    return Conversation(
        conversation_id="conversation_1",
        speakers=["Alice", "Bob"],
        sessions=[
            Session(
                session_id="session_1",
                session_number=1,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                turns=[
                    Turn(
                        turn_id=f"turn_{index}",
                        speaker="Alice" if index % 2 else "Bob",
                        content=f"content {index}",
                    )
                    for index in range(1, n_turns + 1)
                ],
            )
        ],
    )


def _question() -> Question:
    return Question(
        question_id="question_1",
        conversation_id="conversation_1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="What color is Alice's notebook?",
        expected_answer="Blue.",
    )


def _question_evaluation(question: Question) -> QuestionEvaluationRecord:
    return QuestionEvaluationRecord(
        question_evaluation_id="question_eval_1",
        run_id="run_1",
        question_id=question.question_id,
        conversation_id=question.conversation_id,
        category=question.category,
        question_text=question.question_text,
        expected_answer=question.expected_answer,
        is_audited_error=question.is_audited_error,
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
    )


def _memory(memory_id: str) -> Memory:
    return Memory(
        memory_id=memory_id,
        content="Alice bought a blue notebook.",
        metadata={"session_number": 1, "speaker": "Alice"},
        score=0.9,
    )


def _retrieval(question: Question, memories: list[Memory]) -> RetrievalRecord:
    return RetrievalRecord(
        retrieval_id="retrieval_1",
        question_evaluation_id="question_eval_1",
        run_id="run_1",
        question_id=question.question_id,
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        query=question.question_text,
        top_k=10,
        n_returned=len(memories),
        memories=memories,
        retrieval_latency_ms=1.5,
    )


def _response(question: Question) -> Response:
    return Response(
        response_id="response_1",
        run_id="run_1",
        question_id=question.question_id,
        retrieval_id="retrieval_1",
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        model_id="answer-model",
        prompt="prompt",
        answer_text="Blue.",
        input_tokens=10,
        output_tokens=2,
        latency_ms=5.0,
        cost_usd=0.02,
    )


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
        output_tokens=3,
        latency_ms=7.5,
        cost_usd=cost,
        model_id=model_id,
        raw_response=raw_response or {},
    )


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "prompts" / name


def test_canonical_v2_prompt_puts_the_session_date_in_front_of_the_model() -> None:
    # The ingester stored session_timestamp, the provider returned it, and the v1
    # formatter rendered only session number and speaker -- so temporal questions were unanswerable
    # however well retrieval worked, and scored 0.6%. Asserted on the rendered prompt rather than on
    # the prompt file existing, because the file existed and looked correct throughout.
    memory = Memory(
        memory_id="memory_1",
        content="Melanie painted a sunrise.",
        metadata={
            "speaker": "Melanie",
            "session_number": 3,
            "session_timestamp": "2023-05-07T10:00:00+00:00",
        },
        score=0.9,
    )

    prompt = build_generator_prompt(
        _question(),
        [memory],
        template=_prompt_path("generator_canonical_v2.txt").read_text(encoding="utf-8"),
    )

    if "2023" not in prompt:
        raise AssertionError(prompt)
    if "May 07, 2023" not in prompt:
        raise AssertionError(prompt)
    # Session number and speaker survive: the date is added, nothing is traded away for it.
    if "Session 3" not in prompt or "Melanie" not in prompt:
        raise AssertionError(prompt)


def test_canonical_v1_rendering_is_untouched_by_the_v2_formatter() -> None:
    # The reason v2 is a new formatter and a new prompt file rather than a fix to the v1 one: the
    # fingerprint hashes the prompt FILE, not this code. Editing the v1 formatter would change what
    # canonical-v1 measures while its fingerprint stayed identical -- silent drift under a fixed
    # identity, worse than the defect being repaired.
    memory = Memory(
        memory_id="memory_1",
        content="Melanie painted a sunrise.",
        metadata={
            "speaker": "Melanie",
            "session_number": 3,
            "session_timestamp": "2023-05-07T10:00:00+00:00",
        },
        score=0.9,
    )

    prompt = build_generator_prompt(
        _question(),
        [memory],
        template=_prompt_path("generator_canonical_v1.txt").read_text(encoding="utf-8"),
    )

    if "2023" in prompt:
        raise AssertionError(f"canonical-v1 rendering changed: {prompt}")
    if "(Session 3, Speaker: Melanie)" not in prompt:
        raise AssertionError(prompt)
