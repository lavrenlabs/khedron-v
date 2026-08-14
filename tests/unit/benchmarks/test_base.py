from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import get_type_hints

import pytest

from khedron.benchmarks.base import Benchmark
from khedron.types import Conversation, Question, QuestionCategory, Session

NOW = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)


def conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        speakers=["Alice", "Bob"],
        sessions=[Session(session_id="session-1", session_number=0, timestamp=NOW, turns=[])],
    )


def question() -> Question:
    return Question(
        question_id="q-1",
        conversation_id="conv-1",
        category=QuestionCategory.SINGLE_HOP,
        question_text="Where did Alice move?",
        expected_answer="Rome",
    )


class CompleteBenchmark(Benchmark):
    def __init__(self) -> None:
        self.loaded = False

    @property
    def benchmark_type(self) -> str:
        return "synthetic_benchmark"

    @property
    def benchmark_version(self) -> str:
        return "1.0"

    @property
    def dataset_checksum(self) -> str:
        return "0" * 64

    async def load(self) -> None:
        self.loaded = True

    def get_conversations(self) -> list[Conversation]:
        return [conversation()]

    def get_questions(
        self,
        conversation_id: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Question]:
        result = [question()]
        if conversation_id is not None:
            result = [item for item in result if item.conversation_id == conversation_id]
        if categories is not None:
            result = [item for item in result if item.category.value in categories]
        return result

    def get_audit_errors(self) -> set[str]:
        return {"q-bad"}


def test_benchmark_is_abstract() -> None:
    with pytest.raises(TypeError):
        Benchmark()


def test_incomplete_benchmark_subclass_is_abstract() -> None:
    class IncompleteBenchmark(Benchmark):
        @property
        def benchmark_type(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteBenchmark()


@pytest.mark.asyncio
async def test_complete_benchmark_subclass_implements_contract() -> None:
    benchmark = CompleteBenchmark()

    if benchmark.benchmark_type != "synthetic_benchmark":
        raise AssertionError(benchmark.benchmark_type)
    if benchmark.benchmark_version != "1.0":
        raise AssertionError(benchmark.benchmark_version)
    if benchmark.dataset_checksum != "0" * 64:
        raise AssertionError(benchmark.dataset_checksum)

    await benchmark.load()
    conversations = benchmark.get_conversations()
    questions = benchmark.get_questions(conversation_id="conv-1", categories=["single_hop"])
    audit_errors = benchmark.get_audit_errors()

    if benchmark.loaded is not True:
        raise AssertionError(benchmark.loaded)
    if len(conversations) != 1 or conversations[0].conversation_id != "conv-1":
        raise AssertionError(conversations)
    if len(questions) != 1 or questions[0].question_id != "q-1":
        raise AssertionError(questions)
    if audit_errors != {"q-bad"}:
        raise AssertionError(audit_errors)


def test_benchmark_accessor_methods_are_synchronous() -> None:
    if not inspect.iscoroutinefunction(Benchmark.load):
        raise AssertionError(Benchmark.load)

    for method_name in ("get_conversations", "get_questions", "get_audit_errors"):
        if inspect.iscoroutinefunction(getattr(Benchmark, method_name)):
            raise AssertionError(method_name)
        if inspect.iscoroutinefunction(getattr(CompleteBenchmark, method_name)):
            raise AssertionError(method_name)


def test_benchmark_property_names_and_annotations() -> None:
    expected_abstract_methods = {
        "benchmark_type",
        "benchmark_version",
        "dataset_checksum",
        "get_audit_errors",
        "get_conversations",
        "get_questions",
        "load",
    }
    if Benchmark.__abstractmethods__ != expected_abstract_methods:
        raise AssertionError(Benchmark.__abstractmethods__)

    if get_type_hints(Benchmark.get_conversations)["return"] != list[Conversation]:
        raise AssertionError(get_type_hints(Benchmark.get_conversations))
    if get_type_hints(Benchmark.get_questions)["return"] != list[Question]:
        raise AssertionError(get_type_hints(Benchmark.get_questions))
    if get_type_hints(Benchmark.get_audit_errors)["return"] != set[str]:
        raise AssertionError(get_type_hints(Benchmark.get_audit_errors))

    get_questions_signature = inspect.signature(Benchmark.get_questions)
    if get_questions_signature.parameters["conversation_id"].default is not None:
        raise AssertionError(get_questions_signature)
    if get_questions_signature.parameters["categories"].default is not None:
        raise AssertionError(get_questions_signature)
