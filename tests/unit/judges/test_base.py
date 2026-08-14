from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from khedron.judges.base import Judge, JudgeResult
from khedron.types import APICallResult, JudgmentVerdict


def api_call_result() -> APICallResult:
    return APICallResult(
        output='{"verdict": "correct"}',
        input_tokens=20,
        output_tokens=5,
        latency_ms=234.0,
        cost_usd=0.02,
        model_id="synthetic-judge-model-1",
        raw_response={"id": "judge-response-1"},
    )


def judge_result() -> JudgeResult:
    return JudgeResult(
        verdict=JudgmentVerdict.CORRECT,
        score=1.0,
        reasoning="The generated answer matches.",
        api_call=api_call_result(),
    )


class CompleteJudge(Judge):
    @property
    def model_id(self) -> str:
        return "synthetic-judge-model-1"

    @property
    def vendor(self) -> str:
        return "synthetic"

    async def initialize(self) -> None:
        return None

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        if question == "" or expected_answer == "" or generated_answer == "":
            raise AssertionError((question, expected_answer, generated_answer))
        if category is not None and category != "single_hop":
            raise AssertionError(category)
        return judge_result()

    async def close(self) -> None:
        return None


def test_judge_result_validates_score_and_is_frozen() -> None:
    result = judge_result()

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if result.api_call != api_call_result():
        raise AssertionError(result)

    with pytest.raises(ValidationError):
        JudgeResult(
            verdict=JudgmentVerdict.INCORRECT,
            score=-0.01,
            reasoning="Too low.",
            api_call=api_call_result(),
        )

    with pytest.raises(ValidationError):
        JudgeResult(
            verdict=JudgmentVerdict.INCORRECT,
            score=1.01,
            reasoning="Too high.",
            api_call=api_call_result(),
        )

    with pytest.raises(ValidationError):
        result.score = 0.0


def test_judge_is_abstract() -> None:
    with pytest.raises(TypeError):
        Judge()


def test_incomplete_judge_subclass_is_abstract() -> None:
    class IncompleteJudge(Judge):
        @property
        def model_id(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteJudge()


@pytest.mark.asyncio
async def test_complete_judge_subclass_is_awaitable() -> None:
    judge = CompleteJudge()

    if judge.model_id != "synthetic-judge-model-1":
        raise AssertionError(judge.model_id)
    if judge.vendor != "synthetic":
        raise AssertionError(judge.vendor)

    await judge.initialize()
    result = await judge.evaluate(
        question="Where did Alice move?",
        expected_answer="Rome",
        generated_answer="Rome",
        category="single_hop",
    )
    await judge.close()

    if result != judge_result():
        raise AssertionError(result)


def test_judge_async_contract_and_annotations() -> None:
    expected_abstract_methods = {"close", "evaluate", "initialize", "model_id", "vendor"}
    if Judge.__abstractmethods__ != expected_abstract_methods:
        raise AssertionError(Judge.__abstractmethods__)

    for method_name in ("initialize", "evaluate", "close"):
        if not inspect.iscoroutinefunction(getattr(Judge, method_name)):
            raise AssertionError(method_name)

    evaluate_signature = inspect.signature(Judge.evaluate)
    if evaluate_signature.parameters["category"].default is not None:
        raise AssertionError(evaluate_signature)

    if get_type_hints(Judge.evaluate)["return"] is not JudgeResult:
        raise AssertionError(get_type_hints(Judge.evaluate))
