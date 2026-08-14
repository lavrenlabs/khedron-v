from __future__ import annotations

import pytest

from khedron.judges import OpenAIJudge
from khedron.judges.registry import get_judge_class
from khedron.models.base import AnswerModel
from khedron.models.openai import OpenAIModelConfig
from khedron.types import APICallResult, JudgmentVerdict


class FakeOpenAIModel(AnswerModel):
    def __init__(self) -> None:
        self.initialize_count = 0
        self.close_count = 0
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "gpt-4o-mini-judge-test"

    @property
    def vendor(self) -> str:
        return "openai"

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        if max_output_tokens != 32 or temperature != 0.0:
            raise AssertionError((max_output_tokens, temperature))
        self.prompts.append(prompt)
        return APICallResult(
            output='{"verdict": "correct", "score": 1.0, "reasoning": "The answers match."}',
            input_tokens=10,
            output_tokens=6,
            latency_ms=1.0,
            cost_usd=0.0,
            model_id=self.model_id,
        )

    async def close(self) -> None:
        self.close_count += 1


def test_openai_judge_is_registered() -> None:
    if get_judge_class("openai") is not OpenAIJudge:
        raise AssertionError("openai judge was not registered")


@pytest.mark.asyncio
async def test_openai_judge_delegates_lifecycle_and_evaluation() -> None:
    model = FakeOpenAIModel()
    judge = OpenAIJudge(
        OpenAIModelConfig(model_id="gpt-4o-mini-2024-07-18"),
        model=model,
        max_output_tokens=32,
    )

    await judge.initialize()
    result = await judge.evaluate("Question?", "Answer", "Answer", category="single_hop")
    await judge.close()

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if model.initialize_count != 1 or model.close_count != 1:
        raise AssertionError((model.initialize_count, model.close_count))
    if "Question?" not in model.prompts[0]:
        raise AssertionError(model.prompts)
    if judge.vendor != "openai" or judge.model_id != "gpt-4o-mini-judge-test":
        raise AssertionError((judge.vendor, judge.model_id))
