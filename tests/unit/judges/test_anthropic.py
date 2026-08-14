from __future__ import annotations

import pytest

from khedron.judges import AnthropicJudge
from khedron.judges.registry import get_judge_class
from khedron.models.anthropic import AnthropicModelConfig
from khedron.models.base import AnswerModel
from khedron.types import APICallResult, JudgmentVerdict


class FakeAnthropicModel(AnswerModel):
    def __init__(self) -> None:
        self.initialize_count = 0
        self.close_count = 0
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "claude-judge-test"

    @property
    def vendor(self) -> str:
        return "anthropic"

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        if max_output_tokens != 48 or temperature != 0.1:
            raise AssertionError((max_output_tokens, temperature))
        self.prompts.append(prompt)
        return APICallResult(
            output='{"verdict": "incorrect", "score": 0.0, "reasoning": "The answers differ."}',
            input_tokens=10,
            output_tokens=6,
            latency_ms=1.0,
            cost_usd=0.0,
            model_id=self.model_id,
        )

    async def close(self) -> None:
        self.close_count += 1


def test_anthropic_judge_is_registered() -> None:
    if get_judge_class("anthropic") is not AnthropicJudge:
        raise AssertionError("anthropic judge was not registered")


@pytest.mark.asyncio
async def test_anthropic_judge_delegates_lifecycle_and_evaluation() -> None:
    model = FakeAnthropicModel()
    judge = AnthropicJudge(
        AnthropicModelConfig(model_id="claude-sonnet-4-5"),
        model=model,
        max_output_tokens=48,
        temperature=0.1,
    )

    await judge.initialize()
    result = await judge.evaluate("Question?", "Answer", "Different", category="single_hop")
    await judge.close()

    if result.verdict is not JudgmentVerdict.INCORRECT:
        raise AssertionError(result)
    if model.initialize_count != 1 or model.close_count != 1:
        raise AssertionError((model.initialize_count, model.close_count))
    if "Different" not in model.prompts[0]:
        raise AssertionError(model.prompts)
    if judge.vendor != "anthropic" or judge.model_id != "claude-judge-test":
        raise AssertionError((judge.vendor, judge.model_id))
