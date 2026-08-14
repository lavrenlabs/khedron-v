from __future__ import annotations

import pytest

from khedron.judges import GoogleJudge
from khedron.judges.registry import get_judge_class
from khedron.models.base import AnswerModel
from khedron.models.google import GoogleModelConfig
from khedron.types import APICallResult, JudgmentVerdict


class FakeGoogleModel(AnswerModel):
    def __init__(self) -> None:
        self.initialize_count = 0
        self.close_count = 0
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "gemini-judge-test"

    @property
    def vendor(self) -> str:
        return "google"

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        if max_output_tokens != 24 or temperature != 0.3:
            raise AssertionError((max_output_tokens, temperature))
        self.prompts.append(prompt)
        return APICallResult(
            output='{"verdict": "partial", "score": 0.5, "reasoning": "Some facts match."}',
            input_tokens=10,
            output_tokens=6,
            latency_ms=1.0,
            cost_usd=0.0,
            model_id=self.model_id,
        )

    async def close(self) -> None:
        self.close_count += 1


def test_google_judge_is_registered() -> None:
    if get_judge_class("google") is not GoogleJudge:
        raise AssertionError("google judge was not registered")


@pytest.mark.asyncio
async def test_google_judge_delegates_lifecycle_and_evaluation() -> None:
    model = FakeGoogleModel()
    judge = GoogleJudge(
        GoogleModelConfig(model_id="gemini-2.5-flash"),
        model=model,
        max_output_tokens=24,
        temperature=0.3,
    )

    await judge.initialize()
    result = await judge.evaluate("Question?", "Alice and Bob", "Alice", category="multi_hop")
    await judge.close()

    if result.verdict is not JudgmentVerdict.PARTIAL:
        raise AssertionError(result)
    if model.initialize_count != 1 or model.close_count != 1:
        raise AssertionError((model.initialize_count, model.close_count))
    if "multi_hop" not in model.prompts[0]:
        raise AssertionError(model.prompts)
    if judge.vendor != "google" or judge.model_id != "gemini-judge-test":
        raise AssertionError((judge.vendor, judge.model_id))
