from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from khedron.models.base import AnswerModel
from khedron.types import APICallResult


def api_call_result() -> APICallResult:
    return APICallResult(
        output="Rome",
        input_tokens=10,
        output_tokens=2,
        latency_ms=123.0,
        cost_usd=0.01,
        model_id="synthetic-answer-model-1",
        raw_response={"id": "response-1"},
    )


class CompleteAnswerModel(AnswerModel):
    @property
    def model_id(self) -> str:
        return "synthetic-answer-model-1"

    @property
    def vendor(self) -> str:
        return "synthetic"

    async def initialize(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        if prompt == "":
            raise AssertionError(prompt)
        if max_output_tokens != 1024:
            raise AssertionError(max_output_tokens)
        if temperature != 0.0:
            raise AssertionError(temperature)
        return api_call_result()

    async def close(self) -> None:
        return None


def test_answer_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        AnswerModel()


def test_incomplete_answer_model_subclass_is_abstract() -> None:
    class IncompleteAnswerModel(AnswerModel):
        @property
        def model_id(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteAnswerModel()


@pytest.mark.asyncio
async def test_complete_answer_model_subclass_is_awaitable() -> None:
    model = CompleteAnswerModel()

    if model.model_id != "synthetic-answer-model-1":
        raise AssertionError(model.model_id)
    if model.vendor != "synthetic":
        raise AssertionError(model.vendor)

    await model.initialize()
    result = await model.generate("Where did Alice move?")
    await model.close()

    if result != api_call_result():
        raise AssertionError(result)


def test_answer_model_async_contract_and_annotations() -> None:
    expected_abstract_methods = {"close", "generate", "initialize", "model_id", "vendor"}
    if AnswerModel.__abstractmethods__ != expected_abstract_methods:
        raise AssertionError(AnswerModel.__abstractmethods__)

    for method_name in ("initialize", "generate", "close"):
        if not inspect.iscoroutinefunction(getattr(AnswerModel, method_name)):
            raise AssertionError(method_name)

    generate_signature = inspect.signature(AnswerModel.generate)
    if generate_signature.parameters["max_output_tokens"].default != 1024:
        raise AssertionError(generate_signature)
    if generate_signature.parameters["temperature"].default != 0.0:
        raise AssertionError(generate_signature)

    if get_type_hints(AnswerModel.generate)["return"] is not APICallResult:
        raise AssertionError(get_type_hints(AnswerModel.generate))
