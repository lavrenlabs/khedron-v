from __future__ import annotations

from pathlib import Path

from khedron.judges.llm import LLMJudge
from khedron.judges.registry import register_judge
from khedron.models.base import AnswerModel
from khedron.models.openai import OpenAIModel, OpenAIModelConfig

__all__ = ["OpenAIJudge"]


@register_judge("openai")
class OpenAIJudge(LLMJudge):
    """OpenAI-backed judge composed from the OpenAI answer-model adapter."""

    def __init__(
        self,
        config: OpenAIModelConfig,
        *,
        model: AnswerModel | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        prompt_path: Path | None = None,
    ) -> None:
        kwargs = {} if prompt_path is None else {"prompt_path": prompt_path}
        super().__init__(
            model if model is not None else OpenAIModel(config),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs,
        )
