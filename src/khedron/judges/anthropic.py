from __future__ import annotations

from pathlib import Path

from khedron.judges.llm import LLMJudge
from khedron.judges.registry import register_judge
from khedron.models.anthropic import AnthropicModel, AnthropicModelConfig
from khedron.models.base import AnswerModel

__all__ = ["AnthropicJudge"]


@register_judge("anthropic")
class AnthropicJudge(LLMJudge):
    """Anthropic-backed judge composed from the Anthropic answer-model adapter."""

    def __init__(
        self,
        config: AnthropicModelConfig,
        *,
        model: AnswerModel | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        prompt_path: Path | None = None,
    ) -> None:
        kwargs = {} if prompt_path is None else {"prompt_path": prompt_path}
        super().__init__(
            model if model is not None else AnthropicModel(config),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs,
        )
