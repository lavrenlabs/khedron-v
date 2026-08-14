from __future__ import annotations

from pathlib import Path

from khedron.judges.llm import LLMJudge
from khedron.judges.registry import register_judge
from khedron.models.base import AnswerModel
from khedron.models.google import GoogleModel, GoogleModelConfig

__all__ = ["GoogleJudge"]


@register_judge("google")
class GoogleJudge(LLMJudge):
    """Google-backed judge composed from the Google answer-model adapter."""

    def __init__(
        self,
        config: GoogleModelConfig,
        *,
        model: AnswerModel | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        prompt_path: Path | None = None,
    ) -> None:
        kwargs = {} if prompt_path is None else {"prompt_path": prompt_path}
        super().__init__(
            model if model is not None else GoogleModel(config),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs,
        )
