from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from khedron.types import APICallResult, JudgmentVerdict

__all__ = ["Judge", "JudgeResult"]


class JudgeResult(BaseModel):
    """Result of a judging operation."""

    model_config = ConfigDict(frozen=True)

    verdict: JudgmentVerdict
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    api_call: APICallResult


class Judge(ABC):
    """Abstract base for judges."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the judge model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Vendor name."""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize judge resources."""
        raise NotImplementedError

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        """Judge whether a generated answer matches the expected answer."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release judge resources."""
        raise NotImplementedError
