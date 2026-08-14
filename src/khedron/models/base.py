from __future__ import annotations

from abc import ABC, abstractmethod

from khedron.types import APICallResult

__all__ = ["AnswerModel"]


class AnswerModel(ABC):
    """Abstract base for answer-generating models."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Vendor name."""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize model resources."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        """Generate an answer for the given prompt."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release model resources."""
        raise NotImplementedError
