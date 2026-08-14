from __future__ import annotations

from abc import ABC, abstractmethod

from khedron.types import Conversation, Question

__all__ = ["Benchmark"]


class Benchmark(ABC):
    """Abstract base for all benchmarks."""

    @property
    @abstractmethod
    def benchmark_type(self) -> str:
        """Stable identifier for this benchmark."""
        raise NotImplementedError

    @property
    @abstractmethod
    def benchmark_version(self) -> str:
        """Declared version of the benchmark dataset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_checksum(self) -> str:
        """Declared SHA-256 checksum of the benchmark dataset."""
        raise NotImplementedError

    @abstractmethod
    async def load(self) -> None:
        """Load benchmark data."""
        raise NotImplementedError

    @abstractmethod
    def get_conversations(self) -> list[Conversation]:
        """Return all conversations in the benchmark."""
        raise NotImplementedError

    @abstractmethod
    def get_questions(
        self,
        conversation_id: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Question]:
        """Return questions, optionally filtered by conversation and category."""
        raise NotImplementedError

    @property
    def corpus_conversation_count(self) -> int:
        """Conversations the dataset holds before any configured filter narrows it.

        Concrete by default rather than abstract: a benchmark that cannot be narrowed has nothing
        to disclose, and returning the selected count is then exactly right. LoCoMo overrides it,
        because a run restricted to one of ten conversations must say so in its report -- an
        artifact that measured a tenth of the corpus and reads like one that measured all of it is
        the failure mode this methodology version exists to remove.
        """
        return len(self.get_conversations())

    @abstractmethod
    def get_audit_errors(self) -> set[str]:
        """Return question identifiers known to have ground-truth errors."""
        raise NotImplementedError
