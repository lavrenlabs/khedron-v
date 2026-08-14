from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from khedron.analysis.types import FailureAnalysisReport
from khedron.types import (
    ExperimentResult,
    QuestionRecord,
    RunStartedEvent,
    RunStatus,
    SuiteStatus,
)

__all__ = [
    "CostSummary",
    "MethodologyDisclosure",
    "ReportContext",
]

_NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]
_ScoringMode: TypeAlias = Literal["standard", "audited"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class MethodologyDisclosure(_FrozenModel):
    """Methodology and benchmark metadata disclosed in reports."""

    def retrieval_budget_label(self, configured_top_k: int) -> str:
        """Render the retrieval budget, or say it did not apply.

        A full-context baseline discards `top_k`, and `ExperimentConfig` supplies a
        default of 10 whether or not anything used it -- so a report would have stated "Top-K
        retrieval 10" for a run that applied no budget at all. That is an artifact asserting
        something false about its own execution, which is the defect class this methodology
        version exists to remove.
        """
        if not self.retrieval_budget_applies:
            return "not applicable (provider retrieves no subset)"
        return str(configured_top_k)

    @property
    def corpus_is_complete(self) -> bool | None:
        """Whether the run covered the whole corpus, or None when it was not recorded."""
        if (
            self.corpus_conversations_evaluated is None
            or self.corpus_conversations_available is None
        ):
            return None
        return self.corpus_conversations_evaluated >= self.corpus_conversations_available

    @property
    def corpus_scope_label(self) -> str:
        """Corpus coverage, phrased so a partial run cannot read as a complete one.

        A suite may narrow itself to a few conversations for good reasons, but the report it
        produces is otherwise identical to one over the whole corpus: same headline phrasing, same
        Wilson intervals. Over 158 questions those intervals are arithmetically valid and say
        nothing about the system, so the scope has to appear beside the number rather than only in
        the config that produced it.
        """
        if (
            self.corpus_conversations_evaluated is None
            or self.corpus_conversations_available is None
        ):
            return "not recorded"
        if self.corpus_is_complete:
            return f"all {self.corpus_conversations_available} conversations"
        return (
            f"{self.corpus_conversations_evaluated} of "
            f"{self.corpus_conversations_available} conversations -- a partial corpus, so these "
            "scores describe the selected subset and are not a measurement of the benchmark"
        )

    @property
    def scored_scope_label(self) -> str:
        """Human-readable scope for the headline, for templates to render beside a score."""
        if self.scored_categories is None:
            return "all evaluated categories"
        return "the answerable subset (" + ", ".join(self.scored_categories) + ")"

    methodology_version: str
    methodology_profile: str
    scoring_mode: _ScoringMode
    confidence_interval: str
    same_vendor_warning: bool
    benchmark_type: str
    benchmark_version: str
    benchmark_checksum: str
    # The categories the run's overall_* covers. Reports must state it: an "Overall" with no
    # scope cannot be told apart from a figure over every category, and under canonical-v2 the two
    # differ by roughly 19 points. None means every evaluated category.
    scored_categories: list[str] | None = None
    # Whether the provider that produced this run honours a retrieval budget. False means the
    # configured `top_k_retrieval` was never applied, so reports must not print it as though it
    # described the run.
    retrieval_budget_applies: bool = True
    # How much of the corpus the run covered. None on runs recorded before this was captured;
    # reports then say the coverage is unrecorded rather than implying it was complete.
    corpus_conversations_evaluated: int | None = None
    corpus_conversations_available: int | None = None
    methodology_fingerprint: str | None = None
    reference_name: str | None = None
    reference_url: str | None = None
    reference_commit: str | None = None


class IngestionSummary(_FrozenModel):
    """What reached the memory store, and what did not.

    Reported because a lost turn is not visible anywhere in a score: the run answers the same
    questions, computes the same intervals, and simply had less to retrieve from. A prior run
    lost a handful of turns during ingestion and nothing in the resulting numbers said so. The
    suite now tolerates a bounded loss, and a tolerated loss that goes unreported is a silent one.
    """

    n_turns: int = 0
    n_turns_failed: int = 0
    # conversation_id -> turns lost there. Empty on a clean run; the manifest the owner asked for
    # when accepting a non-zero threshold, so the loss can be located rather than merely counted.
    failed_by_conversation: dict[str, int] = {}

    @property
    def failure_rate(self) -> float:
        return self.n_turns_failed / self.n_turns if self.n_turns else 0.0

    @property
    def label(self) -> str:
        if not self.n_turns:
            return "not recorded"
        if not self.n_turns_failed:
            return f"complete ({self.n_turns} turns, none lost)"
        where = ", ".join(
            f"{conversation}: {count}"
            for conversation, count in sorted(self.failed_by_conversation.items())
        )
        return (
            f"{self.n_turns_failed} of {self.n_turns} turns lost "
            f"({100 * self.failure_rate:.3f}%) -- {where}. Retrieval saw less than the corpus."
        )


class CostSummary(_FrozenModel):
    """Report-ready cost totals and breakdowns."""

    total_cost_usd: _NonNegativeFloat
    cost_by_phase: dict[str, _NonNegativeFloat]
    cost_by_model: dict[str, _NonNegativeFloat]


class ReportContext(_FrozenModel):
    """Repository-backed runtime view consumed by report templates."""

    run_status: RunStatus
    run_started_event: RunStartedEvent
    suite_status: SuiteStatus
    experiment_result: ExperimentResult | None
    questions: list[QuestionRecord]
    failed_questions: list[QuestionRecord]
    failure_analysis: FailureAnalysisReport
    methodology: MethodologyDisclosure
    # Why this run's scores may not be published, empty when they may. Carried into the artifact
    # rather than printed to the console: `--force` promised the report would state it, and a
    # warning that lives only in a terminal scrollback is not a disclosure. The report is what
    # leaves this machine.
    withdrawal_reasons: list[str] = Field(default_factory=list)
    cost_summary: CostSummary
    ingestion: IngestionSummary = IngestionSummary()
