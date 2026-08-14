from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeAlias, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from khedron.analysis import ComparisonReport, FailureAnalyzer
from khedron.errors import ConfigurationError, PersistenceError
from khedron.methodology import METHODOLOGY_FINGERPRINT_KEY
from khedron.persistence.repository import RunRepository
from khedron.publishability import resolve_disposition
from khedron.reporting.context import (
    CostSummary,
    IngestionSummary,
    MethodologyDisclosure,
    ReportContext,
)
from khedron.types import (
    APICallRecord,
    ConversationIngestionRecord,
    JudgmentVerdict,
    QuestionRecord,
    RunStartedEvent,
)

__all__ = ["ReportContextBuilder", "ReportGenerator"]

ReportMode: TypeAlias = Literal["standard", "audited"]


class ReportContextBuilder:
    """Assemble repository-backed report contexts for one persisted run."""

    def __init__(self, failure_analyzer: FailureAnalyzer | None = None) -> None:
        self._failure_analyzer = failure_analyzer or FailureAnalyzer()

    def build(
        self,
        run_id: str,
        repository: RunRepository,
        mode: ReportMode = "audited",
    ) -> ReportContext:
        """Build a report-ready context from public repository read APIs."""
        if mode not in {"standard", "audited"}:
            raise ConfigurationError("Unsupported report mode.", mode=mode)

        run_status = repository.get_run_status(run_id)
        run_started_event = repository.get_run_started_event(run_id)
        suite_status = repository.get_suite_status(run_started_event.suite_id)
        experiment_result = repository.get_experiment_result_for_run(run_id)
        questions = repository.get_questions_for_run(run_id)
        api_calls = repository.get_api_calls_for_run(run_id)
        ingestions = repository.get_conversation_ingestions_for_run(run_id)

        return ReportContext(
            # Resolved here rather than passed in, so every report states it whatever the caller
            # remembered to do. A disclosure the caller can forget is not one.
            withdrawal_reasons=list(resolve_disposition(run_started_event).reasons),
            run_status=run_status,
            run_started_event=run_started_event,
            suite_status=suite_status,
            experiment_result=experiment_result,
            questions=questions,
            failed_questions=_failed_questions(run_id, repository, questions),
            failure_analysis=self._failure_analyzer.analyze(run_id, repository),
            methodology=MethodologyDisclosure(
                methodology_version=run_status.methodology_version,
                methodology_profile=run_status.methodology_profile,
                scoring_mode=mode,
                confidence_interval="Wilson 95%",
                same_vendor_warning=(
                    run_started_event.answer_model_vendor == run_started_event.judge_model_vendor
                ),
                benchmark_type=run_started_event.benchmark_type,
                benchmark_version=run_started_event.benchmark_version,
                benchmark_checksum=run_started_event.benchmark_checksum,
                retrieval_budget_applies=_retrieval_budget_applied(run_started_event),
                corpus_conversations_evaluated=_optional_runtime_int(
                    run_started_event.runtime_environment, "corpus_conversations_evaluated"
                ),
                corpus_conversations_available=_optional_runtime_int(
                    run_started_event.runtime_environment, "corpus_conversations_available"
                ),
                scored_categories=_optional_runtime_list(
                    run_started_event.runtime_environment,
                    "methodology_scored_categories",
                ),
                methodology_fingerprint=_optional_runtime_string(
                    run_started_event.runtime_environment,
                    METHODOLOGY_FINGERPRINT_KEY,
                ),
                reference_name=_optional_runtime_string(
                    run_started_event.runtime_environment,
                    "methodology_profile_reference",
                ),
                reference_url=_optional_runtime_string(
                    run_started_event.runtime_environment,
                    "methodology_profile_reference_url",
                ),
                reference_commit=_optional_runtime_string(
                    run_started_event.runtime_environment,
                    "methodology_profile_reference_commit",
                ),
            ),
            cost_summary=_cost_summary(run_status.total_cost_usd, api_calls),
            ingestion=_ingestion_summary(ingestions),
        )


class ReportGenerator:
    """Render supplied report contexts and comparison reports to Markdown."""

    def __init__(self) -> None:
        template_dir = files("khedron.reporting").joinpath("templates")
        self._environment = Environment(  # noqa: S701 - reports are Markdown, not HTML.
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
        )

    def generate_executive_summary(
        self,
        context: ReportContext,
        output_path: Path,
    ) -> None:
        self._render_to_path(
            "executive_summary.md.j2",
            output_path,
            {"context": context},
        )

    def generate_technical_deep_dive(
        self,
        context: ReportContext,
        output_path: Path,
    ) -> None:
        self._render_to_path(
            "technical_deep_dive.md.j2",
            output_path,
            {"context": context},
        )

    def generate_failure_analysis_report(
        self,
        context: ReportContext,
        output_path: Path,
    ) -> None:
        self._render_to_path(
            "failure_analysis.md.j2",
            output_path,
            {"context": context},
        )

    def generate_comparison_report(
        self,
        comparison: ComparisonReport,
        output_path: Path,
    ) -> None:
        self._render_to_path(
            "comparison.md.j2",
            output_path,
            {"comparison": comparison},
        )

    def render_comparison_report(self, comparison: ComparisonReport) -> str:
        """Render a comparison report to Markdown without writing to disk."""

        return self._render_to_string("comparison.md.j2", {"comparison": comparison})

    def _render_to_path(
        self,
        template_name: str,
        output_path: Path,
        variables: Mapping[str, object],
        *,
        generated_at: str | None = None,
    ) -> None:
        normalized = self._render_to_string(template_name, variables, generated_at=generated_at)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(normalized)

    def _render_to_string(
        self,
        template_name: str,
        variables: Mapping[str, object],
        *,
        generated_at: str | None = None,
    ) -> str:
        template = self._environment.get_template(template_name)
        render_variables = dict(variables)
        if generated_at is not None:
            render_variables["generated_at"] = generated_at
        rendered = template.render(**render_variables)
        return rendered.rstrip("\n") + "\n"


def _failed_questions(
    run_id: str,
    repository: RunRepository,
    questions: list[QuestionRecord],
) -> list[QuestionRecord]:
    failed: list[QuestionRecord] = []
    for record in questions:
        if record.verdict is JudgmentVerdict.CORRECT:
            continue
        try:
            failed.append(repository.get_question_record(run_id, record.question_id))
        except PersistenceError:
            failed.append(record)
    return failed


def _cost_summary(total_cost_usd: float, api_calls: list[APICallRecord]) -> CostSummary:
    cost_by_phase: dict[str, float] = {}
    cost_by_model: dict[str, float] = {}
    for record in api_calls:
        cost_by_phase[record.phase] = cost_by_phase.get(record.phase, 0.0) + record.cost_usd
        cost_by_model[record.model_id] = cost_by_model.get(record.model_id, 0.0) + record.cost_usd
    return CostSummary(
        total_cost_usd=total_cost_usd,
        cost_by_phase=cost_by_phase,
        cost_by_model=cost_by_model,
    )


def _ingestion_summary(records: Sequence[ConversationIngestionRecord]) -> IngestionSummary:
    """Total what reached the store, and locate what did not."""
    return IngestionSummary(
        n_turns=sum(record.n_turns for record in records),
        n_turns_failed=sum(record.n_turns_failed for record in records),
        failed_by_conversation={
            record.conversation_id: record.n_turns_failed
            for record in records
            if record.n_turns_failed
        },
    )


def _retrieval_budget_applied(run_started_event: RunStartedEvent) -> bool:
    """Read from the run record whether the provider honoured a retrieval budget."""
    value = run_started_event.runtime_environment.get("provider_honours_top_k")
    return bool(value) if isinstance(value, bool) else True


def _optional_runtime_int(
    runtime_environment: dict[str, object],
    key: str,
) -> int | None:
    """Read a recorded count, tolerating a run recorded before the field existed.

    Absent stays absent rather than defaulting: a run whose corpus coverage was never captured is
    unknown, and reporting unknown as complete is the failure this field exists to prevent.
    """
    value = runtime_environment.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_runtime_list(
    runtime_environment: dict[str, object],
    key: str,
) -> list[str] | None:
    """Read a recorded list of strings, tolerating a run that predates the field."""
    value = runtime_environment.get(key)
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        return None
    return [item for item in items if isinstance(item, str)]


def _optional_runtime_string(environment: dict[str, object], key: str) -> str | None:
    value = environment.get(key)
    return value if isinstance(value, str) and value else None
