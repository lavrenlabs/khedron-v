from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import structlog

from khedron.build import resolve_framework_version
from khedron.cost.tracker import CostTracker
from khedron.errors import ConfigurationError, JudgeError
from khedron.judges.base import Judge
from khedron.methodology import (
    METHODOLOGY_FINGERPRINT_KEY,
    MethodologyRuntimeProfile,
    get_runtime_profile,
    methodology_fingerprint,
)
from khedron.persistence.repository import RunRepository
from khedron.pipeline.judger import judge_response
from khedron.types import (
    Question,
    QuestionEvaluationRecord,
    QuestionRecord,
    Response,
    RetrievalRecord,
    RunCompletedEvent,
    RunStartedEvent,
)
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc

__all__ = ["RejudgeResult", "rejudge_run"]

_LOGGER = structlog.get_logger(__name__)


@dataclass
class _RejudgeContext:
    """The two fields the pipeline's `RunContext` protocol requires, and nothing else."""

    run_id: str
    cost_tracker: CostTracker


@dataclass(frozen=True)
class RejudgeResult:
    """Identifiers and totals of a rejudged run."""

    run_id: str
    source_run_id: str
    n_questions: int
    cost_usd: float


async def rejudge_run(
    *,
    source_run_id: str,
    profile_name: str,
    judge: Judge,
    repository: RunRepository,
    max_cost_usd: float | None = None,
) -> RejudgeResult:
    """Score an existing run's stored answers with a different judge, as a new derived run.

    Why this exists. Comparing two judges by running the benchmark twice confounds what you are
    measuring: the answers are regenerated too, so a difference between the two scores mixes the
    judges' disagreement with the answer model's own variability. Re-scoring the *stored* answers
    removes that entirely -- the graded text is byte-identical, so every difference is the judge.

    That is what makes a same-vendor grading bias measurable rather than merely arguable, and it
    costs the price of the judge calls alone: on the order of a dollar for a full corpus whose
    answers already exist.

    The result is a new run, not an annotation on the old one. It carries its own methodology
    profile -- a profile pins the judge, so a rejudged run belongs to a different profile with a
    different fingerprint by construction -- and its `runtime_environment` records the run it was
    derived from and that no answer was regenerated. The source run is never modified.

    The evaluations, retrievals and responses are copied into the derived run rather than
    referenced, so it is self-contained and a reader can diff the two answer sets and see for
    themselves that only the grading changed.
    """
    profile = get_runtime_profile(profile_name)
    source_started = repository.get_run_started_event(source_run_id)
    source_questions = repository.get_questions_for_run(source_run_id)
    _require_judge_matches_profile(profile, judge, profile_name)
    _require_answers_are_reusable(profile, source_started, source_run_id)

    run_id = generate_ulid()
    sequence = 0
    tracker = CostTracker()
    context = _RejudgeContext(run_id=run_id, cost_tracker=tracker)

    await repository.append_run_event(
        _derived_run_started(
            run_id=run_id,
            sequence=sequence,
            profile=profile,
            source=source_started,
            judge=judge,
        )
    )
    sequence += 1

    # Indexed once. Scanning every response and retrieval per question would be quadratic, and
    # the runs this exists for carry 1,540 of each.
    responses_by_question = {
        response.question_id: response
        for response in repository.get_responses_for_run(source_run_id)
    }
    retrievals_by_question = {
        retrieval.question_id: retrieval
        for retrieval in repository.get_retrievals_for_run(source_run_id)
    }

    n_judged = 0
    for question_record in source_questions:
        response = responses_by_question.get(question_record.question_id)
        if response is None:
            # A question the source run never answered has nothing to re-grade. Skipping it keeps
            # the derived run a strict re-scoring of the same answers; inventing a verdict for it
            # would make the two runs disagree on a question neither judge ever saw.
            continue
        # Minted once and reused: the response persisted into the derived run and the one the judge
        # scores must be the same record, or the judgment references an answer that is not there.
        derived_response = await _copy_question_inputs(
            repository=repository,
            run_id=run_id,
            question_record=question_record,
            response=response,
            retrieval=retrievals_by_question.get(question_record.question_id),
        )
        try:
            await judge_response(
                derived_response,
                _question_from_record(question_record),
                judge,
                repository,
                context,
            )
        except JudgeError:
            # Persisted as an error by the pipeline, exactly as in a normal run.
            pass
        n_judged += 1
        if max_cost_usd is not None and tracker.total_cost_usd() > max_cost_usd:
            raise ConfigurationError(
                "Rejudging exceeded its cost cap.",
                run_id=run_id,
                source_run_id=source_run_id,
                spent_usd=tracker.total_cost_usd(),
                max_cost_usd=max_cost_usd,
            )

    records = repository.get_questions_for_run(run_id)
    await repository.append_run_event(
        _derived_run_completed(
            run_id=run_id,
            sequence=sequence,
            records=records,
            scored_categories=(
                list(profile.scored_categories) if profile.scored_categories is not None else None
            ),
            cost_usd=tracker.total_cost_usd(),
        )
    )
    _LOGGER.info(
        "run_rejudged",
        run_id=run_id,
        source_run_id=source_run_id,
        methodology_profile=profile_name,
        n_questions=n_judged,
        cost_usd=round(tracker.total_cost_usd(), 6),
    )
    return RejudgeResult(
        run_id=run_id,
        source_run_id=source_run_id,
        n_questions=n_judged,
        cost_usd=tracker.total_cost_usd(),
    )


def _require_judge_matches_profile(
    profile: MethodologyRuntimeProfile,
    judge: Judge,
    profile_name: str,
) -> None:
    """Refuse a judge the target profile does not name.

    The whole value of a rejudged run is that its artifact states which judge produced it. A run
    recorded under a profile pinning one judge but graded by another asserts something false about
    itself, which is the defect class this methodology version exists to remove.
    """
    if profile.judge_model_id is not None and judge.model_id != profile.judge_model_id:
        raise ConfigurationError(
            "Judge does not match the model the target profile pins.",
            methodology_profile=profile_name,
            expected=profile.judge_model_id,
            observed=judge.model_id,
        )


# Everything about a profile that shaped the answers rather than the grading. A rejudged run
# inherits the answers and stamps the target profile's fingerprint, so if the target differs on any
# of these the artifact declares a generation that never happened -- a corpus it did not ingest, a
# prompt it was not asked with, a retrieval budget it did not use.
_GENERATION_AFFECTING_FIELDS: Final[tuple[str, ...]] = (
    "answer_model_id",
    "answer_model_type",
    "generator_prompt_path",
    "answer_marker",
    "image_description_policy",
    "top_k_retrieval",
    "evaluation_categories",
    "benchmark_type",
)


def _require_answers_are_reusable(
    profile: MethodologyRuntimeProfile,
    source: RunStartedEvent,
    source_run_id: str,
) -> None:
    """Refuse a target profile that differs from the source on anything but the judge.

    Rejudging holds the answers fixed and varies the grading. Every other field is a claim the
    derived run would make about a generation it did not perform: its fingerprint is the target
    profile's, so a target pinning a different corpus policy would have the artifact declare image
    descriptions it never ingested, and a different generator prompt would have it declare a
    question it was never asked.

    Checked over the whole set rather than the answer model alone. The narrow version admitted
    exactly the case that prompted this: re-scoring answers generated without image descriptions
    under a profile whose policy says they were ingested -- an artifact asserting something false
    about itself, which is the defect class this project exists to remove.
    """
    try:
        source_profile = get_runtime_profile(source.methodology_profile)
    except ConfigurationError:
        # An archived or unknown source profile cannot be compared field by field. The answer model
        # is recorded on the run itself, so that check still binds; the rest cannot be established
        # and the run is refused rather than assumed compatible.
        if (
            profile.answer_model_id is not None
            and source.answer_model_id != profile.answer_model_id
        ):
            raise
        raise ConfigurationError(
            "The source run's methodology profile cannot be resolved, so whether the target "
            "profile differs from it on anything but the judge cannot be established.",
            source_run_id=source_run_id,
            source_profile=source.methodology_profile,
        ) from None

    differing = [
        name
        for name in _GENERATION_AFFECTING_FIELDS
        if getattr(source_profile, name, None) != getattr(profile, name, None)
    ]
    if differing:
        raise ConfigurationError(
            "The target profile differs from the source on fields that shaped the answers, not the "
            "grading, so the derived run would declare a generation it never performed.",
            source_run_id=source_run_id,
            source_profile=source_profile.name,
            target_profile=profile.name,
            differing_fields=differing,
        )


async def _copy_question_inputs(
    *,
    repository: RunRepository,
    run_id: str,
    question_record: QuestionRecord,
    response: Response,
    retrieval: RetrievalRecord | None,
) -> Response:
    """Copy the evaluation, retrieval and answer into the derived run, unchanged but re-identified.

    Copied rather than referenced so the derived run stands on its own: its report reads the same
    fields as any other run's, and a reader can diff the two answer sets to confirm that only the
    grading differs.
    """
    question_evaluation_id = generate_ulid()
    await repository.append_question_evaluation(
        QuestionEvaluationRecord(
            question_evaluation_id=question_evaluation_id,
            run_id=run_id,
            question_id=question_record.question_id,
            conversation_id=question_record.conversation_id,
            category=question_record.category,
            question_text=question_record.question_text,
            expected_answer=question_record.expected_answer,
            is_audited_error=question_record.is_audited_error,
            timestamp=now_utc(),
        )
    )
    retrieval_id = generate_ulid()
    if retrieval is not None:
        await repository.append_retrieval(
            retrieval.model_copy(
                update={
                    "run_id": run_id,
                    "retrieval_id": retrieval_id,
                    # Repointed at the copy, not left pointing at the source run's evaluation:
                    # the index enforces that reference, and a dangling one made every insert
                    # fail into a best-effort warning that the derived run then carried silently.
                    "question_evaluation_id": question_evaluation_id,
                }
            )
        )
    derived_response = _reidentified(response, run_id, retrieval_id)
    await repository.append_response(derived_response)
    return derived_response


def _reidentified(response: Response, run_id: str, retrieval_id: str | None = None) -> Response:
    """Move a stored answer into the derived run, keeping its references inside that run.

    `retrieval_id` matters as much as `run_id`. Copying the retrieval under a new identifier while
    leaving the answer pointing at the source run's makes the derived run reference a record it
    does not contain -- the self-containment this copying exists to provide, silently absent.
    """
    # A new identity, not just a new home. `response_id` is a global lookup key -- `get_response`
    # resolves it with `SELECT run_id FROM question_results WHERE response_id = ?` -- so reusing the
    # source's made one identifier match two rows, and a lookup could hydrate the wrong run's
    # answer. The provenance is kept as data instead.
    update: dict[str, str] = {"run_id": run_id, "response_id": generate_ulid()}
    if retrieval_id is not None:
        update["retrieval_id"] = retrieval_id
    return response.model_copy(update=update)


def _question_from_record(record: QuestionRecord) -> Question:
    return Question(
        question_id=record.question_id,
        conversation_id=record.conversation_id,
        category=record.category,
        question_text=record.question_text,
        expected_answer=record.expected_answer,
        is_audited_error=record.is_audited_error,
    )


def _derived_run_started(
    *,
    run_id: str,
    sequence: int,
    profile: MethodologyRuntimeProfile,
    source: RunStartedEvent,
    judge: Judge,
) -> RunStartedEvent:
    runtime_environment: dict[str, Any] = {
        **source.runtime_environment,
        "methodology_profile_status": profile.status,
        # Its own, under the key every reader uses. It previously wrote a second spelling, so the
        # inherited source fingerprint stayed in place and the report disclosed the wrong run.
        METHODOLOGY_FINGERPRINT_KEY: methodology_fingerprint(profile),
        # The two facts that stop this artifact being mistaken for an independent measurement:
        # its answers were produced elsewhere, and it re-scored them rather than regenerating them.
        "derived_from_run_id": source.run_id,
        "answers_regenerated": False,
        "methodology_scored_categories": (
            list(profile.scored_categories) if profile.scored_categories is not None else None
        ),
    }
    return source.model_copy(
        update={
            "event_id": generate_ulid(),
            "timestamp": now_utc(),
            "run_id": run_id,
            "sequence_number": sequence,
            "experiment_name": f"{source.experiment_name} (rejudged: {profile.name})",
            "judge_model_id": judge.model_id,
            "judge_model_vendor": judge.vendor,
            "methodology_profile": profile.name,
            "framework_version": resolve_framework_version(),
            "runtime_environment": runtime_environment,
        }
    )


def _derived_run_completed(
    *,
    run_id: str,
    sequence: int,
    records: list[QuestionRecord],
    scored_categories: list[str] | None,
    cost_usd: float,
) -> RunCompletedEvent:
    from khedron.analysis.scorer import Scorer

    scores = Scorer().score_run(records, scored_categories=scored_categories)
    n_errored = sum(1 for record in records if record.verdict.value == "error")
    return RunCompletedEvent(
        event_id=generate_ulid(),
        timestamp=now_utc(),
        run_id=run_id,
        sequence_number=sequence,
        status="completed" if n_errored == 0 else "partial",
        n_questions_attempted=len(records),
        n_questions_succeeded=len(records) - n_errored,
        n_questions_errored=n_errored,
        overall_score_standard=scores.overall_standard,
        overall_score_audited=scores.overall_audited,
        by_category_standard={
            key.value: value for key, value in scores.by_category_standard.items()
        },
        by_category_audited={key.value: value for key, value in scores.by_category_audited.items()},
        total_cost_usd=cost_usd,
    )
