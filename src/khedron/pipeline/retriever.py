from __future__ import annotations

from khedron.errors import KhedronError, ProviderError
from khedron.pipeline._runtime import (
    PipelineRepository,
    RunContext,
    error_record_from_exception,
    time_async,
)
from khedron.providers.base import MemoryProvider
from khedron.types import Question, QuestionEvaluationRecord, RetrievalRecord
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc

__all__ = ["retrieve_for_question"]


async def retrieve_for_question(
    question: Question,
    question_evaluation: QuestionEvaluationRecord,
    provider: MemoryProvider,
    top_k: int,
    repository: PipelineRepository,
    run_context: RunContext,
    *,
    recovery_attempt_id: str | None = None,
    ingestion_attempt_id: str | None = None,
) -> RetrievalRecord:
    """Retrieve memories for a question, persist the retrieval, and return it."""

    query = question.question_text
    context = {
        "question_evaluation_id": question_evaluation.question_evaluation_id,
        "question_id": question.question_id,
        "conversation_id": question.conversation_id,
    }
    try:
        memories, latency_ms = await time_async(lambda: provider.search(query, top_k=top_k))
    except Exception as exc:
        project_exc = _provider_exception(exc, "Memory provider retrieval failed", **context)
        await repository.append_error(
            error_record_from_exception(
                exc=project_exc,
                run_id=run_context.run_id,
                phase="retrieve",
                question_id=question.question_id,
                recovered=False,
                context={**context, "retryable": _retryable(project_exc)},
            )
        )
        raise project_exc from exc

    if top_k > 0 and not memories:
        # A search that succeeds but returns nothing is not an exception, so it used to
        # leave no trace at all: the model was handed "No memories retrieved", refused,
        # and the run still reported a clean bill of health. Record it as recovered so it
        # is countable evidence without failing a question the provider answered lawfully.
        # Guarded on top_k: a suite may legitimately request zero memories, and an empty
        # result is then the correct outcome, not a provider fault to log once per question.
        await repository.append_error(
            error_record_from_exception(
                exc=ProviderError("Memory provider returned no memories", **context),
                run_id=run_context.run_id,
                phase="retrieve",
                question_id=question.question_id,
                recovered=True,
                context={**context, "top_k": top_k},
            )
        )

    retrieval = RetrievalRecord(
        retrieval_id=generate_ulid(),
        question_evaluation_id=question_evaluation.question_evaluation_id,
        run_id=run_context.run_id,
        question_id=question.question_id,
        timestamp=now_utc(),
        query=query,
        top_k=top_k,
        n_returned=len(memories),
        memories=memories,
        retrieval_latency_ms=latency_ms,
        recovery_attempt_id=recovery_attempt_id,
        ingestion_attempt_id=ingestion_attempt_id,
    )
    await repository.append_retrieval(retrieval)
    return retrieval


def _provider_exception(exc: Exception, message: str, **context: object) -> ProviderError:
    """Wrap a provider failure without discarding what actually went wrong.

    The last branch used to return a bare `ProviderError(message)`, so any exception that was not
    already a Khedron error lost its type and its message entirely. The persisted record then said
    only "Memory provider retrieval failed", and the stack trace showed the wrapper rather than the
    cause -- which is precisely the information an operator needs at the moment a run dies. A cost
    probe hit exactly this and could not be diagnosed from its own artifacts.

    The raise site chains with `from exc`, so the traceback is intact in-process; this carries the
    cause into the *persisted* record, which is what survives the run.
    """
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, KhedronError):
        return ProviderError(message, **exc.context, **context)
    return ProviderError(
        message,
        **context,
        cause_type=type(exc).__name__,
        cause=str(exc) or repr(exc),
    )


def _retryable(exc: KhedronError) -> bool | None:
    value = exc.context.get("retryable")
    return value if isinstance(value, bool) else None
