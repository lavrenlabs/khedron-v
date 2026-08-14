from __future__ import annotations

from time import perf_counter
from typing import Any

import structlog

from khedron.errors import KhedronError, ProviderError
from khedron.pipeline._runtime import (
    IngestionStats,
    PipelineRepository,
    RunContext,
    elapsed_ms,
    error_record_from_exception,
)
from khedron.providers.base import MemoryProvider
from khedron.types import Conversation, ConversationIngestionRecord, Session, Turn
from khedron.utils.time import now_utc

__all__ = ["ingest_conversation"]

_INGESTION_FAILURE_THRESHOLD = 0.05
# Few enough that a verification search costs nothing next to the add it protects; the turn_id
# filter decides the match, so extra candidates only widen the net.
_RECOVERY_SEARCH_TOP_K = 5
_LOGGER = structlog.get_logger(__name__)


async def ingest_conversation(
    conversation: Conversation,
    provider: MemoryProvider,
    repository: PipelineRepository,
    run_context: RunContext,
    *,
    ingestion_attempt_id: str | None = None,
) -> IngestionStats:
    """Ingest one conversation into the memory provider and persist its summary."""

    started_at = now_utc()
    timer_started_at = perf_counter()
    ordered_sessions = sorted(conversation.sessions, key=lambda session: session.session_number)
    total_turns = sum(len(session.turns) for session in ordered_sessions)
    succeeded = 0
    failed = 0
    memory_ids: list[str] = []
    error_summary: list[dict[str, Any]] = []

    for session in ordered_sessions:
        for turn in _ordered_turns(session):
            try:
                memory_id = await _add_turn_once_only(provider, conversation, session, turn)
            except Exception as exc:
                failed += 1
                error_summary.append(_turn_error_summary(session, turn, exc))
                project_exc = _provider_exception(
                    exc,
                    "Memory provider failed to ingest conversation turn",
                    conversation_id=conversation.conversation_id,
                    session_id=session.session_id,
                    turn_id=turn.turn_id,
                )
                await repository.append_error(
                    error_record_from_exception(
                        exc=project_exc,
                        run_id=run_context.run_id,
                        phase="ingest",
                        question_id=None,
                        recovered=True,
                        context={
                            "conversation_id": conversation.conversation_id,
                            "session_id": session.session_id,
                            "turn_id": turn.turn_id,
                        },
                    )
                )
                _LOGGER.warning(
                    "conversation_turn_ingestion_failed",
                    run_id=run_context.run_id,
                    conversation_id=conversation.conversation_id,
                    session_id=session.session_id,
                    turn_id=turn.turn_id,
                    error=str(exc),
                )
                if _is_unreliable(failed, total_turns):
                    break
                continue
            succeeded += 1
            memory_ids.append(memory_id)
        if _is_unreliable(failed, total_turns):
            break

    total_latency_ms = elapsed_ms(timer_started_at)
    record = ConversationIngestionRecord(
        run_id=run_context.run_id,
        conversation_id=conversation.conversation_id,
        started_at=started_at,
        finished_at=now_utc(),
        n_sessions=len(ordered_sessions),
        n_turns=total_turns,
        n_turns_succeeded=succeeded,
        n_turns_failed=failed,
        total_latency_ms=total_latency_ms,
        avg_latency_per_turn_ms=total_latency_ms / max(1, succeeded + failed),
        error_summary=error_summary,
        ingestion_attempt_id=ingestion_attempt_id,
    )
    await repository.append_conversation_ingestion(record)

    if _is_unreliable(failed, total_turns):
        raise ProviderError(
            "Conversation ingestion is unreliable",
            run_id=run_context.run_id,
            conversation_id=conversation.conversation_id,
            n_turns=total_turns,
            n_turns_failed=failed,
            failure_rate=failed / max(1, total_turns),
        )

    return IngestionStats(record=record, memory_ids=memory_ids)


async def _add_turn_once_only(
    provider: MemoryProvider,
    conversation: Conversation,
    session: Session,
    turn: Turn,
) -> str:
    """Ingest one turn, recovering from a lost reply without ever writing it twice.

    Why this is not a plain retry. The measured failure was a 60-second timeout on the provider's
    `tools/call`, and a timeout says nothing about whether the write happened: the provider may
    have persisted the memory and lost the reply on the way back. Retrying blind would duplicate
    the turn, and a duplicated turn silently changes what retrieval sees -- a worse outcome than
    the lost turn it was trying to repair, because nothing downstream would report it.

    So the failure path asks the provider whether the turn is already there, keyed on the turn's
    own identifier, and only re-adds when it is demonstrably absent. If that question cannot be
    answered -- the verifying search times out too -- the original error is raised unchanged. A
    lost turn is counted honestly and bounded by `max_ingestion_failure_rate`; a duplicate would
    not be counted at all.
    """
    metadata = _turn_metadata(conversation, session, turn)
    try:
        return await provider.add(turn.content, metadata=metadata)
    except Exception as first_error:
        try:
            existing = await _find_ingested_turn(provider, turn, metadata)
        except Exception:
            # Cannot establish whether the write landed. Never risk the duplicate.
            raise first_error from None
        if existing is not None:
            _LOGGER.info(
                "conversation_turn_ingestion_recovered",
                conversation_id=conversation.conversation_id,
                session_id=session.session_id,
                turn_id=turn.turn_id,
                detail="the provider had persisted the turn before the reply was lost",
            )
            return existing
        return await provider.add(turn.content, metadata=metadata)


async def _find_ingested_turn(
    provider: MemoryProvider,
    turn: Turn,
    metadata: dict[str, Any],
) -> str | None:
    """Return the memory id of an already-ingested turn, or None if it is absent.

    Queries with the turn's own text so an exact copy ranks first, then confirms `turn_id` on the
    returned memory: the identifier is what decides, the similarity only narrows the candidates.

    The identifier is re-checked here rather than trusted to `metadata_filter`, because filtering
    is the provider's own business -- `FullContextProvider` ignores it entirely, and a provider
    that ignores it would report every turn as already present, turning each real loss into a
    silent false recovery. The filter is still passed so a provider that can push it down does.
    """
    del metadata
    matches = await provider.search(
        turn.content,
        top_k=_RECOVERY_SEARCH_TOP_K,
        metadata_filter={"turn_id": turn.turn_id},
    )
    for memory in matches:
        if (memory.metadata or {}).get("turn_id") == turn.turn_id:
            return memory.memory_id
    return None


def _ordered_turns(session: Session) -> list[Turn]:
    return list(session.turns)


def _turn_metadata(
    conversation: Conversation,
    session: Session,
    turn: Turn,
) -> dict[str, Any]:
    return {
        "speaker": turn.speaker,
        "session_id": session.session_id,
        "session_number": session.session_number,
        "session_timestamp": session.timestamp.isoformat(),
        "turn_id": turn.turn_id,
        "conversation_id": conversation.conversation_id,
    }


def _turn_error_summary(session: Session, turn: Turn, exc: BaseException) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "session_number": session.session_number,
        "turn_id": turn.turn_id,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _is_unreliable(failed: int, total: int) -> bool:
    return total > 0 and failed / total >= _INGESTION_FAILURE_THRESHOLD


def _provider_exception(exc: Exception, message: str, **context: Any) -> ProviderError:
    """Wrap a provider failure without discarding what actually went wrong.

    The last branch used to return a bare `ProviderError(message)`, so any exception that was not
    already a Khedron error lost its type and its message entirely. The persisted record then said
    only "Memory provider operation failed", and the stack trace showed the wrapper rather than the
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
