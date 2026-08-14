from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime
from time import perf_counter
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from khedron.cost.tracker import CostTracker
from khedron.errors import KhedronError
from khedron.types import (
    APICallRecord,
    APICallResult,
    ConversationIngestionRecord,
    ErrorRecord,
    Judgment,
    Response,
    RetrievalRecord,
)
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc

_ResultT = TypeVar("_ResultT")
_Phase = Literal["initialization", "ingest", "retrieve", "generate", "judge", "analysis"]
_ApiPhase = Literal["ingest", "retrieve", "generate", "judge"]


class RunContext(Protocol):
    run_id: str
    cost_tracker: CostTracker


class PipelineRepository(Protocol):
    async def append_conversation_ingestion(
        self,
        record: ConversationIngestionRecord,
    ) -> None: ...

    async def append_retrieval(self, retrieval: RetrievalRecord) -> None: ...

    async def append_response(self, response: Response) -> None: ...

    async def append_judgment(self, judgment: Judgment) -> None: ...

    async def append_api_call(self, record: APICallRecord) -> None: ...

    async def append_error(self, error: ErrorRecord) -> None: ...


class IngestionStats(BaseModel):
    """Private runtime summary returned by the ingestion stage."""

    model_config = ConfigDict(frozen=True)

    record: ConversationIngestionRecord
    memory_ids: list[str] = Field(default_factory=list)

    @property
    def n_turns_succeeded(self) -> int:
        return self.record.n_turns_succeeded

    @property
    def n_turns_failed(self) -> int:
        return self.record.n_turns_failed

    @property
    def total_latency_ms(self) -> float:
        return self.record.total_latency_ms


def elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000.0)


async def time_async(operation: Callable[[], Awaitable[_ResultT]]) -> tuple[_ResultT, float]:
    started = perf_counter()
    result = await operation()
    return result, elapsed_ms(started)


def api_call_record_from_result(
    *,
    result: APICallResult,
    run_id: str,
    question_id: str | None,
    phase: _ApiPhase,
    vendor: str,
    timestamp: datetime | None = None,
) -> APICallRecord:
    return APICallRecord(
        api_call_id=generate_ulid(),
        run_id=run_id,
        question_id=question_id,
        timestamp=timestamp or now_utc(),
        phase=phase,
        vendor=vendor,
        model_id=result.model_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        status="success",
        attempt_number=1,
    )


def error_record_from_exception(
    *,
    exc: BaseException,
    run_id: str,
    phase: _Phase,
    question_id: str | None,
    recovered: bool,
    context: dict[str, Any] | None = None,
) -> ErrorRecord:
    merged_context: dict[str, Any] = {}
    if isinstance(exc, KhedronError):
        # The redacted view, not `context`: this record is written to errors.jsonl, so copying
        # the raw context persisted the values that the error_message beside it had removed.
        merged_context.update(exc.redacted_context)
    if context is not None:
        merged_context.update(context)
    retryable = merged_context.get("retryable")
    return ErrorRecord(
        error_id=generate_ulid(),
        run_id=run_id,
        timestamp=now_utc(),
        phase=phase,
        question_id=question_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        stack_trace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        context=merged_context,
        recovered=recovered,
        retryable=retryable if isinstance(retryable, bool) else None,
    )
