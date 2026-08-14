from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from pathlib import Path

import pytest

from khedron.errors import KhedronError
from khedron.pacing import DispatchOutcome, RetryDisposition, paced_generate
from khedron.persistence.sqlite_indexer import SqliteIndexer
from khedron.pipeline._runtime import api_call_record_from_result, error_record_from_exception
from khedron.ratelimit import RateLimit, RateLimiter, pacer_name
from khedron.types import APICallResult

# The reserve/dispatch/disposition/retry cycle, verified through a real limiter's bucket state:
# settle updates the held tokens to the real usage, release removes the entry, and keep leaves it
# at the reservation estimate. Entry count and token value together distinguish the three, so these
# assert what actually happened to the account's window rather than that a method was called.


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


_KEY = ("model", pacer_name("openai", "m"))


def _fresh_limiter() -> RateLimiter:
    # A budget large enough that admission never blocks, so a test isolates disposition from
    # throttling; the fake clock keeps the cooldown assertions deterministic.
    return RateLimiter({_KEY: RateLimit(tokens_per_minute=10_000_000)}, clock=_Clock())


def _bucket(limiter: RateLimiter):  # type: ignore[no-untyped-def]
    return limiter._path(pacer_name("openai", "m"), None)[0]


def _result(input_tokens: int, output_tokens: int) -> APICallResult:
    return APICallResult(
        output="ok",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=1.0,
        cost_usd=0.0,
        model_id="m",
    )


def test_api_call_record_carries_cached_input_tokens_from_the_result() -> None:
    """The persisted audit record must copy cached_input_tokens, or the metric is silently lost."""
    result = APICallResult(
        output="ok",
        input_tokens=100,
        output_tokens=25,
        cached_input_tokens=64,
        latency_ms=1.0,
        cost_usd=0.0,
        model_id="m",
    )
    record = api_call_record_from_result(
        result=result, run_id="run-1", question_id="q-1", phase="generate", vendor="openai"
    )
    if record.cached_input_tokens != 64:
        raise AssertionError(record.cached_input_tokens)


async def _drive(
    limiter: RateLimiter | None,
    *,
    dispatch: Callable[[], Awaitable[object]],
    extract: Callable[[object], APICallResult],
    classify: Callable[[Exception], DispatchOutcome],
    max_retries: int,
    sleeps: list[float],
) -> APICallResult:
    async def backoff_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return await paced_generate(
        limiter=limiter,
        vendor="openai",
        model_id="m",
        scope=None,
        estimate=lambda: 100,
        max_retries=max_retries,
        dispatch=dispatch,
        extract=extract,
        classify=classify,
        backoff_sleep=backoff_sleep,
    )


def _terminal(_exc: Exception) -> DispatchOutcome:
    return DispatchOutcome(RetryDisposition.TERMINAL, lambda _n: RuntimeError("terminal"))


def test_a_successful_call_settles_the_hold_to_real_usage() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []

    async def dispatch() -> object:
        return object()

    async def scenario() -> None:
        result = await _drive(
            limiter,
            dispatch=dispatch,
            extract=lambda _r: _result(5, 3),
            classify=_terminal,
            max_retries=1,
            sleeps=sleeps,
        )
        if result.output != "ok" or sleeps:
            raise AssertionError((result, sleeps))
        entries = _bucket(limiter).entries
        if len(entries) != 1 or next(iter(entries.values())).tokens != 8:
            raise AssertionError("a success must settle the hold to the real 5+3 usage")

    asyncio.run(scenario())


def test_a_rate_limit_releases_the_hold_cools_down_and_reserves_again() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []
    calls = {"n": 0}

    async def dispatch() -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429")
        return object()

    def classify(_exc: Exception) -> DispatchOutcome:
        return DispatchOutcome(
            RetryDisposition.RATE_LIMITED, lambda _n: RuntimeError("rl"), retry_after=2.0
        )

    async def scenario() -> None:
        result = await _drive(
            limiter,
            dispatch=dispatch,
            extract=lambda _r: _result(5, 3),
            classify=classify,
            max_retries=1,
            sleeps=sleeps,
        )
        if result.output != "ok":
            raise AssertionError(result)
        bucket = _bucket(limiter)
        # Release leaves exactly one entry (the settled retry); keep would have left two.
        if len(bucket.entries) != 1 or next(iter(bucket.entries.values())).tokens != 8:
            raise AssertionError("a 429 must release the first hold and settle the retry")
        if bucket.cooldown_until <= 1000.0:
            raise AssertionError("a 429 must record a cooldown")
        if len(sleeps) != 1 or sleeps[0] < 2.0:
            raise AssertionError(f"the backoff must honour the 2.0 Retry-After floor, got {sleeps}")

    asyncio.run(scenario())


def test_a_terminal_error_keeps_the_hold() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []

    async def dispatch() -> object:
        raise RuntimeError("auth")

    def classify(_exc: Exception) -> DispatchOutcome:
        return DispatchOutcome(RetryDisposition.TERMINAL, lambda _n: ValueError("terminal"))

    async def scenario() -> None:
        try:
            await _drive(
                limiter,
                dispatch=dispatch,
                extract=lambda _r: _result(5, 3),
                classify=classify,
                max_retries=1,
                sleeps=sleeps,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("a terminal error must be raised")
        entries = _bucket(limiter).entries
        if len(entries) != 1 or next(iter(entries.values())).tokens != 100:
            raise AssertionError("a terminal error must keep the hold at the estimate")

    asyncio.run(scenario())


def test_a_retryable_error_keeps_its_hold_and_reserves_another() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []
    calls = {"n": 0}

    async def dispatch() -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("5xx")
        return object()

    def classify(_exc: Exception) -> DispatchOutcome:
        return DispatchOutcome(RetryDisposition.RETRYABLE, lambda _n: RuntimeError("retry"))

    async def scenario() -> None:
        await _drive(
            limiter,
            dispatch=dispatch,
            extract=lambda _r: _result(5, 3),
            classify=classify,
            max_retries=1,
            sleeps=sleeps,
        )
        # Two physical requests, both counted: the kept first hold plus the settled retry.
        if len(_bucket(limiter).entries) != 2:
            raise AssertionError("a retryable error must keep its hold and reserve again")

    asyncio.run(scenario())


def test_an_extraction_failure_after_a_response_keeps_the_hold() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []

    async def dispatch() -> object:
        return object()  # a real, charged response came back

    def extract(_response: object) -> APICallResult:
        raise ValueError("could not parse the response")

    async def scenario() -> None:
        try:
            await _drive(
                limiter,
                dispatch=dispatch,
                extract=extract,
                classify=_terminal,
                max_retries=0,
                sleeps=sleeps,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("the extraction failure must propagate")
        entries = _bucket(limiter).entries
        if len(entries) != 1 or next(iter(entries.values())).tokens != 100:
            raise AssertionError("an extraction failure after a real response must keep the hold")

    asyncio.run(scenario())


def test_cancellation_keeps_the_hold_and_propagates() -> None:
    limiter = _fresh_limiter()
    sleeps: list[float] = []

    async def dispatch() -> object:
        raise asyncio.CancelledError()

    async def scenario() -> None:
        try:
            await _drive(
                limiter,
                dispatch=dispatch,
                extract=lambda _r: _result(5, 3),
                classify=_terminal,
                max_retries=1,
                sleeps=sleeps,
            )
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation must propagate")
        entries = _bucket(limiter).entries
        if len(entries) != 1 or next(iter(entries.values())).tokens != 100:
            raise AssertionError("cancellation must keep the hold")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("disposition", "expected_retryable"),
    [
        (RetryDisposition.RETRYABLE, 1),
        (RetryDisposition.RATE_LIMITED, 1),
        (RetryDisposition.TERMINAL, 0),
    ],
)
def test_dispatch_retryability_flows_to_error_record_and_sqlite(
    tmp_path: Path,
    disposition: RetryDisposition,
    expected_retryable: int,
) -> None:
    """The adapter classification is the sole source of persisted retryability."""

    async def dispatch() -> object:
        raise RuntimeError("provider failed")

    def classify(_exc: Exception) -> DispatchOutcome:
        return DispatchOutcome(disposition, lambda _attempt: KhedronError("surface failure"))

    async def scenario() -> KhedronError:
        try:
            await _drive(
                None,
                dispatch=dispatch,
                extract=lambda _response: _result(0, 0),
                classify=classify,
                max_retries=0,
                sleeps=[],
            )
        except KhedronError as exc:
            return exc
        raise AssertionError("paced_generate did not surface the dispatch error")

    dispatch_error = asyncio.run(scenario())
    record = error_record_from_exception(
        exc=dispatch_error,
        run_id="run-1",
        phase="generate",
        question_id="q-1",
        recovered=False,
    )
    if record.retryable is not (expected_retryable == 1):
        raise AssertionError(record.retryable)
    non_dispatch_record = error_record_from_exception(
        exc=RuntimeError("local failure"),
        run_id="run-1",
        phase="generate",
        question_id="q-2",
        recovered=False,
    )
    if non_dispatch_record.retryable is not None:
        raise AssertionError(non_dispatch_record.retryable)

    db_path = tmp_path / "benchmark.db"
    indexer = SqliteIndexer(db_path)
    indexer.initialize()
    with closing(sqlite3.connect(db_path)) as connection:
        indexer._insert_errors(connection, [record, non_dispatch_record])
        connection.commit()
        values = {
            row[0]: row[1]
            for row in connection.execute("SELECT question_id, retryable FROM error_log")
        }
    if values != {"q-1": expected_retryable, "q-2": None}:
        raise AssertionError(values)


def test_with_no_limiter_the_call_is_unpaced() -> None:
    sleeps: list[float] = []

    async def dispatch() -> object:
        return object()

    async def scenario() -> None:
        result = await _drive(
            None,
            dispatch=dispatch,
            extract=lambda _r: _result(5, 3),
            classify=_terminal,
            max_retries=1,
            sleeps=sleeps,
        )
        if result.output != "ok":
            raise AssertionError("with no limiter the call must behave exactly as before")

    asyncio.run(scenario())
