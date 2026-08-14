from __future__ import annotations

from khedron.errors import CostExceededError, JudgeError, KhedronError
from khedron.judges.base import Judge
from khedron.pipeline._runtime import (
    PipelineRepository,
    RunContext,
    api_call_record_from_result,
    error_record_from_exception,
)
from khedron.types import Judgment, Question, Response
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc

__all__ = ["judge_response"]


async def judge_response(
    response: Response,
    question: Question,
    judge: Judge,
    repository: PipelineRepository,
    run_context: RunContext,
    *,
    recovery_attempt_id: str | None = None,
) -> Judgment:
    """Judge a generated response, persist audit/cost records, and return the judgment."""

    try:
        result = await judge.evaluate(
            question=question.question_text,
            expected_answer=question.expected_answer,
            generated_answer=response.answer_text,
            category=question.category.value,
        )
    except CostExceededError:
        # A pre-dispatch budget refusal on the judge's paid call: nothing was billed. Propagate it
        # unwrapped so it halts the run cleanly, not demoted to a per-question JudgeError that
        # would let the run keep spending.
        raise
    except Exception as exc:
        project_exc = _judge_exception(
            exc,
            "Judge evaluation failed",
            run_id=run_context.run_id,
            question_id=question.question_id,
            response_id=response.response_id,
        )
        await repository.append_error(
            error_record_from_exception(
                exc=project_exc,
                run_id=run_context.run_id,
                phase="judge",
                question_id=question.question_id,
                recovered=False,
                context={"response_id": response.response_id, "retryable": _retryable(project_exc)},
            )
        )
        raise project_exc from exc

    api_call = api_call_record_from_result(
        result=result.api_call,
        run_id=run_context.run_id,
        question_id=question.question_id,
        phase="judge",
        vendor=judge.vendor,
    )
    await repository.append_api_call(api_call)
    run_context.cost_tracker.record(api_call)

    judgment = Judgment(
        judgment_id=generate_ulid(),
        run_id=run_context.run_id,
        response_id=response.response_id,
        question_id=question.question_id,
        timestamp=now_utc(),
        judge_model_id=result.api_call.model_id,
        prompt=_raw_string(result.api_call.raw_response.get("prompt")),
        raw_judge_output=result.api_call.output,
        parsed_verdict=result.verdict,
        parsed_score=result.score,
        parsed_reasoning=result.reasoning,
        parse_was_successful=True,
        input_tokens=result.api_call.input_tokens,
        output_tokens=result.api_call.output_tokens,
        latency_ms=result.api_call.latency_ms,
        cost_usd=result.api_call.cost_usd,
        recovery_attempt_id=recovery_attempt_id,
    )
    await repository.append_judgment(judgment)
    return judgment


def _raw_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _judge_exception(exc: Exception, message: str, **context: object) -> JudgeError:
    if isinstance(exc, JudgeError):
        return exc
    if isinstance(exc, KhedronError):
        return JudgeError(message, **exc.context, **context)
    return JudgeError(message, **context)


def _retryable(exc: KhedronError) -> bool | None:
    value = exc.context.get("retryable")
    return value if isinstance(value, bool) else None
