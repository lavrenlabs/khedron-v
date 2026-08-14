from __future__ import annotations

from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Any, Final

from khedron.errors import CostExceededError, KhedronError, ModelError
from khedron.models.base import AnswerModel
from khedron.pipeline._runtime import (
    PipelineRepository,
    RunContext,
    api_call_record_from_result,
    error_record_from_exception,
)
from khedron.types import Memory, Question, Response, RetrievalRecord
from khedron.utils.ids import generate_ulid
from khedron.utils.time import now_utc

__all__ = ["CANONICAL_GENERATOR_PROMPT_PATH", "build_generator_prompt", "generate_answer"]

CANONICAL_GENERATOR_PROMPT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "prompts" / "generator_canonical_v1.txt"
)


async def generate_answer(
    question: Question,
    retrieval: RetrievalRecord,
    answer_model: AnswerModel,
    repository: PipelineRepository,
    run_context: RunContext,
    *,
    max_output_tokens: int = 1024,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    answer_marker: str | None = None,
    recovery_attempt_id: str | None = None,
) -> Response:
    """Generate, audit, cost-track, persist, and return an answer response.

    `answer_marker` names the token a prompt requires its final answer to follow. A prompt that asks
    the model to reason in steps and then commit after `ANSWER:` produces an answer that is the text
    after that marker; scoring the whole transcript instead scores the reasoning, which is a
    different thing and usually a more generous one.
    """

    prompt = build_generator_prompt(question, retrieval.memories, template=prompt_template)
    try:
        api_result = await answer_model.generate(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    except CostExceededError:
        # A pre-dispatch budget refusal: reserving this attempt's worst-case cost would breach the
        # cap, so nothing was billed. Propagate it unwrapped -- wrapping into a ModelError would
        # demote a clean, run-halting budget stop into a per-question error and keep the run
        # spending. The run/suite halt path records it; a refused call needs no error record of its
        # own.
        raise
    except Exception as exc:
        project_exc = _model_exception(
            exc,
            "Answer model generation failed",
            run_id=run_context.run_id,
            question_id=question.question_id,
            retrieval_id=retrieval.retrieval_id,
        )
        await repository.append_error(
            error_record_from_exception(
                exc=project_exc,
                run_id=run_context.run_id,
                phase="generate",
                question_id=question.question_id,
                recovered=False,
                context={
                    "retrieval_id": retrieval.retrieval_id,
                    "retryable": _retryable(project_exc),
                },
            )
        )
        raise project_exc from exc

    api_call = api_call_record_from_result(
        result=api_result,
        run_id=run_context.run_id,
        question_id=question.question_id,
        phase="generate",
        vendor=answer_model.vendor,
    )
    await repository.append_api_call(api_call)
    run_context.cost_tracker.record(api_call)

    answer_text, extraction_failed = _extract_answer(api_result.output, answer_marker)
    if extraction_failed:
        # The prompt required a marker and the output never reached it -- almost always because the
        # token budget ended the response mid-reasoning. Raised rather than scored: a truncated
        # transcript is not an answer, and a lenient judge will award it credit for the correct
        # facts it mentions on the way to a conclusion it never wrote -- silently grading truncated
        # transcripts as answers when none of them actually produced one.
        project_exc = ModelError(
            "Answer never reached the marker its prompt requires, so no answer was produced. "
            "Raise max_output_tokens for this profile.",
            run_id=run_context.run_id,
            question_id=question.question_id,
            answer_marker=answer_marker,
            output_tokens=api_result.output_tokens,
            max_output_tokens=max_output_tokens,
        )
        await repository.append_error(
            error_record_from_exception(
                exc=project_exc,
                run_id=run_context.run_id,
                phase="generate",
                question_id=question.question_id,
                recovered=False,
                context={
                    "retrieval_id": retrieval.retrieval_id,
                    "retryable": _retryable(project_exc),
                },
            )
        )
        raise project_exc

    response = Response(
        response_id=generate_ulid(),
        run_id=run_context.run_id,
        question_id=question.question_id,
        retrieval_id=retrieval.retrieval_id,
        timestamp=now_utc(),
        model_id=api_result.model_id,
        prompt=prompt,
        answer_text=answer_text,
        input_tokens=api_result.input_tokens,
        output_tokens=api_result.output_tokens,
        latency_ms=api_result.latency_ms,
        cost_usd=api_result.cost_usd,
        raw_api_response=api_result.raw_response,
        recovery_attempt_id=recovery_attempt_id,
    )
    await repository.append_response(response)
    return response


def _extract_answer(output: str, marker: str | None) -> tuple[str, bool]:
    """Return the answer a marker-bearing output commits to, and whether it never got there.

    Our principle, stated as ours: score the text a prompt commits to, not the reasoning on the way
    there. A prompt that asks the model to work through steps and then answer produces an answer
    after the marker; scoring the whole transcript scores the reasoning, which a permissive grader
    cannot reliably tell apart from a genuine conclusion. The last occurrence wins: a transcript
    can quote its own instructions.
    """
    if marker is None:
        return output, False
    if marker not in output:
        return output, True
    return output.rsplit(marker, 1)[-1].strip(), False


def build_generator_prompt(
    question: Question,
    memories: list[Memory],
    *,
    template: str | None = None,
) -> str:
    selected_template = (
        template
        if template is not None
        else CANONICAL_GENERATOR_PROMPT_PATH.read_text(encoding="utf-8")
    )
    speakers = _speakers_from_memories(memories)
    values = {
        "speaker_a": speakers[0],
        "speaker_b": speakers[1],
        "formatted_memories": _format_retrieved_memories(memories),
        "dated_memories": _format_dated_memories(memories),
        "question_text": question.question_text,
    }
    return selected_template.format(
        **{
            field_name: values[field_name]
            for _, field_name, _, _ in Formatter().parse(selected_template)
            if field_name is not None and field_name in values
        }
    )


def _format_retrieved_memories(memories: list[Memory]) -> str:
    if not memories:
        return "(No memories retrieved.)"
    blocks: list[str] = []
    for index, memory in enumerate(memories, start=1):
        session_number = _metadata_value(memory.metadata, "session_number", "?")
        speaker = _metadata_value(memory.metadata, "speaker", "unknown")
        blocks.append(
            f"[Memory {index}] (Session {session_number}, Speaker: {speaker})\n{memory.content}"
        )
    return "\n\n".join(blocks)


def _format_dated_memories(memories: list[Memory]) -> str:
    """Render memories with the session date the canonical v1 formatter dropped.

    A separate formatter rather than a change to `_format_retrieved_memories`, for a reason that
    is easy to get backwards: the methodology fingerprint hashes the prompt *file*, not this code.
    Editing the v1 formatter would therefore change what `canonical-v1` measures while its
    fingerprint stayed identical -- silent semantic drift under a fixed identity, which is worse
    than the defect being fixed. New behaviour needs a new prompt file, and a new prompt file needs
    its own placeholder.

    The defect this repairs: the ingester stores `session_timestamp`, the provider returns it, and
    v1's formatter rendered only the session number and speaker -- so temporal questions were
    effectively unanswerable however well retrieval worked, scoring near zero regardless of recall
    quality.
    """
    if not memories:
        return "(No memories retrieved.)"
    blocks: list[str] = []
    for index, memory in enumerate(memories, start=1):
        session_number = _metadata_value(memory.metadata, "session_number", "?")
        speaker = _metadata_value(memory.metadata, "speaker", "unknown")
        blocks.append(
            f"[Memory {index}] (Session {session_number}, {_human_date(memory)}, "
            f"Speaker: {speaker})\n{memory.content}"
        )
    return "\n\n".join(blocks)


def _speakers_from_memories(memories: list[Memory]) -> tuple[str, str]:
    speakers: list[str] = []
    for memory in memories:
        speaker = memory.metadata.get("speaker")
        if isinstance(speaker, str) and speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) == 2:
            return speakers[0], speakers[1]
    while len(speakers) < 2:
        speakers.append("the speakers")
    return speakers[0], speakers[1]


def _metadata_value(metadata: dict[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return fallback


def _human_date(memory: Memory) -> str:
    session_timestamp = memory.metadata.get("session_timestamp")
    if isinstance(session_timestamp, str):
        try:
            return datetime.fromisoformat(session_timestamp).strftime("%A, %B %d, %Y")
        except ValueError:
            return session_timestamp[:10]
    if memory.timestamp is not None:
        return memory.timestamp.strftime("%A, %B %d, %Y")
    return "date unknown"


def _model_exception(exc: Exception, message: str, **context: object) -> ModelError:
    if isinstance(exc, ModelError):
        return exc
    if isinstance(exc, KhedronError):
        return ModelError(message, **exc.context, **context)
    return ModelError(message, **context)


def _retryable(exc: KhedronError) -> bool | None:
    value = exc.context.get("retryable")
    return value if isinstance(value, bool) else None
