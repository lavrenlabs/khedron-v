from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeAlias, cast

import structlog

from khedron.judges.base import Judge, JudgeResult
from khedron.models.base import AnswerModel
from khedron.types import APICallResult, JudgmentVerdict

__all__ = [
    "CANONICAL_JUDGE_PROMPT_PATH",
    "REPAIR_INSTRUCTION",
    "LLMJudge",
    "load_judge_prompt",
    "parse_judge_output",
    "render_judge_prompt",
]

JsonObject: TypeAlias = dict[str, Any]

CANONICAL_JUDGE_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "prompts" / "judge_v1.txt"
)
REPAIR_INSTRUCTION = (
    "The previous judge output could not be parsed. Return ONLY valid JSON in "
    'the exact shape {"verdict": "correct|incorrect|partial", "score": 0.0, '
    '"reasoning": "one or two sentences"}. Do not include markdown fences or '
    "any extra text."
)
_UNKNOWN_REASONING = "Judge output could not be parsed after retry."
_LOGGER = structlog.get_logger(__name__)
_PLACEHOLDER_PATTERN = re.compile(
    r"\{question_text\}|\{expected_answer\}|\{generated_answer\}|\{category\}"
    r"|\{question\}|\{answer\}|\{response\}"
)
_FENCED_JSON_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_VALID_VERDICTS = {
    JudgmentVerdict.CORRECT.value,
    JudgmentVerdict.INCORRECT.value,
    JudgmentVerdict.PARTIAL.value,
}
_UNKNOWN_VERDICT_REASONING = "Judge returned a verdict outside the supported enum."


class _MalformedJudgeOutput(ValueError):
    """Raised internally when judge JSON cannot be parsed into a valid result."""


class LLMJudge(Judge):
    """Shared LLM-backed judge implementation."""

    def __init__(
        self,
        model: AnswerModel,
        *,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        prompt_path: Path = CANONICAL_JUDGE_PROMPT_PATH,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._prompt_path = prompt_path
        self._prompt_template: str | None = None
        self._log = _LOGGER.bind(vendor=model.vendor, model_id=model.model_id)

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def vendor(self) -> str:
        return self._model.vendor

    async def initialize(self) -> None:
        await self._model.initialize()

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        prompt = render_judge_prompt(
            self._load_prompt_template(),
            question_text=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            category=category,
        )
        first_call = await self._model.generate(
            prompt,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        try:
            return parse_judge_output(first_call, logger=self._log)
        except _MalformedJudgeOutput:
            retry_call = await self._model.generate(
                f"{prompt}\n\n{REPAIR_INSTRUCTION}",
                max_output_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
            try:
                return parse_judge_output(retry_call, logger=self._log)
            except _MalformedJudgeOutput:
                return JudgeResult(
                    verdict=JudgmentVerdict.UNKNOWN,
                    score=0.0,
                    reasoning=_UNKNOWN_REASONING,
                    api_call=retry_call,
                )

    async def close(self) -> None:
        await self._model.close()

    def _load_prompt_template(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = load_judge_prompt(self._prompt_path)
        return self._prompt_template


def load_judge_prompt(path: Path = CANONICAL_JUDGE_PROMPT_PATH) -> str:
    """Load the canonical judge prompt template."""
    return path.read_text(encoding="utf-8")


def render_judge_prompt(
    template: str,
    *,
    question_text: str,
    expected_answer: str,
    generated_answer: str,
    category: str | None,
) -> str:
    """Render the judge prompt without interpreting JSON braces as format syntax."""
    replacements = {
        "{question_text}": question_text,
        "{expected_answer}": expected_answer,
        "{generated_answer}": generated_answer,
        "{category}": category if category is not None else "unknown",
        "{question}": question_text,
        "{answer}": expected_answer,
        "{response}": generated_answer,
    }
    return _PLACEHOLDER_PATTERN.sub(lambda match: replacements[match.group(0)], template)


def parse_judge_output(
    api_call: APICallResult,
    *,
    logger: Any | None = None,
) -> JudgeResult:
    """Parse one judge model response into a `JudgeResult`."""
    payload = _load_json_object(api_call.output)
    unknown_verdict_result = _unknown_verdict_result(payload, api_call=api_call)
    if unknown_verdict_result is not None:
        return unknown_verdict_result
    verdict = _parse_verdict(payload)
    score = _parse_score(payload, api_call=api_call, logger=logger)
    reasoning = _parse_reasoning(payload)
    return JudgeResult(verdict=verdict, score=score, reasoning=reasoning, api_call=api_call)


def _load_json_object(output: str) -> JsonObject:
    json_text = _json_text_from_output(output)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise _MalformedJudgeOutput("Judge output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise _MalformedJudgeOutput("Judge output JSON is not an object.")
    return cast(JsonObject, payload)


def _json_text_from_output(output: str) -> str:
    match = _FENCED_JSON_PATTERN.match(output)
    if match is None:
        return output
    return match.group("body").strip()


def _parse_verdict(payload: JsonObject) -> JudgmentVerdict:
    verdict = payload.get("verdict")
    if isinstance(verdict, str) and verdict in _VALID_VERDICTS:
        return JudgmentVerdict(verdict)

    label = payload.get("label")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized == "correct":
            return JudgmentVerdict.CORRECT
        if normalized == "wrong":
            return JudgmentVerdict.INCORRECT

    raise _MalformedJudgeOutput("Judge output verdict is invalid.")


def _unknown_verdict_result(
    payload: JsonObject,
    *,
    api_call: APICallResult,
) -> JudgeResult | None:
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or verdict in _VALID_VERDICTS:
        return None
    return JudgeResult(
        verdict=JudgmentVerdict.UNKNOWN,
        score=0.0,
        reasoning=_UNKNOWN_VERDICT_REASONING,
        api_call=api_call,
    )


def _parse_score(
    payload: JsonObject,
    *,
    api_call: APICallResult,
    logger: Any | None,
) -> float:
    score = payload.get("score")
    if score is None:
        label = payload.get("label")
        if isinstance(label, str):
            normalized = label.strip().lower()
            if normalized == "correct":
                return 1.0
            if normalized == "wrong":
                return 0.0
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise _MalformedJudgeOutput("Judge output score is missing or nonnumeric.")
    numeric_score = float(score)
    clamped_score = min(1.0, max(0.0, numeric_score))
    if clamped_score != numeric_score and logger is not None:
        logger.warning(
            "judge_score_clamped",
            original_score=numeric_score,
            clamped_score=clamped_score,
            model_id=api_call.model_id,
        )
    return clamped_score


def _parse_reasoning(payload: JsonObject) -> str:
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise _MalformedJudgeOutput("Judge output reasoning is missing or invalid.")
    return reasoning
