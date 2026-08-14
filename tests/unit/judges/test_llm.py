from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from khedron.judges.llm import (
    CANONICAL_JUDGE_PROMPT_PATH,
    REPAIR_INSTRUCTION,
    LLMJudge,
    load_judge_prompt,
    parse_judge_output,
    render_judge_prompt,
)
from khedron.models.base import AnswerModel
from khedron.types import APICallResult, JudgmentVerdict


def api_call_result(output: str, *, model_id: str = "fake-judge-model") -> APICallResult:
    return APICallResult(
        output=output,
        input_tokens=20,
        output_tokens=8,
        latency_ms=12.5,
        cost_usd=0.001,
        model_id=model_id,
        raw_response={"id": f"{model_id}-response"},
    )


class FakeAnswerModel(AnswerModel):
    def __init__(
        self,
        outputs: list[str],
        *,
        vendor: str = "fake",
        model_id: str = "fake-judge-model",
    ) -> None:
        self._outputs = list(outputs)
        self._vendor = vendor
        self._model_id = model_id
        self.prompts: list[str] = []
        self.max_output_tokens: list[int] = []
        self.temperatures: list[float] = []
        self.initialize_count = 0
        self.close_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return self._vendor

    async def initialize(self) -> None:
        self.initialize_count += 1

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        self.prompts.append(prompt)
        self.max_output_tokens.append(max_output_tokens)
        self.temperatures.append(temperature)
        if not self._outputs:
            raise AssertionError("fake output queue is empty")
        return api_call_result(self._outputs.pop(0), model_id=self._model_id)

    async def close(self) -> None:
        self.close_count += 1


def correct_json(*, score: float = 1.0) -> str:
    return f'{{"verdict": "correct", "score": {score}, "reasoning": "The answers match."}}'


def incorrect_json() -> str:
    return (
        '{"verdict": "incorrect", "score": 0.0, '
        '"reasoning": "The generated answer contradicts the ground truth."}'
    )


def partial_json() -> str:
    return (
        '{"verdict": "partial", "score": 0.5, '
        '"reasoning": "The generated answer includes some but not all facts."}'
    )


def test_canonical_prompt_file_is_materialized() -> None:
    expected = "\n".join(
        [
            (
                "You are evaluating whether an AI assistant's answer to a question "
                "matches the ground truth answer."
            ),
            "",
            "Question: {question_text}",
            "",
            "Ground Truth Answer: {expected_answer}",
            "",
            "Assistant's Answer: {generated_answer}",
            "",
            "Question Category: {category}",
            "",
            (
                "Evaluate whether the assistant's answer is semantically equivalent "
                "to the ground truth answer. Consider only meaning, not exact word matching."
            ),
            "",
            (
                'For "adversarial" category questions, the ground truth is often '
                '"I don\'t know" or similar. A correct answer for these MUST acknowledge '
                "that the information cannot be determined. A confident but wrong answer "
                "is INCORRECT."
            ),
            "",
            (
                'For "temporal" category questions, exact dates and durations matter. '
                "Off-by-one errors are INCORRECT."
            ),
            "",
            (
                "For other categories, semantic equivalence is sufficient. Different "
                "phrasings of the same fact are CORRECT."
            ),
            "",
            "Respond ONLY with valid JSON in this exact format:",
            "{",
            '  "verdict": "correct" | "incorrect" | "partial",',
            '  "score": <number from 0.0 to 1.0>,',
            '  "reasoning": "<your reasoning in one or two sentences>"',
            "}",
            "",
        ]
    )

    if load_judge_prompt(CANONICAL_JUDGE_PROMPT_PATH) != expected:
        raise AssertionError(CANONICAL_JUDGE_PROMPT_PATH)


def test_prompt_rendering_includes_inputs_and_preserves_json_structure(tmp_path: Path) -> None:
    template = load_judge_prompt(CANONICAL_JUDGE_PROMPT_PATH)

    rendered = render_judge_prompt(
        template,
        question_text="Where did Alice move?",
        expected_answer="Alice moved to Rome.",
        generated_answer="Rome",
        category="single_hop",
    )

    for value in (
        "Where did Alice move?",
        "Alice moved to Rome.",
        "Rome",
        "single_hop",
    ):
        if value not in rendered:
            raise AssertionError(rendered)
    if '{"verdict": "correct" | "incorrect" | "partial",' in rendered:
        raise AssertionError(rendered)
    if '"verdict": "correct" | "incorrect" | "partial"' not in rendered:
        raise AssertionError(rendered)
    if "{question_text}" in rendered or "{expected_answer}" in rendered:
        raise AssertionError(rendered)

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(template, encoding="utf-8")
    judge = LLMJudge(FakeAnswerModel([correct_json()]), prompt_path=prompt_path)
    if judge.model_id != "fake-judge-model" or judge.vendor != "fake":
        raise AssertionError(judge)


def test_prompt_rendering_preserves_placeholder_like_input_text() -> None:
    rendered = render_judge_prompt(
        "{question_text}\n{expected_answer}\n{generated_answer}\n{category}",
        question_text="Question mentions {expected_answer}.",
        expected_answer="Expected mentions {generated_answer}.",
        generated_answer="Generated mentions {category}.",
        category="single_hop",
    )

    expected = "\n".join(
        [
            "Question mentions {expected_answer}.",
            "Expected mentions {generated_answer}.",
            "Generated mentions {category}.",
            "single_hop",
        ]
    )
    if rendered != expected:
        raise AssertionError(rendered)


def test_parse_judge_output_accepts_v3_labels() -> None:
    correct = parse_judge_output(
        api_call_result('{"reasoning": "The response identifies Kyoto.", "label": "CORRECT"}')
    )
    wrong = parse_judge_output(
        api_call_result('{"reasoning": "The response names the wrong city.", "label": "WRONG"}')
    )

    if correct.verdict is not JudgmentVerdict.CORRECT or correct.score != 1.0:
        raise AssertionError(correct)
    if wrong.verdict is not JudgmentVerdict.INCORRECT or wrong.score != 0.0:
        raise AssertionError(wrong)


@pytest.mark.asyncio
async def test_lifecycle_delegates_to_composed_model() -> None:
    model = FakeAnswerModel([correct_json()])
    judge = LLMJudge(model)

    await judge.initialize()
    await judge.close()

    if model.initialize_count != 1:
        raise AssertionError(model.initialize_count)
    if model.close_count != 1:
        raise AssertionError(model.close_count)


@pytest.mark.asyncio
async def test_identity_case_returns_correct() -> None:
    model = FakeAnswerModel([correct_json()])
    judge = LLMJudge(model, max_output_tokens=64, temperature=0.2)

    result = await judge.evaluate(
        "Where did Alice move?",
        "Rome",
        "Rome",
        category="single_hop",
    )

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if result.score != 1.0:
        raise AssertionError(result)
    if result.reasoning != "The answers match.":
        raise AssertionError(result)
    if model.max_output_tokens != [64] or model.temperatures != [0.2]:
        raise AssertionError((model.max_output_tokens, model.temperatures))


@pytest.mark.asyncio
async def test_clear_disagreement_returns_incorrect() -> None:
    judge = LLMJudge(FakeAnswerModel([incorrect_json()]))

    result = await judge.evaluate(
        "Where did Alice move?",
        "Rome",
        "Paris",
        category="single_hop",
    )

    if result.verdict is not JudgmentVerdict.INCORRECT:
        raise AssertionError(result)
    if result.score != 0.0:
        raise AssertionError(result)


@pytest.mark.asyncio
async def test_partial_verdict_parses_successfully() -> None:
    judge = LLMJudge(FakeAnswerModel([partial_json()]))

    result = await judge.evaluate(
        "Who came to dinner?",
        "Alice and Bob",
        "Alice",
        category="multi_hop",
    )

    if result.verdict is not JudgmentVerdict.PARTIAL:
        raise AssertionError(result)
    if result.score != 0.5:
        raise AssertionError(result)


@pytest.mark.asyncio
async def test_malformed_json_triggers_one_retry() -> None:
    model = FakeAnswerModel(["not json", correct_json()])
    judge = LLMJudge(model)

    result = await judge.evaluate("Question?", "Expected", "Generated", category="temporal")

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if len(model.prompts) != 2:
        raise AssertionError(model.prompts)
    if REPAIR_INSTRUCTION in model.prompts[0]:
        raise AssertionError(model.prompts[0])
    if REPAIR_INSTRUCTION not in model.prompts[1]:
        raise AssertionError(model.prompts[1])


@pytest.mark.asyncio
async def test_markdown_fenced_json_parses_without_retry() -> None:
    model = FakeAnswerModel([f"```json\n{correct_json()}\n```"])
    judge = LLMJudge(model)

    result = await judge.evaluate("Question?", "Expected", "Expected")

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if len(model.prompts) != 1:
        raise AssertionError(model.prompts)


def test_parse_judge_output_accepts_unlabeled_markdown_fenced_json() -> None:
    result = parse_judge_output(api_call_result(f"```\n{incorrect_json()}\n```"))

    if result.verdict is not JudgmentVerdict.INCORRECT:
        raise AssertionError(result)


@pytest.mark.asyncio
async def test_malformed_json_twice_returns_unknown_with_last_api_call() -> None:
    model = FakeAnswerModel(["not json", "still not json"], model_id="retry-model")
    judge = LLMJudge(model)

    result = await judge.evaluate("Question?", "Expected", "Generated")

    if result.verdict is not JudgmentVerdict.UNKNOWN:
        raise AssertionError(result)
    if result.score != 0.0:
        raise AssertionError(result)
    if "could not be parsed" not in result.reasoning:
        raise AssertionError(result)
    if result.api_call.output != "still not json":
        raise AssertionError(result.api_call)
    if len(model.prompts) != 2:
        raise AssertionError(model.prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_output",
    [
        '{"verdict": "correct", "reasoning": "Missing score."}',
        '{"verdict": "correct", "score": "1.0", "reasoning": "Nonnumeric score."}',
        '{"verdict": "correct", "score": true, "reasoning": "Boolean score."}',
        '{"score": 1.0, "reasoning": "Missing verdict."}',
    ],
)
async def test_semantically_invalid_output_retries_once(bad_output: str) -> None:
    model = FakeAnswerModel([bad_output, correct_json()])
    judge = LLMJudge(model)

    result = await judge.evaluate("Question?", "Expected", "Generated")

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if len(model.prompts) != 2:
        raise AssertionError(model.prompts)


@pytest.mark.asyncio
async def test_invalid_verdict_returns_unknown_without_retry_using_original_api_call() -> None:
    output = '{"verdict": "mostly", "score": 1.0, "reasoning": "Invalid verdict."}'
    model = FakeAnswerModel([output, correct_json()])
    judge = LLMJudge(model)

    result = await judge.evaluate("Question?", "Expected", "Generated")

    if result.verdict is not JudgmentVerdict.UNKNOWN:
        raise AssertionError(result)
    if result.score != 0.0:
        raise AssertionError(result)
    if "outside the supported enum" not in result.reasoning:
        raise AssertionError(result)
    if result.api_call.output != output:
        raise AssertionError(result.api_call)
    if len(model.prompts) != 1:
        raise AssertionError(model.prompts)


@pytest.mark.asyncio
async def test_out_of_range_score_clamps_and_logs_warning_without_retry() -> None:
    model = FakeAnswerModel([correct_json(score=1.5)])
    judge = LLMJudge(model)

    with capture_logs() as logs:
        result = await judge.evaluate("Question?", "Expected", "Generated")

    if result.verdict is not JudgmentVerdict.CORRECT:
        raise AssertionError(result)
    if result.score != 1.0:
        raise AssertionError(result)
    if len(model.prompts) != 1:
        raise AssertionError(model.prompts)
    warning_logs = [
        log
        for log in logs
        if log.get("event") == "judge_score_clamped" and log.get("log_level") == "warning"
    ]
    if len(warning_logs) != 1:
        raise AssertionError(logs)
    if warning_logs[0].get("original_score") != 1.5:
        raise AssertionError(warning_logs)
