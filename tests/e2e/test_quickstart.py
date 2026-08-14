from __future__ import annotations

# ruff: noqa: S101
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import yaml
from typer.testing import CliRunner

from khedron.cli import app
from khedron.judges.base import Judge, JudgeResult
from khedron.judges.registry import JUDGE_REGISTRY
from khedron.models.base import AnswerModel
from khedron.models.registry import MODEL_REGISTRY
from khedron.persistence.repository import RunRepository
from khedron.types import APICallResult, JudgmentVerdict, RunFilters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_CONFIG = PROJECT_ROOT / "experiments" / "quickstart.yaml"
LOCOMO_DATASET_PATH = PROJECT_ROOT / "data" / "locomo" / "locomo10.json"
# The end-to-end run needs the real LoCoMo corpus, which is not vendored (CC BY-NC 4.0). It skips
# when the dataset is absent so offline CI stays green; config validation below needs no corpus.
requires_locomo = pytest.mark.skipif(
    not LOCOMO_DATASET_PATH.exists(),
    reason="LoCoMo dataset absent; run the download script under scripts/ to fetch it.",
)
FULL_LOCOMO_QUESTION_COUNT = 1986
# conv-44 across every evaluated category. It was 11 when quickstart restricted itself to
# multi_hop; canonical-v2 pins the category set, so scope can only be reduced by conversation.
# The conversation changed from conv-30 to conv-44 on 2026-07-31: conv-30 is the only LoCoMo
# conversation containing no open_domain questions, so a quickstart over it silently exercised
# four of the five categories the profile evaluates. The runner now refuses such a corpus, which
# is what turned this from a comment describing the gap into a failing test.
QUICKSTART_SUBSET_QUESTION_COUNT = 158
RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def fake_model_and_judge_registries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Placeholder credentials so preflight's credential check passes: the config declares the
    # openai/anthropic vendors, and preflight legitimately blocks a run whose declared vendors have
    # no key in the environment -- it cannot know the registry was swapped for fakes that read no
    # key. A real operator runs with keys present; these stand in, and the fakes ignore them.
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")
    # Drive the run as if from a clean, committed tree: preflight blocks a dirty/unidentified build
    # and the Runner now refuses one at its execution boundary, and this suite runs from the live
    # checkout, which is routinely dirty in development. Both seams are pinned to the same version.
    monkeypatch.setattr("khedron.preflight.resolve_framework_version", lambda: "0.0.0+testbuild")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+testbuild")
    with _isolated_model_and_judge_registries():
        FakeAnswerModel.reset_observations()
        FakeJudge.reset_observations()
        MODEL_REGISTRY["openai"] = FakeAnswerModel
        JUDGE_REGISTRY["anthropic"] = FakeJudge
        yield


def test_quickstart_config_validates() -> None:
    result = RUNNER.invoke(app, ["validate", "--config", str(QUICKSTART_CONFIG)])

    assert result.exit_code == 0
    assert "Configuration valid" in result.output
    assert "1 experiment(s), 1 run(s)" in result.output


@requires_locomo
def test_quickstart_run_completes_with_real_locomo_and_full_context_fakes(
    tmp_path: Path,
) -> None:
    config_path = _write_tmp_quickstart_config(tmp_path)
    results_dir = tmp_path / "results"

    run = RUNNER.invoke(app, ["run", "--config", str(config_path)])
    assert run.exit_code == 0
    assert "Suite ID:" in run.output
    assert "Total cost USD: 0.000000" in run.output

    suite_id = _suite_id_from_output(run.output)
    repository = RunRepository(
        results_dir=results_dir,
        sqlite_path=results_dir / "benchmark.db",
    )
    suite_status = repository.get_suite_status(suite_id)
    runs = repository.list_runs(RunFilters(suite_id=suite_id))
    assert suite_status.status == "completed"
    assert len(runs) == 1

    run_summary = runs[0]
    run_status = repository.get_run_status(run_summary.run_id)
    questions = repository.get_questions_for_run(run_summary.run_id)

    assert run_summary.status == "completed"
    assert run_summary.provider_type == "full_context"
    assert run_summary.benchmark_type == "locomo"
    assert run_status.n_conversations_processed == 1
    assert run_status.n_questions_attempted == QUICKSTART_SUBSET_QUESTION_COUNT
    assert len(questions) == QUICKSTART_SUBSET_QUESTION_COUNT
    assert len(questions) < FULL_LOCOMO_QUESTION_COUNT
    assert {question.conversation_id for question in questions} == {"conv-44"}
    assert {question.category.value for question in questions} == {
        "single_hop",
        "multi_hop",
        "temporal",
        "open_domain",
        "adversarial",
    }
    assert FakeAnswerModel.generate_calls == [
        (128, 0.0) for _ in range(QUICKSTART_SUBSET_QUESTION_COUNT)
    ]
    assert FakeJudge.evaluate_calls == QUICKSTART_SUBSET_QUESTION_COUNT

    suite = RUNNER.invoke(
        app,
        [
            "inspect",
            "suite",
            "--suite-id",
            suite_id,
            "--results-dir",
            str(results_dir),
        ],
    )
    assert suite.exit_code == 0
    assert "Status: completed" in suite.output
    assert run_summary.run_id in suite.output

    run_inspect = RUNNER.invoke(
        app,
        ["inspect", "run", "--run-id", run_summary.run_id, "--results-dir", str(results_dir)],
    )
    assert run_inspect.exit_code == 0
    assert "Status: completed" in run_inspect.output
    assert f"Questions attempted: {QUICKSTART_SUBSET_QUESTION_COUNT}" in run_inspect.output

    question_inspect = RUNNER.invoke(
        app,
        [
            "inspect",
            "question",
            "--run-id",
            run_summary.run_id,
            "--question-id",
            questions[0].question_id,
            "--results-dir",
            str(results_dir),
        ],
    )
    assert question_inspect.exit_code == 0
    assert f"Question ID: {questions[0].question_id}" in question_inspect.output
    assert f"Category: {questions[0].category.value}" in question_inspect.output
    assert "Verdict: correct" in question_inspect.output

    # Corpus coverage, followed from the run record into the rendered report. The label logic has
    # unit tests and the templates have goldens, but deleting the runner's recording left every
    # one of them green -- this is the only assertion that fails when the wiring breaks. Folded
    # into this test rather than given its own: a second quickstart run costs six minutes to
    # re-prove a run that already happened.
    environment = repository.get_run_started_event(run_summary.run_id).runtime_environment
    assert environment["corpus_conversations_evaluated"] == 1
    assert environment["corpus_conversations_available"] == 10

    report = RUNNER.invoke(
        app,
        [
            "report",
            "--type",
            "executive",
            "--run-id",
            run_summary.run_id,
            "--results-dir",
            str(results_dir),
            "--output",
            str(tmp_path / "summary.md"),
            "--force",
        ],
    )
    assert report.exit_code == 0, report.output
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "1 of 10 conversations" in summary
    assert "not a measurement" in summary


class FakeAnswerModel(AnswerModel):
    generate_calls: ClassVar[list[tuple[int, float]]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_id = str(config["model_id"])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return "fake_openai"

    async def initialize(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        del prompt
        self.generate_calls.append((max_output_tokens, temperature))
        return _api_result(
            output="Fake answer generated without a live model call.",
            model_id=self._model_id,
        )

    async def close(self) -> None:
        return None

    @classmethod
    def reset_observations(cls) -> None:
        cls.generate_calls = []


class FakeJudge(Judge):
    evaluate_calls: ClassVar[int] = 0

    def __init__(self, config: dict[str, Any], *, temperature: float = 0.0) -> None:
        del temperature
        self._model_id = str(config["model_id"])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return "fake_anthropic"

    async def initialize(self) -> None:
        return None

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        del question, expected_answer, generated_answer, category
        type(self).evaluate_calls += 1
        return JudgeResult(
            verdict=JudgmentVerdict.CORRECT,
            score=1.0,
            reasoning="CI-safe fake judge; no live API call was made.",
            api_call=_api_result(
                output=json.dumps({"verdict": JudgmentVerdict.CORRECT.value}),
                model_id=self._model_id,
                raw_response={"prompt": "fake judge prompt"},
            ),
        )

    async def close(self) -> None:
        return None

    @classmethod
    def reset_observations(cls) -> None:
        cls.evaluate_calls = 0


def _api_result(
    *,
    output: str,
    model_id: str,
    raw_response: dict[str, Any] | None = None,
) -> APICallResult:
    return APICallResult(
        output=output,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        cost_usd=0.0,
        model_id=model_id,
        raw_response=raw_response or {},
    )


def _write_tmp_quickstart_config(tmp_path: Path) -> Path:
    raw_config = yaml.safe_load(QUICKSTART_CONFIG.read_text(encoding="utf-8"))
    config = cast(dict[str, Any], raw_config)
    config["output_dir"] = str(tmp_path / "results")

    config_path = tmp_path / "quickstart.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _suite_id_from_output(output: str) -> str:
    match = re.search(r"Suite ID: (\S+)", output)
    assert match is not None
    return match.group(1)


@contextmanager
def _isolated_model_and_judge_registries() -> Iterator[None]:
    model_registry = dict(MODEL_REGISTRY)
    judge_registry = dict(JUDGE_REGISTRY)
    try:
        yield
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(model_registry)
        JUDGE_REGISTRY.clear()
        JUDGE_REGISTRY.update(judge_registry)
