from __future__ import annotations

# ruff: noqa: S101
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from typer.testing import CliRunner

import khedron.cli as cli_module
from khedron.benchmarks.base import Benchmark
from khedron.benchmarks.registry import BENCHMARK_REGISTRY
from khedron.cli import app
from khedron.errors import JudgeError, ModelError
from khedron.judges.base import Judge, JudgeResult
from khedron.judges.registry import JUDGE_REGISTRY
from khedron.models.base import AnswerModel
from khedron.models.registry import MODEL_REGISTRY
from khedron.providers.base import MemoryProvider, ProviderHealthStatus
from khedron.providers.registry import PROVIDER_REGISTRY
from khedron.types import (
    APICallResult,
    Conversation,
    JudgmentVerdict,
    Memory,
    Question,
    QuestionCategory,
    Session,
    Turn,
)

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def skip_methodology_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI end-to-end tests drive fake plugins, which no contractual profile can pin.

    Same reasoning as the runner integration tests: these exercise the command surface, not
    methodology conformance, and a `cli_fake` provider, model and judge cannot satisfy a contract
    that pins real vendors. Conformance has its own tests in tests/unit/methodology.
    """
    # Patched at the source: `config.py` imports this inside the function, so it resolves the
    # attribute on the package at call time rather than holding a module-level reference.
    monkeypatch.setattr(
        "khedron.methodology.validate_suite_methodology_profile", lambda config: None
    )
    monkeypatch.setattr("khedron.runner.validate_suite_methodology_profile", lambda config: None)


@pytest.fixture(autouse=True)
def _identifiable_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the command surface as if from a clean, committed tree.

    preflight blocks a dirty/unidentified build and the Runner now refuses one at its execution
    boundary too. These tests run from the live checkout, which is routinely dirty during
    development, so BOTH seams are pinned to the same clean version -- patching only preflight would
    let the runner stamp (and refuse on) the real dirty version, the check/stamp divergence the
    guard must not have. The build-identity tests below re-patch these to the value they assert on.
    """
    monkeypatch.setattr("khedron.preflight.resolve_framework_version", lambda: "0.0.0+clitest")
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+clitest")


@pytest.fixture(autouse=True)
def fake_registries() -> Iterator[None]:
    with _isolated_registries():
        FakeAnswerModel.reset_observations()
        FakeJudge.reset_observations()
        PROVIDER_REGISTRY["cli_fake"] = FakeProvider
        BENCHMARK_REGISTRY["cli_fake"] = FakeBenchmark
        MODEL_REGISTRY["cli_fake"] = FakeAnswerModel
        JUDGE_REGISTRY["cli_fake"] = FakeJudge
        yield


def test_help_works() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Khedron" in result.output
    assert "run" in result.output


def test_validate_succeeds_for_valid_fake_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = RUNNER.invoke(app, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Configuration valid" in result.output
    assert "1 experiment(s), 1 run(s)" in result.output


def test_validate_fails_cleanly_for_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("runs: 0\nexperiments: []\n", encoding="utf-8")

    result = RUNNER.invoke(app, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration error" in result.output
    assert "Traceback" not in result.output


def test_run_prints_preflight_warnings_before_spending(tmp_path: Path) -> None:
    # An operator about to spend money must see that some spend is untracked, even though
    # the warning does not halt the run. Filtering findings down to blockers before
    # printing left these invisible at exactly the moment they matter.
    config_path = _write_config(tmp_path)
    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "[warning]" in result.output
    # Ordering is the property, not presence: printing the findings after the run would
    # satisfy a substring check while telling the operator nothing they could still act on.
    assert result.output.index("[warning]") < result.output.index("Suite ID:"), result.output


def test_run_spends_nothing_when_preflight_blocks(tmp_path: Path) -> None:
    # The blocker exists to stop the run before the first paid call. Asserting the message
    # alone would pass even if the suite had already run and been reported afterwards, so
    # this asserts the absence of the work: no model call, no judge call, no artifacts.
    FakeAnswerModel.reset_observations()
    FakeJudge.reset_observations()
    config_path = _write_config(
        tmp_path, answer_model_type="openai", answer_model="gpt-4o-does-not-exist"
    )

    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1, result.output
    assert "refusing to run" in result.output
    assert FakeAnswerModel.generate_calls == []
    assert FakeJudge.evaluate_calls == 0
    assert not (tmp_path / "results").exists(), "a refused run must leave no artifacts"


def test_run_blocks_on_an_unidentifiable_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A dirty/unidentified build cannot be reproduced from a commit, so it is refused before the
    # first paid call -- the defect that once cost a full-corpus baseline run.
    FakeAnswerModel.reset_observations()
    FakeJudge.reset_observations()
    monkeypatch.setattr(
        "khedron.preflight.resolve_framework_version", lambda: "0.0.0+abc1234.dirty"
    )
    config_path = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1, result.output
    assert "refusing to run" in result.output
    assert "unidentifiable" in result.output
    assert FakeAnswerModel.generate_calls == []
    assert FakeJudge.evaluate_calls == 0
    assert not (tmp_path / "results").exists(), "a refused run must leave no artifacts"


def test_allow_unidentifiable_build_overrides_only_that_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The override lets a knowingly un-publishable run proceed, and says so loudly. It must not
    # be a way to silence any other blocker. Both seams are dirty so the run exercises the CLI
    # preflight override AND the Runner boundary guard's override branch end-to-end, not just the
    # former with a clean runner underneath.
    monkeypatch.setattr(
        "khedron.preflight.resolve_framework_version", lambda: "0.0.0+abc1234.dirty"
    )
    monkeypatch.setattr("khedron.runner.resolve_framework_version", lambda: "0.0.0+abc1234.dirty")
    config_path = _write_config(tmp_path)

    result = CliRunner().invoke(
        app, ["run", "--config", str(config_path), "--allow-unidentifiable-build"]
    )

    assert result.exit_code == 0, result.output
    assert "un-publishable" in result.output
    assert "Suite ID:" in result.output


def test_run_estimate_inspect_and_db_commands_use_fake_plugins(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    results_dir = tmp_path / "results"

    estimate = RUNNER.invoke(app, ["estimate", "--config", str(config_path)])
    assert estimate.exit_code == 0
    assert "Runs: 1" in estimate.output
    assert "CLI fake experiment" in estimate.output
    assert FakeAnswerModel.generate_calls == []
    assert FakeJudge.evaluate_calls == 0

    run = RUNNER.invoke(app, ["run", "--config", str(config_path)])
    assert run.exit_code == 0
    assert "Suite ID:" in run.output
    assert "Total cost USD: 0.030000" in run.output
    suite_id = _suite_id_from_output(run.output)

    suite = RUNNER.invoke(
        app,
        ["inspect", "suite", "--suite-id", suite_id, "--results-dir", str(results_dir)],
    )
    assert suite.exit_code == 0
    assert "Status: completed" in suite.output
    assert "Run IDs:" in suite.output

    field = RUNNER.invoke(
        app,
        [
            "inspect",
            "suite",
            "--suite-id",
            suite_id,
            "--field",
            "status",
            "--results-dir",
            str(results_dir),
        ],
    )
    assert field.exit_code == 0
    assert field.output == "completed\n"

    unknown_field = RUNNER.invoke(
        app,
        [
            "inspect",
            "suite",
            "--suite-id",
            suite_id,
            "--field",
            "missing",
            "--results-dir",
            str(results_dir),
        ],
    )
    assert unknown_field.exit_code == 1
    assert "Unknown suite field" in unknown_field.output

    run_id = _run_id_from_suite_output(suite.output)

    run_inspect = RUNNER.invoke(
        app,
        ["inspect", "run", "--run-id", run_id, "--results-dir", str(results_dir)],
    )
    assert run_inspect.exit_code == 0
    assert "Status: completed" in run_inspect.output
    assert "Questions attempted: 1" in run_inspect.output
    assert "Answer model: fake-answer-model" in run_inspect.output
    assert "Standard score:" in run_inspect.output

    question = RUNNER.invoke(
        app,
        [
            "inspect",
            "question",
            "--run-id",
            run_id,
            "--question-id",
            "q1",
            "--results-dir",
            str(results_dir),
        ],
    )
    assert question.exit_code == 0
    assert "Question ID: q1" in question.output
    assert "Category: single_hop" in question.output
    assert "Verdict: correct" in question.output
    assert "Generated answer: Blue" in question.output
    assert "Retrieval ID:" in question.output
    assert "Response ID:" in question.output
    assert "Judgment ID:" in question.output

    rebuild = RUNNER.invoke(app, ["db", "rebuild", "--results-dir", str(results_dir)])
    assert rebuild.exit_code == 0
    assert "SQLite rebuilt" in rebuild.output

    version = RUNNER.invoke(app, ["db", "version", "--results-dir", str(results_dir)])
    assert version.exit_code == 0
    assert version.output == "2\n"


def test_report_generates_markdown_for_all_report_types(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    results_dir = tmp_path / "results"

    run = RUNNER.invoke(app, ["run", "--config", str(config_path)])
    assert run.exit_code == 0
    suite_id = _suite_id_from_output(run.output)
    suite = RUNNER.invoke(
        app,
        ["inspect", "suite", "--suite-id", suite_id, "--results-dir", str(results_dir)],
    )
    assert suite.exit_code == 0
    run_id = _run_id_from_suite_output(suite.output)

    cases = [
        ("executive", "# Executive Summary", ["--rebuild"]),
        ("technical", "# Technical Deep Dive", []),
        ("failures", "# Failure Analysis", []),
    ]
    for report_type, expected_heading, extra_args in cases:
        output_path = tmp_path / f"{report_type}.md"
        result = RUNNER.invoke(
            app,
            [
                "report",
                "--run-id",
                run_id,
                "--type",
                report_type,
                "--output",
                str(output_path),
                "--results-dir",
                str(results_dir),
                # Fixture runs use fake plugins under a profile that pins no model, so they are
                # correctly not publishable. These tests exercise report generation; the
                # disposition has its own tests in tests/unit/test_publishability.py.
                "--force",
                *extra_args,
            ],
        )

        assert result.exit_code == 0
        assert "withdrawn/non-publishable" in result.output
        assert "Report written:" in result.output
        assert expected_heading in output_path.read_text(encoding="utf-8")


def test_compare_force_does_not_bypass_withdrawal(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, runs=2)
    results_dir = tmp_path / "results"

    run = RUNNER.invoke(app, ["run", "--config", str(config_path)])
    assert run.exit_code == 0
    suite_id = _suite_id_from_output(run.output)
    suite = RUNNER.invoke(
        app,
        ["inspect", "suite", "--suite-id", suite_id, "--results-dir", str(results_dir)],
    )
    assert suite.exit_code == 0
    run_ids = _run_ids_from_suite_output(suite.output)
    assert len(run_ids) == 2

    output_path = tmp_path / "comparison.md"
    result = RUNNER.invoke(
        app,
        [
            "compare",
            "--run-ids",
            run_ids[0],
            "--run-ids",
            run_ids[1],
            "--output",
            str(output_path),
            "--results-dir",
            str(results_dir),
            "--force",
        ],
    )

    assert result.exit_code == 1
    assert "Refusing to write a report" in result.output
    assert not output_path.exists()


def test_report_and_compare_expected_user_errors_have_no_traceback(
    tmp_path: Path,
) -> None:
    report_output = tmp_path / "report.md"
    comparison_output = tmp_path / "comparison.md"

    missing_report_options = RUNNER.invoke(app, ["report"])
    assert missing_report_options.exit_code == 1
    assert "Missing required option: --run-id" in missing_report_options.output
    assert "Traceback" not in missing_report_options.output

    missing_compare_options = RUNNER.invoke(app, ["compare"])
    assert missing_compare_options.exit_code == 1
    assert "Missing required option: --run-ids" in missing_compare_options.output
    assert "Traceback" not in missing_compare_options.output

    invalid_type = RUNNER.invoke(
        app,
        [
            "report",
            "--run-id",
            "missing-run",
            "--type",
            "invalid",
            "--output",
            str(report_output),
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert invalid_type.exit_code == 1
    assert "Invalid report type" in invalid_type.output
    assert "Traceback" not in invalid_type.output

    missing_run_data = RUNNER.invoke(
        app,
        [
            "report",
            "--run-id",
            "missing-run",
            "--type",
            "executive",
            "--output",
            str(report_output),
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert missing_run_data.exit_code == 1
    # The publishability guard runs first and reports the same class of user error: a run that does
    # not exist has no disposition to establish. What this test protects is that an expected user
    # error exits 1 without a traceback, not which layer names it.
    assert "Unable to establish publishability" in missing_run_data.output
    assert "Traceback" not in missing_run_data.output

    invalid_run_count = RUNNER.invoke(
        app,
        [
            "compare",
            "--run-ids",
            "only-one-run",
            "--output",
            str(comparison_output),
            "--results-dir",
            str(tmp_path / "results"),
        ],
    )
    assert invalid_run_count.exit_code == 1
    assert "exactly two run IDs" in invalid_run_count.output
    assert "Traceback" not in invalid_run_count.output


def test_run_unexpected_error_exits_with_system_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)

    class CrashingRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "Runner", CrashingRunner)

    result = RUNNER.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "Unexpected system error: boom" in result.output


def _write_config(
    tmp_path: Path,
    *,
    runs: int = 1,
    answer_model_type: str = "cli_fake",
    answer_model: str = "fake-answer-model",
) -> Path:
    config_path = tmp_path / "suite.yaml"
    config = {
        "runs": runs,
        "output_dir": str(tmp_path / "results"),
        "methodology_version": "1.0",
        "methodology_profile": "canonical-v1",
        "experiments": [
            {
                "name": "CLI fake experiment",
                "provider": {"type": "cli_fake", "config": {}},
                "benchmark": {
                    "type": "cli_fake",
                    "config": {
                        "questions": [
                            {
                                "id": "q1",
                                "question": "What color is Alice's notebook?",
                                "expected_answer": "Blue",
                                "category": "single_hop",
                            }
                        ]
                    },
                },
                "answer_model": {"type": answer_model_type, "model": answer_model},
                "judge": {"type": "cli_fake", "model": "fake-judge-model"},
                "top_k_retrieval": 10,
                "max_concurrent_questions": 1,
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


class FakeProvider(MemoryProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        del config
        self._memories: list[Memory] = []

    @property
    def provider_type(self) -> str:
        return "cli_fake"

    @property
    def provider_version(self) -> str:
        return "test"

    async def initialize(self) -> None:
        return None

    async def health_check(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(healthy=True, version=self.provider_version)

    async def reset(self) -> None:
        self._memories.clear()

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        memory_id = f"memory-{len(self._memories) + 1}"
        self._memories.append(
            Memory(
                memory_id=memory_id,
                content=content,
                metadata=dict(metadata) if metadata is not None else {},
                score=1.0,
                timestamp=NOW,
            )
        )
        return memory_id

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        del query, top_k, metadata_filter
        return list(self._memories)

    async def close(self) -> None:
        return None


class FakeBenchmark(Benchmark):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._questions: list[Question] = []
        self._conversation = Conversation(
            conversation_id="conversation-1",
            speakers=["Alice", "Bob"],
            sessions=[
                Session(
                    session_id="session-1",
                    session_number=1,
                    timestamp=NOW,
                    turns=[
                        Turn(
                            turn_id="turn-1",
                            speaker="Alice",
                            content="Alice bought a blue notebook.",
                        )
                    ],
                )
            ],
        )

    @property
    def benchmark_type(self) -> str:
        return "cli_fake"

    @property
    def benchmark_version(self) -> str:
        return "test"

    @property
    def dataset_checksum(self) -> str:
        return "sha256:cli-fake"

    async def load(self) -> None:
        self._questions = [
            Question(
                question_id=str(raw["id"]),
                conversation_id="conversation-1",
                category=QuestionCategory(str(raw.get("category", "single_hop"))),
                question_text=str(raw["question"]),
                expected_answer=str(raw.get("expected_answer", "Blue")),
                is_audited_error=bool(raw.get("audited", False)),
            )
            for raw in self._config["questions"]
        ]

    def get_conversations(self) -> list[Conversation]:
        return [self._conversation]

    def get_questions(
        self,
        conversation_id: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Question]:
        questions = list(self._questions)
        if conversation_id is not None:
            questions = [
                question for question in questions if question.conversation_id == conversation_id
            ]
        if categories is not None:
            category_values = set(categories)
            questions = [
                question for question in questions if question.category.value in category_values
            ]
        return questions

    def get_audit_errors(self) -> set[str]:
        return {question.question_id for question in self._questions if question.is_audited_error}


class FakeAnswerModel(AnswerModel):
    generate_calls: ClassVar[list[str]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self._model_id = str(config["model_id"])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vendor(self) -> str:
        return "cli_fake"

    async def initialize(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        del max_output_tokens, temperature
        self.generate_calls.append(prompt)
        if "fail-generate" in prompt:
            raise ModelError("fake generation failed")
        return _api_result(output="Blue", model_id=self._model_id, cost=0.01)

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
        return "cli_fake"

    async def initialize(self) -> None:
        return None

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str | None = None,
    ) -> JudgeResult:
        del question, category
        type(self).evaluate_calls += 1
        if generated_answer == "fail-judge":
            raise JudgeError("fake judge failed")
        verdict = (
            JudgmentVerdict.CORRECT
            if generated_answer == expected_answer
            else JudgmentVerdict.INCORRECT
        )
        return JudgeResult(
            verdict=verdict,
            score=1.0 if verdict is JudgmentVerdict.CORRECT else 0.0,
            reasoning="fake reasoning",
            api_call=_api_result(
                output=json.dumps({"verdict": verdict.value}),
                model_id=self._model_id,
                cost=0.02,
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
    cost: float,
    raw_response: dict[str, Any] | None = None,
) -> APICallResult:
    return APICallResult(
        output=output,
        input_tokens=10,
        output_tokens=2,
        latency_ms=5.0,
        cost_usd=cost,
        model_id=model_id,
        raw_response=raw_response or {},
    )


def _suite_id_from_output(output: str) -> str:
    match = re.search(r"Suite ID: (\S+)", output)
    assert match is not None
    return match.group(1)


def _run_id_from_suite_output(output: str) -> str:
    match = re.search(r"Run IDs: (\S+)", output)
    assert match is not None
    return match.group(1)


def _run_ids_from_suite_output(output: str) -> list[str]:
    match = re.search(r"Run IDs: (.+)", output)
    assert match is not None
    return [run_id.strip() for run_id in match.group(1).split(",")]


@contextmanager
def _isolated_registries() -> Iterator[None]:
    provider_registry = dict(PROVIDER_REGISTRY)
    benchmark_registry = dict(BENCHMARK_REGISTRY)
    model_registry = dict(MODEL_REGISTRY)
    judge_registry = dict(JUDGE_REGISTRY)
    PROVIDER_REGISTRY.clear()
    BENCHMARK_REGISTRY.clear()
    MODEL_REGISTRY.clear()
    JUDGE_REGISTRY.clear()
    try:
        yield
    finally:
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(provider_registry)
        BENCHMARK_REGISTRY.clear()
        BENCHMARK_REGISTRY.update(benchmark_registry)
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(model_registry)
        JUDGE_REGISTRY.clear()
        JUDGE_REGISTRY.update(judge_registry)
