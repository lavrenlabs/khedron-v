from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

__all__ = ["RunProgress", "read_progress"]

# Registered in the canonical-v3 pre-registration §4, prediction 3. Used here rather than a second
# private copy, so the rate a run is monitored against is the rate it will later be judged by.
REFUSAL_MARKERS: Final[tuple[str, ...]] = (
    "don't have enough information",
    "not enough information",
    "do not have enough",
    "cannot determine",
    "can't determine",
    "not specified",
    "not mentioned",
    "no record",
    "don't know",
    "unable to",
)

# A response this close to its ceiling did not stop because it finished. A median output sitting
# exactly at the cap can mean most of a run's answers were truncated mid-reasoning, invalidating
# the whole measurement without any single call raising an error -- this threshold exists to
# surface that on the first handful of questions rather than after the run is paid for.
_TRUNCATION_HARD: Final[float] = 0.95
_TRUNCATION_WARN: Final[float] = 0.80


def _empty_messages() -> list[str]:
    return []


@dataclass
class RunProgress:
    """What a run has done so far, and every reason to stop it now."""

    run_id: str
    profile: str
    n_answered: int
    n_errored: int
    n_judge_parse_failures: int
    cost_usd: float
    max_output_tokens: int | None
    p95_output_tokens: int
    n_at_token_ceiling: int
    n_refusals: int
    stop: list[str] = field(default_factory=_empty_messages)
    warn: list[str] = field(default_factory=_empty_messages)

    @property
    def refusal_rate(self) -> float:
        return self.n_refusals / self.n_answered if self.n_answered else 0.0


def read_progress(
    results_dir: Path, run_id: str, *, expected_cost_usd: float | None = None
) -> RunProgress:
    """Read a run in flight and report whether it should be killed.

    Reads the append-only stream directly rather than the SQLite projection, because the projection
    is refreshed best-effort and a run being monitored is precisely a run that may be about to die.

    It does not abort anything. A monitor that kills runs needs to be right; this one only needs to
    be honest, and the operator decides. What it must never do is report a run as healthy because
    the check that would have caught the problem was not written -- so every `stop` reason below
    corresponds to a defect this project has already paid for.
    """
    run_dir = results_dir / "runs" / run_id
    started = _first_event(run_dir / "lifecycle.jsonl")
    responses = _read(run_dir / "responses.jsonl")
    judgments = _read(run_dir / "judgments.jsonl")
    errors = _read(run_dir / "errors.jsonl")
    api_calls = _read(run_dir / "api_calls.jsonl")

    max_output = _configured_max_output_tokens(started)
    output_tokens = sorted(int(r.get("output_tokens") or 0) for r in responses)
    at_ceiling = (
        sum(1 for t in output_tokens if max_output and t >= _TRUNCATION_HARD * max_output)
        if max_output
        else 0
    )
    refusals = sum(
        1
        for r in responses
        if any(marker in (r.get("answer_text") or "").lower() for marker in REFUSAL_MARKERS)
    )
    cost = sum(float(call.get("cost_usd") or 0.0) for call in api_calls)
    parse_failures = sum(1 for j in judgments if j.get("parse_was_successful") is False)

    progress = RunProgress(
        run_id=run_id,
        profile=str(started.get("methodology_profile", "unknown")),
        n_answered=len(responses),
        n_errored=len(errors),
        n_judge_parse_failures=parse_failures,
        cost_usd=cost,
        max_output_tokens=max_output,
        p95_output_tokens=_percentile(output_tokens, 0.95),
        n_at_token_ceiling=at_ceiling,
        n_refusals=refusals,
    )

    if at_ceiling:
        progress.stop.append(
            f"{at_ceiling} answer(s) ended within {int(_TRUNCATION_HARD * 100)}% of the "
            f"{max_output}-token ceiling: they were cut off, not finished"
        )
    elif max_output and progress.p95_output_tokens >= _TRUNCATION_WARN * max_output:
        progress.warn.append(
            f"p95 output is {progress.p95_output_tokens} of {max_output} tokens; the ceiling is "
            "close enough that later answers may be truncated"
        )
    if errors:
        progress.stop.append(f"{len(errors)} question(s) errored; this stage tolerates none")
    if parse_failures:
        progress.stop.append(f"{parse_failures} judgment(s) failed to parse")
    if expected_cost_usd and progress.n_answered:
        # Projected from what has been spent, not from an estimate made before the run -- the
        # estimate is what was wrong the last time this mattered.
        total_questions = _planned_question_count(started) or progress.n_answered
        projected = cost / progress.n_answered * total_questions
        if projected > 1.5 * expected_cost_usd:
            progress.stop.append(
                f"projected cost ${projected:.2f} is over 150% of the ${expected_cost_usd:.2f} "
                "expected"
            )
        elif projected > 1.2 * expected_cost_usd:
            progress.warn.append(
                f"projected cost ${projected:.2f} against ${expected_cost_usd:.2f} expected"
            )
    return progress


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        decoded = json.loads(line)
        if isinstance(decoded, dict):
            records.append(cast(dict[str, Any], decoded))
    return records


def _first_event(path: Path) -> dict[str, Any]:
    events = _read(path)
    return events[0] if events else {}


def _configured_max_output_tokens(started: dict[str, Any]) -> int | None:
    config = started.get("config")
    if not isinstance(config, dict):
        return None
    typed_config = cast(dict[str, object], config)
    answer_model = typed_config.get("answer_model")
    if not isinstance(answer_model, dict):
        return None
    typed_answer_model = cast(dict[str, object], answer_model)
    value = typed_answer_model.get("max_output_tokens")
    return int(value) if isinstance(value, int) else None


def _planned_question_count(started: dict[str, Any]) -> int | None:
    environment = started.get("runtime_environment")
    if not isinstance(environment, dict):
        return None
    typed_environment = cast(dict[str, object], environment)
    value = typed_environment.get("n_questions_planned")
    return int(value) if isinstance(value, int) else None


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, round(fraction * (len(values) - 1)))
    return values[index]
