from __future__ import annotations

import json
from pathlib import Path

from khedron.progress import read_progress


def _write(run_dir: Path, name: str, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run(tmp_path: Path, *, output_tokens: list[int], max_output: int = 256) -> Path:
    run_dir = tmp_path / "runs" / "R"
    _write(
        run_dir,
        "lifecycle.jsonl",
        [
            {
                "event_type": "run_started",
                "methodology_profile": "canonical-v3",
                "config": {"answer_model": {"max_output_tokens": max_output}},
                "runtime_environment": {},
            }
        ],
    )
    _write(
        run_dir,
        "responses.jsonl",
        [
            {"output_tokens": tokens, "answer_text": f"answer {index}"}
            for index, tokens in enumerate(output_tokens)
        ],
    )
    return tmp_path


def test_an_answer_at_its_token_ceiling_is_a_reason_to_kill_the_run(tmp_path: Path) -> None:
    # The check that catches a real failure mode: a run whose median output sits exactly at the
    # token ceiling means most answers were cut off mid-reasoning and scored as though they were
    # finished. It shows on the first handful of questions.
    results = _run(tmp_path, output_tokens=[22, 31, 256, 44])

    report = read_progress(results, "R")

    if not report.stop:
        raise AssertionError("a truncated answer did not stop the run")
    if "cut off, not finished" not in report.stop[0]:
        raise AssertionError(report.stop)


def test_a_healthy_run_reports_nothing_to_stop_for(tmp_path: Path) -> None:
    results = _run(tmp_path, output_tokens=[22, 31, 19, 44])

    report = read_progress(results, "R")

    if report.stop or report.warn:
        raise AssertionError((report.stop, report.warn))
    if report.n_answered != 4:
        raise AssertionError(report.n_answered)


def test_approaching_the_ceiling_warns_before_it_stops(tmp_path: Path) -> None:
    # 80% of the budget is not yet a defect, but it is the last moment a longer answer would still
    # fit. Warning there is the difference between noticing and paying.
    results = _run(tmp_path, output_tokens=[20, 30, 40, 210])

    report = read_progress(results, "R")

    if report.stop:
        raise AssertionError(report.stop)
    if not report.warn:
        raise AssertionError("approaching the ceiling produced no warning")


def test_refusals_are_counted_with_the_registered_detector(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "R"
    _run(tmp_path, output_tokens=[10, 10])
    _write(
        run_dir,
        "responses.jsonl",
        [
            {"output_tokens": 10, "answer_text": "Blue"},
            {"output_tokens": 10, "answer_text": "I don't have enough information for that."},
        ],
    )

    report = read_progress(tmp_path, "R")

    if report.n_refusals != 1 or abs(report.refusal_rate - 0.5) > 1e-9:
        raise AssertionError((report.n_refusals, report.refusal_rate))
