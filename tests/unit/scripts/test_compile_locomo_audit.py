from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load_compiler_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "compile_locomo_audit.py"
    spec = importlib.util.spec_from_file_location("compile_locomo_audit_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compiler = load_compiler_module()


def set_expected_counts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total: int,
    excluded: int,
    included: int,
) -> None:
    monkeypatch.setattr(compiler, "EXPECTED_UPSTREAM_TOTAL_COUNT", total)
    monkeypatch.setattr(compiler, "EXPECTED_EXCLUDED_COUNT", excluded)
    monkeypatch.setattr(compiler, "EXPECTED_INCLUDED_COUNT", included)


def included_entry(question_id: str, error_type: str = "HALLUCINATION") -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": f"Question for {question_id}?",
        "golden_answer": "published answer",
        "category": 2,
        "error_type": error_type,
        "cited_evidence": ["D1:1"],
        "correct_evidence": ["D1:2"],
        "reasoning": f"Reasoning for {question_id}.",
        "correct_answer": "corrected answer",
    }


def wrong_citation_entry(question_id: str) -> dict[str, object]:
    entry = included_entry(question_id, error_type="WRONG_CITATION")
    entry["reasoning"] = "Citation is wrong but the answer is correct."
    return entry


def test_compile_excludes_wrong_citation_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_expected_counts(monkeypatch, total=3, excluded=1, included=2)
    raw_errors = [
        included_entry("locomo_0_qa10", error_type="TEMPORAL_ERROR"),
        wrong_citation_entry("locomo_0_qa1"),
        included_entry("locomo_0_qa2"),
    ]

    artifact = compiler.compile_audit_errors(raw_errors, "2026-05-05")
    source = artifact["source"]
    errors = artifact["errors"]

    if artifact["schema_version"] != 1:
        raise AssertionError(artifact["schema_version"])
    if source["upstream_total_count"] != 3:
        raise AssertionError(source)
    if source["excluded_error_types"] != ["WRONG_CITATION"]:
        raise AssertionError(source)
    if source["excluded_count"] != 1:
        raise AssertionError(source)
    if source["included_count"] != 2:
        raise AssertionError(source)
    if [error["question_id"] for error in errors] != ["locomo_0_qa2", "locomo_0_qa10"]:
        raise AssertionError(errors)
    if any(error["error_type"] == "WRONG_CITATION" for error in errors):
        raise AssertionError(errors)
    if errors[0]["explanation"] != "Reasoning for locomo_0_qa2.":
        raise AssertionError(errors[0])
    if errors[0]["audited_at"] != "2026-05-05":
        raise AssertionError(errors[0])


def test_compile_defaults_missing_optional_text_fields_to_empty_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_expected_counts(monkeypatch, total=1, excluded=0, included=1)
    raw_errors = [
        {
            "question_id": "locomo_0_qa2",
            "error_type": "HALLUCINATION",
            "reasoning": "The published answer is unsupported.",
        }
    ]

    artifact = compiler.compile_audit_errors(raw_errors, "2026-05-05")
    error = artifact["errors"][0]

    if error["question"] != "":
        raise AssertionError(error)
    if error["golden_answer"] != "":
        raise AssertionError(error)
    if error["correct_answer"] != "":
        raise AssertionError(error)
    if "category" in error:
        raise AssertionError(error)


def test_compile_fails_clearly_when_required_field_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_expected_counts(monkeypatch, total=1, excluded=0, included=1)
    raw_errors = [
        {
            "question_id": "locomo_0_qa2",
            "error_type": "HALLUCINATION",
        }
    ]

    with pytest.raises(compiler.CompileLocomoAuditError, match="reasoning"):
        compiler.compile_audit_errors(raw_errors, "2026-05-05")


def test_compile_hard_stops_on_unexpected_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_expected_counts(monkeypatch, total=156, excluded=57, included=99)

    with pytest.raises(compiler.CompileLocomoAuditError, match="count mismatch"):
        compiler.compile_audit_errors([included_entry("locomo_0_qa2")], "2026-05-05")


def test_main_writes_temp_output_with_mocked_source_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_expected_counts(monkeypatch, total=2, excluded=1, included=1)
    raw_errors = [
        wrong_citation_entry("locomo_0_qa1"),
        included_entry("locomo_0_qa2"),
    ]

    def mocked_download() -> list[dict[str, object]]:
        return raw_errors

    monkeypatch.setattr(compiler, "_download_raw_errors", mocked_download)
    output_path = tmp_path / "locomo_errors.json"

    exit_code = compiler.main(["--retrieved-at", "2026-05-05", "--output-path", str(output_path)])

    if exit_code != 0:
        raise AssertionError(exit_code)

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    if artifact["source"]["included_count"] != 1:
        raise AssertionError(artifact)
    if artifact["errors"][0]["question_id"] != "locomo_0_qa2":
        raise AssertionError(artifact)
