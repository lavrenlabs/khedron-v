from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_verify_setup_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "verify_setup.py"
    spec = importlib.util.spec_from_file_location("verify_setup_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_setup = load_verify_setup_module()


def test_check_python_version_accepts_minimum_and_rejects_older() -> None:
    passed, _ = verify_setup.check_python_version((3, 11))
    if not passed:
        raise AssertionError("3.11 must satisfy the minimum version")
    passed, detail = verify_setup.check_python_version((3, 10))
    if passed:
        raise AssertionError(detail)


def test_check_khedron_importable_in_test_environment() -> None:
    passed, detail = verify_setup.check_khedron_importable()
    if not passed:
        raise AssertionError(detail)


def test_check_dataset_missing_file_fails(tmp_path: Path) -> None:
    passed, detail = verify_setup.check_dataset(tmp_path / "locomo10.json")
    if passed:
        raise AssertionError(detail)
    if "missing" not in detail:
        raise AssertionError(detail)


def test_check_dataset_checksum_mismatch_fails(tmp_path: Path) -> None:
    dataset = tmp_path / "locomo10.json"
    dataset.write_bytes(b"not the canonical dataset")

    passed, detail = verify_setup.check_dataset(dataset)

    if passed:
        raise AssertionError(detail)
    if "checksum mismatch" not in detail:
        raise AssertionError(detail)


def test_check_dataset_matching_checksum_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from khedron.benchmarks import locomo

    payload = b"fake canonical dataset"
    dataset = tmp_path / "locomo10.json"
    dataset.write_bytes(payload)
    monkeypatch.setattr(
        locomo,
        "EXPECTED_DATASET_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    passed, detail = verify_setup.check_dataset(dataset)

    if not passed:
        raise AssertionError(detail)


def test_check_env_keys_reports_names_never_values() -> None:
    secret_value = "sk-super-secret-value"  # noqa: S105 - fake value proving non-disclosure
    environ = {"OPENAI_API_KEY": secret_value, "ANTHROPIC_API_KEY": "sk-ant-secret"}

    passed, detail = verify_setup.check_env_keys(environ)

    if not passed:
        raise AssertionError(detail)
    if secret_value in detail or "sk-ant-secret" in detail:
        raise AssertionError("check output must never contain key values")
    if "OPENAI_API_KEY" not in detail:
        raise AssertionError(detail)


def test_check_env_keys_missing_required_fails() -> None:
    passed, detail = verify_setup.check_env_keys({"OPENAI_API_KEY": "x"})

    if passed:
        raise AssertionError(detail)
    if "ANTHROPIC_API_KEY" not in detail:
        raise AssertionError(detail)


def test_check_env_keys_empty_value_counts_as_missing() -> None:
    passed, detail = verify_setup.check_env_keys({"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "x"})

    if passed:
        raise AssertionError(detail)
    if "OPENAI_API_KEY" not in detail:
        raise AssertionError(detail)
