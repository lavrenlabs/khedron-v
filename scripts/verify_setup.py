"""Verify the local environment is ready for a Khedron benchmark run.

Minimal pre-flight check, in order:

1. Python version is at least 3.11.
2. The ``khedron`` package is importable from the active environment.
3. The LoCoMo dataset exists and matches the framework's expected SHA-256
   checksum (``EXPECTED_DATASET_SHA256`` in ``khedron.benchmarks.locomo``).
4. Required API key environment variables are set. Only presence and the
   variable NAMES are reported; values are never printed.

Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path

MIN_PYTHON_VERSION = (3, 11)
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo" / "locomo10.json"
# The canonical run configs (experiments/quickstart.yaml)
# use OpenAI answer models and an Anthropic judge, so both keys are required.
REQUIRED_ENV_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
OPTIONAL_ENV_KEYS = ("GOOGLE_API_KEY",)
CHUNK_SIZE_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_python_version(version: tuple[int, int] | None = None) -> tuple[bool, str]:
    observed = version if version is not None else sys.version_info[:2]
    expected = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    observed_text = ".".join(str(part) for part in observed)
    if observed >= MIN_PYTHON_VERSION:
        return True, f"Python {observed_text} (>= {expected})"
    return False, f"Python {observed_text} is older than the required {expected}"


def check_khedron_importable() -> tuple[bool, str]:
    try:
        module = importlib.import_module("khedron")
    except Exception as exc:  # report any import failure as a failed check, do not crash
        return False, f"khedron is not importable: {type(exc).__name__}: {exc}"
    location = getattr(module, "__file__", None) or "(unknown location)"
    return True, f"khedron importable from {location}"


def check_dataset(dataset_path: Path | None = None) -> tuple[bool, str]:
    path = dataset_path if dataset_path is not None else DEFAULT_DATASET_PATH
    try:
        locomo_module = importlib.import_module("khedron.benchmarks.locomo")
        expected = str(locomo_module.EXPECTED_DATASET_SHA256).lower()
    except Exception as exc:  # checksum source unavailable is a failed check, not a crash
        return False, (
            "Cannot load the expected dataset checksum from khedron.benchmarks.locomo: "
            f"{type(exc).__name__}: {exc}"
        )
    if not path.is_file():
        return False, (
            f"LoCoMo dataset missing at {path}; run: uv run python scripts/download_locomo.py"
        )
    try:
        observed = sha256_file(path)
    except OSError as exc:
        return False, f"Cannot read LoCoMo dataset at {path}: {exc}"
    if observed != expected:
        return False, (
            f"LoCoMo dataset checksum mismatch at {path}: expected {expected}, observed {observed}"
        )
    return True, f"LoCoMo dataset present at {path} with expected checksum"


def check_env_keys(environ: Mapping[str, str] | None = None) -> tuple[bool, str]:
    env = environ if environ is not None else os.environ
    missing = [key for key in REQUIRED_ENV_KEYS if not env.get(key)]
    optional_missing = [key for key in OPTIONAL_ENV_KEYS if not env.get(key)]
    parts: list[str] = []
    if missing:
        parts.append(f"missing required: {', '.join(missing)}")
    else:
        parts.append(f"required set: {', '.join(REQUIRED_ENV_KEYS)}")
    if optional_missing:
        parts.append(f"optional not set: {', '.join(optional_missing)}")
    return not missing, "; ".join(parts)


def main() -> int:
    checks = (
        ("python", check_python_version()),
        ("khedron import", check_khedron_importable()),
        ("locomo dataset", check_dataset()),
        ("api keys", check_env_keys()),
    )
    failed = False
    for name, (passed, detail) in checks:
        status = "ok" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            failed = True
    if failed:
        print("Setup verification FAILED.", file=sys.stderr)
        return 1
    print("Setup verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
