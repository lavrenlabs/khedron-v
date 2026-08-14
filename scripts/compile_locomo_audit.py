from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

SOURCE_NAME: Final = "dial481/locomo-audit"
SOURCE_URL: Final = "https://github.com/dial481/locomo-audit"
ERRORS_URL: Final = "https://raw.githubusercontent.com/dial481/locomo-audit/main/errors.json"
DEFAULT_OUTPUT_PATH: Final = (
    Path(__file__).resolve().parents[1] / "data" / "audits" / "locomo_errors.json"
)

SCHEMA_VERSION: Final = 1
WRONG_CITATION: Final = "WRONG_CITATION"
EXCLUDED_ERROR_TYPES: Final = (WRONG_CITATION,)
EXPECTED_UPSTREAM_TOTAL_COUNT = 156
EXPECTED_EXCLUDED_COUNT = 57
EXPECTED_INCLUDED_COUNT = 99

OPTIONAL_TEXT_FIELDS: Final = ("question", "golden_answer", "correct_answer")
OPTIONAL_EVIDENCE_FIELDS: Final = ("cited_evidence", "correct_evidence")
QUESTION_ID_PATTERN: Final = re.compile(r"^locomo_(\d+)_qa(\d+)$")


class CompileLocomoAuditError(RuntimeError):
    """Raised when LoCoMo audit data cannot be compiled safely."""


def _validate_retrieved_at(retrieved_at: str) -> str:
    try:
        parsed = dt.date.fromisoformat(retrieved_at)
    except ValueError as exc:
        raise CompileLocomoAuditError(
            f"retrieved_at must use YYYY-MM-DD format, got {retrieved_at!r}."
        ) from exc

    if parsed.isoformat() != retrieved_at:
        raise CompileLocomoAuditError(
            f"retrieved_at must use YYYY-MM-DD format, got {retrieved_at!r}."
        )
    return retrieved_at


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _required_string(
    entry: Mapping[str, object],
    field_name: str,
    entry_index: int,
) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CompileLocomoAuditError(
            f"Audit entry {entry_index} is missing required string field {field_name!r}."
        )
    return value


def _optional_string(
    entry: Mapping[str, object],
    field_name: str,
    entry_index: int,
) -> str:
    value = entry.get(field_name, "")
    if not isinstance(value, str):
        raise CompileLocomoAuditError(
            f"Audit entry {entry_index} field {field_name!r} must be a string when present."
        )
    return value


def _optional_int(
    entry: Mapping[str, object],
    field_name: str,
    entry_index: int,
) -> int | None:
    if field_name not in entry:
        return None

    value = entry[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileLocomoAuditError(
            f"Audit entry {entry_index} field {field_name!r} must be an integer when present."
        )
    return value


def _optional_string_list(
    entry: Mapping[str, object],
    field_name: str,
    entry_index: int,
) -> list[str] | None:
    if field_name not in entry:
        return None

    value = entry[field_name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CompileLocomoAuditError(
            f"Audit entry {entry_index} field {field_name!r} must be a list of strings."
        )
    return list(value)


def _compiled_error(
    entry: Mapping[str, object], entry_index: int, retrieved_at: str
) -> dict[str, object]:
    compiled: dict[str, object] = {
        "question_id": _required_string(entry, "question_id", entry_index),
        "error_type": _required_string(entry, "error_type", entry_index),
    }

    category = _optional_int(entry, "category", entry_index)
    if category is not None:
        compiled["category"] = category

    for field_name in OPTIONAL_TEXT_FIELDS:
        compiled[field_name] = _optional_string(entry, field_name, entry_index)

    compiled["explanation"] = _required_string(entry, "reasoning", entry_index)
    compiled["audited_at"] = retrieved_at

    for field_name in OPTIONAL_EVIDENCE_FIELDS:
        evidence = _optional_string_list(entry, field_name, entry_index)
        if evidence is not None:
            compiled[field_name] = evidence

    return compiled


def _sort_key(error: Mapping[str, object]) -> tuple[int, int, str]:
    question_id = error["question_id"]
    if not isinstance(question_id, str):
        raise CompileLocomoAuditError("Compiled audit error has non-string question_id.")
    match = QUESTION_ID_PATTERN.fullmatch(question_id)
    if match is None:
        return (sys.maxsize, sys.maxsize, question_id)
    return (int(match.group(1)), int(match.group(2)), question_id)


def _verify_counts(
    upstream_total_count: int,
    excluded_count: int,
    included_count: int,
) -> None:
    expected = (
        EXPECTED_UPSTREAM_TOTAL_COUNT,
        EXPECTED_EXCLUDED_COUNT,
        EXPECTED_INCLUDED_COUNT,
    )
    observed = (upstream_total_count, excluded_count, included_count)
    if observed != expected:
        raise CompileLocomoAuditError(
            "LoCoMo audit count mismatch: "
            f"expected total/excluded/included {expected}, observed {observed}."
        )


def compile_audit_errors(
    raw_errors: list[dict[str, object]],
    retrieved_at: str,
) -> dict[str, object]:
    retrieved_at = _validate_retrieved_at(retrieved_at)

    included_errors: list[dict[str, object]] = []
    excluded_count = 0

    for entry_index, entry in enumerate(raw_errors):
        error_type = _required_string(entry, "error_type", entry_index)
        if error_type == WRONG_CITATION:
            excluded_count += 1
            continue
        included_errors.append(_compiled_error(entry, entry_index, retrieved_at))

    included_errors.sort(key=_sort_key)
    upstream_total_count = len(raw_errors)
    included_count = len(included_errors)
    _verify_counts(upstream_total_count, excluded_count, included_count)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "errors_url": ERRORS_URL,
            "retrieved_at": retrieved_at,
            "upstream_total_count": upstream_total_count,
            "excluded_error_types": list(EXCLUDED_ERROR_TYPES),
            "excluded_count": excluded_count,
            "included_count": included_count,
        },
        "errors": included_errors,
    }


def _decode_raw_errors(payload: bytes) -> list[dict[str, object]]:
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, list):
        raise CompileLocomoAuditError("Upstream LoCoMo audit payload must be a JSON list.")

    raw_errors: list[dict[str, object]] = []
    for entry_index, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise CompileLocomoAuditError(
                f"Upstream LoCoMo audit entry {entry_index} must be a JSON object."
            )
        if not all(isinstance(key, str) for key in entry):
            raise CompileLocomoAuditError(
                f"Upstream LoCoMo audit entry {entry_index} has a non-string key."
            )
        raw_errors.append(cast(dict[str, object], entry))
    return raw_errors


def _download_raw_errors() -> list[dict[str, object]]:
    with urllib.request.urlopen(ERRORS_URL, timeout=30) as response:  # noqa: S310
        payload = response.read()
    return _decode_raw_errors(payload)


def write_audit_file(output_path: Path, artifact: dict[str, object]) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile score-corrupting LoCoMo audit errors into the v1 local artifact."
    )
    parser.add_argument(
        "--retrieved-at",
        default=dt.date.today().isoformat(),
        help="Retrieval date in YYYY-MM-DD format. Defaults to today's date.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    args = parser.parse_args(argv)

    try:
        raw_errors = _download_raw_errors()
        artifact = compile_audit_errors(raw_errors, args.retrieved_at)
        output_path = write_audit_file(args.output_path, artifact)
    except (
        CompileLocomoAuditError,
        json.JSONDecodeError,
        OSError,
        urllib.error.URLError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    source = cast(dict[str, object], artifact["source"])
    print(
        "Compiled LoCoMo audit data: "
        f"{source['included_count']} included, "
        f"{source['excluded_count']} excluded, "
        f"{source['upstream_total_count']} upstream total."
    )
    print(f"LoCoMo audit artifact saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
