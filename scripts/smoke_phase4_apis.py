from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from khedron.errors import KhedronError
from khedron.judges import AnthropicJudge, GoogleJudge, OpenAIJudge
from khedron.models import (
    AnthropicModel,
    AnthropicModelConfig,
    GoogleModel,
    GoogleModelConfig,
    OpenAIModel,
    OpenAIModelConfig,
)
from khedron.models.base import AnswerModel
from khedron.tls import configure_os_trust_store
from khedron.types import JudgmentVerdict
from khedron.utils.tokens import estimate_tokens

Vendor = Literal["openai", "anthropic", "google"]
Status = Literal["passed", "failed", "blocked"]

DEFAULT_VENDORS: tuple[Vendor, ...] = ("openai", "anthropic", "google")
API_KEY_ENV: dict[Vendor, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}
SMOKE_MODEL_ENV: dict[Vendor, str] = {
    "openai": "KHEDRON_OPENAI_SMOKE_MODEL",
    "anthropic": "KHEDRON_ANTHROPIC_SMOKE_MODEL",
    "google": "KHEDRON_GOOGLE_SMOKE_MODEL",
}
DEFAULT_SMOKE_MODELS: dict[Vendor, str] = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-2.5-flash-lite",
}
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


@dataclass(frozen=True)
class SmokeResult:
    vendor: Vendor
    check: str
    status: Status
    detail: str
    cost_usd: float = 0.0


def _model_id(vendor: Vendor) -> str:
    value = os.environ.get(SMOKE_MODEL_ENV[vendor])
    if value is None or not value.strip():
        return DEFAULT_SMOKE_MODELS[vendor]
    return value


def _api_key_is_present(vendor: Vendor) -> bool:
    value = os.environ.get(API_KEY_ENV[vendor])
    return value is not None and bool(value.strip())


def _make_model(vendor: Vendor, model_id: str) -> AnswerModel:
    if vendor == "openai":
        return OpenAIModel(OpenAIModelConfig(model_id=model_id))
    if vendor == "anthropic":
        return AnthropicModel(AnthropicModelConfig(model_id=model_id))
    if vendor == "google":
        return GoogleModel(GoogleModelConfig(model_id=model_id))
    raise ValueError(f"Unsupported vendor: {vendor}")


def _make_judge(vendor: Vendor, model_id: str) -> OpenAIJudge | AnthropicJudge | GoogleJudge:
    if vendor == "openai":
        return OpenAIJudge(OpenAIModelConfig(model_id=model_id), max_output_tokens=192)
    if vendor == "anthropic":
        return AnthropicJudge(AnthropicModelConfig(model_id=model_id), max_output_tokens=192)
    if vendor == "google":
        return GoogleJudge(GoogleModelConfig(model_id=model_id), max_output_tokens=192)
    raise ValueError(f"Unsupported vendor: {vendor}")


async def _smoke_answer_model(vendor: Vendor, model_id: str) -> SmokeResult:
    model = _make_model(vendor, model_id)
    try:
        result = await model.generate(
            "Reply with exactly: OK",
            max_output_tokens=16,
            temperature=0.0,
        )
        if not result.output.strip():
            raise RuntimeError("empty model output")
        return SmokeResult(
            vendor=vendor,
            check="answer_model.generate",
            status="passed",
            detail=(
                f"model={result.model_id}; input_tokens={result.input_tokens}; "
                f"output_tokens={result.output_tokens}; cost_usd={result.cost_usd:.8f}"
            ),
            cost_usd=result.cost_usd,
        )
    except Exception as exc:
        return _failed_result(vendor, "answer_model.generate", exc)
    finally:
        await model.close()


async def _smoke_judge(vendor: Vendor, model_id: str) -> SmokeResult:
    judge = _make_judge(vendor, model_id)
    try:
        result = await judge.evaluate(
            question="What city is the expected answer?",
            expected_answer="Rome",
            generated_answer="Rome",
            category="single_hop",
        )
        if result.verdict is not JudgmentVerdict.CORRECT:
            raise RuntimeError(f"identity judge returned {result.verdict.value}")
        return SmokeResult(
            vendor=vendor,
            check="judge.evaluate",
            status="passed",
            detail=(
                f"model={result.api_call.model_id}; verdict={result.verdict.value}; "
                f"score={result.score:.3f}; cost_usd={result.api_call.cost_usd:.8f}"
            ),
            cost_usd=result.api_call.cost_usd,
        )
    except Exception as exc:
        return _failed_result(vendor, "judge.evaluate", exc)
    finally:
        await judge.close()


def _smoke_token_estimation(vendor: Vendor) -> SmokeResult:
    try:
        count = estimate_tokens("hello world", vendor)
        if count <= 0:
            raise RuntimeError(f"nonpositive token estimate: {count}")
        return SmokeResult(
            vendor=vendor,
            check="estimate_tokens",
            status="passed",
            detail=f"tokens={count}",
        )
    except Exception as exc:
        return _failed_result(vendor, "estimate_tokens", exc)


def _blocked_results_for_missing_key(vendor: Vendor) -> list[SmokeResult]:
    key_name = API_KEY_ENV[vendor]
    return [
        SmokeResult(
            vendor=vendor,
            check="api_key",
            status="blocked",
            detail=f"missing required environment variable {key_name}",
        )
    ]


def _failed_result(vendor: Vendor, check: str, exc: Exception) -> SmokeResult:
    error_name = type(exc).__name__
    if isinstance(exc, KhedronError):
        detail = str(exc)
    else:
        detail = f"{error_name}: {exc}"
    return SmokeResult(vendor=vendor, check=check, status="failed", detail=detail)


async def run_smoke(vendors: Sequence[Vendor], *, max_cost_usd: float) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    total_cost = 0.0

    for vendor in vendors:
        if not _api_key_is_present(vendor):
            results.extend(_blocked_results_for_missing_key(vendor))
            continue

        model_id = _model_id(vendor)
        for result in (
            await _smoke_answer_model(vendor, model_id),
            await _smoke_judge(vendor, model_id),
            _smoke_token_estimation(vendor),
        ):
            results.append(result)
            total_cost += result.cost_usd
            if total_cost > max_cost_usd:
                results.append(
                    SmokeResult(
                        vendor=vendor,
                        check="cost_guard",
                        status="blocked",
                        detail=(
                            f"smoke cost guard exceeded: {total_cost:.8f} > {max_cost_usd:.8f}"
                        ),
                    )
                )
                return results

    return results


def write_report(results: Sequence[SmokeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_cost = sum(result.cost_usd for result in results)
    status = "passed" if all(result.status == "passed" for result in results) else "not passed"
    lines = [
        "# Phase 4 Real API Smoke Report",
        "",
        f"- Timestamp: {datetime.now(UTC).isoformat()}",
        f"- Overall status: {status}",
        f"- Reported API cost from adapters: {total_cost:.8f} USD",
        "- Secrets: not recorded",
        "",
        "| Vendor | Check | Status | Detail | Cost USD |",
        "|---|---|---|---|---:|",
    ]
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {result.vendor} | {result.check} | {result.status} | "
            f"{detail} | {result.cost_usd:.8f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_vendors(raw_vendors: Sequence[str]) -> tuple[Vendor, ...]:
    vendors: list[Vendor] = []
    for raw_vendor in raw_vendors:
        normalized = raw_vendor.strip().lower()
        if normalized not in DEFAULT_VENDORS:
            raise argparse.ArgumentTypeError(f"Unsupported vendor: {raw_vendor}")
        vendors.append(cast(Vendor, normalized))
    return tuple(vendors)


def _default_report_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_REPORT_DIR / f"phase-4-real-api-smoke-{timestamp}.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal real-API smoke test for Phase 4 adapters."
    )
    parser.add_argument(
        "--vendor",
        choices=DEFAULT_VENDORS,
        action="append",
        dest="vendors",
        help="Vendor to smoke. May be passed multiple times. Defaults to all vendors.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=1.0,
        help="Stop if adapter-reported cost exceeds this amount. Default: 1.0.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_default_report_path(),
        help="Markdown report path.",
    )
    args = parser.parse_args(argv)

    # This script builds model/judge/token clients directly, bypassing the Khedron CLI,
    # so it must enable OS-trust-store verification itself before any SSL context is
    # created. Without it, a machine behind HTTPS inspection fails every call with
    # CERTIFICATE_VERIFY_FAILED even though the OS trusts the intercepting CA.
    configure_os_trust_store()

    vendors = _parse_vendors(args.vendors or list(DEFAULT_VENDORS))
    results = asyncio.run(run_smoke(vendors, max_cost_usd=args.max_cost_usd))
    write_report(results, args.report)

    for result in results:
        sys.stdout.write(f"{result.vendor} {result.check}: {result.status} - {result.detail}\n")
    sys.stdout.write(f"Report: {args.report}\n")

    return 0 if all(result.status == "passed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
