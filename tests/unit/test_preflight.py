from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from khedron.config import ExperimentSuiteConfig
from khedron.errors import ConfigurationError
from khedron.preflight import preflight_suite
from khedron.providers.registry import register_provider


@register_provider("retriever_probe")
class _RetrieverProbe:
    """A minimal retrieving provider (``honours_top_k`` True) so the retrieval-budget preflight path
    can be exercised without shipping a concrete memory provider. Never instantiated: preflight
    reads the class attribute to decide whether a configured retrieval budget applies."""

    honours_top_k = True


@pytest.fixture(autouse=True)
def _credentials_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test the authenticated environment it was written against.

    Tests asserting "no blockers" predate the credential check and meant "no blockers *of the kind
    under test*". Supplying the keys here keeps them testing that, and leaves the credential tests
    to unset explicitly what they are about. The values are placeholders and are never sent
    anywhere: preflight checks presence and never reads them.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")


@pytest.fixture(autouse=True)
def _identifiable_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a clean, identifiable build.

    preflight_suite reads the running build's version and blocks a dirty/unidentified one. That
    version depends on the git state of the checkout the tests run from -- outside the tests'
    control, and routinely dirty on a developer's tree. Pinning a clean value keeps every
    unrelated "no blockers" assertion deterministic; the build-identity tests re-patch it.
    """
    monkeypatch.setattr("khedron.preflight.resolve_framework_version", lambda: "0.0.0+testbuild")


def _config(**overrides: Any) -> ExperimentSuiteConfig:
    experiment: dict[str, Any] = {
        "name": "probe",
        "provider": {
            "type": overrides.pop("provider_type", "full_context"),
            "config": overrides.pop("provider_config", {}),
        },
        "benchmark": {
            "type": "locomo",
            "config": overrides.pop(
                "benchmark_config",
                {
                    "categories": [
                        "single_hop",
                        "multi_hop",
                        "temporal",
                        "open_domain",
                        "adversarial",
                    ]
                },
            ),
            "audit_mode": "both",
        },
        "answer_model": {
            "type": "openai",
            "model": overrides.pop("answer_model", "gpt-4o-mini-2024-07-18"),
        },
        "judge": {
            "type": "anthropic",
            "model": overrides.pop("judge_model", "claude-haiku-4-5"),
        },
        "top_k_retrieval": 10,
    }
    payload: dict[str, Any] = {
        "runs": 1,
        "methodology_version": "1.0",
        "methodology_profile": "canonical-v1",
        "max_cost_usd": 5.0,
        "experiments": [experiment],
        **overrides,
    }
    return ExperimentSuiteConfig.model_validate(payload)


def _checks(findings: list[Any], severity: str) -> set[str]:
    return {finding.check for finding in findings if finding.severity == severity}


def test_a_dirty_build_is_a_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncommitted tree cannot be reproduced from a commit, so it must not be paid for."""
    monkeypatch.setattr(
        "khedron.preflight.resolve_framework_version", lambda: "0.0.0+5d989e9.dirty"
    )
    findings = preflight_suite(_config())
    if "build_identity" not in _checks(findings, "blocker"):
        raise AssertionError(findings)


def test_an_unidentified_build_is_a_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source outside any git repository is equally unpinnable, and blocks the same way."""
    monkeypatch.setattr("khedron.preflight.resolve_framework_version", lambda: "0.0.0+unidentified")
    findings = preflight_suite(_config())
    if "build_identity" not in _checks(findings, "blocker"):
        raise AssertionError(findings)


def test_a_clean_build_raises_no_build_identity_finding() -> None:
    """A committed, clean tree (the autouse default) is publishable and raises nothing here."""
    findings = preflight_suite(_config())
    if any(finding.check == "build_identity" for finding in findings):
        raise AssertionError(findings)


def test_unpriceable_model_is_a_blocker() -> None:
    # The failure this prevents lands on the first *paid* call: the adapter turns a
    # missing price into a ModelError, so without preflight the money is already gone.
    findings = preflight_suite(_config(answer_model="gpt-4o-does-not-exist"))
    if "model_pricing" not in _checks(findings, "blocker"):
        raise AssertionError(findings)


def test_priceable_models_produce_no_blocker() -> None:
    findings = preflight_suite(_config())
    if _checks(findings, "blocker"):
        raise AssertionError(findings)


def test_a_profile_that_pins_nothing_is_reported() -> None:
    # canonical-v1 leaves models, top_k and categories unpinned, so it certifies nothing
    # about what was measured -- the defect behind the day-1 result.
    findings = preflight_suite(_config())
    if "methodology_contract" not in _checks(findings, "warning"):
        raise AssertionError(findings)


def test_a_contractual_profile_is_not_reported() -> None:
    findings = preflight_suite(
        _config(
            methodology_profile="canonical-v2",
            answer_model="gpt-4o",
            experiments=[
                {
                    "name": "probe",
                    "provider": {"type": "full_context", "config": {}},
                    "benchmark": {
                        "type": "locomo",
                        "config": {
                            "categories": ["multi_hop", "temporal", "open_domain", "single_hop"]
                        },
                        "audit_mode": "standard",
                    },
                    "answer_model": {"type": "openai", "model": "gpt-4o"},
                    "judge": {"type": "openai", "model": "gpt-4o"},
                    "top_k_retrieval": 200,
                }
            ],
        )
    )
    if "methodology_contract" in _checks(findings, "warning"):
        raise AssertionError(findings)


def test_environment_budget_override_is_reported_not_the_config_value(
    monkeypatch: Any,
) -> None:
    # KHEDRON_MAX_BUDGET_USD is what the runner enforces. Reading the config alone made
    # preflight describe a cap the run would not use -- and claim spend was unbounded when
    # the environment had in fact capped it.
    monkeypatch.setenv("KHEDRON_MAX_BUDGET_USD", "2.5")
    findings = preflight_suite(_config(max_cost_usd=None))
    budget = [finding for finding in findings if finding.check == "budget"]
    if not budget or "2.5" not in budget[0].detail:
        raise AssertionError(findings)
    if any("unbounded" in finding.detail for finding in budget):
        raise AssertionError(f"reported unbounded despite an env cap: {findings}")


def test_an_expired_price_is_a_blocker() -> None:
    # An expired introductory price does not fail, it undercounts: the tracker keeps
    # computing and the budget cap enforces a ceiling in the wrong currency. The only
    # symptom is artifacts recording a cost that was never charged.
    findings = preflight_suite(_config(judge_model="claude-sonnet-5"), today=date(2026, 9, 1))
    if "model_pricing" not in _checks(findings, "blocker"):
        raise AssertionError(findings)
    if not any("data/pricing/anthropic.yaml" in finding.detail for finding in findings):
        raise AssertionError("the finding must name the file to edit")


def test_a_price_is_not_blocked_on_its_last_valid_day() -> None:
    findings = preflight_suite(_config(judge_model="claude-sonnet-5"), today=date(2026, 8, 31))
    if _checks(findings, "blocker"):
        raise AssertionError(findings)


def test_a_run_count_below_the_aggregate_minimum_is_reported() -> None:
    # The publication guard admits an aggregate over one run so a smoke test still produces
    # something. That makes the run count the only thing between a smoke test and a number
    # that looks like a measurement, and preflight is where it can still be changed.
    findings = preflight_suite(_config(runs=1))
    if "run_count" not in _checks(findings, "warning"):
        raise AssertionError(findings)
    if "run_count" in _checks(preflight_suite(_config(runs=3)), "warning"):
        raise AssertionError("three runs meet the documented minimum")


def test_an_invalid_environment_budget_fails_closed(monkeypatch: Any) -> None:
    # Fail closed: a typo in the variable must not quietly run the suite uncapped.
    monkeypatch.setenv("KHEDRON_MAX_BUDGET_USD", "five dollars")
    try:
        preflight_suite(_config())
    except ConfigurationError:
        return
    raise AssertionError("an unparseable cap must raise, not be ignored")


def test_unmetered_embedder_and_missing_budget_are_reported() -> None:
    findings = preflight_suite(_config(provider_config={"embedder": "openai"}, max_cost_usd=None))
    budget = [finding for finding in findings if finding.check == "budget"]
    if len(budget) != 2:
        raise AssertionError(findings)


def test_a_provider_that_ignores_top_k_blocks_a_profile_that_pins_it() -> None:
    # A run must not record a retrieval budget it never applied.
    # `FullContextProvider` discards top_k by design, so pinning one cannot describe a
    # full-context baseline. Blocked at preflight rather than at run start, because the operator
    # should learn before launching rather than after.
    findings = preflight_suite(
        _config(
            methodology_profile="canonical-v2",
            methodology_version="2.0",
            provider_type="full_context",
        )
    )

    blockers = [finding for finding in findings if finding.check == "retrieval_budget"]
    if not blockers or blockers[0].severity != "blocker":
        raise AssertionError(findings)
    # The message has to say what to do instead.
    if "unpinned" not in blockers[0].detail:
        raise AssertionError(blockers[0].detail)


def test_a_retrieving_provider_raises_no_budget_finding() -> None:
    # Guards the opposite failure: a check that blocks every configuration is as useless as none.
    findings = preflight_suite(
        _config(
            methodology_profile="canonical-v2",
            methodology_version="2.0",
            provider_type="retriever_probe",
        )
    )

    if any(finding.check == "retrieval_budget" for finding in findings):
        raise AssertionError(findings)


def test_a_restricted_corpus_is_reported() -> None:
    # Found by testing a minimal case: three runs over one conversation preflighted with zero
    # findings, so a smoke test would have produced artifacts indistinguishable from a real
    # measurement. Wilson intervals over 105 questions are arithmetically valid and say nothing.
    findings = preflight_suite(
        _config(
            benchmark_config={
                "conversations": ["conv-30"],
                "categories": [
                    "single_hop",
                    "multi_hop",
                    "temporal",
                    "open_domain",
                    "adversarial",
                ],
            }
        )
    )

    scope = [finding for finding in findings if finding.check == "corpus_scope"]
    if not scope:
        raise AssertionError(findings)
    if "must not be published" not in scope[0].detail:
        raise AssertionError(scope[0].detail)


def test_a_full_corpus_suite_raises_no_scope_finding() -> None:
    findings = preflight_suite(_config())

    if any(finding.check == "corpus_scope" for finding in findings):
        raise AssertionError(findings)


def test_a_missing_credential_is_a_blocker(monkeypatch: Any) -> None:
    # The most basic form of "validates and is still unrunnable", and preflight did not check it:
    # a run can clear preflight with warnings and still die on run_started -> run_failed having
    # measured nothing.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")

    findings = preflight_suite(_config(), today=date(2026, 8, 3))

    blockers = [f for f in findings if f.severity == "blocker" and f.check == "credentials"]
    if len(blockers) != 1:
        raise AssertionError([str(f) for f in findings])
    if "OPENAI_API_KEY" not in blockers[0].detail:
        raise AssertionError(blockers[0].detail)
    # Names the variable, never its contents -- the judge's key is set here and must not appear.
    if "present" in blockers[0].detail:
        raise AssertionError("the finding leaked a credential value")


def test_a_missing_judge_credential_is_reported_separately(monkeypatch: Any) -> None:
    # The expensive shape, and the reason both roles are checked rather than just the first call.
    # A missing judge key fails only when an answer is scored, which on a suite that answers a
    # whole corpus first means paying for every answer and scoring none of them.
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    findings = preflight_suite(_config(), today=date(2026, 8, 3))

    blockers = [f for f in findings if f.severity == "blocker" and f.check == "credentials"]
    if len(blockers) != 1 or "judge" not in blockers[0].detail:
        raise AssertionError([str(f) for f in findings])
    if "ANTHROPIC_API_KEY" not in blockers[0].detail:
        raise AssertionError(blockers[0].detail)


def test_present_credentials_produce_no_blocker(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")

    findings = preflight_suite(_config(), today=date(2026, 8, 3))

    if "credentials" in _checks(findings, "blocker"):
        raise AssertionError([str(f) for f in findings])


def test_a_per_experiment_key_override_is_honoured(monkeypatch: Any) -> None:
    # A suite pointing one vendor at a second key is exactly the configuration a hardcoded
    # default would clear wrongly: the default variable is set, the one actually read is not.
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.delenv("OPENAI_API_KEY_SECONDARY", raising=False)

    config = _config()
    payload = config.model_dump()
    payload["experiments"][0]["answer_model"]["config"] = {
        "api_key_env_var": "OPENAI_API_KEY_SECONDARY"
    }
    findings = preflight_suite(
        ExperimentSuiteConfig.model_validate(payload), today=date(2026, 8, 3)
    )

    blockers = [f for f in findings if f.severity == "blocker" and f.check == "credentials"]
    if len(blockers) != 1 or "OPENAI_API_KEY_SECONDARY" not in blockers[0].detail:
        raise AssertionError([str(f) for f in findings])
