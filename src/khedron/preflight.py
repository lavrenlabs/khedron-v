from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import cast

from khedron.budget import BUDGET_ENV_VAR, resolve_budget_cap_usd
from khedron.build import is_unidentifiable_build, resolve_framework_version
from khedron.config import ExperimentSuiteConfig
from khedron.cost.pricing import compute_cost_usd, load_vendor_pricing
from khedron.errors import ConfigurationError
from khedron.methodology import get_runtime_profile
from khedron.providers.registry import get_provider_class
from khedron.utils.redaction import REDACTED, looks_like_credential

# The advisory minimum for a multi-run AGGREGATE (a mean with measured run-to-run dispersion). A
# single-pass N=1 result is admissible as a labelled non-aggregate descriptive score:
# a sample SD is defined for any n >= 2 (only n = 1 is degenerate), so the reason to prefer >= 3 for
# an aggregate is instability, not impossibility -- and for a fixed-corpus temperature-0 evaluation
# additional runs estimate execution variation rather than narrowing the corpus interval.
_MIN_RUNS_FOR_AGGREGATE = 3

# The variable each adapter reads, mirroring the `api_key_env_var` default on its config model.
# Kept as a table here rather than by importing the adapters, because preflight must answer for a
# vendor whose adapter may not be importable in this environment; a vendor absent from this table
# is reported as unchecked rather than as satisfied.
_VENDOR_CREDENTIAL_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

__all__ = ["PreflightFinding", "preflight_suite"]


@dataclass(frozen=True)
class PreflightFinding:
    """One reason a suite should not be executed, or should be executed knowingly."""

    severity: str  # "blocker" | "warning"
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


def preflight_suite(
    config: ExperimentSuiteConfig, *, today: date | None = None
) -> list[PreflightFinding]:
    """Report what would make a run fail or produce an unpublishable result.

    Why: every defect this catches was discovered *after* spending money. A configuration
    can pass validation, pass the methodology profile, and still be unrunnable (a model the
    pricing table cannot resolve fails on the first paid call) or unpublishable (a profile
    that pins nothing cannot certify what was measured). Blockers must be fixed; warnings
    are for the operator to accept deliberately rather than discover in the artifacts.
    """
    findings: list[PreflightFinding] = []
    findings.extend(_check_build_is_identifiable(resolve_framework_version()))
    findings.extend(_check_credentials_are_present(config))
    findings.extend(_check_models_are_priceable(config))
    findings.extend(_check_prices_are_current(config, today=today or date.today()))
    findings.extend(_check_profile_is_contractual(config))
    findings.extend(_check_run_count_supports_an_aggregate(config))
    findings.extend(_check_retrieval_budget_applies(config))
    findings.extend(_check_corpus_is_complete(config))
    findings.extend(_check_spend_is_bounded(config))
    return findings


def _check_build_is_identifiable(framework_version: str) -> list[PreflightFinding]:
    """Refuse a run whose build cannot be named.

    Every run stamps itself with the version that produced it (``khedron.build``). A dirty working
    tree, or a source that is not in a git repository, yields a ``.dirty``/``+unidentified``
    version: the exact code cannot be established, so the result can never be reproduced from a
    commit and -- under the measurement-card contract and the resume guard (``resume.py`` refuses
    to continue such a build) -- is not a publishable measurement.

    A full-corpus baseline paid for and run from an uncommitted tree records a `.dirty` version
    and can never be published or even resumed, wasting real time and money on an unusable
    artifact. Blocking here converts that into a free, immediate refusal. The remedy is one line
    for the operator: commit or stash every change, then run. The
    override (`--allow-unidentifiable-build`) exists only for a deliberate decision to spend on a
    run that is knowingly un-publishable.
    """
    if is_unidentifiable_build(framework_version):
        return [
            PreflightFinding(
                severity="blocker",
                check="build_identity",
                detail=(
                    f"the framework build is unidentifiable ({framework_version}), so any result "
                    "could not state what produced it and cannot be published or resumed. Commit "
                    "or stash every change and run from a clean tree. Override only with a "
                    "deliberate decision to spend on an un-publishable run."
                ),
            )
        ]
    return []


def _check_credentials_are_present(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    """Refuse a suite whose models cannot authenticate.

    The most basic form of "passes validation and is still unrunnable", and preflight did not
    check it: a suite with no key in the environment cleared preflight with two warnings and then
    died on `run_started → run_failed` having measured nothing.

    The expensive shape is not the one that happened. A *missing answer key* fails on the first
    call, wasting time. A missing **judge** key fails only when the first answer is scored -- and
    on a suite that ingests before it judges, that can be after the whole corpus has been answered
    and paid for, with nothing scored to show for it. So both roles are checked, and both are
    blockers.

    Presence only. The value is never read, compared or reported: the check is `in os.environ`
    with a non-empty value, and the finding names the variable, never its contents.
    """
    findings: list[PreflightFinding] = []
    seen: set[tuple[str, str]] = set()
    for experiment in config.experiments:
        for role, vendor, plugin_config in (
            ("answer_model", experiment.answer_model.type, experiment.answer_model.config),
            ("judge", experiment.judge.type, experiment.judge.config),
        ):
            env_var = _credential_env_var(vendor, plugin_config)
            if env_var is None:
                # An adapter that authenticates some other way, or none at all. Silence is right:
                # inventing a variable name for it would produce a blocker nobody could satisfy.
                continue
            if (role, env_var) in seen:
                continue
            seen.add((role, env_var))
            if not (os.environ.get(env_var) or "").strip():
                findings.append(
                    PreflightFinding(
                        severity="blocker",
                        check="credentials",
                        detail=(
                            f"{experiment.name}: the {role} is {vendor!r}, which reads its key "
                            f"from {env_var}, and that variable is unset or empty. The run would "
                            f"fail on its first {role} call."
                        ),
                    )
                )
    return findings


def _credential_env_var(vendor: str, plugin_config: dict[str, object]) -> str | None:
    """Name the variable an adapter would read, honouring a per-experiment override.

    Reading the override rather than assuming the default, because a suite that points one vendor
    at a second key is exactly the configuration a hardcoded default would clear wrongly.
    """
    override = plugin_config.get("api_key_env_var")
    if isinstance(override, str) and override.strip():
        return override.strip()
    return _VENDOR_CREDENTIAL_ENV_VARS.get(vendor)


def _check_models_are_priceable(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    findings: list[PreflightFinding] = []
    for experiment in config.experiments:
        for role, vendor, model_id in (
            ("answer_model", experiment.answer_model.type, experiment.answer_model.model),
            ("judge", experiment.judge.type, experiment.judge.model),
        ):
            try:
                pricing = load_vendor_pricing(vendor)
            except OSError:
                # No table for this vendor at all. That is the shape of a custom or
                # in-process adapter that never bills, so it is not a blocker -- but it is
                # also how a typo'd vendor looks, so say so rather than staying silent.
                findings.append(
                    PreflightFinding(
                        severity="warning",
                        check="model_pricing",
                        detail=(
                            f"{experiment.name}: {role} vendor '{vendor}' has no pricing "
                            "table, so its spend cannot be recorded. Expected for a custom "
                            "adapter; a typo otherwise."
                        ),
                    )
                )
                continue
            try:
                compute_cost_usd(model_id, input_tokens=1000, output_tokens=100, pricing=pricing)
            except ValueError as exc:
                # The vendor is known and this model is missing from its table: the
                # adapter will turn that into a ModelError on the first paid call.
                findings.append(
                    PreflightFinding(
                        severity="blocker",
                        check="model_pricing",
                        detail=(
                            f"{experiment.name}: {role} '{model_id}' ({vendor}) has no usable "
                            f"price, so the first paid call raises ModelError ({exc})."
                        ),
                    )
                )
    return findings


def _check_prices_are_current(
    config: ExperimentSuiteConfig, *, today: date
) -> list[PreflightFinding]:
    """Block on a price whose announced validity has passed.

    Why a blocker: an expired introductory price does not fail, it undercounts. The tracker
    keeps computing, the budget cap enforces a ceiling in the wrong currency, and the
    artifacts record a cost that was never charged -- all silently. Refusing before any money
    moves turns that into one editable YAML line. A run already under way is not interrupted,
    so a suite that crosses the expiry date mid-flight still bills at the stale figure.
    """
    findings: list[PreflightFinding] = []
    for experiment in config.experiments:
        for role, vendor, model_id in (
            ("answer_model", experiment.answer_model.type, experiment.answer_model.model),
            ("judge", experiment.judge.type, experiment.judge.model),
        ):
            try:
                pricing = load_vendor_pricing(vendor)
            except OSError:
                continue  # reported by _check_models_are_priceable
            entry = pricing.find_for_model(model_id)
            if entry is None or entry.price_valid_until is None:
                continue
            if entry.price_valid_until >= today:
                continue
            findings.append(
                PreflightFinding(
                    severity="blocker",
                    check="model_pricing",
                    detail=(
                        f"{experiment.name}: {role} '{model_id}' ({vendor}) is priced from a "
                        f"table valid only through {entry.price_valid_until.isoformat()}, so "
                        "recorded spend and the budget cap would both be wrong. Update "
                        f"data/pricing/{vendor}.yaml."
                    ),
                )
            )
    return findings


def _provider_ignores_top_k(provider_type: str) -> bool:
    try:
        provider_cls = get_provider_class(provider_type)
    except (KeyError, ConfigurationError):
        return False
    return not getattr(provider_cls, "honours_top_k", True)


def _check_corpus_is_complete(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    """Warn when a suite measures only part of the corpus.

    A restricted corpus can preflight with zero findings even though its artifacts would be
    indistinguishable from a full measurement: the `runs < 3` check does not fire, and nothing
    else looked at how much of the corpus was covered.

    A restricted corpus is a legitimate and useful thing -- rehearsals and smoke tests exist for
    good reasons -- but a run that measured one conversation must not produce a number that reads
    like a run that measured ten. Wilson intervals over 105 questions are arithmetically valid and
    say nothing about the system.
    """
    findings: list[PreflightFinding] = []
    for experiment in config.experiments:
        conversations = experiment.benchmark.config.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        conversation_ids = cast(list[object], conversations)
        findings.append(
            PreflightFinding(
                severity="warning",
                check="corpus_scope",
                detail=(
                    f"{experiment.name}: restricted to {len(conversation_ids)} conversation(s) "
                    f"({', '.join(str(item) for item in conversation_ids)}), so this measures "
                    "part of "
                    "the corpus. Valid for a smoke test; its scores must not be "
                    "published or compared against a full-corpus run."
                ),
            )
        )
    return findings


def _check_retrieval_budget_applies(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    """Block a suite whose provider would ignore the profile's pinned retrieval budget.

    The runner refuses this too, but refusing at run start means the operator learns after
    launching. Preflight exists to say it before any money moves, and the capability is a class
    attribute, so no provider has to be constructed to know the answer.

    This is the contract's section 3 in practice: `FullContextProvider` discards `top_k` by design,
    so a profile pinning a budget cannot describe a full-context baseline. That baseline
    needs a profile that leaves the budget unpinned -- it is measuring something different, and a
    different fingerprint is the honest outcome rather than a nuisance.
    """
    profile = get_runtime_profile(config.methodology_profile)
    if profile.top_k_retrieval is None:
        return []
    findings: list[PreflightFinding] = []
    for experiment in config.experiments:
        if not _provider_ignores_top_k(experiment.provider.type):
            continue
        findings.append(
            PreflightFinding(
                severity="blocker",
                check="retrieval_budget",
                detail=(
                    f"{experiment.name}: provider '{experiment.provider.type}' ignores top_k by "
                    f"design, but profile '{profile.name}' pins it at {profile.top_k_retrieval}. "
                    "The run would record a retrieval budget it never applied. Use a profile that "
                    "leaves top_k_retrieval unpinned for this baseline."
                ),
            )
        )
    return findings


def _check_run_count_supports_an_aggregate(
    config: ExperimentSuiteConfig,
) -> list[PreflightFinding]:
    """Warn when the suite cannot produce a publishable aggregate however well it runs.

    Why: the publication guard admits an aggregate over as few as one run, because refusing
    would leave a single-run smoke test with no result at all. That makes the run count the
    only thing standing between a smoke test and a number that looks like a measurement, and
    it is set before any money is spent -- which is where it should be said out loud.
    """
    if config.runs >= _MIN_RUNS_FOR_AGGREGATE:
        return []
    return [
        PreflightFinding(
            severity="warning",
            check="run_count",
            detail=(
                f"runs={config.runs} is below the {_MIN_RUNS_FOR_AGGREGATE} for a multi-run "
                "aggregate, so its run-to-run dispersion is under-determined at this run count. It "
                "may be published only as a labelled non-aggregate result, never as a measured "
                "multi-run aggregate."
            ),
        )
    ]


def _check_profile_is_contractual(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    profile = get_runtime_profile(config.methodology_profile)
    # An unpinned retrieval budget is only a gap when something would have used it. Every provider
    # in this suite ignoring `top_k` is exactly the case `canonical-v2-baseline` exists for, and
    # reporting it as "certifies nothing" would attach a permanent warning to a correct
    # configuration -- which is how operators learn to skim warnings and miss the real one.
    budget_is_moot = profile.top_k_retrieval is None and all(
        _provider_ignores_top_k(experiment.provider.type) for experiment in config.experiments
    )
    unpinned = [
        name
        for name in (
            "answer_model_id",
            "judge_model_id",
            "top_k_retrieval",
            "evaluation_categories",
            "scored_categories",
        )
        if getattr(profile, name, None) is None
        and not (name == "top_k_retrieval" and budget_is_moot)
    ]
    if not unpinned:
        return []
    return [
        PreflightFinding(
            severity="warning",
            check="methodology_contract",
            detail=(
                f"Profile '{profile.name}' leaves {', '.join(unpinned)} unpinned, so it "
                "constrains nothing there and cannot certify what was measured. Results are "
                "not comparable across runs that chose different values."
            ),
        )
    ]


def _check_spend_is_bounded(config: ExperimentSuiteConfig) -> list[PreflightFinding]:
    findings: list[PreflightFinding] = []
    # The cap the run will actually enforce, not the one written in the YAML:
    # KHEDRON_MAX_BUDGET_USD overrides it, so reading the config alone would report spend as
    # unbounded when the environment had capped it, or name a cap the environment had
    # lowered. A preflight that misdescribes the coming run is worse than none.
    effective_cap_usd = resolve_budget_cap_usd(config.max_cost_usd)
    if effective_cap_usd is None:
        findings.append(
            PreflightFinding(
                severity="warning",
                check="budget",
                detail="No effective spend cap is set, so nothing halts the suite on spend.",
            )
        )
    elif config.max_cost_usd != effective_cap_usd:
        findings.append(
            PreflightFinding(
                severity="warning",
                check="budget",
                detail=(
                    f"{BUDGET_ENV_VAR} overrides the configured cap: "
                    f"{effective_cap_usd} applies, not {config.max_cost_usd}."
                ),
            )
        )
    for experiment in config.experiments:
        embedder = experiment.provider.config.get("embedder")
        if isinstance(embedder, str) and embedder not in ("", "none"):
            # From a plugin bag, the one place a credential legitimately lives, and this
            # string is printed to stdout. Naming it is diagnostics; echoing a credential
            # would be the leak.
            embedder = REDACTED if looks_like_credential(embedder) else embedder
            findings.append(
                PreflightFinding(
                    severity="warning",
                    check="budget",
                    detail=(
                        f"{experiment.name}: provider embedder '{embedder}' bills your "
                        "account through a channel the cost tracker does not observe, so "
                        "max_cost_usd does not bound it."
                    ),
                )
            )
    return findings
