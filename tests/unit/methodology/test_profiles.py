from __future__ import annotations

# ruff: noqa: S101
import re
from dataclasses import fields, replace

import pytest

from khedron.config import (
    AnswerModelConfig,
    BenchmarkConfig,
    ExperimentConfig,
    ExperimentSuiteConfig,
    JudgeConfig,
    ProviderConfig,
)
from khedron.errors import ConfigurationError
from khedron.methodology import (
    get_runtime_profile,
    methodology_fingerprint,
    validate_suite_methodology_profile,
)

_REFERENCE_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
_HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def test_canonical_v1_profile_resolves_with_prompt_artifacts() -> None:
    profile = get_runtime_profile("canonical-v1")

    assert profile.name == "canonical-v1"
    assert profile.status == "ready"
    assert profile.generator_prompt_path.exists()
    assert profile.judge_prompt_path.exists()
    assert _HEX_SHA256.match(methodology_fingerprint(profile))


def _replication_suite(
    *,
    benchmark: BenchmarkConfig | None = None,
    answer_model: AnswerModelConfig | None = None,
    judge: JudgeConfig | None = None,
    top_k_retrieval: int = 200,
) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig(
        methodology_profile="canonical-v2",
        experiments=[
            ExperimentConfig(
                name="v3 replication contract",
                provider=ProviderConfig(type="full_context"),
                benchmark=benchmark
                or BenchmarkConfig(
                    type="locomo",
                    audit_mode="standard",
                    config={
                        "categories": [
                            "multi_hop",
                            "temporal",
                            "open_domain",
                            "single_hop",
                        ]
                    },
                ),
                answer_model=answer_model or AnswerModelConfig(type="openai", model="gpt-4o"),
                judge=judge or JudgeConfig(type="openai", model="gpt-4o"),
                top_k_retrieval=top_k_retrieval,
            )
        ],
    )


def test_fingerprint_is_sensitive_to_the_scored_category_set() -> None:
    # The case DoR review had to add. "Differs from canonical-v1" and "is stable across runs" both
    # pass whether or not these fields enter the hashed payload -- and if they do not, two profiles
    # differing only in what they score share an identity, which is canonical-v1's defect rebuilt
    # inside its successor. This is the assertion that actually constrains the implementation.
    base = get_runtime_profile("canonical-v2")
    narrower = replace(base, scored_categories=("single_hop",))

    if methodology_fingerprint(base) == methodology_fingerprint(narrower):
        raise AssertionError("two profiles scoring different sets share a fingerprint")


def test_fingerprint_is_sensitive_to_the_evaluated_category_set() -> None:
    base = get_runtime_profile("canonical-v2")
    narrower = replace(
        base,
        evaluation_categories=("single_hop", "multi_hop", "temporal"),
        scored_categories=("single_hop",),
    )

    if methodology_fingerprint(base) == methodology_fingerprint(narrower):
        raise AssertionError("two profiles evaluating different sets share a fingerprint")


def test_every_behavioural_profile_field_reaches_the_fingerprint() -> None:
    """Catch a field added to the profile and forgotten in the hashed payload.

    An earlier version of this test asserted the payload version was at least 2, which is a
    literal checked against itself and proves nothing. The real risk is drift: someone adds a
    behaviour-affecting field, does not hash it, and two profiles that measure different things
    start sharing an identity -- exactly what `canonical-v1` did with its category set.

    So this varies each field in turn and records which ones leave the hash unchanged. `status`
    is the one deliberate exclusion: publication readiness is a judgement about a profile, not a
    property of the measurement, and hashing it would change a run's identity when we change our
    minds. Anything else appearing here is a bug in `methodology_fingerprint`.
    """
    base = get_runtime_profile("canonical-v2")
    baseline = methodology_fingerprint(base)
    other_prompt = get_runtime_profile("canonical-v1").generator_prompt_path
    variations: dict[str, object] = {
        "name": "other-name",
        "version": "other-version",
        "reference_name": "other-reference",
        "reference_url": "https://example.invalid/other",
        "reference_commit": "0" * 40,
        "generator_prompt_path": other_prompt,
        "answer_marker": "OTHER-ANSWER:",
        "image_description_policy": "blip_caption_only_v1",
        "judge_prompt_path": other_prompt,
        "benchmark_type": "other-benchmark",
        "evaluation_categories": ("single_hop",),
        "scored_categories": ("single_hop",),
        "audit_mode": "audited",
        "answer_model_type": "other-vendor",
        "answer_model_id": "other-model",
        "judge_type": "other-vendor",
        "judge_model_id": "other-judge",
        "top_k_retrieval": 7,
        "top_k_cutoffs": (1, 2),
        "scoring": "other-scoring",
        "aggregation": "other-aggregation",
        "status": "reserved",
        "requires_methodology_version": "9.9",
        "superseded_by": "some-successor",
    }
    declared = {field.name for field in fields(base)}
    if declared != set(variations):
        raise AssertionError(
            "profile fields changed without updating this sweep; unlisted: "
            f"{sorted(declared - set(variations))}, stale: {sorted(set(variations) - declared)}"
        )

    unhashed = set()
    for name, value in variations.items():
        if name == "evaluation_categories":
            # Narrowing what is asked while still scoring the wider set is an invalid profile and
            # __post_init__ refuses to build it, so this field has to be varied together with the
            # scored set. The `scored_categories` case below isolates that half on its own.
            candidate = replace(
                base, evaluation_categories=("single_hop",), scored_categories=("single_hop",)
            )
        elif isinstance(value, bool):
            # Negated rather than set to a constant: a boolean sweep value that happens to equal
            # the base profile's is a no-op, and the field is then reported as unhashed when it is
            # merely unchanged. `ingests_image_descriptions` hit exactly that once the replication
            # profiles turned it on.
            candidate = replace(base, **{name: not getattr(base, name)})
        else:
            candidate = replace(base, **{name: value})
        if methodology_fingerprint(candidate) == baseline:
            unhashed.add(name)

    # The allowlist: identity and governance are deliberately outside the hash. Renaming a profile
    # or declaring its successor must not change the fingerprint of runs whose measurement was
    # untouched -- `canonical-v3` arriving would otherwise invalidate every canonical-v2 hash for a
    # reason that is not a measurement. `status` was excluded on this argument first.
    expected_unhashed = {
        "status",
        "name",
        "version",
        "reference_name",
        "reference_url",
        "reference_commit",
        "superseded_by",
        "requires_methodology_version",
    }
    if unhashed != expected_unhashed:
        raise AssertionError(
            f"fields absent from the fingerprint: {sorted(unhashed - expected_unhashed)}; "
            f"unexpectedly hashed: {sorted(expected_unhashed - unhashed)}"
        )


def test_a_profile_cannot_score_a_category_it_does_not_evaluate() -> None:
    # Scoring an unasked category computes a score over questions nobody was asked -- a silent zero
    # rather than an error. Checked at construction because profiles are module constants: an
    # incoherent one should be impossible to build, not merely rejected later.
    base = get_runtime_profile("canonical-v2")

    try:
        # Narrowed to one category while scoring another: the scored set is then outside the
        # evaluated set, which is the incoherence. Scoring a category the base already evaluates
        # proves nothing.
        replace(base, evaluation_categories=("single_hop",), scored_categories=("multi_hop",))
    except ConfigurationError as error:
        if "multi_hop" not in str(error):
            raise AssertionError(error) from error
        return
    raise AssertionError("a profile scoring an unevaluated category was accepted")


def test_the_two_profile_name_literals_cannot_drift_apart() -> None:
    # `config.MethodologyProfile` and `profiles.ProfileName` list the same names in two places,
    # because importing one into the other would create a cycle. Adding a profile to one and not
    # the other produces a config that validates and then fails to resolve, or a profile nobody can
    # select -- so the duplication is guarded here instead of being left to memory.
    from typing import get_args

    from khedron.config import MethodologyProfile
    from khedron.methodology.profiles import ProfileName

    if set(get_args(MethodologyProfile)) != set(get_args(ProfileName)):
        raise AssertionError(
            f"config: {sorted(get_args(MethodologyProfile))} != "
            f"profiles: {sorted(get_args(ProfileName))}"
        )


def test_canonical_v2_pins_every_field_the_contract_fixes() -> None:
    # The defect being repaired was a profile that named a methodology and pinned none of it, so
    # the test that matters is that nothing the contract fixes is left open.
    profile = get_runtime_profile("canonical-v2")

    unpinned = [
        name
        for name in (
            "evaluation_categories",
            "scored_categories",
            "answer_model_type",
            "answer_model_id",
            "judge_type",
            "judge_model_id",
            "top_k_retrieval",
            "requires_methodology_version",
        )
        if getattr(profile, name) is None
    ]
    if unpinned:
        raise AssertionError(f"canonical-v2 leaves contract fields open: {unpinned}")
    if profile.top_k_cutoffs != (10, 20, 50, 200):
        raise AssertionError(profile.top_k_cutoffs)
    if profile.scored_categories == profile.evaluation_categories:
        raise AssertionError("adversarial must be evaluated and not scored")
    if "adversarial" not in profile.evaluation_categories:
        raise AssertionError("adversarial must still be asked; it is the diagnostic")
    if "adversarial" in profile.scored_categories:
        raise AssertionError("adversarial must not enter the headline")


def test_canonical_v1_is_refused_and_names_its_successor() -> None:
    from khedron.config import ExperimentSuiteConfig

    suite = ExperimentSuiteConfig.model_validate(
        {
            "methodology_profile": "canonical-v1",
            "methodology_version": "1.0",
            "experiments": [
                {
                    "name": "old",
                    "provider": {"type": "full_context"},
                    "benchmark": {"type": "locomo"},
                    "answer_model": {"type": "openai", "model": "m"},
                    "judge": {"type": "anthropic", "model": "j"},
                }
            ],
        }
    )

    with pytest.raises(ConfigurationError) as exc_info:
        validate_suite_methodology_profile(suite)

    # A refusal that does not say what to use instead is a support ticket.
    if "canonical-v2" not in str(exc_info.value):
        raise AssertionError(exc_info.value)


def test_an_unpinned_profile_field_does_not_demand_the_config_also_be_none() -> None:
    # `None` means the profile does not constrain the field. Comparing it directly made a
    # partially-pinned profile reject every valid configuration -- unnoticed because only
    # fully-pinned profiles ever reached that path.
    from khedron.config import ExperimentSuiteConfig
    from khedron.methodology import profiles as profiles_module

    partial = replace(
        get_runtime_profile("canonical-v2"),
        name="partial",
        superseded_by=None,
        answer_model_id=None,
        judge_model_id=None,
        top_k_retrieval=None,
    )
    suite = ExperimentSuiteConfig.model_validate(
        {
            "methodology_profile": "canonical-v2",
            "methodology_version": "2.0",
            "experiments": [
                {
                    "name": "partial",
                    "provider": {"type": "full_context"},
                    "benchmark": {
                        "type": "locomo",
                        "config": {
                            "categories": [
                                "single_hop",
                                "multi_hop",
                                "temporal",
                                "open_domain",
                                "adversarial",
                            ]
                        },
                        "audit_mode": "both",
                    },
                    "answer_model": {"type": "openai", "model": "anything-at-all"},
                    "judge": {"type": "anthropic", "model": "anything-else"},
                    "top_k_retrieval": 37,
                }
            ],
        }
    )

    original = profiles_module.get_runtime_profile
    profiles_module.get_runtime_profile = lambda name: partial  # type: ignore[assignment]
    try:
        validate_suite_methodology_profile(suite)
    finally:
        profiles_module.get_runtime_profile = original  # type: ignore[assignment]


# The retrieval axis is the one under investigation, so a baseline that differs from its arm on
# anything else is not a baseline -- it is a second experiment reported as a control. Pinned as a
# derived pair rather than by listing each profile's fields: the fields drift, the relationship is
# what must not.
@pytest.mark.parametrize(
    ("arm", "baseline"),
    [("canonical-v2", "canonical-v2-baseline"), ("canonical-v3", "canonical-v3-baseline")],
)
def test_a_baseline_differs_from_its_arm_only_on_the_retrieval_axis(
    arm: str, baseline: str
) -> None:
    left = get_runtime_profile(arm)
    right = get_runtime_profile(baseline)

    differing = {
        field.name
        for field in fields(left)
        if getattr(left, field.name) != getattr(right, field.name)
    }
    # `scoring` and the two identity fields describe the profile; they carry no measurement
    # semantics of their own and every derived profile restates them.
    allowed = {"name", "reference_name", "scoring", "top_k_retrieval", "top_k_cutoffs"}
    if not differing <= allowed:
        raise AssertionError(
            f"{baseline} diverges from {arm} beyond retrieval: {sorted(differing - allowed)}"
        )
    # And it must actually differ where it is supposed to, or the pair is not a contrast at all.
    if right.top_k_retrieval is not None or right.top_k_cutoffs != ():
        raise AssertionError(f"{baseline} still declares a retrieval budget")
    if left.top_k_retrieval is None:
        raise AssertionError(f"{arm} declares no retrieval budget, so the pair contrasts nothing")


def test_the_v3_baseline_carries_the_v3_corpus_and_prompts() -> None:
    # The specific way the arm could be wrong while the test above still passed: derived from
    # canonical-v2-baseline instead of canonical-v3, it would keep the image-free corpus, the
    # scripted-refusal generator and the superseded rubric -- and compare v3 against a v2 control
    # while calling itself v3.
    profile = get_runtime_profile("canonical-v3-baseline")
    v3 = get_runtime_profile("canonical-v3")

    if profile.image_description_policy != v3.image_description_policy:
        raise AssertionError(profile.image_description_policy)
    if profile.generator_prompt_path != v3.generator_prompt_path:
        raise AssertionError(profile.generator_prompt_path)
    if profile.judge_prompt_path != v3.judge_prompt_path:
        raise AssertionError(profile.judge_prompt_path)
    if methodology_fingerprint(profile) == methodology_fingerprint(v3):
        raise AssertionError("the two arms must not share a fingerprint")
