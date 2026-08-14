from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, get_args

from khedron.errors import ConfigurationError
from khedron.methodology.archived import archived_profile

if TYPE_CHECKING:
    from khedron.config import ExperimentConfig, ExperimentSuiteConfig

__all__ = [
    "METHODOLOGY_FINGERPRINT_KEY",
    "RUNNABLE_PROFILE_NAMES",
    "MethodologyRuntimeProfile",
    "get_runtime_profile",
    "methodology_fingerprint",
    "validate_suite_methodology_profile",
]

ProfileName = Literal[
    "canonical-v1",
    "canonical-v2",
    "canonical-v2-baseline",
    "canonical-v3",
    "canonical-v3-baseline",
    "canonical-v3-generator-only",
    "canonical-v3-prior-judge",
    "snap-original",
]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_PROMPTS_DIR: Final[Path] = _PROJECT_ROOT / "data" / "prompts"

# Version of the hashed payload's shape, not of any profile. 2 recorded the category split; 3
# records the move to a positive allowlist of measurement-affecting fields, which drops identity and
# governance from the hash and renames the image field to a named policy. `answer_marker` and
# `ingests_image_descriptions` were added at version 2 *without* a bump, which is why two runs under
# one profile name carry different hashes at the same schema_version -- the omission this number
# exists to prevent, made once already.
FINGERPRINT_PAYLOAD_VERSION: Final[int] = 3
# The one key under which a run records which methodology produced it. A constant because it was
# not one: the runner wrote `methodology_profile_fingerprint`, resume read `methodology_fingerprint`
# and rejudge wrote that second name. So resume's fingerprint guard never fired -- it skips a
# recorded `None` and always found one -- and a rejudged run's report disclosed the fingerprint it
# inherited from its source rather than its own. Two readers, two writers, three spellings.
METHODOLOGY_FINGERPRINT_KEY: Final[str] = "methodology_profile_fingerprint"
# LoCoMo's five categories, and the four that canonical-v2 scores. Adversarial is evaluated and
# excluded from the headline: the always-refuse floor equals the observed score there, so it
# carries no measurable signal about memory (contract section 1.1).
_LOCOMO_ALL_CATEGORIES: Final[tuple[str, ...]] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
)
_LOCOMO_ANSWERABLE_CATEGORIES: Final[tuple[str, ...]] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
)


@dataclass(frozen=True)
class MethodologyRuntimeProfile:
    """Behavior-affecting methodology bundle used by runner and reports."""

    name: str
    version: str
    status: Literal["ready", "reserved"]
    reference_name: str
    reference_url: str
    reference_commit: str | None
    generator_prompt_path: Path
    # The token this profile's generator prompt requires its final answer to follow, or None when
    # the prompt asks for a bare answer. A prompt that instructs the model to reason in steps and
    # commit after `ANSWER:` produces an answer that is the text after that marker; scoring the
    # whole transcript scores the reasoning instead, and a permissive rubric will award it credit
    # for facts mentioned on the way to a conclusion the model never wrote.
    answer_marker: str | None
    # How the descriptions of shared images enter the corpus, if at all. 20.8% of LoCoMo turns
    # carry one and Khedron ingested none, so a score measured under "none" belongs to a corpus
    # missing a fifth of its content.
    #
    # A named policy rather than a boolean, because a boolean cannot tell two renderings apart: the
    # same `True` produced a third party's format and ours, and the fingerprint saw no difference.
    # It would have let the corpus change without the hash noticing -- the defect this whole field
    # exists to prevent, inside the fix for another one. `blip_caption_only_v1` renders the caption
    # and discards the dataset's search query.
    image_description_policy: Literal["none", "blip_caption_only_v1"]
    judge_prompt_path: Path
    benchmark_type: str
    # What is asked. Renamed from `benchmark_categories`, which controlled question loading while
    # reading as though it also controlled scoring -- the ambiguity that made "ask adversarial but
    # keep it out of the headline" inexpressible.
    evaluation_categories: tuple[str, ...] | None
    # What enters `overall_*`. `None` means every evaluated category, preserving the behaviour of
    # every profile written before the split.
    scored_categories: tuple[str, ...] | None
    audit_mode: Literal["standard", "audited", "both"]
    answer_model_type: str | None
    answer_model_id: str | None
    judge_type: str | None
    judge_model_id: str | None
    top_k_retrieval: int | None
    top_k_cutoffs: tuple[int, ...]
    scoring: str
    aggregation: str
    # The suite `methodology_version` this profile requires, or None when the profile does not
    # constrain it. Set for the canonical profiles, whose version *is* our methodology version;
    # left None for replication profiles, whose `version` identifies the reference being
    # reproduced rather than a version of this project's methodology. Conflating the two rejected
    # every valid replication suite.
    requires_methodology_version: str | None
    # Name of the profile that replaces this one, or None while it is current. Expressed as data
    # rather than as `if profile.name == ...` in the validator: a name check is a list that drifts,
    # and the refusal message can name the successor without the validator knowing which profiles
    # exist.
    superseded_by: str | None

    def __post_init__(self) -> None:
        """Refuse a profile that scores a category it never asks.

        Checked at construction rather than at validation, because a profile is a module constant:
        an incoherent one should be impossible to build, not merely rejected later. Scoring a
        category outside the evaluated set would compute a score over questions that were never
        put to the system -- a silent zero rather than an error.
        """
        if self.scored_categories is None or self.evaluation_categories is None:
            return
        unasked = set(self.scored_categories) - set(self.evaluation_categories)
        if unasked:
            raise ConfigurationError(
                "Methodology profile scores categories it does not evaluate.",
                methodology_profile=self.name,
                unasked_categories=sorted(unasked),
            )


_CANONICAL_V1_PROFILE: Final[MethodologyRuntimeProfile] = MethodologyRuntimeProfile(
    name="canonical-v1",
    version="1.0",
    status="ready",
    reference_name="Khedron canonical methodology v1.0",
    reference_url="docs/METHODOLOGY.md",
    reference_commit=None,
    generator_prompt_path=_PROMPTS_DIR / "generator_canonical_v1.txt",
    answer_marker=None,
    image_description_policy="none",
    judge_prompt_path=_PROMPTS_DIR / "judge_v1.txt",
    benchmark_type="locomo",
    evaluation_categories=None,
    scored_categories=None,
    audit_mode="both",
    answer_model_type=None,
    answer_model_id=None,
    judge_type=None,
    judge_model_id=None,
    top_k_retrieval=None,
    top_k_cutoffs=(),
    scoring="canonical binary CORRECT-only scoring with Wilson 95% CI",
    aggregation="micro average with pooled multi-run Wilson intervals",
    requires_methodology_version="1.0",
    superseded_by="canonical-v2",
)

# Every value here is fixed by the v2 verification contract, written and reviewed before this
# profile existed. That ordering is the point: a profile whose values are chosen after seeing a
# score is a rationalisation, not a contract.
_CANONICAL_V2_PROFILE: Final[MethodologyRuntimeProfile] = MethodologyRuntimeProfile(
    name="canonical-v2",
    version="2.0",
    status="ready",
    reference_name="Khedron canonical methodology v2.0",
    reference_url="docs/METHODOLOGY.md",
    reference_commit=None,
    # Renders the session date the v1 formatter dropped, which is why temporal questions scored
    # near zero under v1 however well retrieval worked.
    generator_prompt_path=_PROMPTS_DIR / "generator_canonical_v2.txt",
    answer_marker=None,
    image_description_policy="none",
    # Unchanged from v1 on purpose: the judge's date tolerance is a separate question, and changing
    # generator and judge together would leave neither effect attributable (contract section 5).
    judge_prompt_path=_PROMPTS_DIR / "judge_v1.txt",
    benchmark_type="locomo",
    # Adversarial is still asked -- it is the diagnostic that exposed the problem -- and kept out of
    # the headline, because a system that answers nothing scores 93.72% there and the always-refuse
    # floor equals the observed score to the digit (contract section 1.1).
    evaluation_categories=_LOCOMO_ALL_CATEGORIES,
    scored_categories=_LOCOMO_ANSWERABLE_CATEGORIES,
    audit_mode="both",
    answer_model_type="openai",
    answer_model_id="gpt-4o-mini-2024-07-18",
    # A different vendor than the answer model, and version-pinned rather than a rolling alias, so a
    # vendor cannot change the measurement without anyone editing a file.
    judge_type="anthropic",
    judge_model_id="claude-haiku-4-5-20251001",
    # 200 covers a substantially larger share of a conversation than the initial default did,
    # matching the reference implementation's budget. Chosen deliberately against measured cost.
    top_k_retrieval=200,
    top_k_cutoffs=(10, 20, 50, 200),
    scoring="canonical binary CORRECT-only scoring with Wilson 95% CI",
    aggregation=(
        "micro average over the answerable subset (categories 1-4) with pooled multi-run "
        "Wilson intervals; adversarial reported per category only"
    ),
    requires_methodology_version="2.0",
    superseded_by=None,
)

# canonical-v2 with the retrieval budget unpinned, for providers that do not retrieve a subset.
#
# `FullContextProvider` discards `top_k` by design: it stands in for passing the whole
# history to the model, so honouring a budget would destroy the baseline it exists to be. Running
# it under `canonical-v2` would record a budget it never applied, which preflight refuses.
#
# Its fingerprint therefore differs from `canonical-v2`'s, and that is the honest outcome rather
# than a nuisance: the two measure different things on the retrieval axis, which is exactly the
# axis under investigation. A comparison between them must disclose that difference instead of
# presenting a shared profile name as evidence of a shared measurement.
_CANONICAL_V2_BASELINE_PROFILE: Final[MethodologyRuntimeProfile] = MethodologyRuntimeProfile(
    name="canonical-v2-baseline",
    version="2.0",
    status="ready",
    reference_name="Khedron canonical methodology v2.0, no-retrieval baseline",
    reference_url="docs/METHODOLOGY.md",
    reference_commit=None,
    generator_prompt_path=_PROMPTS_DIR / "generator_canonical_v2.txt",
    answer_marker=None,
    image_description_policy="none",
    judge_prompt_path=_PROMPTS_DIR / "judge_v1.txt",
    benchmark_type="locomo",
    evaluation_categories=_LOCOMO_ALL_CATEGORIES,
    scored_categories=_LOCOMO_ANSWERABLE_CATEGORIES,
    audit_mode="both",
    answer_model_type="openai",
    answer_model_id="gpt-4o-mini-2024-07-18",
    judge_type="anthropic",
    judge_model_id="claude-haiku-4-5-20251001",
    # The two differences from canonical-v2, and the reason this profile exists. The second
    # follows from the first -- retrieval-depth cutoffs describe a sweep a provider returning
    # everything cannot perform -- but it is a second difference, and both enter the fingerprint.
    top_k_retrieval=None,
    top_k_cutoffs=(),
    scoring="canonical binary CORRECT-only scoring with Wilson 95% CI",
    aggregation=(
        "micro average over the answerable subset (categories 1-4) with pooled multi-run "
        "Wilson intervals; adversarial reported per category only"
    ),
    requires_methodology_version="2.0",
    superseded_by=None,
)


# Specified in the v3 preregistration document, written and reviewed before any run under it.
# Three changes from v2, each a repair of a defect this project measured in
# its own instrument: the corpus gains the image descriptions it always had, the generator stops
# handing the model a scripted refusal, and the judge stops penalising a specific answer where the
# ground truth is itself vague.
_CANONICAL_V3_PROFILE: Final[MethodologyRuntimeProfile] = replace(
    _CANONICAL_V2_PROFILE,
    name="canonical-v3",
    version="3.0",
    reference_name="Khedron canonical methodology v3.0",
    reference_url="docs/METHODOLOGY.md",
    generator_prompt_path=_PROMPTS_DIR / "generator_canonical_v3.txt",
    judge_prompt_path=_PROMPTS_DIR / "judge_v3.txt",
    image_description_policy="blip_caption_only_v1",
    scoring="canonical binary CORRECT-only scoring with Wilson 95% CI, vague-ground-truth repair",
    requires_methodology_version="3.0",
    superseded_by=None,
)

# The full-context reference arm under v3, standing to canonical-v3 exactly as
# `canonical-v2-baseline` stands to canonical-v2: the same corpus, generator and rubric, with
# retrieval removed so the model is handed the whole conversation.
#
# It is what the retrieval arm is measured against, and it was referenced by the canonical-v3
# specification before it existed -- the specification named an arm no profile could run. Declaring
# it derived from `_CANONICAL_V3_PROFILE` rather than by editing a copy is what keeps the two arms
# differing on the retrieval axis alone; a hand-written twin drifts the moment either side changes,
# and a baseline that quietly differs on the generator or the rubric measures nothing.
_CANONICAL_V3_BASELINE_PROFILE: Final[MethodologyRuntimeProfile] = replace(
    _CANONICAL_V3_PROFILE,
    name="canonical-v3-baseline",
    reference_name="Khedron canonical methodology v3.0, no-retrieval baseline",
    # As in v2-baseline: no retrieval budget to record, and no depth sweep to describe, because a
    # provider that returns everything performs neither. Both enter the fingerprint.
    top_k_retrieval=None,
    top_k_cutoffs=(),
    scoring=(
        "canonical binary CORRECT-only scoring with Wilson 95% CI, vague-ground-truth repair, "
        "full-context baseline"
    ),
)


# The registered ablation, and the only cell of the validation that needs a profile of its own.
#
# v2's corpus and v2's judge with v3's generator: the generator is the one change of the three that
# is a hypothesis rather than a repair, so it is the one that has to be isolated. Paired against a
# contemporaneous all-v2 run over the same question ids, the only thing that differs is the prompt.
#
# Declared here, in the same commit as the specification that calls for it, so it cannot later be
# mistaken for a convenience profile invented after seeing a number.
_CANONICAL_V3_GENERATOR_ONLY_PROFILE: Final[MethodologyRuntimeProfile] = replace(
    _CANONICAL_V2_PROFILE,
    name="canonical-v3-generator-only",
    version="3.0-ablation",
    reference_name="Khedron canonical methodology v3.0, generator change in isolation",
    reference_url="docs/METHODOLOGY.md",
    generator_prompt_path=_PROMPTS_DIR / "generator_canonical_v3.txt",
    scoring="canonical-v2 protocol with the canonical-v3 generator prompt, for ablation only",
    requires_methodology_version=None,
    superseded_by=None,
)


# The V3/J2 cell: canonical-v3 with the judge it replaces.
#
# Rejudging V3's answers under this isolates the judge repair over byte-identical answers, on the
# corpus the measurement will actually use. And because it shares its generator and judge with
# `canonical-v3-generator-only`, the pair differs in exactly one thing -- the corpus -- so the same
# four cells isolate all three changes instead of two. That is why this profile exists rather than
# the earlier `canonical-v3-prompts-only`, which isolated the judge on a corpus we are leaving
# behind and left the corpus effect unmeasured.
_CANONICAL_V3_PRIOR_JUDGE_PROFILE: Final[MethodologyRuntimeProfile] = replace(
    _CANONICAL_V3_PROFILE,
    name="canonical-v3-prior-judge",
    version="3.0-ablation",
    reference_name="Khedron canonical methodology v3.0, judged under the v2 rubric",
    judge_prompt_path=_PROMPTS_DIR / "judge_v1.txt",
    scoring="canonical-v3 corpus and generator scored under the canonical-v2 rubric, ablation only",
    requires_methodology_version=None,
)


def get_runtime_profile(name: str) -> MethodologyRuntimeProfile:
    """Return a verified runtime profile or raise for reserved/ambiguous names."""

    if name == "canonical-v1":
        return _CANONICAL_V1_PROFILE
    if name == "canonical-v2":
        return _CANONICAL_V2_PROFILE
    if name == "canonical-v3":
        return _CANONICAL_V3_PROFILE
    if name == "canonical-v3-generator-only":
        return _CANONICAL_V3_GENERATOR_ONLY_PROFILE
    if name == "canonical-v3-prior-judge":
        return _CANONICAL_V3_PRIOR_JUDGE_PROFILE
    if name == "canonical-v2-baseline":
        return _CANONICAL_V2_BASELINE_PROFILE
    if name == "canonical-v3-baseline":
        return _CANONICAL_V3_BASELINE_PROFILE
    if name == "snap-original":
        raise ConfigurationError(
            "The snap-original profile is reserved but not implemented as a runtime profile.",
            methodology_profile=name,
        )
    archived = archived_profile(name)
    if archived is not None:
        # Refused here rather than merely absent, so the message says what happened to it. A run
        # recorded under this name stays readable through the archive; it is not runnable again.
        raise ConfigurationError(
            "This methodology profile was archived and cannot be used for new runs. Runs already "
            "recorded under it remain readable.",
            methodology_profile=name,
            archived_on=archived.archived_on,
            reason=archived.reason,
        )
    raise ConfigurationError(
        "Unsupported methodology profile.",
        methodology_profile=name,
        supported_profiles=["canonical-v2", "canonical-v2-baseline"],
        reserved_profiles=["snap-original"],
    )


def validate_suite_methodology_profile(config: ExperimentSuiteConfig) -> None:
    """Validate that a suite's configured profile is behaviorally enforceable."""

    profile = get_runtime_profile(config.methodology_profile)
    # Two fields name the same thing and could disagree: the profile carries a version and the
    # suite declares one. A run whose profile says 2.0 while its recorded methodology version says
    # 1.0 puts two contradicting numbers inside artifacts whose whole purpose is to state what was
    # measured. Made to agree by construction rather than by convention.
    required_version = profile.requires_methodology_version
    if required_version is not None and config.methodology_version != required_version:
        raise ConfigurationError(
            "Suite methodology_version does not match the version this profile requires.",
            methodology_profile=profile.name,
            expected=required_version,
            observed=config.methodology_version,
        )
    for experiment in config.experiments:
        _validate_experiment(profile, experiment)


def methodology_fingerprint(profile: MethodologyRuntimeProfile) -> str:
    """Compute a stable hash of methodology-affecting profile fields."""

    # A positive allowlist of what the measurement depends on, not everything the dataclass holds.
    # Identity and governance were in here -- `name`, `reference_name`, `reference_url`,
    # `reference_commit`, `superseded_by`, `requires_methodology_version` -- so renaming a profile
    # or declaring its successor changed the fingerprint of runs whose measurement was untouched.
    # Every previously recorded run already fails to match a recomputed hash, partly for those
    # reasons. `status` was excluded for exactly this argument; the principle was stated and not
    # applied to the rest.
    payload = {
        "schema_version": FINGERPRINT_PAYLOAD_VERSION,
        "generator_prompt_sha256": _sha256_file(profile.generator_prompt_path),
        "answer_marker": profile.answer_marker,
        "image_description_policy": profile.image_description_policy,
        "judge_prompt_sha256": _sha256_file(profile.judge_prompt_path),
        "benchmark_type": profile.benchmark_type,
        "evaluation_categories": profile.evaluation_categories,
        "scored_categories": profile.scored_categories,
        "audit_mode": profile.audit_mode,
        "answer_model_type": profile.answer_model_type,
        "answer_model_id": profile.answer_model_id,
        "judge_type": profile.judge_type,
        "judge_model_id": profile.judge_model_id,
        "top_k_retrieval": profile.top_k_retrieval,
        "top_k_cutoffs": profile.top_k_cutoffs,
        "scoring": profile.scoring,
        "aggregation": profile.aggregation,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_experiment(
    profile: MethodologyRuntimeProfile,
    experiment: ExperimentConfig,
) -> None:
    if profile.superseded_by is not None:
        # Refused rather than validated leniently. `canonical-v1` pins nothing it names -- models,
        # retrieval budget and category set are all None -- so a run under it certifies nothing
        # about what was measured. Lenient validation is what let an early measurement look
        # governed while being unconstrained; a successor exists, so this is an error and not a
        # warning.
        raise ConfigurationError(
            f"The {profile.name} methodology profile is superseded and cannot be used for new "
            f"runs. Use {profile.superseded_by} instead.",
            methodology_profile=profile.name,
            successor=profile.superseded_by,
        )

    _ensure_prompt_files_exist(profile)
    _require_equal(
        experiment.benchmark.type,
        profile.benchmark_type,
        profile=profile,
        field="benchmark.type",
    )
    _require_equal(
        experiment.benchmark.audit_mode,
        profile.audit_mode,
        profile=profile,
        field="benchmark.audit_mode",
    )
    if profile.evaluation_categories is not None:
        configured_categories = experiment.benchmark.config.get("categories")
        if not isinstance(configured_categories, list):
            raise ConfigurationError(
                "Methodology profile requires an exact benchmark category subset.",
                methodology_profile=profile.name,
                field="benchmark.config.categories",
                expected=list(profile.evaluation_categories),
                observed=configured_categories,
                experiment_name=experiment.name,
            )
        configured_category_set = {
            str(category) for category in cast(list[object], configured_categories)
        }
        if configured_category_set != set(profile.evaluation_categories):
            raise ConfigurationError(
                "Methodology profile requires an exact benchmark category subset.",
                methodology_profile=profile.name,
                field="benchmark.config.categories",
                expected=list(profile.evaluation_categories),
                observed=configured_categories,
                experiment_name=experiment.name,
            )
    _require_equal(
        experiment.answer_model.type,
        profile.answer_model_type,
        profile=profile,
        field="answer_model.type",
    )
    _require_equal(
        experiment.answer_model.model,
        profile.answer_model_id,
        profile=profile,
        field="answer_model.model",
    )
    _require_equal(
        experiment.judge.type,
        profile.judge_type,
        profile=profile,
        field="judge.type",
    )
    _require_equal(
        experiment.judge.model,
        profile.judge_model_id,
        profile=profile,
        field="judge.model",
    )
    _require_equal(
        experiment.top_k_retrieval,
        profile.top_k_retrieval,
        profile=profile,
        field="top_k_retrieval",
    )
    _require_equal(
        experiment.answer_model.temperature,
        0.0,
        profile=profile,
        field="answer_model.temperature",
    )
    _require_equal(
        experiment.judge.temperature,
        0.0,
        profile=profile,
        field="judge.temperature",
    )


def _require_equal(
    observed: object,
    expected: object,
    *,
    profile: MethodologyRuntimeProfile,
    field: str,
) -> None:
    """Require a configured value to match a pinned one, skipping fields the profile leaves open.

    `None` means the profile does not constrain this field, which is what the `| None` on those
    fields has always meant -- but the check compared directly, so an unpinned field demanded the
    configuration also be `None`. That went unnoticed because only fully-pinned profiles ever
    reached this path: `canonical-v1` returned early and pinned nothing. A profile pinning some
    fields and not others would have rejected every valid configuration.
    """
    if expected is None:
        return
    if observed != expected:
        raise ConfigurationError(
            "Configuration is incompatible with methodology profile.",
            methodology_profile=profile.name,
            field=field,
            expected=expected,
            observed=observed,
        )


def _ensure_prompt_files_exist(profile: MethodologyRuntimeProfile) -> None:
    for field, path in (
        ("generator_prompt_path", profile.generator_prompt_path),
        ("judge_prompt_path", profile.judge_prompt_path),
    ):
        if not path.exists():
            raise ConfigurationError(
                "Methodology profile prompt artifact is missing.",
                methodology_profile=profile.name,
                field=field,
                path=str(path),
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigurationError(
            "Unable to read methodology profile prompt artifact.",
            path=str(path),
            error=str(exc),
        ) from exc
    return digest.hexdigest()


def _runnable_profile_names() -> tuple[str, ...]:
    """Every declarable profile name that resolves to a runtime profile.

    Derived from `ProfileName` by asking the lookup, rather than written out a second time: the
    pricing guard iterated a hand-maintained list that had silently omitted both canonical-v2
    profiles -- the two about to be spent on -- so it would have stayed green while the model they
    pin had no resolvable price. Reserved names (`snap-original`) raise by
    design and drop out here, so adding a profile to the Literal and the lookup is enough for
    every check that sweeps all of them to pick it up.
    """
    names: list[str] = []
    for name in get_args(ProfileName):
        try:
            get_runtime_profile(name)
        except ConfigurationError:
            continue
        names.append(name)
    return tuple(names)


RUNNABLE_PROFILE_NAMES: Final[tuple[str, ...]] = _runnable_profile_names()
