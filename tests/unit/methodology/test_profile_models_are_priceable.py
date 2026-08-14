from __future__ import annotations

import pytest

from khedron.cost.pricing import compute_cost_usd, load_vendor_pricing
from khedron.methodology.profiles import RUNNABLE_PROFILE_NAMES, get_runtime_profile

# Derived from the registry rather than written out: the hand-maintained list omitted
# canonical-v2 and canonical-v2-baseline -- the two profiles about to be spent on -- so this
# would have stayed green while the judge model they pin had no resolvable price. A list that
# must be remembered is a list that will be forgotten.
_PROFILES = tuple(get_runtime_profile(name) for name in RUNNABLE_PROFILE_NAMES)


@pytest.mark.parametrize("profile", _PROFILES, ids=lambda p: p.name)
def test_every_model_a_profile_pins_has_a_price(profile: object) -> None:
    # A profile that pins a model the pricing table cannot resolve is unrunnable: the
    # adapters turn a missing price into a ModelError, so the failure lands on the first
    # paid call rather than at config time. a profile pinned the bare alias
    # "gpt-4o" while the table only carried dated snapshots, which made that whole
    # profile impossible to execute without anything reporting why.
    pinned = [
        (getattr(profile, "answer_model_type", None), getattr(profile, "answer_model_id", None)),
        (getattr(profile, "judge_type", None), getattr(profile, "judge_model_id", None)),
    ]
    for vendor, model_id in pinned:
        if vendor is None or model_id is None:
            continue  # the profile leaves this open; nothing to price
        pricing = load_vendor_pricing(vendor)
        # Exercise the real path the adapters use, not just the lookup: compute_cost_usd
        # is what raises when a model is unpriced.
        compute_cost_usd(model_id, input_tokens=1000, output_tokens=100, pricing=pricing)


def test_the_mini_alias_is_not_captured_by_a_gpt_4o_wildcard() -> None:
    # Guards the reason these aliases are listed explicitly: a "gpt-4o*" pattern would
    # also match gpt-4o-mini and bill it at the full gpt-4o rate.
    pricing = load_vendor_pricing("openai")
    mini = compute_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0, pricing=pricing)
    full = compute_cost_usd("gpt-4o", input_tokens=1_000_000, output_tokens=0, pricing=pricing)
    if mini >= full:
        raise AssertionError(f"gpt-4o-mini priced at {mini} but gpt-4o at {full}")
