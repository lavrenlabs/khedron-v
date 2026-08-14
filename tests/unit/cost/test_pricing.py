from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from khedron.cost.pricing import (
    PricingEntry,
    PricingTable,
    compute_cost_usd,
    load_pricing_table,
    load_vendor_pricing,
)

FIXTURE_PATH = Path("tests/fixtures/pricing/test_pricing.yaml")
VENDORS = ("openai", "anthropic", "google")


def test_loads_synthetic_fixture_yaml() -> None:
    table = load_pricing_table(FIXTURE_PATH)

    if table.schema_version != 1:
        raise AssertionError(table)
    if table.vendor != "synthetic":
        raise AssertionError(table)
    if table.find_for_model("synthetic-exact") is None:
        raise AssertionError(table)


def test_loads_all_vendor_yaml_files() -> None:
    for vendor in VENDORS:
        table = load_vendor_pricing(vendor)
        if table.vendor != vendor:
            raise AssertionError(table)
        if table.schema_version != 1:
            raise AssertionError(table)
        if table.source_url is None:
            raise AssertionError(table)
        if len(table.prices) < 2:
            raise AssertionError(table)


def test_pricing_models_are_frozen() -> None:
    entry = PricingEntry(
        model_pattern="frozen-model",
        input_per_million_usd=1.0,
        output_per_million_usd=2.0,
    )
    table = PricingTable(
        vendor="synthetic",
        last_updated="2026-05-04",
        prices=[entry],
    )

    with pytest.raises(ValidationError):
        entry.input_per_million_usd = 3.0
    with pytest.raises(ValidationError):
        table.vendor = "changed"


@pytest.mark.parametrize(
    ("input_per_million_usd", "output_per_million_usd"),
    [(-0.01, 1.0), (1.0, -0.01)],
)
def test_pricing_entry_rejects_negative_prices(
    input_per_million_usd: float,
    output_per_million_usd: float,
) -> None:
    with pytest.raises(ValidationError):
        PricingEntry(
            model_pattern="bad-model",
            input_per_million_usd=input_per_million_usd,
            output_per_million_usd=output_per_million_usd,
        )


def test_exact_model_matching_takes_precedence_over_wildcard() -> None:
    table = load_pricing_table(FIXTURE_PATH)

    exact_entry = table.find_for_model("synthetic-exact")
    wildcard_entry = table.find_for_model("synthetic-other")

    if exact_entry is None:
        raise AssertionError(table)
    if wildcard_entry is None:
        raise AssertionError(table)
    if exact_entry.input_per_million_usd != 2.0:
        raise AssertionError(exact_entry)
    if wildcard_entry.input_per_million_usd != 1.0:
        raise AssertionError(wildcard_entry)


def test_unknown_model_raises_from_compute_cost_usd() -> None:
    table = load_pricing_table(FIXTURE_PATH)

    with pytest.raises(ValueError, match="No pricing for model unknown-model"):
        compute_cost_usd("unknown-model", 1_000, 500, table)


def test_compute_cost_uses_per_million_token_formula() -> None:
    table = load_pricing_table(FIXTURE_PATH)

    cost = compute_cost_usd(
        "synthetic-exact",
        input_tokens=250_000,
        output_tokens=125_000,
        pricing=table,
    )

    if cost != pytest.approx(1.5):
        raise AssertionError(cost)


def test_vendor_pricing_contains_expected_standard_text_entries() -> None:
    expectations = {
        "openai": ("gpt-4o-mini-2024-07-18", 0.15, 0.60),
        "anthropic": ("claude-sonnet-4-5", 3.00, 15.00),
        "google": ("gemini-2.5-pro", 1.25, 10.00),
    }

    for vendor, (model_id, input_price, output_price) in expectations.items():
        table = load_vendor_pricing(vendor)
        entry = table.find_for_model(model_id)
        if entry is None:
            raise AssertionError((vendor, model_id))
        if entry.input_per_million_usd != pytest.approx(input_price):
            raise AssertionError(entry)
        if entry.output_per_million_usd != pytest.approx(output_price):
            raise AssertionError(entry)
