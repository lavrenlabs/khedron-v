from __future__ import annotations

from datetime import date
from fnmatch import fnmatchcase
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PricingEntry",
    "PricingTable",
    "compute_cost_usd",
    "load_pricing_table",
    "load_vendor_pricing",
]


class PricingEntry(BaseModel):
    """Per-token pricing for one model pattern."""

    model_config = ConfigDict(frozen=True)

    model_pattern: str
    input_per_million_usd: float = Field(ge=0.0)
    output_per_million_usd: float = Field(ge=0.0)
    # Last date these figures are known to hold. Introductory and promotional prices expire
    # on an announced date, and a comment saying so cannot stop a run: past that date the
    # tracker keeps computing, silently undercounting spend and under-enforcing the budget
    # cap. Declaring the date makes preflight able to refuse instead of guess. Absent means
    # standard pricing with no announced end.
    price_valid_until: date | None = None


class PricingTable(BaseModel):
    """Versioned vendor pricing table loaded from YAML."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    vendor: str | None = None
    last_updated: date
    source_url: str | None = None
    prices: list[PricingEntry]

    def find_for_model(self, model_id: str) -> PricingEntry | None:
        """Find pricing by exact model ID, then deterministic wildcard pattern order."""

        for entry in self.prices:
            if entry.model_pattern == model_id:
                return entry

        for entry in self.prices:
            if fnmatchcase(model_id, entry.model_pattern):
                return entry
        return None


def load_pricing_table(path: Path) -> PricingTable:
    """Load and validate a pricing table from a YAML file."""

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PricingTable.model_validate(raw_data)


def load_vendor_pricing(
    vendor: str,
    pricing_dir: Path = Path("data/pricing"),
) -> PricingTable:
    """Load pricing for a vendor from ``data/pricing/<vendor>.yaml``."""

    return load_pricing_table(pricing_dir / f"{vendor}.yaml")


def compute_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    pricing: PricingTable,
) -> float:
    """Compute USD cost from input/output token counts and a pricing table."""

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")

    entry = pricing.find_for_model(model_id)
    if entry is None:
        raise ValueError(f"No pricing for model {model_id}")

    input_cost = (input_tokens / 1_000_000) * entry.input_per_million_usd
    output_cost = (output_tokens / 1_000_000) * entry.output_per_million_usd
    return input_cost + output_cost
