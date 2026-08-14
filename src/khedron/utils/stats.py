from __future__ import annotations

from math import sqrt
from statistics import NormalDist


def wilson_score_interval(
    n_successes: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial proportion."""
    if n_total < 0:
        raise ValueError("n_total must be non-negative")
    if n_successes < 0:
        raise ValueError("n_successes must be non-negative")
    if n_successes > n_total:
        raise ValueError("n_successes cannot exceed n_total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_total == 0:
        return (0.0, 0.0)

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = n_successes / n_total
    z_squared = z * z
    denominator = 1.0 + z_squared / n_total
    center = (proportion + z_squared / (2.0 * n_total)) / denominator
    margin = (
        z
        * sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * n_total)) / n_total)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))
