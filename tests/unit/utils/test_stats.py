from __future__ import annotations

import pytest

from khedron.utils.stats import wilson_score_interval


def _check_close(actual: float, expected: float) -> None:
    if abs(actual - expected) >= 0.0001:
        raise AssertionError(f"{actual} != {expected}")


def test_wilson_score_interval_matches_phase_zero_golden_value() -> None:
    low, high = wilson_score_interval(82, 100)
    _check_close(low, 0.7333)
    _check_close(high, 0.8830)


def test_wilson_score_interval_matches_balanced_golden_value() -> None:
    low, high = wilson_score_interval(50, 100)
    _check_close(low, 0.4038)
    _check_close(high, 0.5962)


def test_wilson_score_interval_handles_zero_total() -> None:
    low, high = wilson_score_interval(0, 0)
    _check_close(low, 0.0)
    _check_close(high, 0.0)


def test_wilson_score_interval_handles_no_successes() -> None:
    low, high = wilson_score_interval(0, 100)
    _check_close(low, 0.0)
    _check_close(high, 0.0370)


def test_wilson_score_interval_handles_all_successes() -> None:
    low, high = wilson_score_interval(100, 100)
    _check_close(low, 0.9630)
    _check_close(high, 1.0)


def test_wilson_score_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(101, 100)
