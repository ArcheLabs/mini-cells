from __future__ import annotations

import numpy as np

from minicells.language_recruitment_response import RECRUITMENT_GRID, normalized_regret, summarize_response


def test_recruitment_grid_is_ordered_and_contains_key_scales() -> None:
    grid = np.asarray(RECRUITMENT_GRID, dtype=float)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0.0)
    for required in (1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 5e-1):
        assert required in RECRUITMENT_GRID


def test_activation_barrier_requires_small_recruitment_harm_and_full_benefit() -> None:
    grid = np.asarray(RECRUITMENT_GRID, dtype=float)
    value = np.interp(grid, [0.0, 0.02, 0.10, 1.0], [0.0, -0.03, 0.02, 0.40])
    summary = summarize_response(grid, value)
    assert summary.full_beneficial
    assert summary.activation_barrier
    assert summary.min_small_value < 0.0
    assert summary.first_positive_recruitment is not None
    assert summary.nonmonotonic


def test_monotone_helpful_curve_has_no_activation_barrier() -> None:
    grid = np.asarray(RECRUITMENT_GRID, dtype=float)
    value = 0.5 * grid
    summary = summarize_response(grid, value)
    assert summary.full_beneficial
    assert not summary.activation_barrier
    assert not summary.nonmonotonic
    assert summary.best_recruitment == 1.0


def test_initial_harm_without_full_benefit_is_not_called_activation_barrier() -> None:
    grid = np.asarray(RECRUITMENT_GRID, dtype=float)
    value = -0.2 * grid
    summary = summarize_response(grid, value)
    assert not summary.full_beneficial
    assert not summary.activation_barrier


def test_normalized_regret_is_zero_for_best_choice() -> None:
    assert normalized_regret(2.0, 2.0) == 0.0
    assert abs(normalized_regret(2.0, 1.0) - 0.5) < 1e-12
