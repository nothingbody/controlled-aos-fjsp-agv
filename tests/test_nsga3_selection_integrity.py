"""Integrity tests for NSGA-III niching and scale-free spread."""

import numpy as np
import pytest

from src.algorithm.nsga3.selection import (
    _niching_select,
    _normalize_nsga3_objectives,
    nsga3_select,
)
from src.utils.metrics import compute_spread


class _LastChoiceRng:
    """Deterministic test RNG that always chooses the last candidate."""

    @staticmethod
    def choice(values):
        values = np.asarray(values)
        return values[-1]


def test_niching_excludes_reference_directions_without_candidates():
    # Direction 2 has the smallest nominal count but no split-front candidate.
    # Direction 1 is the only eligible direction and its closest point is 11.
    counts = np.array([1, 0, 0], dtype=int)
    chosen = _niching_select(
        [10, 11],
        np.array([1, 1]),
        np.array([0.4, 0.1]),
        counts,
        remaining=1,
        rng=_LastChoiceRng(),
    )

    assert chosen == [11]
    assert counts.tolist() == [1, 1, 0]


def test_niching_uses_nearest_for_empty_niche_and_random_for_occupied_niche():
    nearest_counts = np.array([0], dtype=int)
    nearest = _niching_select(
        [20, 21],
        np.array([0, 0]),
        np.array([0.05, 0.8]),
        nearest_counts,
        remaining=1,
        rng=_LastChoiceRng(),
    )
    assert nearest == [20]

    occupied_counts = np.array([2], dtype=int)
    random_pick = _niching_select(
        [20, 21],
        np.array([0, 0]),
        np.array([0.05, 0.8]),
        occupied_counts,
        remaining=1,
        rng=_LastChoiceRng(),
    )
    assert random_pick == [21]


def test_nsga3_selection_is_reproducible_with_explicit_rng():
    objectives = np.array(
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
    )
    refs = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    selected_a = nsga3_select(objectives, 3, refs, rng=np.random.default_rng(7))
    selected_b = nsga3_select(objectives, 3, refs, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(selected_a, selected_b)


def test_nsga3_rejects_zero_reference_direction():
    with pytest.raises(ValueError, match="non-zero"):
        nsga3_select(np.array([[1.0, 2.0], [2.0, 1.0]]), 1, np.zeros((1, 2)))


def test_nsga3_hyperplane_normalization_uses_extreme_point_intercepts():
    objectives = np.array(
        [
            [4.0, 1.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 1.0, 4.0],
            [2.0, 2.0, 2.0],
        ]
    )

    normalized = _normalize_nsga3_objectives(objectives, [0, 1, 2, 3])

    np.testing.assert_allclose(normalized[:3], np.eye(3))
    np.testing.assert_allclose(normalized[3], np.full(3, 1.0 / 3.0))


def test_nsga3_hyperplane_normalization_has_degenerate_front_fallback():
    objectives = np.array(
        [[1.0, 2.0, 5.0], [2.0, 3.0, 5.0], [3.0, 4.0, 5.0]]
    )

    normalized = _normalize_nsga3_objectives(objectives, [0])

    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized[:, 2], 0.0)


def test_nsga3_hyperplane_normalization_is_scale_and_translation_invariant():
    objectives = np.array(
        [
            [4.0, 1.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 1.0, 4.0],
            [2.0, 2.0, 2.0],
        ]
    )
    transformed = (
        objectives * np.array([1000.0, 0.01, 7.0])
        + np.array([9.0, -3.0, 2.0])
    )

    baseline = _normalize_nsga3_objectives(objectives, [0, 1, 2, 3])
    changed = _normalize_nsga3_objectives(transformed, [0, 1, 2, 3])

    np.testing.assert_allclose(changed, baseline)


def test_spread_deduplicates_and_is_invariant_to_scale_and_translation():
    front = np.array(
        [[1.0, 100.0], [2.0, 80.0], [3.0, 20.0], [4.0, 0.0]]
    )
    with_duplicates = np.vstack([front, front[1], front[1], front[3]])
    transformed = with_duplicates * np.array([1000.0, 0.001]) + np.array([5.0, -3.0])

    baseline = compute_spread(front)
    assert compute_spread(with_duplicates) == pytest.approx(baseline)
    assert compute_spread(transformed) == pytest.approx(baseline)


def test_spread_returns_zero_when_fewer_than_three_unique_points_remain():
    front = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    assert compute_spread(front) == 0.0
