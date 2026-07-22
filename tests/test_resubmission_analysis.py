import numpy as np

from scripts.analyze_resubmission_v4 import holm_adjust, rank_biserial


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = np.array([0.01, 0.04, 0.02])
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))


def test_rank_biserial_direction_matches_target_minus_comparator():
    target = np.array([4.0, 5.0, 6.0])
    comparator = np.array([1.0, 2.0, 3.0])
    assert rank_biserial(target, comparator) == 1.0
    assert rank_biserial(comparator, target) == -1.0
