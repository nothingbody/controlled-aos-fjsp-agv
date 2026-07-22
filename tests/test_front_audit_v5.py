import numpy as np
import pytest

from scripts.audit_fronts_v5 import audit_front_matrix


def test_front_audit_deduplicates_and_computes_tradeoff_correlation():
    objectives = np.array(
        [
            [1.0, 4.0, 2.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 2.0],
            [4.0, 1.0, 2.0],
            [2.0, 3.0, 2.0],
        ]
    )

    result = audit_front_matrix(objectives)

    assert result["front_points_unique_nondominated"] == 4
    assert result["pearson_cmax_tec"] == pytest.approx(-1.0)
    assert result["spearman_cmax_tec"] == pytest.approx(-1.0)
    assert result["correlation_status"] == "computed"


def test_front_audit_marks_small_front_without_fabricating_correlation():
    result = audit_front_matrix(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))

    assert result["front_points_unique_nondominated"] == 2
    assert np.isnan(result["pearson_cmax_tec"])
    assert result["correlation_status"] == "insufficient_variation"
