import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_resubmission_v5 import (
    PROTOCOL,
    add_common_hv,
    bootstrap_ci,
    collapse_seeds,
    holm_adjust,
    load_front,
    main_inference,
    stable_unique_rows,
    summarize_main,
    validate_completeness,
)


def _write_front(path: Path, objectives):
    with path.open("wb") as stream:
        pickle.dump({"objectives": np.asarray(objectives, dtype=float)}, stream)


def _base_rows(methods=("AdaptiveSAOS", "UCBOnly"), budgets=(50,), seeds=(42, 43)):
    rows = []
    for instance in ("I1", "I2"):
        for budget in budgets:
            for method in methods:
                for seed in seeds:
                    rows.append(
                        {
                            "Protocol": PROTOCOL,
                            "dataset": "D",
                            "instance": instance,
                            "variant": method,
                            "reward_scheme": "composite",
                            "seed": seed,
                            "Budget": budget,
                            "front_pickle": f"{instance}_{budget}_{method}_{seed}.pkl",
                        }
                    )
    return pd.DataFrame(rows)


def test_front_objectives_are_deduplicated_after_loading(tmp_path):
    path = tmp_path / "front.pkl"
    _write_front(path, [[1, 2, 3], [1, 2, 3], [2, 1, 4]])

    front, raw_count = load_front(path)

    assert raw_count == 3
    assert front.shape == (2, 3)
    assert np.array_equal(front, stable_unique_rows(front))


def test_common_hv_uses_shared_instance_budget_3d_scale(tmp_path):
    a_path = tmp_path / "a.pkl"
    b_path = tmp_path / "b.pkl"
    _write_front(a_path, [[0, 2, 9], [0, 2, 9], [2, 0, 8]])
    _write_front(b_path, [[1, 1, 7]])
    frame = pd.DataFrame(
        [
            {
                "dataset": "D",
                "instance": "I",
                "Budget": 100,
                "variant": "A",
                "front_pickle": str(a_path),
            },
            {
                "dataset": "D",
                "instance": "I",
                "Budget": 100,
                "variant": "B",
                "front_pickle": str(b_path),
            },
        ]
    )

    enriched, normalization = add_common_hv(frame, tmp_path / "runs.csv", tmp_path)

    assert normalization[0]["ideal"] == [0.0, 0.0, 7.0]
    assert normalization[0]["nadir"] == [2.0, 2.0, 9.0]
    assert normalization[0]["reference"] == [1.1, 1.1, 1.1]
    assert enriched.loc[0, "Front_points_raw"] == 3
    assert enriched.loc[0, "Front_points_unique"] == 2
    # Exact 3-D union volumes after the common normalization.
    assert enriched.loc[0, "HV_common"] == pytest.approx(0.076)
    assert enriched.loc[1, "HV_common"] == pytest.approx(0.396)


def test_main_aggregation_collapses_seeds_before_instance_inference():
    frame = pd.DataFrame(
        [
            # I1: a seed outlier must not become a third independent block.
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "A", "seed": 1, "HV_common": 0.0},
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "A", "seed": 2, "HV_common": 100.0},
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "A", "seed": 3, "HV_common": 2.0},
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "B", "seed": 1, "HV_common": 1.0},
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "B", "seed": 2, "HV_common": 1.0},
            {"dataset": "D", "instance": "I1", "Budget": 100, "variant": "B", "seed": 3, "HV_common": 1.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "A", "seed": 1, "HV_common": 3.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "A", "seed": 2, "HV_common": 3.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "A", "seed": 3, "HV_common": 3.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "B", "seed": 1, "HV_common": 4.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "B", "seed": 2, "HV_common": 4.0},
            {"dataset": "D", "instance": "I2", "Budget": 100, "variant": "B", "seed": 3, "HV_common": 4.0},
        ]
    )

    medians = collapse_seeds(frame)
    a_i1 = medians.query("instance == 'I1' and variant == 'A'").iloc[0]
    analysis = main_inference(
        medians,
        target="A",
        comparators=["B"],
        bootstrap_reps=100,
    )

    assert a_i1["HV_common"] == 2.0
    assert len(medians) == 4
    assert analysis["pairwise"].iloc[0]["instances"] == 2
    assert analysis["pairwise"].iloc[0]["wins"] == 1
    assert analysis["pairwise"].iloc[0]["losses"] == 1


def test_e3_holm_family_is_joint_across_budgets():
    rows = []
    methods = ("T", "A", "B")
    for budget in (50, 100):
        for index in range(8):
            for method_index, method in enumerate(methods):
                rows.append(
                    {
                        "dataset": "D",
                        "instance": f"I{index}",
                        "Budget": budget,
                        "variant": method,
                        "HV_common": float(index + (2 - method_index) * (budget / 100)),
                        "seed_count": 2,
                    }
                )
    medians = pd.DataFrame(rows)

    result = main_inference(
        medians,
        target="T",
        comparators=["A", "B"],
        bootstrap_reps=100,
    )["pairwise"]

    assert len(result) == 4
    assert result["comparison_family"].nunique() == 1
    assert set(result["holm_family_size"]) == {4}
    assert np.allclose(result["p_holm"], holm_adjust(result["p_raw"]))


def test_deterministic_instance_bootstrap():
    first = bootstrap_ci([1.0, 2.0, 9.0], reps=500, seed=77)
    second = bootstrap_ci([1.0, 2.0, 9.0], reps=500, seed=77)
    assert first == second


def test_main_summary_reports_iqr_and_all_declared_secondary_metrics():
    medians = pd.DataFrame(
        {
            "dataset": ["D"] * 4,
            "instance": ["I1", "I2", "I3", "I4"],
            "variant": ["A"] * 4,
            "HV_common": [0.1, 0.2, 0.3, 0.4],
            "Cmax_best": [10, 20, 30, 40],
            "TEC_best": [1, 2, 3, 4],
            "WB_best": [4, 3, 2, 1],
            "NSol": [5, 6, 7, 8],
            "Time": [11, 12, 13, 14],
        }
    )

    summary = summarize_main(medians, bootstrap_reps=100, bootstrap_seed=5).iloc[0]

    for metric in ("HV_common", "Cmax_best", "TEC_best", "WB_best", "NSol", "Time"):
        assert f"{metric}_q1_of_instance_medians" in summary.index
        assert f"{metric}_q3_of_instance_medians" in summary.index
        assert f"{metric}_iqr_of_instance_medians" in summary.index
    assert summary["HV_common_iqr_of_instance_medians"] == pytest.approx(0.15)


def test_completeness_rejects_duplicate_primary_key():
    frame = _base_rows()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate primary keys"):
        validate_completeness(
            duplicated,
            expected_seed_count=2,
            expected_methods=["AdaptiveSAOS", "UCBOnly"],
            expected_budgets=["50"],
            expected_instance_count=2,
        )


def test_completeness_rejects_missing_expected_cell():
    frame = _base_rows()
    missing = frame.drop(frame.index[0]).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="incomplete expected analysis grid"):
        validate_completeness(
            missing,
            expected_seed_count=2,
            expected_methods=["AdaptiveSAOS", "UCBOnly"],
            expected_budgets=["50"],
            expected_instance_count=2,
        )


def test_protocol_is_v5_only():
    frame = _base_rows()
    frame["Protocol"] = "saos_bc_onpolicy_ppo_v4_20260720"
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        validate_completeness(frame, expected_seed_count=2)
