import numpy as np
import pandas as pd
import pytest

from experiments.run_e4_replication_v8 import CONFIGS, FORMAL_SEEDS, SELECTED_INSTANCES
from scripts.analyze_e4_replication_v8 import (
    TIE_TOL,
    V6_SHARED_CONTRASTS,
    add_indicators,
    inferential_analysis,
    instance_seed_medians,
    preflight,
    replication_comparison,
    signed_rank_randomization,
    validate_payload_diagnostics,
    validate_grid,
)


def _formal_rows() -> pd.DataFrame:
    rows = []
    variants_at_100 = [item["variant"] for item in CONFIGS[100]]
    variants_at_200 = [item["variant"] for item in CONFIGS[200]]
    effects = {
        "UCBOnly": 0.0,
        "BasePaddedNoBC_R16": 1.0,
        "BasePaddedBC_R16": 2.0,
        "EnhancedNoBC_R16": 3.0,
        "EnhancedBC_R8": 4.0,
        "EnhancedBC_R16": 5.0,
        "EnhancedBC_R32": 6.0,
    }
    for dataset, instances in SELECTED_INSTANCES.items():
        for instance_index, instance in enumerate(instances):
            for budget, variants in ((100, variants_at_100), (200, variants_at_200)):
                for variant in variants:
                    for seed in FORMAL_SEEDS:
                        value = effects[variant] + instance_index / 100 + seed / 10000
                        rows.append({
                            "Protocol": "saos_e4_replication_v8_20260722",
                            "dataset": dataset,
                            "instance": instance,
                            "variant": variant,
                            "Budget": budget,
                            "seed": seed,
                            "Population_size": 100,
                            "Max_generations": budget,
                            "Worker_count": 40,
                            "Initial_evaluations": 100,
                            "Offspring_evaluations": 100 * budget,
                            "Config_hash": "config",
                            "Code_hash": "code",
                            "Design_hash": "design",
                            "Input_hash": "input",
                            "Reference_snapshot_sha256": "reference",
                            "Front_sha256": "front",
                            "Front_semantic_sha256": "semantic",
                            "front_pickle": "front.pkl",
                            "PPO_update_count": 0,
                            "PPO_samples": 0,
                            "PPO_action_effective_updates": 0,
                            "PPO_terminal_full_updates": 0,
                            "PPO_terminal_residual_updates": 0,
                            "BC_epoch_loss": "[]",
                            "BC_epoch_accuracy": "[]",
                            "BC_confusion_matrix": "[]",
                            "HV_fixed": value,
                            "IGDplus_fixed": 10.0 - value,
                        })
    return pd.DataFrame(rows)


def test_validate_grid_requires_exactly_1350_unique_formal_keys():
    frame = _formal_rows()
    audit = validate_grid(
        frame,
        hashes={
            "code_hash": "code", "design_hash": "design",
            "input_hash": "input", "reference_snapshot_sha256": "reference",
        },
        verify_config_hashes=False,
    )
    assert audit["rows"] == audit["unique_keys"] == 1350

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_grid(duplicate, hashes=audit, verify_config_hashes=False)

    with pytest.raises(RuntimeError, match="grid mismatch|1350"):
        validate_grid(frame.iloc[:-1], hashes=audit, verify_config_hashes=False)

    fractional = frame.copy()
    fractional["Population_size"] = fractional["Population_size"].astype(float)
    fractional.loc[0, "Population_size"] = 100.5
    with pytest.raises(RuntimeError, match="strictly integral"):
        validate_grid(fractional, hashes=audit, verify_config_hashes=False)


def test_instance_medians_require_all_five_seeds():
    frame = _formal_rows()
    medians = instance_seed_medians(frame)
    assert len(medians) == 270
    assert set(medians["seed_count"]) == {5}

    missing_seed = frame.drop(index=frame.index[0])
    with pytest.raises(RuntimeError, match="five seeds|seed grid"):
        instance_seed_medians(missing_seed)


def test_signed_rank_uses_fixed_seed_sign_flip_for_30_instances():
    differences = np.arange(1.0, 31.0)
    first = signed_rank_randomization(differences, reps=5_000, seed=8128)
    second = signed_rank_randomization(differences, reps=5_000, seed=8128)
    assert first == second
    assert first[2] == 30
    assert "fixed-seed sign flip" in first[3]

    ties = signed_rank_randomization(np.zeros(30), reps=10, seed=1)
    assert ties[:3] == (0.0, 1.0, 0)
    assert ties[3] == "all ties"


def test_frozen_families_have_expected_sizes_holm_and_oriented_effects():
    medians = instance_seed_medians(_formal_rows())
    comparisons, blocks = inferential_analysis(
        medians, bootstrap_reps=200, sign_flip_reps=2_000
    )
    hv = comparisons[comparisons["metric"] == "HV_fixed"]
    assert hv.groupby("family").size().to_dict() == {
        "M100_bc": 2,
        "M100_interaction": 1,
        "M100_state": 2,
        "M200_rollout": 2,
        "UCB_replication": 3,
    }
    assert len(blocks) == 10 * 2 * 30
    assert (comparisons["wins"] + comparisons["ties"] + comparisons["losses"] == 30).all()
    assert (comparisons["p_holm_within_family"] >= comparisons["p_raw"] - TIE_TOL).all()
    assert (
        comparisons.loc[comparisons["metric"] == "IGDplus_fixed", "orientation"]
        == "positive favors lhs (lower raw metric is better)"
    ).all()


def test_interaction_is_a_separate_single_test():
    medians = instance_seed_medians(_formal_rows())
    comparisons, _ = inferential_analysis(
        medians, bootstrap_reps=50, sign_flip_reps=200
    )
    interaction = comparisons[
        (comparisons["metric"] == "HV_fixed")
        & (comparisons["family"] == "M100_interaction")
    ].iloc[0]
    assert interaction["contrast"] == "state_by_bc_difference_in_differences"
    assert interaction["holm_family_size"] == 1


def test_fixed_box_excludes_outside_points_but_igd_uses_the_full_front(tmp_path):
    row = {
        "dataset": "Brandimarte", "instance": "Mk02.fjs",
        "variant": "UCBOnly", "Budget": 100, "seed": 52,
    }
    points = np.array([[0.5, 0.5, 0.5], [1.2, 0.2, 0.2]])
    key = ("Brandimarte", "Mk02.fjs", 100)
    snapshot = {
        "normalization": {
            key: {"ideal": [0, 0, 0], "nadir": [1, 1, 1], "reference": [1.1, 1.1, 1.1]}
        },
        "reference_sets": {key: np.array([[1.2, 0.2, 0.2]])},
    }
    enriched, audit = add_indicators([(row, points, tmp_path / "front.pkl")], snapshot)
    assert enriched.loc[0, "HV_fixed"] == pytest.approx(0.6 ** 3)
    assert enriched.loc[0, "IGDplus_fixed"] == pytest.approx(0.0)
    assert audit.loc[0, "fixed_inside_points"] == 1
    assert audit.loc[0, "fixed_excluded_points"] == 1
    assert audit.loc[0, "above_fixed_cmax"] == 1


def test_failure_marker_blocks_preflight_before_any_analysis(tmp_path):
    (tmp_path / "pipeline_failed.txt").write_text("failed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failure marker"):
        preflight(tmp_path)


def test_payload_diagnostics_reject_missing_trace_and_mixed_policy_versions(tmp_path):
    row = {
        "Budget": 2, "PPO_update_count": 1, "PPO_samples": 2,
        "PPO_action_effective_updates": 1, "PPO_terminal_full_updates": 0,
        "PPO_terminal_residual_updates": 0, "BC_epoch_loss": "[]",
        "BC_epoch_accuracy": "[]", "BC_confusion_matrix": "[]",
    }
    payload = {
        "operator_sequence": [0, 1], "reward_sequence": [0.1, 0.2],
        "hv_delta_sequence": [0.0, 0.1], "behavior_cloning_stats": {},
        "ppo_update_stats": [{
            "behavior_policy_version": 0, "updated_policy_version": 1,
            "rollout_size": 2, "update_context": "pre_action",
        }],
    }
    validate_payload_diagnostics(payload, row, tmp_path / "front.pkl")
    missing = dict(payload)
    missing.pop("reward_sequence")
    with pytest.raises(RuntimeError, match="reward_sequence"):
        validate_payload_diagnostics(missing, row, tmp_path / "front.pkl")
    mixed = dict(payload)
    mixed["ppo_update_stats"] = [dict(payload["ppo_update_stats"][0], updated_policy_version=2)]
    with pytest.raises(RuntimeError, match="policy versions"):
        validate_payload_diagnostics(mixed, row, tmp_path / "front.pkl")


def test_shared_v6_contrasts_report_direction_and_familywise_reproduction(tmp_path):
    medians = instance_seed_medians(_formal_rows())
    v8, _ = inferential_analysis(medians, bootstrap_reps=20, sign_flip_reps=50)
    v8_hv = v8[v8["metric"] == "HV_fixed"].set_index("contrast")
    rows = []
    for v8_name, v6_name in V6_SHARED_CONTRASTS.items():
        current = v8_hv.loc[v8_name]
        rows.append({
            "metric": "HV_fixed", "contrast": v6_name,
            "median_delta_oriented": current["median_delta_oriented"],
            "p_holm_within_family": current["p_holm_within_family"],
        })
    path = tmp_path / "v6.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = replication_comparison(v8, path)
    assert len(result) == 9
    assert result["direction_reproduced"].all()
    assert result["familywise_decision_reproduced"].all()
