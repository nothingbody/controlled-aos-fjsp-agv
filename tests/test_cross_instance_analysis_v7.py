import numpy as np
import pandas as pd

from scripts.analyze_cross_instance_pretraining_v7 import (
    amortization_tables,
    hierarchical_fold_bootstrap,
    online_data_economy,
    pretraining_chain_cost,
    signed_rank_randomization,
)


def test_signed_rank_exact_known_small_case():
    statistic, p_value, nonzero, method = signed_rank_randomization(
        np.asarray([1.0, 2.0]), reps=10, seed=3
    )
    assert statistic == 0.0
    assert p_value == 0.5
    assert nonzero == 2
    assert method == "exact sign enumeration"


def test_signed_rank_all_ties_is_neutral():
    statistic, p_value, nonzero, method = signed_rank_randomization(
        np.zeros(50), reps=20, seed=4
    )
    assert (statistic, p_value, nonzero, method) == (0.0, 1.0, 0, "all ties")


def test_hierarchical_bootstrap_preserves_constant_effect():
    block = pd.DataFrame({
        "Fold": np.repeat(np.arange(1, 6), 10),
        "delta": np.ones(50),
    })
    assert hierarchical_fold_bootstrap(block, reps=100, seed=11) == (1.0, 1.0)


def test_hierarchical_bootstrap_requires_formal_cluster_shape():
    block = pd.DataFrame({"Fold": [1, 2], "delta": [1.0, 2.0]})
    try:
        hierarchical_fold_bootstrap(block, reps=10, seed=1)
    except ValueError as error:
        assert "five folds of ten" in str(error)
    else:
        raise AssertionError("invalid fold layout was accepted")


def _pretraining_fixture():
    rows = []
    for fold in range(1, 6):
        for replica in range(5):
            cumulative = 0
            for episode in range(80):
                cumulative += 20_100
                rows.append({
                    "Fold": fold, "Replica": replica,
                    "Pass": episode // 40 + 1, "Position": episode % 40 + 1,
                    "Pretrain_seed": 7042 + replica,
                    "Cumulative_objective_evaluations": cumulative,
                    "Cumulative_PPO_samples": (episode + 1) * 192,
                    "Cumulative_PPO_collected_transitions": (episode + 1) * 193,
                    "Cumulative_PPO_discarded_singletons": episode + 1,
                    "Cumulative_PPO_updates": (episode + 1) * 12,
                    "Cumulative_PPO_optimizer_steps": (episode + 1) * 48,
                    "Cumulative_CPU_seconds": float((episode + 1) * 10),
                    "Cumulative_wall_seconds": float((episode + 1) * 11),
                })
    return pd.DataFrame(rows)


def test_pretraining_chain_cost_uses_terminal_cumulative_rows():
    terminal = pretraining_chain_cost(_pretraining_fixture())
    assert len(terminal) == 25
    assert set(terminal["Episodes"]) == {80}
    assert set(terminal["Generations"]) == {16_000}
    assert set(terminal["Cumulative_objective_evaluations"]) == {1_608_000}


def _evaluation_fixture():
    rows = []
    variants = (
        "UCBOnly", "ScratchNoBC_R16", "XPrePPO_Frozen", "XPrePPO_Online_R16"
    )
    for budget in (50, 100, 200):
        for variant in variants:
            rows.append({
                "variant": variant, "Budget": budget,
                "Initial_evaluations": 100,
                "Offspring_evaluations": 100 * budget,
                "Online_PPO_samples": 0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 32,
                "Online_PPO_collected_transitions": 0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 33,
                "Online_PPO_discarded_singletons": 0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 1,
                "PPO_controlled_actions": 0 if variant == "UCBOnly" else 40,
                "Online_PPO_updates": 0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 2,
                "Online_PPO_optimizer_steps": 0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 8,
                "learning_time": 0.0 if variant in {"UCBOnly", "XPrePPO_Frozen"} else 1.0,
                "elapsed_seconds": 10.0,
                "cpu_seconds": {
                    "UCBOnly": 9.0,
                    "ScratchNoBC_R16": 12.0,
                    "XPrePPO_Frozen": 9.5,
                    "XPrePPO_Online_R16": 10.0,
                }[variant],
                "Offline_checkpoint_bytes": 1000 if variant.startswith("XPre") else 0,
            })
    return pd.DataFrame(rows)


def test_data_economy_and_break_even_keep_offline_cost_explicit():
    evaluation = _evaluation_fixture()
    online = online_data_economy(evaluation)
    scratch = online[(online["variant"] == "ScratchNoBC_R16") & (online["Budget"] == 100)].iloc[0]
    assert scratch["consumed_transition_fraction"] == 0.8
    amortized, break_even = amortization_tables(
        evaluation, pretraining_chain_cost(_pretraining_fixture())
    )
    one = amortized[
        (amortized["variant"] == "XPrePPO_Online_R16")
        & (amortized["Budget"] == 100)
        & (amortized["deployments_per_checkpoint"] == 1)
    ].iloc[0]
    assert one["offline_objective_evaluations_per_deployment"] == 1_608_000
    ucb = break_even[
        (break_even["Budget"] == 100) & (break_even["comparator"] == "UCBOnly")
    ].iloc[0]
    scratch_be = break_even[
        (break_even["Budget"] == 100)
        & (break_even["comparator"] == "ScratchNoBC_R16")
    ].iloc[0]
    assert np.isnan(ucb["CPU_break_even_deployments"])
    assert scratch_be["CPU_break_even_deployments"] == 400.0
