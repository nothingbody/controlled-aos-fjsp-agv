"""Regression tests for the UCB-to-PPO training boundary."""

import numpy as np
import torch

from experiments.run_revision_aos import HybridSelector, N_OPS
from src.algorithm.grl.ppo_agent import PPOAgent


def test_behavior_warm_start_does_not_pollute_on_policy_buffer():
    torch.manual_seed(3)
    rng = np.random.RandomState(3)
    agent = PPOAgent(state_dim=6, action_dim=3, batch_size=4, device="cpu")
    states = rng.normal(size=(12, 6)).astype(np.float32)
    actions = np.arange(12) % 3

    stats = agent.warm_start_behavior(states, actions, n_epochs=2, batch_size=4)

    assert len(agent.buffer) == 0
    assert stats["bc_samples"] == 12
    assert 0.0 <= stats["bc_final_accuracy"] <= 1.0
    assert len(stats["bc_epoch_loss"]) == 2
    assert len(stats["bc_epoch_accuracy"]) == 2
    assert np.isfinite(stats["bc_pre_post_kl"])
    assert np.asarray(stats["bc_confusion_matrix"]).shape == (3, 3)
    assert stats["bc_pre_loss"] >= 0.0
    assert 0.0 <= stats["bc_pre_accuracy"] <= 1.0


def test_controller_rng_streams_are_independent_of_bc_shuffle():
    kwargs = dict(
        state_dim=6,
        action_dim=3,
        device="cpu",
        network_seed=101,
        action_seed=202,
        ppo_seed=404,
    )
    first = PPOAgent(**kwargs, bc_seed=303)
    second = PPOAgent(**kwargs, bc_seed=999)
    states = np.arange(72, dtype=np.float32).reshape(12, 6) / 10.0
    actions = np.arange(12) % 3

    first.warm_start_behavior(states, actions, n_epochs=0, batch_size=4)
    second.warm_start_behavior(states, actions, n_epochs=0, batch_size=4)
    probe = np.ones(6, dtype=np.float32)

    assert [first.select(probe) for _ in range(20)] == [
        second.select(probe) for _ in range(20)
    ]


def test_ucb_demonstrations_are_separate_from_ppo_rollouts():
    torch.manual_seed(5)
    selector = HybridSelector(
        np.random.RandomState(5),
        transition_mode="adaptive",
        min_per_op=2,
        min_buffer=2 * N_OPS,
        min_stagnation=0,
        rollout_size=4,
        device="cpu",
    )

    for gen in range(2 * N_OPS):
        op_id = selector.select(gen, 30, 0, 0.1, 0.0)
        selector.update(op_id, 0.1)

    assert not selector.switched
    assert len(selector.demo_states) == 2 * N_OPS
    assert len(selector.ppo.buffer) == 0

    op_id = selector.select(2 * N_OPS, 30, 0, 0.1, 0.0)

    assert selector.switched
    assert selector.bc_stats["bc_samples"] == 2 * N_OPS
    assert len(selector.ppo.buffer) == 1
    assert selector.ppo.buffer.actions[-1] == op_id
    assert np.isfinite(selector.ppo.buffer.log_probs[-1])


def test_no_bc_ablation_switches_without_supervised_updates():
    selector = HybridSelector(
        np.random.RandomState(9),
        transition_mode="adaptive",
        min_per_op=2,
        min_buffer=2 * N_OPS,
        min_stagnation=0,
        rollout_size=4,
        use_behavior_cloning=False,
        device="cpu",
    )
    for gen in range(2 * N_OPS):
        op_id = selector.select(gen, 30, 0, 0.1, 0.0)
        selector.update(op_id, 0.1)

    selector.select(2 * N_OPS, 30, 0, 0.1, 0.0)

    assert selector.switched
    assert selector.bc_stats == {}
    assert len(selector.ppo.buffer) == 1


def test_ppo_update_records_training_diagnostics():
    torch.manual_seed(11)
    rng = np.random.RandomState(11)
    agent = PPOAgent(state_dim=6, action_dim=3, batch_size=4, n_epochs=2, device="cpu")
    for _ in range(4):
        state = rng.normal(size=6).astype(np.float32)
        agent.select_and_store(state)
        agent.store_reward(float(rng.normal()))
    stats = agent.update()

    for key in (
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "optimizer_steps",
        "advantage_std_raw",
        "return_std",
        "explained_variance_pre",
    ):
        assert key in stats
        assert np.isfinite(stats[key])
