import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.run_cross_instance_pretraining_v7 import (
    ENHANCED_STATE_DIM,
    FORMAL_EVAL_ROWS,
    FORMAL_PRETRAIN_ROWS,
    FROZEN_TEST_SPLIT,
    CrossInstanceSelector,
    PROTOCOL,
    atomic_pickle,
    build_training_orders,
    checkpoint_marker_path,
    checkpoint_path,
    formal_design,
    generated_test_split,
    front_semantic_hash,
    instance_input_records,
    load_all_instances,
    load_checkpoint,
    is_deduplicated_nondominated,
    journal_evaluation_row,
    recover_evaluation_journal,
    run_pretraining_chain,
    file_sha256,
    tensor_state_semantic_hash,
    training_keys_for_fold,
    validate_split,
    write_checkpoint,
)
from src.algorithm.grl.ppo_agent import PPOAgent


@pytest.fixture(scope="module")
def benchmark_context():
    instances = load_all_instances()
    inputs = instance_input_records()
    split = validate_split(instances, inputs)
    return instances, inputs, split


def test_split_is_generated_without_outcomes_and_matches_frozen_table(
    benchmark_context,
):
    instances, _, split = benchmark_context
    assert generated_test_split(instances) == split == {
        fold: {
            family: tuple(sorted(names)) for family, names in families.items()
        }
        for fold, families in FROZEN_TEST_SPLIT.items()
    }
    assert all(len(split[fold]["Brandimarte"]) == 2 for fold in split)
    assert all(len(split[fold]["Hurink_edata"]) == 8 for fold in split)


def test_split_is_exhaustive_and_disjoint_by_identity_and_content_hash(
    benchmark_context,
):
    instances, inputs, split = benchmark_context
    all_keys = {(dataset, name) for dataset, name, _ in instances}
    observed_keys = []
    observed_hashes = []
    for fold in split:
        keys = {
            (family, name)
            for family, names in split[fold].items()
            for name in names
        }
        observed_keys.extend(keys)
        observed_hashes.extend(
            inputs[key]["canonical_token_sha256"] for key in keys
        )
        training = set(training_keys_for_fold(fold, all_keys))
        assert len(training) == 40
        assert not training & keys
    assert len(observed_keys) == len(set(observed_keys)) == 50
    assert set(observed_keys) == all_keys
    assert len(observed_hashes) == len(set(observed_hashes)) == 50


def test_formal_grid_sizes_are_frozen(benchmark_context):
    _, _, split = benchmark_context
    design = formal_design(split)
    assert design["expected_pretraining_rows"] == FORMAL_PRETRAIN_ROWS == 2000
    assert design["expected_evaluation_rows"] == FORMAL_EVAL_ROWS == 6000
    assert design["transfer"] == (
        "actor_and_critic_parameters_fresh_optimizer_empty_buffer"
    )


def test_training_orders_are_deterministic_and_replica_specific(benchmark_context):
    instances, _, _ = benchmark_context
    keys = {(dataset, name) for dataset, name, _ in instances}
    training = training_keys_for_fold(1, keys)
    first = build_training_orders(1, 0, training)
    repeated = build_training_orders(1, 0, training)
    other = build_training_orders(1, 1, training)
    assert first == repeated
    assert first != other
    assert all(set(order) == set(training) for order in first)
    assert len(first) == 2 and all(len(order) == 40 for order in first)


def _make_selector_state(seed=123):
    selector = CrossInstanceSelector(
        np.random.RandomState(seed),
        rng_seeds={
            "network": seed,
            "action": seed + 1,
            "bc": seed + 2,
            "ppo": seed + 3,
        },
    )
    return selector.ppo.export_training_state()


def test_checkpoint_roundtrip_binds_metadata_and_weights(tmp_path):
    initial = _make_selector_state(101)
    trained = copy.deepcopy(initial)
    with torch.no_grad():
        trained["policy"]["actor.0.bias"][0] += 0.25
    payload = {
        "protocol": "saos_cross_instance_pretrained_ppo_v7_20260722",
        "metadata": {
            "fold": 1,
            "replica": 0,
            "completed_episodes": 80,
            "code_hash": "code",
            "design_hash": "design",
            "input_hash": "input",
            "split_hash": "split",
        },
        "initial_training_state": initial,
        "training_state": trained,
        "episode_records": [],
    }
    path = tmp_path / "checkpoint.pt"
    marker = write_checkpoint(path, payload)
    assert checkpoint_marker_path(path).is_file()
    loaded, observed_marker = load_checkpoint(
        path,
        expected={
            "fold": 1,
            "replica": 0,
            "completed_episodes": 80,
            "code_hash": "code",
            "design_hash": "design",
            "input_hash": "input",
            "split_hash": "split",
        },
    )
    assert observed_marker == marker
    assert tensor_state_semantic_hash(loaded["training_state"]["policy"]) == (
        marker["weight_semantic_sha256"]
    )


def test_tampered_checkpoint_is_rejected(tmp_path):
    state = _make_selector_state(202)
    payload = {
        "protocol": "saos_cross_instance_pretrained_ppo_v7_20260722",
        "metadata": {"fold": 2, "replica": 1, "completed_episodes": 1},
        "initial_training_state": state,
        "training_state": state,
        "episode_records": [],
    }
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(path, payload)
    marker = json.loads(checkpoint_marker_path(path).read_text(encoding="utf-8"))
    object_path = checkpoint_marker_path(path).parent / marker["object_path"]
    with object_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        load_checkpoint(path)


def test_checkpoint_pointer_can_be_recreated_without_partial_overwrite(tmp_path):
    state = _make_selector_state(203)
    payload = {
        "protocol": PROTOCOL,
        "metadata": {"fold": 2, "replica": 1, "completed_episodes": 1},
        "initial_training_state": state,
        "training_state": state,
        "episode_records": [],
    }
    path = tmp_path / "checkpoint.pt"
    first = write_checkpoint(path, payload)
    checkpoint_marker_path(path).unlink()
    assert len(list((tmp_path / "objects").glob("*.pt"))) == 1
    second = write_checkpoint(path, payload)
    assert (
        first["weight_semantic_sha256"]
        == second["weight_semantic_sha256"]
    )
    loaded, _ = load_checkpoint(path)
    assert loaded["metadata"] == payload["metadata"]


def test_transfer_load_resets_optimizer_and_rollout_buffer():
    state = _make_selector_state(303)
    selector = CrossInstanceSelector(
        np.random.RandomState(1),
        training_state=state,
        load_optimizer=False,
        frozen=False,
        rng_seeds={"network": 8, "action": 9, "bc": 10, "ppo": 11},
    )
    assert len(selector.ppo.buffer) == 0
    assert selector.ppo.state_dim == ENHANCED_STATE_DIM
    assert selector.ppo.optimizer.state_dict()["state"] == {}
    assert tensor_state_semantic_hash(selector.ppo.policy.state_dict()) == (
        tensor_state_semantic_hash(state["policy"])
    )


def test_transfer_rejects_hyperparameter_mismatch():
    state = _make_selector_state(304)
    state["hyperparameters"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="incompatible PPO hyperparameters"):
        CrossInstanceSelector(
            np.random.RandomState(1),
            training_state=state,
            load_optimizer=False,
            rng_seeds={"network": 8, "action": 9, "bc": 10, "ppo": 11},
        )


def test_frozen_selector_never_collects_test_rollouts():
    state = _make_selector_state(404)
    selector = CrossInstanceSelector(
        np.random.RandomState(2),
        training_state=state,
        load_optimizer=False,
        frozen=True,
        rng_seeds={"network": 12, "action": 13, "bc": 14, "ppo": 15},
    )
    selector.switched = True
    selector.transition_gen = 0
    action = selector.select(0, 10, 0, 0.1, 0.0)
    selector.update(action, 0.5)
    selector.finalize()
    assert len(selector.ppo.buffer) == 0
    assert selector.ppo.policy_version == state["policy_version"]


def test_ppo_update_rejects_mixed_behavior_policy_versions():
    agent = PPOAgent(
        state_dim=3, action_dim=2, network_seed=1, action_seed=2, ppo_seed=3
    )
    agent.select_and_store(np.zeros(3, dtype=np.float32))
    agent.store_reward(0.1)
    agent.select_and_store(np.ones(3, dtype=np.float32))
    agent.store_reward(0.2)
    agent.buffer.policy_versions[-1] += 1
    with pytest.raises(RuntimeError, match="mixes behavior-policy versions"):
        agent.update()


def test_ppo_update_records_policy_version_and_clears_buffer():
    agent = PPOAgent(
        state_dim=3, action_dim=2, network_seed=4, action_seed=5, ppo_seed=6
    )
    for reward in (0.1, 0.2):
        agent.select_and_store(np.full(3, reward, dtype=np.float32))
        agent.store_reward(reward)
    stats = agent.update()
    assert stats["behavior_policy_version"] == 0
    assert stats["updated_policy_version"] == 1
    assert agent.policy_version == 1
    assert len(agent.buffer) == 0


def test_front_integrity_gate_rejects_duplicate_or_dominated_points():
    assert is_deduplicated_nondominated(np.array([[1.0, 2.0, 3.0]]))
    assert not is_deduplicated_nondominated(
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    )
    assert not is_deduplicated_nondominated(
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    )


def test_atomic_evaluation_journal_recovers_interrupted_csv(tmp_path):
    objectives = np.array([[1.0, 2.0, 3.0]], dtype=float)
    hashes = {
        "code_hash": "code",
        "design_hash": "design",
        "input_hash": "input",
        "split_hash": "split",
        "reference_snapshot_sha256": "reference",
    }
    front = tmp_path / "fronts" / "one.pkl"
    payload = {
        "protocol": PROTOCOL,
        "fold": 1,
        "dataset": "Brandimarte",
        "instance": "Mk04.fjs",
        "variant": "UCBOnly",
        "budget": 2,
        "seed": 42,
        "replica": -1,
        "code_hash": hashes["code_hash"],
        "design_hash": hashes["design_hash"],
        "input_hash": hashes["input_hash"],
        "split_hash": hashes["split_hash"],
        "reference_snapshot_sha256": hashes["reference_snapshot_sha256"],
        "checkpoint_sha256": "none",
        "objectives": objectives,
    }
    atomic_pickle(front, payload)
    row = {
        "Protocol": PROTOCOL,
        "Fold": 1,
        "dataset": "Brandimarte",
        "instance": "Mk04.fjs",
        "variant": "UCBOnly",
        "Budget": 2,
        "seed": 42,
        "Replica": -1,
        "Code_hash": hashes["code_hash"],
        "Design_hash": hashes["design_hash"],
        "Input_hash": hashes["input_hash"],
        "Split_hash": hashes["split_hash"],
        "Reference_snapshot_sha256": hashes["reference_snapshot_sha256"],
        "Checkpoint_sha256": "none",
        "Front_sha256": file_sha256(front),
        "Front_semantic_sha256": front_semantic_hash(objectives),
        "front_pickle": str(front),
    }
    journal_evaluation_row(tmp_path, row)
    assert not (tmp_path / "runs.csv").exists()
    recovered = recover_evaluation_journal(tmp_path, hashes)
    first_csv = (tmp_path / "runs.csv").read_bytes()
    assert len(recovered) == 1
    recovered_again = recover_evaluation_journal(tmp_path, hashes)
    assert len(recovered_again) == 1
    assert (tmp_path / "runs.csv").read_bytes() == first_csv


def test_interrupted_pretraining_resume_matches_clean_scientific_state(tmp_path):
    hashes = ("code", "design", "input", "split")
    reference = "reference"

    def task(out_dir):
        return (
            1, 0, str(out_dir), *hashes,
            2, 2, 1, reference,
        )

    clean_dir = tmp_path / "clean"
    interrupted_dir = tmp_path / "interrupted"
    clean = run_pretraining_chain(task(clean_dir))
    run_pretraining_chain(task(interrupted_dir))
    pass1_marker = checkpoint_marker_path(
        checkpoint_path(interrupted_dir, 1, 0, "pass1")
    )
    progress_marker = checkpoint_marker_path(
        checkpoint_path(interrupted_dir, 1, 0, "progress")
    )
    progress_marker.write_bytes(pass1_marker.read_bytes())
    checkpoint_marker_path(
        checkpoint_path(interrupted_dir, 1, 0, "terminal")
    ).unlink()
    resumed = run_pretraining_chain(task(interrupted_dir))
    assert (
        clean["checkpoint_marker"]["weight_semantic_sha256"]
        == resumed["checkpoint_marker"]["weight_semantic_sha256"]
    )
    clean_scientific = [
        (row["Pass"], row["Position"], row["dataset"], row["instance"],
         row["Front_semantic_sha256"], row["PPO_samples"], row["PPO_updates"])
        for row in clean["episode_records"]
    ]
    resumed_scientific = [
        (row["Pass"], row["Position"], row["dataset"], row["instance"],
         row["Front_semantic_sha256"], row["PPO_samples"], row["PPO_updates"])
        for row in resumed["episode_records"]
    ]
    assert clean_scientific == resumed_scientific


def test_optimizer_state_survives_cross_episode_export_and_continuation():
    source = PPOAgent(
        state_dim=3, action_dim=2, batch_size=2,
        network_seed=10, action_seed=11, ppo_seed=12,
    )
    for value in (0.1, 0.2):
        source.select_and_store(np.full(3, value, dtype=np.float32))
        source.store_reward(value)
    source.update()
    exported = source.export_training_state()
    assert exported["optimizer"]["state"]

    continued_a = PPOAgent(
        state_dim=3, action_dim=2, batch_size=2,
        network_seed=20, action_seed=21, ppo_seed=22,
    )
    continued_b = PPOAgent(
        state_dim=3, action_dim=2, batch_size=2,
        network_seed=30, action_seed=21, ppo_seed=22,
    )
    continued_a.load_training_state(exported, load_optimizer=True)
    continued_b.load_training_state(exported, load_optimizer=True)
    for agent in (continued_a, continued_b):
        for value in (0.3, 0.4):
            agent.select_and_store(np.full(3, value, dtype=np.float32))
            agent.store_reward(value)
        agent.update()
    assert tensor_state_semantic_hash(continued_a.policy.state_dict()) == (
        tensor_state_semantic_hash(continued_b.policy.state_dict())
    )
    assert continued_a.policy_version == continued_b.policy_version == 2
