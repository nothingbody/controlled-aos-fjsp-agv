import numpy as np
import torch

from experiments.run_mechanism_robustness_v6 import (
    BASE_STATE_DIM,
    ENHANCED_STATE_DIM,
    AuditedHybridSelector,
    CONFIGS,
    PROTOCOL,
    nearest_neighbor_label_disagreement,
)
from scripts.analyze_mechanism_robustness_v6 import exact_signed_rank
from scripts.analyze_resubmission_v5 import holm_adjust


def test_prespecified_grid_has_1100_runs():
    runs = 10 * 10 * (len(CONFIGS[100]) + len(CONFIGS[200]))
    assert runs == 1100
    assert PROTOCOL == "saos_mechanism_robustness_v6_20260722"


def test_enhanced_state_contains_finite_context_features():
    selector = AuditedHybridSelector(
        np.random.RandomState(7),
        state_mode="enhanced",
        transition_mode="adaptive",
        min_per_op=3,
        min_buffer=30,
        use_behavior_cloning=False,
        rollout_size=8,
        device="cpu",
    )
    state = selector.build_state(0, 100, 0, 0.1, 0.0, switched=False)
    assert BASE_STATE_DIM == 24
    assert ENHANCED_STATE_DIM == 78
    assert state.shape == (ENHANCED_STATE_DIM,)
    assert np.isfinite(state).all()


def test_original_state_remains_the_frozen_24_dimensions():
    selector = AuditedHybridSelector(
        np.random.RandomState(8),
        state_mode="original24",
        transition_mode="adaptive",
        min_per_op=3,
        min_buffer=30,
        use_behavior_cloning=False,
        rollout_size=8,
        device="cpu",
    )
    state = selector.build_state(0, 100, 0, 0.1, 0.0, switched=False)
    assert state.shape == (BASE_STATE_DIM,)


def test_padded_base_matches_enhanced_network_dimension_without_new_information():
    selector = AuditedHybridSelector(
        np.random.RandomState(9),
        state_mode="base_padded",
        transition_mode="adaptive",
        min_per_op=3,
        min_buffer=30,
        use_behavior_cloning=False,
        rollout_size=8,
        device="cpu",
    )
    state = selector.build_state(0, 100, 0, 0.1, 0.0, switched=False)
    assert state.shape == (ENHANCED_STATE_DIM,)
    assert np.count_nonzero(state[BASE_STATE_DIM:]) == 0


def test_padded_and_enhanced_models_have_identical_seeded_initialization():
    common = dict(
        transition_mode="adaptive",
        min_per_op=3,
        min_buffer=30,
        use_behavior_cloning=False,
        rollout_size=8,
        rng_seeds={"network": 11, "action": 12, "bc": 13, "ppo": 14},
        device="cpu",
    )
    padded = AuditedHybridSelector(
        np.random.RandomState(10), state_mode="base_padded", **common
    )
    enhanced = AuditedHybridSelector(
        np.random.RandomState(10), state_mode="enhanced", **common
    )
    padded_state = padded.ppo.policy.state_dict()
    enhanced_state = enhanced.ppo.policy.state_dict()
    assert padded_state.keys() == enhanced_state.keys()
    assert sum(value.numel() for value in padded_state.values()) == sum(
        value.numel() for value in enhanced_state.values()
    )
    assert all(
        torch.equal(padded_state[key], enhanced_state[key]) for key in padded_state
    )


def test_nearest_neighbor_disagreement_is_bounded():
    states = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0]], dtype=np.float32)
    actions = [0, 0, 1]
    value = nearest_neighbor_label_disagreement(states, actions)
    assert 0.0 <= value <= 1.0


def test_exact_signed_rank_and_holm_known_cases():
    statistic, p_value, n_nonzero = exact_signed_rank([1.0, 2.0])
    assert statistic == 0.0
    assert p_value == 0.5
    assert n_nonzero == 2
    assert exact_signed_rank([0.0, 0.0]) == (0.0, 1.0, 0)
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
