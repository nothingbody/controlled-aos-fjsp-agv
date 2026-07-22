"""Prespecified SA-AOS mechanism-robustness experiment (v6).

The frozen design is documented in ``SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md``.
This runner writes only to a new v6 directory and never modifies v5 results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.loader import load_benchmark_set
from experiments.run_revision_aos import (
    BaseSelector,
    DATASETS,
    HybridSelector,
    N_OPS,
    UCBSelector,
    apply_operator,
    compute_reward,
    evaluate_archive,
    initialize_population,
    make_rng_streams,
    normalized_entropy,
    population_diversity,
    update_archive,
    update_generation_archive,
)
from src.algorithm.nsga3.selection import (
    compute_hypervolume,
    generate_reference_points,
    nsga3_select,
)


PROTOCOL = "saos_mechanism_robustness_v6_20260722"
DEFAULT_OUT = "results/resubmission/v6_mechanism"
REFERENCE_SNAPSHOT = "results/resubmission/v6_mechanism/frozen_reference_snapshot.pkl"
SELECTED_INSTANCES = {
    "Brandimarte": ["Mk01.fjs", "Mk03.fjs", "Mk05.fjs", "Mk08.fjs", "Mk10.fjs"],
    "Hurink_edata": ["la01.fjs", "la10.fjs", "la20.fjs", "la30.fjs", "la40.fjs"],
}

CONFIGS = {
    100: [
        {"variant": "UCBOnly", "state_mode": "none", "use_bc": False, "rollout": 0},
        {"variant": "Original24BC_R16", "state_mode": "original24", "use_bc": True, "rollout": 16},
        {"variant": "Original24NoBC_R16", "state_mode": "original24", "use_bc": False, "rollout": 16},
        {"variant": "BasePaddedBC_R16", "state_mode": "base_padded", "use_bc": True, "rollout": 16},
        {"variant": "BasePaddedNoBC_R16", "state_mode": "base_padded", "use_bc": False, "rollout": 16},
        {"variant": "EnhancedBC_R16", "state_mode": "enhanced", "use_bc": True, "rollout": 16},
        {"variant": "EnhancedNoBC_R16", "state_mode": "enhanced", "use_bc": False, "rollout": 16},
    ],
    200: [
        {"variant": "UCBOnly", "state_mode": "none", "use_bc": False, "rollout": 0},
        {"variant": "EnhancedBC_R8", "state_mode": "enhanced", "use_bc": True, "rollout": 8},
        {"variant": "EnhancedBC_R16", "state_mode": "enhanced", "use_bc": True, "rollout": 16},
        {"variant": "EnhancedBC_R32", "state_mode": "enhanced", "use_bc": True, "rollout": 32},
    ],
}

BASE_STATE_DIM = 2 * N_OPS + 4
ENHANCED_STATE_DIM = BASE_STATE_DIM + 5 * N_OPS + 4
MODEL_STATE_DIM = ENHANCED_STATE_DIM

FORMAL_SEED_START = 42
FORMAL_SEED_END = 52
FORMAL_POP_SIZE = 100
FORMAL_BUDGETS = (100, 200)
FORMAL_WORKERS = 40
CODE_FILES = (
    "experiments/run_mechanism_robustness_v6.py",
    "experiments/run_revision_aos.py",
    "src/algorithm/grl/ppo_agent.py",
    "src/algorithm/nsga3/encoding.py",
    "src/algorithm/nsga3/crossover.py",
    "src/algorithm/nsga3/mutation.py",
    "src/algorithm/nsga3/decoding.py",
    "src/algorithm/nsga3/selection.py",
    "src/problem/instance.py",
    "src/problem/energy_model.py",
    "src/utils/metrics.py",
    "data/loader.py",
    "scripts/analyze_mechanism_robustness_v6.py",
    "scripts/build_mechanism_reference_snapshot_v6.py",
    "SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md",
)
INPUT_FILES = tuple(
    f"{DATASETS[dataset]}/{instance}"
    for dataset, instances in SELECTED_INSTANCES.items()
    for instance in instances
)


def controller_rng_seeds(seed):
    sequence = np.random.SeedSequence([int(seed), 0x5A0A5])
    values = sequence.generate_state(4, dtype=np.uint32)
    return {
        name: int(value)
        for name, value in zip(("network", "action", "bc", "ppo"), values)
    }


def canonical_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def formal_design():
    return {
        "protocol": PROTOCOL,
        "population_size": FORMAL_POP_SIZE,
        "budgets": list(FORMAL_BUDGETS),
        "seed_start": FORMAL_SEED_START,
        "seed_end_exclusive": FORMAL_SEED_END,
        "instances": SELECTED_INSTANCES,
        "configs": CONFIGS,
        "expected_rows": 1100,
        "workers": FORMAL_WORKERS,
    }


def configuration_hash(config, budget, pop_size):
    return canonical_hash(
        {
            "protocol": PROTOCOL,
            "config": config,
            "budget": int(budget),
            "population_size": int(pop_size),
            "reward_scheme": "composite",
            "min_per_op": 3,
            "min_buffer": 30,
        }
    )


def _ucb_context(selector: "AuditedHybridSelector", max_gen: int, switched: bool):
    ucb = selector.ucb
    recent = ucb.history[-ucb.window :]
    recent_counts = np.zeros(N_OPS, dtype=np.float32)
    recent_means = np.zeros(N_OPS, dtype=np.float32)
    for op_id, reward in recent:
        recent_counts[op_id] += 1.0
        recent_means[op_id] += float(reward)
    nonzero = recent_counts > 0
    recent_means[nonzero] /= recent_counts[nonzero]
    total = float(recent_counts.sum())

    raw_scores = np.zeros(N_OPS, dtype=np.float32)
    if total > 0:
        raw_scores[nonzero] = recent_means[nonzero] + float(ucb.c) * np.sqrt(
            np.log(max(total, 1.0)) / recent_counts[nonzero]
        )
    finite_max = float(raw_scores[nonzero].max()) if np.any(nonzero) else 0.0
    raw_scores[~nonzero] = finite_max + 1.0
    score_features = np.tanh(raw_scores).astype(np.float32)

    forced_mask = (ucb.op_counts < ucb.min_count).astype(np.float32)
    missing_mask = (~nonzero).astype(np.float32)
    forced_active = float(np.any(forced_mask > 0))
    transition_coverage_ready = float(
        np.all(ucb.op_counts >= selector.min_per_op)
        and len(ucb.history) >= selector.min_buffer
    )
    recent_coverage_fraction = float(np.mean(nonzero))
    sorted_scores = np.sort(raw_scores)
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0

    extras = np.concatenate(
        [
            recent_counts / max(float(ucb.window), 1.0),
            ucb.op_counts.astype(np.float32) / max(float(max_gen), 1.0),
            score_features,
            forced_mask,
            missing_mask,
            np.asarray(
                [
                    forced_active,
                    transition_coverage_ready,
                    recent_coverage_fraction,
                    math.tanh(margin),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    diagnostics = {
        "forced_active": bool(forced_active),
        "transition_coverage_ready": bool(transition_coverage_ready),
        "raw_margin": margin,
    }
    return extras, diagnostics


class AuditedHybridSelector(HybridSelector):
    """Hybrid selector with either the frozen base state or enhanced UCB context."""

    def __init__(self, rng, *, state_mode: str, **kwargs):
        if state_mode not in {"original24", "base_padded", "enhanced"}:
            raise ValueError(f"unsupported state_mode={state_mode}")
        self.state_mode = state_mode
        self.demo_ucb_margins = []
        self.demo_forced_coverage = []
        self.ppo_enhanced_features = []
        self._last_context_diagnostics = {}
        state_dim = BASE_STATE_DIM if state_mode == "original24" else MODEL_STATE_DIM
        super().__init__(rng, state_dim=state_dim, **kwargs)

    def build_state(self, gen, max_gen, stagnation, diversity, hv_trend, switched=False):
        base = BaseSelector.build_state(
            self, gen, max_gen, stagnation, diversity, hv_trend, switched
        )
        _, diagnostics = _ucb_context(self, max_gen, switched)
        self._last_context_diagnostics = diagnostics
        if self.state_mode == "original24":
            return base
        if self.state_mode == "base_padded":
            state = np.concatenate(
                [base, np.zeros(MODEL_STATE_DIM - BASE_STATE_DIM, dtype=np.float32)]
            )
            return state.astype(np.float32)
        extras, diagnostics = _ucb_context(self, max_gen, switched)
        self._last_context_diagnostics = diagnostics
        if switched:
            self.ppo_enhanced_features.append(extras.copy())
        state = np.concatenate([base, extras]).astype(np.float32)
        if len(state) != ENHANCED_STATE_DIM:
            raise RuntimeError(f"enhanced state has {len(state)} features")
        return state

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        action = super().select(gen, max_gen, stagnation, diversity, hv_trend)
        if self.last_phase == "UCB":
            self.demo_ucb_margins.append(
                float(self._last_context_diagnostics.get("raw_margin", 0.0))
            )
            self.demo_forced_coverage.append(
                bool(self._last_context_diagnostics.get("forced_active", False))
            )
        return action


def nearest_neighbor_label_disagreement(states, actions):
    if len(states) < 2:
        return 0.0
    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(actions, dtype=int)
    std = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
    distances = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(y != y[nearest]))


def enhanced_feature_summary(selector):
    if not isinstance(selector, AuditedHybridSelector) or not selector.ppo_enhanced_features:
        return {}
    x = np.asarray(selector.ppo_enhanced_features, dtype=np.float64)
    std = x.std(axis=0)
    return {
        "samples": int(len(x)),
        "mean": x.mean(axis=0).tolist(),
        "std": std.tolist(),
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "constant_feature_fraction": float(np.mean(std < 1e-12)),
        "near_zero_fraction": float(np.mean(np.isclose(x, 0.0, atol=1e-8))),
        "near_one_fraction": float(np.mean(np.isclose(x, 1.0, atol=1e-8))),
    }


def make_selector(config, controller_rng, rng_seeds):
    if config["variant"] == "UCBOnly":
        return UCBSelector(controller_rng, reward_scheme="composite")
    return AuditedHybridSelector(
        controller_rng,
        state_mode=str(config["state_mode"]),
        reward_scheme="composite",
        transition_mode="adaptive",
        min_per_op=3,
        min_buffer=30,
        use_behavior_cloning=bool(config["use_bc"]),
        rollout_size=int(config["rollout"]),
        rng_seeds=rng_seeds,
        device="cpu",
    )


def load_selected_instances():
    loaded = []
    for dataset, path in DATASETS.items():
        wanted = set(SELECTED_INSTANCES[dataset])
        for inst_name, instance in load_benchmark_set(path, num_agv=3):
            if inst_name in wanted:
                loaded.append((dataset, inst_name, instance))
    expected = sum(len(names) for names in SELECTED_INSTANCES.values())
    if len(loaded) != expected:
        found = [(dataset, name) for dataset, name, _ in loaded]
        raise RuntimeError(f"loaded {len(loaded)} of {expected} selected instances: {found}")
    return loaded


def run_single(task):
    (
        dataset, inst_name, instance, config, seed, pop_size, max_gen, out_dir,
        code_hash, design_hash, input_hash, config_hash, worker_count,
    ) = task
    init_rng, evolution_rng, controller_rng = make_rng_streams(seed)
    rng_seeds = controller_rng_seeds(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    start = time.time()

    ref_points = generate_reference_points(3, 8)
    population = initialize_population(instance, pop_size, init_rng)
    selector = make_selector(config, controller_rng, rng_seeds)
    archive = update_archive([], population)

    pop_objs = np.asarray([c.objectives.to_array() for c in population])
    online_reference = pop_objs.max(axis=0) * 1.1
    prev_hv = compute_hypervolume(pop_objs, online_reference)
    prev_best_cmax = float(pop_objs[:, 0].min())
    stagnation = 0
    hv_window = deque(maxlen=20)
    op_sequence = []
    reward_sequence = []
    hv_delta_sequence = []

    for gen in range(max_gen):
        pop_objs = np.asarray([c.objectives.to_array() for c in population])
        diversity = population_diversity(pop_objs)
        hv_trend = float(np.mean(hv_window)) if hv_window else 0.0
        op_id = selector.select(gen, max_gen, stagnation, diversity, hv_trend)
        offspring = apply_operator(population, op_id, pop_size, 5, evolution_rng)
        for chromosome in offspring:
            chromosome._revision_uid = id(chromosome)

        combined = population + offspring
        combined_objs = np.asarray([c.objectives.to_array() for c in combined])
        archive = update_generation_archive(archive, population, offspring)
        selected = nsga3_select(
            combined_objs, pop_size, ref_points, rng=evolution_rng
        )
        new_population = [combined[index] for index in selected]

        selected_ids = {
            getattr(chromosome, "_revision_uid", None) for chromosome in new_population
        }
        survived = sum(
            getattr(chromosome, "_revision_uid", None) in selected_ids
            for chromosome in offspring
        )
        survival = survived / max(len(offspring), 1)
        new_objs = np.asarray([c.objectives.to_array() for c in new_population])
        current_hv = compute_hypervolume(new_objs, online_reference)
        hv_delta = float(
            np.clip(
                (current_hv - prev_hv) / max(abs(prev_hv), 1e-10), -1.0, 1.0
            )
        )
        current_best_cmax = float(new_objs[:, 0].min())
        cmax_delta = float(
            np.clip(
                (prev_best_cmax - current_best_cmax)
                / max(abs(prev_best_cmax), 1e-10),
                -1.0,
                1.0,
            )
        )
        reward = compute_reward(
            "composite", survival, hv_delta, cmax_delta, gen, max_gen
        )
        selector.update(op_id, reward)

        hv_window.append(hv_delta)
        op_sequence.append(int(op_id))
        reward_sequence.append(float(reward))
        hv_delta_sequence.append(hv_delta)
        stagnation = stagnation + 1 if current_hv <= prev_hv * 1.001 else 0
        prev_hv = current_hv
        prev_best_cmax = current_best_cmax
        population = new_population

    if hasattr(selector, "finalize"):
        selector.finalize()

    metrics = evaluate_archive(archive)
    objectives = metrics.pop("objectives")
    if len(objectives) == 0 or not np.isfinite(objectives).all():
        raise RuntimeError("empty or non-finite final front")
    counts = np.bincount(np.asarray(op_sequence, dtype=int), minlength=N_OPS)
    last_counts = np.bincount(
        np.asarray(op_sequence[-20:], dtype=int), minlength=N_OPS
    )
    ppo_stats = list(getattr(selector, "ppo_update_stats", []))
    bc_stats = dict(getattr(selector, "bc_stats", {}))
    contexts = [str(item.get("update_context", "unknown")) for item in ppo_stats]

    def context_sum(key, context):
        return float(
            sum(float(item.get(key, 0.0)) for item in ppo_stats
                if item.get("update_context") == context)
        )

    demo_actions = list(getattr(selector, "demo_actions", []))
    demo_states = list(getattr(selector, "demo_states", []))
    demo_counts = np.bincount(np.asarray(demo_actions, dtype=int), minlength=N_OPS)
    feature_summary = enhanced_feature_summary(selector)

    pkl_dir = Path(out_dir) / "fronts"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    variant = str(config["variant"])
    pkl_path = pkl_dir / f"{dataset}_{Path(inst_name).stem}_{variant}_g{max_gen}_seed{seed}.pkl"
    temporary_path = pkl_path.with_suffix(
        pkl_path.suffix + f".tmp-{os.getpid()}"
    )
    with open(temporary_path, "wb") as stream:
        pickle.dump(
            {
                "protocol": PROTOCOL,
                "dataset": dataset,
                "instance": inst_name,
                "variant": variant,
                "budget": max_gen,
                "seed": seed,
                "config": config,
                "config_hash": config_hash,
                "code_hash": code_hash,
                "design_hash": design_hash,
                "input_hash": input_hash,
                "population_size": int(pop_size),
                "rng_seeds": rng_seeds,
                "objectives": objectives,
                "operator_sequence": op_sequence,
                "reward_sequence": reward_sequence,
                "hv_delta_sequence": hv_delta_sequence,
                "ppo_update_stats": ppo_stats,
                "behavior_cloning_stats": bc_stats,
                "demo_actions": demo_actions,
                "demo_ucb_margins": list(
                    getattr(selector, "demo_ucb_margins", [])
                ),
                "demo_forced_coverage": list(
                    getattr(selector, "demo_forced_coverage", [])
                ),
                "enhanced_feature_summary": feature_summary,
            },
            stream,
        )
    os.replace(temporary_path, pkl_path)
    front_hash = file_sha256(pkl_path)

    def mean_update_stat(key):
        values = [float(item[key]) for item in ppo_stats if key in item]
        return float(np.mean(values)) if values else 0.0

    row = {
        "Protocol": PROTOCOL,
        "dataset": dataset,
        "instance": inst_name,
        "variant": variant,
        "Budget": int(max_gen),
        "seed": int(seed),
        "Population_size": int(pop_size),
        "Max_generations": int(max_gen),
        "Config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "Config_hash": config_hash,
        "Code_hash": code_hash,
        "Design_hash": design_hash,
        "Input_hash": input_hash,
        "Worker_count": int(worker_count),
        "Front_sha256": front_hash,
        "Network_seed": int(rng_seeds["network"]),
        "Action_seed": int(rng_seeds["action"]),
        "BC_seed": int(rng_seeds["bc"]),
        "PPO_seed": int(rng_seeds["ppo"]),
        "state_mode": str(config["state_mode"]),
        "state_dim": int(
            ENHANCED_STATE_DIM
            if config["state_mode"] == "enhanced"
            else MODEL_STATE_DIM if config["state_mode"] == "base_padded"
            else BASE_STATE_DIM if config["state_mode"] == "original24" else 0
        ),
        "use_bc": bool(config["use_bc"]),
        "rollout_size_planned": int(config["rollout"]),
        **metrics,
        "Time": float(time.time() - start),
        "Learning_time": float(getattr(selector, "learning_time_seconds", 0.0)),
        "BC_time": float(getattr(selector, "bc_time_seconds", 0.0)),
        "Execution_device": "cpu",
        "Transition_gen": int(getattr(selector, "transition_gen", -1)),
        "Transition_reason": str(getattr(selector, "transition_reason", "none")),
        "Entropy_all": normalized_entropy(counts),
        "Entropy_last20": normalized_entropy(last_counts),
        "Operator_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        "Demo_samples": int(len(demo_actions)),
        "Demo_action_counts": json.dumps(demo_counts.tolist(), separators=(",", ":")),
        "Demo_label_entropy": normalized_entropy(demo_counts),
        "Demo_nn_disagreement": nearest_neighbor_label_disagreement(
            demo_states, demo_actions
        ),
        "Demo_ucb_margin_median": float(
            np.median(getattr(selector, "demo_ucb_margins", [0.0]))
        ) if demo_actions else 0.0,
        "Demo_forced_fraction": float(
            np.mean(getattr(selector, "demo_forced_coverage", [0.0]))
        ) if demo_actions else 0.0,
        "BC_pre_loss": float(bc_stats.get("bc_pre_loss", 0.0)),
        "BC_pre_accuracy": float(bc_stats.get("bc_pre_accuracy", 0.0)),
        "BC_final_loss": float(bc_stats.get("bc_final_loss", 0.0)),
        "BC_final_accuracy": float(bc_stats.get("bc_final_accuracy", 0.0)),
        "BC_pre_post_KL": float(bc_stats.get("bc_pre_post_kl", 0.0)),
        "BC_epoch_loss": json.dumps(bc_stats.get("bc_epoch_loss", []), separators=(",", ":")),
        "BC_epoch_accuracy": json.dumps(bc_stats.get("bc_epoch_accuracy", []), separators=(",", ":")),
        "BC_confusion_matrix": json.dumps(bc_stats.get("bc_confusion_matrix", []), separators=(",", ":")),
        "PPO_update_count": int(len(ppo_stats)),
        "PPO_action_effective_updates": int(contexts.count("pre_action")),
        "PPO_terminal_full_updates": int(contexts.count("terminal_full")),
        "PPO_terminal_residual_updates": int(contexts.count("terminal_residual")),
        "PPO_samples": int(sum(int(item.get("rollout_size", 0)) for item in ppo_stats)),
        "PPO_optimizer_steps": int(sum(int(item.get("optimizer_steps", 0)) for item in ppo_stats)),
        "PPO_effective_samples": int(context_sum("rollout_size", "pre_action")),
        "PPO_terminal_full_samples": int(context_sum("rollout_size", "terminal_full")),
        "PPO_terminal_residual_samples": int(context_sum("rollout_size", "terminal_residual")),
        "PPO_effective_optimizer_steps": int(context_sum("optimizer_steps", "pre_action")),
        "PPO_terminal_full_optimizer_steps": int(context_sum("optimizer_steps", "terminal_full")),
        "PPO_terminal_residual_optimizer_steps": int(context_sum("optimizer_steps", "terminal_residual")),
        "PPO_effective_update_time": context_sum("update_seconds", "pre_action"),
        "PPO_terminal_full_update_time": context_sum("update_seconds", "terminal_full"),
        "PPO_terminal_residual_update_time": context_sum("update_seconds", "terminal_residual"),
        "PPO_phase_actions": int(
            max_gen - int(getattr(selector, "transition_gen", max_gen))
            if int(getattr(selector, "transition_gen", -1)) >= 0 else 0
        ),
        "PPO_discarded_singletons": int(
            getattr(selector, "ppo_discarded_singletons", 0)
        ),
        "PPO_rollout_sizes": json.dumps(
            [int(item.get("rollout_size", 0)) for item in ppo_stats],
            separators=(",", ":"),
        ),
        "PPO_policy_loss_mean": mean_update_stat("policy_loss"),
        "PPO_value_loss_mean": mean_update_stat("value_loss"),
        "PPO_entropy_mean": mean_update_stat("entropy"),
        "PPO_approx_KL_mean": mean_update_stat("approx_kl"),
        "PPO_clip_fraction_mean": mean_update_stat("clip_fraction"),
        "PPO_gradient_norm_mean": mean_update_stat("gradient_norm"),
        "PPO_explained_variance_pre_mean": mean_update_stat("explained_variance_pre"),
        "PPO_advantage_std_raw_mean": mean_update_stat("advantage_std_raw"),
        "PPO_return_std_mean": mean_update_stat("return_std"),
        "Enhanced_feature_samples": int(feature_summary.get("samples", 0)),
        "Enhanced_constant_feature_fraction": float(
            feature_summary.get("constant_feature_fraction", 0.0)
        ),
        "Enhanced_near_zero_fraction": float(
            feature_summary.get("near_zero_fraction", 0.0)
        ),
        "Enhanced_near_one_fraction": float(
            feature_summary.get("near_one_fraction", 0.0)
        ),
        "front_pickle": str(pkl_path),
    }
    if (
        len(reward_sequence) > 2
        and np.std(reward_sequence) > 1e-12
        and np.std(hv_delta_sequence) > 1e-12
    ):
        row["Reward_HV_corr"] = float(
            np.corrcoef(reward_sequence, hv_delta_sequence)[0, 1]
        )
    else:
        row["Reward_HV_corr"] = 0.0
    return row


def existing_keys(
    csv_path, *, code_hash, design_hash, input_hash, pop_size, worker_count
):
    if not Path(csv_path).exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        budget = int(row["Budget"])
        variant = row["variant"]
        config = next(
            item for item in CONFIGS[budget] if item["variant"] == variant
        )
        expected_config_hash = configuration_hash(config, budget, pop_size)
        if (
            row.get("Protocol") != PROTOCOL
            or row.get("Code_hash") != code_hash
            or row.get("Design_hash") != design_hash
            or row.get("Input_hash") != input_hash
            or row.get("Config_hash") != expected_config_hash
            or int(row.get("Population_size", -1)) != int(pop_size)
            or int(row.get("Worker_count", -1)) != int(worker_count)
            or int(row.get("Max_generations", -1)) != budget
        ):
            raise RuntimeError(
                "existing CSV is incompatible with the frozen run manifest: "
                f"{row.get('dataset')}/{row.get('instance')}/{variant}/g{budget}/seed{row.get('seed')}"
            )
        front_path = Path(row.get("front_pickle", ""))
        if (
            not front_path.is_file()
            or file_sha256(front_path) != row.get("Front_sha256")
        ):
            raise RuntimeError(f"existing front is missing or modified: {front_path}")
    return {
        (row["dataset"], row["instance"], row["variant"], int(row["Budget"]), int(row["seed"]))
        for row in rows
    }


def append_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(args):
    instances = load_selected_instances()
    wanted_budgets = {int(value) for value in args.budgets.split(",") if value.strip()}
    wanted_variants = (
        {value.strip() for value in args.variants.split(",") if value.strip()}
        if args.variants
        else None
    )
    done = existing_keys(
        args.result_file,
        code_hash=args.code_hash,
        design_hash=args.design_hash,
        input_hash=args.input_hash,
        pop_size=args.pop_size,
        worker_count=args.workers,
    )
    tasks = []
    for budget in sorted(wanted_budgets):
        if budget not in CONFIGS:
            raise ValueError(f"unsupported budget {budget}")
        configs = [
            config
            for config in CONFIGS[budget]
            if wanted_variants is None or config["variant"] in wanted_variants
        ]
        for dataset, inst_name, instance in instances:
            for config in configs:
                for seed in range(args.seed_start, args.seed_end):
                    key = (dataset, inst_name, config["variant"], budget, seed)
                    if key not in done:
                        tasks.append(
                            (
                                dataset,
                                inst_name,
                                instance,
                                config,
                                seed,
                                args.pop_size,
                                budget,
                                args.out_dir,
                                args.code_hash,
                                args.design_hash,
                                args.input_hash,
                                configuration_hash(config, budget, args.pop_size),
                                args.workers,
                            )
                        )
    return tasks


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_sha256(path):
    """Hash parsed benchmark content independently of line-ending conversion."""
    tokens = Path(path).read_text(encoding="utf-8").split()
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def code_manifest():
    missing = [path for path in CODE_FILES if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"missing code files for hashing: {missing}")
    files = {path: file_sha256(path) for path in CODE_FILES}
    return {"files": files, "code_hash": canonical_hash(files)}


def input_manifest():
    missing = [path for path in INPUT_FILES if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"missing frozen benchmark inputs: {missing}")
    files = {
        path: {
            "canonical_token_sha256": canonical_text_sha256(path),
            "raw_sha256": file_sha256(path),
        }
        for path in INPUT_FILES
    }
    semantic_files = {
        path: record["canonical_token_sha256"] for path, record in files.items()
    }
    return {
        "files": files,
        "input_hash": canonical_hash(semantic_files),
        "hash_basis": "UTF-8 whitespace-token stream",
    }


def build_run_manifest():
    code = code_manifest()
    inputs = input_manifest()
    snapshot = Path(REFERENCE_SNAPSHOT)
    if not snapshot.is_file():
        raise RuntimeError(
            f"frozen reference snapshot is required before formal execution: {snapshot}"
        )
    design = formal_design()
    return {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "design": design,
        "design_hash": canonical_hash(design),
        "code_hash": code["code_hash"],
        "code_files": code["files"],
        "input_hash": inputs["input_hash"],
        "input_files": inputs["files"],
        "reference_snapshot": REFERENCE_SNAPSHOT,
        "reference_snapshot_sha256": file_sha256(snapshot),
    }


def write_or_validate_run_manifest(out_dir):
    path = Path(out_dir) / "run_manifest.json"
    current = build_run_manifest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = (
            "protocol", "hostname", "platform", "python", "numpy", "torch",
            "design_hash", "code_hash", "reference_snapshot_sha256",
            "input_hash",
        )
        mismatches = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in stable_keys if existing.get(key) != current.get(key)
        }
        if mismatches:
            raise RuntimeError(f"refusing incompatible resume: {mismatches}")
        return existing
    if Path(out_dir, "runs.csv").exists():
        raise RuntimeError("runs.csv exists without an immutable run_manifest.json")
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def verify_complete(result_file, *, code_hash, design_hash, input_hash):
    with open(result_file, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = int(formal_design()["expected_rows"])
    keys = [
        (
            row["dataset"],
            row["instance"],
            row["variant"],
            int(row["Budget"]),
            int(row["seed"]),
        )
        for row in rows
    ]
    if len(rows) != expected or len(set(keys)) != expected:
        raise RuntimeError(
            f"incomplete grid rows={len(rows)} unique={len(set(keys))} expected={expected}"
        )
    protocols = {row["Protocol"] for row in rows}
    if protocols != {PROTOCOL}:
        raise RuntimeError(f"protocol mismatch: {protocols}")
    if {row.get("Code_hash") for row in rows} != {code_hash}:
        raise RuntimeError("code hash mismatch in completed rows")
    if {row.get("Design_hash") for row in rows} != {design_hash}:
        raise RuntimeError("design hash mismatch in completed rows")
    if {row.get("Input_hash") for row in rows} != {input_hash}:
        raise RuntimeError("benchmark input hash mismatch in completed rows")
    front_manifest_records = []
    for row in rows:
        path = Path(row["front_pickle"])
        if not path.is_file():
            raise RuntimeError(f"missing front: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        observed_front_hash = file_sha256(path)
        if observed_front_hash != row.get("Front_sha256"):
            raise RuntimeError(
                f"front hash mismatch in {path}: {observed_front_hash} != "
                f"{row.get('Front_sha256')}"
            )
        front_manifest_records.append(f"{path}:{observed_front_hash}")
        checks = {
            "protocol": (payload.get("protocol"), PROTOCOL),
            "dataset": (payload.get("dataset"), row["dataset"]),
            "instance": (payload.get("instance"), row["instance"]),
            "variant": (payload.get("variant"), row["variant"]),
            "budget": (int(payload.get("budget", -1)), int(row["Budget"])),
            "seed": (int(payload.get("seed", -1)), int(row["seed"])),
            "config_hash": (payload.get("config_hash"), row["Config_hash"]),
            "code_hash": (payload.get("code_hash"), code_hash),
            "design_hash": (payload.get("design_hash"), design_hash),
            "input_hash": (payload.get("input_hash"), input_hash),
            "population_size": (
                int(payload.get("population_size", -1)),
                FORMAL_POP_SIZE,
            ),
        }
        mismatches = {
            key: {"front": actual, "expected": expected_value}
            for key, (actual, expected_value) in checks.items()
            if actual != expected_value
        }
        objectives = np.asarray(payload.get("objectives", []), dtype=float)
        if mismatches:
            raise RuntimeError(f"front metadata mismatch in {path}: {mismatches}")
        if (
            objectives.ndim != 2
            or objectives.shape[1] < 3
            or len(objectives) == 0
            or not np.isfinite(objectives[:, :3]).all()
        ):
            raise RuntimeError(f"invalid front objectives in {path}: {objectives.shape}")
    return {
        "rows": len(rows), "unique_keys": len(set(keys)), "protocol": PROTOCOL,
        "code_hash": code_hash, "design_hash": design_hash,
        "input_hash": input_hash,
        "front_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(front_manifest_records)).encode("utf-8")
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--budgets", default="100,200")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.result_file is None:
        args.result_file = str(Path(args.out_dir) / "runs.csv")

    if not args.smoke:
        frozen = {
            "out_dir": args.out_dir == DEFAULT_OUT,
            "result_file": Path(args.result_file) == Path(DEFAULT_OUT) / "runs.csv",
            "budgets": args.budgets == "100,200",
            "variants": args.variants is None,
            "pop_size": args.pop_size == FORMAL_POP_SIZE,
            "seed_start": args.seed_start == FORMAL_SEED_START,
            "seed_end": args.seed_end == FORMAL_SEED_END,
            "workers": args.workers == FORMAL_WORKERS,
        }
        if not all(frozen.values()):
            raise SystemExit(f"formal protocol arguments are frozen: {frozen}")
        manifest = write_or_validate_run_manifest(args.out_dir)
        args.code_hash = manifest["code_hash"]
        args.design_hash = manifest["design_hash"]
        args.input_hash = manifest["input_hash"]
    else:
        if Path(args.out_dir) == Path(DEFAULT_OUT):
            raise SystemExit("smoke runs must use a directory other than the formal output")
        code = code_manifest()
        inputs = input_manifest()
        args.code_hash = code["code_hash"]
        args.input_hash = inputs["input_hash"]
        args.design_hash = canonical_hash(
            {"smoke": True, "pop_size": args.pop_size, "budgets": args.budgets,
             "seed_start": args.seed_start, "seed_end": args.seed_end,
             "variants": args.variants}
        )
    tasks = build_tasks(args)
    print(
        f"[v6] protocol={PROTOCOL} pending={len(tasks)} result={args.result_file}",
        flush=True,
    )
    failures = []
    start = time.time()
    workers = max(1, min(args.workers, len(tasks) or 1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_single, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                row = future.result()
                append_row(args.result_file, row)
                status = (
                    f"{row['dataset']}/{row['instance']} {row['variant']} "
                    f"g={row['Budget']} seed={row['seed']} T={row['Time']:.1f}s"
                )
            except Exception as exc:
                dataset, inst_name, _, config, seed, _, budget, *_ = task
                failure = {
                    "dataset": dataset,
                    "instance": inst_name,
                    "variant": config["variant"],
                    "budget": budget,
                    "seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                status = f"ERROR {dataset}/{inst_name} {config['variant']} g={budget} seed={seed}: {exc!r}"
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            eta = (len(tasks) - completed) / max(rate, 1e-9)
            print(f"[{completed}/{len(tasks)}] {status} ETA={eta/60:.1f}min", flush=True)

    failure_path = Path(args.out_dir) / "failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failures)} tasks failed; see {failure_path}")
    if failure_path.exists():
        failure_path.unlink()

    if args.smoke:
        print("[v6] partial/smoke run finished; completeness gate skipped", flush=True)
        return
    completion = verify_complete(
        args.result_file,
        code_hash=args.code_hash,
        design_hash=args.design_hash,
        input_hash=args.input_hash,
    )
    completion["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    completion["elapsed_seconds_this_invocation"] = time.time() - start
    (Path(args.out_dir) / "pipeline_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(f"[v6] COMPLETE rows={completion['rows']}", flush=True)


if __name__ == "__main__":
    main()
