"""Prespecified cross-instance PPO pretraining experiment (E5/v7).

The frozen design is documented in
``SCI_Paper/CROSS_INSTANCE_PRETRAINING_PROTOCOL_V7.md``. Smoke runs and formal
runs must use disjoint output directories. No v5/v6 result is modified.
"""

from __future__ import annotations

import argparse
import copy
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

# A formal worker is one scientific run and one CPU thread.  Do not inherit a
# host-level BLAS setting that silently multiplies threads per worker.
for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.loader import load_benchmark_set
from experiments.run_mechanism_robustness_v6 import (
    AuditedHybridSelector,
    ENHANCED_STATE_DIM,
    canonical_hash,
    canonical_text_sha256,
    controller_rng_seeds,
    file_sha256,
)
from experiments.run_revision_aos import (
    BaseSelector,
    DATASETS,
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


PROTOCOL = "saos_cross_instance_pretrained_ppo_v7_20260722"
DEFAULT_OUT = "results/resubmission/v7_cross_instance"
REFERENCE_SNAPSHOT = (
    "results/resubmission/v7_cross_instance/frozen_reference_snapshot.pkl"
)
FORMAL_WORKERS = 40
FORMAL_POP_SIZE = 100
FORMAL_BUDGETS = (50, 100, 200)
FORMAL_EVAL_SEEDS = tuple(range(42, 52))
FORMAL_REPLICAS = tuple(range(5))
FORMAL_PRETRAIN_SEEDS = tuple(range(7042, 7047))
FORMAL_PRETRAIN_PASSES = 2
FORMAL_PRETRAIN_BUDGET = 200
FORMAL_ROLLOUT = 16
FORMAL_PRETRAIN_ROWS = 2000
FORMAL_EVAL_ROWS = 6000

VARIANTS = (
    "UCBOnly",
    "ScratchNoBC_R16",
    "XPrePPO_Frozen",
    "XPrePPO_Online_R16",
)

FROZEN_TEST_SPLIT = {
    1: {
        "Brandimarte": ("Mk04.fjs", "Mk10.fjs"),
        "Hurink_edata": (
            "la05.fjs", "la10.fjs", "la15.fjs", "la16.fjs",
            "la24.fjs", "la26.fjs", "la34.fjs", "la37.fjs",
        ),
    },
    2: {
        "Brandimarte": ("Mk02.fjs", "Mk03.fjs"),
        "Hurink_edata": (
            "la03.fjs", "la07.fjs", "la13.fjs", "la19.fjs",
            "la23.fjs", "la27.fjs", "la31.fjs", "la36.fjs",
        ),
    },
    3: {
        "Brandimarte": ("Mk06.fjs", "Mk07.fjs"),
        "Hurink_edata": (
            "la04.fjs", "la09.fjs", "la12.fjs", "la17.fjs",
            "la25.fjs", "la28.fjs", "la35.fjs", "la39.fjs",
        ),
    },
    4: {
        "Brandimarte": ("Mk01.fjs", "Mk08.fjs"),
        "Hurink_edata": (
            "la01.fjs", "la06.fjs", "la11.fjs", "la18.fjs",
            "la22.fjs", "la29.fjs", "la32.fjs", "la40.fjs",
        ),
    },
    5: {
        "Brandimarte": ("Mk05.fjs", "Mk09.fjs"),
        "Hurink_edata": (
            "la02.fjs", "la08.fjs", "la14.fjs", "la20.fjs",
            "la21.fjs", "la30.fjs", "la33.fjs", "la38.fjs",
        ),
    },
}

CODE_FILES = (
    "experiments/run_cross_instance_pretraining_v7.py",
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
    "scripts/build_cross_instance_reference_snapshot_v7.py",
    "SCI_Paper/CROSS_INSTANCE_PRETRAINING_PROTOCOL_V7.md",
)


def stable_seed(*parts) -> int:
    text = "|".join(str(value) for value in (PROTOCOL, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_pickle(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_torch_save(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def tensor_state_semantic_hash(state_dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def front_semantic_hash(objectives) -> str:
    array = np.asarray(objectives, dtype=np.float64)
    order = np.lexsort(tuple(array[:, index] for index in reversed(range(array.shape[1]))))
    canonical = np.ascontiguousarray(array[order])
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def is_deduplicated_nondominated(objectives, tolerance=1e-12) -> bool:
    points = np.asarray(objectives, dtype=np.float64)
    if len(np.unique(points, axis=0)) != len(points):
        return False
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        if len(others) and np.any(
            np.all(others <= point + tolerance, axis=1)
            & np.any(others < point - tolerance, axis=1)
        ):
            return False
    return True


def nondominated_subset(objectives, tolerance=1e-12):
    points = np.unique(np.asarray(objectives, dtype=np.float64), axis=0)
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        if len(others) and np.any(
            np.all(others <= point + tolerance, axis=1)
            & np.any(others < point - tolerance, axis=1)
        ):
            keep[index] = False
    return points[keep]


def load_all_instances():
    loaded = []
    for dataset, path in DATASETS.items():
        for instance_name, instance in load_benchmark_set(path, num_agv=3):
            loaded.append((dataset, instance_name, instance))
    if len(loaded) != 50:
        raise RuntimeError(f"expected 50 benchmark instances, loaded {len(loaded)}")
    return loaded


def instance_input_records():
    records = {}
    for dataset, names in (("Brandimarte", range(1, 11)), ("Hurink_edata", range(1, 41))):
        prefix = "Mk" if dataset == "Brandimarte" else "la"
        for number in names:
            name = f"{prefix}{number:02d}.fjs"
            path = Path(DATASETS[dataset]) / name
            if not path.is_file():
                raise RuntimeError(f"missing benchmark input: {path}")
            records[(dataset, name)] = {
                "path": path.as_posix(),
                "resolved_path": str(path.resolve()),
                "canonical_token_sha256": canonical_text_sha256(path),
                "raw_sha256": file_sha256(path),
            }
    return records


def generated_test_split(instances):
    """Generate the frozen split from size blocks and protocol-salted hashes."""
    by_family = {"Brandimarte": [], "Hurink_edata": []}
    for dataset, name, instance in instances:
        by_family[dataset].append((int(instance.total_operations), name))
    folds = {fold: {family: [] for family in by_family} for fold in range(1, 6)}
    for family, records in by_family.items():
        ordered = sorted(records, key=lambda item: (item[0], item[1]))
        if len(ordered) % 5:
            raise RuntimeError(f"family {family} cannot be divided into five-instance blocks")
        for block_id, start in enumerate(range(0, len(ordered), 5), start=1):
            block = ordered[start:start + 5]
            block = sorted(
                block,
                key=lambda item: hashlib.sha256(
                    f"{PROTOCOL}|{family}|{block_id}|{item[1]}".encode("utf-8")
                ).hexdigest(),
            )
            for fold, (_, name) in enumerate(block, start=1):
                folds[fold][family].append(name)
    return {
        fold: {family: tuple(sorted(names)) for family, names in families.items()}
        for fold, families in folds.items()
    }


def validate_split(instances, input_records):
    generated = generated_test_split(instances)
    frozen = {
        fold: {family: tuple(sorted(names)) for family, names in families.items()}
        for fold, families in FROZEN_TEST_SPLIT.items()
    }
    if generated != frozen:
        raise RuntimeError(f"generated fold split differs from frozen table: {generated}")
    all_keys = {(dataset, name) for dataset, name, _ in instances}
    resolved_paths = [record["resolved_path"] for record in input_records.values()]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise RuntimeError("two benchmark keys resolve to the same filesystem identity")
    observed = []
    content_hashes = []
    for fold in range(1, 6):
        fold_keys = {
            (family, name)
            for family, names in frozen[fold].items()
            for name in names
        }
        if len(fold_keys) != 10:
            raise RuntimeError(f"fold {fold} has {len(fold_keys)} test instances")
        observed.extend(fold_keys)
        fold_hashes = [input_records[key]["canonical_token_sha256"] for key in fold_keys]
        test_paths = {input_records[key]["resolved_path"] for key in fold_keys}
        training_paths = {
            input_records[key]["resolved_path"] for key in (all_keys - fold_keys)
        }
        if test_paths & training_paths:
            raise RuntimeError(f"resolved-path leakage detected in fold {fold}")
        if len(set(fold_hashes)) != len(fold_hashes):
            raise RuntimeError(f"duplicate benchmark content inside test fold {fold}")
        content_hashes.extend(fold_hashes)
    if set(observed) != all_keys or len(observed) != len(all_keys):
        raise RuntimeError("fold tests are not an exhaustive one-time partition")
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError("two benchmark files share the same canonical content hash")
    return frozen


def fold_for_instance(dataset, instance_name) -> int:
    matches = [
        fold for fold, families in FROZEN_TEST_SPLIT.items()
        if instance_name in families[dataset]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"instance has {len(matches)} test-fold assignments: {dataset}/{instance_name}")
    return int(matches[0])


def training_keys_for_fold(fold, all_keys):
    test = {
        (family, name)
        for family, names in FROZEN_TEST_SPLIT[int(fold)].items()
        for name in names
    }
    training = sorted(set(all_keys) - test)
    if len(training) != 40 or len(test) != 10 or set(training) & test:
        raise RuntimeError(f"invalid fold {fold} training/test membership")
    return training


def build_training_orders(fold, replica, training_keys, passes=FORMAL_PRETRAIN_PASSES):
    orders = []
    for pass_index in range(1, int(passes) + 1):
        rng = np.random.RandomState(stable_seed("training-order", fold, replica, pass_index))
        permutation = rng.permutation(len(training_keys))
        orders.append([training_keys[int(index)] for index in permutation])
    return orders


def formal_design(split):
    return {
        "protocol": PROTOCOL,
        "population_size": FORMAL_POP_SIZE,
        "test_budgets": list(FORMAL_BUDGETS),
        "test_seeds": list(FORMAL_EVAL_SEEDS),
        "variants": list(VARIANTS),
        "pretrain_seeds": list(FORMAL_PRETRAIN_SEEDS),
        "replicas": list(FORMAL_REPLICAS),
        "pretrain_passes": FORMAL_PRETRAIN_PASSES,
        "pretrain_budget": FORMAL_PRETRAIN_BUDGET,
        "rollout": FORMAL_ROLLOUT,
        "state_dim": ENHANCED_STATE_DIM,
        "test_split": split,
        "expected_pretraining_rows": FORMAL_PRETRAIN_ROWS,
        "expected_evaluation_rows": FORMAL_EVAL_ROWS,
        "workers": FORMAL_WORKERS,
        "transfer": "actor_and_critic_parameters_fresh_optimizer_empty_buffer",
        "reference_snapshot": REFERENCE_SNAPSHOT,
    }


class CrossInstanceSelector(AuditedHybridSelector):
    """Enhanced-state no-BC selector with explicit scratch/transfer modes."""

    def __init__(self, rng, *, training_state=None, load_optimizer=False, frozen=False,
                 rng_seeds=None):
        self.frozen_after_handover = bool(frozen)
        super().__init__(
            rng,
            state_mode="enhanced",
            reward_scheme="composite",
            transition_mode="adaptive",
            min_per_op=3,
            min_buffer=30,
            use_behavior_cloning=False,
            rollout_size=FORMAL_ROLLOUT,
            rng_seeds=rng_seeds,
            device="cpu",
        )
        if training_state is not None:
            self.ppo.load_training_state(training_state, load_optimizer=load_optimizer)
        if len(self.ppo.buffer):
            raise RuntimeError("loaded selector has a nonempty PPO buffer")

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        if not self.frozen_after_handover:
            return super().select(gen, max_gen, stagnation, diversity, hv_trend)
        if not self.switched and self._should_switch(gen, max_gen, stagnation):
            self.switched = True
            self.transition_gen = gen
        if self.switched:
            state = self.build_state(gen, max_gen, stagnation, diversity, hv_trend, True)
            action = self.ppo.select(state)
            self.last_state = state
            self.last_action = int(action)
            self.last_gen = gen
            self.last_max_gen = max_gen
            self.last_phase = "PPO"
            return int(action)
        state = self.build_state(gen, max_gen, stagnation, diversity, hv_trend, False)
        action = self.ucb.select(gen, max_gen, stagnation, diversity, hv_trend)
        self.demo_states.append(state.copy())
        self.demo_actions.append(int(action))
        self.last_state = state
        self.last_action = int(action)
        self.last_gen = gen
        self.last_max_gen = max_gen
        self.last_phase = "UCB"
        return int(action)

    def update(self, op_id, reward):
        if not self.frozen_after_handover:
            return super().update(op_id, reward)
        BaseSelector.update(self, op_id, reward)
        self.ucb.update(op_id, reward)

    def finalize(self):
        if not self.frozen_after_handover:
            return super().finalize()
        if len(self.ppo.buffer):
            raise RuntimeError("frozen selector unexpectedly collected PPO rollouts")


def make_cross_instance_selector(
    variant, controller_rng, rng_seeds, *, checkpoint_payload=None,
):
    if variant == "UCBOnly":
        return UCBSelector(controller_rng, reward_scheme="composite")
    if checkpoint_payload is None:
        raise ValueError(f"{variant} requires a checkpoint payload")
    if variant == "ScratchNoBC_R16":
        state = checkpoint_payload["initial_training_state"]
        return CrossInstanceSelector(
            controller_rng, training_state=state, load_optimizer=False,
            frozen=False, rng_seeds=rng_seeds,
        )
    if variant == "XPrePPO_Frozen":
        return CrossInstanceSelector(
            controller_rng, training_state=checkpoint_payload["training_state"],
            load_optimizer=False, frozen=True, rng_seeds=rng_seeds,
        )
    if variant == "XPrePPO_Online_R16":
        return CrossInstanceSelector(
            controller_rng, training_state=checkpoint_payload["training_state"],
            load_optimizer=False, frozen=False, rng_seeds=rng_seeds,
        )
    raise ValueError(f"unsupported v7 variant: {variant}")


def _mean_stat(records, key):
    values = [float(record[key]) for record in records if key in record]
    return float(np.mean(values)) if values else 0.0


def execute_episode(instance, selector, *, init_rng, evolution_rng, pop_size, max_gen):
    """Run one fixed-backbone episode and return audited scientific outputs."""
    start = time.time()
    cpu_start = time.process_time()
    ref_points = generate_reference_points(3, 8)
    population = initialize_population(instance, int(pop_size), init_rng)
    archive = update_archive([], population)
    pop_objs = np.asarray([chromosome.objectives.to_array() for chromosome in population])
    online_reference = pop_objs.max(axis=0) * 1.1
    previous_hv = compute_hypervolume(pop_objs, online_reference)
    previous_best_cmax = float(pop_objs[:, 0].min())
    stagnation = 0
    hv_window = deque(maxlen=20)
    operator_sequence = []
    reward_sequence = []
    hv_delta_sequence = []

    for generation in range(int(max_gen)):
        pop_objs = np.asarray([chromosome.objectives.to_array() for chromosome in population])
        diversity = population_diversity(pop_objs)
        hv_trend = float(np.mean(hv_window)) if hv_window else 0.0
        operator_id = selector.select(
            generation, max_gen, stagnation, diversity, hv_trend
        )
        offspring = apply_operator(population, operator_id, pop_size, 5, evolution_rng)
        if len(offspring) != int(pop_size):
            raise RuntimeError(
                f"generation {generation} produced {len(offspring)} offspring, expected {pop_size}"
            )
        for chromosome in offspring:
            chromosome._revision_uid = id(chromosome)
        combined = population + offspring
        combined_objs = np.asarray([chromosome.objectives.to_array() for chromosome in combined])
        archive = update_generation_archive(archive, population, offspring)
        selected = nsga3_select(combined_objs, pop_size, ref_points, rng=evolution_rng)
        new_population = [combined[index] for index in selected]
        selected_ids = {
            getattr(chromosome, "_revision_uid", None) for chromosome in new_population
        }
        survived = sum(
            getattr(chromosome, "_revision_uid", None) in selected_ids
            for chromosome in offspring
        )
        survival = survived / max(len(offspring), 1)
        new_objs = np.asarray([chromosome.objectives.to_array() for chromosome in new_population])
        current_hv = compute_hypervolume(new_objs, online_reference)
        hv_delta = float(np.clip(
            (current_hv - previous_hv) / max(abs(previous_hv), 1e-10), -1.0, 1.0
        ))
        current_best_cmax = float(new_objs[:, 0].min())
        cmax_delta = float(np.clip(
            (previous_best_cmax - current_best_cmax)
            / max(abs(previous_best_cmax), 1e-10), -1.0, 1.0
        ))
        reward = compute_reward(
            "composite", survival, hv_delta, cmax_delta, generation, max_gen
        )
        selector.update(operator_id, reward)
        hv_window.append(hv_delta)
        operator_sequence.append(int(operator_id))
        reward_sequence.append(float(reward))
        hv_delta_sequence.append(float(hv_delta))
        stagnation = stagnation + 1 if current_hv <= previous_hv * 1.001 else 0
        previous_hv = current_hv
        previous_best_cmax = current_best_cmax
        population = new_population

    if hasattr(selector, "finalize"):
        selector.finalize()
    if hasattr(selector, "ppo") and len(selector.ppo.buffer):
        raise RuntimeError("episode ended with a nonempty PPO rollout buffer")
    metrics = evaluate_archive(archive)
    objectives = np.asarray(metrics.pop("objectives"), dtype=np.float64)
    if (
        objectives.ndim != 2 or objectives.shape[1] < 3 or len(objectives) == 0
        or not np.isfinite(objectives[:, :3]).all()
    ):
        raise RuntimeError(f"invalid final front: shape={objectives.shape}")
    objectives = nondominated_subset(objectives[:, :3])
    if not is_deduplicated_nondominated(objectives):
        raise RuntimeError("final front is not a deduplicated nondominated set")
    ppo_stats = list(getattr(selector, "ppo_update_stats", []))
    for update in ppo_stats:
        if int(update.get("updated_policy_version", -1)) != int(
            update.get("behavior_policy_version", -2)
        ) + 1:
            raise RuntimeError(f"invalid PPO policy-version audit: {update}")
    counts = np.bincount(np.asarray(operator_sequence, dtype=int), minlength=N_OPS)
    consumed_samples = int(
        sum(int(item.get("rollout_size", 0)) for item in ppo_stats)
    )
    discarded_singletons = int(getattr(selector, "ppo_discarded_singletons", 0))
    collected_transitions = consumed_samples + discarded_singletons
    transition_generation = int(getattr(selector, "transition_gen", -1))
    ppo_controlled_actions = (
        int(max_gen) - transition_generation if transition_generation >= 0 else 0
    )
    result = {
        **metrics,
        "objectives": objectives,
        "front_semantic_sha256": front_semantic_hash(objectives),
        "operator_sequence": operator_sequence,
        "reward_sequence": reward_sequence,
        "hv_delta_sequence": hv_delta_sequence,
        "operator_counts": counts.tolist(),
        "operator_entropy": normalized_entropy(counts),
        "transition_generation": transition_generation,
        "transition_reason": str(getattr(selector, "transition_reason", "none")),
        "ppo_update_stats": ppo_stats,
        "ppo_samples": consumed_samples,
        "ppo_consumed_samples": consumed_samples,
        "ppo_collected_transitions": collected_transitions,
        "ppo_discarded_singletons": discarded_singletons,
        "ppo_controlled_actions": ppo_controlled_actions,
        "ppo_updates": int(len(ppo_stats)),
        "ppo_optimizer_steps": int(sum(int(item.get("optimizer_steps", 0)) for item in ppo_stats)),
        "ppo_policy_loss_mean": _mean_stat(ppo_stats, "policy_loss"),
        "ppo_value_loss_mean": _mean_stat(ppo_stats, "value_loss"),
        "ppo_entropy_mean": _mean_stat(ppo_stats, "entropy"),
        "ppo_approx_kl_mean": _mean_stat(ppo_stats, "approx_kl"),
        "learning_time": float(getattr(selector, "learning_time_seconds", 0.0)),
        "elapsed_seconds": float(time.time() - start),
        "cpu_seconds": float(time.process_time() - cpu_start),
        "initial_evaluations": int(pop_size),
        "offspring_evaluations": int(pop_size) * int(max_gen),
    }
    return result


def checkpoint_path(out_dir, fold, replica, label="terminal"):
    return Path(out_dir) / "checkpoints" / f"fold{int(fold)}_rep{int(replica)}_{label}.pt"


def checkpoint_marker_path(path):
    return Path(str(path) + ".complete.json")


def write_checkpoint(path, payload):
    path = Path(path)
    if len(payload["training_state"].get("policy", {})) == 0:
        raise RuntimeError("refusing to write checkpoint with empty policy state")
    object_dir = path.parent / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    temporary = object_dir / f".tmp-{os.getpid()}-{time.time_ns()}.pt"
    torch.save(payload, temporary)
    checkpoint_hash = file_sha256(temporary)
    object_path = object_dir / f"{checkpoint_hash}.pt"
    if object_path.is_file():
        if file_sha256(object_path) != checkpoint_hash:
            raise RuntimeError(f"content-addressed checkpoint collision: {object_path}")
        temporary.unlink()
    else:
        os.replace(temporary, object_path)
    marker = {
        "protocol": PROTOCOL,
        "checkpoint": path.as_posix(),
        "object_path": f"objects/{object_path.name}",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": int(object_path.stat().st_size),
        "weight_semantic_sha256": tensor_state_semantic_hash(
            payload["training_state"]["policy"]
        ),
        "initial_weight_semantic_sha256": tensor_state_semantic_hash(
            payload["initial_training_state"]["policy"]
        ),
        "metadata_sha256": canonical_hash(payload["metadata"]),
        "fold": int(payload["metadata"]["fold"]),
        "replica": int(payload["metadata"]["replica"]),
        "completed_episodes": int(payload["metadata"]["completed_episodes"]),
        "policy_version": int(payload["training_state"].get("policy_version", -1)),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(checkpoint_marker_path(path), marker)
    return marker


def load_checkpoint(path, *, expected=None):
    path = Path(path)
    marker_path = checkpoint_marker_path(path)
    if not marker_path.is_file():
        raise RuntimeError(f"checkpoint completion pointer missing: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    object_path = marker_path.parent / marker.get("object_path", "")
    if not object_path.is_file():
        raise RuntimeError(f"checkpoint object missing: {object_path}")
    observed_file_hash = file_sha256(object_path)
    if observed_file_hash != marker.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint file hash mismatch: {object_path}")
    if int(object_path.stat().st_size) != int(marker.get("checkpoint_size_bytes", -1)):
        raise RuntimeError(f"checkpoint size mismatch: {object_path}")
    payload = torch.load(object_path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"checkpoint protocol mismatch: {path}")
    metadata = dict(payload.get("metadata", {}))
    if canonical_hash(metadata) != marker.get("metadata_sha256"):
        raise RuntimeError(f"checkpoint metadata hash mismatch: {path}")
    semantic = tensor_state_semantic_hash(payload["training_state"]["policy"])
    if semantic != marker.get("weight_semantic_sha256"):
        raise RuntimeError(f"checkpoint semantic weight hash mismatch: {path}")
    initial_semantic = tensor_state_semantic_hash(
        payload["initial_training_state"]["policy"]
    )
    if initial_semantic != marker.get("initial_weight_semantic_sha256"):
        raise RuntimeError(f"checkpoint initial-weight hash mismatch: {path}")
    if expected:
        mismatches = {
            key: {"observed": metadata.get(key), "expected": value}
            for key, value in expected.items() if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"checkpoint metadata mismatch in {path}: {mismatches}")
    architecture = payload["training_state"].get("architecture", {})
    expected_architecture = {
        "state_dim": ENHANCED_STATE_DIM, "action_dim": N_OPS, "hidden": 128,
    }
    if architecture != expected_architecture:
        raise RuntimeError(
            f"checkpoint architecture mismatch: {architecture} != "
            f"{expected_architecture}"
        )
    expected_hyperparameters = {
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "n_epochs": 4, "batch_size": FORMAL_ROLLOUT,
        "entropy_coef": 0.01, "value_coef": 0.5,
    }
    observed_hyperparameters = payload["training_state"].get(
        "hyperparameters", {}
    )
    if observed_hyperparameters != expected_hyperparameters:
        raise RuntimeError(
            f"checkpoint hyperparameter mismatch: {observed_hyperparameters} != "
            f"{expected_hyperparameters}"
        )
    return payload, marker


def _checkpoint_metadata(
    *, fold, replica, pretrain_seed, training_orders, input_records,
    code_hash, design_hash, input_hash, split_hash, completed_episodes,
    cumulative, label, initial_rng_seeds, episode_rng_records,
):
    training_keys = sorted({key for order in training_orders for key in order})
    return {
        "protocol": PROTOCOL,
        "fold": int(fold),
        "replica": int(replica),
        "pretrain_seed": int(pretrain_seed),
        "label": str(label),
        "completed_episodes": int(completed_episodes),
        "training_orders": [
            [[dataset, name] for dataset, name in order] for order in training_orders
        ],
        "training_instances": [
            {
                "dataset": dataset,
                "instance": name,
                "canonical_token_sha256": input_records[(dataset, name)][
                    "canonical_token_sha256"
                ],
            }
            for dataset, name in training_keys
        ],
        "training_instance_set_sha256": canonical_hash(training_keys),
        "seed_schema": "sha256(protocol|phase|fold|replica|pass|dataset|instance)",
        "initial_rng_seeds": {
            key: int(value) for key, value in initial_rng_seeds.items()
        },
        "episode_rng_manifest_sha256": canonical_hash(episode_rng_records),
        "code_hash": code_hash,
        "design_hash": design_hash,
        "input_hash": input_hash,
        "split_hash": split_hash,
        "configuration": {
            "population_size": FORMAL_POP_SIZE,
            "generations_per_episode": FORMAL_PRETRAIN_BUDGET,
            "rollout": FORMAL_ROLLOUT,
            "state_dim": ENHANCED_STATE_DIM,
            "behavior_cloning": False,
            "passes": FORMAL_PRETRAIN_PASSES,
        },
        "cumulative": dict(cumulative),
    }


def _pretraining_episode_row(
    *, fold, replica, pretrain_seed, pass_index, position, dataset,
    instance_name, episode_seed, episode_rng_seeds, episode_budget, result,
    cumulative, hashes,
):
    return {
        "Protocol": PROTOCOL,
        "Fold": int(fold),
        "Replica": int(replica),
        "Pretrain_seed": int(pretrain_seed),
        "Pass": int(pass_index),
        "Position": int(position),
        "dataset": dataset,
        "instance": instance_name,
        "Episode_seed": int(episode_seed),
        "Network_seed": int(episode_rng_seeds["network"]),
        "Action_seed": int(episode_rng_seeds["action"]),
        "BC_seed": int(episode_rng_seeds["bc"]),
        "PPO_seed": int(episode_rng_seeds["ppo"]),
        "Population_size": FORMAL_POP_SIZE,
        "Budget": int(episode_budget),
        "Initial_evaluations": int(result["initial_evaluations"]),
        "Offspring_evaluations": int(result["offspring_evaluations"]),
        "PPO_samples": int(result["ppo_samples"]),
        "PPO_collected_transitions": int(result["ppo_collected_transitions"]),
        "PPO_discarded_singletons": int(result["ppo_discarded_singletons"]),
        "PPO_updates": int(result["ppo_updates"]),
        "PPO_optimizer_steps": int(result["ppo_optimizer_steps"]),
        "Transition_gen": int(result["transition_generation"]),
        "Operator_entropy": float(result["operator_entropy"]),
        "Front_semantic_sha256": result["front_semantic_sha256"],
        "Time": float(result["elapsed_seconds"]),
        "CPU_time": float(result["cpu_seconds"]),
        "Learning_time": float(result["learning_time"]),
        "Cumulative_objective_evaluations": int(cumulative["objective_evaluations"]),
        "Cumulative_PPO_samples": int(cumulative["ppo_samples"]),
        "Cumulative_PPO_collected_transitions": int(
            cumulative["ppo_collected_transitions"]
        ),
        "Cumulative_PPO_discarded_singletons": int(
            cumulative["ppo_discarded_singletons"]
        ),
        "Cumulative_PPO_updates": int(cumulative["ppo_updates"]),
        "Cumulative_PPO_optimizer_steps": int(cumulative["ppo_optimizer_steps"]),
        "Cumulative_CPU_seconds": float(cumulative["cpu_seconds"]),
        "Cumulative_wall_seconds": float(cumulative["wall_seconds"]),
        "Code_hash": hashes["code_hash"],
        "Design_hash": hashes["design_hash"],
        "Input_hash": hashes["input_hash"],
        "Split_hash": hashes["split_hash"],
        "Reference_snapshot_sha256": hashes["reference_snapshot_sha256"],
    }


def run_pretraining_chain(task):
    (
        fold, replica, out_dir, code_hash, design_hash, input_hash, split_hash,
        pretrain_passes, pretrain_budget, training_limit,
        reference_snapshot_sha256,
    ) = task
    torch.set_num_threads(1)
    random.seed(stable_seed("python", "pretrain", fold, replica))
    np.random.seed(stable_seed("numpy", "pretrain", fold, replica))
    torch.manual_seed(stable_seed("torch", "pretrain", fold, replica))
    instances = load_all_instances()
    lookup = {(dataset, name): instance for dataset, name, instance in instances}
    input_records = instance_input_records()
    validate_split(instances, input_records)
    training_keys = training_keys_for_fold(fold, lookup)
    if training_limit:
        training_keys = training_keys[: int(training_limit)]
    orders = build_training_orders(fold, replica, training_keys, passes=pretrain_passes)
    pretrain_seed = FORMAL_PRETRAIN_SEEDS[int(replica)]
    terminal_path = checkpoint_path(out_dir, fold, replica, "terminal")
    if terminal_path.is_file() or checkpoint_marker_path(terminal_path).is_file():
        terminal, marker = load_checkpoint(
            terminal_path,
            expected={
                "fold": int(fold), "replica": int(replica),
                "code_hash": code_hash, "design_hash": design_hash,
                "input_hash": input_hash, "split_hash": split_hash,
                "completed_episodes": len(orders) * len(training_keys),
                "label": "terminal",
            },
        )
        expected_orders = [
            [[dataset, name] for dataset, name in order] for order in orders
        ]
        if terminal["metadata"].get("training_orders") != expected_orders:
            raise RuntimeError("terminal checkpoint training order mismatch")
        records = list(terminal.get("episode_records", []))
        if len(records) != len(orders) * len(training_keys):
            raise RuntimeError("terminal checkpoint episode-record count mismatch")
        rng_records = list(terminal.get("episode_rng_records", []))
        if len(rng_records) != len(records):
            raise RuntimeError("terminal checkpoint RNG-record count mismatch")
        if canonical_hash(rng_records) != terminal["metadata"].get(
            "episode_rng_manifest_sha256"
        ):
            raise RuntimeError("terminal checkpoint RNG manifest mismatch")
        return {
            "fold": int(fold), "replica": int(replica),
            "episode_records": records, "checkpoint_marker": marker,
        }
    initial_rng_seeds = {
        "network": stable_seed("network-init", fold, replica, pretrain_seed),
        "action": stable_seed("action-init", fold, replica, pretrain_seed),
        "bc": stable_seed("bc-init", fold, replica, pretrain_seed),
        "ppo": stable_seed("ppo-init", fold, replica, pretrain_seed),
    }
    initial_selector = CrossInstanceSelector(
        np.random.RandomState(stable_seed("controller-init", fold, replica)),
        rng_seeds=initial_rng_seeds,
    )
    initial_training_state = initial_selector.ppo.export_training_state()
    training_state = copy.deepcopy(initial_training_state)
    records = []
    episode_rng_records = []
    cumulative = {
        "objective_evaluations": 0,
        "ppo_samples": 0,
        "ppo_collected_transitions": 0,
        "ppo_discarded_singletons": 0,
        "ppo_updates": 0,
        "ppo_optimizer_steps": 0,
        "cpu_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    flattened = [
        (pass_index, position, key)
        for pass_index, order in enumerate(orders, start=1)
        for position, key in enumerate(order, start=1)
    ]
    progress_path = checkpoint_path(out_dir, fold, replica, "progress")
    if progress_path.is_file() or checkpoint_marker_path(progress_path).is_file():
        progress, _ = load_checkpoint(
            progress_path,
            expected={
                "fold": int(fold), "replica": int(replica),
                "code_hash": code_hash, "design_hash": design_hash,
                "input_hash": input_hash, "split_hash": split_hash,
            },
        )
        if progress["metadata"]["training_orders"] != [
            [[dataset, name] for dataset, name in order] for order in orders
        ]:
            raise RuntimeError("progress checkpoint training order mismatch")
        training_state = progress["training_state"]
        initial_training_state = progress["initial_training_state"]
        records = list(progress.get("episode_records", []))
        episode_rng_records = list(progress.get("episode_rng_records", []))
        cumulative = dict(progress["metadata"]["cumulative"])
        start_index = int(progress["metadata"]["completed_episodes"])
        if len(records) != start_index or len(episode_rng_records) != start_index:
            raise RuntimeError("progress checkpoint record count mismatch")
    else:
        start_index = 0

    hashes = {
        "code_hash": code_hash, "design_hash": design_hash,
        "input_hash": input_hash, "split_hash": split_hash,
        "reference_snapshot_sha256": reference_snapshot_sha256,
    }
    for episode_index in range(start_index, len(flattened)):
        pass_index, position, (dataset, instance_name) = flattened[episode_index]
        episode_seed = stable_seed(
            "pretrain-episode", fold, replica, pass_index, dataset, instance_name
        )
        init_rng, evolution_rng, controller_rng = make_rng_streams(episode_seed)
        episode_rng_seeds = {
            "network": initial_rng_seeds["network"],
            "action": stable_seed("action", fold, replica, pass_index, dataset, instance_name),
            "bc": stable_seed("bc", fold, replica, pass_index, dataset, instance_name),
            "ppo": stable_seed("ppo", fold, replica, pass_index, dataset, instance_name),
        }
        episode_rng_records.append({
            "pass": int(pass_index), "position": int(position),
            "dataset": dataset, "instance": instance_name,
            "episode_seed": int(episode_seed),
            "rng_seeds": {key: int(value) for key, value in episode_rng_seeds.items()},
        })
        selector = CrossInstanceSelector(
            controller_rng,
            training_state=training_state,
            load_optimizer=True,
            frozen=False,
            rng_seeds=episode_rng_seeds,
        )
        result = execute_episode(
            lookup[(dataset, instance_name)], selector,
            init_rng=init_rng, evolution_rng=evolution_rng,
            pop_size=FORMAL_POP_SIZE, max_gen=pretrain_budget,
        )
        training_state = selector.ppo.export_training_state()
        cumulative["objective_evaluations"] += (
            int(result["initial_evaluations"]) + int(result["offspring_evaluations"])
        )
        cumulative["ppo_samples"] += int(result["ppo_samples"])
        cumulative["ppo_collected_transitions"] += int(
            result["ppo_collected_transitions"]
        )
        cumulative["ppo_discarded_singletons"] += int(
            result["ppo_discarded_singletons"]
        )
        cumulative["ppo_updates"] += int(result["ppo_updates"])
        cumulative["ppo_optimizer_steps"] += int(result["ppo_optimizer_steps"])
        cumulative["cpu_seconds"] += float(result["cpu_seconds"])
        cumulative["wall_seconds"] += float(result["elapsed_seconds"])
        records.append(_pretraining_episode_row(
            fold=fold, replica=replica, pretrain_seed=pretrain_seed,
            pass_index=pass_index, position=position, dataset=dataset,
            instance_name=instance_name, episode_seed=episode_seed,
            episode_rng_seeds=episode_rng_seeds,
            episode_budget=pretrain_budget,
            result=result, cumulative=cumulative, hashes=hashes,
        ))
        completed = episode_index + 1
        label = "pass1" if completed == len(orders[0]) else "progress"
        metadata = _checkpoint_metadata(
            fold=fold, replica=replica, pretrain_seed=pretrain_seed,
            training_orders=orders, input_records=input_records,
            code_hash=code_hash, design_hash=design_hash, input_hash=input_hash,
            split_hash=split_hash, completed_episodes=completed,
            cumulative=cumulative, label=label,
            initial_rng_seeds=initial_rng_seeds,
            episode_rng_records=episode_rng_records,
        )
        payload = {
            "protocol": PROTOCOL,
            "metadata": metadata,
            "initial_training_state": initial_training_state,
            "training_state": training_state,
            "episode_records": records,
            "episode_rng_records": episode_rng_records,
        }
        write_checkpoint(progress_path, payload)
        if completed == len(orders[0]):
            write_checkpoint(checkpoint_path(out_dir, fold, replica, "pass1"), payload)

    terminal_metadata = _checkpoint_metadata(
        fold=fold, replica=replica, pretrain_seed=pretrain_seed,
        training_orders=orders, input_records=input_records,
        code_hash=code_hash, design_hash=design_hash, input_hash=input_hash,
        split_hash=split_hash, completed_episodes=len(flattened),
        cumulative=cumulative, label="terminal",
        initial_rng_seeds=initial_rng_seeds,
        episode_rng_records=episode_rng_records,
    )
    terminal_payload = {
        "protocol": PROTOCOL,
        "metadata": terminal_metadata,
        "initial_training_state": initial_training_state,
        "training_state": training_state,
        "episode_records": records,
        "episode_rng_records": episode_rng_records,
    }
    marker = write_checkpoint(terminal_path, terminal_payload)
    return {
        "fold": int(fold), "replica": int(replica),
        "episode_records": records, "checkpoint_marker": marker,
    }


_CHECKPOINT_CACHE = {}


def cached_checkpoint(path, expected):
    key = (str(path), tuple(sorted(expected.items())))
    if key not in _CHECKPOINT_CACHE:
        _CHECKPOINT_CACHE[key] = load_checkpoint(path, expected=expected)
    return _CHECKPOINT_CACHE[key]


def _parameter_l2(state_a, state_b):
    total = 0.0
    for name in state_a:
        difference = (
            state_a[name].detach().cpu().double()
            - state_b[name].detach().cpu().double()
        )
        total += float(torch.sum(difference * difference).item())
    return float(math.sqrt(total))


def run_evaluation_task(task):
    (
        dataset, instance_name, instance, fold, variant, budget, seed, replica,
        out_dir, code_hash, design_hash, input_hash, split_hash, worker_count,
        expected_pretraining_episodes, reference_snapshot_sha256,
    ) = task
    torch.set_num_threads(1)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    init_rng, evolution_rng, controller_rng = make_rng_streams(seed)
    rng_seeds = controller_rng_seeds(seed)
    checkpoint_payload = None
    checkpoint_marker = None
    checkpoint_role = "none"
    if variant != "UCBOnly":
        path = checkpoint_path(out_dir, fold, replica, "terminal")
        checkpoint_payload, checkpoint_marker = cached_checkpoint(
            path,
            {
                "fold": int(fold), "replica": int(replica),
                "code_hash": code_hash, "design_hash": design_hash,
                "input_hash": input_hash, "split_hash": split_hash,
                "completed_episodes": int(expected_pretraining_episodes),
            },
        )
        training_records = checkpoint_payload["metadata"]["training_instances"]
        training_keys = {
            (record["dataset"], record["instance"]) for record in training_records
        }
        training_hashes = {
            record["canonical_token_sha256"] for record in training_records
        }
        current_hash = canonical_text_sha256(Path(DATASETS[dataset]) / instance_name)
        if (dataset, instance_name) in training_keys or current_hash in training_hashes:
            raise RuntimeError(
                f"held-out leakage detected for fold {fold}: {dataset}/{instance_name}"
            )
        checkpoint_role = (
            "step0_initialization" if variant == "ScratchNoBC_R16"
            else "pretrained_parameters"
        )

    selector = make_cross_instance_selector(
        variant, controller_rng, rng_seeds, checkpoint_payload=checkpoint_payload
    )
    if hasattr(selector, "ppo"):
        initial_policy = {
            name: tensor.detach().cpu().clone()
            for name, tensor in selector.ppo.policy.state_dict().items()
        }
        initial_policy_hash = tensor_state_semantic_hash(initial_policy)
    else:
        initial_policy = None
        initial_policy_hash = "none"
    result = execute_episode(
        instance, selector, init_rng=init_rng, evolution_rng=evolution_rng,
        pop_size=FORMAL_POP_SIZE, max_gen=budget,
    )
    if hasattr(selector, "ppo"):
        final_policy = {
            name: tensor.detach().cpu().clone()
            for name, tensor in selector.ppo.policy.state_dict().items()
        }
        final_policy_hash = tensor_state_semantic_hash(final_policy)
        parameter_drift = _parameter_l2(initial_policy, final_policy)
        final_policy_version = int(selector.ppo.policy_version)
    else:
        final_policy_hash = "none"
        parameter_drift = 0.0
        final_policy_version = -1

    front_dir = Path(out_dir) / "fronts"
    front_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"fold{fold}_{dataset}_{Path(instance_name).stem}_{variant}_"
        f"g{budget}_seed{seed}_rep{replica}.pkl"
    )
    front_path = front_dir / filename
    front_payload = {
        "protocol": PROTOCOL,
        "fold": int(fold), "dataset": dataset, "instance": instance_name,
        "variant": variant, "budget": int(budget), "seed": int(seed),
        "replica": int(replica), "code_hash": code_hash,
        "design_hash": design_hash, "input_hash": input_hash,
        "split_hash": split_hash, "population_size": FORMAL_POP_SIZE,
        "reference_snapshot_sha256": reference_snapshot_sha256,
        "checkpoint_sha256": (
            checkpoint_marker["checkpoint_sha256"] if checkpoint_marker else "none"
        ),
        "checkpoint_weight_semantic_sha256": (
            checkpoint_marker["weight_semantic_sha256"] if checkpoint_marker else "none"
        ),
        "rng_seeds": rng_seeds,
        "objectives": result["objectives"],
        "operator_sequence": result["operator_sequence"],
        "reward_sequence": result["reward_sequence"],
        "hv_delta_sequence": result["hv_delta_sequence"],
        "ppo_update_stats": result["ppo_update_stats"],
        "initial_policy_semantic_sha256": initial_policy_hash,
        "final_policy_semantic_sha256": final_policy_hash,
    }
    atomic_pickle(front_path, front_payload)
    front_file_hash = file_sha256(front_path)
    metrics = {
        key: value for key, value in result.items()
        if key not in {
            "objectives", "operator_sequence", "reward_sequence",
            "hv_delta_sequence", "ppo_update_stats",
        }
    }
    offline = (
        checkpoint_payload["metadata"]["cumulative"]
        if checkpoint_payload is not None
        and variant in {"XPrePPO_Frozen", "XPrePPO_Online_R16"}
        else {}
    )
    row = {
        "Protocol": PROTOCOL,
        "Fold": int(fold), "dataset": dataset, "instance": instance_name,
        "variant": variant, "Budget": int(budget), "seed": int(seed),
        "Replica": int(replica), "Population_size": FORMAL_POP_SIZE,
        "Worker_count": int(worker_count),
        "Initial_evaluations": int(result["initial_evaluations"]),
        "Offspring_evaluations": int(result["offspring_evaluations"]),
        "Offline_pretraining_episodes": (
            int(checkpoint_payload["metadata"]["completed_episodes"])
            if offline else 0
        ),
        "Offline_objective_evaluations": int(offline.get("objective_evaluations", 0)),
        "Offline_PPO_samples": int(offline.get("ppo_samples", 0)),
        "Offline_PPO_collected_transitions": int(
            offline.get("ppo_collected_transitions", 0)
        ),
        "Offline_PPO_discarded_singletons": int(
            offline.get("ppo_discarded_singletons", 0)
        ),
        "Offline_PPO_updates": int(offline.get("ppo_updates", 0)),
        "Offline_PPO_optimizer_steps": int(offline.get("ppo_optimizer_steps", 0)),
        "Offline_CPU_seconds": float(offline.get("cpu_seconds", 0.0)),
        "Offline_wall_seconds": float(offline.get("wall_seconds", 0.0)),
        "Online_PPO_samples": int(result["ppo_samples"]),
        "Online_PPO_collected_transitions": int(
            result["ppo_collected_transitions"]
        ),
        "Online_PPO_discarded_singletons": int(
            result["ppo_discarded_singletons"]
        ),
        "PPO_controlled_actions": int(result["ppo_controlled_actions"]),
        "Online_PPO_updates": int(result["ppo_updates"]),
        "Online_PPO_optimizer_steps": int(result["ppo_optimizer_steps"]),
        "Transition_gen": int(result["transition_generation"]),
        "Transition_reason": result["transition_reason"],
        "Operator_entropy": float(result["operator_entropy"]),
        "Policy_parameter_L2_drift": float(parameter_drift),
        "Policy_version_final": int(final_policy_version),
        "Initial_policy_semantic_sha256": initial_policy_hash,
        "Final_policy_semantic_sha256": final_policy_hash,
        "Checkpoint_role": checkpoint_role,
        "Checkpoint_sha256": (
            checkpoint_marker["checkpoint_sha256"] if checkpoint_marker else "none"
        ),
        "Checkpoint_weight_semantic_sha256": (
            checkpoint_marker["weight_semantic_sha256"] if checkpoint_marker else "none"
        ),
        "Checkpoint_size_bytes": (
            int(checkpoint_marker["checkpoint_size_bytes"])
            if checkpoint_marker else 0
        ),
        "Offline_checkpoint_bytes": (
            int(checkpoint_marker["checkpoint_size_bytes"])
            if checkpoint_marker is not None
            and variant in {"XPrePPO_Frozen", "XPrePPO_Online_R16"}
            else 0
        ),
        "Front_sha256": front_file_hash,
        "Front_semantic_sha256": result["front_semantic_sha256"],
        "front_pickle": front_path.as_posix(),
        "Code_hash": code_hash, "Design_hash": design_hash,
        "Input_hash": input_hash, "Split_hash": split_hash,
        "Reference_snapshot_sha256": reference_snapshot_sha256,
        **{
            key: value for key, value in metrics.items()
            if key not in {
                "initial_evaluations", "offspring_evaluations",
                "ppo_samples", "ppo_updates", "ppo_optimizer_steps",
                "ppo_consumed_samples", "ppo_collected_transitions",
                "ppo_discarded_singletons", "ppo_controlled_actions",
                "transition_generation", "transition_reason", "operator_entropy",
                "front_semantic_sha256",
            }
        },
    }
    return row


def code_manifest():
    missing = [path for path in CODE_FILES if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"missing v7 code files: {missing}")
    files = {path: file_sha256(path) for path in CODE_FILES}
    return {"files": files, "code_hash": canonical_hash(files)}


def input_manifest(input_records):
    semantic = {
        f"{dataset}/{name}": record["canonical_token_sha256"]
        for (dataset, name), record in sorted(input_records.items())
    }
    return {
        "files": {
            f"{dataset}/{name}": record
            for (dataset, name), record in sorted(input_records.items())
        },
        "input_hash": canonical_hash(semantic),
        "hash_basis": "UTF-8 whitespace-token stream",
    }


def build_run_manifest(*, smoke=False, smoke_design=None):
    instances = load_all_instances()
    input_records = instance_input_records()
    split = validate_split(instances, input_records)
    code = code_manifest()
    inputs = input_manifest(input_records)
    split_hash = canonical_hash(split)
    design = formal_design(split) if not smoke else dict(smoke_design or {})
    snapshot = Path(REFERENCE_SNAPSHOT)
    if not snapshot.is_file():
        raise RuntimeError(
            f"frozen v7 reference snapshot is required before execution: {snapshot}"
        )
    return {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "design": design,
        "design_hash": canonical_hash(design),
        "split_hash": split_hash,
        "code_hash": code["code_hash"],
        "code_files": code["files"],
        "input_hash": inputs["input_hash"],
        "input_files": inputs["files"],
        "reference_snapshot": REFERENCE_SNAPSHOT,
        "reference_snapshot_sha256": file_sha256(snapshot),
    }


def write_or_validate_manifest(out_dir, *, smoke=False, smoke_design=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    current = build_run_manifest(smoke=smoke, smoke_design=smoke_design)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable = (
            "protocol", "hostname", "platform", "python", "numpy", "torch",
            "design_hash", "split_hash", "code_hash", "input_hash",
            "reference_snapshot_sha256",
        )
        mismatches = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in stable if existing.get(key) != current.get(key)
        }
        if mismatches:
            raise RuntimeError(f"refusing incompatible v7 resume: {mismatches}")
        return existing
    result_files = [out_dir / "pretraining_runs.csv", out_dir / "runs.csv"]
    if any(path.is_file() for path in result_files):
        raise RuntimeError("v7 result CSV exists without immutable run_manifest.json")
    atomic_json(path, current)
    return current


def write_csv_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    if any(list(row.keys()) != fieldnames for row in rows):
        raise RuntimeError(f"inconsistent CSV fields for {path}")
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluation_journal_path(out_dir, row):
    digest = canonical_hash(list(evaluation_key(row)))
    return Path(out_dir) / "evaluation_journal" / f"{digest}.json"


def journal_evaluation_row(out_dir, row):
    atomic_json(evaluation_journal_path(out_dir, row), row)


def recover_evaluation_journal(out_dir, hashes):
    """Atomically compact per-run journals into runs.csv after interruption."""
    out_dir = Path(out_dir)
    if (out_dir / "runs.csv").is_file():
        existing_evaluation_keys(out_dir / "runs.csv", hashes)
    row_map = {
        evaluation_key(row): row
        for row in read_csv_rows(out_dir / "runs.csv")
    }
    for journal in sorted((out_dir / "evaluation_journal").glob("*.json")):
        row = json.loads(journal.read_text(encoding="utf-8"))
        key = evaluation_key(row)
        expected_name = canonical_hash(list(key)) + ".json"
        if journal.name != expected_name:
            raise RuntimeError(f"evaluation journal key/name mismatch: {journal}")
        row_map[key] = row
    if row_map:
        rows = sorted(row_map.values(), key=evaluation_key)
        write_csv_atomic(out_dir / "runs.csv", rows)
        existing_evaluation_keys(out_dir / "runs.csv", hashes)
    return row_map


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pretraining_key(row):
    return (
        int(row["Fold"]), int(row["Replica"]), int(row["Pass"]),
        int(row["Position"]), row["dataset"], row["instance"],
    )


def evaluation_key(row):
    return (
        int(row["Fold"]), row["dataset"], row["instance"], row["variant"],
        int(row["Budget"]), int(row["seed"]), int(row["Replica"]),
    )


def verify_pretraining_rows(rows, *, expected, hashes):
    keys = [pretraining_key(row) for row in rows]
    if len(rows) != int(expected) or len(set(keys)) != int(expected):
        raise RuntimeError(
            f"pretraining grid incomplete rows={len(rows)} unique={len(set(keys))} "
            f"expected={expected}"
        )
    for row in rows:
        for column, expected_hash in (
            ("Code_hash", hashes["code_hash"]),
            ("Design_hash", hashes["design_hash"]),
            ("Input_hash", hashes["input_hash"]),
            ("Split_hash", hashes["split_hash"]),
            ("Reference_snapshot_sha256", hashes["reference_snapshot_sha256"]),
        ):
            if row.get(column) != expected_hash:
                raise RuntimeError(f"pretraining {column} mismatch in key {pretraining_key(row)}")
        if int(row["Initial_evaluations"]) != FORMAL_POP_SIZE:
            raise RuntimeError("pretraining initial-evaluation budget mismatch")
        expected_offspring = FORMAL_POP_SIZE * int(row["Budget"])
        if int(row["Offspring_evaluations"]) != expected_offspring:
            raise RuntimeError("pretraining offspring-evaluation budget mismatch")
    return {"rows": len(rows), "unique_keys": len(set(keys))}


def existing_evaluation_keys(path, hashes):
    rows = read_csv_rows(path)
    keys = [evaluation_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate evaluation primary keys in runs.csv")
    for row in rows:
        if row.get("Protocol") != PROTOCOL:
            raise RuntimeError("evaluation protocol mismatch")
        for column, expected_hash in (
            ("Code_hash", hashes["code_hash"]),
            ("Design_hash", hashes["design_hash"]),
            ("Input_hash", hashes["input_hash"]),
            ("Split_hash", hashes["split_hash"]),
            ("Reference_snapshot_sha256", hashes["reference_snapshot_sha256"]),
        ):
            if row.get(column) != expected_hash:
                raise RuntimeError(f"evaluation {column} mismatch in key {evaluation_key(row)}")
        front = Path(row["front_pickle"])
        if not front.is_file() or file_sha256(front) != row.get("Front_sha256"):
            raise RuntimeError(f"evaluation front missing or modified: {front}")
        with front.open("rb") as stream:
            payload = pickle.load(stream)
        payload_checks = {
            "protocol": (payload.get("protocol"), PROTOCOL),
            "fold": (int(payload.get("fold", -1)), int(row["Fold"])),
            "dataset": (payload.get("dataset"), row["dataset"]),
            "instance": (payload.get("instance"), row["instance"]),
            "variant": (payload.get("variant"), row["variant"]),
            "budget": (int(payload.get("budget", -1)), int(row["Budget"])),
            "seed": (int(payload.get("seed", -1)), int(row["seed"])),
            "replica": (int(payload.get("replica", -2)), int(row["Replica"])),
            "code_hash": (payload.get("code_hash"), hashes["code_hash"]),
            "design_hash": (payload.get("design_hash"), hashes["design_hash"]),
            "input_hash": (payload.get("input_hash"), hashes["input_hash"]),
            "split_hash": (payload.get("split_hash"), hashes["split_hash"]),
            "reference_snapshot_sha256": (
                payload.get("reference_snapshot_sha256"),
                hashes["reference_snapshot_sha256"],
            ),
            "checkpoint_sha256": (
                payload.get("checkpoint_sha256"), row["Checkpoint_sha256"]
            ),
        }
        mismatches = {
            name: {"observed": observed, "expected": expected}
            for name, (observed, expected) in payload_checks.items()
            if observed != expected
        }
        if mismatches:
            raise RuntimeError(f"evaluation front metadata mismatch in {front}: {mismatches}")
        objectives = np.asarray(payload.get("objectives", []), dtype=float)
        if (
            objectives.ndim != 2 or objectives.shape[1] != 3 or len(objectives) == 0
            or not np.isfinite(objectives).all()
            or front_semantic_hash(objectives) != row.get("Front_semantic_sha256")
            or not is_deduplicated_nondominated(objectives)
        ):
            raise RuntimeError(f"evaluation front objective audit failed: {front}")
        if row["variant"] != "UCBOnly":
            checkpoint = checkpoint_path(
                path.parent, int(row["Fold"]), int(row["Replica"]), "terminal"
            )
            marker_path = checkpoint_marker_path(checkpoint)
            if not marker_path.is_file():
                raise RuntimeError(f"referenced evaluation checkpoint is missing: {checkpoint}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            object_path = marker_path.parent / marker.get("object_path", "")
            if (
                marker.get("checkpoint_sha256") != row["Checkpoint_sha256"]
                or not object_path.is_file()
                or file_sha256(object_path) != row["Checkpoint_sha256"]
            ):
                raise RuntimeError(f"evaluation checkpoint changed after run: {checkpoint}")
    return set(keys)


def build_pretraining_tasks(args, hashes):
    return [
        (
            fold, replica, args.out_dir,
            hashes["code_hash"], hashes["design_hash"], hashes["input_hash"],
            hashes["split_hash"], args.pretrain_passes, args.pretrain_budget,
            args.training_limit,
            hashes["reference_snapshot_sha256"],
        )
        for fold in args.folds
        for replica in args.replicas
    ]


def run_pretraining_phase(args, hashes):
    tasks = build_pretraining_tasks(args, hashes)
    print(f"[v7/pretrain] chains={len(tasks)}", flush=True)
    failures = []
    chain_results = []
    start = time.time()
    workers = max(1, min(args.workers, len(tasks) or 1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_pretraining_chain, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                result = future.result()
                chain_results.append(result)
                status = f"fold={result['fold']} replica={result['replica']}"
            except Exception as exc:
                failure = {
                    "phase": "pretrain", "fold": task[0], "replica": task[1],
                    "error": repr(exc), "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                status = f"ERROR fold={task[0]} replica={task[1]}: {exc!r}"
            elapsed = time.time() - start
            eta = (len(tasks) - completed) * elapsed / max(completed, 1)
            print(
                f"[v7/pretrain {completed}/{len(tasks)}] {status} ETA={eta/60:.1f}min",
                flush=True,
            )
    if failures:
        atomic_json(Path(args.out_dir) / "pretraining_failures.json", failures)
        raise RuntimeError(f"{len(failures)} pretraining chains failed")
    rows = [row for result in chain_results for row in result["episode_records"]]
    rows.sort(key=pretraining_key)
    expected = (
        len(args.folds) * len(args.replicas) * args.pretrain_passes
        * (args.training_limit or 40)
    )
    verification = verify_pretraining_rows(rows, expected=expected, hashes=hashes)
    write_csv_atomic(Path(args.out_dir) / "pretraining_runs.csv", rows)
    verification.update({
        "protocol": PROTOCOL,
        "checkpoint_count": len(chain_results),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    atomic_json(Path(args.out_dir) / "pretraining_complete.json", verification)
    return verification


def build_evaluation_tasks(args, hashes):
    instances = load_all_instances()
    input_records = instance_input_records()
    validate_split(instances, input_records)
    done = existing_evaluation_keys(Path(args.out_dir) / "runs.csv", hashes)
    tasks = []
    per_fold_seen = {fold: 0 for fold in args.folds}
    for dataset, instance_name, instance in instances:
        fold = fold_for_instance(dataset, instance_name)
        if fold not in args.folds:
            continue
        if args.eval_limit and per_fold_seen[fold] >= args.eval_limit:
            continue
        per_fold_seen[fold] += 1
        for variant in args.variants:
            for budget in args.budgets:
                for seed in args.eval_seeds:
                    replica = -1 if variant == "UCBOnly" else (int(seed) - 42) % 5
                    if replica not in args.replicas and variant != "UCBOnly":
                        continue
                    key = (
                        fold, dataset, instance_name, variant, int(budget),
                        int(seed), int(replica),
                    )
                    if key not in done:
                        tasks.append((
                            dataset, instance_name, instance, fold, variant,
                            int(budget), int(seed), int(replica), args.out_dir,
                            hashes["code_hash"], hashes["design_hash"],
                            hashes["input_hash"], hashes["split_hash"], args.workers,
                            args.pretrain_passes * (args.training_limit or 40),
                            hashes["reference_snapshot_sha256"],
                        ))
    return tasks


def run_evaluation_phase(args, hashes):
    if not (Path(args.out_dir) / "pretraining_complete.json").is_file():
        raise RuntimeError("evaluation cannot start before pretraining_complete.json")
    row_map = recover_evaluation_journal(args.out_dir, hashes)
    tasks = build_evaluation_tasks(args, hashes)
    print(f"[v7/eval] pending={len(tasks)}", flush=True)
    failures = []
    start = time.time()
    workers = max(1, min(args.workers, len(tasks) or 1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_evaluation_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                row = future.result()
                journal_evaluation_row(args.out_dir, row)
                row_map[evaluation_key(row)] = row
                if completed % 20 == 0:
                    write_csv_atomic(
                        Path(args.out_dir) / "runs.csv",
                        sorted(row_map.values(), key=evaluation_key),
                    )
                status = (
                    f"fold={row['Fold']} {row['dataset']}/{row['instance']} "
                    f"{row['variant']} g={row['Budget']} seed={row['seed']} "
                    f"T={float(row['elapsed_seconds']):.1f}s"
                )
            except Exception as exc:
                failure = {
                    "phase": "evaluation", "dataset": task[0], "instance": task[1],
                    "fold": task[3], "variant": task[4], "budget": task[5],
                    "seed": task[6], "replica": task[7], "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                status = (
                    f"ERROR fold={task[3]} {task[0]}/{task[1]} {task[4]} "
                    f"g={task[5]} seed={task[6]}: {exc!r}"
                )
            elapsed = time.time() - start
            eta = (len(tasks) - completed) * elapsed / max(completed, 1)
            print(
                f"[v7/eval {completed}/{len(tasks)}] {status} ETA={eta/60:.1f}min",
                flush=True,
            )
    if failures:
        atomic_json(Path(args.out_dir) / "evaluation_failures.json", failures)
        raise RuntimeError(f"{len(failures)} evaluation runs failed")
    if row_map:
        write_csv_atomic(
            Path(args.out_dir) / "runs.csv",
            sorted(row_map.values(), key=evaluation_key),
        )


def verify_formal_complete(out_dir, hashes):
    pretrain_rows = read_csv_rows(Path(out_dir) / "pretraining_runs.csv")
    pretrain = verify_pretraining_rows(
        pretrain_rows, expected=FORMAL_PRETRAIN_ROWS, hashes=hashes
    )
    terminal_markers = []
    for fold in range(1, 6):
        for replica in FORMAL_REPLICAS:
            path = checkpoint_path(out_dir, fold, replica, "terminal")
            _, marker = load_checkpoint(
                path,
                expected={
                    "fold": fold, "replica": replica,
                    "code_hash": hashes["code_hash"],
                    "design_hash": hashes["design_hash"],
                    "input_hash": hashes["input_hash"],
                    "split_hash": hashes["split_hash"],
                    "completed_episodes": 80,
                },
            )
            terminal_markers.append(marker)
    rows = read_csv_rows(Path(out_dir) / "runs.csv")
    keys = [evaluation_key(row) for row in rows]
    if len(rows) != FORMAL_EVAL_ROWS or len(set(keys)) != FORMAL_EVAL_ROWS:
        raise RuntimeError(
            f"evaluation grid incomplete rows={len(rows)} unique={len(set(keys))} "
            f"expected={FORMAL_EVAL_ROWS}"
        )
    existing_evaluation_keys(Path(out_dir) / "runs.csv", hashes)
    expected_keys = set()
    for dataset, names in (("Brandimarte", range(1, 11)), ("Hurink_edata", range(1, 41))):
        prefix = "Mk" if dataset == "Brandimarte" else "la"
        for number in names:
            instance_name = f"{prefix}{number:02d}.fjs"
            fold = fold_for_instance(dataset, instance_name)
            for variant in VARIANTS:
                for budget in FORMAL_BUDGETS:
                    for seed in FORMAL_EVAL_SEEDS:
                        replica = -1 if variant == "UCBOnly" else (seed - 42) % 5
                        expected_keys.add((
                            fold, dataset, instance_name, variant, budget, seed, replica
                        ))
    if set(keys) != expected_keys:
        missing = list(expected_keys - set(keys))[:5]
        extra = list(set(keys) - expected_keys)[:5]
        raise RuntimeError(f"evaluation grid mismatch missing={missing} extra={extra}")
    checkpoint_manifest = hashlib.sha256(
        "\n".join(sorted(marker["checkpoint_sha256"] for marker in terminal_markers))
        .encode("ascii")
    ).hexdigest()
    front_manifest = hashlib.sha256(
        "\n".join(sorted(row["Front_sha256"] for row in rows)).encode("ascii")
    ).hexdigest()
    if any((Path(out_dir) / name).is_file() for name in (
        "pretraining_failures.json", "evaluation_failures.json"
    )):
        raise RuntimeError("failure marker exists; formal completion is prohibited")
    return {
        "protocol": PROTOCOL,
        "pretraining_rows": pretrain["rows"],
        "pretraining_unique_keys": pretrain["unique_keys"],
        "evaluation_rows": len(rows),
        "evaluation_unique_keys": len(set(keys)),
        "terminal_checkpoints": len(terminal_markers),
        "checkpoint_manifest_sha256": checkpoint_manifest,
        "front_manifest_sha256": front_manifest,
        **hashes,
    }


def parse_int_list(text):
    return tuple(int(value.strip()) for value in str(text).split(",") if value.strip())


def parse_str_list(text):
    return tuple(value.strip() for value in str(text).split(",") if value.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--phase", choices=("all", "pretrain", "evaluate"), default="all")
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--replicas", default="0,1,2,3,4")
    parser.add_argument("--pretrain-passes", type=int, default=FORMAL_PRETRAIN_PASSES)
    parser.add_argument("--pretrain-budget", type=int, default=FORMAL_PRETRAIN_BUDGET)
    parser.add_argument("--training-limit", type=int, default=0)
    parser.add_argument("--budgets", default="50,100,200")
    parser.add_argument("--eval-seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.folds = parse_int_list(args.folds)
    args.replicas = parse_int_list(args.replicas)
    args.budgets = parse_int_list(args.budgets)
    args.eval_seeds = parse_int_list(args.eval_seeds)
    args.variants = parse_str_list(args.variants)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.smoke:
        frozen = {
            "out_dir": Path(args.out_dir).resolve() == Path(DEFAULT_OUT).resolve(),
            "workers": args.workers == FORMAL_WORKERS,
            "folds": args.folds == (1, 2, 3, 4, 5),
            "replicas": args.replicas == FORMAL_REPLICAS,
            "pretrain_passes": args.pretrain_passes == FORMAL_PRETRAIN_PASSES,
            "pretrain_budget": args.pretrain_budget == FORMAL_PRETRAIN_BUDGET,
            "training_limit": args.training_limit == 0,
            "budgets": args.budgets == FORMAL_BUDGETS,
            "eval_seeds": args.eval_seeds == FORMAL_EVAL_SEEDS,
            "variants": args.variants == VARIANTS,
            "eval_limit": args.eval_limit == 0,
        }
        if not all(frozen.values()):
            raise SystemExit(f"formal v7 arguments are frozen: {frozen}")
        manifest = write_or_validate_manifest(args.out_dir)
    else:
        if Path(args.out_dir).resolve() == Path(DEFAULT_OUT).resolve():
            raise SystemExit("smoke runs must not use the formal v7 output directory")
        smoke_design = {
            "smoke": True, "folds": args.folds, "replicas": args.replicas,
            "pretrain_passes": args.pretrain_passes,
            "pretrain_budget": args.pretrain_budget,
            "training_limit": args.training_limit, "budgets": args.budgets,
            "eval_seeds": args.eval_seeds, "variants": args.variants,
            "eval_limit": args.eval_limit, "workers": args.workers,
        }
        manifest = write_or_validate_manifest(
            args.out_dir, smoke=True, smoke_design=smoke_design
        )
    hashes = {
        "code_hash": manifest["code_hash"],
        "design_hash": manifest["design_hash"],
        "input_hash": manifest["input_hash"],
        "split_hash": manifest["split_hash"],
        "reference_snapshot_sha256": manifest["reference_snapshot_sha256"],
    }
    print(
        f"[v7] protocol={PROTOCOL} phase={args.phase} out={args.out_dir}",
        flush=True,
    )
    if args.phase in {"all", "pretrain"}:
        run_pretraining_phase(args, hashes)
    if args.phase in {"all", "evaluate"}:
        run_evaluation_phase(args, hashes)
    if args.smoke:
        print("[v7] smoke execution complete; formal completeness gate skipped", flush=True)
        return
    if args.phase == "all" or args.phase == "evaluate":
        completion = verify_formal_complete(args.out_dir, hashes)
        completion["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(Path(args.out_dir) / "pipeline_complete.json", completion)
        print(
            f"[v7] COMPLETE pretraining={completion['pretraining_rows']} "
            f"evaluation={completion['evaluation_rows']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
