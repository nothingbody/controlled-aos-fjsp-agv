"""Operator-family ablation for the SA-AOS module.

The experiment keeps the same NSGA-III backbone, SA-AOS transition rule, reward
definition, population, generation budget, and decoder. It only removes one
operator family from the library, addressing the revision request for machine,
AGV, and speed operator ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.loader import load_benchmark_set
from experiments.run_revision_aos import (
    ALL_OPERATORS,
    DATASETS,
    HybridSelector,
    apply_operator,
    evaluate_archive,
    initialize_population,
    normalized_entropy,
    update_archive,
)
from src.algorithm.nsga3.selection import compute_hypervolume, generate_reference_points, nsga3_select


N_OPS = len(ALL_OPERATORS)

OPERATOR_VARIANTS = [
    {
        "variant": "FullLibrary",
        "description": "Full 10-operator library",
        "allowed_ops": list(range(N_OPS)),
    },
    {
        "variant": "NoMachineOperator",
        "description": "Remove UniformMA and MachineReassign",
        "allowed_ops": [op for op in range(N_OPS) if op not in {2, 7}],
    },
    {
        "variant": "NoAGVOperator",
        "description": "Remove UniformAGV and AGVReassign",
        "allowed_ops": [op for op in range(N_OPS) if op not in {3, 8}],
    },
    {
        "variant": "NoSpeedOperator",
        "description": "Remove SpeedAdjust",
        "allowed_ops": [op for op in range(N_OPS) if op not in {9}],
    },
]


class MappedHybridSelector:
    def __init__(self, rng, allowed_ops, device="cpu"):
        self.allowed_ops = list(allowed_ops)
        self.inner = HybridSelector(
            rng,
            n_ops=len(self.allowed_ops),
            reward_scheme="composite",
            transition_mode="adaptive",
            min_per_op=3,
            min_buffer=48,
            ucb_c=1.0,
            window=50,
            lr=3e-4,
            device=device,
        )
        self.last_local_op = None

    @property
    def transition_gen(self):
        return self.inner.transition_gen

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        local_op = self.inner.select(gen, max_gen, stagnation, diversity, hv_trend)
        self.last_local_op = int(local_op)
        return int(self.allowed_ops[local_op])

    def update(self, actual_op, reward):
        if self.last_local_op is None:
            raise RuntimeError("update called before select")
        self.inner.update(self.last_local_op, reward)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def compute_composite_reward(survival, hv_delta, cmax_delta):
    return float(0.5 * survival + 0.3 * hv_delta + 0.2 * cmax_delta)


def load_instances(limit=None):
    loaded = []
    for dataset, path in DATASETS.items():
        for inst_name, instance in load_benchmark_set(path, num_agv=3):
            loaded.append((dataset, inst_name, instance))
    if limit:
        return loaded[:limit]
    return loaded


def existing_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["dataset"], row["instance"], row["variant"], int(row["seed"])))
    return keys


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_single(task):
    dataset, inst_name, instance, config, seed, pop_size, max_gen, out_dir = task
    rng = np.random.RandomState(seed)
    start = time.time()
    selector = MappedHybridSelector(rng, config["allowed_ops"], device="cpu")
    ref_points = generate_reference_points(3, 8)
    population = initialize_population(instance, pop_size, rng)
    archive = update_archive([], population)

    pop_objs = np.array([c.objectives.to_array() for c in population])
    ref_point = pop_objs.max(axis=0) * 1.1
    prev_hv = compute_hypervolume(pop_objs, ref_point)
    prev_best_cmax = pop_objs[:, 0].min()
    stagnation = 0
    hv_window = deque(maxlen=20)
    op_sequence = []
    reward_sequence = []
    hv_delta_sequence = []

    for gen in range(max_gen):
        pop_objs = np.array([c.objectives.to_array() for c in population])
        diversity = float(np.std(pop_objs, axis=0).mean() / (np.mean(pop_objs) + 1e-10))
        hv_trend = float(np.mean(hv_window)) if hv_window else 0.0

        op_id = selector.select(gen, max_gen, stagnation, diversity, hv_trend)
        offspring = apply_operator(population, op_id, pop_size, 5, rng)
        for chrom in offspring:
            chrom._revision_uid = id(chrom)

        combined = population + offspring
        combined_objs = np.array([c.objectives.to_array() for c in combined])
        selected = nsga3_select(combined_objs, pop_size, ref_points)
        new_population = [combined[i] for i in selected]

        selected_ids = {getattr(c, "_revision_uid", None) for c in new_population}
        survived = sum(1 for c in offspring if getattr(c, "_revision_uid", None) in selected_ids)
        survival = survived / max(len(offspring), 1)

        new_objs = np.array([c.objectives.to_array() for c in new_population])
        current_hv = compute_hypervolume(new_objs, ref_point)
        hv_delta = np.clip((current_hv - prev_hv) / max(abs(prev_hv), 1e-10), -1.0, 1.0)
        current_best_cmax = new_objs[:, 0].min()
        cmax_delta = np.clip(
            (prev_best_cmax - current_best_cmax) / max(abs(prev_best_cmax), 1e-10),
            -1.0,
            1.0,
        )
        reward = compute_composite_reward(survival, hv_delta, cmax_delta)
        selector.update(op_id, reward)

        hv_window.append(float(hv_delta))
        op_sequence.append(int(op_id))
        reward_sequence.append(float(reward))
        hv_delta_sequence.append(float(hv_delta))

        stagnation = 0 if current_hv > prev_hv * 1.001 else stagnation + 1
        prev_hv = current_hv
        prev_best_cmax = current_best_cmax
        population = new_population
        archive = update_archive(archive, population)

    metrics = evaluate_archive(archive)
    counts = np.bincount(op_sequence, minlength=N_OPS)
    last_counts = np.bincount(op_sequence[-20:], minlength=N_OPS)
    if (
        len(reward_sequence) > 2
        and np.std(reward_sequence) > 1e-12
        and np.std(hv_delta_sequence) > 1e-12
    ):
        reward_hv_corr = float(np.corrcoef(reward_sequence, hv_delta_sequence)[0, 1])
    else:
        reward_hv_corr = 0.0

    variant = config["variant"]
    pkl_dir = Path(out_dir) / "fronts"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / f"{dataset}_{Path(inst_name).stem}_{safe_name(variant)}_seed{seed}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "dataset": dataset,
                "instance": inst_name,
                "variant": variant,
                "seed": seed,
                "objectives": metrics["objectives"],
                "operator_sequence": op_sequence,
                "reward_sequence": reward_sequence,
                "hv_delta_sequence": hv_delta_sequence,
                "config": config,
            },
            f,
        )

    metrics.pop("objectives")
    return {
        "dataset": dataset,
        "instance": inst_name,
        "variant": variant,
        "description": config["description"],
        "removed_ops": json.dumps([op for op in range(N_OPS) if op not in config["allowed_ops"]]),
        "allowed_ops": json.dumps(config["allowed_ops"]),
        "reward_scheme": "composite",
        "seed": seed,
        **metrics,
        "Time": float(time.time() - start),
        "Entropy_all": normalized_entropy(counts),
        "Entropy_last20": normalized_entropy(last_counts),
        "Transition_gen": int(selector.transition_gen),
        "Reward_HV_corr": reward_hv_corr,
        "Operator_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        "front_pickle": str(pkl_path),
    }


def build_tasks(args):
    variants = OPERATOR_VARIANTS
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [cfg for cfg in variants if cfg["variant"] in wanted]
    instances = load_instances(limit=args.limit_instances)
    seeds = list(range(args.seed_start, args.seed_end))
    done = existing_keys(args.result_file)
    tasks = []
    for dataset, inst_name, instance in instances:
        for config in variants:
            for seed in seeds:
                key = (dataset, inst_name, config["variant"], seed)
                if key in done:
                    continue
                tasks.append((dataset, inst_name, instance, config, seed, args.pop_size, args.max_gen, args.out_dir))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/revision/operator_ablation")
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--limit-instances", type=int, default=None)
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.result_file is None:
        args.result_file = os.path.join(args.out_dir, "operator_ablation_runs.csv")

    tasks = build_tasks(args)
    total = len(tasks)
    print(f"[operator-ablation] tasks={total} result={args.result_file}", flush=True)
    if total == 0:
        print("[operator-ablation] nothing to do", flush=True)
        return

    start = time.time()
    completed = 0
    n_workers = max(1, min(args.workers, total, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(run_single, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            completed += 1
            try:
                row = future.result()
                append_row(args.result_file, row)
                msg = (
                    f"[{completed}/{total}] {row['dataset']}/{row['instance']} "
                    f"{row['variant']} seed={row['seed']} HV={row['HV_local']:.3g} "
                    f"Cmax={row['Cmax_best']:.1f} NSol={row['NSol']} T={row['Time']:.1f}s"
                )
            except Exception as exc:
                dataset, inst_name, _, config, seed, *_ = task
                msg = f"[{completed}/{total}] ERROR {dataset}/{inst_name} {config['variant']} seed={seed}: {exc!r}"
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / max(rate, 1e-9)
            print(f"{msg} ETA={eta/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
