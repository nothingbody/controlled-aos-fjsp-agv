"""SA-AOS parameter sensitivity experiments for the major revision.

This runner reuses the isolated AOS backbone from ``run_revision_aos.py`` and
changes only one SA-AOS parameter at a time. By default it runs Mk06 with
10 seeds, which matches the manuscript's representative-parameter-sensitivity
setting while covering the reviewer-requested SA-AOS parameters.
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
    DATASETS,
    HybridSelector,
    apply_operator,
    crowding_distance,
    evaluate_archive,
    initialize_population,
    normalized_entropy,
    update_archive,
)
from src.algorithm.nsga3.selection import (
    compute_hypervolume,
    generate_reference_points,
    nsga3_select,
)


DEFAULT_CONFIG = {
    "variant": "Default",
    "parameter": "Default",
    "setting": "W=50,c=1.0,n_min=3,B_min=48,weights=0.5/0.3/0.2,lr=3e-4,adaptive",
    "window": 50,
    "ucb_c": 1.0,
    "min_per_op": 3,
    "min_buffer": 48,
    "reward_weights": (0.5, 0.3, 0.2),
    "ppo_lr": 3e-4,
    "transition_mode": "adaptive",
}


SENSITIVITY_CONFIGS = [
    DEFAULT_CONFIG,
    {"variant": "W20", "parameter": "UCB window W", "setting": "20", "window": 20},
    {"variant": "W80", "parameter": "UCB window W", "setting": "80", "window": 80},
    {"variant": "W100", "parameter": "UCB window W", "setting": "100", "window": 100},
    {"variant": "c0.5", "parameter": "UCB coefficient c", "setting": "0.5", "ucb_c": 0.5},
    {"variant": "c1.5", "parameter": "UCB coefficient c", "setting": "1.5", "ucb_c": 1.5},
    {"variant": "c2.0", "parameter": "UCB coefficient c", "setting": "2.0", "ucb_c": 2.0},
    {"variant": "nmin1", "parameter": "n_min", "setting": "1", "min_per_op": 1},
    {"variant": "nmin5", "parameter": "n_min", "setting": "5", "min_per_op": 5},
    {"variant": "nmin10", "parameter": "n_min", "setting": "10", "min_per_op": 10},
    {"variant": "B32", "parameter": "B_min", "setting": "32", "min_buffer": 32},
    {"variant": "B64", "parameter": "B_min", "setting": "64", "min_buffer": 64},
    {"variant": "B96", "parameter": "B_min", "setting": "96", "min_buffer": 96},
    {
        "variant": "wSurvHeavy",
        "parameter": "reward weights",
        "setting": "0.7/0.2/0.1",
        "reward_weights": (0.7, 0.2, 0.1),
    },
    {
        "variant": "wHVHeavy",
        "parameter": "reward weights",
        "setting": "0.3/0.5/0.2",
        "reward_weights": (0.3, 0.5, 0.2),
    },
    {
        "variant": "wCmaxHeavy",
        "parameter": "reward weights",
        "setting": "0.3/0.2/0.5",
        "reward_weights": (0.3, 0.2, 0.5),
    },
    {
        "variant": "wBalanced",
        "parameter": "reward weights",
        "setting": "1/3/1/3/1/3",
        "reward_weights": (1 / 3, 1 / 3, 1 / 3),
    },
    {"variant": "lr1e-4", "parameter": "PPO learning rate", "setting": "1e-4", "ppo_lr": 1e-4},
    {"variant": "lr1e-3", "parameter": "PPO learning rate", "setting": "1e-3", "ppo_lr": 1e-3},
    {
        "variant": "FixedTransition48",
        "parameter": "transition rule",
        "setting": "fixed@48",
        "transition_mode": "fixed",
        "fixed_transition": 48,
    },
    {
        "variant": "RandomTransition",
        "parameter": "transition rule",
        "setting": "random@20-80%",
        "transition_mode": "random",
    },
]


def merged_config(config: dict) -> dict:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return merged


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def weighted_reward(weights, survival, hv_delta, cmax_delta):
    alpha, beta, gamma = weights
    return float(alpha * survival + beta * hv_delta + gamma * cmax_delta)


def load_instances(instance_filter: str | None = "Mk06.fjs"):
    wanted = None
    if instance_filter:
        wanted = {item.strip() for item in instance_filter.split(",") if item.strip()}
    loaded = []
    for dataset, path in DATASETS.items():
        for inst_name, instance in load_benchmark_set(path, num_agv=3):
            stem = Path(inst_name).stem
            keys = {inst_name, stem, f"{dataset}/{inst_name}", f"{dataset}/{stem}"}
            if wanted is None or keys.intersection(wanted):
                loaded.append((dataset, inst_name, instance))
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
    dataset, inst_name, instance, raw_config, seed, pop_size, max_gen, out_dir = task
    config = merged_config(raw_config)
    rng = np.random.RandomState(seed)
    start = time.time()

    transition_mode = config["transition_mode"]
    random_transition = None
    if transition_mode == "random":
        random_transition = int(rng.randint(max(2, max_gen // 5), max(3, int(max_gen * 0.8))))

    selector = HybridSelector(
        rng,
        reward_scheme="sensitivity",
        transition_mode=transition_mode,
        fixed_transition=int(config.get("fixed_transition", max(1, int(max_gen * 0.48)))),
        random_transition=random_transition,
        min_per_op=int(config["min_per_op"]),
        min_buffer=int(config["min_buffer"]),
        ucb_c=float(config["ucb_c"]),
        window=int(config["window"]),
        lr=float(config["ppo_lr"]),
        device="cpu",
    )

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
        reward = weighted_reward(config["reward_weights"], survival, hv_delta, cmax_delta)
        selector.update(op_id, reward)

        hv_window.append(float(hv_delta))
        op_sequence.append(int(op_id))
        reward_sequence.append(float(reward))
        hv_delta_sequence.append(float(hv_delta))

        if current_hv > prev_hv * 1.001:
            stagnation = 0
        else:
            stagnation += 1
        prev_hv = current_hv
        prev_best_cmax = current_best_cmax
        population = new_population
        archive = update_archive(archive, population)

    metrics = evaluate_archive(archive)
    counts = np.bincount(op_sequence, minlength=len(selector.op_counts))
    last_counts = np.bincount(op_sequence[-20:], minlength=len(selector.op_counts))
    if (
        len(reward_sequence) > 2
        and np.std(reward_sequence) > 1e-12
        and np.std(hv_delta_sequence) > 1e-12
    ):
        reward_hv_corr = float(np.corrcoef(reward_sequence, hv_delta_sequence)[0, 1])
    else:
        reward_hv_corr = 0.0

    pkl_dir = Path(out_dir) / "fronts"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    variant = str(config["variant"])
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
    alpha, beta, gamma = config["reward_weights"]
    row = {
        "dataset": dataset,
        "instance": inst_name,
        "variant": variant,
        "parameter": config["parameter"],
        "setting": config["setting"],
        "seed": seed,
        "W": int(config["window"]),
        "ucb_c": float(config["ucb_c"]),
        "n_min": int(config["min_per_op"]),
        "B_min": int(config["min_buffer"]),
        "reward_alpha": float(alpha),
        "reward_beta": float(beta),
        "reward_gamma": float(gamma),
        "ppo_lr": float(config["ppo_lr"]),
        "transition_mode": transition_mode,
        "planned_transition": int(random_transition if random_transition is not None else config.get("fixed_transition", -1)),
        **metrics,
        "Time": float(time.time() - start),
        "Entropy_all": normalized_entropy(counts),
        "Entropy_last20": normalized_entropy(last_counts),
        "Transition_gen": int(selector.transition_gen),
        "Reward_HV_corr": reward_hv_corr,
        "Operator_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        "front_pickle": str(pkl_path),
    }
    return row


def build_tasks(args):
    configs = SENSITIVITY_CONFIGS
    if args.variants:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        configs = [cfg for cfg in configs if cfg["variant"] in wanted]
    instances = load_instances(args.instances)
    seeds = list(range(args.seed_start, args.seed_end))
    done = existing_keys(args.result_file)
    tasks = []
    for dataset, inst_name, instance in instances:
        for config in configs:
            for seed in seeds:
                key = (dataset, inst_name, config["variant"], seed)
                if key in done:
                    continue
                tasks.append((dataset, inst_name, instance, config, seed, args.pop_size, args.max_gen, args.out_dir))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/revision/sensitivity")
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--instances", default="Mk06.fjs")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.result_file is None:
        args.result_file = os.path.join(args.out_dir, "saos_parameter_sensitivity_runs.csv")

    tasks = build_tasks(args)
    total = len(tasks)
    print(f"[sensitivity] tasks={total} result={args.result_file}", flush=True)
    if total == 0:
        print("[sensitivity] nothing to do", flush=True)
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
