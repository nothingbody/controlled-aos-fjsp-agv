"""Budget/stage-stress experiments for AOS comparison.

This runner reuses the revision AOS implementation while adding a budget
dimension. It is intended to test whether the stage-aware UCB-to-PPO controller
behaves differently from strong UCB-like baselines when the search has a
short, medium, or longer late-stage exploitation window.

Only the operator-selection module is changed. The NSGA-III backbone,
encoding, operator library, decoding, population size, and instance set are
inherited from ``experiments.run_revision_aos``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_revision_aos import load_instances, run_single


DEFAULT_VARIANTS = [
    "UniformFixed",
    "UCBOnly",
    "PPOOnly",
    "FixedUCBPPO",
    "RandomUCBPPO",
    "AdaptiveNoBC",
    "AdaptiveSAOS",
]

DEFAULT_BUDGETS = [50, 100, 200]


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def existing_keys(csv_path: Path) -> set[tuple[str, str, str, str, int, int]]:
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str, str, str, int, int]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(
                (
                    row["dataset"],
                    row["instance"],
                    row["variant"],
                    row["reward_scheme"],
                    int(row["seed"]),
                    int(row.get("Budget", row.get("max_gen", 0))),
                )
            )
    return keys


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(args: argparse.Namespace):
    instances = load_instances(limit=args.limit_instances)
    seeds = list(range(args.seed_start, args.seed_end))
    variants = parse_str_list(args.variants, DEFAULT_VARIANTS)
    budgets = parse_int_list(args.budgets, DEFAULT_BUDGETS)
    done = existing_keys(Path(args.result_file))

    tasks = []
    for budget in budgets:
        budget_dir = Path(args.out_dir) / f"gen{budget:03d}"
        for dataset, inst_name, instance in instances:
            for variant in variants:
                for seed in seeds:
                    key = (
                        dataset,
                        inst_name,
                        variant,
                        args.reward_scheme,
                        seed,
                        budget,
                    )
                    if key in done:
                        continue
                    task = (
                        dataset,
                        inst_name,
                        instance,
                        variant,
                        args.reward_scheme,
                        seed,
                        args.pop_size,
                        budget,
                        str(budget_dir),
                    )
                    tasks.append((budget, task))
    return tasks


def run_budget_task(payload):
    budget, task = payload
    row = run_single(task)
    row["Budget"] = int(budget)
    row["StressGroup"] = "budget_stage"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/revision/aos_budget_stress")
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--limit-instances", type=int, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--reward-scheme", default="composite")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.result_file is None:
        args.result_file = str(out_dir / "aos_budget_stress_runs.csv")

    tasks = build_tasks(args)
    total = len(tasks)
    variants = parse_str_list(args.variants, DEFAULT_VARIANTS)
    budgets = parse_int_list(args.budgets, DEFAULT_BUDGETS)
    print(
        "[aos-budget-stress] "
        f"tasks={total} budgets={budgets} variants={variants} "
        f"result={args.result_file}",
        flush=True,
    )
    if total == 0:
        print("[aos-budget-stress] nothing to do", flush=True)
        return

    start = time.time()
    completed = 0
    n_workers = max(1, min(args.workers, total, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(run_budget_task, task): task for task in tasks}
        for future in as_completed(future_map):
            budget, task = future_map[future]
            completed += 1
            try:
                row = future.result()
                append_row(Path(args.result_file), row)
                msg = (
                    f"[{completed}/{total}] gen={budget} "
                    f"{row['dataset']}/{row['instance']} {row['variant']} "
                    f"seed={row['seed']} HV={row['HV_local']:.3g} "
                    f"Cmax={row['Cmax_best']:.1f} NSol={row['NSol']} "
                    f"T={row['Time']:.1f}s"
                )
            except Exception as exc:
                _, raw_task = budget, task
                dataset, inst_name, _, variant, reward_scheme, seed, *_ = raw_task
                msg = (
                    f"[{completed}/{total}] ERROR gen={budget} "
                    f"{dataset}/{inst_name} {variant} {reward_scheme} "
                    f"seed={seed}: {exc!r}"
                )
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / max(rate, 1e-9)
            print(f"{msg} ETA={eta/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
