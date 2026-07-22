"""Audit the calibrated v5 layout and energy construct before a full rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_benchmark_set
from src.algorithm.nsga3.encoding import random_chromosome
from src.problem.energy_model import compute_agv_energy, compute_machine_energy


DATASETS = {
    "Brandimarte": "data/benchmarks/brandimarte",
    "Hurink_edata": "data/benchmarks/hurink_edata",
}


def finite_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def audit_instance(dataset: str, filename: str, instance, samples: int,
                   base_seed: int) -> dict:
    seed_sequence = np.random.SeedSequence(
        [int(base_seed), int(instance.extension_seed)]
    )
    rng_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    rng = np.random.RandomState(rng_seed)
    makespans = []
    energies = []
    agv_shares = []

    for _ in range(samples):
        chromosome = random_chromosome(instance, rng)
        objectives = chromosome.objectives
        machine = compute_machine_energy(
            instance, chromosome.schedule, objectives.makespan
        )
        agv = compute_agv_energy(instance, chromosome.schedule)
        machine_total = sum(item["total"] for item in machine.values())
        agv_total = sum(item["total"] for item in agv.values())
        total = machine_total + agv_total
        makespans.append(objectives.makespan)
        energies.append(total)
        agv_shares.append(agv_total / total if total > 0 else 0.0)

    one_unit_loaded = [
        instance.transport_energy(1.0, speed, loaded=True)
        for speed in instance.speed_levels
    ]
    metadata = instance.extension_metadata
    return {
        "dataset": dataset,
        "instance": filename,
        "samples": int(samples),
        "transport_time_ratio_target": metadata["transport_time_ratio_target"],
        "transport_time_ratio_realized": metadata["transport_time_ratio_realized"],
        "agv_energy_share_median": float(np.median(agv_shares)),
        "agv_energy_share_q25": float(np.quantile(agv_shares, 0.25)),
        "agv_energy_share_q75": float(np.quantile(agv_shares, 0.75)),
        "cmax_energy_correlation": finite_correlation(
            np.asarray(makespans), np.asarray(energies)
        ),
        "loaded_energy_per_distance_low": float(one_unit_loaded[0]),
        "loaded_energy_per_distance_mid": float(one_unit_loaded[1]),
        "loaded_energy_per_distance_high": float(one_unit_loaded[2]),
        "speed_energy_strictly_increasing": bool(
            one_unit_loaded[0] < one_unit_loaded[1] < one_unit_loaded[2]
        ),
        "all_objectives_finite": bool(
            np.isfinite(makespans).all() and np.isfinite(energies).all()
        ),
    }


def quantiles(values) -> dict:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if data.size == 0:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "min": float(np.min(data)),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "q75": float(np.quantile(data, 0.75)),
        "max": float(np.max(data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/resubmission/v5/energy_audit"),
    )
    args = parser.parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3")

    records = []
    for dataset, directory in DATASETS.items():
        for filename, instance in load_benchmark_set(directory, num_agv=3):
            records.append(
                audit_instance(dataset, filename, instance, args.samples, args.seed)
            )

    frame = pd.DataFrame(records)
    if len(frame) != 50:
        raise SystemExit(f"expected 50 instances, found {len(frame)}")
    summary = {
        "protocol": "saos_bc_onpolicy_ppo_v5_20260720",
        "instance_count": int(len(frame)),
        "samples_per_instance": int(args.samples),
        "transport_time_ratio": quantiles(frame["transport_time_ratio_realized"]),
        "agv_energy_share": quantiles(frame["agv_energy_share_median"]),
        "within_instance_cmax_energy_correlation": quantiles(
            frame["cmax_energy_correlation"]
        ),
        "all_speed_energy_orderings_valid": bool(
            frame["speed_energy_strictly_increasing"].all()
        ),
        "all_objectives_finite": bool(frame["all_objectives_finite"].all()),
        "interpretation": (
            "Construct diagnostics from random feasible schedules; these are a "
            "pre-run validity audit, not algorithm-performance results."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "energy_construct_by_instance.csv", index=False)
    (args.out_dir / "energy_construct_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
