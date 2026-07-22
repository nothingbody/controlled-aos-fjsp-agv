"""Analyze the frozen E5/v7 cross-instance PPO experiment.

The script deliberately refuses partial output.  It validates the formal v7
completion chain, recomputes all quality indicators from the saved fronts, and
keeps offline pretraining cost separate from held-out deployment cost.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_cross_instance_pretraining_v7 import (  # noqa: E402
    FORMAL_BUDGETS,
    FORMAL_EVAL_ROWS,
    FORMAL_EVAL_SEEDS,
    FORMAL_POP_SIZE,
    FORMAL_PRETRAIN_ROWS,
    PROTOCOL,
    VARIANTS,
    file_sha256,
    front_semantic_hash,
    is_deduplicated_nondominated,
    verify_formal_complete,
)
from scripts.analyze_resubmission_v5 import (  # noqa: E402
    bootstrap_ci,
    holm_adjust,
    nondominated,
    rank_biserial_from_diff,
    stable_unique_rows,
)


DEFAULT_ROOT = ROOT / "results/resubmission/v7_cross_instance"
DEFAULT_OUT = DEFAULT_ROOT / "analysis"
REFERENCE_NAME = "frozen_reference_snapshot.pkl"
TIE_TOL = 1e-12
BOOTSTRAP_REPS = 10_000
SIGN_FLIP_REPS = 1_000_000

PRIMARY = (
    (100, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g100"),
    (100, "XPrePPO_Online_R16", "ScratchNoBC_R16", "online_minus_scratch_g100"),
)
SECONDARY = (
    (50, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g50"),
    (50, "XPrePPO_Online_R16", "ScratchNoBC_R16", "online_minus_scratch_g50"),
    (200, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g200"),
    (200, "XPrePPO_Online_R16", "ScratchNoBC_R16", "online_minus_scratch_g200"),
)
MECHANISM = tuple(
    (budget, "XPrePPO_Online_R16", "XPrePPO_Frozen", f"online_minus_frozen_g{budget}")
    for budget in FORMAL_BUDGETS
)


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_front(raw: object, experiment_root: Path) -> Path:
    value = Path(str(raw))
    candidates = (value, experiment_root / value, ROOT / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate v7 front: {raw}")


def load_snapshot(experiment_root: Path, run_manifest: dict) -> dict:
    path = experiment_root / REFERENCE_NAME
    if not path.is_file():
        raise RuntimeError(f"missing frozen reference snapshot: {path}")
    observed = file_sha256(path)
    expected = run_manifest.get("reference_snapshot_sha256")
    if observed != expected:
        raise RuntimeError(f"reference snapshot hash mismatch: {observed} != {expected}")
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    if snapshot.get("snapshot_protocol") != "saos_v7_fixed_v5_reference_20260722":
        raise RuntimeError("unexpected v7 reference-snapshot protocol")
    if len(snapshot.get("normalization", {})) != 150:
        raise RuntimeError("v7 snapshot must contain 150 normalization blocks")
    if len(snapshot.get("reference_sets", {})) != 150:
        raise RuntimeError("v7 snapshot must contain 150 IGD+ reference sets")
    return snapshot


def preflight(experiment_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    failures = [
        path for path in (
            experiment_root / "pretraining_failures.json",
            experiment_root / "evaluation_failures.json",
            experiment_root / "pipeline_failed.txt",
        ) if path.exists()
    ]
    if failures:
        raise RuntimeError(f"failure marker(s) prohibit analysis: {failures}")
    required = {
        "manifest": experiment_root / "run_manifest.json",
        "completion": experiment_root / "pipeline_complete.json",
        "pretraining": experiment_root / "pretraining_runs.csv",
        "evaluation": experiment_root / "runs.csv",
    }
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"formal v7 is incomplete; missing {missing}")
    manifest = load_json(required["manifest"])
    completion = load_json(required["completion"])
    if manifest.get("protocol") != PROTOCOL or completion.get("protocol") != PROTOCOL:
        raise RuntimeError("v7 protocol mismatch")
    hashes = {
        key: manifest[key]
        for key in (
            "code_hash", "design_hash", "input_hash", "split_hash",
            "reference_snapshot_sha256",
        )
    }
    verified = verify_formal_complete(experiment_root, hashes)
    for key, value in verified.items():
        if key in completion and completion[key] != value:
            raise RuntimeError(
                f"completion marker differs from fresh verification for {key}: "
                f"{completion[key]} != {value}"
            )
    pretraining = pd.read_csv(required["pretraining"])
    evaluation = pd.read_csv(required["evaluation"])
    if len(pretraining) != FORMAL_PRETRAIN_ROWS or len(evaluation) != FORMAL_EVAL_ROWS:
        raise RuntimeError(
            f"row-count mismatch: pretraining={len(pretraining)}, evaluation={len(evaluation)}"
        )
    pre_key = ["Fold", "Replica", "Pass", "Position", "dataset", "instance"]
    eval_key = ["Fold", "dataset", "instance", "variant", "Budget", "seed", "Replica"]
    if pretraining.duplicated(pre_key).any() or evaluation.duplicated(eval_key).any():
        raise RuntimeError("duplicate v7 primary keys")
    if set(evaluation["variant"].astype(str)) != set(VARIANTS):
        raise RuntimeError("evaluation controller grid mismatch")
    if set(pd.to_numeric(evaluation["Budget"]).astype(int)) != set(FORMAL_BUDGETS):
        raise RuntimeError("evaluation budget grid mismatch")
    if set(pd.to_numeric(evaluation["seed"]).astype(int)) != set(FORMAL_EVAL_SEEDS):
        raise RuntimeError("evaluation seed grid mismatch")
    if set(pd.to_numeric(evaluation["Worker_count"]).astype(int)) != {40}:
        raise RuntimeError("formal v7 must record 40 workers")
    if set(pd.to_numeric(evaluation["Population_size"]).astype(int)) != {FORMAL_POP_SIZE}:
        raise RuntimeError("formal v7 population-size mismatch")
    for column, key in (
        ("Code_hash", "code_hash"), ("Design_hash", "design_hash"),
        ("Input_hash", "input_hash"), ("Split_hash", "split_hash"),
        ("Reference_snapshot_sha256", "reference_snapshot_sha256"),
    ):
        if set(evaluation[column].astype(str)) != {manifest[key]}:
            raise RuntimeError(f"evaluation {column} differs from run manifest")
        if set(pretraining[column].astype(str)) != {manifest[key]}:
            raise RuntimeError(f"pretraining {column} differs from run manifest")
    snapshot = load_snapshot(experiment_root, manifest)
    return pretraining, evaluation, manifest, snapshot


def normalize(points: np.ndarray, record: dict) -> np.ndarray:
    ideal = np.asarray(record["ideal"], dtype=float)
    nadir = np.asarray(record["nadir"], dtype=float)
    scale = np.where(nadir > ideal, nadir - ideal, 1.0)
    return (np.asarray(points, dtype=float) - ideal) / scale


def igd_plus(reference: np.ndarray, approximation: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    distances = []
    for target in reference:
        deviation = np.maximum(approximation - target, 0.0)
        distances.append(np.linalg.norm(deviation, axis=1).min())
    return float(np.mean(distances))


def load_front(row: dict, experiment_root: Path) -> tuple[np.ndarray, Path]:
    path = locate_front(row["front_pickle"], experiment_root)
    if file_sha256(path) != str(row["Front_sha256"]):
        raise RuntimeError(f"front file hash mismatch: {path}")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    checks = {
        "protocol": (payload.get("protocol"), PROTOCOL),
        "fold": (int(payload.get("fold", -1)), int(row["Fold"])),
        "dataset": (payload.get("dataset"), row["dataset"]),
        "instance": (payload.get("instance"), row["instance"]),
        "variant": (payload.get("variant"), row["variant"]),
        "budget": (int(payload.get("budget", -1)), int(row["Budget"])),
        "seed": (int(payload.get("seed", -1)), int(row["seed"])),
        "replica": (int(payload.get("replica", -2)), int(row["Replica"])),
        "code_hash": (payload.get("code_hash"), row["Code_hash"]),
        "design_hash": (payload.get("design_hash"), row["Design_hash"]),
        "input_hash": (payload.get("input_hash"), row["Input_hash"]),
        "split_hash": (payload.get("split_hash"), row["Split_hash"]),
        "reference_snapshot_sha256": (
            payload.get("reference_snapshot_sha256"),
            row["Reference_snapshot_sha256"],
        ),
    }
    mismatch = {
        key: {"payload": observed, "csv": expected}
        for key, (observed, expected) in checks.items() if observed != expected
    }
    if mismatch:
        raise RuntimeError(f"front metadata mismatch in {path}: {mismatch}")
    points = np.asarray(payload.get("objectives", []), dtype=float)
    if (
        points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
        or not np.isfinite(points).all()
        or not is_deduplicated_nondominated(points)
        or front_semantic_hash(points) != str(row["Front_semantic_sha256"])
    ):
        raise RuntimeError(f"front scientific audit failed: {path}")
    return points, path


def add_indicators(
    frame: pd.DataFrame, experiment_root: Path, snapshot: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    audits: list[dict] = []
    for row in frame.to_dict(orient="records"):
        key = (row["dataset"], row["instance"], int(row["Budget"]))
        record = snapshot["normalization"][key]
        reference_set = np.asarray(snapshot["reference_sets"][key], dtype=float)
        points, path = load_front(row, experiment_root)
        projected = nondominated(stable_unique_rows(normalize(points, record)))
        fixed_reference = np.asarray(record["reference"], dtype=float)
        inside = np.all(projected <= fixed_reference + TIE_TOL, axis=1)
        fixed_points = projected[inside]
        expanded_reference = np.full(3, 1.5, dtype=float)
        expanded_valid = bool(np.all(projected <= expanded_reference + TIE_TOL))
        row["HV_fixed"] = (
            float(HV(ref_point=fixed_reference)(nondominated(fixed_points)))
            if len(fixed_points) else np.nan
        )
        row["HV_expanded_1p5"] = (
            float(HV(ref_point=expanded_reference)(projected))
            if expanded_valid else np.nan
        )
        row["IGDplus_fixed"] = igd_plus(reference_set, projected)
        rows.append(row)
        above = projected > fixed_reference + TIE_TOL
        audits.append({
            "Fold": int(row["Fold"]), "dataset": row["dataset"],
            "instance": row["instance"], "variant": row["variant"],
            "Budget": int(row["Budget"]), "seed": int(row["seed"]),
            "Replica": int(row["Replica"]), "front_points": int(len(projected)),
            "fixed_inside_points": int(inside.sum()),
            "fixed_excluded_points": int((~inside).sum()),
            "fixed_invalid": bool(not inside.any()),
            "above_fixed_cmax": int(above[:, 0].sum()),
            "above_fixed_energy": int(above[:, 1].sum()),
            "above_fixed_workload": int(above[:, 2].sum()),
            "expanded_1p5_invalid": bool(not expanded_valid),
            "normalized_min": float(projected.min()),
            "normalized_max": float(projected.max()),
            "front_path": str(path),
        })
    enriched = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if enriched["HV_fixed"].isna().any():
        invalid = enriched.loc[
            enriched["HV_fixed"].isna(),
            ["dataset", "instance", "variant", "Budget", "seed"],
        ]
        raise RuntimeError(f"confirmatory fixed-box HV invalid runs: {invalid.head().to_dict('records')}")
    return enriched, audit


def instance_medians(enriched: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "HV_fixed", "HV_expanded_1p5", "IGDplus_fixed", "Cmax_best",
        "TEC_best", "WB_best", "NSol", "elapsed_seconds", "cpu_seconds",
        "learning_time", "Initial_evaluations", "Offspring_evaluations",
        "Online_PPO_samples", "Online_PPO_collected_transitions",
        "Online_PPO_discarded_singletons", "PPO_controlled_actions",
        "Online_PPO_updates", "Online_PPO_optimizer_steps", "Transition_gen",
        "Operator_entropy", "Policy_parameter_L2_drift",
    ]
    metrics = [name for name in metrics if name in enriched.columns]
    return (
        enriched.groupby(
            ["Fold", "dataset", "instance", "Budget", "variant"], sort=True
        )[metrics].median().reset_index()
    )


def signed_rank_randomization(
    difference: np.ndarray, *, reps: int = SIGN_FLIP_REPS, seed: int = 20260722,
) -> tuple[float, float, int, str]:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > TIE_TOL)]
    if len(values) == 0:
        return 0.0, 1.0, 0, "all ties"
    ranks = rankdata(np.abs(values), method="average")
    total = float(ranks.sum())
    positive = float(ranks[values > 0].sum())
    statistic = min(positive, total - positive)
    deviation = abs(positive - total / 2.0)
    if len(values) <= 20:
        extreme = 0
        permutations = 2 ** len(values)
        for signs in itertools.product((False, True), repeat=len(values)):
            signed_positive = float(ranks[np.asarray(signs, dtype=bool)].sum())
            extreme += abs(signed_positive - total / 2.0) >= deviation - TIE_TOL
        return statistic, float(extreme / permutations), len(values), "exact sign enumeration"
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk = 20_000
    while completed < reps:
        size = min(chunk, reps - completed)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8)
        signed_positive = signs @ ranks
        extreme += int(np.sum(np.abs(signed_positive - total / 2.0) >= deviation - TIE_TOL))
        completed += size
    p_value = (extreme + 1.0) / (reps + 1.0)
    return statistic, float(p_value), len(values), f"fixed-seed sign flip ({reps} resamples)"


def hierarchical_fold_bootstrap(
    block: pd.DataFrame, *, reps: int = BOOTSTRAP_REPS, seed: int = 20260722,
) -> tuple[float, float]:
    required = {"Fold", "delta"}
    if required - set(block.columns):
        raise ValueError(f"hierarchical bootstrap requires {required}")
    grouped = {
        int(fold): values["delta"].to_numpy(dtype=float)
        for fold, values in block.groupby("Fold", sort=True)
    }
    folds = np.asarray(sorted(grouped), dtype=int)
    if len(folds) != 5 or any(len(grouped[int(fold)]) != 10 for fold in folds):
        raise ValueError("formal hierarchical bootstrap requires five folds of ten instances")
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for replicate in range(reps):
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        sampled_values = []
        for fold in sampled_folds:
            values = grouped[int(fold)]
            sampled_values.append(rng.choice(values, size=len(values), replace=True))
        estimates[replicate] = float(np.median(np.concatenate(sampled_values)))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def contrast_block(
    medians: pd.DataFrame, budget: int, lhs: str, rhs: str, metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    subset = medians[pd.to_numeric(medians["Budget"]).astype(int) == int(budget)]
    pivot = subset.pivot(
        index=["Fold", "dataset", "instance"], columns="variant", values=metric
    ).reset_index()
    if lhs not in pivot or rhs not in pivot:
        raise RuntimeError(f"missing contrast columns for {lhs} vs {rhs} at g={budget}")
    orientation = 1.0 if higher_is_better else -1.0
    pivot["delta"] = orientation * (pivot[lhs] - pivot[rhs])
    if len(pivot) != 50 or pivot["delta"].isna().any():
        raise RuntimeError(f"contrast block is incomplete for {lhs} vs {rhs} at g={budget}")
    return pivot


def comparison_record(
    medians: pd.DataFrame, *, family: str, budget: int, lhs: str, rhs: str,
    contrast: str, metric: str = "HV_fixed", higher_is_better: bool = True,
    bootstrap_reps: int = BOOTSTRAP_REPS, sign_flip_reps: int = SIGN_FLIP_REPS,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    block = contrast_block(medians, budget, lhs, rhs, metric, higher_is_better)
    delta = block["delta"].to_numpy(dtype=float)
    seed = deterministic_seed(PROTOCOL, family, contrast, metric)
    statistic, p_raw, nonzero, test_method = signed_rank_randomization(
        delta, reps=sign_flip_reps, seed=seed
    )
    ci_low, ci_high = bootstrap_ci(
        delta, reps=bootstrap_reps, seed=seed, statistic=np.median
    )
    h_low, h_high = hierarchical_fold_bootstrap(
        block[["Fold", "delta"]], reps=bootstrap_reps, seed=seed + 1
    )
    leave_rows = []
    for omitted in sorted(block["Fold"].unique()):
        value = float(np.median(block.loc[block["Fold"] != omitted, "delta"]))
        leave_rows.append({
            "family": family, "contrast": contrast, "metric": metric,
            "Budget": int(budget), "omitted_fold": int(omitted),
            "median_delta_oriented": value,
        })
    leave = pd.DataFrame(leave_rows)
    record = {
        "family": family, "contrast": contrast, "metric": metric,
        "Budget": int(budget), "lhs": lhs, "rhs": rhs,
        "orientation": "positive favors lhs", "n_instances": int(len(delta)),
        "median_delta_oriented": float(np.median(delta)),
        "instance_bootstrap_ci95_low": ci_low,
        "instance_bootstrap_ci95_high": ci_high,
        "hierarchical_fold_bootstrap_ci95_low": h_low,
        "hierarchical_fold_bootstrap_ci95_high": h_high,
        "wins": int(np.sum(delta > TIE_TOL)),
        "ties": int(np.sum(np.abs(delta) <= TIE_TOL)),
        "losses": int(np.sum(delta < -TIE_TOL)),
        "rank_biserial": rank_biserial_from_diff(delta, TIE_TOL),
        "signed_rank_statistic": statistic, "p_raw": p_raw,
        "n_nonzero": int(nonzero), "test_method": test_method,
        "leave_one_fold_out_min": float(leave["median_delta_oriented"].min()),
        "leave_one_fold_out_max": float(leave["median_delta_oriented"].max()),
    }
    block = block.assign(family=family, contrast=contrast, metric=metric, Budget=budget)
    return record, block, leave


def inferential_analysis(
    medians: pd.DataFrame, *, bootstrap_reps: int, sign_flip_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    blocks = []
    leave = []
    for family, definitions in (
        ("primary_g100", PRIMARY),
        ("secondary_g50_g200", SECONDARY),
        ("mechanism_online_frozen", MECHANISM),
    ):
        for budget, lhs, rhs, contrast in definitions:
            record, block, omitted = comparison_record(
                medians, family=family, budget=budget, lhs=lhs, rhs=rhs,
                contrast=contrast, bootstrap_reps=bootstrap_reps,
                sign_flip_reps=sign_flip_reps,
            )
            records.append(record); blocks.append(block); leave.append(omitted)
    table = pd.DataFrame(records)
    table["p_holm_within_family"] = np.nan
    for family, indices in table.groupby("family").groups.items():
        table.loc[list(indices), "p_holm_within_family"] = holm_adjust(
            table.loc[list(indices), "p_raw"].to_numpy(dtype=float)
        )
    table["instance_holm_significant"] = table["p_holm_within_family"] < 0.05
    table["superiority_gate"] = (
        (table["median_delta_oriented"] > 0)
        & table["instance_holm_significant"]
        & (table["hierarchical_fold_bootstrap_ci95_low"] > 0)
        & (table["leave_one_fold_out_min"] > 0)
    )
    return table, pd.concat(blocks, ignore_index=True), pd.concat(leave, ignore_index=True)


def fold_replica_summaries(enriched: pd.DataFrame) -> pd.DataFrame:
    raw = enriched.copy()
    raw["PairReplica"] = (pd.to_numeric(raw["seed"]).astype(int) - 42) % 5
    keys = ["Fold", "dataset", "instance", "Budget", "seed", "PairReplica"]
    pivot = raw.pivot(index=keys, columns="variant", values="HV_fixed").reset_index()
    definitions = list(PRIMARY) + list(SECONDARY) + list(MECHANISM)
    records = []
    for budget, lhs, rhs, contrast in definitions:
        block = pivot[pivot["Budget"] == budget].copy()
        block["delta"] = block[lhs] - block[rhs]
        for (fold, replica), values in block.groupby(["Fold", "PairReplica"], sort=True):
            records.append({
                "contrast": contrast, "Budget": int(budget), "Fold": int(fold),
                "Replica": int(replica), "n_instance_seed_pairs": int(len(values)),
                "median_delta": float(values["delta"].median()),
                "mean_delta": float(values["delta"].mean()),
            })
    return pd.DataFrame(records).drop_duplicates(
        ["contrast", "Budget", "Fold", "Replica"]
    )


def pretraining_chain_cost(pretraining: pd.DataFrame) -> pd.DataFrame:
    ordered = pretraining.sort_values(["Fold", "Replica", "Pass", "Position"])
    terminal = ordered.groupby(["Fold", "Replica"], sort=True).tail(1).copy()
    if len(terminal) != 25:
        raise RuntimeError(f"expected 25 terminal pretraining rows, observed {len(terminal)}")
    keep = [
        "Fold", "Replica", "Pretrain_seed", "Cumulative_objective_evaluations",
        "Cumulative_PPO_samples", "Cumulative_PPO_collected_transitions",
        "Cumulative_PPO_discarded_singletons", "Cumulative_PPO_updates",
        "Cumulative_PPO_optimizer_steps", "Cumulative_CPU_seconds",
        "Cumulative_wall_seconds",
    ]
    terminal = terminal[keep].reset_index(drop=True)
    terminal["Episodes"] = 80
    terminal["Generations"] = 80 * 200
    return terminal


def online_data_economy(enriched: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "Initial_evaluations", "Offspring_evaluations", "Online_PPO_samples",
        "Online_PPO_collected_transitions", "Online_PPO_discarded_singletons",
        "PPO_controlled_actions", "Online_PPO_updates", "Online_PPO_optimizer_steps",
        "learning_time", "elapsed_seconds", "cpu_seconds", "Offline_checkpoint_bytes",
    ]
    fields = [field for field in fields if field in enriched.columns]
    summary = enriched.groupby(["variant", "Budget"], sort=True)[fields].median().reset_index()
    if {"Online_PPO_samples", "PPO_controlled_actions"} <= set(summary.columns):
        denominator = summary["PPO_controlled_actions"].replace(0, np.nan)
        summary["consumed_transition_fraction"] = summary["Online_PPO_samples"] / denominator
        summary["updates_per_100_controlled_actions"] = (
            100.0 * summary["Online_PPO_updates"] / denominator
        )
    return summary


def amortization_tables(
    enriched: pd.DataFrame, chain_cost: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offline_cpu = float(chain_cost["Cumulative_CPU_seconds"].median())
    offline_evaluations = float(chain_cost["Cumulative_objective_evaluations"].median())
    offline_samples = float(chain_cost["Cumulative_PPO_samples"].median())
    online = online_data_economy(enriched)
    rows = []
    for budget in FORMAL_BUDGETS:
        block = online[online["Budget"] == budget].set_index("variant")
        for variant in ("XPrePPO_Frozen", "XPrePPO_Online_R16"):
            online_eval = float(
                block.loc[variant, "Initial_evaluations"]
                + block.loc[variant, "Offspring_evaluations"]
            )
            online_cpu = float(block.loc[variant, "cpu_seconds"])
            for deployments in (1, 10, 50, 60, 100, 1000):
                rows.append({
                    "Budget": int(budget), "variant": variant,
                    "deployments_per_checkpoint": int(deployments),
                    "offline_objective_evaluations_per_deployment": offline_evaluations / deployments,
                    "offline_PPO_samples_per_deployment": offline_samples / deployments,
                    "offline_CPU_seconds_per_deployment": offline_cpu / deployments,
                    "online_objective_evaluations": online_eval,
                    "online_CPU_seconds": online_cpu,
                    "total_objective_evaluations_amortized": online_eval + offline_evaluations / deployments,
                    "total_CPU_seconds_amortized": online_cpu + offline_cpu / deployments,
                })
    break_even = []
    for budget in FORMAL_BUDGETS:
        block = online[online["Budget"] == budget].set_index("variant")
        target_cpu = float(block.loc["XPrePPO_Online_R16", "cpu_seconds"])
        for comparator in ("UCBOnly", "ScratchNoBC_R16"):
            comparator_cpu = float(block.loc[comparator, "cpu_seconds"])
            online_saving = comparator_cpu - target_cpu
            deployments = offline_cpu / online_saving if online_saving > 0 else math.nan
            break_even.append({
                "Budget": int(budget), "target": "XPrePPO_Online_R16",
                "comparator": comparator, "median_offline_CPU_seconds": offline_cpu,
                "target_online_CPU_seconds": target_cpu,
                "comparator_online_CPU_seconds": comparator_cpu,
                "online_CPU_saving_per_deployment": online_saving,
                "CPU_break_even_deployments": deployments,
                "objective_evaluation_break_even": math.nan,
                "reason": (
                    "defined from positive online CPU saving"
                    if online_saving > 0 else
                    "undefined: pretrained PPO has no positive online CPU saving"
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(break_even)


def performance_summary(medians: pd.DataFrame) -> pd.DataFrame:
    fields = ["HV_fixed", "HV_expanded_1p5", "IGDplus_fixed"]
    records = []
    for (variant, budget), block in medians.groupby(["variant", "Budget"], sort=True):
        row = {"variant": variant, "Budget": int(budget), "n_instances": int(len(block))}
        for field in fields:
            values = block[field].dropna().to_numpy(dtype=float)
            row[f"{field}_median"] = float(np.median(values)) if len(values) else math.nan
            row[f"{field}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{field}_valid_instances"] = int(len(values))
        records.append(row)
    return pd.DataFrame(records)


def plot_results(
    comparisons: pd.DataFrame, data_economy: pd.DataFrame, out_dir: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    selected = comparisons[comparisons["contrast"].str.contains("online_minus_(ucb|scratch)", regex=True)].copy()
    selected = selected.sort_values(["Budget", "rhs"])
    labels = [
        f"g{int(row.Budget)}\nvs {'UCB' if row.rhs == 'UCBOnly' else 'scratch'}"
        for row in selected.itertuples(index=False)
    ]
    x = np.arange(len(selected))
    values = selected["median_delta_oriented"].to_numpy(dtype=float)
    low = selected["hierarchical_fold_bootstrap_ci95_low"].to_numpy(dtype=float)
    high = selected["hierarchical_fold_bootstrap_ci95_high"].to_numpy(dtype=float)
    axes[0].errorbar(
        x, values, yerr=[values - low, high - values], fmt="o", color="#2F5597",
        ecolor="#7F8FA6", capsize=3,
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Median paired HV difference")
    axes[0].set_title("(a) Held-out effects; hierarchical 95% intervals")

    order = ["ScratchNoBC_R16", "XPrePPO_Frozen", "XPrePPO_Online_R16"]
    colors = ["#8C8C8C", "#4C956C", "#D9822B"]
    for variant, color in zip(order, colors):
        block = data_economy[data_economy["variant"] == variant]
        axes[1].plot(
            block["Budget"], block["Online_PPO_samples"], marker="o",
            label=variant.replace("XPrePPO_", "pretrained ").replace("ScratchNoBC_R16", "scratch"),
            color=color,
        )
    axes[1].set_xlabel("Generation budget")
    axes[1].set_ylabel("Median consumed online PPO transitions")
    axes[1].set_title("(b) Realized online data supply")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_instance_effects_and_data_economy.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cross_instance_effects_and_data_economy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--sign-flip-reps", type=int, default=SIGN_FLIP_REPS)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0 or args.sign_flip_reps <= 0:
        raise SystemExit("resample counts must be positive")
    experiment_root = args.experiment_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pretraining, evaluation, manifest, snapshot = preflight(experiment_root)
    enriched, reference_audit = add_indicators(evaluation, experiment_root, snapshot)
    medians = instance_medians(enriched)
    comparisons, contrast_blocks, leave_one_fold_out = inferential_analysis(
        medians, bootstrap_reps=args.bootstrap_reps,
        sign_flip_reps=args.sign_flip_reps,
    )
    fold_replica = fold_replica_summaries(enriched)
    chain_cost = pretraining_chain_cost(pretraining)
    data_economy = online_data_economy(enriched)
    amortization, break_even = amortization_tables(enriched, chain_cost)
    summary = performance_summary(medians)

    outputs = {
        "runs_with_fixed_indicators.csv": enriched,
        "reference_orthant_audit.csv": reference_audit,
        "instance_seed_medians.csv": medians,
        "prespecified_comparisons.csv": comparisons,
        "contrast_instance_blocks.csv": contrast_blocks,
        "leave_one_fold_out.csv": leave_one_fold_out,
        "fold_replica_summaries.csv": fold_replica,
        "pretraining_chain_cost.csv": chain_cost,
        "online_data_economy.csv": data_economy,
        "offline_cost_amortization.csv": amortization,
        "cpu_break_even.csv": break_even,
        "performance_summary.csv": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    plot_results(comparisons, data_economy, args.out_dir)

    analysis_manifest = {
        "protocol": PROTOCOL,
        "analysis_role": "prespecified E5/v7 held-out and data-economy analysis",
        "run_manifest_sha256": file_sha256(experiment_root / "run_manifest.json"),
        "pipeline_complete_sha256": file_sha256(experiment_root / "pipeline_complete.json"),
        "reference_snapshot_sha256": manifest["reference_snapshot_sha256"],
        "pretraining_rows": int(len(pretraining)),
        "evaluation_rows": int(len(evaluation)),
        "instance_blocks": int(medians[["Fold", "dataset", "instance"]].drop_duplicates().shape[0]),
        "bootstrap_reps": int(args.bootstrap_reps),
        "sign_flip_reps": int(args.sign_flip_reps),
        "fixed_reference_invalid_runs": int(reference_audit["fixed_invalid"].sum()),
        "fixed_reference_excluded_points": int(reference_audit["fixed_excluded_points"].sum()),
        "expanded_reference_invalid_runs": int(reference_audit["expanded_1p5_invalid"].sum()),
        "superiority_gates_passed": comparisons.loc[
            comparisons["superiority_gate"], "contrast"
        ].tolist(),
        "outputs": {name: file_sha256(args.out_dir / name) for name in outputs},
    }
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis_manifest, indent=2))


if __name__ == "__main__":
    main()
