"""Strict instance-blocked analysis for the SA-AOS v5 resubmission protocol.

The primary inferential unit is an instance.  Seed-level results are retained as
supplementary diagnostics, but seeds are first collapsed with a median inside
each instance/method/budget cell for all main tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithm.nsga3.selection import non_dominated_sort


PROTOCOL = "saos_bc_onpolicy_ppo_v5_20260720"
HV_OBJECTIVE_COLUMNS = (0, 1, 2)  # makespan, total energy, workload imbalance
HV_REFERENCE = np.array([1.1, 1.1, 1.1], dtype=float)
# This is a numerical equality threshold, not a claim of practical equivalence.
DEFAULT_TIE_TOLERANCE = 1e-12
DEFAULT_BOOTSTRAP_SEED = 20260720
E1_METHODS = [
    "Random",
    "UniformFixed",
    "ProbabilityMatching",
    "AdaptivePursuit",
    "UCBOnly",
    "PPOOnly",
    "FixedUCBPPO",
    "RandomUCBPPO",
    "AdaptiveNoBC",
    "AdaptiveSAOS",
]
E2_METHODS = [
    "R1_survival_only",
    "R2_hv_only",
    "R3_cmax_only",
    "R4_survival_hv",
    "R5_composite",
    "R6_adaptive_weight",
]
E3_METHODS = [
    "UniformFixed",
    "UCBOnly",
    "PPOOnly",
    "FixedUCBPPO",
    "RandomUCBPPO",
    "AdaptiveNoBC",
    "AdaptiveSAOS",
]
E3_BUDGETS = ["50", "100", "200"]


def stable_unique_rows(points: np.ndarray) -> np.ndarray:
    """Remove exact duplicate objective vectors while preserving first order."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"objective matrix must be N x M with M>=2, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("objective matrix contains non-finite values")
    if len(points) <= 1:
        return points.copy()
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]


def nondominated(points: np.ndarray) -> np.ndarray:
    points = stable_unique_rows(points)
    if len(points) <= 1:
        return points
    return points[non_dominated_sort(points)[0]]


def locate_front(raw_path: str, csv_path: Path, search_root: Path) -> Path:
    candidate = Path(str(raw_path))
    attempts = [candidate, csv_path.parent / candidate, search_root / candidate]
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    matches = list(search_root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous front path {raw_path}: {len(matches)} matches")
    raise FileNotFoundError(raw_path)


def load_front(path: Path) -> tuple[np.ndarray, int]:
    """Load a front and return de-duplicated objectives plus raw row count."""
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "objectives" not in payload:
        raise ValueError(f"front payload lacks 'objectives': {path}")
    raw = np.asarray(payload["objectives"], dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"unexpected objective shape {raw.shape} in {path}")
    if len(raw) == 0:
        raise ValueError(f"empty objective front in {path}")
    unique = stable_unique_rows(raw)
    return unique, int(len(raw))


def _budget_columns(frame: pd.DataFrame) -> list[str]:
    return ["Budget"] if "Budget" in frame.columns else []


def _analysis_cell_columns(frame: pd.DataFrame) -> list[str]:
    return ["dataset", "instance", *_budget_columns(frame), "variant"]


def _canonical_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _budget_label(value) -> str:
    if pd.isna(value):
        raise RuntimeError("Budget contains a missing value")
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def validate_completeness(
    frame: pd.DataFrame,
    *,
    expected_seed_count: int | None = 10,
    expected_seed_values: Sequence[int] | None = None,
    expected_methods: Sequence[str] | None = None,
    expected_budgets: Sequence[str] | None = None,
    expected_instance_count: int | None = None,
) -> dict:
    """Validate protocol, primary keys, and the complete factorial run grid."""
    required = {
        "Protocol",
        "dataset",
        "instance",
        "variant",
        "reward_scheme",
        "seed",
        "front_pickle",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"missing required columns: {missing_columns}")
    if frame.empty:
        raise RuntimeError("result CSV is empty")

    protocols = sorted(frame["Protocol"].astype(str).unique().tolist())
    if protocols != [PROTOCOL]:
        raise RuntimeError(f"protocol mismatch: {protocols}; expected only {PROTOCOL}")

    key_cols = ["dataset", "instance", "variant", "reward_scheme", "seed"]
    key_cols += _budget_columns(frame)
    duplicate_mask = frame.duplicated(key_cols, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, key_cols].head(5).to_dict(orient="records")
        raise RuntimeError(
            f"duplicate primary keys: {int(duplicate_mask.sum())} rows; examples={examples}"
        )

    seeds_numeric = pd.to_numeric(frame["seed"], errors="raise")
    if not np.all(np.equal(seeds_numeric, np.floor(seeds_numeric))):
        raise RuntimeError("seed values must be integers")
    observed_seeds = sorted(int(value) for value in seeds_numeric.unique())
    if expected_seed_values is not None:
        wanted_seeds = sorted(int(value) for value in expected_seed_values)
        if observed_seeds != wanted_seeds:
            raise RuntimeError(
                f"seed grid mismatch: observed={observed_seeds}, expected={wanted_seeds}"
            )
    else:
        if expected_seed_count is not None and len(observed_seeds) != expected_seed_count:
            raise RuntimeError(
                f"expected {expected_seed_count} distinct seeds, found {len(observed_seeds)}: "
                f"{observed_seeds}"
            )
        wanted_seeds = observed_seeds

    observed_methods = sorted(frame["variant"].astype(str).unique().tolist())
    wanted_methods = sorted(expected_methods) if expected_methods is not None else observed_methods
    if observed_methods != wanted_methods:
        raise RuntimeError(
            f"method grid mismatch: observed={observed_methods}, expected={wanted_methods}"
        )

    instance_frame = frame[["dataset", "instance"]].drop_duplicates()
    instances = [tuple(row) for row in instance_frame.itertuples(index=False, name=None)]
    if expected_instance_count is not None and len(instances) != expected_instance_count:
        raise RuntimeError(
            f"expected {expected_instance_count} instances, found {len(instances)}"
        )

    has_budget = "Budget" in frame.columns
    if has_budget:
        observed_budget_labels = sorted({_budget_label(value) for value in frame["Budget"]})
        wanted_budget_labels = (
            sorted(str(value) for value in expected_budgets)
            if expected_budgets is not None
            else observed_budget_labels
        )
        if observed_budget_labels != wanted_budget_labels:
            raise RuntimeError(
                "budget grid mismatch: "
                f"observed={observed_budget_labels}, expected={wanted_budget_labels}"
            )
    else:
        if expected_budgets not in (None, [], ["default"]):
            raise RuntimeError("expected budgets were supplied but the CSV has no Budget column")
        observed_budget_labels = ["default"]
        wanted_budget_labels = ["default"]

    # A main analysis unit must have exactly one row per seed.  This catches a
    # method accidentally emitted under multiple reward labels even when the
    # more detailed primary key remains unique.
    analysis_key = ["dataset", "instance", *_budget_columns(frame), "variant", "seed"]
    analysis_duplicates = frame.duplicated(analysis_key, keep=False)
    if analysis_duplicates.any():
        examples = frame.loc[analysis_duplicates, analysis_key].head(5).to_dict(orient="records")
        raise RuntimeError(f"duplicate analysis units: {examples}")

    missing_units = []
    seed_set = set(wanted_seeds)
    for dataset, instance in instances:
        instance_rows = frame[
            (frame["dataset"] == dataset) & (frame["instance"] == instance)
        ]
        for budget_label in wanted_budget_labels:
            if has_budget:
                budget_rows = instance_rows[
                    instance_rows["Budget"].map(_budget_label) == budget_label
                ]
            else:
                budget_rows = instance_rows
            for method in wanted_methods:
                unit = budget_rows[budget_rows["variant"].astype(str) == method]
                actual = set(int(value) for value in unit["seed"].tolist())
                if actual != seed_set or len(unit) != len(seed_set):
                    missing_units.append(
                        {
                            "dataset": dataset,
                            "instance": instance,
                            "Budget": budget_label,
                            "variant": method,
                            "missing_seeds": sorted(seed_set - actual),
                            "unexpected_seeds": sorted(actual - seed_set),
                            "rows": int(len(unit)),
                        }
                    )
    if missing_units:
        raise RuntimeError(
            f"incomplete expected analysis grid: {len(missing_units)} cells; "
            f"examples={missing_units[:5]}"
        )

    return {
        "protocol": PROTOCOL,
        "rows": int(len(frame)),
        "primary_key": key_cols,
        "duplicate_primary_keys": 0,
        "instances": int(len(instances)),
        "instance_ids": [f"{dataset}/{instance}" for dataset, instance in instances],
        "methods": wanted_methods,
        "budgets": wanted_budget_labels,
        "seeds": wanted_seeds,
        "expected_cells": int(
            len(instances) * len(wanted_budget_labels) * len(wanted_methods)
        ),
        "status": "complete",
    }


def add_common_hv(
    frame: pd.DataFrame,
    csv_path: Path,
    search_root: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Compute exact 3-D HV using one scale per instance/budget block."""
    frame = frame.copy()
    fronts: dict[int, np.ndarray] = {}
    raw_counts: dict[int, int] = {}
    path_cache: dict[Path, tuple[np.ndarray, int]] = {}
    failures = []
    for idx, row in frame.iterrows():
        try:
            path = locate_front(row["front_pickle"], csv_path, search_root)
            if path not in path_cache:
                path_cache[path] = load_front(path)
            front, raw_count = path_cache[path]
            fronts[idx] = front
            raw_counts[idx] = raw_count
        except Exception as exc:  # report all missing/corrupt files together
            failures.append(
                {"row": _canonical_scalar(idx), "path": row["front_pickle"], "error": repr(exc)}
            )
    if failures:
        raise RuntimeError(f"{len(failures)} front files failed: {failures[:5]}")

    frame["Front_points_raw"] = pd.Series(raw_counts)
    frame["Front_points_unique"] = pd.Series({idx: len(front) for idx, front in fronts.items()})
    frame["HV_common"] = np.nan
    normalization = []
    group_cols = ["dataset", "instance", *_budget_columns(frame)]

    for keys, group in frame.groupby(group_cols, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        pooled = np.vstack(
            [fronts[idx][:, HV_OBJECTIVE_COLUMNS] for idx in group.index]
        )
        pooled = stable_unique_rows(pooled)
        ideal = pooled.min(axis=0)
        nadir = pooled.max(axis=0)
        scale = np.where(nadir > ideal, nadir - ideal, 1.0)

        for idx in group.index:
            projected = stable_unique_rows(fronts[idx][:, HV_OBJECTIVE_COLUMNS])
            normalized = (projected - ideal) / scale
            normalized = nondominated(normalized)
            frame.loc[idx, "HV_common"] = float(
                HV(ref_point=HV_REFERENCE)(normalized)
            )

        normalization.append(
            {
                **{
                    column: _canonical_scalar(value)
                    for column, value in zip(group_cols, key_values)
                },
                "objective_indices": list(HV_OBJECTIVE_COLUMNS),
                "ideal": ideal.tolist(),
                "nadir": nadir.tolist(),
                "reference": HV_REFERENCE.tolist(),
                "pooled_unique_points": int(len(pooled)),
                "runs": int(len(group)),
            }
        )

    if frame["HV_common"].isna().any():
        raise RuntimeError("HV_common contains missing values after computation")
    return frame, normalization


def collapse_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Take seed medians inside every instance/method/budget analysis cell."""
    group_cols = _analysis_cell_columns(frame)
    preferred_metrics = [
        "HV_common",
        "Cmax_best",
        "TEC_best",
        "WB_best",
        "NSol",
        "Time",
        "Transition_gen",
        "PPO_update_count",
        "BC_final_accuracy",
        "Entropy_last20",
    ]
    metrics = [name for name in preferred_metrics if name in frame.columns]
    numeric = frame[metrics].apply(pd.to_numeric, errors="raise")
    working = frame[group_cols + ["seed"]].copy()
    working[metrics] = numeric
    medians = (
        working.groupby(group_cols, sort=True, dropna=False)[metrics]
        .median()
        .reset_index()
    )
    counts = (
        working.groupby(group_cols, sort=True, dropna=False)
        .size()
        .reset_index(name="seed_count")
    )
    return medians.merge(counts, on=group_cols, validate="one_to_one")


def rank_biserial_from_diff(
    differences: Iterable[float], tolerance: float = DEFAULT_TIE_TOLERANCE
) -> float:
    diff = np.asarray(list(differences), dtype=float)
    diff = diff[np.abs(diff) > tolerance]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    return float((positive - negative) / ranks.sum())


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment, returned in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if len(p_values) == 0:
        return np.array([], dtype=float)
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _wilcoxon_from_diff(diff: np.ndarray, tolerance: float) -> tuple[float, float]:
    non_ties = np.asarray(diff, dtype=float)
    non_ties = non_ties[np.abs(non_ties) > tolerance]
    if len(non_ties) == 0:
        return 0.0, 1.0
    result = wilcoxon(non_ties, alternative="two-sided", zero_method="wilcox")
    p_value = float(result.pvalue)
    return float(result.statistic), p_value if np.isfinite(p_value) else 1.0


def bootstrap_ci(
    values: Iterable[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    reps: int = 10000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Deterministic percentile CI from resampling independent instances."""
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    if len(values) == 1:
        point = float(statistic(values))
        return point, point
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    estimates = np.array([statistic(values[index]) for index in draws], dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1 - alpha]))


def _comparison_seed(base_seed: int, *parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "big")) % (2**32))


def main_inference(
    seed_medians: pd.DataFrame,
    *,
    target: str,
    comparators: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TIE_TOLERANCE,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Run instance-blocked Friedman, ranks, and target comparisons."""
    budget_cols = _budget_columns(seed_medians)
    strata = (
        seed_medians.groupby("Budget", sort=True, dropna=False)
        if budget_cols
        else [("default", seed_medians)]
    )
    methods = sorted(seed_medians["variant"].astype(str).unique().tolist())
    if target not in methods:
        raise RuntimeError(f"target {target!r} is absent; methods={methods}")
    comparator_list = list(comparators) if comparators is not None else [m for m in methods if m != target]
    if len(set(comparator_list)) != len(comparator_list):
        raise RuntimeError("comparators contain duplicates")
    invalid = sorted(set(comparator_list) - (set(methods) - {target}))
    if invalid:
        raise RuntimeError(f"invalid comparators: {invalid}")

    friedman_rows = []
    rank_rows = []
    pairwise_rows = []
    for budget, subset in strata:
        pivot = subset.pivot(
            index=["dataset", "instance"], columns="variant", values="HV_common"
        ).sort_index(axis=1)
        if pivot.isna().any().any():
            raise RuntimeError(f"main inference has unmatched instance cells for budget={budget}")
        ordered_methods = list(pivot.columns)
        arrays = [pivot[method].to_numpy(dtype=float) for method in ordered_methods]
        if len(ordered_methods) >= 3 and len(pivot) >= 2:
            stacked = np.column_stack(arrays)
            if np.all(np.ptp(stacked, axis=1) <= tolerance):
                statistic, p_value = 0.0, 1.0
            else:
                result = friedmanchisquare(*arrays)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            friedman_rows.append(
                {
                    "Budget": _canonical_scalar(budget),
                    "independent_blocks": int(len(pivot)),
                    "methods": json.dumps(ordered_methods, separators=(",", ":")),
                    "friedman_statistic": statistic,
                    "friedman_p": p_value,
                }
            )

        for (dataset, instance), row in pivot.iterrows():
            ranks = rankdata(-row.to_numpy(dtype=float), method="average")
            for method, value, rank in zip(ordered_methods, row, ranks):
                rank_rows.append(
                    {
                        "dataset": dataset,
                        "instance": instance,
                        "Budget": _canonical_scalar(budget),
                        "variant": method,
                        "HV_common": float(value),
                        "rank": float(rank),
                    }
                )

        for comparator in comparator_list:
            diff = pivot[target].to_numpy(dtype=float) - pivot[comparator].to_numpy(dtype=float)
            statistic, p_raw = _wilcoxon_from_diff(diff, tolerance)
            ci_low, ci_high = bootstrap_ci(
                diff,
                statistic=np.median,
                reps=bootstrap_reps,
                seed=_comparison_seed(bootstrap_seed, budget, target, comparator),
            )
            pairwise_rows.append(
                {
                    "Budget": _canonical_scalar(budget),
                    "target": target,
                    "comparator": comparator,
                    "independent_unit": "instance",
                    "instances": int(len(diff)),
                    "numerical_tie_tolerance": float(tolerance),
                    "wins": int(np.sum(diff > tolerance)),
                    "ties": int(np.sum(np.abs(diff) <= tolerance)),
                    "losses": int(np.sum(diff < -tolerance)),
                    "median_difference": float(np.median(diff)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "rank_biserial": rank_biserial_from_diff(diff, tolerance),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                    "comparison_family": "main_target_vs_comparators_all_budgets",
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        # One predeclared family across all budgets.  In particular, E3 is not
        # silently split into separate low/medium/high-budget Holm families.
        pairwise["p_holm"] = holm_adjust(pairwise["p_raw"].to_numpy(dtype=float))
        pairwise["holm_family_size"] = int(len(pairwise))
        pairwise["significant_0_05"] = pairwise["p_holm"] < 0.05

    instance_ranks = pd.DataFrame(rank_rows)
    mean_by_budget = (
        instance_ranks.groupby(["Budget", "variant"], sort=True)["rank"]
        .agg(mean_rank="mean", rank_sd="std", independent_blocks="count")
        .reset_index()
        .sort_values(["Budget", "mean_rank", "variant"])
    )
    overall_mean = (
        instance_ranks.groupby("variant", sort=True)["rank"]
        .agg(overall_mean_rank="mean", rank_sd="std", instance_budget_blocks="count")
        .reset_index()
        .sort_values(["overall_mean_rank", "variant"])
    )
    return {
        "friedman": pd.DataFrame(friedman_rows),
        "pairwise": pairwise,
        "instance_ranks": instance_ranks,
        "mean_ranks_by_budget": mean_by_budget,
        "overall_mean_ranks": overall_mean,
    }


def summarize_main(
    seed_medians: pd.DataFrame,
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Instance-level descriptive summaries for all declared outcome metrics."""
    group_cols = [*_budget_columns(seed_medians), "variant"]
    rows = []
    for keys, group in seed_medians.groupby(group_cols, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = {column: _canonical_scalar(value) for column, value in zip(group_cols, key_values)}
        row["independent_unit"] = "instance"
        row["instances"] = int(len(group))
        metrics = [
            metric
            for metric in ("HV_common", "Cmax_best", "TEC_best", "WB_best", "NSol", "Time")
            if metric in group.columns
        ]
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ci_low, ci_high = bootstrap_ci(
                values,
                statistic=np.mean,
                reps=bootstrap_reps,
                seed=_comparison_seed(
                    bootstrap_seed,
                    *key_values,
                    metric,
                    "method_summary",
                ),
            )
            row.update(
                {
                    f"{metric}_mean_of_instance_medians": float(np.mean(values)),
                    f"{metric}_median_of_instance_medians": float(median),
                    f"{metric}_q1_of_instance_medians": float(q1),
                    f"{metric}_q3_of_instance_medians": float(q3),
                    f"{metric}_iqr_of_instance_medians": float(q3 - q1),
                    f"{metric}_bootstrap_mean_ci95_low": ci_low,
                    f"{metric}_bootstrap_mean_ci95_high": ci_high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def supplementary_seed_analysis(
    frame: pd.DataFrame,
    *,
    target: str,
    comparators: Sequence[str],
    tolerance: float,
) -> pd.DataFrame:
    """Describe nested instance-seed contrasts without pseudo-replicated tests."""
    strata = (
        frame.groupby("Budget", sort=True, dropna=False)
        if "Budget" in frame.columns
        else [("default", frame)]
    )
    rows = []
    for budget, subset in strata:
        pivot = subset.pivot(
            index=["dataset", "instance", "seed"],
            columns="variant",
            values="HV_common",
        )
        for comparator in comparators:
            if target not in pivot or comparator not in pivot:
                raise RuntimeError(
                    f"supplementary comparison missing {target} or {comparator} at budget={budget}"
                )
            matched = pivot[[target, comparator]].dropna()
            diff = matched[target].to_numpy(dtype=float) - matched[comparator].to_numpy(dtype=float)
            rows.append(
                {
                    "Budget": _canonical_scalar(budget),
                    "target": target,
                    "comparator": comparator,
                    "inference_status": "descriptive_only_no_independence_assumed",
                    "nested_unit": "instance_seed",
                    "matched_instance_seed_rows": int(len(diff)),
                    "numerical_tie_tolerance": float(tolerance),
                    "wins": int(np.sum(diff > tolerance)),
                    "ties": int(np.sum(np.abs(diff) <= tolerance)),
                    "losses": int(np.sum(diff < -tolerance)),
                    "median_difference": float(np.median(diff)),
                    "q1_difference": float(np.quantile(diff, 0.25)),
                    "q3_difference": float(np.quantile(diff, 0.75)),
                    "rank_biserial": rank_biserial_from_diff(diff, tolerance),
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    csv_path: Path,
    out_dir: Path,
    *,
    search_root: Path,
    target: str = "AdaptiveSAOS",
    comparators: Sequence[str] | None = None,
    expected_seed_count: int | None = 10,
    expected_seed_values: Sequence[int] | None = None,
    expected_methods: Sequence[str] | None = None,
    expected_budgets: Sequence[str] | None = None,
    expected_instance_count: int | None = 50,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    frame = pd.read_csv(csv_path)
    observed_methods = set(frame.get("variant", pd.Series(dtype=str)).astype(str))
    if expected_methods is None:
        if "Budget" in frame.columns:
            expected_methods = E3_METHODS
        elif observed_methods & set(E2_METHODS):
            expected_methods = E2_METHODS
        else:
            expected_methods = E1_METHODS
    if expected_budgets is None and "Budget" in frame.columns:
        expected_budgets = E3_BUDGETS
    completeness = validate_completeness(
        frame,
        expected_seed_count=expected_seed_count,
        expected_seed_values=expected_seed_values,
        expected_methods=expected_methods,
        expected_budgets=expected_budgets,
        expected_instance_count=expected_instance_count,
    )
    frame, normalization = add_common_hv(frame, csv_path, search_root)
    seed_medians = collapse_seeds(frame)
    methods = completeness["methods"]
    resolved_comparators = (
        list(comparators)
        if comparators is not None
        else [method for method in methods if method != target]
    )
    inference = main_inference(
        seed_medians,
        target=target,
        comparators=resolved_comparators,
        tolerance=tie_tolerance,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    main_summary = summarize_main(
        seed_medians,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    supplementary = supplementary_seed_analysis(
        frame,
        target=target,
        comparators=resolved_comparators,
        tolerance=tie_tolerance,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "runs_with_common_hv.csv", index=False)
    seed_medians.to_csv(out_dir / "seed_medians_main.csv", index=False)
    main_summary.to_csv(out_dir / "main_summary_by_method.csv", index=False)
    inference["friedman"].to_csv(out_dir / "friedman_main.csv", index=False)
    inference["pairwise"].to_csv(out_dir / "main_pairwise_vs_target.csv", index=False)
    inference["instance_ranks"].to_csv(out_dir / "main_instance_ranks.csv", index=False)
    inference["mean_ranks_by_budget"].to_csv(out_dir / "mean_ranks_by_budget.csv", index=False)
    inference["overall_mean_ranks"].to_csv(out_dir / "overall_mean_ranks.csv", index=False)
    supplementary.to_csv(out_dir / "supplementary_seed_level_pairwise.csv", index=False)
    (out_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    (out_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "hv_objectives": ["Cmax", "TEC", "workload_imbalance"],
        "hv_reference": HV_REFERENCE.tolist(),
        "main_inference_unit": "instance after within-cell seed median",
        "supplementary_seed_results_are_main_inference": False,
        "target": target,
        "comparators": resolved_comparators,
        "holm_main_family": "target vs all predefined comparators across all budgets",
        "numerical_tie_tolerance": tie_tolerance,
        "tie_interpretation": "floating-point equality only; not practical equivalence",
        "bootstrap": {
            "unit": "instance",
            "method": "deterministic percentile bootstrap",
            "reps": bootstrap_reps,
            "seed": bootstrap_seed,
        },
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "completeness": completeness,
        "normalization": normalization,
        "seed_medians": seed_medians,
        "main_summary": main_summary,
        **inference,
        "supplementary": supplementary,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("."))
    parser.add_argument("--target", default="AdaptiveSAOS")
    parser.add_argument("--comparators", default=None)
    parser.add_argument("--expected-methods", default=None)
    parser.add_argument("--expected-budgets", default=None)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--expected-seed-values",
        default=",".join(str(value) for value in range(42, 52)),
    )
    parser.add_argument("--expected-instances", type=int, default=50)
    parser.add_argument("--tie-tolerance", type=float, default=DEFAULT_TIE_TOLERANCE)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    seed_values = _parse_csv_list(args.expected_seed_values)
    result = run_analysis(
        args.csv,
        args.out_dir,
        search_root=args.search_root,
        target=args.target,
        comparators=_parse_csv_list(args.comparators),
        expected_seed_count=args.expected_seeds,
        expected_seed_values=[int(value) for value in seed_values] if seed_values else None,
        expected_methods=_parse_csv_list(args.expected_methods),
        expected_budgets=_parse_csv_list(args.expected_budgets),
        expected_instance_count=args.expected_instances,
        tie_tolerance=args.tie_tolerance,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result["completeness"], indent=2))


if __name__ == "__main__":
    main()
