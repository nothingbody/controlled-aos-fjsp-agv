"""Analyze corrected SA-AOS runs with common-reference HV and matched tests."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort


PROTOCOL = "saos_bc_onpolicy_ppo_v4_20260720"


def nondominated(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points
    fronts = non_dominated_sort(points)
    return points[fronts[0]]


def locate_front(raw_path: str, csv_path: Path, search_root: Path) -> Path:
    candidate = Path(raw_path)
    attempts = [candidate, csv_path.parent / candidate, search_root / candidate]
    for attempt in attempts:
        if attempt.exists():
            return attempt
    matches = list(search_root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous front path {raw_path}: {len(matches)} matches")
    raise FileNotFoundError(raw_path)


def load_front(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    points = np.asarray(payload["objectives"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"unexpected objective shape {points.shape} in {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"non-finite objective in {path}")
    return nondominated(points)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.abs(diff) > 1e-15]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    return float((positive - negative) / ranks.sum())


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def add_common_hv(frame: pd.DataFrame, csv_path: Path, search_root: Path):
    fronts = {}
    missing = []
    for idx, row in frame.iterrows():
        try:
            path = locate_front(row["front_pickle"], csv_path, search_root)
            fronts[idx] = load_front(path)
        except Exception as exc:
            missing.append({"row": int(idx), "path": row["front_pickle"], "error": repr(exc)})
    if missing:
        raise RuntimeError(f"{len(missing)} front files failed: {missing[:3]}")

    group_cols = ["dataset", "instance"]
    if "Budget" in frame.columns:
        group_cols.append("Budget")
    frame["HV_common"] = np.nan
    normalization = []

    for keys, group in frame.groupby(group_cols, sort=True):
        pooled = np.vstack([fronts[idx] for idx in group.index])
        ideal = pooled.min(axis=0)
        nadir = pooled.max(axis=0)
        scale = np.where(nadir > ideal, nadir - ideal, 1.0)
        reference = np.full(3, 1.1, dtype=float)
        for idx in group.index:
            normalized = (fronts[idx] - ideal) / scale
            frame.loc[idx, "HV_common"] = compute_hypervolume(
                nondominated(normalized), reference
            )
        key_values = keys if isinstance(keys, tuple) else (keys,)
        normalization.append(
            {
                **dict(zip(group_cols, key_values)),
                "ideal": ideal.tolist(),
                "nadir": nadir.tolist(),
                "reference": reference.tolist(),
                "pooled_points": int(len(pooled)),
            }
        )
    return frame, normalization


def summarize(frame: pd.DataFrame, extra_group=None):
    group_cols = ([] if extra_group is None else list(extra_group)) + ["variant"]
    rows = []
    for keys, group in frame.groupby(group_cols, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        for metric in ["HV_common", "Cmax_best", "TEC_best", "WB_best", "Time"]:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q3"] = float(np.quantile(values, 0.75))
            row[f"{metric}_mean"] = float(np.mean(values))
        row["runs"] = int(len(group))
        row["transition_median"] = float(np.median(group["Transition_gen"].astype(float)))
        row["ppo_updates_median"] = float(np.median(group["PPO_update_count"].astype(float)))
        row["bc_final_accuracy_median"] = float(np.median(group["BC_final_accuracy"].astype(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def matched_analysis(frame: pd.DataFrame, target: str, extra_group=None):
    extra_group = [] if extra_group is None else list(extra_group)
    strata = frame.groupby(extra_group, dropna=False) if extra_group else [((), frame)]
    all_pairwise = []
    omnibus = []

    for stratum_keys, subset in strata:
        index_cols = ["dataset", "instance", "seed"]
        pivot = subset.pivot_table(index=index_cols, columns="variant", values="HV_common", aggfunc="first")
        pivot = pivot.dropna(axis=0, how="any")
        variants = list(pivot.columns)
        stratum_values = stratum_keys if isinstance(stratum_keys, tuple) else (stratum_keys,)
        stratum = dict(zip(extra_group, stratum_values))

        if len(variants) >= 3 and len(pivot) >= 2:
            statistic, p_value = friedmanchisquare(
                *[pivot[name].to_numpy() for name in variants]
            )
            omnibus.append(
                {
                    **stratum,
                    "matched_cells": int(len(pivot)),
                    "variants": json.dumps(variants),
                    "friedman_statistic": float(statistic),
                    "friedman_p": float(p_value),
                }
            )

        if target not in pivot.columns:
            continue
        target_values = pivot[target].to_numpy()
        local_rows = []
        for comparator in variants:
            if comparator == target:
                continue
            other = pivot[comparator].to_numpy()
            diff = target_values - other
            try:
                test = wilcoxon(target_values, other, alternative="two-sided", zero_method="wilcox")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            except ValueError:
                p_value = 1.0
                statistic = 0.0
            local_rows.append(
                {
                    **stratum,
                    "target": target,
                    "comparator": comparator,
                    "matched_cells": int(len(diff)),
                    "wins": int(np.sum(diff > 1e-12)),
                    "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                    "losses": int(np.sum(diff < -1e-12)),
                    "median_difference": float(np.median(diff)),
                    "rank_biserial": rank_biserial(target_values, other),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                }
            )
        adjusted = holm_adjust([row["p_raw"] for row in local_rows])
        for row, p_adjusted in zip(local_rows, adjusted):
            row["p_holm"] = float(p_adjusted)
            row["significant_0_05"] = bool(p_adjusted < 0.05)
            all_pairwise.append(row)
    return pd.DataFrame(all_pairwise), omnibus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("."))
    parser.add_argument("--target", default="AdaptiveSAOS")
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.csv)
    if set(frame["Protocol"].astype(str)) != {PROTOCOL}:
        raise RuntimeError(
            f"protocol mismatch: {sorted(frame['Protocol'].astype(str).unique())}"
        )

    key_cols = ["dataset", "instance", "variant", "reward_scheme", "seed"]
    if "Budget" in frame.columns:
        key_cols.append("Budget")
    duplicate_count = int(frame.duplicated(key_cols).sum())
    if duplicate_count:
        raise RuntimeError(f"duplicate result keys: {duplicate_count}")

    frame, normalization = add_common_hv(frame, args.csv, args.search_root)
    extra_group = ["Budget"] if "Budget" in frame.columns else []
    summary = summarize(frame, extra_group=extra_group)
    pairwise, omnibus = matched_analysis(
        frame, target=args.target, extra_group=extra_group
    )

    cell_counts = (
        frame.groupby(extra_group + ["dataset", "instance", "variant"])
        .size()
        .reset_index(name="runs")
    )
    incomplete = cell_counts[cell_counts["runs"] != args.expected_seeds]
    completeness = {
        "protocol": PROTOCOL,
        "rows": int(len(frame)),
        "duplicate_keys": duplicate_count,
        "unique_instances": int(frame[["dataset", "instance"]].drop_duplicates().shape[0]),
        "variants": sorted(frame["variant"].unique().tolist()),
        "seeds": sorted(int(value) for value in frame["seed"].unique()),
        "incomplete_cells": incomplete.to_dict(orient="records"),
    }

    frame.to_csv(args.out_dir / "runs_with_common_hv.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_vs_target.csv", index=False)
    cell_counts.to_csv(args.out_dir / "cell_counts.csv", index=False)
    (args.out_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    (args.out_dir / "friedman.json").write_text(
        json.dumps(omnibus, indent=2), encoding="utf-8"
    )
    (args.out_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    print(json.dumps(completeness, indent=2))


if __name__ == "__main__":
    main()
