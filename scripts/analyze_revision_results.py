"""Analyze major-revision AOS and reward experiments.

The revision runner stores one CSV row per run and one pickle per final
non-dominated front. This script recomputes comparable hypervolume values using
one normalized reference point per instance, then writes summary tables,
average ranks, pairwise Wilcoxon tests, and compact comparison figures.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METRIC_DIRECTIONS = {
    "HV_unified": "max",
    "Cmax_best": "min",
    "TEC_best": "min",
    "WB_best": "min",
    "NSol": "max",
    "Time": "min",
    "Entropy_all": "max",
    "Entropy_last20": "max",
    "Reward_HV_corr": "max",
}


def hv2d_min(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume for minimization."""
    if points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    area = 0.0
    best_y = math.inf
    prev_x = None
    kept = []
    for x, y in pts:
        if y < best_y:
            kept.append((x, y))
            best_y = y
    for i, (x, y) in enumerate(kept):
        next_x = kept[i + 1][0] if i + 1 < len(kept) else ref[0]
        width = max(0.0, next_x - x)
        height = max(0.0, ref[1] - y)
        area += width * height
    return float(area)


def hv3d_min(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3D hypervolume for minimization by slicing on objective 1."""
    if points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs = np.unique(pts[:, 0])
    hv = 0.0
    for i, x in enumerate(xs):
        next_x = xs[i + 1] if i + 1 < len(xs) else ref[0]
        width = max(0.0, next_x - x)
        if width <= 0:
            continue
        active = pts[pts[:, 0] <= x, 1:3]
        hv += width * hv2d_min(active, ref[1:3])
    return float(hv)


def resolve_front_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def load_fronts(df: pd.DataFrame, root: Path) -> list[np.ndarray]:
    fronts: list[np.ndarray] = []
    for path_value in df["front_pickle"].astype(str):
        path = resolve_front_path(path_value, root)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        objs = np.asarray(payload.get("objectives", []), dtype=float)
        if objs.ndim != 2 or objs.shape[1] != 3:
            objs = np.empty((0, 3), dtype=float)
        fronts.append(objs)
    return fronts


def add_unified_hv(df: pd.DataFrame, fronts: list[np.ndarray]) -> pd.DataFrame:
    df = df.copy()
    hv_values = np.zeros(len(df), dtype=float)
    ref_values = {}
    for (dataset, instance), idx in df.groupby(["dataset", "instance"]).groups.items():
        idx_list = list(idx)
        stacked = [fronts[i] for i in idx_list if len(fronts[i]) > 0]
        if not stacked:
            continue
        all_objs = np.vstack(stacked)
        ideal = all_objs.min(axis=0)
        nadir = all_objs.max(axis=0)
        denom = nadir - ideal
        denom[denom < 1e-12] = 1.0
        ref = np.full(3, 1.1, dtype=float)
        ref_values[f"{dataset}/{instance}"] = {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "normalized_ref": ref.tolist(),
        }
        for i in idx_list:
            objs = fronts[i]
            if len(objs) == 0:
                hv_values[i] = 0.0
                continue
            norm = (objs - ideal) / denom
            hv_values[i] = hv3d_min(norm, ref)
    df["HV_unified"] = hv_values
    df.attrs["reference_points"] = ref_values
    return df


def ci95(series: pd.Series) -> float:
    n = series.count()
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / math.sqrt(n))


def summarize_by_variant(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    metrics = [m for m in METRIC_DIRECTIONS if m in df.columns]
    rows = []
    grouped = df.groupby("variant")
    for variant in variants:
        if variant not in grouped.groups:
            continue
        group = grouped.get_group(variant)
        row = {"variant": variant, "n": int(len(group))}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_ci95"] = ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def average_ranks(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    key_cols = ["dataset", "instance", "seed"]
    rank_frames = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index=key_cols, columns="variant", values=metric, aggfunc="mean")
        present = [v for v in variants if v in pivot.columns]
        pivot = pivot[present].dropna()
        if pivot.empty:
            continue
        ranks = pivot.rank(axis=1, ascending=(direction == "min"), method="average")
        avg = ranks.mean(axis=0)
        for variant, value in avg.items():
            rank_frames.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "average_rank": float(value),
                    "n_pairs": int(len(ranks)),
                }
            )
    rank_df = pd.DataFrame(rank_frames)
    if rank_df.empty:
        return rank_df
    return rank_df.sort_values(["metric", "average_rank", "variant"]).reset_index(drop=True)


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for value in a:
        gt += int(np.sum(value > b))
        lt += int(np.sum(value < b))
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon(df: pd.DataFrame, variants: list[str], baseline: str) -> pd.DataFrame:
    key_cols = ["dataset", "instance", "seed"]
    rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index=key_cols, columns="variant", values=metric, aggfunc="mean")
        if baseline not in pivot.columns:
            continue
        for variant in variants:
            if variant == baseline or variant not in pivot.columns:
                continue
            paired = pivot[[baseline, variant]].dropna()
            if paired.empty:
                continue
            base = paired[baseline].astype(float)
            comp = paired[variant].astype(float)
            diff = base - comp
            if direction == "min":
                # Positive means the baseline is better for all reported diffs.
                diff = comp - base
            nonzero = diff[np.abs(diff) > 1e-12]
            if len(nonzero) == 0:
                stat = 0.0
                pvalue = 1.0
            else:
                stat, pvalue = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "comparator": variant,
                    "n_pairs": int(len(paired)),
                    "baseline_mean": float(base.mean()),
                    "comparator_mean": float(comp.mean()),
                    "baseline_better_fraction": float((diff > 0).mean()),
                    "mean_better_diff": float(diff.mean()),
                    "median_better_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(pvalue),
                    "cliffs_delta_baseline_vs_comparator": cliffs_delta(base, comp),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, variants: list[str], out_path: Path, title: str) -> None:
    present = [v for v in variants if v in set(df["variant"])]
    if not present:
        return
    means = df.groupby("variant")["HV_unified"].mean().sort_values(ascending=False)
    order = [v for v in means.index if v in present]
    entropy_col = "Entropy_last20" if "Entropy_last20" in df.columns else "Entropy_all"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    hv_data = [df.loc[df["variant"] == v, "HV_unified"].astype(float).values for v in order]
    try:
        axes[0].boxplot(hv_data, tick_labels=order, showfliers=False)
    except TypeError:
        axes[0].boxplot(hv_data, labels=order, showfliers=False)
    axes[0].set_ylabel("Unified normalized HV")
    axes[0].set_title("Final Pareto quality")
    axes[0].tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    ent_data = [df.loc[df["variant"] == v, entropy_col].astype(float).values for v in order]
    try:
        axes[1].boxplot(ent_data, tick_labels=order, showfliers=False)
    except TypeError:
        axes[1].boxplot(ent_data, labels=order, showfliers=False)
    axes[1].set_ylabel(entropy_col)
    axes[1].set_title("Operator selection behavior")
    axes[1].tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def analyze_one(
    csv_path: Path,
    root: Path,
    out_dir: Path,
    prefix: str,
    baseline: str,
    variants: list[str] | None = None,
    replacement_csv: Path | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    if replacement_csv is not None:
        replacement = pd.read_csv(replacement_csv)
        replace_variants = set(replacement["variant"].astype(str))
        df = df[~df["variant"].astype(str).isin(replace_variants)]
        df = pd.concat([df, replacement], ignore_index=True)
    if variants is None:
        variants = list(dict.fromkeys(df["variant"].astype(str)))
    fronts = load_fronts(df, root)
    df = add_unified_hv(df, fronts)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = out_dir / f"{prefix}_enriched_runs.csv"
    summary_path = out_dir / f"{prefix}_summary_by_variant.csv"
    ranks_path = out_dir / f"{prefix}_average_ranks.csv"
    wilcoxon_path = out_dir / f"{prefix}_wilcoxon_vs_{baseline}.csv"
    refs_path = out_dir / f"{prefix}_hv_reference_points.json"
    fig_path = out_dir / f"{prefix}_comparison.png"

    df.to_csv(enriched_path, index=False)
    summarize_by_variant(df, variants).to_csv(summary_path, index=False)
    average_ranks(df, variants).to_csv(ranks_path, index=False)
    paired_wilcoxon(df, variants, baseline).to_csv(wilcoxon_path, index=False)
    refs_path.write_text(json.dumps(df.attrs.get("reference_points", {}), indent=2), encoding="utf-8")
    plot_comparison(df, variants, fig_path, prefix.replace("_", " ").title())

    return {
        "prefix": prefix,
        "runs": int(len(df)),
        "variants": variants,
        "baseline": baseline,
        "enriched": str(enriched_path),
        "summary": str(summary_path),
        "average_ranks": str(ranks_path),
        "wilcoxon": str(wilcoxon_path),
        "references": str(refs_path),
        "figure": str(fig_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/analysis"))
    parser.add_argument("--aos-csv", type=Path)
    parser.add_argument("--aos-replacement-csv", type=Path)
    parser.add_argument("--reward-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    manifest = []
    if args.aos_csv:
        manifest.append(
            analyze_one(
                args.aos_csv,
                root,
                out_dir,
                "aos",
                "AdaptiveSAOS",
                [
                    "Random",
                    "UniformFixed",
                    "ProbabilityMatching",
                    "AdaptivePursuit",
                    "UCBOnly",
                    "PPOOnly",
                    "FixedUCBPPO",
                    "RandomUCBPPO",
                    "AdaptiveSAOS",
                ],
                args.aos_replacement_csv,
            )
        )
    if args.reward_csv:
        manifest.append(
            analyze_one(
                args.reward_csv,
                root,
                out_dir,
                "reward",
                "R5_composite",
                [
                    "R1_survival_only",
                    "R2_hv_only",
                    "R3_cmax_only",
                    "R4_survival_hv",
                    "R5_composite",
                    "R6_adaptive_weight",
                ],
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
