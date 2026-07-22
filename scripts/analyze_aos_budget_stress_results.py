"""Analyze AOS budget/stage-stress experiments.

The runner stores one row per (budget, dataset, instance, variant, seed). This
script recomputes normalized hypervolume with a unified reference point per
(budget, dataset, instance), then reports mean/CI tables, average ranks, and
paired Wilcoxon tests against AdaptiveSAOS within each budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.analyze_revision_results import (
    METRIC_DIRECTIONS,
    cliffs_delta,
    hv3d_min,
    load_fronts,
)


DEFAULT_VARIANTS = [
    "UniformFixed",
    "UCBOnly",
    "PPOOnly",
    "FixedUCBPPO",
    "RandomUCBPPO",
    "AdaptiveSAOS",
]


def ci95(values: pd.Series) -> float:
    n = values.count()
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(n))


def parse_variants(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_VARIANTS)
    return [part.strip() for part in value.split(",") if part.strip()]


def add_budget_hv(df: pd.DataFrame, fronts: list[np.ndarray]) -> pd.DataFrame:
    df = df.copy()
    df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce").astype(int)
    hv_values = np.zeros(len(df), dtype=float)
    ref_values = {}
    group_cols = ["Budget", "dataset", "instance"]
    for (budget, dataset, instance), idx in df.groupby(group_cols).groups.items():
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
        ref_values[f"gen{budget}/{dataset}/{instance}"] = {
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


def summarize_by_budget_variant(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    metrics = [m for m in METRIC_DIRECTIONS if m in df.columns]
    rows = []
    for budget in sorted(df["Budget"].dropna().unique()):
        budget_df = df[df["Budget"] == budget]
        grouped = budget_df.groupby("variant")
        for variant in variants:
            if variant not in grouped.groups:
                continue
            group = grouped.get_group(variant)
            row = {"Budget": int(budget), "variant": variant, "n": int(len(group))}
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_ci95"] = ci95(values)
            rows.append(row)
    return pd.DataFrame(rows)


def average_ranks_by_budget(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    key_cols = ["dataset", "instance", "seed"]
    rows = []
    for budget in sorted(df["Budget"].dropna().unique()):
        budget_df = df[df["Budget"] == budget]
        for metric, direction in METRIC_DIRECTIONS.items():
            if metric not in budget_df.columns:
                continue
            pivot = budget_df.pivot_table(
                index=key_cols, columns="variant", values=metric, aggfunc="mean"
            )
            present = [v for v in variants if v in pivot.columns]
            pivot = pivot[present].dropna()
            if pivot.empty:
                continue
            ranks = pivot.rank(axis=1, ascending=(direction == "min"), method="average")
            for variant, value in ranks.mean(axis=0).items():
                rows.append(
                    {
                        "Budget": int(budget),
                        "metric": metric,
                        "variant": variant,
                        "average_rank": float(value),
                        "n_pairs": int(len(ranks)),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Budget", "metric", "average_rank", "variant"]
    )


def paired_wilcoxon_by_budget(
    df: pd.DataFrame, variants: list[str], baseline: str
) -> pd.DataFrame:
    key_cols = ["dataset", "instance", "seed"]
    rows = []
    for budget in sorted(df["Budget"].dropna().unique()):
        budget_df = df[df["Budget"] == budget]
        for metric, direction in METRIC_DIRECTIONS.items():
            if metric not in budget_df.columns:
                continue
            pivot = budget_df.pivot_table(
                index=key_cols, columns="variant", values=metric, aggfunc="mean"
            )
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
                    diff = comp - base
                nonzero = diff[np.abs(diff) > 1e-12]
                if len(nonzero) == 0:
                    stat = 0.0
                    pvalue = 1.0
                else:
                    stat, pvalue = stats.wilcoxon(
                        nonzero, alternative="two-sided", zero_method="wilcox"
                    )
                rows.append(
                    {
                        "Budget": int(budget),
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


def best_counts(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows = []
    key_cols = ["Budget", "dataset", "instance", "seed"]
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index=key_cols, columns="variant", values=metric, aggfunc="mean")
        present = [v for v in variants if v in pivot.columns]
        pivot = pivot[present].dropna()
        if pivot.empty:
            continue
        if direction == "min":
            winners = pivot.idxmin(axis=1)
        else:
            winners = pivot.idxmax(axis=1)
        counts = winners.groupby(level=0).value_counts()
        for (budget, variant), count in counts.items():
            rows.append(
                {
                    "Budget": int(budget),
                    "metric": metric,
                    "variant": variant,
                    "best_count": int(count),
                    "total_pairs": int((winners.index.get_level_values(0) == budget).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["Budget", "metric", "variant"])


def plot_hv_budget(summary: pd.DataFrame, out_path: Path, variants: list[str]) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    for variant in variants:
        sub = summary[summary["variant"] == variant].sort_values("Budget")
        if sub.empty:
            continue
        ax.errorbar(
            sub["Budget"],
            sub["HV_unified_mean"],
            yerr=sub["HV_unified_ci95"],
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=variant,
        )
    ax.set_xlabel("Generation budget")
    ax.set_ylabel("Unified normalized HV")
    ax.set_title("AOS budget/stage-stress comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/analysis_aos_budget_stress"))
    parser.add_argument("--baseline", default="AdaptiveSAOS")
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variants)

    df = pd.read_csv(args.csv)
    fronts = load_fronts(df, root)
    df = add_budget_hv(df, fronts)

    enriched_path = out_dir / "aos_budget_stress_enriched_runs.csv"
    summary_path = out_dir / "aos_budget_stress_summary_by_budget_variant.csv"
    ranks_path = out_dir / "aos_budget_stress_average_ranks_by_budget.csv"
    wilcoxon_path = out_dir / f"aos_budget_stress_wilcoxon_vs_{args.baseline}.csv"
    best_path = out_dir / "aos_budget_stress_best_counts.csv"
    refs_path = out_dir / "aos_budget_stress_hv_reference_points.json"
    fig_path = out_dir / "aos_budget_stress_hv_by_budget.png"
    manifest_path = out_dir / "aos_budget_stress_manifest.json"

    summary = summarize_by_budget_variant(df, variants)
    df.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)
    average_ranks_by_budget(df, variants).to_csv(ranks_path, index=False)
    paired_wilcoxon_by_budget(df, variants, args.baseline).to_csv(wilcoxon_path, index=False)
    best_counts(df, variants).to_csv(best_path, index=False)
    refs_path.write_text(json.dumps(df.attrs.get("reference_points", {}), indent=2), encoding="utf-8")
    plot_hv_budget(summary, fig_path, variants)

    manifest = {
        "runs": int(len(df)),
        "budgets": sorted(int(v) for v in df["Budget"].unique()),
        "variants": variants,
        "baseline": args.baseline,
        "enriched": str(enriched_path),
        "summary": str(summary_path),
        "average_ranks": str(ranks_path),
        "wilcoxon": str(wilcoxon_path),
        "best_counts": str(best_path),
        "references": str(refs_path),
        "figure": str(fig_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
