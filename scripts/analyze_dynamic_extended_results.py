"""Analyze expanded dynamic-rescheduling revision results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHOD_ORDER = ["RightShift", "LocalRepair", "RollingHorizonEA", "FullReschedule", "GradedResponse"]
LOWER_BETTER = ["cmax_deviation_pct", "tec_deviation_pct", "stability_f4", "avg_response_time"]


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def effect_size_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) < 3:
        return np.nan
    stat, p = stats.wilcoxon(diff)
    z = stats.norm.isf(p / 2.0)
    return z / np.sqrt(len(diff))


def format_mean_ci(values, digits=2):
    values = np.asarray(values, dtype=float)
    return f"{np.nanmean(values):.{digits}f} +/- {ci95(values):.{digits}f}"


def add_average_rank(df, metric):
    ranks = []
    keys = ["instance", "scenario", "intensity", "seed"]
    for _, group in df.groupby(keys):
        group = group.copy()
        group["rank"] = group[metric].rank(method="average", ascending=True)
        ranks.append(group[["instance", "scenario", "intensity", "seed", "method", "rank"]])
    if not ranks:
        return pd.DataFrame()
    return pd.concat(ranks, ignore_index=True)


def wilcoxon_vs_graded(df, metric):
    rows = []
    keys = ["instance", "scenario", "intensity", "seed"]
    pivot = df.pivot_table(index=keys, columns="method", values=metric, aggfunc="mean")
    if "GradedResponse" not in pivot.columns:
        return pd.DataFrame()
    for method in METHOD_ORDER:
        if method == "GradedResponse" or method not in pivot.columns:
            continue
        paired = pivot[[method, "GradedResponse"]].dropna()
        if len(paired) < 3:
            p_value = np.nan
            stat = np.nan
            r_value = np.nan
        else:
            try:
                stat, p_value = stats.wilcoxon(paired[method], paired["GradedResponse"], alternative="two-sided")
            except ValueError:
                stat, p_value = np.nan, 1.0
            r_value = effect_size_r(paired[method], paired["GradedResponse"])
        rows.append({
            "metric": metric,
            "method": method,
            "n_pairs": len(paired),
            "method_mean": paired[method].mean() if len(paired) else np.nan,
            "graded_mean": paired["GradedResponse"].mean() if len(paired) else np.nan,
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "effect_size_r": r_value,
        })
    return pd.DataFrame(rows)


def friedman_by_block(df, metric):
    rows = []
    keys = ["scenario", "intensity"]
    for (scenario, intensity), group in df.groupby(keys):
        pivot = group.pivot_table(
            index=["instance", "seed"],
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        cols = [m for m in METHOD_ORDER if m in pivot.columns]
        pivot = pivot[cols].dropna()
        if len(cols) < 3 or len(pivot) < 3:
            stat, p_value = np.nan, np.nan
        else:
            stat, p_value = stats.friedmanchisquare(*[pivot[col].values for col in cols])
        rows.append({
            "metric": metric,
            "scenario": scenario,
            "intensity": intensity,
            "n_blocks": len(pivot),
            "methods": ",".join(cols),
            "friedman_chi2": stat,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def build_summary(df):
    rank_df = add_average_rank(df, "cmax_deviation_pct")
    rank_mean = rank_df.groupby("method")["rank"].mean().rename("avg_cmax_rank") if not rank_df.empty else pd.Series(dtype=float)

    rows = []
    for method in METHOD_ORDER:
        group = df[df["method"] == method]
        if group.empty:
            continue
        row = {
            "method": method,
            "runs": len(group),
            "cmax_deviation_pct": format_mean_ci(group["cmax_deviation_pct"], 2),
            "tec_deviation_pct": format_mean_ci(group["tec_deviation_pct"], 2),
            "stability_f4": format_mean_ci(group["stability_f4"], 2),
            "avg_response_time": format_mean_ci(group["avg_response_time"], 3),
            "success_rate": f"{group['success'].mean():.3f}",
            "avg_events": f"{group['num_events'].mean():.2f}",
            "avg_cmax_rank": f"{rank_mean.get(method, np.nan):.3f}",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_scenario_intensity_summary(df):
    rows = []
    for (scenario, intensity, method), group in df.groupby(["scenario", "intensity", "method"]):
        rows.append({
            "scenario": scenario,
            "intensity": intensity,
            "method": method,
            "runs": len(group),
            "cmax_deviation_pct_mean": group["cmax_deviation_pct"].mean(),
            "cmax_deviation_pct_ci95": ci95(group["cmax_deviation_pct"]),
            "tec_deviation_pct_mean": group["tec_deviation_pct"].mean(),
            "stability_f4_mean": group["stability_f4"].mean(),
            "avg_response_time_mean": group["avg_response_time"].mean(),
            "success_rate": group["success"].mean(),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["scenario", "intensity", "method"])


def plot_summary(df, out_dir: Path):
    summary = build_scenario_intensity_summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=200)
    metrics = [
        ("cmax_deviation_pct_mean", "Cmax deviation (%)"),
        ("stability_f4_mean", "Stability f4"),
        ("avg_response_time_mean", "Response time (s)"),
    ]
    x = np.arange(len(METHOD_ORDER))
    colors = ["#666666", "#999999", "#BBBBBB", "#DDDDDD", "#333333"]
    hatches = ["//", "\\\\", "..", "xx", ""]
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = []
        errs = []
        for method in METHOD_ORDER:
            group = summary[summary["method"] == method]
            vals.append(group[metric].mean())
            if metric == "cmax_deviation_pct_mean":
                errs.append(group["cmax_deviation_pct_ci95"].mean())
            else:
                errs.append(0)
        bars = ax.bar(x, vals, yerr=errs if metric == "cmax_deviation_pct_mean" else None,
                      color=colors, edgecolor="black", linewidth=0.8, capsize=3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_extended_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "dynamic_extended_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/revision/dynamic_extended/dynamic_extended_runs.csv")
    parser.add_argument("--out-dir", default="results/revision/analysis_dynamic_extended")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["success"].fillna(0).astype(int) == 1].copy()
    df.to_csv(out_dir / "dynamic_extended_successful_runs.csv", index=False)

    summary = build_summary(df)
    by_block = build_scenario_intensity_summary(df)
    wilcoxon = pd.concat([wilcoxon_vs_graded(df, metric) for metric in LOWER_BETTER], ignore_index=True)
    friedman = pd.concat([friedman_by_block(df, metric) for metric in LOWER_BETTER], ignore_index=True)

    summary.to_csv(out_dir / "dynamic_extended_compact_table.csv", index=False)
    by_block.to_csv(out_dir / "dynamic_extended_by_scenario_intensity.csv", index=False)
    wilcoxon.to_csv(out_dir / "dynamic_extended_wilcoxon_vs_GradedResponse.csv", index=False)
    friedman.to_csv(out_dir / "dynamic_extended_friedman.csv", index=False)
    plot_summary(df, out_dir)

    manifest = {
        "input_csv": args.csv,
        "successful_rows": int(len(df)),
        "methods": METHOD_ORDER,
        "outputs": [
            "dynamic_extended_successful_runs.csv",
            "dynamic_extended_compact_table.csv",
            "dynamic_extended_by_scenario_intensity.csv",
            "dynamic_extended_wilcoxon_vs_GradedResponse.csv",
            "dynamic_extended_friedman.csv",
            "dynamic_extended_summary.png",
            "dynamic_extended_summary.pdf",
        ],
    }
    pd.Series(manifest).to_json(out_dir / "dynamic_extended_manifest.json", indent=2)
    print(f"Analyzed {len(df)} successful rows")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
