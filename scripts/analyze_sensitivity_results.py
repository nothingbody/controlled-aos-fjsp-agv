"""Analyze SA-AOS parameter sensitivity results for the revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_revision_results import analyze_one


DEFAULT_VARIANTS = [
    "Default",
    "W20",
    "W80",
    "W100",
    "c0.5",
    "c1.5",
    "c2.0",
    "nmin1",
    "nmin5",
    "nmin10",
    "B32",
    "B64",
    "B96",
    "wSurvHeavy",
    "wHVHeavy",
    "wCmaxHeavy",
    "wBalanced",
    "lr1e-4",
    "lr1e-3",
    "FixedTransition48",
    "RandomTransition",
]


def fmt_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def build_compact_table(out_dir: Path) -> Path:
    enriched = pd.read_csv(out_dir / "sensitivity_enriched_runs.csv")
    summary = pd.read_csv(out_dir / "sensitivity_summary_by_variant.csv")
    ranks = pd.read_csv(out_dir / "sensitivity_average_ranks.csv")
    try:
        wilcoxon = pd.read_csv(out_dir / "sensitivity_wilcoxon_vs_Default.csv")
    except EmptyDataError:
        wilcoxon = pd.DataFrame(
            columns=["metric", "comparator", "p_value", "cliffs_delta_baseline_vs_comparator"]
        )

    meta_cols = [
        "variant",
        "parameter",
        "setting",
        "W",
        "ucb_c",
        "n_min",
        "B_min",
        "reward_alpha",
        "reward_beta",
        "reward_gamma",
        "ppo_lr",
        "transition_mode",
    ]
    meta = enriched[meta_cols].drop_duplicates("variant")
    run_means = enriched.groupby("variant", as_index=False).agg(
        Transition_gen_mean=("Transition_gen", "mean")
    )
    hv_ranks = ranks[ranks["metric"] == "HV_unified"][["variant", "average_rank"]]
    if wilcoxon.empty:
        hv_wilcox = pd.DataFrame(
            columns=["variant", "p_value", "cliffs_delta_baseline_vs_comparator"]
        )
    else:
        hv_wilcox = wilcoxon[wilcoxon["metric"] == "HV_unified"][
            ["comparator", "p_value", "cliffs_delta_baseline_vs_comparator"]
        ].rename(columns={"comparator": "variant"})

    compact = meta.merge(summary, on="variant", how="left")
    compact = compact.merge(run_means, on="variant", how="left")
    compact = compact.merge(hv_ranks, on="variant", how="left")
    compact = compact.merge(hv_wilcox, on="variant", how="left")
    compact.loc[compact["variant"] == "Default", "p_value"] = 1.0
    compact.loc[compact["variant"] == "Default", "cliffs_delta_baseline_vs_comparator"] = 0.0
    compact["HV"] = compact.apply(
        lambda r: fmt_mean_std(r["HV_unified_mean"], r["HV_unified_std"]),
        axis=1,
    )
    compact["Cmax"] = compact["Cmax_best_mean"].map(lambda x: f"{x:.1f}")
    compact["TEC"] = compact["TEC_best_mean"].map(lambda x: f"{x:.1f}")
    compact["NSol"] = compact["NSol_mean"].map(lambda x: f"{x:.1f}")
    compact["Transition"] = compact["Transition_gen_mean"].map(lambda x: f"{x:.1f}")
    compact["Entropy(last20)"] = compact["Entropy_last20_mean"].map(lambda x: f"{x:.3f}")
    compact["HV rank"] = compact["average_rank"].map(lambda x: f"{x:.3f}")
    compact["p vs default"] = compact["p_value"].map(lambda x: "--" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))

    order = {variant: i for i, variant in enumerate(DEFAULT_VARIANTS)}
    compact["_order"] = compact["variant"].map(order).fillna(999)
    compact = compact.sort_values(["_order", "variant"])
    columns = [
        "parameter",
        "setting",
        "variant",
        "HV",
        "HV rank",
        "Cmax",
        "TEC",
        "NSol",
        "Transition",
        "Entropy(last20)",
        "p vs default",
    ]
    path = out_dir / "sensitivity_compact_table.csv"
    compact[columns].to_csv(path, index=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/analysis_sensitivity"))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = analyze_one(
        args.csv,
        args.root.resolve(),
        args.out_dir,
        "sensitivity",
        "Default",
        DEFAULT_VARIANTS,
    )
    compact_path = build_compact_table(args.out_dir)
    manifest["compact_table"] = str(compact_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
