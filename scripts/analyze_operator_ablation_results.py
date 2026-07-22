"""Analyze operator-family ablation results for the revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_revision_results import analyze_one


VARIANTS = [
    "FullLibrary",
    "NoMachineOperator",
    "NoAGVOperator",
    "NoSpeedOperator",
]


def build_compact_table(out_dir: Path) -> Path:
    enriched = pd.read_csv(out_dir / "operator_ablation_enriched_runs.csv")
    summary = pd.read_csv(out_dir / "operator_ablation_summary_by_variant.csv")
    ranks = pd.read_csv(out_dir / "operator_ablation_average_ranks.csv")
    wilcoxon = pd.read_csv(out_dir / "operator_ablation_wilcoxon_vs_FullLibrary.csv")

    meta = enriched[["variant", "description", "removed_ops", "allowed_ops"]].drop_duplicates("variant")
    hv_rank = ranks[ranks["metric"] == "HV_unified"][["variant", "average_rank"]]
    hv_wilcox = wilcoxon[wilcoxon["metric"] == "HV_unified"][
        ["comparator", "p_value", "cliffs_delta_baseline_vs_comparator"]
    ].rename(columns={"comparator": "variant"})
    transition = enriched.groupby("variant", as_index=False).agg(
        Transition_gen_mean=("Transition_gen", "mean")
    )

    compact = meta.merge(summary, on="variant", how="left")
    compact = compact.merge(hv_rank, on="variant", how="left")
    compact = compact.merge(hv_wilcox, on="variant", how="left")
    compact = compact.merge(transition, on="variant", how="left")
    compact.loc[compact["variant"] == "FullLibrary", "p_value"] = 1.0
    compact.loc[compact["variant"] == "FullLibrary", "cliffs_delta_baseline_vs_comparator"] = 0.0
    compact["HV"] = compact.apply(lambda r: f"{r['HV_unified_mean']:.4f} +/- {r['HV_unified_std']:.4f}", axis=1)
    compact["Cmax"] = compact["Cmax_best_mean"].map(lambda x: f"{x:.1f}")
    compact["TEC"] = compact["TEC_best_mean"].map(lambda x: f"{x:.1f}")
    compact["NSol"] = compact["NSol_mean"].map(lambda x: f"{x:.1f}")
    compact["HV rank"] = compact["average_rank"].map(lambda x: f"{x:.3f}")
    compact["Transition"] = compact["Transition_gen_mean"].map(lambda x: f"{x:.1f}")
    compact["Entropy(last20)"] = compact["Entropy_last20_mean"].map(lambda x: f"{x:.3f}")
    compact["p vs full"] = compact["p_value"].map(
        lambda x: "--" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")
    )
    order = {v: i for i, v in enumerate(VARIANTS)}
    compact["_order"] = compact["variant"].map(order).fillna(999)
    compact = compact.sort_values(["_order", "variant"])
    columns = [
        "variant",
        "description",
        "HV",
        "HV rank",
        "Cmax",
        "TEC",
        "NSol",
        "Transition",
        "Entropy(last20)",
        "p vs full",
    ]
    path = out_dir / "operator_ablation_compact_table.csv"
    compact[columns].to_csv(path, index=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/analysis_operator_ablation"))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = analyze_one(
        args.csv,
        args.root.resolve(),
        args.out_dir,
        "operator_ablation",
        "FullLibrary",
        VARIANTS,
    )
    compact = build_compact_table(args.out_dir)
    manifest["compact_table"] = str(compact)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "operator_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
