"""Render the exact E5 held-out-performance figure from frozen analysis data.

This plot-only script does not modify or rerun the optimizer or statistical
analysis.  It standardizes manuscript-facing controller names while retaining
repository identifiers in the source table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY = {
    "UCBOnly": ("UCB-only", "#1F5A9D", "o", "-"),
    "ScratchNoBC_R16": ("Scratch adaptive PPO", "#4D4D4D", "s", "--"),
    "XPrePPO_Frozen": ("Frozen transferred PPO", "#3A8F95", "^", "-."),
    "XPrePPO_Online_R16": ("Online transferred PPO", "#B64342", "D", ":"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "results/resubmission/v7_cross_instance/analysis/"
            "performance_summary.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../paper_rewriting_output/cor_submission"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.source.resolve())
    required = {"variant", "Budget", "HV_fixed_median"}
    if not required.issubset(data.columns):
        raise RuntimeError(f"Missing required columns: {required - set(data.columns)}")
    selected = data.loc[data["variant"].isin(DISPLAY)].copy()
    if len(selected) != 12:
        raise RuntimeError(f"Expected 12 controller-budget rows; found {len(selected)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    for variant, (label, color, marker, linestyle) in DISPLAY.items():
        block = selected.loc[selected["variant"] == variant].sort_values("Budget")
        ax.plot(
            block["Budget"],
            block["HV_fixed_median"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4.5,
        )

    ax.set_xlabel("Generation budget")
    ax.set_ylabel("Median fixed-reference HV")
    ax.set_xticks([50, 100, 200])
    ax.set_ylim(0.925, 1.105)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_e5_heldout_performance"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
