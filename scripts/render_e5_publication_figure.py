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
    "UCBOnly": ("UCB", "UCB-only", "#1F5A9D", "o"),
    "ScratchNoBC_R16": ("Scratch", "Scratch adaptive PPO", "#4D4D4D", "s"),
    "XPrePPO_Frozen": ("Frozen", "Frozen transferred PPO", "#3A8F95", "^"),
    "XPrePPO_Online_R16": ("Online", "Online transferred PPO", "#B64342", "D"),
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
    budgets = [50, 100, 200]
    variants = list(DISPLAY)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.95),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, budget in zip(axes, budgets):
        block = selected.loc[selected["Budget"].astype(int) == budget].set_index(
            "variant"
        )
        for x_position, variant in enumerate(variants):
            short_label, full_label, color, marker = DISPLAY[variant]
            value = float(block.loc[variant, "HV_fixed_median"])
            ax.scatter(
                x_position,
                value,
                s=36,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                marker=marker,
                zorder=3,
                label=full_label,
            )
        ax.set_title(f"$G={budget}$", fontsize=9.5, pad=5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(
            [DISPLAY[variant][0] for variant in variants],
            rotation=28,
            ha="right",
        )
        ax.set_xlim(-0.55, len(variants) - 0.45)
        ax.set_ylim(0.925, 1.105)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.85)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Median fixed-reference HV")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.90)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_e5_heldout_performance"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
