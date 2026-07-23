"""Render publication figures from frozen E4 analysis tables.

This plot-only script changes no scientific indicator.  It uses the amended
v6.1 CSV outputs and standardizes the visible behavior-cloning terminology to
"in-sample demonstration agreement".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results/resubmission/v6_mechanism/analysis_v6_1"


def render_mechanism_summary(analysis_dir: Path, out_dir: Path) -> None:
    medians = pd.read_csv(analysis_dir / "instance_seed_medians.csv")
    pairwise = pd.read_csv(analysis_dir / "pairwise_fixed_box_hv.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    m100 = pairwise[pairwise["Budget"] == 100]
    labels = [
        "State (BC)",
        "State (no BC)",
        "BC (base)",
        "BC (enhanced)",
        "State × BC",
        "Enhanced vs UCB",
    ]
    wanted = [
        "enhanced_vs_padded_base_with_bc",
        "enhanced_vs_padded_base_without_bc",
        "bc_effect_padded_base_state",
        "bc_effect_enhanced_state",
        "state_by_bc_difference_in_differences",
        "enhanced16_vs_ucb_g100",
    ]
    lookup = m100.set_index("contrast")
    values = np.asarray([lookup.loc[name, "median_delta_oriented"] for name in wanted])
    low = np.asarray([lookup.loc[name, "bootstrap_ci_low"] for name in wanted])
    high = np.asarray([lookup.loc[name, "bootstrap_ci_high"] for name in wanted])
    xpos = np.arange(len(values))
    axes[0].errorbar(
        xpos,
        values,
        yerr=[values - low, high - values],
        fmt="o",
        color="#2F5597",
        ecolor="#7F8FA6",
        capsize=3,
    )
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set_xticks(xpos, labels, rotation=45, ha="right")
    axes[0].set_ylabel("Median paired ΔHV")
    axes[0].set_title("(a) G = 100 mechanism contrasts")

    updates = medians[medians["Budget"] == 200]
    order = ["EnhancedBC_R8", "EnhancedBC_R16", "EnhancedBC_R32"]
    data = [
        updates.loc[
            updates["variant"] == variant, "PPO_action_effective_updates"
        ].to_numpy()
        for variant in order
    ]
    axes[1].boxplot(data, tick_labels=["R8", "R16", "R32"], showfliers=False)
    axes[1].set_ylabel("Action-effective PPO updates")
    axes[1].set_title("(b) Updates before later actions")

    selected = medians[
        (medians["Budget"] == 100)
        & medians["variant"].isin(["BasePaddedBC_R16", "EnhancedBC_R16"])
    ]
    padded = selected[selected["variant"] == "BasePaddedBC_R16"].sort_values(
        ["dataset", "instance"]
    )
    enhanced = selected[selected["variant"] == "EnhancedBC_R16"].sort_values(
        ["dataset", "instance"]
    )
    axes[2].scatter(
        padded["BC_final_accuracy"],
        enhanced["BC_final_accuracy"],
        color="#4C956C",
        s=24,
    )
    lower = max(0.0, float(selected["BC_final_accuracy"].min()) - 0.04)
    upper = min(1.0, float(selected["BC_final_accuracy"].max()) + 0.04)
    limits = [lower, upper]
    axes[2].plot(limits, limits, ls="--", color="#777777", lw=0.8)
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel("Padded-base demo. agreement")
    axes[2].set_ylabel("Enhanced-state demo. agreement")
    axes[2].set_title("(c) In-sample imitation fit")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v6_mechanism_robustness.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "fig_v6_mechanism_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def render_bc_diagnostics(analysis_dir: Path, out_dir: Path) -> None:
    curves = pd.read_csv(analysis_dir / "bc_epoch_curve_summary.csv")
    confusion = pd.read_csv(analysis_dir / "bc_confusion_summary.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    variants = ["BasePaddedBC_R16", "EnhancedBC_R16"]
    labels = ["Capacity-matched padded", "Enhanced UCB context"]
    colors = ["#4C78A8", "#4C956C"]
    for variant, label, color in zip(variants, labels, colors):
        data = curves[
            (curves["Budget"] == 100) & (curves["variant"] == variant)
        ].sort_values("epoch")
        epoch = data["epoch"].to_numpy(dtype=float)
        axes[0, 0].plot(epoch, data["loss_median"], color=color, lw=1.6, label=label)
        axes[0, 0].fill_between(
            epoch,
            data["loss_q25"],
            data["loss_q75"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axes[0, 1].plot(
            epoch,
            data["accuracy_median"],
            color=color,
            lw=1.6,
            label=label,
        )
        axes[0, 1].fill_between(
            epoch,
            data["accuracy_q25"],
            data["accuracy_q75"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axes[0, 0].set(xlabel="BC epoch", ylabel="Training cross-entropy")
    axes[0, 0].set_title("(a) In-sample BC loss")
    axes[0, 1].set(
        xlabel="BC epoch", ylabel="Top-one in-sample demo. agreement"
    )
    axes[0, 1].set_title("(b) In-sample demonstration agreement")
    for axis in axes[0]:
        axis.legend(frameon=False, loc="best")
        axis.grid(axis="y", alpha=0.2)

    action_labels = ["C1", "C2", "C3", "C4", "C5", "M1", "M2", "M3", "M4", "M5"]
    matrices = []
    for variant in variants:
        data = confusion[
            (confusion["Budget"] == 100) & (confusion["variant"] == variant)
        ]
        matrix = (
            data.pivot(
                index="true_action",
                columns="predicted_action",
                values="row_proportion",
            )
            .reindex(index=range(10), columns=range(10))
            .to_numpy(dtype=float)
        )
        matrices.append(matrix)
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices)
    image_handle = None
    for axis, matrix, title in zip(
        axes[1],
        matrices,
        ["(c) Padded-state confusion", "(d) Enhanced-state confusion"],
    ):
        image_handle = axis.imshow(
            matrix, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal"
        )
        axis.set_xticks(range(10), action_labels, rotation=45, ha="right")
        axis.set_yticks(range(10), action_labels)
        axis.set_xlabel("Predicted action")
        axis.set_ylabel("UCB demonstration action")
        axis.set_title(title)
    fig.colorbar(
        image_handle,
        ax=[axes[1, 0], axes[1, 1]],
        label="Row-normalized proportion",
        shrink=0.82,
    )
    fig.savefig(out_dir / "fig_v6_bc_training_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "fig_v6_bc_training_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_mechanism_summary(args.analysis_dir.resolve(), args.out_dir.resolve())
    render_bc_diagnostics(args.analysis_dir.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
