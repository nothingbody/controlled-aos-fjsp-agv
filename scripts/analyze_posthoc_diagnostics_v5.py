"""Post-hoc diagnostics for the frozen SA-AOS v5 experiment.

This script does not rerun an optimizer.  It derives two reviewer-facing
diagnostics from the frozen CSV and front-pickle artifacts:

1. behavior-cloning (BC) training accuracy versus the within-demonstration
   majority-action baseline; and
2. exploratory instance-level heterogeneity in the paired common-reference
   hypervolume difference, SA-AOS minus UCB-only.

Outputs are written to
``results/resubmission/v5/posthoc_diagnostics``.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT_ROOT = ROOT / "results" / "resubmission" / "v5"
OUT = RESULT_ROOT / "posthoc_diagnostics"
N_ACTIONS = 10


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def read_front(relative_path: str) -> dict:
    path = ROOT / Path(relative_path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def action_baseline_rows(runs: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows: list[dict] = []
    selected = runs.loc[runs["variant"].eq("AdaptiveSAOS")].copy()
    for record in selected.to_dict("records"):
        n_demo = int(record["BC_samples"])
        if n_demo <= 0:
            continue
        payload = read_front(record["front_pickle"])
        labels = np.asarray(payload["operator_sequence"][:n_demo], dtype=int)
        counts = np.bincount(labels, minlength=N_ACTIONS)
        probabilities = counts[counts > 0] / labels.size
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        majority = float(counts.max() / labels.size)
        row = {
            "experiment": experiment,
            "dataset": record["dataset"],
            "instance": record["instance"],
            "seed": int(record["seed"]),
            "budget": int(record.get("Budget", 100)),
            "bc_samples": n_demo,
            "observed_actions": int(np.count_nonzero(counts)),
            "majority_action_share": majority,
            "normalized_label_entropy": entropy / math.log(N_ACTIONS),
            "bc_final_training_accuracy": float(record["BC_final_accuracy"]),
            "bc_accuracy_minus_majority": (
                float(record["BC_final_accuracy"]) - majority
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bc(bc: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "bc_samples",
        "observed_actions",
        "majority_action_share",
        "normalized_label_entropy",
        "bc_final_training_accuracy",
        "bc_accuracy_minus_majority",
    ]
    rows: list[dict] = []
    for (experiment, budget), group in bc.groupby(["experiment", "budget"]):
        row: dict[str, float | int | str] = {
            "experiment": experiment,
            "budget": int(budget),
            "runs": int(len(group)),
            "fraction_bc_above_majority": float(
                (group["bc_accuracy_minus_majority"] > 0).mean()
            ),
            "fraction_bc_equal_or_below_majority": float(
                (group["bc_accuracy_minus_majority"] <= 0).mean()
            ),
        }
        for metric in metrics:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_q25"] = float(group[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(group[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["experiment", "budget"])


def load_instance_features() -> pd.DataFrame:
    from data.loader import load_benchmark_set

    directory_map = {
        "Brandimarte": ROOT / "data" / "benchmarks" / "brandimarte",
        "Hurink_edata": ROOT / "data" / "benchmarks" / "hurink_edata",
    }
    rows: list[dict] = []
    for dataset, directory in directory_map.items():
        for filename, instance in load_benchmark_set(str(directory)):
            rows.append(
                {
                    "dataset": dataset,
                    "instance": filename,
                    "jobs": int(instance.num_jobs),
                    "machines": int(instance.num_machines),
                    "operations": int(instance.total_operations),
                }
            )
    return pd.DataFrame(rows)


def build_heterogeneity_table(
    e1: pd.DataFrame, energy: pd.DataFrame, bc_e1: pd.DataFrame
) -> pd.DataFrame:
    keys = ["dataset", "instance"]
    seed_medians = (
        e1.groupby(keys + ["variant"], as_index=False)
        .agg(
            hv=("HV_common", "median"),
            front_size=("NSol", "median"),
            transition_gen=("Transition_gen", "median"),
            ppo_updates=("PPO_update_count", "median"),
            bc_training_accuracy=("BC_final_accuracy", "median"),
        )
    )
    pivot = seed_medians.pivot(index=keys, columns="variant", values="hv")
    paired = (
        pivot[["AdaptiveSAOS", "UCBOnly"]]
        .rename(
            columns={"AdaptiveSAOS": "hv_sa_aos", "UCBOnly": "hv_ucb_only"}
        )
        .reset_index()
    )
    paired["delta_hv_sa_aos_minus_ucb"] = (
        paired["hv_sa_aos"] - paired["hv_ucb_only"]
    )

    saos = seed_medians.loc[
        seed_medians["variant"].eq("AdaptiveSAOS"),
        keys
        + [
            "front_size",
            "transition_gen",
            "ppo_updates",
            "bc_training_accuracy",
        ],
    ]
    bc_instance = (
        bc_e1.groupby(keys, as_index=False)
        .agg(
            majority_action_share=("majority_action_share", "median"),
            normalized_label_entropy=("normalized_label_entropy", "median"),
            bc_accuracy_minus_majority=("bc_accuracy_minus_majority", "median"),
        )
    )
    features = load_instance_features()
    energy_keep = energy[
        keys
        + [
            "agv_energy_share_median",
            "cmax_energy_correlation",
            "transport_time_ratio_realized",
        ]
    ].copy()
    table = paired.merge(saos, on=keys, validate="one_to_one")
    table = table.merge(features, on=keys, validate="one_to_one")
    table = table.merge(energy_keep, on=keys, validate="one_to_one")
    table = table.merge(bc_instance, on=keys, validate="one_to_one")
    table["family"] = np.where(
        table["dataset"].eq("Brandimarte"), "Mk", "la"
    )
    return table.sort_values(keys).reset_index(drop=True)


def heterogeneity_correlations(table: pd.DataFrame) -> pd.DataFrame:
    outcome = "delta_hv_sa_aos_minus_ucb"
    predictors = [
        "jobs",
        "machines",
        "operations",
        "front_size",
        "agv_energy_share_median",
        "cmax_energy_correlation",
        "transition_gen",
        "ppo_updates",
        "bc_training_accuracy",
        "majority_action_share",
        "normalized_label_entropy",
        "bc_accuracy_minus_majority",
    ]
    rows: list[dict] = []
    for predictor in predictors:
        valid = table[[outcome, predictor]].dropna()
        rho, p_value = spearmanr(valid[outcome], valid[predictor])
        rows.append(
            {
                "predictor": predictor,
                "n": int(len(valid)),
                "spearman_rho": float(rho),
                "p_unadjusted": float(p_value),
            }
        )
    adjusted = holm_adjust([row["p_unadjusted"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm_exploratory"] = p_adjusted
    return pd.DataFrame(rows).sort_values("p_unadjusted").reset_index(drop=True)


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for family, group in table.groupby("family"):
        delta = group["delta_hv_sa_aos_minus_ucb"]
        rows.append(
            {
                "family": family,
                "instances": int(len(group)),
                "sa_aos_wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "sa_aos_losses": int((delta < 0).sum()),
                "delta_hv_median": float(delta.median()),
                "delta_hv_q25": float(delta.quantile(0.25)),
                "delta_hv_q75": float(delta.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def plot_paired_differences(table: pd.DataFrame) -> None:
    plot = table.sort_values("delta_hv_sa_aos_minus_ucb").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.axhline(0.0, color="0.25", linewidth=0.9)
    family_styles = {
        "Mk": {"color": "#D55E00", "marker": "o", "label": "Brandimarte (Mk)"},
        "la": {"color": "#0072B2", "marker": "^", "label": "Hurink edata (la)"},
    }
    for family, style in family_styles.items():
        mask = plot["family"].eq(family)
        ax.scatter(
            np.flatnonzero(mask),
            plot.loc[mask, "delta_hv_sa_aos_minus_ucb"],
            color=style["color"],
            marker=style["marker"],
            s=30,
            edgecolors="white",
            linewidths=0.45,
            label=style["label"],
            zorder=3,
        )
    ax.set_xlabel("Instances ordered by paired difference")
    ax.set_ylabel(r"Common-reference $\Delta$HV (SA-AOS $-$ UCB-only)")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(OUT / "paired_difference_sa_aos_vs_ucb.pdf")
    fig.savefig(OUT / "paired_difference_sa_aos_vs_ucb.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e1 = pd.read_csv(RESULT_ROOT / "e1_aos" / "analysis" / "runs_with_common_hv.csv")
    e3 = pd.read_csv(RESULT_ROOT / "e3_budget" / "runs.csv")
    energy = pd.read_csv(RESULT_ROOT / "energy_audit" / "energy_construct_by_instance.csv")

    bc_e1 = action_baseline_rows(e1, "E1")
    bc_e3 = action_baseline_rows(e3, "E3")
    bc = pd.concat([bc_e1, bc_e3], ignore_index=True)
    bc_summary = summarize_bc(bc)
    bc.to_csv(OUT / "bc_run_diagnostics.csv", index=False)
    bc_summary.to_csv(OUT / "bc_summary.csv", index=False)

    heterogeneity = build_heterogeneity_table(e1, energy, bc_e1)
    correlations = heterogeneity_correlations(heterogeneity)
    families = family_summary(heterogeneity)
    heterogeneity.to_csv(OUT / "instance_heterogeneity.csv", index=False)
    correlations.to_csv(OUT / "instance_heterogeneity_spearman.csv", index=False)
    families.to_csv(OUT / "instance_family_summary.csv", index=False)
    plot_paired_differences(heterogeneity)

    summary = {
        "protocol": str(e1["Protocol"].iloc[0]),
        "analysis_status": "exploratory_posthoc_no_optimizer_rerun",
        "bc": bc_summary.to_dict("records"),
        "family_heterogeneity": families.to_dict("records"),
        "spearman": correlations.to_dict("records"),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(bc_summary.to_string(index=False))
    print("\nFamily summary")
    print(families.to_string(index=False))
    print("\nExploratory Spearman correlations")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
