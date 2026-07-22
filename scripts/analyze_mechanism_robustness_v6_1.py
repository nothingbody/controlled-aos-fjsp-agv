"""Transparent amended analysis after the v6 fixed-reference gate stopped.

This script does not alter or replace the executed v6 protocol.  It preserves
the frozen v5 normalization and original 1.1 reference box, reports excluded
out-of-box points, and adds a post-audit 1.5-reference sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_mechanism_robustness_v6 as base


DEFAULT_IN = ROOT / "results/resubmission/v6_mechanism/runs.csv"
DEFAULT_OUT = ROOT / "results/resubmission/v6_mechanism/analysis_v6_1"
AMENDMENT = ROOT / "SCI_Paper/MECHANISM_ROBUSTNESS_ANALYSIS_AMENDMENT_V6_1.md"
EXPANDED_REFERENCE = np.asarray([1.5, 1.5, 1.5], dtype=float)
OBJECTIVE_LABELS = ("Cmax", "Energy", "Workload")
SELECTED_INSTANCES = {
    "Brandimarte": ("Mk01.fjs", "Mk03.fjs", "Mk05.fjs", "Mk08.fjs", "Mk10.fjs"),
    "Hurink_edata": ("la01.fjs", "la10.fjs", "la20.fjs", "la30.fjs", "la40.fjs"),
}
BENCHMARK_DIRS = {
    "Brandimarte": ROOT / "data/benchmarks/brandimarte",
    "Hurink_edata": ROOT / "data/benchmarks/hurink_edata",
}
BC_VARIANTS = {
    "Original24BC_R16", "BasePaddedBC_R16", "EnhancedBC_R16",
    "EnhancedBC_R8", "EnhancedBC_R32",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_front_manifest(frame: pd.DataFrame, completion: dict) -> str:
    records = [
        f"{row.front_pickle}:{row.Front_sha256}"
        for row in frame.itertuples(index=False)
    ]
    observed = hashlib.sha256(
        "\n".join(sorted(records)).encode("utf-8")
    ).hexdigest()
    if observed != completion.get("front_manifest_sha256"):
        raise RuntimeError("CSV front manifest differs from completion marker")
    return observed


def amended_indicators(frame, csv_path, normalization, references):
    enriched, audit, feature = base.add_indicators(
        frame, csv_path, normalization, references
    )
    fixed_values = []
    expanded_values = []
    outside_counts = []
    inside_counts = []
    outside_dimension_counts = []
    outside_dimension_maxima = []

    for row in frame.to_dict(orient="records"):
        key = (row["dataset"], row["instance"], int(row["Budget"]))
        points, _, _, _ = base.load_objectives(
            row["front_pickle"], csv_path, expected=row
        )
        normalized = base.stable_unique_rows(
            base.normalize(points, normalization[key])
        )
        projected = base.nondominated(normalized)
        fixed_reference = np.asarray(normalization[key]["reference"], dtype=float)
        inside_mask = np.all(projected <= fixed_reference + 1e-12, axis=1)
        inside = projected[inside_mask]
        above = projected > fixed_reference + 1e-12
        dimension_counts = np.sum(above, axis=0).astype(int)
        dimension_maxima = [
            float(np.max(projected[above[:, index], index]))
            if dimension_counts[index] else None
            for index in range(projected.shape[1])
        ]
        if len(inside) == 0:
            raise RuntimeError(
                f"no point remains inside fixed reference for {key}"
            )
        if not np.all(projected <= EXPANDED_REFERENCE + 1e-12):
            raise RuntimeError("expanded reference 1.5 does not dominate all points")

        fixed_values.append(float(HV(ref_point=fixed_reference)(inside)))
        expanded_values.append(
            float(HV(ref_point=EXPANDED_REFERENCE)(projected))
        )
        outside_counts.append(int(np.sum(~inside_mask)))
        inside_counts.append(int(np.sum(inside_mask)))
        outside_dimension_counts.append(
            json.dumps(dict(zip(OBJECTIVE_LABELS, dimension_counts.tolist())))
        )
        outside_dimension_maxima.append(
            json.dumps(dict(zip(OBJECTIVE_LABELS, dimension_maxima)))
        )

    enriched["HV_fixed"] = fixed_values
    enriched["HV_expanded_1p5"] = expanded_values
    enriched["Outside_reference_points"] = outside_counts
    audit["outside_reference_points"] = outside_counts
    audit["inside_reference_points"] = inside_counts
    audit["outside_dimension_counts"] = outside_dimension_counts
    audit["outside_dimension_maxima"] = outside_dimension_maxima
    audit["amended_fixed_box_hv_valid"] = np.asarray(inside_counts) > 0
    return enriched, audit, feature


def instance_selection_audit():
    """Describe the frozen E4 subset from source files, without using outcomes."""
    records = []
    for dataset, directory in BENCHMARK_DIRS.items():
        dimensions = []
        for path in sorted(directory.glob("*.fjs")):
            lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
            jobs, machines = int(lines[0][0]), int(lines[0][1])
            operations = sum(int(lines[index + 1][0]) for index in range(jobs))
            dimensions.append((path.name, jobs, machines, operations))
        operation_values = pd.Series([row[3] for row in dimensions], dtype=float)
        ranks = operation_values.rank(method="average").to_numpy(dtype=float)
        percentiles = 100.0 * (ranks - 1.0) / max(len(dimensions) - 1, 1)
        selected = set(SELECTED_INSTANCES[dataset])
        for row, percentile in zip(dimensions, percentiles):
            if row[0] not in selected:
                continue
            if percentile < 33.333:
                stratum = "lower operation-count stratum"
            elif percentile < 66.667:
                stratum = "central operation-count stratum"
            else:
                stratum = "upper operation-count stratum"
            records.append(
                {
                    "dataset": dataset,
                    "instance": row[0],
                    "jobs": row[1],
                    "machines": row[2],
                    "operations": row[3],
                    "operation_count_percentile_within_family": percentile,
                    "selection_rationale": (
                        f"pre-outcome coverage of the {stratum}; "
                        "selection did not use controller performance"
                    ),
                }
            )
    result = pd.DataFrame(records)
    expected = sum(len(value) for value in SELECTED_INSTANCES.values())
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} selected instances, observed {len(result)}")
    return result


def exploratory_r8_ucb_contrasts(medians):
    """Post-hoc R8-versus-UCB transparency check, outside confirmatory families."""
    records = []
    endpoints = [
        ("HV_fixed", True, "amended primary scale used descriptively"),
        ("HV_expanded_1p5", True, "post-audit expanded-reference sensitivity"),
        ("IGDplus_fixed", False, "prespecified supportive sensitivity"),
    ]
    for offset, (metric, higher_is_better, endpoint_role) in enumerate(endpoints):
        pivot = medians.pivot_table(
            index=["dataset", "instance", "Budget"],
            columns="variant", values=metric,
        ).reset_index()
        record = base.comparison_record(
            pivot, 200, "EnhancedBC_R8", "UCBOnly",
            "exploratory_R8_UCB", "rollout8_vs_ucb_g200_posthoc",
            metric, 1.0 if higher_is_better else -1.0, 91 + offset,
        )
        record["endpoint_role"] = endpoint_role
        record["multiplicity_note"] = (
            "post hoc and unadjusted; outside every confirmatory family; "
            "not usable for variant selection or superiority claims"
        )
        records.append(record)
    return pd.DataFrame(records)


def instance_medians(frame):
    metrics = [
        "HV_fixed", "HV_expanded_1p5", "IGDplus_fixed", "Cmax_best",
        "TEC_best", "WB_best", "NSol", "Time", "Learning_time",
        "Transition_gen", "BC_pre_accuracy", "BC_final_accuracy",
        "BC_pre_loss", "BC_final_loss", "BC_pre_post_KL",
        "Demo_nn_disagreement", "Demo_ucb_margin_median",
        "Demo_forced_fraction", "PPO_action_effective_updates",
        "PPO_terminal_full_updates", "PPO_terminal_residual_updates",
        "PPO_optimizer_steps", "PPO_approx_KL_mean", "PPO_entropy_mean",
        "PPO_gradient_norm_mean", "PPO_explained_variance_pre_mean",
        "Reward_HV_corr", "Outside_reference_points",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]
    return (
        frame.groupby(
            ["dataset", "instance", "Budget", "variant"], sort=True
        )[metrics]
        .median()
        .reset_index()
    )


def diagnostic_contrasts(medians):
    """Return explicitly exploratory mechanism diagnostics.

    These contrasts diagnose imitation and controller mechanics; they are not
    members of the confirmatory optimization-performance families.
    """
    specifications = [
        ("BC_final_accuracy", 1.0, "enhanced_minus_padded_bc_accuracy"),
        ("BC_final_loss", -1.0, "enhanced_minus_padded_bc_loss"),
        ("BC_pre_post_KL", 1.0, "enhanced_minus_padded_pre_post_kl"),
        ("Demo_nn_disagreement", -1.0, "enhanced_minus_padded_nn_disagreement"),
    ]
    records = []
    for offset, (metric, orientation, name) in enumerate(specifications):
        pivot = medians[medians["Budget"] == 100].pivot_table(
            index=["dataset", "instance"], columns="variant", values=metric
        )
        raw_delta = (
            pivot["EnhancedBC_R16"] - pivot["BasePaddedBC_R16"]
        ).dropna().to_numpy(dtype=float)
        oriented = orientation * raw_delta
        statistic, p_raw, n_nonzero = base.exact_signed_rank(
            oriented, base.TIE_TOL
        )
        ci_low, ci_high = base.bootstrap_ci(
            raw_delta, seed=20260801 + offset
        )
        records.append(
            {
                "contrast": name,
                "metric": metric,
                "Budget": 100,
                "lhs": "EnhancedBC_R16",
                "rhs": "BasePaddedBC_R16",
                "n_instances": int(len(raw_delta)),
                "median_raw_delta_lhs_minus_rhs": float(np.median(raw_delta)),
                "bootstrap_ci_low_raw_delta": ci_low,
                "bootstrap_ci_high_raw_delta": ci_high,
                "favorable_direction": "higher" if orientation > 0 else "lower",
                "wins_in_favorable_direction": int(
                    np.sum(oriented > base.TIE_TOL)
                ),
                "ties": int(np.sum(np.abs(oriented) <= base.TIE_TOL)),
                "losses_in_favorable_direction": int(
                    np.sum(oriented < -base.TIE_TOL)
                ),
                "rank_biserial_favorable": base.rank_biserial_from_diff(
                    oriented, base.TIE_TOL
                ),
                "wilcoxon_statistic": statistic,
                "p_raw_unadjusted": p_raw,
                "n_nonzero": n_nonzero,
                "endpoint_role": "exploratory mechanism diagnostic",
                "multiplicity_note": (
                    "unadjusted; not a confirmatory performance test"
                ),
                "test_method": (
                    "exact sign enumeration of average signed ranks"
                ),
            }
        )
    return pd.DataFrame(records)


def bc_training_summaries(frame):
    """Expand archived BC traces and aggregate terminal confusion matrices."""
    curve_rows = []
    confusion_rows = []
    bc = frame[frame["variant"].isin(BC_VARIANTS) & frame["use_bc"].astype(bool)]
    for row in bc.itertuples(index=False):
        losses = json.loads(row.BC_epoch_loss)
        accuracies = json.loads(row.BC_epoch_accuracy)
        if len(losses) != len(accuracies):
            raise RuntimeError("BC loss and accuracy traces differ in length")
        for epoch, (loss, accuracy) in enumerate(
            zip(losses, accuracies), start=1
        ):
            curve_rows.append(
                {
                    "dataset": row.dataset,
                    "instance": row.instance,
                    "Budget": int(row.Budget),
                    "variant": row.variant,
                    "seed": int(row.seed),
                    "epoch": epoch,
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                }
            )
        confusion = np.asarray(json.loads(row.BC_confusion_matrix), dtype=int)
        if confusion.shape != (10, 10):
            raise RuntimeError(
                f"unexpected BC confusion shape {confusion.shape}"
            )
        for true_action in range(10):
            for predicted_action in range(10):
                confusion_rows.append(
                    {
                        "Budget": int(row.Budget),
                        "variant": row.variant,
                        "true_action": true_action,
                        "predicted_action": predicted_action,
                        "count": int(confusion[true_action, predicted_action]),
                    }
                )

    curves = pd.DataFrame(curve_rows)
    curve_summary = (
        curves.groupby(["Budget", "variant", "epoch"], sort=True)
        .agg(
            runs=("loss", "size"),
            loss_median=("loss", "median"),
            loss_q25=("loss", lambda x: float(np.quantile(x, 0.25))),
            loss_q75=("loss", lambda x: float(np.quantile(x, 0.75))),
            accuracy_median=("accuracy", "median"),
            accuracy_q25=("accuracy", lambda x: float(np.quantile(x, 0.25))),
            accuracy_q75=("accuracy", lambda x: float(np.quantile(x, 0.75))),
        )
        .reset_index()
    )
    confusion = (
        pd.DataFrame(confusion_rows)
        .groupby(
            ["Budget", "variant", "true_action", "predicted_action"],
            sort=True,
        )["count"]
        .sum()
        .reset_index()
    )
    row_totals = confusion.groupby(
        ["Budget", "variant", "true_action"]
    )["count"].transform("sum")
    confusion["row_proportion"] = confusion["count"] / row_totals
    return curves, curve_summary, confusion


def plot_summary(medians, pairwise, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    m100 = pairwise[pairwise["Budget"] == 100]
    labels = [
        "State (BC)", "State (no BC)", "BC (base)", "BC (enhanced)",
        "State x BC", "Enhanced - UCB",
    ]
    wanted = [
        "enhanced_vs_padded_base_with_bc",
        "enhanced_vs_padded_base_without_bc",
        "bc_effect_padded_base_state", "bc_effect_enhanced_state",
        "state_by_bc_difference_in_differences", "enhanced16_vs_ucb_g100",
    ]
    lookup = m100.set_index("contrast")
    values = np.asarray(
        [lookup.loc[name, "median_delta_oriented"] for name in wanted]
    )
    low = np.asarray([lookup.loc[name, "bootstrap_ci_low"] for name in wanted])
    high = np.asarray([lookup.loc[name, "bootstrap_ci_high"] for name in wanted])
    x = np.arange(len(values))
    axes[0].errorbar(
        x, values, yerr=[values - low, high - values], fmt="o",
        color="#2F5597", ecolor="#7F8FA6", capsize=3,
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("Median paired delta HV")
    axes[0].set_title("(a) M100 mechanism contrasts")

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
        padded["BC_final_accuracy"], enhanced["BC_final_accuracy"],
        color="#4C956C", s=24,
    )
    lower = max(
        0.0,
        float(selected["BC_final_accuracy"].min()) - 0.04,
    )
    upper = min(
        1.0,
        float(selected["BC_final_accuracy"].max()) + 0.04,
    )
    limits = [lower, upper]
    axes[2].plot(limits, limits, ls="--", color="gray", lw=0.8)
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel("Padded-base BC accuracy")
    axes[2].set_ylabel("Enhanced-state BC accuracy")
    axes[2].set_title("(c) In-sample imitation fit")

    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_robustness_summary.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "mechanism_robustness_summary.png", dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bc_training_diagnostics(curve_summary, confusion, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.2), constrained_layout=True
    )
    variants = ["BasePaddedBC_R16", "EnhancedBC_R16"]
    labels = ["Capacity-matched padded", "Enhanced UCB context"]
    colors = ["#4C78A8", "#4C956C"]
    for variant, label, color in zip(variants, labels, colors):
        data = curve_summary[
            (curve_summary["Budget"] == 100)
            & (curve_summary["variant"] == variant)
        ].sort_values("epoch")
        epoch = data["epoch"].to_numpy(dtype=float)
        axes[0, 0].plot(
            epoch, data["loss_median"], color=color, lw=1.6, label=label
        )
        axes[0, 0].fill_between(
            epoch, data["loss_q25"], data["loss_q75"],
            color=color, alpha=0.16, linewidth=0,
        )
        axes[0, 1].plot(
            epoch, data["accuracy_median"], color=color, lw=1.6,
            label=label,
        )
        axes[0, 1].fill_between(
            epoch, data["accuracy_q25"], data["accuracy_q75"],
            color=color, alpha=0.16, linewidth=0,
        )
    axes[0, 0].set(xlabel="BC epoch", ylabel="Training cross-entropy")
    axes[0, 0].set_title("(a) In-sample BC loss")
    axes[0, 1].set(xlabel="BC epoch", ylabel="Top-one training accuracy")
    axes[0, 1].set_title("(b) In-sample BC accuracy")
    for axis in axes[0]:
        axis.legend(frameon=False, loc="best")
        axis.grid(axis="y", alpha=0.2)

    action_labels = ["C1", "C2", "C3", "C4", "C5",
                     "M1", "M2", "M3", "M4", "M5"]
    matrices = []
    for variant in variants:
        data = confusion[
            (confusion["Budget"] == 100)
            & (confusion["variant"] == variant)
        ]
        matrix = data.pivot(
            index="true_action", columns="predicted_action",
            values="row_proportion",
        ).reindex(index=range(10), columns=range(10)).to_numpy(dtype=float)
        matrices.append(matrix)
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices)
    image_handle = None
    for axis, matrix, title in zip(
        axes[1], matrices,
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
        image_handle, ax=[axes[1, 0], axes[1, 1]],
        label="Row-normalized proportion", shrink=0.82,
    )
    fig.savefig(out_dir / "bc_training_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "bc_training_diagnostics.png", dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not AMENDMENT.is_file():
        raise RuntimeError(f"missing analysis amendment: {AMENDMENT}")
    run_manifest, completion = base.preflight(args.input)
    frame = pd.read_csv(args.input)
    completeness = base.validate_grid(frame, run_manifest)
    front_manifest_hash = verify_front_manifest(frame, completion)
    snapshot = base.load_frozen_snapshot(run_manifest)
    normalization = snapshot["normalization"]
    references = snapshot["reference_sets"]

    enriched, audit, feature = amended_indicators(
        frame, args.input, normalization, references
    )
    failed_gate_runs = int((~audit["reference_dominates_front"]).sum())
    if failed_gate_runs != 4:
        raise RuntimeError(
            f"amendment expected 4 failed-gate runs, observed {failed_gate_runs}"
        )
    if not audit["amended_fixed_box_hv_valid"].all():
        raise RuntimeError("amended fixed-box HV is not valid for every run")

    medians = instance_medians(enriched)
    selection_audit = instance_selection_audit()
    diagnostics = diagnostic_contrasts(medians)
    bc_curves, bc_curve_summary, bc_confusion = bc_training_summaries(enriched)
    pairwise_fixed = base.comparisons(
        medians, metric="HV_fixed", higher_is_better=True
    )
    pairwise_expanded = base.comparisons(
        medians, metric="HV_expanded_1p5", higher_is_better=True
    )
    pairwise_igd = base.comparisons(
        medians, metric="IGDplus_fixed", higher_is_better=False
    )
    pairwise_fixed["endpoint_role"] = "amended primary fixed-box HV"
    pairwise_expanded["endpoint_role"] = "post-audit expanded-reference sensitivity"
    pairwise_igd["endpoint_role"] = "prespecified supportive sensitivity"
    pairwise = pd.concat(
        [pairwise_fixed, pairwise_expanded, pairwise_igd], ignore_index=True
    )
    exploratory_r8_ucb = exploratory_r8_ucb_contrasts(medians)

    summary = (
        medians.groupby(["Budget", "variant"], sort=True)
        .agg(
            instances=("instance", "size"),
            hv_fixed_median=("HV_fixed", "median"),
            hv_expanded_median=("HV_expanded_1p5", "median"),
            igdplus_median=("IGDplus_fixed", "median"),
            effective_updates_median=("PPO_action_effective_updates", "median"),
            bc_pre_accuracy_median=("BC_pre_accuracy", "median"),
            bc_final_accuracy_median=("BC_final_accuracy", "median"),
            bc_pre_loss_median=("BC_pre_loss", "median"),
            bc_final_loss_median=("BC_final_loss", "median"),
            bc_kl_median=("BC_pre_post_KL", "median"),
            learning_time_median=("Learning_time", "median"),
        )
        .reset_index()
    )
    family_summary = (
        medians.groupby(["dataset", "Budget", "variant"], sort=True)
        .agg(
            instances=("instance", "size"),
            hv_fixed_median=("HV_fixed", "median"),
            hv_expanded_median=("HV_expanded_1p5", "median"),
            igdplus_median=("IGDplus_fixed", "median"),
            effective_updates_median=("PPO_action_effective_updates", "median"),
        )
        .reset_index()
    )

    enriched.to_csv(args.out_dir / "runs_with_amended_indicators.csv", index=False)
    audit.to_csv(args.out_dir / "reference_and_truncation_audit.csv", index=False)
    affected_columns = [
        "dataset", "instance", "variant", "Budget", "seed", "front_points",
        "outside_reference_points", "inside_reference_points",
        "outside_dimension_counts", "outside_dimension_maxima", "front_sha256",
        "amended_fixed_box_hv_valid",
    ]
    audit.loc[
        audit["outside_reference_points"] > 0, affected_columns
    ].to_csv(args.out_dir / "affected_reference_fronts.csv", index=False)
    selection_audit.to_csv(
        args.out_dir / "e4_instance_selection_audit.csv", index=False
    )
    feature.to_csv(args.out_dir / "enhanced_feature_audit.csv", index=False)
    medians.to_csv(args.out_dir / "instance_seed_medians.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_all_endpoints.csv", index=False)
    pairwise_fixed.to_csv(args.out_dir / "pairwise_fixed_box_hv.csv", index=False)
    pairwise_expanded.to_csv(
        args.out_dir / "pairwise_expanded_reference_hv.csv", index=False
    )
    pairwise_igd.to_csv(args.out_dir / "pairwise_igdplus.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_variant.csv", index=False)
    family_summary.to_csv(
        args.out_dir / "summary_by_instance_family.csv", index=False
    )
    diagnostics.to_csv(
        args.out_dir / "mechanism_diagnostic_contrasts.csv", index=False
    )
    exploratory_r8_ucb.to_csv(
        args.out_dir / "exploratory_r8_vs_ucb.csv", index=False
    )
    bc_curves.to_csv(args.out_dir / "bc_epoch_curves_by_run.csv", index=False)
    bc_curve_summary.to_csv(
        args.out_dir / "bc_epoch_curve_summary.csv", index=False
    )
    bc_confusion.to_csv(
        args.out_dir / "bc_confusion_summary.csv", index=False
    )
    plot_summary(medians, pairwise_fixed, args.out_dir)
    plot_bc_training_diagnostics(
        bc_curve_summary, bc_confusion, args.out_dir
    )

    analysis_manifest = {
        **completeness,
        "analysis_protocol": "saos_v6_analysis_amendment_1_20260722",
        "original_reference_gate_failed_runs": failed_gate_runs,
        "outside_reference_points": int(audit["outside_reference_points"].sum()),
        "outside_reference_coordinates": int(
            audit["coordinates_above_reference"].sum()
        ),
        "affected_reference_audit_rows": int(
            (audit["outside_reference_points"] > 0).sum()
        ),
        "exploratory_r8_ucb_reported": True,
        "exploratory_r8_ucb_confirmatory": False,
        "fixed_reference": [1.1, 1.1, 1.1],
        "expanded_reference": EXPANDED_REFERENCE.tolist(),
        "front_manifest_sha256": front_manifest_hash,
        "reference_snapshot_sha256": run_manifest["reference_snapshot_sha256"],
        "amendment_sha256": sha256(AMENDMENT),
        "analysis_script_sha256": sha256(Path(__file__)),
        "status": "complete_with_declared_reference_amendment",
    }
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis_manifest, indent=2))


if __name__ == "__main__":
    main()
