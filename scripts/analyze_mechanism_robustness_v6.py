"""Analyze the prespecified v6 SA-AOS mechanism-robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_mechanism_robustness_v6 import (
    CONFIGS, PROTOCOL, SELECTED_INSTANCES, code_manifest, input_manifest,
)
from scripts.analyze_resubmission_v5 import (
    bootstrap_ci, holm_adjust, locate_front, nondominated,
    rank_biserial_from_diff, stable_unique_rows,
)


DEFAULT_IN = ROOT / "results/resubmission/v6_mechanism/runs.csv"
DEFAULT_OUT = ROOT / "results/resubmission/v6_mechanism/analysis"
REFERENCE_SNAPSHOT = ROOT / "results/resubmission/v6_mechanism/frozen_reference_snapshot.pkl"
TIE_TOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_snapshot(run_manifest):
    if not REFERENCE_SNAPSHOT.is_file():
        raise RuntimeError(f"missing frozen reference snapshot: {REFERENCE_SNAPSHOT}")
    observed_hash = sha256(REFERENCE_SNAPSHOT)
    expected_hash = run_manifest.get("reference_snapshot_sha256")
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"reference snapshot hash mismatch: {observed_hash} != {expected_hash}"
        )
    with REFERENCE_SNAPSHOT.open("rb") as stream:
        snapshot = pickle.load(stream)
    if snapshot.get("snapshot_protocol") != "saos_v6_fixed_v5_reference_20260722":
        raise RuntimeError("unexpected fixed-reference snapshot protocol")
    return snapshot


def validate_grid(frame, run_manifest):
    required = {
        "Protocol", "dataset", "instance", "variant", "Budget", "seed",
        "Population_size", "Max_generations", "Config_hash", "Code_hash",
        "Design_hash", "Input_hash", "Worker_count", "Front_sha256",
        "front_pickle",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    if set(frame["Protocol"].astype(str)) != {PROTOCOL}:
        raise RuntimeError(f"protocol mismatch: {sorted(frame['Protocol'].unique())}")
    key = ["dataset", "instance", "variant", "Budget", "seed"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise RuntimeError(f"duplicate keys: {frame.loc[duplicates, key].head().to_dict('records')}")
    expected = {
        (dataset, instance, config["variant"], budget, seed)
        for dataset, names in SELECTED_INSTANCES.items()
        for instance in names
        for budget, configs in CONFIGS.items()
        for config in configs
        for seed in range(42, 52)
    }
    observed = {
        (row.dataset, row.instance, row.variant, int(row.Budget), int(row.seed))
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError(
            f"grid mismatch missing={len(expected-observed)} unexpected={len(observed-expected)}"
        )
    if len(frame) != 1100:
        raise RuntimeError(f"expected 1100 rows, found {len(frame)}")
    if set(frame["Code_hash"].astype(str)) != {run_manifest["code_hash"]}:
        raise RuntimeError("CSV code hash does not match run manifest")
    if set(frame["Design_hash"].astype(str)) != {run_manifest["design_hash"]}:
        raise RuntimeError("CSV design hash does not match run manifest")
    if set(frame["Input_hash"].astype(str)) != {run_manifest["input_hash"]}:
        raise RuntimeError("CSV benchmark input hash does not match run manifest")
    if set(pd.to_numeric(frame["Population_size"])) != {100}:
        raise RuntimeError("population size differs from frozen value 100")
    if set(pd.to_numeric(frame["Worker_count"])) != {40}:
        raise RuntimeError("worker count differs from frozen value 40")
    if not np.array_equal(
        pd.to_numeric(frame["Max_generations"]).to_numpy(),
        pd.to_numeric(frame["Budget"]).to_numpy(),
    ):
        raise RuntimeError("Max_generations and Budget differ")
    return {
        "rows": int(len(frame)),
        "unique_keys": int(len(observed)),
        "instances": 10,
        "seeds": list(range(42, 52)),
        "budgets": [100, 200],
        "status": "complete", "code_hash": run_manifest["code_hash"],
        "design_hash": run_manifest["design_hash"],
        "input_hash": run_manifest["input_hash"],
    }


def load_objectives(raw_path, csv_path, expected=None):
    path = locate_front(str(raw_path), csv_path, ROOT)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if expected is not None:
        checks = {
            "protocol": (payload.get("protocol"), expected["Protocol"]),
            "dataset": (payload.get("dataset"), expected["dataset"]),
            "instance": (payload.get("instance"), expected["instance"]),
            "variant": (payload.get("variant"), expected["variant"]),
            "budget": (int(payload.get("budget", -1)), int(expected["Budget"])),
            "seed": (int(payload.get("seed", -1)), int(expected["seed"])),
            "config_hash": (payload.get("config_hash"), expected["Config_hash"]),
            "code_hash": (payload.get("code_hash"), expected["Code_hash"]),
            "design_hash": (payload.get("design_hash"), expected["Design_hash"]),
            "input_hash": (payload.get("input_hash"), expected["Input_hash"]),
            "population_size": (
                int(payload.get("population_size", -1)),
                int(expected["Population_size"]),
            ),
        }
        mismatches = {
            key: {"payload": actual, "csv": wanted}
            for key, (actual, wanted) in checks.items() if actual != wanted
        }
        if mismatches:
            raise RuntimeError(f"front metadata mismatch in {path}: {mismatches}")
    objectives = stable_unique_rows(np.asarray(payload["objectives"], dtype=float))
    if len(objectives) == 0 or objectives.shape[1] < 3:
        raise RuntimeError(f"invalid front in {path}: {objectives.shape}")
    if not np.isfinite(objectives[:, :3]).all():
        raise RuntimeError(f"non-finite objectives in {path}")
    observed_hash = sha256(path)
    if expected is not None and observed_hash != expected["Front_sha256"]:
        raise RuntimeError(
            f"front hash mismatch in {path}: {observed_hash} != "
            f"{expected['Front_sha256']}"
        )
    return objectives[:, :3], payload, path, observed_hash


def normalize(points, record):
    ideal = np.asarray(record["ideal"], dtype=float)
    nadir = np.asarray(record["nadir"], dtype=float)
    scale = np.where(nadir > ideal, nadir - ideal, 1.0)
    return (np.asarray(points, dtype=float) - ideal) / scale


def igd_plus(reference, approximation):
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    distances = []
    for target in reference:
        positive_deviation = np.maximum(approximation - target, 0.0)
        distances.append(np.linalg.norm(positive_deviation, axis=1).min())
    return float(np.mean(distances))


def add_indicators(frame, csv_path, normalization, references):
    rows = []
    audit = []
    feature_records = []
    for row in frame.to_dict(orient="records"):
        key = (row["dataset"], row["instance"], int(row["Budget"]))
        points, payload, path, front_hash = load_objectives(
            row["front_pickle"], csv_path, expected=row
        )
        normalized = stable_unique_rows(normalize(points, normalization[key]))
        projected = nondominated(normalized)
        reference_point = np.asarray(normalization[key]["reference"], dtype=float)
        dominated_by_reference = bool(np.all(projected <= reference_point + 1e-12))
        audit_row = {
            "dataset": row["dataset"],
            "instance": row["instance"],
            "variant": row["variant"],
            "Budget": int(row["Budget"]),
            "seed": int(row["seed"]),
            "normalized_min": float(projected.min()),
            "normalized_max": float(projected.max()),
            "coordinates_below_zero": int(np.sum(projected < 0.0)),
            "coordinates_above_one": int(np.sum(projected > 1.0)),
            "coordinates_above_reference": int(np.sum(projected > reference_point)),
            "reference_dominates_front": dominated_by_reference,
            "front_points": int(len(projected)),
            "front_path": str(path),
            "front_sha256": front_hash,
        }
        audit.append(audit_row)
        row["HV_fixed"] = (
            float(HV(ref_point=reference_point)(projected))
            if dominated_by_reference else np.nan
        )
        row["IGDplus_fixed"] = igd_plus(references[key], projected)
        rows.append(row)

        feature = payload.get("enhanced_feature_summary", {})
        for index, (mean, std, minimum, maximum) in enumerate(
            zip(
                feature.get("mean", []), feature.get("std", []),
                feature.get("min", []), feature.get("max", []),
            )
        ):
            feature_records.append(
                {
                    "dataset": row["dataset"], "instance": row["instance"],
                    "variant": row["variant"], "Budget": int(row["Budget"]),
                    "seed": int(row["seed"]), "feature_index": int(index + 25),
                    "mean": float(mean), "std": float(std),
                    "min": float(minimum), "max": float(maximum),
                    "constant": bool(abs(float(std)) < 1e-12),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(feature_records)


def seed_medians(frame):
    metrics = [
        "HV_fixed", "IGDplus_fixed", "Cmax_best", "TEC_best", "WB_best", "NSol",
        "Time", "Learning_time", "Transition_gen", "BC_final_accuracy",
        "BC_pre_post_KL", "Demo_nn_disagreement", "PPO_action_effective_updates",
        "PPO_terminal_full_updates", "PPO_terminal_residual_updates",
        "PPO_optimizer_steps", "PPO_approx_KL_mean", "PPO_entropy_mean",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]
    return (
        frame.groupby(["dataset", "instance", "Budget", "variant"], sort=True)[metrics]
        .median().reset_index()
    )


def exact_signed_rank(diff, tolerance=TIE_TOL):
    values = np.asarray(diff, dtype=float)
    values = values[np.abs(values) > tolerance]
    if len(values) == 0:
        return 0.0, 1.0, 0
    ranks = rankdata(np.abs(values), method="average")
    total = float(ranks.sum())
    observed_positive = float(ranks[values > 0].sum())
    observed_deviation = abs(observed_positive - total / 2.0)
    extreme = 0
    permutations = 2 ** len(values)
    for signs in itertools.product((False, True), repeat=len(values)):
        positive = float(ranks[np.asarray(signs, dtype=bool)].sum())
        if abs(positive - total / 2.0) >= observed_deviation - 1e-12:
            extreme += 1
    statistic = min(observed_positive, total - observed_positive)
    return statistic, float(extreme / permutations), int(len(values))


def comparison_record(
    pivot, budget, lhs, rhs, family, contrast, metric, orientation, seed_offset=0
):
    block = pivot[pivot["Budget"] == budget]
    delta = (
        orientation * (block[lhs] - block[rhs])
    ).dropna().to_numpy(dtype=float)
    statistic, p_raw, n_nonzero = exact_signed_rank(delta, TIE_TOL)
    ci_low, ci_high = bootstrap_ci(delta, seed=20260722 + seed_offset)
    return {
        "metric": metric, "family": family, "contrast": contrast, "Budget": budget,
        "lhs": lhs, "rhs": rhs, "n_instances": int(len(delta)),
        "median_delta_oriented": float(np.median(delta)),
        "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
        "wins": int(np.sum(delta > TIE_TOL)),
        "ties": int(np.sum(np.abs(delta) <= TIE_TOL)),
        "losses": int(np.sum(delta < -TIE_TOL)),
        "rank_biserial": rank_biserial_from_diff(delta, TIE_TOL),
        "wilcoxon_statistic": statistic, "p_raw": p_raw,
        "n_nonzero": n_nonzero,
        "test_method": "exact sign enumeration of average signed ranks",
    }


def comparisons(medians, metric="HV_fixed", higher_is_better=True):
    orientation = 1.0 if higher_is_better else -1.0
    pivot = medians.pivot_table(
        index=["dataset", "instance", "Budget"], columns="variant", values=metric
    ).reset_index()
    records = [
        comparison_record(pivot, 100, "EnhancedBC_R16", "BasePaddedBC_R16", "M100_state", "enhanced_vs_padded_base_with_bc", metric, orientation, 1),
        comparison_record(pivot, 100, "EnhancedNoBC_R16", "BasePaddedNoBC_R16", "M100_state", "enhanced_vs_padded_base_without_bc", metric, orientation, 2),
        comparison_record(pivot, 100, "BasePaddedBC_R16", "BasePaddedNoBC_R16", "M100_bc", "bc_effect_padded_base_state", metric, orientation, 3),
        comparison_record(pivot, 100, "EnhancedBC_R16", "EnhancedNoBC_R16", "M100_bc", "bc_effect_enhanced_state", metric, orientation, 4),
        comparison_record(pivot, 200, "EnhancedBC_R8", "EnhancedBC_R16", "M200_rollout", "rollout8_vs_16", metric, orientation, 5),
        comparison_record(pivot, 200, "EnhancedBC_R32", "EnhancedBC_R16", "M200_rollout", "rollout32_vs_16", metric, orientation, 6),
        comparison_record(pivot, 100, "EnhancedBC_R16", "UCBOnly", "UCB_boundary", "enhanced16_vs_ucb_g100", metric, orientation, 7),
        comparison_record(pivot, 200, "EnhancedBC_R16", "UCBOnly", "UCB_boundary", "enhanced16_vs_ucb_g200", metric, orientation, 8),
    ]

    block = pivot[pivot["Budget"] == 100]
    interaction = orientation * (
        (block["EnhancedBC_R16"] - block["EnhancedNoBC_R16"])
        - (block["BasePaddedBC_R16"] - block["BasePaddedNoBC_R16"])
    ).dropna().to_numpy(dtype=float)
    statistic, p_raw, n_nonzero = exact_signed_rank(interaction, TIE_TOL)
    ci_low, ci_high = bootstrap_ci(interaction, seed=20260731)
    records.append(
        {
            "metric": metric, "family": "M100_interaction", "contrast": "state_by_bc_difference_in_differences",
            "Budget": 100, "lhs": "(EnhancedBC-EnhancedNoBC)",
            "rhs": "(BasePaddedBC-BasePaddedNoBC)", "n_instances": int(len(interaction)),
            "median_delta_oriented": float(np.median(interaction)),
            "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            "wins": int(np.sum(interaction > TIE_TOL)),
            "ties": int(np.sum(np.abs(interaction) <= TIE_TOL)),
            "losses": int(np.sum(interaction < -TIE_TOL)),
            "rank_biserial": rank_biserial_from_diff(interaction, TIE_TOL),
            "wilcoxon_statistic": statistic, "p_raw": p_raw,
            "n_nonzero": n_nonzero,
            "test_method": "exact sign enumeration of average signed ranks",
        }
    )
    result = pd.DataFrame(records)
    result["p_holm_within_family"] = np.nan
    for family, indices in result.groupby("family").groups.items():
        result.loc[list(indices), "p_holm_within_family"] = holm_adjust(
            result.loc[list(indices), "p_raw"].to_numpy()
        )
    return result


def plot_summary(medians, pairwise, out_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    m100 = pairwise[pairwise["Budget"] == 100]
    labels = [
        "State (BC)", "State (no BC)", "BC (base)", "BC (enhanced)",
        "State×BC", "Enhanced−UCB",
    ]
    wanted = [
        "enhanced_vs_padded_base_with_bc", "enhanced_vs_padded_base_without_bc",
        "bc_effect_padded_base_state", "bc_effect_enhanced_state",
        "state_by_bc_difference_in_differences", "enhanced16_vs_ucb_g100",
    ]
    lookup = m100.set_index("contrast")
    values = [lookup.loc[name, "median_delta_oriented"] for name in wanted]
    low = [lookup.loc[name, "bootstrap_ci_low"] for name in wanted]
    high = [lookup.loc[name, "bootstrap_ci_high"] for name in wanted]
    x = np.arange(len(values))
    axes[0].errorbar(
        x, values, yerr=[np.asarray(values) - low, np.asarray(high) - values],
        fmt="o", color="#2F5597", ecolor="#7F8FA6", capsize=3,
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("Median paired ΔHV")
    axes[0].set_title("(a) M100 mechanism contrasts")

    updates = medians[medians["Budget"] == 200]
    order = ["EnhancedBC_R8", "EnhancedBC_R16", "EnhancedBC_R32"]
    data = [
        updates.loc[updates["variant"] == variant, "PPO_action_effective_updates"].to_numpy()
        for variant in order
    ]
    axes[1].boxplot(data, labels=["R8", "R16", "R32"], showfliers=False)
    axes[1].set_ylabel("Action-effective PPO updates")
    axes[1].set_title("(b) Updates before later actions")

    bc = medians[(medians["Budget"] == 100) & medians["variant"].isin(["BasePaddedBC_R16", "EnhancedBC_R16"])]
    base = bc[bc["variant"] == "BasePaddedBC_R16"].sort_values(["dataset", "instance"])
    enhanced = bc[bc["variant"] == "EnhancedBC_R16"].sort_values(["dataset", "instance"])
    axes[2].scatter(base["BC_final_accuracy"], enhanced["BC_final_accuracy"], color="#4C956C", s=24)
    limits = [0, max(1.0, float(bc["BC_final_accuracy"].max()) + 0.05)]
    axes[2].plot(limits, limits, ls="--", color="gray", lw=0.8)
    axes[2].set_xlim(limits); axes[2].set_ylim(limits)
    axes[2].set_xlabel("Base-state BC accuracy")
    axes[2].set_ylabel("Enhanced-state BC accuracy")
    axes[2].set_title("(c) In-sample imitation fit")

    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_robustness_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "mechanism_robustness_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def preflight(input_path):
    root = input_path.parent
    run_manifest_path = root / "run_manifest.json"
    completion_path = root / "pipeline_complete.json"
    failure_path = root / "failures.json"
    if failure_path.exists():
        raise RuntimeError(f"failure marker exists: {failure_path}")
    if not run_manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError("run_manifest.json and pipeline_complete.json are required")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if run_manifest.get("protocol") != PROTOCOL or completion.get("protocol") != PROTOCOL:
        raise RuntimeError("manifest protocol mismatch")
    if int(completion.get("rows", -1)) != 1100:
        raise RuntimeError(f"completion marker has rows={completion.get('rows')}")
    if completion.get("code_hash") != run_manifest.get("code_hash"):
        raise RuntimeError("completion/run-manifest code hashes differ")
    if completion.get("design_hash") != run_manifest.get("design_hash"):
        raise RuntimeError("completion/run-manifest design hashes differ")
    if completion.get("input_hash") != run_manifest.get("input_hash"):
        raise RuntimeError("completion/run-manifest input hashes differ")
    observed_code_hash = code_manifest()["code_hash"]
    if observed_code_hash != run_manifest.get("code_hash"):
        raise RuntimeError(
            f"current code differs from executed code: {observed_code_hash}"
        )
    observed_input_hash = input_manifest()["input_hash"]
    if observed_input_hash != run_manifest.get("input_hash"):
        raise RuntimeError(
            f"current benchmark inputs differ from executed inputs: {observed_input_hash}"
        )
    return run_manifest, completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest, completion = preflight(args.input)
    frame = pd.read_csv(args.input)
    completeness = validate_grid(frame, run_manifest)
    front_manifest_records = [
        f"{row.front_pickle}:{row.Front_sha256}"
        for row in frame.itertuples(index=False)
    ]
    observed_front_manifest_hash = hashlib.sha256(
        "\n".join(sorted(front_manifest_records)).encode("utf-8")
    ).hexdigest()
    if observed_front_manifest_hash != completion.get("front_manifest_sha256"):
        raise RuntimeError("CSV front manifest differs from completion marker")
    snapshot = load_frozen_snapshot(run_manifest)
    normalization = snapshot["normalization"]
    references = snapshot["reference_sets"]
    enriched, audit, feature = add_indicators(frame, args.input, normalization, references)
    audit.to_csv(args.out_dir / "fixed_reference_audit.csv", index=False)
    if not audit["reference_dominates_front"].all():
        failures = audit.loc[~audit["reference_dominates_front"]]
        raise RuntimeError(
            f"fixed reference fails to dominate {len(failures)} runs; inference stopped"
        )

    medians = seed_medians(enriched)
    pairwise_hv = comparisons(medians, metric="HV_fixed", higher_is_better=True)
    pairwise_igd = comparisons(
        medians, metric="IGDplus_fixed", higher_is_better=False
    )
    pairwise_hv["endpoint_role"] = "confirmatory primary"
    pairwise_igd["endpoint_role"] = "supportive sensitivity"
    pairwise = pd.concat([pairwise_hv, pairwise_igd], ignore_index=True)
    summary = (
        medians.groupby(["Budget", "variant"], sort=True)
        .agg(
            instances=("instance", "size"),
            hv_median=("HV_fixed", "median"),
            hv_q25=("HV_fixed", lambda x: x.quantile(0.25)),
            hv_q75=("HV_fixed", lambda x: x.quantile(0.75)),
            igdplus_median=("IGDplus_fixed", "median"),
            effective_updates_median=("PPO_action_effective_updates", "median"),
            terminal_updates_median=("PPO_terminal_residual_updates", "median"),
            bc_accuracy_median=("BC_final_accuracy", "median"),
            bc_kl_median=("BC_pre_post_KL", "median"),
            learning_time_median=("Learning_time", "median"),
        ).reset_index()
    )

    enriched.to_csv(args.out_dir / "runs_with_fixed_indicators.csv", index=False)
    medians.to_csv(args.out_dir / "instance_seed_medians.csv", index=False)
    pairwise.to_csv(args.out_dir / "prespecified_pairwise_all_metrics.csv", index=False)
    pairwise_hv.to_csv(args.out_dir / "prespecified_pairwise_hv.csv", index=False)
    pairwise_igd.to_csv(args.out_dir / "prespecified_pairwise_igdplus.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_variant.csv", index=False)
    feature.to_csv(args.out_dir / "enhanced_feature_audit.csv", index=False)
    family_summary = (
        medians.groupby(["dataset", "Budget", "variant"], sort=True)
        .agg(
            instances=("instance", "size"),
            hv_median=("HV_fixed", "median"),
            igdplus_median=("IGDplus_fixed", "median"),
            effective_updates_median=("PPO_action_effective_updates", "median"),
        ).reset_index()
    )
    family_summary.to_csv(args.out_dir / "summary_by_instance_family.csv", index=False)
    completeness.update(
        {
            "reference_failures": int((~audit["reference_dominates_front"]).sum()),
            "coordinates_below_zero": int(audit["coordinates_below_zero"].sum()),
            "coordinates_above_one": int(audit["coordinates_above_one"].sum()),
            "coordinates_above_reference": int(audit["coordinates_above_reference"].sum()),
            "reference_snapshot_sha256": run_manifest["reference_snapshot_sha256"],
            "fixed_reference_sets": {
                f"{dataset}/{instance}/g{budget}": int(len(points))
                for (dataset, instance, budget), points in references.items()
            },
        }
    )
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    plot_summary(medians, pairwise_hv, args.out_dir)
    print(json.dumps(completeness, indent=2))


if __name__ == "__main__":
    main()
