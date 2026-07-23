#!/usr/bin/env python3
"""Read-only diagnostics requested during the MC1/MC3/MC5 manuscript revision.

The script does not rerun an optimizer or alter any frozen result.  It:

1. counts action-effective PPO updates from archived rollout ledgers;
2. places adaptive SA-AOS, generation-zero PPO-only, the E4 rollout-8
   intervention, and E5 pretraining on one data-supply ladder;
3. compares archived SA-AOS and AdaptiveNoBC behavior at the E1 budget;
4. reconciles the 50-generation handover and effective-update counts; and
5. reports checkpoint-dependent instance-level E5 paired sensitivities.

The E5 instance-level tests are explicitly descriptive.  Instances within a
fold share a checkpoint and the five checkpoint training sets overlap, so the
reported p-values are not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def parse_rollouts(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def action_effective_updates(row: pd.Series, budget_col: str = "Budget") -> int:
    transition = int(row["Transition_gen"])
    budget = int(row[budget_col])
    if str(row.get("variant", "")) == "PPOOnly":
        transition = 0
    if transition < 0:
        return 0
    controlled_actions = budget - transition
    cumulative = np.cumsum(parse_rollouts(row["PPO_rollout_sizes"]))
    return int(np.sum(cumulative < controlled_actions))


def paired_summary(
    frame: pd.DataFrame,
    lhs: str,
    rhs: str,
    metric: str,
    keys: list[str],
) -> dict[str, object]:
    wide = frame.pivot_table(index=keys, columns="variant", values=metric, aggfunc="median")
    diff = (wide[lhs] - wide[rhs]).dropna()
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.empty:
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
        )
    return {
        "lhs": lhs,
        "rhs": rhs,
        "metric": metric,
        "n_blocks": int(diff.size),
        "lhs_median": float(wide[lhs].median()),
        "rhs_median": float(wide[rhs].median()),
        "paired_median_delta": float(diff.median()),
        "wins": int((diff > 1e-12).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
        "losses": int((diff < -1e-12).sum()),
        "wilcoxon_p_raw_descriptive": p_value,
    }


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * (len(ranked) - np.arange(len(ranked))))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return pd.Series(adjusted, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "results" / "resubmission" / "revision_readonly_mc"
    )
    output.mkdir(parents=True, exist_ok=True)

    v5 = root / "results" / "resubmission" / "v5"
    e1 = pd.read_csv(v5 / "e1_aos" / "analysis" / "runs_with_common_hv.csv")
    e3 = pd.read_csv(v5 / "e3_budget" / "runs.csv")
    e4 = pd.read_csv(root / "results" / "resubmission" / "v6_mechanism" / "runs.csv")
    e5_blocks = pd.read_csv(
        root
        / "results"
        / "resubmission"
        / "v7_cross_instance"
        / "analysis"
        / "contrast_instance_blocks.csv"
    )
    e5_runs = pd.read_csv(
        root
        / "results"
        / "resubmission"
        / "v7_cross_instance"
        / "analysis"
        / "runs_with_fixed_indicators.csv"
    )

    e3["PPO_action_effective_updates_readonly"] = e3.apply(action_effective_updates, axis=1)
    pressure_rows: list[dict[str, object]] = []
    for budget in (50, 100, 200):
        for variant in ("AdaptiveSAOS", "PPOOnly"):
            group = e3[(e3["Budget"] == budget) & (e3["variant"] == variant)]
            pressure_rows.append(
                {
                    "evidence_block": "E3",
                    "controller": variant,
                    "budget": budget,
                    "n_runs": int(len(group)),
                    "median_transition_gen": float(group["Transition_gen"].median()),
                    "median_terminal_inclusive_updates": float(group["PPO_update_count"].median()),
                    "median_action_effective_updates": float(
                        group["PPO_action_effective_updates_readonly"].median()
                    ),
                    "offline_objective_evaluations": 0,
                    "interpretation": (
                        "adaptive handover"
                        if variant == "AdaptiveSAOS"
                        else "generation-zero PPO pressure test"
                    ),
                }
            )

    e4_r8 = e4[(e4["variant"] == "EnhancedBC_R8") & (e4["Budget"] == 200)]
    pressure_rows.append(
        {
            "evidence_block": "E4",
            "controller": "EnhancedBC_R8",
            "budget": 200,
            "n_runs": int(len(e4_r8)),
            "median_transition_gen": float(e4_r8["Transition_gen"].median()),
            "median_terminal_inclusive_updates": float(e4_r8["PPO_update_count"].median()),
            "median_action_effective_updates": float(
                e4_r8["PPO_action_effective_updates"].median()
            ),
            "offline_objective_evaluations": 0,
            "interpretation": "short-rollout within-run PPO pressure test",
        }
    )
    pressure_rows.append(
        {
            "evidence_block": "E5",
            "controller": "Two-pass transferred PPO",
            "budget": 200,
            "n_runs": 2000,
            "median_transition_gen": np.nan,
            "median_terminal_inclusive_updates": 16369,
            "median_action_effective_updates": np.nan,
            "offline_objective_evaluations": 40200000,
            "interpretation": "cross-instance data-supply pressure test; totals, not medians",
        }
    )
    pd.DataFrame(pressure_rows).to_csv(output / "ppo_data_supply_ladder.csv", index=False)

    e1_subset = e1[e1["variant"].isin(["AdaptiveSAOS", "AdaptiveNoBC"])].copy()
    behavior_metrics = [
        "HV_common",
        "Entropy_all",
        "Entropy_last20",
        "PPO_entropy_mean",
    ]
    behavior = pd.DataFrame(
        [
            paired_summary(
                e1_subset,
                "AdaptiveSAOS",
                "AdaptiveNoBC",
                metric,
                ["dataset", "instance"],
            )
            for metric in behavior_metrics
        ]
    )
    behavior.to_csv(output / "e1_bc_vs_nobc_behavior.csv", index=False)

    short = e3[(e3["Budget"] == 50) & (e3["variant"] == "AdaptiveSAOS")].copy()
    short["effective_updates"] = short["PPO_action_effective_updates_readonly"]
    reconciliation = {
        "n_runs": int(len(short)),
        "guard_handover_runs": int((short["Transition_reason"] == "coverage_latest_guard").sum()),
        "stagnation_handover_runs": int(
            (short["Transition_reason"] == "coverage_stagnation").sum()
        ),
        "zero_action_effective_update_runs": int((short["effective_updates"] == 0).sum()),
        "one_action_effective_update_runs": int((short["effective_updates"] == 1).sum()),
        "guard_runs_with_zero_effective_updates": int(
            (
                (short["Transition_reason"] == "coverage_latest_guard")
                & (short["effective_updates"] == 0)
            ).sum()
        ),
        "stagnation_runs_with_zero_effective_updates": int(
            (
                (short["Transition_reason"] == "coverage_stagnation")
                & (short["effective_updates"] == 0)
            ).sum()
        ),
        "stagnation_runs_with_one_effective_update": int(
            (
                (short["Transition_reason"] == "coverage_stagnation")
                & (short["effective_updates"] == 1)
            ).sum()
        ),
        "transition_generation_counts": {
            str(int(k)): int(v)
            for k, v in short["Transition_gen"].value_counts().sort_index().items()
        },
    }
    (output / "g50_update_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    e5_rows: list[dict[str, object]] = []
    for contrast, group in e5_blocks[
        e5_blocks["contrast"].str.match(r"online_minus_(ucb|scratch)_g(50|100|200)")
    ].groupby("contrast"):
        diff = group["delta"].dropna()
        nonzero = diff[np.abs(diff) > 1e-12]
        p_value = float(
            wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
        )
        e5_rows.append(
            {
                "contrast": contrast,
                "budget": int(group["Budget"].iloc[0]),
                "n_checkpoint_dependent_instances": int(diff.size),
                "median_delta": float(diff.median()),
                "mean_delta": float(diff.mean()),
                "wins": int((diff > 1e-12).sum()),
                "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
                "wilcoxon_p_raw_descriptive": p_value,
                "independence_warning": (
                    "instances share fold-specific checkpoints and the five training sets overlap"
                ),
            }
        )
    e5 = pd.DataFrame(e5_rows).sort_values(["budget", "contrast"]).reset_index(drop=True)
    e5["holm_adjusted_descriptive"] = holm_adjust(e5["wilcoxon_p_raw_descriptive"])
    e5.to_csv(output / "e5_instance_level_descriptive_sensitivity.csv", index=False)

    mixed_rows: list[dict[str, object]] = []
    for comparator in ("UCBOnly", "ScratchNoBC_R16"):
        for budget in (50, 100, 200):
            subset = e5_runs[
                (e5_runs["Budget"] == budget)
                & e5_runs["variant"].isin(["XPrePPO_Online_R16", comparator])
            ]
            match_keys = ["Fold", "dataset", "instance", "Budget", "seed"]
            if comparator != "UCBOnly":
                match_keys.append("Replica")
            wide = (
                subset.pivot_table(
                    index=match_keys,
                    columns="variant",
                    values="HV_fixed",
                    aggfunc="first",
                )
                .dropna()
                .reset_index()
            )
            if "Replica" not in wide:
                replica_map = e5_runs[
                    (e5_runs["Budget"] == budget)
                    & (e5_runs["variant"] == "XPrePPO_Online_R16")
                ][["Fold", "dataset", "instance", "Budget", "seed", "Replica"]]
                wide = wide.merge(
                    replica_map,
                    on=["Fold", "dataset", "instance", "Budget", "seed"],
                    how="left",
                    validate="one_to_one",
                )
            wide["delta"] = wide["XPrePPO_Online_R16"] - wide[comparator]
            wide["instance_key"] = (
                wide["dataset"].astype(str) + "/" + wide["instance"].astype(str)
            )
            wide["checkpoint_key"] = (
                "F" + wide["Fold"].astype(str) + "R" + wide["Replica"].astype(str)
            )
            model = smf.mixedlm(
                "delta ~ 1",
                wide,
                groups=wide["instance_key"],
                re_formula="1",
                vc_formula={"checkpoint": "0+C(checkpoint_key)"},
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
            ci_low, ci_high = fit.conf_int().loc["Intercept"].tolist()
            mixed_rows.append(
                {
                    "comparator": comparator,
                    "budget": budget,
                    "n_seed_level_paired_differences": int(len(wide)),
                    "fixed_intercept_mean_delta": float(fit.params["Intercept"]),
                    "wald_se": float(fit.bse["Intercept"]),
                    "wald_ci95_low": float(ci_low),
                    "wald_ci95_high": float(ci_high),
                    "wald_p_descriptive": float(fit.pvalues["Intercept"]),
                    "converged": bool(fit.converged),
                    "boundary_warning": any(
                        isinstance(item.message, ConvergenceWarning) for item in caught
                    ),
                    "random_structure": (
                        "instance random intercept; fold-replica checkpoint variance component"
                    ),
                    "unmodeled_dependence": (
                        "overlap among the five checkpoint training sets"
                    ),
                }
            )
    pd.DataFrame(mixed_rows).to_csv(
        output / "e5_run_level_mixed_sensitivity.csv",
        index=False,
    )

    manifest = {
        "script": "scripts/audit_revision_mc1_mc3_mc5.py",
        "read_only_inputs": [
            "results/resubmission/v5/e1_aos/analysis/runs_with_common_hv.csv",
            "results/resubmission/v5/e3_budget/runs.csv",
            "results/resubmission/v6_mechanism/runs.csv",
            "results/resubmission/v7_cross_instance/analysis/contrast_instance_blocks.csv",
            "results/resubmission/v7_cross_instance/analysis/runs_with_fixed_indicators.csv",
        ],
        "outputs": sorted(path.name for path in output.iterdir()),
        "inferential_boundary": (
            "All added analyses are descriptive/read-only. E5 instance-level p-values "
            "are not confirmatory because checkpoint dependence and overlapping fold "
            "training sets violate an independent-instance interpretation."
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
