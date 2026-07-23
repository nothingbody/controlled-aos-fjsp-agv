"""Generate read-only evidence used by the main(18) manuscript revision.

The script does not rerun an optimizer.  It summarizes final archive-cap
contacts, budget-specific enhanced-state constant-coordinate fractions, and
the E4 state-by-BC interaction from frozen CSV artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "resubmission" / "revision_audits" / "main18"

RUN_TABLES = {
    "E1": ROOT / "results/resubmission/v5/e1_aos/runs.csv",
    "E2": ROOT / "results/resubmission/v5/e2_reward/runs.csv",
    "E3": ROOT / "results/resubmission/v5/e3_budget/runs.csv",
    "E4": ROOT / "results/resubmission/v6_mechanism/runs.csv",
    "E5": ROOT / "results/resubmission/v7_cross_instance/runs.csv",
    "E4-R": ROOT / "results/resubmission/v8_e4_replication/runs.csv",
}


def bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    estimates = np.median(values[draws], axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def archive_cap_summary() -> pd.DataFrame:
    records = []
    for experiment, path in RUN_TABLES.items():
        frame = pd.read_csv(path, usecols=["NSol"])
        sizes = pd.to_numeric(frame["NSol"], errors="raise")
        records.append(
            {
                "experiment": experiment,
                "runs": int(len(frame)),
                "final_archive_size_100": int((sizes == 100).sum()),
                "final_archive_size_100_percent": float(100 * (sizes == 100).mean()),
                "maximum_saved_archive_size": int(sizes.max()),
            }
        )
    output = pd.DataFrame(records)
    output.loc[len(output)] = {
        "experiment": "Total",
        "runs": int(output["runs"].sum()),
        "final_archive_size_100": int(output["final_archive_size_100"].sum()),
        "final_archive_size_100_percent": float(
            100 * output["final_archive_size_100"].sum() / output["runs"].sum()
        ),
        "maximum_saved_archive_size": int(output["maximum_saved_archive_size"].max()),
    }
    return output


def enhanced_constant_summary() -> pd.DataFrame:
    path = RUN_TABLES["E4"]
    frame = pd.read_csv(
        path,
        usecols=[
            "Budget",
            "state_mode",
            "Enhanced_feature_samples",
            "Enhanced_constant_feature_fraction",
        ],
    )
    selected = frame.loc[
        (frame["state_mode"] == "enhanced")
        & (pd.to_numeric(frame["Enhanced_feature_samples"], errors="coerce") > 0)
    ].copy()
    selected["fraction"] = pd.to_numeric(
        selected["Enhanced_constant_feature_fraction"], errors="raise"
    )
    return (
        selected.groupby("Budget", sort=True)["fraction"]
        .agg(runs="size", median="median", mean="mean", minimum="min", maximum="max")
        .reset_index()
    )


def interaction_audit() -> tuple[pd.DataFrame, dict[str, object]]:
    path = (
        ROOT
        / "results/resubmission/v6_mechanism/analysis_v6_1/"
        "instance_seed_medians.csv"
    )
    frame = pd.read_csv(path)
    frame = frame.loc[pd.to_numeric(frame["Budget"]).astype(int) == 100]
    pivot = frame.pivot(
        index=["dataset", "instance"], columns="variant", values="HV_fixed"
    ).reset_index()
    pivot["enhanced_bc_minus_padded_bc"] = (
        pivot["EnhancedBC_R16"] - pivot["BasePaddedBC_R16"]
    )
    pivot["enhanced_nobc_minus_padded_nobc"] = (
        pivot["EnhancedNoBC_R16"] - pivot["BasePaddedNoBC_R16"]
    )
    pivot["state_by_bc_interaction"] = (
        pivot["enhanced_bc_minus_padded_bc"]
        - pivot["enhanced_nobc_minus_padded_nobc"]
    )
    audit = pivot[
        [
            "dataset",
            "instance",
            "enhanced_bc_minus_padded_bc",
            "enhanced_nobc_minus_padded_nobc",
            "state_by_bc_interaction",
        ]
    ].copy()

    state_bc = audit["enhanced_bc_minus_padded_bc"].to_numpy(dtype=float)
    interaction = audit["state_by_bc_interaction"].to_numpy(dtype=float)
    state_ci = bootstrap_median(state_bc, seed=20260723)
    interaction_ci = bootstrap_median(interaction, seed=20260731)
    summary = {
        "interaction_definition": (
            "(EnhancedBC-BasePaddedBC)-"
            "(EnhancedNoBC-BasePaddedNoBC)"
        ),
        "state_with_bc": {
            "median": float(np.median(state_bc)),
            "bootstrap_seed": 20260723,
            "percentile_ci95": list(state_ci),
        },
        "state_by_bc_interaction": {
            "median": float(np.median(interaction)),
            "bootstrap_seed": 20260731,
            "percentile_ci95": list(interaction_ci),
        },
        "coincident_displayed_endpoints": bool(
            np.allclose(state_ci, interaction_ci, rtol=0.0, atol=1e-15)
        ),
        "interpretation": (
            "The two statistics use different instance vectors and independent "
            "bootstrap seeds. Their percentile endpoints coincide because the "
            "small-sample bootstrap quantiles select the same observed values."
        ),
    }
    return audit, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive_cap_summary().to_csv(OUT / "final_archive_cap_summary.csv", index=False)
    enhanced_constant_summary().to_csv(
        OUT / "enhanced_constant_fraction_by_budget.csv", index=False
    )
    interaction_rows, interaction_summary = interaction_audit()
    interaction_rows.to_csv(OUT / "e4_state_by_bc_interaction_rows.csv", index=False)
    (OUT / "e4_state_by_bc_interaction_summary.json").write_text(
        json.dumps(interaction_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote revision evidence to {OUT}")


if __name__ == "__main__":
    main()
