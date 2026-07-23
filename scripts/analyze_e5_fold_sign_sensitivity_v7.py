"""Dependent fold-sign sensitivity for the frozen E5/v7 held-out experiment.

The original E5 analysis reduces ten seeds within each held-out instance and
reports instance-level paired tests.  Instances within a fold nevertheless
share a training set and five terminal checkpoints.  This read-only supplement
therefore reduces the data to five descriptive fold summaries:

1. compute the paired controller difference for each held-out instance;
2. take the median of the ten instance differences within each fold;
3. enumerate all 2**5 signs of the five fold summaries around their mean;
4. apply Holm correction within the three prespecified E5 families.

The five training sets are not independent: every pair overlaps on 30 of 40
training instances.  The 32-sign enumeration is therefore reported as a
low-resolution sensitivity, not as an exact randomization test.  Its minimum
attainable two-sided sensitivity value is 2/32 = 0.0625.  The reduction avoids
reinterpreting the 50 checkpoint-dependent instance summaries as independent
replications, but it does not create five independent experiments.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "resubmission"
    / "v7_cross_instance"
    / "analysis"
    / "instance_seed_medians.csv"
)
DEFAULT_OUT = DEFAULT_INPUT.parent

FAMILIES = {
    "primary_g100": (
        (100, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g100"),
        (
            100,
            "XPrePPO_Online_R16",
            "ScratchNoBC_R16",
            "online_minus_scratch_g100",
        ),
    ),
    "secondary_g50_g200": (
        (50, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g50"),
        (
            50,
            "XPrePPO_Online_R16",
            "ScratchNoBC_R16",
            "online_minus_scratch_g50",
        ),
        (200, "XPrePPO_Online_R16", "UCBOnly", "online_minus_ucb_g200"),
        (
            200,
            "XPrePPO_Online_R16",
            "ScratchNoBC_R16",
            "online_minus_scratch_g200",
        ),
    ),
    "mechanism_online_frozen": tuple(
        (
            budget,
            "XPrePPO_Online_R16",
            "XPrePPO_Frozen",
            f"online_minus_frozen_g{budget}",
        )
        for budget in (50, 100, 200)
    ),
}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def enumerate_fold_signs(values: np.ndarray) -> tuple[float, float, int]:
    """Two-sided 32-sign sensitivity using the absolute mean fold effect."""

    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("E5 fold-sign sensitivity requires five finite fold effects")
    observed = float(abs(np.mean(values)))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    statistics_array = np.asarray(statistics)
    p_value = float(
        np.mean(statistics_array >= observed - 10 * np.finfo(float).eps)
    )
    return observed, p_value, int(len(statistics_array))


def contrast_fold_effects(
    medians: pd.DataFrame,
    *,
    family: str,
    budget: int,
    lhs: str,
    rhs: str,
    contrast: str,
) -> tuple[pd.DataFrame, dict]:
    """Build ten-instance fold summaries for one fixed-reference HV contrast."""

    block = medians[pd.to_numeric(medians["Budget"]).astype(int) == int(budget)]
    pivot = block.pivot(
        index=["Fold", "dataset", "instance"],
        columns="variant",
        values="HV_fixed",
    ).reset_index()
    if lhs not in pivot or rhs not in pivot:
        raise RuntimeError(f"missing {lhs} or {rhs} for {contrast}")
    pivot["delta"] = pivot[lhs] - pivot[rhs]
    if len(pivot) != 50 or pivot["delta"].isna().any():
        raise RuntimeError(f"incomplete 50-instance block for {contrast}")

    rows = []
    for fold, group in pivot.groupby("Fold", sort=True):
        if len(group) != 10:
            raise RuntimeError(f"fold {fold} in {contrast} has {len(group)} instances")
        rows.append(
            {
                "family": family,
                "contrast": contrast,
                "Budget": int(budget),
                "lhs": lhs,
                "rhs": rhs,
                "Fold": int(fold),
                "n_instances": int(len(group)),
                "fold_median_delta": float(np.median(group["delta"])),
                "fold_mean_delta": float(np.mean(group["delta"])),
                "fold_wins": int(np.sum(group["delta"] > 1e-12)),
                "fold_ties": int(np.sum(np.abs(group["delta"]) <= 1e-12)),
                "fold_losses": int(np.sum(group["delta"] < -1e-12)),
            }
        )
    fold_table = pd.DataFrame(rows)
    values = fold_table["fold_median_delta"].to_numpy(dtype=float)
    statistic, p_raw, assignments = enumerate_fold_signs(values)
    record = {
        "family": family,
        "contrast": contrast,
        "Budget": int(budget),
        "lhs": lhs,
        "rhs": rhs,
        "orientation": "positive favors lhs",
        "n_fold_summaries": int(len(values)),
        "training_instances_per_fold": 40,
        "pairwise_training_overlap": 30,
        "dependence_note": "fold training sets overlap; sensitivity is not an exact randomization test",
        "fold_effect_definition": "median of 10 instance-level seed medians",
        "median_of_fold_effects": float(np.median(values)),
        "mean_of_fold_effects": float(np.mean(values)),
        "minimum_fold_effect": float(np.min(values)),
        "maximum_fold_effect": float(np.max(values)),
        "positive_folds": int(np.sum(values > 1e-12)),
        "zero_folds": int(np.sum(np.abs(values) <= 1e-12)),
        "negative_folds": int(np.sum(values < -1e-12)),
        "test_statistic_abs_mean": statistic,
        "sign_assignments": assignments,
        "p_raw_fold_sign_sensitivity": p_raw,
    }
    return fold_table, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    medians = pd.read_csv(args.input)
    required = {
        "Fold",
        "dataset",
        "instance",
        "variant",
        "Budget",
        "HV_fixed",
    }
    missing = required.difference(medians.columns)
    if missing:
        raise RuntimeError(f"missing required columns: {sorted(missing)}")
    if medians.duplicated(
        ["Fold", "dataset", "instance", "variant", "Budget"]
    ).any():
        raise RuntimeError("instance_seed_medians.csv has duplicate analysis keys")

    fold_frames = []
    records = []
    for family, definitions in FAMILIES.items():
        for budget, lhs, rhs, contrast in definitions:
            fold_table, record = contrast_fold_effects(
                medians,
                family=family,
                budget=budget,
                lhs=lhs,
                rhs=rhs,
                contrast=contrast,
            )
            fold_frames.append(fold_table)
            records.append(record)

    inference = pd.DataFrame(records)
    inference["p_holm_within_family"] = np.nan
    for family, indices in inference.groupby("family").groups.items():
        index = list(indices)
        inference.loc[index, "p_holm_within_family"] = holm_adjust(
            inference.loc[index, "p_raw_fold_sign_sensitivity"].to_numpy(dtype=float)
        )
    inference["fold_sign_sensitivity_below_0_05"] = (
        inference["p_holm_within_family"] < 0.05
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.out_dir / "fold_effect_summaries.csv"
    inference_path = args.out_dir / "fold_sign_sensitivity.csv"
    pd.concat(fold_frames, ignore_index=True).to_csv(fold_path, index=False)
    inference.to_csv(inference_path, index=False)
    print(inference.to_string(index=False))
    print(f"wrote {fold_path}")
    print(f"wrote {inference_path}")


if __name__ == "__main__":
    main()
