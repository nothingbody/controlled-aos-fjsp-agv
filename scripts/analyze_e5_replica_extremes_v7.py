"""Read-only E5 sensitivity analyses from the frozen v7 test ledger.

This script does not rerun the optimizer.  It reports:

1. paired HV effects separately for the five checkpoint-replica/assigned-seed
   pairs; and
2. paired percentage changes in the three single-objective extreme values of
   each saved final front; and
3. an explicit denominator audit plus absolute paired changes, so percentage
   changes are never formed from zero or nonpositive comparator extrema.

Checkpoint replica and its two assigned test seeds are nested in the frozen
design and therefore cannot be separated by this analysis.  The front-extreme
values need not be attained by the same schedule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_ONLINE = "XPrePPO_Online_R16"
VARIANT_UCB = "UCBOnly"
VARIANT_SCRATCH = "ScratchNoBC_R16"

PAIR_KEYS = ["Fold", "dataset", "instance", "Budget", "seed"]
INSTANCE_KEYS = ["Fold", "dataset", "instance", "Budget"]


def _paired_rows(
    runs: pd.DataFrame,
    left: str,
    right: str,
    columns: list[str],
) -> pd.DataFrame:
    left_rows = runs.loc[
        runs["variant"] == left, PAIR_KEYS + ["Replica"] + columns
    ].copy()
    right_rows = runs.loc[
        runs["variant"] == right, PAIR_KEYS + ["Replica"] + columns
    ].copy()
    paired = left_rows.merge(
        right_rows,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    expected = runs.loc[runs["variant"] == left, PAIR_KEYS].shape[0]
    if len(paired) != expected:
        raise RuntimeError(
            f"Incomplete pairing for {left} versus {right}: "
            f"{len(paired)} of {expected} rows"
        )
    if right != VARIANT_UCB and not np.array_equal(
        paired["Replica_left"].to_numpy(), paired["Replica_right"].to_numpy()
    ):
        raise RuntimeError(f"Checkpoint-replica mismatch for {left} versus {right}")
    paired["Replica"] = paired["Replica_left"].astype(int)
    return paired


def checkpoint_replica_sensitivity(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for contrast, right in (
        ("online_minus_ucb", VARIANT_UCB),
        ("online_minus_scratch", VARIANT_SCRATCH),
    ):
        paired = _paired_rows(runs, VARIANT_ONLINE, right, ["HV_fixed"])
        paired["delta"] = paired["HV_fixed_left"] - paired["HV_fixed_right"]
        for (budget, replica), group in paired.groupby(["Budget", "Replica"], sort=True):
            instance_delta = (
                group.groupby(["Fold", "dataset", "instance"], sort=False)["delta"]
                .median()
                .to_numpy(dtype=float)
            )
            if len(instance_delta) != 50:
                raise RuntimeError(
                    f"Expected 50 instance blocks for {contrast}, G={budget}, "
                    f"replica={replica}; found {len(instance_delta)}"
                )
            rows.append(
                {
                    "contrast": contrast,
                    "Budget": int(budget),
                    "Replica": int(replica),
                    "n_instances": int(len(instance_delta)),
                    "median_delta_HV_fixed": float(np.median(instance_delta)),
                    "q25_delta_HV_fixed": float(np.quantile(instance_delta, 0.25)),
                    "q75_delta_HV_fixed": float(np.quantile(instance_delta, 0.75)),
                    "wins": int(np.sum(instance_delta > 1e-12)),
                    "ties": int(np.sum(np.abs(instance_delta) <= 1e-12)),
                    "losses": int(np.sum(instance_delta < -1e-12)),
                }
            )
    detail = pd.DataFrame(rows).sort_values(["contrast", "Budget", "Replica"])
    summary = (
        detail.groupby(["contrast", "Budget"], sort=True)
        .agg(
            min_replica_median_delta=("median_delta_HV_fixed", "min"),
            max_replica_median_delta=("median_delta_HV_fixed", "max"),
            n_negative_replica_medians=(
                "median_delta_HV_fixed",
                lambda values: int(np.sum(np.asarray(values) < 0.0)),
            ),
        )
        .reset_index()
    )
    return detail, summary


def front_extreme_sensitivity(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = {
        "Cmax_best": "best_makespan",
        "TEC_best": "best_energy_index",
        "WB_best": "best_workload_imbalance",
    }
    cells = (
        runs.groupby(INSTANCE_KEYS + ["variant"], sort=False)[list(metrics)]
        .median()
        .reset_index()
    )
    block_rows: list[pd.DataFrame] = []
    denominator_rows: list[dict[str, object]] = []
    for contrast, right in (
        ("online_minus_ucb", VARIANT_UCB),
        ("online_minus_scratch", VARIANT_SCRATCH),
    ):
        left_cells = cells.loc[cells["variant"] == VARIANT_ONLINE].drop(
            columns="variant"
        )
        right_cells = cells.loc[cells["variant"] == right].drop(columns="variant")
        paired = left_cells.merge(
            right_cells,
            on=INSTANCE_KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("_left", "_right"),
        )
        if len(paired) != 150:
            raise RuntimeError(
                f"Expected 150 instance-budget blocks for {contrast}; found {len(paired)}"
            )
        for source, label in metrics.items():
            denominator = paired[f"{source}_right"].to_numpy(dtype=float)
            nonpositive = int(np.sum(denominator <= 0.0))
            denominator_rows.append(
                {
                    "contrast": contrast,
                    "front_extreme": label,
                    "n_instance_budget_blocks": int(len(denominator)),
                    "minimum_comparator_extreme": float(np.min(denominator)),
                    "maximum_comparator_extreme": float(np.max(denominator)),
                    "nonpositive_denominators": nonpositive,
                }
            )
            if nonpositive:
                raise RuntimeError(f"Nonpositive denominator found for {source}")
            absolute = (
                paired[f"{source}_left"].to_numpy(dtype=float) - denominator
            )
            relative = (
                100.0 * absolute / denominator
            )
            block_rows.append(
                pd.DataFrame(
                    {
                        **{key: paired[key] for key in INSTANCE_KEYS},
                        "contrast": contrast,
                        "front_extreme": label,
                        "absolute_change": absolute,
                        "comparator_extreme": denominator,
                        "relative_change_percent": relative,
                    }
                )
            )
    blocks = pd.concat(block_rows, ignore_index=True)
    summary = (
        blocks.groupby(["contrast", "Budget", "front_extreme"], sort=True)
        .agg(
            n_instances=("relative_change_percent", "size"),
            median_relative_change_percent=("relative_change_percent", "median"),
            median_absolute_change=("absolute_change", "median"),
            q25_relative_change_percent=(
                "relative_change_percent",
                lambda values: float(np.quantile(values, 0.25)),
            ),
            q75_relative_change_percent=(
                "relative_change_percent",
                lambda values: float(np.quantile(values, 0.75)),
            ),
            online_better=(
                "relative_change_percent",
                lambda values: int(np.sum(np.asarray(values) < -1e-12)),
            ),
            ties=(
                "relative_change_percent",
                lambda values: int(np.sum(np.abs(np.asarray(values)) <= 1e-12)),
            ),
            online_worse=(
                "relative_change_percent",
                lambda values: int(np.sum(np.asarray(values) > 1e-12)),
            ),
        )
        .reset_index()
    )
    denominator_audit = pd.DataFrame(denominator_rows)
    denominator_audit = (
        denominator_audit.groupby(["contrast", "front_extreme"], sort=True)
        .agg(
            n_instance_budget_blocks=("n_instance_budget_blocks", "sum"),
            minimum_comparator_extreme=("minimum_comparator_extreme", "min"),
            maximum_comparator_extreme=("maximum_comparator_extreme", "max"),
            nonpositive_denominators=("nonpositive_denominators", "sum"),
        )
        .reset_index()
    )
    return blocks, summary, denominator_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/resubmission/v7_cross_instance"),
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    analysis_dir = result_dir / "analysis"
    source = analysis_dir / "runs_with_fixed_indicators.csv"
    runs = pd.read_csv(source)
    if len(runs) != 6000:
        raise RuntimeError(f"Expected 6,000 frozen E5 test rows; found {len(runs)}")
    if runs[PAIR_KEYS + ["variant"]].duplicated().any():
        raise RuntimeError("Duplicate frozen E5 test key detected")

    replica_detail, replica_summary = checkpoint_replica_sensitivity(runs)
    extreme_blocks, extreme_summary, denominator_audit = front_extreme_sensitivity(
        runs
    )

    replica_detail.to_csv(
        analysis_dir / "checkpoint_replica_pair_sensitivity.csv", index=False
    )
    replica_summary.to_csv(
        analysis_dir / "checkpoint_replica_pair_ranges.csv", index=False
    )
    extreme_blocks.to_csv(
        analysis_dir / "front_extreme_instance_blocks.csv", index=False
    )
    extreme_summary.to_csv(
        analysis_dir / "front_extreme_relative_changes.csv", index=False
    )
    denominator_audit.to_csv(
        analysis_dir / "front_extreme_denominator_audit.csv", index=False
    )

    print(replica_summary.to_string(index=False))
    print()
    print(extreme_summary.to_string(index=False))
    print()
    print(denominator_audit.to_string(index=False))


if __name__ == "__main__":
    main()
