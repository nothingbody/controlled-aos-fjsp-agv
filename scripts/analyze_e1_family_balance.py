"""Read-only family-balanced sensitivity for the frozen E1 SA-AOS--UCB contrast.

The primary analysis gives each of the 50 instances equal weight, so the
40-instance Hurink edata family contributes 80% of the blocks.  This script
retains the frozen within-instance seed medians, reports each family's paired
effect separately, and then gives each family one-half weight descriptively.
It performs no new optimizer run and introduces no additional hypothesis test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "AdaptiveSAOS"
COMPARATOR = "UCBOnly"
TOL = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "results/resubmission/v5/e1_aos/analysis/seed_medians_main.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/resubmission/v5/posthoc_diagnostics/"
            "family_equal_weight_sensitivity.csv"
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    data = pd.read_csv(source)
    selected = data.loc[
        data["variant"].isin([TARGET, COMPARATOR]),
        ["dataset", "instance", "variant", "HV_common"],
    ]
    wide = selected.pivot(
        index=["dataset", "instance"],
        columns="variant",
        values="HV_common",
    ).reset_index()
    if len(wide) != 50 or wide[[TARGET, COMPARATOR]].isna().any().any():
        raise RuntimeError("Expected 50 complete frozen E1 instance pairs")
    wide["delta_HV"] = wide[TARGET] - wide[COMPARATOR]

    rows: list[dict[str, object]] = []
    for family, group in wide.groupby("dataset", sort=True):
        values = group["delta_HV"].to_numpy(dtype=float)
        rows.append(
            {
                "summary": str(family),
                "n_instances": int(len(values)),
                "median_delta_HV": float(np.median(values)),
                "win_rate": float(np.mean(values > TOL)),
                "tie_rate": float(np.mean(np.abs(values) <= TOL)),
                "loss_rate": float(np.mean(values < -TOL)),
            }
        )

    family_rows = pd.DataFrame(rows)
    rows.append(
        {
            "summary": "equal_family_weight",
            "n_instances": 50,
            "median_delta_HV": float(family_rows["median_delta_HV"].mean()),
            "win_rate": float(family_rows["win_rate"].mean()),
            "tie_rate": float(family_rows["tie_rate"].mean()),
            "loss_rate": float(family_rows["loss_rate"].mean()),
        }
    )
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out.resolve(), index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
