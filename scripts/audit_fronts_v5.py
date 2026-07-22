"""Audit saved v5 nondominated fronts and their Cmax--energy trade-off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_resubmission_v5 import (
    PROTOCOL,
    load_front,
    locate_front,
    nondominated,
)


def audit_front_matrix(objectives: np.ndarray) -> dict:
    """Return size and within-front Cmax--TEC correlations."""
    front = nondominated(np.asarray(objectives, dtype=float))
    result = {
        "front_points_unique_nondominated": int(len(front)),
        "pearson_cmax_tec": np.nan,
        "spearman_cmax_tec": np.nan,
        "correlation_status": "insufficient_variation",
    }
    if (
        len(front) >= 3
        and np.ptp(front[:, 0]) > 1e-12
        and np.ptp(front[:, 1]) > 1e-12
    ):
        result["pearson_cmax_tec"] = float(np.corrcoef(front[:, 0], front[:, 1])[0, 1])
        result["spearman_cmax_tec"] = float(
            spearmanr(front[:, 0], front[:, 1]).statistic
        )
        result["correlation_status"] = "computed"
    return result


def _distribution(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "n": int(len(values)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_audit(csv_path: Path, search_root: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(csv_path)
    if set(frame["Protocol"].astype(str)) != {PROTOCOL}:
        raise RuntimeError("front audit accepts only the frozen v5 protocol")

    rows = []
    mismatches = []
    for source_row in frame.itertuples(index=False):
        path = locate_front(source_row.front_pickle, csv_path, search_root)
        front, raw_count = load_front(path)
        metrics = audit_front_matrix(front)
        expected_nsol = int(source_row.NSol)
        actual_nsol = metrics["front_points_unique_nondominated"]
        if expected_nsol != actual_nsol:
            mismatches.append(
                {
                    "front": str(path),
                    "reported_nsol": expected_nsol,
                    "audited_nsol": actual_nsol,
                }
            )
        row = {
            "dataset": source_row.dataset,
            "instance": source_row.instance,
            "variant": source_row.variant,
            "reward_scheme": source_row.reward_scheme,
            "seed": int(source_row.seed),
            "front_pickle_resolved": str(path),
            "front_points_raw": int(raw_count),
            **metrics,
        }
        if hasattr(source_row, "Budget"):
            row["Budget"] = int(source_row.Budget)
        rows.append(row)

    if mismatches:
        raise RuntimeError(f"NSol/front mismatches: {mismatches[:5]}")

    details = pd.DataFrame(rows)
    summary = {
        "protocol": PROTOCOL,
        "runs": int(len(details)),
        "fronts_with_computable_correlation": int(
            (details["correlation_status"] == "computed").sum()
        ),
        "pearson_cmax_tec": _distribution(details["pearson_cmax_tec"]),
        "spearman_cmax_tec": _distribution(details["spearman_cmax_tec"]),
        "front_size": _distribution(details["front_points_unique_nondominated"]),
        "interpretation": (
            "Correlations are descriptive within-final-front diagnostics, not "
            "independent inferential observations."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / "front_correlation_audit.csv", index=False)
    (out_dir / "front_correlation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.csv, args.search_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
