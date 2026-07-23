"""Post-hoc behavioral audit of the frozen E5 transfer protocol.

This script uses only fields already saved by the formal E5 evaluation.  It
does not alter the prespecified E5 performance analysis or its multiplicity
families.  The output describes whether transferred and scratch PPO realize
different action distributions and whether online updating changes the
transferred behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = ROOT / "results" / "resubmission" / "v7_cross_instance"
INPUT_CSV = E5_ROOT / "analysis" / "runs_with_fixed_indicators.csv"
OUTPUT_CSV = E5_ROOT / "analysis" / "posthoc_transfer_behavior_summary.csv"

OPERATOR_NAMES = (
    "POX",
    "JBX",
    "UniformMA",
    "UniformAGV",
    "TwoPoint",
    "Swap",
    "Insert",
    "MachineReassign",
    "AGVReassign",
    "SpeedAdjust",
)
UNIFORM_MA_INDEX = OPERATOR_NAMES.index("UniformMA")
MATCH_KEYS = ["Fold", "dataset", "instance", "Budget", "seed", "Replica"]
INSTANCE_KEYS = ["Fold", "dataset", "instance"]


def _parse_counts(value: str) -> np.ndarray:
    counts = np.asarray(json.loads(value), dtype=float)
    if counts.shape != (len(OPERATOR_NAMES),):
        raise ValueError(f"Expected {len(OPERATOR_NAMES)} operator counts, got {counts.shape}")
    if np.any(counts < 0) or not np.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError("Operator counts must be finite, nonnegative, and nonempty")
    return counts


def _add_behavior_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_counts"] = out["operator_counts"].map(_parse_counts)
    out["uniform_ma_share"] = out["_counts"].map(
        lambda counts: float(counts[UNIFORM_MA_INDEX] / counts.sum())
    )
    out["dominant_operator"] = out["_counts"].map(lambda counts: int(np.argmax(counts)))
    return out


def _instance_median(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.groupby(INSTANCE_KEYS, sort=True)[columns].median()


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one descriptive row for each E5 evaluation budget."""

    required = {
        *MATCH_KEYS,
        "variant",
        "operator_counts",
        "Operator_entropy",
        "ppo_entropy_mean",
        "HV_fixed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required E5 columns: {missing}")

    data = _add_behavior_columns(frame)
    rows: list[dict[str, float | int | str]] = []

    for budget in sorted(int(value) for value in data["Budget"].unique()):
        scratch = data[
            (data["Budget"] == budget) & (data["variant"] == "ScratchNoBC_R16")
        ].set_index(MATCH_KEYS)
        online = data[
            (data["Budget"] == budget) & (data["variant"] == "XPrePPO_Online_R16")
        ].set_index(MATCH_KEYS)
        frozen = data[
            (data["Budget"] == budget) & (data["variant"] == "XPrePPO_Frozen")
        ].set_index(MATCH_KEYS)

        if not scratch.index.equals(online.index) or not frozen.index.equals(online.index):
            raise ValueError(f"Unmatched E5 transfer cells at budget {budget}")

        js_rows = []
        for index, scratch_counts, online_counts in zip(
            scratch.index, scratch["_counts"], online["_counts"], strict=True
        ):
            scratch_prob = scratch_counts / scratch_counts.sum()
            online_prob = online_counts / online_counts.sum()
            js_rows.append(
                {
                    "Fold": index[0],
                    "dataset": index[1],
                    "instance": index[2],
                    "js_divergence": float(
                        jensenshannon(scratch_prob, online_prob, base=2.0) ** 2
                    ),
                }
            )
        instance_js = (
            pd.DataFrame(js_rows)
            .groupby(INSTANCE_KEYS, sort=True)["js_divergence"]
            .median()
        )

        diagnostic_columns = [
            "Operator_entropy",
            "uniform_ma_share",
            "ppo_entropy_mean",
            "HV_fixed",
        ]
        scratch_instance = _instance_median(scratch.reset_index(), diagnostic_columns)
        online_instance = _instance_median(online.reset_index(), diagnostic_columns)
        delta = online_instance - scratch_instance
        rho, p_value = spearmanr(delta["uniform_ma_share"], delta["HV_fixed"])

        rows.append(
            {
                "analysis_role": "posthoc_descriptive_transfer_behavior",
                "budget": budget,
                "n_run_pairs": len(scratch),
                "n_instance_blocks": len(scratch_instance),
                "scratch_operator_entropy_median": float(
                    scratch_instance["Operator_entropy"].median()
                ),
                "online_operator_entropy_median": float(
                    online_instance["Operator_entropy"].median()
                ),
                "scratch_uniform_ma_share_median": float(
                    scratch_instance["uniform_ma_share"].median()
                ),
                "online_uniform_ma_share_median": float(
                    online_instance["uniform_ma_share"].median()
                ),
                "scratch_policy_entropy_median": float(
                    scratch_instance["ppo_entropy_mean"].median()
                ),
                "online_policy_entropy_median": float(
                    online_instance["ppo_entropy_mean"].median()
                ),
                "js_divergence_vs_scratch_median": float(instance_js.median()),
                "online_frozen_exact_count_match_fraction": float(
                    (online["operator_counts"] == frozen["operator_counts"]).mean()
                ),
                "uniform_ma_dominant_online_fraction": float(
                    (online["dominant_operator"] == UNIFORM_MA_INDEX).mean()
                ),
                "spearman_delta_uniform_ma_vs_delta_hv": float(rho),
                "spearman_uncorrected_p": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    frame = pd.read_csv(INPUT_CSV)
    summary = build_summary(frame)
    summary.to_csv(OUTPUT_CSV, index=False, float_format="%.10g")
    print(summary.to_string(index=False))
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
