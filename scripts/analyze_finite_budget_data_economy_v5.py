"""Audit terminal-inclusive versus action-effective PPO learning in v5 E3."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/resubmission/v5/e3_budget/runs.csv"
DEFAULT_OUT = ROOT / "results/resubmission/v5/e3_budget/analysis"
EXPECTED_ROWS = 10_500
ROLLOUT = 16


def parse_rollout_sizes(value) -> list[int]:
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or any(int(item) < 2 for item in parsed):
        raise ValueError(f"invalid PPO rollout sizes: {value}")
    return [int(item) for item in parsed]


def audit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Protocol", "dataset", "instance", "variant", "seed", "Budget",
        "Transition_gen", "PPO_update_count", "PPO_rollout_sizes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing E3 columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} E3 rows, observed {len(frame)}")
    key = ["dataset", "instance", "variant", "Budget", "seed"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate E3 primary keys")
    learned = frame[pd.to_numeric(frame["Transition_gen"], errors="coerce").notna()].copy()
    learned = learned[pd.to_numeric(learned["Transition_gen"]) >= 0].copy()
    learned["Budget"] = pd.to_numeric(learned["Budget"]).astype(int)
    learned["Transition_gen"] = pd.to_numeric(learned["Transition_gen"]).astype(int)
    learned["PPO_controlled_actions_derived"] = np.maximum(
        0, learned["Budget"] - learned["Transition_gen"]
    ).astype(int)
    n_actions = learned["PPO_controlled_actions_derived"]
    learned["PPO_action_effective_updates_derived"] = np.maximum(
        0, (n_actions - 1) // ROLLOUT
    ).astype(int)
    residual = n_actions % ROLLOUT
    learned["PPO_total_updates_expected"] = (
        n_actions // ROLLOUT + (residual >= 2).astype(int)
    ).astype(int)
    learned["PPO_consumed_samples_expected"] = (
        ROLLOUT * (n_actions // ROLLOUT) + residual * (residual >= 2).astype(int)
    ).astype(int)
    learned["PPO_rollout_sizes_parsed"] = learned["PPO_rollout_sizes"].map(
        parse_rollout_sizes
    )
    learned["PPO_consumed_samples_observed"] = learned[
        "PPO_rollout_sizes_parsed"
    ].map(sum).astype(int)
    observed_updates = pd.to_numeric(learned["PPO_update_count"]).astype(int)
    if not np.array_equal(
        observed_updates.to_numpy(), learned["PPO_total_updates_expected"].to_numpy()
    ):
        raise RuntimeError("terminal-inclusive PPO update formula does not match v5 log")
    if not np.array_equal(
        learned["PPO_consumed_samples_observed"].to_numpy(),
        learned["PPO_consumed_samples_expected"].to_numpy(),
    ):
        raise RuntimeError("consumed-transition formula does not match v5 rollout log")
    learned["Terminal_only_update_count"] = (
        learned["PPO_total_updates_expected"]
        - learned["PPO_action_effective_updates_derived"]
    )
    return learned


def summarize(audited: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (variant, budget), block in audited.groupby(["variant", "Budget"], sort=True):
        records.append({
            "variant": variant,
            "Budget": int(budget),
            "runs": int(len(block)),
            "median_transition_generation": float(block["Transition_gen"].median()),
            "median_PPO_controlled_actions": float(
                block["PPO_controlled_actions_derived"].median()
            ),
            "median_terminal_inclusive_updates": float(
                block["PPO_total_updates_expected"].median()
            ),
            "median_action_effective_updates": float(
                block["PPO_action_effective_updates_derived"].median()
            ),
            "median_consumed_transitions": float(
                block["PPO_consumed_samples_observed"].median()
            ),
            "runs_with_zero_action_effective_updates": int(
                (block["PPO_action_effective_updates_derived"] == 0).sum()
            ),
            "runs_with_terminal_only_update": int(
                (block["Terminal_only_update_count"] > 0).sum()
            ),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    audited = audit_rows(frame)
    summary = summarize(audited)
    saos = audited[audited["variant"] == "AdaptiveSAOS"].copy()
    if len(saos) != 1_500:
        raise RuntimeError(f"expected 1,500 SA-AOS E3 rows, observed {len(saos)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_columns = [
        "Protocol", "dataset", "instance", "variant", "seed", "Budget",
        "Transition_gen", "PPO_controlled_actions_derived",
        "PPO_total_updates_expected", "PPO_action_effective_updates_derived",
        "PPO_consumed_samples_observed", "Terminal_only_update_count",
    ]
    saos[row_columns].to_csv(
        args.out_dir / "saos_actionable_learning_audit.csv", index=False
    )
    summary.to_csv(args.out_dir / "ppo_data_economy_by_variant_budget.csv", index=False)
    saos_summary = summary[summary["variant"] == "AdaptiveSAOS"].to_dict("records")
    manifest = {
        "source": str(args.input),
        "source_rows": int(len(frame)),
        "audited_PPO_rows": int(len(audited)),
        "rollout_length": ROLLOUT,
        "formula": {
            "controlled_actions": "max(0, G - T_c)",
            "action_effective_updates": "max(0, floor((n_PPO - 1) / L))",
            "terminal_inclusive_updates": "floor(n_PPO/L) + I[n_PPO mod L >= 2]",
        },
        "saos_summary": saos_summary,
    }
    (args.out_dir / "finite_budget_data_economy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
