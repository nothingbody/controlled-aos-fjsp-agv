"""Read-only process audit of the 25 frozen E5/v7 checkpoint chains.

Only the pass-1 and terminal checkpoint objects are required.  The formal v7
ledger did not retain episode reward/HV trajectories or per-update PPO losses,
KL, clip fraction, or value diagnostics.  This script therefore audits what is
actually recoverable without rerunning optimization:

* marker and object integrity;
* 40/80 episode and pass ordering;
* actor/critic parameter drift from initialization to pass 1 and terminal;
* parameter and optimizer-state finiteness;
* pass-wise realized action-count entropy, transition generation, samples, and
  updates from the retained episode records.

It does not treat parameter drift or count entropy as an optimization-quality
measure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    ROOT / "results" / "resubmission" / "v7_cross_instance" / "checkpoints"
)
DEFAULT_OUT = (
    ROOT / "results" / "resubmission" / "v7_cross_instance" / "analysis"
)
PROTOCOL = "saos_cross_instance_pretrained_ppo_v7_20260722"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_marker(checkpoint_dir: Path, fold: int, replica: int, role: str) -> dict:
    path = checkpoint_dir / f"fold{fold}_rep{replica}_{role}.pt.complete.json"
    if not path.is_file():
        raise RuntimeError(f"missing checkpoint marker: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected protocol in {path}")
    if int(payload["fold"]) != fold or int(payload["replica"]) != replica:
        raise RuntimeError(f"fold/replica mismatch in {path}")
    return payload


def load_object(checkpoint_dir: Path, marker: dict) -> dict:
    path = checkpoint_dir / str(marker["object_path"])
    if not path.is_file():
        raise RuntimeError(f"missing content-addressed checkpoint object: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != marker["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    if path.stat().st_size != int(marker["checkpoint_size_bytes"]):
        raise RuntimeError(f"checkpoint size mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected checkpoint protocol: {path}")
    return payload


def flatten_policy(policy: dict, prefix: str) -> torch.Tensor:
    tensors = [
        value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        for key, value in sorted(policy.items())
        if key.startswith(prefix)
    ]
    if not tensors:
        raise RuntimeError(f"no {prefix} parameters in policy")
    return torch.cat(tensors)


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        raise RuntimeError("policy vector shape mismatch")
    delta = b - a
    a_norm = float(torch.linalg.vector_norm(a))
    b_norm = float(torch.linalg.vector_norm(b))
    drift = float(torch.linalg.vector_norm(delta))
    denominator = max(a_norm * b_norm, np.finfo(float).tiny)
    cosine = float(torch.dot(a, b) / denominator)
    return {
        "l2_drift": drift,
        "relative_l2_drift": drift / max(a_norm, np.finfo(float).tiny),
        "start_l2_norm": a_norm,
        "end_l2_norm": b_norm,
        "cosine_similarity": cosine,
    }


def all_nested_tensors_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_nested_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_nested_tensors_finite(item) for item in value)
    return True


def stage_metrics(
    initial: dict, pass1: dict, terminal: dict, prefix: str
) -> dict[str, float]:
    v0 = flatten_policy(initial["policy"], prefix)
    v1 = flatten_policy(pass1["policy"], prefix)
    v2 = flatten_policy(terminal["policy"], prefix)
    result: dict[str, float] = {}
    for label, start, end in (
        ("initial_to_pass1", v0, v1),
        ("pass1_to_terminal", v1, v2),
        ("initial_to_terminal", v0, v2),
    ):
        for metric, value in vector_metrics(start, end).items():
            result[f"{prefix.rstrip('.')}_{label}_{metric}"] = value
    result[f"{prefix.rstrip('.')}_parameter_count"] = int(v0.numel())
    return result


def validate_episode_records(records: list[dict], fold: int, replica: int) -> None:
    if len(records) != 80:
        raise RuntimeError(f"fold {fold} replica {replica}: expected 80 records")
    observed = [
        (int(row["Pass"]), int(row["Position"])) for row in records
    ]
    expected = [(pass_id, position) for pass_id in (1, 2) for position in range(1, 41)]
    if observed != expected:
        raise RuntimeError(f"fold {fold} replica {replica}: pass order mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    chain_rows = []
    pass_rows = []
    for fold in range(1, 6):
        for replica in range(5):
            pass1_marker = load_marker(args.checkpoint_dir, fold, replica, "pass1")
            terminal_marker = load_marker(args.checkpoint_dir, fold, replica, "terminal")
            pass1 = load_object(args.checkpoint_dir, pass1_marker)
            terminal = load_object(args.checkpoint_dir, terminal_marker)
            if int(pass1_marker["completed_episodes"]) != 40:
                raise RuntimeError("pass-1 marker does not contain 40 episodes")
            if int(terminal_marker["completed_episodes"]) != 80:
                raise RuntimeError("terminal marker does not contain 80 episodes")
            records = list(terminal["episode_records"])
            validate_episode_records(records, fold, replica)

            initial = terminal["initial_training_state"]
            pass1_state = pass1["training_state"]
            terminal_state = terminal["training_state"]
            if pass1["initial_training_state"]["policy"].keys() != initial["policy"].keys():
                raise RuntimeError("initial-state schema mismatch")

            row = {
                "Fold": fold,
                "Replica": replica,
                "pass1_completed_episodes": 40,
                "terminal_completed_episodes": 80,
                "initial_policy_version": int(initial["policy_version"]),
                "pass1_policy_version": int(pass1_state["policy_version"]),
                "terminal_policy_version": int(terminal_state["policy_version"]),
                "pass1_checkpoint_sha256": pass1_marker["checkpoint_sha256"],
                "terminal_checkpoint_sha256": terminal_marker["checkpoint_sha256"],
                "initial_weight_semantic_sha256": terminal_marker[
                    "initial_weight_semantic_sha256"
                ],
                "pass1_weight_semantic_sha256": pass1_marker[
                    "weight_semantic_sha256"
                ],
                "terminal_weight_semantic_sha256": terminal_marker[
                    "weight_semantic_sha256"
                ],
                "pass1_policy_finite": all_nested_tensors_finite(
                    pass1_state["policy"]
                ),
                "terminal_policy_finite": all_nested_tensors_finite(
                    terminal_state["policy"]
                ),
                "pass1_optimizer_finite": all_nested_tensors_finite(
                    pass1_state["optimizer"]
                ),
                "terminal_optimizer_finite": all_nested_tensors_finite(
                    terminal_state["optimizer"]
                ),
            }
            row.update(stage_metrics(initial, pass1_state, terminal_state, "actor."))
            row.update(stage_metrics(initial, pass1_state, terminal_state, "critic."))
            chain_rows.append(row)

            record_frame = pd.DataFrame(records)
            for pass_id, group in record_frame.groupby("Pass", sort=True):
                entropy = pd.to_numeric(group["Operator_entropy"], errors="raise")
                pass_rows.append(
                    {
                        "Fold": fold,
                        "Replica": replica,
                        "Pass": int(pass_id),
                        "n_episodes": int(len(group)),
                        "operator_count_entropy_median": float(entropy.median()),
                        "operator_count_entropy_first10_median": float(
                            entropy.iloc[:10].median()
                        ),
                        "operator_count_entropy_last10_median": float(
                            entropy.iloc[-10:].median()
                        ),
                        "transition_generation_median": float(
                            pd.to_numeric(group["Transition_gen"]).median()
                        ),
                        "ppo_samples_total": int(
                            pd.to_numeric(group["PPO_samples"]).sum()
                        ),
                        "ppo_updates_total": int(
                            pd.to_numeric(group["PPO_updates"]).sum()
                        ),
                        "objective_evaluations_total": int(
                            (
                                pd.to_numeric(group["Initial_evaluations"])
                                + pd.to_numeric(group["Offspring_evaluations"])
                            ).sum()
                        ),
                    }
                )

    chains = pd.DataFrame(chain_rows).sort_values(["Fold", "Replica"])
    passes = pd.DataFrame(pass_rows).sort_values(["Fold", "Replica", "Pass"])
    if len(chains) != 25 or len(passes) != 50:
        raise RuntimeError("formal audit requires 25 chains and 50 pass summaries")
    if not chains[
        [
            "pass1_policy_finite",
            "terminal_policy_finite",
            "pass1_optimizer_finite",
            "terminal_optimizer_finite",
        ]
    ].all(axis=None):
        raise RuntimeError("non-finite checkpoint or optimizer tensor detected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chain_path = args.out_dir / "pretraining_checkpoint_audit.csv"
    pass_path = args.out_dir / "pretraining_pass_diagnostics.csv"
    chains.to_csv(chain_path, index=False)
    passes.to_csv(pass_path, index=False)
    print(chains.describe(include="all").transpose().to_string())
    print(passes.groupby("Pass").median(numeric_only=True).to_string())
    print(f"wrote {chain_path}")
    print(f"wrote {pass_path}")


if __name__ == "__main__":
    main()
