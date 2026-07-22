"""Freeze v5 normalization and IGD+ reference sets for the E5/v7 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/resubmission/v7_cross_instance/frozen_reference_snapshot.pkl"
NORMALIZATION = ROOT / "results/resubmission/v5/e3_budget/analysis/normalization.json"
RUNS = ROOT / "results/resubmission/v5/e3_budget/runs.csv"
BUDGETS = (50, 100, 200)
EXPECTED_INSTANCES = 50
EXPECTED_RUNS_PER_BLOCK = 70


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_unique(points):
    points = np.asarray(points, dtype=float)
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]


def nondominated(points):
    points = stable_unique(points)
    indices = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[np.asarray(indices, dtype=int)]


def locate(raw):
    raw = Path(str(raw))
    for candidate in (raw, RUNS.parent / raw, ROOT / raw):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def load_front(path, expected):
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    checks = {
        "dataset": (payload.get("dataset"), expected.dataset),
        "instance": (payload.get("instance"), expected.instance),
        "variant": (payload.get("variant"), expected.variant),
        "seed": (int(payload.get("seed", -1)), int(expected.seed)),
    }
    mismatches = {
        key: {"observed": observed, "expected": wanted}
        for key, (observed, wanted) in checks.items() if observed != wanted
    }
    if mismatches:
        raise RuntimeError(f"v5 front metadata mismatch in {path}: {mismatches}")
    points = np.asarray(payload.get("objectives", []), dtype=float)
    if (
        points.ndim != 2 or points.shape[1] < 3 or len(points) == 0
        or not np.isfinite(points[:, :3]).all()
    ):
        raise RuntimeError(f"invalid v5 front {path}: {points.shape}")
    return stable_unique(points[:, :3])


def build_snapshot():
    normalization_records = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    normalization = {
        (record["dataset"], record["instance"], int(record["Budget"])): record
        for record in normalization_records
        if int(record["Budget"]) in BUDGETS
    }
    if len(normalization) != EXPECTED_INSTANCES * len(BUDGETS):
        raise RuntimeError(
            f"expected {EXPECTED_INSTANCES * len(BUDGETS)} normalization blocks, "
            f"observed {len(normalization)}"
        )
    frame = pd.read_csv(RUNS)
    frame = frame[pd.to_numeric(frame["Budget"]).isin(BUDGETS)].copy()
    key_columns = ["dataset", "instance", "variant", "Budget", "seed"]
    if frame.duplicated(key_columns, keep=False).any():
        raise RuntimeError("duplicate source keys in frozen v5 E3 runs")
    references = {}
    front_hash_records = []
    for key, record in sorted(normalization.items()):
        dataset, instance, budget = key
        subset = frame[
            (frame["dataset"] == dataset)
            & (frame["instance"] == instance)
            & (pd.to_numeric(frame["Budget"]) == budget)
        ].sort_values(["variant", "seed"])
        if len(subset) != EXPECTED_RUNS_PER_BLOCK:
            raise RuntimeError(
                f"incomplete v5 E3 source for {dataset}/{instance}/g{budget}: "
                f"{len(subset)} != {EXPECTED_RUNS_PER_BLOCK}"
            )
        ideal = np.asarray(record["ideal"], dtype=float)
        nadir = np.asarray(record["nadir"], dtype=float)
        scale = np.where(nadir > ideal, nadir - ideal, 1.0)
        normalized = []
        for row in subset.itertuples(index=False):
            path = locate(row.front_pickle)
            normalized.append((load_front(path, row) - ideal) / scale)
            front_hash_records.append(f"{path.relative_to(ROOT)}:{sha256(path)}")
        references[key] = nondominated(np.vstack(normalized))
    front_digest = hashlib.sha256(
        "\n".join(sorted(front_hash_records)).encode("utf-8")
    ).hexdigest()
    return {
        "snapshot_protocol": "saos_v7_fixed_v5_reference_20260722",
        "normalization": normalization,
        "reference_sets": references,
        "source_manifest": {
            "normalization_sha256": sha256(NORMALIZATION),
            "runs_sha256": sha256(RUNS),
            "front_count": len(front_hash_records),
            "front_composite_sha256": front_digest,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable snapshot: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    manifest = {
        "snapshot_protocol": snapshot["snapshot_protocol"],
        "snapshot_sha256": sha256(args.output),
        "blocks": len(snapshot["reference_sets"]),
        "reference_points": {
            f"{dataset}/{instance}/g{budget}": int(len(points))
            for (dataset, instance, budget), points
            in snapshot["reference_sets"].items()
        },
        "source_manifest": snapshot["source_manifest"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "snapshot_sha256": manifest["snapshot_sha256"],
        "blocks": manifest["blocks"],
        "front_count": manifest["source_manifest"]["front_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
