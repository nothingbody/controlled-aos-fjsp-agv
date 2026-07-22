"""Create the immutable v5 normalization and IGD+ reference snapshot for v6."""

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
OUT = ROOT / "results/resubmission/v6_mechanism/frozen_reference_snapshot.pkl"
E1_NORMALIZATION = ROOT / "results/resubmission/v5/e1_aos/analysis/normalization.json"
E3_NORMALIZATION = ROOT / "results/resubmission/v5/e3_budget/analysis/normalization.json"
E1_RUNS = ROOT / "results/resubmission/v5/e1_aos/runs.csv"
E3_RUNS = ROOT / "results/resubmission/v5/e3_budget/runs.csv"

SELECTED = {
    "Brandimarte": ["Mk01.fjs", "Mk03.fjs", "Mk05.fjs", "Mk08.fjs", "Mk10.fjs"],
    "Hurink_edata": ["la01.fjs", "la10.fjs", "la20.fjs", "la30.fjs", "la40.fjs"],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
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


def load_normalization():
    records = []
    for record in json.loads(E1_NORMALIZATION.read_text(encoding="utf-8")):
        records.append({**record, "Budget": 100, "source": "v5_E1"})
    for record in json.loads(E3_NORMALIZATION.read_text(encoding="utf-8")):
        if int(record.get("Budget", -1)) == 200:
            records.append({**record, "Budget": 200, "source": "v5_E3"})
    return {
        (record["dataset"], record["instance"], int(record["Budget"])): record
        for record in records
        if record["instance"] in SELECTED.get(record["dataset"], [])
    }


def locate(raw, csv_path):
    raw = Path(str(raw))
    for candidate in (raw, csv_path.parent / raw, ROOT / raw):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def load_front(path, expected=None):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if expected is not None:
        for key, wanted in (
            ("dataset", expected.dataset),
            ("instance", expected.instance),
            ("variant", expected.variant),
            ("seed", int(expected.seed)),
        ):
            observed = payload.get(key)
            if key == "seed" and observed is not None:
                observed = int(observed)
            if observed != wanted:
                raise RuntimeError(
                    f"v5 payload mismatch in {path}: {key}={observed!r}, "
                    f"expected {wanted!r}"
                )
    points = np.asarray(payload["objectives"], dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise RuntimeError(f"invalid front {path}: {points.shape}")
    if not np.isfinite(points[:, :3]).all():
        raise RuntimeError(f"non-finite front {path}")
    return stable_unique(points[:, :3])


def build_snapshot():
    normalization = load_normalization()
    references = {}
    front_hash_records = []
    for budget, csv_path in ((100, E1_RUNS), (200, E3_RUNS)):
        frame = pd.read_csv(csv_path)
        if "Budget" in frame.columns:
            frame = frame[pd.to_numeric(frame["Budget"]) == budget]
        key_columns = ["dataset", "instance", "variant", "seed"]
        if "Budget" in frame.columns:
            key_columns.append("Budget")
        if frame.duplicated(key_columns, keep=False).any():
            raise RuntimeError(f"duplicate v5 source keys in {csv_path}")
        expected_per_instance = 100 if budget == 100 else 70
        for dataset, names in SELECTED.items():
            for instance in names:
                key = (dataset, instance, budget)
                record = normalization[key]
                ideal = np.asarray(record["ideal"], dtype=float)
                nadir = np.asarray(record["nadir"], dtype=float)
                scale = np.where(nadir > ideal, nadir - ideal, 1.0)
                subset = frame[
                    (frame["dataset"] == dataset) & (frame["instance"] == instance)
                ].sort_values(["variant", "seed"])
                if len(subset) != expected_per_instance:
                    raise RuntimeError(
                        f"incomplete v5 source grid for {dataset}/{instance}/g{budget}: "
                        f"{len(subset)} != {expected_per_instance}"
                    )
                normalized = []
                for row in subset.itertuples(index=False):
                    path = locate(row.front_pickle, csv_path)
                    points = load_front(path, expected=row)
                    normalized.append((points - ideal) / scale)
                    front_hash_records.append(
                        f"{path.relative_to(ROOT)}:{sha256(path)}"
                    )
                references[key] = nondominated(np.vstack(normalized))

    front_digest = hashlib.sha256(
        "\n".join(sorted(front_hash_records)).encode("utf-8")
    ).hexdigest()
    source_manifest = {
        "e1_normalization_sha256": sha256(E1_NORMALIZATION),
        "e3_normalization_sha256": sha256(E3_NORMALIZATION),
        "e1_runs_sha256": sha256(E1_RUNS),
        "e3_runs_sha256": sha256(E3_RUNS),
        "front_count": len(front_hash_records),
        "front_composite_sha256": front_digest,
    }
    return {
        "snapshot_protocol": "saos_v6_fixed_v5_reference_20260722",
        "normalization": normalization,
        "reference_sets": references,
        "source_manifest": source_manifest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable snapshot: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot()
    with args.output.open("wb") as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "snapshot_protocol": snapshot["snapshot_protocol"],
        "snapshot_sha256": sha256(args.output),
        "blocks": len(snapshot["reference_sets"]),
        "reference_points": {
            f"{dataset}/{instance}/g{budget}": int(len(points))
            for (dataset, instance, budget), points in snapshot["reference_sets"].items()
        },
        "source_manifest": snapshot["source_manifest"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
