"""Freeze v5 normalization and IGD+ references for the E4-R/v8 instances."""

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
OUT = ROOT / "results/resubmission/v8_e4_replication/frozen_reference_snapshot.pkl"
NORMALIZATION = ROOT / "results/resubmission/v5/e3_budget/analysis/normalization.json"
RUNS = ROOT / "results/resubmission/v5/e3_budget/runs.csv"
BUDGETS = (100, 200)
EXPECTED_RUNS_PER_BLOCK = 70
SELECTED_INSTANCES = {
    "Brandimarte": (
        "Mk02.fjs", "Mk04.fjs", "Mk06.fjs", "Mk07.fjs", "Mk09.fjs",
    ),
    "Hurink_edata": (
        "la03.fjs", "la05.fjs", "la06.fjs", "la07.fjs", "la08.fjs",
        "la09.fjs", "la12.fjs", "la13.fjs", "la14.fjs", "la15.fjs",
        "la17.fjs", "la18.fjs", "la19.fjs", "la22.fjs", "la24.fjs",
        "la25.fjs", "la28.fjs", "la29.fjs", "la33.fjs", "la34.fjs",
        "la35.fjs", "la36.fjs", "la37.fjs", "la38.fjs", "la39.fjs",
    ),
}
ORIGINAL_E4 = {
    "Brandimarte": {"Mk01.fjs", "Mk03.fjs", "Mk05.fjs", "Mk08.fjs", "Mk10.fjs"},
    "Hurink_edata": {"la01.fjs", "la10.fjs", "la20.fjs", "la30.fjs", "la40.fjs"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_unique(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]


def nondominated(points: np.ndarray) -> np.ndarray:
    points = stable_unique(points)
    indices = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[np.asarray(indices, dtype=int)]


def locate(raw: object) -> Path:
    raw = Path(str(raw))
    for candidate in (raw, RUNS.parent / raw, ROOT / raw):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def load_front(path: Path, expected) -> np.ndarray:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    checks = {
        "dataset": (payload.get("dataset"), expected.dataset),
        "instance": (payload.get("instance"), expected.instance),
        "variant": (payload.get("variant"), expected.variant),
        "seed": (int(payload.get("seed", -1)), int(expected.seed)),
    }
    mismatch = {
        key: {"observed": observed, "expected": wanted}
        for key, (observed, wanted) in checks.items() if observed != wanted
    }
    if mismatch:
        raise RuntimeError(f"v5 front metadata mismatch in {path}: {mismatch}")
    points = np.asarray(payload.get("objectives", []), dtype=float)
    if (
        points.ndim != 2 or points.shape[1] < 3 or len(points) == 0
        or not np.isfinite(points[:, :3]).all()
    ):
        raise RuntimeError(f"invalid v5 front {path}: {points.shape}")
    return stable_unique(points[:, :3])


def validate_selection() -> None:
    if sum(map(len, SELECTED_INSTANCES.values())) != 30:
        raise RuntimeError("v8 selection must contain exactly 30 instances")
    for family, names in SELECTED_INSTANCES.items():
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate v8 names in {family}")
        overlap = set(names) & ORIGINAL_E4[family]
        if overlap:
            raise RuntimeError(f"v8 overlaps original E4 in {family}: {overlap}")


def build_snapshot() -> dict:
    validate_selection()
    all_normalization = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    wanted = {
        (family, instance, budget)
        for family, names in SELECTED_INSTANCES.items()
        for instance in names
        for budget in BUDGETS
    }
    normalization = {
        (record["dataset"], record["instance"], int(record["Budget"])): record
        for record in all_normalization
        if (record["dataset"], record["instance"], int(record["Budget"])) in wanted
    }
    if set(normalization) != wanted:
        raise RuntimeError(
            f"v8 normalization grid mismatch missing={sorted(wanted-set(normalization))[:5]}"
        )
    frame = pd.read_csv(RUNS)
    key_columns = ["dataset", "instance", "variant", "Budget", "seed"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("duplicate v5 E3 source keys")
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
                f"incomplete v5 source for {dataset}/{instance}/g{budget}: {len(subset)}"
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
    return {
        "snapshot_protocol": "saos_v8_fixed_v5_reference_20260722",
        "normalization": normalization,
        "reference_sets": references,
        "selected_instances": SELECTED_INSTANCES,
        "source_manifest": {
            "normalization_sha256": sha256(NORMALIZATION),
            "runs_sha256": sha256(RUNS),
            "front_count": len(front_hash_records),
            "front_composite_sha256": hashlib.sha256(
                "\n".join(sorted(front_hash_records)).encode("utf-8")
            ).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable v8 snapshot: {args.output}")
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
        "selected_instances": {
            family: list(names) for family, names in SELECTED_INSTANCES.items()
        },
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
