"""Run the frozen 30-instance E4 mechanism replication (E4-R/v8)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import experiments.run_mechanism_robustness_v6 as v6
from experiments.run_cross_instance_pretraining_v7 import (
    atomic_json,
    atomic_pickle,
    canonical_hash,
    file_sha256,
    front_semantic_hash,
    is_deduplicated_nondominated,
    nondominated_subset,
)


PROTOCOL = "saos_e4_replication_v8_20260722"
DEFAULT_OUT = "results/resubmission/v8_e4_replication"
REFERENCE_SNAPSHOT = (
    "results/resubmission/v8_e4_replication/frozen_reference_snapshot.pkl"
)
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
CONFIGS = {
    100: (
        {"variant": "UCBOnly", "state_mode": "none", "use_bc": False, "rollout": 0},
        {"variant": "BasePaddedBC_R16", "state_mode": "base_padded", "use_bc": True, "rollout": 16},
        {"variant": "BasePaddedNoBC_R16", "state_mode": "base_padded", "use_bc": False, "rollout": 16},
        {"variant": "EnhancedBC_R16", "state_mode": "enhanced", "use_bc": True, "rollout": 16},
        {"variant": "EnhancedNoBC_R16", "state_mode": "enhanced", "use_bc": False, "rollout": 16},
    ),
    200: (
        {"variant": "UCBOnly", "state_mode": "none", "use_bc": False, "rollout": 0},
        {"variant": "EnhancedBC_R8", "state_mode": "enhanced", "use_bc": True, "rollout": 8},
        {"variant": "EnhancedBC_R16", "state_mode": "enhanced", "use_bc": True, "rollout": 16},
        {"variant": "EnhancedBC_R32", "state_mode": "enhanced", "use_bc": True, "rollout": 32},
    ),
}
FORMAL_POP_SIZE = 100
FORMAL_BUDGETS = (100, 200)
FORMAL_SEEDS = tuple(range(52, 57))
FORMAL_WORKERS = 40
FORMAL_ROWS = 1350
CODE_FILES = (
    "experiments/run_e4_replication_v8.py",
    "experiments/run_mechanism_robustness_v6.py",
    "experiments/run_cross_instance_pretraining_v7.py",
    "experiments/run_revision_aos.py",
    "src/algorithm/grl/ppo_agent.py",
    "src/algorithm/nsga3/encoding.py",
    "src/algorithm/nsga3/crossover.py",
    "src/algorithm/nsga3/mutation.py",
    "src/algorithm/nsga3/decoding.py",
    "src/algorithm/nsga3/selection.py",
    "src/problem/instance.py",
    "src/problem/energy_model.py",
    "src/utils/metrics.py",
    "data/loader.py",
    "scripts/build_e4_replication_reference_snapshot_v8.py",
    "SCI_Paper/E4_REPLICATION_PROTOCOL_V8.md",
)


def configure_v6_module() -> None:
    """Bind imported v6 mechanics to the immutable v8 design in every process."""
    v6.PROTOCOL = PROTOCOL
    v6.DEFAULT_OUT = DEFAULT_OUT
    v6.REFERENCE_SNAPSHOT = REFERENCE_SNAPSHOT
    v6.SELECTED_INSTANCES = {
        family: list(names) for family, names in SELECTED_INSTANCES.items()
    }
    v6.CONFIGS = {budget: [dict(item) for item in configs] for budget, configs in CONFIGS.items()}
    v6.FORMAL_SEED_START = FORMAL_SEEDS[0]
    v6.FORMAL_SEED_END = FORMAL_SEEDS[-1] + 1
    v6.FORMAL_POP_SIZE = FORMAL_POP_SIZE
    v6.FORMAL_BUDGETS = FORMAL_BUDGETS
    v6.FORMAL_WORKERS = FORMAL_WORKERS
    v6.INPUT_FILES = tuple(
        f"{v6.DATASETS[family]}/{instance}"
        for family, names in SELECTED_INSTANCES.items()
        for instance in names
    )


def validate_design() -> None:
    if sum(len(names) for names in SELECTED_INSTANCES.values()) != 30:
        raise RuntimeError("v8 must contain exactly 30 instances")
    for family, names in SELECTED_INSTANCES.items():
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate v8 instances in {family}")
        overlap = set(names) & ORIGINAL_E4[family]
        if overlap:
            raise RuntimeError(f"v8 overlaps original E4: {family}/{sorted(overlap)}")
    expected = sum(
        len(SELECTED_INSTANCES[family]) * len(CONFIGS[budget]) * len(FORMAL_SEEDS)
        for family in SELECTED_INSTANCES for budget in FORMAL_BUDGETS
    )
    if expected != FORMAL_ROWS:
        raise RuntimeError(f"v8 design gives {expected} rows, expected {FORMAL_ROWS}")


def code_manifest() -> dict:
    missing = [path for path in CODE_FILES if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"missing v8 code files: {missing}")
    files = {path: file_sha256(path) for path in CODE_FILES}
    return {"files": files, "code_hash": canonical_hash(files)}


def input_manifest() -> dict:
    configure_v6_module()
    return v6.input_manifest()


def formal_design() -> dict:
    return {
        "protocol": PROTOCOL,
        "replication_role": "outcome-informed prospective replication on instances unused in E4",
        "population_size": FORMAL_POP_SIZE,
        "budgets": list(FORMAL_BUDGETS),
        "seeds": list(FORMAL_SEEDS),
        "instances": {key: list(value) for key, value in SELECTED_INSTANCES.items()},
        "excluded_original_e4": {
            key: sorted(value) for key, value in ORIGINAL_E4.items()
        },
        "configs": {str(key): list(value) for key, value in CONFIGS.items()},
        "expected_rows": FORMAL_ROWS,
        "workers": FORMAL_WORKERS,
    }


def build_manifest(*, smoke: bool = False, smoke_design: dict | None = None) -> dict:
    validate_design()
    code = code_manifest()
    inputs = input_manifest()
    snapshot = Path(REFERENCE_SNAPSHOT)
    if not snapshot.is_file():
        raise RuntimeError(f"missing immutable v8 reference snapshot: {snapshot}")
    design = formal_design() if not smoke else dict(smoke_design or {})
    return {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "design": design,
        "design_hash": canonical_hash(design),
        "code_hash": code["code_hash"],
        "code_files": code["files"],
        "input_hash": inputs["input_hash"],
        "input_files": inputs["files"],
        "reference_snapshot": REFERENCE_SNAPSHOT,
        "reference_snapshot_sha256": file_sha256(snapshot),
    }


def write_or_validate_manifest(
    out_dir: Path, *, smoke: bool = False, smoke_design: dict | None = None,
) -> dict:
    path = Path(out_dir) / "run_manifest.json"
    current = build_manifest(smoke=smoke, smoke_design=smoke_design)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable = (
            "protocol", "hostname", "platform", "python", "numpy", "torch",
            "design_hash", "code_hash", "input_hash", "reference_snapshot_sha256",
        )
        mismatch = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in stable if existing.get(key) != current.get(key)
        }
        if mismatch:
            raise RuntimeError(f"refusing incompatible v8 resume: {mismatch}")
        return existing
    if any((Path(out_dir) / name).exists() for name in ("runs.csv", "pipeline_complete.json")):
        raise RuntimeError("v8 result exists without an immutable run manifest")
    atomic_json(path, current)
    return current


def config_hash(config: dict, budget: int) -> str:
    configure_v6_module()
    return v6.configuration_hash(config, budget, FORMAL_POP_SIZE)


def evaluation_key(row: dict) -> tuple:
    return (
        row["dataset"], row["instance"], row["variant"],
        int(row["Budget"]), int(row["seed"]),
    )


def journal_path(out_dir: Path, row: dict) -> Path:
    digest = canonical_hash(list(evaluation_key(row)))
    return Path(out_dir) / "evaluation_journal" / f"{digest}.json"


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing to write empty v8 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise RuntimeError("inconsistent v8 CSV fields")
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def load_existing_rows(out_dir: Path, hashes: dict) -> dict[tuple, dict]:
    row_map: dict[tuple, dict] = {}
    csv_path = Path(out_dir) / "runs.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = evaluation_key(row)
                if key in row_map:
                    raise RuntimeError(f"duplicate v8 CSV key: {key}")
                row_map[key] = row
    for path in sorted((Path(out_dir) / "evaluation_journal").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = evaluation_key(row)
        if path.name != canonical_hash(list(key)) + ".json":
            raise RuntimeError(f"v8 journal key/name mismatch: {path}")
        row_map[key] = row
    for key, row in row_map.items():
        checks = (
            ("Protocol", PROTOCOL), ("Code_hash", hashes["code_hash"]),
            ("Design_hash", hashes["design_hash"]), ("Input_hash", hashes["input_hash"]),
            ("Reference_snapshot_sha256", hashes["reference_snapshot_sha256"]),
        )
        if any(str(row.get(name)) != str(expected) for name, expected in checks):
            raise RuntimeError(f"v8 row hash/protocol mismatch: {key}")
        expected_config = next(
            item for item in CONFIGS[int(row["Budget"])]
            if item["variant"] == row["variant"]
        )
        if row.get("Config_hash") != config_hash(expected_config, int(row["Budget"])):
            raise RuntimeError(f"v8 row config mismatch: {key}")
        front = Path(row["front_pickle"])
        if not front.is_file() or file_sha256(front) != row.get("Front_sha256"):
            raise RuntimeError(f"v8 front missing or modified: {front}")
    if row_map:
        write_csv_atomic(csv_path, sorted(row_map.values(), key=evaluation_key))
    return row_map


def run_single_v8(task):
    """Spawn-safe adapter around the frozen v6 scientific mechanics."""
    configure_v6_module()
    (
        dataset, instance_name, _instance, config, seed, pop_size, budget,
        out_dir, code_hash, design_hash, input_hash, _config_hash, worker_count,
        reference_snapshot_sha256,
    ) = task
    cpu_start = time.process_time()
    base_task = task[:-1]
    row = v6.run_single(base_task)
    row["Initial_evaluations"] = int(pop_size)
    row["Offspring_evaluations"] = int(pop_size) * int(budget)
    row["CPU_time"] = float(time.process_time() - cpu_start)
    row["Reference_snapshot_sha256"] = reference_snapshot_sha256
    row["Replication_role"] = "outcome-informed_new-instance_replication"
    path = Path(row["front_pickle"])
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    points = nondominated_subset(np.asarray(payload["objectives"], dtype=float)[:, :3])
    if not is_deduplicated_nondominated(points):
        raise RuntimeError("v8 final front canonicalization failed")
    payload["objectives"] = points
    payload["reference_snapshot_sha256"] = reference_snapshot_sha256
    payload["replication_role"] = row["Replication_role"]
    payload["front_semantic_sha256"] = front_semantic_hash(points)
    atomic_pickle(path, payload)
    row["Front_sha256"] = file_sha256(path)
    row["Front_semantic_sha256"] = payload["front_semantic_sha256"]
    if "NSol" in row:
        row["NSol"] = len(points)
    return row


def load_instances() -> list[tuple]:
    configure_v6_module()
    instances = v6.load_selected_instances()
    observed = {
        family: {name for dataset, name, _ in instances if dataset == family}
        for family in SELECTED_INSTANCES
    }
    expected = {family: set(names) for family, names in SELECTED_INSTANCES.items()}
    if observed != expected:
        raise RuntimeError(f"v8 loaded-instance mismatch: {observed}")
    return instances


def build_tasks(args, hashes: dict, completed: set[tuple]) -> list[tuple]:
    tasks = []
    instances = load_instances()
    per_family_seen = {family: 0 for family in SELECTED_INSTANCES}
    for dataset, instance_name, instance in instances:
        if args.instance_limit and per_family_seen[dataset] >= args.instance_limit:
            continue
        per_family_seen[dataset] += 1
        for budget in args.budgets:
            for config in CONFIGS[budget]:
                if args.variants and config["variant"] not in args.variants:
                    continue
                for seed in args.seeds:
                    key = (dataset, instance_name, config["variant"], budget, seed)
                    if key in completed:
                        continue
                    tasks.append((
                        dataset, instance_name, instance, dict(config), int(seed),
                        int(args.pop_size), int(budget), str(args.out_dir),
                        hashes["code_hash"], hashes["design_hash"], hashes["input_hash"],
                        config_hash(config, budget), int(args.workers),
                        hashes["reference_snapshot_sha256"],
                    ))
    return tasks


def verify_complete(out_dir: Path, hashes: dict) -> dict:
    row_map = load_existing_rows(out_dir, hashes)
    expected = {
        (dataset, instance, config["variant"], budget, seed)
        for dataset, names in SELECTED_INSTANCES.items()
        for instance in names
        for budget, configs in CONFIGS.items()
        for config in configs
        for seed in FORMAL_SEEDS
    }
    if set(row_map) != expected or len(row_map) != FORMAL_ROWS:
        raise RuntimeError(
            f"incomplete v8 grid rows={len(row_map)} missing={len(expected-set(row_map))} "
            f"extra={len(set(row_map)-expected)}"
        )
    front_hashes = []
    for key, row in row_map.items():
        if int(row["Initial_evaluations"]) != FORMAL_POP_SIZE:
            raise RuntimeError(f"v8 initial-evaluation mismatch: {key}")
        if int(row["Offspring_evaluations"]) != FORMAL_POP_SIZE * int(row["Budget"]):
            raise RuntimeError(f"v8 offspring-evaluation mismatch: {key}")
        path = Path(row["front_pickle"])
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        points = np.asarray(payload.get("objectives", []), dtype=float)
        checks = {
            "protocol": (payload.get("protocol"), PROTOCOL),
            "dataset": (payload.get("dataset"), row["dataset"]),
            "instance": (payload.get("instance"), row["instance"]),
            "variant": (payload.get("variant"), row["variant"]),
            "budget": (int(payload.get("budget", -1)), int(row["Budget"])),
            "seed": (int(payload.get("seed", -1)), int(row["seed"])),
            "code_hash": (payload.get("code_hash"), hashes["code_hash"]),
            "design_hash": (payload.get("design_hash"), hashes["design_hash"]),
            "input_hash": (payload.get("input_hash"), hashes["input_hash"]),
            "reference_snapshot_sha256": (
                payload.get("reference_snapshot_sha256"),
                hashes["reference_snapshot_sha256"],
            ),
            "front_semantic_sha256": (
                front_semantic_hash(points), row["Front_semantic_sha256"],
            ),
        }
        mismatch = {
            name: {"observed": observed, "expected": wanted}
            for name, (observed, wanted) in checks.items() if observed != wanted
        }
        if mismatch or not is_deduplicated_nondominated(points):
            raise RuntimeError(f"v8 front audit failed {path}: {mismatch}")
        for update in payload.get("ppo_update_stats", []):
            behavior = update.get("behavior_policy_version")
            updated = update.get("updated_policy_version")
            if behavior is not None and int(updated) != int(behavior) + 1:
                raise RuntimeError(f"v8 mixed-policy update in {path}")
        front_hashes.append(row["Front_sha256"])
    if any((Path(out_dir) / name).exists() for name in ("failures.json", "pipeline_failed.txt")):
        raise RuntimeError("v8 failure marker exists")
    return {
        "protocol": PROTOCOL,
        "rows": len(row_map), "unique_keys": len(row_map),
        "instances": 30, "seeds": list(FORMAL_SEEDS),
        "front_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(front_hashes)).encode("ascii")
        ).hexdigest(),
        **hashes,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def parse_strings(text: str | None) -> tuple[str, ...]:
    return tuple(value.strip() for value in (text or "").split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--pop-size", type=int, default=FORMAL_POP_SIZE)
    parser.add_argument("--budgets", default="100,200")
    parser.add_argument("--seeds", default="52,53,54,55,56")
    parser.add_argument("--variants", default="")
    parser.add_argument("--instance-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.budgets = parse_ints(args.budgets)
    args.seeds = parse_ints(args.seeds)
    args.variants = parse_strings(args.variants)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.smoke:
        frozen = {
            "out_dir": args.out_dir.resolve() == Path(DEFAULT_OUT).resolve(),
            "workers": args.workers == FORMAL_WORKERS,
            "pop_size": args.pop_size == FORMAL_POP_SIZE,
            "budgets": args.budgets == FORMAL_BUDGETS,
            "seeds": args.seeds == FORMAL_SEEDS,
            "variants": not args.variants,
            "instance_limit": args.instance_limit == 0,
        }
        if not all(frozen.values()):
            raise SystemExit(f"formal v8 arguments are frozen: {frozen}")
        manifest = write_or_validate_manifest(args.out_dir)
    else:
        if args.out_dir.resolve() == Path(DEFAULT_OUT).resolve():
            raise SystemExit("v8 smoke output cannot use the formal directory")
        smoke_design = {
            "smoke": True, "workers": args.workers, "pop_size": args.pop_size,
            "budgets": args.budgets, "seeds": args.seeds,
            "variants": args.variants, "instance_limit": args.instance_limit,
        }
        manifest = write_or_validate_manifest(
            args.out_dir, smoke=True, smoke_design=smoke_design
        )
    hashes = {
        key: manifest[key] for key in (
            "code_hash", "design_hash", "input_hash", "reference_snapshot_sha256"
        )
    }
    row_map = load_existing_rows(args.out_dir, hashes)
    tasks = build_tasks(args, hashes, set(row_map))
    print(
        f"[v8] protocol={PROTOCOL} pending={len(tasks)} out={args.out_dir}",
        flush=True,
    )
    failures = []
    start = time.time()
    workers = max(1, min(args.workers, len(tasks) or 1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_single_v8, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                row = future.result()
                atomic_json(journal_path(args.out_dir, row), row)
                row_map[evaluation_key(row)] = row
                if completed % 20 == 0:
                    write_csv_atomic(
                        args.out_dir / "runs.csv",
                        sorted(row_map.values(), key=evaluation_key),
                    )
                status = (
                    f"{row['dataset']}/{row['instance']} {row['variant']} "
                    f"g={row['Budget']} seed={row['seed']} T={float(row['Time']):.1f}s"
                )
            except Exception as error:
                dataset, instance, _, config, seed, _, budget, *_ = task
                failures.append({
                    "dataset": dataset, "instance": instance,
                    "variant": config["variant"], "budget": budget, "seed": seed,
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
                status = (
                    f"ERROR {dataset}/{instance} {config['variant']} "
                    f"g={budget} seed={seed}: {error!r}"
                )
            elapsed = time.time() - start
            eta = (len(tasks) - completed) * elapsed / max(completed, 1)
            print(f"[v8 {completed}/{len(tasks)}] {status} ETA={eta/60:.1f}min", flush=True)
    if failures:
        atomic_json(args.out_dir / "failures.json", failures)
        raise SystemExit(f"{len(failures)} v8 tasks failed")
    if row_map:
        write_csv_atomic(
            args.out_dir / "runs.csv", sorted(row_map.values(), key=evaluation_key)
        )
    if args.smoke:
        print("[v8] smoke complete; formal completeness gate skipped", flush=True)
        return
    completion = verify_complete(args.out_dir, hashes)
    completion["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    completion["elapsed_seconds_this_invocation"] = time.time() - start
    atomic_json(args.out_dir / "pipeline_complete.json", completion)
    print(f"[v8] COMPLETE rows={completion['rows']}", flush=True)


if __name__ == "__main__":
    main()
