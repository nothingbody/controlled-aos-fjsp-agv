"""Formal analysis for the frozen E4-R/v8 new-instance replication.

The entry point is deliberately fail closed: it does not create analysis output
until the 1,350-row result grid, completion/hash chain, immutable reference
snapshot, and every saved Pareto front have passed a fresh audit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_e4_replication_v8 import (  # noqa: E402
    CODE_FILES,
    CONFIGS,
    FORMAL_BUDGETS,
    FORMAL_POP_SIZE,
    FORMAL_ROWS,
    FORMAL_SEEDS,
    FORMAL_WORKERS,
    PROTOCOL,
    SELECTED_INSTANCES,
    canonical_hash,
    config_hash,
    file_sha256,
    formal_design,
    front_semantic_hash,
    is_deduplicated_nondominated,
)
from scripts.analyze_resubmission_v5 import (  # noqa: E402
    bootstrap_ci,
    holm_adjust,
    nondominated,
    rank_biserial_from_diff,
    stable_unique_rows,
)


DEFAULT_ROOT = ROOT / "results/resubmission/v8_e4_replication"
DEFAULT_OUT = DEFAULT_ROOT / "analysis"
DEFAULT_V6_COMPARISONS = (
    ROOT / "results/resubmission/v6_mechanism/analysis_v6_1/pairwise_fixed_box_hv.csv"
)
REFERENCE_NAME = "frozen_reference_snapshot.pkl"
SNAPSHOT_PROTOCOL = "saos_v8_fixed_v5_reference_20260722"
TIE_TOL = 1e-12
BOOTSTRAP_REPS = 10_000
SIGN_FLIP_REPS = 1_000_000

STATE = (
    (100, "EnhancedBC_R16", "BasePaddedBC_R16", "enhanced_vs_padded_with_bc"),
    (100, "EnhancedNoBC_R16", "BasePaddedNoBC_R16", "enhanced_vs_padded_without_bc"),
)
BC = (
    (100, "BasePaddedBC_R16", "BasePaddedNoBC_R16", "bc_minus_no_bc_padded"),
    (100, "EnhancedBC_R16", "EnhancedNoBC_R16", "bc_minus_no_bc_enhanced"),
)
ROLLOUT = (
    (200, "EnhancedBC_R8", "EnhancedBC_R16", "rollout8_minus_rollout16"),
    (200, "EnhancedBC_R32", "EnhancedBC_R16", "rollout32_minus_rollout16"),
)
UCB = (
    (100, "EnhancedBC_R16", "UCBOnly", "enhanced16_minus_ucb_g100"),
    (200, "EnhancedBC_R16", "UCBOnly", "enhanced16_minus_ucb_g200"),
    (200, "EnhancedBC_R8", "UCBOnly", "enhanced8_minus_ucb_g200"),
)


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _expected_keys() -> set[tuple[str, str, str, int, int]]:
    return {
        (dataset, instance, config["variant"], int(budget), int(seed))
        for dataset, instances in SELECTED_INSTANCES.items()
        for instance in instances
        for budget, configs in CONFIGS.items()
        for config in configs
        for seed in FORMAL_SEEDS
    }


def validate_grid(
    frame: pd.DataFrame, *, hashes: dict, verify_config_hashes: bool = True,
) -> dict:
    """Validate the exact frozen grid and all row-level execution hashes."""
    required = {
        "Protocol", "dataset", "instance", "variant", "Budget", "seed",
        "Population_size", "Max_generations", "Worker_count",
        "Initial_evaluations", "Offspring_evaluations", "Config_hash",
        "Code_hash", "Design_hash", "Input_hash", "Reference_snapshot_sha256",
        "Front_sha256", "Front_semantic_sha256", "front_pickle",
        "PPO_update_count", "PPO_samples", "PPO_action_effective_updates",
        "PPO_terminal_full_updates", "PPO_terminal_residual_updates",
        "BC_epoch_loss", "BC_epoch_accuracy", "BC_confusion_matrix",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing required v8 columns: {sorted(missing)}")
    key_columns = ["dataset", "instance", "variant", "Budget", "seed"]
    duplicates = frame.duplicated(key_columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, key_columns].head().to_dict("records")
        raise RuntimeError(f"duplicate v8 result keys: {sample}")

    observed = {
        (str(row.dataset), str(row.instance), str(row.variant),
         int(row.Budget), int(row.seed))
        for row in frame.itertuples(index=False)
    }
    expected = _expected_keys()
    if len(frame) != FORMAL_ROWS or observed != expected:
        raise RuntimeError(
            f"v8 grid mismatch: rows={len(frame)} unique={len(observed)} "
            f"missing={len(expected - observed)} unexpected={len(observed - expected)}"
        )
    if set(frame["Protocol"].astype(str)) != {PROTOCOL}:
        raise RuntimeError("v8 CSV protocol mismatch")
    hash_columns = (
        ("Code_hash", "code_hash"), ("Design_hash", "design_hash"),
        ("Input_hash", "input_hash"),
        ("Reference_snapshot_sha256", "reference_snapshot_sha256"),
    )
    for column, key in hash_columns:
        if key not in hashes or set(frame[column].astype(str)) != {str(hashes[key])}:
            raise RuntimeError(f"v8 CSV {column} does not match the run manifest")

    numeric = {}
    for column in (
        "Budget", "seed", "Population_size", "Max_generations", "Worker_count",
        "Initial_evaluations", "Offspring_evaluations",
    ):
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"v8 CSV {column} is not integral") from error
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"v8 CSV {column} is not strictly integral")
        numeric[column] = values.astype(int)
    if set(numeric["Population_size"]) != {FORMAL_POP_SIZE}:
        raise RuntimeError("v8 population size differs from the frozen value 100")
    if set(numeric["Worker_count"]) != {FORMAL_WORKERS}:
        raise RuntimeError("v8 worker count differs from the frozen value 40")
    if not np.array_equal(numeric["Max_generations"], numeric["Budget"]):
        raise RuntimeError("v8 Max_generations differs from Budget")
    if set(numeric["Initial_evaluations"]) != {FORMAL_POP_SIZE}:
        raise RuntimeError("v8 initial-evaluation count mismatch")
    expected_offspring = FORMAL_POP_SIZE * numeric["Budget"].to_numpy()
    if not np.array_equal(numeric["Offspring_evaluations"], expected_offspring):
        raise RuntimeError("v8 offspring-evaluation count mismatch")

    if verify_config_hashes:
        expected_hashes = {
            (int(budget), config["variant"]): config_hash(config, int(budget))
            for budget, configs in CONFIGS.items() for config in configs
        }
        wrong = [
            (row.dataset, row.instance, row.variant, int(row.Budget), int(row.seed))
            for row in frame.itertuples(index=False)
            if str(row.Config_hash) != expected_hashes[(int(row.Budget), row.variant)]
        ]
        if wrong:
            raise RuntimeError(f"v8 configuration hash mismatch: {wrong[:3]}")
    return {
        "rows": int(len(frame)), "unique_keys": int(len(observed)),
        "instances": 30, "seeds": list(FORMAL_SEEDS),
        "budgets": list(FORMAL_BUDGETS), "status": "complete",
        **{key: hashes[key] for _, key in hash_columns},
    }


def _canonical_token_hash(path: Path) -> str:
    tokens = path.read_text(encoding="utf-8").split()
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def verify_current_sources(manifest: dict) -> None:
    code_files = manifest.get("code_files")
    if not isinstance(code_files, dict) or set(code_files) != set(CODE_FILES):
        raise RuntimeError("run manifest has an incomplete v8 code-file inventory")
    actual_code = {}
    for relative, recorded_hash in code_files.items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != str(recorded_hash):
            raise RuntimeError(f"current v8 code differs from executed code: {relative}")
        actual_code[relative] = str(recorded_hash)
    if canonical_hash(actual_code) != manifest.get("code_hash"):
        raise RuntimeError("run-manifest code hash is not canonical for its file inventory")

    input_files = manifest.get("input_files")
    if not isinstance(input_files, dict) or not input_files:
        raise RuntimeError("run manifest has no benchmark input-file inventory")
    semantic = {}
    for relative, record in input_files.items():
        path = ROOT / relative
        if not path.is_file() or not isinstance(record, dict):
            raise RuntimeError(f"missing current benchmark input: {relative}")
        observed = _canonical_token_hash(path)
        if observed != record.get("canonical_token_sha256"):
            raise RuntimeError(f"current benchmark input differs from executed input: {relative}")
        semantic[relative] = observed
    if canonical_hash(semantic) != manifest.get("input_hash"):
        raise RuntimeError("run-manifest input hash is not canonical for its file inventory")


def load_snapshot(experiment_root: Path, manifest: dict) -> dict:
    path = experiment_root / REFERENCE_NAME
    if not path.is_file():
        raise RuntimeError(f"missing immutable v8 reference snapshot: {path}")
    observed_hash = file_sha256(path)
    if observed_hash != manifest.get("reference_snapshot_sha256"):
        raise RuntimeError("v8 reference snapshot hash mismatch")
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    if snapshot.get("snapshot_protocol") != SNAPSHOT_PROTOCOL:
        raise RuntimeError("unexpected v8 reference snapshot protocol")
    expected = {
        (dataset, instance, budget)
        for dataset, instances in SELECTED_INSTANCES.items()
        for instance in instances for budget in FORMAL_BUDGETS
    }
    normalization = snapshot.get("normalization", {})
    references = snapshot.get("reference_sets", {})
    if set(normalization) != expected or set(references) != expected:
        raise RuntimeError("v8 reference snapshot must contain the exact 60 frozen blocks")
    selected = {
        family: tuple(names)
        for family, names in snapshot.get("selected_instances", {}).items()
    }
    if selected != SELECTED_INSTANCES:
        raise RuntimeError("v8 reference snapshot instance selection mismatch")
    for key in expected:
        record = normalization[key]
        for field in ("ideal", "nadir", "reference"):
            values = np.asarray(record.get(field, []), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid snapshot normalization {key}/{field}")
        ideal = np.asarray(record["ideal"], dtype=float)
        nadir = np.asarray(record["nadir"], dtype=float)
        fixed_reference = np.asarray(record["reference"], dtype=float)
        if not np.all(nadir > ideal):
            raise RuntimeError(f"non-positive frozen normalization range for {key}")
        if not np.allclose(fixed_reference, np.full(3, 1.1), rtol=0.0, atol=TIE_TOL):
            raise RuntimeError(f"snapshot fixed reference is not (1.1,1.1,1.1) for {key}")
        reference = np.asarray(references[key], dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference):
            raise RuntimeError(f"invalid empty IGD+ reference set for {key}")
        if not np.isfinite(reference).all():
            raise RuntimeError(f"non-finite IGD+ reference set for {key}")
        if (
            len(stable_unique_rows(reference)) != len(reference)
            or len(nondominated(reference)) != len(reference)
        ):
            raise RuntimeError(f"IGD+ reference set is not deduplicated nondominated for {key}")
    return snapshot


def preflight(experiment_root: Path) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    experiment_root = Path(experiment_root).resolve()
    failures = [
        path for path in (
            experiment_root / "failures.json",
            experiment_root / "pipeline_failed.txt",
        ) if path.exists()
    ]
    if failures:
        raise RuntimeError(f"failure marker(s) prohibit v8 analysis: {failures}")
    paths = {
        "manifest": experiment_root / "run_manifest.json",
        "completion": experiment_root / "pipeline_complete.json",
        "runs": experiment_root / "runs.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"formal v8 output is incomplete; missing {missing}")
    manifest = load_json(paths["manifest"])
    completion = load_json(paths["completion"])
    if manifest.get("protocol") != PROTOCOL or completion.get("protocol") != PROTOCOL:
        raise RuntimeError("v8 manifest/completion protocol mismatch")
    frozen_design = formal_design()
    if manifest.get("design") != frozen_design:
        raise RuntimeError("run manifest differs from the frozen v8 design")
    if manifest.get("design_hash") != canonical_hash(frozen_design):
        raise RuntimeError("run manifest has an invalid v8 design hash")
    required_hashes = (
        "code_hash", "design_hash", "input_hash", "reference_snapshot_sha256",
    )
    hashes = {}
    for key in required_hashes:
        if not manifest.get(key) or completion.get(key) != manifest.get(key):
            raise RuntimeError(f"completion/run-manifest {key} mismatch")
        hashes[key] = manifest[key]
    if int(completion.get("rows", -1)) != FORMAL_ROWS:
        raise RuntimeError("v8 completion marker does not certify 1350 rows")
    if int(completion.get("unique_keys", -1)) != FORMAL_ROWS:
        raise RuntimeError("v8 completion marker does not certify 1350 unique keys")
    verify_current_sources(manifest)
    snapshot = load_snapshot(experiment_root, manifest)
    frame = pd.read_csv(paths["runs"])
    completeness = validate_grid(frame, hashes=hashes)
    return frame, manifest, completion, snapshot, completeness


def locate_front(raw: object, experiment_root: Path) -> Path:
    value = Path(str(raw))
    for candidate in (value, experiment_root / value, ROOT / value):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"cannot locate v8 front: {raw}")


def validate_payload_diagnostics(payload: dict, row: dict, path: Path) -> None:
    """Re-audit the saved mechanism traces and the single-policy update boundary."""
    sequence_fields = ("operator_sequence", "reward_sequence", "hv_delta_sequence")
    for field in sequence_fields:
        values = payload.get(field)
        if not isinstance(values, (list, tuple)) or len(values) != int(row["Budget"]):
            raise RuntimeError(f"missing or incomplete {field} in {path}")
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise RuntimeError(f"non-finite {field} in {path}")
    operators = np.asarray(payload["operator_sequence"], dtype=float)
    if not np.equal(operators, np.floor(operators)).all() or not np.all(
        (operators >= 0) & (operators < 10)
    ):
        raise RuntimeError(f"invalid operator sequence in {path}")

    bc_stats = payload.get("behavior_cloning_stats")
    ppo_stats = payload.get("ppo_update_stats")
    if not isinstance(bc_stats, dict) or not isinstance(ppo_stats, list):
        raise RuntimeError(f"missing BC/PPO diagnostic records in {path}")
    previous_updated = None
    allowed_contexts = {"pre_action", "terminal_full", "terminal_residual"}
    for index, update in enumerate(ppo_stats):
        if not isinstance(update, dict):
            raise RuntimeError(f"invalid PPO update record {index} in {path}")
        try:
            behavior = int(update["behavior_policy_version"])
            updated = int(update["updated_policy_version"])
            rollout_size = int(update["rollout_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"incomplete PPO update record {index} in {path}") from error
        if updated != behavior + 1 or (
            previous_updated is not None and behavior != previous_updated
        ):
            raise RuntimeError(f"mixed or discontinuous policy versions in {path}")
        if rollout_size < 2 or update.get("update_context") not in allowed_contexts:
            raise RuntimeError(f"invalid on-policy update boundary in {path}")
        previous_updated = updated

    scalar_checks = {
        "PPO_update_count": len(ppo_stats),
        "PPO_samples": sum(int(item["rollout_size"]) for item in ppo_stats),
        "PPO_action_effective_updates": sum(
            item.get("update_context") == "pre_action" for item in ppo_stats
        ),
        "PPO_terminal_full_updates": sum(
            item.get("update_context") == "terminal_full" for item in ppo_stats
        ),
        "PPO_terminal_residual_updates": sum(
            item.get("update_context") == "terminal_residual" for item in ppo_stats
        ),
    }
    for field, expected in scalar_checks.items():
        if field not in row or int(row[field]) != int(expected):
            raise RuntimeError(f"CSV/payload {field} mismatch in {path}")
    json_checks = {
        "BC_epoch_loss": bc_stats.get("bc_epoch_loss", []),
        "BC_epoch_accuracy": bc_stats.get("bc_epoch_accuracy", []),
        "BC_confusion_matrix": bc_stats.get("bc_confusion_matrix", []),
    }
    for field, expected in json_checks.items():
        try:
            observed = json.loads(str(row[field]))
        except (KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"missing valid CSV {field} in {path}") from error
        if observed != expected:
            raise RuntimeError(f"CSV/payload {field} mismatch in {path}")


def audit_fronts(
    frame: pd.DataFrame, experiment_root: Path, hashes: dict, completion: dict,
) -> list[tuple[dict, np.ndarray, Path]]:
    observed_manifest = hashlib.sha256(
        "\n".join(sorted(frame["Front_sha256"].astype(str))).encode("ascii")
    ).hexdigest()
    if observed_manifest != completion.get("front_manifest_sha256"):
        raise RuntimeError("v8 CSV front manifest differs from completion marker")
    audited = []
    for row in frame.to_dict(orient="records"):
        path = locate_front(row["front_pickle"], experiment_root)
        if file_sha256(path) != str(row["Front_sha256"]):
            raise RuntimeError(f"v8 front file hash mismatch: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        checks = {
            "protocol": (payload.get("protocol"), PROTOCOL),
            "dataset": (payload.get("dataset"), row["dataset"]),
            "instance": (payload.get("instance"), row["instance"]),
            "variant": (payload.get("variant"), row["variant"]),
            "budget": (int(payload.get("budget", -1)), int(row["Budget"])),
            "seed": (int(payload.get("seed", -1)), int(row["seed"])),
            "config_hash": (payload.get("config_hash"), row["Config_hash"]),
            "code_hash": (payload.get("code_hash"), hashes["code_hash"]),
            "design_hash": (payload.get("design_hash"), hashes["design_hash"]),
            "input_hash": (payload.get("input_hash"), hashes["input_hash"]),
            "reference_snapshot_sha256": (
                payload.get("reference_snapshot_sha256"),
                hashes["reference_snapshot_sha256"],
            ),
            "population_size": (
                int(payload.get("population_size", -1)), FORMAL_POP_SIZE,
            ),
        }
        mismatch = {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in checks.items() if observed != expected
        }
        points = np.asarray(payload.get("objectives", []), dtype=float)
        valid = (
            points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
            and np.isfinite(points).all()
            and is_deduplicated_nondominated(points)
            and front_semantic_hash(points) == str(row["Front_semantic_sha256"])
        )
        if mismatch or not valid:
            raise RuntimeError(f"v8 front scientific audit failed {path}: {mismatch}")
        validate_payload_diagnostics(payload, row, path)
        audited.append((row, points, path))
    return audited


def normalize(points: np.ndarray, record: dict) -> np.ndarray:
    ideal = np.asarray(record["ideal"], dtype=float)
    nadir = np.asarray(record["nadir"], dtype=float)
    scale = np.where(nadir > ideal, nadir - ideal, 1.0)
    return (np.asarray(points, dtype=float) - ideal) / scale


def igd_plus(reference: np.ndarray, approximation: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    distances = []
    for target in reference:
        deviation = np.maximum(approximation - target, 0.0)
        distances.append(np.linalg.norm(deviation, axis=1).min())
    return float(np.mean(distances))


def add_indicators(
    audited: list[tuple[dict, np.ndarray, Path]], snapshot: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audits = []
    for row, points, path in audited:
        key = (row["dataset"], row["instance"], int(row["Budget"]))
        record = snapshot["normalization"][key]
        projected = nondominated(stable_unique_rows(normalize(points, record)))
        fixed_reference = np.asarray(record["reference"], dtype=float)
        inside = np.all(projected <= fixed_reference + TIE_TOL, axis=1)
        fixed_points = nondominated(projected[inside]) if inside.any() else np.empty((0, 3))
        expanded_reference = np.full(3, 1.5, dtype=float)
        expanded_valid = bool(np.all(projected <= expanded_reference + TIE_TOL))
        enriched = dict(row)
        enriched["HV_fixed"] = (
            float(HV(ref_point=fixed_reference)(fixed_points))
            if len(fixed_points) else math.nan
        )
        enriched["HV_expanded_1p5"] = (
            float(HV(ref_point=expanded_reference)(projected))
            if expanded_valid else math.nan
        )
        enriched["IGDplus_fixed"] = igd_plus(
            np.asarray(snapshot["reference_sets"][key], dtype=float), projected
        )
        rows.append(enriched)
        above = projected > fixed_reference + TIE_TOL
        audits.append({
            "dataset": row["dataset"], "instance": row["instance"],
            "variant": row["variant"], "Budget": int(row["Budget"]),
            "seed": int(row["seed"]), "front_points": int(len(projected)),
            "fixed_inside_points": int(inside.sum()),
            "fixed_excluded_points": int((~inside).sum()),
            "fixed_invalid": bool(not inside.any()),
            "above_fixed_cmax": int(above[:, 0].sum()),
            "above_fixed_energy": int(above[:, 1].sum()),
            "above_fixed_workload": int(above[:, 2].sum()),
            "expanded_1p5_invalid": bool(not expanded_valid),
            "normalized_min": float(projected.min()),
            "normalized_max": float(projected.max()), "front_path": str(path),
        })
    enriched = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if enriched["HV_fixed"].isna().any():
        invalid = enriched.loc[
            enriched["HV_fixed"].isna(),
            ["dataset", "instance", "variant", "Budget", "seed"],
        ]
        raise RuntimeError(
            f"fixed-box HV has no eligible point for {len(invalid)} runs: "
            f"{invalid.head().to_dict('records')}"
        )
    return enriched, audit


def instance_seed_medians(enriched: pd.DataFrame) -> pd.DataFrame:
    group = ["dataset", "instance", "Budget", "variant"]
    required = set(group) | {"seed", "HV_fixed", "IGDplus_fixed"}
    if required - set(enriched.columns):
        raise RuntimeError(f"cannot form v8 medians; missing {sorted(required-set(enriched.columns))}")
    seed_sets = enriched.groupby(group, sort=True)["seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected = tuple(FORMAL_SEEDS)
    if len(seed_sets) != 270 or any(values != expected for values in seed_sets):
        raise RuntimeError("every v8 instance/configuration/budget cell must contain all five seeds")
    preferred = [
        "HV_fixed", "HV_expanded_1p5", "IGDplus_fixed", "Cmax_best",
        "TEC_best", "WB_best", "NSol", "Time", "CPU_time", "Learning_time",
        "Initial_evaluations", "Offspring_evaluations", "Transition_gen",
        "BC_final_accuracy", "BC_pre_post_KL", "Demo_nn_disagreement",
        "PPO_samples", "PPO_action_effective_updates", "PPO_terminal_full_updates",
        "PPO_terminal_residual_updates", "PPO_optimizer_steps",
        "PPO_approx_KL_mean", "PPO_entropy_mean",
    ]
    metrics = [metric for metric in preferred if metric in enriched.columns]
    numeric = enriched[metrics].apply(pd.to_numeric, errors="raise")
    working = enriched[group + ["seed"]].copy()
    working[metrics] = numeric
    medians = working.groupby(group, sort=True)[metrics].median().reset_index()
    medians["seed_count"] = len(FORMAL_SEEDS)
    return medians


def signed_rank_randomization(
    difference: np.ndarray, *, reps: int = SIGN_FLIP_REPS, seed: int = 20260722,
) -> tuple[float, float, int, str]:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > TIE_TOL)]
    if len(values) == 0:
        return 0.0, 1.0, 0, "all ties"
    ranks = rankdata(np.abs(values), method="average")
    total = float(ranks.sum())
    positive = float(ranks[values > 0].sum())
    statistic = min(positive, total - positive)
    deviation = abs(positive - total / 2.0)
    if len(values) <= 20:
        extreme = 0
        for signs in itertools.product((False, True), repeat=len(values)):
            signed_positive = float(ranks[np.asarray(signs, dtype=bool)].sum())
            extreme += abs(signed_positive - total / 2.0) >= deviation - TIE_TOL
        permutations = 2 ** len(values)
        return statistic, float(extreme / permutations), len(values), "exact paired Wilcoxon sign enumeration"
    if reps <= 0:
        raise ValueError("sign-flip reps must be positive")
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < reps:
        size = min(20_000, reps - completed)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8)
        signed_positive = signs @ ranks
        extreme += int(np.sum(np.abs(signed_positive - total / 2.0) >= deviation - TIE_TOL))
        completed += size
    p_value = (extreme + 1.0) / (reps + 1.0)
    return statistic, float(p_value), len(values), f"paired Wilcoxon fixed-seed sign flip ({reps} resamples)"


def _ordinary_contrast(
    medians: pd.DataFrame, budget: int, lhs: str, rhs: str, metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    subset = medians[pd.to_numeric(medians["Budget"]).astype(int) == int(budget)]
    pivot = subset.pivot(
        index=["dataset", "instance"], columns="variant", values=metric
    ).reset_index()
    if lhs not in pivot or rhs not in pivot:
        raise RuntimeError(f"missing v8 contrast columns for {lhs} vs {rhs} at g={budget}")
    orientation = 1.0 if higher_is_better else -1.0
    pivot["lhs_value"] = pivot[lhs]
    pivot["rhs_value"] = pivot[rhs]
    pivot["delta"] = orientation * (pivot[lhs] - pivot[rhs])
    if len(pivot) != 30 or pivot["delta"].isna().any():
        raise RuntimeError(f"incomplete 30-instance contrast for {lhs} vs {rhs} at g={budget}")
    return pivot[["dataset", "instance", "lhs_value", "rhs_value", "delta"]]


def _interaction_contrast(medians: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    subset = medians[pd.to_numeric(medians["Budget"]).astype(int) == 100]
    pivot = subset.pivot(
        index=["dataset", "instance"], columns="variant", values=metric
    ).reset_index()
    wanted = (
        "EnhancedBC_R16", "EnhancedNoBC_R16",
        "BasePaddedBC_R16", "BasePaddedNoBC_R16",
    )
    if any(name not in pivot for name in wanted):
        raise RuntimeError("missing v8 state-by-BC interaction columns")
    pivot["lhs_value"] = pivot["EnhancedBC_R16"] - pivot["EnhancedNoBC_R16"]
    pivot["rhs_value"] = pivot["BasePaddedBC_R16"] - pivot["BasePaddedNoBC_R16"]
    orientation = 1.0 if higher_is_better else -1.0
    pivot["delta"] = orientation * (pivot["lhs_value"] - pivot["rhs_value"])
    if len(pivot) != 30 or pivot["delta"].isna().any():
        raise RuntimeError("incomplete 30-instance v8 interaction contrast")
    return pivot[["dataset", "instance", "lhs_value", "rhs_value", "delta"]]


def _comparison_record(
    block: pd.DataFrame, *, family: str, contrast: str, budget: int,
    lhs: str, rhs: str, metric: str, higher_is_better: bool,
    bootstrap_reps: int, sign_flip_reps: int,
) -> dict:
    delta = block["delta"].to_numpy(dtype=float)
    seed = deterministic_seed(PROTOCOL, family, contrast, metric)
    statistic, p_raw, nonzero, method = signed_rank_randomization(
        delta, reps=sign_flip_reps, seed=seed
    )
    low, high = bootstrap_ci(delta, reps=bootstrap_reps, seed=seed, statistic=np.median)
    return {
        "family": family, "contrast": contrast, "metric": metric,
        "endpoint_role": (
            "confirmatory primary" if metric == "HV_fixed" else "supportive sensitivity"
        ),
        "Budget": int(budget), "lhs": lhs, "rhs": rhs,
        "orientation": (
            "positive favors lhs" if higher_is_better
            else "positive favors lhs (lower raw metric is better)"
        ),
        "n_instances": int(len(delta)),
        "median_delta_oriented": float(np.median(delta)),
        "instance_bootstrap_ci95_low": low,
        "instance_bootstrap_ci95_high": high,
        "wins": int(np.sum(delta > TIE_TOL)),
        "ties": int(np.sum(np.abs(delta) <= TIE_TOL)),
        "losses": int(np.sum(delta < -TIE_TOL)),
        "rank_biserial": rank_biserial_from_diff(delta, TIE_TOL),
        "signed_rank_statistic": statistic, "p_raw": p_raw,
        "n_nonzero": int(nonzero), "test_method": method,
    }


def inferential_analysis(
    medians: pd.DataFrame, *, bootstrap_reps: int = BOOTSTRAP_REPS,
    sign_flip_reps: int = SIGN_FLIP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bootstrap_reps <= 0 or sign_flip_reps <= 0:
        raise ValueError("resample counts must be positive")
    records = []
    blocks = []
    definitions = (
        ("M100_state", STATE), ("M100_bc", BC),
        ("M200_rollout", ROLLOUT), ("UCB_replication", UCB),
    )
    for metric, higher_is_better in (("HV_fixed", True), ("IGDplus_fixed", False)):
        for family, contrasts in definitions:
            for budget, lhs, rhs, contrast in contrasts:
                block = _ordinary_contrast(
                    medians, budget, lhs, rhs, metric, higher_is_better
                )
                records.append(_comparison_record(
                    block, family=family, contrast=contrast, budget=budget,
                    lhs=lhs, rhs=rhs, metric=metric,
                    higher_is_better=higher_is_better,
                    bootstrap_reps=bootstrap_reps, sign_flip_reps=sign_flip_reps,
                ))
                blocks.append(block.assign(
                    family=family, contrast=contrast, metric=metric, Budget=budget,
                    lhs=lhs, rhs=rhs,
                ))
        family = "M100_interaction"
        contrast = "state_by_bc_difference_in_differences"
        block = _interaction_contrast(medians, metric, higher_is_better)
        lhs = "(EnhancedBC_R16-EnhancedNoBC_R16)"
        rhs = "(BasePaddedBC_R16-BasePaddedNoBC_R16)"
        records.append(_comparison_record(
            block, family=family, contrast=contrast, budget=100,
            lhs=lhs, rhs=rhs, metric=metric, higher_is_better=higher_is_better,
            bootstrap_reps=bootstrap_reps, sign_flip_reps=sign_flip_reps,
        ))
        blocks.append(block.assign(
            family=family, contrast=contrast, metric=metric, Budget=100,
            lhs=lhs, rhs=rhs,
        ))
    result = pd.DataFrame(records)
    result["p_holm_within_family"] = np.nan
    result["holm_family_size"] = 0
    for (_, _), indices in result.groupby(["metric", "family"], sort=False).groups.items():
        indices = list(indices)
        result.loc[indices, "p_holm_within_family"] = holm_adjust(
            result.loc[indices, "p_raw"].to_numpy(dtype=float)
        )
        result.loc[indices, "holm_family_size"] = len(indices)
    result["holm_family_size"] = result["holm_family_size"].astype(int)
    result["holm_significant_0_05"] = result["p_holm_within_family"] < 0.05
    return result, pd.concat(blocks, ignore_index=True)


V6_SHARED_CONTRASTS = {
    "enhanced_vs_padded_with_bc": "enhanced_vs_padded_base_with_bc",
    "enhanced_vs_padded_without_bc": "enhanced_vs_padded_base_without_bc",
    "bc_minus_no_bc_padded": "bc_effect_padded_base_state",
    "bc_minus_no_bc_enhanced": "bc_effect_enhanced_state",
    "rollout8_minus_rollout16": "rollout8_vs_16",
    "rollout32_minus_rollout16": "rollout32_vs_16",
    "enhanced16_minus_ucb_g100": "enhanced16_vs_ucb_g100",
    "enhanced16_minus_ucb_g200": "enhanced16_vs_ucb_g200",
    "state_by_bc_difference_in_differences": "state_by_bc_difference_in_differences",
}


def _effect_direction(value: float) -> str:
    if value > TIE_TOL:
        return "positive"
    if value < -TIE_TOL:
        return "negative"
    return "tie"


def replication_comparison(v8_comparisons: pd.DataFrame, v6_path: Path) -> pd.DataFrame:
    """Compare shared v8 HV contrasts with their separately analyzed v6 results."""
    v6_path = Path(v6_path)
    if not v6_path.is_file():
        raise RuntimeError(f"missing v6 comparison table required for replication audit: {v6_path}")
    v6 = pd.read_csv(v6_path)
    required = {"contrast", "median_delta_oriented", "p_holm_within_family"}
    if required - set(v6.columns):
        raise RuntimeError(f"v6 comparison table lacks {sorted(required-set(v6.columns))}")
    if "metric" in v6.columns:
        v6 = v6[v6["metric"].astype(str) == "HV_fixed"]
    if v6.duplicated("contrast").any():
        raise RuntimeError("v6 comparison table has duplicate HV contrast rows")
    v6 = v6.set_index("contrast")
    v8 = v8_comparisons[v8_comparisons["metric"] == "HV_fixed"].set_index("contrast")
    rows = []
    for v8_name, v6_name in V6_SHARED_CONTRASTS.items():
        if v8_name not in v8.index or v6_name not in v6.index:
            raise RuntimeError(f"missing shared v6/v8 replication contrast: {v6_name}/{v8_name}")
        new = v8.loc[v8_name]
        old = v6.loc[v6_name]
        v8_effect = float(new["median_delta_oriented"])
        v6_effect = float(old["median_delta_oriented"])
        v8_p = float(new["p_holm_within_family"])
        v6_p = float(old["p_holm_within_family"])
        if not np.isfinite([v8_effect, v6_effect, v8_p, v6_p]).all():
            raise RuntimeError(f"non-finite v6/v8 replication result for {v8_name}")
        v8_direction = _effect_direction(v8_effect)
        v6_direction = _effect_direction(v6_effect)
        v8_decision = bool(v8_p < 0.05)
        v6_decision = bool(v6_p < 0.05)
        rows.append({
            "v8_family": new["family"], "v8_contrast": v8_name,
            "v6_contrast": v6_name, "v6_median_delta_oriented": v6_effect,
            "v8_median_delta_oriented": v8_effect,
            "v6_direction": v6_direction, "v8_direction": v8_direction,
            "direction_reproduced": bool(v6_direction == v8_direction),
            "v6_p_holm_within_family": v6_p,
            "v8_p_holm_within_family": v8_p,
            "v6_familywise_significant_0_05": v6_decision,
            "v8_familywise_significant_0_05": v8_decision,
            "familywise_decision_reproduced": bool(v6_decision == v8_decision),
            "interpretation": (
                "shared prespecified v6/v8 contrast"
                if v8_name != "enhanced16_minus_ucb_g200"
                else "shared outcome-informed UCB boundary contrast"
            ),
        })
    return pd.DataFrame(rows)


def performance_summary(medians: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    records = []
    for keys, block in medians.groupby(group, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, keys))
        row["instances"] = int(len(block))
        for metric in ("HV_fixed", "HV_expanded_1p5", "IGDplus_fixed"):
            if metric not in block:
                continue
            values = block[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if len(values) else math.nan
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if len(values) else math.nan
            row[f"{metric}_valid_instances"] = int(len(values))
        records.append(row)
    return pd.DataFrame(records)


def _write_outputs(out_dir: Path, outputs: dict[str, pd.DataFrame]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, frame in outputs.items():
        path = out_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        hashes[name] = file_sha256(path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--sign-flip-reps", type=int, default=SIGN_FLIP_REPS)
    parser.add_argument("--v6-comparisons", type=Path, default=DEFAULT_V6_COMPARISONS)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0 or args.sign_flip_reps <= 0:
        raise SystemExit("resample counts must be positive")

    experiment_root = args.experiment_root.resolve()
    frame, manifest, completion, snapshot, completeness = preflight(experiment_root)
    hashes = {
        key: manifest[key] for key in (
            "code_hash", "design_hash", "input_hash", "reference_snapshot_sha256",
        )
    }
    audited = audit_fronts(frame, experiment_root, hashes, completion)
    enriched, reference_audit = add_indicators(audited, snapshot)
    medians = instance_seed_medians(enriched)
    comparisons, contrast_blocks = inferential_analysis(
        medians, bootstrap_reps=args.bootstrap_reps,
        sign_flip_reps=args.sign_flip_reps,
    )
    reproduction = replication_comparison(comparisons, args.v6_comparisons)
    summary = performance_summary(medians, ["Budget", "variant"])
    family_summary = performance_summary(medians, ["dataset", "Budget", "variant"])
    outputs = {
        "runs_with_fixed_indicators.csv": enriched,
        "reference_orthant_audit.csv": reference_audit,
        "instance_seed_medians.csv": medians,
        "prespecified_comparisons.csv": comparisons,
        "prespecified_pairwise_hv.csv": comparisons[comparisons["metric"] == "HV_fixed"],
        "prespecified_pairwise_igdplus.csv": comparisons[comparisons["metric"] == "IGDplus_fixed"],
        "contrast_instance_blocks.csv": contrast_blocks,
        "v6_v8_replication_comparison.csv": reproduction,
        "summary_by_variant.csv": summary,
        "summary_by_instance_family.csv": family_summary,
    }
    output_hashes = _write_outputs(args.out_dir, outputs)
    analysis_manifest = {
        **completeness,
        "protocol": PROTOCOL,
        "analysis_role": "prespecified E4-R/v8 new-instance replication analysis",
        "run_manifest_sha256": file_sha256(experiment_root / "run_manifest.json"),
        "pipeline_complete_sha256": file_sha256(experiment_root / "pipeline_complete.json"),
        "runs_csv_sha256": file_sha256(experiment_root / "runs.csv"),
        "front_manifest_sha256": completion["front_manifest_sha256"],
        "reference_snapshot_sha256": manifest["reference_snapshot_sha256"],
        "instance_seed_cells": int(len(medians)),
        "bootstrap_reps": int(args.bootstrap_reps),
        "sign_flip_reps": int(args.sign_flip_reps),
        "fixed_reference_invalid_runs": int(reference_audit["fixed_invalid"].sum()),
        "fixed_reference_excluded_points": int(reference_audit["fixed_excluded_points"].sum()),
        "expanded_reference_invalid_runs": int(reference_audit["expanded_1p5_invalid"].sum()),
        "v6_comparisons_sha256": file_sha256(args.v6_comparisons),
        "shared_v6_v8_contrasts": int(len(reproduction)),
        "direction_reproduced": reproduction.loc[
            reproduction["direction_reproduced"], "v8_contrast"
        ].tolist(),
        "familywise_decision_reproduced": reproduction.loc[
            reproduction["familywise_decision_reproduced"], "v8_contrast"
        ].tolist(),
        "holm_families": {
            family: int(size) for family, size in
            comparisons[comparisons["metric"] == "HV_fixed"]
            .groupby("family").size().items()
        },
        "outputs": output_hashes,
    }
    manifest_path = args.out_dir / "analysis_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis_manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(analysis_manifest, indent=2))


if __name__ == "__main__":
    main()
