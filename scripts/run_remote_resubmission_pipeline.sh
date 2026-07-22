#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/saos_resubmission_v5}"
PY="${PY:-$ROOT/.venv/bin/python}"
PROTOCOL="saos_bc_onpolicy_ppo_v5_20260720"

cd "$ROOT"

if [[ -z "${WORKERS:-}" ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "$quota" == "max" ]]; then
        WORKERS="$(nproc)"
    else
        WORKERS="$((quota / period))"
    fi
fi
WORKERS="$((WORKERS > 0 ? WORKERS : 1))"

PIPELINE_DIR="results/resubmission/v5"
mkdir -p "$PIPELINE_DIR"
FAILED_MARKER="$PIPELINE_DIR/pipeline_failed.txt"
COMPLETE_MARKER="$PIPELINE_DIR/pipeline_complete.json"
LOCK_FILE="$PIPELINE_DIR/pipeline.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'another resubmission pipeline already holds %s\n' "$LOCK_FILE" >&2
    exit 1
fi
rm -f "$FAILED_MARKER" "$COMPLETE_MARKER"
trap 'printf "failed_at=%s line=%s\n" "$(date -Iseconds)" "$LINENO" > "$FAILED_MARKER"' ERR

validate_partial_resume() {
    local csv_path="$1"
    [[ ! -f "$csv_path" ]] && return 0
    "$PY" - "$csv_path" "$PROTOCOL" <<'PY'
import sys
import pandas as pd

path, protocol = sys.argv[1:]
frame = pd.read_csv(path)
key = ["dataset", "instance", "variant", "reward_scheme", "seed"]
if "Budget" in frame.columns:
    key.append("Budget")
if frame.duplicated(key).any():
    raise SystemExit(f"partial resume contains duplicate primary keys: {path}")
if set(frame["Protocol"].astype(str)) != {protocol}:
    raise SystemExit(f"partial resume protocol mismatch: {path}")
print(f"resume preflight {path}: existing_rows={len(frame)}")
PY
}

validate_run() {
    local csv_path="$1"
    local expected_rows="$2"
    local expected_instances="$3"
    local expected_seeds="$4"
    "$PY" - "$csv_path" "$expected_rows" "$expected_instances" "$expected_seeds" "$PROTOCOL" <<'PY'
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

csv_path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_instances = int(sys.argv[3])
expected_seeds = int(sys.argv[4])
protocol = sys.argv[5]

frame = pd.read_csv(csv_path)
key = ["dataset", "instance", "variant", "reward_scheme", "seed"]
if "Budget" in frame.columns:
    key.append("Budget")
errors = []
if len(frame) != expected_rows:
    errors.append(f"rows={len(frame)} expected={expected_rows}")
if frame.duplicated(key).any():
    errors.append(f"duplicate_keys={int(frame.duplicated(key).sum())}")
if set(frame["Protocol"].astype(str)) != {protocol}:
    errors.append(f"protocols={sorted(frame['Protocol'].astype(str).unique())}")
if frame[["dataset", "instance"]].drop_duplicates().shape[0] != expected_instances:
    errors.append("unexpected instance count")
if frame["seed"].nunique() != expected_seeds:
    errors.append("unexpected seed count")
missing_fronts = [value for value in frame["front_pickle"] if not Path(value).is_file()]
if missing_fronts:
    errors.append(f"missing_fronts={len(missing_fronts)}")
duplicate_fronts = 0
nsol_mismatches = 0
for row in frame.itertuples(index=False):
    front_path = Path(row.front_pickle)
    if not front_path.is_file():
        continue
    with front_path.open("rb") as stream:
        objectives = np.asarray(pickle.load(stream)["objectives"], dtype=float)
    unique_count = len(np.unique(objectives, axis=0))
    duplicate_fronts += int(unique_count != len(objectives))
    nsol_mismatches += int(int(row.NSol) != unique_count)
if duplicate_fronts:
    errors.append(f"fronts_with_duplicate_objectives={duplicate_fronts}")
if nsol_mismatches:
    errors.append(f"nsol_mismatches={nsol_mismatches}")
if errors:
    raise SystemExit("; ".join(errors))
print(
    f"validated {csv_path}: rows={len(frame)} instances={expected_instances} "
    f"seeds={expected_seeds} fronts={len(frame)}"
)
PY
}

analyze_run() {
    local csv_path="$1"
    local out_dir="$2"
    local search_root="$3"
    local seeds="$4"
    local target="$5"
    "$PY" scripts/analyze_resubmission_v5.py "$csv_path" \
        --out-dir "$out_dir" \
        --search-root "$search_root" \
        --expected-seeds "$seeds" \
        --target "$target"
}

audit_fronts() {
    local csv_path="$1"
    local out_dir="$2"
    local search_root="$3"
    "$PY" scripts/audit_fronts_v5.py "$csv_path" \
        --out-dir "$out_dir" \
        --search-root "$search_root"
}

printf '[%s] pipeline start workers=%s protocol=%s\n' "$(date -Iseconds)" "$WORKERS" "$PROTOCOL"
mkdir -p "$PIPELINE_DIR/e1_aos"
validate_partial_resume "$PIPELINE_DIR/e1_aos/runs.csv"
"$PY" experiments/run_revision_aos.py \
    --experiment aos \
    --out-dir "$PIPELINE_DIR/e1_aos" \
    --result-file "$PIPELINE_DIR/e1_aos/runs.csv" \
    --pop-size 100 --max-gen 100 \
    --seed-start 42 --seed-end 52 \
    --workers "$WORKERS" \
    > "$PIPELINE_DIR/e1_aos/run.log" 2>&1
validate_run "$PIPELINE_DIR/e1_aos/runs.csv" 5000 50 10
analyze_run \
    "$PIPELINE_DIR/e1_aos/runs.csv" \
    "$PIPELINE_DIR/e1_aos/analysis" \
    "$PIPELINE_DIR/e1_aos" 10 AdaptiveSAOS
audit_fronts \
    "$PIPELINE_DIR/e1_aos/runs.csv" \
    "$PIPELINE_DIR/e1_aos/front_audit" \
    "$PIPELINE_DIR/e1_aos"

mkdir -p "$PIPELINE_DIR/e2_reward"
validate_partial_resume "$PIPELINE_DIR/e2_reward/runs.csv"
"$PY" experiments/run_revision_aos.py \
    --experiment reward \
    --out-dir "$PIPELINE_DIR/e2_reward" \
    --result-file "$PIPELINE_DIR/e2_reward/runs.csv" \
    --pop-size 100 --max-gen 100 \
    --seed-start 42 --seed-end 52 \
    --workers "$WORKERS" \
    > "$PIPELINE_DIR/e2_reward/run.log" 2>&1
validate_run "$PIPELINE_DIR/e2_reward/runs.csv" 3000 50 10
analyze_run \
    "$PIPELINE_DIR/e2_reward/runs.csv" \
    "$PIPELINE_DIR/e2_reward/analysis" \
    "$PIPELINE_DIR/e2_reward" 10 R5_composite
audit_fronts \
    "$PIPELINE_DIR/e2_reward/runs.csv" \
    "$PIPELINE_DIR/e2_reward/front_audit" \
    "$PIPELINE_DIR/e2_reward"

mkdir -p "$PIPELINE_DIR/e3_budget"
validate_partial_resume "$PIPELINE_DIR/e3_budget/runs.csv"
"$PY" experiments/run_revision_aos_budget_stress.py \
    --out-dir "$PIPELINE_DIR/e3_budget" \
    --result-file "$PIPELINE_DIR/e3_budget/runs.csv" \
    --pop-size 100 \
    --seed-start 42 --seed-end 52 \
    --workers "$WORKERS" \
    > "$PIPELINE_DIR/e3_budget/run.log" 2>&1
validate_run "$PIPELINE_DIR/e3_budget/runs.csv" 10500 50 10
analyze_run \
    "$PIPELINE_DIR/e3_budget/runs.csv" \
    "$PIPELINE_DIR/e3_budget/analysis" \
    "$PIPELINE_DIR/e3_budget" 10 AdaptiveSAOS
audit_fronts \
    "$PIPELINE_DIR/e3_budget/runs.csv" \
    "$PIPELINE_DIR/e3_budget/front_audit" \
    "$PIPELINE_DIR/e3_budget"

"$PY" - "$COMPLETE_MARKER" "$PROTOCOL" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol": sys.argv[2],
            "experiments": {"E1": 5000, "E2": 3000, "E3": 10500},
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
rm -f "$FAILED_MARKER"
printf '[%s] pipeline complete\n' "$(date -Iseconds)"
