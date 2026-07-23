# Reproducibility guide

## 1. Obtain the immutable source release

```bash
git clone https://github.com/nothingbody/controlled-aos-fjsp-agv.git
cd controlled-aos-fjsp-agv
git checkout v8.1.0-reproducibility
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints-postcampaign-20260723.txt
python -m pytest -q
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## 2. Download the E5 raw assets

Download the three assets from
[`v8.1.0-reproducibility`](https://github.com/nothingbody/controlled-aos-fjsp-agv/releases/tag/v8.1.0-reproducibility)
into one directory:

- `e5-v7-fronts-6000.tar.gz`
- `e5-v7-evaluation-journals-6000.tar.gz`
- `e5-v7-checkpoints-pass1-terminal-50.tar.gz`

Verify them before extraction:

```bash
python scripts/verify_reproducibility_release.py /path/to/downloaded/assets
```

The verifier checks archive SHA-256 values, byte sizes, safe member paths,
exact file counts, and all 50 checkpoint-object hashes against their bundled
metadata.

Extract each archive under the E5 result directory:

```bash
tar -xzf /path/to/assets/e5-v7-fronts-6000.tar.gz \
  -C results/resubmission/v7_cross_instance
tar -xzf /path/to/assets/e5-v7-evaluation-journals-6000.tar.gz \
  -C results/resubmission/v7_cross_instance
mkdir -p results/resubmission/v7_cross_instance/checkpoints
tar -xzf /path/to/assets/e5-v7-checkpoints-pass1-terminal-50.tar.gz \
  -C results/resubmission/v7_cross_instance/checkpoints
```

The checkpoint archive contains pass-1 and terminal objects for all 25
training chains. Superseded per-episode progress objects are intentionally
omitted; they are not inputs to any reported analysis.

## 3. Reproduce the archived analyses

```bash
python scripts/analyze_mechanism_robustness_v6_1.py
python scripts/analyze_cross_instance_pretraining_v7.py
python scripts/analyze_e5_transfer_behavior.py
python scripts/analyze_e5_fold_sign_sensitivity_v7.py
python scripts/analyze_e5_replica_extremes_v7.py
python scripts/audit_e5_checkpoint_process_v7.py \
  --checkpoint-dir results/resubmission/v7_cross_instance/checkpoints
python scripts/analyze_e4_replication_v8.py
python scripts/audit_revision_mc1_mc3_mc5.py
```

The complete E1--E3 pipeline and the E4/E5/E4-R runners are computationally
expensive. Run them only when a full re-execution is intended; their commands
and worker options are documented in `README.md` and the frozen protocol
files.

## 4. Integrity expectations

The checked-out repository and release assets together provide:

| Block | Run/pretraining ledger | Raw fronts | Journals or block logs | Checkpoint evidence |
|---|---:|---:|---:|---:|
| E1 | 5,000 | 5,000 | block log and run ledger | not applicable |
| E2 | 3,000 | 3,000 | block log and run ledger | not applicable |
| E3 | 10,500 | 10,500 | block log and run ledger | not applicable |
| E4 | 1,100 | 1,100 | block log and run ledger | not loaded |
| E5 | 6,000 test + 2,000 pretraining | 6,000 | 6,000 | 25 pass-1 + 25 terminal |
| E4-R | 1,350 | 1,350 | 1,350 | not loaded |

The 26,950 execution rows include 3,500 E3 rows at 100 generations that
repeat E1 design cells as a consistency bridge; these are not treated as
independent scientific evidence. Check each block's `pipeline_complete.json`
and run manifest before interpreting results.

## 5. Scope of the public archive

The public archive contains executable source, configurations, benchmark
inputs, frozen protocols, result ledgers, raw fronts, run/evaluation logs,
analysis outputs, and the checkpoint evidence needed by the reported E5
audit. Manuscript files, submission packages, response letters, graphical
abstracts, and journal-production assets are intentionally excluded.

No root open-source license has yet been granted for the original project
code. The bundled third-party benchmark files retain their upstream license
and attribution.

The Pareto fronts are Python pickle files and the checkpoint objects are
loaded by PyTorch. Both formats can execute code during deserialization. Do
not load them unless the release hashes have first passed
`scripts/verify_reproducibility_release.py`.
