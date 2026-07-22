# Controlled Evaluation of Stage-Aware Adaptive Operator Selection for FJSP-AGV

This repository contains the code, frozen protocols, run logs, Pareto fronts,
and analysis artifacts for the controlled evaluation of
stage-aware adaptive operator selection in multi-objective flexible job-shop
scheduling with automated guided vehicles (FJSP-AGV). The archive covers the
18,500-run primary campaign and the separately frozen 1,100-run E4 mechanism
study: 19,600 optimizer runs in total.

## Scope

The study evaluates **static, energy-aware flexible job-shop scheduling with
AGVs** under a fixed NSGA-III backbone and a fixed ten-operator library. The
three optimization objectives are makespan, a normalized total-energy index,
and workload imbalance. The repository is an empirical audit package rather
than a claim of a generally superior solver.

The revised scope does **not** include dynamic job arrivals, machine or AGV failures, conflict-free AGV routing, critical-path PPO, graph encoding, or graded rescheduling.

## Method

Stage-Aware Adaptive Operator Selection (SA-AOS) uses:

1. an early sliding-window UCB controller to ensure operator coverage;
2. behavior cloning of the UCB state-action records to warm-start the actor;
3. an adaptive handover to PPO;
4. strictly on-policy PPO rollouts after the handover.

No pretrained checkpoint or cross-instance policy transfer is used. Every instance-seed run initializes its actor, critic, optimizers, and buffers independently.

## Frozen experiment protocol

- Protocol ID: `saos_bc_onpolicy_ppo_v5_20260720`
- Benchmarks: 10 Brandimarte instances and 40 Hurink edata instances
- Seeds: 42--51
- Population size: 100
- Execution device: CPU, one numerical-library thread per worker
- Primary metric: three-objective common-reference hypervolume
- Main inference unit: benchmark instance after taking the within-cell median over seeds
- Statistical tests: Friedman and paired Wilcoxon signed-rank tests with Holm correction

| Experiment | Purpose | Runs |
|---|---|---:|
| E1 | Ten-controller comparison at 100 generations | 5,000 |
| E2 | Six reward definitions with AdaptiveSAOS | 3,000 |
| E3 | Seven controllers at 50, 100, and 200 generations | 10,500 |
| E4 | State information, behavior cloning, and rollout mechanisms | 1,100 |
| **Total** | Primary campaign plus the separately frozen mechanism study | **19,600** |

The primary design and analysis rules are in
[`SCI_Paper/RESUBMISSION_EXPERIMENT_PROTOCOL.md`](SCI_Paper/RESUBMISSION_EXPERIMENT_PROTOCOL.md).
The E4 protocol and the dated reference-point amendment are in
[`SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md`](SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md)
and
[`SCI_Paper/MECHANISM_ROBUSTNESS_ANALYSIS_AMENDMENT_V6_1.md`](SCI_Paper/MECHANISM_ROBUSTNESS_ANALYSIS_AMENDMENT_V6_1.md).

## Main empirical finding

The final results do not support uniform SA-AOS superiority.

- At 50 generations, AdaptiveSAOS ranks close to UCBOnly and significantly outperforms several PPO or fixed-transition controls.
- At 100 and 200 generations, UCBOnly has a better mean rank, and the paired common-HV comparison favors UCBOnly after Holm correction.
- The behavior-cloning handover changes the operator-selection distribution, but it does not provide a stable optimization advantage over `AdaptiveNoBC`.
- The six reward definitions in E2 are not significantly different in the instance-blocked Friedman test.
- In E4, exact UCB-context features improve in-sample behavior-cloning fit but do not yield a detected hypervolume gain.
- Rollout 8 produces more action-effective PPO updates than rollout 16 or 32, but its prespecified rollout contrast does not survive Holm correction.
- The exploratory rollout-8 versus UCB-only comparison is post hoc, unadjusted, and outside every confirmatory family; it does not support a superiority claim.

The evidence therefore supports a **conditional design boundary** for
within-run UCB-to-BC-to-PPO operator control. E4 diagnoses mechanisms using ten
inferential instances; its nonsignificant contrasts establish neither absence
of an effect nor practical equivalence.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Full validated pipeline; choose a worker count permitted by the host.
WORKERS=12 bash scripts/run_remote_resubmission_pipeline.sh

# The E4 runner is resumable and validates the frozen hashes before execution.
python experiments/run_mechanism_robustness_v6.py --workers 40

# Reproduce the amended E4 analysis from the archived runs and fronts.
python scripts/analyze_mechanism_robustness_v6_1.py
```

The full pipeline resumes from existing unique result keys, validates the frozen protocol, rejects duplicates, recomputes common-reference hypervolume, performs instance-blocked statistics, and audits every saved Pareto front.

## Repository layout

```text
configs/                         Experiment configuration
data/                            Benchmark and extension data
experiments/                     E1/E2/E3 experiment runners
src/                             FJSP-AGV model, NSGA-III, operators, SA-AOS
scripts/                         Validation, analysis, audit, and plotting tools
tests/                           Integrity and regression tests
SCI_Paper/                       Experiment protocols and reproducibility audits
results/resubmission/v5/         E1--E3 CSVs, fronts, logs, statistics, and audits
results/resubmission/v6_mechanism/ E4 raw results, fronts, audits, and amended analysis
```

The primary completion marker is
`results/resubmission/v5/pipeline_complete.json`. Final validation found
exactly 5,000 E1 rows, 3,000 E2 rows, and 10,500 E3 rows, with zero duplicate
keys, zero missing front files, and no numeric NaN or infinity values.

The E4 completion marker is
`results/resubmission/v6_mechanism/pipeline_complete.json`: 1,100 rows and
1,100 unique run keys. The front manifest SHA-256 is
`6add7c98069a9b1c6fac27e4387b6cf75c3a2f43e1713c54f589606aa9da4d72`.
The final analysis manifest records the fixed-reference audit, all four
affected fronts and 12 out-of-box points, the expanded-reference sensitivity,
IGD+, behavior-cloning diagnostics, and the explicitly post-hoc R8--UCB table.
No run was deleted and no coordinate was clipped.

The versioned code-and-data archive is available as
[GitHub release v6.0.1](https://github.com/nothingbody/controlled-aos-fjsp-agv/releases/tag/v6.0.1).

Manuscript text, submission packages, response letters, graphical abstracts,
and journal-production assets are intentionally excluded from this public
repository.

## Data interpretation

The source FJSP benchmarks do not provide factory layouts or measured power data. AGV layouts and energy parameters are deterministic synthetic extensions, and energy is reported as a normalized index rather than kWh. Results support comparisons under the declared model; they should not be interpreted as measured industrial energy savings.

The 130 raw Brandimarte and Hurink benchmark files were byte-verified against the MIT-licensed SchedulingLab `fjsp-instances` repository at commit `ac4c3402312bfbeafcf4472d78be567d4e6b46ab`. Only the 10 Brandimarte and 40 Hurink edata instances are part of the frozen v5 experiment. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the bundled upstream license for provenance and attribution.

Saved Pareto fronts use Python pickle files for exact archival compatibility. Pickle can execute code while loading; load these files only from a trusted checkout. The CSV summaries and audit tables can be inspected without deserializing the fronts.

## License

No open-source license has been selected for the original project code yet. Copyright remains with the author unless a root license file is added explicitly. The benchmark files under `data/benchmarks/` retain their upstream MIT license and attribution.
