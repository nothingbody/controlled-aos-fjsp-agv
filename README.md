# Controlled Finite-Budget Evaluation of UCB-to-BC-to-PPO Operator Selection

This repository contains the code, frozen protocols, run logs, Pareto fronts,
and analysis artifacts for the controlled evaluation of
stage-aware adaptive operator selection in multi-objective flexible job-shop
scheduling with automated guided vehicles (FJSP-AGV). The archive covers the
18,500-run primary campaign, the separately frozen 1,100-run E4 mechanism
study, 6,000 held-out E5 evaluations, and an outcome-informed 1,350-run E4-R
replication: 26,950 test runs in total. E5 also records 2,000 offline
pretraining episodes separately.

## Scope

The study evaluates **static, energy-aware flexible job-shop scheduling with
AGVs** under a fixed NSGA-III backbone and a fixed ten-operator library. The
three optimization objectives are makespan, a normalized total-energy index,
and workload imbalance. The repository is an empirical audit package rather
than a claim of a generally superior solver.

The selector acts at **generation level**: one operator is selected once per
generation and produces all 100 offspring. Consequently, 50, 100, and 200
generations expose only 50, 100, and 200 selector decisions. This decision
granularity aligns one reward with one population transition, but it is not
implied by the objective-evaluation budget; finer offspring-batch or
parent-pair control could supply substantially more action samples without
more schedule evaluations. The reported boundary is therefore specific to
this generation-level contract.

The revised scope does **not** include dynamic job arrivals, machine or AGV failures, conflict-free AGV routing, critical-path PPO, graph encoding, or graded rescheduling.

## Method

Stage-Aware Adaptive Operator Selection (SA-AOS) uses:

1. an early sliding-window UCB controller to ensure operator coverage;
2. behavior cloning of the UCB state-action records to warm-start the actor;
3. an adaptive handover to PPO;
4. strictly on-policy PPO rollouts after the handover.

The completed E1--E4 studies use no pretrained checkpoint or cross-instance
policy transfer: every instance--seed run initializes its actor, critic,
optimizers, and buffers independently. A separately prespecified E5 protocol
tests cross-instance PPO pretraining under strict out-of-fold evaluation;
its offline training cost is recorded separately from the held-out deployment
budget.

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
| E5 | Four controllers under five-fold held-out transfer evaluation | 6,000 |
| E4-R | New-instance replication of the E4 mechanism contrasts | 1,350 |
| **Total** | Completed controlled test runs | **26,950** |

The E5 total excludes its 2,000 pretraining episodes because those episodes
are not pooled with held-out runs as inferential replicates.

The primary design and analysis rules are in
[`SCI_Paper/RESUBMISSION_EXPERIMENT_PROTOCOL.md`](SCI_Paper/RESUBMISSION_EXPERIMENT_PROTOCOL.md).
The E4 protocol and the dated reference-point amendment are in
[`SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md`](SCI_Paper/MECHANISM_ROBUSTNESS_PROTOCOL_V6.md)
and
[`SCI_Paper/MECHANISM_ROBUSTNESS_ANALYSIS_AMENDMENT_V6_1.md`](SCI_Paper/MECHANISM_ROBUSTNESS_ANALYSIS_AMENDMENT_V6_1.md).

## Prespecified extensions

Two prospective extensions were frozen after E1--E4. E5 and E4-R are complete,
and their prespecified analyses have passed their integrity gates.

- **E5/v7** changes only the source of PPO parameters. Five cross-fitted
  training folds provide strictly held-out policies for 50 test instances.
  The frozen design contains 2,000 pretraining episodes and 6,000 held-out
  test runs. The completed output contains exactly 2,000 unique pretraining
  keys and 6,000 unique evaluation keys, validates 25 terminal checkpoint
  chains through recorded file and semantic hashes, and has no failure marker. See
  [`SCI_Paper/CROSS_INSTANCE_PRETRAINING_PROTOCOL_V7.md`](SCI_Paper/CROSS_INSTANCE_PRETRAINING_PROTOCOL_V7.md).
- **E4-R/v8** is an outcome-informed prospective replication on 30 new
  inferential instances, excluding all ten instances used in the original E4.
  The completed output contains exactly 1,350 unique run keys, 1,350 Pareto
  fronts, 1,350 evaluation journals, and no failure marker. See
  [`SCI_Paper/E4_REPLICATION_PROTOCOL_V8.md`](SCI_Paper/E4_REPLICATION_PROTOCOL_V8.md).

## Main empirical finding

The final results do not support uniform SA-AOS superiority.

- At 50 generations, AdaptiveSAOS ranks close to UCBOnly and significantly outperforms several PPO or fixed-transition controls.
- At 100 and 200 generations, UCBOnly has a better mean rank, and the paired common-HV comparison favors UCBOnly after Holm correction.
- The behavior-cloning handover changes the operator-selection distribution, but it does not provide a stable optimization advantage over `AdaptiveNoBC`. A read-only E1 audit finds lower action-count entropy for SA-AOS than for the no-BC control in all 50 instance blocks; this is a persistent behavioral signature, not a causal attribution.
- The median number of action-effective PPO updates is 0, 1, and 8 for SA-AOS at 50, 100, and 200 generations. Generation-zero PPO-only increases those medians to 3, 6, and 12, and the E4 rollout-8 intervention reaches 15 at 200 generations; neither pressure test establishes learned-controller superiority.
- The six reward definitions in E2 are not significantly different in the instance-blocked Friedman test.
- In E4, exact UCB-context features improve in-sample behavior-cloning fit but do not yield a detected hypervolume gain.
- Rollout 8 produces more action-effective PPO updates than rollout 16 or 32, but its prespecified rollout contrast does not survive Holm correction.
- The exploratory rollout-8 versus UCB-only comparison is post hoc, unadjusted, and outside every confirmatory family; it does not support a superiority claim.
- In E5, descriptive online-transferred-PPO effects are adverse relative to UCB-only and scratch adaptive PPO in all five held-out fold summaries at all three budgets. Every pair of fold-specific training sets overlaps on 30 of 40 instances, so the 32-sign enumeration is reported only as a low-resolution sensitivity, not as an exact randomization test. Its raw value is 0.0625; Holm-adjusted sensitivities are 0.125 in the primary family and 0.250 in the secondary family.
- Every leave-one-fold-out transfer estimate is negative, while the dependent fold-sign sensitivity does not distinguish online fine-tuning from freezing the transferred policy.
- All pretraining episodes use 200 generations. The 50- and 100-generation tests therefore combine cross-instance and cross-horizon transfer; only the 200-generation test is horizon matched.
- E5 pretraining consumes 2,000 episodes and 40.2 million objective evaluations; more offline data does not recover this frozen transfer design.
- A post-hoc E5 behavior audit finds that the transferred policy remains strongly concentrated on `UniformMA`, whereas scratch PPO stays close to maximum ten-action entropy. This identifies a persistent behavioral prior but does not establish that the concentration causes the held-out HV loss.
- Read-only instance-level and run-level mixed-model sensitivities preserve the adverse E5 direction. They remain descriptive because instances share fold-specific checkpoints, training sets overlap, and all six mixed models issue variance-boundary warnings.
- In E4-R, none of the ten fixed-reference hypervolume contrasts is significant after Holm correction. The three PPO-versus-UCB medians remain negative; their raw two-sided p-values are 0.0325--0.0382, but their adjusted p-values are 0.0974.
- Across the nine contrasts shared with E4, E4-R reproduces seven effect directions and eight familywise significance decisions. The original generation-200 PPO-versus-UCB significance does not replicate, although its adverse direction does.

The completed baseline family contains random selection, fixed cycling,
probability matching, adaptive pursuit, coverage-aware sliding-window UCB,
PPO-only, and UCB-to-PPO handover controls. It does **not** contain Thompson
sampling, EXP3, dynamic MAB, Ropke--Pisinger segment-based ALNS weighting, or a
same-backbone reproduction of an external imitation-learning selector.
Accordingly, the results compare specified controllers rather than bandit,
adaptive-weighting, or learned-selector method classes.

The evidence therefore supports a **conditional design boundary** for
within-run and cross-instance UCB-to-PPO operator control. E4 diagnoses
mechanisms using ten inferential instances; E4-R repeats the frozen contrast
family on 30 new inferential instances; E5 has five dependent fold summaries
containing 50 descriptive out-of-fold instance blocks under one frozen transfer
protocol. E4-R was specified after the original E4
outcomes were known and is therefore a bounded, outcome-informed replication,
not an independent discovery study. Nonsignificant mechanism contrasts
establish neither absence of an effect nor practical equivalence.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q

# Full validated pipeline; choose a worker count permitted by the host.
WORKERS=12 bash scripts/run_remote_resubmission_pipeline.sh

# The E4 runner is resumable and validates the frozen hashes before execution.
python experiments/run_mechanism_robustness_v6.py --workers 40

# Reproduce the amended E4 analysis from the archived runs and fronts.
python scripts/analyze_mechanism_robustness_v6_1.py

# E5/v7: run the frozen pretraining and held-out evaluation pipeline, then
# analyze only after the completion gate accepts all 8,000 records.
python experiments/run_cross_instance_pretraining_v7.py --phase all
python scripts/analyze_cross_instance_pretraining_v7.py
python scripts/analyze_e5_transfer_behavior.py
python scripts/analyze_e5_fold_sign_sensitivity_v7.py
python scripts/analyze_e5_replica_extremes_v7.py

# Read-only MC1/MC3/MC5 revision audit: PPO data supply, BC/no-BC behavior,
# G=50 count reconciliation, and descriptive E5 sensitivities.
python scripts/audit_revision_mc1_mc3_mc5.py

# Checkpoint-process audit requires the archived pass-1 and terminal checkpoint
# objects; it never changes an optimizer state.
python scripts/audit_e5_checkpoint_process_v7.py --checkpoint-dir /path/to/checkpoints

# E4-R/v8: reproduce the completed new-instance replication and analysis.
python experiments/run_e4_replication_v8.py
python scripts/analyze_e4_replication_v8.py
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
results/resubmission/v7_cross_instance/ E5 ledgers, held-out rows, and analysis
results/resubmission/v8_e4_replication/ E4-R frozen reference and formal outputs
results/resubmission/revision_readonly_mc/ Read-only revision diagnostics
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

The E5 completion marker is
`results/resubmission/v7_cross_instance/pipeline_complete.json`: 2,000 unique
pretraining records, 25 terminal-checkpoint hash records, and 6,000 unique
held-out test keys. The E4-R completion marker is
`results/resubmission/v8_e4_replication/pipeline_complete.json`: 1,350 unique
run keys, 1,350 fronts, and 1,350 evaluation journals. Its front-manifest
SHA-256 is
`803c72ffe5bf5434ad5f0d55579ad08c5de1f418ade562b23a541c2d2c8dec47`,
and the fixed-reference confirmatory audit reports no invalid run.

The additional E5 audit tables in
`results/resubmission/v7_cross_instance/analysis/` report dependent fold-sign
sensitivity, checkpoint-replica/assigned-seed sensitivities, front-extreme
denominator checks, and pass-1/terminal checkpoint parameter diagnostics. The
archived checkpoint ledgers did not retain episode reward/HV, policy loss,
value loss, clip fraction, approximate KL, or intermediate checkpoints; those
training curves cannot be reconstructed retrospectively and are not claimed.

The read-only revision audit in
`results/resubmission/revision_readonly_mc/` reports the PPO data-supply
ladder, E1 BC-versus-no-BC behavioral contrasts, the 50-generation handover
count reconciliation, and descriptive E5 instance-level and mixed-model
sensitivities. It does not alter any frozen run, front, checkpoint, protocol,
or confirmatory family.

The earlier primary-plus-E4 archive remains available as
[GitHub release v6.0.1](https://github.com/nothingbody/controlled-aos-fjsp-agv/releases/tag/v6.0.1).
The completed E5 and E4-R evidence packages are published through the current
repository history and its accompanying pull request.

Manuscript text, submission packages, response letters, graphical abstracts,
and journal-production assets are intentionally excluded from this public
repository.

## Data interpretation

The source FJSP benchmarks do not provide factory layouts or measured power data. AGV layouts and energy parameters are deterministic synthetic extensions, and energy is reported as a normalized index rather than kWh. Results support comparisons under the declared model; they should not be interpreted as measured industrial energy savings.

The 130 raw Brandimarte and Hurink benchmark files were byte-verified against the MIT-licensed SchedulingLab `fjsp-instances` repository at commit `ac4c3402312bfbeafcf4472d78be567d4e6b46ab`. Only the 10 Brandimarte and 40 Hurink edata instances are part of the frozen v5 experiment. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the bundled upstream license for provenance and attribution.

Saved Pareto fronts use Python pickle files for exact archival compatibility. Pickle can execute code while loading; load these files only from a trusted checkout. The CSV summaries and audit tables can be inspected without deserializing the fronts.

## License

No open-source license has been selected for the original project code yet. Copyright remains with the author unless a root license file is added explicitly. The benchmark files under `data/benchmarks/` retain their upstream MIT license and attribution.


