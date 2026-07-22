# Resubmission experiment protocol

Protocol ID: `saos_bc_onpolicy_ppo_v5_20260720`

This document freezes the corrected experiment design before the full rerun. Results produced by an earlier protocol must not be appended, merged, or used to fill revised tables.

## Scope

The resubmission evaluates static energy-aware FJSP-AGV with a fixed NSGA-III backbone and a fixed ten-operator library. Critical-path PPO, graph encoding, dynamic job arrivals, machine failures, AGV failures, and graded rescheduling are excluded.

## Data

- Brandimarte `Mk01`–`Mk10`: 10 instances.
- Hurink edata `la01`–`la40`: 40 instances.
- Three AGVs and speed tiers `0.50`, `0.75`, and `1.00` for every instance.
- Total AGV energy includes both loaded travel and empty travel to the next pickup location.
- Synthetic AGV and energy parameters use an instance-specific deterministic seed derived from SHA-256 of `dataset-directory/filename` plus base seed 42.
- Unit-square layouts are scaled per instance so that the median inter-location travel time at speed `0.75` is exactly `0.30` of the median eligible processing time. This prevents a fixed coordinate range from making transportation negligible on some benchmarks and dominant on others.
- Machine power uses a source-inspired normalized construction: `phi ~ U[5,10]`, processing power `phi`, standby power `phi/4`, and setup power `phi/2`. Setup duration is `U[0.05,0.15]` times the instance median processing time.
- Loaded AGV power is `P_load(v)=xi*v^2`, where `xi ~ U[4,6]`; empty power is `0.6*P_load(v)`. Thus trip energy is `P(v)*distance/v`, and higher speed saves time but consumes more energy per unit distance.
- Machine standby energy is counted from that machine's first processing start to its final completion, excluding processing and setup periods. Machines are treated as powered down outside this active window.
- Because the source benchmarks have no physical layout or power measurements, total energy is reported as a normalized energy index, not as kWh. No industrial-energy generalization is claimed.
- The full source-file hashes and generated parameters are stored in `results/resubmission/manifests/benchmark_and_extension_manifest.json`.

## Common optimizer budget

- population size: 100;
- default generations: 100;
- independent algorithm seeds: 42–51;
- NSGA-III reference divisions: 8;
- tournament size: 5;
- execution device: CPU;
- one CPU thread inside each worker process.
- initialization quotas are exactly 30% SPT, 20% processing-energy oriented, 20% load balancing, and 30% random; the energy-oriented machine choice minimizes `P_proc[u] * p_iju`;
- environmental selection uses Deb--Jain extreme-point/intercept normalization and standard split-front niching;
- the bounded archive is updated from every evaluated parent and offspring before environmental truncation.

## Corrected SA-AOS training protocol

- UCB window: 50; exploration coefficient: 1.0.
- Transition: every operator selected at least three times, at least 30 UCB demonstrations, and either five consecutive generations without 0.1% HV improvement or the 70% latest-transition guard. The realized generation and trigger reason are recorded.
- Behavior cloning: actor only, 100 epochs, minibatch 32, Adam learning rate `1e-3`.
- PPO: initialized independently for every instance-seed run; no checkpoint; no cross-instance transfer.
- PPO rollout: 16 on-policy generations; four PPO epochs; actor/critic learning rate `3e-4`; gamma 0.99; GAE lambda 0.95; clipping 0.2.
- Nonterminal rollouts bootstrap from the next-state critic value.
- Python, NumPy, and PyTorch receive the same run seed.

## Experiment E1: controller comparison

Ten controllers are compared on all 50 instances and ten seeds (5,000 runs): Random, UniformFixed, ProbabilityMatching, AdaptivePursuit, UCBOnly, PPOOnly, FixedUCBPPO, RandomUCBPPO, AdaptiveNoBC, and AdaptiveSAOS.

The encoding, decoder, initialization, population, operator set, reward, and environmental selection are held constant. `AdaptiveNoBC` is the direct ablation of the supervised handover.

```bash
python experiments/run_revision_aos.py \
  --experiment aos \
  --out-dir results/resubmission/v5/e1_aos \
  --result-file results/resubmission/v5/e1_aos/runs.csv \
  --pop-size 100 --max-gen 100 \
  --seed-start 42 --seed-end 52 --workers 12
```

## Experiment E2: reward ablation

Six reward definitions are compared with AdaptiveSAOS on all 50 instances and ten seeds (3,000 runs): survival only, HV only, makespan only, survival plus HV, the default composite reward, and generation-adaptive weights.

```bash
python experiments/run_revision_aos.py \
  --experiment reward \
  --out-dir results/resubmission/v5/e2_reward \
  --result-file results/resubmission/v5/e2_reward/runs.csv \
  --pop-size 100 --max-gen 100 \
  --seed-start 42 --seed-end 52 --workers 12
```

## Experiment E3: budget and PPO sufficiency

Seven controllers—UniformFixed, UCBOnly, PPOOnly, FixedUCBPPO, RandomUCBPPO, AdaptiveNoBC, and AdaptiveSAOS—are evaluated at 50, 100, and 200 generations on all 50 instances and ten seeds (10,500 runs). This test is designed to answer whether 100 generations provide enough post-transition PPO records and whether the conclusion changes with a longer horizon. Each row must report transition generation, behavior-cloning diagnostics, PPO update count, rollout sizes, policy loss, value loss, and entropy.

```bash
python experiments/run_revision_aos_budget_stress.py \
  --out-dir results/resubmission/v5/e3_budget \
  --result-file results/resubmission/v5/e3_budget/runs.csv \
  --pop-size 100 --budgets 50,100,200 \
  --variants UniformFixed,UCBOnly,PPOOnly,FixedUCBPPO,RandomUCBPPO,AdaptiveNoBC,AdaptiveSAOS \
  --seed-start 42 --seed-end 52 --workers 12
```

## Analysis rules fixed before results

1. Recompute three-objective hypervolume with one common min-max transformation and reference point `(1.1, 1.1, 1.1)` per instance and budget, using the union of final nondominated points from every compared method and seed.
2. For primary inference, first take the median across seeds within each instance-method-budget cell, then treat the 50 benchmark instances as independent matched blocks. Nested instance-seed contrasts are descriptive only and receive no supplementary `p`-values.
3. Report median and interquartile range, mean rank, win/tie/loss, instance-bootstrap 95% confidence intervals, Friedman omnibus tests, and paired Wilcoxon tests with Holm correction. Report rank-biserial effect size for every pairwise contrast.
4. Report an effect size for every pairwise contrast; do not interpret significance without magnitude.
5. Plot median and interquartile learning diagnostics across seeds. A single representative seed may illustrate a trajectory but cannot support a convergence claim.
6. Separate optimization outcomes from mechanism diagnostics. Behavior-cloning accuracy or reduced entropy does not by itself demonstrate better scheduling quality.
7. Report negative and null findings. In particular, do not claim SA-AOS dominance if UCBOnly, UniformFixed, or another coverage-preserving controller is statistically comparable or better.
8. The `1e-12` win/tie/loss threshold is a numerical-equality tolerance only; it is not interpreted as practical equivalence.

## Smoke and completeness gates

Before a full launch:

- run two instances, two seeds, and all E1 variants;
- confirm every output row has the protocol ID;
- confirm every stored Pareto front contains unique objective vectors and that `NSol` equals the stored unique-row count;
- audit all 50 synthetic extensions: realized median travel/processing ratio, AGV energy share, speed-energy ordering, and sampled within-front Cmax-energy correlation must be reported before the full launch;
- confirm `AdaptiveSAOS` has at least 30 behavior-cloning samples, a non-forced transition generation, and on-policy PPO updates at 100 generations;
- confirm UCB-stage records never enter the PPO rollout buffer;
- confirm every expected result key is unique;
- recompute metrics from saved Pareto fronts.

After a full launch, verify exact row counts, zero duplicate keys, zero failed tasks, 50 unique benchmark hashes, ten seeds per instance-method cell, and a complete Pareto-front pickle for every CSV row.
