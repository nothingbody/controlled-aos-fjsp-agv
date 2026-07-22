# Prespecified cross-instance PPO pretraining experiment (E5/v7)

**Protocol identifier:** `saos_cross_instance_pretrained_ppo_v7_20260722`  
**Frozen before v7 result inspection:** 2026-07-22  
**Formal output:** `results/resubmission/v7_cross_instance`  
**Purpose:** test whether experience accumulated on other scheduling instances changes the finite-budget boundary observed for within-run, from-scratch PPO.

## 1. Scientific scope

E5 changes one factor only: the source of the PPO parameters available at deployment. The FJSP--AGV extension, encoding, decoder, initialization mixture, population size, ten-operator library, composite reward, archive, NSGA-III environmental selection, and adaptive UCB-to-PPO handover remain frozen. The enhanced 78-dimensional state and rollout length 16 are inherited from the prespecified E4 reference configuration; neither was selected from E4 outcomes.

The primary question is whether cross-instance PPO parameter pretraining improves held-out, test-budget performance relative to (i) coverage-preserving UCB and (ii) the same PPO architecture trained from scratch within a run. This is not a total-compute comparison: offline pretraining is an additional cost and is reported separately.

## 2. Five-fold instance split

The 10 Brandimarte and 40 Hurink-edata instances are split without using HV, ranks, controller diagnostics, or any E1--E4 outcome. Within each benchmark family, instances are ordered by `(total_operations, filename)`, divided into consecutive blocks of five, salt-sorted within each block by SHA-256 of `protocol | family | block | filename`, and assigned once to folds 1--5. Each fold contains two Brandimarte and eight Hurink instances for testing; its remaining 40 instances are the corresponding training set.

| Fold | Brandimarte test instances | Hurink-edata test instances |
|---|---|---|
| 1 | Mk04, Mk10 | la05, la10, la15, la16, la24, la26, la34, la37 |
| 2 | Mk02, Mk03 | la03, la07, la13, la19, la23, la27, la31, la36 |
| 3 | Mk06, Mk07 | la04, la09, la12, la17, la25, la28, la35, la39 |
| 4 | Mk01, Mk08 | la01, la06, la11, la18, la22, la29, la32, la40 |
| 5 | Mk05, Mk09 | la02, la08, la14, la20, la21, la30, la33, la38 |

The split implementation must reproduce this table and verify disjointness both by canonical path identity and by whitespace-token benchmark-content SHA-256. A result for instance `x` is admissible only when `x` and its content hash are absent from the referenced checkpoint's training set. These are held out from the corresponding v7 checkpoint, but they are not described as author-unseen benchmarks because earlier experiment outcomes on the benchmark collection have already been inspected.

## 3. Pretraining protocol

Five independent checkpoint replicas are trained per fold. Replica seeds are fixed as 7042--7046. Each replica completes two passes over the 40 training instances, giving 80 sequential episodes and 2,000 pretraining episodes in total:

`5 folds x 5 replicas x 2 passes x 40 training instances = 2,000 episodes`.

Each pass uses an independently and deterministically shuffled instance order derived by `numpy.random.SeedSequence` from protocol, fold, replica, and pass identifiers. The terminal checkpoint after pass 2 is the only checkpoint used for evaluation; pass-1 artifacts are retained for audit and cannot be selected after observing test performance.

Each pretraining episode uses population 100, 200 generations, enhanced state dimension 78, rollout 16, learning rate `3e-4`, discount 0.99, GAE 0.95, clipping 0.2, four PPO epochs, entropy coefficient 0.01, and value coefficient 0.5. Behavior cloning is disabled. Actor, critic, and Adam optimizer state persist across training episodes. Population, archive, UCB history, rollout buffer, action RNG, and instance state reset at every episode boundary. `finalize()` is called before transfer to the next episode, and a checkpoint may be written only when the rollout buffer is empty.

Different fold--replica chains may run in parallel. Episodes within one chain are sequential. Asynchronous rollouts from multiple workers are never aggregated into one changing policy.

## 4. Checkpoint contract

The full audit checkpoint stores actor, critic, optimizer, architecture, hyperparameters, fold, replica, both ordered pass lists, canonical training-instance hashes, cumulative objective evaluations, PPO transitions, update counts, optimizer steps, RNG derivation metadata, protocol hash, code hash, design hash, and input hash. It is written atomically and accompanied by a completion marker.

Two integrity digests are required:

1. a file SHA-256 for tamper detection;
2. a semantic weight SHA-256 computed from sorted tensor name, dtype, shape, and contiguous CPU bytes.

Formal evaluation loads actor and critic parameters from the terminal checkpoint but creates a fresh Adam optimizer and empty rollout buffer. Thus the evaluated treatment is parameter transfer rather than continuation of training-set optimizer moments. Test action, PPO-permutation, initialization, and evolution streams are newly derived from the evaluation seed. Loading a checkpoint must not advance any evaluation RNG.

## 5. Held-out test controllers and budgets

Four controllers are evaluated:

1. `UCBOnly`: the frozen coverage-preserving low-data baseline;
2. `ScratchNoBC_R16`: enhanced-state PPO with no BC and no cross-instance parameter training;
3. `XPrePPO_Frozen`: the fold-specific pretrained actor is sampled after the same UCB handover, with no test-instance gradient update;
4. `XPrePPO_Online_R16`: the fold-specific pretrained actor and critic are loaded, a fresh optimizer is created, and strictly on-policy test-instance updates resume after handover.

For `ScratchNoBC_R16`, the untrained network uses the same replica-specific step-0 network seed from which the matched pretrained checkpoint originated. Evaluation seeds 42--51 are fixed; checkpoint replica is `(evaluation_seed - 42) mod 5`, so each checkpoint is used by exactly two evaluation seeds. Initialization, evolution, and controller-action random streams are paired across applicable controllers for each instance and evaluation seed.

Test budgets are 50, 100, and 200 generations with population 100 and exactly 100 offspring per generation. The formal held-out grid contains 6,000 runs:

`50 instances x 10 seeds x 4 controllers x 3 budgets = 6,000 runs`.

The complete E5 execution therefore contains 2,000 pretraining episodes and 6,000 held-out test runs. Pretraining rows and held-out rows are stored separately and are never pooled as inferential replicates.

## 6. On-policy and random-stream gates

Every stored PPO transition carries a policy-version identifier. A PPO update may consume transitions from exactly one policy version, after which the version increments and the buffer is cleared. No pretraining transition is loaded into a held-out run. Each test worker loads an independent immutable checkpoint copy. The checkpoint records the RNG derivation schema, the initial network/action/BC/PPO seeds, and the four derived seeds for every training episode.

Network initialization, stochastic actions, PPO permutations, benchmark initialization, evolutionary variation, fold assignment, and training-order shuffling use separately derived streams. Python's process-randomized `hash()` is prohibited. Worker processes must run with one Torch/BLAS thread; the formal worker cap is 40.

## 7. Outcomes and frozen references

The primary outcome is three-objective common-reference HV recomputed from saved final nondominated fronts. Ideal, nadir, and reference values are imported unchanged from the corresponding frozen v5 budget snapshot. New v7 fronts do not change normalization or the reference point. For the fixed `(1.1,1.1,1.1)` box, a point outside the reference orthant has an empty dominated hyperrectangle and is excluded from that HV calculation; coordinates are never clipped or winsorized, and excluded-point counts are reported by objective and run. A run is invalid for confirmatory HV if no point remains inside the fixed box. Expanded-reference `(1.5,1.5,1.5)` HV is a prespecified sensitivity analysis only and is marked invalid if that reference fails to dominate a new v7 point. IGD+ against the corresponding frozen v5 nondominated reference set is supportive only.

Each row also stores makespan, energy index, workload balance, front size, transition generation, action entropy, online PPO samples and updates, policy drift, learning time, and total CPU time.

Data-economy accounting must distinguish:

- initial population evaluations;
- test-run offspring evaluations;
- offline pretraining episodes, generations, objective evaluations, PPO transitions, updates, optimizer steps, CPU time, and checkpoint size;
- online test PPO-controlled actions, collected transitions, update-consumed samples, discarded terminal singletons, updates, optimizer steps, and learning time;
- amortized offline cost per deployment and break-even deployment count, when defined.

Pretraining cost is never counted as zero and cannot support a claim of total computational superiority without an explicit matched-total-compute analysis.

## 8. Confirmatory statistics

Within every instance, controller, and budget, the ten evaluation seeds are collapsed by their median. The 50 out-of-fold instances are the matched performance blocks for the prespecified Wilcoxon summaries. Because ten instances in a fold share the same five pretrained checkpoints, ordinary instance bootstrap intervals are not treated as sufficient evidence of independence.

The 100-generation primary family contains two two-sided paired tests with Holm correction:

1. `XPrePPO_Online_R16 - UCBOnly`;
2. `XPrePPO_Online_R16 - ScratchNoBC_R16`.

The secondary budget family repeats those two contrasts at 50 and 200 generations (four Holm-adjusted tests). The mechanism family compares `XPrePPO_Online_R16 - XPrePPO_Frozen` at 50, 100, and 200 generations (three Holm-adjusted tests).

For every contrast report median paired difference, 10,000-replicate instance bootstrap 95% interval, wins/ties/losses, paired rank-biserial effect, two-sided paired Wilcoxon statistic, raw p-value, and family-adjusted p-value. Exact or fixed-seed sign-flip computation is used when ties/zeros make an asymptotic result inappropriate. In addition, a mandatory hierarchical sensitivity resamples the five folds and then instances within sampled folds; fold-by-replica summaries and leave-one-fold-out effect estimates are reported. With only five checkpoint clusters, the clustered analysis is acknowledged as low-resolution. A superiority claim requires a positive prespecified effect, Holm significance in the instance-blocked analysis, and no direction reversal in the hierarchical interval or any leave-one-fold-out estimate. Benchmark-family summaries remain exploratory. A nonsignificant result is not interpreted as equivalence.

## 9. Integrity and completion gates

Before any outcome contrast is inspected, require:

- exact split reproduction, exhaustive membership, and zero train--test overlap by identity and content hash;
- exactly 25 complete terminal checkpoints and 2,000 unique pretraining episode keys;
- exactly 6,000 unique held-out run keys and one corresponding front per run;
- exact controller, budget, seed, fold, checkpoint, and instance grids;
- finite, nonempty, deduplicated, nondominated three-objective fronts;
- one protocol, code, design, split, input, and checkpoint hash chain;
- empty rollout buffers at checkpoint and test-run boundaries;
- single-policy-version PPO updates only;
- equal online offspring evaluations across controllers at each test budget;
- explicit pretraining and online data-economy fields;
- deterministic clean-run versus interrupted-resume equivalence on scientific outputs; each completed evaluation is first written as an atomic per-key journal and only then compacted atomically into `runs.csv`;
- no failure marker when a completion marker is written.

Any failed task is recorded. Formal result directories cannot be reused for smoke tests, and no v5/v6 result is modified or mixed into v7 rows.

## 10. Interpretation boundary

If cross-instance online PPO is superior, the supported claim is that transferable parameter initialization can improve this frozen selector under the tested within-run budgets; it does not establish a general PPO advantage or total-compute superiority. If it is not superior, the supported claim is limited to this frozen two-pass, enhanced-state, rollout-16 cross-instance PPO protocol. It does not rule out offline RL, meta-learning, alternative state representations, different benchmark families, or larger pretraining corpora.
