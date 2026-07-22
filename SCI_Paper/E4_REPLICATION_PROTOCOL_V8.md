# Prospective E4 mechanism replication on new instances (E4-R/v8)

**Protocol identifier:** saos_e4_replication_v8_20260722  
**Frozen before v8 outcome generation:** 2026-07-22  
**Formal output:** results/resubmission/v8_e4_replication  
**Status:** design frozen; execution starts only after E5/v7 releases the formal compute allocation.

## 1. Purpose and interpretation

E4-R tests whether the state, behavior-cloning (BC), and rollout observations from
the ten-instance E4 mechanism study reproduce on 30 instances that were not used
in E4. The design is explicitly **outcome-informed replication**: the original
E4 results were known when the contrasts below were selected. It is not described
as an independent discovery experiment or as part of the original E4
preregistration.

The NSGA-III backbone, population 100, decoder, archive, ten-operator library,
composite reward, handover rule, 78-dimensional state construction, actor/critic
widths, BC procedure, and PPO hyperparameters remain unchanged from v6. E4-R
does not tune a successful controller.

## 2. New-instance selection

All ten original E4 instances are excluded:

- Brandimarte: Mk01, Mk03, Mk05, Mk08, Mk10;
- Hurink edata: la01, la10, la20, la30, la40.

All five remaining Brandimarte instances are included. The 35 remaining Hurink
instances are ordered by (total operations, filename), divided into five
consecutive blocks of seven, salt-sorted within each block by SHA-256 of
'saos_e4_replication_v8_20260722 | block | filename', and the first five in every
block are selected. The resulting 30-instance set is frozen as:

| Family | Frozen instances |
|---|---|
| Brandimarte | Mk02, Mk04, Mk06, Mk07, Mk09 |
| Hurink edata | la03, la05, la06, la07, la08, la09, la12, la13, la14, la15, la17, la18, la19, la22, la24, la25, la28, la29, la33, la34, la35, la36, la37, la38, la39 |

This produces disjoint E4 and E4-R instance sets. It does not make E4-R
author-unseen relative to E1--E3, because primary-campaign results on the benchmark
collection had already been inspected.

## 3. Configurations and run count

Evaluation seeds are 52--56. Five repeats are used because the inferential unit
is the instance and the purpose is to expand from 10 to 30 independent scheduling
problems, not to maximize nested repeats.

At 100 generations, E4-R repeats the capacity-matched 2x2 design and retains UCB:

1. BasePaddedBC_R16;
2. BasePaddedNoBC_R16;
3. EnhancedBC_R16;
4. EnhancedNoBC_R16;
5. UCBOnly.

This block contains 30 instances x 5 seeds x 5 configurations = 750 runs.

At 200 generations, E4-R repeats the rollout design:

1. EnhancedBC_R8;
2. EnhancedBC_R16;
3. EnhancedBC_R32;
4. UCBOnly.

This block contains 30 instances x 5 seeds x 4 configurations = 600 runs.
The formal total is exactly 1,350 runs.

The original 24-dimensional implementation controls are not rerun. Their v6
role was implementation verification; the replication concentrates compute on
the capacity-matched and update-frequency questions.

## 4. Outcomes and frozen references

The primary outcome is common-reference three-objective HV recomputed from saved
deduplicated nondominated fronts. Normalization, the (1.1,1.1,1.1) fixed box,
outside-orthant exclusion rule, expanded (1.5,1.5,1.5) sensitivity, and frozen
v5 IGD+ reference sets follow E5/v7. A v8-local immutable reference snapshot is
built from v5 before any v8 outcome exists; v8 fronts never change its bounds or
reference sets.

Every row retains the v6 BC curves, confusion counts, UCB-context diagnostics,
PPO action-effective updates, terminal-only updates, consumed transitions,
optimizer steps, learning time, operator sequence, reward trace, and final front.

## 5. Prespecified statistical families

Seeds are collapsed by the median within instance--configuration--budget cells.
The 30 new instances are matched blocks. Every contrast reports median paired
difference, a 10,000-instance-bootstrap interval, wins/ties/losses,
rank-biserial effect, two-sided signed-rank result, and Holm-adjusted p-value.
A nonsignificant result is not equivalence.

The 100-generation state family (Holm size 2) is:

1. enhanced minus padded with BC;
2. enhanced minus padded without BC.

The 100-generation BC family (Holm size 2) is:

1. BC minus no-BC under the padded state;
2. BC minus no-BC under the enhanced state.

The state-by-BC difference-in-differences is a separate single test.

The 200-generation rollout family (Holm size 2) is:

1. rollout 8 minus rollout 16;
2. rollout 32 minus rollout 16.

The outcome-informed UCB replication family (Holm size 3) is:

1. enhanced rollout 16 minus UCB at 100 generations;
2. enhanced rollout 16 minus UCB at 200 generations;
3. enhanced rollout 8 minus UCB at 200 generations.

The third comparison directly follows the descriptive v6 R8 result and is labeled
as a prospective replication of an observed pattern, not an original confirmatory
hypothesis.

For each shared contrast, E4-R additionally reports whether direction and
familywise decision reproduce v6. A pooled 40-instance estimate may be shown as
a descriptive synthesis only; formal v6 and v8 families remain separate.

## 6. Completion and integrity gates

Analysis is prohibited unless all conditions hold:

- exact reproduction of the frozen 30-instance set and zero overlap with E4;
- exactly 1,350 unique result keys and one saved front per key;
- population 100, budgets 100/200, seeds 52--56, and the exact configuration grid;
- one protocol, code, design, input, configuration, front, and reference hash chain;
- finite, nonempty, deduplicated nondominated fronts;
- equal initial and offspring evaluations within every budget;
- single-policy-version PPO updates and the v6 on-policy buffer boundary;
- deterministic smoke and interrupted-resume checks before formal launch;
- no failure marker when the completion marker is written.

No v5, v6, or v7 row is copied into the v8 result file. Earlier results may be
used only in the separately labeled replication comparison after v8 completes.
