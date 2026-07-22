# Data-source audit for the SA-AOS resubmission

Audit date: 2026-07-20; public-archive verification updated 2026-07-22

## What exists locally

The repository contains 130 raw `.fjs` benchmark files under `data/benchmarks/`:

| Dataset directory | Files | Local range | Total bytes | Used by the revised core experiments |
|:--|--:|:--|--:|:--|
| `brandimarte` | 10 | `Mk01.fjs`–`Mk10.fjs` | 19,048 | Yes |
| `hurink_edata` | 40 | `la01.fjs`–`la40.fjs` | 48,282 | Yes |
| `hurink_rdata` | 40 | `la01.fjs`–`la40.fjs` | 73,081 | No |
| `hurink_vdata` | 40 | `la01.fjs`–`la40.fjs` | 152,884 | No |

The 50 instances used in the revised paper therefore have local raw source files. They are not reconstructed from tables in the manuscript.

## Provenance currently documented in the code

`data/loader.py` identifies the format as the SchedulingLab zero-indexed FJSP format and names two retrieval locations: `https://github.com/SchedulingLab/fjsp-instances` and `https://github.com/PyJobShop/FJSPLIB`. The v5 manifest records 50 distinct SHA-256 source hashes, source byte counts, parsed sizes, deterministic extension seeds, and every generated field.

For the public-archive verification on 2026-07-22, all 130 local `.fjs` files were compared byte-for-byte with the corresponding files in the SchedulingLab collection at commit `ac4c3402312bfbeafcf4472d78be567d4e6b46ab`; all 130 matched. That upstream repository distributes the collection under the MIT License (Copyright (c) 2022 SchedulingLab). The repository now includes `THIRD_PARTY_NOTICES.md` and `data/benchmarks/LICENSE.SchedulingLab-MIT`. The original historical download timestamp and commit remain unavailable, but the released bytes and current upstream reference commit are fully recorded.

## Which fields are original and which are synthetic

The `.fjs` files supply jobs, operations, eligible machines, and processing times. They do not supply AGV layouts, speed tiers, machine power, setup power, idle power, or setup times. For the corrected rerun, the loader derives a deterministic extension seed from `dataset-directory/filename` plus base seed 42, and appends:

- three AGVs in the current revised AOS runner;
- speed tiers `[0.50, 0.75, 1.00]`;
- loading-station and machine coordinates sampled in a unit square and scaled so that median travel time at speed `0.75` equals `0.30` of the instance median eligible processing time;
- base machine power `phi` sampled from `U(5,10)`, with processing, standby, and setup power fixed to `phi`, `phi/4`, and `phi/2`;
- setup time sampled from `U(0.05,0.15)` times the instance median processing time;
- loaded AGV coefficient `xi` sampled from `U(4,6)`, with `P_load(v)=xi*v^2` and `P_empty(v)=0.6*xi*v^2`;
- an explicit active-window boundary for machine standby accounting.

The earlier statements that pairwise distances were sampled from `U(5,50)` or that layouts used an uncalibrated `[0,100]^2` range must not be reused.

The former objective evaluator also omitted empty-travel AGV energy even though empty travel was included in timing. The corrected protocol stores empty and loaded distance per task and includes both components in total energy. All earlier TEC and HV values are therefore invalid for the resubmission.

## Existing result data

Raw CSV, Pareto-front pickle files, statistical summaries, and figures from the previous revision exist under `results/revision/`. A paused v4 pilot is preserved separately with a verified archive hash. These files establish that engineering runs occurred, but they are not valid evidence for the corrected method because of controller-data, archive-duplication, random-stream, transport-scale, energy-construct, NSGA-III, and statistical-unit defects. The formal run must use `results/resubmission/v5`, protocol `saos_bc_onpolicy_ppo_v5_20260720`, and must never append or merge older rows.

## Archival status

The release archive preserves the v5 manifest and deterministic extension parameters, the exact deployed source tree, environment and hardware records, one formal row per instance-method-seed key, final nondominated objective arrays, and the scripts used to recompute common-reference hypervolume and instance-blocked statistics. Third-party benchmark provenance, the verification commit, and the upstream MIT license are recorded in `THIRD_PARTY_NOTICES.md`. The Git commit identifier is added when the release repository is initialized.
