# v4 pilot pause and preservation audit

## Status

- Remote project: `/root/saos_resubmission_v4`
- Pause time: 2026-07-20 11:44 CST
- Verified stopped processes: E1 parent PID 4788 and its 12 worker processes
- Remaining matching v4 experiment processes after stop: 0
- Intended use: engineering pilot and audit history only; excluded from manuscript evidence

## Preserved output

- Complete E1 CSV rows: 1,020 (1,021 lines including the header)
- E1 front pickle files: 1,031
- Fronts written before their CSV row during termination: 11
- Files below `results/resubmission/v4`: 1,086
- Uncompressed result bytes: 5,584,595
- `Traceback|Exception|Error` matches in E1 and pipeline logs: 0

The 11 extra front files are an expected consequence of stopping worker processes after a front was atomically written but before the parent appended the corresponding CSV row. They must not be interpreted as complete runs or merged into v5.

## Archive integrity

- Remote archive: `/root/saos_v4_pilot_paused_20260720_1144CST.tar.gz`
- Local archive: `results/resubmission/deploy/saos_v4_pilot_paused_20260720_1144CST.tar.gz`
- Archive bytes: 2,487,062
- SHA-256 on both hosts: `61abcf056e93adcb25eeb5efb474ccdd69c39c3d42a71a3c3b5bf7a9b2248885`

## Why v4 is invalid for inference

1. Final archives contained duplicate objective vectors, so archive capacity, `NSol`, and spread were distorted.
2. Fixed `[0,100]^2` layouts created strongly inconsistent transport intensity across benchmark families.
3. Full-makespan machine idle accounting made total energy nearly redundant with makespan.
4. The hybrid initialization proportions did not match the declared 30/20/20/30 design.
5. Controller construction could consume the shared random stream before population initialization, weakening common-random-number fairness.
6. NSGA-III last-front niching did not follow the declared Deb-Jain selection rule.
7. The primary statistical analysis treated instance-seed pairs as independent blocks instead of collapsing seeds within each benchmark instance.
8. Early runtime rows were contaminated by oversubscription from an earlier 80-worker launch.

All corrected formal results must carry protocol `saos_bc_onpolicy_ppo_v5_20260720` and be written under `results/resubmission/v5`.
