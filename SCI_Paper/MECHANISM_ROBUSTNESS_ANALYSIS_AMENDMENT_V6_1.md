# Analysis amendment for the v6 mechanism-robustness experiment

**Amendment identifier:** `saos_v6_analysis_amendment_1_20260722`  
**Recorded after:** the fixed-reference audit and before inspecting any controller-level performance contrast  
**Reason:** the prespecified audit found 4 of 1,100 run fronts containing points outside the fixed hypervolume reference orthant. Across these runs, 12 objective coordinates exceeded 1.1; the largest normalized coordinate was 1.455882. The original analysis stopped as required and produced no controller comparison.

## Amended primary hypervolume

The v5 normalization and reference point `(1.1, 1.1, 1.1)` remain unchanged. Before hypervolume calculation, points with any coordinate greater than the reference point are excluded from that run's approximation set. Coordinates are not clipped or winsorized. This is the bounded hypervolume of the approximation set inside the fixed reference orthant: an excluded point has an empty dominated hyperrectangle relative to the reference and therefore contributes no volume. The number of excluded points and affected runs is reported.

The prespecified instance aggregation, paired contrasts, exact signed-rank sign enumeration, Holm families, bootstrap summaries, and HV role as the only primary inferential endpoint remain unchanged. The original all-points-dominated gate is reported as failed; the amended analysis is not presented as if the gate had passed.

## Sensitivity analyses

1. Hypervolume is recomputed without point exclusion using a post-audit expanded reference point `(1.5, 1.5, 1.5)`, which dominates every observed normalized v6 point. This result is explicitly post hoc and supportive.
2. Frozen-reference IGD+ remains the prespecified supportive sensitivity indicator and is not used independently to declare a positive result.

Controller-level claims are accepted only if their direction and substantive interpretation are stable across the amended fixed-box HV, expanded-reference HV, and IGD+ sensitivity results. Any disagreement is reported rather than resolved by selecting a favorable indicator.

## Integrity

The executed code, design, benchmark-input, result-front, combined-front-manifest, and v5 reference-snapshot hashes remain unchanged. The amendment and amended analysis script receive separate SHA-256 hashes in the analysis manifest. No optimization run is repeated, removed, or replaced.
