# Execution environment

The formal campaign ran on Linux/x86-64 CPUs with one numerical-library
thread per worker. The primary v5 environment manifest records Python 3.12.3,
PyTorch 2.11.0+cu128, and `cuda_available=false`; see
[`results/resubmission/v5/manifests/environment_manifest.json`](results/resubmission/v5/manifests/environment_manifest.json).
The v7 and v8 run manifests record the protocol, code-file hashes, input hash,
design hash, and reference-snapshot hash used by those blocks.

`requirements.txt` is the supported installation specification, not a claim
that one post-campaign package snapshot applies retroactively to every
experiment block. The server environment inspected after campaign completion
on 2026-07-23 contained:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| Linux | 5.15, x86-64, glibc 2.39 |
| NumPy | 2.4.3 |
| pandas | 3.0.3 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| pymoo | 0.6.2 |
| PyTorch | 2.11.0+cu128 (CPU execution) |
| PyYAML | 6.0.3 |
| statsmodels | 0.14.6 (read-only mixed-model audit environment) |
| pytest | 9.1.1 |

The remote experiment environment did not contain `statsmodels`; version
0.14.6 was used in the separate read-only environment that produced the
reported mixed-model sensitivity audit. For the closest tested direct-stack
compatibility, install with:

```bash
pip install -r requirements.txt -c constraints-postcampaign-20260723.txt
```

The constraint file intentionally identifies itself as a post-campaign direct
stack, not a full transitive or pre-run lockfile. The immutable run manifests
and their per-file hashes are the authoritative provenance records for the
executed code. Exact byte-for-byte regeneration of floating-point fronts can
still depend on operating-system, BLAS, and library details. Scientific
reproduction should therefore verify the declared design, row counts,
pairing, direction and magnitude of effects, and analysis outputs rather than
requiring identical pickle bytes.

The E4/v6 run used the earlier PPO-agent implementation preserved at tag
[`v6.0.1`](https://github.com/nothingbody/controlled-aos-fjsp-agv/tree/v6.0.1).
The current reproducibility release preserves the v7/v8 implementation and
adds the E5 raw release assets without rewriting any frozen result.
