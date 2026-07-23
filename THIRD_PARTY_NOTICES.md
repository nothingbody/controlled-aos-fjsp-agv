# Third-Party Data Notices

## Flexible job-shop benchmark instances

The files under `data/benchmarks/` originate from the SchedulingLab
`fjsp-instances` collection:

- Source: https://github.com/SchedulingLab/fjsp-instances
- Reference commit used for release verification:
  `ac4c3402312bfbeafcf4472d78be567d4e6b46ab`
- Upstream license: MIT License, Copyright (c) 2022 SchedulingLab
- Bundled license copy: `data/benchmarks/LICENSE.SchedulingLab-MIT`

Before this release, all 130 local `.fjs` files were compared byte-for-byte
with the corresponding upstream `.txt` files at the reference commit (with
only the local extension and directory naming changed). All 130 files matched.

The benchmark instances were originally introduced in the papers cited by the
upstream collection, including Brandimarte (1993) and Hurink, Jurisch, and
Thole (1994). Please cite the original benchmark papers as well as the
upstream collection when reusing the data.

Only these 50 files are used by the frozen v5 protocol:

- `data/benchmarks/brandimarte/Mk01.fjs` through `Mk10.fjs`
- `data/benchmarks/hurink_edata/la01.fjs` through `la40.fjs`

The Hurink rdata and vdata files are retained for provenance and future
comparisons but are not used in the reported 18,500-run experiment.

## Project licensing boundary

The upstream MIT license and SchedulingLab copyright apply to the third-party
benchmark files. The repository's root MIT License separately covers the
original project code and documentation owned by Qingyou Wang; it does not
replace or remove the upstream notices for the benchmark files. Manuscript and
journal-submission files are not distributed in this repository.
