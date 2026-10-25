# Changelog

## 3.3.1 — 2026-10-13

First public release, accompanying the manuscript *How Many Samples Does a
Generative Evaluation Need? Intrinsic Dimension and the Design of
Distributional Comparisons*.

### Library

- The resolution floor of the exact two-sample W₂ distance and its power-law
  fit, with the curvature test for a single exponent, the severity of a
  non-rejection, and the sample-size prescription as an interval with a
  calibrated level.
- Reporting: the floor-corrected skill score against reference generators,
  two-sided floor tests, the null-calibrated leakage screen, the minimum
  detectable difference at a stated power, and equivalence testing.
- Certification: the reference tables r(d, N) for N up to 32 768, the
  calibration of the slope estimate of intrinsic dimension, the
  certification limit in measured units, and the plan-pullback bound.
- A command-line interface (`certgen design | resolve | evaluate | limit`).

### Evidence

- One experiment script per measurement, each writing one artifact to
  `results/`; every number in the manuscript is computed from these
  artifacts by `tools/paper_numbers.py`.
- `tools/reproduce.py` re-runs the experiments in a copy of the project and
  compares every regenerated artifact with the stored one.
- The provenance of the benchmark data: OpenML identifiers, versions and
  checksums of the tables, and digests of every analysed matrix.
- Lean 4 proofs of the formal statements, with their axioms pinned at
  compile time, and a Mathlib bridge that instantiates them over the reals.
- Every reference verified against two bibliographic indexes
  (`docs/bibliography.md`).
- A test suite covering the library, the manuscript's numbers, references,
  figures and PDF, the Lean build records, and the repository itself.
