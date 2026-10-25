# Experiments

One script per measurement. Every script writes one artifact to `results/`,
and every number in the manuscript is computed from these artifacts by
`tools/paper_numbers.py`. Run a script from this directory or from the
project root, for example `python experiments/stress_tests.py`.

Runtimes are wall times on one core of an Intel Core i7-14700HX. Scripts marked
*network* download the benchmark tables from OpenML on first use
(scikit-learn caches them); `plan_pullback_vae.py` needs `torch`.

## The floor and its estimation (Section 3)

| script | artifact | measures | runtime |
|---|---|---|---|
| `copula_examples.py` | `copula_examples.json` | floor law and resolvable changes of the running copula example; unbiased statistics and sliced W₂; skill of fresh draws; spread under a mean shift | 7 min |
| `slope_estimator_bias.py` | `slope_estimator_bias.json` | bias of the slope dimension estimate on isotropic Gaussians | 3 min |
| `ot_cost.py` | `ot_cost.json` | wall time of one exact two-sample W₂ | 1 min |

## Designing an evaluation (Section 4)

| script | artifact | measures | runtime |
|---|---|---|---|
| `stress_tests.py` | `stress_tests.json` | memorisation, goodness of fit against curvature, tied observations, floor proxy of the paired standard deviation | 1.5 min |
| `curvature_test_size.py` | `curvature_test_size.json` | size of the curvature test against single power laws | 8 min |
| `curvature_test_power.py` | `curvature_test_power.json` | power of the curvature test by grid design and floor precision | 10 min |
| `interval_coverage.py` | `interval_coverage.json` | coverage and level calibration of the sample-size interval; the interval on the benchmark cells | 1.5 min |
| `inference_checks.py` | `inference_checks.json` | shared random numbers across the grid; power of the two-standard-deviation rule; coverage of the exponent interval | 2 min |

## Reporting an evaluation (Section 5)

| script | artifact | measures | runtime |
|---|---|---|---|
| `reporting_endpoints.py` | `reporting_endpoints.json` | skill endpoints and the T_self ceiling | 1 min |
| `leakage_screen.py` | `leakage_screen.json` | detection and false alarms of the leakage screen | seconds |
| `equivalence_test.py` | `equivalence_test.json` | operating characteristics of the equivalence test | 20 s |

## Latent-generative certificates (Sections 6 and 7)

| script | artifact | measures | runtime |
|---|---|---|---|
| `reference_tables.py` | `reference_tables.json` | floor ratio r(d, N) and slope-estimate calibration on isotropic Gaussians, N ≤ 8 192, and the crossings r = 1 | 1.4 h |
| `reference_tables_large_n.py` | `reference_tables_large_n.json`, `reference_tables_large_n_solves.json` | the same tables at N = 16 384 and 32 768 under a memory budget | 46 CPU h |
| `two_sample_vs_one_sample.py` | `two_sample_vs_one_sample.json` | two-sample against one-sample distance | 4 min |
| `transfer_smooth_laws.py` | `transfer_smooth_laws.json` | transfer of the tables to anisotropic, factor and manifold laws | 10 s |
| `transfer_probe.py` | `transfer_probe.json` | transfer to heavy-tailed, lattice and mixed laws | 10 s |
| `tail_index.py` | `tail_index.json` | Hill tail index of the benchmark cells and reference laws (network) | 10 s |
| `plan_pullback_vae.py` | `plan_pullback_vae.json` | tightness of the plan-pullback bound on trained VAE decoders | 2.5 min |

## Empirical study (Section 8)

| script | artifact | measures | runtime |
|---|---|---|---|
| `benchmark_sources.py` | `benchmark_sources.json` | OpenML identifiers, versions and checksums of the benchmark tables; digests of the encoded matrices (network) | 1 min |
| `benchmark_floor_laws.py` | `benchmark_floor_laws.json` | floor laws and dimension estimates of the eleven benchmark cells (network) | 25 min |
| `benchmark_diagnostics.py` | `benchmark_diagnostics.json` | excluded candidate tables; correlations of adult (network) | 4 min |
| `half_fit_ranges.py` | `half_fit_ranges.json` | agreement of the half-grid prescriptions | seconds |
| `severity.py` | `severity.json` | severity thresholds with their combined interval | 15 s |
| `resolution_atlas.py` | `resolution_atlas.json` | the resolution atlas (network) | 1.8 h |
| `certification_margins.py` | `certification_margins.json` | certification margins with joint uncertainty | 25 s |

`half_fit_ranges.py`, `severity.py` and `interval_coverage.py` read the stored
sweeps of `benchmark_floor_laws.json`; `certification_margins.py` reads the
two transfer studies. `_benchmarks.py` loads and encodes the benchmark tables and rebuilds a stored floor law
and `_artifacts.py` reads and writes artifacts.

Neural-network training is seeded but not bit-identical across
floating-point platforms; `python experiments/plan_pullback_vae.py --out
NAME.json` writes a rerun beside the reference artifact for comparison.
