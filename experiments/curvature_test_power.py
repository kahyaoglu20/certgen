"""Power and size of the curvature test by grid design and floor precision
(Section 4.4).

Floor profiles are simulated at the replicate level and passed through
certgen.fit_floor_law (its `_replicates` argument), so every trial uses the
library's own standard errors, weighted half-slopes and p-value. The
alternative is the two-scale floor n^(-1/2) + 0.35 n^(-1/8); the null is the
single power law n^(-0.2). Two grid families are compared:

  range_extension  4 to 12 points doubling from n = 125 (the range grows);
  densified        6 to 12 points equally spaced in log n over [125, 4000].

The floor precision is set by the median standard error of log f
(1%, 3%, 5%), which spans the precision reached on the benchmark tables.

    python experiments/curvature_test_power.py

Writes results/curvature_test_power.json (about 10 minutes).
"""
from __future__ import annotations

import time

import numpy as np

import _artifacts as art
import certgen as cg

TRIALS = 200
N_REP = 12
# SE(median) = sqrt(pi/2) * replicate_cv / sqrt(n_rep)
MEDIAN_SES = (0.01, 0.03, 0.05)


def make_grid(kind: str, k: int) -> np.ndarray:
    if kind == "range_extension":
        return np.array([125.0 * 2 ** i for i in range(k)])
    lo, hi = 125.0, 4000.0
    ratio = (hi / lo) ** (1.0 / (k - 1))
    return np.array([lo * ratio ** i for i in range(k)])


def trial(rng, grid, f0, cv):
    """One fit of a profile with lognormal replicate noise of scale cv."""
    reps = {}
    for n in grid:
        mu = float(np.interp(n, grid, f0))
        reps[int(n)] = np.exp(np.log(max(mu, 1e-12))
                              + rng.normal(0.0, cv, N_REP))
    return cg.fit_floor_law(_replicates=reps, n_grid=[int(n) for n in grid],
                            metric="w2", n_rep=N_REP, seed=0,
                            drift_boot=200)


def run_cell(kind, k, median_se, rng):
    grid = make_grid(kind, k)
    f_alt = grid ** -0.5 + 0.35 * grid ** -0.125
    f_null = grid ** -0.2
    rep_cv = median_se * np.sqrt(N_REP) / np.sqrt(np.pi / 2.0)
    rej_alt = rej_null = 0
    ok_alt = ok_null = 0
    for t in range(TRIALS):
        law = trial(rng, grid, f_alt, rep_cv)
        if np.isfinite(law.drift_p_value):
            ok_alt += 1
            rej_alt += int(law.drift_p_value < 0.05)
        law = trial(rng, grid, f_null, rep_cv)
        if np.isfinite(law.drift_p_value):
            ok_null += 1
            rej_null += int(law.drift_p_value < 0.05)
    return {"grid_kind": kind, "n_points": k,
            "n_min": float(grid.min()), "n_max": float(grid.max()),
            "median_se": median_se, "replicate_cv": float(rep_cv),
            "power": rej_alt / max(ok_alt, 1),
            "size": rej_null / max(ok_null, 1),
            "trials": TRIALS,
            "finite_power_trials": ok_alt, "finite_size_trials": ok_null}


def main():
    t0 = time.time()
    cells = []
    print(f"alternative n^-1/2 + 0.35 n^-1/8; {TRIALS} trials per cell; "
          f"n_rep={N_REP}")
    for mse in MEDIAN_SES:
        for kind, ks in (("range_extension", (4, 6, 8, 10, 12)),
                         ("densified", (6, 8, 10, 12))):
            for k in ks:
                rng = np.random.default_rng(1000 * int(mse * 1000) + k)
                cell = run_cell(kind, k, mse, rng)
                cells.append(cell)
                print(f"  SE={mse:.2f} {kind:<16} k={k:>2} "
                      f"n=[{cell['n_min']:.0f},{cell['n_max']:.0f}]  "
                      f"power={cell['power']:.3f}  size={cell['size']:.3f}",
                      flush=True)
    art.save("curvature_test_power.json", {
        "_definition": ("power and size at nominal 0.05 of the curvature test "
                        "of certgen.fit_floor_law by grid family, number of "
                        "grid points and median standard error of log f"),
        "_alternative": "n^-1/2 + 0.35 n^-1/8",
        "_null": "n^-0.2",
        "_n_rep": N_REP,
        "_drift_boot": 200,
        "cells": cells,
        "elapsed_s": time.time() - t0}, indent=2)


if __name__ == "__main__":
    main()
