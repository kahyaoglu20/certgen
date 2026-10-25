"""Size of the curvature test against exact single power laws (Section 4.3).

The rejection rate of the curvature test of certgen.fit_floor_law is
measured on isotropic Gaussian laws of nominal dimension 2 to 8 (N = 1024,
n_rep = 12, six grid points), which have a single exponent by construction,
together with the mean signed log drift log(d_low / d_high) of the null
profiles. Below d_eff = 4 the decay carries a logarithmic correction, which
appears as a systematically positive drift.

    python experiments/curvature_test_size.py [--quick]

Writes results/curvature_test_size.json (about 8 minutes).
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from scipy import stats

import _artifacts as art
import certgen as cg

warnings.filterwarnings("ignore")

N, N_REP = 1024, 12
DIMS = (2, 3, 4, 6, 8)
QUICK = "--quick" in sys.argv
TRIALS = 40 if QUICK else 200
TRIALS_BY_D = {} if QUICK else {6: 175, 8: 175}
for _a in sys.argv[1:]:
    if _a.startswith("--dims="):
        DIMS = tuple(int(x) for x in _a.split("=")[1].split(","))
    if _a.startswith("--trials="):
        TRIALS = int(_a.split("=")[1])
        TRIALS_BY_D = {}


def _law(d, seed):
    return cg.Standardizer().fit_transform(
        np.random.default_rng(seed).normal(size=(N, d)))


def cell(d, trials, seed0):
    rej, ps, de, lo, hi = 0, [], [], [], []
    for t in range(trials):
        L = cg.fit_floor_law(_law(d, seed0 + t), n_rep=N_REP, seed=t)
        p = L.drift_p_value
        if not np.isfinite(p):
            continue
        ps.append(p)
        de.append(L.d_eff)
        rej += int(p < 0.05)
        if np.isfinite(L.d_eff_low) and np.isfinite(L.d_eff_high):
            lo.append(L.d_eff_low)
            hi.append(L.d_eff_high)
    n = len(ps)
    s = rej / max(n, 1)
    r = np.log(np.asarray(lo) / np.asarray(hi)) if lo else np.array([np.nan])
    tt = stats.ttest_1samp(r, 0.0) if len(r) > 2 else None
    return {"nominal_d": d, "d_eff_median": float(np.median(de)),
            "trials": n, "size": float(s),
            "se": float(np.sqrt(s * (1 - s) / max(n, 1))),
            "median_p": float(np.median(ps)),
            "mean_log_drift": float(np.mean(r)),
            "drift_t": float(tt.statistic) if tt else float("nan"),
            "drift_p": float(tt.pvalue) if tt else float("nan")}


def main():
    t0 = time.time()
    print(f"isotropic Gaussians, N={N}, n_rep={N_REP}, trials per cell "
          f"{ {d: TRIALS_BY_D.get(d, TRIALS) for d in DIMS} }")
    print(f"{'nominal d':>10} {'d_eff':>7} {'size':>7} {'+/-':>6} "
          f"{'mean log drift':>15} {'t':>7}")
    cells = []
    for d in DIMS:
        c = cell(d, TRIALS_BY_D.get(d, TRIALS), 60_000 + 1000 * d)
        cells.append(c)
        print(f"{d:>10} {c['d_eff_median']:>7.2f} {c['size']:>7.3f} "
              f"{c['se']:>6.3f} {c['mean_log_drift']:>15.4f} "
              f"{c['drift_t']:>7.2f}", flush=True)
    sizes = [c["size"] for c in cells]
    art.save("curvature_test_size.json", {
        "_definition": ("rejection rate at nominal 0.05 of the curvature test "
                        "of certgen.fit_floor_law on laws with a single "
                        "exponent, and the mean signed log drift of their "
                        "profiles"),
        "N": N, "n_rep": N_REP,
        "trials_per_cell": {str(d): TRIALS_BY_D.get(d, TRIALS) for d in DIMS},
        "cells": cells,
        "size_min": min(sizes), "size_max": max(sizes),
        "elapsed_s": round(time.time() - t0, 1)}, indent=2)


if __name__ == "__main__":
    main()
