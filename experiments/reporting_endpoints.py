"""Skill endpoints and the T_self ceiling (Section 5.1).

  skill_T1           skill relative to T1 from certgen.evaluate for four
                     generators on an eight-dimensional Gaussian copula
                     (rho = 0.8, N = 1000 real rows, n = 500 per side): a fresh
                     draw from the law, and samples of T2, T1 and T0; median
                     and range over 20 seeds;
  t_self_over_floor  distance between a real reference of 500 rows and 500
                     rows resampled from the real rows outside the reference,
                     or from the whole sample, divided by the floor at n = 500.

    python experiments/reporting_endpoints.py

Writes results/reporting_endpoints.json (about 1 minute).
"""
from __future__ import annotations

import json
import time

import numpy as np

import _artifacts as art
import certgen as cg
from certgen.baselines import Standardizer
from certgen.discrepancy import w2_exact

D, RHO, N_REAL, N_SIDE, R, SEED = 8, 0.8, 1000, 500, 20, 20260930


def generators(X, rng):
    mu, S = X.mean(0), np.cov(X, rowvar=False)
    t2 = rng.multivariate_normal(mu, S, size=N_SIDE)
    t1 = np.column_stack([rng.choice(X[:, j], size=N_SIDE, replace=True)
                          for j in range(X.shape[1])])
    t0 = np.repeat(mu[None, :], N_SIDE, axis=0)
    return {"T2 sample": t2, "T1 sample": t1, "T0 sample": t0}


def main():
    t_start = time.time()
    skills = {k: [] for k in ("fresh draw", "T2 sample", "T1 sample", "T0 sample")}
    self_disjoint, self_whole = [], []
    for r in range(R):
        rng = np.random.default_rng(SEED + r)
        X = cg.data.gaussian_copula(N_REAL, D, rho=RHO, seed=SEED + r)
        fresh = cg.data.gaussian_copula(N_SIDE, D, rho=RHO, seed=10_000 + SEED + r)
        gens = {"fresh draw": fresh, **generators(X, rng)}
        for name, G in gens.items():
            rep = cg.evaluate(X, G, n_per_side=N_SIDE, n_rep=20, seed=r)
            skills[name].append(float(rep.skill["T1_marginals"]))
        # T_self relative to the floor, on standardised data
        Z = Standardizer().fit_transform(X)
        idx = rng.permutation(N_REAL)
        ref, rest = Z[idx[:N_SIDE]], Z[idx[N_SIDE:]]
        floor = cg.resolution_floor(Z, N_SIDE, n_rep=20, seed=r).median
        a = rest[rng.integers(0, len(rest), N_SIDE)]
        b = Z[rng.integers(0, N_REAL, N_SIDE)]
        self_disjoint.append(w2_exact(ref, a) / floor)
        self_whole.append(w2_exact(ref, b) / floor)
    out = {
        "_definition": ("(a) skill relative to T1 from certgen.evaluate on "
                        "gaussian_copula(d=8, rho=0.8), N=1000 real rows, "
                        "n=500 per side, median and range over R seeds; (b) "
                        "T_self distance to a real reference of n=500 rows, "
                        "resampled from the disjoint remainder or from the "
                        "whole sample, divided by the floor at n=500"),
        "config": {"d": D, "rho": RHO, "N_real": N_REAL, "n_per_side": N_SIDE,
                   "R": R, "seed": SEED},
        "skill_T1": {k: {"median": float(np.median(v)), "min": float(np.min(v)),
                         "max": float(np.max(v))} for k, v in skills.items()},
        "t_self_over_floor": {
            "disjoint_remainder": {"median": float(np.median(self_disjoint)),
                                   "min": float(np.min(self_disjoint)),
                                   "max": float(np.max(self_disjoint))},
            "whole_sample": {"median": float(np.median(self_whole)),
                             "min": float(np.min(self_whole)),
                             "max": float(np.max(self_whole))}},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    art.save("reporting_endpoints.json", out)
    print(json.dumps({k: out[k] for k in ("skill_T1", "t_self_over_floor")},
                     indent=1))


if __name__ == "__main__":
    main()
