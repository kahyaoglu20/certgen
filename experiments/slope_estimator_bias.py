"""Bias of the slope dimension estimate on isotropic Gaussians (Section 3.4).

For nominal dimensions 1 to 12, the slope estimate of
certgen.dimension.wasserstein_dimension is computed on six independent
samples of N = 4096 standard normal observations. In dimension below
2p = 4 the two-sample W2 rate approaches the parametric n^(-1/2), so the
estimate saturates near 2.

    python experiments/slope_estimator_bias.py

Writes results/slope_estimator_bias.json (about 3 minutes).
"""
from __future__ import annotations

import time

import numpy as np

import _artifacts as art
from certgen.dimension import wasserstein_dimension

N = 4096
N_REP = 3
SEEDS = 6
NOMINAL = (1, 2, 3, 4, 6, 8, 12)


def main():
    t0 = time.time()
    print(f"isotropic Gaussians, N={N}, n_rep={N_REP}, {SEEDS} seeds per cell")
    print(f"{'nominal d':>10} {'d_hat':>9} {'sd':>7} {'bias':>9}")
    cells = []
    for d in NOMINAL:
        reps = [wasserstein_dimension(
            np.random.default_rng(100 + s).normal(size=(N, d)),
            n_rep=N_REP, seed=s) for s in range(SEEDS)]
        m, sd = float(np.mean(reps)), float(np.std(reps, ddof=1))
        cells.append({"nominal_d": d, "d_hat": m, "sd": sd,
                      "rel_bias": m / d - 1.0, "exponent": 1.0 / m})
        print(f"{d:>10} {m:>9.3f} {sd:>7.3f} {m/d-1:>8.1%}")
    art.save("slope_estimator_bias.json", {
        "_definition": ("mean and standard deviation over seeds of the slope "
                        "dimension estimate on isotropic Gaussians of known "
                        "nominal dimension"),
        "N": N, "n_rep": N_REP, "seeds": SEEDS,
        "cells": cells, "saturated": bool(cells[0]["d_hat"] > 1.6),
        "elapsed_s": round(time.time() - t0, 1)}, indent=2)


if __name__ == "__main__":
    main()
