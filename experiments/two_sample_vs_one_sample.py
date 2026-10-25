"""Two-sample against one-sample distance on isotropic Gaussians
(Section 6.2, Appendix B.3).

For d in {2, 4, 8, 16, 32} and N in {256, 1024}, the two-sample distance
W2(X_N, X'_N) (exact assignment) is compared with the distance between X_N
and a reference sample of M = 2000 observations (exact Kantorovich linear
program, HiGHS), which stands in for the population. The ratio of their
means relates the floor ratio to the statistical term of a certificate.

    python experiments/two_sample_vs_one_sample.py

Writes results/two_sample_vs_one_sample.json (about 4 minutes).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

import _artifacts as art
from certgen.discrepancy import w2_exact


def w2_empirical(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> float:
    """Exact W2 between two uniform empirical measures (possibly unequal n)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    if n == m:
        return w2_exact(X, Y, max_n=n, seed=seed)
    C = cdist(X, Y, "sqeuclidean")
    c = C.ravel()
    j_idx = np.tile(np.arange(m), n)
    i_idx = np.repeat(np.arange(n), m)
    k = i_idx * m + j_idx
    rows = np.concatenate([i_idx, n + j_idx])
    cols = np.concatenate([k, k])
    data = np.ones(2 * n * m)
    A = coo_matrix((data, (rows, cols)), shape=(n + m, n * m)).tocsr()
    b = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"OT LP failed: {res.message}")
    return float(np.sqrt(max(res.fun, 0.0)))


def main():
    # at N = 1024 the linear program is expensive, so the d-grid is thinned
    d_grid = [2, 4, 8, 16, 32]
    plan = [
        {"N": 256, "d_list": d_grid, "n_rep": 3},
        {"N": 1024, "d_list": [2, 8, 32], "n_rep": 1},
    ]
    rng = np.random.default_rng(0)
    cells = []
    t0 = time.time()
    for block in plan:
        N = block["N"]
        n_rep = block["n_rep"]
        M_spec = min(4000, max(4 * N, 2000))
        M = min(M_spec, 2000)
        for d in block["d_list"]:
            two, one = [], []
            for r in range(n_rep):
                Xn = rng.normal(size=(N, d))
                Xn2 = rng.normal(size=(N, d))
                Xm = rng.normal(size=(M, d))
                two.append(w2_exact(Xn, Xn2, max_n=N, seed=r))
                one.append(w2_empirical(Xn, Xm, seed=r))
            two, one = np.asarray(two), np.asarray(one)
            ratio = two / np.maximum(one, 1e-15)
            rec = {
                "d": d,
                "N": N,
                "M": M,
                "M_specified": M_spec,
                "n_rep": n_rep,
                "two_sample_mean": float(two.mean()),
                "one_sample_proxy_mean": float(one.mean()),
                "ratio_mean": float(ratio.mean()),
                "ratio_std": float(ratio.std(ddof=1)) if len(ratio) > 1 else 0.0,
                "heuristic_2_to_the_1_over_d": float(2.0 ** (1.0 / d)),
                "two_sample_values": [float(x) for x in two],
                "one_sample_proxy_values": [float(x) for x in one],
            }
            cells.append(rec)
            print(
                f"  d={d:<3} N={N:<5} M={M:<5}  two={rec['two_sample_mean']:.4f}  "
                f"one={rec['one_sample_proxy_mean']:.4f}  "
                f"ratio={rec['ratio_mean']:.3f}  2^(1/d)={rec['heuristic_2_to_the_1_over_d']:.3f}"
            )
    elapsed = time.time() - t0
    out = {
        "_definition": (
            "two_sample = W2(X_N, X'_N) exact assignment (w2_exact); "
            "one_sample_proxy = W2(X_N, X_M) exact Kantorovich LP, "
            "X_N, X'_N, X_M iid N(0, I_d)."
        ),
        "_caveat": (
            "the one-sample proxy is the distance to a large empirical "
            "measure (M = 2000), not to the population; at N = 1024 the "
            "d-grid is thinned and n_rep = 1"
        ),
        "elapsed_seconds": elapsed,
        "cells": cells,
        "ratio_by_d_N": {
            f"d{c['d']}_N{c['N']}": {
                "ratio_mean": c["ratio_mean"],
                "heuristic_2_to_the_1_over_d": c["heuristic_2_to_the_1_over_d"],
            }
            for c in cells
        },
    }
    art.save("two_sample_vs_one_sample.json", out, indent=2)
    return out


if __name__ == "__main__":
    main()
