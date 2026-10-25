"""
Two-sample discrepancies between empirical measures.

Every function compares two finite samples. With equal sizes and uniform
weights, the 2-Wasserstein distance between two empirical measures is a
linear assignment problem and is computed exactly. The plug-in distance is
biased upward as an estimator of the population distance; that bias is the
resolution floor (certgen.floor), and its decay rate is set by the intrinsic
dimension of the data (Weed & Bach, Bernoulli 25(4A), 2019).

w2_exact         exact assignment (scipy linear_sum_assignment); no
                 regularisation, no slicing.
energy_distance  unbiased energy distance (Szekely & Rizzo 2013).
mmd_rbf          square root of the unbiased MMD^2 with an RBF kernel whose
                 bandwidth is the median off-diagonal pairwise distance of the
                 real sample (Gretton et al. 2012).
sliced_w2        sliced 2-Wasserstein distance, for comparison only; it
                 underestimates W2 strongly in high dimension.

The unbiased statistics have expectation zero under the same-law null, so
they have no floor (FLOOR_IS_POSITIVE); sliced W2 has a floor whose decay is
not governed by the dimension (DIMENSION_GOVERNED_RATE).
"""
from __future__ import annotations

import warnings

from typing import Callable, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = ["w2_exact", "energy_distance", "mmd_rbf", "sliced_w2",
           "DISCREPANCIES", "get_discrepancy"]

_DEFAULT_MAX_N = 4000


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), n, replace=False)]


def w2_exact(X: np.ndarray, Y: np.ndarray, max_n: int = _DEFAULT_MAX_N,
             seed: int = 0) -> float:
    """Exact 2-Wasserstein distance between two uniform empirical measures.

    Both samples are subsampled to n = min(len(X), len(Y), max_n) and the
    assignment problem between them is solved exactly. When an input exceeds
    `max_n` a RuntimeWarning states the size actually used; pass a larger
    `max_n` to compute at the full size.
    """
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_n)
    if n < 2:
        raise ValueError("need at least 2 points per sample")
    if len(X) > max_n or len(Y) > max_n:
        warnings.warn(
            f"w2_exact: inputs of size {len(X)}x{len(Y)} exceed max_n={max_n} "
            f"and were subsampled to n={n}. The value returned is the exact "
            "W2 at n={} per side, NOT at the input size; do not record "
            "n_per_side above max_n.".format(n),
            RuntimeWarning, stacklevel=2)
    X, Y = _subsample(X, n, rng), _subsample(Y, n, rng)
    C = cdist(X, Y, "sqeuclidean")
    i, j = linear_sum_assignment(C)
    return float(np.sqrt(C[i, j].mean()))


def energy_distance(X: np.ndarray, Y: np.ndarray, max_n: int = _DEFAULT_MAX_N,
                    seed: int = 0) -> float:
    """Unbiased energy distance 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||."""
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, max_n, rng), _subsample(Y, max_n, rng)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("need at least 2 points per sample")
    dxy = cdist(X, Y).mean()
    dxx = cdist(X, X).sum() / (n * (n - 1))
    dyy = cdist(Y, Y).sum() / (m * (m - 1))
    return float(max(2 * dxy - dxx - dyy, 0.0))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: Optional[float] = None,
            max_n: int = _DEFAULT_MAX_N, seed: int = 0) -> float:
    """Square root of the unbiased MMD^2 with an RBF kernel.

    Bandwidth is the median pairwise distance of the REAL sample, computed on
    the off-diagonal (the zero diagonal would bias the median down). Fitting
    on pooled data would let the model under test influence its own metric.
    """
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, max_n, rng), _subsample(Y, max_n, rng)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("need at least 2 points per sample")
    if bandwidth is None:
        sub = _subsample(X, 500, rng)
        D = cdist(sub, sub)
        tri = D[np.triu_indices(len(sub), k=1)]
        med = float(np.median(tri)) if len(tri) else 0.0
        bandwidth = med if med > 0 else 1.0
    k = lambda A, B: np.exp(-cdist(A, B, "sqeuclidean") / (2 * bandwidth ** 2))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    val = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
           - 2.0 * Kxy.mean())
    return float(np.sqrt(max(val, 0.0)))


def sliced_w2(X: np.ndarray, Y: np.ndarray, n_proj: int = 512,
              max_n: int = _DEFAULT_MAX_N, seed: int = 0) -> float:
    """Sliced 2-Wasserstein distance over `n_proj` random directions."""
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_n)
    X, Y = _subsample(X, n, rng), _subsample(Y, n, rng)
    th = rng.normal(size=(X.shape[1], n_proj))
    th /= np.linalg.norm(th, axis=0, keepdims=True)
    px, py = np.sort(X @ th, axis=0), np.sort(Y @ th, axis=0)
    return float(np.sqrt(((px - py) ** 2).mean()))


DISCREPANCIES: Dict[str, Callable] = {
    "w2": w2_exact, "energy": energy_distance,
    "mmd": mmd_rbf, "sliced_w2": sliced_w2,
}

# Whether the same-law expectation of a statistic is positive. The plug-in W2
# satisfies E W2(X_n, X'_n) > 0 = W2(P, P); the unbiased statistics have
# expectation zero under the same-law null, a large fraction of their null
# replicates is clipped to zero, and log(floor) is undefined. For them the
# design quantity is the null variance (a classical two-sample power
# analysis), which this package does not compute.
FLOOR_IS_POSITIVE: Dict[str, bool] = {
    "w2": True,          # biased upward under the null; rate n^(-1/d*)
    "sliced_w2": True,   # also biased upward, but see the module docstring
    "energy": False,     # unbiased U-statistic; null mean exactly 0
    "mmd": False,        # unbiased MMD^2; null mean exactly 0
}

DIMENSION_GOVERNED_RATE: Dict[str, bool] = {
    "w2": True,
    "sliced_w2": False,  # projection kills the dimension dependence
    "energy": False,
    "mmd": False,
}


def get_discrepancy(name: str) -> Callable:
    if name not in DISCREPANCIES:
        raise KeyError(f"unknown metric '{name}'; available: {sorted(DISCREPANCIES)}")
    return DISCREPANCIES[name]
