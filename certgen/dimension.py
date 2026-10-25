"""
Intrinsic-dimension estimates.

The rate at which the empirical W2 distance decays is set by an intrinsic
dimension of the data (Weed & Bach, Bernoulli 25(4A):2620-2648, 2019), and
measures can behave differently at different scales. Three estimators are
reported together:

  wasserstein_dimension  slope of log W2(X_n, X'_n) against log n over three
                         disjoint-split sizes (N/8, N/4, N/2), the transport
                         analogue of Block et al. (JMLR 23(313), 2022);
  twonn_dimension        two-nearest-neighbour estimator (Facco et al. 2017);
  mle_dimension          maximum-likelihood estimator (Levina & Bickel 2004),
                         local estimates combined as in Pope et al. (2021).

Their spread, the ratio of the largest to the smallest estimate, indicates
disagreement between the transport-based and the geometric view of the data.

The slope estimate saturates in low dimension, where the decay approaches
the parametric rate n^(-1/2): on isotropic Gaussians it is biased upward in
low dimension and downward in high dimension (EXPONENT_SATURATION_TABLE).
The geometric estimators need distinct points; zero nearest-neighbour
distances are excluded and the fraction of tied points is reported with
every estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import t as _student_t

__all__ = ["MIN_N_TRUSTWORTHY", "DimensionReport", "wasserstein_dimension",
           "twonn_dimension", "mle_dimension", "intrinsic_dimension",
           "tie_fraction", "D_IDENTIFIED_MIN", "RATE_REGIME_MIN_D",
           "tail_index_hill", "HEAVY_TAIL_INDEX",
           "EXPONENT_SATURATION_TABLE"]

# Below this size the slope estimate has a worst-case bias above 25%.
MIN_N_TRUSTWORTHY = 2048

# For W_p the dimension-governed regime is d > 2p; p = 2 throughout.
RATE_REGIME_MIN_D = 4.0

# Below this value the slope estimate has saturated: laws of intrinsic
# dimension at most 2 all return values near 2.
D_IDENTIFIED_MIN = 2.5

# Mean slope estimate on isotropic Gaussians (N = 4096, n_rep = 3, six
# seeds), nominal -> measured. Source: results/slope_estimator_bias.json.
EXPONENT_SATURATION_TABLE = {
    1: 2.119, 2: 2.632, 3: 3.222, 4: 4.191, 6: 6.060, 8: 7.701, 12: 10.775,
}

# Tail-index threshold below which the reference tables of certgen.latent
# are not validated: on a multivariate Student-t law with 3 degrees of
# freedom the floor ratio exceeds the table by 44% at the measured dimension
# (results/transfer_probe.json).
HEAVY_TAIL_INDEX = 5.0


def _w2(X: np.ndarray, Y: np.ndarray) -> float:
    C = cdist(X, Y, "sqeuclidean")
    i, j = linear_sum_assignment(C)
    return float(np.sqrt(C[i, j].mean()))


def wasserstein_dimension(X: np.ndarray, n_rep: int = 3, seed: int = 0) -> float:
    """Slope dimension estimate d_hat = -1 / slope of log W2 against log n.

    Uses `n_rep` random disjoint splits at each of n in (N//8, N//4, N//2).
    """
    X = np.asarray(X, float)
    N = len(X)
    window: Sequence[int] = (N // 8, N // 4, N // 2)
    if window[0] < 8:
        raise ValueError(f"N={N} too small; window would be {window}")
    rng = np.random.default_rng(seed)
    log_n, log_w = [], []
    for s in window:
        for _ in range(n_rep):
            idx = rng.permutation(N)[: 2 * s]
            log_n.append(np.log(s))
            log_w.append(np.log(_w2(X[idx[:s]], X[idx[s:]])))
    slope = np.polyfit(log_n, log_w, 1)[0]
    if slope >= 0:
        raise RuntimeError("non-decreasing W2 with n: the sample is degenerate "
                           "or contains duplicated rows")
    return float(-1.0 / slope)


def tail_index_hill(X: np.ndarray, frac: float = 0.05) -> float:
    """Hill estimate of the tail index of the distances ||x - median||.

    Computed from the largest `frac` of those distances. A screening device
    without a formal guarantee: the estimator is consistent for polynomial
    tails and merely large for light ones. Values below HEAVY_TAIL_INDEX mark
    laws for which the reference tables are not validated.
    """
    X = np.asarray(X, float)
    r = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    r = np.sort(r)[::-1]
    k = max(10, int(frac * len(r)))
    if len(r) <= k or r[k] <= 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(r[:k] / r[k])))


def tie_fraction(X: np.ndarray) -> float:
    """Fraction of points whose nearest neighbour is at distance zero."""
    X = np.asarray(X, float)
    dist, _ = cKDTree(X).query(X, k=2)
    return float((dist[:, 1] <= 0).mean())


def twonn_dimension(X: np.ndarray, discard_tail: float = 0.1) -> float:
    """Two-nearest-neighbour estimator of Facco et al. (2017).

    With mu the ratio of the second to the first nearest-neighbour distance
    and F its empirical distribution function, the estimate is the slope of
    the line through the origin fitted to (log mu, -log(1 - F(mu))), after
    the largest `discard_tail` fraction of the ratios is discarded. Points
    whose nearest neighbour coincides with them are excluded; on quantised
    data these are the points of dense regions, so the estimate is then
    biased, and `tie_fraction` reports how many were excluded.
    """
    X = np.asarray(X, float)
    dist, _ = cKDTree(X).query(X, k=3)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    if ok.sum() < 16:
        raise ValueError("fewer than 16 points have a strictly positive "
                         "nearest-neighbour distance; the estimator is not "
                         "defined on this sample")
    mu = np.sort(r2[ok] / r1[ok])
    F = np.arange(1, len(mu) + 1) / len(mu)
    keep = F < (1.0 - discard_tail)
    x, y = np.log(mu[keep]), -np.log(1.0 - F[keep])
    return float(np.dot(x, y) / np.dot(x, x))


def mle_dimension(X: np.ndarray, k: int = 10) -> float:
    """Maximum-likelihood estimator of Levina & Bickel (2004).

    The local estimate at a point is the inverse of the mean of
    log(T_k / T_j) over its j = 1, ..., k-1 nearest neighbours, T_j being the
    distance to the j-th. The local estimates are combined by averaging
    their inverses, the form used by Pope et al. (ICLR 2021), which is less
    biased than their plain average. Points with a coincident neighbour are
    excluded rather than assigned a small positive distance, which would
    drive the estimate towards zero.
    """
    X = np.asarray(X, float)
    dist, _ = cKDTree(X).query(X, k=k + 1)
    d = dist[:, 1:]
    ok = d[:, 0] > 0
    if ok.sum() < 16:
        raise ValueError("fewer than 16 points have a strictly positive "
                         "nearest-neighbour distance; the estimator is not "
                         "defined on this sample")
    d = d[ok]
    mean_log = np.log(d[:, -1][:, None] / d[:, :-1]).mean(1)
    return float(1.0 / np.mean(mean_log))


@dataclass
class DimensionReport:
    """Result of `intrinsic_dimension`.

    Uncertainty fields of the slope estimate:

      wasserstein_sd            spread of one estimate under re-splitting the
                                same data; a property of the estimator at N
      wasserstein_se            wasserstein_sd / sqrt(n_boot), the Monte-Carlo
                                error of the reported mean
      wasserstein_sd_subsample  spread across subsamples of half the data,
                                which includes the variability of the data
      d_ci90_from_exponent      90% interval built on 1/d_hat, the linearly
                                estimated quantity, with the Student-t
                                quantile on n_boot - 1 degrees of freedom
    """
    wasserstein: float
    twonn: float
    mle: float
    n_samples: int
    d_ambient: int
    warnings: List[str] = field(default_factory=list)
    wasserstein_sd: float = float("nan")
    tie_fraction: float = float("nan")
    wasserstein_se: float = float("nan")
    wasserstein_sd_subsample: float = float("nan")
    wasserstein_sd_subsample_scaled: float = float("nan")
    n_boot: int = 0
    subsample_m: int = 0
    exponent: float = float("nan")
    exponent_sd: float = float("nan")
    exponent_se: float = float("nan")
    d_ci90_from_exponent: tuple = (float("nan"), float("nan"))
    d_ci90_quantile: float = float("nan")
    tail_index: float = float("nan")

    @property
    def rate_regime(self) -> str:
        """'parametric' (d_hat below D_IDENTIFIED_MIN), 'crossover' (below
        4) or 'dimension_governed'."""
        d = self.wasserstein
        if not np.isfinite(d):
            return "undefined"
        if d < D_IDENTIFIED_MIN:
            return "parametric"
        if d < RATE_REGIME_MIN_D:
            return "crossover"
        return "dimension_governed"

    @property
    def spread(self) -> float:
        v = [self.wasserstein, self.twonn, self.mle]
        return float(max(v) / max(min(v), 1e-9))

    @property
    def lattice_suspected(self) -> bool:
        """True when more than 1% of points have a coincident neighbour."""
        return bool(np.isfinite(self.tie_fraction) and self.tie_fraction > 0.01)

    @property
    def usable_for_sizing(self) -> bool:
        return (self.n_samples >= MIN_N_TRUSTWORTHY and self.spread <= 2.0
                and not self.lattice_suspected)

    def summary(self) -> str:
        sd = (f" +/- {self.wasserstein_sd:.2f}"
              if np.isfinite(self.wasserstein_sd) else "")
        L = [f"d_hat (slope estimate) = {self.wasserstein:.2f}{sd}"
             f"   [{self.rate_regime}]",
             f"   two-NN = {self.twonn:.2f}   MLE = {self.mle:.2f}   "
             f"spread = {self.spread:.2f}x   (N={self.n_samples}, "
             f"ambient={self.d_ambient}, ties={self.tie_fraction:.1%})"]
        if np.isfinite(self.d_ci90_from_exponent[0]):
            L.append(f"   90% interval = "
                     f"[{self.d_ci90_from_exponent[0]:.2f}, "
                     f"{self.d_ci90_from_exponent[1]:.2f}]")
        for w in self.warnings:
            L.append(f"   [!] {w}")
        return "\n".join(L)


def intrinsic_dimension(X: np.ndarray, n_rep: int = 3, seed: int = 0,
                        n_boot: int = 8,
                        data_bootstrap: int = 0) -> DimensionReport:
    """All three estimators, their spread and the diagnostics that qualify them.

    The slope estimate is averaged over `n_boot` re-splits of X. With
    `data_bootstrap` > 1 it is also recomputed on that many subsamples of
    half the data, which measures the variability of the data rather than
    of the splitting.
    """
    X = np.asarray(X, float)
    dw_reps = np.array([wasserstein_dimension(X, n_rep=n_rep, seed=seed + s)
                        for s in range(n_boot)])
    dw = float(dw_reps.mean())
    dw_sd = float(dw_reps.std(ddof=1)) if len(dw_reps) > 1 else float("nan")
    dw_se = (dw_sd / np.sqrt(n_boot)) if np.isfinite(dw_sd) else float("nan")

    # interval on the exponent 1/d, the quantity the regression estimates
    ex_reps = 1.0 / dw_reps
    ex = float(ex_reps.mean())
    ex_sd = float(ex_reps.std(ddof=1)) if len(ex_reps) > 1 else float("nan")
    ex_se = (ex_sd / np.sqrt(n_boot)) if np.isfinite(ex_sd) else float("nan")
    q_ci = (float(_student_t.ppf(0.95, n_boot - 1)) if n_boot > 1
            else float("nan"))
    if np.isfinite(ex_se) and ex_se > 0 and np.isfinite(q_ci) \
            and ex > q_ci * ex_se:
        d_ci = (float(1.0 / (ex + q_ci * ex_se)),
                float(1.0 / (ex - q_ci * ex_se)))
    else:
        d_ci = (float("nan"), float("nan"))

    dw_sub = dw_sub_scaled = float("nan")
    m = 0
    if data_bootstrap and data_bootstrap > 1:
        m = len(X) // 2
        srng = np.random.default_rng(seed + 7777)
        sub = []
        for b in range(data_bootstrap):
            idx = srng.choice(len(X), m, replace=False)
            try:
                sub.append(wasserstein_dimension(X[idx], n_rep=n_rep, seed=b))
            except Exception:
                continue
        if len(sub) > 1:
            dw_sub = float(np.std(np.asarray(sub), ddof=1))
            dw_sub_scaled = float(dw_sub * np.sqrt(m / max(len(X), 1)))
    ties = tie_fraction(X)
    tail = tail_index_hill(X)
    d2, dm = twonn_dimension(X), mle_dimension(X)
    warn: List[str] = []
    if len(X) < MIN_N_TRUSTWORTHY:
        warn.append(f"N={len(X)} < {MIN_N_TRUSTWORTHY}: the slope estimate has "
                    "a worst-case bias above 25% at this size.")
    spread = max(dw, d2, dm) / max(min(dw, d2, dm), 1e-9)
    if ties > 0.01:
        warn.append(
            f"{ties:.1%} of points have a coincident nearest neighbour; the "
            "law is quantised or contains duplicate rows, which biases both "
            "geometric estimators. Dequantise or deduplicate before reading "
            "their disagreement as geometry.")
    if spread > 2.0:
        cause = ("check the tie fraction first" if ties > 0.01
                 else "the law may lack a single governing exponent")
        warn.append(f"the estimators disagree by {spread:.1f}x ({cause}); "
                    "report a range rather than a single dimension.")
    if np.isfinite(tail) and tail < HEAVY_TAIL_INDEX:
        warn.append(
            f"tail-index estimate {tail:.2f} < {HEAVY_TAIL_INDEX:g}: polynomial "
            "tails, for which the reference tables of certgen.latent understate "
            "the floor ratio (by 44% for a Student-t law with 3 degrees of "
            "freedom), so certification margins computed from this d_hat are "
            "optimistic.")
    if dw > 0.5 * X.shape[1]:
        warn.append("d_hat approaches the ambient dimension: the sample may be "
                    "too small to reveal low-dimensional structure.")
    if dw < D_IDENTIFIED_MIN:
        warn.append(
            f"d_hat={dw:.2f} is at the saturation level of the slope estimate "
            f"(below {D_IDENTIFIED_MIN}); every law of dimension at most 2 "
            "returns a value near 2, so this is not a dimension estimate.")
    elif dw < RATE_REGIME_MIN_D:
        warn.append(
            f"d_hat={dw:.2f} < 4: the decay is in the crossover between the "
            "dimension-governed and the parametric rate, and the value cannot "
            "be identified with the Weed-Bach dimension.")
    if np.isfinite(dw_sub) and np.isfinite(dw_sd) and dw_sub > 2.0 * dw_sd:
        warn.append(
            f"the subsample spread ({dw_sub:.2f} at m={m}) exceeds twice the "
            f"re-split spread ({dw_sd:.2f}): uncertainty about this law's "
            "dimension is dominated by data variability; use "
            "wasserstein_sd_subsample.")
    return DimensionReport(dw, d2, dm, len(X), X.shape[1], warn,
                           wasserstein_sd=dw_sd, tie_fraction=ties,
                           wasserstein_se=dw_se,
                           wasserstein_sd_subsample=dw_sub,
                           wasserstein_sd_subsample_scaled=dw_sub_scaled,
                           n_boot=int(n_boot), subsample_m=int(m),
                           exponent=ex, exponent_sd=ex_sd, exponent_se=ex_se,
                           d_ci90_from_exponent=d_ci,
                           d_ci90_quantile=q_ci, tail_index=tail)
