"""
Reference generators and column standardisation.

The reference generators are computed from the real data and need no
training:

  T0      all mass at the real mean, the best constant generator;
  T1      each column resampled independently (the `perm` reference of Yang
          et al., arXiv:2403.10424), which keeps every marginal and no
          dependence;
  T2      the Gaussian with the real mean and covariance;
  T_self  a resample of real rows (their `self` reference), used as a ceiling.

The skill score of certgen.report divides by the distance of T0, T1 or T2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["Standardizer", "baseline_mean", "baseline_marginals",
           "baseline_gaussian", "baseline_self", "BASELINES", "get_baseline"]


@dataclass
class Standardizer:
    """Column standardisation fitted on the REAL sample only.

    Optimal-transport discrepancies are not scale invariant, so columns must
    be placed on a common footing. Fitting the transform on pooled
    real+synthetic data would let the model under test influence its own
    evaluation; the transform is part of the protocol, not of the candidate.
    """
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        s = X.std(0)
        self.scale_ = np.where(s > 1e-12, s, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer.fit must be called first")
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map standardised rows back to the original units.

        Exact inverse of `transform`, including on constant columns, whose
        scale is stored as 1.0.
        """
        if self.mean_ is None:
            raise RuntimeError("Standardizer.fit must be called first")
        return np.asarray(Z, float) * self.scale_ + self.mean_


def baseline_mean(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """T0: all mass at the mean.

    argmin_c E||x - c||^2 = E[x], so no constant generator is closer under a
    squared-cost transport distance; its distance to the data is the data
    scale tau = (tr Cov)^(1/2).
    """
    return np.tile(np.asarray(X, float).mean(0), (n, 1))


def baseline_marginals(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """T1: product of the empirical marginals.

    Reproduces every univariate distribution and no dependence between
    columns; a generator that does not beat T1 has captured no joint
    structure.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    return np.column_stack([rng.choice(X[:, j], n, replace=True)
                            for j in range(X.shape[1])])


def baseline_gaussian(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """T2: Gaussian with the empirical mean and covariance.

    Reproduces first and second moments and nothing beyond them.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    mu = X.mean(0)
    cov = np.atleast_2d(np.cov(X, rowvar=False)) + 1e-10 * np.eye(X.shape[1])
    return rng.multivariate_normal(mu, cov, size=n, method="cholesky")


def baseline_self(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """T_self: n rows resampled with replacement from X.

    A ceiling rather than a skill denominator. `certgen.evaluate` draws it
    from the rows outside the reference subsample, because rows shared with
    the reference pull its distance below the floor.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    return X[rng.choice(len(X), n, replace=True)]


BASELINES: Dict[str, Callable] = {
    "T0_mean": baseline_mean,
    "T1_marginals": baseline_marginals,
    "T2_gaussian": baseline_gaussian,
    "T_self": baseline_self,
}


def get_baseline(name: str) -> Callable:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'; available: {sorted(BASELINES)}")
    return BASELINES[name]
