"""Synthetic laws used by the experiments and the tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["gaussian_copula", "curved_manifold"]


def gaussian_copula(n: int, d: int = 8, rho: float = 0.8,
                    seed: int = 0) -> np.ndarray:
    """Gaussian copula with bounded, non-Gaussian marginals.

    Correlation rho^|i-j| between coordinates i and j; with U_j = Phi(Z_j),
    even coordinates are exp(2 U_j), skewed on (1, e^2), and odd coordinates
    are tan(0.6 pi (U_j - 1/2)), a Cauchy shape truncated to
    (-tan(0.3 pi), tan(0.3 pi)).
    """
    rng = np.random.default_rng(seed)
    C = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    U = norm.cdf(rng.multivariate_normal(np.zeros(d), C, size=n))
    return np.column_stack([
        np.exp(2 * U[:, j]) if j % 2 == 0
        else np.tan(np.pi * (U[:, j] - 0.5) * 0.6) for j in range(d)])


class curved_manifold:
    """Fixed smooth embedding of R^m into R^D.

    The embedding is drawn once, from a random stream independent of the
    sample size, so every call of `sample` draws from the same law.
    """

    def __init__(self, m: int, D: int = 64, seed: int = 0):
        g = np.random.default_rng(10_000 + seed)
        self.m, self.D = m, D
        self.A = g.normal(size=(m, D)) / np.sqrt(m)
        self.B = g.normal(size=(m, D)) / np.sqrt(m)
        ref = np.random.default_rng(999).normal(size=(4096, m))
        self._scale = float((ref @ self.A + 0.3 * np.tanh(ref) @ self.B).std())

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        t = np.random.default_rng(20_000 + seed).normal(size=(n, self.m))
        return (t @ self.A + 0.3 * np.tanh(t) @ self.B) / self._scale
