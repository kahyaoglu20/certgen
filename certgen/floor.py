"""
The resolution floor: the distance between two disjoint real samples of the
same size.

Two independent samples from one law are at a strictly positive W2 distance.
Repeating the split gives the replicate distribution of that distance, which
approximates the null distribution of the statistic when a generated sample
is drawn from the data law. A generated sample whose distance to real data
lies inside it is, at that sample size, indistinguishable from a second real
sample; differences between models smaller than its spread cannot be
resolved.

The replicate distribution supports screening and design. For a formal
comparison of two models against a shared reference sample use the
relative-similarity test of Bounliphone et al. (ICLR 2016), which models the
correlation that the shared reference induces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .discrepancy import get_discrepancy

__all__ = ["FloorReport", "resolution_floor"]


@dataclass
class FloorReport:
    """Replicate distribution of the distance between two same-law samples."""
    values: np.ndarray
    n_per_side: int
    metric: str
    n_rep: int

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        return float(self.values.std(ddof=1)) if len(self.values) > 1 else 0.0

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.values, q))

    @property
    def q95(self) -> float:
        return self.quantile(0.95)

    @property
    def cv(self) -> float:
        """Coefficient of variation of the replicate distribution."""
        m = self.median
        return float(self.std / m) if m > 0 else float("nan")

    def p_value(self, observed: float) -> float:
        """Upper-tail Monte-Carlo p-value P(floor >= observed).

        Computed as (count + 1) / (n_rep + 1). A small value indicates a
        generator that is distinguishable from real data. The test rejects
        only upwards: a sample that shares points with the reference falls
        below the floor, so read `p_value_low` alongside it.
        """
        n = len(self.values)
        if n == 0:
            return float("nan")
        count = int(np.sum(self.values >= observed))
        return float((count + 1) / (n + 1))

    def p_value_low(self, observed: float) -> float:
        """Lower-tail Monte-Carlo p-value P(floor <= observed).

        A generator that draws fresh samples from the data law produces an
        ordinary draw from the replicate distribution. A small value means the
        generated sample is closer to the reference than a second real sample
        would be, the signature of leakage. The smallest attainable value is
        1 / (n_rep + 1), so n_rep >= 19 is needed to reach 0.05.
        """
        n = len(self.values)
        if n == 0:
            return float("nan")
        count = int(np.sum(self.values <= observed))
        return float((count + 1) / (n + 1))

    def equivalence(self, observed: float, delta: float, alpha: float = 0.05) -> dict:
        """Two one-sided tests of |D_obs - floor| < delta at level alpha.

        The margin delta must be fixed before the generator is evaluated.
        Under each one-sided null the replicate distribution is shifted by
        +delta or -delta:

          too bad:   reject when P(F <= D_obs - delta) <= alpha
          too good:  reject when P(F >= D_obs + delta) <= alpha

        and equivalence is declared when both are rejected. At delta = 0 the
        two p-values are those of the two-sided floor test. `delta_min`, half
        the width of the central (1 - 2 alpha) range of the replicates, is the
        smallest margin the design can support.

        Returns p_too_bad, p_too_good, equivalent, feasible, delta, alpha,
        delta_min, n_rep and floor_median.
        """
        v = np.asarray(self.values, float)
        n = len(v)
        if n == 0 or delta < 0:
            return {"note": "no replicates or negative delta",
                    "equivalent": False, "feasible": False}
        qa = float(np.quantile(v, alpha))
        q1a = float(np.quantile(v, 1.0 - alpha))
        delta_min = (q1a - qa) / 2.0
        feasible = 2.0 * delta > (q1a - qa)
        p_A = float((int(np.sum(v <= observed - delta)) + 1) / (n + 1))
        p_B = float((int(np.sum(v >= observed + delta)) + 1) / (n + 1))
        return {
            "p_too_bad": p_A,
            "p_too_good": p_B,
            "equivalent": bool(p_A <= alpha and p_B <= alpha),
            "feasible": bool(feasible),
            "delta": float(delta), "alpha": float(alpha),
            "delta_min": float(delta_min),
            "n_rep": int(n),
            "floor_median": float(np.median(v)),
        }

    def sd_difference_at(self, level: float) -> float:
        """Floor proxy for the standard deviation of a paired difference
        D_A - D_B between two models whose distances sit near `level`.

        The proxy sqrt(2) * CV(floor) * level assumes that the spread of D is
        proportional to its level and treats the two distances as
        independent. It ignores the positive correlation induced by a shared
        reference sample and therefore overstates the paired spread.
        """
        lv = float(level)
        if not np.isfinite(lv) or lv <= 0:
            return float("nan")
        return float(np.sqrt(2.0) * self.cv * lv)

    def mdd(self, level: float, alpha: float = 0.05, power: float = 0.80) -> float:
        """Minimum detectable difference at distance level `level`,

            MDD = (z_{1-alpha/2} + z_{power}) * sd_difference_at(level).
        """
        from scipy.stats import norm
        if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
            raise ValueError("alpha and power must be in (0, 1)")
        sd = self.sd_difference_at(level)
        if not np.isfinite(sd):
            return float("nan")
        z = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
        return float(z * sd)

    def summary(self) -> str:
        return (f"floor[{self.metric}] n/side={self.n_per_side} reps={self.n_rep}\n"
                f"  median={self.median:.6f}  q95={self.quantile(0.95):.6f}  "
                f"sd={self.std:.3g}  CV={self.cv*100:.2f}%\n"
                f"  minimum detectable difference at the floor level "
                f"(alpha=0.05, power=0.80) = {self.mdd(self.median):.6f}")


def resolution_floor(X: np.ndarray, n_per_side: Optional[int] = None,
                     metric: str = "w2", n_rep: int = 20, seed: int = 0,
                     **kw) -> FloorReport:
    """Replicate distribution of the distance between disjoint subsamples.

    Each replicate draws two disjoint subsamples of size `n_per_side`
    (default N // 2) from X. The subsamples must be disjoint: shared points
    pull the distance towards zero.
    """
    X = np.asarray(X, float)
    fn = get_discrepancy(metric)
    rng = np.random.default_rng(seed)
    N = len(X)
    n = int(n_per_side or N // 2)
    if n < 2:
        raise ValueError(f"n_per_side={n} too small")
    if 2 * n > N:
        raise ValueError(f"need 2*n_per_side={2*n} <= N={N}; lower n_per_side")
    vals = np.empty(n_rep)
    for r in range(n_rep):
        idx = rng.permutation(N)[: 2 * n]
        vals[r] = fn(X[idx[:n]], X[idx[n:]], seed=r, **kw)
    return FloorReport(vals, n, metric, n_rep)
