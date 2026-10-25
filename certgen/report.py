"""
Floor-corrected reporting of a generated sample.

With f the floor median at the evaluation size and D_B the distance of a
reference generator B,

    V     = (D_obs - f) / (D_B - f)
    skill = 1 - V

so that skill is 1 for a generator as close to the real sample as a second
real sample and 0 for a generator as distant as B. The reference generators
(certgen.baselines) are T0 (the mean), T1 (independent columns, the `perm`
reference of Yang et al., arXiv:2403.10424) and T2 (a Gaussian with the real
mean and covariance); T_self (a resample of real rows outside the reference)
is reported as a ceiling.

`evaluate` reports both tail probabilities of D_obs under the replicate
distribution, a null-calibrated leakage screen (certgen.leakage) and the
minimum detectable difference at alpha = 0.05 and power 0.80. It does not
test whether one model beats another; for that use the relative-similarity
test of Bounliphone et al. (ICLR 2016).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .baselines import Standardizer, baseline_self, get_baseline
from .leakage import LeakageReport, leakage_check
from .discrepancy import get_discrepancy
from .floor import FloorReport, resolution_floor

__all__ = ["EvaluationReport", "evaluate", "DEFAULT_BASELINES"]

DEFAULT_BASELINES = ("T0_mean", "T1_marginals", "T2_gaussian")
D_SELF_RATIO_FLAG = 0.5       # warn if the observed distance < 0.5 * D_self


@dataclass
class EvaluationReport:
    """Result of `evaluate`."""
    metric: str
    n_per_side: int
    d_ambient: int
    observed: float
    floor: FloorReport
    baselines: Dict[str, float]
    V: Dict[str, float]
    p_value: float
    mdd: float
    warnings: List[str] = field(default_factory=list)
    p_value_low: float = float("nan")
    duplicate_fraction: float = float("nan")
    leakage_suspected: bool = False
    d_self: float = float("nan")
    resolving_power: Dict[str, float] = field(default_factory=dict)
    baseline_sd: Dict[str, float] = field(default_factory=dict)
    leakage: Optional[LeakageReport] = None

    @property
    def skill(self) -> Dict[str, float]:
        return {k: 1.0 - v for k, v in self.V.items()}

    def skill_ci(self, level: float = 0.95, n_boot: int = 4000,
                 seed: int = 0) -> Dict[str, tuple]:
        """Percentile interval for the skill against each reference generator.

        The floor median appears in numerator and denominator of V, so it is
        resampled jointly: f by bootstrap of the floor replicates, D_obs and
        D_B as Normal(value, CV(floor) * level). The interval conditions on
        this real sample and is conservative to the extent that the floor
        proxy overstates the spread of D.
        """
        rng = np.random.default_rng(seed)
        v = np.asarray(self.floor.values, float)
        if len(v) < 3:
            return {b: (float("nan"), float("nan")) for b in self.V}
        cv = self.floor.cv
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        f_b = np.median(v[idx], axis=1)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        out: Dict[str, tuple] = {}
        for b, D_B in self.baselines.items():
            if b not in self.V:
                continue
            d_obs = self.observed + rng.normal(
                0.0, cv * max(self.observed, self.floor.median), n_boot)
            d_b = D_B + rng.normal(0.0, cv * max(D_B, self.floor.median), n_boot)
            denom = d_b - f_b
            with np.errstate(divide="ignore", invalid="ignore"):
                sk = 1.0 - (d_obs - f_b) / denom
            sk = sk[np.isfinite(sk) & (denom > 0)]
            out[b] = ((float(np.quantile(sk, lo_q)),
                       float(np.quantile(sk, hi_q))) if len(sk) > 10
                      else (float("nan"), float("nan")))
        return out

    @property
    def valid(self) -> bool:
        """False when leakage is suspected; skill is then not a fidelity
        statement."""
        return not self.leakage_suspected

    def to_dict(self) -> dict:
        return {"metric": self.metric, "n_per_side": self.n_per_side,
                "d_ambient": self.d_ambient, "observed": self.observed,
                "floor_median": self.floor.median, "floor_sd": self.floor.std,
                "p_value": self.p_value, "p_value_low": self.p_value_low,
                "mdd": self.mdd,
                "duplicate_fraction": self.duplicate_fraction,
                "leakage_suspected": self.leakage_suspected,
                "d_self": self.d_self,
                "resolving_power": self.resolving_power,
                "baseline_sd": self.baseline_sd,
                "leakage": (self.leakage.to_dict() if self.leakage else None),
                "baselines": self.baselines, "V": self.V,
                "skill": self.skill,
                "skill_ci95": {k: list(v) for k, v in self.skill_ci().items()},
                "warnings": self.warnings}

    def summary(self) -> str:
        L = [f"metric={self.metric}  n/side={self.n_per_side}  "
             f"d={self.d_ambient}",
             f"  observed D            = {self.observed:.6f}",
             f"  floor median          = {self.floor.median:.6f}  "
             f"(q95 {self.floor.q95:.6f}, sd {self.floor.std:.3g})",
             f"  p(floor >= observed)  = {self.p_value:.3f}",
             f"  p(floor <= observed)  = {self.p_value_low:.3f}   "
             f"(small => leakage)",
             f"  min detectable diff   = {self.mdd:.6f}   "
             f"(alpha=0.05, power=0.80)",
             "  ---- reference generators ----"]
        try:
            _ci = self.skill_ci()
        except Exception:
            _ci = {}
        for k in self.baselines:
            if k in self.V:
                c = _ci.get(k)
                ci_s = (f"  [{c[0]:+.3f}, {c[1]:+.3f}]"
                        if c and np.isfinite(c[0]) else "")
                L.append(f"  {k:<16} D={self.baselines[k]:.6f}   "
                         f"V={self.V[k]:+.3f}   skill={1 - self.V[k]:+.3f}{ci_s}")
            else:
                L.append(f"  {k:<16} D={self.baselines[k]:.6f}   (ceiling)")
        if np.isfinite(self.d_self):
            L.append(f"  T_self (ceiling) D={self.d_self:.6f}")
        for w in self.warnings:
            L.append(f"  [!] {w}")
        return "\n".join(L)


def evaluate(X_real: np.ndarray, X_gen: np.ndarray, metric: str = "w2",
             n_per_side: Optional[int] = None, n_rep: int = 20,
             standardize: bool = True, seed: int = 0,
             baselines: Sequence[str] = DEFAULT_BASELINES,
             n_baseline_rep: int = 3,
             **kw) -> EvaluationReport:
    """Floor-corrected evaluation of a generated sample against real data.

    Columns are standardised on the real sample only. The floor is measured at
    the evaluation size n, which must satisfy 2n <= len(X_real) and
    n <= len(X_gen). The observed distance and every reference generator are
    computed against one shared real subsample; each stochastic reference is
    averaged over `n_baseline_rep` draws and its spread is reported.
    """
    X_real = np.asarray(X_real, float)
    X_gen = np.asarray(X_gen, float)
    if X_real.ndim != 2 or X_gen.ndim != 2:
        raise ValueError("expected 2-D arrays (n_samples, n_features)")
    if X_real.shape[1] != X_gen.shape[1]:
        raise ValueError(f"column mismatch: real has {X_real.shape[1]}, "
                         f"generated has {X_gen.shape[1]}")

    if standardize:
        st = Standardizer().fit(X_real)
        X_real, X_gen = st.transform(X_real), st.transform(X_gen)

    fn = get_discrepancy(metric)
    N = len(X_real)
    n = int(n_per_side or min(N // 2, len(X_gen)))
    rng = np.random.default_rng(seed)

    if n < 2:
        raise ValueError(f"n_per_side={n} too small")
    if 2 * n > N:
        raise ValueError(
            f"need 2*n_per_side={2*n} <= len(X_real)={N} for disjoint floor "
            f"splits; lower n_per_side")
    if len(X_gen) < n:
        raise ValueError(
            f"n_per_side={n} exceeds the generated sample ({len(X_gen)} rows); "
            "the floor and the observed distance must be measured at the same "
            f"size. Pass n_per_side<={len(X_gen)} or generate more samples.")

    floor = resolution_floor(X_real, n, metric=metric, n_rep=n_rep,
                             seed=seed, **kw)
    f = floor.median

    # one shared real reference subsample for the observed D and every baseline
    _perm = rng.permutation(N)
    _ref_idx = _perm[:n]
    _rest_idx = _perm[n:]
    Xr = X_real[_ref_idx]
    Xg = X_gen[rng.permutation(len(X_gen))[:n]]
    observed = fn(Xr, Xg, seed=seed, **kw)

    base_D: Dict[str, float] = {}
    base_sd: Dict[str, float] = {}
    V: Dict[str, float] = {}
    for b in baselines:
        if b == "T_self":
            continue
        draws = [fn(Xr, get_baseline(b)(X_real, n, seed=seed + 977 * k),
                    seed=seed + 977 * k, **kw)
                 for k in range(max(1, int(n_baseline_rep)))]
        base_D[b] = float(np.mean(draws))
        base_sd[b] = float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0
        denom = base_D[b] - f
        V[b] = float((observed - f) / denom) if abs(denom) > 1e-12 else float("nan")

    # T_self: a resample of the real rows outside the reference subsample
    _rs = np.random.default_rng(seed + 4242)
    _disjoint = X_real[_rest_idx]
    if len(_disjoint) >= max(16, n // 4):
        X_self = _disjoint[_rs.choice(len(_disjoint), n, replace=True)]
    else:
        X_self = baseline_self(X_real, n, seed=seed)
    d_self = float(fn(Xr, X_self, seed=seed, **kw))

    warn: List[str] = []
    p = floor.p_value(observed)
    p_low = floor.p_value_low(observed)

    leak_rep = leakage_check(X_real, X_gen, seed=seed)
    dup_frac = leak_rep.overlap_gen
    leak = bool(leak_rep.suspected or p_low <= 0.05)
    if leak:
        bits = list(leak_rep.reasons)
        if p_low <= 0.05:
            bits.append(f"the observed distance lies in the lower tail of the "
                        f"floor (p={p_low:.3f}).")
        warn.append(
            "leakage suspected: " + " ".join(bits) + " The generated sample is "
            "closer to the reference than an independent real sample, so the "
            "skill does not measure fidelity. Check the train/evaluation split.")
    for w in leak_rep.warnings:
        warn.append(f"leakage screen: {w}")
    if np.isfinite(d_self) and d_self > 0 and observed < D_SELF_RATIO_FLAG * d_self:
        warn.append(
            f"observed D={observed:.4g} is below {D_SELF_RATIO_FLAG:g} x "
            f"D_self={d_self:.4g}, a resample of real rows outside the "
            "reference; a generator drawing fresh samples should not be this "
            "close.")
    if p_low <= 0.05:
        warn.append(
            f"the observed distance lies in the lower tail of the floor "
            f"(p_value_low={p_low:.3f}); read the leakage screen.")
    elif any(v < 0 for v in V.values()):
        warn.append(
            "skill > 1 against at least one reference generator while "
            f"p_value_low={p_low:.3f}; a generator drawing from the data law "
            "exceeds 1 about half the time, so this is not evidence of "
            "leakage. Report indistinguishability, not skill above 1.")
    if p > 0.05 and not leak:
        warn.append(f"the observed distance lies inside the floor (p={p:.2f}): "
                    "the model is indistinguishable from real data at this n.")
    if V.get("T1_marginals", np.inf) > 0.9:
        warn.append("V >= 0.9 against independent columns: little evidence "
                    "that the model captures dependence between columns.")
    for _b, _D in base_D.items():
        _sd_b = base_sd.get(_b, 0.0)
        _span = _D - f
        if _sd_b > 0 and _span < 2 * _sd_b:
            warn.append(
                f"reference {_b} lies {_span:.4g} above the floor while its "
                f"own seed-to-seed sd is {_sd_b:.4g}; the skill denominator is "
                "dominated by noise. Raise n_baseline_rep.")
        if _span <= 0:
            warn.append(
                f"no resolving power: reference {_b} scores D={_D:.4g}, at or "
                f"below the floor {f:.4g}, so it is indistinguishable from real "
                f"data at n={n} and the skill against it is undefined.")
        elif np.isfinite(m_ := floor.mdd(max(_D, f))) and _span < m_:
            warn.append(
                f"reference {_b} lies {_span:.4g} above the floor, less than "
                f"the minimum detectable difference {m_:.4g} at this n; no "
                "ordering of models against it is supported.")
    if f > 0.5 * base_D.get("T0_mean", np.inf):
        warn.append("the floor exceeds half the T0 distance: n is too small "
                    "for this dimension (see certgen.design).")

    mdd = floor.mdd(max(observed, f))
    resolving_power = {b: float((D - f) / m) if (m := floor.mdd(max(D, f))) > 0
                       else float("nan") for b, D in base_D.items()}
    return EvaluationReport(metric, n, X_real.shape[1], observed, floor,
                            base_D, V, p, mdd, warn,
                            p_value_low=p_low, duplicate_fraction=dup_frac,
                            leakage_suspected=leak, d_self=d_self,
                            resolving_power=resolving_power,
                            baseline_sd=base_sd, leakage=leak_rep)
