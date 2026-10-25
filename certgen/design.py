"""
Prospective design of an evaluation from real data alone.

The floor of the plug-in W2 statistic decays as

    floor(n) ~ C * n^(-1/d_eff)

with an effective exponent d_eff set by the intrinsic dimension of the data
(Weed & Bach, Bernoulli 25(4A), 2019). `fit_floor_law` fits C and d_eff on a
geometric grid of per-side sizes, and the fitted law is inverted in two ways:

  * `required_n`        the size at which the floor reaches a target level;
  * `resolvable_effect` the smallest change in distance that two
                        evaluations at size n resolve at (alpha, power).

Inverting a fitted exponent is meaningful only where a single power law
describes the decay and the exponent lies in the dimension-governed regime
d_eff >= 4. `FloorLaw.single_exponent_ok` tests the first condition with a
parametric-bootstrap curvature test; when it fails, or when d_eff < 4, the
sizing functions return nan and `FloorLaw.required_n_range` reports how far
the half-grid fits disagree. `FloorLaw.required_n_interval` propagates the
fit's sampling error, and `FloorLaw.severity` / `severity_panel` state which
departures from a single exponent a non-rejection excludes.
"""
from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .discrepancy import DIMENSION_GOVERNED_RATE, FLOOR_IS_POSITIVE
from .floor import FloorReport, resolution_floor

__all__ = ["FloorLaw", "fit_floor_law", "floor_law_from_profile", "required_n",
           "required_n_sensitivity",
           "resolvable_effect", "power_curve", "combine_severity",
           "GATE_SIZE_BY_D_EFF", "GATE_POWER_GRID_6PT",
           "INTERVAL_LEVEL_CALIBRATION", "RATE_REGIME_MIN_D"]

# Size of the curvature test at nominal 0.05 against exact single power laws
# (isotropic Gaussians, N = 1024, n_rep = 12, six grid points), keyed by the
# median d_eff of the null law. Source: results/curvature_test_size.json.
# Below d_eff = 4 a single power law is misspecified (the rate theory gives no
# pure power law there), so rejections there detect a real departure.
GATE_SIZE_BY_D_EFF = {
    2.69: 0.105, 3.43: 0.12, 4.27: 0.025, 5.92: 0.04, 7.44: 0.023,
}

# Power of the six-point curvature test over [125, 4000] against the
# two-scale floor n^(-1/2) + 0.35 n^(-1/8), keyed by the median standard
# error of log f. Source: results/curvature_test_power.json.
GATE_POWER_GRID_6PT = {0.01: 0.885, 0.03: 0.18, 0.05: 0.08}

# Nominal level at which `required_n_interval` delivers the requested
# coverage, as a median over the regime in which it was measured: six-point
# grid over [125, 4000], n_rep = 12, d in {4, 5, 6}, replicate coefficient
# of variation in {0.03, 0.08, 0.13}. Source: results/interval_coverage.json.
INTERVAL_LEVEL_CALIBRATION = {0.8: 0.82, 0.85: 0.8689, 0.9: 0.9125, 0.95: 0.9561}

# Measured delivered coverage of the uncalibrated interval at nominal 0.90.
INTERVAL_COVERAGE_AT_090 = {"min": 0.86, "max": 0.92, "median": 0.88}

# For W_p the dimension-governed regime is d > 2p; p = 2 throughout.
RATE_REGIME_MIN_D = 4.0

_CALIBRATION_REGIME = ("6-point grid, n_rep in [8,24], d_eff in [3.5,7.0], "
                       "median replicate CV in [0.02,0.16]")
_Z90 = 1.6448536269514722


def combine_severity(thresholds, cis) -> dict:
    """Two-component 90% interval for a severity threshold.

    thresholds : per-seed interpolated thresholds t_s
    cis        : per-seed binomial 90% intervals [lo_s, hi_s]

        var_within  = mean_s(((hi_s - lo_s) / (2 z90))^2)
        var_between = sample variance of {t_s}   (ddof = 1)
        interval    = median(t_s) +/- z90 sqrt(var_within + var_between)
    """
    t = np.asarray([x for x in thresholds if np.isfinite(x)], float)
    if t.size == 0:
        nan = float("nan")
        return dict(median=nan, sd_within=nan, sd_between=nan, sd_total=nan,
                    lo=nan, hi=nan, width_binomial_only=nan,
                    width_combined=nan, inflation=nan, n_seeds=0)
    half = []
    for c in cis:
        if c and np.all(np.isfinite(c)):
            half.append((c[1] - c[0]) / (2.0 * _Z90))
    sd_within = float(np.sqrt(np.mean(np.square(half)))) if half else 0.0
    sd_between = float(np.std(t, ddof=1)) if t.size > 1 else 0.0
    sd_total = float(np.sqrt(sd_within ** 2 + sd_between ** 2))
    med = float(np.median(t))
    w_binom = 2.0 * _Z90 * sd_within
    w_comb = 2.0 * _Z90 * sd_total
    return dict(median=med, sd_within=sd_within, sd_between=sd_between,
                sd_total=sd_total, lo=med - _Z90 * sd_total,
                hi=med + _Z90 * sd_total, width_binomial_only=w_binom,
                width_combined=w_comb,
                inflation=(w_comb / w_binom) if w_binom > 0 else float("nan"),
                n_seeds=int(t.size))


def _half_slices(k: int):
    """Index ranges of the lower and upper halves of a k-point grid."""
    if k >= 4:
        return slice(0, k // 2), slice(k - k // 2, k)
    return slice(0, 2), slice(k - 2, k)


def _wls_line(x, y, w=None):
    """(slope, intercept) of the line minimising sum_i w_i (y_i - a - b x_i)^2.

    numpy.polyfit applies its weights to the unsquared residuals, so the
    square roots of the weights are passed. Without valid weights the fit is
    unweighted.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if w is not None:
        w = np.asarray(w, float)
        if np.all(np.isfinite(w)) and np.all(w >= 0) and np.max(w) > 0:
            s, b = np.polyfit(x, y, 1, w=np.sqrt(w))
            return float(s), float(b)
    s, b = np.polyfit(x, y, 1)
    return float(s), float(b)


def _exponent(ns, fs, ws) -> float:
    """Effective exponent -1/slope of a weighted log-log fit; inf when the
    floor does not decrease, nan with fewer than two points."""
    if len(ns) < 2:
        return float("nan")
    s, _ = _wls_line(np.log(ns), np.log(fs), ws)
    return float(-1.0 / s) if s < 0 else float("inf")


@dataclass
class FloorLaw:
    """Fitted law floor(n) = C * n^(-1/d_eff), with the sweep it came from."""
    C: float
    d_eff: float
    n_grid: np.ndarray
    floors: np.ndarray
    stds: np.ndarray
    metric: str
    r2: float
    warnings: List[str] = field(default_factory=list)
    d_eff_low: float = float("nan")
    d_eff_high: float = float("nan")
    drift_p_value: float = float("nan")
    n_rep: int = 0
    se_log: Optional[np.ndarray] = None

    @property
    def exponent_drift(self) -> float:
        """Ratio of the larger to the smaller half-grid exponent.

        Descriptive; the calibrated statistic is `drift_p_value`.
        """
        a, b = self.d_eff_low, self.d_eff_high
        if not (np.isfinite(a) and np.isfinite(b)) or min(a, b) <= 0:
            return float("nan")
        return float(max(a, b) / min(a, b))

    @property
    def single_exponent_ok(self) -> bool:
        """Admissibility: R^2 > 0.95 and a curvature p-value of at least 0.05.

        A missing p-value counts as a failure. R^2 alone has almost no power
        against curvature on a short log-log grid, which is why the curvature
        test carries the decision.
        """
        p = self.drift_p_value
        return bool(self.r2 > 0.95 and np.isfinite(p) and p >= 0.05)

    def required_n_range(self, target_floor: float) -> dict:
        """Invert the floor under the low-half, high-half and full fits.

        The half fits use the same weighted least squares as the full fit.
        Their disagreement shows how strongly a prescription depends on the
        part of the grid it extrapolates.

        Returns n_low_fit, n_high_fit, n_full_fit, the ratio of the two half
        prescriptions, the admissibility flag `ok`, the half exponents and the
        target.
        """
        if target_floor <= 0:
            raise ValueError("target_floor must be positive")
        lo_idx, hi_idx = _half_slices(len(self.n_grid))

        def _fit_Cd(ns, fs, ws):
            ns = np.asarray(ns, float)
            fs = np.asarray(fs, float)
            if len(ns) < 2 or np.any(fs <= 0):
                return float("nan"), float("nan")
            slope, intercept = _wls_line(np.log(ns), np.log(fs), ws)
            if slope >= 0:
                return float("nan"), float("nan")
            return float(np.exp(intercept)), float(-1.0 / slope)

        def _invert(C, d):
            if not (np.isfinite(C) and np.isfinite(d) and d > 0 and C > 0):
                return float("nan")
            return float((C / target_floor) ** d)

        wts = (self.floors / np.maximum(self.stds, 1e-15)) ** 2
        C_lo, d_lo = _fit_Cd(self.n_grid[lo_idx], self.floors[lo_idx],
                             wts[lo_idx])
        C_hi, d_hi = _fit_Cd(self.n_grid[hi_idx], self.floors[hi_idx],
                             wts[hi_idx])
        n_low = _invert(C_lo, d_lo)
        n_high = _invert(C_hi, d_hi)
        n_full = _invert(self.C, self.d_eff)
        finite = [x for x in (n_low, n_high) if np.isfinite(x) and x > 0]
        ratio = (float(max(finite) / min(finite)) if len(finite) == 2
                 else float("nan"))
        return {
            "n_low_fit": n_low,
            "n_high_fit": n_high,
            "n_full_fit": n_full,
            "ratio": ratio,
            "ok": bool(self.single_exponent_ok),
            "d_eff_low": float(self.d_eff_low),
            "d_eff_high": float(self.d_eff_high),
            "target_floor": float(target_floor),
        }

    def required_n_interval(self, target_floor: float, level: float = 0.90,
                            n_mc: int = 4000, seed: int = 0,
                            calibrate: bool = False) -> dict:
        """Interval for the sample size at which the floor reaches a target.

        The weighted least-squares fit of log f on log n gives the covariance
        of (slope, intercept). Coefficient pairs are drawn from the
        multivariate t distribution with k - 2 degrees of freedom centred at
        the estimate, mapped to (C, d) = (e^intercept, -1/slope), and the
        prescription n = (C/t)^d is evaluated for each draw; the interval is
        a pair of quantiles. Draws with a non-negative slope are discarded and
        their fraction is reported.

        On known power laws the construction delivers, at nominal level
        0.90, the coverage recorded in INTERVAL_COVERAGE_AT_090. With
        `calibrate=True`,
        `level` is read as the requested delivered coverage and the nominal
        level is taken from INTERVAL_LEVEL_CALIBRATION; outside the regime in
        which that map was measured the calibration is not applied and
        `calibration_refused` says why.

        Returns n_point, n_lo, n_med, n_hi, d_lo, d_hi, discard_fraction, ok,
        target_floor, n_mc, level, calibrated, calibration_refused,
        requested_level, applied_level, in_calibration_regime,
        calibration_regime, observed_regime and delivered_coverage.
        """
        if target_floor <= 0:
            raise ValueError("target_floor must be positive")
        logn = np.log(np.asarray(self.n_grid, float))
        logf = np.log(np.asarray(self.floors, float))
        w = (np.asarray(self.floors, float)
             / np.maximum(np.asarray(self.stds, float), 1e-15)) ** 2
        if not (np.all(np.isfinite(w)) and np.max(w) > 0 and len(logn) >= 3):
            return {"n_point": float("nan"), "ok": False,
                    "note": "degenerate weights; interval undefined"}
        # weighted least-squares covariance sigma^2 (X'WX)^-1, dof-corrected
        X = np.column_stack([logn, np.ones_like(logn)])
        WX = X * w[:, None]
        XtWX = X.T @ WX
        resid = logf - X @ np.linalg.solve(XtWX, WX.T @ logf)
        dof = max(len(logn) - 2, 1)
        sigma2 = float((w * resid ** 2).sum() / dof)
        cov = sigma2 * np.linalg.inv(XtWX)
        beta = np.array([-1.0 / self.d_eff, np.log(self.C)])
        # multivariate t with nu = k - 2: beta + chol(cov) z / sqrt(u / nu)
        rng = np.random.default_rng(seed + 77)
        nu = max(len(logn) - 2, 1)
        L = np.linalg.cholesky(cov)
        z = rng.normal(size=(n_mc, 2))
        u = rng.chisquare(nu, size=n_mc)
        draws = beta + (z @ L.T) / np.sqrt(u / nu)[:, None]
        slopes, intercepts = draws[:, 0], draws[:, 1]
        ok_mask = slopes < 0
        d_draws = -1.0 / slopes[ok_mask]
        C_draws = np.exp(intercepts[ok_mask])
        with np.errstate(over="ignore", invalid="ignore"):
            # far extrapolations overflow to inf, which lands in the upper tail
            n_draws = np.power(C_draws / target_floor, d_draws)
        n_draws = np.where(np.isfinite(n_draws), n_draws, np.inf)

        level_requested = float(level)
        cvs = (np.asarray(self.stds, float)
               / np.maximum(np.asarray(self.floors, float), 1e-15))
        cv_med = float(np.median(cvs)) if cvs.size else float("nan")
        in_regime = bool(
            len(self.n_grid) == 6
            and 8 <= int(self.n_rep) <= 24
            and 3.5 <= float(self.d_eff) <= 7.0
            and 0.02 <= cv_med <= 0.16)
        refused = None
        if calibrate and not in_regime:
            calibrate = False
            refused = ("law outside the measured calibration regime ("
                       + _CALIBRATION_REGIME + "); interval returned "
                       "uncalibrated at the requested nominal level")
        if calibrate:
            xs = sorted(INTERVAL_LEVEL_CALIBRATION)
            ys = [INTERVAL_LEVEL_CALIBRATION[k] for k in xs]
            level = float(np.clip(np.interp(level_requested, xs, ys),
                                  0.5, 0.999))
        q = (1.0 - level) / 2.0
        return {
            "n_point": required_n(self, target_floor),
            "n_lo": float(np.quantile(n_draws, q)),
            "n_med": float(np.quantile(n_draws, 0.5)),
            "n_hi": float(np.quantile(n_draws, 1.0 - q)),
            "d_lo": float(np.quantile(d_draws, q)),
            "d_hi": float(np.quantile(d_draws, 1.0 - q)),
            "discard_fraction": float(1.0 - ok_mask.mean()),
            "ok": bool(self.single_exponent_ok),
            "target_floor": float(target_floor),
            "n_mc": int(n_mc), "level": float(level),
            "calibrated": bool(calibrate),
            "calibration_refused": refused,
            "requested_level": level_requested,
            "applied_level": float(level),
            "in_calibration_regime": in_regime,
            "calibration_regime": _CALIBRATION_REGIME,
            "observed_regime": {"n_points": int(len(self.n_grid)),
                                "n_rep": int(self.n_rep),
                                "d_eff": float(self.d_eff),
                                "cv_median": cv_med},
            "delivered_coverage_at_nominal_090": dict(INTERVAL_COVERAGE_AT_090),
        }

    def severity(self, drift_boot: int = 400, trials: int = 200,
                 seed: int = 0) -> dict:
        """Which departures from a single exponent the curvature test detects.

        The alternative for a drift factor delta is a piecewise power law,
        continuous at the geometric midpoint of the grid, with exponents
        d_eff * sqrt(delta) on the lower half and d_eff / sqrt(delta) on the
        upper half, perturbed by the measured noise of the profile. The
        rejection rate is estimated on the grid delta in {1.1, 1.25, 1.5, 2, 3}
        and the thresholds `drift_50` and `drift_80`, at which it reaches 1/2
        and 0.8, are interpolated linearly in log delta. `drift_50_ci90` is
        the binomial Monte-Carlo interval at this seed; `severity_panel`
        adds the between-seed component.

        Returns drift_observed, drift_grid, rejection_rate,
        rejection_rate_mcse, monotone, drift_50, drift_80, drift_50_ci90,
        drift_80_ci90, power_at_observed and n_grid_points.
        """
        if not np.isfinite(self.drift_p_value):
            return {"note": "no curvature p-value; severity undefined"}
        rng = np.random.default_rng(seed + 5150)
        k = len(self.n_grid)
        lo_idx, hi_idx = _half_slices(k)
        logn = np.log(self.n_grid)
        w = (self.floors / np.maximum(self.stds, 1e-15)) ** 2
        cv = self.stds / np.maximum(self.floors, 1e-15)
        se_log = (np.asarray(self.se_log, float)
                  if getattr(self, "se_log", None) is not None
                  and np.all(np.isfinite(np.asarray(self.se_log, float)))
                  else np.sqrt(np.pi / 2.0) * cv / np.sqrt(max(self.n_rep, 1)))
        base = np.log(self.C) - logn / self.d_eff

        def _drift(f):
            a = _exponent(self.n_grid[lo_idx], f[lo_idx], w[lo_idx])
            b = _exponent(self.n_grid[hi_idx], f[hi_idx], w[hi_idx])
            if not (np.isfinite(a) and np.isfinite(b) and min(a, b) > 0):
                return float("nan")
            return float(max(a, b) / min(a, b))

        # null: drifts of profiles drawn from the fitted single law, drawn once
        null_drift = np.empty(drift_boot)
        for t in range(drift_boot):
            fb = np.exp(base + rng.normal(0.0, se_log))
            null_drift[t] = _drift(fb)
        null_drift = null_drift[np.isfinite(null_drift)]

        def _alt_floors(true_drift):
            d_lo = self.d_eff * np.sqrt(true_drift)
            d_hi = self.d_eff / np.sqrt(true_drift)
            n_mid = float(np.exp(logn.mean()))
            f_mid = float(self.C * np.power(n_mid, -1.0 / self.d_eff))
            return np.where(
                self.n_grid <= n_mid,
                f_mid * np.power(self.n_grid / n_mid, -1.0 / d_lo),
                f_mid * np.power(self.n_grid / n_mid, -1.0 / d_hi))

        def _reject_rate(true_drift):
            f_alt = _alt_floors(true_drift)
            rej = 0
            for _ in range(trials):
                f = np.exp(np.log(f_alt) + rng.normal(0.0, se_log))
                if np.any(f <= 0):
                    continue
                obs = _drift(f)
                if not np.isfinite(obs):
                    continue
                p = (np.sum(null_drift >= obs) + 1) / (len(null_drift) + 1)
                rej += int(p < 0.05)
            return rej / max(trials, 1)

        grid = [1.1, 1.25, 1.5, 2.0, 3.0]
        rates = [_reject_rate(g) for g in grid]
        mcse = float(np.sqrt(0.25 / max(trials, 1)))
        viol = sum(1 for i in range(len(rates) - 1)
                   if rates[i + 1] < rates[i] - 3 * mcse)

        def _cross(target, rr):
            rr = np.asarray(rr, float)
            lg = np.log(np.asarray(grid, float))
            if rr[-1] < target:
                return float("nan")          # not reached on this grid
            if rr[0] >= target:
                return float(grid[0])
            j = int(np.argmax(rr >= target))
            r0, r1 = rr[j - 1], rr[j]
            if r1 <= r0:
                return float(grid[j])
            f = (target - r0) / (r1 - r0)
            return float(np.exp(lg[j - 1] + f * (lg[j] - lg[j - 1])))

        # binomial interval: perturb each rate by its standard error
        rng_ci = np.random.default_rng(seed + 991)
        se_vec = np.sqrt(np.clip(np.asarray(rates, float), 0, 1)
                         * (1 - np.clip(np.asarray(rates, float), 0, 1))
                         / max(trials, 1))
        boot50, boot80 = [], []
        for _ in range(600):
            pert = np.clip(np.asarray(rates, float)
                           + rng_ci.normal(0.0, np.maximum(se_vec, 1e-9)), 0, 1)
            pert = np.maximum.accumulate(pert)   # power curves are monotone
            boot50.append(_cross(0.5, pert))
            boot80.append(_cross(0.8, pert))

        def _ci(v):
            v = np.asarray(v, float)
            f = v[np.isfinite(v)]
            if f.size < 30:
                return [float("nan"), float("nan")]
            return [float(np.quantile(f, 0.05)), float(np.quantile(f, 0.95))]

        return {"drift_observed": float(self.exponent_drift),
                "drift_grid": grid, "rejection_rate": rates,
                "rejection_rate_mcse": mcse, "monotone": bool(viol == 0),
                "drift_50": _cross(0.5, rates), "drift_80": _cross(0.8, rates),
                "drift_50_ci90": _ci(boot50), "drift_80_ci90": _ci(boot80),
                "power_at_observed": float(np.interp(
                    self.exponent_drift, grid, rates))
                if np.isfinite(self.exponent_drift) else float("nan"),
                "n_grid_points": int(k)}

    def severity_panel(self, seeds: Sequence[int] = tuple(range(8)),
                       drift_boot: int = 600, trials: int = 200) -> dict:
        """Severity threshold with its combined Monte-Carlo interval.

        Runs `severity(drift_boot, trials, seed=s)` for every seed and
        combines the per-seed thresholds with `combine_severity`, so that the
        interval contains both the binomial and the between-seed component.

        Returns drift_observed, drift_50 (median over seeds), drift_50_ci90
        (combined), drift_50_ci90_within (binomial only), sd_within,
        sd_between, sd_total, inflation, per-seed values, observed_inside_ci,
        n_seeds and settings.
        """
        ts, cis, obs = [], [], float("nan")
        for sd in seeds:
            sv = self.severity(drift_boot=drift_boot, trials=trials, seed=sd)
            ts.append(sv.get("drift_50", float("nan")))
            cis.append(sv.get("drift_50_ci90"))
            obs = sv.get("drift_observed", obs)
        comb = combine_severity(ts, cis)
        inside = bool(np.isfinite(comb["lo"]) and np.isfinite(obs)
                      and comb["lo"] <= obs <= comb["hi"])
        return {"drift_observed": float(obs),
                "drift_50": comb["median"],
                "drift_50_ci90": [comb["lo"], comb["hi"]],
                "drift_50_ci90_within": [comb["median"] - _Z90 * comb["sd_within"],
                                         comb["median"] + _Z90 * comb["sd_within"]],
                "sd_within": comb["sd_within"], "sd_between": comb["sd_between"],
                "sd_total": comb["sd_total"], "inflation": comb["inflation"],
                "per_seed_drift_50": [float(x) for x in ts],
                "per_seed_ci90": cis, "observed_inside_ci": inside,
                "n_seeds": comb["n_seeds"],
                "settings": {"seeds": list(seeds), "drift_boot": drift_boot,
                             "trials": trials}}

    def floor_at(self, n: float) -> float:
        return float(self.C * np.power(n, -1.0 / self.d_eff))

    def std_at(self, n: float) -> float:
        """Floor standard deviation at n, at the median coefficient of
        variation of the fitted grid."""
        cv = self.stds / np.maximum(self.floors, 1e-15)
        return float(self.floor_at(n) * np.median(cv))

    def summary(self) -> str:
        L = [f"floor law [{self.metric}]:  floor(n) = {self.C:.4f} * "
             f"n^(-1/{self.d_eff:.2f})   R^2={self.r2:.4f}",
             f"  fitted on n = {[int(x) for x in self.n_grid]}",
             f"  half-grid exponents: {self.d_eff_low:.2f} (low n), "
             f"{self.d_eff_high:.2f} (high n); drift {self.exponent_drift:.2f}x, "
             f"curvature p = {self.drift_p_value:.3f}; "
             f"admissible = {self.single_exponent_ok}"]
        for w in self.warnings:
            L.append(f"  [!] {w}")
        return "\n".join(L)


def fit_floor_law(X: Optional[np.ndarray] = None,
                  n_grid: Optional[Sequence[int]] = None,
                  metric: str = "w2", n_rep: int = 20, seed: int = 0,
                  drift_boot: int = 400, _replicates=None, **kw) -> FloorLaw:
    """Fit floor(n) = C n^(-1/d_eff) from repeated disjoint splits of X.

    The default grid has six sizes N/64, N/32, ..., N/2. At each size the
    floor is the median of `n_rep` replicates; the law is fitted by weighted
    least squares on log-log axes with weights (floor / sd)^2. The curvature
    test compares the ratio of the exponents fitted to the two halves of the
    grid with its distribution under the fitted law, simulated with the
    bootstrap standard error of each log-median.

    `_replicates`, a dict {n: replicate values}, replaces the sweep on X and
    runs every later step unchanged. It exists to measure the size and power
    of the test through this function, and is not meant for real data.
    """
    if _replicates is None and X is None:
        raise ValueError("either X or _replicates must be provided")
    if _replicates is not None:
        X = None
    X = None if X is None else np.asarray(X, float)
    N = len(X) if X is not None else 2 * max(int(k) for k in _replicates)

    if not FLOOR_IS_POSITIVE.get(metric, True):
        raise ValueError(
            f"metric={metric!r} has no resolution floor: it is an unbiased "
            "statistic whose same-law expectation is zero, so log(floor) is "
            "undefined. For such statistics the design quantity is the null "
            "variance (a classical two-sample power analysis). Use metric='w2'.")
    if not DIMENSION_GOVERNED_RATE.get(metric, False):
        warnings.warn(
            f"metric={metric!r} does not have a dimension-governed floor rate; "
            "d_eff is a descriptive exponent without a dimension "
            "interpretation.", RuntimeWarning, stacklevel=2)

    if n_grid is None:
        top = N // 2
        n_grid = [max(16, int(top / f)) for f in (32, 16, 8, 4, 2, 1)]
        n_grid = sorted(set(n for n in n_grid if 2 * n <= N))
    n_grid = np.asarray(sorted(set(int(n) for n in n_grid)), dtype=float)
    if len(n_grid) < 3:
        raise ValueError(f"need >=3 distinct sizes with 2n<=N={N}; got {n_grid}")

    floors, stds, se_boot = [], [], []
    _srng = np.random.default_rng(seed + 13)
    for n in n_grid:
        if _replicates is not None and int(n) in _replicates:
            fr = FloorReport(values=np.asarray(_replicates[int(n)], float),
                             n_per_side=int(n), metric=metric,
                             n_rep=len(_replicates[int(n)]))
        elif _replicates is not None:
            raise KeyError(
                f"_replicates has no entry for grid point n={int(n)}; "
                f"provided: {sorted(int(k) for k in _replicates)}")
        else:
            fr = resolution_floor(X, int(n), metric=metric, n_rep=n_rep,
                                  seed=seed, **kw)
        floors.append(fr.median)
        stds.append(max(fr.std, 1e-12))
        # standard error of the log-median by nonparametric bootstrap
        v = np.asarray(fr.values, float)
        if len(v) >= 3 and fr.median > 0:
            bidx = _srng.integers(0, len(v), size=(2000, len(v)))
            bmed = np.median(v[bidx], axis=1)
            se_boot.append(max(float(np.std(np.log(np.maximum(bmed, 1e-300)),
                                            ddof=1)), 1e-12))
        else:
            se_boot.append(float("nan"))
    return floor_law_from_profile(
        n_grid, np.asarray(floors, float), np.asarray(stds, float),
        se_log=np.asarray(se_boot, float), n_rep=n_rep, metric=metric,
        seed=seed, drift_boot=drift_boot,
        n_obs=len(X) if X is not None else None)


def floor_law_from_profile(n_grid: Sequence[float], floors: Sequence[float],
                           stds: Sequence[float],
                           se_log: Optional[Sequence[float]] = None,
                           n_rep: int = 20, metric: str = "w2", seed: int = 0,
                           drift_boot: int = 400,
                           n_obs: Optional[int] = None) -> FloorLaw:
    """Fit the floor law to a measured profile and run the curvature test.

    n_grid, floors and stds are the grid of per-side sizes, the floor (median
    of the replicates) and the standard deviation of the replicates at each
    size; se_log is the standard error of each log-median (by default
    sqrt(pi/2) CV / sqrt(n_rep), the large-sample value for a median).

    The law is fitted by weighted least squares of log f on log n with
    weights (f / s)^2, the inverse variances of log f up to a constant. The
    exponents of the lower and upper halves of the grid are fitted with the
    same weights, and the curvature p-value compares the ratio of the larger
    to the smaller with its distribution over `drift_boot` profiles drawn
    from the fitted law with independent normal errors of standard deviation
    se_log on the log scale. fit_floor_law measures the profile on data and
    calls this function; it is also the way to rebuild the law of a stored
    profile.
    """
    n_grid = np.asarray(n_grid, float)
    floors = np.asarray(floors, float)
    stds = np.maximum(np.asarray(stds, float), 1e-12)
    if np.any(floors <= 0):
        bad = [int(n) for n, f in zip(n_grid, floors) if f <= 0]
        raise ValueError(
            f"the measured floor is non-positive at n = {bad} with "
            f"metric={metric!r}; log(floor) is undefined. Check for duplicated "
            "rows or a degenerate column.")
    cv_grid = stds / floors
    default_se = np.sqrt(np.pi / 2.0) * cv_grid / np.sqrt(max(int(n_rep), 1))
    if se_log is None:
        se_log = default_se
    else:
        se_log = np.asarray(se_log, float)
        se_log = np.where(np.isfinite(se_log), se_log, default_se)

    # weighted least squares on log-log axes, weights (floor / sd)^2
    logn, logf = np.log(n_grid), np.log(floors)
    w = (floors / stds) ** 2
    slope, intercept = _wls_line(logn, logf, w)
    pred = intercept + slope * logn
    ybar = float(np.average(logf, weights=w))
    ss_res = float((w * (logf - pred) ** 2).sum())
    ss_tot = float((w * (logf - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lo_idx, hi_idx = _half_slices(len(n_grid))
    d_low = _exponent(n_grid[lo_idx], floors[lo_idx], w[lo_idx])
    d_high = _exponent(n_grid[hi_idx], floors[hi_idx], w[hi_idx])
    drift = (max(d_low, d_high) / min(d_low, d_high)
             if np.isfinite(d_low) and np.isfinite(d_high)
             and min(d_low, d_high) > 0 else float("nan"))

    # curvature test: drifts of profiles simulated from the fitted law with
    # the standard error of each log-median
    drift_p = float("nan")
    if np.isfinite(drift) and np.all(np.isfinite(se_log)) and np.all(se_log > 0):
        brng = np.random.default_rng(seed + 991)
        null = np.empty(drift_boot)
        for t in range(drift_boot):
            fb = np.exp(pred + brng.normal(0.0, se_log))
            a = _exponent(n_grid[lo_idx], fb[lo_idx], w[lo_idx])
            b = _exponent(n_grid[hi_idx], fb[hi_idx], w[hi_idx])
            null[t] = (max(a, b) / min(a, b)
                       if np.isfinite(a) and np.isfinite(b) and min(a, b) > 0
                       else np.inf)
        drift_p = float((np.sum(null >= drift) + 1) / (drift_boot + 1))

    warn: List[str] = []
    if not np.isfinite(drift_p):
        warn.append("the curvature test produced no p-value (undefined drift "
                    "or degenerate standard errors); the profile is not "
                    "admissible.")
    if np.isfinite(drift_p) and drift_p < 0.05:
        warn.append(
            f"the half-grid exponents differ by {drift:.2f}x ({d_low:.2f} at "
            f"low n, {d_high:.2f} at high n; curvature p = {drift_p:.3f}): a "
            "single power law does not describe this floor. Report the "
            "half-fit range instead of a prescription.")
    cv_med = float(np.median(cv_grid))
    if cv_med > 0 and (float(cv_grid.max()) / max(float(cv_grid.min()), 1e-15) > 2.0
                       or (float(np.std(cv_grid, ddof=1)) if len(cv_grid) > 1
                           else 0.0) / cv_med > 0.25):
        warn.append(
            f"the floor's coefficient of variation varies across the grid "
            f"(min {cv_grid.min():.3g}, max {cv_grid.max():.3g}, median "
            f"{cv_med:.3g}); std_at assumes it is constant.")
    if slope >= 0:
        warn.append("the floor does not decrease with n; the fit is "
                    "meaningless. Check for duplicated rows or a degenerate "
                    "column.")
        d_eff = float("inf")
    else:
        d_eff = float(-1.0 / slope)
    if r2 < 0.95:
        warn.append(f"R^2 = {r2:.3f} < 0.95: a single power law does not "
                    "describe this floor.")
    if np.isfinite(d_eff) and d_eff < RATE_REGIME_MIN_D:
        warn.append(
            f"d_eff = {d_eff:.2f} < 4: the decay is in the crossover between "
            "the dimension-governed and the parametric rate, where the rate "
            "theory gives no pure power law. d_eff is descriptive here and no "
            "prescription is returned.")
    elif np.isfinite(d_eff) and d_eff < 6.0:
        warn.append(
            f"d_eff = {d_eff:.2f} is close to the rate-regime boundary 4; mild "
            "curvature may reflect the approach to that boundary rather than "
            "multi-scale geometry.")
    if np.isfinite(drift_p) and drift_p >= 0.05:
        warn.append(
            f"a single exponent is not rejected (p = {drift_p:.3f}). The power "
            "of the test depends on the precision of the floor (between "
            f"{GATE_POWER_GRID_6PT[0.05]:.3f} and {GATE_POWER_GRID_6PT[0.01]:.2f} "
            "at median standard errors of 5% and 1%); FloorLaw.severity() "
            "states which departures this non-rejection excludes.")
    if n_obs is not None and n_obs < 2048:
        warn.append(f"N={n_obs}: the grid is short and d_eff is poorly determined.")
    return FloorLaw(float(np.exp(intercept)), d_eff, n_grid, floors, stds,
                    metric, r2, warn, d_eff_low=d_low, d_eff_high=d_high,
                    drift_p_value=drift_p, n_rep=int(n_rep),
                    se_log=np.asarray(se_log, float))


def _sizing_admissible(law: FloorLaw) -> bool:
    """A single exponent is admitted and lies in the dimension-governed regime."""
    return bool(law.single_exponent_ok and np.isfinite(law.d_eff)
                and law.d_eff >= RATE_REGIME_MIN_D)


def required_n(law: FloorLaw, target_floor: float) -> float:
    """Per-side size at which the floor reaches `target_floor`.

    Inverts floor(n) = C n^(-1/d_eff). Returns nan when the profile is not
    admissible or d_eff < 4; use `law.required_n_range` there. The fitted
    exponent is not corrected for the bias of the slope estimator, and the
    prescription grows like (C / target)^d_eff.
    """
    if target_floor <= 0:
        raise ValueError("target_floor must be positive")
    if not _sizing_admissible(law):
        return float("nan")
    return float((law.C / target_floor) ** law.d_eff)


def required_n_sensitivity(law: FloorLaw, target_floor: float,
                           rel_d_err: float = 0.05) -> dict:
    """Range of `required_n` under a relative exponent error of `rel_d_err`.

    A relative error eps in d_eff multiplies the prescription by
    (C / target)^(+-eps * d_eff).
    """
    n0 = required_n(law, target_floor)
    if not np.isfinite(n0):
        return {"n": float("nan"), "note": "single exponent not identified"}
    lo = float((law.C / target_floor) ** (law.d_eff * (1 - rel_d_err)))
    hi = float((law.C / target_floor) ** (law.d_eff * (1 + rel_d_err)))
    return {"n": n0, "n_low": lo, "n_high": hi,
            "ratio": float(hi / lo) if lo > 0 else float("nan"),
            "rel_d_err": rel_d_err, "d_eff": float(law.d_eff),
            "C_over_target": float(law.C / target_floor)}


def resolvable_effect(law: FloorLaw, n: int, alpha: float = 0.05,
                      power: float = 0.80,
                      reference_level: Optional[float] = None) -> float:
    """Smallest change in expected distance resolvable at size n.

        Delta(n) = (z_{1-alpha/2} + z_{power}) * sqrt(2) * s(n),

    with s(n) the floor's standard deviation extrapolated on the fitted law
    at the median coefficient of variation of the grid. With
    `reference_level`, the coefficient of variation is applied at that
    distance level instead, which is appropriate for models that sit well
    above the floor. Returns nan when the profile is not admissible or
    d_eff < 4, as for `required_n`.
    """
    if not _sizing_admissible(law):
        return float("nan")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    f = law.floor_at(n)
    sd = law.std_at(n)
    if reference_level is not None:
        lv = float(reference_level)
        if not np.isfinite(lv) or lv <= 0:
            raise ValueError("reference_level must be a positive discrepancy")
        cv = sd / f if f > 0 else float("nan")
        sd = cv * lv
    from scipy.stats import norm
    mult = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    return float(mult * np.sqrt(2.0) * sd)


def power_curve(law: FloorLaw, effect_size: float,
                n_grid: Optional[Sequence[int]] = None, alpha: float = 0.05,
                power: float = 0.80,
                reference_level: Optional[float] = None) -> dict:
    """Resolvable effect across sample sizes, and the smallest n at which
    `effect_size` becomes resolvable. All entries are nan when the profile is
    not admissible or d_eff < 4."""
    if n_grid is None:
        n_grid = np.unique(np.round(np.logspace(2, 6, 40)).astype(int))
    n_grid = np.asarray(n_grid, dtype=float)
    if not _sizing_admissible(law):
        res = np.full(len(n_grid), np.nan)
        return {"n_grid": n_grid, "resolvable": res, "effect_size": effect_size,
                "n_required": float("nan"),
                "single_exponent_ok": False}
    res = np.array([resolvable_effect(law, int(n), alpha, power,
                                      reference_level=reference_level)
                    for n in n_grid])
    ok = np.where(res <= effect_size)[0]
    return {"n_grid": n_grid, "resolvable": res, "effect_size": effect_size,
            "n_required": float(n_grid[ok[0]]) if len(ok) else float("nan"),
            "single_exponent_ok": True}
