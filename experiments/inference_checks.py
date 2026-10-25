"""Three measurements behind inferential statements of the paper.

1. Shared random numbers across the grid (Section 4.6). fit_floor_law uses
   the same seed at every grid size, so the subsamples at different n are
   nested. On a five-dimensional Gaussian table (4000 rows, grid
   {125, 250, 500, 1000}, n_rep = 8, 30 seeds) this measures the correlation
   of the log-floor medians across grid sizes and the variance of the drift
   statistic, under shared and under independent seeds.
2. Power of the two-standard-deviation rule (Section 5.4). A difference is
   declared when the observed gap exceeds 2 x the floor proxy of the paired
   standard deviation; the rule is applied at a true gap of exactly that size
   on a six-dimensional copula (n = 500 per side, 400 replicates).
3. Coverage of the exponent interval (Section 9). 90% intervals for a mean
   of eight replicates with the normal and with the Student-t quantile,
   4000 trials.

    python experiments/inference_checks.py

Writes results/inference_checks.json (about 2 minutes).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats

import _artifacts as art
import certgen as cg
from certgen.discrepancy import w2_exact

warnings.filterwarnings("ignore")

N_ROWS, D, GRID, N_REP, N_PANEL = 4000, 5, (125, 250, 500, 1000), 8, 30
SEEDS = list(range(N_PANEL))


def _median_floors(X, base_seed, shared: bool):
    """Log-floor medians at each grid size, with the same seed at every size
    (shared) or a different one (independent)."""
    out = []
    for k, n in enumerate(GRID):
        s = base_seed if shared else base_seed + 7919 * k
        fr = cg.resolution_floor(X, n, metric="w2", n_rep=N_REP, seed=s)
        out.append(float(np.log(fr.median)))
    return np.asarray(out)


def _half_slopes(logf):
    logn = np.log(np.asarray(GRID, float))
    d = []
    for idx in ([0, 1], [2, 3]):
        s = np.polyfit(logn[idx], logf[idx], 1)[0]
        d.append(-1.0 / s if s < 0 else np.nan)
    return d


def shared_random_numbers():
    X = cg.Standardizer().fit_transform(
        np.random.default_rng(2024).normal(size=(N_ROWS, D)))
    crn = np.asarray([_median_floors(X, s, True) for s in SEEDS])
    ind = np.asarray([_median_floors(X, s, False) for s in SEEDS])
    corr_crn = np.corrcoef(crn.T)
    off_crn = corr_crn[np.triu_indices(len(GRID), k=1)]
    off_ind = np.corrcoef(ind.T)[np.triu_indices(len(GRID), k=1)]

    def _drift_var(panels):
        dr = []
        for lf in panels:
            a, b = _half_slopes(lf)
            if np.isfinite(a) and np.isfinite(b) and min(a, b) > 0:
                dr.append(max(a, b) / min(a, b))
        dr = np.asarray(dr)
        return float(np.var(dr, ddof=1)), float(np.mean(dr))

    var_crn, mean_crn = _drift_var(crn)
    var_ind, mean_ind = _drift_var(ind)
    return {
        "logfloor_median_correlation_crn": {
            "matrix": [[round(float(v), 3) for v in row] for row in corr_crn],
            "offdiag_mean": float(off_crn.mean()),
            "offdiag_min": float(off_crn.min()),
            "offdiag_max": float(off_crn.max())},
        "logfloor_median_correlation_independent": {
            "offdiag_mean": float(off_ind.mean()),
            "offdiag_min": float(off_ind.min()),
            "offdiag_max": float(off_ind.max())},
        "drift_statistic": {
            "var_crn": var_crn, "mean_crn": mean_crn,
            "var_independent": var_ind, "mean_independent": mean_ind,
            "var_ratio_crn_over_independent": (
                float(var_crn / var_ind) if var_ind > 0 else float("nan"))}}


def power_two_sd_rule():
    N_ROWS2, D2, N_SIDE, R2 = 6000, 6, 500, 400
    raw = cg.data.gaussian_copula(N_ROWS2, D2, seed=21)
    st = cg.Standardizer().fit(raw)
    Xs = st.transform(raw)
    fl = cg.resolution_floor(Xs, N_SIDE, "w2", n_rep=30, seed=0)
    level = 0.8038                          # mean distance of fresh draws
    gap = 2.0 * fl.sd_difference_at(level)  # two floor-proxy standard deviations

    def _fresh(gg, k):
        return st.transform(cg.data.gaussian_copula(
            k, D2, seed=int(gg.integers(1, 10 ** 6))))
    declare = 0
    for r in range(R2):
        gg = np.random.default_rng(900 + r)
        Xr = Xs[gg.permutation(len(Xs))[:N_SIDE]]
        dA = w2_exact(Xr, _fresh(gg, N_SIDE), seed=r)
        dB = w2_exact(Xr, _fresh(np.random.default_rng(4000 + r), N_SIDE),
                      seed=r)
        declare += int(abs(dA - dB + gap) > gap)
    return {"setting": {"law": "gaussian_copula", "d": D2, "N": N_ROWS2,
                        "n_per_side": N_SIDE, "R": R2, "level": level},
            "measured_power": float(declare / R2),
            "analytic": 0.5,
            "mcse": float(np.sqrt(0.25 / R2))}


def exponent_interval_coverage():
    true_ex, sd, NB, TRIALS = 1.0 / 5.0, 0.5, 8, 4000
    z_q, t_q = 1.645, stats.t.ppf(0.95, NB - 1)
    cov_z = cov_t = 0
    rng = np.random.default_rng(11)
    for _ in range(TRIALS):
        reps = true_ex + rng.normal(0.0, sd, NB)
        ex, se = reps.mean(), reps.std(ddof=1) / np.sqrt(NB)
        if se > 0:
            cov_z += int(abs(ex - true_ex) <= z_q * se)
            cov_t += int(abs(ex - true_ex) <= t_q * se)
    return {"n_replicates": NB, "z_quantile": z_q, "t_quantile": float(t_q),
            "simulated_cover_z": float(cov_z / TRIALS),
            "simulated_cover_t": float(cov_t / TRIALS)}


def main():
    t0 = time.time()
    out = {"_definition": ("shared random numbers across the floor grid, the "
                           "power of the two-standard-deviation rule, and "
                           "the coverage of the exponent interval"),
           "_regime": {"rows": N_ROWS, "d": D, "grid": list(GRID),
                       "n_rep": N_REP, "panel_seeds": N_PANEL}}
    out["crn_vs_independent"] = shared_random_numbers()
    out["power_two_sd_rule"] = power_two_sd_rule()
    out["exponent_ci_quantile"] = exponent_interval_coverage()
    out["elapsed_s"] = round(time.time() - t0, 1)
    v = out["crn_vs_independent"]["drift_statistic"]
    print(f"Var(drift) shared/independent = "
          f"{v['var_ratio_crn_over_independent']:.3f}")
    print(f"power of the 2-sd rule = {out['power_two_sd_rule']['measured_power']:.3f}")
    q = out["exponent_ci_quantile"]
    print(f"exponent interval coverage: z {q['simulated_cover_z']:.4f}, "
          f"t {q['simulated_cover_t']:.4f}")
    art.save("inference_checks.json", out, indent=2)


if __name__ == "__main__":
    main()
