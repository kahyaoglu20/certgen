"""Four stress tests of the procedures on synthetic laws.

memorisation (Section 5.2)
    Skill, tail p-values and leakage verdict of evaluate() for a generator
    that returns half of the training rows, an exact copy of the training
    data, and a fresh draw from the same six-dimensional copula.
goodness_of_fit_power (Section 4.2)
    Against the two-scale floor n^(-1/2) + 0.35 n^(-1/8), with 3% noise on
    log f: the probability that R^2 < 0.95 and the power of the calibrated
    curvature test on grids of 4, 6 and 8 points; on the single power law
    n^(-0.2), the size of the calibrated test and the rejection rate of a
    fixed threshold of 1.25 on the drift.
tied_observations (Section 9)
    The three dimension estimators on a six-dimensional copula quantised to
    50, 20 and 10 levels per column, with zero nearest-neighbour distances
    excluded (the library's rule) and, for comparison, replaced by 1e-12.
mdd_proxy (Section 5.4)
    The floor proxy of the paired standard deviation, at the floor level and
    rescaled to the level of the models, against the standard deviation of
    D_A - D_B measured over 200 replicates for three generators.

    python experiments/stress_tests.py

Writes results/stress_tests.json (about 2 minutes).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

import _artifacts as art
import certgen as cg
from certgen.discrepancy import w2_exact


def slope_d(ns, fs):
    s = np.polyfit(np.log(ns), np.log(fs), 1)[0]
    return float(-1.0 / s) if s < 0 else float("inf")


def memorisation():
    X = cg.data.gaussian_copula(1200, 6, seed=11)
    cases = {}
    for tag, gen in [("memoriser_subset", X[:600]),
                     ("exact_copy", X.copy()),
                     ("fresh_draw", cg.data.gaussian_copula(1200, 6, seed=777))]:
        r = cg.evaluate(X, gen, n_per_side=400, n_rep=10)
        cases[tag] = {"skill_T1": r.skill["T1_marginals"],
                      "skill_T2": r.skill["T2_gaussian"],
                      "p_hi": r.p_value, "p_lo": r.p_value_low,
                      "dup_fraction": r.duplicate_fraction,
                      "leakage_suspected": r.leakage_suspected,
                      "valid": r.valid}
        print(f"  {tag:<18} skill(T2)={r.skill['T2_gaussian']:+.3f} "
              f"p_lo={r.p_value_low:.3f} valid={r.valid}")
    return cases


SIG = 0.03                       # standard deviation of log f at each grid point


def _fit_and_test(gg, f0, rs, boot=250):
    """R^2, observed drift and calibrated curvature p-value of one profile."""
    f = f0 * np.exp(rs.normal(0, SIG, size=len(gg)))
    sl, b = np.polyfit(np.log(gg), np.log(f), 1)
    pred = b + sl * np.log(gg)
    r2 = 1 - ((np.log(f) - pred) ** 2).sum() / \
        ((np.log(f) - np.log(f).mean()) ** 2).sum()
    k = len(gg)
    lo, hi = (slice(0, k // 2), slice(k - k // 2, k))
    a, c = slope_d(gg[lo], f[lo]), slope_d(gg[hi], f[hi])
    obs = max(a, c) / min(a, c)
    null = np.empty(boot)
    for t in range(boot):
        fb = np.exp(pred + rs.normal(0.0, SIG, size=len(gg)))
        a2, c2 = slope_d(gg[lo], fb[lo]), slope_d(gg[hi], fb[hi])
        null[t] = max(a2, c2) / min(a2, c2)
    return r2, obs, float((np.sum(null >= obs) + 1) / (boot + 1))


def goodness_of_fit_power():
    grids = {
        "4-point": np.array([500., 1000., 2000., 4000.]),
        "6-point": np.array([125., 250., 500., 1000., 2000., 4000.]),
        "8-point": np.array([125., 250., 500., 1000., 2000., 4000.,
                             8000., 16000.]),
    }
    rows = []
    for gname, gg in grids.items():
        rr = np.random.default_rng(0)
        f2 = gg ** -0.5 + 0.35 * gg ** -0.125
        pr, pd = [], []
        for _ in range(150):
            r2v, _, pv = _fit_and_test(gg, f2, rr)
            pr.append(r2v < 0.95)
            pd.append(pv < 0.05)
        fp, fixed = [], []
        for _ in range(150):
            _, obs, pv = _fit_and_test(gg, gg ** -0.2, rr)
            fp.append(pv < 0.05)
            fixed.append(obs > 1.25)
        rows.append({"grid": gname, "n_points": len(gg),
                     "power_r2": float(np.mean(pr)),
                     "power_drift": float(np.mean(pd)),
                     "size_drift": float(np.mean(fp)),
                     "rejection_fixed_cut_1.25_under_null": float(np.mean(fixed))})
        print(f"  {gname}: P(R^2<0.95)={np.mean(pr):.3f}  power={np.mean(pd):.3f}  "
              f"size={np.mean(fp):.3f}  fixed 1.25 cut on the null={np.mean(fixed):.3f}")
    return {"noise_sd_log_floor": SIG, "trials": 150, "by_grid_length": rows}


def _mle_clamped(X, k=10):
    """Maximum-likelihood estimate with zero distances replaced by 1e-12."""
    dist, _ = cKDTree(X).query(X, k=k + 1)
    d = np.maximum(dist[:, 1:], 1e-12)
    inv = np.log(d[:, -1][:, None] / d[:, :-1]).mean(1)
    return float(1.0 / np.mean(inv))


def tied_observations():
    base = cg.Standardizer().fit_transform(cg.data.gaussian_copula(4000, 6, seed=5))
    rows = []
    for levels in (None, 50, 20, 10):
        if levels is None:
            Xq, tag = base.copy(), "none"
        else:
            Xq = np.empty_like(base)
            for j in range(base.shape[1]):
                q = np.unique(np.quantile(base[:, j], np.linspace(0, 1, levels + 1)))
                i = np.clip(np.searchsorted(q, base[:, j]) - 1, 0, len(q) - 2)
                Xq[:, j] = 0.5 * (q[i] + q[i + 1])
            tag = str(levels)
        Xs = cg.Standardizer().fit_transform(Xq)
        d = cg.intrinsic_dimension(Xs, n_boot=2)
        mle_c = _mle_clamped(Xs)
        est = [d.wasserstein, d.twonn, mle_c]
        rows.append({"levels": tag, "ties": d.tie_fraction, "W": d.wasserstein,
                     "twonn": d.twonn, "mle": d.mle, "spread": d.spread,
                     "mle_zero_distance_replaced": mle_c,
                     "spread_zero_distance_replaced": float(max(est) / max(min(est), 1e-9)),
                     "lattice_suspected": d.lattice_suspected})
        print(f"  levels {tag:>4}: ties {d.tie_fraction:.1%}  spread "
              f"{d.spread:.2f}x (excluded)  "
              f"{rows[-1]['spread_zero_distance_replaced']:.2f}x (replaced)")
    return rows


def mdd_proxy():
    raw = cg.data.gaussian_copula(6000, 6, seed=21)
    st = cg.Standardizer().fit(raw)
    Xs = st.transform(raw)
    n, R = 500, 200
    fl = cg.resolution_floor(Xs, n, "w2", n_rep=30, seed=0)
    muX, covX = Xs.mean(0), np.cov(Xs, rowvar=False)
    models = {
        "fresh draw": lambda gg, k: st.transform(
            cg.data.gaussian_copula(k, 6, seed=int(gg.integers(1, 10 ** 6)))),
        "T2 gaussian": lambda gg, k: gg.multivariate_normal(muX, covX, size=k),
        "T1 marginals": lambda gg, k: np.column_stack(
            [gg.choice(Xs[:, j], k) for j in range(Xs.shape[1])]),
    }
    cells = []
    for name, gen in models.items():
        dA, dB = [], []
        for r in range(R):
            gg = np.random.default_rng(900 + r)
            Xr = Xs[gg.permutation(len(Xs))[:n]]
            dA.append(w2_exact(Xr, gen(gg, n), seed=r))
            dB.append(w2_exact(Xr, gen(np.random.default_rng(4000 + r), n), seed=r))
        dA, dB = np.array(dA), np.array(dB)
        sd_pair = float((dA - dB).std(ddof=1))
        mean_D = float(dA.mean())
        cells.append({"model": name, "mean_D": mean_D, "sd_paired": sd_pair,
                      "ratio_floor_level": float(np.sqrt(2.0) * fl.std / sd_pair),
                      "ratio_level_scaled": fl.sd_difference_at(mean_D) / sd_pair})
        print(f"  {name:<13} E[D]={mean_D:.4f}  sd(D_A-D_B)={sd_pair:.5f}  "
              f"floor-level proxy {cells[-1]['ratio_floor_level']:.2f}x  "
              f"level-scaled proxy {cells[-1]['ratio_level_scaled']:.2f}x")
    return {"n_per_side": n, "replicates": R, "floor_median": fl.median,
            "floor_cv": fl.cv, "cells": cells}


def main():
    t0 = time.time()
    out = {"_definition": ("stress tests of the skill score, the specification "
                           "test, the dimension estimators and the floor proxy "
                           "on synthetic laws")}
    print("memorisation")
    out["memorisation"] = memorisation()
    print("goodness of fit against curvature")
    out["goodness_of_fit_power"] = goodness_of_fit_power()
    print("tied observations")
    out["tied_observations"] = tied_observations()
    print("floor proxy of the paired standard deviation")
    out["mdd_proxy"] = mdd_proxy()
    out["elapsed_s"] = round(time.time() - t0, 1)
    art.save("stress_tests.json", out, indent=2)


if __name__ == "__main__":
    main()
