"""Worked examples on synthetic Gaussian copulas (Sections 3 to 5).

floor_law            floor law of a six-dimensional copula (N = 4000, seed 6,
                     n_rep = 20) and the resolvable changes it implies at
                     alpha = 0.05 and power 0.80: at n = 500 and 5000, and the
                     n needed to resolve 0.043 (Sections 3.4 and 4.1);
sliced_exponent      the effective exponent of sliced W2 on the same data
                     (Section 3.3);
unbiased_statistics  fraction of the null replicates of the energy distance
                     and of the MMD that are zero after clipping, on a
                     standardised eight-dimensional copula (N = 4000,
                     n in {125, 500, 2000}, 100 replicates; Section 3.3);
sliced_underestimate ratio of sliced W2 to exact W2 between two independent
                     samples of 1000 standard normal points in dimension 64
                     (Section 3.5);
skill_above_one      generators drawing from the data law (copulas of
                     dimension 4 and 8, 2000 real rows, n = 500 per side,
                     200 trials): the probability of a skill above 1 and of
                     V < -0.05 against each reference generator (Section 5.2);
shift_spread         on the copula of the equivalence study, the standard
                     deviation of D for fresh draws and for fresh draws
                     shifted by a fixed vector whose expected distance lies
                     one equivalence margin above the floor (Section 5.4).

    python experiments/copula_examples.py

Writes results/copula_examples.json (about 7 minutes).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import norm

import _artifacts as art
import certgen as cg
from certgen.data import gaussian_copula
from certgen.discrepancy import sliced_w2, w2_exact

warnings.filterwarnings("ignore")
ALPHA, POWER = 0.05, 0.80
TARGET_EFFECT = 0.043


def floor_law_example():
    X = gaussian_copula(4000, d=6, rho=0.8, seed=6)
    law = cg.fit_floor_law(X, n_rep=20, seed=0)
    # the coefficient of variation at which FloorLaw.std_at extrapolates
    cv = float(np.median(law.stds / np.maximum(law.floors, 1e-15)))
    z = float(norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER))
    # resolvable(n) = z sqrt(2) cv C n^(-1/d_eff), solved for n
    n_req = float((z * np.sqrt(2.0) * cv * law.C / TARGET_EFFECT) ** law.d_eff)
    sliced = cg.fit_floor_law(X, metric="sliced_w2", n_rep=20, seed=0)
    return ({"law": {"C": float(law.C), "d_eff": float(law.d_eff),
                     "r2": float(law.r2), "drift_p": float(law.drift_p_value),
                     "single_exponent_ok": bool(law.single_exponent_ok),
                     "cv_median": cv},
             "alpha": ALPHA, "power": POWER,
             "resolvable_effect": {str(n): float(cg.resolvable_effect(
                 law, n, alpha=ALPHA, power=POWER)) for n in (500, 5000)},
             "target_effect": TARGET_EFFECT,
             "n_required_for_target": n_req},
            {"d_eff_sliced": float(sliced.d_eff), "r2_sliced": float(sliced.r2),
             "d_eff_exact": float(law.d_eff)})


def unbiased_statistics():
    X = cg.Standardizer().fit_transform(gaussian_copula(4000, d=8, rho=0.8, seed=8))
    out = {}
    for metric in ("energy", "mmd"):
        out[metric] = {}
        for n in (125, 500, 2000):
            fr = cg.resolution_floor(X, n, metric=metric, n_rep=100, seed=0)
            out[metric][str(n)] = {
                "fraction_zero": float(np.mean(fr.values <= 0.0)),
                "median": fr.median}
    fr_all = [v["fraction_zero"] for m in out.values() for v in m.values()]
    out["range_energy"] = [min(v["fraction_zero"] for v in out["energy"].values()),
                           max(v["fraction_zero"] for v in out["energy"].values())]
    out["range_mmd"] = [min(v["fraction_zero"] for v in out["mmd"].values()),
                        max(v["fraction_zero"] for v in out["mmd"].values())]
    out["range_all"] = [min(fr_all), max(fr_all)]
    return out


def sliced_underestimate():
    ratios = []
    for s in range(5):
        g = np.random.default_rng(640 + s)
        X, Y = g.normal(size=(1000, 64)), g.normal(size=(1000, 64))
        ratios.append(sliced_w2(X, Y, seed=s) / w2_exact(X, Y, seed=s))
    return {"d": 64, "n": 1000, "seeds": 5,
            "ratio_mean": float(np.mean(ratios)),
            "underestimate": float(1.0 - np.mean(ratios))}


def skill_above_one(trials=200):
    out = {}
    for d in (4, 8):
        above, flag = 0, {"T0_mean": 0, "T1_marginals": 0, "T2_gaussian": 0}
        for t in range(trials):
            X = gaussian_copula(2000, d=d, seed=1000 + t)
            G = gaussian_copula(500, d=d, seed=50_000 + t)
            rep = cg.evaluate(X, G, n_per_side=500, n_rep=20, seed=t)
            above += int(rep.skill["T2_gaussian"] > 1.0)
            for b in flag:
                flag[b] += int(rep.V[b] < -0.05)
        out[str(d)] = {"trials": trials,
                       "p_skill_T2_above_1": above / trials,
                       "p_V_below_minus_005": {b: v / trials for b, v in flag.items()}}
        print(f"  d={d}: P(skill > 1) = {above / trials:.3f}, "
              f"P(V < -0.05) = {[round(v / trials, 3) for v in flag.values()]}",
              flush=True)
    rates = [v for c in out.values() for v in c["p_V_below_minus_005"].values()]
    out["p_V_below_minus_005_range"] = [min(rates), max(rates)]
    return out


def shift_spread(trials=200):
    d, N, n = 6, 2400, 600
    X = gaussian_copula(N, d, seed=3)
    floor = cg.resolution_floor(X, n, metric="w2", n_rep=20, seed=11)
    delta = 0.25 * floor.median
    target = floor.median + delta
    Xr = X[:n]
    u = np.ones(d) / np.sqrt(d)

    def mean_D(s, k=40):
        return float(np.mean([w2_exact(Xr, gaussian_copula(n, d, seed=70_000 + j)
                                       + s * u) for j in range(k)]))
    lo, hi = 0.0, 2.0 * delta
    while mean_D(hi) < target:
        hi *= 2.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_D(mid) < target else (lo, mid)
    shift = 0.5 * (lo + hi)
    fresh = np.array([w2_exact(Xr, gaussian_copula(n, d, seed=90_000 + t))
                      for t in range(trials)])
    shifted = np.array([w2_exact(Xr, gaussian_copula(n, d, seed=90_000 + t) + shift * u)
                        for t in range(trials)])
    return {"d": d, "N": N, "n_per_side": n, "trials": trials,
            "floor_median": floor.median, "margin": delta,
            "shift_norm": shift,
            "mean_D_fresh": float(fresh.mean()), "mean_D_shifted": float(shifted.mean()),
            "sd_D_fresh": float(fresh.std(ddof=1)),
            "sd_D_shifted": float(shifted.std(ddof=1)),
            "sd_ratio": float(shifted.std(ddof=1) / fresh.std(ddof=1)),
            "variance_ratio": float(shifted.var(ddof=1) / fresh.var(ddof=1))}


def main():
    t0 = time.time()
    out = {"_definition": ("worked examples on synthetic Gaussian copulas "
                           "quoted in Sections 3 to 5")}
    print("floor law and sliced exponent")
    out["floor_law"], out["sliced_exponent"] = floor_law_example()
    print("unbiased statistics")
    out["unbiased_statistics"] = unbiased_statistics()
    print("sliced underestimate")
    out["sliced_underestimate"] = sliced_underestimate()
    print("skill above one")
    out["skill_above_one"] = skill_above_one()
    print("spread under a mean shift")
    out["shift_spread"] = shift_spread()
    out["elapsed_s"] = round(time.time() - t0, 1)
    art.save("copula_examples.json", out, indent=2)


if __name__ == "__main__":
    main()
