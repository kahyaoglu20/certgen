"""Operating characteristics of the equivalence test (Section 5.5).

On a six-dimensional Gaussian copula (N = 2400, n = 600 per side,
n_rep = 20 floor replicates) with the margin eta = 0.25 x floor median:

  * statistics placed exactly at the boundaries floor +/- eta, and at
    +/- 2 eta, built from the floor replicates themselves;
  * fresh draws from the copula (a generator drawing from the data law);
  * an exact copy of the reference sample (a memoriser);
  * the upper-tail p-value of 200 fresh generators;
  * the reduction of the test to the two-sided floor test at eta = 0.

    python experiments/equivalence_test.py

Writes results/equivalence_test.json (under a minute).
"""
from __future__ import annotations

import numpy as np

import _artifacts as art
import certgen as cg


def main():
    d, N, n, n_rep = 6, 2400, 600, 20
    trials = 400
    alpha = 0.05
    X = cg.data.gaussian_copula(N, d, seed=3)
    floor = cg.resolution_floor(X, n, metric="w2", n_rep=n_rep, seed=11)
    fv = np.asarray(floor.values, float)
    f_med = floor.median
    kappa = 0.25
    delta = kappa * f_med

    res = {"_definition": ("operating characteristics of "
                           "FloorReport.equivalence on a six-dimensional "
                           "Gaussian copula"),
           "config": {"d": d, "N": N, "n_per_side": n, "n_rep": n_rep,
                      "delta": float(delta), "kappa": kappa,
                      "floor_median": float(f_med), "trials": trials,
                      "boundary": "statistic space: floor replicate +/- delta"},
           "alpha": alpha}
    print(f"floor median {f_med:.4f}, delta = {kappa} x floor = {delta:.4f}, "
          f"delta_min = {floor.equivalence(f_med, delta)['delta_min']:.4f}")

    def stat_draw(kind, t):
        i = (t * 37 + 11) % len(fv)
        return {"boundary +delta": float(fv[i] + delta),
                "boundary -delta": float(fv[i] - delta),
                "far +2delta": float(fv[i] + 2 * delta),
                "far -2delta": float(fv[i] - 2 * delta)}[kind]

    rows = {}
    for label in ("boundary +delta", "boundary -delta",
                  "far +2delta", "far -2delta"):
        eq = 0
        pA, pB = [], []
        for t in range(trials):
            r = floor.equivalence(stat_draw(label, t), float(delta))
            eq += int(r["equivalent"])
            pA.append(r["p_too_bad"])
            pB.append(r["p_too_good"])
        rows[label] = {"equiv_rate": eq / trials,
                       "mean_p_too_bad": float(np.mean(pA)),
                       "mean_p_too_good": float(np.mean(pB))}
        print(f"  {label:<26} P(equivalent) = {eq / trials:.3f}")

    eq = 0
    pA, pB = [], []
    for t in range(trials):
        G = cg.data.gaussian_copula(n, d, seed=9000 + 7 * t)
        D = float(cg.w2_exact(X[:n], G))
        r = floor.equivalence(D, float(delta))
        eq += int(r["equivalent"])
        pA.append(r["p_too_bad"])
        pB.append(r["p_too_good"])
    rows["honest (fresh draws)"] = {"equiv_rate": eq / trials,
                                    "mean_p_too_bad": float(np.mean(pA)),
                                    "mean_p_too_good": float(np.mean(pB))}
    print(f"  {'fresh draws':<26} P(equivalent) = {eq / trials:.3f}")

    # a memoriser that returns the reference rows themselves
    eq = 0
    Dm_list = []
    for t in range(trials):
        Dm = float(cg.w2_exact(X[:n], X[:n]))
        Dm_list.append(Dm)
        r = floor.equivalence(Dm, float(delta))
        eq += int(r["equivalent"])
    rows["reference copy (memoriser)"] = {
        "equiv_rate": eq / trials,
        "D": float(np.mean(Dm_list))}
    print(f"  {'reference copy':<26} P(equivalent) = {eq / trials:.3f}")

    p_hi = np.array([floor.p_value(
        float(cg.w2_exact(X[:n],
                          cg.data.gaussian_copula(n, d, seed=12000 + t))))
                     for t in range(200)])
    res["p_hi_uniformity"] = {
        "frac_le_05": float(np.mean(p_hi <= 0.05)),
        "median": float(np.median(p_hi)),
        "n": 200}

    checks = {
        "size_hi": rows["boundary +delta"]["equiv_rate"] <= alpha + 0.03,
        "size_lo": rows["boundary -delta"]["equiv_rate"] <= alpha + 0.03,
        "power_honest": rows["honest (fresh draws)"]["equiv_rate"] >= 0.80,
        "far_hi": rows["far +2delta"]["equiv_rate"] <= 0.02,
        "far_lo": rows["far -2delta"]["equiv_rate"] <= 0.02,
        "far_memoriser": rows["reference copy (memoriser)"]["equiv_rate"] <= 0.02,
        "p_hi_not_anticonservative": res["p_hi_uniformity"]["frac_le_05"] <= alpha + 0.03,
    }
    G0 = cg.data.gaussian_copula(n, d, seed=4242)
    D0 = float(cg.w2_exact(X[:n], G0))
    r0 = floor.equivalence(D0, 0.0)
    checks["degenerate"] = bool(
        abs(r0["p_too_bad"] - floor.p_value_low(D0)) < 1e-12
        and abs(r0["p_too_good"] - floor.p_value(D0)) < 1e-12)
    for k, v in checks.items():
        print(f"  {k:<32} {'ok' if v else 'FAIL'}")
    res["rates"] = rows
    res["checks"] = checks
    art.save("equivalence_test.json", res, indent=2)


if __name__ == "__main__":
    main()
