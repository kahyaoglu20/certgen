"""Operating characteristics of the null-calibrated leakage screen
(Section 5.3).

  by_table_size       an exact memoriser (500 copied rows) of tables of
                      1000 to 60000 rows;
  by_perturbation     memorisers that add Gaussian noise of increasing scale
                      to the copied rows, relative to the median
                      nearest-neighbour distance between real rows;
  discrete_laws       generators drawing fresh samples from four discrete or
                      mixed laws, whose real rows coincide;
  identifiability     exact memorisers and fresh draws on purely categorical
                      laws of increasing support size.

    python experiments/leakage_screen.py

Writes results/leakage_screen.json (seconds).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import _artifacts as art
import certgen as cg
from certgen.leakage import leakage_check


def disc(n, nc, lv, n_num, seed):
    """nc uniform categorical columns with lv levels, plus n_num normal columns."""
    g = np.random.default_rng(seed)
    c = g.integers(0, lv, size=(n, nc)).astype(float)
    return np.column_stack([c, g.normal(size=(n, n_num))]) if n_num else c


def by_table_size():
    rows = []
    for Nr in (1000, 3000, 10000, 30000, 60000):
        X = cg.Standardizer().fit_transform(cg.data.gaussian_copula(Nr, 6, seed=3))
        flag = bool(leakage_check(X, X[:500], seed=0).suspected)
        rows.append({"N_real": Nr, "flagged": flag})
        print(f"  N_real={Nr:>6}  memoriser flagged={flag}")
    return rows


def by_perturbation():
    X = cg.Standardizer().fit_transform(cg.data.gaussian_copula(3000, 6, seed=11))
    med_nn = float(np.median(cKDTree(X).query(X, k=2)[0][:, 1]))
    rng = np.random.default_rng(4)
    cells = []
    for sig in (0.0, 1e-3, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4):
        Xg = X[:800] + rng.normal(0, sig, size=(800, X.shape[1]))
        rep = leakage_check(X, Xg, seed=0)
        cells.append({"sigma": sig, "rel": sig / med_nn,
                      "flagged": bool(rep.suspected),
                      "median_dcr_ratio": rep.dcr_median_ratio})
        print(f"  sigma/median NN={sig / med_nn:6.3f}  flagged={rep.suspected}")
    escaped = [c["rel"] for c in cells if not c["flagged"]]
    return {"median_nn": med_nn, "cells": cells,
            "escape_rel": min(escaped) if escaped else float("inf")}


def discrete_laws():
    rows = []
    for tag, nc, lv, nn in [("3 cat(4) + 3 numeric", 3, 4, 3),
                            ("5 cat(4) + 1 numeric", 5, 4, 1),
                            ("6 cat(3), no numeric", 6, 3, 0),
                            ("4 cat(2), no numeric", 4, 2, 0)]:
        Xr, Xg = disc(3000, nc, lv, nn, 1), disc(800, nc, lv, nn, 999)
        st = cg.Standardizer().fit(Xr)
        flag = bool(leakage_check(st.transform(Xr), st.transform(Xg),
                                  seed=0).suspected)
        rows.append({"case": tag, "flagged": flag})
        print(f"  {tag:<24} fresh draw flagged={flag}")
    return rows


def identifiability():
    rows = []
    for nc, lv in [(4, 2), (6, 3), (6, 5), (8, 5), (10, 6)]:
        Xr = disc(3000, nc, lv, 0, 1)
        st = cg.Standardizer().fit(Xr)
        a = st.transform(Xr)
        h = st.transform(disc(800, nc, lv, 0, 999))
        rows.append({"cols": nc, "levels": lv, "support": lv ** nc,
                     "ratio": 3000 / lv ** nc,
                     "memoriser": bool(leakage_check(a, a[:800], seed=0).suspected),
                     "honest": bool(leakage_check(a, h, seed=0).suspected)})
        print(f"  n/|support|={rows[-1]['ratio']:9.4f}  memoriser flagged="
              f"{rows[-1]['memoriser']}  fresh draw flagged={rows[-1]['honest']}")
    return rows


def main():
    out = {"_definition": ("detection rates and false alarms of "
                           "certgen.leakage_check")}
    print("table size")
    out["by_table_size"] = by_table_size()
    print("perturbed memorisers")
    out["by_perturbation"] = by_perturbation()
    print("discrete laws")
    out["discrete_laws"] = discrete_laws()
    print("identifiability on categorical supports")
    out["identifiability"] = identifiability()
    art.save("leakage_screen.json", out, indent=2)


if __name__ == "__main__":
    main()
