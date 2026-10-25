"""Coverage of the sample-size interval and its level calibration
(Sections 4.6 and 8.3).

Floor profiles are drawn from known power laws C n^(-1/d) with C = 3 on the
six-point grid {125, ..., 4000} (n_rep = 12, lognormal replicate noise) and
passed through certgen.fit_floor_law. Three measurements:

  1. coverage of the 90% interval of FloorLaw.required_n_interval for the
     target floors at n = 1000 (inside the grid) and n = 40000 (10x beyond
     it), and of the exponent interval, for d in {4, 5, 6, 8} and replicate
     coefficients of variation {0.03, 0.08, 0.13};
  2. delivered coverage at six nominal levels and its inversion, the map
     certgen.design.INTERVAL_LEVEL_CALIBRATION;
  3. the interval on the eleven benchmark cells of
     results/benchmark_floor_laws.json, at targets n = 1000 to 40000.

    python experiments/interval_coverage.py               # about 2 minutes
    python experiments/interval_coverage.py --real-only   # step 3 only

Writes results/interval_coverage.json.
"""
from __future__ import annotations

import sys

import numpy as np

import _artifacts as art
import certgen as cg
from _benchmarks import law_from_record

TRIALS = 200
N_REP = 12
LEVEL = 0.90
LEVELS = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975)


def one_trial(rng, grid, d_known, cv, seed):
    reps = {}
    for n in grid:
        mu = 3.0 * float(n) ** (-1.0 / d_known)
        reps[int(n)] = np.exp(np.log(mu) + rng.normal(0.0, cv, N_REP))
    return cg.fit_floor_law(_replicates=reps, n_grid=[int(n) for n in grid],
                            metric="w2", n_rep=N_REP, seed=seed,
                            drift_boot=100)


def coverage_cells(grid):
    cells = []
    for d_known in (4.0, 5.0, 6.0, 8.0):
        for cv in (0.03, 0.08, 0.13):
            rng = np.random.default_rng(int(d_known * 100) + int(cv * 1000))
            cover_int = cover_ext = cover_d = 0
            widths_int, widths_ext = [], []
            for t in range(TRIALS):
                law = one_trial(rng, grid, d_known, cv, seed=t)
                if not np.isfinite(law.drift_p_value):
                    continue
                t_int = 3.0 * 1000.0 ** (-1.0 / d_known)
                t_ext = 3.0 * 40000.0 ** (-1.0 / d_known)
                n_true_int = (3.0 / t_int) ** d_known
                n_true_ext = (3.0 / t_ext) ** d_known
                iv_int = law.required_n_interval(float(t_int), level=LEVEL,
                                                 seed=t)
                iv_ext = law.required_n_interval(float(t_ext), level=LEVEL,
                                                 seed=t)
                cover_int += int(iv_int["n_lo"] <= n_true_int <= iv_int["n_hi"])
                cover_ext += int(iv_ext["n_lo"] <= n_true_ext <= iv_ext["n_hi"])
                cover_d += int(iv_int["d_lo"] <= d_known <= iv_int["d_hi"])
                widths_int.append(iv_int["n_hi"] / max(iv_int["n_lo"], 1e-9))
                widths_ext.append(iv_ext["n_hi"] / max(iv_ext["n_lo"], 1e-9))
            n_ok = max(len(widths_int), 1)
            cell = {"d_known": d_known, "replicate_cv": cv,
                    "coverage_n_interior": cover_int / n_ok,
                    "coverage_n_extrapolated": cover_ext / n_ok,
                    "coverage_exponent": cover_d / n_ok,
                    "width_interior_med": float(np.median(widths_int)),
                    "width_extrapolated_med": float(np.median(widths_ext))}
            cells.append(cell)
            print(f"  d={d_known:.0f} cv={cv:.2f}: coverage interior "
                  f"{cell['coverage_n_interior']:.3f}, extrapolated "
                  f"{cell['coverage_n_extrapolated']:.3f}, exponent "
                  f"{cell['coverage_exponent']:.3f}", flush=True)
    return cells


def level_calibration(grid):
    cal_cells = []
    for d_known in (4.0, 5.0, 6.0):
        for cv in (0.03, 0.08, 0.13):
            rng = np.random.default_rng(7000 + int(d_known * 100)
                                        + int(cv * 1000))
            cov_by_level = {lv: 0 for lv in LEVELS}
            cov_ext_by_level = {lv: 0 for lv in LEVELS}
            n_ok = 0
            for t in range(TRIALS):
                law = one_trial(rng, grid, d_known, cv, seed=t)
                if not np.isfinite(law.drift_p_value):
                    continue
                t_int = 3.0 * 1000.0 ** (-1.0 / d_known)
                t_ext = 3.0 * 40000.0 ** (-1.0 / d_known)
                n_true_int = (3.0 / t_int) ** d_known
                n_true_ext = (3.0 / t_ext) ** d_known
                n_ok += 1
                for lv in LEVELS:
                    # the same seed gives the same draws; only the quantiles move
                    iv = law.required_n_interval(float(t_int), level=lv, seed=t)
                    cov_by_level[lv] += int(iv["n_lo"] <= n_true_int <= iv["n_hi"])
                    iv = law.required_n_interval(float(t_ext), level=lv, seed=t)
                    cov_ext_by_level[lv] += int(
                        iv["n_lo"] <= n_true_ext <= iv["n_hi"])
            cal_cells.append({
                "d_known": d_known, "replicate_cv": cv,
                "delivered_interior": {str(lv): cov_by_level[lv] / max(n_ok, 1)
                                       for lv in LEVELS},
                "delivered_extrapolated": {
                    str(lv): cov_ext_by_level[lv] / max(n_ok, 1)
                    for lv in LEVELS}})
    # per level: min / median / max over cells of the mean of both targets
    agg = {}
    for lv in LEVELS:
        pooled = [0.5 * (c["delivered_interior"][str(lv)]
                         + c["delivered_extrapolated"][str(lv)])
                  for c in cal_cells]
        agg[str(lv)] = {"min": round(float(np.min(pooled)), 3),
                        "median": round(float(np.median(pooled)), 3),
                        "max": round(float(np.max(pooled)), 3)}
    # inverse map: the nominal level whose median delivered coverage equals
    # each requested level (linear interpolation on the monotonised curve)
    x_nom = np.array(LEVELS)
    y_mono = np.maximum.accumulate(
        np.array([agg[str(lv)]["median"] for lv in LEVELS]))
    inverse = {str(dt): round(float(np.interp(dt, y_mono, x_nom)), 4)
               for dt in (0.80, 0.85, 0.90, 0.95)}
    for lv in LEVELS:
        a = agg[str(lv)]
        print(f"  nominal {lv:.3f}: delivered {a['median']:.3f} "
              f"[{a['min']:.3f}, {a['max']:.3f}]")
    return {
        "_definition": ("delivered coverage of the interval as a function of "
                        "its nominal level on known single-exponent laws, and "
                        "the nominal level that delivers each requested "
                        "coverage"),
        "regime": ("d in {4,5,6}, replicate CV in {0.03,0.08,0.13}, 6-point "
                   "grid [125,4000], n_rep=12, 200 trials per cell, targets "
                   "n=1000 and n=40000"),
        "levels": list(LEVELS),
        "delivered_by_level": agg,
        "inverse_map_requested_to_nominal": inverse,
        "cells": cal_cells}


def benchmark_intervals():
    src = art.load("benchmark_floor_laws.json")
    n_rep = int(src["_config"].get("n_rep", 12))
    real = []
    for rec in src["records"]:
        law = law_from_record(rec, n_rep)
        rows = {"dataset": rec["dataset"],
                "single_exponent_ok": bool(rec["single_exponent_ok"]),
                "targets": {}}
        for n_t in (1000, 4000, 10000, 40000):
            t = law.floor_at(float(n_t))
            iv = law.required_n_interval(float(t), level=LEVEL, seed=5)
            rows["targets"][f"floor_at_n={n_t}"] = {
                "target_floor": float(t),
                "n_point": iv["n_point"], "n_lo": iv["n_lo"],
                "n_med": iv["n_med"], "n_hi": iv["n_hi"],
                "width_ratio": iv["n_hi"] / max(iv["n_lo"], 1e-9)}
        real.append(rows)
        print(f"  {rec['dataset']:<20} width ratio at n=10000 "
              f"{rows['targets']['floor_at_n=10000']['width_ratio']:.2f}, "
              f"at n=40000 {rows['targets']['floor_at_n=40000']['width_ratio']:.2f}")
    return real


def main():
    grid = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    if "--real-only" in sys.argv:
        out = art.load("interval_coverage.json")
    else:
        out = {"_definition": ("coverage of FloorLaw.required_n_interval on "
                               "known power laws, its level calibration, and "
                               "the interval on the benchmark cells"),
               "trials": TRIALS, "n_rep": N_REP, "level": LEVEL}
        print("coverage on known power laws")
        out["cells"] = coverage_cells(grid)
        print("level calibration")
        out["level_calibration"] = level_calibration(grid)
    print("benchmark cells")
    out["real_table_source"] = "benchmark_floor_laws.json"
    out["real_tables"] = benchmark_intervals()
    art.save("interval_coverage.json", out, indent=2)


if __name__ == "__main__":
    main()
