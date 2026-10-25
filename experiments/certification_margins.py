"""Certification margins of the admissible benchmark cells
(Section 8.5, Appendix B.6).

For each admissible cell of results/benchmark_floor_laws.json, 8000 joint
Monte-Carlo draws combine three sources of uncertainty:

  (a) the measured dimension, d_hat ~ Normal(mean, subsample sd);
  (b) a law-to-law residual in r, resampled from the in-band cells of
      results/transfer_smooth_laws.json;
  (c) the calibration map D_HAT_CALIBRATION, the cells of the two sizes
      that bracket the cell's row count perturbed by their standard errors.

Each draw converts d_hat to nominal units at the number of rows the
estimator saw, and evaluates the reference tables at the cell's own size per
side. The margin in dimension units is the law-adjusted crossing divided by
the drawn nominal dimension; the margin in r units is 1/r at that dimension.
A second set of r-unit margins divides by 1 + the largest heavy-tail
residual of results/transfer_probe.json.

    python experiments/certification_margins.py

Writes results/certification_margins.json (under a minute).
"""
from __future__ import annotations

import numpy as np

import _artifacts as art
import certgen as cg
from certgen.latent import (D_HAT_CALIBRATION, D_HAT_CALIBRATION_NREP,
                            D_HAT_CALIBRATION_SD, TABULATED_N)

N_MC = 8000
SEED = 20260905
D_GRID = np.exp(np.linspace(np.log(2.0), np.log(128.0), 2000))

_NS = np.array(sorted(D_HAT_CALIBRATION), float)
_DN = np.array(sorted(D_HAT_CALIBRATION[int(_NS[0])]), float)
_MEAN = np.array([[D_HAT_CALIBRATION[int(n)][int(d)] for d in _DN] for n in _NS])
_SE = np.array([[D_HAT_CALIBRATION_SD[int(n)][int(d)] for d in _DN]
                for n in _NS]) / np.sqrt(D_HAT_CALIBRATION_NREP)


def crossing_nominal(residual: float, N: int) -> float:
    """Nominal dimension at which r (1 + residual) reaches 1 at size N."""
    r = np.array([cg.floor_ratio(d, N) for d in D_GRID]) * (1.0 + residual)
    ok = np.where(r < 1.0)[0]
    return float(D_GRID[ok[-1]]) if len(ok) else float("nan")


def _bracket(N):
    """Indices of the two tabulated sizes that bracket N (clamped)."""
    j = int(np.clip(np.searchsorted(_NS, float(N)), 1, len(_NS) - 1))
    return [j - 1, j]


def meas_to_nom_perturbed(d_hat, N, rng):
    """measured_to_nominal against a perturbed copy of the calibration map.

    Only the two rows that enter the interpolation at N are perturbed, so the
    draws do not depend on rows of the table far from N.
    """
    rows = _bracket(N)
    tab = np.maximum(_MEAN[rows] + rng.normal(0.0, _SE[rows]), 1e-6)
    dh = np.exp([np.interp(np.log(max(float(N), 1.0)), np.log(_NS[rows]),
                           np.log(tab[:, j])) for j in range(len(_DN))])
    order = np.argsort(dh)
    return float(np.exp(np.interp(np.log(max(d_hat, 1e-9)),
                                  np.log(dh[order]), np.log(_DN[order]))))


def main():
    transfer = art.load("transfer_smooth_laws.json")
    band = [c for c in transfer["cells"] if c.get("in_target_dhat_band")]
    residuals = np.array([c["residual_vs_table_at_dhat"] for c in band])
    fams = sorted({c["meta"]["family"] for c in band})
    adj = []
    for c in band:
        cnv = crossing_nominal(c["residual_vs_table_at_dhat"], c["N"])
        adj.append({"law": c["law"], "family": c["meta"]["family"],
                    "N": c["N"], "residual": c["residual_vs_table_at_dhat"],
                    "crossing_nominal": cnv,
                    "crossing_measured": float(cg.nominal_to_measured(cnv, c["N"]))})
    cn = np.array([a["crossing_nominal"] for a in adj])
    cm = np.array([a["crossing_measured"] for a in adj])

    recs = [r for r in art.load("benchmark_floor_laws.json")["records"]
            if r.get("single_exponent_ok")]
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'cell':<18} {'d_hat':>6} {'q05 d-units':>12} {'q05 r-units':>12}")
    for r in recs:
        xing_cache = {}
        dh = float(r["d_wasserstein"])
        N_rows = int(r["n"])            # rows seen by the estimator
        N_side = int(r["n"]) // 2       # per-side size of the two-sample floor
        sd_used = float(r["d_W_sd_subsample_scaled"])
        margins = np.empty(N_MC)
        margins_r = np.empty(N_MC)
        dnoms = np.empty(N_MC)
        for b in range(N_MC):
            dh_b = max(dh + rng.normal(0.0, sd_used), 0.5)
            d_nom = meas_to_nom_perturbed(dh_b, N_rows, rng)
            resid_b = adj[int(rng.integers(0, len(adj)))]["residual"]
            key = round(float(resid_b), 4)
            if key not in xing_cache:
                xing_cache[key] = crossing_nominal(resid_b, N_side)
            dnoms[b] = d_nom
            margins[b] = xing_cache[key] / max(d_nom, 1e-9)
            r_b = cg.floor_ratio(max(d_nom, 1e-9), N_side) * (1.0 + resid_b)
            margins_r[b] = 1.0 / max(r_b, 1e-12)
        p_bad = float((margins < 1.0).mean())
        p_vac = float((margins_r < 1.0).mean())
        rows.append({"dataset": r["dataset"], "d_hat": dh,
                     "N_rows_estimator": N_rows, "N_side_floor": N_side,
                     "crossing_nominal_at_N_side_median": (
                         float(np.median(list(xing_cache.values())))
                         if xing_cache else float("nan")),
                     "sd_used": sd_used, "sd_source": "subsample",
                     "d_nominal_median": float(np.median(dnoms)),
                     "margin_median": float(np.median(margins)),
                     "margin_q05": float(np.quantile(margins, 0.05)),
                     "margin_q95": float(np.quantile(margins, 0.95)),
                     "p_margin_below_1": p_bad,
                     "p_margin_below_1_report": (f"< {1.0 / N_MC:.5f}" if p_bad == 0.0
                                                 else f"{p_bad:.5f}"),
                     "margin_r_units_median": float(np.median(margins_r)),
                     "margin_r_units_q05": float(np.quantile(margins_r, 0.05)),
                     "margin_r_units_q95": float(np.quantile(margins_r, 0.95)),
                     "p_vacuous_r": p_vac,
                     "p_vacuous_r_report": (f"< {1.0 / N_MC:.5f}" if p_vac == 0.0
                                            else f"{p_vac:.5f}"),
                     "n_mc": int(N_MC)})
        print(f"{r['dataset']:<18} {dh:>6.2f} {rows[-1]['margin_q05']:>11.2f}x "
              f"{rows[-1]['margin_r_units_q05']:>11.2f}x")

    heavy = art.load("transfer_probe.json")["families"]["heavy_tailed"]
    worst = max(heavy["all_range"])
    out = {
        "_definition": (
            "joint Monte-Carlo margins of the admissible benchmark cells over "
            "the measured dimension, the law-to-law residual in r and the "
            "calibration map; margins in dimension units (crossing / nominal "
            "dimension) and in r units (1/r)"),
        "tabulated_N": list(TABULATED_N),
        "n_mc": N_MC, "seed": SEED,
        "band_residuals": {
            "n_cells": int(len(residuals)),
            "median": float(np.median(residuals)),
            "min": float(residuals.min()), "max": float(residuals.max()),
            "by_family": {f: {
                "n": int(sum(1 for c in band if c["meta"]["family"] == f)),
                "median": float(np.median([c["residual_vs_table_at_dhat"]
                                           for c in band
                                           if c["meta"]["family"] == f]))}
                for f in fams}},
        "law_adjusted_crossings": adj,
        "crossing_nominal": {"min": float(cn.min()), "median": float(np.median(cn)),
                             "max": float(cn.max())},
        "crossing_measured": {"min": float(cm.min()), "median": float(np.median(cm)),
                              "max": float(cm.max())},
        "per_table": rows,
        "tightest_margin_q05": float(min(x["margin_q05"] for x in rows)),
        "largest_p_margin_below_1": float(max(x["p_margin_below_1"] for x in rows)),
        "heavy_tail_sensitivity": {
            "worst_heavy_tail_residual": worst,
            "source": "results/transfer_probe.json (families.heavy_tailed)",
            "margin_r_units_q05_if_applied": {
                t["dataset"]: t["margin_r_units_q05"] / (1.0 + worst)
                for t in rows}},
    }
    art.save("certification_margins.json", out, indent=2)


if __name__ == "__main__":
    main()
