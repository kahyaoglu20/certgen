"""Agreement of the half-grid prescriptions on the benchmark cells
(Section 8.3).

For each cell of results/benchmark_floor_laws.json the fitted floor law is
rebuilt from the stored sweep (grid, floor medians and standard
deviations), and FloorLaw.required_n_range is evaluated at four targets, the
fitted floor at n = 500, 1000, 2000 and 4000. The rebuilt fit is checked
against the stored one.

    python experiments/half_fit_ranges.py

Writes results/half_fit_ranges.json (seconds; no download).
"""
from __future__ import annotations

import numpy as np

import _artifacts as art
from _benchmarks import law_from_record


def main():
    src = art.load("benchmark_floor_laws.json")
    n_rep = int(src["_config"].get("n_rep", 12))
    out = {"_definition": ("sample sizes prescribed by the low-half, "
                           "high-half and full fits of each benchmark cell "
                           "(FloorLaw.required_n_range) at four target floors"),
           "_source": "benchmark_floor_laws.json",
           "n_rep": n_rep, "tables": []}
    print(f"{'cell':<20} {'worst half-fit ratio':>21}")
    for rec in src["records"]:
        law = law_from_record(rec, n_rep)
        consistent = all(
            np.isclose(getattr(law, a), rec[b], rtol=1e-9, atol=0.0,
                       equal_nan=True)
            for a, b in (("d_eff", "d_eff"), ("d_eff_low", "d_eff_low"),
                         ("d_eff_high", "d_eff_high"), ("r2", "r2"),
                         ("drift_p_value", "drift_p")))
        rows = {}
        for n_t in (500, 1000, 2000, 4000):
            t = law.floor_at(n_t)
            r = law.required_n_range(float(t))
            rows[f"target_floor_at_n={n_t}"] = {
                "target_floor": float(t),
                "wls": {kk: r[kk] for kk in
                        ("n_low_fit", "n_high_fit", "n_full_fit", "ratio",
                         "d_eff_low", "d_eff_high", "ok")}}
        worst = max(rows[kk]["wls"]["ratio"] for kk in rows)
        out["tables"].append({
            "dataset": rec["dataset"],
            "d_eff": float(rec["d_eff"]), "r2": float(rec["r2"]),
            "drift": float(rec["drift"]), "drift_p": float(rec["drift_p"]),
            "single_exponent_ok": bool(rec["single_exponent_ok"]),
            "reproduces_stored_fit": bool(consistent),
            "targets": rows,
            "worst_ratio_wls": float(worst)})
        print(f"{rec['dataset']:<20} {worst:>20.2f}x")
    art.save("half_fit_ranges.json", out, indent=2)


if __name__ == "__main__":
    main()
