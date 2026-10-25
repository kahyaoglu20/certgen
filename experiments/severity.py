"""Severity thresholds of the benchmark cells with their combined interval
(Sections 4.5 and 8.3).

For each cell of results/benchmark_floor_laws.json the fitted floor law is
rebuilt from the stored sweep, and FloorLaw.severity is run for eight seeds
(drift_boot = 600, trials = 200). The per-seed thresholds drift_50 and their
binomial intervals are combined into a 90% interval,

    median(t_s) +/- z_0.95 sqrt(var_within + var_between),

with var_within the mean squared binomial half-width / z_0.95 and
var_between the sample variance of the thresholds across seeds. The
artifact also records whether the observed drift of each cell lies inside
its interval.

    python experiments/severity.py [--seeds 8] [--boot 600]

Writes results/severity.json (under a minute; no download).
"""
from __future__ import annotations

import argparse

import numpy as np

import _artifacts as art
from _benchmarks import law_from_record
from certgen.design import combine_severity


def law_from(rec, n_rep=12):
    """The FloorLaw of a stored profile, with its bootstrap standard errors."""
    return law_from_record(rec, n_rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    src = art.load("benchmark_floor_laws.json")
    n_rep = int(src["_config"]["n_rep"])
    print(f"seeds {seeds}, drift_boot={args.boot}, trials={args.trials}")
    print(f"{'cell':<20} {'drift':>7} {'drift_50':>9} {'90% interval':>18} "
          f"{'inflation':>9} {'inside':>7}")
    out = []
    for rec in src["records"]:
        law = law_from(rec, n_rep)
        ts, cis, obs = [], [], float("nan")
        for sd in seeds:
            sv = law.severity(drift_boot=args.boot, trials=args.trials, seed=sd)
            ts.append(sv.get("drift_50", float("nan")))
            cis.append(sv.get("drift_50_ci90"))
            obs = sv.get("drift_observed", obs)
        comb = combine_severity(ts, cis)
        inside = (np.isfinite(comb["lo"]) and comb["lo"] <= obs <= comb["hi"])
        if np.isfinite(comb["median"]):
            print(f"{rec['dataset']:<20} {obs:7.3f} {comb['median']:9.3f} "
                  f"[{comb['lo']:7.3f},{comb['hi']:7.3f}] "
                  f"{comb['inflation']:9.2f} {str(inside):>7}")
        else:
            print(f"{rec['dataset']:<20} {obs:7.3f}   undefined (not admissible)")
        out.append({"dataset": rec["dataset"],
                    "drift_observed": obs,
                    "per_seed_drift_50": [None if not np.isfinite(x) else x
                                          for x in ts],
                    "drift_50_median": comb["median"],
                    "sd_within_binomial": comb["sd_within"],
                    "sd_between_seed": comb["sd_between"],
                    "sd_total": comb["sd_total"],
                    "drift_50_ci90_combined": [comb["lo"], comb["hi"]],
                    "width_binomial_only": comb["width_binomial_only"],
                    "width_combined": comb["width_combined"],
                    "interval_inflation_vs_binomial_only": comb["inflation"],
                    "observed_drift_inside_combined_ci": bool(inside)
                    if np.isfinite(comb["median"]) else None,
                    "n_seeds": comb["n_seeds"]})
    art.save("severity.json", {
        "_definition": ("severity threshold drift_50 of each benchmark cell "
                        "over a panel of seeds, with the combined binomial and "
                        "between-seed 90% interval"),
        "_source": "benchmark_floor_laws.json",
        "settings": {"seeds": list(seeds), "drift_boot": args.boot,
                     "trials": args.trials},
        "combination_rule": (
            "sd_total = sqrt(mean_s(((hi_s-lo_s)/(2*z90))^2) + var_s(t_s)); "
            "interval = median(t_s) +/- z90*sd_total, z90 = 1.6449"),
        "tables": out}, indent=2)


if __name__ == "__main__":
    main()
