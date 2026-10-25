"""Floor laws and dimension estimates of the eleven benchmark cells
(Section 8.2).

Each cell (experiments/_benchmarks.py) is standardised and subsampled to
8000 rows (seed 0). The floor law is fitted by certgen.fit_floor_law on the
six-point grid {125, ..., 4000} with n_rep = 12, and the three dimension
estimators are run by certgen.intrinsic_dimension with eight re-splits and a
six-fold subsample bootstrap. The stored sweep (grid, floor medians,
standard deviations and bootstrap standard errors of the log-medians) lets
every later analysis be recomputed without the data.

    python experiments/benchmark_floor_laws.py                 # about 25 minutes
    python experiments/benchmark_floor_laws.py --only=adult:E1 --merge

Downloads the tables from OpenML on first use. Writes
results/benchmark_floor_laws.json.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import _artifacts as art
import certgen as cg
from _benchmarks import CELLS, Tables, subsample

MAX_N, N_REP, N_BOOT, SEED = 8000, 12, 8, 0
DATA_BOOT = 6


def tag_of(cell: str) -> str:
    """Record name of a cell; covertype has identical E1 and E2 encodings."""
    return "covertype:E1=E2" if cell == "covertype:E1" else cell


def measure(X: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    X = cg.Standardizer().fit_transform(subsample(X, MAX_N, SEED))
    law = cg.fit_floor_law(X, metric="w2", n_rep=N_REP, seed=SEED)
    dim = cg.intrinsic_dimension(X, n_boot=N_BOOT, seed=SEED,
                                 data_bootstrap=DATA_BOOT)
    print(law.summary())
    print(dim.summary())
    return {
        "dataset": tag, "n": int(len(X)), "d_ambient": int(X.shape[1]),
        "C": law.C, "d_eff": law.d_eff, "d_eff_low": law.d_eff_low,
        "d_eff_high": law.d_eff_high, "r2": law.r2,
        "drift": law.exponent_drift, "drift_p": law.drift_p_value,
        "single_exponent_ok": bool(law.single_exponent_ok),
        "n_grid": [int(v) for v in law.n_grid],
        "floors": [float(v) for v in law.floors],
        "floor_stds": [float(v) for v in law.stds],
        "floor_cv": [float(a / b) for a, b in zip(law.stds, law.floors)],
        "floor_se_log": ([float(v) for v in law.se_log]
                         if law.se_log is not None
                         and np.all(np.isfinite(np.asarray(law.se_log, float)))
                         else None),
        "d_wasserstein": dim.wasserstein, "d_W_sd": dim.wasserstein_sd,
        "d_W_se_mc": dim.wasserstein_se,
        "d_W_sd_subsample": dim.wasserstein_sd_subsample,
        "d_W_sd_subsample_scaled": dim.wasserstein_sd_subsample_scaled,
        "d_W_subsample_m": dim.subsample_m,
        "exponent": dim.exponent, "exponent_se": dim.exponent_se,
        "d_twonn": dim.twonn, "d_mle": dim.mle, "spread": dim.spread,
        "tie_fraction": dim.tie_fraction,
        "lattice_suspected": bool(dim.lattice_suspected),
        "usable_for_sizing": bool(dim.usable_for_sizing),
        "rate_regime": dim.rate_regime,
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv):
    only = None
    for a in argv:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
    tables, recs, pending = Tables(), [], []
    for cell, source, enc in CELLS:
        tag = tag_of(cell)
        if only is not None and tag not in only:
            continue
        print(f"\n### {tag}", flush=True)
        try:
            X = tables.encoded(source, enc)
        except Exception as e:  # recorded in the artifact
            print(f"  not loaded: {type(e).__name__}: {e}", flush=True)
            pending.append({"dataset": tag,
                            "reason": f"load failed: {type(e).__name__}: {e}"})
            continue
        recs.append(measure(X, tag))
    out = {"_definition": ("floor law and dimension estimates of each "
                           "benchmark cell; the stored sweep allows offline "
                           "recomputation of every derived quantity"),
           "_when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "_config": dict(max_n=MAX_N, n_rep=N_REP, n_boot=N_BOOT, seed=SEED,
                           metric="w2", data_bootstrap=DATA_BOOT),
           "pending": pending,
           "records": recs}
    if "--merge" in argv and art.exists("benchmark_floor_laws.json"):
        prev = art.load("benchmark_floor_laws.json")
        keep = [r for r in prev.get("records", [])
                if r["dataset"] not in {x["dataset"] for x in recs}]
        out["records"] = sorted(keep + recs, key=lambda r: r["dataset"])
    art.save("benchmark_floor_laws.json", out, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
