"""Cost of one exact two-sample W2 (Section 3.5, Appendix B.2).

Wall time of a dense squared-Euclidean cost matrix followed by an exact
linear assignment (scipy linear_sum_assignment) on standard normal samples,
for n in {512, ..., 8192} per side at d = 8 and d = 32; the median of the
repeats is reported, and the growth exponent in n is fitted for n >= 1024.
The cost matrix occupies 8 n^2 bytes.

    python experiments/ot_cost.py            # about a minute
    python experiments/ot_cost.py --quick    # n <= 4096

Writes results/ot_cost.json. Timings depend on the machine; the artifact
records the processor.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import _artifacts as art


def main(argv):
    ns = (512, 1024, 2048, 4096) if "--quick" in argv else (512, 1024, 2048,
                                                          4096, 8192)
    rows = []
    rng = np.random.default_rng(20260930)
    for d in (8, 32):
        for n in ns:
            ts = []
            for _ in range(3 if n <= 4096 else 2):
                X, Y = rng.normal(size=(n, d)), rng.normal(size=(n, d))
                t0 = time.perf_counter()
                C = cdist(X, Y, "sqeuclidean")
                i, j = linear_sum_assignment(C)
                float(np.sqrt(C[i, j].mean()))
                ts.append(time.perf_counter() - t0)
            rows.append({"d": d, "n": n, "seconds_median": float(np.median(ts)),
                         "seconds_all": [round(t, 3) for t in ts],
                         "cost_matrix_GB": 8.0 * n * n / 1e9})
            print(f"d={d:>3} n={n:>5}  {np.median(ts):8.2f} s", flush=True)
    fits = {}
    for d in (8, 32):
        r = [x for x in rows if x["d"] == d and x["n"] >= 1024]
        b = np.polyfit(np.log([x["n"] for x in r]),
                       np.log([x["seconds_median"] for x in r]), 1)[0]
        fits[str(d)] = float(b)
    art.save("ot_cost.json", {
        "_definition": ("wall time of one exact two-sample W2 (dense "
                        "squared-Euclidean cost matrix and scipy "
                        "linear_sum_assignment), median of repeats"),
        "cpu": art.processor_name(), "python": sys.version.split()[0],
        "numpy": np.__version__, "cells": rows,
        "time_exponent_in_n_(n>=1024)": fits})


if __name__ == "__main__":
    main(sys.argv[1:])
