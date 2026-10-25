"""Diagnostics of the benchmark data (Sections 8.1 and 8.4).

exclusions    for each candidate excluded by the screening (electricity,
              eeg-eye-state, MiniBooNE): the E1 encoding capped at 8000 rows
              (seed 0) and standardised, the largest absolute z-score, the
              median W2 between disjoint splits at n = 1000, 2000 and 4000
              (12 replicates), and the floor law of certgen.fit_floor_law;
correlations  pairwise absolute correlations of the six numeric columns of
              adult on the data path of the resolution atlas (20000 rows,
              seed 0).

    python experiments/benchmark_diagnostics.py [--only=electricity,adult]

Downloads the tables from OpenML on first use. Writes
results/benchmark_diagnostics.json; with --only, only the named parts are
recomputed and the others are kept.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

import _artifacts as art
import certgen as cg
from certgen.discrepancy import w2_exact
from _benchmarks import EXCLUDED, Tables, subsample

warnings.filterwarnings("ignore")
NAME = "benchmark_diagnostics.json"


def exclusion(tables, source):
    X = cg.Standardizer().fit_transform(
        np.asarray(subsample(tables.encoded(source, "E1"), 8000, 0), float))
    rng = np.random.default_rng(0)
    med = {}
    for s in (1000, 2000, 4000):
        vals = [w2_exact(X[idx[:s]], X[idx[s:]], max_n=s, seed=r) for r, idx in
                [(r, rng.permutation(len(X))[:2 * s]) for r in range(12)]]
        med[str(s)] = float(np.median(vals))
    out = {"rows": int(len(X)), "columns": int(X.shape[1]),
           "max_abs_z": float(np.abs(X).max()),
           "w2_medians_by_side": med, "n_rep": 12}
    try:
        law = cg.fit_floor_law(X, n_rep=12, seed=0)
        out["floor_law"] = {"n_grid": [int(v) for v in law.n_grid],
                            "floors": [float(v) for v in law.floors],
                            "d_eff": float(law.d_eff), "r2": float(law.r2)}
    except ValueError as e:
        out["floor_law"] = {"error": str(e)[:200]}
    return out


def correlations(tables):
    X = subsample(tables.encoded("adult", "E1"), 20000, 0)
    A = np.abs(np.corrcoef(np.asarray(X, float), rowvar=False))
    tri = A[np.triu_indices_from(A, k=1)]
    return {"table": "adult:E1",
            "max_pairwise_abs_corr": float(tri.max()),
            "median_pairwise_abs_corr": float(np.median(tri)),
            "n_columns": int(X.shape[1]), "n_rows": int(len(X))}


def main(argv):
    only = None
    for a in argv:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
    out = (art.load(NAME) if only and art.exists(NAME) else
           {"_definition": ("exclusion diagnostics of the candidate tables set "
                            "aside before the study, and the correlation "
                            "structure of adult:E1"),
            "exclusions": {}})
    tables = Tables()
    for tag, source, _ in EXCLUDED:
        if only is not None and source not in only:
            continue
        print(f"exclusion candidate {tag}", flush=True)
        out["exclusions"][tag] = exclusion(tables, source)
        print(f"  medians {out['exclusions'][tag]['w2_medians_by_side']}")
    if only is None or "adult" in only:
        out["correlations"] = correlations(tables)
        print(f"adult:E1 |corr| max {out['correlations']['max_pairwise_abs_corr']:.3f}, "
              f"median {out['correlations']['median_pairwise_abs_corr']:.3f}")
    art.save(NAME, out, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
