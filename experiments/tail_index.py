"""Tail-index estimates of the benchmark cells and of reference laws
(Section 6.5).

The Hill estimate (certgen.tail_index_hill, largest 5% of the distances from
the median) is computed on an 8000-row subsample (seed 0) of every benchmark
cell, on the data path of the resolution atlas, and on four reference laws
of the transfer studies (four seeds each).

    python experiments/tail_index.py

Downloads the tables from OpenML on first use. Writes
results/tail_index.json.
"""
from __future__ import annotations

import numpy as np

import _artifacts as art
import certgen as cg
import resolution_atlas as atlas
from transfer_probe import RotLaplace, StudentT
from transfer_smooth_laws import AnisotropicGaussian

N_SUB = 8000


def main():
    rows = []
    for tag, X, err in atlas.load():
        if X is None:
            rows.append({"table": tag, "error": err})
            continue
        g = np.random.default_rng(0)
        Xs = X[g.choice(len(X), min(N_SUB, len(X)), replace=False)]
        t = cg.tail_index_hill(Xs)
        rows.append({"table": tag, "rows": int(len(Xs)), "tail_index_hill": t,
                     "max_abs_z": float(np.abs(Xs).max()),
                     "below_threshold": bool(t < cg.HEAVY_TAIL_INDEX)})
        print(f"{tag:<20} {t:6.2f}", flush=True)
    refs = []
    for name, law in (("student_t_nu3", StudentT(3.0, law_seed=3)),
                      ("student_t_nu5", StudentT(5.0, law_seed=5)),
                      ("laplace_rot", RotLaplace(law_seed=1)),
                      ("aniso_gaussian_0.90", AnisotropicGaussian(64, 0.9, 1))):
        v = [cg.tail_index_hill(law(N_SUB, s)) for s in range(4)]
        refs.append({"law": name, "tail_index_hill": v,
                     "median": float(np.median(v))})
        print(f"{name:<20} {np.median(v):6.2f}", flush=True)
    t = [r["tail_index_hill"] for r in rows if "tail_index_hill" in r]
    art.save("tail_index.json", {
        "_definition": ("Hill estimate (largest 5%) of the tail index of "
                        "||x - median|| on standardised 8000-row subsamples of "
                        "the benchmark cells and on reference laws"),
        "threshold": cg.HEAVY_TAIL_INDEX, "tables": rows,
        "reference_laws": refs,
        "summary": {"min": float(min(t)), "max": float(max(t)),
                    "below_threshold": [r["table"] for r in rows
                                        if r.get("below_threshold")]}})


if __name__ == "__main__":
    main()
