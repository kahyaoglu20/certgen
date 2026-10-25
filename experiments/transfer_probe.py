"""Transfer of the reference tables to heavy-tailed and lattice laws
(Section 6.5, Appendix B.5).

The protocol of transfer_smooth_laws.py (fixed law geometry, a same-law
guard, r over four sample pairs, d_hat on an independent sample, residual
against floor_ratio_measured) is applied to

  heavy-tailed  multivariate Student-t (nu = 3, 5) with the anisotropic
                decay-0.90 shape, and a rotated Laplace law;
  lattice       the factor models (k = 20, 28) rounded to a grid of step
                0.5 or 1 in every coordinate;
  mixed type    the k = 20 factor model with half of the coordinates rounded.

Verdict rule, fixed before the run: a family transfers if every residual in
the band 9 <= d_hat <= 28 lies within +/-12%.

    python experiments/transfer_probe.py

Writes results/transfer_probe.json (under a minute).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr

import _artifacts as art
import certgen as cg
from certgen.dimension import wasserstein_dimension
from transfer_smooth_laws import (AnisotropicGaussian, FactorModel,
                                  same_law_check, two_sample_r)

N_GRID = (256, 1024)
N_REP = 4
N_REP_DHAT = 3
SEED = 20260931
BAND = (9.0, 28.0)
TRANSFER_TOL = 0.12


class StudentT:
    """Multivariate t with the shape of AnisotropicGaussian(decay)."""

    def __init__(self, nu: float, D: int = 64, decay: float = 0.90, law_seed: int = 0):
        self.base = AnisotropicGaussian(D=D, decay=decay, law_seed=law_seed)
        self.nu = nu

    def __call__(self, n, seed):
        Z = self.base(n, seed)
        w = np.random.default_rng(70_000 + seed).chisquare(self.nu, size=n)
        return Z / np.sqrt(w / self.nu)[:, None]


class RotLaplace:
    """Product Laplace law with decaying scales in a fixed rotated basis."""

    def __init__(self, D: int = 64, decay: float = 0.90, law_seed: int = 0):
        g = np.random.default_rng(80_000 + law_seed)
        self.sd = np.sqrt(decay ** np.arange(D)) / np.sqrt(2.0)
        self.Q, _ = np.linalg.qr(g.normal(size=(D, D)))
        self.D = D

    def __call__(self, n, seed):
        r = np.random.default_rng(90_000 + seed)
        return (r.laplace(size=(n, self.D)) * self.sd) @ self.Q.T


class Quantised:
    """The first `frac` of the coordinates of `base` rounded to step h."""

    def __init__(self, base, h: float, frac: float = 1.0):
        self.base, self.h, self.frac = base, h, frac

    def __call__(self, n, seed):
        X = self.base(n, seed)
        k = int(round(self.frac * X.shape[1]))
        X = X.copy()
        X[:, :k] = np.round(X[:, :k] / self.h) * self.h
        return X


def rank_same_law_check(draw, N=1024, seed=0) -> float:
    """Same-law guard on rank correlations, for laws without finite fourth
    moments."""
    A, B = draw(N, seed), draw(N, seed + 1)
    Ra = spearmanr(A).correlation
    Rb = spearmanr(B).correlation
    return float(np.linalg.norm(Ra - Rb) / max(np.linalg.norm(Ra), 1e-15))


def make_laws():
    laws = []
    for nu in (3.0, 5.0):
        laws.append((f"student_t_nu{int(nu)}", StudentT(nu, law_seed=int(nu)),
                     {"family": "heavy_tailed", "kind": "student_t", "nu": nu,
                      "D": 64, "decay": 0.90}))
    laws.append(("laplace_rot", RotLaplace(law_seed=1),
                 {"family": "heavy_tailed", "kind": "laplace", "D": 64,
                  "decay": 0.90}))
    for k in (20, 28):
        for h in (0.5, 1.0):
            laws.append((f"factor_k{k}_lattice_h{h}",
                         Quantised(FactorModel(k=k, D=64, noise=0.25, law_seed=k), h),
                         {"family": "lattice", "k": k, "h": h, "D": 64}))
    laws.append(("factor_k20_mixed_h1.0",
                 Quantised(FactorModel(k=20, D=64, noise=0.25, law_seed=20), 1.0, 0.5),
                 {"family": "mixed_type", "k": 20, "h": 1.0, "frac_quantised": 0.5,
                  "D": 64}))
    return laws


def main():
    t0 = time.time()
    cells = []
    print(f"{'law':<26} {'N':>5} {'d_hat':>7} {'r':>8} {'table':>8} {'resid':>8}")
    for li, (name, draw, meta) in enumerate(make_laws()):
        cov_chk = same_law_check(draw)
        rank_chk = rank_same_law_check(draw)
        guard = rank_chk if meta.get("nu", 99) <= 3 else cov_chk
        meta = dict(meta, same_law_cov_diff=cov_chk, same_law_rank_diff=rank_chk)
        if guard > 0.45:
            raise RuntimeError(f"{name}: same-law guard failed ({guard:.2f})")
        for N in N_GRID:
            s0 = SEED + 10_000 * li + N
            cr = two_sample_r(draw, N, N_REP, s0)
            Xd = draw(N, s0 + 777)
            try:
                dhat = float(wasserstein_dimension(Xd, n_rep=N_REP_DHAT, seed=0))
            except RuntimeError as e:      # W2 not decreasing in n
                dhat = float("nan")
                meta = dict(meta, dhat_error=str(e)[:120])
            r_tbl = (float(cg.floor_ratio_measured(dhat, N)) if np.isfinite(dhat)
                     else float("nan"))
            resid = cr["r_mean"] / r_tbl - 1.0 if np.isfinite(r_tbl) else float("nan")
            cells.append({"law": name, "meta": meta, "N": N, "d_hat": dhat,
                          "r": cr, "table_r_at_dhat": r_tbl,
                          "residual_vs_table_at_dhat": resid,
                          "tie_fraction": float(cg.tie_fraction(Xd)),
                          "in_band": bool(np.isfinite(dhat)
                                          and BAND[0] <= dhat <= BAND[1])})
            print(f"{name:<26} {N:>5} {dhat:>7.2f} {cr['r_mean']:>8.4f} "
                  f"{r_tbl:>8.4f} {100 * resid:>+7.1f}%", flush=True)
    fams = {}
    for c in cells:
        fams.setdefault(c["meta"]["family"], []).append(c)
    verdict = {}
    for f, cs in fams.items():
        band = [c["residual_vs_table_at_dhat"] for c in cs if c["in_band"]]
        allr = [c["residual_vs_table_at_dhat"] for c in cs
                if np.isfinite(c["residual_vs_table_at_dhat"])]
        verdict[f] = {
            "cells": len(cs), "band_cells": len(band),
            "band_median": float(np.median(band)) if band else float("nan"),
            "band_range": [float(min(band)), float(max(band))] if band else [],
            "all_range": [float(min(allr)), float(max(allr))] if allr else [],
            "transfers": bool(band and max(abs(x) for x in band) <= TRANSFER_TOL)}
    art.save("transfer_probe.json", {
        "_definition": ("floor ratio on heavy-tailed, lattice and mixed-type "
                        "laws; residual = r / floor_ratio_measured(d_hat, N) - 1"),
        "_verdict_rule": (f"a family transfers if every residual with d_hat in "
                          f"{list(BAND)} lies within +/-{TRANSFER_TOL:.0%}"),
        "config": {"N_grid": list(N_GRID), "n_rep": N_REP,
                   "n_rep_dhat": N_REP_DHAT, "seed": SEED, "band": list(BAND)},
        "cells": cells, "families": verdict,
        "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
