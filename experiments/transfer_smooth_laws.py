"""Transfer of the reference tables to non-isotropic smooth laws
(Section 6.5, Appendix B.5).

The floor ratio r = W2(X_N, X'_N) / sqrt(tr Cov) is measured on three
families of laws in 64 ambient dimensions,

  anisotropic Gaussians  eigenvalues decaying geometrically with ratio
                         0.95, 0.90 or 0.85, in a random rotated basis;
  factor models          k in {20, 28} Gaussian factors, random loadings,
                         isotropic noise of standard deviation 0.25;
  curved manifolds       images of m in {16, 24, 32} dimensional Gaussian
                         latents under a fixed smooth random embedding,

at N in {256, 1024} with four replicates, together with the slope estimate
d_hat on an independent sample. The geometry of each law is drawn once, and
the covariances of two samples are compared before use. The residual
r / floor_ratio_measured(d_hat, N) - 1 is reported for every cell, and
isotropic Gaussians in dimensions 16, 22 and 25 measure the noise of the
reference itself.

    python experiments/transfer_smooth_laws.py

Writes results/transfer_smooth_laws.json (under a minute).
"""
from __future__ import annotations

import time

import numpy as np

import _artifacts as art
import certgen as cg
from certgen.discrepancy import w2_exact
from certgen.dimension import wasserstein_dimension

N_GRID = (256, 1024)
N_REP = 4
N_REP_DHAT = 3
SEED = 20260904
MAX_N_OT = 1024
BAND = (12.0, 28.0)
SAME_LAW_TOL = 0.45

_LIMITS = {N: cg.certification_limit(N, with_uncertainty=False)
           for N in (256, 1024, 2048)}
CROSSING_MEASURED = {N: round(c["d_star_limit_measured"], 1)
                     for N, c in _LIMITS.items()}
CROSSING_NOMINAL = {N: round(c["d_star_limit"], 1) for N, c in _LIMITS.items()}


def scale_trcov(X: np.ndarray) -> float:
    Xc = X - X.mean(0)
    return float(np.sqrt(np.mean(np.sum(Xc ** 2, axis=1))))


class AnisotropicGaussian:
    """Gaussian with eigenvalues decay**k in a fixed random rotated basis."""

    def __init__(self, D: int = 64, decay: float = 0.9, law_seed: int = 0):
        g = np.random.default_rng(30_000 + law_seed)
        self.D, self.decay = D, decay
        self.sd = np.sqrt(decay ** np.arange(D))
        self.Q, _ = np.linalg.qr(g.normal(size=(D, D)))

    def __call__(self, n: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(40_000 + seed)
        return (r.normal(size=(n, self.D)) * self.sd) @ self.Q.T


class FactorModel:
    """Gaussian factor model with a fixed random loading matrix."""

    def __init__(self, k: int = 20, D: int = 64, noise: float = 0.25,
                 law_seed: int = 0):
        g = np.random.default_rng(50_000 + law_seed)
        self.A = g.normal(size=(k, D)) / np.sqrt(k)
        self.k, self.D, self.noise = k, D, noise

    def __call__(self, n: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(60_000 + seed)
        return (r.normal(size=(n, self.k)) @ self.A
                + self.noise * r.normal(size=(n, self.D)))


def same_law_check(draw, N: int = 1024, seed: int = 0) -> float:
    """Relative Frobenius difference of the covariances of two samples."""
    A, B = draw(N, seed), draw(N, seed + 1)
    Ca, Cb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    return float(np.linalg.norm(Ca - Cb) / max(np.linalg.norm(Ca), 1e-15))


def make_laws():
    """(name, draw(n, seed), metadata) for every law of the study."""
    laws = []
    for decay in (0.95, 0.90, 0.85):
        laws.append((f"aniso_decay{decay:.2f}",
                     AnisotropicGaussian(D=64, decay=decay,
                                         law_seed=int(decay * 100)),
                     {"family": "anisotropic_gaussian", "D": 64,
                      "decay": decay}))
    for k in (20, 28):
        laws.append((f"factor_k{k}",
                     FactorModel(k=k, D=64, noise=0.25, law_seed=k),
                     {"family": "factor_model", "k": k, "D": 64,
                      "noise": 0.25}))
    for m in (16, 24, 32):
        man = cg.data.curved_manifold(m, D=64, seed=m)

        def make(man=man):
            def draw(n, seed):
                return man.sample(n, seed=seed)
            return draw

        laws.append((f"curved_m{m}", make(), {"family": "curved_manifold",
                                              "m": m, "D": 64}))
    return laws


def two_sample_r(draw_fn, N: int, n_rep: int, seed0: int) -> dict:
    rs, w2s, scales = [], [], []
    for r in range(n_rep):
        X = draw_fn(N, seed0 + 2 * r)
        Xp = draw_fn(N, seed0 + 2 * r + 1)
        scale = scale_trcov(np.vstack([X, Xp]))
        w2 = w2_exact(X, Xp, max_n=min(N, MAX_N_OT), seed=r)
        w2s.append(float(w2))
        scales.append(float(scale))
        rs.append(float(w2 / max(scale, 1e-15)))
    rs = np.asarray(rs, float)
    return {"r_mean": float(rs.mean()),
            "r_sd": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
            "w2_mean": float(np.mean(w2s)),
            "scale_mean": float(np.mean(scales)),
            "n_rep": int(n_rep),
            "values": [float(v) for v in rs]}


def main():
    t0 = time.time()
    cells = []
    print(f"{'law':<22} {'N':>5} {'d_hat':>7} {'r':>8} {'table':>8} {'resid':>8}")
    for li, (name, draw, meta) in enumerate(make_laws()):
        slc = same_law_check(draw)
        meta = dict(meta, same_law_cov_diff=slc)
        if slc > SAME_LAW_TOL:
            raise RuntimeError(f"law '{name}': the covariances of two samples "
                               f"differ by {slc:.2f} (tolerance {SAME_LAW_TOL})")
        for N in N_GRID:
            seed0 = SEED + 10_000 * li + N
            cell_r = two_sample_r(draw, N, N_REP, seed0)
            dhat = float(wasserstein_dimension(draw(N, seed0 + 777),
                                               n_rep=N_REP_DHAT, seed=0))
            r_tbl = float(cg.floor_ratio_measured(dhat, N))
            nom_key = meta.get("m") or meta.get("k")
            r_nom = float(cg.floor_ratio(nom_key, N)) if nom_key else float("nan")
            resid = cell_r["r_mean"] / max(r_tbl, 1e-15) - 1.0
            crossing = CROSSING_MEASURED[N]
            cells.append({
                "law": name, "meta": meta, "N": int(N), "d_hat": dhat,
                "r": cell_r, "table_r_at_dhat": r_tbl,
                "table_r_at_nominal_label": r_nom,
                "residual_vs_table_at_dhat": float(resid),
                "crossing_measured": crossing,
                "dhat_over_crossing": float(dhat / crossing),
                "near_crossing": bool(abs(dhat - crossing) / crossing < 0.35),
                "in_target_dhat_band": bool(BAND[0] <= dhat <= BAND[1])})
            print(f"{name:<22} {N:>5} {dhat:>7.2f} {cell_r['r_mean']:>8.4f} "
                  f"{r_tbl:>8.4f} {100 * resid:>+7.1f}%", flush=True)

    iso_cells = []
    for d in (16, 22, 25):
        for N in N_GRID:
            def gauss(n, s, d=d):
                return np.random.default_rng(s).normal(size=(n, d))
            g = two_sample_r(gauss, N, N_REP, SEED + 50_000 + 100 * d + N)
            Xg = gauss(N, SEED + 51_000 + 100 * d + N)
            dhat = float(wasserstein_dimension(Xg, n_rep=N_REP_DHAT, seed=0))
            r_tbl = float(cg.floor_ratio_measured(dhat, N))
            iso_cells.append({
                "law": f"isotropic_d{d}", "d_ambient": d, "N": int(N),
                "d_hat": dhat, "r_mean": g["r_mean"], "table_r_at_dhat": r_tbl,
                "residual_vs_table_at_dhat": float(g["r_mean"] / max(r_tbl, 1e-15) - 1.0)})

    band = [c for c in cells if c["in_target_dhat_band"]]
    residuals = [c["residual_vs_table_at_dhat"] for c in band]
    out = {
        "_definition": ("floor ratio r = W2(X_N, X'_N) / sqrt(tr Cov) on "
                        "non-isotropic laws and its residual against the "
                        "isotropic reference table at the measured d_hat"),
        "_config": dict(n_grid=list(N_GRID), n_rep=N_REP, n_rep_dhat=N_REP_DHAT,
                        seed=SEED, max_n_ot=MAX_N_OT,
                        crossing_measured=CROSSING_MEASURED,
                        crossing_nominal=CROSSING_NOMINAL,
                        target_dhat_band=list(BAND)),
        "_elapsed_s": round(time.time() - t0, 1),
        "cells": cells,
        "isotropic_reference_cells": iso_cells,
        "near_crossing_cells": [c["law"] + f"@N={c['N']}" for c in cells
                                if c["near_crossing"]],
        "band_residual_summary": {
            "n_cells": len(band),
            "median_residual": float(np.median(residuals)) if residuals else None,
            "mean_residual": float(np.mean(residuals)) if residuals else None,
            "min_residual": float(np.min(residuals)) if residuals else None,
            "max_residual": float(np.max(residuals)) if residuals else None,
            "residuals": [float(x) for x in residuals]},
    }
    s = out["band_residual_summary"]
    print(f"band {BAND}: {s['n_cells']} cells, median residual "
          f"{100 * s['median_residual']:+.1f}%, range "
          f"[{100 * s['min_residual']:+.1f}%, {100 * s['max_residual']:+.1f}%]")
    art.save("transfer_smooth_laws.json", out, indent=2)


if __name__ == "__main__":
    main()
