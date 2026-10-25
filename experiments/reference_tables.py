"""Reference tables of the floor ratio and of the slope estimate
(Sections 6.3 and 6.4, Appendix B.4).

On isotropic Gaussian laws N(0, I_d), with 20 replicates per cell:

  r(d, N)      W2(X_N, X'_N) / sqrt(d) for two independent samples of N rows,
               exact assignment, d in {1, 2, 4, ..., 128};
  d_hat(d, N)  the slope estimate certgen.wasserstein_dimension(X, n_rep=3)
               on one sample of N rows, d in {2, 4, ..., 128};

for N in {256, 1024, 2048, 4096, 8192}. Larger N are measured by
reference_tables_large_n.py with the same protocol and merged into the same
artifact. For every N the crossing r = 1 is located in nominal dimension by
log-log interpolation, mapped to measured units through the d_hat table, and
given a 90% interval from 2000 recomputations with every cell of both tables
perturbed by its standard error.

The artifact is rewritten after every cell, so an interrupted run resumes.

    python experiments/reference_tables.py            # about 1.5 CPU hours
    python experiments/reference_tables.py --quick    # N <= 2048

Writes results/reference_tables.json; tools/write_library_tables.py copies
the tables into certgen/latent.py.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import _artifacts as art
from certgen.discrepancy import w2_exact
from certgen.dimension import wasserstein_dimension
from certgen.latent import crossing_loglog

D_GRID_R = (1, 2, 4, 8, 16, 32, 64, 128)
D_GRID_DHAT = (2, 4, 8, 16, 32, 64, 128)
N_GRID = (256, 1024, 2048, 4096, 8192)
N_QUICK = (256, 1024, 2048)
N_REP_R = 20
N_REP_DHAT = 20
SEED = 20260930
NAME = "reference_tables.json"


def _rng(*key):
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def r_cell(N: int, d: int, n_rep: int) -> dict:
    vals = []
    for r in range(n_rep):
        g = _rng(1, N, d, r)
        X = g.normal(size=(N, d))
        Y = g.normal(size=(N, d))
        vals.append(w2_exact(X, Y, max_n=N, seed=r) / np.sqrt(d))
    v = np.asarray(vals, float)
    return {"N": N, "d": d, "n_rep": n_rep, "r_mean": float(v.mean()),
            "r_sd": float(v.std(ddof=1)), "r_se": float(v.std(ddof=1) / np.sqrt(n_rep)),
            "values": [float(x) for x in v]}


def dhat_cell(N: int, d: int, n_rep: int) -> dict:
    vals = []
    for r in range(n_rep):
        g = _rng(2, N, d, r)
        X = g.normal(size=(N, d))
        vals.append(wasserstein_dimension(X, n_rep=3, seed=r))
    v = np.asarray(vals, float)
    return {"N": N, "d": d, "n_rep": n_rep, "d_hat_mean": float(v.mean()),
            "d_hat_sd": float(v.std(ddof=1)),
            "d_hat_se": float(v.std(ddof=1) / np.sqrt(n_rep)),
            "values": [float(x) for x in v]}


def _load():
    return art.load(NAME) if art.exists(NAME) else {"r_cells": [], "dhat_cells": []}


def _save(state):
    """Checkpoint the artifact, with the cells in (N, d) order whatever the
    order in which they were measured."""
    for key in ("r_cells", "dhat_cells"):
        state[key].sort(key=lambda c: (c["N"], c["d"]))
    art.save(NAME, state, quiet=True)


def _have(cells, N, d):
    return any(c["N"] == N and c["d"] == d for c in cells)


def crossing(r_tab: dict, dhat_tab: dict, N: int, rng=None) -> tuple:
    """Largest nominal d with r(d, N) < 1 (log-log interpolation in d), and
    the same crossing mapped to measured units through the d-hat table.
    r_tab: {d: r}; dhat_tab: {d: d_hat}."""
    ds = np.array(sorted(r_tab), float)
    rs = np.array([r_tab[int(k)] for k in ds], float)
    lim = crossing_loglog(ds, rs)
    if not np.isfinite(lim):
        return float("nan"), float("nan")
    dn = np.array(sorted(dhat_tab), float)
    dh = np.array([dhat_tab[int(k)] for k in dn], float)
    lim_meas = float(np.exp(np.interp(np.log(lim), np.log(dn), np.log(dh))))
    return lim, lim_meas


def summarise(state):
    """Crossings r = 1 of every complete N, with 90% intervals."""
    rc = {(c["N"], c["d"]): c for c in state["r_cells"]}
    dc = {(c["N"], c["d"]): c for c in state["dhat_cells"]}
    Ns = sorted({c["N"] for c in state["r_cells"]})
    cross = []
    for N in Ns:
        r_tab = {d: rc[(N, d)]["r_mean"] for d in D_GRID_R if (N, d) in rc}
        d_tab = {d: dc[(N, d)]["d_hat_mean"] for d in D_GRID_DHAT if (N, d) in dc}
        if len(r_tab) < len(D_GRID_R) or len(d_tab) < len(D_GRID_DHAT):
            continue
        lim, lim_m = crossing(r_tab, d_tab, N)
        g = _rng(9, N)
        draws_n, draws_m = [], []
        for _ in range(2000):
            rt = {d: max(1e-6, r_tab[d] + g.normal() * rc[(N, d)]["r_se"]) for d in r_tab}
            dt = {d: max(1e-3, d_tab[d] + g.normal() * dc[(N, d)]["d_hat_se"]) for d in d_tab}
            # monotone rearrangement (both tables are increasing in d)
            ks = sorted(rt)
            rv = np.maximum.accumulate([rt[k] for k in ks])
            rt = dict(zip(ks, rv))
            ks = sorted(dt)
            dv = np.maximum.accumulate([dt[k] for k in ks])
            dt = dict(zip(ks, dv))
            a, b = crossing(rt, dt, N)
            draws_n.append(a)
            draws_m.append(b)
        dn, dm = np.asarray(draws_n), np.asarray(draws_m)
        dn, dm = dn[np.isfinite(dn)], dm[np.isfinite(dm)]
        # sensitivity of the crossing to a systematic error in r
        ds_ = np.array(sorted(r_tab), float)
        rs_ = np.array([r_tab[int(k)] for k in ds_], float)
        _r = lambda d: np.exp(np.interp(np.log(d), np.log(ds_), np.log(rs_)))  # noqa: E731
        sens = float((np.log(1.03) - np.log(0.97))
                     / (np.log(_r(lim * 1.03)) - np.log(_r(lim * 0.97))))
        cross.append({
            "N": N, "d_star_limit_nominal": lim, "d_star_limit_measured": lim_m,
            "dlogd_dlogr_at_crossing": sens,
            "nominal_ci90": [float(np.quantile(dn, 0.05)), float(np.quantile(dn, 0.95))]
            if dn.size > 100 else [float("nan")] * 2,
            "measured_ci90": [float(np.quantile(dm, 0.05)), float(np.quantile(dm, 0.95))]
            if dm.size > 100 else [float("nan")] * 2})
    state["crossings"] = cross
    return state


def main(argv):
    Ns = N_QUICK if "--quick" in argv else N_GRID
    state = _load()
    state.update({
        "_definition": (
            "r(d, N) = W2(X_N, X'_N) / sqrt(d) for independent X_N, X'_N of N "
            "rows each from N(0, I_d), exact assignment; d_hat(d, N) = "
            "certgen.wasserstein_dimension(X, n_rep=3) on X ~ N(0, I_d) with "
            "N rows. Per-cell mean, SD, SE and the replicate values; crossings "
            "r = 1 in nominal and measured units with 90% intervals."),
        "n_rep_r": N_REP_R, "n_rep_dhat": N_REP_DHAT, "seed": SEED,
        # cells at larger N, added by reference_tables_large_n.py, are kept
        "N_grid": sorted(set(Ns) | {c["N"] for c in state["r_cells"]}),
        "d_grid_r": list(D_GRID_R), "d_grid_dhat": list(D_GRID_DHAT)})
    t0 = time.time()
    for N in Ns:
        for d in D_GRID_R:
            if _have(state["r_cells"], N, d):
                continue
            t = time.time()
            c = r_cell(N, d, N_REP_R)
            c["seconds"] = round(time.time() - t, 1)
            state["r_cells"].append(c)
            _save(state)
            print(f"r     N={N:>5} d={d:>3}  r={c['r_mean']:.4f} +/- {c['r_se']:.4f}"
                  f"  ({c['seconds']}s)", flush=True)
        for d in D_GRID_DHAT:
            if _have(state["dhat_cells"], N, d):
                continue
            t = time.time()
            c = dhat_cell(N, d, N_REP_DHAT)
            c["seconds"] = round(time.time() - t, 1)
            state["dhat_cells"].append(c)
            _save(state)
            print(f"d_hat N={N:>5} d={d:>3}  d_hat={c['d_hat_mean']:.3f} +/- "
                  f"{c['d_hat_se']:.3f}  ({c['seconds']}s)", flush=True)
    state = summarise(state)
    state["elapsed_this_run_s"] = round(time.time() - t0, 1)
    _save(state)
    for c in state["crossings"]:
        print(f"crossing N={c['N']:>5}: nominal {c['d_star_limit_nominal']:.1f} "
              f"{[round(x, 1) for x in c['nominal_ci90']]}  measured "
              f"{c['d_star_limit_measured']:.1f} {[round(x, 1) for x in c['measured_ci90']]}")
    return state


if __name__ == "__main__":
    main(sys.argv[1:])
