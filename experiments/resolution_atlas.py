"""The resolution atlas (Section 8.4).

For every benchmark cell (up to 20000 rows, standardised) and every per-side
size n in {250, 500, 1000, 2000, 4000}:

  floor      median of n_rep = 20 disjoint-split distances;
  D_T1       mean distance of five independent samples of the
             independent-columns generator T1 to a real reference sample,
             with its standard deviation over the five samples;
  sd_pair    standard deviation over 30 replicates of W2(Xr, A) - W2(Xr, B),
             with A and B independent samples of T1 and Xr a fresh real
             reference of n rows, and its 90% chi-square interval;
  rho        (z_0.975 + z_0.80) sd_pair / (D_T1 - floor), the smallest
             resolvable skill difference in units of the skill range;
  proxy      the same quantity with sd_pair replaced by the floor proxy
             sqrt(2) CV(floor) D_T1.

Cells are checkpointed in the artifact, so an interrupted run resumes.

    python experiments/resolution_atlas.py                    # about 2 CPU hours
    python experiments/resolution_atlas.py --only=adult:E1,letter:E1

Downloads the tables from OpenML on first use. Writes
results/resolution_atlas.json.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.stats import chi2, norm

import _artifacts as art
import certgen as cg
from certgen.floor import resolution_floor
from _benchmarks import CELLS, Tables, subsample

ALPHA, POWER = 0.05, 0.80
MAX_ROWS = 20000
N_GRID = (250, 500, 1000, 2000, 4000)
N_REP = 20
N_BASE_REP = 5
R_PAIR = 30
SEED = 0
NAME = "resolution_atlas.json"


def not_admissible() -> set:
    """Cells whose floor law is not admissible in benchmark_floor_laws.json
    (encodings are named as there, with covertype's single cell as E1)."""
    recs = art.load("benchmark_floor_laws.json")["records"]
    return {r["dataset"].replace("E1=E2", "E1") for r in recs
            if not r["single_exponent_ok"]}


def load(only=None):
    """(tag, standardised data or None, error) for every benchmark cell."""
    tables, out = Tables(), []
    for tag, source, enc in CELLS:
        if only and tag not in only:
            continue
        try:
            X = tables.encoded(source, enc)
        except Exception as e:  # recorded in the artifact
            print(f"  not loaded {tag}: {type(e).__name__}: {e}", flush=True)
            out.append((tag, None, f"{type(e).__name__}: {e}"))
            continue
        out.append((tag, cg.Standardizer().fit_transform(
            subsample(X, MAX_ROWS, SEED)), None))
    return out


def floor_and_t1(tag, X, n):
    fr = resolution_floor(X, n, metric="w2", n_rep=N_REP, seed=SEED)
    f, cv = fr.median, fr.cv
    rng = np.random.default_rng(SEED)
    Xr = X[rng.permutation(len(X))[:n]]
    t1s = [cg.w2_exact(Xr, cg.baseline_marginals(X, n, seed=s), max_n=n, seed=s)
           for s in range(N_BASE_REP)]
    D_T1, D_T1_sd = float(np.mean(t1s)), float(np.std(t1s, ddof=1))
    span = D_T1 - f
    proxy_mdd = fr.mdd(D_T1, ALPHA, POWER)
    return {"table": tag, "n_per_side": int(n), "floor": float(f),
            "floor_cv": float(cv), "D_T1": D_T1,
            "D_T1_sd_over_seeds": D_T1_sd,
            "span_T1_exceeds_baseline_noise": bool(span > 2 * D_T1_sd),
            "skill_span_T1": float(span),
            "dskill_T1_over_T1_p80": float(proxy_mdd / span) if span > 0
            else float("nan"),
            "alpha": ALPHA, "power": POWER}


def paired_sd(X, n):
    """Standard deviation of W2(Xr, A) - W2(Xr, B) for independent T1 samples
    A and B against a shared fresh reference Xr, over R_PAIR replicates."""
    diffs, d_all = [], []
    for k in range(R_PAIR):
        g = np.random.default_rng([SEED, 48, n, k])
        Xr = X[g.permutation(len(X))[:n]]
        A = cg.baseline_marginals(X, n, seed=100000 + 2 * k)
        B = cg.baseline_marginals(X, n, seed=100001 + 2 * k)
        dA = cg.w2_exact(Xr, A, max_n=n, seed=k)
        dB = cg.w2_exact(Xr, B, max_n=n, seed=k)
        diffs.append(dA - dB)
        d_all.extend([dA, dB])
    diffs = np.asarray(diffs, float)
    sd = float(diffs.std(ddof=1))
    lo = sd * np.sqrt((R_PAIR - 1) / chi2.ppf(0.95, R_PAIR - 1))
    hi = sd * np.sqrt((R_PAIR - 1) / chi2.ppf(0.05, R_PAIR - 1))
    return {"sd_pair_true": sd, "sd_pair_true_ci90": [float(lo), float(hi)],
            "r_pair": R_PAIR, "mean_D_T1_pairs": float(np.mean(d_all)),
            "diff_mean": float(diffs.mean())}


def summarise(state):
    z = float(norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER))
    cells = state["cells"]
    for c in cells:
        sd = c.get("sd_pair_true")
        if sd is not None and c["skill_span_T1"] > 0:
            c["mdd_at_T1_p80_true"] = z * sd
            c["dskill_T1_over_T1_p80_true"] = z * sd / c["skill_span_T1"]
            lo, hi = c["sd_pair_true_ci90"]
            c["dskill_T1_over_T1_p80_true_ci90"] = [z * lo / c["skill_span_T1"],
                                                    z * hi / c["skill_span_T1"]]
            proxy_sd = np.sqrt(2.0) * c["floor_cv"] * c["D_T1"]
            c["proxy_over_true_sd"] = float(proxy_sd / sd) if sd > 0 else float("nan")
        else:
            c["dskill_T1_over_T1_p80_true"] = float("nan")

    def agg(sel, key):
        v = np.array([c[key] for c in sel if np.isfinite(c.get(key, np.nan))], float)
        if v.size == 0:
            return {}
        return {"cells": int(v.size), "min": float(v.min()),
                "median": float(np.median(v)), "max": float(v.max()),
                "frac_ge_1": float(np.mean(v >= 1.0)),
                "argmin": [(c["table"], c["n_per_side"]) for c in sel
                           if c.get(key) == float(v.min())][0]}
    excluded = not_admissible()
    admissible = [c for c in cells if c["table"] not in excluded]
    state["summary"] = {
        "not_admissible": sorted(excluded),
        "extended_proxy_p80": agg(admissible, "dskill_T1_over_T1_p80"),
        "extended_true_p80": agg(admissible, "dskill_T1_over_T1_p80_true"),
        "all_incl_shuttle_true_p80": agg(cells, "dskill_T1_over_T1_p80_true"),
        "proxy_over_true_sd": agg(cells, "proxy_over_true_sd"),
        "cells_where_T1_span_inside_baseline_noise": [
            (c["table"], c["n_per_side"]) for c in cells
            if not c["span_T1_exceeds_baseline_noise"]],
    }
    return state


def main(argv):
    only = None
    for a in argv:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
    state = (art.load(NAME) if art.exists(NAME)
             else {"cells": [], "skipped": []})
    state.update({
        "_definition": (
            "per cell (table, n): floor, distance of the independent-columns "
            "generator T1, measured paired standard deviation at the T1 "
            "level, and the smallest resolvable skill difference rho = "
            "(z_0.975 + z_0.80) sd_pair / (D_T1 - floor), measured and with "
            "the floor proxy"),
        "_scope": ("n <= 4000 per side; tables capped at 20000 rows; the "
                   "generator is held fixed, so rho is a lower bound on the "
                   "difference required when trained models are compared"),
        "n_grid": list(N_GRID), "n_rep": N_REP, "n_base_rep": N_BASE_REP,
        "r_pair": R_PAIR, "seed": SEED, "max_rows": MAX_ROWS})
    done = {(c["table"], c["n_per_side"]) for c in state["cells"]}
    for tag, X, err in load(only):
        if X is None:
            state["skipped"].append({"table": tag, "reason": err})
            art.save(NAME, state, quiet=True)
            continue
        for n in N_GRID:
            if (tag, n) in done or 2 * n > len(X):
                continue
            t = time.time()
            rec = floor_and_t1(tag, X, n)
            rec.update(paired_sd(X, n))
            rec["rows_used"] = int(len(X))
            rec["ambient_dim"] = int(X.shape[1])
            rec["seconds"] = round(time.time() - t, 1)
            state["cells"].append(rec)
            art.save(NAME, state, quiet=True)
            print(f"{tag:<18} n={n:>5} floor={rec['floor']:.4f} "
                  f"D_T1={rec['D_T1']:.4f} sd_pair={rec['sd_pair_true']:.4f} "
                  f"({rec['seconds']} s)", flush=True)
    art.save(NAME, summarise(state))


if __name__ == "__main__":
    main(sys.argv[1:])
