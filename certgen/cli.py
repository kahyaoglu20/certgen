"""
Command-line interface.

    python -m certgen design  real.csv [--target 0.5] [--n-rep 12] [--severity]
    python -m certgen resolve real.csv --n 1000 [--level L] [--alpha .05 --power .8]
    python -m certgen evaluate real.csv generated.csv [--n 2000]
    python -m certgen limit   2048 [--dhat 12.3]
    python -m certgen version

Every subcommand accepts --json for machine-readable output. CSV files are
read with numpy (header row optional, numeric columns only; non-numeric
columns are dropped with a notice -- choose and document an encoding for
mixed-type tables yourself, see the paper's mixed-type section). Rows with
missing values are dropped. Columns are standardised on the REAL sample only.

Every number is computed by the library function of the same name; a
prescription that the library withholds is printed as "not identified".
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

from . import __version__
from .baselines import Standardizer
from .design import fit_floor_law, required_n, resolvable_effect
from .latent import certification_limit, vacuity_margin
from .report import evaluate


def _read_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
    delim = "," if first.count(",") >= first.count(";") else ";"
    try:
        [float(x) for x in first.strip().split(delim) if x.strip()]
        skip = 0
    except ValueError:
        skip = 1
    raw = np.genfromtxt(path, delimiter=delim, skip_header=skip, dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    keep = ~np.all(~np.isfinite(raw), axis=0)       # drop all-NaN (text) columns
    dropped = int((~keep).sum())
    X = raw[:, keep]
    X = X[np.isfinite(X).all(axis=1)]
    if dropped:
        print(f"[certgen] {path}: dropped {dropped} non-numeric column(s)",
              file=sys.stderr)
    if len(X) < 64:
        raise SystemExit(f"{path}: only {len(X)} complete numeric rows")
    return X


def _fmt(x: float, nd: int = 4) -> str:
    return "not identified" if not np.isfinite(x) else f"{x:.{nd}g}"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def cmd_design(a) -> dict:
    X = Standardizer().fit_transform(_read_csv(a.real))
    law = fit_floor_law(X, n_rep=a.n_rep, seed=a.seed)
    out = {"rows": int(len(X)), "columns": int(X.shape[1]),
           "C": law.C, "d_eff": law.d_eff, "r2": law.r2,
           "d_eff_low": law.d_eff_low, "d_eff_high": law.d_eff_high,
           "drift_p_value": law.drift_p_value,
           "single_exponent_ok": law.single_exponent_ok,
           "n_grid": law.n_grid, "floors": law.floors,
           "warnings": law.warnings}
    if a.target is not None:
        out["required_n"] = required_n(law, a.target)
        out["required_n_range"] = law.required_n_range(a.target)
        iv = law.required_n_interval(a.target, calibrate=True)
        out["required_n_interval"] = {k: iv[k] for k in
                                      ("n_lo", "n_med", "n_hi", "applied_level",
                                       "calibrated", "calibration_refused")}
    if a.severity:
        sp = law.severity_panel(seeds=tuple(range(a.seeds)))
        out["severity"] = {k: sp[k] for k in ("drift_observed", "drift_50",
                                               "drift_50_ci90", "sd_within",
                                               "sd_between", "n_seeds")}
    if not a.json:
        print(law.summary())
        if a.target is not None:
            print(f"required_n(target={a.target}) = {_fmt(out['required_n'])}")
            rr = out["required_n_range"]
            print(f"  half-fit range: low {_fmt(rr['n_low_fit'])}  high "
                  f"{_fmt(rr['n_high_fit'])}  ratio {_fmt(rr['ratio'])}")
            iv = out["required_n_interval"]
            print(f"  interval (requested delivered 0.90): [{_fmt(iv['n_lo'])}, "
                  f"{_fmt(iv['n_hi'])}]"
                  + (" (outside the calibration regime: uncalibrated)"
                     if iv["calibration_refused"] else ""))
        if a.severity:
            s = out["severity"]
            print(f"severity: drift_50 = {_fmt(s['drift_50'])} combined CI90 "
                  f"[{_fmt(s['drift_50_ci90'][0])}, {_fmt(s['drift_50_ci90'][1])}]"
                  f" over {s['n_seeds']} seeds; observed drift "
                  f"{_fmt(s['drift_observed'])}")
    return out


def cmd_resolve(a) -> dict:
    X = Standardizer().fit_transform(_read_csv(a.real))
    law = fit_floor_law(X, n_rep=a.n_rep, seed=a.seed)
    eff = resolvable_effect(law, a.n, reference_level=a.level,
                            alpha=a.alpha, power=a.power)
    out = {"n": a.n, "alpha": a.alpha, "power": a.power,
           "reference_level": a.level, "resolvable_effect": eff,
           "single_exponent_ok": law.single_exponent_ok}
    if not a.json:
        print(f"smallest resolvable difference at n = {a.n} per side "
              f"(alpha = {a.alpha}, power = {a.power}): {_fmt(eff)}")
    return out


def cmd_evaluate(a) -> dict:
    Xr, Xg = _read_csv(a.real), _read_csv(a.generated)
    n = a.n or min(len(Xr) // 2, len(Xg))
    rep = evaluate(Xr, Xg[:n] if len(Xg) > n else Xg, n_per_side=n,
                   seed=a.seed)
    out = _jsonable(rep.to_dict())
    if not a.json:
        print(rep.summary())
    return out


def cmd_limit(a) -> dict:
    cl = certification_limit(a.N)
    out = {"N": a.N, "d_star_limit_nominal": cl["d_star_limit"],
           "d_star_limit_measured": cl["d_star_limit_measured"],
           "in_tabulated_range": cl["in_tabulated_range"],
           "warnings": cl["warnings"],
           "note": ("limits what can be certified, not what can be achieved")}
    if "d_star_limit_measured_ci90" in cl:
        out["d_star_limit_measured_ci90"] = cl["d_star_limit_measured_ci90"]
    if a.dhat is not None:
        out["vacuity_margin"] = vacuity_margin(a.dhat, a.N)
    if not a.json:
        print(f"certification limit at N = {a.N}: measured-units d_hat "
              f"{_fmt(out['d_star_limit_measured'])} (nominal "
              f"{_fmt(out['d_star_limit_nominal'])})")
        for w in cl["warnings"]:
            print(f"  [!] {w}")
        print("  The limit concerns what can be certified, not what can be achieved.")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="certgen", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, real=True):
        if real:
            sp.add_argument("real", help="CSV of real rows (numeric columns)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--json", action="store_true", help="print JSON")

    d = sub.add_parser("design", help="fit the floor law; optional sizing")
    common(d)
    d.add_argument("--n-rep", type=int, default=12)
    d.add_argument("--target", type=float, default=None,
                   help="target ABSOLUTE floor for required_n")
    d.add_argument("--severity", action="store_true")
    d.add_argument("--seeds", type=int, default=8)
    d.set_defaults(fn=cmd_design)

    r = sub.add_parser("resolve", help="smallest resolvable difference at n")
    common(r)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--n-rep", type=int, default=12)
    r.add_argument("--level", type=float, default=None,
                   help="discrepancy level the models sit at (e.g. D_T1)")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--power", type=float, default=0.80)
    r.set_defaults(fn=cmd_resolve)

    e = sub.add_parser("evaluate", help="floor-corrected report + leakage")
    common(e)
    e.add_argument("generated", help="CSV of generated rows")
    e.add_argument("--n", type=int, default=None)
    e.set_defaults(fn=cmd_evaluate)

    lm = sub.add_parser("limit", help="certification limit at N")
    lm.add_argument("N", type=int)
    lm.add_argument("--dhat", type=float, default=None)
    lm.add_argument("--json", action="store_true")
    lm.set_defaults(fn=cmd_limit)

    v = sub.add_parser("version")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=lambda a: {"version": __version__} if a.json
                   else print(f"certgen {__version__}") or {"version": __version__})

    a = p.parse_args(argv)
    out = a.fn(a)
    if getattr(a, "json", False):
        print(json.dumps(_jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
