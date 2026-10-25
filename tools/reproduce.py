"""Re-run every experiment in a copy of the project and compare the results.

The project is copied to a temporary directory, so results/ is never
modified. Every script in experiments/ is run there, and every regenerated
artifact is compared with the stored one: numeric values to a relative
tolerance of 1e-6 and every other value exactly, with timing and platform
fields excluded.

    python tools/reproduce.py                 # offline experiments, 2 jobs
    python tools/reproduce.py --quick         # without the slow experiments
    python tools/reproduce.py --network       # also the OpenML experiments
    python tools/reproduce.py --fresh         # recompute cached table cells
    python tools/reproduce.py --only severity,stress_tests --log out.json

Writes results/reproduction_log.json (a partial run with --only needs --log).

Status of a script: ok | failed | timeout. Status of an artifact:
identical | reproduced (within tolerance) | differs | platform-dependent
(see PLATFORM_DEPENDENT) | not-regenerated.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# experiments that download data from OpenML
NETWORK = {"benchmark_floor_laws.py", "resolution_atlas.py", "tail_index.py",
           "benchmark_sources.py",
           "benchmark_diagnostics.py"}
# experiments that take five minutes or more on one core
SLOW = {"copula_examples.py", "curvature_test_size.py", "curvature_test_power.py"}
# experiments that keep their measured cells in their artifact; the copy
# reuses them unless --fresh is given (from scratch, the reference tables
# take about 1.5 CPU hours and the resolution atlas about 2)
RESUMABLE = {"reference_tables.py": "reference_tables.json",
             "resolution_atlas.py": "resolution_atlas.json"}
# leaves that depend on the last bit of floating-point sums because the
# scenario places the statistic exactly on a decision boundary; excluded by
# path, with the reason logged (no value quoted in the manuscript is listed)
BOUNDARY_SENSITIVE = {
    "equivalence_test.json": (
        ("/rates/boundary +delta/", "/rates/boundary -delta/"),
        "the boundary scenarios place D exactly at floor +/- delta, so which "
        "replicates fall on either side of the threshold depends on rounding"),
}


def _vae_claims(a) -> bool:
    return bool(a["summary"]["all_ppc_valid"]) and all(
        c["ppc_over_exact"] <= 1.2 and c["lipschitz_over_exact"] >= 10
        for c in a["cells"])


# artifacts of trained neural networks: seeded and single-threaded, but
# training amplifies last-bit differences between floating-point platforms. A
# difference is accepted only if the claims built on the artifact still hold.
PLATFORM_DEPENDENT = {
    "plan_pullback_vae.json": (
        "trained VAEs: training amplifies last-bit differences between "
        "floating-point platforms. Claims checked: every bound valid, "
        "PPC/exact <= 1.2 and Lipschitz/exact >= 10 in every cell",
        _vae_claims),
}
VOLATILE = ("elapsed", "wall", "seconds", "_when", "time", "date",
            "timestamp", "path", "host", "platform", "torch", "cpu",
            "python", "numpy", "scipy")
REL_TOL = 1e-6


def _leaves(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(t in str(k).lower() for t in VOLATILE):
                continue
            yield from _leaves(v, f"{prefix}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def compare(a, b, skip=()) -> dict:
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    excluded = sorted(k for k in set(la) | set(lb) if k.startswith(tuple(skip)))
    for k in excluded:
        la.pop(k, None)
        lb.pop(k, None)
    worst, nnum, mism = 0.0, 0, []
    for k in sorted(set(la) | set(lb)):
        x, y = la.get(k, "<absent>"), lb.get(k, "<absent>")
        if isinstance(x, bool) or isinstance(y, bool) or not (
                isinstance(x, (int, float)) and isinstance(y, (int, float))):
            if x != y:
                mism.append(k)
            continue
        nnum += 1
        if isinstance(x, float) and math.isnan(x) and isinstance(y, float) and math.isnan(y):
            continue
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
    status = ("identical" if worst == 0 and not mism else
              "reproduced" if worst <= REL_TOL and not mism else "differs")
    out = {"status": status, "numeric_leaves": nnum, "max_rel_diff": worst,
           "structural_mismatches": mism[:10], "n_structural_mismatches": len(mism)}
    if excluded:
        out["boundary_sensitive_excluded"] = excluded
    return out


def run_one(work, script, timeout):
    t0 = time.time()
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        p = subprocess.run([sys.executable, script],
                           cwd=os.path.join(work, "experiments"),
                           capture_output=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        rc, out = p.returncode, p.stdout + p.stderr
        status = "ok" if rc == 0 else "failed"
    except subprocess.TimeoutExpired:
        rc, out, status = -1, "", "timeout"
    return {"script": f"experiments/{script}", "status": status, "rc": rc,
            "seconds": round(time.time() - t0, 1),
            "tail": out.strip().splitlines()[-3:] if out else []}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--only", default=None,
                    help="comma-separated script-name prefixes")
    ap.add_argument("--log", default=None,
                    help="log path; required with --only")
    a = ap.parse_args(argv)
    if a.only and not a.log:
        ap.error("--only needs --log: a partial run must not replace the full log")
    work = tempfile.mkdtemp(prefix="certgen_reproduce_")
    shutil.copytree(ROOT, os.path.join(work, "p"),
                    ignore=shutil.ignore_patterns(".git", ".lake", "__pycache__",
                                                  "*.egg-info", ".pytest_cache"))
    wroot = os.path.join(work, "p")
    if a.fresh:
        for art in RESUMABLE.values():
            p = os.path.join(wroot, "results", art)
            if art == "reference_tables.json":
                # the cells at large N come from reference_tables_large_n.py
                # and its stored solves, not from reference_tables.py
                with open(p, encoding="utf-8") as fh:
                    tab = json.load(fh)
                for key in ("r_cells", "dhat_cells"):
                    tab[key] = [c for c in tab[key] if c.get("source") == "large_n"]
                tab.pop("crossings", None)
                with open(p, "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(tab, fh, indent=1)
            else:
                os.remove(p)
    before = {f: os.path.getmtime(os.path.join(wroot, "results", f))
              for f in os.listdir(os.path.join(wroot, "results"))}
    scripts = []
    for f in sorted(os.listdir(os.path.join(ROOT, "experiments"))):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        if (f in NETWORK and not a.network) or (a.quick and f in SLOW):
            continue
        if a.only and not any(f.startswith(x.strip()) for x in a.only.split(",")):
            continue
        scripts.append(f)
    # second stage: scripts that read artifacts written in the first
    order = {"certification_margins.py": 1, "half_fit_ranges.py": 1,
             "severity.py": 1, "interval_coverage.py": 1,
             "reference_tables_large_n.py": 1}
    print(f"reproduce: {len(scripts)} experiments in {wroot} with {a.jobs} jobs",
          flush=True)
    t0 = time.time()
    runs = []
    for stage in (0, 1):
        batch = [s for s in scripts if order.get(s, 0) == stage]
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(lambda s: run_one(wroot, s, a.timeout), batch):
                runs.append(r)
                print(f"  {r['status']:<8} {r['seconds']:>8.1f}s  {r['script']}",
                      flush=True)
    arts = []
    for f in sorted(os.listdir(os.path.join(ROOT, "results"))):
        if not f.endswith(".json") or f == "reproduction_log.json":
            continue
        new_p = os.path.join(wroot, "results", f)
        if not os.path.exists(new_p) or os.path.getmtime(new_p) == before.get(f):
            arts.append({"artifact": f, "status": "not-regenerated"})
            continue
        with open(os.path.join(ROOT, "results", f), encoding="utf-8") as fh:
            old = json.load(fh)
        with open(new_p, encoding="utf-8") as fh:
            new = json.load(fh)
        skip, why = BOUNDARY_SENSITIVE.get(f, ((), ""))
        c = compare(old, new, skip)
        if skip:
            c["boundary_sensitive_reason"] = why
        if c["status"] == "differs" and f in PLATFORM_DEPENDENT:
            reason, claims = PLATFORM_DEPENDENT[f]
            if claims(new):
                c["status"], c["reason"] = "platform-dependent", reason
        c["artifact"] = f
        arts.append(c)
    summary = {
        "scripts": len(runs),
        "ok": sum(r["status"] == "ok" for r in runs),
        "failed": [r["script"] for r in runs if r["status"] != "ok"],
        "artifacts_regenerated": sum(c["status"] != "not-regenerated" for c in arts),
        "artifacts_identical_or_reproduced": sum(
            c["status"] in ("identical", "reproduced") for c in arts),
        "artifacts_differing": [c["artifact"] for c in arts if c["status"] == "differs"],
        "artifacts_platform_dependent": [c["artifact"] for c in arts
                                         if c["status"] == "platform-dependent"],
        "wall_seconds": round(time.time() - t0, 1),
        "mode": {"quick": a.quick, "network": a.network, "jobs": a.jobs,
                 "fresh": a.fresh, "only": a.only},
        "python": sys.version.split()[0],
    }
    out = {"_definition": ("every experiment run in a copy of the project and "
                           "every regenerated artifact compared with the "
                           f"stored one (relative tolerance {REL_TOL})"),
           "summary": summary, "scripts": runs, "artifacts": arts}
    with open(a.log or os.path.join(ROOT, "results", "reproduction_log.json"),
              "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps(summary, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if not (summary["failed"] or summary["artifacts_differing"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
