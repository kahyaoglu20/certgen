"""Reference tables at N = 16 384 and 32 768 under a memory budget
(Section 6.3, Appendix B.4).

The protocol is that of reference_tables.py, cell by cell:

  r task     (N, d, rep): X, X' ~ N(0, I_d) with N rows each, drawn from
             SeedSequence([20260930, 1, N, d, rep]); value =
             w2_exact(X, X', max_n=N, seed=rep) / sqrt(d);
  d_hat task (N, d, rep): X ~ N(0, I_d) with N rows from
             SeedSequence([20260930, 2, N, d, rep]); value =
             wasserstein_dimension(X, n_rep=3, seed=rep), whose largest
             solve has N/2 rows per side;

for d in (1, 2, 4, ..., 128) (r) and (2, ..., 128) (d_hat), rep = 0..19.
One exact solve at N rows per side holds an N x N float64 cost matrix
(2.1 GB at N = 16 384, 8.6 GB at N = 32 768), so every task runs as its own
process and starts only when its memory estimate fits under
min(budget, available memory - reserve); smaller tasks fill the gaps while a
large one waits (EASY backfilling). Each task writes one JSON shard, so an
interrupted run resumes.

    python reference_tables_large_n.py run --N 16384 32768 --budget-gb 20
            --workers 10 --shards DIR [--status DIR] [--calibrate]
        Execute the missing tasks. With --calibrate, six solves at N = 8192
        whose values are stored in results/reference_tables.json are
        recomputed first, as a reproduction check on the machine at hand.
        An optional DIR/control.json changes N, budget_gb, workers or
        reserve_gb of a running session, or stops it ("stop": true).
    python reference_tables_large_n.py collect --shards DIR [--N 16384 ...]
        Gather the shards into results/reference_tables_large_n_solves.json.
    python reference_tables_large_n.py merge
        Insert the completed cells into results/reference_tables.json and
        recompute the crossings.
    python reference_tables_large_n.py
        Rebuild results/reference_tables_large_n.json (per-cell statistics,
        timing, memory and the reproduction check) from the solves, and
        verify that results/reference_tables.json carries the same cells.

The script also runs from a stand-alone directory that holds a copy of the
certgen package next to it.
"""
from __future__ import annotations

import os

# one thread per task: the solver is single-threaded and parallelism comes
# from running several tasks at once
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, ".."), HERE):
    if os.path.isdir(os.path.join(_p, "certgen")) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

SEED = 20260930
D_GRID_R = (1, 2, 4, 8, 16, 32, 64, 128)
D_GRID_DHAT = (2, 4, 8, 16, 32, 64, 128)
N_REP = 20
N_LARGE = (16384, 32768)
# reproduction-check solves: (kind, N, d, rep) stored in reference_tables.json
CALIBRATION = [("r", 8192, 8, 0), ("r", 8192, 8, 1), ("r", 8192, 64, 0),
               ("r", 8192, 1, 0), ("dhat", 8192, 8, 0), ("dhat", 8192, 64, 0)]
RESULTS = os.path.join(HERE, "..", "results")
SOLVES = os.path.join(RESULTS, "reference_tables_large_n_solves.json")
TABLES = os.path.join(RESULTS, "reference_tables_large_n.json")
EXT = os.path.join(RESULTS, "reference_tables.json")
GB = 1e9


def _rng(*key):
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def task_id(t):
    return "%s_%d_%d_%02d" % t


def mem_gb(t) -> float:
    """Memory estimate of one task: the dense float64 cost matrix of its
    largest solve, the samples (two N x d arrays for r, one for d_hat) and
    0.15 GB of interpreter overhead. Measured peak working sets stay below
    it."""
    kind, N, d = t[0], t[1], t[2]
    n = N if kind == "r" else N // 2
    samples = (2 if kind == "r" else 1) * 8.0 * N * d
    return (8.0 * n * n + samples) / GB + 0.15


def compute(t) -> float:
    from certgen.discrepancy import w2_exact
    from certgen.dimension import wasserstein_dimension
    kind, N, d, rep = t
    if kind == "r":
        g = _rng(1, N, d, rep)
        X = g.normal(size=(N, d))
        Y = g.normal(size=(N, d))
        return float(w2_exact(X, Y, max_n=N, seed=rep) / np.sqrt(d))
    g = _rng(2, N, d, rep)
    X = g.normal(size=(N, d))
    return float(wasserstein_dimension(X, n_rep=3, seed=rep))


def _peak_rss_gb() -> float:
    """Peak resident memory of this process (Windows: peak working set;
    POSIX: ru_maxrss)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):
            return round(mi.peak_wset / GB, 3)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB, 3)
    except Exception:
        return float("nan")


def _host():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "system": platform.system(),
            "machine": platform.machine(), "processor": platform.processor()}


def _worker(t, shard_dir, tag):
    try:
        import psutil
        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(5)
    except Exception:
        pass
    t0 = time.time()
    v = compute(t)
    rec = {"kind": t[0], "N": t[1], "d": t[2], "rep": t[3], "value": v,
           "seconds": round(time.time() - t0, 2), "peak_rss_gb": _peak_rss_gb(),
           "set": tag, "host": _host(),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.join(shard_dir, task_id(t) + ".json")
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh)
    os.replace(path + ".tmp", path)


def grid(Ns):
    out = []
    for N in Ns:
        out += [("r", N, d, r) for d in D_GRID_R for r in range(N_REP)]
        out += [("dhat", N, d, r) for d in D_GRID_DHAT for r in range(N_REP)]
    return out


def _available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / GB
    except Exception:
        return float("inf")


def _committed_gb(pid) -> float:
    """Memory a running task has committed so far (Windows: private bytes;
    elsewhere: RSS); 0 when it cannot be read."""
    try:
        import psutil
        mi = psutil.Process(pid).memory_info()
        return getattr(mi, "private", mi.rss) / GB
    except Exception:
        return 0.0


def _expected_seconds(t, done_times):
    """Mean duration of the tasks of the same kind and N finished in this
    session, or None before the first one has finished."""
    v = done_times.get((t[0], t[1]))
    return float(np.mean(v)) if v else None


def _shadow(running, m_head, cap, done_times):
    """EASY backfilling: the earliest time the blocked head task fits (from
    the expected end of the running tasks) and the memory still spare at that
    moment. The time is inf while any task it waits for has no estimate."""
    ends = []
    for t, (p, m, s0) in running.items():
        dur = _expected_seconds(t, done_times)
        ends.append((s0 + dur if dur is not None else float("inf"), m))
    use = sum(m for _, m in ends)
    shadow = time.time()
    for end, m in sorted(ends, key=lambda x: x[0]):
        if use + m_head <= cap:
            break
        use -= m
        shadow = max(shadow, end)
    return shadow, max(0.0, cap - use - m_head)


def _keep_awake():
    if sys.platform == "win32":     # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def cmd_run(a):
    os.makedirs(a.shards, exist_ok=True)
    status_dir = a.status or a.shards
    os.makedirs(status_dir, exist_ok=True)
    log = open(os.path.join(status_dir, "run.log"), "a", buffering=1, encoding="utf-8", newline="\n")

    def say(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")

    def build_todo(Ns, calibrate, exclude=()):
        todo = ([(t, "calibration") for t in CALIBRATION] if calibrate else [])
        # largest tasks first within each N: they are the hardest to place
        for N in sorted(Ns):
            g = sorted(grid([N]), key=lambda t: (-mem_gb(t), t[0], t[2], t[3]))
            todo += [(t, "grid") for t in g]
        return [(t, s) for t, s in todo if t not in exclude and not os.path.exists(
            os.path.join(a.shards, task_id(t) + ".json"))]

    control_path = os.path.join(status_dir, "control.json")

    def read_control():
        try:
            with open(control_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    ctl = read_control()
    Ns = sorted(set(ctl.get("N", a.N)))
    todo = build_todo(Ns, a.calibrate)
    total = len(todo)
    say(f"run: {total} tasks to do for N = {Ns}; budget {a.budget_gb} GB, "
        f"reserve {a.reserve_gb} GB, at most {a.workers} workers; host {_host()}")
    ctx = mp.get_context("spawn")
    running = {}      # task -> (process, mem, start)
    failures = {}
    done_times = {}
    t_start = time.time()
    last_status = 0.0
    last_control = time.time()
    blocked_said = None
    while True:
        _keep_awake()
        if time.time() - last_control > 30:
            # control.json (optional) lets an operator change the grid, the
            # memory budget or the worker count of a running session, or end it
            last_control = time.time()
            ctl = read_control()
            a.budget_gb = float(ctl.get("budget_gb", a.budget_gb))
            a.workers = int(ctl.get("workers", a.workers))
            a.reserve_gb = float(ctl.get("reserve_gb", a.reserve_gb))
            newN = sorted(set(ctl.get("N", Ns)))
            if newN != Ns:
                Ns = newN
                queued = {t for t, _ in todo}
                todo += [x for x in build_todo(Ns, False, exclude=set(running))
                         if x[0] not in queued]
                say(f"control: N = {Ns}; {len(todo)} tasks queued")
            if ctl.get("stop"):
                say("control: stop requested; waiting for running tasks")
                todo = []
        if not todo and not running:
            if read_control().get("stay") and not read_control().get("stop"):
                time.sleep(10)
                continue
            break
        for t, (p, m, s0) in list(running.items()):
            if not p.is_alive():
                p.join()
                del running[t]
                ok = os.path.exists(os.path.join(a.shards, task_id(t) + ".json"))
                if ok:
                    done_times.setdefault((t[0], t[1]), []).append(time.time() - s0)
                    say(f"done   {task_id(t):<22} {time.time() - s0:8.1f}s")
                else:
                    failures[t] = failures.get(t, 0) + 1
                    say(f"FAILED {task_id(t)} (exit {p.exitcode}, attempt {failures[t]})")
                    if failures[t] < 2:
                        todo.insert(0, (t, "grid"))
        in_use = sum(m for _, m, _ in running.values())
        # a task reserves its estimate when it starts but allocates it seconds
        # later; until then the OS still reports that memory as available, so
        # the unallocated part is subtracted or the next task overcommits
        pending = sum(max(0.0, m - _committed_gb(p.pid)) for p, m, _ in running.values())
        cap = min(a.budget_gb, in_use + _available_gb() - pending - a.reserve_gb)
        shadow = spare = None       # set once the head task is blocked
        i = 0
        while i < len(todo) and len(running) < a.workers:
            t, tag = todo[i]
            m = mem_gb(t)
            fits = in_use + m <= cap
            if fits and shadow is not None:
                # the head is blocked: a later task may start only if it ends
                # before the head can start or fits beside the head then
                dur = _expected_seconds(t, done_times)
                if dur is not None and time.time() + dur <= shadow < float("inf"):
                    pass
                elif m <= spare:
                    spare -= m
                else:
                    fits = False
            if fits:
                p = ctx.Process(target=_worker, args=(t, a.shards, tag))
                p.start()
                running[t] = (p, m, time.time())
                in_use += m
                todo.pop(i)
                blocked_said = None
                say(f"start  {task_id(t):<22} mem {m:5.2f} GB  in use {in_use:5.2f} GB")
                continue
            if i == 0:
                if m > a.budget_gb:
                    say(f"cannot place {task_id(t)}: needs {m:.2f} GB, budget "
                        f"{a.budget_gb:.2f} GB")
                    return 2
                if not running and blocked_said != task_id(t):
                    # memory is short for now (another program is using it):
                    # wait for it instead of ending an unattended run
                    blocked_said = task_id(t)
                    say(f"waiting: {task_id(t)} needs {m:.2f} GB, {cap:.2f} GB "
                        "usable now")
                shadow, spare = _shadow(running, m, cap, done_times)
            i += 1
        if time.time() - last_status > 30:
            last_status = time.time()
            st = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "elapsed_h": round((time.time() - t_start) / 3600, 3),
                  "remaining_tasks": len(todo), "running": [
                      {"task": task_id(t), "mem_gb": round(m, 2),
                       "seconds": round(time.time() - s0)}
                      for t, (p, m, s0) in running.items()],
                  "mean_seconds": {f"{k[0]}_{k[1]}": round(float(np.mean(v)), 1)
                                   for k, v in done_times.items()},
                  "done_counts": {f"{k[0]}_{k[1]}": len(v)
                                  for k, v in done_times.items()},
                  "failures": {task_id(t): n for t, n in failures.items()},
                  "in_use_gb": round(in_use, 2),
                  "available_gb": round(_available_gb(), 2)}
            with open(os.path.join(status_dir, "status.json.tmp"), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(st, fh, indent=1)
            os.replace(os.path.join(status_dir, "status.json.tmp"),
                       os.path.join(status_dir, "status.json"))
        time.sleep(2)
    say(f"run finished in {(time.time() - t_start) / 3600:.2f} h; "
        f"{len(failures)} tasks failed at least once")
    return 0


def cmd_collect(a):
    recs = []
    for f in sorted(os.listdir(a.shards)):
        if f.endswith(".json") and f[:2] in ("r_", "dh"):
            with open(os.path.join(a.shards, f), encoding="utf-8") as fh:
                rec = json.load(fh)
            # --N limits the grid records to the listed N (a release can ship
            # the N that are complete while larger N are still running)
            if a.N and rec["set"] == "grid" and rec["N"] not in a.N:
                continue
            recs.append(rec)
    recs.sort(key=lambda r: (r["set"], r["kind"], r["N"], r["d"], r["rep"]))
    out = {"_definition": ("one record per solve task: kind r -> W2/sqrt(d) "
                           "at N rows per side; kind dhat -> "
                           "wasserstein_dimension at N rows; set "
                           "'calibration' recomputes values stored in "
                           "reference_tables.json"),
           "records": recs}
    with open(SOLVES, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(f"collected {len(recs)} records -> {SOLVES}")


def build_cells(recs):
    """Cells in the schema of reference_tables.json (mean, sample SD, SE,
    values in replicate order) for every (kind, N, d) with all replicates."""
    from collections import defaultdict
    groups = defaultdict(dict)
    for r in recs:
        if r["set"] == "grid":
            groups[(r["kind"], r["N"], r["d"])][r["rep"]] = r
    r_cells, dhat_cells = [], []
    for (kind, N, d), reps in sorted(groups.items()):
        if sorted(reps) != list(range(N_REP)):
            continue
        v = np.array([reps[i]["value"] for i in range(N_REP)], float)
        secs = float(sum(reps[i]["seconds"] for i in range(N_REP)))
        sd = float(v.std(ddof=1))
        if kind == "r":
            r_cells.append({"N": N, "d": d, "n_rep": N_REP, "r_mean": float(v.mean()),
                            "r_sd": sd, "r_se": sd / np.sqrt(N_REP),
                            "values": [float(x) for x in v],
                            "seconds": round(secs, 1), "source": "large_n"})
        else:
            dhat_cells.append({"N": N, "d": d, "n_rep": N_REP,
                               "d_hat_mean": float(v.mean()), "d_hat_sd": sd,
                               "d_hat_se": sd / np.sqrt(N_REP),
                               "values": [float(x) for x in v],
                               "seconds": round(secs, 1), "source": "large_n"})
    return r_cells, dhat_cells


def calibration_check(recs, ext):
    rows = []
    for r in recs:
        if r["set"] != "calibration":
            continue
        key = "r_cells" if r["kind"] == "r" else "dhat_cells"
        cell = next((c for c in ext[key] if c["N"] == r["N"] and c["d"] == r["d"]), None)
        if cell is None:
            continue
        stored = cell["values"][r["rep"]]
        rows.append({"kind": r["kind"], "N": r["N"], "d": r["d"], "rep": r["rep"],
                     "stored": stored, "recomputed": r["value"],
                     "rel_diff": abs(r["value"] / stored - 1.0),
                     "host": r["host"]["system"]})
    return rows


def cmd_merge(a):
    sys.path.insert(0, HERE)
    import reference_tables
    with open(SOLVES, encoding="utf-8") as fh:
        recs = json.load(fh)["records"]
    with open(EXT, encoding="utf-8") as fh:
        st = json.load(fh)
    r_cells, dhat_cells = build_cells(recs)
    for key, new in (("r_cells", r_cells), ("dhat_cells", dhat_cells)):
        keep = [c for c in st[key]
                if not any(c["N"] == n["N"] and c["d"] == n["d"] for n in new)]
        st[key] = sorted(keep + new, key=lambda c: (c["N"], c["d"]))
    st["N_grid"] = sorted({c["N"] for c in st["r_cells"]})
    st = reference_tables.summarise(st)
    with open(EXT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(st, fh, indent=1)
    print(f"merged {len(r_cells)} r cells and {len(dhat_cells)} d_hat cells")
    for c in st["crossings"]:
        print(f"crossing N={c['N']:>5}: nominal {c['d_star_limit_nominal']:.2f} "
              f"measured {c['d_star_limit_measured']:.2f} "
              f"{[round(x, 2) for x in c['measured_ci90']]}")


def cmd_tables(a):
    with open(SOLVES, encoding="utf-8") as fh:
        recs = json.load(fh)["records"]
    with open(EXT, encoding="utf-8") as fh:
        ext = json.load(fh)
    r_cells, dhat_cells = build_cells(recs)
    agree = True
    for key, new in (("r_cells", r_cells), ("dhat_cells", dhat_cells)):
        for c in new:
            e = next((x for x in ext[key] if x["N"] == c["N"] and x["d"] == c["d"]), None)
            agree &= e is not None and e["values"] == c["values"]
    grid_recs = [r for r in recs if r["set"] == "grid"]
    timing = {}
    for kind in ("r", "dhat"):
        for N in sorted({r["N"] for r in grid_recs}):
            s = [r["seconds"] for r in grid_recs if r["kind"] == kind and r["N"] == N]
            m = [r["peak_rss_gb"] for r in grid_recs if r["kind"] == kind and r["N"] == N]
            if s:
                timing[f"{kind}_{N}"] = {
                    "tasks": len(s), "median_s": float(np.median(s)),
                    "max_s": float(np.max(s)), "total_cpu_h": float(np.sum(s) / 3600),
                    "peak_rss_gb_max": float(np.nanmax(m))}
    cal = calibration_check(recs, ext)
    hosts = sorted({json.dumps(r["host"], sort_keys=True) for r in recs})
    out = {"_definition": ("reference-table cells at N > 8192 rebuilt from "
                           "results/reference_tables_large_n_solves.json, with "
                           "the cost of the solves and the reproduction check"),
           "r_cells": r_cells, "dhat_cells": dhat_cells,
           "cells_match_reference_tables": bool(agree),
           "timing": timing, "calibration": cal,
           "calibration_max_rel_diff": float(max([c["rel_diff"] for c in cal] or [float("nan")])),
           "hosts": [json.loads(h) for h in hosts]}
    with open(TABLES, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(r_cells)} r cells, {len(dhat_cells)} d_hat cells; match "
          f"reference_tables.json: {agree}; calibration max rel diff "
          f"{out['calibration_max_rel_diff']:.2e}")
    return 0 if agree else 1


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--N", type=int, nargs="+", default=list(N_LARGE))
    r.add_argument("--budget-gb", type=float, default=20.0)
    r.add_argument("--reserve-gb", type=float, default=2.5)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    r.add_argument("--shards", required=True)
    r.add_argument("--status", default=None)
    r.add_argument("--calibrate", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--shards", required=True)
    c.add_argument("--N", type=int, nargs="+", default=None)
    sub.add_parser("merge")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "collect":
        return cmd_collect(a)
    if a.cmd == "merge":
        return cmd_merge(a)
    return cmd_tables(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
