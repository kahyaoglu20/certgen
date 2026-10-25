"""Draw every figure of the manuscript from the artifacts in results/.

    python tools/make_figures.py

Writes paper/figures/*.pdf and results/figures_manifest.json, which records
for every figure the sha256 of each artifact it is drawn from;
tests/test_manuscript.py fails when an artifact changes after its figure was
drawn.

Fonts are embedded as TrueType (Type 42). Categorical series use the order
blue, orange, aqua; ordered magnitudes use one sequential blue ramp; every
series also has its own marker or line style, so no information is carried
by colour alone.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
SEQ = ["#b5d0f0", "#7fb0e6", "#4a8fdc", "#2a78d6", "#1b5aa8", "#123f78"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None, None
    with open(p, "rb") as fh:
        raw = fh.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), metadata={"CreationDate": None,
                                                   "ModDate": None})
    plt.close(fig)


def fig_floor_laws():
    a, h = _load("benchmark_floor_laws.json")
    if a is None:
        return None
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    n_bad = sum(not r["single_exponent_ok"] for r in a["records"])
    for r in a["records"]:
        n = np.asarray(r["n_grid"], float)
        f = np.asarray(r["floors"], float)
        bad = not r["single_exponent_ok"]
        ax.plot(n, f, marker="s" if bad else "o", ms=3.2 if not bad else 4,
                color=ORANGE if bad else GRAY, lw=1.6 if bad else 0.9,
                alpha=1.0 if bad else 0.85, zorder=3 if bad else 2)
    ax.plot([], [], marker="s", ms=4, color=ORANGE, lw=1.6,
            label=f"not admissible ({n_bad})")
    ax.plot([], [], marker="o", ms=3.2, color=GRAY, lw=0.9,
            label=f"admissible ({len(a['records']) - n_bad})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
    ax.xaxis.set_major_locator(FixedLocator([125, 250, 500, 1000, 2000, 4000]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("per-side sample size $n$")
    ax.set_ylabel("floor $f(n)$ (median $W_2$)")
    ax.grid(True, which="major")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, fontsize=7, handlelength=1.6, columnspacing=1.2)
    _save(fig, "fig_floor_laws.pdf")
    return {"benchmark_floor_laws.json": h}


def fig_curvature_test():
    g, h1 = _load("curvature_test_size.json")
    p, h2 = _load("curvature_test_power.json")
    if g is None or p is None:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    d = np.array([c["d_eff_median"] for c in g["cells"]])
    s = np.array([c["size"] for c in g["cells"]])
    se = np.array([c["se"] for c in g["cells"]])
    a1.errorbar(d, s, yerr=1.96 * se, color=BLUE, marker="o", capsize=2,
                lw=1.4, label="empirical size (95% MC interval)")
    a1.axhline(0.05, color=INK, ls="--", lw=0.8, label="nominal 0.05")
    a1.vlines(4.0, 0.0, 0.17, color=GRAY, ls=":", lw=0.9)    # ends below the legend
    a1.set_ylim(0, 0.235)
    a1.text(4.08, 0.118, r"$d_{\mathrm{eff}}=4$", color="#52514e", fontsize=8, va="center", ha="left")
    a1.set_xlabel(r"$d_{\mathrm{eff}}$ of the single-exponent law")
    a1.set_ylabel("rejection rate")
    a1.set_title("(a) size under single-exponent laws", loc="left")
    a1.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.4,
               borderaxespad=0.1)
    a1.grid(True)
    styles = {0.01: (BLUE, "o", "-"), 0.03: (ORANGE, "s", "--"),
              0.05: (AQUA, "^", "-.")}
    for se_lvl, (col, mk, ls) in styles.items():
        cells = sorted([c for c in p["cells"] if c["grid_kind"] == "densified"
                        and abs(c["median_se"] - se_lvl) < 1e-9],
                       key=lambda c: c["n_points"])
        a2.plot([c["n_points"] for c in cells], [c["power"] for c in cells],
                color=col, marker=mk, ls=ls,
                label=f"floor median-SE {int(se_lvl * 100)}%")
    a2.set_xlabel("grid points within $[125, 4000]$")
    a2.set_ylabel("power (two-scale alternative)")
    a2.set_ylim(0, 1.02)
    a2.set_title("(b) power against a two-scale law", loc="left")
    a2.legend(frameon=False, loc="center right")
    a2.grid(True)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig_curvature_test.pdf")
    return {"curvature_test_size.json": h1, "curvature_test_power.json": h2}


def _ramp(n):
    """n colours along the sequential ramp SEQ[1:], interpolated in RGB, so
    every N gets its own step (for n = 5 these are exactly SEQ[1:])."""
    anchors = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in SEQ[1:]], float)
    t = np.linspace(0, len(anchors) - 1, n)
    rgb = [[np.interp(v, np.arange(len(anchors)), anchors[:, k]) for k in range(3)]
           for v in t]
    return ["#%02x%02x%02x" % tuple(int(round(x)) for x in c) for c in rgb]


def fig_reference_tables():
    e, h = _load("reference_tables.json")
    if e is None or not e.get("crossings"):
        return None
    rc = e["r_cells"]
    Ns = sorted({c["N"] for c in rc})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ramp = _ramp(len(Ns))
    for i, N in enumerate(Ns):
        cs = sorted([c for c in rc if c["N"] == N], key=lambda c: c["d"])
        col = ramp[i]
        a1.plot([c["d"] for c in cs], [c["r_mean"] for c in cs], marker="o",
                ms=3, color=col, label=f"$N={N}$")
    a1.axhline(1.0, color=INK, ls="--", lw=0.8)
    a1.set_xscale("log", base=2)
    a1.set_xlabel("nominal dimension $d$ (isotropic Gaussian)")
    a1.set_ylabel(r"floor ratio $r(d,N)$")
    a1.set_title("(a) floor ratio against dimension", loc="left")
    a1.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.2)
    a1.grid(True)
    cr = sorted(e["crossings"], key=lambda c: c["N"])
    x = [c["N"] for c in cr]
    y = [c["d_star_limit_measured"] for c in cr]
    lo = [c["d_star_limit_measured"] - c["measured_ci90"][0] for c in cr]
    hi = [c["measured_ci90"][1] - c["d_star_limit_measured"] for c in cr]
    a2.errorbar(x, y, yerr=[lo, hi], color=BLUE, marker="o", capsize=2,
                label="measured-units limit (90% CI)")
    for xi, yi in zip(x, y):
        a2.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color="#52514e")
    a2.set_xscale("log", base=2)
    a2.set_xlim(min(x) / 1.3, max(x) * 1.35)
    a2.set_ylim(min(y) - 3.5, max(y) + 2.5)
    a2.set_xlabel("sample size $N$")
    a2.set_ylabel(r"limit on measured $\hat d$")
    a2.set_title("(b) certification limit, measured units", loc="left")
    a2.legend(frameon=False, loc="upper left")
    a2.grid(True)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig_reference_tables.pdf")
    return {"reference_tables.json": h}


def fig_resolution_atlas():
    a, h = _load("resolution_atlas.json")
    if a is None or "summary" not in a:
        return None
    excluded = set(a["summary"].get("not_admissible", []))
    cells = [c for c in a["cells"] if c["table"] not in excluded]
    ns = sorted({c["n_per_side"] for c in cells})
    tables = []
    for c in cells:
        if c["table"] not in tables:
            tables.append(c["table"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 3.1),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    # (a) measured-SD resolution per table: one line per table, direct labels
    ends = []
    for t in tables:
        cs = sorted((c for c in cells if c["table"] == t),
                    key=lambda c: c["n_per_side"])
        x = [c["n_per_side"] for c in cs]
        y = [c["dskill_T1_over_T1_p80_true"] for c in cs]
        ax.plot(x, y, color=BLUE, lw=1.0, marker="o", ms=2.6, alpha=0.9)
        ends.append([y[-1], t])
    ends.sort()
    ly = [np.log10(e[0]) for e in ends]
    for i in range(1, len(ly)):              # minimal vertical repel (log units)
        ly[i] = max(ly[i], ly[i - 1] + 0.19)
    for (v, t), yy in zip(ends, ly):
        ax.annotate(t, (4000, v), xytext=(4700, 10 ** yy), fontsize=6,
                    color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.4))
    ax.axhline(1.0, color=INK, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(220, 22000)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("per-side sample size $n$")
    ax.set_ylabel("resolvable skill difference")
    ax.set_title("(a) resolution per table", loc="left")
    ax.grid(True, which="major")
    # (b) proxy SD over measured paired SD, every cell
    for j, n in enumerate(ns):
        v = np.array([c["proxy_over_true_sd"] for c in a["cells"]
                      if c["n_per_side"] == n], float)
        xs = j + np.linspace(-0.18, 0.18, len(v))
        bx.scatter(xs, v, s=10, color=BLUE, linewidths=0)
        bx.plot([j - 0.28, j + 0.28], [np.median(v)] * 2, color=INK, lw=1.4)
    bx.axhline(1.0, color=INK, ls="--", lw=0.8)
    bx.set_yscale("log")
    bx.set_xticks(range(len(ns)))
    bx.set_xticklabels([str(n) for n in ns])
    bx.set_xlabel("per-side sample size $n$")
    bx.set_ylabel("proxy SD / measured paired SD")
    bx.set_title("(b) proxy SD against measured SD", loc="left")
    bx.grid(True, axis="y")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_resolution_atlas.pdf")
    return {"resolution_atlas.json": h}


def fig_plan_pullback():
    a, h = _load("plan_pullback_vae.json")
    if a is None:
        return None
    rows = [("plan-pullback bound", "ppc_over_exact", BLUE, "o"),
            ("Lipschitz route, local lower bound on $L$",
             "local_lipschitz_over_exact", AQUA, "^"),
            ("Lipschitz route, spectral-norm product",
             "lipschitz_over_exact", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (lab, key, col, mk) in enumerate(rows):
        v = np.array([c[key] for c in a["cells"]], float)
        y = np.full(len(v), len(rows) - 1 - i) + np.linspace(-0.12, 0.12, len(v))
        ax.scatter(v, y, s=12, color=col, marker=mk, linewidths=0)
        ax.plot([np.median(v)] * 2, [y.min() - 0.12, y.max() + 0.12], color=INK,
                lw=1.2)
    ax.axvline(1.0, color=INK, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
    ax.set_xlabel("bound / exact output-space $W_2$ (log scale)")
    ax.grid(True, axis="x")
    _save(fig, "fig_plan_pullback.pdf")
    return {"plan_pullback_vae.json": h}


def main():
    man = {}
    for fn in (fig_floor_laws, fig_curvature_test, fig_reference_tables,
               fig_resolution_atlas, fig_plan_pullback):
        name = fn.__name__ + ".pdf"
        deps = fn()
        if deps is None:
            print(f"  skip {name}: input artifact absent or incomplete")
            continue
        man[name] = {"inputs": deps}
        print(f"  wrote paper/figures/{name}")
    with open(os.path.join(RES, "figures_manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"_definition": ("sha256 of every artifact each manuscript "
                                   "figure is drawn from (tools/make_figures.py)"),
                   "figures": man}, fh, indent=1)


if __name__ == "__main__":
    main()
