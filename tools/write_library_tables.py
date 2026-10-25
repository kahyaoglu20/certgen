"""Copy the reference tables from results/reference_tables.json into
certgen/latent.py.

    python tools/write_library_tables.py           # rewrite the marked block
    python tools/write_library_tables.py --check   # exit 1 if it differs

The block between `# <<REFERENCE-TABLES-BEGIN>>` and
`# <<REFERENCE-TABLES-END>>` is generated (r rounded to 4 decimals, d_hat to
3). Only sizes N for which both tables are complete are written.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAT = os.path.join(ROOT, "certgen", "latent.py")
ART = os.path.join(ROOT, "results", "reference_tables.json")
BEGIN, END = "# <<REFERENCE-TABLES-BEGIN>>\n", "# <<REFERENCE-TABLES-END>>"


def _dict(cells, key, nd):
    lines = []
    for N in sorted({c["N"] for c in cells}):
        row = sorted([c for c in cells if c["N"] == N], key=lambda c: c["d"])
        body = ", ".join(f"{c['d']}: {round(c[key], nd):.{nd}f}" for c in row)
        lines.append(f"    {N}: {{{body}}},")
    return "{\n" + "\n".join(lines) + "\n}"


def block() -> str:
    with open(ART, encoding="utf-8") as fh:
        a = json.load(fh)
    rc, dc = a["r_cells"], a["dhat_cells"]
    full = [N for N in sorted({c["N"] for c in rc})
            if {c["d"] for c in rc if c["N"] == N} == set(a["d_grid_r"])
            and {c["d"] for c in dc if c["N"] == N} == set(a["d_grid_dhat"])]
    rc = [c for c in rc if c["N"] in full]
    dc = [c for c in dc if c["N"] in full]
    return "\n".join([
        f"FLOOR_RATIO = {_dict(rc, 'r_mean', 4)}",
        f"FLOOR_RATIO_SD = {_dict(rc, 'r_sd', 4)}",
        f"FLOOR_RATIO_NREP = {a['n_rep_r']}",
        f"D_HAT_CALIBRATION = {_dict(dc, 'd_hat_mean', 3)}",
        f"D_HAT_CALIBRATION_SD = {_dict(dc, 'd_hat_sd', 3)}",
        f"D_HAT_CALIBRATION_NREP = {a['n_rep_dhat']}",
        f"TABULATED_N = {tuple(full)}",
    ]) + "\n"


def main(argv):
    with open(LAT, encoding="utf-8") as fh:
        src = fh.read()
    i, j = src.index(BEGIN) + len(BEGIN), src.index(END)
    new = block()
    if "--check" in argv:
        return 0 if src[i:j] == new else 1
    with open(LAT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src[:i] + new + src[j:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
