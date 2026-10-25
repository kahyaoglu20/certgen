"""Compile paper/main.tex to paper/main.pdf.

    python tools/build_paper.py

Uses latexmk when available and otherwise pdflatex, bibtex, pdflatex,
pdflatex. The build fails on a TeX error and on any warning of the final
pass: undefined references or citations, overfull or underfull boxes, and
LaTeX, package or font warnings. Intermediate files are removed afterwards;
main.pdf and main.bbl are kept.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "paper")
INTERMEDIATE = (".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc")
WARNING = re.compile(r"^(Overfull|Underfull|LaTeX Warning|LaTeX Font Warning|"
                     r"Package \S+ Warning|Class \S+ Warning|pdfTeX warning)")


def run(cmd) -> int:
    p = subprocess.run(cmd, cwd=PAPER, capture_output=True, encoding="utf-8",
                       errors="replace")
    if p.returncode != 0:
        errors = [line for line in p.stdout.splitlines() if line.startswith("!")]
        print("\n".join(errors[:20]) or p.stdout[-2000:])
    return p.returncode


def log_warnings(path: str) -> list:
    """Warnings of the last TeX pass recorded in main.log."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="latin-1") as fh:
        lines = fh.read().splitlines()
    return [line.strip() for line in lines if WARNING.match(line)]


def main() -> int:
    if shutil.which("latexmk"):
        rc = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                  "main.tex"])
    else:
        rc = 0
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
            rc = rc or run(cmd)
    warnings = log_warnings(os.path.join(PAPER, "main.log"))
    for ext in INTERMEDIATE:
        path = os.path.join(PAPER, "main" + ext)
        if os.path.exists(path):
            os.remove(path)
    for w in warnings:
        print(w)
    if rc:
        print("build failed")
    elif warnings:
        print(f"paper/main.pdf built with {len(warnings)} warnings")
    else:
        print("paper/main.pdf built without warnings")
    return 1 if rc or warnings else 0


if __name__ == "__main__":
    sys.exit(main())
