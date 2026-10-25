"""Comparison of the manuscript's numbers with tools/paper_numbers.py.

Used by tests/test_manuscript.py, and runnable on its own to list every
discrepancy while the manuscript is edited:

    python tests/manuscript_numbers.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX = os.path.join(ROOT, "paper", "main.tex")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import paper_numbers  # noqa: E402

# commands whose arguments are not running text
_ARGUMENT_COMMANDS = (r"\\(?:label|ref|cref|Cref|eqref|cite[a-z]*|url|includegraphics"
                      r"|begin|end|hspace|vspace|vskip|cmidrule|setlength|bibliography"
                      r"|bibliographystyle|jmlrheading|ShortHeadings|firstpageno)"
                      r"(?:\[[^\]]*\])*(?:\([^)]*\))?\{[^}]*\}")


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.split("\n"))


def body(tex: str) -> str:
    """The document from \\begin{document} to \\end{document}, without comments."""
    t = strip_comments(tex)
    return t[t.index(r"\begin{document}"):t.index(r"\end{document}")]


def normalise(t: str) -> str:
    t = t.replace("\\,", "").replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t)


def missing_values(tex: str, numbers: dict) -> list:
    """Keys whose value does not occur in the manuscript."""
    flat = normalise(body(tex))
    out = []
    for k, v in numbers.items():
        nv = normalise(v)
        if nv not in flat and nv.replace("\\times", "") not in flat:
            out.append(f"{k} = {v}")
    return out


def literals(tex: str) -> set:
    t = re.sub(_ARGUMENT_COMMANDS, " ", body(tex))
    t = re.sub(r"\d*\.?\d+\\(?:textwidth|linewidth|columnwidth)", " ", t)
    t = normalise(t).replace("\\%", "%")
    return set(re.findall(r"(?<![\w.\\])(\d+(?:\.\d+)?)", t))


def unaccounted_literals(tex: str, numbers: dict) -> list:
    """Numeric literals (decimals, or integers of 10 and above) of the body
    that are neither computed values, design parameters nor cited figures."""
    known = set(paper_numbers.DESIGN_PARAMETERS) | set(paper_numbers.CITED_FIGURES)
    for v in numbers.values():
        known |= set(re.findall(r"\d+(?:\.\d+)?", normalise(str(v))))
    sig = [x for x in literals(tex) if "." in x or int(x) >= 10]
    return sorted(set(sig) - known, key=float)


def main():
    tex = open(TEX, encoding="utf-8").read()
    nums = paper_numbers.all_numbers()
    miss = missing_values(tex, nums)
    unacc = unaccounted_literals(tex, nums)
    print(f"{len(nums)} computed values; {len(miss)} missing from the manuscript")
    for m in miss:
        print("  missing:", m)
    print(f"{len(unacc)} unaccounted literals: {unacc}")
    return 1 if miss or unacc else 0


if __name__ == "__main__":
    sys.exit(main())
