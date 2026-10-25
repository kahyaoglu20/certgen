# Manuscript

*How Many Samples Does a Generative Evaluation Need? Intrinsic Dimension and
the Design of Distributional Comparisons*, typeset with the official JMLR
style.

| file | contents |
|---|---|
| `main.tex` | the manuscript (ASCII source) |
| `references.bib` | the references; each entry is verified by `tools/verify_bibliography.py` |
| `jmlr2e.sty` | the JMLR style file, so that the source compiles on its own |
| `figures/` | the five figures, drawn from `results/` by `tools/make_figures.py` |
| `main.pdf`, `main.bbl` | the compiled manuscript and its bibliography |

## Building

```bash
python tools/make_figures.py      # after a change to an artifact
python tools/build_paper.py       # latexmk, or pdflatex + bibtex
```

## Numbers, references and figures

Every measured number in the text is computed from `results/` by
`tools/paper_numbers.py`, which also lists the design parameters and the
figures quoted from cited works. `tests/test_manuscript.py` requires every
computed number to appear in the text and every numeric literal of the text
to be one of them; it also checks the citations against `references.bib` and
the verification registry, the figures against the artifacts they are drawn
from, the fonts and the numbering of the compiled PDF, and the software
version.

## Repository address

The availability paragraph of Section 9 prints the address set by the
`\repourl` macro near the top of `main.tex`.
