# Maintaining the project

How the parts of the repository depend on each other, which checks hold them
together, and the steps for the usual changes.

## How results reach the manuscript

```
certgen/ (library) ──> experiments/*.py ──> results/*.json ──┬──> tools/paper_numbers.py ──> paper/main.tex
                                                              ├──> tools/make_figures.py  ──> paper/figures/
                                                              └──> tools/write_library_tables.py ──> certgen/latent.py
```

- Every experiment writes exactly one artifact named after it, and its
  docstring states the artifact, the paper section and the runtime.
- Every number of the manuscript is computed from the artifacts by
  `tools/paper_numbers.py`. The only other numbers allowed in the text are
  the design parameters and the figures quoted from cited works, both listed
  at the end of that file.
- The figures and the reference tables of the library are generated from
  the artifacts, never edited by hand.

## What the tests enforce

`python -m pytest` checks, among other things:

| area | check |
|---|---|
| library | behaviour of every public function; constants equal the artifacts they were measured in |
| manuscript | every computed number appears in the text and every numeric literal has a source; ASCII source; software version; citations against `references.bib`; the verification record of every reference; figures drawn from the current artifacts; embedded fonts; numbering of the formal statements; the PDF is newer than its sources |
| Lean | the build records certify the current proofs; no `sorry`; every exported result has its axioms pinned |
| repository | no undefined names or unused imports; every artifact has a generator and every experiment is listed in its README; the reproduction log reports no failure; the README example runs; package metadata; no reference to material outside the project |

## Usual changes

**A change in the library.** Run the tests, rerun every experiment that
uses the changed function, then `python tools/make_figures.py`,
`python tools/build_paper.py` and `python tests/manuscript_numbers.py`; the
last lists every number of the text that changed. Update the text, rebuild,
and finish with a full `python tools/reproduce.py --network --fresh`, which
rewrites `results/reproduction_log.json`.

**A new experiment.** Write `experiments/<name>.py` with a docstring that
names `results/<name>.json`, add a row to `experiments/README.md`, add the
numbers it contributes to `tools/paper_numbers.py`, and list it in
`NETWORK` or `SLOW` of `tools/reproduce.py` when it downloads data or runs
for five minutes or more.

**A new reference.** Add the entry to `paper/references.bib`, verify it with
`python tools/verify_bibliography.py --only KEY` (Crossref, arXiv or the
publisher's page, then OpenAlex), and rewrite the log with
`python tools/verify_bibliography.py --log`. An entry no index covers is
checked by hand and recorded in `MANUAL` of that tool with its source.

**The manuscript.** `python tools/build_paper.py` fails on any TeX error and
on any warning of the final pass, including overfull and underfull boxes.
The address of the software repository is the `\repourl` macro near the
top of `paper/main.tex`.

**A release.** Set the version in `pyproject.toml`, `certgen/__init__.py` and
`CITATION.cff`, add the entry to `CHANGELOG.md`, update the version sentence
of Section 9 of the manuscript (a test compares it with the package), and
rebuild the PDF.

## Environment notes

- Python 3.9 or later; `pip install -e ".[dev,figures,experiments]"`
  installs everything the tests, figures and experiments need. The
  benchmark tables are cached by scikit-learn after the first download.
- The manuscript needs TeX Live (or MiKTeX) with `latexmk`, `bibtex`,
  `pdffonts` and `pdftotext`.
- The proofs need Lean 4.15.0 through `elan`. `tools/build_lean.py` asks
  elan for the toolchain of `formal/lean-toolchain` and calls its `lake`
  directly. The Mathlib bridge is built once in a working copy of `formal/`
  (`lake build` in its `bridge/` directory, about an hour from source on
  a desktop processor, or `lake exe cache get` to download the compiled
  library) and then rebuilt in seconds.
- On Windows, two problems can appear while Mathlib is prepared. The
  downloads of `lake exe cache get`, of the Reservoir lookup and of elan go
  through `curl`, which fails where the certificate-revocation servers are
  unreachable; requiring Mathlib by its git URL, as `bridge/lakefile.toml`
  does, and building from source avoids them. ProofWidgets compiles its
  JavaScript with npm, and lake starts `npm.cmd` in a way that makes the npm
  launcher look for itself in the wrong directory; a two-line `npm.cmd`
  placed first on `PATH` for the build, which calls `node` on the absolute
  path of `npm-cli.js`, resolves it.
