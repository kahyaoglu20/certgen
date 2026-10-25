# Reproducing the results

Everything the manuscript reports is computed from the artifacts in
`results/`, and every artifact is written by one script in `experiments/`
(see [`experiments/README.md`](../experiments/README.md)). This page lists
the steps from a fresh checkout to a rebuilt manuscript.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -e ".[dev,figures,experiments]"
```

The library needs only NumPy and SciPy. The extras add scikit-learn and
pandas (benchmark tables from OpenML), PyTorch (the trained-decoder study),
matplotlib (figures), and pytest, pyflakes and certifi (tests and the
bibliography check). Building the manuscript needs a TeX distribution with
`latexmk` or `pdflatex` and `bibtex`; building the proofs needs Lean 4.15.0
through `elan`.

## Checking the stored results

```bash
python -m pytest
```

The suite runs in a few minutes, without network access. Besides the
library, it checks that every number computed from the artifacts appears in
`paper/main.tex` and that every numeric literal there is accounted for, that
the references match their verification records, that the figures were
drawn from the current artifacts, that the compiled PDF is current, and that
the recorded Lean builds certify the current proofs.

## Re-running the experiments

```bash
python tools/reproduce.py                 # offline experiments, two jobs
python tools/reproduce.py --quick         # without the three slow ones
python tools/reproduce.py --network       # also the OpenML experiments
```

`tools/reproduce.py` copies the project to a temporary directory, runs the
experiments there, and compares every regenerated artifact with the stored
one: numbers to a relative tolerance of 10⁻⁶, every other value exactly,
timing and platform fields excepted; `results/` itself is never modified.
The outcome is written to `results/reproduction_log.json`. The reference
tables and the resolution atlas keep their measured cells in their
artifacts, and the copy reuses them unless `--fresh` is given (from
scratch, the reference tables take about 1.5 CPU hours). With two jobs, the
offline run takes about 25 minutes on a current desktop processor.

Two kinds of difference are expected and reported as such. Neural-network
training is seeded but not bit-identical across floating-point platforms,
so the trained-decoder study is compared on its conclusions. Two
boundary scenarios of the equivalence test place the statistic exactly on
the decision threshold, so their rates depend on the last bit of a sum; no
value quoted in the manuscript depends on them.

To regenerate a single artifact in place, run its script, for example

```bash
python experiments/stress_tests.py
```

## The reference tables at large N

The tables at N = 16 384 and 32 768 hold an N × N cost matrix per exact
solve (2.1 GB and 8.6 GB), so `experiments/reference_tables_large_n.py`
runs every cell as its own process under a memory budget and writes one
shard per cell; an interrupted run resumes.

```bash
python experiments/reference_tables_large_n.py run --N 16384 32768 --budget-gb 20 --workers 10 --shards shards/
python experiments/reference_tables_large_n.py collect --shards shards/ --N 16384 32768
python experiments/reference_tables_large_n.py merge
python experiments/reference_tables_large_n.py
python tools/write_library_tables.py
```

The last two commands rebuild the per-cell summary and copy the completed
tables into `certgen/latent.py`; `python tools/write_library_tables.py
--check` confirms that the library and the artifact agree.

## Rebuilding the manuscript

```bash
python tools/make_figures.py
python tools/build_paper.py
```

`python tools/paper_numbers.py` prints every number the manuscript takes
from the artifacts, grouped by section. After a change to an artifact,
`python tests/manuscript_numbers.py` lists every computed number that the
text no longer contains and every numeric literal of the text without a
source.

## Verifying the references

```bash
python tools/verify_bibliography.py             # queries Crossref, arXiv and OpenAlex
python tools/verify_bibliography.py --log       # rewrites docs/bibliography.md
```

The test suite runs the offline comparison of `paper/references.bib` with
the stored records.

## Rebuilding the proofs

```bash
python tools/build_lean.py                      # the Mathlib-free core
python tools/build_lean.py --bridge-dir DIR     # also the Mathlib bridge
```

See [`formal/README.md`](../formal/README.md). The core builds in under a
minute without network access. The bridge is built in DIR, a working copy
of `formal/` whose `bridge/` has been built once with `lake build`, which
fetches Mathlib and its dependencies at the commits of
`bridge/lake-manifest.json` and compiles the part of Mathlib the bridge
imports (`lake exe cache get` downloads it compiled instead).
