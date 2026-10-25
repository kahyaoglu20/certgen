# certgen

> **Preprint — work in progress.** The manuscript in `paper/` is an unpublished draft
> prepared for submission to the *Journal of Machine Learning Research*. It has not yet
> been peer-reviewed or reviewed by my advisor, and its methods, results and wording may
> change. Please cite it as a preprint.

[Manuscript (PDF)](paper/main.pdf) · [Project page](https://kahyaoglu20.github.io/#certgen) · [Portfolio](https://kahyaoglu20.github.io)

**Prospective design and floor-corrected reporting for evaluations of
generative models based on the 2-Wasserstein distance.**

Two independent samples of the same data are at a strictly positive W₂
distance from each other. This *resolution floor* is what an ideal generator
would still score, and it decays with the sample size at a rate set by the
intrinsic dimension of the data. `certgen` measures the floor on the real
data alone, before any model is trained, and uses it to answer three
questions:

- **Design.** How many samples does an evaluation need to resolve a
  difference of a given size, and is the sample-size rule valid for these
  data?
- **Reporting.** How good is a generator relative to simple reference
  generators, is its sample suspiciously close to the training data, and
  which differences between models are detectable at stated error rates?
- **Certification.** Above which intrinsic dimension can no latent-generative
  Wasserstein certificate of the standard form be informative at a given
  sample size?

The methods, their operating characteristics and an empirical study on public
tabular benchmarks are described in the accompanying manuscript,
*How Many Samples Does a Generative Evaluation Need? Intrinsic Dimension and
the Design of Distributional Comparisons* (`paper/main.pdf`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # the library: numpy and scipy only
pip install -e ".[dev,figures,experiments]"   # tests, figures, all experiments
```

## Usage

```python
import certgen as cg

# Design: fit the floor law on the real data and size the evaluation.
law = cg.fit_floor_law(X_real)
print(law.summary())
if law.single_exponent_ok:
    print(law.required_n_interval(target_floor=0.05, calibrate=True))
    print(cg.resolvable_effect(law, n=2000, alpha=0.05, power=0.80))
else:
    print(law.required_n_range(target_floor=0.05))

# Reporting: evaluate a generated sample.
report = cg.evaluate(X_real, X_generated)
print(report.summary())            # skill, both tail p-values, leakage screen

# Certification: the dimension limit at N observations.
limit = cg.certification_limit(4096)
print(limit["d_star_limit_measured"], limit["d_star_limit_measured_ci90"])
print(cg.intrinsic_dimension(X_real).summary())
```

The same functions are available on the command line:

```bash
certgen design real.csv --target 0.05 --severity
certgen resolve real.csv --n 2000
certgen evaluate real.csv generated.csv
certgen limit 4096 --dhat 12.3
```

Sample sizes are withheld (returned as NaN) when the floor profile does not
follow a single power law or when its exponent lies below the
dimension-governed regime; `FloorLaw.required_n_range` then shows how far the
half-grid prescriptions disagree.

## Repository

| folder | contents |
|---|---|
| `certgen/` | the library |
| `experiments/` | one script per measurement reported in the manuscript; see [`experiments/README.md`](experiments/README.md) |
| `results/` | the measured artifacts, one JSON file per experiment |
| `paper/` | the manuscript (`main.tex`, `references.bib`, `figures/`, `main.pdf`) |
| `formal/` | Lean 4 proofs of the formal statements; see [`formal/README.md`](formal/README.md) |
| `tools/` | builds of the manuscript's numbers, figures and PDF, the Lean build, the bibliography verification and the reproduction run |
| `tests/` | the test suite |
| `docs/` | [reproducing the results](docs/reproducing.md), [maintaining the project](docs/maintaining.md) and the [bibliographic verification](docs/bibliography.md) |

## Verification

```bash
python -m pytest                    # library, manuscript, Lean records, repository (a few minutes)
python tools/reproduce.py           # re-run every offline experiment and compare (about 25 minutes)
```

The test suite checks the library's behaviour; that every number in the
manuscript is computed from an artifact and every numeric literal in it is
accounted for; that every reference is verified in two bibliographic indexes;
that every figure is drawn from the current artifacts; and that the recorded
Lean builds certify the current proofs. `tools/reproduce.py` re-executes the
experiments in a copy of the project and compares every regenerated artifact
with the stored one. See [`docs/reproducing.md`](docs/reproducing.md).

## Scope

The framework uses W₂ because its plug-in estimator is biased upward under
the same-law null; unbiased statistics such as the energy distance and the
MMD have no floor, and `fit_floor_law` refuses them. It ranks no generators.
The leakage screen protects the validity of a fidelity statement and is not a
privacy guarantee. The certification limit concerns what can be proved about
a generator at a given sample size, not how good the generator is.

The latent-generative Wasserstein certificate analysed here is that of Dao et
al. (2023) and Lee et al. (2025); the rate of the floor is that of Weed and
Bach (2019). [`CHANGELOG.md`](CHANGELOG.md) describes the current release.
