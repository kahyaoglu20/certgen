# CertgenFormal

Lean 4 proofs of the formal statements of *How Many Samples Does a
Generative Evaluation Need?*. The main development has no dependencies and
builds on Lean 4.15.0 without warnings; `bridge/` is a separate project that
instantiates it over the real numbers of Mathlib v4.15.0.

The manuscript contains two kinds of claim: theorems, which follow from
stated hypotheses, and measurements (fitted exponents, coverages, powers,
tightness ratios), which no proof assistant can establish.
`CertgenFormal/Boundary.lean` states the boundary between the two and lists
every measured quantity that is deliberately not formalised.

## What is proved

| result | file | manuscript |
|---|---|---|
| A discrepancy between distinct objects is strictly positive; an unbiased estimator of a discrepancy that vanishes on the diagonal takes negative values | `Discrepancy.lean` | Proposition 1 |
| Sensitivity of the sample-size prescription, as exact identities in log coordinates; the sizing interface returns a value only when the curvature test admits the fit | `Sizing.lean`, `Gate.lean` | Proposition 2, Sections 4.2 and 9 |
| The vacuity criterion with the statistical term in its antecedent; the two-sample surrogate relates to that term only up to a factor of two, also in expectation | `Vacuity.lean`, `Probability.lean` | Proposition 3, Remark 2 |
| The plan-pullback bound for any bijection, and its comparison with the Lipschitz bound | `Pullback.lean` | Lemma 1, Proposition 4 |
| The pushforward of a coupling is a coupling, and the population transport cost is at most the expected cost of the pushed-forward plan | `Population.lean` | Lemma 1 |
| Validity of both Monte-Carlo p-value tails in the form (count + 1)/(n_rep + 1); the significance floor 1/(n_rep + 1) and the requirement n_rep >= 19 for p <= 0.05 | `MonteCarlo.lean` | Section 3.2 |
| The minimum-detectable-difference power algebra, the fifty-percent power of the two-standard-deviation rule, and the severity bounds | `Quantile.lean`, `Severity.lean` | Sections 4.5 and 5.4 |
| Scale invariance and monotonicity of the skill score; its limits V -> W2(P,Q)/W2(P,Q_B) and V -> 0 as the floor vanishes | `Skill.lean`, `Analysis.lean` | Definition 6 |
| Endpoint propagation and width growth of the prescription interval | `Interval.lean` | Section 4.6 |
| The mean is the optimal constant generator; no detector separates equal laws | `Probability.lean` | Sections 5.1 and 5.3 |
| Between two uniform empirical measures of equal size, the transport problem has an optimal solution given by an assignment (from Mathlib's Birkhoff theorem) | `Kantorovich.lean`, `bridge/BridgeKantorovich.lean` | Section 3.5 |

In the Mathlib-free core the last result is proved conditionally on the
integral-optimum property `HasIntegralOptimum`, which the bridge proves over
`Real` for every ground cost.

## The trusted base

| class | file | laws |
|---|---|---|
| `LinOrdField` | `Order.lean` | 17: a nontrivial linearly ordered field without division |
| `LinOrdFieldDiv` | `DivField.lean` | 2 more: inversion and its cancellation |
| `Discrepancy` | `Discrepancy.lean` | 4: the metric laws |
| `Expectation` | `Probability.lean` | 4: monotonicity, additivity, constants, scalar homogeneity |
| `SymCdf` | `Quantile.lean` | 8: a symmetric distribution function with a monotone quantile |
| `HasSqrt` | `Severity.lean` | 3: non-negativity and squaring |
| `Analysis` | `Analysis.lean` | 3: positivity, monotonicity and unboundedness of powers |

The bridge discharges every interface over `Real`: `Real` models the
arithmetic base, every metric space gives a `Discrepancy`, `Real.sqrt`
models `HasSqrt`, finite weighted sums model `Expectation`, the uniform law
on [-1, 1] with clamped quantile models `SymCdf`, and `Real.rpow` models
`Analysis`. Mathlib v4.15.0 has no `Real.erf`, and the interface's global
quantile monotonicity excludes cdfs with full support, so the Gaussian
law is not instantiated; the bridge's header records both facts.

## Axiom audit

`Boundary.lean` runs `#print axioms` on fifty-one exported results and pins
each output with `#guard_msgs`, so a proof that used `sorry` or acquired a
new axiom fails the build. Most results, including Propositions 2 to 4, the
Monte-Carlo validity theorems and the power algebra, use no axioms at all;
the few that use classical reasoning are listed with their axioms in the
file. The six theorems of `bridge/BridgeKantorovich.lean` are pinned in the
same way.

## Building

```bash
elan toolchain install leanprover/lean4:v4.15.0
lake build                      # the core: under a minute, no network
cd bridge && lake build         # the bridge: Mathlib v4.15.0 and its dependencies
```

`bridge/lake-manifest.json` pins Mathlib and every dependency to a commit.
`lake exe cache get` in `bridge/` downloads the compiled Mathlib (about
5 GB) instead of building the roughly two thousand modules the bridge
imports from source.

`python tools/build_lean.py [--bridge-dir DIR]` builds the core in a
temporary copy and the bridge in DIR, a working copy of `formal/` prepared
once as above, and records the outcome in `results/lean_build.json` and
`results/lean_bridge_build.json`: the number of Lean warnings and a digest of
the sources without comments. `tests/test_formal.py` checks that the records
certify the current code.

## Layout

```
CertgenFormal/
  Order.lean, DivField.lean    the arithmetic base
  Discrepancy.lean             metric base, Proposition 1, list sums
  Sizing.lean, Gate.lean       Proposition 2, the admissibility rule
  Vacuity.lean                 Proposition 3, surrogate and statistical term
  Pullback.lean, Population.lean   Lemma 1, Proposition 4
  Probability.lean             expectations, the factor-two bridge, the mean
  MonteCarlo.lean              Monte-Carlo p-values
  Quantile.lean, Severity.lean power algebra and severity
  Skill.lean, Analysis.lean    the skill score and its limits
  Interval.lean                the prescription interval
  Kantorovich.lean             plans, matchings, the integral-optimum reduction
  Boundary.lean                the formalisation boundary and the axiom audit
bridge/
  Bridge.lean                  every interface over Real
  BridgeKantorovich.lean       the integral optimum over Real
```
