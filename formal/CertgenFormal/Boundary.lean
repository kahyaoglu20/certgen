import CertgenFormal.Sizing
import CertgenFormal.Vacuity
import CertgenFormal.Pullback
import CertgenFormal.Probability
import CertgenFormal.MonteCarlo
import CertgenFormal.Quantile
import CertgenFormal.Severity
import CertgenFormal.Skill
import CertgenFormal.Interval
import CertgenFormal.Gate
import CertgenFormal.Analysis
import CertgenFormal.Population
import CertgenFormal.Kantorovich

/-
# The formalisation boundary

This file is the point of the development. It states, and where possible
machine-checks, **what has been formally verified and what has not**.

A formalisation that does not say where it stops invites the reader to
assume it stops nowhere. For a project whose central argument is that an
extrapolation must be licensed by a tested specification, shipping a Lean
development without an explicit boundary would be the same error one level
up.

## What is proved

Everything listed in the axiom audit below: the propositions and the lemma
of the paper, the statistical layer, and the analysis, population and
Kantorovich layers --

* the factor-½ bridge between the two-sample surrogate and the one-sample
  statistical term, in expectation form, from a four-law expectation
  interface (monotonicity, additivity, constants, scalar homogeneity);
* the Fréchet mean's optimality among constant generators: the T0 baseline
  of the paper, with the classical variance-plus-shift decomposition;
* the identifiability limit: no detector separates equal laws;
* the Davison–Hinkley validity of both Monte-Carlo p-value tails, the
  equivalence of the pooled form with the code's `(count+1)/(n+1)`, the
  significance floor `1/(n_rep+1)`, and the `n_rep ≥ 19` budget line;
* the MDD power algebra, the fifty-percent power of the
  two-standard-deviation rule, and the severity bounds;
* the severity alternative's well-posedness: positive exponents, ratio
  exactly drift, slopes that cannot flip, midpoint anchoring;
* the skill score's scale invariance, monotonicity and asymptotics;
* interval endpoint propagation, width growth, and the exactness of the
  extrapolation label;
* the admissibility gate's fail-closed discipline: a NaN is not a pass,
  and a failed gate makes the sizing interface return nothing;
* the pushforward-coupling lemma at the population level, and the
  reduction of the Kantorovich value to the assignment value under the
  integral-optimum property.

Each `#print axioms` below is checked at compile time by `#guard_msgs`. If
any proof anywhere above were to acquire a `sorry`, `sorryAx` would appear
in its axiom list and **the build would fail**. Completeness is therefore
enforced at compile time, not claimed in prose. Fifty-one results are
pinned.

## What is assumed, and where

The trusted base is a short list of class declarations:

1. `Certgen.LinOrdField` (`Order.lean`) -- seventeen laws of a nontrivial
   linearly ordered field: sixteen field and order laws, plus
   nontriviality.
2. `Certgen.Discrepancy` (`Discrepancy.lean`) -- four axioms: the metric
   laws. `W₂` on measures with finite second moment is the intended model.
3. `Certgen.LinOrdFieldDiv` (`DivField.lean`) -- two axioms beyond
   `LinOrdField`: inversion and its cancellation. Used by the statistical
   layer (ratios), not by the core propositions, which remain
   division-free.
4. The statistical interfaces: `Expectation` (four laws),
   `SymCdf` (eight laws), `HasSqrt` (three laws). Each is stated at the
   strength the corresponding theorem needs and no more.
5. The analysis layer: `Analysis` (three laws beyond `LinOrdFieldDiv`:
   positivity of powers, monotonicity in the base, unboundedness of
   `n ↦ (n : α)^s` for positive exponents) -- the interface behind the
   skill-score asymptotics. The whole of items 1-5 is discharged by
   `Real` in the compiled Mathlib bridge (`bridge/`).

Nothing else is assumed. The development does **not** assume the
Weed–Bach rate, any central limit theorem, or any property of Gaussian or
empirical measures.

## What is NOT formalised, and cannot be

The following are **measurements, not theorems**, and no proof assistant
can establish them. They are reported in the paper with their artifacts and
their Monte-Carlo error, which is the appropriate standard of evidence for
an empirical claim:

* that the floor of a real benchmark table obeys a single power law over a
  given grid (what the curvature test *tests*; where it rejects, the
  sizing interface returns nothing, which is itself a proved property);
* the value of any fitted exponent, constant, or interval width;
* the delivered coverage of the interval construction (§4.6);
* the size and power of the curvature test (§4.3, §4.4), and the severity
  thresholds with their combined within-seed and between-seed intervals
  (§4.5);
* the resolution atlas (§8.4);
* the measured tightness of the plan-pullback bound and of the Lipschitz
  route (§7.2), and the ratio of the two-sample to the one-sample distance;
* the correlation induced by shared random numbers across the grid;
* the saturation of the dimension estimate against the nominal dimension,
  the reference tables r(d, N), their crossings, and the calibration map
  between them (§6.3, §6.4).

## The analysis, population and Kantorovich layers

Of the three results the paper states that need more than the trusted base
(limits, powers, or measure theory), two are proved in full:

* the asymptotic n-invariance of the skill score (V → D(P,Q)/D(P,Q_B) for
  fixed Q ≠ P, V → 0 for Q = P) -- `Analysis.lean` extends the trusted
  base by exactly three power laws (positivity, monotonicity in the base,
  and unboundedness of `n ↦ (n:α)^s` for positive exponents), defines the
  classical ε–N limit for sequences, proves the limit algebra (constants,
  sums, negatives, differences, quotients with the denominator bounded
  away from zero), proves the power-law floor `C·(n+1)^(-s)` vanishes, and
  derives both halves of the skill asymptotic;
* the population pushforward-coupling lemma -- `Population.lean` proves,
  from the four-law expectation interface alone, that the pushforward of
  a coupling is a coupling (`pushforward_is_coupling`) and therefore that
  the population transport cost is at most the expected cost of the
  pushed-forward plan (`ppc_population`). The full measure-theoretic
  generality is the interface's own: whatever satisfies the four laws of
  `Expectation` satisfies the lemma, and Lebesgue integration does so
  wherever the integrals exist.

The third, the exact-W₂/Kantorovich equivalence (the integral-optimum
theorem behind computing exact W₂ by assignment), is formalised in
`Kantorovich.lean` up to its combinatorial core: the plan and matching
structures and their costs are defined, the cycle-signature machinery --
point indicators, counting sums, rotation invariance, and Fubini for two
list sums -- is proved from the ordered-field axioms alone, and the
integral-optimum property itself is stated as `HasIntegralOptimum`, with
the reduction `assignment_le_plan_of_integral_optimum` proved: under the
property a minimum-cost matching lower-bounds every fractional plan.
Over the real numbers the property is a theorem: the bridge file
`bridge/BridgeKantorovich.lean` proves `hasIntegralOptimum_real` for every
ground cost and every pair of duplicate-free index lists of equal length,
from Mathlib's Birkhoff--von Neumann theorem, so all three results hold
over `ℝ`. In the Mathlib-free core the property remains an assumption of
the reduction (the walk-shift-peel recursion is not formalised) -- stated,
not papered over. The analysis layer's three power laws are discharged by
`Real.rpow` in the Mathlib bridge (`bridge/`, Lean 4.15.0, Mathlib
v4.15.0), which also discharges the rest of the trusted base against
`Real` (arithmetic, metric discrepancies, `Real.sqrt`, finite
expectations, a symmetric distribution function with explicit quantile),
so every theorem of this development holds, verbatim, over the reals. The
bridge's `SymCdf` instance is the uniform law on `[-1,1]` with clamped
quantile: Mathlib v4.15.0 has no `Real.erf`, and the Gaussian quantile,
being surjective onto `α`, can never satisfy the interface's *global*
quantile monotonicity -- a strength of the interface worth knowing,
explained in the bridge's header.

The relationship between the two halves is the honest one: the theorems say
what follows *if* the empirical premises hold, and the paper measures
whether they hold. `Sizing.requiredLogN_eq_none` and `Gate.gate_nan_not_pass`
are the formal statements that when a premise is not established, the
software returns nothing at all.
-/

namespace Certgen
namespace Boundary

/-! ## Axiom audit

Each `#guard_msgs` below pins the exact axiom list of one exported theorem.
`sorryAx` in any list fails the build. -/

-- Core: Proposition 1(a), the resolution floor's deterministic base.
/-- info: 'Certgen.Discrepancy.pos_of_ne' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Discrepancy.pos_of_ne

-- Core: Proposition 1(b), unbiased estimators must take negative values.
/-- info: 'Certgen.Replicates.exists_neg_of_sum_zero' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Replicates.exists_neg_of_sum_zero

-- Core: Proposition 2, sizing sensitivity (constant perturbation).
/-- info: 'Certgen.Sizing.log_required_perturb_const' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Sizing.log_required_perturb_const

-- Core: Proposition 2, sizing sensitivity (exponent perturbation).
/-- info: 'Certgen.Sizing.log_required_perturb_exponent' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Sizing.log_required_perturb_exponent

-- Core: Proposition 2, the two sensitivities stated together.
/-- info: 'Certgen.Sizing.sensitivity_asymmetry' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Sizing.sensitivity_asymmetry

-- Core: monotonicity of the sizing rule.
/-- info: 'Certgen.Sizing.log_required_mono' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Sizing.log_required_mono

-- Core: the fail-closed sizing interface (NaN case).
/-- info: 'Certgen.Sizing.requiredLogN_eq_none' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.Sizing.requiredLogN_eq_none

-- Core: the fail-closed sizing interface (characterisation).
/-- info: 'Certgen.Sizing.requiredLogN_isSome_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.Sizing.requiredLogN_isSome_iff

-- Core: the interface returns the inversion and nothing else.
/-- info: 'Certgen.Sizing.requiredLogN_eq_raw' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.Sizing.requiredLogN_eq_raw

-- Core: Proposition 3, the vacuity criterion.
/-- info: 'Certgen.Certificate.vacuous_of_statistical_exceeds' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Certificate.vacuous_of_statistical_exceeds

-- Core: the contrapositive a practitioner uses.
/-- info: 'Certgen.Certificate.statistical_le_of_nonVacuous' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Certificate.statistical_le_of_nonVacuous

-- Core: the surrogate does not transfer.
/-- info: 'Certgen.surrogate_does_not_transfer' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.surrogate_does_not_transfer

-- Core: the factor-½ bridge, positive form.
/-- info: 'Certgen.exists_half_bridge' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.exists_half_bridge

-- Core: Proposition 4, the plan-pullback bound.
/-- info: 'Certgen.ppc_le_cost' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.ppc_le_cost

-- Core: the Lipschitz step is optional.
/-- info: 'Certgen.ppc_le_lipschitz' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.ppc_le_lipschitz

-- Core: validity never requires optimality.
/-- info: 'Certgen.ppc_valid_for_any_bijection' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.ppc_valid_for_any_bijection

-- Statistical layer: the integrated triangle inequality.
/-- info: 'Certgen.ExpectationOps.E_dist_triangle' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.ExpectationOps.E_dist_triangle

-- Statistical layer: the factor-½ bridge, in expectation.
/-- info: 'Certgen.ExpectationOps.E_one_sample_ge_half' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.ExpectationOps.E_one_sample_ge_half

-- Statistical layer: the identifiability limit.
/-- info: 'Certgen.ExpectationOps.no_detector_separates' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.ExpectationOps.no_detector_separates

-- Statistical layer: the Fréchet decomposition (T0 baseline).
/-- info: 'Certgen.ExpectationOps.frechet_decomposition' depends on axioms: [Quot.sound] -/
#guard_msgs in
#print axioms Certgen.ExpectationOps.frechet_decomposition

-- Statistical layer: the Fréchet mean is optimal (T0 optimal among
-- constant generators).
/-- info: 'Certgen.ExpectationOps.frechet_optimal' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.ExpectationOps.frechet_optimal

-- Statistical layer: validity of the upper-tail Monte-Carlo p-value.
/-- info: 'Certgen.MC.mc_upper_valid' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.MC.mc_upper_valid

-- Statistical layer: validity of the lower-tail Monte-Carlo p-value.
/-- info: 'Certgen.MC.mc_lower_valid' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.MC.mc_lower_valid

-- Statistical layer: the pooled form is exactly the code's p-value.
/-- info: 'Certgen.MC.pUp_pool_code' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.MC.pUp_pool_code

-- Statistical layer: the significance floor 1/(n_rep+1).
/-- info: 'Certgen.MC.pUpCode_min' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.MC.pUpCode_min

-- Statistical layer: the n_rep >= 19 budget line.
/-- info: 'Certgen.MC.n_rep_floor_19' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.MC.n_rep_floor_19

-- Statistical layer: the MDD power algebra.
/-- info: 'Certgen.Power.mdd_power' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Power.mdd_power

-- Statistical layer: the two-standard-deviation rule has fifty percent power.
/-- info: 'Certgen.Power.power_at_critical_value' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Power.power_at_critical_value

-- Statistical layer: the severity lower bound.
/-- info: 'Certgen.Power.severity_lower' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.Power.severity_lower

-- Statistical layer: the severity upper bound.
/-- info: 'Certgen.Power.severity_upper' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Power.severity_upper

-- Statistical layer: square roots of positives are positive.
/-- info: 'Certgen.Severity.sqrt_pos' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Severity.sqrt_pos

-- Statistical layer: no slope of the severity alternative can flip.
/-- info: 'Certgen.Severity.slopes_negative' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Severity.slopes_negative

-- Statistical layer: the half-slope ratio is exactly the drift.
/-- info: 'Certgen.Severity.ratio_is_drift' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Severity.ratio_is_drift

-- Statistical layer: midpoint anchoring of the alternative.
/-- info: 'Certgen.Severity.anchored' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Severity.anchored

-- Statistical layer: scale invariance of the skill score.
/-- info: 'Certgen.Skill.vacuity_scale' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Skill.vacuity_scale

-- Statistical layer: monotonicity of the skill score.
/-- info: 'Certgen.Skill.vacuity_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Skill.vacuity_mono

-- Statistical layer: interval endpoint propagation.
/-- info: 'Certgen.Interval.logn_bounds' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Interval.logn_bounds

-- Statistical layer: width grows linearly in the log-gap.
/-- info: 'Certgen.Interval.width_grows' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Interval.width_grows

-- Statistical layer: the extrapolation label is exact.
/-- info: 'Certgen.Interval.extrapolation_label' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Interval.extrapolation_label

-- Statistical layer: a NaN is not a pass.
/-- info: 'Certgen.Gate.gate_nan_not_pass' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Gate.gate_nan_not_pass

-- Statistical layer: the pass condition, unfolded.
/-- info: 'Certgen.Gate.gate_pass_char' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Gate.gate_pass_char

-- Statistical layer: the gate composes with the sizing interface,
-- fail-closed.
/-- info: 'Certgen.Gate.gate_fail_closed_through_sizing' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Certgen.Gate.gate_fail_closed_through_sizing

-- Analysis layer: the power-law floor vanishes.
/-- info: 'Certgen.Analysis.powLaw_tendsto_zero' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Analysis.powLaw_tendsto_zero

-- Analysis layer: the skill-score asymptotic (boundary theorem 1 of 3).
/-- info: 'Certgen.Analysis.vacuity_tendsto' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Analysis.vacuity_tendsto

-- Analysis layer: the equal-law case, V → 0.
/-- info: 'Certgen.Analysis.vacuity_tendsto_self' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Analysis.vacuity_tendsto_self

-- Analysis layer: the composed asymptotic under the paper's floor model.
/-- info: 'Certgen.Analysis.skill_asymptotic' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Analysis.skill_asymptotic

-- Analysis layer: the composed equal-law asymptotic.
/-- info: 'Certgen.Analysis.skill_asymptotic_self' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Certgen.Analysis.skill_asymptotic_self

-- Population layer: the pushforward of a coupling is a coupling
-- (boundary theorem 2 of 3).
/-- info: 'Certgen.Population.pushforward_is_coupling' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Population.pushforward_is_coupling

-- Population layer: the plan-pullback bound, population form.
/-- info: 'Certgen.Population.ppc_population' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Population.ppc_population

-- Population layer: validity does not require coupling optimality.
/-- info: 'Certgen.Population.ppc_population_any_coupling' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Population.ppc_population_any_coupling

-- Kantorovich layer: the reduction of the value equivalence to the
-- integral-optimum property. The property itself is proved over Real in
-- bridge/BridgeKantorovich.lean (hasIntegralOptimum_real).
/-- info: 'Certgen.Kantorovich.assignment_le_plan_of_integral_optimum' does not depend on any axioms -/
#guard_msgs in
#print axioms Certgen.Kantorovich.assignment_le_plan_of_integral_optimum

/-! ## A worked model

To show the interfaces are not vacuous: `Unit` is not a model of
`LinOrdField` (`zero_ne_one` fails), and the identity assignment witnesses
attainment for `IsEmpiricalW` whenever the identity's cost is minimal, so
the empirical transport predicate is satisfiable. The compiled Mathlib
bridge (`bridge/Bridge.lean`, `bridge/BridgeKantorovich.lean`) discharges the
full trusted base against `Real` and metric spaces. -/

/-- The identity assignment always witnesses attainment when the cost of the
identity is the claimed optimum, so `IsEmpiricalW` is not vacuous. -/
example {α : Type} [LinOrdField α] {ι X : Type} (c : X → X → α)
    (z z' : ι → X) (idx : List ι)
    (hmin : ∀ sigma : Assignment ι,
      cost c z z' Assignment.id' idx ≤ cost c z z' sigma idx) :
    IsEmpiricalW c z z' idx (cost c z z' Assignment.id' idx) :=
  { le_cost := hmin, attained := ⟨Assignment.id', rfl⟩ }

end Boundary
end Certgen
