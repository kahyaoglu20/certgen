import CertgenFormal.Sizing

/-
# The admissibility gate, fail-closed

`FloorLaw.single_exponent_ok` (paper §4.2) decides whether a single power
law is an adequate description of a measured floor. Two conditions:

* `R² > 0.95` — the fit is close;
* a finite drift p-value at least 0.05 — the low-half/high-half exponent
  drift is consistent with the single-exponent null.

The second condition does the work: against a floor with no single
exponent anywhere, such as `n^(−1/2) + 0.35 · n^(−1/8)`, the R² of a short
grid stays high, so goodness of fit is no specification test (§4.4). A
fixed threshold on the drift does not work either, since it fires on many
genuine power laws; the calibrated object is the parametric-bootstrap
p-value.

**Fail-closed**: a missing drift p-value is *not* a pass, so a failed
bootstrap cannot certify a single exponent. This file formalises that
discipline: the drift p-value is an `Option`, a
`none` (the NaN of the numerical world) can never satisfy the gate, and the
gate's verdict composes with the sizing interface of `Sizing.lean`, where a
failed gate makes the prescription return nothing at all.
-/

namespace Certgen

namespace Gate

open LinOrdField

universe u

variable {α : Type u} [LinOrdField α]

/-- The gate's inputs: the coefficient of determination and the
parametric-bootstrap drift p-value, as an `Option` whose `none` is the
missing (NaN) p-value of a failed bootstrap. -/
structure GateInputs (α : Type u) [LinOrdField α] where
  /-- The coefficient of determination of the single-law fit. -/
  r2 : α
  /-- The drift p-value, if the bootstrap produced one. -/
  driftP : Option α

/-- The admissibility verdict: close fit, and a finite drift p-value
clearing the threshold. The thresholds are parameters (the production
values are `0.95` and `0.05`); what is proved below is the *shape* of the
rule, in particular its behaviour on missing p-values. -/
def gateAdmits (r2thr pthr : α) (g : GateInputs α) : Prop :=
  r2thr < g.r2 ∧ ∃ p, g.driftP = some p ∧ pthr ≤ p

/-- **A NaN is not a pass.** With no drift p-value at all, the gate
refuses, whatever the fit quality: the fail-closed property, stated as a
theorem. -/
theorem gate_nan_not_pass (r2thr pthr : α) (r : α) :
    ¬ gateAdmits r2thr pthr { r2 := r, driftP := none } := by
  intro hadmits
  have ⟨_, ⟨p, hp, _⟩⟩ := hadmits
  cases hp

/-- **The pass condition, unfolded.** The gate admits the fit exactly when
both conditions hold; in particular a passing gate supplies the p-value it
used, so every admission carries its evidence with it. -/
theorem gate_pass_char (r2thr pthr : α) (g : GateInputs α) :
    gateAdmits r2thr pthr g ↔
      r2thr < g.r2 ∧ ∃ p, g.driftP = some p ∧ pthr ≤ p :=
  Iff.rfl

/-- **The gate composes with the sizing interface, fail-closed.** If the
fit record's `admissible` flag is false, the sizing interface of
`Sizing.lean` returns no prescription: no default, no fallback, no silently
extrapolated value. This is the theorem `Sizing.requiredLogN_eq_none`
restated at the gate's level: the discipline is not a convention but a
proved property of the interface. -/
theorem gate_fail_closed_through_sizing {f : FloorFit α}
    (h : f.admissible = false) : f.requiredLogN = none :=
  Sizing.requiredLogN_eq_none h

end Gate
end Certgen
