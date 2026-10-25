import CertgenFormal.Discrepancy

/-
# Prospective sample-size determination

Two things are formalised here.

**Proposition 2 (sensitivity of the prescription).** The certgen sizing rule
inverts a fitted power law `floor(n) = C * n^(-1/d_eff)` to obtain
`n*(phi) = (C/phi)^d_eff`. The paper states the sensitivity of `n*` to the
fitted pair in derivative form. In *logarithmic coordinates* the same
content is exact rather than first-order, and needs no analysis at all:
writing `L = log n*`, `k = log C - log phi`, and `d = d_eff`,

    L = d * k,

so `L` is linear in `k` with slope `d`, and linear in `d` with slope `k`.
The two perturbation statements the paper needs follow as identities:
perturbing `log C` additively by `eps` adds `d * eps` to `L`, and
perturbing `d` multiplicatively by `(1 + delta)` multiplies `L` by
`(1 + delta)`. The second is the one that matters in practice, because the
elasticity of the prescription with respect to the fitted exponent is `L`
itself, so the same relative error in `d_eff` costs more the further the
target lies from the fitted grid.

Working in log coordinates is not a dodge. `log n*` is the quantity the
implementation actually propagates, and the identities below are exactly
the ones its interval construction uses.

**The fail-closed property.** The certgen interface returns *no number* when
the curvature gate rejects the single-exponent hypothesis. That is a
property of the program, not of the mathematics, and it is exactly the kind
of property a proof assistant is good at: `requiredLogN` below is total,
returns an `Option`, and is proved to return `none` on precisely the
inadmissible fits. On the eleven benchmark cells the certgen paper reports,
this path fired once, on `shuttle:E1`.
-/

namespace Certgen

open LinOrdField

universe u

variable {α : Type u} [LinOrdField α]

/-- A fitted floor law, in the coordinates the implementation uses.

`logC` and `logPhi` are the log of the fitted constant and of the target
floor level; `dEff` is the fitted exponent. `admissible` records the verdict
of the curvature gate: `true` exactly when a single exponent is identified
over the fitting grid. -/
structure FloorFit (α : Type u) [LinOrdField α] where
  /-- Log of the fitted constant `C`. -/
  logC : α
  /-- Log of the target floor level `phi`. -/
  logPhi : α
  /-- The fitted decay exponent `d_eff`. -/
  dEff : α
  /-- Verdict of the curvature gate: is a single exponent identified? -/
  admissible : Bool

namespace FloorFit

variable (f : FloorFit α)

/-- The log-scale "distance to target", `log C - log phi`. -/
def gap : α := f.logC - f.logPhi

/-- `log n*`, the log of the prescribed sample size, as a bare quantity.

This is *not* the interface: see `requiredLogN`, which refuses to return it
when the exponent is not identified. -/
def logRequiredRaw : α := f.dEff * f.gap

/-- **The sizing interface, fail-closed.**

Returns the prescription only when the curvature gate has certified that a
single exponent governs the decay over the fitting grid. Where it has not,
the prescription has no referent -- `d_eff` is not a parameter of any law --
and the correct output is no number at all. -/
def requiredLogN : Option α :=
  if f.admissible then some f.logRequiredRaw else none

end FloorFit

namespace Sizing

open FloorFit

/-! ## The fail-closed theorems -/

/-- The interface returns nothing when the exponent is not identified. -/
theorem requiredLogN_eq_none {f : FloorFit α} (h : f.admissible = false) :
    f.requiredLogN = none := by
  simp [FloorFit.requiredLogN, h]

/-- The interface returns a number exactly when the gate admits the fit.
There is no third behaviour: no default, no fallback, no silently
extrapolated value. -/
theorem requiredLogN_isSome_iff {f : FloorFit α} :
    (∃ v : α, f.requiredLogN = some v) ↔ f.admissible = true := by
  constructor
  · intro ⟨v, hv⟩
    by_cases h : f.admissible
    · exact h
    · rw [FloorFit.requiredLogN, if_neg (by simp [h])] at hv
      exact absurd hv (by simp)
  · intro h
    exact ⟨f.logRequiredRaw, by simp [FloorFit.requiredLogN, h]⟩

/-- Whenever the interface does return a number, that number is the
inversion of the fitted law and nothing else. -/
theorem requiredLogN_eq_raw {f : FloorFit α} {v : α}
    (hv : f.requiredLogN = some v) : v = f.dEff * f.gap := by
  by_cases h : f.admissible
  · rw [FloorFit.requiredLogN, if_pos h] at hv
    exact (Option.some.inj hv).symm
  · rw [FloorFit.requiredLogN, if_neg (by simp [h])] at hv
    exact absurd hv (by simp)

/-! ## Proposition 2: sensitivity of the prescription

Each statement below is an identity, exact for every perturbation size, not
a first-order approximation. -/

/-- **Linearity in the constant.** Adding `eps` to `log C` adds
`d_eff * eps` to `log n*`.

This is the exact form of `d(log n*)/d(log C) = d_eff`. -/
theorem log_required_perturb_const (f : FloorFit α) (eps : α) :
    ({ f with logC := f.logC + eps } : FloorFit α).logRequiredRaw
      = f.logRequiredRaw + f.dEff * eps := by
  show f.dEff * (f.logC + eps - f.logPhi) = f.dEff * (f.logC - f.logPhi)
    + f.dEff * eps
  rw [sub_eq, sub_eq, add_assoc, add_comm eps (-f.logPhi), ← add_assoc,
      distrib]

/-- **Homogeneity in the exponent.** Scaling `d_eff` by `(1 + delta)`
scales `log n*` by `(1 + delta)`.

This is the exact form of the elasticity statement: the elasticity of
`log n*` with respect to `d_eff` is one, hence the elasticity of `n*` is
`log n*` itself, which is why a fixed relative error in the fitted exponent
is more expensive the further the target lies from the fitted grid. It is
the mechanism behind the growth of the interval widths with the distance
of the target from the grid (§4.6). -/
theorem log_required_perturb_exponent (f : FloorFit α) (delta : α) :
    ({ f with dEff := f.dEff * (1 + delta) } : FloorFit α).logRequiredRaw
      = f.logRequiredRaw * (1 + delta) := by
  show f.dEff * (1 + delta) * f.gap = f.dEff * f.gap * (1 + delta)
  rw [mul_assoc, mul_comm (1 + delta) f.gap, ← mul_assoc]

/-- **The two sensitivities are genuinely different.** Perturbing the
constant changes `log n*` by an amount independent of the target, while
perturbing the exponent changes it by an amount proportional to `log n*`.
Stated together, so that the asymmetry is visible. -/
theorem sensitivity_asymmetry (f : FloorFit α) (eps delta : α) :
    ({ f with logC := f.logC + eps } : FloorFit α).logRequiredRaw
        - f.logRequiredRaw = f.dEff * eps
    ∧ ({ f with dEff := f.dEff * (1 + delta) } : FloorFit α).logRequiredRaw
        - f.logRequiredRaw = f.logRequiredRaw * delta := by
  constructor
  · rw [log_required_perturb_const, sub_eq,
        add_comm f.logRequiredRaw (f.dEff * eps), add_assoc, add_neg',
        add_zero']
  · rw [log_required_perturb_exponent, distrib, mul_one', sub_eq,
        add_comm f.logRequiredRaw (f.logRequiredRaw * delta), add_assoc,
        add_neg', add_zero']

/-- **Monotonicity.** For a fixed positive gap, a larger fitted exponent
prescribes a larger sample size. The sizing rule is therefore never
self-contradictory: it cannot recommend fewer samples for a harder target. -/
theorem log_required_mono {f g : FloorFit α}
    (hgap : f.gap = g.gap) (hpos : (0 : α) ≤ f.gap)
    (hd : f.dEff ≤ g.dEff) : f.logRequiredRaw ≤ g.logRequiredRaw := by
  show f.dEff * f.gap ≤ g.dEff * g.gap
  rw [← hgap]
  exact mul_le_mul_of_nonneg_right hd hpos

end Sizing
end Certgen
