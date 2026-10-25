import CertgenFormal.DivField

/-
# The severity alternative parameterization

`FloorLaw.severity` quantifies what a non-rejection of the curvature test
excludes by sweeping a family of alternative floor laws indexed by an
exponent *drift*. Bending a single power law about the grid midpoint in log
space can **flip the sign of the slope** for large drift on tables whose
total log-decay is small, and the rejection rate then collapses exactly
where power should be highest. The alternative is therefore parameterised
by the half-slope *ratio* directly, as a piecewise two-exponent law with

    d_low  = d_eff · √drift     (slope −1/d_low, always negative)
    d_high = d_eff / √drift     (slope −1/d_high, always negative)

This file proves the three properties of that parameterisation which make it
well-posed, and nothing else: both effective exponents are positive; the
realised half-slope ratio (the ratio of the effective exponents) is exactly
`drift`; and both log-log slopes are negative, so no slope can flip. The
measurement of the gate's power against this family remains a Monte-Carlo
statement and is not formalised here.
-/

namespace Certgen

namespace Severity

open LinOrdField LinOrdFieldDiv

universe u

/-- A field with a square root: three axioms beyond `LinOrdFieldDiv`. -/
class HasSqrt (α : Type u) [LinOrdFieldDiv α] where
  /-- The square root. -/
  sqrt : α → α
  /-- Square roots are nonnegative. -/
  sqrt_nonneg : ∀ a, (0 : α) ≤ sqrt a
  /-- Squaring the root recovers a nonnegative argument. -/
  sq_sqrt : ∀ {a : α}, (0 : α) ≤ a → sqrt a * sqrt a = a

variable {α : Type u} [LinOrdFieldDiv α] [HasSqrt α]

open HasSqrt

omit [HasSqrt α] in
theorem inv_ne_zero' {a : α} (ha : a ≠ (0 : α)) : inv a ≠ (0 : α) := by
  intro he
  have h : a * inv a = (1 : α) := mul_inv_cancel ha
  rw [he, mul_zero] at h
  exact zero_ne_one h

omit [HasSqrt α] in
theorem inv_unique {c X : α} (h : c * X = (1 : α)) : X = inv c := by
  have hc : c ≠ (0 : α) := by
    intro he
    rw [he, zero_mul] at h
    exact zero_ne_one h
  calc X = (1 : α) * X := (one_mul X).symm
    _ = (inv c * c) * X := by rw [inv_mul_cancel hc]
    _ = inv c * (c * X) := by rw [mul_assoc]
    _ = inv c * (1 : α) := by rw [h]
    _ = inv c := mul_one' _

theorem sqrt_pos {a : α} (ha : (0 : α) < a) : (0 : α) < sqrt a := by
  have h1 : (0 : α) ≤ sqrt a := sqrt_nonneg a
  cases Classical.em ((0 : α) < sqrt a) with
  | inl h => exact h
  | inr h =>
      exfalso
      have hz : sqrt a = (0 : α) := le_antisymm (le_of_not_lt h) h1
      have h2 : (0 : α) < a := ha
      rw [← sq_sqrt (le_of_lt ha)] at h2
      rw [hz, mul_zero] at h2
      exact lt_irrefl (0 : α) h2

omit [HasSqrt α] in
theorem neg_lt_zero {x : α} (h : (0 : α) < x) : -x < (0 : α) := by
  apply lt_of_sub_pos
  have e : (0 : α) - (-x) = x := by rw [sub_eq, neg_neg, zero_add]
  rw [e]
  exact h

/-- The low-`n` effective exponent of the alternative: `d_eff · √drift`. -/
def altLo (d drift : α) : α := d * sqrt drift

/-- The high-`n` effective exponent of the alternative: `d_eff / √drift`. -/
def altHi (d drift : α) : α := d / sqrt drift

theorem altLo_pos (d drift : α) (hd : (0 : α) < d) (hdr : (0 : α) < drift) :
    (0 : α) < altLo d drift := mul_pos hd (sqrt_pos hdr)

theorem altHi_pos (d drift : α) (hd : (0 : α) < d) (hdr : (0 : α) < drift) :
    (0 : α) < altHi d drift := div_pos hd (sqrt_pos hdr)

/-- **No slope can flip.** Both log-log slopes of the piecewise alternative,
`−1/d_low` and `−1/d_high`, are strictly negative, for any positive `d_eff`
and any positive drift. A family bent about the grid midpoint lacks this
property, and its rejection rate collapses at exactly the drifts where
power should be highest. -/
theorem slopes_negative (d drift : α) (hd : (0 : α) < d) (hdr : (0 : α) < drift) :
    -(1 / altLo d drift) < (0 : α) ∧ -(1 / altHi d drift) < (0 : α) := by
  constructor
  · exact neg_lt_zero (div_pos zero_lt_one (altLo_pos d drift hd hdr))
  · exact neg_lt_zero (div_pos zero_lt_one (altHi_pos d drift hd hdr))

/-- **The realised half-slope ratio is exactly `drift`.**

The drift statistic that the gate computes is the ratio of the two
half-fit effective exponents, `max/min`; for the alternative family this
ratio is `altLo / altHi`, and the parameterisation delivers it *exactly*,
not approximately: this is what makes the family well-posed as a yardstick
for severity. -/
theorem ratio_is_drift (d drift : α) (hd : (0 : α) < d) (hdr : (0 : α) < drift) :
    altLo d drift / altHi d drift = drift := by
  have hdn : d ≠ (0 : α) := ne_zero_of_pos hd
  have hr : sqrt drift ≠ (0 : α) := ne_zero_of_pos (sqrt_pos hdr)
  have hinv : inv (inv (sqrt drift)) = sqrt drift :=
    (inv_unique (inv_mul_cancel hr)).symm
  show d * sqrt drift / (d / sqrt drift) = drift
  calc d * sqrt drift / (d / sqrt drift)
      = d * sqrt drift * inv (d * inv (sqrt drift)) := rfl
    _ = d * sqrt drift * (inv (inv (sqrt drift)) * inv d) := by
        rw [inv_mul' hdn (inv_ne_zero' hr)]
    _ = d * (sqrt drift * (inv (inv (sqrt drift)) * inv d)) := by
        rw [mul_assoc]
    _ = d * ((sqrt drift * inv (inv (sqrt drift))) * inv d) := by
        rw [← mul_assoc (sqrt drift) (inv (inv (sqrt drift))) (inv d)]
    _ = d * ((sqrt drift * sqrt drift) * inv d) := by rw [hinv]
    _ = d * (drift * inv d) := by rw [sq_sqrt (le_of_lt hdr)]
    _ = d * (inv d * drift) := by rw [mul_comm drift (inv d)]
    _ = (d * inv d) * drift := by rw [← mul_assoc]
    _ = (1 : α) * drift := by rw [mul_inv_cancel hdn]
    _ = drift := one_mul drift

/-- **Continuity at the geometric midpoint.** Both pieces of the alternative
are anchored at the same value `f_mid` at the midpoint `ln_mid`, so the law
is well-defined there: the piecewise definition cannot jump. -/
theorem anchored (fmid lnmid d drift : α) (_hd : (0 : α) < d)
    (_hdr : (0 : α) < drift) :
    fmid - (lnmid - lnmid) / altLo d drift = fmid
      ∧ fmid - (lnmid - lnmid) / altHi d drift = fmid := by
  constructor
  · have e : lnmid - lnmid = (0 : α) := sub_self _
    rw [e]
    show fmid - (0 : α) / altLo d drift = fmid
    have e2 : (0 : α) / altLo d drift = (0 : α) := by
      show (0 : α) * inv (altLo d drift) = (0 : α)
      rw [zero_mul]
    rw [e2, sub_zero]
  · have e : lnmid - lnmid = (0 : α) := sub_self _
    rw [e]
    show fmid - (0 : α) / altHi d drift = fmid
    have e2 : (0 : α) / altHi d drift = (0 : α) := by
      show (0 : α) * inv (altHi d drift) = (0 : α)
      rw [zero_mul]
    rw [e2, sub_zero]

end Severity
end Certgen
