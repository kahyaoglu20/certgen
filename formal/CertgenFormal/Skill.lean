import CertgenFormal.DivField

/-
# The skill score: scale invariance

The certgen reporting layer (§5.1) scores a generator by the floor-corrected
vacuity

    V = (D_obs − f) / (D_B − f),     skill = 1 − V,

where `f` is the floor median and `D_B` a trivial baseline's discrepancy.
Because the floor is subtracted from numerator and denominator alike, `V` is
*dimensionless*: it survives a rescaling of the data, which a raw discrepancy
does not (Chong–Forsyth: fixed-n comparisons of a biased metric are
model-dependently unreliable). This file proves the invariance and the
elementary monotonicity fact.

The asymptotic statement of the paper -- for fixed Q ≠ P, V converges to
D(P,Q)/D(P,Q_B) as the floor vanishes, and to 0 when Q = P -- is proved
in `Analysis.lean`, from a three-law power interface and the classical
ε–N definition of convergence; see `Analysis.vacuity_tendsto` and
`Analysis.skill_asymptotic`.
-/

namespace Certgen

namespace Skill

open LinOrdField LinOrdFieldDiv

universe u

variable {α : Type u} [LinOrdFieldDiv α]

/-- The floor-corrected vacuity ratio `V = (D − f)/(D_B − f)`. -/
def vacuity (D f DB : α) : α := (D - f) / (DB - f)

/-- **Scale invariance.** Rescaling the data by any positive factor leaves
`V` unchanged: subtracting the floor removes the units in which the raw
discrepancy is measured, which is exactly the property that makes skill
scores comparable across sample sizes and tables. -/
theorem vacuity_scale (D f DB t : α) (ht : (0 : α) < t)
    (hden : DB - f ≠ (0 : α)) :
    vacuity (t * D) (t * f) (t * DB) = vacuity D f DB := by
  have e1 : ∀ x y : α, t * x - t * y = t * (x - y) := by
    intro x y
    rw [sub_eq, sub_eq, distrib, neg_mul_right]
  show (t * D - t * f) / (t * DB - t * f) = (D - f) / (DB - f)
  rw [e1 D f, e1 DB f, div_mul_mul (ne_zero_of_pos ht) hden]

/-- **Monotonicity in the observed discrepancy.** With the floor and the
baseline fixed, a larger observed discrepancy means a larger vacuity: worse
generators score worse, with no crossing of levels. -/
theorem vacuity_mono (D1 D2 f DB : α) (hden : (0 : α) < DB - f)
    (h : D1 ≤ D2) : vacuity D1 f DB ≤ vacuity D2 f DB := by
  have hsub : D1 - f ≤ D2 - f := by
    have h1 : -f + D1 ≤ -f + D2 := add_le_add_left h (-f)
    calc D1 - f = D1 + -f := sub_eq D1 f
      _ = -f + D1 := add_comm D1 (-f)
      _ ≤ -f + D2 := h1
      _ = D2 + -f := add_comm (-f) D2
      _ = D2 - f := (sub_eq D2 f).symm
  apply (div_le_iff hden).mpr
  show D1 - f ≤ (D2 - f) / (DB - f) * (DB - f)
  rw [div_mul_cancel (ne_zero_of_pos hden)]
  exact hsub

end Skill
end Certgen
