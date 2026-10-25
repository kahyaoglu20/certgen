import CertgenFormal.DivField

/-
# The power algebra of a two-sided z-test

The certgen paper's §5.4 fixes the minimum detectable difference with a
stated power: `MDD(α, power) = (z_{1−α/2} + z_{power}) · sd_paired`. This file
proves the probability content of that statement at exactly the abstraction
a proof assistant can honour: a **symmetric distribution function with an
explicit quantile function**, eight laws, no measure theory.

The standardized-effect framing avoids `√2 · sd / √n` entirely: `g` below is
the true effect measured in whatever units make the test statistic standard
under the null. The conversion from physical effect to `g` is where the
paper's proxy for the paired standard deviation enters, and that is an
empirical input, not a theorem.

Results:

* `mdd_power` -- an effect at or beyond `z_{1−α/2} + zq power` is detected
  with probability at least that power: the theorem behind
  `FloorReport.mdd`, which sets the minimum detectable difference at a
  stated power.
* `power_at_critical_value` -- the conventional two-standard-deviation rule, the
  `z_{1−α/2}` term alone, sits at fifty percent power up to the `α/2`
  slack of the far tail: `1/2 ≤ power ≤ 1/2 + α/2`.
* `severity_lower` / `severity_upper` -- the deterministic core of the
  severity statement of §4.5: below the critical effect the power stays
  essentially at one half; at it, it crosses one half. What the gate detects
  at a given drift is then a measurement, not a theorem.
-/

namespace Certgen

namespace Power

open LinOrdField LinOrdFieldDiv

universe u

/-- A symmetric distribution function with an explicit quantile function.

Eight laws: `cdf` is monotone, symmetric about the origin, and valued in
`[0, 1]`; `zq` inverts `cdf` on the open unit interval, is monotone, and
places the median at the origin. `Real` with the error function and its
inverse is the intended model; `bridge/Bridge.lean` explains why the
compiled instance uses the uniform law instead. -/
class SymCdf (α : Type u) extends LinOrdFieldDiv α where
  /-- The distribution function. -/
  cdf : α → α
  /-- The distribution function is monotone. -/
  cdf_mono : ∀ x y : α, x ≤ y → cdf x ≤ cdf y
  /-- Symmetry about the origin. -/
  cdf_symm : ∀ x : α, cdf (-x) = 1 - cdf x
  /-- The distribution function is nonnegative. -/
  cdf_nonneg : ∀ x : α, (0 : α) ≤ cdf x
  /-- The distribution function is at most one. -/
  cdf_le_one : ∀ x : α, cdf x ≤ (1 : α)
  /-- The quantile function. -/
  zq : α → α
  /-- `zq` inverts `cdf` on the open unit interval. -/
  zq_char : ∀ {t : α}, (0 : α) < t → t < (1 : α) → cdf (zq t) = t
  /-- The quantile function is monotone. -/
  zq_mono : ∀ {s t : α}, s ≤ t → zq s ≤ zq t
  /-- The median sits at the origin. -/
  zq_half : zq (1 / ((1 : α) + 1)) = (0 : α)

variable {α : Type u} [SymCdf α]

open SymCdf

/-! ### Halves and cancellation helpers -/

theorem one_add_ne_zero : (1 : α) + 1 ≠ (0 : α) := by
  intro he
  have hlt : (0 : α) + 1 < (1 : α) + 1 := add_lt_add_right zero_lt_one 1
  rw [he, zero_add] at hlt
  exact zero_ne_one (le_antisymm (le_of_lt hlt) zero_le_one).symm

theorem zero_lt_two : (0 : α) < 1 + 1 :=
  (lt_iff _ _).mpr ⟨zero_le_two, fun he => one_add_ne_zero he.symm⟩

theorem half_add : (1 : α) / ((1 : α) + 1) + 1 / ((1 : α) + 1) = (1 : α) := by
  rw [show (1 : α) / ((1 : α) + 1) + 1 / ((1 : α) + 1) = (1 + 1) / ((1 : α) + 1) from
      div_add_div_same,
    div_self one_add_ne_zero]

theorem half_pos : (0 : α) < 1 / ((1 : α) + 1) :=
  div_pos zero_lt_one zero_lt_two

theorem half_nonneg : (0 : α) ≤ 1 / ((1 : α) + 1) :=
  le_of_lt half_pos

theorem half_le_one : (1 : α) / ((1 : α) + 1) ≤ (1 : α) := by
  have h : (1 : α) / ((1 : α) + 1) ≤ 1 / ((1 : α) + 1) + 1 / ((1 : α) + 1) :=
    le_add_of_nonneg_right half_nonneg
  rw [half_add] at h
  exact h

theorem half_lt_one : 1 / ((1 : α) + 1) < (1 : α) := by
  cases Classical.em (1 / ((1 : α) + 1) = (1 : α)) with
  | inl he =>
      have hha : (1 : α) / ((1 : α) + 1) + 1 / ((1 : α) + 1) = 1 := half_add
      rw [he] at hha
      have h4 : (1 : α) + 1 = (0 : α) + 1 := by rw [hha, zero_add]
      have h3 : (1 : α) = (0 : α) := add_right_cancel h4
      exact absurd h3 one_ne_zero
  | inr hne => exact (lt_iff _ _).mpr ⟨half_le_one, hne⟩

theorem sub_add_cancel' (a b : α) : a - b + b = a := by
  rw [sub_eq, add_assoc, neg_add, add_zero']

theorem add_sub_cancel2 (b a : α) : b + (a - b) = a := by
  rw [add_comm b (a - b)]
  exact sub_add_cancel' a b

theorem sub_sub_cancel' (a b : α) : a - (a - b) = b := by
  apply add_right_cancel (c := a - b)
  calc (a - (a - b)) + (a - b) = a := sub_add_cancel' a (a - b)
    _ = b + (a - b) := (add_sub_cancel2 b a).symm

theorem add_le_add_right_cancel' {a b c : α} (h : a + c ≤ b + c) : a ≤ b := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hlt : b < a := lt_of_not_le (fun hle => hcon hle)
  exact not_lt_of_le h (add_lt_add_right hlt c)

theorem sub_le_sub_right {c x y : α} (h : x ≤ y) : c - y ≤ c - x := by
  have e1 : (c - y) + x ≤ (c - x) + x := by
    calc (c - y) + x ≤ (c - y) + y := add_le_add_left h (c - y)
      _ = c := sub_add_cancel' c y
      _ = (c - x) + x := (sub_add_cancel' c x).symm
  exact add_le_add_right_cancel' e1

theorem eq_div_of_mul_eq_one {a b : α} (hb : b ≠ (0 : α))
    (h : b * a = (1 : α)) : a = 1 / b := by
  rw [one_div_eq_inv]
  calc a = (inv b * b) * a := by rw [inv_mul_cancel hb, one_mul]
    _ = inv b * (b * a) := by rw [mul_assoc]
    _ = inv b * (1 : α) := by rw [h]
    _ = inv b := mul_one' (inv b)

/-! ### The symmetric CDF at the origin, and the critical value -/

theorem cdf_zero : cdf (0 : α) = 1 / ((1 : α) + 1) := by
  have h := cdf_symm (0 : α)
  rw [neg_zero] at h
  have h2 : cdf (0 : α) + cdf (0 : α) = 1 := by
    have e1 : cdf (0 : α) + cdf (0 : α) = (1 - cdf (0 : α)) + cdf (0 : α) :=
      congrArg (fun x => x + cdf (0 : α)) h
    rw [e1, sub_add_cancel']
  have h3 : (1 + 1) * cdf (0 : α) = 1 := by rw [two_mul]; exact h2
  exact eq_div_of_mul_eq_one one_add_ne_zero h3

theorem half_le {a : α} : a / ((1 : α) + 1) ≤ 1 / ((1 : α) + 1) ↔ a ≤ (1 : α) := by
  constructor
  · intro h
    have h1 : a ≤ (1 : α) / ((1 : α) + 1) * (1 + 1) := (div_le_iff zero_lt_two).mp h
    rw [div_mul_cancel one_add_ne_zero] at h1
    exact h1
  · intro h
    apply (div_le_iff zero_lt_two).mpr
    calc a ≤ (1 : α) := h
      _ = (1 : α) / ((1 : α) + 1) * (1 + 1) := (div_mul_cancel one_add_ne_zero 1).symm

theorem zq_nonneg {t : α} (ht : 1 / ((1 : α) + 1) ≤ t) : (0 : α) ≤ zq t := by
  rw [← zq_half]
  exact zq_mono ht

theorem crit_nonneg {a : α} (ha1 : a < (1 : α)) :
    (0 : α) ≤ zq (1 - a / ((1 : α) + 1)) := by
  apply zq_nonneg
  have hhalf : (1 : α) - 1 / ((1 : α) + 1) = 1 / ((1 : α) + 1) := by
    apply add_right_cancel (c := 1 / ((1 : α) + 1))
    rw [sub_add_cancel', half_add]
  rw [← hhalf]
  exact sub_le_sub_right ((half_le).mpr (le_of_lt ha1))

theorem crit_pos {a : α} (_ha0 : (0 : α) < a) (ha1 : a < (1 : α)) :
    (0 : α) < 1 - a / ((1 : α) + 1) := by
  have hle : a / ((1 : α) + 1) ≤ 1 / ((1 : α) + 1) := (half_le).mpr (le_of_lt ha1)
  have hlt : a / ((1 : α) + 1) < (1 : α) := lt_of_le_of_lt hle half_lt_one
  exact sub_pos_of_lt hlt

theorem sub_lt_iff' {a b c : α} : a - c < b ↔ a < b + c := by
  constructor
  · intro h
    calc a = (a - c) + c := (sub_add_cancel' a c).symm
      _ < b + c := add_lt_add_right h c
  · intro h
    have h2 : (a - c) + c < b + c := by rw [sub_add_cancel' a c]; exact h
    exact lt_of_add_lt_add_right h2

theorem crit_lt_one {a : α} (ha0 : (0 : α) < a) :
    1 - a / ((1 : α) + 1) < (1 : α) := by
  have hlt : (0 : α) < a / ((1 : α) + 1) := div_pos ha0 zero_lt_two
  rw [sub_lt_iff']
  exact lt_add_of_pos_right hlt

theorem crit_char {a : α} (ha0 : (0 : α) < a) (ha1 : a < (1 : α)) :
    cdf (zq (1 - a / ((1 : α) + 1))) = 1 - a / ((1 : α) + 1) :=
  zq_char (crit_pos ha0 ha1) (crit_lt_one ha0)

/-! ### The power function -/

/-- The two-sided z-test power at standardized effect `g` and level `a`:
the rejection probability of the level-`a` two-sided test when the true
effect is `g` standard units. -/
def power2 (g a : α) : α :=
  cdf (g - zq (1 - a / ((1 : α) + 1))) + cdf (-g - zq (1 - a / ((1 : α) + 1)))

theorem power2_ge_first (g a : α) :
    cdf (g - zq (1 - a / ((1 : α) + 1))) ≤ power2 g a :=
  le_add_of_nonneg_right (cdf_nonneg _)

/-! ### The MDD power algebra -/

/-- **The MDD power algebra.** An effect at or beyond the sum of the
significance quantile and the power quantile is detected with probability at
least the stated power: the theorem behind `mdd_at_power`. The hypotheses
`0 < b < 1` make `b` the stated power; the paper's type-II error is `1 − b`. -/
theorem mdd_power {a b g : α} (hb0 : (0 : α) < b) (hb1 : b < 1)
    (hg : zq (1 - a / ((1 : α) + 1)) + zq b ≤ g) : b ≤ power2 g a := by
  have hgc : zq b ≤ g - zq (1 - a / ((1 : α) + 1)) := by
    have e1 : zq b + zq (1 - a / ((1 : α) + 1))
        ≤ (g - zq (1 - a / ((1 : α) + 1))) + zq (1 - a / ((1 : α) + 1)) := by
      rw [sub_add_cancel', add_comm]
      exact hg
    exact add_le_add_right_cancel' e1
  have h1 : b ≤ cdf (g - zq (1 - a / ((1 : α) + 1))) := by
    rw [← zq_char hb0 hb1]
    exact cdf_mono _ _ hgc
  exact le_trans h1 (power2_ge_first g a)

/-- **The two-standard-deviation rule has fifty percent power.** At a true
effect exactly equal to the significance quantile -- the `z_{1−α/2}` term
with no power term -- the detection probability is between one half and
`1/2 + α/2`. The lower bound is the paper's analytical power of one half;
the upper bound is the far tail's contribution, which is negligible for the
conventional `α = 0.05` but is stated exactly. -/
theorem power_at_critical_value {a : α} (ha0 : (0 : α) < a) (ha1 : a < (1 : α)) :
    1 / ((1 : α) + 1) ≤ power2 (zq (1 - a / ((1 : α) + 1))) a
      ∧ power2 (zq (1 - a / ((1 : α) + 1))) a ≤ 1 / ((1 : α) + 1) + a / ((1 : α) + 1) := by
  constructor
  · have hself : zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)) = (0 : α) :=
      sub_self _
    have h1 : cdf (zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)))
        = 1 / ((1 : α) + 1) := by rw [hself, cdf_zero]
    have h2 := power2_ge_first (zq (1 - a / ((1 : α) + 1))) a
    rw [h1] at h2
    exact h2
  · have hc := crit_nonneg ha1
    have hself : zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)) = (0 : α) :=
      sub_self _
    have hfar : cdf (-zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)))
        ≤ cdf (-zq (1 - a / ((1 : α) + 1))) := by
      apply cdf_mono
      have h3 := sub_le_sub_right (c := -zq (1 - a / ((1 : α) + 1))) hc
      rw [sub_zero] at h3
      calc -zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1))
          = -(zq (1 - a / ((1 : α) + 1))) - zq (1 - a / ((1 : α) + 1)) := rfl
        _ ≤ -(zq (1 - a / ((1 : α) + 1))) := h3
    have hsymm : cdf (-zq (1 - a / ((1 : α) + 1))) = a / ((1 : α) + 1) := by
      rw [cdf_symm, crit_char ha0 ha1, sub_sub_cancel']
    have h2 : cdf (-zq (1 - a / ((1 : α) + 1))) ≤ a / ((1 : α) + 1) := by
      rw [hsymm]
      exact le_refl _
    show cdf (zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)))
        + cdf (-zq (1 - a / ((1 : α) + 1)) - zq (1 - a / ((1 : α) + 1)))
        ≤ 1 / ((1 : α) + 1) + a / ((1 : α) + 1)
    rw [hself, cdf_zero]
    exact add_le_add (le_refl _) (le_trans hfar h2)

/-- **The severity lower bound.** At or beyond the critical effect the
two-sided power is at least one half: the 50%-detection threshold of the
paper's severity construction is at most the critical value. -/
theorem severity_lower {a g : α} (hg : zq (1 - a / ((1 : α) + 1)) ≤ g) :
    1 / ((1 : α) + 1) ≤ power2 g a := by
  have hgc : (0 : α) ≤ g - zq (1 - a / ((1 : α) + 1)) := sub_nonneg_of_le hg
  have h1 : cdf ((0 : α)) ≤ cdf (g - zq (1 - a / ((1 : α) + 1))) := cdf_mono _ _ hgc
  rw [cdf_zero] at h1
  exact le_trans h1 (power2_ge_first g a)

/-- **The severity upper bound.** Between zero and the critical effect the
two-sided power stays essentially at one half: at most `1/2 + α/2`. This is
the deterministic content of "what a non-rejection excludes": departures
smaller than the critical value are not detected more than half the time,
up to the far tail. -/
theorem severity_upper {a g : α} (hg0 : (0 : α) ≤ g)
    (hg : g ≤ zq (1 - a / ((1 : α) + 1))) (ha0 : (0 : α) < a) (ha1 : a < (1 : α)) :
    power2 g a ≤ 1 / ((1 : α) + 1) + a / ((1 : α) + 1) := by
  have hle0 : g - zq (1 - a / ((1 : α) + 1)) ≤ (0 : α) := by
    have h2 : -(zq (1 - a / ((1 : α) + 1))) + g ≤ (0 : α) := by
      have h3 := add_le_add_left hg (-(zq (1 - a / ((1 : α) + 1))))
      rw [neg_add] at h3
      exact h3
    rw [sub_eq, add_comm]
    exact h2
  have h1' : cdf (g - zq (1 - a / ((1 : α) + 1))) ≤ 1 / ((1 : α) + 1) := by
    rw [← cdf_zero]
    exact cdf_mono _ _ hle0
  have hneg : cdf (-g - zq (1 - a / ((1 : α) + 1))) ≤ a / ((1 : α) + 1) := by
    have hle : -g - zq (1 - a / ((1 : α) + 1)) ≤ -zq (1 - a / ((1 : α) + 1)) := by
      have h3 := sub_le_sub_right (c := -zq (1 - a / ((1 : α) + 1))) hg0
      rw [sub_zero] at h3
      calc -g - zq (1 - a / ((1 : α) + 1))
          = -(zq (1 - a / ((1 : α) + 1))) - g := by
              rw [sub_eq (-g) (zq (1 - a / ((1 : α) + 1))),
                  sub_eq (-(zq (1 - a / ((1 : α) + 1)))) g,
                  add_comm (-g) (-(zq (1 - a / ((1 : α) + 1))))]
        _ ≤ -(zq (1 - a / ((1 : α) + 1))) := h3
    have hsymm : cdf (-zq (1 - a / ((1 : α) + 1))) = a / ((1 : α) + 1) := by
      rw [cdf_symm, crit_char ha0 ha1, sub_sub_cancel']
    have hle2 : cdf (-zq (1 - a / ((1 : α) + 1))) ≤ a / ((1 : α) + 1) := by
      rw [hsymm]
      exact le_refl _
    exact le_trans (cdf_mono _ _ hle) hle2
  show cdf (g - zq (1 - a / ((1 : α) + 1))) + cdf (-g - zq (1 - a / ((1 : α) + 1)))
      ≤ 1 / ((1 : α) + 1) + a / ((1 : α) + 1)
  exact add_le_add h1' hneg

end Power
end Certgen
