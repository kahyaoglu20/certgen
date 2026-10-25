import CertgenFormal.Order

/-
# Discrepancies, replicates, and finite sums

This file fixes the second half of the trusted base: what is assumed about a
distributional discrepancy such as `W₂`.

The interface `Discrepancy` below lists exactly four properties: the
statistic is non-negative, symmetric, satisfies the triangle inequality, and
vanishes precisely on the diagonal. These are the metric axioms, and they
are all the certgen results need. In particular the development never
assumes a rate, never assumes a limit law, and never assumes anything about
the geometry of the underlying space, which is why the results below hold
for any metric discrepancy and not only for `W₂`.

The rate `n^(-1/d*)` of Weed and Bach is deliberately **not** axiomatised
here. It is an empirical input to the certgen protocol, not a hypothesis of
any theorem proved in this development; see `Boundary.lean`.
-/

namespace Certgen

open LinOrdField

universe u v

/-- A metric discrepancy between objects of type `M`, valued in `α`.

`W₂` restricted to measures with finite second moment is the intended
model. So is any other metric: the results proved from this interface are
therefore statements about metric discrepancies in general. -/
class Discrepancy (α : Type u) (M : Type v) [LinOrdField α] where
  /-- The discrepancy itself. -/
  d : M → M → α
  /-- Discrepancies are non-negative. -/
  nonneg : ∀ x y : M, (0 : α) ≤ d x y
  /-- Discrepancies are symmetric. -/
  symm : ∀ x y : M, d x y = d y x
  /-- The triangle inequality. -/
  triangle : ∀ x y z : M, d x z ≤ d x y + d y z
  /-- A discrepancy vanishes exactly on the diagonal. -/
  eq_zero_iff : ∀ x y : M, d x y = 0 ↔ x = y

/-- `dist α x y` is the discrepancy, with the scalar type named explicitly
so that it can be inferred at every use site. -/
abbrev dist (α : Type u) {M : Type v} [LinOrdField α] [Discrepancy α M]
    (x y : M) : α := Discrepancy.d x y

namespace Discrepancy

variable {α : Type u} {M : Type v} [LinOrdField α] [Discrepancy α M]

/-- **A discrepancy between distinct objects is strictly positive.**

This is the deterministic core of the resolution floor: the floor is
positive because two disjoint halves of a real sample are, with probability
one, distinct as empirical measures. -/
theorem pos_of_ne {x y : M} (h : x ≠ y) : (0 : α) < dist α x y := by
  rcases Classical.em (dist α x y = (0 : α)) with hz | hnz
  · exact absurd ((eq_zero_iff (α := α) x y).1 hz) h
  · exact (lt_iff _ _).2 ⟨nonneg (α := α) x y, fun he => hnz he.symm⟩

/-- **The factor-one-half bridge.**

The triangle inequality bounds a *two-sample* discrepancy by the **sum** of
two one-sample discrepancies, never by either one alone. Consequently one
cannot conclude that a one-sample term is large merely because the
two-sample statistic is: at least one of the two is at least half of it,
and that is the strongest conclusion available.

Stating it as a theorem makes the stronger claim -- that the two-sample
surrogate *is* the statistical term -- underivable here. -/
theorem not_both_less_than_half (x y p : M) :
    ¬ ((1 + 1 : α) * dist α x p < dist α x y ∧
       (1 + 1 : α) * dist α p y < dist α x y) := by
  intro hcon
  obtain ⟨h1, h2⟩ := hcon
  have hsum : (1 + 1 : α) * dist α x p + (1 + 1 : α) * dist α p y
      < dist α x y + dist α x y := add_lt_add h1 h2
  have htri : dist α x y ≤ dist α x p + dist α p y := triangle x p y
  have hmono : (1 + 1 : α) * dist α x y
      ≤ (1 + 1 : α) * (dist α x p + dist α p y) :=
    mul_le_mul_of_nonneg_left htri zero_le_two
  rw [distrib, two_mul] at hmono
  exact lt_irrefl _ (lt_of_le_of_lt hmono hsum)

end Discrepancy

/-! ## Finite replicate collections

The certgen floor is the median of a finite collection of replicate
discrepancies, and the statements about unbiased estimators concern the
*sum* of such a collection. Lists are the right finite structure here: they
are in Lean's core, so this file needs no library beyond `Order.lean`. -/

namespace Replicates

open LinOrdField

variable {α : Type u} [LinOrdField α]

/-- The sum of a finite list of replicate values. -/
def sum : List α → α
  | [] => 0
  | x :: xs => x + sum xs

@[simp] theorem sum_nil : sum ([] : List α) = 0 := rfl

@[simp] theorem sum_cons (x : α) (xs : List α) :
    sum (x :: xs) = x + sum xs := rfl

/-- A sum of non-negative replicates is non-negative. -/
theorem sum_nonneg {xs : List α} (h : ∀ x ∈ xs, (0 : α) ≤ x) :
    (0 : α) ≤ sum xs := by
  induction xs with
  | nil => exact le_refl 0
  | cons a as ih =>
      exact add_nonneg (h a (List.mem_cons_self a as))
        (ih fun x hx => h x (List.mem_cons_of_mem a hx))

/-- If every replicate is non-negative and at least one is strictly
positive, the sum is strictly positive. -/
theorem sum_pos_of_mem_pos : ∀ {xs : List α} {a : α},
    a ∈ xs → (0 : α) < a → (∀ x ∈ xs, (0 : α) ≤ x) → (0 : α) < sum xs := by
  intro xs
  induction xs with
  | nil => intro a hmem _ _; exact absurd hmem (by simp)
  | cons b bs ih =>
      intro a hmem hpos hnn
      rcases List.mem_cons.1 hmem with hb | hbs
      · have hb0 : (0 : α) < b := hb ▸ hpos
        exact add_pos_of_pos_of_nonneg hb0
          (sum_nonneg fun x hx => hnn x (List.mem_cons_of_mem b hx))
      · have hrest : (0 : α) < sum bs :=
          ih hbs hpos (fun x hx => hnn x (List.mem_cons_of_mem b hx))
        have hb0 : (0 : α) ≤ b := hnn b (List.mem_cons_self b bs)
        rw [sum_cons]
        have h := add_lt_add_of_lt_of_le hrest hb0
        rw [add_zero', add_comm (sum bs) b] at h
        exact h

/-- **Proposition 1(b): an unbiased estimator must take negative values.**

If the replicate values sum to zero (the estimator is unbiased for a
discrepancy that vanishes on the diagonal) and at least one replicate is
strictly positive, then some replicate is strictly negative.

This is why the certgen protocol cannot be built on an unbiased
`U`-statistic such as the energy distance or `MMD`: on the same-law null a
positive fraction of the replicates is negative, so `log(floor)` is
undefined on that fraction and the floor law has nothing to fit (§3.3
reports the measured fractions). -/
theorem exists_neg_of_sum_zero {xs : List α} {a : α}
    (hmem : a ∈ xs) (hpos : (0 : α) < a) (hsum : sum xs = 0) :
    ∃ x ∈ xs, x < (0 : α) := by
  rcases Classical.em (∃ x ∈ xs, x < (0 : α)) with hex | hcon
  · exact hex
  exfalso
  have hnn : ∀ x ∈ xs, (0 : α) ≤ x :=
    fun x hx => le_of_not_lt (fun hlt => hcon ⟨x, hx, hlt⟩)
  have := sum_pos_of_mem_pos hmem hpos hnn
  rw [hsum] at this
  exact lt_irrefl _ this

end Replicates
end Certgen
