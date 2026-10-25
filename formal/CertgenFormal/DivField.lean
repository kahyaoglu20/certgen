import CertgenFormal.Order

/-
# The division layer

The five core results of this development (`Discrepancy.lean`, `Sizing.lean`,
`Vacuity.lean`, `Pullback.lean`) are deliberately division-free: none of their
statements can fail through a hidden division by zero, and that fact is
visible in the absence of `inv` from `LinOrdField`.

The statistical layer formalised for the certgen project as a whole does,
however, need ratios: a Monte-Carlo p-value is a count over a number of
replicates, the skill score is a ratio of floor-corrected differences, and
the TOST feasibility margin is a half-width. This file extends the trusted
base by exactly the two axioms that make field operations invertible on
nonzero elements:

* `inv` -- a multiplicative inverse operation;
* `mul_inv_cancel` -- `a * inv a = 1` whenever `a ≠ 0`.

Everything else about division -- the order-cancellation lemmas, the absolute
value, the natural-number embedding and its order reflection, and the bounded
floor lemma used by the Monte-Carlo p-value argument -- is *derived* from those
two axioms and `LinOrdField`. The axiom count of the core is untouched; the
statistical layer is honest about what it adds.

One consequence is worth recording. `LinOrdField` alone does not imply that
the ring is an integral domain: the lexicographic product `ℤ × ℤ` with
componentwise operations satisfies every axiom of `Order.lean` and has zero
divisors (the element `(0, 1)` is positive with square zero). With
`mul_inv_cancel` the cancellation laws hold and `mul_pos` below becomes a
theorem rather than a hidden assumption. The division layer therefore has
strictly more theorems, not merely more notation.
-/

namespace Certgen

open LinOrdField

universe u

/-- A linearly ordered field with multiplicative inverses.

Two axioms beyond `LinOrdField`: an inverse operation, and its cancellation
law on nonzero elements. Nothing is claimed about `inv 0`: division by zero is
uncontrolled, and every use of division below carries an explicit nonzero
hypothesis. -/
class LinOrdFieldDiv (α : Type u) extends LinOrdField α where
  /-- Multiplicative inversion. -/
  inv : α → α
  /-- Every nonzero element is inverted by `inv`. -/
  mul_inv_cancel : ∀ {a : α}, a ≠ (0 : α) → a * inv a = (1 : α)

namespace LinOrdFieldDiv

variable {α : Type u} [LinOrdFieldDiv α]

/-- Division, as multiplication by the inverse. -/
def div (a b : α) : α := a * inv b

instance : HDiv α α α := ⟨div⟩

/-! ### Auxiliary ordered-field lemmas -/

theorem one_ne_zero : (1 : α) ≠ (0 : α) := fun he => zero_ne_one he.symm

theorem ne_zero_of_pos {a : α} (h : (0 : α) < a) : a ≠ (0 : α) :=
  fun he => ((lt_iff _ _).1 h).2 he.symm

theorem lt_ne {a b : α} (h : a < b) : a ≠ b := ((lt_iff a b).1 h).2

theorem add_sub_cancel (a b : α) : a + (b - a) = b := by
  rw [sub_eq, add_comm b (-a), ← add_assoc, add_neg', zero_add]

theorem eq_of_sub_eq_zero {a b : α} (h : b - a = (0 : α)) : a = b := by
  rw [← add_sub_cancel a b, h, add_zero']

theorem sub_pos_of_lt {a b : α} (h : a < b) : (0 : α) < b - a := by
  rw [lt_iff]
  exact ⟨sub_nonneg_of_le (le_of_lt h),
    fun he => ((lt_iff a b).1 h).2 (eq_of_sub_eq_zero he.symm)⟩

theorem lt_of_sub_pos {a b : α} (h : (0 : α) < b - a) : a < b := by
  rw [lt_iff]
  refine ⟨le_of_sub_nonneg (le_of_lt h), ?_⟩
  intro he
  exact ((lt_iff (0 : α) (b - a)).1 h).2 (by rw [he, sub_self])

theorem neg_le_neg {a b : α} (h : a ≤ b) : -b ≤ -a := by
  have h1 : (0 : α) ≤ b - a := sub_nonneg_of_le h
  rw [sub_eq] at h1
  have h2 : (0 : α) ≤ -a - (-b) := by
    rw [sub_eq, neg_neg, add_comm (-a) b]; exact h1
  exact le_of_sub_nonneg h2

theorem neg_nonneg_of_nonpos {a : α} (h : a ≤ (0 : α)) : (0 : α) ≤ -a := by
  have := sub_nonneg_of_le h
  rwa [sub_eq, zero_add] at this

theorem neg_unique {a u : α} (h : a + u = (0 : α)) : u = -a := by
  apply add_left_cancel (a := a)
  rw [add_neg' a]
  exact h

theorem neg_add_dist (x y : α) : -(x + y) = -x + -y := by
  have h0 : x + y + (-x + -y) = (0 : α) := by
    rw [add_assoc x y (-x + -y), add_comm (-x) (-y), ← add_assoc y (-y) (-x),
        add_neg' y, zero_add (-x), add_neg' x]
  exact (neg_unique h0).symm

/-! ### Inverses and cancellation -/

theorem inv_mul_cancel {a : α} (ha : a ≠ (0 : α)) : inv a * a = (1 : α) := by
  rw [mul_comm]; exact mul_inv_cancel ha

theorem inv_mul_cancel_left {a c : α} (ha : a ≠ (0 : α)) : inv a * (a * c) = c := by
  rw [← mul_assoc, inv_mul_cancel ha, one_mul]

theorem mul_right_cancel' {c x y : α} (hc : c ≠ (0 : α)) (h : c * x = c * y) : x = y := by
  have h1 : inv c * (c * x) = inv c * (c * y) := by rw [h]
  rw [inv_mul_cancel_left hc, inv_mul_cancel_left hc] at h1
  exact h1

theorem mul_left_cancel' {c x y : α} (hc : c ≠ (0 : α)) (h : x * c = y * c) : x = y := by
  apply mul_right_cancel' (c := c) hc
  rw [mul_comm c x, mul_comm c y]; exact h

theorem inv_unique {c X : α} (h : c * X = (1 : α)) : X = inv c := by
  have hc : c ≠ (0 : α) := by
    intro he
    rw [he, zero_mul] at h
    exact zero_ne_one h
  calc X = (1 : α) * X := (one_mul X).symm
    _ = (inv c * c) * X := by rw [inv_mul_cancel hc]
    _ = inv c * (c * X) := by rw [mul_assoc]
    _ = inv c * (1 : α) := by rw [h]
    _ = inv c := mul_one _

theorem mul_ne_zero {a b : α} (ha : a ≠ (0 : α)) (hb : b ≠ (0 : α)) :
    a * b ≠ (0 : α) := by
  intro he
  have h1 : inv a * (a * b) = inv a * (0 : α) := by rw [he, mul_zero]
  rw [inv_mul_cancel_left ha, mul_zero] at h1
  exact hb h1

theorem inv_mul' {a b : α} (ha : a ≠ (0 : α)) (hb : b ≠ (0 : α)) :
    inv (a * b) = inv b * inv a := by
  symm
  apply inv_unique
  calc a * b * (inv b * inv a) = a * (b * (inv b * inv a)) := by
        rw [mul_assoc a b (inv b * inv a)]
    _ = a * ((b * inv b) * inv a) := by rw [← mul_assoc b (inv b) (inv a)]
    _ = a * ((1 : α) * inv a) := by rw [mul_inv_cancel (a := b) hb]
    _ = a * inv a := by rw [one_mul]
    _ = (1 : α) := mul_inv_cancel (a := a) ha

theorem inv_one : inv (1 : α) = (1 : α) := by
  have h : (1 : α) * inv (1 : α) = (1 : α) :=
    mul_inv_cancel (a := (1 : α)) one_ne_zero
  rwa [one_mul] at h

/-! ### Positivity and monotone multiplication -/

/-- A product of strictly positive elements is strictly positive.

In the division-free core only `0 ≤ a * b` is available; strict positivity
needs cancellation, which needs inverses. This is the lemma that fails in a
lexicographically ordered ring with zero divisors. -/
theorem mul_pos {a b : α} (ha : (0 : α) < a) (hb : (0 : α) < b) : (0 : α) < a * b := by
  have hnn : (0 : α) ≤ a * b := mul_nonneg (le_of_lt ha) (le_of_lt hb)
  refine Classical.byContradiction (fun hcon => ?_)
  have hz : a * b = (0 : α) := le_antisymm (le_of_not_lt hcon) hnn
  have hb0 : b = (0 : α) := by
    have h1 : inv a * (a * b) = inv a * (0 : α) := by rw [hz, mul_zero]
    rw [inv_mul_cancel_left (ne_zero_of_pos ha), mul_zero] at h1
    exact h1
  exact lt_ne hb hb0.symm

/-- The inverse of a positive element is positive. -/
theorem inv_pos {a : α} (ha : (0 : α) < a) : (0 : α) < inv a := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hle : inv a ≤ (0 : α) := le_of_not_lt hcon
  have h1 : a * inv a ≤ a * (0 : α) := mul_le_mul_of_nonneg_left hle (le_of_lt ha)
  rw [mul_inv_cancel (a := a) (ne_zero_of_pos ha), mul_zero] at h1
  exact one_ne_zero (le_antisymm h1 zero_le_one)

theorem mul_lt_mul_right_of_lt {b x y : α} (hb : (0 : α) < b) (h : y < x) :
    y * b < x * b := by
  have hp : (0 : α) < (x - y) * b := mul_pos (sub_pos_of_lt h) hb
  have he : (x - y) * b = x * b - y * b := by
    rw [sub_eq, distrib_right, neg_mul, sub_eq]
  apply lt_of_sub_pos
  rw [← he]; exact hp

theorem mul_le_mul_right_cancel {b x y : α} (hb : (0 : α) < b) (h : x * b ≤ y * b) :
    x ≤ y := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hxy : y < x := lt_of_not_le (fun hle => hcon hle)
  exact not_lt_of_le h (mul_lt_mul_right_of_lt hb hxy)

/-! ### Division facts -/

theorem div_mul_cancel {b : α} (hb : b ≠ (0 : α)) (a : α) : a / b * b = a := by
  show a * inv b * b = a
  rw [mul_assoc a (inv b) b, mul_comm (inv b) b, mul_inv_cancel hb, mul_one']

theorem mul_div_cancel {b : α} (hb : b ≠ (0 : α)) (a : α) : a * b / b = a := by
  show a * b * inv b = a
  rw [mul_assoc a b (inv b), mul_inv_cancel hb, mul_one']

theorem div_self {a : α} (ha : a ≠ (0 : α)) : a / a = (1 : α) :=
  mul_inv_cancel ha

theorem div_one (a : α) : a / (1 : α) = a := by
  show a * inv (1 : α) = a
  rw [inv_one, mul_one']

theorem one_div_eq_inv (a : α) : (1 : α) / a = inv a := by
  show (1 : α) * inv a = inv a
  rw [one_mul]

theorem div_pos {a b : α} (ha : (0 : α) < a) (hb : (0 : α) < b) : (0 : α) < a / b :=
  mul_pos ha (inv_pos hb)

theorem div_nonneg {a b : α} (ha : (0 : α) ≤ a) (hb : (0 : α) < b) :
    (0 : α) ≤ a / b :=
  mul_nonneg ha (le_of_lt (inv_pos hb))

/-- Multiplying numerator and denominator by the same nonzero factor leaves a
quotient unchanged. -/
theorem div_mul_mul {a b t : α} (ht : t ≠ (0 : α)) (hb : b ≠ (0 : α)) :
    t * a / (t * b) = a / b := by
  have e1 : inv (t * b) = inv b * inv t := inv_mul' ht hb
  have e2 : t * (inv t * (a * inv b)) = a * inv b := by
    calc t * (inv t * (a * inv b)) = (t * inv t) * (a * inv b) := by
          rw [mul_assoc t (inv t) (a * inv b)]
      _ = (1 : α) * (a * inv b) := by rw [mul_inv_cancel (a := t) ht]
      _ = a * inv b := one_mul _
  calc t * a / (t * b) = (t * a) * (inv b * inv t) := by
        show (t * a) * inv (t * b) = (t * a) * (inv b * inv t)
        rw [e1]
    _ = t * (a * (inv b * inv t)) := by rw [mul_assoc t a (inv b * inv t)]
    _ = t * ((a * inv b) * inv t) := by rw [← mul_assoc a (inv b) (inv t)]
    _ = t * (inv t * (a * inv b)) := by rw [mul_comm (a * inv b) (inv t)]
    _ = a * inv b := e2
    _ = a / b := rfl

theorem div_add_div_same {d a b : α} : a / d + b / d = (a + b) / d := by
  show a * inv d + b * inv d = (a + b) * inv d
  rw [← distrib_right]

theorem sub_div_same {d a b : α} : a / d - b / d = (a - b) / d := by
  have h : (a - b) / d + b / d = a / d := by
    rw [div_add_div_same, sub_eq, add_assoc, neg_add, add_zero']
  rw [← h, sub_eq, add_assoc, add_neg', add_zero']

/-- **The order-cancellation law for division.**

`a / b ≤ c` if and only if `a ≤ c * b`, for a positive denominator. This is
the form in which every inequality between ratios is manipulated below; it is
what makes the Monte-Carlo p-value argument a statement about integer counts. -/
theorem div_le_iff {b : α} (hb : (0 : α) < b) {a c : α} : a / b ≤ c ↔ a ≤ c * b := by
  constructor
  · intro h
    have h2 := mul_le_mul_of_nonneg_right h (le_of_lt hb)
    rwa [div_mul_cancel (ne_zero_of_pos hb)] at h2
  · intro h
    have h2 : a / b * b ≤ c * b := by
      rw [div_mul_cancel (ne_zero_of_pos hb)]; exact h
    exact mul_le_mul_right_cancel hb h2

theorem le_div_iff {b : α} (hb : (0 : α) < b) {a c : α} : a ≤ c / b ↔ a * b ≤ c := by
  constructor
  · intro h
    have h2 := mul_le_mul_of_nonneg_right h (le_of_lt hb)
    rwa [div_mul_cancel (ne_zero_of_pos hb)] at h2
  · intro h
    have h2 : a * b ≤ c / b * b := by
      rw [div_mul_cancel (ne_zero_of_pos hb)]; exact h
    exact mul_le_mul_right_cancel hb h2

theorem one_div_le_one_div_of_le {a b : α} (ha : (0 : α) < a) (h : a ≤ b) :
    (1 : α) / b ≤ (1 : α) / a := by
  have hb : (0 : α) < b := lt_of_lt_of_le ha h
  apply (le_div_iff ha).mpr
  have h2 : (1 : α) / b * a = a / b := by
    show (1 : α) * inv b * a = a * inv b
    rw [mul_comm ((1 : α) * inv b) a, one_mul]
  rw [h2, (div_le_iff hb), one_mul]
  exact h

theorem one_div_inj {a b : α} (ha : (0 : α) < a) (hb : (0 : α) < b)
    (h : (1 : α) / a = (1 : α) / b) : a = b := by
  have hib : inv b ≠ (0 : α) := by
    intro he
    apply (one_ne_zero (α := α))
    have h1 : b * inv b = (1 : α) := mul_inv_cancel (a := b) (ne_zero_of_pos hb)
    rw [← h1, he, mul_zero]
  apply mul_right_cancel' (c := inv b) hib
  calc inv b * a = a * inv b := mul_comm (inv b) a
    _ = a * ((1 : α) / b) := by rw [one_div_eq_inv]
    _ = a * ((1 : α) / a) := by rw [← h]
    _ = (1 : α) := by
        rw [one_div_eq_inv]; exact mul_inv_cancel (a := a) (ne_zero_of_pos ha)
    _ = inv b * b := by
        rw [mul_comm]; exact (mul_inv_cancel (a := b) (ne_zero_of_pos hb)).symm

theorem one_div_lt_one_div_of_lt {a b : α} (ha : (0 : α) < a) (h : a < b) :
    (1 : α) / b < (1 : α) / a := by
  have hb : (0 : α) < b := lt_of_lt_of_le ha (le_of_lt h)
  rw [lt_iff]
  refine ⟨one_div_le_one_div_of_le ha (le_of_lt h), ?_⟩
  intro he
  exact lt_ne h (one_div_inj hb ha he).symm

/-! ### The natural-number embedding

Counts arrive as natural numbers (`List.length`, replicate counts) while the
level `α` carries the order. The embedding below is the standard
`n ↦ n • 1` recursion, proved monotone and order-reflecting from the axioms;
no archimedean property is claimed or used anywhere in this development. -/

def toα : Nat → α
  | 0 => (0 : α)
  | n + 1 => toα n + (1 : α)

theorem toα_zero : toα 0 = (0 : α) := rfl
theorem toα_succ (n : Nat) : toα (n + 1) = toα n + (1 : α) := rfl
theorem toα_one : toα 1 = (1 : α) := by
  have h : toα (0 + 1) = toα 0 + (1 : α) := toα_succ 0
  rw [Nat.zero_add, toα_zero, zero_add] at h
  exact h

theorem toα_nonneg (n : Nat) : (0 : α) ≤ toα n := by
  induction n with
  | zero => exact le_refl _
  | succ m ih => exact add_nonneg ih zero_le_one

theorem toα_add (m n : Nat) : (toα (m + n) : α) = toα m + toα n := by
  induction m with
  | zero => rw [Nat.zero_add, toα_zero, zero_add]
  | succ m ih =>
      rw [Nat.succ_add, toα_succ, toα_succ, ih, add_assoc, add_assoc,
          add_comm (1 : α) (toα n)]

theorem toα_mono {m n : Nat} (h : m ≤ n) : (toα m : α) ≤ toα n := by
  have hd := Nat.le.dest h
  match hd with
  | ⟨k, hk⟩ =>
      rw [← hk, toα_add]
      exact le_add_of_nonneg_right (toα_nonneg k)

theorem zero_lt_one : (0 : α) < (1 : α) :=
  (lt_iff (0 : α) (1 : α)).mpr ⟨zero_le_one, fun he => one_ne_zero he.symm⟩

theorem toα_strict {m n : Nat} (h : m < n) : (toα m : α) < toα n := by
  have h1 : toα m + (1 : α) ≤ toα n := by
    have h2 : (toα (m + 1) : α) ≤ toα n := toα_mono (Nat.succ_le_of_lt h)
    rw [toα_succ] at h2
    exact h2
  exact lt_of_lt_of_le (lt_add_of_pos_right zero_lt_one) h1

theorem toα_le_iff {m n : Nat} : (toα m : α) ≤ toα n ↔ m ≤ n := by
  constructor
  · intro h
    cases Nat.lt_trichotomy m n with
    | inl hlt => exact Nat.le_of_lt hlt
    | inr he =>
        cases he with
        | inl heq => exact Nat.le_of_eq heq
        | inr hgt => exact absurd (toα_strict hgt) (not_lt_of_le h)
  · exact toα_mono

theorem toα_lt_iff {m n : Nat} : (toα m : α) < toα n ↔ m < n := by
  constructor
  · intro h
    cases Nat.lt_trichotomy m n with
    | inl hlt => exact hlt
    | inr he =>
        cases he with
        | inl heq => exact absurd h (by subst heq; exact lt_irrefl (toα m))
        | inr hgt => exact absurd (toα_strict hgt) (not_lt_of_le (le_of_lt h))
  · exact toα_strict

/-! ### The absolute value -/

open Classical in
/-- The absolute value, defined by cases on the sign.

The case split needs `le_total`, not a decidable comparison, so the definition
is classical and noncomputable; nothing in this development computes. -/
noncomputable def abv (a : α) : α := if (0 : α) ≤ a then a else -a

theorem abv_of_nonneg {a : α} (h : (0 : α) ≤ a) : abv a = a := if_pos h

theorem abv_of_nonpos {a : α} (h : a ≤ (0 : α)) : abv a = -a := by
  unfold abv
  by_cases h0 : (0 : α) ≤ a
  · rw [if_pos h0]
    have ha0 : a = (0 : α) := le_antisymm h h0
    rw [ha0, neg_zero]
  · rw [if_neg h0]

theorem abv_nonneg (a : α) : (0 : α) ≤ abv a := by
  by_cases h : (0 : α) ≤ a
  · rw [abv_of_nonneg h]; exact h
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h)
    rw [abv_of_nonpos ha]
    exact neg_nonneg_of_nonpos ha

theorem abv_zero : abv (0 : α) = (0 : α) := if_pos (le_refl _)

theorem abv_one : abv (1 : α) = (1 : α) := if_pos zero_le_one

theorem abv_neg (a : α) : abv (-a) = abv a := by
  by_cases h : (0 : α) ≤ a
  · have hna : -a ≤ (0 : α) := by
      rw [← neg_zero]
      exact neg_le_neg h
    rw [abv_of_nonpos hna, abv_of_nonneg h, neg_neg]
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h)
    have h' : (0 : α) ≤ -a := by
      rw [← neg_zero]
      exact neg_le_neg ha
    rw [abv_of_nonpos ha, abv_of_nonneg h']

theorem neg_eq_zero {a : α} : -a = (0 : α) ↔ a = (0 : α) := by
  constructor
  · intro h
    rw [← neg_neg a, h, neg_zero]
  · intro h; rw [h, neg_zero]

theorem abv_eq_zero {a : α} (h : abv a = (0 : α)) : a = (0 : α) := by
  by_cases h0 : (0 : α) ≤ a
  · rw [abv_of_nonneg h0] at h; exact h
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h0)
    rw [abv_of_nonpos ha, neg_eq_zero] at h
    exact h

theorem abv_pos {a : α} : (0 : α) < abv a ↔ a ≠ (0 : α) := by
  constructor
  · intro h
    intro he
    have h2 : abv a = (0 : α) := by rw [he, abv_zero]
    exact ((lt_iff _ _).1 h).2 h2.symm
  · intro h
    exact (lt_iff _ _).mpr ⟨abv_nonneg a, fun he => h (abv_eq_zero he.symm)⟩

theorem le_abv_self (a : α) : a ≤ abv a := by
  by_cases h : (0 : α) ≤ a
  · rw [abv_of_nonneg h]; exact le_refl a
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h)
    rw [abv_of_nonpos ha]
    exact le_trans ha (neg_nonneg_of_nonpos ha)

theorem neg_abv_le (a : α) : -(abv a) ≤ a := by
  by_cases h : (0 : α) ≤ a
  · rw [abv_of_nonneg h]
    have h2 : (0 : α) ≤ a + a := by
      have := add_le_add h h
      rwa [zero_add] at this
    exact le_of_sub_nonneg (by rw [sub_eq, neg_neg]; exact h2)
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h)
    rw [abv_of_nonpos ha, neg_neg]
    exact le_refl a

theorem abv_triangle (a b : α) : abv (a + b) ≤ abv a + abv b := by
  have h1 : a + b ≤ abv a + abv b := add_le_add (le_abv_self a) (le_abv_self b)
  have h2 : -(abv a + abv b) ≤ a + b := by
    have := add_le_add (neg_abv_le a) (neg_abv_le b)
    rwa [← neg_add_dist] at this
  by_cases h : (0 : α) ≤ a + b
  · rw [abv_of_nonneg h]; exact h1
  · rw [abv_of_nonpos (le_of_lt (lt_of_not_le h)),
        ← neg_neg (abv a + abv b)]
    exact neg_le_neg h2

theorem neg_mul_right (a b : α) : a * -b = -(a * b) := by
  rw [mul_comm a (-b), neg_mul, mul_comm b a]

theorem abv_mul (a b : α) : abv (a * b) = abv a * abv b := by
  by_cases ha : (0 : α) ≤ a
  · by_cases hb : (0 : α) ≤ b
    · rw [abv_of_nonneg (mul_nonneg ha hb), abv_of_nonneg ha, abv_of_nonneg hb]
    · have hb' : b ≤ (0 : α) := le_of_lt (lt_of_not_le hb)
      have h : (0 : α) ≤ a * -b := mul_nonneg ha (neg_nonneg_of_nonpos hb')
      rw [neg_mul_right] at h
      have hab : a * b ≤ (0 : α) := by
        have h2 := neg_le_neg h
        rw [neg_zero, neg_neg] at h2
        exact h2
      rw [abv_of_nonpos hab, abv_of_nonneg ha, abv_of_nonpos hb',
          ← neg_mul_right]
  · have ha' : a ≤ (0 : α) := le_of_lt (lt_of_not_le ha)
    by_cases hb : (0 : α) ≤ b
    · have h : (0 : α) ≤ -a * b := mul_nonneg (neg_nonneg_of_nonpos ha') hb
      rw [neg_mul] at h
      have hab : a * b ≤ (0 : α) := by
        have h2 := neg_le_neg h
        rw [neg_zero, neg_neg] at h2
        exact h2
      rw [abv_of_nonpos hab, abv_of_nonpos ha', abv_of_nonneg hb, ← neg_mul]
    · have hb' : b ≤ (0 : α) := le_of_lt (lt_of_not_le hb)
      have hnn : (0 : α) ≤ (-a) * (-b) :=
        mul_nonneg (neg_nonneg_of_nonpos ha') (neg_nonneg_of_nonpos hb')
      have he : (-a) * (-b) = a * b := by rw [neg_mul, neg_mul_right, neg_neg]
      rw [abv_of_nonneg (he ▸ hnn), abv_of_nonpos ha', abv_of_nonpos hb', he]

theorem abv_inv {a : α} (ha : a ≠ (0 : α)) : abv (inv a) = 1 / abv a := by
  have hab : abv a ≠ (0 : α) := fun he => ha (abv_eq_zero he)
  have h : abv a * abv (inv a) = (1 : α) := by
    rw [← abv_mul, mul_inv_cancel ha, abv_one]
  calc abv (inv a) = (1 : α) * abv (inv a) := (one_mul _).symm
    _ = (inv (abv a) * abv a) * abv (inv a) := by rw [inv_mul_cancel hab]
    _ = inv (abv a) * (abv a * abv (inv a)) := by rw [mul_assoc]
    _ = inv (abv a) * (1 : α) := by rw [h]
    _ = inv (abv a) := mul_one _
    _ = (1 : α) / abv a := (one_div_eq_inv _).symm

theorem abv_div {a b : α} (hb : b ≠ (0 : α)) : abv (a / b) = abv a / abv b := by
  show abv (a * inv b) = abv a * inv (abv b)
  rw [abv_mul, abv_inv hb, ← one_div_eq_inv]

/-! ### The bounded floor lemma

No archimedean property is assumed anywhere. But every element bounded by an
embedded natural number does have an integral floor among the naturals up to
that bound, and the proof is a plain induction on the bound. This is exactly
what the Monte-Carlo p-value validity theorem needs: the p-value's numerator
is an integer count bounded by the number of replicates. -/

theorem sub_add_cancel (a : α) : a - (1 : α) + (1 : α) = a := by
  rw [sub_eq, add_assoc, neg_add, add_zero']

theorem lt_of_add_lt_add_right {a b c : α} (h : a + c < b + c) : a < b := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hle : b ≤ a := le_of_not_lt hcon
  exact not_lt_of_le (add_le_add_right hle c) h

theorem sub_one_lt_iff {a b : α} : a - (1 : α) < b ↔ a < b + (1 : α) := by
  constructor
  · intro h
    calc a = a - (1 : α) + (1 : α) := (sub_add_cancel a).symm
      _ < b + (1 : α) := add_lt_add_right h _
  · intro h
    have h2 : a - (1 : α) + (1 : α) < b + (1 : α) := by
      rw [sub_add_cancel a]; exact h
    exact lt_of_add_lt_add_right h2

/-- **Bounded floor.** If `0 ≤ r < toα n` then `r` has an integral floor `k`
with `toα k ≤ r < toα (k+1)`.

The proof is induction on the natural bound; no archimedean axiom is used,
and none is needed, because `r` is bounded by an embedded natural. -/
theorem floor_bounded : ∀ (r : α) (n : Nat), (0 : α) ≤ r → r < toα n →
    ∃ k : Nat, toα k ≤ r ∧ r < toα (k + 1) := by
  intro r n
  induction n generalizing r with
  | zero =>
      intro h1 h2
      rw [toα_zero] at h2
      exact absurd h2 (not_lt_of_le h1)
  | succ m ih =>
      intro h1 h2
      cases Classical.em (r < (1 : α)) with
      | inl hlt =>
          refine ⟨0, ?_, ?_⟩
          · rw [toα_zero]; exact h1
          · rw [toα_succ, toα_zero, zero_add]; exact hlt
      | inr hge =>
          have hge' : (1 : α) ≤ r := le_of_not_lt hge
          have h1' : (0 : α) ≤ r - (1 : α) := sub_nonneg_of_le hge'
          have h2' : r - (1 : α) < toα m := (sub_one_lt_iff (a := r) (b := toα m)).mpr (by
            rw [← toα_succ]; exact h2)
          have ih2 := ih (r - (1 : α)) h1' h2'
          match ih2 with
          | ⟨k, hk1, hk2⟩ =>
              refine ⟨k + 1, ?_, ?_⟩
              · rw [toα_succ, ← sub_add_cancel r]
                exact add_le_add_right hk1 (1 : α)
              · rw [toα_succ, ← sub_add_cancel r]
                exact add_lt_add_right hk2 (1 : α)

end LinOrdFieldDiv
end Certgen
