/-
# The numeric base

Everything in this development is parametric in an abstract linearly ordered
field `α`. This file is the **entire arithmetic trusted base**: the class
`LinOrdField` below lists every algebraic and order fact the rest of the
development is permitted to use. Nothing else is assumed about numbers.

Why abstract rather than `Real`? Because `Real` and its order-field
instances live in Mathlib, and importing Mathlib would put roughly 1.5
million lines between a reader and the facts actually being relied on. A
trusted base must be small enough to audit and explicit enough to
re-check: seventeen laws on one screen satisfy that, `import Mathlib`
does not.

The Mathlib bridge (`bridge/Bridge.lean`) discharges these laws against
Mathlib's `ℝ`, so every theorem in this development transfers to `ℝ`
unchanged.
-/

namespace Certgen

universe u

/-- A linearly ordered field.

The axioms are the standard ones. They are stated explicitly, rather than
inherited from a hierarchy, so that the reader can see the complete list of
arithmetic facts this development relies on. Division is deliberately
**absent**: no result below needs it, which is itself worth knowing, since
it means none of the mathematics can fail through a hidden division by
zero. -/
class LinOrdField (α : Type u) extends
    Add α, Mul α, Neg α, Sub α, LT α, LE α where
  /-- The additive identity. -/
  zero : α
  /-- The multiplicative identity. -/
  one : α
  -- additive abelian group
  add_assoc : ∀ a b c : α, a + b + c = a + (b + c)
  add_comm : ∀ a b : α, a + b = b + a
  add_zero : ∀ a : α, a + zero = a
  add_neg : ∀ a : α, a + -a = zero
  sub_eq : ∀ a b : α, a - b = a + -b
  -- commutative monoid, distributing over addition
  mul_assoc : ∀ a b c : α, a * b * c = a * (b * c)
  mul_comm : ∀ a b : α, a * b = b * a
  mul_one : ∀ a : α, a * one = a
  distrib : ∀ a b c : α, a * (b + c) = a * b + a * c
  -- linear order
  le_refl : ∀ a : α, a ≤ a
  le_trans : ∀ {a b c : α}, a ≤ b → b ≤ c → a ≤ c
  le_antisymm : ∀ {a b : α}, a ≤ b → b ≤ a → a = b
  le_total : ∀ a b : α, a ≤ b ∨ b ≤ a
  lt_iff : ∀ a b : α, a < b ↔ a ≤ b ∧ a ≠ b
  -- compatibility of the order with the operations
  add_le_add_left : ∀ {a b : α}, a ≤ b → ∀ c : α, c + a ≤ c + b
  mul_nonneg : ∀ {a b : α}, zero ≤ a → zero ≤ b → zero ≤ a * b
  /-- Nontriviality. Without it `0 ≤ 1` is not derivable, and the
  factor-of-two bridge in `Discrepancy.lean` needs it. -/
  zero_ne_one : (zero : α) ≠ one

namespace LinOrdField

variable {α : Type u} [LinOrdField α]

instance : OfNat α 0 := ⟨zero⟩
instance : OfNat α 1 := ⟨one⟩

theorem zero_def : (0 : α) = zero := rfl
theorem one_def : (1 : α) = one := rfl

/-! The axioms are stated with the class fields `zero` and `one`. The
numeral forms below are definitionally the same statements, and are what the
rewriting tactics below actually use. -/

theorem add_zero' (a : α) : a + (0 : α) = a := add_zero a
theorem add_neg' (a : α) : a + -a = (0 : α) := add_neg a
theorem mul_one' (a : α) : a * (1 : α) = a := mul_one a

/-! ### Derived arithmetic

Each lemma below is proved from the axioms above and from nothing else.
-/

theorem zero_add (a : α) : (0 : α) + a = a := by
  rw [add_comm]; exact add_zero' a

theorem neg_add (a : α) : -a + a = (0 : α) := by
  rw [add_comm]; exact add_neg' a

theorem add_left_cancel {a b c : α} (h : a + b = a + c) : b = c := by
  have h2 : -a + (a + b) = -a + (a + c) := by rw [h]
  rw [← add_assoc, ← add_assoc, neg_add, zero_add, zero_add] at h2
  exact h2

theorem add_right_cancel {a b c : α} (h : a + c = b + c) : a = b := by
  apply add_left_cancel (a := c)
  rw [add_comm c a, add_comm c b]; exact h

theorem mul_zero (a : α) : a * (0 : α) = 0 := by
  have h : a * 0 + a * 0 = a * 0 + 0 := by
    rw [← distrib, zero_add, add_zero']
  exact add_left_cancel h

theorem zero_mul (a : α) : (0 : α) * a = 0 := by
  rw [mul_comm]; exact mul_zero a

theorem one_mul (a : α) : (1 : α) * a = a := by
  rw [mul_comm]; exact mul_one' a

theorem distrib_right (a b c : α) : (a + b) * c = a * c + b * c := by
  rw [mul_comm, distrib, mul_comm c a, mul_comm c b]

theorem neg_mul (a b : α) : -a * b = -(a * b) := by
  apply add_left_cancel (a := a * b)
  rw [add_neg', ← distrib_right, add_neg', zero_mul]

theorem neg_zero : -(0 : α) = 0 := by
  apply add_left_cancel (a := (0 : α)); rw [add_neg', add_zero']

theorem sub_self (a : α) : a - a = (0 : α) := by
  rw [sub_eq]; exact add_neg' a

theorem sub_zero (a : α) : a - (0 : α) = a := by
  rw [sub_eq, neg_zero, add_zero']

/-! ### Derived order facts -/

theorem le_of_eq {a b : α} (h : a = b) : a ≤ b := h ▸ le_refl a

theorem lt_of_lt_of_le {a b c : α} (h1 : a < b) (h2 : b ≤ c) : a < c := by
  rw [lt_iff] at h1 ⊢
  refine ⟨le_trans h1.1 h2, ?_⟩
  intro hac
  exact h1.2 (le_antisymm h1.1 (le_trans h2 (le_of_eq hac.symm)))

theorem le_of_lt {a b : α} (h : a < b) : a ≤ b := ((lt_iff a b).1 h).1

theorem lt_of_le_of_lt {a b c : α} (h1 : a ≤ b) (h2 : b < c) : a < c := by
  rw [lt_iff] at h2 ⊢
  refine ⟨le_trans h1 h2.1, ?_⟩
  intro hac
  exact h2.2 (le_antisymm h2.1 (le_trans (le_of_eq hac.symm) h1))

theorem lt_irrefl (a : α) : ¬ a < a := by
  intro h; exact ((lt_iff a a).1 h).2 rfl

theorem lt_trans {a b c : α} (h1 : a < b) (h2 : b < c) : a < c :=
  lt_of_lt_of_le h1 (le_of_lt h2)

theorem not_lt_of_le {a b : α} (h : a ≤ b) : ¬ b < a := by
  intro hlt
  exact ((lt_iff b a).1 hlt).2 (le_antisymm (le_of_lt hlt) h)

theorem lt_of_not_le {a b : α} (h : ¬ a ≤ b) : b < a := by
  rcases le_total a b with hab | hba
  · exact absurd hab h
  · rw [lt_iff]; exact ⟨hba, fun he => h (le_of_eq he.symm)⟩

theorem le_of_not_lt {a b : α} (h : ¬ a < b) : b ≤ a := by
  rcases le_total b a with h1 | h2
  · exact h1
  · rcases Classical.em (a = b) with he | hne
    · exact le_of_eq he.symm
    · exact absurd ((lt_iff a b).2 ⟨h2, hne⟩) h

theorem add_le_add_right {a b : α} (h : a ≤ b) (c : α) : a + c ≤ b + c := by
  rw [add_comm a c, add_comm b c]; exact add_le_add_left h c

theorem add_le_add {a b c d : α} (h1 : a ≤ b) (h2 : c ≤ d) :
    a + c ≤ b + d :=
  le_trans (add_le_add_right h1 c) (add_le_add_left h2 b)

theorem add_lt_add_left {a b : α} (h : a < b) (c : α) : c + a < c + b := by
  rw [lt_iff] at h ⊢
  refine ⟨add_le_add_left h.1 c, ?_⟩
  intro he; exact h.2 (add_left_cancel he)

theorem add_lt_add_right {a b : α} (h : a < b) (c : α) : a + c < b + c := by
  rw [add_comm a c, add_comm b c]; exact add_lt_add_left h c

theorem add_lt_add_of_lt_of_le {a b c d : α} (h1 : a < b) (h2 : c ≤ d) :
    a + c < b + d :=
  lt_of_lt_of_le (add_lt_add_right h1 c) (add_le_add_left h2 b)

theorem add_lt_add {a b c d : α} (h1 : a < b) (h2 : c < d) : a + c < b + d :=
  add_lt_add_of_lt_of_le h1 (le_of_lt h2)

theorem add_pos {a b : α} (ha : (0 : α) < a) (hb : (0 : α) < b) :
    (0 : α) < a + b := by
  have h := add_lt_add_of_lt_of_le ha (le_of_lt hb)
  rwa [add_zero'] at h

theorem add_pos_of_pos_of_nonneg {a b : α} (ha : (0 : α) < a)
    (hb : (0 : α) ≤ b) : (0 : α) < a + b := by
  have h := add_lt_add_of_lt_of_le ha hb
  rwa [add_zero'] at h

theorem add_nonneg {a b : α} (ha : (0 : α) ≤ a) (hb : (0 : α) ≤ b) :
    (0 : α) ≤ a + b := by
  have h := add_le_add ha hb
  rwa [add_zero'] at h

theorem le_add_of_nonneg_right {a b : α} (hb : (0 : α) ≤ b) : a ≤ a + b := by
  have h := add_le_add_left hb a
  rwa [add_zero'] at h

theorem lt_add_of_pos_right {a b : α} (hb : (0 : α) < b) : a < a + b := by
  have h := add_lt_add_left hb a
  rwa [add_zero'] at h

theorem sub_nonneg_of_le {a b : α} (h : a ≤ b) : (0 : α) ≤ b - a := by
  have h2 := add_le_add_right h (-a)
  rwa [add_neg', ← sub_eq] at h2

theorem le_of_sub_nonneg {a b : α} (h : (0 : α) ≤ b - a) : a ≤ b := by
  have h2 := add_le_add_right h a
  rw [zero_add, sub_eq, add_assoc, neg_add, add_zero'] at h2
  exact h2

theorem mul_le_mul_of_nonneg_left {a b c : α} (h : a ≤ b)
    (hc : (0 : α) ≤ c) : c * a ≤ c * b := by
  apply le_of_sub_nonneg
  have hba : (0 : α) ≤ b - a := sub_nonneg_of_le h
  have hneg : c * -a = -(c * a) := by
    rw [mul_comm c (-a), neg_mul, mul_comm a c]
  have hd : c * (b - a) = c * b - c * a := by
    rw [sub_eq, distrib, hneg, sub_eq]
  rw [← hd]
  exact mul_nonneg hc hba

theorem mul_le_mul_of_nonneg_right {a b c : α} (h : a ≤ b)
    (hc : (0 : α) ≤ c) : a * c ≤ b * c := by
  rw [mul_comm a c, mul_comm b c]; exact mul_le_mul_of_nonneg_left h hc

theorem neg_neg (a : α) : - -a = a := by
  apply add_left_cancel (a := -a)
  rw [add_neg', neg_add]

theorem zero_le_one : (0 : α) ≤ 1 := by
  rcases le_total (0 : α) 1 with h | h
  · exact h
  · exfalso
    have hn : (0 : α) ≤ -1 := by
      have := sub_nonneg_of_le h
      rwa [sub_eq, zero_add] at this
    have hsq : (0 : α) ≤ -1 * -1 := mul_nonneg hn hn
    have he : (-1 : α) * -1 = 1 := by rw [neg_mul, one_mul, neg_neg]
    rw [he] at hsq
    exact zero_ne_one (le_antisymm hsq h)

theorem zero_le_two : (0 : α) ≤ 1 + 1 := add_nonneg zero_le_one zero_le_one

theorem two_mul (a : α) : (1 + 1 : α) * a = a + a := by
  rw [distrib_right, one_mul]

end LinOrdField
end Certgen
