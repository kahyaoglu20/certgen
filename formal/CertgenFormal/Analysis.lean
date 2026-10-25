import CertgenFormal.DivField
import CertgenFormal.Skill

/-
# The analysis layer: limits and powers

Three theorems of the paper need more than the trusted base provides --
limits, powers, or measure theory:

* the asymptotic n-invariance of the skill score (§5.1): as the floor
  vanishes, V → D(P,Q)/D(P,Q_B) for fixed Q ≠ P, and V → 0 for Q = P;
* the population pushforward-coupling lemma (§7.1), in full generality --
  `Pullback.lean` proves the finite case in which the quantity is
  computed;
* the exact-W₂/Kantorovich equivalence (§3.5): with equal sample sizes and
  uniform weights the Kantorovich LP has an integral optimal solution, so
  it reduces to a linear assignment problem.

This file supplies the layer for the first of the three: a notion of
convergence for sequences in `α`, and a **power operation** with three laws.
`Population.lean` and `Kantorovich.lean` discharge the other two. The layer
is minimal in the sense the rest of this development is minimal: nothing is
assumed beyond what the theorems need.

* **Limits.** `Tendsto u l` says: for every positive ε there is an index
  beyond which `u n` differs from `l` by less than ε. This is a *definition*,
  not an axiom, and the algebra of limits below (constants, sums, negatives,
  differences, quotients) is *proved* from the ordered-field axioms already
  in the trusted base. No completeness, no topology, no filters.

* **Powers.** The floor of the paper is a fitted power law, `C · n^(−s)` with
  `s > 0` (s = 1/2 in the CLT regime, s = 1/d_eff in the rate regime). The
  class `Analysis` below extends `LinOrdFieldDiv` by a power operation
  `rpow : α → α → α` and exactly three laws: positivity, monotonicity in the
  base, and unboundedness of `n ↦ (n : α)^s` for positive exponents. Those
  three laws are what the vanishing of the power-law floor needs, and they
  are all discharged by `Real.rpow` in the Mathlib bridge
  (`bridge/`), so the layer closes over the reals.

The archimedean property is deliberately *not* assumed: it is not needed.
The unboundedness law `rpow_unbounded` plays that role for the power law,
and nothing else in the development escapes the finite.
-/

namespace Certgen

open LinOrdField LinOrdFieldDiv

universe u

/-! ## The power interface -/

/-- A linearly ordered division field with real powers, at exactly the
strength the floor-law asymptotics need.

Three laws beyond `LinOrdFieldDiv`: powers of positives (with positive
exponent) are positive; powers are monotone in the base; and the powers of
the embedded naturals are unbounded, for every positive exponent. `Real`
with `Real.rpow` is the intended model, discharged in the bridge. -/
class Analysis (α : Type u) extends LinOrdFieldDiv α where
  /-- The power operation `x ↦ x^s`. -/
  rpow : α → α → α
  /-- A positive base to a positive power is positive. -/
  rpow_pos : ∀ {x s : α}, (0 : α) < x → (0 : α) < s → (0 : α) < rpow x s
  /-- Powers are monotone in the base, for a positive exponent. -/
  rpow_mono : ∀ {x y s : α}, (0 : α) ≤ x → x ≤ y → (0 : α) < s → rpow x s ≤ rpow y s
  /-- The powers of the embedded naturals are unbounded, for a positive
  exponent. This is the archimedean-style law the vanishing of the floor
  needs; no other archimedean property is assumed anywhere. -/
  rpow_unbounded : ∀ {s : α}, (0 : α) < s → ∀ b : α, ∃ N : Nat, b < rpow (toα N) s

namespace Analysis

variable {α : Type u} [Analysis α]

/-! ## Convergence of sequences

A definition, not an axiom: the classical ε–N statement, valued in the
absolute value of `DivField.lean`. Everything below is proved from the
ordered-field axioms. -/

/-- `Tendsto u l`: for every positive ε there is an N such that every term
beyond N is within ε of `l`. -/
def Tendsto (u : Nat → α) (l : α) : Prop :=
  ∀ ε : α, (0 : α) < ε → ∃ N : Nat, ∀ n : Nat, N ≤ n → abv (u n - l) < ε

/-- Convergence respects pointwise equality of sequences. -/
theorem tendsto_congr {u v : Nat → α} {l : α} (huv : ∀ n, u n = v n)
    (hu : Tendsto u l) : Tendsto v l := by
  intro ε hε
  obtain ⟨N, hN⟩ := hu ε hε
  refine ⟨N, fun n hn => ?_⟩
  rw [← huv n]
  exact hN n hn

/-- Convergence respects equality of the limit. -/
theorem tendsto_limit_eq {u : Nat → α} {l m : α} (hlm : l = m)
    (hu : Tendsto u l) : Tendsto u m := by
  subst hlm; exact hu

/-- **Constants converge** to themselves. -/
theorem tendsto_const (c : α) : Tendsto (fun _ => c) c := by
  intro ε hε
  refine ⟨0, fun _ _ => ?_⟩
  have hsub : c - c = (0 : α) := sub_self c
  rw [hsub, abv_zero]
  exact hε

/-! ### Halves, subtraction identities, and small ordered-field facts

All of this is field algebra; no limits are involved. -/

theorem two_pos : (0 : α) < 1 + 1 := by
  have h : (0 : α) + (1 : α) < (1 : α) + (1 : α) := add_lt_add_right zero_lt_one (1 : α)
  rw [zero_add] at h
  exact lt_trans zero_lt_one h

theorem two_ne_zero : (1 : α) + 1 ≠ (0 : α) := by
  intro he
  have hlt : (1 : α) < (1 : α) + 1 := lt_add_of_pos_right zero_lt_one
  rw [he] at hlt
  exact lt_irrefl _ (lt_of_lt_of_le hlt zero_le_one)

/-- Twice half of ε is ε; the split used by every limit proof below. -/
theorem half_half (ε : α) : ε / (1 + 1) + ε / (1 + 1) = ε := by
  rw [div_add_div_same, ← two_mul ε, mul_comm ((1 : α) + 1) ε,
      mul_div_cancel two_ne_zero ε]

theorem half_pos {ε : α} (h : (0 : α) < ε) : (0 : α) < ε / (1 + 1) :=
  div_pos h two_pos

/-- `a - b + b = a`. -/
theorem subadd (a b : α) : a - b + b = a := by
  rw [sub_eq, add_assoc, neg_add, add_zero']

/-- If `x ≤ y` then `c - y ≤ c - x`. -/
theorem add_le_add_right_cancel' {a b c : α} (h : a + c ≤ b + c) : a ≤ b := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hlt : b < a := lt_of_not_le (fun hle => hcon hle)
  exact not_lt_of_le h (add_lt_add_right hlt c)

theorem sub_le_sub_right' {c x y : α} (h : x ≤ y) : c - y ≤ c - x := by
  have h1 : c - y + x ≤ c - y + y := add_le_add_left h (c - y)
  rw [subadd c y] at h1
  have hEq : c - y + x ≤ c - x + x := by
    calc c - y + x ≤ c := h1
      _ = c - x + x := (subadd c x).symm
  exact add_le_add_right_cancel' hEq

/-- `a - b = -(b - a)`. -/
theorem neg_sub' (a b : α) : a - b = -(b - a) := by
  rw [sub_eq a b, sub_eq b a, neg_add_dist, neg_neg a, add_comm a (-b)]

/-- If `x < y` then `c - y < c - x` (strict). -/
theorem sub_lt_sub_right' {c x y : α} (h : x < y) : c - y < c - x := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hle : c - x ≤ c - y := le_of_not_lt hcon
  have heq : c - y = c - x := le_antisymm (sub_le_sub_right' (le_of_lt h)) hle
  have e1 : (c - x) + x = (c - x) + y := by
    rw [subadd c x, ← heq, subadd c y]
  exact lt_ne h (add_left_cancel e1)

/-- If `a ≤ b + c` then `a - c ≤ b`. -/
theorem sub_le_of_le_add {a b c : α} (h : a ≤ b + c) : a - c ≤ b := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hlt : b < a - c := lt_of_not_le (fun hle => hcon hle)
  have h1 : b + c < a - c + c := add_lt_add_right hlt c
  rw [subadd a c] at h1
  exact lt_irrefl _ (lt_of_le_of_lt h h1)

/-- Strict multiplication on the left: `y < x` and `0 < c` give `c*y < c*x`. -/
theorem mul_lt_mul_left_of_pos {c x y : α} (hc : (0 : α) < c) (h : y < x) :
    c * y < c * x := by
  have h1 : (0 : α) < c * (x - y) := mul_pos hc (sub_pos_of_lt h)
  have h2 : c * x - c * y = c * (x - y) := by
    rw [sub_eq (c * x) (c * y), ← neg_mul_right, ← distrib, sub_eq x y]
  apply lt_of_sub_pos
  rw [h2]
  exact h1

/-- The difference identity behind the limit law for sums:
`(a + b) − (c + d) = (a − c) + (b − d)`. -/
theorem sub_split_add (a b c d : α) :
    a + b - (c + d) = (a - c) + (b - d) := by
  have key : (a + -c) + (b + -d) = a + b + (-c + -d) := by
    rw [add_assoc a (-c) (b + -d), ← add_assoc (-c) b (-d), add_comm (-c) b,
        add_assoc b (-c) (-d), ← add_assoc a b (-c + -d)]
  rw [sub_eq (a + b) (c + d), neg_add_dist, sub_eq a c, sub_eq b d]
  exact key.symm

/-- The difference identity behind the limit law for differences:
`(a − b) − (c − d) = (a − c) + −(b − d)`, stated with the second error
negated so that the triangle inequality applies to it directly. -/
theorem sub_split (a b c d : α) : a - b - (c - d) = (a - c) + -(b - d) := by
  have h := sub_split_add a (-b) c (-d)
  rw [sub_eq (-b) (-d), neg_neg d, sub_eq (a + -b) (c + -d)] at h
  have hneg : -(b - d) = -b + d := by
    rw [sub_eq b d, neg_add_dist, neg_neg d]
  calc a - b - (c - d) = a + -b + (-(c + -d)) := by
        rw [sub_eq a b, sub_eq c d, sub_eq (a + -b) (c + -d)]
    _ = (a - c) + (-b + d) := h
    _ = (a - c) + -(b - d) := by rw [← hneg]

/-! ### The limit laws -/

/-- **Sums of convergent sequences converge** to the sum of the limits. -/
theorem tendsto_add {u v : Nat → α} {l m : α} (hu : Tendsto u l) (hv : Tendsto v m) :
    Tendsto (fun n => u n + v n) (l + m) := by
  intro ε hε
  have hη := half_pos hε
  obtain ⟨N₁, h₁⟩ := hu _ hη
  obtain ⟨N₂, h₂⟩ := hv _ hη
  refine ⟨N₁ + N₂, fun n hn => ?_⟩
  have hn1 : N₁ ≤ n := Nat.le_trans (Nat.le_add_right N₁ N₂) hn
  have hn2 : N₂ ≤ n := Nat.le_trans (Nat.le_add_left N₂ N₁) hn
  have hu' := h₁ n hn1
  have hv' := h₂ n hn2
  rw [sub_split_add (u n) (v n) l m]
  refine lt_of_le_of_lt (abv_triangle (u n - l) (v n - m)) ?_
  have hlt : abv (u n - l) + abv (v n - m) < ε / (1 + 1) + ε / (1 + 1) :=
    add_lt_add hu' hv'
  rw [half_half ε] at hlt
  exact hlt

/-- **Negatives converge** to the negative of the limit. -/
theorem tendsto_neg {u : Nat → α} {l : α} (hu : Tendsto u l) :
    Tendsto (fun n => -u n) (-l) := by
  intro ε hε
  obtain ⟨N, hN⟩ := hu ε hε
  refine ⟨N, fun n hn => ?_⟩
  have hsub : -u n - -l = -(u n - l) := by
    rw [sub_eq (-u n) (-l), sub_eq (u n) l, ← neg_add_dist]
  rw [hsub, abv_neg]
  exact hN n hn

/-- **Differences of convergent sequences converge** to the difference of
the limits. -/
theorem tendsto_sub {u v : Nat → α} {l m : α} (hu : Tendsto u l) (hv : Tendsto v m) :
    Tendsto (fun n => u n - v n) (l - m) := by
  intro ε hε
  have hη := half_pos hε
  obtain ⟨N₁, h₁⟩ := hu _ hη
  obtain ⟨N₂, h₂⟩ := hv _ hη
  refine ⟨N₁ + N₂, fun n hn => ?_⟩
  have hn1 : N₁ ≤ n := Nat.le_trans (Nat.le_add_right N₁ N₂) hn
  have hn2 : N₂ ≤ n := Nat.le_trans (Nat.le_add_left N₂ N₁) hn
  have hu' := h₁ n hn1
  have hv' := h₂ n hn2
  rw [sub_split (u n) (v n) l m]
  refine lt_of_le_of_lt (abv_triangle (u n - l) (-(v n - m))) ?_
  have hneg : abv (-(v n - m)) = abv (v n - m) := abv_neg (v n - m)
  rw [hneg]
  have hlt : abv (u n - l) + abv (v n - m) < ε / (1 + 1) + ε / (1 + 1) :=
    add_lt_add hu' hv'
  rw [half_half ε] at hlt
  exact hlt

/-! ### Eventual bounds, and the quotient law

Division needs the denominator to stay eventually bounded away from zero,
and that is a *consequence* of convergence to a nonzero limit, via the
reverse triangle inequality. -/

/-- A convergent sequence is eventually bounded: beyond some index, every
term's absolute value is at most the limit's plus one. -/
theorem tendsto_eventually_abv_bound {v : Nat → α} {m : α} (hv : Tendsto v m) :
    ∃ N : Nat, ∀ n : Nat, N ≤ n → abv (v n) ≤ abv m + 1 := by
  obtain ⟨N, hN⟩ := hv 1 zero_lt_one
  refine ⟨N, fun n hn => ?_⟩
  have h := hN n hn
  have e : m + (v n - m) = v n := add_sub_cancel m (v n)
  have htri : abv (m + (v n - m)) ≤ abv m + abv (v n - m) := abv_triangle m (v n - m)
  rw [e] at htri
  refine le_trans htri ?_
  have h1 : abv m + abv (v n - m) < abv m + 1 := add_lt_add_left h (abv m)
  exact le_of_lt h1

/-- **A sequence converging to a nonzero limit is eventually bounded away
from zero**: beyond some index, `|m| / 2 < |v n|`. -/
theorem tendsto_eventually_bounded_away {v : Nat → α} {m : α} (hv : Tendsto v m)
    (hm : m ≠ (0 : α)) :
    ∃ N : Nat, ∀ n : Nat, N ≤ n → abv m / (1 + 1) < abv (v n) := by
  have hampos : (0 : α) < abv m := (abv_pos).mpr hm
  have hhalf : (0 : α) < abv m / (1 + 1) := div_pos hampos two_pos
  obtain ⟨N, hN⟩ := hv _ hhalf
  refine ⟨N, fun n hn => ?_⟩
  have h := hN n hn
  have e : m + (v n - m) = v n := add_sub_cancel m (v n)
  have hrev : abv m ≤ abv (m + (v n - m)) + abv (v n - m) := by
    have h1 := abv_triangle (m + (v n - m)) (-(v n - m))
    rw [abv_neg] at h1
    have e2 : m + (v n - m) + -(v n - m) = m := by
      rw [add_assoc m (v n - m) (-(v n - m)), add_neg', add_zero']
    rw [e2] at h1
    exact h1
  rw [e] at hrev
  have hhalfsub : abv m - abv m / (1 + 1) = abv m / (1 + 1) := by
    refine add_right_cancel (c := abv m / (1 + 1)) ?_
    rw [subadd (abv m) (abv m / (1 + 1)), half_half (abv m)]
  have hstep1 : abv m - abv m / (1 + 1) ≤ abv m - abv (v n - m) :=
    sub_le_sub_right' (le_of_lt h)
  have hstep2 : abv m - abv (v n - m) ≤ abv (v n) :=
    sub_le_of_le_add hrev
  have hlt : abv m / (1 + 1) < abv m - abv (v n - m) := by
    rw [← hhalfsub]
    exact sub_lt_sub_right' h
  exact lt_of_lt_of_le hlt hstep2

/-! ### Quotients -/

/-- `x * (y / z) = x * y / z`. -/
theorem mul_div_assoc' (x y z : α) : x * (y / z) = x * y / z := by
  show x * (y * inv z) = x * y * inv z
  rw [← mul_assoc x y (inv z)]

/-- `(a / b) / c = a / (b * c)`, for nonzero `b` and `c`. -/
theorem div_div_mul (a b c : α) (hb : b ≠ (0 : α)) (hc : c ≠ (0 : α)) :
    a / b / c = a / (b * c) := by
  show a * inv b * inv c = a * inv (b * c)
  rw [inv_mul' hb hc, mul_assoc, mul_comm (inv b) (inv c)]

/-- A positive element is nonzero. -/
theorem pos_ne {a : α} (h : (0 : α) < a) : a ≠ (0 : α) :=
  fun he => ((lt_iff (0 : α) a).1 h).2 he.symm

/-- **Quotients of convergent sequences converge** to the quotient of the
limits, provided the limiting denominator is nonzero. This is the law the
skill-score asymptotics run on: a floor-corrected ratio whose denominator
converges to a nonzero limit converges to the ratio of the limits. -/
theorem tendsto_div {u v : Nat → α} {l m : α} (hu : Tendsto u l) (hv : Tendsto v m)
    (hm : m ≠ (0 : α)) : Tendsto (fun n => u n / v n) (l / m) := by
  have hampos : (0 : α) < abv m := (abv_pos).mpr hm
  have hsumpos : (0 : α) < abv m + abv l := by
    refine (lt_iff _ _).mpr ⟨add_nonneg (abv_nonneg m) (abv_nonneg l), fun he => pos_ne hampos ?_⟩
    have h1 : abv m ≤ abv m + abv l := le_add_of_nonneg_right (abv_nonneg l)
    rw [← he] at h1
    exact le_antisymm h1 (abv_nonneg m)
  have hsumne : abv m + abv l ≠ (0 : α) := pos_ne hsumpos
  intro ε hε
  have hDpos : (0 : α) < abv m * (abv m / (1 + 1)) :=
    mul_pos hampos (div_pos hampos two_pos)
  have hDne : abv m * (abv m / (1 + 1)) ≠ (0 : α) := pos_ne hDpos
  have hη : (0 : α) < ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)) :=
    div_pos (mul_pos hε hDpos) (mul_pos two_pos hsumpos)
  obtain ⟨N₁, h₁⟩ := tendsto_eventually_bounded_away hv hm
  obtain ⟨N₃, h₃⟩ := hu _ hη
  obtain ⟨N₄, h₄⟩ := hv _ hη
  refine ⟨N₁ + N₃ + N₄, fun n hn => ?_⟩
  have hn1 : N₁ ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_add_right N₁ N₃) (Nat.le_add_right (N₁ + N₃) N₄)) hn
  have hn3 : N₃ ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_add_left N₃ N₁) (Nat.le_add_right (N₁ + N₃) N₄)) hn
  have hn4 : N₄ ≤ n :=
    Nat.le_trans (Nat.le_add_left N₄ (N₁ + N₃)) hn
  have hb : abv m / (1 + 1) < abv (v n) := h₁ n hn1
  have hu' := h₃ n hn3
  have hv' := h₄ n hn4
  have hvne : v n ≠ (0 : α) := by
    intro he
    rw [he, abv_zero] at hb
    exact lt_irrefl _ (lt_of_le_of_lt (le_of_lt (div_pos hampos two_pos)) hb)
  have havpos : (0 : α) < abv (v n) :=
    (abv_pos).mpr hvne
  have hdenpos : (0 : α) < abv m * abv (v n) := mul_pos hampos havpos
  have hdennz : abv m * abv (v n) ≠ (0 : α) := pos_ne hdenpos
  -- the error identity: u n / v n − l / m = (m·u n − l·v n) / (m·v n)
  have hsplit : u n / v n - l / m = (m * u n - l * v n) / (m * v n) := by
    calc u n / v n - l / m
        = (m * u n) / (m * v n) - (v n * l) / (v n * m) := by
          rw [div_mul_mul hm hvne, div_mul_mul hvne hm]
      _ = (m * u n) / (m * v n) - (l * v n) / (m * v n) := by
          rw [mul_comm (v n) l, mul_comm (v n) m]
      _ = (m * u n - l * v n) / (m * v n) := sub_div_same
  -- the numerator identity: m·u n − l·v n = m·(u n − l) + l·(m − v n)
  have hmSub : ∀ a b : α, m * (a - b) = m * a - m * b := by
    intro a b; rw [sub_eq, distrib, neg_mul_right, sub_eq]
  have hlSub : ∀ a b : α, l * (a - b) = l * a - l * b := by
    intro a b; rw [sub_eq, distrib, neg_mul_right, sub_eq]
  have hreg : ∀ a p q : α, a + -p + (p + -q) = a + -q := by
    intro a p q
    calc a + -p + (p + -q) = a + (-p + (p + -q)) := add_assoc a (-p) (p + -q)
      _ = a + ((-p + p) + -q) := by rw [← add_assoc (-p) p (-q)]
      _ = a + ((0 : α) + -q) := by rw [neg_add p]
      _ = a + -q := by rw [zero_add]
  have hnumid : m * u n - l * v n = m * (u n - l) + l * (m - v n) := by
    rw [hmSub, hlSub, mul_comm l m, sub_eq (m * u n) (m * l),
        sub_eq (m * l) (l * v n), sub_eq (m * u n) (l * v n)]
    exact (hreg (m * u n) (m * l) (l * v n)).symm
  -- the numerator bound, in absolute values
  have hnum : abv (m * u n - l * v n)
      ≤ abv m * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
        + abv l * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l))) := by
    have htri : abv (m * u n - l * v n)
        ≤ abv (m * (u n - l)) + abv (l * (m - v n)) := by
      rw [hnumid]
      exact abv_triangle (m * (u n - l)) (l * (m - v n))
    rw [abv_mul, abv_mul] at htri
    have hb1 : abv m * abv (u n - l)
        ≤ abv m * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l))) :=
      mul_le_mul_of_nonneg_left (le_of_lt hu') (abv_nonneg m)
    have hmv : abv (m - v n) = abv (v n - m) := by
      rw [neg_sub' (m) (v n), abv_neg]
    have hb2 : abv l * abv (m - v n)
        ≤ abv l * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l))) := by
      rw [hmv]
      exact mul_le_mul_of_nonneg_left (le_of_lt hv') (abv_nonneg l)
    exact le_trans htri (add_le_add hb1 hb2)
  -- the denominator bound
  have hdenle : abv m * (abv m / (1 + 1)) ≤ abv m * abv (v n) :=
    le_of_lt (mul_lt_mul_left_of_pos hampos hb)
  -- assembling: |error| = numerator / denominator ≤ … ≤ ε/(1+1) < ε
  have habs : abv (u n / v n - l / m)
      = abv (m * u n - l * v n) / (abv m * abv (v n)) := by
    rw [hsplit, abv_div (mul_ne_zero hm hvne), abv_mul]
  rw [habs]
  have hstep1 : abv (m * u n - l * v n) / (abv m * abv (v n))
      ≤ (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
          / (abv m * abv (v n)) := by
    refine (div_le_iff hdenpos).mpr ?_
    rw [div_mul_cancel hdennz]
    have hsplit2 : abv m * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
        + abv l * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
        = (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l))) :=
      (distrib_right (abv m) (abv l) _).symm
    rw [← hsplit2]
    exact hnum
  have hstep2 : (abv m + abv l)
      * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
      / (abv m * abv (v n))
      ≤ (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
          / (abv m * (abv m / (1 + 1))) := by
    refine (le_div_iff hDpos).mpr ?_
    calc (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
          / (abv m * abv (v n)) * (abv m * (abv m / (1 + 1)))
        ≤ (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
          / (abv m * abv (v n)) * (abv m * abv (v n)) :=
          mul_le_mul_of_nonneg_left hdenle (div_nonneg
            (mul_nonneg (add_nonneg (abv_nonneg m) (abv_nonneg l)) (le_of_lt hη))
            hdenpos)
      _ = (abv m + abv l)
          * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l))) :=
          div_mul_cancel hdennz _
  have hvalue : (abv m + abv l)
      * (ε * (abv m * (abv m / (1 + 1))) / ((1 + 1) * (abv m + abv l)))
      / (abv m * (abv m / (1 + 1)))
      = ε / (1 + 1) := by
    rw [mul_div_assoc' (abv m + abv l) (ε * (abv m * (abv m / (1 + 1))))
        ((1 + 1) * (abv m + abv l)),
      mul_comm ((1 : α) + 1) (abv m + abv l), div_mul_mul hsumne two_ne_zero,
      div_div_mul (ε * (abv m * (abv m / (1 + 1)))) (1 + 1)
        (abv m * (abv m / (1 + 1))) two_ne_zero hDne,
      mul_comm ε (abv m * (abv m / (1 + 1))),
      mul_comm ((1 : α) + 1) (abv m * (abv m / (1 + 1))),
      div_mul_mul hDne two_ne_zero]
  have hfinal : abv (m * u n - l * v n) / (abv m * abv (v n)) ≤ ε / (1 + 1) :=
    le_trans (le_trans hstep1 hstep2) (le_of_eq hvalue)
  refine lt_of_le_of_lt hfinal ?_
  have hle : ε / (1 + 1) ≤ ε := by
    refine (div_le_iff two_pos).mpr ?_
    rw [distrib ε 1 1, mul_one' ε]
    exact le_add_of_nonneg_right (le_of_lt hε)
  have hne : ε / (1 + 1) ≠ ε := by
    intro he
    have h4 : ε = ε * (1 + 1) := by
      have h5 : ε / (1 + 1) * (1 + 1) = ε * (1 + 1) := by rw [he]
      rw [div_mul_cancel two_ne_zero ε] at h5
      exact h5
    rw [distrib ε 1 1, mul_one' ε] at h4
    have h6 : ε + (0 : α) = ε + ε := by rw [add_zero']; exact h4
    have h7 : (0 : α) = ε := add_left_cancel h6
    rw [h7] at hε
    exact lt_irrefl ε hε
  exact (lt_iff _ _).mpr ⟨hle, hne⟩

/-! ## The power-law floor -/

/-- The embedded successor of a natural is positive. -/
theorem toα_succ_pos (n : Nat) : (0 : α) < toα (n + 1) := by
  have h1 : (1 : α) + (0 : α) ≤ (1 : α) + toα n := add_le_add_left (toα_nonneg n) (1 : α)
  rw [add_zero'] at h1
  rw [toα_succ n, add_comm (toα n) (1 : α)]
  exact lt_of_lt_of_le zero_lt_one h1

/-- The power of an embedded successor is positive, for a positive
exponent. -/
theorem rpow_toα_pos {s : α} (hs : (0 : α) < s) (n : Nat) :
    (0 : α) < rpow (toα (n + 1)) s :=
  rpow_pos (toα_succ_pos n) hs

/-- The paper's fitted floor law, as a sequence: `C · (n+1)^(−s)`.

The `n + 1` avoids the degenerate `0^s` at the zeroth term; the paper's n
is the sample size, which is at least one. -/
def powLaw (C s : α) : Nat → α :=
  fun n => C * inv (rpow (toα (n + 1)) s)

/-- The power-law floor is nonnegative whenever the constant is. -/
theorem powLaw_nonneg {C s : α} (hC : (0 : α) ≤ C) (hs : (0 : α) < s) (n : Nat) :
    (0 : α) ≤ powLaw C s n := by
  have hR : (0 : α) < rpow (toα (n + 1)) s := rpow_toα_pos hs n
  exact mul_nonneg hC (le_of_lt (inv_pos hR))

/-- **The power-law floor vanishes.** For every positive exponent and
nonnegative constant, `C · (n+1)^(−s) → 0` as `n → ∞`.

This is the analytic fact the paper's asymptotics run on: the fitted floor
decreases to zero, so subtracting it is a first-order correction that
vanishes. The proof is the three power laws plus the ordered-field axioms:
unboundedness of `(n : α)^s` turns `C/ε` into a threshold, monotonicity
carries it to every larger index, and inversion converts it back to the
smallness of the floor. -/
theorem powLaw_tendsto_zero {C s : α} (hC : (0 : α) ≤ C) (hs : (0 : α) < s) :
    Tendsto (powLaw C s) (0 : α) := by
  intro ε hε
  rcases Classical.em (C = (0 : α)) with hC0 | hC0
  · refine ⟨0, fun n _ => ?_⟩
    have hz : powLaw C s n = (0 : α) := by
      show C * inv (rpow (toα (n + 1)) s) = (0 : α)
      rw [hC0, zero_mul]
    rw [hz, sub_self, abv_zero]
    exact hε
  · have hCpos : (0 : α) < C := by
      rcases Classical.em ((0 : α) < C) with h | h
      · exact h
      · exact absurd (le_antisymm hC (le_of_not_lt h)).symm hC0
    obtain ⟨N₀, hN₀⟩ := rpow_unbounded hs (C / ε)
    refine ⟨N₀, fun n hn => ?_⟩
    have hRn : (0 : α) < rpow (toα (n + 1)) s := rpow_toα_pos hs n
    have hmono : rpow (toα N₀) s ≤ rpow (toα (n + 1)) s :=
      rpow_mono (toα_nonneg N₀)
        (toα_mono (Nat.le_trans hn (Nat.le_succ n))) hs
    have hbound : C / ε < rpow (toα (n + 1)) s := lt_of_lt_of_le hN₀ hmono
    have hRε : C < ε * rpow (toα (n + 1)) s := by
      have h1 : C / ε * ε < rpow (toα (n + 1)) s * ε :=
        mul_lt_mul_right_of_lt hε hbound
      rw [div_mul_cancel (ne_zero_of_pos hε) C] at h1
      rw [mul_comm (rpow (toα (n + 1)) s) ε] at h1
      exact h1
    have hfinal : C * inv (rpow (toα (n + 1)) s) < ε := by
      refine Classical.byContradiction (fun hcon => ?_)
      have hle : ε ≤ C * inv (rpow (toα (n + 1)) s) := le_of_not_lt hcon
      have e1 : C * inv (rpow (toα (n + 1)) s) * rpow (toα (n + 1)) s = C := by
        rw [mul_assoc, mul_comm (inv (rpow (toα (n + 1)) s)) (rpow (toα (n + 1)) s),
          mul_inv_cancel (ne_zero_of_pos hRn), mul_one']
      have h1 : ε * rpow (toα (n + 1)) s
          ≤ C * inv (rpow (toα (n + 1)) s) * rpow (toα (n + 1)) s :=
        mul_le_mul_of_nonneg_right hle (le_of_lt hRn)
      rw [e1] at h1
      exact lt_irrefl _ (lt_of_le_of_lt h1 hRε)
    have hnn : (0 : α) ≤ powLaw C s n := powLaw_nonneg hC hs n
    rw [sub_zero, abv_of_nonneg hnn]
    exact hfinal

/-! ## The skill-score asymptotics (the first of the three boundary
theorems) -/

/-- **The asymptotic n-invariance of the skill score.** If the floor
sequence converges to zero, then the floor-corrected vacuity

    V n = (D − f n) / (D_B − f n)

converges to `D / D_B`, for any baseline discrepancy `D_B ≠ 0`. The
finite-n statistic is *not* n-invariant -- subtracting the floor is a
first-order correction, not an exact cancellation -- but asymptotically the
floor drops out of the ratio entirely. This is the statement of the paper's
§5.1 in its general form: for a fixed Q ≠ P the limit is the ratio of raw
discrepancies `D(P,Q)/D(P,Q_B)`, independent of n. -/
theorem vacuity_tendsto {f : Nat → α} {D DB : α} (hf : Tendsto f (0 : α))
    (hDB : DB ≠ (0 : α)) :
    Tendsto (fun n => Skill.vacuity D (f n) DB) (D / DB) := by
  have hnum : Tendsto (fun n => D - f n) (D - (0 : α)) :=
    tendsto_sub (tendsto_const D) hf
  have hden : Tendsto (fun n => DB - f n) (DB - (0 : α)) :=
    tendsto_sub (tendsto_const DB) hf
  have hDB0 : DB - (0 : α) ≠ (0 : α) := by
    rw [sub_zero]
    exact hDB
  have hmain := tendsto_div hnum hden hDB0
  refine tendsto_limit_eq ?_ hmain
  rw [sub_zero D, sub_zero DB]

/-- **The equal-law case.** When the observed discrepancy is zero (Q = P),
the vacuity ratio converges to zero: an honest generator's skill tends to
perfection as the floor vanishes. This is the `V → 0` half of the paper's
§5.1 statement. -/
theorem vacuity_tendsto_self {f : Nat → α} {DB : α} (hf : Tendsto f (0 : α))
    (hDB : DB ≠ (0 : α)) :
    Tendsto (fun n => Skill.vacuity (0 : α) (f n) DB) (0 : α) := by
  have h := vacuity_tendsto (D := (0 : α)) hf hDB
  refine tendsto_limit_eq ?_ h
  have hzero : (0 : α) / DB = (0 : α) := by
    show (0 : α) * inv DB = (0 : α)
    exact zero_mul (inv DB)
  rw [hzero]

/-- **The composed asymptotic: the paper's fitted floor.** With the floor
sequence given by the power law `C · (n+1)^(−s)`, the vacuity ratio
converges to `D / D_B` (and to zero when `D = 0`). This is the exact
statement of the paper's §5.1 asymptotics under its own floor model. -/
theorem skill_asymptotic {D DB C s : α} (hC : (0 : α) ≤ C) (hs : (0 : α) < s)
    (hDB : DB ≠ (0 : α)) :
    Tendsto (fun n => Skill.vacuity D (powLaw C s n) DB) (D / DB) :=
  vacuity_tendsto (powLaw_tendsto_zero hC hs) hDB

/-- **The composed asymptotic, equal-law case.** -/
theorem skill_asymptotic_self {DB C s : α} (hC : (0 : α) ≤ C) (hs : (0 : α) < s)
    (hDB : DB ≠ (0 : α)) :
    Tendsto (fun n => Skill.vacuity (0 : α) (powLaw C s n) DB) (0 : α) :=
  vacuity_tendsto_self (powLaw_tendsto_zero hC hs) hDB

end Analysis
end Certgen
