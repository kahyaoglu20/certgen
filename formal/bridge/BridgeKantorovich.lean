import Bridge
import CertgenFormal.Kantorovich
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Convex.Birkhoff

/-!
# Boundary theorem 3 over the reals: the Kantorovich integral optimum

`CertgenFormal/Kantorovich.lean` (zero dependencies) states the
integral-optimum property `HasIntegralOptimum` -- every fractional
transport plan between two equal-size uniform empirical measures is
dominated in cost by a matching -- and reduces the Kantorovich/assignment
value equivalence to it. This file proves the property over `ℝ`, in the
development's own
list-based terms (`hasIntegralOptimum_real`), for every ground cost and
every pair of duplicate-free index lists of equal length.

The route is not the walk-shift-peel recursion sketched in the core file
but Birkhoff's theorem, which Mathlib v4.15.0 provides
(`exists_eq_sum_perm_of_mem_doublyStochastic`): a doubly stochastic matrix
is a convex combination of permutation matrices; the transport cost is
linear in the plan; a convex combination is at least its smallest term.
The transfer then reads the development's `Plan` as a doubly stochastic
matrix (row and column sums from `Plan.row`/`Plan.col`, the list sum
identified with `List.sum`) and a permutation back as a `Matching`.

`empirical_w2_attained_by_assignment` is the instance the paper uses
(§3.5): squared-Euclidean cost between two point clouds -- the exact
two-sample `W₂` of uniform empirical measures of equal size is attained by
an assignment, which is what the Hungarian algorithm computes and why no
entropic regularisation is needed.

The zero-dependency core still carries the property as an assumption of
its reduction: a Mathlib-free proof (the cycle-shifting recursion) is not
formalised. Every theorem here is pinned below by `#guard_msgs`.
-/

open Matrix Finset

namespace CertgenBridge.Kantorovich

variable {n : Type*} [Fintype n] [DecidableEq n]

set_option linter.unusedSectionVars false

/-- Transport cost of a (fractional) plan `M` under ground cost `c`. -/
noncomputable def planCost (c M : Matrix n n ℝ) : ℝ := ∑ i, ∑ j, M i j * c i j

/-- Cost of the assignment `σ`. -/
noncomputable def assignCost (c : Matrix n n ℝ) (σ : Equiv.Perm n) : ℝ := ∑ i, c i (σ i)

theorem planCost_sum_smul {ι : Type*} (s : Finset ι) (c : Matrix n n ℝ)
    (w : ι → ℝ) (P : ι → Matrix n n ℝ) :
    planCost c (∑ k ∈ s, w k • P k) = ∑ k ∈ s, w k * planCost c (P k) := by
  unfold planCost
  have h1 : ∀ i j, (∑ k ∈ s, w k • P k) i j * c i j
      = ∑ k ∈ s, w k * (P k i j * c i j) := by
    intro i j
    rw [Matrix.sum_apply, Finset.sum_mul]
    refine Finset.sum_congr rfl (fun k _ => ?_)
    rw [Matrix.smul_apply, smul_eq_mul]; ring
  simp only [h1, Finset.mul_sum]
  calc (∑ i, ∑ j, ∑ k ∈ s, w k * (P k i j * c i j))
      = ∑ i, ∑ k ∈ s, ∑ j, w k * (P k i j * c i j) := by
        refine Finset.sum_congr rfl (fun i _ => ?_)
        exact Finset.sum_comm
    _ = ∑ k ∈ s, ∑ i, ∑ j, w k * (P k i j * c i j) := Finset.sum_comm

theorem planCost_permMatrix (c : Matrix n n ℝ) (σ : Equiv.Perm n) :
    planCost c (σ.permMatrix ℝ) = assignCost c σ := by
  unfold planCost assignCost
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Finset.sum_eq_single (σ i)]
  · simp [Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply]
  · intro j _ hj
    simp [Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, Ne.symm hj]
  · intro h; exact absurd (Finset.mem_univ _) h


/-- A cost-minimising assignment exists (the permutation group is finite
and nonempty). -/
theorem exists_min_assign (c : Matrix n n ℝ) :
    ∃ σ₀ : Equiv.Perm n, ∀ σ : Equiv.Perm n, assignCost c σ₀ ≤ assignCost c σ := by
  obtain ⟨σ₀, -, h⟩ := Finset.exists_min_image (Finset.univ : Finset (Equiv.Perm n))
    (assignCost c) Finset.univ_nonempty
  exact ⟨σ₀, fun σ => h σ (Finset.mem_univ σ)⟩

/-- **Integral-optimum theorem (boundary theorem 3, over `ℝ`).** Every
fractional transport plan between two uniform empirical measures of equal
size -- a doubly stochastic matrix -- is dominated in cost by a
permutation. Proof: Birkhoff's theorem writes the plan as a convex
combination of permutation matrices; the cost is linear; a convex
combination is at least its smallest term. -/
theorem integral_optimum (c M : Matrix n n ℝ) (hM : M ∈ doublyStochastic ℝ n) :
    ∃ σ : Equiv.Perm n, assignCost c σ ≤ planCost c M := by
  obtain ⟨w, hw0, hw1, hwM⟩ := exists_eq_sum_perm_of_mem_doublyStochastic hM
  obtain ⟨σ₀, hmin⟩ := exists_min_assign c
  refine ⟨σ₀, ?_⟩
  rw [← hwM, planCost_sum_smul]
  simp only [planCost_permMatrix]
  calc assignCost c σ₀ = ∑ σ, w σ * assignCost c σ₀ := by
        rw [← Finset.sum_mul, hw1, one_mul]
    _ ≤ ∑ σ, w σ * assignCost c σ :=
        Finset.sum_le_sum (fun σ _ => mul_le_mul_of_nonneg_left (hmin σ) (hw0 σ))

/-- **The Kantorovich LP over uniform equal-size empirical measures has an
integral optimal solution.** Some permutation matrix is a feasible plan
whose cost is no larger than that of any feasible plan; hence the LP value
equals the assignment value, which the Hungarian algorithm computes. -/
theorem kantorovich_has_integral_optimum (c : Matrix n n ℝ) :
    ∃ σ₀ : Equiv.Perm n, σ₀.permMatrix ℝ ∈ doublyStochastic ℝ n ∧
      ∀ M ∈ doublyStochastic ℝ n, planCost c (σ₀.permMatrix ℝ) ≤ planCost c M := by
  obtain ⟨σ₀, hmin⟩ := exists_min_assign c
  refine ⟨σ₀, permMatrix_mem_doublyStochastic, fun M hM => ?_⟩
  obtain ⟨σ, hσ⟩ := integral_optimum c M hM
  rw [planCost_permMatrix]
  exact le_trans (hmin σ) hσ

/-- The value form: the infimum of the plan cost over doubly stochastic
matrices is attained and equals the minimum assignment cost. -/
theorem kantorovich_value_eq_assignment_value (c : Matrix n n ℝ) :
    ∃ σ₀ : Equiv.Perm n,
      (∀ σ : Equiv.Perm n, assignCost c σ₀ ≤ assignCost c σ) ∧
      (∀ M ∈ doublyStochastic ℝ n, assignCost c σ₀ ≤ planCost c M) ∧
      planCost c (σ₀.permMatrix ℝ) = assignCost c σ₀ := by
  obtain ⟨σ₀, hmin⟩ := exists_min_assign c
  refine ⟨σ₀, hmin, fun M hM => ?_, planCost_permMatrix c σ₀⟩
  obtain ⟨σ, hσ⟩ := integral_optimum c M hM
  exact le_trans (hmin σ) hσ

/-- The instance the paper uses (§3.5): squared-Euclidean ground cost
between two point clouds of equal size. The exact two-sample `W₂²` of the
uniform empirical measures (the LP over couplings, i.e. doubly stochastic
plans scaled by `1/n`) is attained by a permutation. -/
theorem empirical_w2_attained_by_assignment {p : ℕ}
    (x y : n → EuclideanSpace ℝ (Fin p)) :
    ∃ σ₀ : Equiv.Perm n, ∀ M ∈ doublyStochastic ℝ n,
      (∑ i, ‖x i - y (σ₀ i)‖ ^ 2) ≤ ∑ i, ∑ j, M i j * ‖x i - y j‖ ^ 2 := by
  obtain ⟨σ₀, -, hle, -⟩ :=
    kantorovich_value_eq_assignment_value (fun i j => ‖x i - y j‖ ^ 2)
  exact ⟨σ₀, fun M hM => hle M hM⟩


/-! ## Transfer to the list-based statement of `CertgenFormal.Kantorovich` -/

section Transfer

open Certgen

variable {ι : Type*} [DecidableEq ι]

/-- The development's list sum is `List.sum` over `ℝ`. -/
theorem rsum_eq (l : List ℝ) : Certgen.Replicates.sum l = l.sum := by
  induction l with
  | nil => rfl
  | cons a as ih =>
      rw [Certgen.Replicates.sum_cons, List.sum_cons, ih]

/-- Sums over the pair list `lprod` are iterated sums. -/
theorem sum_lprod (F : ι × ι → ℝ) : ∀ (l₁ l₂ : List ι),
    ((Certgen.Kantorovich.lprod l₁ l₂).map F).sum
      = (l₁.map (fun i => (l₂.map (fun j => F (i, j))).sum)).sum := by
  intro l₁
  induction l₁ with
  | nil => intro l₂; rfl
  | cons a as ih =>
      intro l₂
      show ((l₂.map (fun b => (a, b)) ++ Certgen.Kantorovich.lprod as l₂).map F).sum = _
      rw [List.map_append, List.sum_append, ih, List.map_map, List.map_cons, List.sum_cons]
      rfl

/-- Column index of the square matrix: position `b` of `idx₂`. -/
def colIdx (idx₁ idx₂ : List ι) (hlen : idx₁.length = idx₂.length)
    (b : Fin idx₁.length) : ι :=
  idx₂[b.1]'(hlen ▸ b.2)

theorem sum_col (idx₁ idx₂ : List ι) (hlen : idx₁.length = idx₂.length) (f : ι → ℝ) :
    ∑ b : Fin idx₁.length, f (colIdx idx₁ idx₂ hlen b) = (idx₂.map f).sum := by
  rw [← Fin.sum_univ_get' idx₂ f]
  exact Fintype.sum_equiv (finCongr hlen) _ _ (fun b => rfl)

theorem sum_row (idx₁ : List ι) (f : ι → ℝ) :
    ∑ a : Fin idx₁.length, f (idx₁[a.1]) = (idx₁.map f).sum :=
  Fin.sum_univ_get' idx₁ f

/-- **Boundary theorem 3 over `ℝ`, in the development's own terms.** For
duplicate-free index lists of equal length, every fractional plan in the
sense of `CertgenFormal.Kantorovich.Plan` is dominated in cost by a
`Matching` -- the property `HasIntegralOptimum` holds for every ground
cost. Proof: transport the plan to a doubly stochastic matrix, apply
`integral_optimum` (Birkhoff--von Neumann), and read the permutation back
as a matching. -/
theorem hasIntegralOptimum_real (γ : ι → ι → ℝ) {idx₁ idx₂ : List ι}
    (h₁ : idx₁.Nodup) (h₂ : idx₂.Nodup) (hlen : idx₁.length = idx₂.length) :
    Certgen.Kantorovich.HasIntegralOptimum γ idx₁ idx₂ := by
  intro m hm
  classical
  let col := colIdx idx₁ idx₂ hlen
  let M : Matrix (Fin idx₁.length) (Fin idx₁.length) ℝ :=
    fun a b => m (idx₁[a.1]) (col b)
  let c : Matrix (Fin idx₁.length) (Fin idx₁.length) ℝ :=
    fun a b => γ (idx₁[a.1]) (col b)
  have hcolmem : ∀ b, col b ∈ idx₂ := fun b => List.getElem_mem _
  have hrowmem : ∀ a : Fin idx₁.length, idx₁[a.1] ∈ idx₁ := fun a => List.getElem_mem _
  -- M is doubly stochastic
  have hM : M ∈ doublyStochastic ℝ (Fin idx₁.length) := by
    rw [mem_doublyStochastic_iff_sum]
    refine ⟨fun a b => hm.nonneg _ (hrowmem a) _ (hcolmem b), fun a => ?_, fun b => ?_⟩
    · show ∑ b, m (idx₁[a.1]) (col b) = 1
      rw [sum_col idx₁ idx₂ hlen (fun j => m (idx₁[a.1]) j), ← rsum_eq]
      exact hm.row _ (hrowmem a)
    · show ∑ a : Fin idx₁.length, m (idx₁[a.1]) (col b) = 1
      rw [sum_row idx₁ (fun i => m i (col b)), ← rsum_eq]
      exact hm.col _ (hcolmem b)
  obtain ⟨σ, hσ⟩ := integral_optimum c M hM
  -- the matching read back from σ
  let pos : ∀ i, i ∈ idx₁ → Fin idx₁.length :=
    fun i h => ⟨idx₁.indexOf i, List.indexOf_lt_length.2 h⟩
  let μ : ι → ι := fun i => if h : i ∈ idx₁ then col (σ (pos i h)) else i
  have hpos_get : ∀ a : Fin idx₁.length, pos (idx₁[a.1]) (hrowmem a) = a := by
    intro a
    apply Fin.ext
    have := List.get_indexOf h₁ a
    simpa [pos, List.get_eq_getElem] using this
  have hget_pos : ∀ i (h : i ∈ idx₁), idx₁[(pos i h).1] = i := by
    intro i h
    have := List.indexOf_get (l := idx₁) (a := i) (List.indexOf_lt_length.2 h)
    simp only [List.get_eq_getElem] at this
    exact this
  have hμ_get : ∀ a : Fin idx₁.length, μ (idx₁[a.1]) = col (σ a) := by
    intro a
    simp only [μ, dif_pos (hrowmem a), hpos_get a]
  refine ⟨μ, ⟨fun i hi => ?_, fun i hi i' hi' he => ?_⟩, ?_⟩
  · simp only [μ, dif_pos hi]; exact hcolmem _
  · simp only [μ, dif_pos hi, dif_pos hi'] at he
    have hb : (σ (pos i hi)).1 = (σ (pos i' hi')).1 := by
      have h2 := (List.Nodup.getElem_inj_iff h₂).1 he
      exact h2
    have hσeq : σ (pos i hi) = σ (pos i' hi') := Fin.ext hb
    have hpeq : pos i hi = pos i' hi' := σ.injective hσeq
    rw [← hget_pos i hi, ← hget_pos i' hi', hpeq]
  · -- costs
    have hm_cost : Certgen.Kantorovich.mcost γ idx₁ μ = assignCost c σ := by
      unfold Certgen.Kantorovich.mcost assignCost
      rw [rsum_eq, ← sum_row idx₁ (fun i => γ i (μ i))]
      refine Finset.sum_congr rfl (fun a _ => ?_)
      rw [hμ_get a]
    have hp_cost : Certgen.Kantorovich.pcost γ idx₁ idx₂ m = planCost c M := by
      unfold Certgen.Kantorovich.pcost planCost
      rw [rsum_eq, sum_lprod (fun p => m p.1 p.2 * γ p.1 p.2),
        ← sum_row idx₁ (fun i => (idx₂.map (fun j => m i j * γ i j)).sum)]
      refine Finset.sum_congr rfl (fun a _ => ?_)
      rw [← sum_col idx₁ idx₂ hlen (fun j => m (idx₁[a.1]) j * γ (idx₁[a.1]) j)]
    rw [hm_cost, hp_cost]
    exact hσ

/-- The value equivalence over `ℝ` in the development's own terms: a
minimum-cost matching lower-bounds every fractional plan. -/
theorem assignment_le_plan_real (γ : ι → ι → ℝ) {idx₁ idx₂ : List ι}
    (h₁ : idx₁.Nodup) (h₂ : idx₂.Nodup) (hlen : idx₁.length = idx₂.length)
    {μ₀ : ι → ι}
    (hmin : ∀ μ : ι → ι, Certgen.Kantorovich.Matching idx₁ idx₂ μ →
      Certgen.Kantorovich.mcost γ idx₁ μ₀ ≤ Certgen.Kantorovich.mcost γ idx₁ μ)
    (m : ι → ι → ℝ) (hm : Certgen.Kantorovich.Plan idx₁ idx₂ m) :
    Certgen.Kantorovich.mcost γ idx₁ μ₀ ≤ Certgen.Kantorovich.pcost γ idx₁ idx₂ m :=
  Certgen.Kantorovich.assignment_le_plan_of_integral_optimum
    (hasIntegralOptimum_real γ h₁ h₂ hlen) hmin m hm

end Transfer

end CertgenBridge.Kantorovich

/-- info: 'CertgenBridge.Kantorovich.integral_optimum' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.integral_optimum

/-- info: 'CertgenBridge.Kantorovich.kantorovich_has_integral_optimum' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.kantorovich_has_integral_optimum

/--
info: 'CertgenBridge.Kantorovich.kantorovich_value_eq_assignment_value' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.kantorovich_value_eq_assignment_value

/--
info: 'CertgenBridge.Kantorovich.empirical_w2_attained_by_assignment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.empirical_w2_attained_by_assignment

/-- info: 'CertgenBridge.Kantorovich.hasIntegralOptimum_real' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.hasIntegralOptimum_real

/-- info: 'CertgenBridge.Kantorovich.assignment_le_plan_real' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms CertgenBridge.Kantorovich.assignment_le_plan_real

