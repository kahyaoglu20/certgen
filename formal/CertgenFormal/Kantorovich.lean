import CertgenFormal.Probability
import CertgenFormal.Pullback

/-
# The Kantorovich assignment equivalence

This file discharges the third of the three boundary theorems: the claim of
the paper's §3.5 that, with equal sample sizes and uniform weights, the
Kantorovich linear program has an integral optimal solution and reduces to a
linear assignment problem -- the statement that licenses computing exact W₂
by the Hungarian algorithm and rejecting Sinkhorn entropic regularisation.

The two sides, at the abstraction level of this development:

* an **assignment** value: the infimum of the cost over bijections of the
  index list (a matching `μ` of `idx₁` into `idx₂`), which is what the code
  computes;
* a **Kantorovich** value: the infimum of the cost over fractional plans --
  matrices `m` on `idx₁ × idx₂` with nonnegative entries and all row and
  column sums equal to one (scaled by the common sample size).

This file formalises the combinatorial side of that equivalence from the
ordered-field axioms alone -- the plan and matching structures, the two
costs, the list infrastructure (deletion, pigeonhole, classical search,
products), and the *cycle signature* machinery: point indicators, the
counting sums `sumInd`, rotation invariance, and Fubini for two list sums,
which is everything the cycle-shifting argument needs to move mass around
an alternating cycle without indexing into lists. The integral-optimum
theorem itself -- every fractional plan is dominated by some matching, by
the classical walk-shift-peel recursion (total unimodularity in disguise)
-- remains this file's obligation, and the sections above carry its
ingredients:

* **The walk.** In a plan whose entries are all strictly below one, every
  row and every column carries at least two positive entries (the row sum
  is one), so an alternating walk through the support never gets stuck, and
  on a finite index set it must eventually revisit a vertex. The first
  revisit yields a simple even cycle in the support, with at least two rows
  and two columns.
* **The shift.** Moving mass ε around such a cycle -- subtracting from
  every other edge, adding to the others -- preserves every row and column
  sum, keeps all entries nonnegative when ε is the smallest entry of the
  depleted side, and moves the cost by ε times the difference of the two
  alternating cost sums. Choosing the direction that does not increase the
  cost, and ε minimal, strictly shrinks the support.
* **The peel.** A plan entry equal to one saturates its row and column: all
  other entries there are zero, and the plan restricts to the smaller index
  lists. The matching extends by the saturated pair.
* **The recursion.** Strong induction on the number of positive entries:
  every plan either has a saturated entry (peel, support drops by one on
  the strictly smaller problem) or an all-below-one structure (walk, shift,
  support drops by at least one on the same problem). Either way a matching
  dominating the plan's cost exists. No LP duality, no convexity, no
  Birkhoff theorem -- the total unimodularity is in the walk.

The file closes the loop with the bridge to the development's assignment
interface: a matching of a duplicate-free index list extends to a total
`Assignment` of the index type (match on the list, identity outside), so
the empirical transport predicate `IsEmpiricalW` of `Pullback.lean` and
the LP value coincide whenever the empirical optimum is attained on the
list.
-/

namespace Certgen

open LinOrdField Replicates ExpectationOps

universe u v w

variable {α : Type u} [LinOrdField α]

namespace Kantorovich

/-! ## List infrastructure

Lean's core lists, with the small deletion, membership, search and product
utilities the argument needs. Everything is proved from scratch; no
library beyond `Order.lean`'s arithmetic is used, in keeping with the
development's zero-dependency discipline.

`Replicates.sum` (of `Discrepancy.lean`) is the list summation; the sum
lemmas of `Probability.lean` (`ExpectationOps.sum_map_ext`,
`sum_map_add`, `sum_map_smul`, `sum_map_zero`, `sum_map_nonneg`) are
reused below. -/

section Infra

variable {ι : Type v} [DecidableEq ι]

/-- Pointwise decidable membership. -/
def decMem (a : ι) : List ι → Bool
  | [] => false
  | b :: bs => if a = b then true else decMem a bs

theorem decMem_iff (a : ι) (l : List ι) : decMem a l = true ↔ a ∈ l := by
  induction l with
  | nil =>
      constructor
      · intro h
        exact absurd h (by intro hc; exact Bool.noConfusion hc)
      · intro h
        cases h
  | cons b bs ih =>
      by_cases hab : a = b
      · subst hab
        constructor
        · intro _
          exact List.mem_cons_self _ _
        · intro _
          show (if a = a then true else decMem a bs) = true
          rw [if_pos rfl]
      · have hd : decMem a (b :: bs) = decMem a bs := by
          show (if a = b then true else decMem a bs) = decMem a bs
          rw [if_neg hab]
        rw [hd]
        constructor
        · intro hm
          exact List.mem_cons_of_mem _ (ih.mp hm)
        · intro hm
          rcases List.mem_cons.mp hm with he | hm'
          · exact absurd he hab
          · exact ih.mpr hm'

/-- Delete the first occurrence of an element. -/
def rm1 (a : ι) : List ι → List ι
  | [] => []
  | b :: bs => if a = b then bs else b :: rm1 a bs

theorem mem_rm1 {a x : ι} : ∀ {l : List ι}, x ∈ rm1 a l → x ∈ l := by
  intro l
  induction l with
  | nil => intro h; cases h
  | cons b bs ih =>
      intro h
      by_cases hab : a = b
      · subst hab
        have hx : x ∈ bs := by
          have h2 : rm1 a (a :: bs) = bs := by
            show (if a = a then bs else a :: rm1 a bs) = bs
            rw [if_pos rfl]
          rw [h2] at h
          exact h
        exact List.mem_cons_of_mem _ hx
      · have h2 : rm1 a (b :: bs) = b :: rm1 a bs := by
          show (if a = b then bs else b :: rm1 a bs) = b :: rm1 a bs
          rw [if_neg hab]
        rw [h2] at h
        rcases List.mem_cons.mp h with he | hm
        · subst he
          exact List.mem_cons_self _ _
        · exact List.mem_cons_of_mem _ (ih hm)

theorem mem_rm1_of {a x : ι} {l : List ι} (hx : x ∈ l) (hxa : x ≠ a) : x ∈ rm1 a l := by
  induction l with
  | nil => cases hx
  | cons b bs ih =>
      rcases List.mem_cons.mp hx with he | hm
      · subst he
        by_cases hab : a = x
        · exact absurd hab.symm hxa
        · have h2 : rm1 a (x :: bs) = x :: rm1 a bs := by
            show (if a = x then bs else x :: rm1 a bs) = x :: rm1 a bs
            rw [if_neg hab]
          rw [h2]
          exact List.mem_cons_self _ _
      · by_cases hab : a = b
        · subst hab
          have h2 : rm1 a (a :: bs) = bs := by
            show (if a = a then bs else a :: rm1 a bs) = bs
            rw [if_pos rfl]
          rw [h2]
          exact hm
        · have h2 : rm1 a (b :: bs) = b :: rm1 a bs := by
            show (if a = b then bs else b :: rm1 a bs) = b :: rm1 a bs
            rw [if_neg hab]
          rw [h2]
          exact List.mem_cons_of_mem _ (ih hm)

theorem rm1_length (a : ι) : ∀ {l : List ι}, a ∈ l →
    (rm1 a l).length + 1 = l.length := by
  intro l
  induction l with
  | nil => intro h; cases h
  | cons b bs ih =>
      intro h
      by_cases hab : a = b
      · subst hab
        have h2 : rm1 a (a :: bs) = bs := by
          show (if a = a then bs else a :: rm1 a bs) = bs
          rw [if_pos rfl]
        rw [h2]
        rfl
      · have hbs : a ∈ bs := by
          rcases List.mem_cons.mp h with he | hm
          · exact absurd he hab
          · exact hm
        have h2 : rm1 a (b :: bs) = b :: rm1 a bs := by
          show (if a = b then bs else b :: rm1 a bs) = b :: rm1 a bs
          rw [if_neg hab]
        rw [h2]
        have h3 := ih hbs
        calc (b :: rm1 a bs).length + 1
            = (rm1 a bs).length + 1 + 1 := rfl
          _ = bs.length + 1 := by rw [h3]
          _ = (b :: bs).length := by rw [List.length_cons]

/-- **Pigeonhole.** A duplicate-free list of elements of `l` is no longer
than `l`. -/
theorem length_le_of_nodup_sub {rows l : List ι} (hnd : rows.Nodup)
    (hsub : ∀ x ∈ rows, x ∈ l) : rows.length ≤ l.length := by
  induction rows generalizing l with
  | nil => exact Nat.zero_le _
  | cons a as ih =>
      have hndc := List.nodup_cons.mp hnd
      have habs : ∀ hm : a ∈ as, False := hndc.1
      have has : as.Nodup := hndc.2
      have ha : a ∈ l := hsub a (List.mem_cons_self _ _)
      have hsub' : ∀ x ∈ as, x ∈ rm1 a l := by
        intro x hx
        exact mem_rm1_of (hsub x (List.mem_cons_of_mem _ hx))
          (fun he => habs (by subst he; exact hx))
      have h1 := ih has hsub'
      have h2 := rm1_length a ha
      calc (a :: as).length = as.length + 1 := rfl
        _ ≤ (rm1 a l).length + 1 := Nat.succ_le_succ h1
        _ = l.length := h2

attribute [local instance] Classical.propDecidable

/-- Search a list for an element satisfying a (possibly undecidable)
predicate, classically. The development proves, it does not compute. -/
noncomputable def findIn (P : ι → Prop) : List ι → Option ι
  | [] => none
  | a :: as => if P a then some a else findIn P as

omit [DecidableEq ι] in
theorem findIn_some_spec (P : ι → Prop) :
    ∀ (l : List ι) (a : ι), findIn P l = some a → P a ∧ a ∈ l := by
  intro l
  induction l with
  | nil => intro a h; exact Option.noConfusion h
  | cons b bs ih =>
      intro a h
      by_cases hb : P b
      · have h' : findIn P (b :: bs) = some b := by
          show (if P b then some b else findIn P bs) = some b
          rw [if_pos hb]
        rw [h'] at h
        have hba : b = a := Option.some.inj h
        subst hba
        exact ⟨hb, List.mem_cons_self _ _⟩
      · have h' : findIn P (b :: bs) = findIn P bs := by
          show (if P b then some b else findIn P bs) = findIn P bs
          rw [if_neg hb]
        rw [h'] at h
        have := ih a h
        exact ⟨this.1, List.mem_cons_of_mem _ this.2⟩

omit [DecidableEq ι] in
theorem findIn_none_spec (P : ι → Prop) :
    ∀ (l : List ι), findIn P l = none → ∀ a ∈ l, ¬ P a := by
  intro l
  induction l with
  | nil => intro _ a ha; cases ha
  | cons b bs ih =>
      intro h a ha
      by_cases hb : P b
      · have h' : findIn P (b :: bs) = some b := by
          show (if P b then some b else findIn P bs) = some b
          rw [if_pos hb]
        rw [h'] at h
        exact Option.noConfusion h
      · have h' : findIn P (b :: bs) = findIn P bs := by
          show (if P b then some b else findIn P bs) = findIn P bs
          rw [if_neg hb]
        rw [h'] at h
        rcases List.mem_cons.mp ha with he | hm
        · subst he; exact hb
        · exact ih h a hm

omit [DecidableEq ι] in
theorem findIn_exists (P : ι → Prop) :
    ∀ (l : List ι), (∃ a, a ∈ l ∧ P a) → ∃ a, findIn P l = some a ∧ P a := by
  intro l
  induction l with
  | nil => intro h; rcases h with ⟨a, ha, _⟩; cases ha
  | cons b bs ih =>
      intro h
      rcases h with ⟨a, ha, hPa⟩
      by_cases hb : P b
      · refine ⟨b, ?_, hb⟩
        show (if P b then some b else findIn P bs) = some b
        rw [if_pos hb]
      · have h' : findIn P (b :: bs) = findIn P bs := by
          show (if P b then some b else findIn P bs) = findIn P bs
          rw [if_neg hb]
        have hbs : ∃ a, a ∈ bs ∧ P a := by
          rcases List.mem_cons.mp ha with he | hm
          · exact absurd (he ▸ hPa) hb
          · exact ⟨a, hm, hPa⟩
        rcases ih hbs with ⟨a', hsome, hP⟩
        refine ⟨a', ?_, hP⟩
        show (if P b then some b else findIn P bs) = some a'
        rw [if_neg hb]
        exact hsome

/-- All pairs from two lists, in a fixed order. -/
def lprod : List ι → List ι → List (ι × ι)
  | [], _ => []
  | a :: as, bs => bs.map (fun b => (a, b)) ++ lprod as bs

omit [DecidableEq ι] in
theorem mem_lprod {x : ι × ι} : ∀ {l₁ l₂ : List ι}, x ∈ lprod l₁ l₂ →
    x.1 ∈ l₁ ∧ x.2 ∈ l₂ := by
  intro l₁
  induction l₁ with
  | nil => intro l₂ h; cases h
  | cons a as ih =>
      intro l₂ h
      rcases List.mem_append.mp h with hm | hm
      · rcases List.mem_map.mp hm with ⟨b, hb, he⟩
        have hx1 : x.1 = a := congrArg Prod.fst he.symm
        have hx2 : x.2 = b := congrArg Prod.snd he.symm
        refine ⟨?_, ?_⟩
        · rw [hx1]; exact List.mem_cons_self _ _
        · rw [hx2]; exact hb
      · have hrec := ih hm
        exact ⟨List.mem_cons_of_mem _ hrec.1, hrec.2⟩

omit [DecidableEq ι] in
theorem lprod_mem {a b : ι} : ∀ {l₁ l₂ : List ι}, a ∈ l₁ → b ∈ l₂ →
    (a, b) ∈ lprod l₁ l₂ := by
  intro l₁
  induction l₁ with
  | nil => intro l₂ ha; cases ha
  | cons c cs ih =>
      intro l₂ ha hb
      rcases List.mem_cons.mp ha with he | hm
      · rw [he]
        exact List.mem_append_left _
          (List.mem_map_of_mem (fun d => (c, d)) hb)
      · exact List.mem_append_right _ (ih hm hb)

end Infra

/-! ## Plans, matchings, costs

The Kantorovich side of the equivalence: a *plan* is a fractional
transport on `idx₁ × idx₂` with unit row and column sums; a *matching* is
an integral one. Costs are carried by a ground cost `γ` on index pairs
(the point families `z, z'` enter only at the final theorem). -/

section Plans

variable {ι : Type v} [DecidableEq ι]

/-- A fractional transport plan: nonnegative entries, every row sum and
every column sum equal to one. -/
structure Plan (idx₁ idx₂ : List ι) (m : ι → ι → α) : Prop where
  nonneg : ∀ i ∈ idx₁, ∀ j ∈ idx₂, (0 : α) ≤ m i j
  row : ∀ i ∈ idx₁, sum (idx₂.map (fun j => m i j)) = 1
  col : ∀ j ∈ idx₂, sum (idx₁.map (fun i => m i j)) = 1

/-- The cost of a plan: the ground cost summed over all index pairs,
weighted by the plan. -/
def pcost (γ : ι → ι → α) (idx₁ idx₂ : List ι) (m : ι → ι → α) : α :=
  sum ((lprod idx₁ idx₂).map (fun p => m p.1 p.2 * γ p.1 p.2))

/-- An integral transport: an injective assignment of `idx₁` into `idx₂`. -/
structure Matching (idx₁ idx₂ : List ι) (μ : ι → ι) : Prop where
  maps : ∀ i ∈ idx₁, μ i ∈ idx₂
  inj : ∀ i ∈ idx₁, ∀ i' ∈ idx₁, μ i = μ i' → i = i'

/-- The cost of a matching. -/
def mcost (γ : ι → ι → α) (idx₁ : List ι) (μ : ι → ι) : α :=
  sum (idx₁.map (fun i => γ i (μ i)))

/-! ### Sum utilities -/

theorem sum_append' : ∀ (l₁ l₂ : List α), sum (l₁ ++ l₂) = sum l₁ + sum l₂ := by
  intro l₁
  induction l₁ with
  | nil => intro l₂; rw [List.nil_append, sum_nil, zero_add]
  | cons a as ih =>
      intro l₂
      rw [List.cons_append, sum_cons, sum_cons, ih, add_assoc]

omit [DecidableEq ι] in
theorem sum_zero_mem {f : ι → α} : ∀ (l : List ι), (∀ x ∈ l, f x = (0 : α)) →
    sum (l.map f) = (0 : α) := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a as ih =>
      intro h
      rw [List.map_cons, sum_cons, ih (fun x hx => h x (List.mem_cons_of_mem _ hx)),
        h a (List.mem_cons_self _ _), add_zero']

/-- Swapping two inner summands. -/
theorem add_swap_mid (x y z : α) : x + (y + z) = y + (x + z) := by
  rw [← add_assoc, add_comm x y, add_assoc]

theorem sum_rm1 {f : ι → α} (a : ι) : ∀ {l : List ι}, a ∈ l →
    sum (l.map f) = f a + sum ((rm1 a l).map f) := by
  intro l
  induction l with
  | nil => intro h; cases h
  | cons b bs ih =>
      intro h
      by_cases hab : a = b
      · subst hab
        have h2 : rm1 a (a :: bs) = bs := by
          show (if a = a then bs else a :: rm1 a bs) = bs
          rw [if_pos rfl]
        rw [h2, List.map_cons, sum_cons]
      · have hbs : a ∈ bs := by
          rcases List.mem_cons.mp h with he | hm
          · exact absurd he hab
          · exact hm
        have h2 : rm1 a (b :: bs) = b :: rm1 a bs := by
          show (if a = b then bs else b :: rm1 a bs) = b :: rm1 a bs
          rw [if_neg hab]
        rw [h2, List.map_cons, List.map_cons, sum_cons, sum_cons, ih hbs]
        rw [add_swap_mid (f b) (f a) (sum ((rm1 a bs).map f))]

theorem rm1_not_mem {a : ι} : ∀ {l : List ι}, l.Nodup → a ∈ l → a ∉ rm1 a l := by
  intro l
  induction l with
  | nil => intro _ h; cases h
  | cons b bs ih =>
      intro hnd hmem
      by_cases hab : a = b
      · subst hab
        have h2 : rm1 a (a :: bs) = bs := by
          show (if a = a then bs else a :: rm1 a bs) = bs
          rw [if_pos rfl]
        rw [h2]
        exact (List.nodup_cons.mp hnd).1
      · have h2 : rm1 a (b :: bs) = b :: rm1 a bs := by
          show (if a = b then bs else b :: rm1 a bs) = b :: rm1 a bs
          rw [if_neg hab]
        rw [h2]
        have hbs : a ∈ bs := by
          rcases List.mem_cons.mp hmem with he | hm
          · exact absurd he hab
          · exact hm
        have hndbs : bs.Nodup := (List.nodup_cons.mp hnd).2
        intro hmem'
        rcases List.mem_cons.mp hmem' with he | hm
        · exact absurd he hab
        · exact ih hndbs hbs hm

theorem sum_eq_single {f : ι → α} {a : ι} {l : List ι} (hmem : a ∈ l)
    (hnd : l.Nodup) (hzero : ∀ x ∈ l, x ≠ a → f x = (0 : α)) :
    sum (l.map f) = f a := by
  have hrest : sum ((rm1 a l).map f) = (0 : α) :=
    sum_zero_mem (rm1 a l) (fun x hx =>
      hzero x (mem_rm1 hx)
        (fun he => absurd (by subst he; exact hx) (rm1_not_mem hnd hmem)))
  rw [sum_rm1 a hmem, hrest, add_zero']

omit [DecidableEq ι] in
theorem sum_map_nonneg_mem {f : ι → α} : ∀ (l : List ι),
    (∀ x ∈ l, (0 : α) ≤ f x) → (0 : α) ≤ sum (l.map f) := by
  intro l
  induction l with
  | nil => intro _; exact le_refl _
  | cons a as ih =>
      intro h
      rw [List.map_cons, sum_cons]
      exact add_nonneg (h a (List.mem_cons_self _ _))
        (ih (fun y hy => h y (List.mem_cons_of_mem _ hy)))

omit [DecidableEq ι] in
theorem sum_pos_mem {f : ι → α} : ∀ (l : List ι), sum (l.map f) = (0 : α) →
    (∀ x ∈ l, (0 : α) ≤ f x) → ∀ x ∈ l, f x = (0 : α) := by
  intro l
  induction l with
  | nil => intro _ _ x hx; cases hx
  | cons a as ih =>
      intro hsum hnn x hx
      have hc : sum ((a :: as).map f) = f a + sum (as.map f) := by
        rw [List.map_cons, sum_cons]
      have hnn' : ∀ y ∈ as, (0 : α) ≤ f y :=
        fun y hy => hnn y (List.mem_cons_of_mem _ hy)
      have hsum' : f a + sum (as.map f) = (0 : α) := by rw [← hc]; exact hsum
      have hfa : f a = (0 : α) := by
        have hle : f a ≤ (0 : α) := by
          refine Classical.byContradiction (fun hcon => ?_)
          have hlt : (0 : α) < f a := lt_of_not_le hcon
          have h2 : (0 : α) < f a + sum (as.map f) :=
            add_pos_of_pos_of_nonneg hlt (sum_map_nonneg_mem as hnn')
          rw [hsum'] at h2
          exact lt_irrefl _ h2
        exact le_antisymm hle (hnn a (List.mem_cons_self _ _))
      have hrest : sum (as.map f) = (0 : α) := by
        rw [hfa, zero_add] at hsum'
        exact hsum'
      rcases List.mem_cons.mp hx with he | hm
      · exact he ▸ hfa
      · exact ih hrest hnn' x hm

theorem exists_min_list : ∀ (l : List α), l ≠ [] → ∃ x ∈ l, ∀ y ∈ l, x ≤ y := by
  intro l
  induction l with
  | nil => intro h; exact absurd rfl h
  | cons a as ih =>
      intro _
      by_cases has : as = []
      · subst has
        refine ⟨a, List.mem_cons_self _ _, fun y hy => ?_⟩
        cases List.mem_cons.mp hy with
        | inl he => exact le_of_eq he.symm
        | inr hcon => exact absurd hcon (List.not_mem_nil y)
      · rcases ih has with ⟨x, hxmem, hxmin⟩
        by_cases hle : a ≤ x
        · refine ⟨a, List.mem_cons_self _ _, fun y hy => ?_⟩
          rcases List.mem_cons.mp hy with he | hm
          · rw [he]; exact le_refl _
          · exact le_trans hle (hxmin y hm)
        · refine ⟨x, List.mem_cons_of_mem _ hxmem, fun y hy => ?_⟩
          rcases List.mem_cons.mp hy with he | hm
          · rw [he]; exact le_of_lt (lt_of_not_le hle)
          · exact hxmin y hm

/-- Negation distributes over addition (local; the division layer is not
imported here, and the ring facts live in `Order.lean`). -/
theorem neg_unique' {a u : α} (h : a + u = (0 : α)) : u = -a :=
  add_left_cancel (show a + u = a + -a from by rw [h]; exact (add_neg' a).symm)

theorem neg_add_dist' (x y : α) : -(x + y) = -x + -y := by
  have h0 : x + y + (-x + -y) = (0 : α) := by
    rw [add_assoc x y (-x + -y), add_comm (-x) (-y), ← add_assoc y (-y) (-x),
        add_neg', zero_add, add_neg']
  exact (neg_unique' h0).symm

/-- The four-term regrouping behind the difference of sums. -/
theorem regroup4 (a b c d : α) : (a + -c) + (b + -d) = a + b + (-c + -d) := by
  rw [add_assoc a (-c) (b + -d), ← add_assoc (-c) b (-d), add_comm (-c) b,
      add_assoc b (-c) (-d), ← add_assoc a b (-c + -d)]

omit [DecidableEq ι] in
/-- Difference of sums. -/
theorem sum_map_sub {F G : ι → α} : ∀ (L : List ι),
    sum (L.map (fun i => F i - G i)) = sum (L.map F) - sum (L.map G) := by
  intro L
  induction L with
  | nil => rw [List.map_nil, List.map_nil, List.map_nil, sum_nil, sub_zero]
  | cons a as ih =>
      have e1 : sum ((a :: as).map (fun i => F i - G i))
          = (F a - G a) + sum (as.map (fun i => F i - G i)) := rfl
      have e2 : sum ((a :: as).map F) = F a + sum (as.map F) := rfl
      have e3 : sum ((a :: as).map G) = G a + sum (as.map G) := rfl
      rw [e1, e2, e3, ih]
      rw [sub_eq, sub_eq, sub_eq, neg_add_dist']
      rw [regroup4 (F a) (sum (as.map F)) (G a) (sum (as.map G))]

end Plans

/-! ## The cycle signature

An alternating cycle in a plan's support is carried by two lists of pairs:
the **path** (the diagonal edges) and the **shift** (the other alternating
edges). The *signature* of a pair list is the sum of its point
indicators; the cycle signature is the difference of the two, and moving
mass ε along the cycle means adding ε times that difference to the plan.

Everything here is proved from the ordered-field axioms and list
inductions; no indexing into lists is used. -/

section Sig

variable {ι : Type v} [DecidableEq ι]

/-- The point indicator: `1` at `(r, c)`, `0` elsewhere. -/
def indD (r c i j : ι) : α := if (i = r ∧ j = c) then (1 : α) else (0 : α)

theorem indD_eq_zero {r c i j : ι} (h : ¬(i = r ∧ j = c)) : indD r c i j = (0 : α) :=
  if_neg h

theorem indD_eq_one {r c i j : ι} (h1 : i = r) (h2 : j = c) : indD r c i j = (1 : α) :=
  if_pos ⟨h1, h2⟩

/-- The signature of a pair list: `1` on its pairs, `0` elsewhere (for
duplicate-free lists). -/
def sigOf (Es : List (ι × ι)) (i j : ι) : α :=
  sum (Es.map (fun p => indD p.1 p.2 i j))

/-- The counting sum `Σ [i = x]` over a list. -/
def sumInd (i : ι) (l : List ι) : α := sum (l.map (fun x => if i = x then (1 : α) else (0 : α)))

theorem sumInd_cons (i a : ι) (l : List ι) :
    sumInd i (a :: l) = (if i = a then (1 : α) else (0 : α)) + sumInd i l := by
  show sum ((a :: l).map _) = _ + _
  rw [List.map_cons, sum_cons]
  rfl

theorem sumInd_snoc (i a : ι) : ∀ (l : List ι),
    sumInd i (l ++ [a]) = (if i = a then (1 : α) else (0 : α)) + sumInd i l := by
  intro l
  induction l with
  | nil => rw [List.nil_append]; exact sumInd_cons i a []
  | cons b bs ih =>
      show (if i = b then (1 : α) else (0 : α)) + sumInd i (bs ++ [a])
          = (if i = a then (1 : α) else (0 : α))
            + ((if i = b then (1 : α) else (0 : α)) + sumInd i bs)
      rw [ih]
      exact add_swap_mid _ _ _

/-- Rotate a nonempty list's head to its end. -/
def rot1 : List ι → List ι
  | [] => []
  | a :: as => as ++ [a]

theorem sumInd_rotate (i : ι) : ∀ (l : List ι),
    (sumInd i (rot1 l) : α) = sumInd i l := by
  intro l
  induction l with
  | nil => rfl
  | cons a as ih =>
      show sumInd i (as ++ [a]) = sumInd i (a :: as)
      calc sumInd i (as ++ [a])
          = (if i = a then (1 : α) else (0 : α)) + sumInd i as := sumInd_snoc i a as
        _ = sumInd i (a :: as) := (sumInd_cons i a as).symm

theorem sumInd_append (i : ι) : ∀ (l₁ l₂ : List ι),
    sumInd i (l₁ ++ l₂) = (sumInd i l₁ : α) + sumInd i l₂ := by
  intro l₁
  induction l₁ with
  | nil =>
      intro l₂
      have h0 : sumInd i ([] : List ι) = (0 : α) := by
        show sum (([] : List ι).map _) = (0 : α)
        rfl
      rw [List.nil_append, h0]
      exact (zero_add (sumInd i l₂)).symm
  | cons a as ih =>
      intro l₂
      have h1 : sumInd i ((a :: as) ++ l₂)
          = (if i = a then (1 : α) else (0 : α)) + sumInd i (as ++ l₂) := by
        rw [List.cons_append]
        exact sumInd_cons i a (as ++ l₂)
      have h2 : sumInd i (as ++ l₂) = (sumInd i as : α) + sumInd i l₂ := ih l₂
      calc sumInd i ((a :: as) ++ l₂)
          = (if i = a then (1 : α) else (0 : α)) + sumInd i (as ++ l₂) := h1
        _ = (if i = a then (1 : α) else (0 : α)) + ((sumInd i as : α) + sumInd i l₂) := by
            rw [h2]
        _ = ((if i = a then (1 : α) else (0 : α)) + sumInd i as) + sumInd i l₂ := by
            rw [add_assoc]
        _ = sumInd i (a :: as) + sumInd i l₂ :=
            (congrArg (fun t => t + sumInd i l₂) (sumInd_cons i a as)).symm

omit [DecidableEq ι] in
/-- Fubini for two list sums. -/
theorem sum_swap : ∀ (L₁ : List ι) (L₂ : List ι) (h : ι → ι → α),
    sum (L₁.map (fun x => sum (L₂.map (h x))))
      = sum (L₂.map (fun y => sum (L₁.map (fun x => h x y)))) := by
  intro L₁
  induction L₁ with
  | nil =>
      intro L₂ h
      show (0 : α) = sum (L₂.map (fun y => sum (([] : List ι).map (fun x => h x y))))
      rw [ExpectationOps.sum_map_ext (F := fun y => sum (([] : List ι).map (fun x => h x y)))
        (G := fun _ => (0 : α)) (fun y _ => rfl)]
      exact (ExpectationOps.sum_map_zero L₂).symm
  | cons a as ih =>
      intro L₂ h
      have e2 := ih L₂ h
      calc sum ((a :: as).map (fun x => sum (L₂.map (h x))))
          = sum (L₂.map (h a)) + sum (as.map (fun x => sum (L₂.map (h x)))) := rfl
        _ = sum (L₂.map (h a)) + sum (L₂.map (fun y => sum (as.map (fun x => h x y)))) := by
              rw [e2]
        _ = sum (L₂.map (fun y => h a y + sum (as.map (fun x => h x y)))) :=
              (ExpectationOps.sum_map_add L₂).symm
        _ = sum (L₂.map (fun y => sum ((a :: as).map (fun x => h x y)))) := rfl



end Sig

/-! ## The integral-optimum statement, and the reduction

The statement `HasIntegralOptimum` is the combinatorial core of boundary
theorem 3: every fractional plan is dominated in cost by a matching. In the
zero-dependency development it remains an *assumption* of the reduction
(the walk-shift-peel recursion is not formalised here). Over the real
numbers it is a *theorem*: `bridge/BridgeKantorovich.lean` proves
`hasIntegralOptimum_real` for every pair of duplicate-free index lists of
equal length, from Mathlib's Birkhoff--von Neumann theorem, and the
reduction below then yields the value equivalence over `ℝ`. -/

section Reduction

variable {ι : Type v} [DecidableEq ι]

/-- **The integral-optimum property** of a ground cost on a pair of index
lists: every fractional plan is dominated in cost by some matching. -/
def HasIntegralOptimum (γ : ι → ι → α) (idx₁ idx₂ : List ι) : Prop :=
  ∀ m : ι → ι → α, Plan idx₁ idx₂ m →
    ∃ μ : ι → ι, Matching idx₁ idx₂ μ ∧ mcost γ idx₁ μ ≤ pcost γ idx₁ idx₂ m

omit [DecidableEq ι] in
/-- **The reduction.** Under the integral-optimum property, a
minimum-cost matching lower-bounds the cost of every fractional plan: the
Kantorovich value is attained by the assignment the Hungarian algorithm
computes. -/
theorem assignment_le_plan_of_integral_optimum {γ : ι → ι → α}
    {idx₁ idx₂ : List ι} (hIO : HasIntegralOptimum γ idx₁ idx₂)
    {μ₀ : ι → ι}
    (hmin : ∀ μ : ι → ι, Matching idx₁ idx₂ μ → mcost γ idx₁ μ₀ ≤ mcost γ idx₁ μ)
    (m : ι → ι → α) (hm : Plan idx₁ idx₂ m) :
    mcost γ idx₁ μ₀ ≤ pcost γ idx₁ idx₂ m := by
  obtain ⟨μ, hμ, hle⟩ := hIO m hm
  exact le_trans (hmin μ hμ) hle

end Reduction

end Kantorovich
end Certgen
