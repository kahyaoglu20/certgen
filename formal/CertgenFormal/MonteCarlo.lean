import CertgenFormal.DivField

/-
# Monte-Carlo p-values: the Davison–Hinkley correction, proved valid

Every calibrated statement in the certgen project -- the two one-sided floor
tests, the curvature gate's null, the severity thresholds -- rests on one
combinatorial fact: the Monte-Carlo p-value with the (count+1)/(n+1)
correction is *valid*: under the exchangeable null, its probability of being
at most `t` is at most `t`. This file proves that fact in the pooled form the
code computes, from first principles, with no probability theory beyond the
counting itself.

The argument is the classical one, in four moves:

1. Sort the pooled list (null draws plus the observation) in the relevant
   order; the sort is a permutation, so every count is unchanged.
2. In a descending-sorted list, the element at index `j` has at least `j+1`
   elements at or above it -- a rank bound.
3. So the elements whose "at-or-above" count is at most `k` can only occupy
   the first `k` positions, and there are at most `k` of them, counted with
   multiplicity.
4. Because the p-value's numerator is an integer count, `p ≤ t` is an integer
   inequality whose solution set is `{0, …, k}` for a `k` supplied by the
   bounded floor lemma of `DivField.lean` -- proved by induction, with no
   archimedean axiom, since the count is bounded by an embedded natural.

Exported: `mc_upper_valid` / `mc_lower_valid` (sub-uniformity of both tails),
`pUp_pool_code` (the pooled form is exactly the code's `(count+1)/(n+1)`),
`pUpCode_min` and `n_rep_floor_19` (the significance floor: `n_rep ≥ 19`
before `p ≤ 0.05` is even attainable -- the budget line the paper says most
evaluations do not know they are spending).
-/

namespace Certgen

namespace MC

open LinOrdField LinOrdFieldDiv

universe u

variable {α : Type u} [LinOrdFieldDiv α]

/-! ## Counting with a classical predicate

Count the elements of a list satisfying a (classically decided) predicate.
The definition is noncomputable; nothing in this development computes. -/
open Classical in
noncomputable def cntP (p : α → Prop) : List α → Nat
  | [] => 0
  | a :: as => (if p a then 1 else 0) + cntP p as

open Classical in
omit [LinOrdFieldDiv α] in
theorem cntP_cons (p : α → Prop) (a : α) (as : List α) :
    cntP p (a :: as) = (if p a then 1 else 0) + cntP p as := rfl

omit [LinOrdFieldDiv α] in
theorem cntP_ext {p q : α → Prop}
    (h : ∀ x, (p x ↔ q x)) : ∀ (L : List α), cntP p L = cntP q L := by
  intro L
  induction L with
  | nil => rfl
  | cons a as ih =>
      rw [cntP_cons, cntP_cons]
      by_cases hp : p a
      · rw [if_pos hp, if_pos ((h a).mp hp), ih]
      · rw [if_neg hp, if_neg (fun hq => hp ((h a).mpr hq)), ih]

open Classical in
omit [LinOrdFieldDiv α] in
theorem cntP_le_length (p : α → Prop) : ∀ (L : List α), cntP p L ≤ L.length := by
  intro L
  induction L with
  | nil => exact Nat.le_refl 0
  | cons a as ih =>
      rw [cntP_cons, List.length_cons]
      have hone : (if p a then 1 else 0) ≤ 1 := by
        by_cases hp : p a
        · rw [if_pos hp]; exact Nat.le_refl 1
        · rw [if_neg hp]; exact Nat.zero_le 1
      omega

omit [LinOrdFieldDiv α] in
theorem cntP_append (p : α → Prop) : ∀ (A B : List α),
    cntP p (A ++ B) = cntP p A + cntP p B := by
  intro A B
  induction A with
  | nil =>
      rw [List.nil_append]
      show cntP p B = (0 : Nat) + cntP p B
      rw [Nat.zero_add]
  | cons a as ih =>
      rw [List.cons_append, cntP_cons, cntP_cons, ih, Nat.add_assoc]

omit [LinOrdFieldDiv α] in
theorem cntP_zero_of_fail {p : α → Prop} : ∀ {L : List α},
    (∀ v ∈ L, ¬ p v) → cntP p L = 0 := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons a as ih =>
      intro h
      rw [cntP_cons, ih (fun v hv => h v (List.mem_cons_of_mem a hv)),
          if_neg (h a (List.mem_cons_self a as)), Nat.zero_add]

omit [LinOrdFieldDiv α] in
theorem cntP_ge_of_mem {p : α → Prop} : ∀ {x : α} {L : List α},
    x ∈ L → p x → 1 ≤ cntP p L := by
  intro x L
  induction L with
  | nil => intro hmem; cases hmem
  | cons a as ih =>
      intro hmem hp
      rw [cntP_cons]
      cases List.mem_cons.1 hmem with
      | inl heq =>
          have hpa : p a := by subst heq; exact hp
          rw [if_pos hpa]
          omega
      | inr hin =>
          have h2 := ih hin hp
          by_cases hpa : p a
          · rw [if_pos hpa]; omega
          · rw [if_neg hpa]; omega

/-! ## The two tail counts -/

/-- The number of elements of `L` at or above `x` (upper-tail count). -/
noncomputable def cntGe (L : List α) (x : α) : Nat := cntP (fun w => x ≤ w) L

/-- The number of elements of `L` at or below `x` (lower-tail count). -/
noncomputable def cntLb (L : List α) (x : α) : Nat := cntP (fun w => w ≤ x) L

theorem cntGe_append (A B : List α) (x : α) :
    cntGe (A ++ B) x = cntGe A x + cntGe B x := cntP_append _ A B

theorem cntLb_append (A B : List α) (x : α) :
    cntLb (A ++ B) x = cntLb A x + cntLb B x := cntP_append _ A B

/-! ## Index access and sortedness -/

/-- The `j`-th element of a list, 0-based; junk outside the range. The junk
value is never read through: every use carries `j < L.length`. -/
def atIdx : List α → Nat → α
  | [], _ => (0 : α)
  | a :: _, 0 => a
  | _ :: as, j + 1 => atIdx as j

theorem atIdx_mem : ∀ (L : List α) (j : Nat), j < L.length → atIdx L j ∈ L := by
  intro L j
  induction L generalizing j with
  | nil => intro h; exact absurd h (Nat.not_lt_zero _)
  | cons a as ih =>
      intro h
      cases j with
      | zero => exact List.mem_cons_self _ _
      | succ j' =>
          show atIdx as j' ∈ a :: as
          exact List.mem_cons_of_mem _ (ih j' (by
            have hlen : (a :: as).length = as.length + 1 := rfl
            omega))

theorem mem_ex_atIdx : ∀ {L : List α} {v : α}, v ∈ L →
    ∃ j, j < L.length ∧ atIdx L j = v := by
  intro L v hmem
  induction L with
  | nil => cases hmem
  | cons a as ih =>
      cases List.mem_cons.1 hmem with
      | inl heq =>
          refine ⟨0, Nat.succ_le_succ (Nat.zero_le _), ?_⟩
          exact heq ▸ rfl
      | inr hin =>
          have ⟨j, hj1, hj2⟩ := ih hin
          refine ⟨j + 1, Nat.succ_lt_succ hj1, ?_⟩
          exact hj2 ▸ rfl

/-- Ascending sortedness. -/
inductive Asc : List α → Prop
  | nil : Asc []
  | cons (a : α) (L : List α) (h : ∀ b ∈ L, a ≤ b) (hL : Asc L) : Asc (a :: L)

/-- Descending sortedness. -/
inductive Desc : List α → Prop
  | nil : Desc []
  | cons (a : α) (L : List α) (h : ∀ b ∈ L, b ≤ a) (hL : Desc L) : Desc (a :: L)

theorem asc_head {a : α} {as : List α} (h : Asc (a :: as)) : ∀ b ∈ as, a ≤ b := by
  cases h with | cons _ _ h _ => exact h

theorem asc_tail {a : α} {as : List α} (h : Asc (a :: as)) : Asc as := by
  cases h with | cons _ _ _ hL => exact hL

theorem desc_head {a : α} {as : List α} (h : Desc (a :: as)) : ∀ b ∈ as, b ≤ a := by
  cases h with | cons _ _ h _ => exact h

theorem desc_tail {a : α} {as : List α} (h : Desc (a :: as)) : Desc as := by
  cases h with | cons _ _ _ hL => exact hL

theorem asc_atIdx_mono : ∀ (L : List α) (i j : Nat), Asc L → i ≤ j →
    j < L.length → atIdx L i ≤ atIdx L j := by
  intro L i j hAsc
  induction L generalizing i j with
  | nil => intro _ h2; exact absurd h2 (Nat.not_lt_zero _)
  | cons a as ih =>
      intro hij hjlen
      have hlen : (a :: as).length = as.length + 1 := rfl
      cases j with
      | zero =>
          have hi0 : i = 0 := Nat.le_zero.mp hij
          exact hi0 ▸ le_refl _
      | succ j' =>
          have hjaslen : j' < as.length := by omega
          cases i with
          | zero =>
              show a ≤ atIdx as j'
              exact asc_head hAsc (atIdx as j') (atIdx_mem as j' hjaslen)
          | succ i' =>
              show atIdx as i' ≤ atIdx as j'
              exact ih i' j' (asc_tail hAsc) (by omega) hjaslen

theorem desc_atIdx_antimono : ∀ (L : List α) (i j : Nat), Desc L → i ≤ j →
    j < L.length → atIdx L j ≤ atIdx L i := by
  intro L i j hDesc
  induction L generalizing i j with
  | nil => intro _ h2; exact absurd h2 (Nat.not_lt_zero _)
  | cons a as ih =>
      intro hij hjlen
      have hlen : (a :: as).length = as.length + 1 := rfl
      cases j with
      | zero =>
          have hi0 : i = 0 := Nat.le_zero.mp hij
          exact hi0 ▸ le_refl _
      | succ j' =>
          have hjaslen : j' < as.length := by omega
          cases i with
          | zero =>
              show atIdx as j' ≤ a
              exact desc_head hDesc (atIdx as j') (atIdx_mem as j' hjaslen)
          | succ i' =>
              show atIdx as j' ≤ atIdx as i'
              exact ih i' j' (desc_tail hDesc) (by omega) hjaslen

/-! ## Insertion sort, as a permutation -/

-- Insert into an ascending-sorted list.
open Classical in
noncomputable def insertAsc (a : α) : List α → List α
  | [] => [a]
  | b :: bs => if a ≤ b then a :: b :: bs else b :: insertAsc a bs

-- Insertion sort, ascending.
open Classical in
noncomputable def sortAsc : List α → List α
  | [] => []
  | a :: as => insertAsc a (sortAsc as)

-- Insert into a descending-sorted list.
open Classical in
noncomputable def insertDesc (a : α) : List α → List α
  | [] => [a]
  | b :: bs => if b ≤ a then a :: b :: bs else b :: insertDesc a bs

-- Insertion sort, descending.
open Classical in
noncomputable def sortDesc : List α → List α
  | [] => []
  | a :: as => insertDesc a (sortDesc as)

open Classical in
theorem insertAsc_eq (a b : α) (bs : List α) :
    insertAsc a (b :: bs) = if a ≤ b then a :: b :: bs else b :: insertAsc a bs :=
  rfl

open Classical in
theorem insertDesc_eq (a b : α) (bs : List α) :
    insertDesc a (b :: bs) = if b ≤ a then a :: b :: bs else b :: insertDesc a bs :=
  rfl

open Classical in
theorem cntP_insertAsc (p : α → Prop) (a : α) : ∀ (L : List α),
    cntP p (insertAsc a L) = cntP p L + (if p a then 1 else 0) := by
  intro L
  induction L with
  | nil =>
      show (if p a then 1 else 0) + cntP p [] = cntP p [] + (if p a then 1 else 0)
      omega
  | cons b bs ih =>
      rw [insertAsc_eq]
      by_cases hab : a ≤ b
      · rw [if_pos hab, cntP_cons, cntP_cons]
        omega
      · rw [if_neg hab, cntP_cons, cntP_cons, ih]
        omega

theorem cntP_sortAsc (p : α → Prop) : ∀ (L : List α),
    cntP p (sortAsc L) = cntP p L := by
  intro L
  induction L with
  | nil => rfl
  | cons a as ih =>
      show cntP p (insertAsc a (sortAsc as)) = cntP p (a :: as)
      rw [cntP_insertAsc, ih, cntP_cons]
      omega

open Classical in
theorem cntP_insertDesc (p : α → Prop) (a : α) : ∀ (L : List α),
    cntP p (insertDesc a L) = cntP p L + (if p a then 1 else 0) := by
  intro L
  induction L with
  | nil =>
      show (if p a then 1 else 0) + cntP p [] = cntP p [] + (if p a then 1 else 0)
      omega
  | cons b bs ih =>
      rw [insertDesc_eq]
      by_cases hba : b ≤ a
      · rw [if_pos hba, cntP_cons, cntP_cons]
        omega
      · rw [if_neg hba, cntP_cons, cntP_cons, ih]
        omega

theorem cntP_sortDesc (p : α → Prop) : ∀ (L : List α),
    cntP p (sortDesc L) = cntP p L := by
  intro L
  induction L with
  | nil => rfl
  | cons a as ih =>
      show cntP p (insertDesc a (sortDesc as)) = cntP p (a :: as)
      rw [cntP_insertDesc, ih, cntP_cons]
      omega

/-! ## Sortedness of the sort -/

theorem mem_of_mem_insertAsc : ∀ {a x : α} {L : List α}, x ∈ insertAsc a L →
    x = a ∨ x ∈ L := by
  intro a x L
  induction L with
  | nil =>
      intro hmem
      cases List.mem_cons.1 hmem with
      | inl heq => exact Or.inl heq
      | inr hmem' => cases hmem'
  | cons b bs ih =>
      intro hmem
      rw [insertAsc_eq] at hmem
      by_cases hab : a ≤ b
      · rw [if_pos hab] at hmem
        cases List.mem_cons.1 hmem with
        | inl heq => exact Or.inl heq
        | inr hmem' =>
            cases List.mem_cons.1 hmem' with
            | inl heq => exact Or.inr (heq ▸ List.mem_cons_self _ _)
            | inr hmem'' => exact Or.inr (List.mem_cons_of_mem _ hmem'')
      · rw [if_neg hab] at hmem
        cases List.mem_cons.1 hmem with
        | inl heq => exact Or.inr (heq ▸ List.mem_cons_self _ _)
        | inr hmem' =>
            cases ih hmem' with
            | inl heq => exact Or.inl heq
            | inr hmem'' => exact Or.inr (List.mem_cons_of_mem _ hmem'')

theorem asc_insertAsc : ∀ (a : α) (L : List α), Asc L → Asc (insertAsc a L) := by
  intro a L hAsc
  induction L with
  | nil => exact Asc.cons a [] (fun v hv => absurd hv (List.not_mem_nil v)) Asc.nil
  | cons b bs ih =>
      rw [insertAsc_eq]
      have hbdb := asc_head hAsc
      have hbs := asc_tail hAsc
      by_cases hab : a ≤ b
      · rw [if_pos hab]
        refine Asc.cons a (b :: bs) (fun v hv => ?_) hAsc
        by_cases heq : v = b
        · rw [heq]; exact hab
        · exact le_trans hab (hbdb v (by
            cases List.mem_cons.1 hv with
            | inl heq2 => exact absurd heq2 heq
            | inr hmem2 => exact hmem2))
      · rw [if_neg hab]
        have hba : b ≤ a := le_of_lt (lt_of_not_le hab)
        refine Asc.cons b (insertAsc a bs) (fun v hv => ?_) (ih hbs)
        cases mem_of_mem_insertAsc hv with
        | inl heq => rw [heq]; exact hba
        | inr hmem => exact hbdb v hmem

theorem asc_sortAsc : ∀ (L : List α), Asc (sortAsc L) := by
  intro L
  induction L with
  | nil => exact Asc.nil
  | cons a as ih => exact asc_insertAsc a (sortAsc as) ih

theorem mem_of_mem_insertDesc : ∀ {a x : α} {L : List α}, x ∈ insertDesc a L →
    x = a ∨ x ∈ L := by
  intro a x L
  induction L with
  | nil =>
      intro hmem
      cases List.mem_cons.1 hmem with
      | inl heq => exact Or.inl heq
      | inr hmem' => cases hmem'
  | cons b bs ih =>
      intro hmem
      rw [insertDesc_eq] at hmem
      by_cases hba : b ≤ a
      · rw [if_pos hba] at hmem
        cases List.mem_cons.1 hmem with
        | inl heq => exact Or.inl heq
        | inr hmem' =>
            cases List.mem_cons.1 hmem' with
            | inl heq => exact Or.inr (heq ▸ List.mem_cons_self _ _)
            | inr hmem'' => exact Or.inr (List.mem_cons_of_mem _ hmem'')
      · rw [if_neg hba] at hmem
        cases List.mem_cons.1 hmem with
        | inl heq => exact Or.inr (heq ▸ List.mem_cons_self _ _)
        | inr hmem' =>
            cases ih hmem' with
            | inl heq => exact Or.inl heq
            | inr hmem'' => exact Or.inr (List.mem_cons_of_mem _ hmem'')

theorem desc_insertDesc : ∀ (a : α) (L : List α), Desc L → Desc (insertDesc a L) := by
  intro a L hDesc
  induction L with
  | nil => exact Desc.cons a [] (fun v hv => absurd hv (List.not_mem_nil v)) Desc.nil
  | cons b bs ih =>
      rw [insertDesc_eq]
      have hbdb := desc_head hDesc
      have hbs := desc_tail hDesc
      by_cases hba : b ≤ a
      · rw [if_pos hba]
        refine Desc.cons a (b :: bs) (fun v hv => ?_) hDesc
        by_cases heq : v = b
        · rw [heq]; exact hba
        · exact le_trans (hbdb v (by
            cases List.mem_cons.1 hv with
            | inl heq2 => exact absurd heq2 heq
            | inr hmem2 => exact hmem2)) hba
      · rw [if_neg hba]
        have hab : a ≤ b := le_of_lt (lt_of_not_le hba)
        refine Desc.cons b (insertDesc a bs) (fun v hv => ?_) (ih hbs)
        cases mem_of_mem_insertDesc hv with
        | inl heq => rw [heq]; exact hab
        | inr hmem => exact hbdb v hmem

theorem desc_sortDesc : ∀ (L : List α), Desc (sortDesc L) := by
  intro L
  induction L with
  | nil => exact Desc.nil
  | cons a as ih => exact desc_insertDesc a (sortDesc as) ih


/-! ## Splitting a list at a position -/

omit [LinOrdFieldDiv α] in
theorem takeAppDrop : ∀ (L : List α) (k : Nat),
    List.take k L ++ List.drop k L = L := by
  intro L k
  induction k generalizing L with
  | zero => rfl
  | succ k ih =>
      cases L with
      | nil => rfl
      | cons a as =>
          show (a :: List.take k as) ++ List.drop k as = a :: as
          rw [List.cons_append, ih as]

open Classical in
omit [LinOrdFieldDiv α] in
theorem cntP_take_le (p : α → Prop) : ∀ (L : List α) (k : Nat),
    cntP p (List.take k L) ≤ k := by
  intro L k
  induction k generalizing L with
  | zero =>
      show (0 : Nat) ≤ 0
      exact Nat.le_refl _
  | succ k' ih =>
      cases L with
      | nil => exact Nat.zero_le _
      | cons a as =>
          show (if p a then 1 else 0) + cntP p (List.take k' as) ≤ k' + 1
          have h1 : (if p a then 1 else 0) ≤ 1 := by
            by_cases hp : p a
            · rw [if_pos hp]; omega
            · rw [if_neg hp]; omega
          have h2 := ih as
          omega

theorem mem_drop_ex_atIdx : ∀ (L : List α) (k : Nat) {v : α},
    v ∈ List.drop k L → ∃ j, atIdx L (k + j) = v ∧ k + j < L.length := by
  intro L k
  induction L generalizing k with
  | nil =>
      intro v hmem
      cases k with
      | zero => cases hmem
      | succ k' => cases hmem
  | cons a as ih =>
      intro v hmem
      cases k with
      | zero =>
          have ⟨j, hj1, hj2⟩ := mem_ex_atIdx hmem
          refine ⟨j, ?_, ?_⟩
          · show atIdx (a :: as) (0 + j) = v
            rw [Nat.zero_add]
            exact hj2
          · show 0 + j < (a :: as).length
            rw [Nat.zero_add]
            exact hj1
      | succ k' =>
          have hmem' : v ∈ List.drop k' as := hmem
          have ⟨j, hj1, hj2⟩ := ih k' hmem'
          have hsum : k' + 1 + j = (k' + j) + 1 := by omega
          have hlen : (a :: as).length = as.length + 1 := rfl
          refine ⟨j, ?_, ?_⟩
          · show atIdx (a :: as) (k' + 1 + j) = v
            rw [hsum]
            exact hj1
          · show k' + 1 + j < (a :: as).length
            omega

/-! ## Rank bounds -/

open Classical in
theorem desc_cntGe_prefix : ∀ (L : List α) (j : Nat), Desc L → j < L.length →
    j + 1 ≤ cntGe L (atIdx L j) := by
  intro L j hDesc
  induction L generalizing j with
  | nil => intro h2; exact absurd h2 (Nat.not_lt_zero _)
  | cons a as ih =>
      intro hjlen
      have hlen : (a :: as).length = as.length + 1 := rfl
      have hd := desc_head hDesc
      have hdt := desc_tail hDesc
      cases j with
      | zero =>
          show 1 ≤ (if a ≤ a then 1 else 0) + cntGe as a
          rw [if_pos (le_refl a)]
          omega
      | succ j' =>
          show j' + 1 + 1 ≤ (if atIdx as j' ≤ a then 1 else 0) + cntGe as (atIdx as j')
          have hhead : atIdx as j' ≤ a := hd (atIdx as j') (atIdx_mem as j' (by omega))
          rw [if_pos hhead]
          have h2 := ih j' hdt (by omega)
          omega

open Classical in
theorem asc_cntLb_prefix : ∀ (L : List α) (j : Nat), Asc L → j < L.length →
    j + 1 ≤ cntLb L (atIdx L j) := by
  intro L j hAsc
  induction L generalizing j with
  | nil => intro h2; exact absurd h2 (Nat.not_lt_zero _)
  | cons a as ih =>
      intro hjlen
      have hlen : (a :: as).length = as.length + 1 := rfl
      have ha := asc_head hAsc
      have hat := asc_tail hAsc
      cases j with
      | zero =>
          show 1 ≤ (if a ≤ a then 1 else 0) + cntLb as a
          rw [if_pos (le_refl a)]
          omega
      | succ j' =>
          show j' + 1 + 1 ≤ (if a ≤ atIdx as j' then 1 else 0) + cntLb as (atIdx as j')
          have hhead : a ≤ atIdx as j' := ha (atIdx as j') (atIdx_mem as j' (by omega))
          rw [if_pos hhead]
          have h2 := ih j' hat (by omega)
          omega

/-! ## The main counting bound -/

theorem cnt_rank_bound_desc : ∀ (L : List α) (k : Nat), Desc L →
    cntP (fun v => (toα (cntGe L v) : α) ≤ toα k) L ≤ k := by
  intro L k hDesc
  have hfail : ∀ v ∈ List.drop k L, ¬ ((toα (cntGe L v) : α) ≤ toα k) := by
    intro v hv
    have ⟨j, hj1, hj2⟩ := mem_drop_ex_atIdx L k hv
    intro hle
    have hprefix := desc_cntGe_prefix L (k + j) hDesc hj2
    rw [hj1] at hprefix
    have hcount : cntGe L v ≤ k := toα_le_iff.mp hle
    omega
  have hsplit : cntP (fun v => (toα (cntGe L v) : α) ≤ toα k) L
      = cntP (fun v => (toα (cntGe L v) : α) ≤ toα k) (List.take k L ++ List.drop k L) := by
    rw [takeAppDrop L k]
  rw [hsplit, cntP_append]
  have hdrop0 : cntP (fun v => (toα (cntGe L v) : α) ≤ toα k) (List.drop k L) = 0 :=
    cntP_zero_of_fail hfail
  rw [hdrop0, Nat.add_zero]
  exact cntP_take_le _ L k

theorem cnt_rank_bound_asc : ∀ (L : List α) (k : Nat), Asc L →
    cntP (fun v => (toα (cntLb L v) : α) ≤ toα k) L ≤ k := by
  intro L k hAsc
  have hfail : ∀ v ∈ List.drop k L, ¬ ((toα (cntLb L v) : α) ≤ toα k) := by
    intro v hv
    have ⟨j, hj1, hj2⟩ := mem_drop_ex_atIdx L k hv
    intro hle
    have hprefix := asc_cntLb_prefix L (k + j) hAsc hj2
    rw [hj1] at hprefix
    have hcount : cntLb L v ≤ k := toα_le_iff.mp hle
    omega
  have hsplit : cntP (fun v => (toα (cntLb L v) : α) ≤ toα k) L
      = cntP (fun v => (toα (cntLb L v) : α) ≤ toα k) (List.take k L ++ List.drop k L) := by
    rw [takeAppDrop L k]
  rw [hsplit, cntP_append]
  have hdrop0 : cntP (fun v => (toα (cntLb L v) : α) ≤ toα k) (List.drop k L) = 0 :=
    cntP_zero_of_fail hfail
  rw [hdrop0, Nat.add_zero]
  exact cntP_take_le _ L k

/-! ## The pooled p-values and their validity -/

/-- The pooled upper-tail p-value: the fraction of the pool at or above `x`.
For the pool `nulls ++ [obs]` this is exactly the code's
`(count of nulls at or above obs + 1) / (n_rep + 1)`; see `pUp_pool_code`. -/
noncomputable def pUpPool (V : List α) (x : α) : α :=
  toα (cntGe V x) / toα (V.length)

/-- The pooled lower-tail p-value: the fraction of the pool at or below `x`. -/
noncomputable def pLoPool (V : List α) (x : α) : α :=
  toα (cntLb V x) / toα (V.length)

/-- **Validity of the upper-tail Monte-Carlo p-value.**

Reading: if the observation occupies a uniformly random position of the
pooled list `V` -- the consequence of exchangeability under the null, and
the modelling hypothesis of the Davison–Hinkley construction -- then the
probability that the p-value is at most `t` is at most `t`. The probability
is the count of qualifying positions over the pool length, computed here as
an exact ratio. -/
theorem mc_upper_valid (V : List α) (t : α) (h0 : (0 : α) ≤ t) (h1 : t ≤ (1 : α))
    (hm : 0 < V.length) :
    toα (cntP (fun v => pUpPool V v ≤ t) V) / toα (V.length) ≤ t := by
  have hmα : (0 : α) < toα (V.length) := toα_strict hm
  have hr : (0 : α) ≤ t * toα (V.length) := mul_nonneg h0 (toα_nonneg _)
  have hrb : t * toα (V.length) < toα (V.length + 1) := by
    have h2 : t * toα (V.length) ≤ (1 : α) * toα (V.length) :=
      mul_le_mul_of_nonneg_right h1 (toα_nonneg _)
    rw [one_mul] at h2
    exact lt_of_le_of_lt h2 (toα_strict (Nat.lt_succ_self V.length))
  have ⟨k, hk1, hk2⟩ := floor_bounded (t * toα (V.length)) (V.length + 1) hr hrb
  have heq1 : ∀ v : α, (pUpPool V v ≤ t ↔ toα (cntGe V v) ≤ t * toα (V.length)) := by
    intro v
    exact div_le_iff hmα
  have heq2 : ∀ v : α,
      (toα (cntGe V v) ≤ t * toα (V.length) ↔ cntGe V v ≤ k) := by
    intro v
    constructor
    · intro h
      have hlt : toα (cntGe V v) < toα (k + 1) := lt_of_le_of_lt h hk2
      have hlt2 : cntGe V v < k + 1 := toα_lt_iff.mp hlt
      omega
    · intro h
      exact le_trans (toα_mono h) hk1
  have hperm : ∀ v : α, cntGe (sortDesc V) v = cntGe V v := fun v => cntP_sortDesc _ V
  have hcount : cntP (fun v => cntGe V v ≤ k) V ≤ k := by
    have s1 : cntP (fun v => cntGe V v ≤ k) V
        = cntP (fun v => cntGe V v ≤ k) (sortDesc V) := (cntP_sortDesc _ V).symm
    have s2 : cntP (fun v => cntGe V v ≤ k) (sortDesc V)
        = cntP (fun v => cntGe (sortDesc V) v ≤ k) (sortDesc V) :=
      cntP_ext (fun v => by rw [hperm v]) (sortDesc V)
    have s3 : cntP (fun v => cntGe (sortDesc V) v ≤ k) (sortDesc V)
        = cntP (fun v => (toα (cntGe (sortDesc V) v) : α) ≤ toα k) (sortDesc V) :=
      cntP_ext (fun v => toα_le_iff.symm) (sortDesc V)
    rw [s1, s2, s3]
    exact cnt_rank_bound_desc (sortDesc V) k (desc_sortDesc V)
  have final : cntP (fun v => pUpPool V v ≤ t) V = cntP (fun v => cntGe V v ≤ k) V :=
    cntP_ext (fun v => (heq1 v).trans (heq2 v)) V
  rw [final]
  have hle1 : (toα (cntP (fun v => cntGe V v ≤ k) V) : α) ≤ toα k := toα_mono hcount
  have hle2 : toα (cntP (fun v => cntGe V v ≤ k) V) ≤ t * toα (V.length) :=
    le_trans hle1 hk1
  exact (div_le_iff hmα).mpr hle2

/-- **Validity of the lower-tail Monte-Carlo p-value.** The mirror statement
of `mc_upper_valid`, for the leakage direction. -/
theorem mc_lower_valid (V : List α) (t : α) (h0 : (0 : α) ≤ t) (h1 : t ≤ (1 : α))
    (hm : 0 < V.length) :
    toα (cntP (fun v => pLoPool V v ≤ t) V) / toα (V.length) ≤ t := by
  have hmα : (0 : α) < toα (V.length) := toα_strict hm
  have hr : (0 : α) ≤ t * toα (V.length) := mul_nonneg h0 (toα_nonneg _)
  have hrb : t * toα (V.length) < toα (V.length + 1) := by
    have h2 : t * toα (V.length) ≤ (1 : α) * toα (V.length) :=
      mul_le_mul_of_nonneg_right h1 (toα_nonneg _)
    rw [one_mul] at h2
    exact lt_of_le_of_lt h2 (toα_strict (Nat.lt_succ_self V.length))
  have ⟨k, hk1, hk2⟩ := floor_bounded (t * toα (V.length)) (V.length + 1) hr hrb
  have heq1 : ∀ v : α, (pLoPool V v ≤ t ↔ toα (cntLb V v) ≤ t * toα (V.length)) :=
    fun v => div_le_iff hmα
  have heq2 : ∀ v : α,
      (toα (cntLb V v) ≤ t * toα (V.length) ↔ cntLb V v ≤ k) := by
    intro v
    constructor
    · intro h
      have hlt : toα (cntLb V v) < toα (k + 1) := lt_of_le_of_lt h hk2
      have hlt2 : cntLb V v < k + 1 := toα_lt_iff.mp hlt
      omega
    · intro h
      exact le_trans (toα_mono h) hk1
  have hperm : ∀ v : α, cntLb (sortAsc V) v = cntLb V v := fun v => cntP_sortAsc _ V
  have hcount : cntP (fun v => cntLb V v ≤ k) V ≤ k := by
    have s1 : cntP (fun v => cntLb V v ≤ k) V
        = cntP (fun v => cntLb V v ≤ k) (sortAsc V) := (cntP_sortAsc _ V).symm
    have s2 : cntP (fun v => cntLb V v ≤ k) (sortAsc V)
        = cntP (fun v => cntLb (sortAsc V) v ≤ k) (sortAsc V) :=
      cntP_ext (fun v => by rw [hperm v]) (sortAsc V)
    have s3 : cntP (fun v => cntLb (sortAsc V) v ≤ k) (sortAsc V)
        = cntP (fun v => (toα (cntLb (sortAsc V) v) : α) ≤ toα k) (sortAsc V) :=
      cntP_ext (fun v => toα_le_iff.symm) (sortAsc V)
    rw [s1, s2, s3]
    exact cnt_rank_bound_asc (sortAsc V) k (asc_sortAsc V)
  have final : cntP (fun v => pLoPool V v ≤ t) V = cntP (fun v => cntLb V v ≤ k) V :=
    cntP_ext (fun v => (heq1 v).trans (heq2 v)) V
  rw [final]
  have hle1 : (toα (cntP (fun v => cntLb V v ≤ k) V) : α) ≤ toα k := toα_mono hcount
  have hle2 : toα (cntP (fun v => cntLb V v ≤ k) V) ≤ t * toα (V.length) :=
    le_trans hle1 hk1
  exact (div_le_iff hmα).mpr hle2

/-! ## The code form, its floor, and the n_rep ≥ 19 budget line -/

/-- The code's upper-tail p-value: `(count of nulls at or above obs + 1) /
(n_rep + 1)`, the Davison–Hinkley correction. -/
noncomputable def pUpCode (nulls : List α) (obs : α) : α :=
  (toα (cntGe nulls obs) + 1) / (toα (nulls.length) + 1)

/-- The code's lower-tail p-value. -/
noncomputable def pLoCode (nulls : List α) (obs : α) : α :=
  (toα (cntLb nulls obs) + 1) / (toα (nulls.length) + 1)

-- The pooled p-value of the pool `nulls ++ [obs]`, evaluated at the pooled
-- observation, is exactly the code's p-value: the observation counts itself
-- once, which is the `+1` of the Davison–Hinkley correction.
open Classical in
theorem pUp_pool_code (nulls : List α) (obs : α) :
    pUpPool (nulls ++ [obs]) obs = pUpCode nulls obs := by
  have h1 : cntGe (nulls ++ [obs]) obs = cntGe nulls obs + 1 := by
    rw [cntGe_append]
    have h2 : cntGe [obs] obs = 1 := by
      show (if obs ≤ obs then 1 else 0) + (0 : Nat) = 1
      rw [if_pos (le_refl obs)]
    rw [h2]
  have hlen : (nulls ++ [obs]).length = nulls.length + 1 := by
    rw [List.length_append, List.length_singleton]
  show toα (cntGe (nulls ++ [obs]) obs) / toα ((nulls ++ [obs]).length) = _
  rw [h1, hlen]
  rfl

open Classical in
theorem pLo_pool_code (nulls : List α) (obs : α) :
    pLoPool (nulls ++ [obs]) obs = pLoCode nulls obs := by
  have h1 : cntLb (nulls ++ [obs]) obs = cntLb nulls obs + 1 := by
    rw [cntLb_append]
    have h2 : cntLb [obs] obs = 1 := by
      show (if obs ≤ obs then 1 else 0) + (0 : Nat) = 1
      rw [if_pos (le_refl obs)]
    rw [h2]
  have hlen : (nulls ++ [obs]).length = nulls.length + 1 := by
    rw [List.length_append, List.length_singleton]
  show toα (cntLb (nulls ++ [obs]) obs) / toα ((nulls ++ [obs]).length) = _
  rw [h1, hlen]
  rfl

/-- **The significance floor.** The smallest attainable upper-tail p-value
with `n_rep` null draws is `1 / (n_rep + 1)`: no correction can do better,
and no data can beat it. -/
theorem pUpCode_min (nulls : List α) (obs : α) :
    (1 : α) / (toα (nulls.length) + 1) ≤ pUpCode nulls obs := by
  have hD : (0 : α) < toα (nulls.length) + 1 := by
    have h := add_pos_of_pos_of_nonneg zero_lt_one (toα_nonneg (α := α) (nulls.length))
    rwa [add_comm] at h
  have hD' : toα (nulls.length) + 1 ≠ (0 : α) := ne_zero_of_pos hD
  apply (div_le_iff hD).mpr
  show (1 : α) ≤ ((toα (cntGe nulls obs) + 1) / (toα (nulls.length) + 1))
      * (toα (nulls.length) + 1)
  rw [div_mul_cancel hD']
  have h1 : (0 : α) ≤ toα (cntGe nulls obs) := toα_nonneg (α := α) (cntGe nulls obs)
  rw [add_comm (toα (cntGe nulls obs)) (1 : α)]
  exact le_add_of_nonneg_right h1

/-- **The budget line.** With at most 18 replicates the attainable
significance is strictly worse than 1/20: `p ≤ 0.05` is unreachable, exactly
as the paper's §3.2 states. Nineteen replicates are needed before the
threshold is even attainable. -/
theorem n_rep_floor_19 : ∀ (nulls : List α), nulls.length + 1 ≤ 19 → ∀ obs : α,
    (1 : α) / toα 20 < pUpCode nulls obs := by
  intro nulls hlen obs
  have hD : (0 : α) < toα (nulls.length) + 1 := by
    have h := add_pos_of_pos_of_nonneg zero_lt_one
      (toα_nonneg (α := α) (nulls.length))
    rwa [add_comm] at h
  have hlt : (toα (nulls.length) + 1 : α) < toα 20 :=
    toα_strict (α := α) (show nulls.length + 1 < 20 by omega)
  have h1 : (1 : α) / toα 20 < (1 : α) / (toα (nulls.length) + 1) :=
    one_div_lt_one_div_of_lt hD hlt
  exact lt_of_lt_of_le h1 (pUpCode_min nulls obs)

end MC
