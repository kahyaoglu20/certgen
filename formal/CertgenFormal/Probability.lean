import CertgenFormal.Discrepancy

/-
# Expectations, the factor-½ bridge in expectation, and the Fréchet mean

This file introduces the probability content the certgen project needs, at the
level of abstraction a proof assistant can honour: an **expectation
functional** on an outcome space `Ω`, valued in the ordered field `α`, with
exactly four laws -- monotonicity, additivity, constants, and scalar
homogeneity. No measure theory, no sigma-algebras, no integrability
conditions are assumed; whatever holds of an `E` satisfying these four laws
holds of Lebesgue integration whenever the integrals exist.

Three results are proved from this interface, all division-free:

* **The factor-½ bridge in expectation form.** The triangle inequality
  for a metric discrepancy gives `E d(x,y) ≤ E d(x,p) + E d(p,y)`; when the
  two halves are identically distributed (exchangeability of the two-sample
  split), this becomes `E d(x,y) ≤ 2 · E d(x,p)`. That factor of two is the
  slack between the two statistics, whose actual ratio the paper measures,
  and the reason the two-sample statistic is a *surrogate* for the
  statistical term and never an identity for it.

* **The Fréchet mean is optimal among constant generators.** For squared
  coordinate cost, `E Σᵢ (Xᵢ − cᵢ)²` is minimised over `c` by the coordinate
  means, with the classical decomposition into variance plus shift-squared.
  This is the formal content of the paper's T0 baseline: the point mass at
  the data mean has squared population discrepancy `tr Cov` (the data
  scale, squared), and no other constant generator does better.

* **No detector separates equal laws.** If two generators induce the same
  distribution of observed samples -- stated here as equality of the two
  expectation functionals, which is exactly the strength the argument needs
  -- then every event's probability, in particular every test's rejection
  probability, is the same under both. This is the formal content of the
  paper's identifiability limit (§5.3): where a law's support is finite and
  densely sampled, memorisation is not detectable by any statistic at all;
  the remedy there is a disjoint training split, not a better statistic.

The indicator definition is classical, hence `noncomputable`; this
development proves, it does not compute.
-/

namespace Certgen

open LinOrdField

universe u v w

/-! ## The expectation interface -/

/-- An expectation functional: monotone, additive, unital on constants, and
homogeneous under scalar multiplication. Four laws, nothing else. -/
class Expectation (α : Type u) (Ω : Type v) [LinOrdField α] where
  /-- The expectation of an `α`-valued function on the outcome space. -/
  E : (Ω → α) → α
  /-- Expectations respect pointwise order. -/
  mono : ∀ {f g : Ω → α}, (∀ ω, f ω ≤ g ω) → E f ≤ E g
  /-- Expectations are additive. -/
  add : ∀ {f g : Ω → α}, E (fun ω => f ω + g ω) = E f + E g
  /-- The expectation of a constant is that constant. -/
  const : ∀ (c : α), E (fun _ => c) = c
  /-- Scalars pass through expectations. -/
  smul : ∀ (c : α) (f : Ω → α), E (fun ω => c * f ω) = c * E f

namespace ExpectationOps

variable {α : Type u} {Ω : Type v} [LinOrdField α] [Expectation α Ω]

open Expectation Replicates

theorem E_zero : E (fun _ : Ω => (0 : α)) = (0 : α) := const 0

theorem E_one : E (fun _ : Ω => (1 : α)) = (1 : α) := const 1

theorem E_neg (f : Ω → α) : E (fun ω => -f ω) = -E f := by
  have hfun : (fun ω => -f ω) = (fun ω => -(1 : α) * f ω) :=
    funext (fun ω => by rw [neg_mul, one_mul])
  rw [hfun, smul (c := -(1 : α)) (f := f), neg_mul, one_mul]

theorem E_sub (f g : Ω → α) : E (fun ω => f ω - g ω) = E f - E g := by
  have hfun : (fun ω => f ω - g ω) = (fun ω => f ω + -g ω) :=
    funext (fun ω => sub_eq (f ω) (g ω))
  rw [hfun, add (f := fun ω => f ω) (g := fun ω => -g ω), E_neg g, sub_eq]

theorem E_nonneg {f : Ω → α} (h : ∀ ω, (0 : α) ≤ f ω) : (0 : α) ≤ E f := by
  have := mono (f := fun _ => (0 : α)) (g := f) h
  rwa [E_zero] at this

/-! ## Events and their probabilities -/

open Classical in
/-- The indicator of an event, as an `α`-valued function on `Ω`. -/
noncomputable def ind (A : Ω → Prop) : Ω → α := fun ω => if A ω then 1 else 0

/-- The probability of an event: the expectation of its indicator. -/
noncomputable def prob (A : Ω → Prop) : α := E (ind A)

omit [Expectation α Ω] in
theorem ind_bounds (A : Ω → Prop) (ω : Ω) :
    (0 : α) ≤ ind A ω ∧ ind A ω ≤ (1 : α) := by
  by_cases h : A ω
  · rw [ind, if_pos h]
    exact ⟨zero_le_one, le_refl _⟩
  · rw [ind, if_neg h]
    exact ⟨le_refl _, zero_le_one⟩

omit [Expectation α Ω] in
theorem ind_le_of_impl {A B : Ω → Prop} (h : ∀ ω, A ω → B ω) (ω : Ω) :
    (ind A ω : α) ≤ ind B ω := by
  by_cases hA : A ω
  · have hB : B ω := h ω hA
    rw [ind, if_pos hA, ind, if_pos hB]
    exact le_refl _
  · have hA0 : ind A ω = (0 : α) := by rw [ind, if_neg hA]
    rw [hA0]
    exact (ind_bounds B ω).1

theorem prob_nonneg (A : Ω → Prop) : (0 : α) ≤ prob A :=
  E_nonneg (fun ω => (ind_bounds A ω).1)

theorem prob_le_one (A : Ω → Prop) : prob A ≤ (1 : α) := by
  have := mono (f := ind A) (g := fun _ : Ω => (1 : α)) (fun ω => (ind_bounds A ω).2)
  rwa [E_one] at this

theorem prob_mono {A B : Ω → Prop} (h : ∀ ω, A ω → B ω) : (prob A : α) ≤ prob B :=
  mono (fun ω => ind_le_of_impl h ω)

/-- **Union bound**, from linearity and pointwise subadditivity. -/
theorem prob_union_le (A B : Ω → Prop) :
    (prob (fun ω => A ω ∨ B ω) : α) ≤ prob A + prob B := by
  have hpt : ∀ ω, (ind (fun ω' => A ω' ∨ B ω') ω : α) ≤ ind A ω + ind B ω := by
    intro ω
    by_cases h1 : A ω
    · by_cases h2 : B ω
      · rw [ind, if_pos (Or.inl h1), ind, if_pos h1, ind, if_pos h2]
        exact le_add_of_nonneg_right zero_le_one
      · rw [ind, if_pos (Or.inl h1), ind, if_pos h1, ind, if_neg h2, add_zero']
        exact le_refl _
    · by_cases h2 : B ω
      · have hA0 : ind A ω = (0 : α) := by rw [ind, if_neg h1]
        rw [hA0, ind, if_pos (Or.inr h2), ind, if_pos h2, zero_add]
        exact le_refl _
      · have hA0 : ind A ω = (0 : α) := by rw [ind, if_neg h1]
        have hB0 : ind B ω = (0 : α) := by rw [ind, if_neg h2]
        rw [hA0, hB0, ind, if_neg (fun hc => hc.elim h1 h2), add_zero']
        exact le_refl _
  have hE := mono (f := ind (fun ω' => A ω' ∨ B ω'))
    (g := fun ω => ind A ω + ind B ω) hpt
  rwa [add (f := ind A) (g := ind B)] at hE

/-! ## The factor-½ bridge in expectation -/

section HalfBridge

variable {M : Type w} [Discrepancy α M]

/-- **Pointwise triangle, integrated.** For random objects `x`, `y` and a fixed
reference `p₀`, the expected two-sample discrepancy is at most the sum of the
two expected one-sample discrepancies. No independence is used: the triangle
inequality holds pointwise whatever the coupling. -/
theorem E_dist_triangle (x y : Ω → M) (p₀ : M) :
    E (fun ω => dist α (x ω) (y ω))
      ≤ E (fun ω => dist α (x ω) p₀) + E (fun ω => dist α (p₀) (y ω)) := by
  have hpt : ∀ ω : Ω, dist α (x ω) (y ω)
      ≤ dist α (x ω) p₀ + dist α (p₀) (y ω) :=
    fun ω => Discrepancy.triangle (α := α) (x ω) p₀ (y ω)
  have hE := mono (f := fun ω => dist α (x ω) (y ω))
    (g := fun ω => dist α (x ω) p₀ + dist α (p₀) (y ω)) hpt
  rwa [add (f := fun ω => dist α (x ω) p₀) (g := fun ω => dist α (p₀) (y ω))] at hE

/-- **The factor-½ bridge (expectation form).**

If the two halves of a two-sample split are identically distributed -- the
exchangeability hypothesis, stated here at exactly the strength used: the two
one-sample terms have equal expectations -- then

    E d(x, y)  ≤  2 · E d(x, p₀),

so the one-sample statistical term is at least *half* the two-sample
statistic, and in general no more than that half can be recovered: the
two-sample quantity is a surrogate for the statistical term, never an
identity for it, and the gap is measured rather than assumed. -/
theorem E_one_sample_ge_half (x y : Ω → M) (p₀ : M)
    (hexch : E (fun ω => dist α (x ω) p₀) = E (fun ω => dist α (p₀) (y ω))) :
    E (fun ω => dist α (x ω) (y ω)) ≤ (1 + 1) * E (fun ω => dist α (x ω) p₀) := by
  have h := E_dist_triangle (α := α) x y p₀
  rw [← hexch, ← two_mul] at h
  exact h

end HalfBridge

/-! ## No detector separates equal laws -/

/-- **The identifiability limit.** If two generators induce the same
distribution of observed samples -- equality of the two expectation
functionals -- then every event's probability is identical under the two, so
no test, however constructed, can distinguish them. In the paper's leakage
screen this is the boundary case: where `n / |support| ≈ 1`, memorisation is
not identifiable from an honest draw by any detector, and the remedy is a
disjoint training split rather than a better statistic. -/
theorem no_detector_separates {α : Type u} {Ω : Type v} [LinOrdField α]
    (E₁ E₂ : Expectation α Ω)
    (h : ∀ f : Ω → α, E₁.E f = E₂.E f) (A : Ω → Prop) :
    E₁.E (ind A) = E₂.E (ind A) := h _

/-! ## Auxiliary order lemmas (division-free) -/

theorem neg_mul_right' (a b : α) : a * -b = -(a * b) := by
  rw [mul_comm a (-b), neg_mul, mul_comm b a]

theorem sq_nonneg {a : α} : (0 : α) ≤ a * a := by
  by_cases h : (0 : α) ≤ a
  · exact mul_nonneg h h
  · have ha : a ≤ (0 : α) := le_of_lt (lt_of_not_le h)
    have hnn : (0 : α) ≤ -a := by
      have := sub_nonneg_of_le ha
      rwa [sub_eq, zero_add] at this
    have hm := mul_nonneg hnn hnn
    rw [neg_mul, neg_mul_right', neg_neg] at hm
    exact hm

/-! ## Sum lemmas -/

section Sums

variable {ι : Type w}

theorem sum_map_ext {F G : ι → α} : ∀ {L : List ι},
    (∀ i ∈ L, F i = G i) → sum (L.map F) = sum (L.map G) := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons i rest ih =>
      intro h
      rw [List.map_cons, List.map_cons, sum_cons, sum_cons,
          h i (List.mem_cons_self i rest),
          ih (fun j hj => h j (List.mem_cons_of_mem i hj))]

theorem sum_map_add {F G : ι → α} : ∀ (L : List ι),
    sum (L.map (fun i => F i + G i)) = sum (L.map F) + sum (L.map G) := by
  intro L
  induction L with
  | nil => rw [List.map_nil, List.map_nil, List.map_nil, sum_nil, add_zero']
  | cons i rest ih =>
      rw [List.map_cons, List.map_cons, List.map_cons, sum_cons, sum_cons,
          sum_cons, ih]
      calc (F i + G i) + (sum (rest.map F) + sum (rest.map G))
          = F i + (G i + (sum (rest.map F) + sum (rest.map G))) := by
            rw [add_assoc]
        _ = F i + ((G i + sum (rest.map F)) + sum (rest.map G)) := by
            rw [add_assoc (G i) (sum (rest.map F)) (sum (rest.map G))]
        _ = F i + ((sum (rest.map F) + G i) + sum (rest.map G)) := by
            rw [add_comm (G i) (sum (rest.map F))]
        _ = F i + (sum (rest.map F) + (G i + sum (rest.map G))) := by
            rw [add_assoc (sum (rest.map F)) (G i) (sum (rest.map G))]
        _ = (F i + sum (rest.map F)) + (G i + sum (rest.map G)) := by
            rw [add_assoc (F i) (sum (rest.map F)) (G i + sum (rest.map G))]

theorem sum_map_smul {c : α} {F : ι → α} : ∀ (L : List ι),
    sum (L.map (fun i => c * F i)) = c * sum (L.map F) := by
  intro L
  induction L with
  | nil => rw [List.map_nil, List.map_nil, sum_nil, mul_zero]
  | cons i rest ih =>
      rw [List.map_cons, List.map_cons, sum_cons, sum_cons, ih, ← distrib]

theorem sum_map_zero (L : List ι) :
    sum (L.map (fun _ => (0 : α))) = (0 : α) := by
  induction L with
  | nil => rfl
  | cons i rest ih => rw [List.map_cons, sum_cons, ih, add_zero']

theorem sum_map_nonneg {F : ι → α} (hF : ∀ i, (0 : α) ≤ F i) : ∀ (L : List ι),
    (0 : α) ≤ sum (L.map F) := by
  intro L
  induction L with
  | nil => rw [List.map_nil, sum_nil]; exact le_refl _
  | cons i rest ih => rw [List.map_cons, sum_cons]; exact add_nonneg (hF i) ih

/-! ## The Fréchet mean and the T0 baseline -/

/-- The squared-cost objective: `E Σᵢ (Xᵢ − cᵢ)²` over the enumerated
coordinates, for a constant generator `c`. -/
def totalSq (L : List ι) (X : Ω → ι → α) (c : ι → α) : α :=
  E (fun ω => sum (L.map (fun i => (X ω i - c i) * (X ω i - c i))))

/-- The coordinate mean: the expectation of each coordinate. -/
def meanOf (X : Ω → ι → α) (i : ι) : α := E (fun ω => X ω i)

/-- The square decomposition behind the Fréchet mean: for any `u`, the mean
`m` and a candidate `c`, `(u − c)² = (u − m)² + (m − c)² + 2(u − m)(m − c)`. -/
theorem sq_decomp (u m c : α) :
    (u - c) * (u - c)
      = (u - m) * (u - m)
        + ((m - c) * (m - c) + (1 + 1) * ((u - m) * (m - c))) := by
  have e1 : u - c = (u - m) + (m - c) := by
    rw [sub_eq u c, sub_eq u m, sub_eq m c, add_assoc u (-m) (m + -c),
        ← add_assoc (-m) m (-c), neg_add m, zero_add (-c)]
  have hsplit : ((u - m) + (m - c)) * ((u - m) + (m - c))
      = (u - m) * (u - m)
        + ((1 + 1) * ((u - m) * (m - c)) + (m - c) * (m - c)) := by
    rw [distrib ((u - m) + (m - c)) (u - m) (m - c),
        distrib_right (u - m) (m - c) (u - m),
        distrib_right (u - m) (m - c) (m - c)]
    calc (u - m) * (u - m) + (m - c) * (u - m)
        + ((u - m) * (m - c) + (m - c) * (m - c))
        = (u - m) * (u - m) + ((m - c) * (u - m)
            + ((u - m) * (m - c) + (m - c) * (m - c))) := by
          rw [add_assoc ((u - m) * (u - m)) ((m - c) * (u - m))
            ((u - m) * (m - c) + (m - c) * (m - c))]
        _ = (u - m) * (u - m) + (((m - c) * (u - m) + (m - c) * (u - m))
            + (m - c) * (m - c)) := by
          rw [mul_comm (u - m) (m - c),
              ← add_assoc ((m - c) * (u - m)) ((m - c) * (u - m))
                ((m - c) * (m - c))]
        _ = (u - m) * (u - m) + ((1 + 1) * ((m - c) * (u - m))
            + (m - c) * (m - c)) := by
          rw [← two_mul]
        _ = (u - m) * (u - m)
            + ((1 + 1) * ((u - m) * (m - c)) + (m - c) * (m - c)) := by
          rw [mul_comm (m - c) (u - m)]
  rw [e1, hsplit]
  rw [add_comm ((1 + 1) * ((u - m) * (m - c))) ((m - c) * (m - c))]

/-- **The Fréchet decomposition.** The squared objective at a constant `c`
equals the squared objective at the coordinate means plus a sum of squared
shifts. Everything beyond the four expectation laws is field algebra: expand
the square, interchange `E` with the finite sum, and observe that the cross
term carries `E[Xᵢ − mᵢ] = 0` in each coordinate. -/
theorem frechet_decomposition (L : List ι) (X : Ω → ι → α) (c : ι → α) :
    totalSq L X c
      = totalSq L X (fun i => meanOf X i)
        + sum (L.map (fun i => (meanOf X i - c i) * (meanOf X i - c i))) := by
  -- Step 1: expectations interchange with the finite sums.
  have EsumL : ∀ (F : ι → Ω → α),
      E (fun ω => sum (L.map (fun i => F i ω)))
        = sum (L.map (fun i => E (fun ω => F i ω))) := by
    intro F
    induction L with
    | nil =>
        show E (fun _ : Ω => (0 : α)) = (0 : α)
        exact E_zero
    | cons i rest ih =>
        have h1 : E (fun ω => F i ω + sum (rest.map (fun j => F j ω)))
            = E (fun ω => F i ω) + E (fun ω => sum (rest.map (fun j => F j ω))) :=
          add (f := fun ω => F i ω)
              (g := fun ω => sum (rest.map (fun j => F j ω)))
        calc E (fun ω => sum ((i :: rest).map (fun j => F j ω)))
            = E (fun ω => F i ω + sum (rest.map (fun j => F j ω))) := rfl
          _ = E (fun ω => F i ω) + E (fun ω => sum (rest.map (fun j => F j ω))) := h1
          _ = E (fun ω => F i ω) + sum (rest.map (fun j => E (fun ω => F j ω))) := by
              rw [ih]
          _ = sum ((i :: rest).map (fun j => E (fun ω => F j ω))) := rfl
  -- Step 2: the cross term vanishes coordinatewise.
  have cross_zero : ∀ i : ι,
      E (fun ω => (X ω i - meanOf X i) * (meanOf X i - c i)) = (0 : α) := by
    intro i
    have hcomm : (fun ω => (X ω i - meanOf X i) * (meanOf X i - c i))
        = (fun ω => (meanOf X i - c i) * (X ω i - meanOf X i)) :=
      funext (fun ω => mul_comm _ _)
    have hm : meanOf X i = E (fun ω => X ω i) := rfl
    rw [hcomm, smul (c := meanOf X i - c i) (f := fun ω => X ω i - meanOf X i),
        E_sub, const, hm, sub_self, mul_zero]
  -- Step 3: the pointwise identity, summed.
  have key : ∀ ω : Ω,
      sum (L.map (fun i => (X ω i - c i) * (X ω i - c i)))
        = sum (L.map (fun i => (X ω i - meanOf X i) * (X ω i - meanOf X i)))
          + (sum (L.map (fun i => (meanOf X i - c i) * (meanOf X i - c i)))
            + (1 + 1) * sum (L.map
                (fun i => (X ω i - meanOf X i) * (meanOf X i - c i)))) := by
    intro ω
    have hsq : ∀ i : ι, (X ω i - c i) * (X ω i - c i)
        = (X ω i - meanOf X i) * (X ω i - meanOf X i)
          + ((meanOf X i - c i) * (meanOf X i - c i)
            + (1 + 1) * ((X ω i - meanOf X i) * (meanOf X i - c i))) :=
      fun i => sq_decomp (X ω i) (meanOf X i) (c i)
    have hext : sum (L.map (fun i => (X ω i - c i) * (X ω i - c i)))
        = sum (L.map (fun i => (X ω i - meanOf X i) * (X ω i - meanOf X i)
          + ((meanOf X i - c i) * (meanOf X i - c i)
            + (1 + 1) * ((X ω i - meanOf X i) * (meanOf X i - c i))))) :=
      sum_map_ext (F := fun i => (X ω i - c i) * (X ω i - c i))
        (G := fun i => (X ω i - meanOf X i) * (X ω i - meanOf X i)
          + ((meanOf X i - c i) * (meanOf X i - c i)
            + (1 + 1) * ((X ω i - meanOf X i) * (meanOf X i - c i))))
        (fun i _ => hsq i)
    rw [hext, sum_map_add, sum_map_add,
        sum_map_smul (c := (1 + 1 : α))
          (F := fun i => (X ω i - meanOf X i) * (meanOf X i - c i))]
  -- Step 4: assemble inside E, by named steps written in reduced form.
  have hfun : (fun ω => sum (L.map (fun i => (X ω i - c i) * (X ω i - c i))))
      = (fun ω => sum (L.map (fun i => (X ω i - meanOf X i)
            * (X ω i - meanOf X i)))
          + (sum (L.map (fun i => (meanOf X i - c i) * (meanOf X i - c i)))
            + (1 + 1) * sum (L.map
                (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))) :=
    funext key
  have e1 : E (fun ω => sum (L.map (fun i => (X ω i - meanOf X i)
            * (X ω i - meanOf X i)))
          + (sum (L.map (fun i => (meanOf X i - c i) * (meanOf X i - c i)))
            + (1 + 1) * sum (L.map
                (fun i => (X ω i - meanOf X i) * (meanOf X i - c i)))))
      = E (fun ω => sum (L.map (fun i => (X ω i - meanOf X i)
            * (X ω i - meanOf X i))))
        + E (fun ω => sum (L.map (fun i => (meanOf X i - c i)
            * (meanOf X i - c i)))
          + (1 + 1) * sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i)))) :=
    add (f := fun ω => sum (L.map (fun i => (X ω i - meanOf X i)
              * (X ω i - meanOf X i))))
       (g := fun ω => sum (L.map (fun i => (meanOf X i - c i)
              * (meanOf X i - c i)))
            + (1 + 1) * sum (L.map
                (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))
  have e2 : E (fun ω => sum (L.map (fun i => (meanOf X i - c i)
            * (meanOf X i - c i)))
          + (1 + 1) * sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))
      = E (fun _ : Ω => sum (L.map (fun i => (meanOf X i - c i)
            * (meanOf X i - c i))))
        + E (fun ω => (1 + 1) * sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i)))) :=
    add (f := fun _ : Ω => sum (L.map (fun i => (meanOf X i - c i)
              * (meanOf X i - c i))))
       (g := fun ω => (1 + 1) * sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))
  have e3 : E (fun ω => (1 + 1) * sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))
      = (1 + 1) * E (fun ω => sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i)))) :=
    smul (c := (1 + 1 : α))
       (f := fun ω => sum (L.map
              (fun i => (X ω i - meanOf X i) * (meanOf X i - c i))))
  have eD : E (fun _ : Ω => sum (L.map (fun i => (meanOf X i - c i)
      * (meanOf X i - c i))))
      = sum (L.map (fun i => (meanOf X i - c i) * (meanOf X i - c i))) :=
    const _
  have hcross0 : sum (L.map (fun i => E (fun ω => (X ω i - meanOf X i)
      * (meanOf X i - c i))))
      = sum (L.map (fun _ => (0 : α))) :=
    sum_map_ext (F := fun i => E (fun ω => (X ω i - meanOf X i)
        * (meanOf X i - c i)))
      (G := fun _ => (0 : α)) (L := L) (fun i _ => cross_zero i)
  have eB : totalSq L X (fun i => meanOf X i)
      = E (fun ω => sum (L.map (fun i => (X ω i - meanOf X i)
          * (X ω i - meanOf X i)))) := rfl
  have cform : totalSq L X c
      = E (fun ω => sum (L.map (fun i => (X ω i - c i) * (X ω i - c i)))) := rfl
  rw [cform, hfun, e1, e2, e3, eD]
  rw [EsumL (fun i ω => (X ω i - meanOf X i) * (meanOf X i - c i))]
  rw [hcross0, sum_map_zero, mul_zero, add_zero', eB]

end Sums

/-- **The Fréchet mean is optimal among constant generators.** No constant
`c` beats the coordinate means under squared cost: the T0 baseline of the
certgen paper, formalised. The paper's operational statement -- the point
mass at the data mean has population discrepancy the data scale, and no
constant generator does better -- is exactly this inequality read through
the square root. -/
theorem frechet_optimal {α : Type u} {Ω : Type v} [LinOrdField α] [Expectation α Ω]
    {ι : Type w} (L : List ι) (X : Ω → ι → α) (c : ι → α) :
    totalSq L X (fun i => meanOf X i) ≤ totalSq L X c := by
  have h := frechet_decomposition L X c
  have hnn : (0 : α) ≤ sum (L.map
    (fun i => (meanOf X i - c i) * (meanOf X i - c i))) :=
    sum_map_nonneg (fun i => sq_nonneg) L
  rw [h]
  exact le_add_of_nonneg_right hnn

end ExpectationOps
end Certgen
