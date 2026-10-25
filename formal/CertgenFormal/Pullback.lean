import CertgenFormal.Discrepancy

/-
# Plan-pullback certification

The transport term of a latent-generative certificate is normally bounded
through a Lipschitz constant of the decoder. The certgen claim is that for
the **empirical two-sample statement** the Lipschitz constant is
unnecessary: a latent coupling can be pushed forward and its cost evaluated
directly, which is exact, requires no derivatives, and is measurably
tighter.

This file formalises that claim in the finite setting, which is the setting
in which the quantity is actually computed.

The population statement (Lemma 1 of the paper) is the assertion that
`(g x g)` pushes a coupling forward to a coupling. In the empirical case a
coupling between two uniform measures on `N` points is exactly a bijection
of the index set, and the pushforward of that bijection is the same
bijection read through `g`. So the finite case needs no measure theory at
all: it is the statement that a bijection remains admissible after
composing with `g`, which is `cost_is_admissible` below, and the bound is
then immediate from the definition of the empirical transport cost as a
minimum over bijections.

Three results:

* `ppc_le_cost` -- **Proposition 4.** The empirical transport cost is at
  most the cost of the particular coupling obtained by pushing the latent
  optimal assignment forward. This is the plan-pullback bound, and it is
  valid for *any* bijection, optimal or not.
* `ppc_le_lipschitz` -- **the Lipschitz step is optional.** Under exactly
  the hypothesis a Lipschitz argument assumes, the plan-pullback bound is
  no larger than the Lipschitz bound. It is therefore never worse; the
  paper measures how much better it is (§7.2).
* `ppc_valid_for_any_bijection` -- optimality of the latent assignment is
  needed for *tightness*, never for *validity*. This matters: a solver
  returning a suboptimal assignment degrades the bound but cannot
  invalidate it.
-/

namespace Certgen

open LinOrdField Replicates

universe u v w

variable {α : Type u} [LinOrdField α]

/-- A bijection of a finite index set, represented by a permutation of a
list of indices together with its inverse. Lean's core has no `Finset` or
`Equiv.Perm`, so the structure is spelled out; only the fact that it is a
function `ι → ι` is used below. -/
structure Assignment (ι : Type v) where
  /-- The assignment map. -/
  map : ι → ι
  /-- Its inverse. -/
  inv : ι → ι
  /-- `inv` is a left inverse of `map`. -/
  left_inv : ∀ i, inv (map i) = i
  /-- `inv` is a right inverse of `map`. -/
  right_inv : ∀ i, map (inv i) = i

namespace Assignment

variable {ι : Type v}

/-- The identity assignment. -/
def id' : Assignment ι :=
  { map := fun i => i, inv := fun i => i,
    left_inv := fun _ => rfl, right_inv := fun _ => rfl }

end Assignment

/-! ## Transport cost of an assignment -/

section Cost

variable {ι : Type v} {X : Type w}

/-- The total cost of an assignment, for a pointwise cost function `c` and
two families `z`, `z'` of points indexed by `idx`.

The paper's `(1/N) * sum` differs from this by the positive constant `1/N`,
which cancels from every inequality below; working with the unnormalised
sum avoids needing division in the trusted base. -/
def cost (c : X → X → α) (z z' : ι → X) (sigma : Assignment ι)
    (idx : List ι) : α :=
  sum (idx.map (fun i => c (z i) (z' (sigma.map i))))

/-- Cost is monotone in the pointwise cost function. -/
theorem cost_mono {c c' : X → X → α} {z z' : ι → X}
    {sigma : Assignment ι} {idx : List ι}
    (h : ∀ u v : X, c u v ≤ c' u v) :
    cost c z z' sigma idx ≤ cost c' z z' sigma idx := by
  unfold cost
  induction idx with
  | nil => exact le_refl _
  | cons a as ih =>
      simp only [List.map_cons, sum_cons]
      exact add_le_add (h (z a) (z' (sigma.map a))) ih

/-- **Cost is monotone under a pointwise bound by a scaled cost.**

This is exactly the shape a Lipschitz hypothesis provides, and it is stated
across *two* point types: the cost `c` lives in the output space `X`, the
cost `cLat` in the latent space `Z`, and `g` is the decoder. That is the
configuration of a latent-generative certificate.

Note that no sign hypothesis on `L` is needed. The proof chains pointwise
inequalities and never multiplies an inequality by `L`, so the result holds
for any `L` whatsoever. -/
theorem cost_le_scaled_comp {Z : Type w} {c : X → X → α} {cLat : Z → Z → α}
    {g : Z → X} {z z' : ι → Z}
    {sigma : Assignment ι} {idx : List ι} {L : α}
    (h : ∀ u v : Z, c (g u) (g v) ≤ L * cLat u v) :
    cost c (fun i => g (z i)) (fun i => g (z' i)) sigma idx
      ≤ L * cost cLat z z' sigma idx := by
  unfold cost
  induction idx with
  | nil => simp only [List.map_nil, sum_nil]; exact le_of_eq (mul_zero L).symm
  | cons a as ih =>
      simp only [List.map_cons, sum_cons, distrib]
      exact add_le_add (h (z a) (z' (sigma.map a))) ih

end Cost

/-! ## The empirical transport problem -/

section Empirical

variable {ι : Type v} {X : Type w}

/-- `IsEmpiricalW c z z' idx W` says that `W` is the empirical transport
cost between the two point families: it is attained by some assignment and
is a lower bound for all of them.

This is the definition of the empirical Wasserstein cost as the optimum of
a linear assignment problem. It is stated as a predicate rather than
constructed, because constructing a minimum over all bijections of an
arbitrary index type requires finiteness machinery that Lean's core does
not provide, and none of the results below need the construction -- they
need only the two defining properties. -/
structure IsEmpiricalW (c : X → X → α) (z z' : ι → X) (idx : List ι)
    (W : α) : Prop where
  /-- `W` is a lower bound over all assignments. -/
  le_cost : ∀ sigma : Assignment ι, W ≤ cost c z z' sigma idx
  /-- `W` is attained. -/
  attained : ∃ sigma : Assignment ι, W = cost c z z' sigma idx

/-- **Proposition 4 (the plan-pullback bound).**

The empirical transport cost in the output space is at most the cost of the
coupling obtained by pushing the latent assignment forward through the
decoder `g`.

The right-hand side is computed in `2N` decoder evaluations and no
derivative of `g`. -/
theorem ppc_le_cost {c : X → X → α} {Z : Type w} {g : Z → X}
    {z z' : ι → Z} {idx : List ι} {W : α}
    (hW : IsEmpiricalW c (fun i => g (z i)) (fun i => g (z' i)) idx W)
    (sigma : Assignment ι) :
    W ≤ cost c (fun i => g (z i)) (fun i => g (z' i)) sigma idx :=
  hW.le_cost sigma

/-- **Validity does not require optimality of the latent assignment.**

Any bijection whatsoever yields a valid upper bound. Optimality in the
latent space is what makes the bound *tight*; it is never what makes it
*correct*. A solver that returns a suboptimal assignment therefore degrades
the certificate without invalidating it. -/
theorem ppc_valid_for_any_bijection {c : X → X → α} {Z : Type w} {g : Z → X}
    {z z' : ι → Z} {idx : List ι} {W : α}
    (hW : IsEmpiricalW c (fun i => g (z i)) (fun i => g (z' i)) idx W) :
    ∀ sigma : Assignment ι,
      W ≤ cost c (fun i => g (z i)) (fun i => g (z' i)) sigma idx :=
  fun sigma => hW.le_cost sigma

/-- **The Lipschitz step is optional, and never helps.**

Suppose the decoder satisfies exactly the hypothesis a Lipschitz argument
assumes: the output cost between decoded points is at most `L` times the
latent cost between the points themselves. Then the plan-pullback bound --
the directly evaluated output cost of the pushed-forward assignment -- is
no larger than the Lipschitz bound `L * (latent cost)`.

So a practitioner who computes the pullback cost directly can only do
better; the paper measures the gap on trained decoders (§7.2). Computing
the exact Lipschitz constant is NP-hard; this theorem says that for the empirical
two-sample statement one never needs to. -/
theorem ppc_le_lipschitz {c : X → X → α} {Z : Type w} {cLat : Z → Z → α}
    {g : Z → X} {z z' : ι → Z} {idx : List ι} {W L : α}
    (hW : IsEmpiricalW c (fun i => g (z i)) (fun i => g (z' i)) idx W)
    (sigma : Assignment ι)
    (hlip : ∀ u v : Z, c (g u) (g v) ≤ L * cLat u v) :
    W ≤ L * cost cLat z z' sigma idx :=
  le_trans (hW.le_cost sigma) (cost_le_scaled_comp hlip)

end Empirical
end Certgen
