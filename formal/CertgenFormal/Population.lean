import CertgenFormal.Probability

/-
# The population pushforward-coupling lemma

This file discharges the second of the three boundary theorems: the
population form of the paper's Lemma (§7.1),

    For any coupling π of the latent measures Σ and P and any measurable
    decoder g, the plan (g ⊗ g)#π is a coupling of g#Σ and g#P, and
    therefore
        W₂(g#Σ, g#P)² ≤ ∫ ‖g(z) − g(z′)‖² dπ(z, z′).

`Pullback.lean` proved the finite case, in which the quantity is computed:
there a coupling between two uniform empirical measures is a bijection of
the index set, and the lemma is the statement that a bijection remains
admissible after composing with `g`. The population case is what the finite
case approximates, and the paper proves it with one line of measure theory:
a pushforward of a coupling is a coupling, and `W₂²` is an infimum over
couplings.

Here the same two lines are theorems at the level of abstraction the
`Expectation` interface provides. A **coupling** of random elements `X` and
`Y` is a pair `(X', Y')` on a common outcome space whose one-marginal has
the law of `X` and whose two-marginal has the law of `Y`; a **law** is the
pushforward of the expectation functional, which at this abstraction level
is the operation `F ↦ E (F ∘ X)` itself -- exactly the strength the argument
needs, and no more. Whatever holds of an `E` satisfying the four laws holds
of Lebesgue integration wherever the integrals exist, so the two theorems
below are the population lemma for standard probability, with the
measurability clauses as usual in the model, not the interface.

Two remarks on scope, so that nothing is smuggled:

* `W₂` is characterised, not constructed: `IsPopW` below says that `W` is a
  lower bound for the expected cost of *every* coupling and the greatest
  such. This is the same discipline as `IsEmpiricalW` in `Pullback.lean`:
  constructing the infimum needs finiteness machinery that no result here
  needs, and a specification keeps the statement honest about what is
  assumed. The bridge (`bridge/`) discharges the whole
  expectation interface, and thereby these theorems, over finite
  probability spaces with real weights.

* The quantification over coupling spaces ranges over `Type`. Every model
  in this development (finite index sets, `Real`, function spaces over
  them) lives there, and Lean structures cannot quantify over all
  universes in a field; the restriction is stated here rather than hidden.

The cost `c` is kept abstract: the paper's `‖g(z) − g(z′)‖²` is the
intended instantiation, and nothing below uses more than the four
expectation laws, so the lemma holds for every cost the interface's models
can express.
-/

namespace Certgen

open LinOrdField

universe u v₁ v₂ w

variable {α : Type u} [LinOrdField α]

namespace Population

/-! ## Couplings

A random element's law, at this abstraction level, is the pushforward of
the expectation functional onto the value space. Two random elements are
coupled by a pair on a common space if their one- and two-marginals match
those laws. -/

/-- `IsCoupling X Y X' Y'`: the pair `(X', Y')` on a common outcome space
couples the random elements `X` and `Y`, which may live on different
spaces: the pushforward law of `X'` is that of `X`, and the pushforward
law of `Y'` is that of `Y`. Laws are compared through *all* `α`-valued
test functions, which is exactly the strength the `Expectation`
interface's own quantification provides. -/
structure IsCoupling (α : Type u) [LinOrdField α] {M : Type w}
    {Ω₁ : Type v₁} {Ω₂ : Type v₂} [Expectation α Ω₁] [Expectation α Ω₂]
    {Ω' : Type} [Expectation α Ω']
    (X : Ω₁ → M) (Y : Ω₂ → M) (X' Y' : Ω' → M) : Prop where
  /-- The first marginal has the law of `X`. -/
  fst : ∀ F : M → α,
    @Expectation.E α Ω' _ _ (fun ω => F (X' ω))
      = @Expectation.E α Ω₁ _ _ (fun ω => F (X ω))
  /-- The second marginal has the law of `Y`. -/
  snd : ∀ F : M → α,
    @Expectation.E α Ω' _ _ (fun ω => F (Y' ω))
      = @Expectation.E α Ω₂ _ _ (fun ω => F (Y ω))

/-- `IsPopW c X Y W`: `W` is the population transport cost between the laws
of `X` and `Y` for the cost `c` -- a lower bound for the expected cost of
every coupling, and the greatest such (so the value, not merely a bound,
once any approximating sequence of couplings exists). -/
structure IsPopW (α : Type u) [LinOrdField α] {M : Type w}
    {Ω₁ : Type v₁} {Ω₂ : Type v₂} [Expectation α Ω₁] [Expectation α Ω₂]
    (c : M → M → α) (X : Ω₁ → M) (Y : Ω₂ → M) (W : α) : Prop where
  /-- `W` is a lower bound for the expected cost of every coupling. -/
  le : ∀ {Ω' : Type} [Expectation α Ω'] (X' Y' : Ω' → M),
    IsCoupling α X Y X' Y' →
    W ≤ @Expectation.E α Ω' _ _ (fun ω => c (X' ω) (Y' ω))
  /-- `W` is the greatest such lower bound. -/
  ge : ∀ W' : α, (∀ {Ω' : Type} [Expectation α Ω'] (X' Y' : Ω' → M),
      IsCoupling α X Y X' Y' →
      W' ≤ @Expectation.E α Ω' _ _ (fun ω => c (X' ω) (Y' ω))) → W' ≤ W

/-! ## The lemma -/

/-- **The pushforward of a coupling is a coupling.** Composing a coupling's
legs with any decoder preserves the marginal laws: pushforward commutes
with taking laws. This is the measure-theoretic half of the paper's Lemma
(§7.1) -- `(g ⊗ g)#π` is a coupling of `g#Σ` and `g#P` -- proved here at
the level of expectation functionals: the law of `g ∘ X'` is the law of
`X'` read through the test functions `F ∘ g`, which is the law of `g ∘ X`
by the coupling's first marginal. The proof is one function composition;
no measurability is needed, because the interface never restricted the
test functions. -/
theorem pushforward_is_coupling {M : Type w} {LZ : Type w}
    {Ω₁ : Type v₁} {Ω₂ : Type v₂}
    [Expectation α Ω₁] [Expectation α Ω₂]
    {Ω' : Type} [Expectation α Ω']
    {Z : Ω₁ → LZ} {Z' : Ω₂ → LZ} {X' Y' : Ω' → LZ}
    (hc : IsCoupling α Z Z' X' Y') (g : LZ → M) :
    IsCoupling α (fun ω => g (Z ω)) (fun ω => g (Z' ω))
      (fun ω => g (X' ω)) (fun ω => g (Y' ω)) where
  fst F := hc.fst (fun z => F (g z))
  snd F := hc.snd (fun z => F (g z))

/-- **The plan-pullback bound, population form.** The population transport
cost between the pushed laws is at most the expected cost of the
pushed-forward coupling:

    W (g ∘ Z, g ∘ Z') ≤ E' [ c (g X', g Y') ].

Combined with `pushforward_is_coupling` this is exactly the paper's
inequality `W₂(g#Σ, g#P)² ≤ ∫ ‖g(z) − g(z′)‖² dπ`: the infimum defining
`W` ranges over all couplings of the pushed laws, the pushed-forward plan
is one of them, and no Lipschitz constant of `g` appears anywhere. The
empirical instantiation -- take the coupling to be the latent-optimal
assignment, read through `g` -- is `Pullback.ppc_le_cost`, and the two
together say the empirical bound converges to the population bound at the
rate the pushed coupling approximates the latent optimum. -/
theorem ppc_population {M : Type w} {LZ : Type w}
    {Ω₁ : Type v₁} {Ω₂ : Type v₂}
    [Expectation α Ω₁] [Expectation α Ω₂]
    {Ω' : Type} [Expectation α Ω']
    {Z : Ω₁ → LZ} {Z' : Ω₂ → LZ} {X' Y' : Ω' → LZ}
    (hc : IsCoupling α Z Z' X' Y') (g : LZ → M)
    {c : M → M → α}
    {W : α} (hW : IsPopW α c (fun ω => g (Z ω)) (fun ω => g (Z' ω)) W) :
    W ≤ @Expectation.E α Ω' _ _ (fun ω => c (g (X' ω)) (g (Y' ω))) :=
  hW.le (fun ω => g (X' ω)) (fun ω => g (Y' ω)) (pushforward_is_coupling hc g)

/-- **Validity does not require optimality of the coupling** (population
form). Any coupling of the latent laws yields a valid upper bound;
optimality is what makes the bound tight, never what makes it correct --
the population twin of `Pullback.ppc_valid_for_any_bijection`. -/
theorem ppc_population_any_coupling {M : Type w} {LZ : Type w}
    {Ω₁ : Type v₁} {Ω₂ : Type v₂}
    [Expectation α Ω₁] [Expectation α Ω₂]
    {Ω' : Type} [Expectation α Ω']
    {Z : Ω₁ → LZ} {Z' : Ω₂ → LZ} {X' Y' : Ω' → LZ}
    (hc : IsCoupling α Z Z' X' Y') (g : LZ → M)
    {c : M → M → α}
    {W : α} (hW : IsPopW α c (fun ω => g (Z ω)) (fun ω => g (Z' ω)) W) :
    W ≤ @Expectation.E α Ω' _ _ (fun ω => c (g (X' ω)) (g (Y' ω))) :=
  ppc_population hc g hW

end Population
end Certgen
