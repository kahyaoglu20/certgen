import CertgenFormal.Discrepancy

/-
# The certification limit

A latent-generative Wasserstein certificate of the standard form bounds the
quantity of interest by a sum of three terms: a statistical term that the
practitioner cannot control, a computable transport term, and a term
controlled by the number of synthetic draws. A bound is **vacuous** when it
exceeds the trivial scale of the data, in the sense of Dziugaite and Roy:
a guarantee larger than the quantity it bounds says nothing.

Two results are proved here.

* `vacuous_of_statistical_exceeds` is Proposition 3 of the certgen paper.
  It is one line, and it is stated with the antecedent on the *statistical
  term*.

* `surrogate_does_not_transfer` records, as a theorem, why the antecedent
  may **not** be replaced by the computable two-sample surrogate: the
  triangle inequality gives only a factor-one-half bridge between the
  two-sample quantity and the one-sample statistical term, and their actual
  ratio is a measurement. A criterion stated in terms of the surrogate
  would not follow.
-/

namespace Certgen

open LinOrdField

universe u v

variable {α : Type u} [LinOrdField α]

/-- The three-term decomposition of a latent-generative certificate.

`statistical` is the term governed by the sample size and the intrinsic
dimension; `transport` is the term computed exactly in `Pullback.lean`;
`draws` is controlled by drawing more synthetic samples. `trivial` is the
scale of the data, against which vacuity is judged. -/
structure Certificate (α : Type u) [LinOrdField α] where
  /-- The irreducible statistical term. -/
  statistical : α
  /-- The computable transport term. -/
  transport : α
  /-- The term controlled by the number of synthetic draws. -/
  draws : α
  /-- The trivial scale of the data. -/
  trivialScale : α
  /-- All three terms are discrepancies, hence non-negative. -/
  transport_nonneg : (0 : α) ≤ transport
  /-- All three terms are discrepancies, hence non-negative. -/
  draws_nonneg : (0 : α) ≤ draws

namespace Certificate

variable (c : Certificate α)

/-- The value of the certificate: the sum of the three terms. -/
def bound : α := c.statistical + c.transport + c.draws

/-- A certificate is non-vacuous when it is at most the trivial scale. -/
def NonVacuous : Prop := c.bound ≤ c.trivialScale

/-- **Proposition 3 (vacuity criterion).**

If the statistical term alone exceeds the trivial scale, then no bound of
this form is non-vacuous -- whatever the practitioner does about the other
two terms, and however many synthetic samples are drawn.

The antecedent is the statistical term. It is *not* the computable
surrogate; see `surrogate_does_not_transfer`. -/
theorem vacuous_of_statistical_exceeds
    (h : c.trivialScale < c.statistical) : ¬ c.NonVacuous := by
  intro hnv
  have h1 : c.statistical ≤ c.statistical + c.transport :=
    le_add_of_nonneg_right c.transport_nonneg
  have h2 : c.statistical + c.transport
      ≤ c.statistical + c.transport + c.draws :=
    le_add_of_nonneg_right c.draws_nonneg
  have hle : c.statistical ≤ c.trivialScale :=
    le_trans (le_trans h1 h2) hnv
  exact not_lt_of_le hle h

/-- The contrapositive, in the form a practitioner uses it: a non-vacuous
certificate certifies that the statistical term was within the trivial
scale, which is a statement about the *measurement*, not about the model. -/
theorem statistical_le_of_nonVacuous (h : c.NonVacuous) :
    c.statistical ≤ c.trivialScale := by
  have h1 : c.statistical ≤ c.statistical + c.transport :=
    le_add_of_nonneg_right c.transport_nonneg
  have h2 : c.statistical + c.transport
      ≤ c.statistical + c.transport + c.draws :=
    le_add_of_nonneg_right c.draws_nonneg
  exact le_trans (le_trans h1 h2) h

end Certificate

/-! ## Why the surrogate may not be substituted for the term -/

section Surrogate

variable {M : Type v} [Discrepancy α M]

/-- **The surrogate does not transfer without its bridge factor.**

The computable quantity is the two-sample discrepancy `dist x y` between
two disjoint halves. The statistical term is a one-sample discrepancy
`dist x p` to the population `p`. By the triangle inequality all one can
conclude from a large two-sample quantity is that *one* of the two
one-sample terms is at least half of it.

Concretely: it is impossible for both one-sample terms to be strictly less
than half the two-sample quantity. Therefore an argument of the shape
"the surrogate crosses one, hence the statistical term crosses one" is not
available; the bridge factor must be supplied, and in the certgen study it
was measured rather than assumed. -/
theorem surrogate_does_not_transfer (x y p : M) :
    ¬ ((1 + 1 : α) * dist α x p < dist α x y ∧
       (1 + 1 : α) * dist α p y < dist α x y) :=
  Discrepancy.not_both_less_than_half x y p

/-- The positive form of the same fact: at least one one-sample term is at
least half the two-sample quantity. This is the *entire* content the
triangle inequality licenses, and it is what the certgen paper states. -/
theorem exists_half_bridge (x y p : M) :
    dist α x y ≤ (1 + 1 : α) * dist α x p ∨
    dist α x y ≤ (1 + 1 : α) * dist α p y := by
  rcases Classical.em (dist α x y ≤ (1 + 1 : α) * dist α x p) with h | h
  · exact Or.inl h
  · refine Or.inr ?_
    rcases Classical.em (dist α x y ≤ (1 + 1 : α) * dist α p y) with h2 | h2
    · exact h2
    · exact absurd ⟨lt_of_not_le h, lt_of_not_le h2⟩
        (surrogate_does_not_transfer (α := α) x y p)

end Surrogate
end Certgen
