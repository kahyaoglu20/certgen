import CertgenFormal.Sizing
import CertgenFormal.DivField

/-
# Interval propagation and the extrapolation label

`required_n_interval` (paper §4.6) propagates the WLS `(log C, d_eff)`
uncertainty through the inversion `log n* = d_eff · (log C − log φ)` by
multivariate-t draws. The distribution theory of that construction is
empirical (its delivered coverage is measured, not proved);
what is proved here are the deterministic algebraic facts the construction
rests on and the labelling discipline around it:

* endpoint propagation: if `d_eff` and the log-gap are each known only to
  lie in an interval, the prescribed log sample size lies between the
  product of the lower endpoints and the product of the upper endpoints
  (given a nonnegative gap -- the operative case, since the gap is the log
  distance to a target below the fitted constant);
* the width of the prescription grows linearly with the log-gap, so the
  intervals quoted far beyond the fitted grid are honestly labelled by the
  extrapolation factor;
* the extrapolation label itself: a target `c` times the grid maximum is
  labelled by exactly `c`.

These are the facts behind the interval widths the paper reports at
targets beyond the measured grid.
-/

namespace Certgen

namespace Interval

open LinOrdField LinOrdFieldDiv

universe u

variable {α : Type u} [LinOrdFieldDiv α]

/-- **Endpoint propagation.** With the exponent in `[d1, d2]`, the log-gap
in `[g1, g2]`, and a nonnegative gap, the prescription `d · gap` lies in
`[d1 · g1, d2 · g2]`. -/
theorem logn_bounds (d1 d2 g1 g2 d g : α)
    (hd1 : d1 ≤ d) (hd2 : d ≤ d2) (hg1 : (0 : α) ≤ g1) (hg1' : g1 ≤ g)
    (hg2 : g ≤ g2) (hd1' : (0 : α) ≤ d1) :
    d1 * g1 ≤ d * g ∧ d * g ≤ d2 * g2 := by
  have hdpos : (0 : α) ≤ d := le_trans hd1' hd1
  constructor
  · have hA : d1 * g1 ≤ d * g1 := mul_le_mul_of_nonneg_right hd1 hg1
    have hB : d * g1 ≤ d * g := mul_le_mul_of_nonneg_left hg1' hdpos
    exact le_trans hA hB
  · have hg2' : (0 : α) ≤ g2 :=
      le_trans hg1 (le_trans hg1' hg2)
    have hC : d * g ≤ d * g2 := mul_le_mul_of_nonneg_left hg2 hdpos
    have hD : d * g2 ≤ d2 * g2 := mul_le_mul_of_nonneg_right hd2 hg2'
    exact le_trans hC hD

/-- **Width grows linearly in the log-gap.** For a fixed exponent
uncertainty, the width of the propagated interval is the exponent width
times the log-gap; a target further from the fitted constant gives a wider
prescription, with no bound on the growth other than the gap itself. -/
theorem width_grows (d1 d2 g1 g2 : α) (hd : d1 ≤ d2) (hg : g1 ≤ g2)
    (_hd1 : (0 : α) ≤ d1) : (d2 - d1) * g1 ≤ (d2 - d1) * g2 :=
  mul_le_mul_of_nonneg_left hg (sub_nonneg_of_le hd)

/-- **The extrapolation label is exact.** A target equal to `c` times the
grid maximum is labelled by exactly the factor `c`: the extrapolation
factors the paper quotes are ratios, not approximations. -/
theorem extrapolation_label (target gridmax c : α)
    (hgrid : (0 : α) < gridmax) (htarget : target = c * gridmax) :
    target / gridmax = c := by
  rw [htarget]
  exact mul_div_cancel (ne_zero_of_pos hgrid) c

end Interval
end Certgen
