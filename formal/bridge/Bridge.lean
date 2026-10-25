import CertgenFormal.Boundary
import CertgenFormal.Analysis
import Mathlib.Data.Real.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The standard model: the trusted base over the real numbers

This file closes the standard model: it discharges every interface of the
trusted base against Mathlib's `Real`, so that each theorem of
`CertgenFormal` holds, verbatim, over the reals.

* `LinOrdFieldDiv Real` (hence `LinOrdField Real`): the seventeen
  arithmetic axioms plus inversion, discharged by Mathlib's ordered-field
  lemmas;
* `Discrepancy Real M` for every metric space `M`: the metric laws;
* `HasSqrt Real`: `Real.sqrt`;
* `Expectation Real Ω` for every finite probability space with real
  masses: the expectation is the finite weighted sum, with monotonicity,
  additivity, constants and scalar homogeneity from `Finset.sum`;
* `SymCdf Real`: a symmetric distribution function with an explicit
  quantile. The intended model of the paper's z-test is the Gaussian,
  whose distribution function is the error function, and Mathlib v4.15.0
  does not provide `Real.erf`. Moreover, the interface's *global* quantile
  monotonicity rules out every strictly increasing cdf with full support
  -- logistic and Gaussian alike -- because such a quantile is surjective
  onto `ℝ`. The instance is therefore discharged on the **uniform law on
  `[-1, 1]` with a clamped quantile** (see `uniCdf` below). Every theorem
  of `Quantile.lean` -- the MDD power algebra, the fifty-percent power of
  the two-standard-deviation rule, the severity bounds -- holds for the
  two-sided level-α test built on any law satisfying the interface, the
  uniform one included; the Gaussian needs both `Real.erf` and a
  weakening of the monotonicity law to the open unit interval.
* `Analysis Real`: `Real.rpow` with positivity, monotonicity in the base,
  and unboundedness of `n ↦ (n : ℝ)^s` for positive exponents -- the
  three laws behind the vanishing of the power-law floor.

With these instances the analysis layer's theorems hold over ℝ: the
power-law floor `C · (n+1)^(-s)` tends to zero, the floor-corrected
vacuity tends to `D / D_B`, and the skill asymptotics of §5.1 are
theorems about real numbers. The file closes with such instantiations
as examples.
-/

namespace Certgen

open scoped Classical

/-! ## The arithmetic base -/

/-- The real numbers are a model of the division-capable trusted base. -/
noncomputable instance instLinOrdFieldReal : LinOrdField Real where
  zero := 0
  one := 1
  add_assoc := by intro a b c; ring
  add_comm := by intro a b; ring
  add_zero := by intro a; ring
  add_neg := by intro a; ring
  sub_eq := by intro a b; ring
  mul_assoc := by intro a b c; ring
  mul_comm := by intro a b; ring
  mul_one := by intro a; ring
  distrib := by intro a b c; ring
  le_refl := le_refl
  le_trans := le_trans
  le_antisymm := le_antisymm
  le_total := le_total
  lt_iff := fun a b => lt_iff_le_and_ne
  add_le_add_left := by intro a b h c; exact add_le_add_left h c
  mul_nonneg := fun ha hb => mul_nonneg ha hb
  zero_ne_one := zero_ne_one

/-- Inversion closes the arithmetic base. -/
noncomputable instance instLinOrdFieldDivReal : LinOrdFieldDiv Real where
  __ := instLinOrdFieldReal
  inv := Inv.inv
  mul_inv_cancel := fun ha => mul_inv_cancel₀ ha

/-! ## Metric discrepancies -/

/-- Any metric space is a model of `Discrepancy` valued in the reals. -/
noncomputable instance instMetricDiscrepancy {M : Type*} [MetricSpace M] :
    Discrepancy Real M where
  d := fun x y => Dist.dist x y
  nonneg := fun _ _ => dist_nonneg
  symm := fun x y => dist_comm x y
  triangle := fun x y z => dist_triangle x y z
  eq_zero_iff := fun _ _ => dist_eq_zero

/-! ## Square roots -/

/-- The reals with `Real.sqrt` model `HasSqrt`. -/
noncomputable instance instHasSqrtReal : Severity.HasSqrt Real where
  sqrt := Real.sqrt
  sqrt_nonneg := Real.sqrt_nonneg
  sq_sqrt := fun ha => Real.mul_self_sqrt ha

/-! ## Finite expectations -/

/-- A finite probability space with real masses models `Expectation`: the
expectation is the finite weighted sum. -/
noncomputable instance instFintypeExpectation {Ω : Type*} [Fintype Ω]
    (P : Ω → Real) (hP : ∀ i, 0 ≤ P i) (hsum : ∑ i, P i = 1) :
    Expectation Real Ω where
  E := fun f => ∑ i, P i * f i
  mono := by
    intro f g hfg
    refine Finset.sum_le_sum ?_
    intro i _
    exact mul_le_mul_of_nonneg_left (hfg i) (hP i)
  add := by
    intro f g
    have h1 : ∀ i : Ω, P i * (f i + g i) = P i * f i + P i * g i := fun i => by ring
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun i _ => h1 i)
  const := by
    intro c
    show ∑ i, P i * c = c
    have h1 : ∀ i : Ω, P i * c = c * P i := fun i => mul_comm (P i) c
    have h2 : ∑ i, P i * c = ∑ i, c * P i := Finset.sum_congr rfl (fun i _ => h1 i)
    have h3 : c * ∑ i, P i = ∑ i, c * P i := Finset.mul_sum _ _ c
    rw [h2, ← h3, hsum, mul_one]
  smul := by
    intro c f
    show ∑ i, P i * (c * f i) = c * ∑ i, P i * f i
    have h1 : ∀ i : Ω, P i * (c * f i) = c * (P i * f i) := fun i => by ring
    have h2 : ∑ i, P i * (c * f i) = ∑ i, c * (P i * f i) := Finset.sum_congr rfl (fun i _ => h1 i)
    have h3 : c * ∑ i, P i * f i = ∑ i, c * (P i * f i) := Finset.mul_sum _ _ c
    rw [h2, ← h3]

/-! ## A symmetric distribution with explicit quantile -/

/-- The uniform distribution function on `[-1, 1]`, extended by clamping:
symmetric about the origin, monotone, `[0,1]`-valued.

This is the model that closes the `SymCdf` interface as stated. Two
facts explain the choice:

* the Gaussian needs `Real.erf`, which Mathlib v4.15.0 does not provide;
* more subtly, the interface's `zq_mono` requires the quantile to be
  monotone on *all* of `α`, not only on `(0, 1)`. For a strictly
  increasing cdf with full support -- Gaussian or logistic -- the
  quantile maps `(0,1)` onto all of `α`, and no real-valued extension can
  dominate its own range, so the Gaussian quantile can never satisfy the
  law as stated. The uniform law on a bounded interval admits a clamped
  quantile (`zq t = 2t - 1` inside `(0,1)`, clamped to `±1` outside),
  which is globally monotone and inverts the cdf on the open unit
  interval, and that is what is discharged here.

Every theorem of `Quantile.lean` -- the MDD power algebra, the
fifty-percent power of the two-standard-deviation rule, the severity
bounds -- therefore holds for the two-sided level-α test built on this
law. -/
noncomputable def uniCdf (x : Real) : Real :=
  if x ≤ -1 then 0 else if x < 1 then (x + 1) / 2 else 1

/-- The clamped quantile of the uniform law. -/
noncomputable def uniZq (t : Real) : Real :=
  if t ≤ 0 then -1 else if t < 1 then 2 * t - 1 else 1

theorem uniCdf_neg (z : Real) (hz : z ≤ -1) : uniCdf z = 0 := by
  unfold uniCdf
  rw [if_pos hz]

theorem uniCdf_mid (z : Real) (hz1 : ¬(z ≤ -1)) (hz2 : z < 1) : uniCdf z = (z + 1) / 2 := by
  unfold uniCdf
  rw [if_neg hz1, if_pos hz2]

theorem uniCdf_pos (z : Real) (hz : ¬(z < 1)) : uniCdf z = 1 := by
  unfold uniCdf
  by_cases hz1 : z ≤ -1
  · exact absurd (lt_of_le_of_lt hz1 (by norm_num : (-1:Real) < 1)) hz
  · rw [if_neg hz1, if_neg hz]

theorem uniZq_neg (t : Real) (ht : t ≤ 0) : uniZq t = -1 := by
  unfold uniZq
  rw [if_pos ht]

theorem uniZq_mid (t : Real) (ht1 : ¬(t ≤ 0)) (ht2 : t < 1) : uniZq t = 2 * t - 1 := by
  unfold uniZq
  rw [if_neg ht1, if_pos ht2]

theorem uniZq_pos (t : Real) (ht : ¬(t < 1)) : uniZq t = 1 := by
  unfold uniZq
  by_cases ht1 : t ≤ 0
  · exact absurd (lt_of_le_of_lt ht1 (by norm_num : (0:Real) < 1)) ht
  · rw [if_neg ht1, if_neg ht]

noncomputable instance instSymCdfReal : Power.SymCdf Real where
  __ := instLinOrdFieldDivReal
  cdf := uniCdf
  cdf_mono := by
    intro x y hxy
    rcases le_or_lt x (-1 : Real) with hx | hx
    · rw [uniCdf_neg x hx]
      rcases le_or_lt y (-1 : Real) with hy | hy
      · rw [uniCdf_neg y hy]
      · rcases lt_or_le y (1 : Real) with hy2 | hy2
        · rw [uniCdf_mid y (by linarith) hy2]
          linarith
        · rw [uniCdf_pos y (by linarith)]
          linarith
    · rcases lt_or_le x (1 : Real) with hx2 | hx2
      · rcases le_or_lt y (-1 : Real) with hy | hy
        · rw [uniCdf_neg y hy]
          linarith
        · rcases lt_or_le y (1 : Real) with hy2 | hy2
          · rw [uniCdf_mid x (by linarith) hx2, uniCdf_mid y (by linarith) hy2]
            linarith
          · rw [uniCdf_mid x (by linarith) hx2, uniCdf_pos y (by linarith)]
            linarith
      · rcases le_or_lt y (-1 : Real) with hy | hy
        · rw [uniCdf_neg y hy]
          linarith
        · rcases lt_or_le y (1 : Real) with hy2 | hy2
          · rw [uniCdf_pos x (by linarith), uniCdf_mid y (by linarith) hy2]
            linarith
          · rw [uniCdf_pos x (by linarith), uniCdf_pos y (by linarith)]
  cdf_symm := by
    intro x
    rcases le_or_lt x (-1 : Real) with hx | hx
    · rw [uniCdf_neg x hx, uniCdf_pos (-x) (by linarith)]
      norm_num
    · rcases lt_or_le x (1 : Real) with hx2 | hx2
      · rw [uniCdf_mid x (by linarith) hx2, uniCdf_mid (-x) (by linarith) (by linarith)]
        ring
      · rw [uniCdf_pos x (by linarith), uniCdf_neg (-x) (by linarith)]
        norm_num
  cdf_nonneg := by
    intro x
    rcases le_or_lt x (-1 : Real) with hx | hx
    · rw [uniCdf_neg x hx]
    · rcases lt_or_le x (1 : Real) with hx2 | hx2
      · rw [uniCdf_mid x (by linarith) hx2]
        linarith
      · rw [uniCdf_pos x (by linarith)]
        exact zero_le_one
  cdf_le_one := by
    intro x
    rcases le_or_lt x (-1 : Real) with hx | hx
    · rw [uniCdf_neg x hx]
      exact zero_le_one
    · rcases lt_or_le x (1 : Real) with hx2 | hx2
      · rw [uniCdf_mid x (by linarith) hx2]
        linarith
      · rw [uniCdf_pos x (by linarith)]
  zq := uniZq
  zq_char := by
    intro t ht0 ht1
    rw [uniZq_mid t (by linarith) ht1]
    rw [uniCdf_mid (2 * t - 1) (by linarith : ¬((2 * t - 1 : Real) ≤ -1))
      (by linarith : (2 * t - 1 : Real) < 1)]
    ring
  zq_mono := by
    intro s t hst
    rcases le_or_lt s (0 : Real) with hs | hs
    · rw [uniZq_neg s hs]
      rcases le_or_lt t (0 : Real) with ht | ht
      · rw [uniZq_neg t ht]
      · rcases lt_or_le t (1 : Real) with ht2 | ht2
        · rw [uniZq_mid t (by linarith) ht2]
          linarith
        · rw [uniZq_pos t (by linarith)]
          linarith
    · rcases lt_or_le s (1 : Real) with hs2 | hs2
      · rcases le_or_lt t (0 : Real) with ht | ht
        · linarith
        · rcases lt_or_le t (1 : Real) with ht2 | ht2
          · rw [uniZq_mid s (by linarith) hs2, uniZq_mid t (by linarith) ht2]
            linarith
          · rw [uniZq_mid s (by linarith) hs2, uniZq_pos t (by linarith)]
            linarith
      · rw [uniZq_pos s (by linarith)]
        rcases le_or_lt t (0 : Real) with ht | ht
        · linarith
        · rcases lt_or_le t (1 : Real) with ht2 | ht2
          · rw [uniZq_mid t (by linarith) ht2]
            linarith
          · rw [uniZq_pos t (by linarith)]
  zq_half := by
    have h : (1 / (1 + 1) : Real) = 1 / 2 := by norm_num
    rw [h, uniZq_mid (1 / 2) (by norm_num) (by norm_num)]
    norm_num

/-! ## Real powers -/

/-- The reals with `Real.rpow` model the analysis layer's power
interface. -/
noncomputable instance instAnalysisReal : Analysis Real where
  __ := instLinOrdFieldDivReal
  rpow := fun x s => Real.rpow x s
  rpow_pos := fun hx _ => Real.rpow_pos_of_pos hx _
  rpow_mono := fun hx hxy hs => Real.rpow_le_rpow hx hxy (le_of_lt hs)
  rpow_unbounded := by
    intro s hs b
    -- b is dominated by exp (log (max b 1) + 1), which is dominated in
    -- turn by (N : ℝ)^s for a large enough natural N.
    have hto : ∀ (M : Nat), (LinOrdFieldDiv.toα M : Real) = (M : Real) := by
      intro M
      induction M with
      | zero => rw [LinOrdFieldDiv.toα_zero, Nat.cast_zero]
      | succ m ih => rw [LinOrdFieldDiv.toα_succ, Nat.cast_succ, ih]
    obtain ⟨N, hN⟩ := exists_nat_gt (Real.exp ((Real.log (max b 1) + 1) / s))
    have hNpos : (0:Real) < (N:Real) := by
      rcases N with _ | N'
      · exact absurd hN (by have := Real.exp_pos ((Real.log (max b 1) + 1) / s); simp at hN; linarith)
      · exact_mod_cast Nat.succ_pos N'
    refine ⟨N, ?_⟩
    have hkey : (N:Real)^s = Real.exp (Real.log N * s) := Real.rpow_def_of_pos hNpos s
    have hNbig : (Real.log (max b 1) + 1) / s ≤ Real.log N := by
      have hle : Real.exp ((Real.log (max b 1) + 1) / s) ≤ (N:Real) := le_of_lt hN
      have hlog := Real.log_le_log (Real.exp_pos _) hle
      rwa [Real.log_exp] at hlog
    have h1 : Real.log (max b 1) + 1 ≤ Real.log N * s := by
      have hla : (Real.log (max b 1) + 1) / s * s = Real.log (max b 1) + 1 := by
        field_simp
      calc Real.log (max b 1) + 1 = (Real.log (max b 1) + 1) / s * s := hla.symm
        _ ≤ Real.log N * s := mul_le_mul_of_nonneg_right hNbig (le_of_lt hs)
    have h3 : Real.exp (Real.log (max b 1) + 1) ≤ Real.exp (Real.log N * s) :=
      Real.exp_le_exp.mpr h1
    have h5 : Real.exp (Real.log (max b 1)) = max b 1 := by
      refine Real.exp_log ?_
      rcases lt_or_le 0 (max b 1) with hpos | hnonpos
      · exact hpos
      · exfalso
        have h6 : max b 1 = 0 :=
          le_antisymm hnonpos (le_trans (by norm_num : (0:Real) ≤ 1) (le_max_right b 1))
        have h7 : (1:Real) ≤ max b 1 := le_max_right b 1
        rw [h6] at h7
        norm_num at h7
    have hble : (0:Real) < max b 1 :=
      lt_of_lt_of_le (by norm_num : (0:Real) < 1) (le_max_right b 1)
    have h9 : Real.exp (Real.log (max b 1)) < Real.exp (Real.log (max b 1)) * Real.exp 1 := by
      rw [h5]
      have h10 : (1:Real) < Real.exp 1 := by
        have h11 : Real.exp 0 < Real.exp 1 := Real.exp_strictMono (by norm_num)
        rwa [Real.exp_zero] at h11
      calc max b 1 = max b 1 * 1 := by ring
        _ < max b 1 * Real.exp 1 := mul_lt_mul_of_pos_left h10 hble
    show b < Real.rpow (LinOrdFieldDiv.toα N) s
    rw [hto]
    calc b ≤ max b 1 := le_max_left b 1
      _ = Real.exp (Real.log (max b 1)) := h5.symm
      _ < Real.exp (Real.log (max b 1)) * Real.exp 1 := h9
      _ = Real.exp (Real.log (max b 1) + 1) := by rw [← Real.exp_add]
      _ ≤ Real.exp (Real.log N * s) := h3
      _ = (N:Real)^s := hkey.symm

/-! ## Instantiated theorems

Two read-outs: the vacuity criterion over the reals, and the analysis
layer's asymptotics as statements about real numbers. -/

/-- The vacuity criterion over `Real`: a certificate whose statistical
term exceeds its trivial scale is vacuous. -/
example (c : Certificate Real) (h : c.trivialScale < c.statistical) :
    ¬ c.NonVacuous :=
  c.vacuous_of_statistical_exceeds h

/-- The power-law floor vanishes over `Real`: for every positive exponent
and constant, `C · (n+1)^(-s) → 0` in the classical ε–N sense. -/
example (C s : Real) (hC : 0 ≤ C) (hs : 0 < s) :
    Analysis.Tendsto (Analysis.powLaw C s) 0 :=
  Analysis.powLaw_tendsto_zero hC hs

/-- The skill asymptotic over `Real`: the floor-corrected vacuity ratio
converges to the ratio of raw discrepancies, independent of n; and to
zero in the equal-law case. -/
example (D DB C s : Real) (hC : 0 ≤ C) (hs : 0 < s) (hDB : DB ≠ 0) :
    Analysis.Tendsto
      (fun n => Skill.vacuity D (Analysis.powLaw C s n) DB) (D / DB) :=
  Analysis.skill_asymptotic hC hs hDB

end Certgen
