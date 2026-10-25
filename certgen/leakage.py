"""Null-calibrated leakage screening.

A generator that returns training rows produces a sample closer to a real
reference sample than a second real sample would be. The screen splits the
real data into disjoint halves A (the reference) and B. B is a genuine sample
from the data law that shares no row with A, so the distances from B to its
nearest neighbours in A, and the fraction of B's rows that occur verbatim in
A, describe what a generator drawing fresh samples produces. The generated
sample is compared with this null; no absolute distance threshold is used.
On discrete data, where real rows legitimately coincide, the null contains
the coincidences, and a generator that reproduces common combinations is not
flagged.

Two limits apply to any screen of this kind: a memoriser that perturbs every
copied record by a large enough amount escapes it, and when the sample covers
a finite support densely, memorisation cannot be distinguished from fresh
sampling at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["LeakageReport", "leakage_check", "DCR_QUANTILE", "DCR_FRAC_TOL",
           "DCR_MEDIAN_TOL", "OVERLAP_TOL"]

# Operating points, defined relative to the null: a generated sample is
# flagged when it is closer to the reference than the null by more than these
# margins.
DCR_QUANTILE = 0.05      # "unusually close" = below this quantile of the null DCR
DCR_FRAC_TOL = 3.0       # flag if that happens >3x as often as the null rate
DCR_MEDIAN_TOL = 0.75    # flag if median(DCR_gen) < 0.75 x median(DCR_null)
OVERLAP_TOL = 3.0        # flag if verbatim overlap >3x the null overlap
MIN_NULL_N = 32          # below this the null is too noisy to calibrate against


@dataclass
class LeakageReport:
    """Result of `leakage_check`.

    `suspected` is a screening decision against a null built from disjoint
    real halves, and it inherits that null's sampling noise; a marginal flag
    is a prompt to inspect the train/evaluation split, not a p-value.
    """
    suspected: bool
    median_dcr_gen: float = float("nan")
    median_dcr_null: float = float("nan")
    dcr_median_ratio: float = float("nan")
    frac_close_gen: float = float("nan")
    frac_close_null: float = float("nan")
    dcr_frac_ratio: float = float("nan")
    overlap_gen: float = float("nan")
    overlap_null: float = float("nan")
    overlap_ratio: float = float("nan")
    n_reference: int = 0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("suspected", "median_dcr_gen", "median_dcr_null",
                 "dcr_median_ratio", "frac_close_gen", "frac_close_null",
                 "dcr_frac_ratio", "overlap_gen", "overlap_null",
                 "overlap_ratio", "n_reference", "reasons", "warnings")}

    def summary(self) -> str:
        L = [f"leakage: suspected={self.suspected}  "
             f"(reference n={self.n_reference})",
             f"  median DCR  gen={self.median_dcr_gen:.6g}  "
             f"null={self.median_dcr_null:.6g}  "
             f"ratio={self.dcr_median_ratio:.3f}  (flag < {DCR_MEDIAN_TOL})",
             f"  frac close  gen={self.frac_close_gen:.3%}  "
             f"null={self.frac_close_null:.3%}  "
             f"ratio={self.dcr_frac_ratio:.2f}x  (flag > {DCR_FRAC_TOL}x)",
             f"  verbatim    gen={self.overlap_gen:.3%}  "
             f"null={self.overlap_null:.3%}  "
             f"ratio={self.overlap_ratio:.2f}x  (flag > {OVERLAP_TOL}x)"]
        for r in self.reasons:
            L.append(f"  [!] {r}")
        for w in self.warnings:
            L.append(f"  [?] {w}")
        return "\n".join(L)


def _row_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Fraction of rows of B that appear verbatim in A."""
    try:
        va = np.ascontiguousarray(A).view(
            [("", A.dtype)] * A.shape[1]).ravel()
        vb = np.ascontiguousarray(B).view(
            [("", B.dtype)] * B.shape[1]).ravel()
        return float(np.isin(vb, va).mean())
    except Exception:
        return float("nan")


def leakage_check(X_real: np.ndarray, X_gen: np.ndarray,
                  seed: int = 0, n_reference: Optional[int] = None
                  ) -> LeakageReport:
    """Compare the generated sample's closeness to real data with a null.

    The real sample is split into a reference half and a disjoint null half.
    Distances to the closest reference record (DCR) are computed for every
    generated row and every null row with a k-d tree over the full reference
    half, together with the verbatim row overlap.
    """
    X_real = np.asarray(X_real, float)
    X_gen = np.asarray(X_gen, float)
    rng = np.random.default_rng(seed)
    warn: List[str] = []

    n_r = len(X_real)
    if n_r < 2 * MIN_NULL_N or len(X_gen) < 1:
        return LeakageReport(False, n_reference=0, warnings=[
            f"real sample has {n_r} rows; at least {2*MIN_NULL_N} are needed to "
            "build a disjoint same-law null. Leakage was NOT checked."])

    perm = rng.permutation(n_r)
    half = n_r // 2
    if n_reference is not None:
        half = int(min(max(n_reference, MIN_NULL_N), n_r - MIN_NULL_N))
    ref, null = X_real[perm[:half]], X_real[perm[half:]]

    tree = cKDTree(ref)
    dcr_gen = tree.query(X_gen, k=1)[0]
    dcr_null = tree.query(null, k=1)[0]

    med_gen = float(np.median(dcr_gen))
    med_null = float(np.median(dcr_null))
    # "unusually close" is defined by the null's own lower tail; on a
    # discrete law whose null DCR is mostly zero the threshold is zero
    thresh = float(np.quantile(dcr_null, DCR_QUANTILE))
    frac_gen = float((dcr_gen <= thresh).mean())
    frac_null = float((dcr_null <= thresh).mean())

    ov_gen = _row_overlap(ref, X_gen)
    ov_null = _row_overlap(ref, null)

    reasons: List[str] = []
    if med_null > 0:
        med_ratio = med_gen / med_null
        if med_ratio < DCR_MEDIAN_TOL:
            reasons.append(
                f"median distance-to-closest-record is {med_ratio:.2f}x the "
                f"same-law null ({med_gen:.4g} vs {med_null:.4g}): generated "
                "points sit closer to the reference than independent real "
                "points do.")
    else:
        med_ratio = float("nan")
        warn.append("median null DCR is 0 (the real law has coincident rows), "
                    "so the median-ratio test is not identified; the "
                    "frequency and overlap tests still apply.")

    frac_ratio = (frac_gen / frac_null) if frac_null > 0 else (
        float("inf") if frac_gen > 0 else float("nan"))
    if np.isfinite(frac_ratio) and frac_ratio > DCR_FRAC_TOL:
        reasons.append(
            f"{frac_gen:.1%} of generated points fall below the null's "
            f"{DCR_QUANTILE:.0%} DCR quantile, {frac_ratio:.1f}x the "
            f"null rate of {frac_null:.1%}.")
    elif np.isinf(frac_ratio):
        reasons.append(
            f"{frac_gen:.1%} of generated points are closer to the reference "
            "than ANY null point is.")

    ov_ratio = float("nan")
    if np.isfinite(ov_gen) and np.isfinite(ov_null):
        if ov_null > 0:
            ov_ratio = ov_gen / ov_null
            if ov_ratio > OVERLAP_TOL:
                reasons.append(
                    f"verbatim row overlap is {ov_gen:.1%} against a null of "
                    f"{ov_null:.1%} ({ov_ratio:.1f}x). Some overlap is normal "
                    "for discrete columns; this much is not.")
        elif ov_gen > 0.01:
            ov_ratio = float("inf")
            reasons.append(
                f"{ov_gen:.1%} of generated rows appear verbatim in the "
                "reference while no independent real row does.")

    if len(null) < 128:
        warn.append(f"null half has only {len(null)} rows; the calibration is "
                    "noisy and a marginal flag should not be trusted.")
    if np.isfinite(ov_null) and ov_null > 0.5:
        warn.append(
            f"{ov_null:.0%} of independent real rows already appear in the "
            "reference: the support is finite and densely sampled, so "
            "memorisation cannot be distinguished from fresh sampling by this "
            "or any other detector. Use a training split disjoint from the "
            "evaluation data.")

    return LeakageReport(bool(reasons), med_gen, med_null, med_ratio,
                         frac_gen, frac_null, frac_ratio,
                         ov_gen, ov_null, ov_ratio, len(ref), reasons, warn)
