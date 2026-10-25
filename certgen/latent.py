"""
Latent-generative models: an empirical certificate without a Lipschitz
constant, and the sample-size limit on what any certificate can show.

Plan-pullback bound
-------------------
Published Wasserstein certificates for latent-generative models (Dao et al.,
arXiv:2307.08698; Lee et al., arXiv:2501.09876) bound the empirical term
W2(g#Sigma_N, g#Sigma'_N) by L * W2(Sigma_N, Sigma'_N), with L a Lipschitz constant of
the decoder g. The constant enters through one step: for any coupling pi of
the latent measures, (g x g)#pi couples the decoded measures, so

    W2(g#Sigma, g#Sigma')^2 <= integral of ||g(z) - g(z')||^2 dpi(z, z').

For uniform empirical measures of equal size the latent-optimal coupling is
an assignment, and the right-hand side can be evaluated exactly by decoding
both endpoints of every assigned pair. `plan_pullback` does this; it needs
no derivative of g and no Lipschitz constant. A Lipschitz constant remains
necessary for statements about latent points that were not drawn.

Certification limit
-------------------
A standard certificate splits W2(Ups, Ups~) into a statistical term
W2(Ups, Ups_N), an empirical term and a sampling term. The statistical term
depends only on the data and on N, and a bound is vacuous when it exceeds the
data scale tau. Its expected size is approximated by the two-sample floor
ratio

    r(d, N) = E W2(X_N, X'_N) / tau,

tabulated below on isotropic Gaussian laws (FLOOR_RATIO, indexed by nominal
dimension). The surrogate is not an identity with the statistical term; the
measured ratio between the two is TWO_SAMPLE_VS_ONE_SAMPLE.

Real data are characterised by a measured slope estimate d_hat
(`certgen.dimension.wasserstein_dimension`), which is biased upward in low
and downward in high dimension. D_HAT_CALIBRATION maps nominal dimension to
the expected d_hat, so that `certification_limit` can state the limit in the
units a practitioner measures. The tables are generated from
results/reference_tables.json by tools/write_library_tables.py; outside the
tabulated N they are clamped, not extrapolated.

The limit concerns what can be certified at a given sample size, not how
good a generator is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = ["PPCReport", "plan_pullback", "FLOOR_RATIO", "FLOOR_RATIO_SD",
           "D_HAT_CALIBRATION", "D_HAT_CALIBRATION_SD", "TABULATED_N",
           "TWO_SAMPLE_VS_ONE_SAMPLE", "floor_ratio", "floor_ratio_measured",
           "nominal_to_measured", "measured_to_nominal", "vacuity_margin",
           "certification_limit", "crossing_loglog"]


@dataclass
class PPCReport:
    """Result of `plan_pullback`."""
    ppc_bound: float
    exact_w2_output: Optional[float]
    lipschitz_bound: Optional[float]
    w2_latent: float
    n: int
    warnings: List[str] = field(default_factory=list)

    @property
    def tightness(self) -> Optional[float]:
        """Plan-pullback bound divided by the exact output-space distance."""
        if self.exact_w2_output is None:
            return None
        if self.exact_w2_output == 0.0:
            return float("inf") if self.ppc_bound > 0 else float("nan")
        return self.ppc_bound / self.exact_w2_output

    def summary(self) -> str:
        L = [f"plan-pullback bound      = {self.ppc_bound:.6f}   (n={self.n})",
             f"  W2 in latent space     = {self.w2_latent:.6f}"]
        if self.exact_w2_output is not None:
            L.append(f"  exact output-space W2  = {self.exact_w2_output:.6f}"
                     f"   -> bound is {self.tightness:.2f}x the exact value")
        if self.lipschitz_bound is not None:
            L.append(f"  Lipschitz bound        = {self.lipschitz_bound:.6f}"
                     f"   -> {self.lipschitz_bound / self.ppc_bound:,.0f}x looser")
        for w in self.warnings:
            L.append(f"  [!] {w}")
        return "\n".join(L)


def plan_pullback(Z_data: np.ndarray, Z_gen: np.ndarray,
                  decode: Callable[[np.ndarray], np.ndarray],
                  L_cert: Optional[float] = None, compute_exact: bool = True,
                  output_weight: float = 1.0) -> PPCReport:
    """Upper bound on the output-space W2 without a Lipschitz constant.

    Parameters
    ----------
    Z_data, Z_gen : (n, d_Z) latent codes of equal length, so that the latent
        transport problem is an assignment problem.
    decode : maps (n, d_Z) latent codes to (n, P) outputs.
    L_cert : optional certified Lipschitz constant; when given, the report
        also contains the classical bound L_cert * W2(latent).
    compute_exact : also solve the output-space assignment problem, so that
        the tightness of the bound is measured rather than assumed.
    output_weight : quadrature weight w in ||u||^2 = w * sum_p u_p^2 (h^2 on a
        grid, 1 for plain vectors); used by both computations.
    """
    Z_data, Z_gen = np.asarray(Z_data, float), np.asarray(Z_gen, float)
    if len(Z_data) != len(Z_gen):
        raise ValueError("plan_pullback needs equal sample sizes so the latent "
                         "OT problem is an exact assignment problem")
    n = len(Z_data)
    C = cdist(Z_data, Z_gen, "sqeuclidean")
    i, j = linear_sum_assignment(C)
    w2_lat = float(np.sqrt(C[i, j].mean()))

    U = np.asarray(decode(Z_data), float).reshape(n, -1)
    V = np.asarray(decode(Z_gen), float).reshape(n, -1)
    diff = U[i] - V[j]
    ppc = float(np.sqrt((output_weight * np.sum(diff ** 2, 1)).mean()))

    exact = None
    if compute_exact:
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u, v>: one product, no n x n x P array
        G = output_weight * (U @ V.T)
        nu = output_weight * np.sum(U ** 2, 1)
        nv = output_weight * np.sum(V ** 2, 1)
        Co = nu[:, None] + nv[None, :] - 2.0 * G
        ii, jj = linear_sum_assignment(Co)
        exact = float(np.sqrt(np.maximum(Co[ii, jj], 0).mean()))

    warn: List[str] = []
    if exact is not None and ppc < exact - 1e-9:
        warn.append("bound below the exact W2, which is impossible; check that "
                    "output_weight matches between the two computations.")
    if L_cert is not None and L_cert * w2_lat < ppc - 1e-9:
        warn.append("Lipschitz bound below the plan-pullback bound: L_cert is "
                    "not a valid Lipschitz constant of the decoder.")
    return PPCReport(ppc, exact,
                     None if L_cert is None else L_cert * w2_lat,
                     w2_lat, n, warn)


# Mean ratio of the two-sample distance W2(X_N, X'_N) to the distance between
# X_N and a large reference sample, on isotropic Gaussians; keys are (d, N).
# Source: results/two_sample_vs_one_sample.json.
TWO_SAMPLE_VS_ONE_SAMPLE = {
    (2, 256): 1.160, (4, 256): 1.106, (8, 256): 1.066,
    (16, 256): 1.027, (32, 256): 1.017,
    (2, 1024): 1.040, (8, 1024): 1.025, (32, 1024): 1.007,
}

# Reference tables on isotropic Gaussian laws, 20 replicates per cell:
#   FLOOR_RATIO[N][d]          mean of W2(X_N, X'_N) / sqrt(d)
#   D_HAT_CALIBRATION[N][d]    mean slope estimate d_hat at nominal dimension d
# with their per-cell standard deviations. Generated by
# tools/write_library_tables.py from results/reference_tables.json; do not
# edit by hand.
# <<REFERENCE-TABLES-BEGIN>>
FLOOR_RATIO = {
    256: {1: 0.1581, 2: 0.2360, 4: 0.4354, 8: 0.6807, 16: 0.8948, 32: 1.0520, 64: 1.1609, 128: 1.2385},
    1024: {1: 0.0751, 2: 0.1374, 4: 0.3153, 8: 0.5676, 16: 0.8038, 32: 0.9858, 64: 1.1171, 128: 1.2076},
    2048: {1: 0.0517, 2: 0.1074, 4: 0.2677, 8: 0.5194, 16: 0.7650, 32: 0.9574, 64: 1.0972, 128: 1.1940},
    4096: {1: 0.0372, 2: 0.0783, 4: 0.2265, 8: 0.4753, 16: 0.7288, 32: 0.9312, 64: 1.0779, 128: 1.1805},
    8192: {1: 0.0307, 2: 0.0588, 4: 0.1914, 8: 0.4351, 16: 0.6941, 32: 0.9051, 64: 1.0597, 128: 1.1679},
    16384: {1: 0.0213, 2: 0.0444, 4: 0.1618, 8: 0.3987, 16: 0.6617, 32: 0.8804, 64: 1.0420, 128: 1.1556},
    32768: {1: 0.0130, 2: 0.0332, 4: 0.1367, 8: 0.3657, 16: 0.6310, 32: 0.8571, 64: 1.0250, 128: 1.1438},
}
FLOOR_RATIO_SD = {
    256: {1: 0.0570, 2: 0.0272, 4: 0.0137, 8: 0.0110, 16: 0.0056, 32: 0.0062, 64: 0.0041, 128: 0.0039},
    1024: {1: 0.0179, 2: 0.0089, 4: 0.0033, 8: 0.0045, 16: 0.0033, 32: 0.0038, 64: 0.0022, 128: 0.0017},
    2048: {1: 0.0124, 2: 0.0057, 4: 0.0030, 8: 0.0027, 16: 0.0019, 32: 0.0019, 64: 0.0013, 128: 0.0014},
    4096: {1: 0.0079, 2: 0.0036, 4: 0.0018, 8: 0.0012, 16: 0.0017, 32: 0.0008, 64: 0.0012, 128: 0.0008},
    8192: {1: 0.0073, 2: 0.0033, 4: 0.0012, 8: 0.0011, 16: 0.0012, 32: 0.0007, 64: 0.0006, 128: 0.0006},
    16384: {1: 0.0037, 2: 0.0020, 4: 0.0006, 8: 0.0007, 16: 0.0006, 32: 0.0006, 64: 0.0005, 128: 0.0004},
    32768: {1: 0.0025, 2: 0.0017, 4: 0.0004, 8: 0.0007, 16: 0.0004, 32: 0.0003, 64: 0.0004, 128: 0.0003},
}
FLOOR_RATIO_NREP = 20
D_HAT_CALIBRATION = {
    256: {2: 2.838, 4: 4.335, 8: 7.598, 16: 12.517, 32: 19.875, 64: 31.388, 128: 48.666},
    1024: {2: 2.674, 4: 4.406, 8: 7.457, 16: 13.064, 32: 21.607, 64: 35.159, 128: 53.781},
    2048: {2: 2.517, 4: 4.211, 8: 7.739, 16: 13.487, 32: 22.176, 64: 36.394, 128: 56.440},
    4096: {2: 2.568, 4: 4.207, 8: 7.786, 16: 13.739, 32: 23.208, 64: 38.349, 128: 59.316},
    8192: {2: 2.418, 4: 4.212, 8: 7.747, 16: 13.920, 32: 23.909, 64: 38.473, 128: 60.861},
    16384: {2: 2.475, 4: 4.135, 8: 7.850, 16: 14.270, 32: 24.355, 64: 39.672, 128: 62.867},
    32768: {2: 2.375, 4: 4.153, 8: 7.850, 16: 14.426, 32: 24.963, 64: 40.963, 128: 65.512},
}
D_HAT_CALIBRATION_SD = {
    256: {2: 0.614, 4: 0.684, 8: 1.233, 16: 1.399, 32: 2.474, 64: 4.472, 128: 7.479},
    1024: {2: 0.489, 4: 0.350, 8: 0.482, 16: 1.116, 32: 2.080, 64: 3.047, 128: 5.029},
    2048: {2: 0.335, 4: 0.226, 8: 0.260, 16: 0.596, 32: 1.306, 64: 2.494, 128: 3.219},
    4096: {2: 0.290, 4: 0.174, 8: 0.271, 16: 0.585, 32: 1.085, 64: 1.466, 128: 2.161},
    8192: {2: 0.213, 4: 0.140, 8: 0.162, 16: 0.342, 32: 0.623, 64: 0.931, 128: 1.881},
    16384: {2: 0.158, 4: 0.090, 8: 0.122, 16: 0.263, 32: 0.486, 64: 0.652, 128: 1.859},
    32768: {2: 0.174, 4: 0.065, 8: 0.101, 16: 0.232, 32: 0.403, 64: 0.683, 128: 1.261},
}
D_HAT_CALIBRATION_NREP = 20
TABULATED_N = (256, 1024, 2048, 4096, 8192, 16384, 32768)
# <<REFERENCE-TABLES-END>>

TABLE_SOURCE = "results/reference_tables.json"


def _interp_loglog(x, xs, ys):
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def _calib_arrays(N: int):
    """Nominal grid and expected d_hat at N, interpolated in (log N, log d)."""
    N = float(max(N, 1.0))
    cal = D_HAT_CALIBRATION
    Ns = np.array(sorted(cal.keys()), dtype=float)
    dn = np.array(sorted(cal[int(Ns[0])]), dtype=float)
    dh = np.empty_like(dn)
    logN, logNs = np.log(N), np.log(Ns)
    for j, d in enumerate(dn):
        y = np.array([cal[int(nk)][int(d)] for nk in Ns], dtype=float)
        dh[j] = np.exp(np.interp(logN, logNs, np.log(y)))
    return dn, dh


def nominal_to_measured(d_nominal: float, N: int) -> float:
    """Expected d_hat of an isotropic Gaussian of nominal dimension d at size N."""
    dn, dh = _calib_arrays(N)
    return _interp_loglog(max(float(d_nominal), 1e-9), dn, dh)


def measured_to_nominal(d_hat: float, N: int) -> float:
    """Nominal dimension whose expected d_hat at size N equals `d_hat`.

    The calibration is monotone over the tabulated range, so the inverse is
    well defined there; outside it the value is clamped.
    """
    dn, dh = _calib_arrays(N)
    return _interp_loglog(max(float(d_hat), 1e-9), dh, dn)


def floor_ratio(d_star: float, N: int) -> float:
    """Floor ratio r(d, N) at nominal dimension `d_star`.

    Interpolates FLOOR_RATIO in (log N, log d) and clamps outside the table.
    For a measured d_hat use `floor_ratio_measured`.
    """
    d_star = float(max(d_star, 1.0))
    N = float(max(N, 1.0))
    Ns = np.array(sorted(FLOOR_RATIO.keys()), dtype=float)
    r_at_N = np.empty(len(Ns))
    logd = np.log(d_star)
    for i, nkey in enumerate(Ns):
        tbl = FLOOR_RATIO[int(nkey)]
        ds = np.array(sorted(tbl), dtype=float)
        rs = np.array([tbl[int(x)] for x in ds], dtype=float)
        r_at_N[i] = np.exp(np.interp(logd, np.log(ds), np.log(rs)))
    return float(np.exp(np.interp(np.log(N), np.log(Ns), np.log(r_at_N))))


def floor_ratio_measured(d_hat: float, N: int) -> float:
    """Floor ratio at a measured slope estimate `d_hat`.

    Maps d_hat to nominal dimension through D_HAT_CALIBRATION, then reads
    FLOOR_RATIO. This is the function to use on real data.
    """
    return floor_ratio(measured_to_nominal(d_hat, N), N)


def vacuity_margin(d_hat: float, N: int) -> dict:
    """Distance from vacuity of a data set with measured dimension `d_hat`.

    Returns the margin in the units that decide vacuity, 1/r, together with
    the margin in dimension units (crossing / nominal dimension). Because r is
    flat in d near the crossing, the dimension-unit margin is several times
    larger; the r-unit margin is the relevant one.
    """
    r = floor_ratio_measured(float(d_hat), int(N))
    lim = certification_limit(int(N), with_uncertainty=False)
    d_nom = measured_to_nominal(float(d_hat), int(N))
    return {
        "d_hat": float(d_hat), "N": int(N),
        "d_nominal": float(d_nom),
        "r": float(r),
        "margin_r_units": float(1.0 / r) if r > 0 else float("inf"),
        "margin_d_units": (float(lim["d_star_limit"] / d_nom)
                           if d_nom > 0 else float("nan")),
        "d_star_limit_nominal": float(lim["d_star_limit"]),
        "d_star_limit_measured": float(lim["d_star_limit_measured"]),
        "in_tabulated_range": bool(lim["in_tabulated_range"]),
    }


def certification_limit(N: int, d_grid: Optional[np.ndarray] = None,
                        with_uncertainty: bool = True, n_boot: int = 4000,
                        seed: int = 0) -> dict:
    """Largest dimension at which the floor ratio r(d, N) is below 1.

    Returns two limits in different units:

      d_star_limit           nominal dimension, the index of FLOOR_RATIO;
      d_star_limit_measured  the same limit in units of the slope estimate
                             d_hat, the quantity measured on real data.

    Compare a measured d_hat with `d_star_limit_measured`. With
    `with_uncertainty`, every cell of FLOOR_RATIO and D_HAT_CALIBRATION is
    perturbed by its standard error, and the 90% intervals of the limit are
    returned as `d_star_limit_ci90` and `d_star_limit_measured_ci90`.

    For N outside TABULATED_N both tables are clamped: the limit is the one
    measured at the nearest tabulated N, `in_tabulated_range` is False and a
    warning says so.
    """
    TN = TABULATED_N
    if d_grid is None:
        dmax = max(max(v) for v in FLOOR_RATIO.values())
        d_grid = np.exp(np.linspace(np.log(1), np.log(dmax), 4000))
    r = np.array([floor_ratio(d, N) for d in d_grid])
    Ns, dr, R, _ = _table_array(FLOOR_RATIO, FLOOR_RATIO_SD, FLOOR_RATIO_NREP)
    lim = crossing_loglog(dr, _row_at(R, Ns, N))
    lim_meas = (nominal_to_measured(lim, N) if np.isfinite(lim)
                else float("nan"))

    warnings_: List[str] = []
    if N < min(TN) or N > max(TN):
        edge = min(TN) if N < min(TN) else max(TN)
        warnings_.append(
            f"N={N} is outside the tabulated range {TN}; the tables are "
            f"clamped, so this is the limit measured at N={edge}. Above the "
            "table the true limit is larger.")

    out = {"N": N, "d_grid": d_grid, "r": r,
           "d_star_limit": lim,
           "d_star_limit_measured": lim_meas,
           "in_tabulated_range": bool(min(TN) <= N <= max(TN)),
           "table_source": TABLE_SOURCE,
           "warnings": warnings_}

    if with_uncertainty and np.isfinite(lim):
        out.update(_limit_interval(N, n_boot, seed))
    return out


def crossing_loglog(ds, rs) -> float:
    """Largest d at which the floor ratio is below 1, with the ratio
    interpolated linearly in (log d, log r) between the tabulated points;
    nan when the ratio is below 1 nowhere or up to the last point."""
    ld, lr = np.log(np.asarray(ds, float)), np.log(np.asarray(rs, float))
    below = np.where(lr < 0.0)[0]
    if len(below) == 0 or below[-1] == len(ld) - 1:
        return float("nan")
    i = int(below[-1])
    return float(np.exp(ld[i] - lr[i] * (ld[i + 1] - ld[i]) / (lr[i + 1] - lr[i])))


def _row_at(T, Ns, N):
    """A table interpolated to size N in (log N, log value), clamped."""
    logN = np.log(max(float(N), 1.0))
    return np.exp([np.interp(logN, np.log(Ns), np.log(T[:, j]))
                   for j in range(T.shape[1])])


def _table_array(tab, sd, nrep):
    Ns = np.array(sorted(tab), float)
    ds = np.array(sorted(tab[int(Ns[0])]), float)
    mean = np.array([[tab[int(n)][int(d)] for d in ds] for n in Ns])
    se = np.array([[sd[int(n)][int(d)] for d in ds] for n in Ns]) / np.sqrt(nrep)
    return Ns, ds, mean, se


def _limit_interval(N: int, n_boot: int, seed: int) -> dict:
    """90% intervals of the limit with both tables perturbed.

    Every cell of FLOOR_RATIO and D_HAT_CALIBRATION is perturbed by its
    standard error, each perturbed row is made monotone in d, the tables are
    interpolated to N, and the crossing is located again and mapped to
    measured units.
    """
    rng = np.random.default_rng(seed)
    Ns, dr, R, RS = _table_array(FLOOR_RATIO, FLOOR_RATIO_SD, FLOOR_RATIO_NREP)
    Nc, dc, H, HS = _table_array(D_HAT_CALIBRATION, D_HAT_CALIBRATION_SD,
                                 D_HAT_CALIBRATION_NREP)
    nom, meas = [], []
    for _ in range(n_boot):
        Rb = np.maximum.accumulate(np.maximum(R + rng.normal(size=R.shape) * RS,
                                              1e-6), axis=1)
        Hb = np.maximum.accumulate(np.maximum(H + rng.normal(size=H.shape) * HS,
                                              1e-3), axis=1)
        lim_b = crossing_loglog(dr, _row_at(Rb, Ns, N))
        if not np.isfinite(lim_b):
            continue
        nom.append(lim_b)
        meas.append(_interp_loglog(lim_b, dc, _row_at(Hb, Nc, N)))
    nom, meas = np.asarray(nom), np.asarray(meas)
    if meas.size < 100:
        nan = (float("nan"), float("nan"))
        return {"d_star_limit_ci90": nan, "d_star_limit_measured_ci90": nan,
                "d_star_limit_measured_sd": float("nan")}
    return {"d_star_limit_ci90": (float(np.quantile(nom, 0.05)),
                                  float(np.quantile(nom, 0.95))),
            "d_star_limit_measured_ci90": (float(np.quantile(meas, 0.05)),
                                           float(np.quantile(meas, 0.95))),
            "d_star_limit_measured_sd": float(meas.std(ddof=1))}
