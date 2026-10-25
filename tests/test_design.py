"""Floor laws, the curvature test, sizing, intervals and severity."""
import numpy as np
import pytest
from scipy import stats

import certgen as cg
from certgen.design import INTERVAL_LEVEL_CALIBRATION
from conftest import artifact


def test_floor_law_on_a_copula(law6):
    assert law6.r2 > 0.90
    assert 0 < law6.d_eff < 100
    assert len(law6.n_grid) >= 6
    assert law6.floor_at(4000) < law6.floor_at(400)
    assert law6.single_exponent_ok


def test_admissibility_rule(law6):
    p = law6.drift_p_value
    assert 0.0 < p <= 1.0
    assert law6.single_exponent_ok == (law6.r2 > 0.95 and np.isfinite(p) and p >= 0.05)


def test_sizing_functions(law6):
    small, large = cg.resolvable_effect(law6, 500), cg.resolvable_effect(law6, 5000)
    assert np.isfinite(small) and large < small
    assert np.isfinite(cg.power_curve(law6, effect_size=small / 2)["n_required"])
    n = cg.required_n(law6, law6.floor_at(2000) * 0.5)
    assert np.isfinite(n) and n > 0
    assert (cg.resolvable_effect(law6, 1000, reference_level=2.0)
            > cg.resolvable_effect(law6, 1000, reference_level=1.0))


def test_resolvable_effect_uses_the_median_coefficient_of_variation(law6):
    cv = np.median(law6.stds / law6.floors)
    assert law6.std_at(3000) == pytest.approx(cv * law6.floor_at(3000), rel=1e-12)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    assert cg.resolvable_effect(law6, 3000) == pytest.approx(
        z * np.sqrt(2.0) * cv * law6.floor_at(3000), rel=1e-12)


def test_weighted_fit_uses_inverse_variance_weights():
    """The weights multiply the squared residuals, as in the manuscript."""
    from certgen.design import _wls_line
    rng = np.random.default_rng(3)
    x = np.log([125.0, 250, 500, 1000, 2000, 4000])
    y = 1.2 - x / 5 + rng.normal(0.0, 0.02, x.size)
    w = rng.uniform(1.0, 50.0, x.size)
    X = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    assert _wls_line(x, y, w) == pytest.approx((slope, intercept), rel=1e-10)


def test_profile_fit_reproduces_the_fit_on_data(law6):
    law = cg.floor_law_from_profile(law6.n_grid, law6.floors, law6.stds,
                                    se_log=law6.se_log, n_rep=law6.n_rep)
    for a in ("C", "d_eff", "r2", "d_eff_low", "d_eff_high", "drift_p_value"):
        assert getattr(law, a) == pytest.approx(getattr(law6, a), rel=1e-12)


def test_every_sizing_function_is_withheld_below_the_rate_boundary():
    n = np.array([125.0, 250, 500, 1000, 2000, 4000])
    f = 2.0 * n ** (-1.0 / 3.0)
    law = cg.floor_law_from_profile(n, f, 0.03 * f, n_rep=12)
    assert law.single_exponent_ok and law.d_eff == pytest.approx(3.0)
    assert not np.isfinite(cg.required_n(law, law.floor_at(8000.0)))
    assert not np.isfinite(cg.resolvable_effect(law, 1000))
    assert not np.isfinite(cg.power_curve(law, 0.01)["n_required"])


def _with_p(law, p):
    return cg.FloorLaw(C=law.C, d_eff=law.d_eff, n_grid=law.n_grid,
                       floors=law.floors, stds=law.stds, metric=law.metric,
                       r2=law.r2, warnings=[], d_eff_low=law.d_eff_low,
                       d_eff_high=law.d_eff_high, drift_p_value=p,
                       n_rep=law.n_rep)


def test_sizing_is_withheld_for_an_inadmissible_profile(law6):
    bad = _with_p(law6, 0.01)
    target = law6.floor_at(law6.n_grid[-1]) * 0.5
    assert not bad.single_exponent_ok
    assert not np.isfinite(cg.required_n(bad, target))
    assert not np.isfinite(cg.resolvable_effect(bad, 1000))
    assert not np.isfinite(cg.power_curve(bad, 0.05)["n_required"])
    rng = bad.required_n_range(target)
    assert np.isfinite(rng["n_full_fit"]) and rng["ok"] is False


def test_missing_curvature_p_value_is_not_admissible(law6):
    assert not _with_p(law6, float("nan")).single_exponent_ok


def test_sizing_is_withheld_below_the_rate_boundary():
    X = cg.Standardizer().fit_transform(np.random.default_rng(21).normal(size=(2048, 3)))
    law = cg.fit_floor_law(X, n_rep=12, seed=0)
    assert law.d_eff < 4.0
    assert not np.isfinite(cg.required_n(law, law.floor_at(1000.0) * 0.5))
    iv = law.required_n_interval(law.floor_at(1000.0) * 0.5)
    assert not np.isfinite(iv["n_point"]) and np.isfinite(iv["n_lo"])


def test_curvature_test_over_rejects_below_the_rate_boundary():
    """Replicates the d = 2 cell of results/curvature_test_size.json."""
    rejections, p_values = 0, []
    for t in range(200):
        X = cg.Standardizer().fit_transform(
            np.random.default_rng(4000 + t).normal(size=(1024, 2)))
        law = cg.fit_floor_law(X, metric="w2", n_rep=12, seed=t)
        if np.isfinite(law.drift_p_value):
            p_values.append(law.drift_p_value)
            rejections += int(law.drift_p_value < 0.05)
    assert rejections / len(p_values) > 0.05
    assert np.median(p_values) > 0.25


def test_interval_brackets_the_prescription(gaussian_law):
    iv = gaussian_law.required_n_interval(gaussian_law.floor_at(4000), level=0.90, seed=0)
    assert iv["n_lo"] <= iv["n_point"] <= iv["n_hi"]
    assert 0.0 <= iv["discard_fraction"] <= 1.0


def test_interval_uses_the_t_scale_of_the_fit(gaussian_law):
    """The exponent interval matches the textbook t interval of the WLS fit."""
    law = gaussian_law
    iv = law.required_n_interval(law.floor_at(4000), level=0.90, seed=0)
    logn, logf = np.log(law.n_grid), np.log(law.floors)
    w = (law.floors / law.stds) ** 2
    X = np.column_stack([logn, np.ones_like(logn)])
    XtWX = X.T @ (X * w[:, None])
    beta = np.linalg.solve(XtWX, (X * w[:, None]).T @ logf)
    nu = len(logn) - 2
    s2 = float((w * (logf - X @ beta) ** 2).sum() / nu)
    se = float(np.sqrt((s2 * np.linalg.inv(XtWX))[0, 0]))
    t = stats.t.ppf(0.95, nu)
    lo, hi = sorted([-1.0 / (beta[0] - t * se), -1.0 / (beta[0] + t * se)])
    assert 0.75 <= (iv["d_hi"] - iv["d_lo"]) / (hi - lo) <= 1.35


def test_level_calibration(gaussian_law):
    law = gaussian_law
    keys = sorted(INTERVAL_LEVEL_CALIBRATION)
    assert all(INTERVAL_LEVEL_CALIBRATION[k] > k for k in keys)
    plain = law.required_n_interval(law.floor_at(4000), level=0.90, seed=0)
    cal = law.required_n_interval(law.floor_at(4000), level=0.90, seed=0, calibrate=True)
    assert cal["in_calibration_regime"] and cal["calibrated"]
    assert cal["applied_level"] == pytest.approx(
        np.interp(0.90, keys, [INTERVAL_LEVEL_CALIBRATION[k] for k in keys]))
    assert cal["n_lo"] <= plain["n_lo"] and cal["n_hi"] >= plain["n_hi"]


def test_calibration_is_not_applied_outside_its_regime():
    X = cg.Standardizer().fit_transform(np.random.default_rng(7).normal(size=(2048, 10)))
    law = cg.fit_floor_law(X, n_rep=3, seed=0)
    cal = law.required_n_interval(law.floor_at(4000), calibrate=True, seed=0)
    plain = law.required_n_interval(law.floor_at(4000), calibrate=False, seed=0)
    assert not cal["in_calibration_regime"] and not cal["calibrated"]
    assert isinstance(cal["calibration_refused"], str)
    assert cal["n_lo"] == plain["n_lo"] and cal["n_hi"] == plain["n_hi"]


def test_severity_thresholds_are_interpolated_with_an_interval():
    X = cg.Standardizer().fit_transform(np.random.default_rng(7).normal(size=(2048, 5)))
    law = cg.fit_floor_law(X, n_rep=12, seed=0)
    sv = [law.severity(seed=s) for s in (0, 1, 2)]
    d50 = [v["drift_50"] for v in sv]
    assert all(np.isfinite(d50))
    assert not all(x in sv[0]["drift_grid"] for x in d50)
    lo, hi = sv[0]["drift_50_ci90"]
    assert lo <= sv[0]["drift_50"] <= hi
    assert max(d50) - min(d50) < 0.5


def test_severity_panel_reproduces_the_stored_benchmark_result():
    from severity import law_from
    rec = next(r for r in artifact("benchmark_floor_laws.json")["records"]
               if r["dataset"] == "letter:E1")
    stored = next(t for t in artifact("severity.json")["tables"]
                  if t["dataset"] == "letter:E1")
    panel = law_from(rec).severity_panel()
    assert panel["drift_50"] == pytest.approx(stored["drift_50_median"], rel=1e-9)
    assert panel["drift_50_ci90"] == pytest.approx(stored["drift_50_ci90_combined"], rel=1e-9)


def test_library_constants_equal_their_artifacts():
    size = {round(c["d_eff_median"], 2): round(c["size"], 3)
            for c in artifact("curvature_test_size.json")["cells"]}
    assert size == cg.GATE_SIZE_BY_D_EFF
    power = {c["median_se"]: c["power"] for c in artifact("curvature_test_power.json")["cells"]
             if c["grid_kind"] == "densified" and c["n_points"] == 6}
    assert {k: round(v, 3) for k, v in power.items()} == pytest.approx(
        cg.GATE_POWER_GRID_6PT, abs=5e-3)
    inverse = artifact("interval_coverage.json")["level_calibration"][
        "inverse_map_requested_to_nominal"]
    assert {float(k): v for k, v in inverse.items()} == INTERVAL_LEVEL_CALIBRATION
    cov = artifact("interval_coverage.json")
    delivered = ([c["coverage_n_interior"] for c in cov["cells"]]
                 + [c["coverage_n_extrapolated"] for c in cov["cells"]])
    at090 = cov["level_calibration"]["delivered_by_level"]["0.9"]
    assert cg.design.INTERVAL_COVERAGE_AT_090 == pytest.approx(
        {"min": min(min(delivered), at090["min"]),
         "max": max(max(delivered), at090["max"]),
         "median": float(np.median(delivered))}, abs=5e-4)
