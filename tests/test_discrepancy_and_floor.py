"""Two-sample discrepancies and the resolution floor."""
import numpy as np
import pytest

import certgen as cg

RNG = np.random.default_rng(0)
A, B = RNG.normal(size=(300, 5)), RNG.normal(size=(300, 5))
C = RNG.normal(size=(300, 5)) + 2.0


@pytest.mark.parametrize("metric", ["w2", "energy", "mmd", "sliced_w2"])
def test_same_law_closer_than_shifted(metric):
    f = cg.get_discrepancy(metric)
    assert f(A, B, seed=1) < f(A, C, seed=1)


def test_w2_self_distance_is_zero():
    assert cg.w2_exact(A, A.copy()) < 1e-9


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        cg.get_discrepancy("nope")


def test_w2_warns_when_inputs_are_subsampled():
    X = RNG.normal(size=(60, 2))
    with pytest.warns(RuntimeWarning):
        cg.w2_exact(X, X[::-1], max_n=40)


def test_floor_is_positive_and_decreases_with_n():
    X = cg.data.gaussian_copula(2400, 8, seed=1)
    small = cg.resolution_floor(X, 300, "w2", n_rep=12)
    large = cg.resolution_floor(X, 900, "w2", n_rep=8)
    assert small.median > 0
    assert large.median < small.median


def test_floor_needs_disjoint_splits():
    X = cg.data.gaussian_copula(100, 3, seed=1)
    with pytest.raises(ValueError):
        cg.resolution_floor(X, 60)


def test_monte_carlo_p_values_are_never_zero():
    fl = cg.resolution_floor(cg.data.gaussian_copula(800, 4, seed=3), 200, "w2", n_rep=20)
    assert fl.p_value(1e9) == pytest.approx(1.0 / (fl.n_rep + 1))
    assert fl.p_value_low(-1.0) == pytest.approx(1.0 / (fl.n_rep + 1))
    assert 0.0 < fl.p_value(fl.median) <= 1.0


def test_mdd_scales_with_level_and_power():
    fl = cg.resolution_floor(cg.data.gaussian_copula(1200, 6, seed=11), 300, "w2", n_rep=20)
    assert fl.mdd(2.0) > fl.mdd(1.0) > 0
    assert fl.mdd(1.0, power=0.9) > fl.mdd(1.0, power=0.8)
    # (z_0.975 + z_0.8) times the proxy standard deviation
    assert fl.mdd(1.0) == pytest.approx(2.8015852 * fl.sd_difference_at(1.0), rel=1e-6)


def test_equivalence_at_zero_margin_is_the_two_sided_floor_test():
    X = cg.data.gaussian_copula(2400, 6, seed=3)
    fl = cg.resolution_floor(X, 600, n_rep=20, seed=11)
    D = float(cg.w2_exact(X[:600], cg.data.gaussian_copula(600, 6, seed=4242)))
    r = fl.equivalence(D, 0.0)
    assert r["p_too_bad"] == pytest.approx(fl.p_value_low(D))
    assert r["p_too_good"] == pytest.approx(fl.p_value(D))


def test_equivalence_rejects_a_reference_copy():
    X = cg.data.gaussian_copula(2400, 6, seed=3)
    fl = cg.resolution_floor(X, 600, n_rep=20, seed=11)
    assert not fl.equivalence(0.0, 0.25 * fl.median)["equivalent"]


def test_unbiased_statistics_have_no_floor_law():
    X = cg.data.gaussian_copula(800, 4, seed=2)
    with pytest.raises(ValueError):
        cg.fit_floor_law(X, metric="mmd", n_rep=4)
