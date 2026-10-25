"""Floor-corrected reporting and leakage screening."""
import numpy as np
import pytest

import certgen as cg


@pytest.fixture(scope="module")
def endpoints():
    X = cg.data.gaussian_copula(2000, 8, seed=2)
    gens = {
        "fresh": cg.data.gaussian_copula(1000, 8, seed=99),
        "T2": cg.baseline_gaussian(X, 1000, 0),
        "T1": cg.baseline_marginals(X, 1000, 0),
        "T0": cg.baseline_mean(X, 1000, 0),
    }
    return {k: cg.evaluate(X, g, n_per_side=500, n_rep=10) for k, g in gens.items()}


def test_skill_endpoints(endpoints):
    assert abs(endpoints["fresh"].V["T1_marginals"]) < 0.35
    assert abs(endpoints["T1"].V["T1_marginals"] - 1) < 0.35


def test_reference_generators_are_ordered(endpoints):
    d = {k: r.observed for k, r in endpoints.items()}
    assert d["T0"] > d["T1"] > d["T2"] > d["fresh"]


def test_fresh_draw_is_indistinguishable(endpoints):
    assert endpoints["fresh"].p_value > 0.05
    assert endpoints["fresh"].valid


def test_skill_interval_brackets_the_skill(endpoints):
    r = endpoints["T2"]
    lo, hi = r.skill_ci()["T1_marginals"]
    assert lo <= r.skill["T1_marginals"] <= hi


def test_column_mismatch_raises():
    X = cg.data.gaussian_copula(400, 6, seed=1)
    with pytest.raises(ValueError):
        cg.evaluate(X, X[:, :4])


def test_floor_and_observed_distance_use_the_same_size():
    X = cg.data.gaussian_copula(1600, 6, seed=17)
    with pytest.raises(ValueError, match="generated sample"):
        cg.evaluate(X, X[:200], n_per_side=500, n_rep=6, seed=0)


@pytest.fixture(scope="module")
def memorisation():
    X = cg.data.gaussian_copula(1200, 6, seed=11)
    return {"subset": cg.evaluate(X, X[:600], n_per_side=400, n_rep=10),
            "copy": cg.evaluate(X, X.copy(), n_per_side=400, n_rep=10),
            "fresh": cg.evaluate(X, cg.data.gaussian_copula(1200, 6, seed=777),
                                 n_per_side=400, n_rep=10)}


def test_memorisers_are_flagged(memorisation):
    assert memorisation["subset"].leakage_suspected
    assert memorisation["copy"].leakage_suspected
    assert not memorisation["subset"].valid


def test_fresh_draw_is_not_flagged(memorisation):
    assert not memorisation["fresh"].leakage_suspected
    assert memorisation["fresh"].valid


def test_exact_copy_overlaps_far_more_than_the_null(memorisation):
    lk = memorisation["copy"].leakage
    assert lk.overlap_ratio > 3.0 or np.isinf(lk.overlap_ratio)


def test_t_self_is_a_ceiling():
    rep = cg.evaluate(cg.data.gaussian_copula(3000, 6, seed=11),
                      cg.data.gaussian_copula(1500, 6, seed=555),
                      n_per_side=600, n_rep=12)
    assert rep.d_self >= 0.97 * rep.floor.median


def test_jittered_memoriser_is_caught():
    X = cg.Standardizer().fit_transform(cg.data.gaussian_copula(3000, 6, seed=11))
    noise = np.random.default_rng(0).normal(0, 0.05, size=(800, 6))
    assert cg.leakage_check(X, X[:800] + noise, seed=0).suspected


def test_exact_memoriser_is_caught_on_a_large_table():
    X = cg.Standardizer().fit_transform(cg.data.gaussian_copula(30000, 6, seed=3))
    assert cg.leakage_check(X, X[:500], seed=0).suspected


def _categorical(n, columns, levels, seed):
    return np.random.default_rng(seed).integers(0, levels, size=(n, columns)).astype(float)


def test_fresh_draw_from_a_discrete_law_is_not_flagged():
    X = _categorical(3000, 6, 3, 1)
    st = cg.Standardizer().fit(X)
    rep = cg.leakage_check(st.transform(X), st.transform(_categorical(800, 6, 3, 999)),
                           seed=0)
    assert not rep.suspected
    assert any("cannot be distinguished" in w for w in rep.warnings)


def test_memoriser_on_a_sparse_discrete_support_is_caught():
    X = _categorical(3000, 8, 5, 1)
    Z = cg.Standardizer().fit_transform(X)
    assert cg.leakage_check(Z, Z[:800], seed=0).suspected
