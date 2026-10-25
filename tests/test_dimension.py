"""Intrinsic-dimension estimators."""
import numpy as np
import pytest
from scipy import stats

import certgen as cg
from conftest import artifact


@pytest.mark.parametrize("m", [2, 4, 8])
def test_some_estimator_recovers_the_manifold_dimension(m):
    Y = cg.data.curved_manifold(m, 24, seed=m).sample(2048, seed=0)
    rep = cg.intrinsic_dimension(Y)
    best = min(abs(rep.wasserstein - m), abs(rep.twonn - m), abs(rep.mle - m)) / m
    assert best < 0.35


def test_disagreement_is_reported():
    rep = cg.intrinsic_dimension(cg.data.curved_manifold(2, 24, seed=2).sample(2048, seed=0))
    assert rep.spread <= 2.0 or any("disagree" in w for w in rep.warnings)


def test_small_samples_are_not_usable_for_sizing():
    Y = cg.data.curved_manifold(4, 24, seed=4).sample(512)
    assert not cg.intrinsic_dimension(Y).usable_for_sizing


@pytest.fixture(scope="module")
def quantised():
    Xc = cg.Standardizer().fit_transform(cg.data.gaussian_copula(3000, 6, seed=5))
    Xq = np.empty_like(Xc)
    for j in range(Xc.shape[1]):
        q = np.unique(np.quantile(Xc[:, j], np.linspace(0, 1, 11)))
        i = np.clip(np.searchsorted(q, Xc[:, j]) - 1, 0, len(q) - 2)
        Xq[:, j] = 0.5 * (q[i] + q[i + 1])
    return (cg.intrinsic_dimension(Xc, n_boot=2),
            cg.intrinsic_dimension(cg.Standardizer().fit_transform(Xq), n_boot=2))


def test_ties_are_detected(quantised):
    continuous, lattice = quantised
    assert continuous.tie_fraction < 0.01 < lattice.tie_fraction
    assert lattice.lattice_suspected and not continuous.lattice_suspected
    assert not lattice.usable_for_sizing


def test_maximum_likelihood_estimate_survives_ties(quantised):
    assert quantised[1].mle > 3.0


def test_uncertainty_fields():
    rep = cg.intrinsic_dimension(cg.data.gaussian_copula(2500, d=5, seed=19),
                                 n_boot=8, data_bootstrap=4)
    assert np.isfinite(rep.wasserstein_sd)
    assert rep.wasserstein_se < rep.wasserstein_sd
    assert np.isfinite(rep.wasserstein_sd_subsample)


def test_exponent_interval_uses_the_t_quantile():
    X = cg.data.curved_manifold(4, D=12, seed=3).sample(2048, seed=1)
    rep = cg.intrinsic_dimension(X, n_boot=6)
    assert rep.d_ci90_quantile == pytest.approx(stats.t.ppf(0.95, 5))
    assert rep.d_ci90_from_exponent[0] < 1.0 / rep.exponent < rep.d_ci90_from_exponent[1]


def test_saturation_table_equals_its_artifact():
    cells = artifact("slope_estimator_bias.json")["cells"]
    assert {c["nominal_d"]: round(c["d_hat"], 3) for c in cells} == \
        cg.EXPONENT_SATURATION_TABLE


def test_standardizer_inverse():
    X = cg.data.curved_manifold(4, D=12, seed=3).sample(500, seed=1) * 3.0 + 7.0
    st = cg.Standardizer().fit(X)
    assert np.allclose(st.inverse_transform(st.transform(X)), X)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_twonn_recovers_the_dimension_of_a_gaussian(d):
    X = np.random.default_rng(d).normal(size=(4000, d))
    assert cg.twonn_dimension(X) == pytest.approx(d, rel=0.08)
