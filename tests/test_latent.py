"""Plan-pullback bound, reference tables and the certification limit."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import certgen as cg
from certgen import latent
from conftest import ROOT, artifact


@pytest.fixture(scope="module")
def decoder():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(16, 64)) * 0.4
    return (lambda Z: np.tanh(Z @ W)), float(np.linalg.svd(W, compute_uv=False)[0])


def test_plan_pullback_is_valid_and_tighter_than_lipschitz(decoder):
    decode, L = decoder
    rng = np.random.default_rng(1)
    Zd = rng.normal(size=(300, 16))
    Zg = rng.normal(size=(300, 16)) * 1.08 + 0.15
    rep = cg.plan_pullback(Zd, Zg, decode, L_cert=L)
    assert rep.ppc_bound >= rep.exact_w2_output - 1e-9
    assert rep.ppc_bound < rep.lipschitz_bound
    assert rep.tightness < 3.0


def test_plan_pullback_needs_equal_sizes(decoder):
    decode, _ = decoder
    Z = np.zeros((10, 16))
    with pytest.raises(ValueError):
        cg.plan_pullback(Z, Z[:5], decode)


def test_tightness_edge_cases():
    assert np.isinf(cg.PPCReport(1.2, 0.0, None, 0.5, 40).tightness)
    assert cg.PPCReport(1.2, None, None, 0.5, 40).tightness is None


def test_library_tables_equal_their_artifact():
    a = artifact("reference_tables.json")
    for c in a["r_cells"]:
        if c["N"] in latent.TABULATED_N:
            assert latent.FLOOR_RATIO[c["N"]][c["d"]] == round(c["r_mean"], 4)
            assert latent.FLOOR_RATIO_SD[c["N"]][c["d"]] == round(c["r_sd"], 4)
    for c in a["dhat_cells"]:
        if c["N"] in latent.TABULATED_N:
            assert latent.D_HAT_CALIBRATION[c["N"]][c["d"]] == round(c["d_hat_mean"], 3)
            assert latent.D_HAT_CALIBRATION_SD[c["N"]][c["d"]] == round(c["d_hat_sd"], 3)


def test_library_tables_are_generated_from_the_artifact():
    import write_library_tables
    assert write_library_tables.main(["--check"]) == 0


def test_two_sample_ratios_equal_their_artifact():
    cells = artifact("two_sample_vs_one_sample.json")["cells"]
    stored = {(c["d"], c["N"]): round(c["ratio_mean"], 3) for c in cells}
    assert stored == latent.TWO_SAMPLE_VS_ONE_SAMPLE


def test_floor_ratio_interpolates_between_sizes():
    assert cg.floor_ratio(16, 1024) == pytest.approx(latent.FLOOR_RATIO[1024][16])
    values = [cg.floor_ratio(16, N) for N in (256, 512, 768, 1024)]
    assert values == sorted(values, reverse=True)


def test_certification_limit_grows_with_n():
    limits = [cg.certification_limit(N, with_uncertainty=False)["d_star_limit"]
              for N in (256, 512, 1024, 2048)]
    assert limits == sorted(limits)


def test_measured_units_limit_and_its_interval():
    c = cg.certification_limit(1024)
    assert c["d_star_limit_measured"] < c["d_star_limit"]
    lo, hi = c["d_star_limit_measured_ci90"]
    assert lo < c["d_star_limit_measured"] < hi


def test_limit_matches_the_crossings_of_the_artifact():
    for c in artifact("reference_tables.json")["crossings"]:
        if c["N"] in latent.TABULATED_N:
            lim = cg.certification_limit(c["N"], with_uncertainty=False)
            assert lim["d_star_limit"] == pytest.approx(c["d_star_limit_nominal"], abs=0.15)


def test_out_of_table_sizes_are_clamped_and_flagged():
    top = max(latent.TABULATED_N)
    out = cg.certification_limit(2 * top, with_uncertainty=False)
    edge = cg.certification_limit(top, with_uncertainty=False)
    assert not out["in_tabulated_range"] and out["warnings"]
    assert out["d_star_limit_measured"] == pytest.approx(edge["d_star_limit_measured"])


def test_calibration_round_trips_and_is_monotone():
    assert cg.measured_to_nominal(cg.nominal_to_measured(16.0, 1024), 1024) == \
        pytest.approx(16.0, abs=0.5)
    assert cg.nominal_to_measured(32, 1024) > cg.nominal_to_measured(8, 1024)
    assert cg.floor_ratio_measured(16.0, 1024) == pytest.approx(
        cg.floor_ratio(cg.measured_to_nominal(16.0, 1024), 1024))


def test_vacuity_margin_units():
    m = cg.vacuity_margin(9.0, 4000)
    assert m["margin_d_units"] > m["margin_r_units"] > 1.0


def test_command_line_interface():
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "certgen", "limit", "2048", "--json"],
                       capture_output=True, encoding="utf-8", cwd=ROOT, env=env,
                       timeout=300)
    assert p.returncode == 0
    assert "d_star_limit_measured" in json.loads(p.stdout)


def test_crossing_is_exact_under_log_linear_interpolation():
    ds = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    rs = 0.25 * ds ** 0.75                       # r = 1 at d = 4^(4/3)
    assert latent.crossing_loglog(ds, rs) == pytest.approx(4.0 ** (4.0 / 3.0), rel=1e-12)
    assert np.isnan(latent.crossing_loglog(ds, 0.1 * np.ones(5)))
    assert np.isnan(latent.crossing_loglog(ds, 2.0 * np.ones(5)))


def test_limit_interval_contains_the_limit():
    lim = cg.certification_limit(3000, n_boot=600)
    lo, hi = lim["d_star_limit_ci90"]
    assert lo < lim["d_star_limit"] < hi
    lo, hi = lim["d_star_limit_measured_ci90"]
    assert lo < lim["d_star_limit_measured"] < hi
