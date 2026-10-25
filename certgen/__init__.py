"""
certgen -- prospective design and floor-corrected reporting for evaluations of
generative models based on the 2-Wasserstein distance.

Main entry points
-----------------
resolution_floor      replicate distribution of the distance between two
                      disjoint real subsamples
fit_floor_law         floor(n) = C n^(-1/d_eff) with a curvature test
required_n            per-side size at which the floor reaches a target
FloorLaw.required_n_interval / required_n_range / severity_panel
resolvable_effect     smallest resolvable change in distance at (alpha, power)
evaluate              floor-corrected skill, two-sided floor test, leakage
                      screen and minimum detectable difference
intrinsic_dimension   slope, two-NN and maximum-likelihood estimates
plan_pullback         empirical latent certificate without a Lipschitz constant
certification_limit   dimension above which a latent certificate is vacuous

Command line: ``python -m certgen --help``.
"""
from .discrepancy import (DIMENSION_GOVERNED_RATE, DISCREPANCIES,
                          FLOOR_IS_POSITIVE, energy_distance,
                          get_discrepancy, mmd_rbf, sliced_w2, w2_exact)
from .baselines import (BASELINES, Standardizer, baseline_gaussian,
                        baseline_marginals, baseline_mean, baseline_self,
                        get_baseline)
from .leakage import LeakageReport, leakage_check
from .floor import FloorReport, resolution_floor
from .design import (GATE_POWER_GRID_6PT, GATE_SIZE_BY_D_EFF,
                     INTERVAL_LEVEL_CALIBRATION, FloorLaw, combine_severity,
                     fit_floor_law, floor_law_from_profile, power_curve,
                     required_n,
                     required_n_sensitivity, resolvable_effect)
from .dimension import (D_IDENTIFIED_MIN, EXPONENT_SATURATION_TABLE,
                        HEAVY_TAIL_INDEX, RATE_REGIME_MIN_D, DimensionReport,
                        intrinsic_dimension, mle_dimension, tail_index_hill,
                        tie_fraction, twonn_dimension, wasserstein_dimension)
from .latent import (D_HAT_CALIBRATION, FLOOR_RATIO, TABULATED_N,
                     TWO_SAMPLE_VS_ONE_SAMPLE, PPCReport, certification_limit,
                     floor_ratio, floor_ratio_measured, measured_to_nominal,
                     nominal_to_measured, plan_pullback, vacuity_margin)
from .report import EvaluationReport, evaluate
from . import data

__version__ = "3.3.1"
__all__ = [
    "DISCREPANCIES", "FLOOR_IS_POSITIVE", "DIMENSION_GOVERNED_RATE",
    "w2_exact", "energy_distance", "mmd_rbf", "sliced_w2", "get_discrepancy",
    "BASELINES", "Standardizer", "baseline_mean", "baseline_marginals",
    "baseline_gaussian", "baseline_self", "get_baseline",
    "LeakageReport", "leakage_check",
    "FloorReport", "resolution_floor",
    "FloorLaw", "fit_floor_law", "floor_law_from_profile", "required_n",
    "required_n_sensitivity",
    "resolvable_effect", "power_curve", "combine_severity",
    "GATE_SIZE_BY_D_EFF", "GATE_POWER_GRID_6PT", "INTERVAL_LEVEL_CALIBRATION",
    "DimensionReport", "intrinsic_dimension", "wasserstein_dimension",
    "twonn_dimension", "mle_dimension", "tie_fraction", "tail_index_hill",
    "D_IDENTIFIED_MIN", "RATE_REGIME_MIN_D", "EXPONENT_SATURATION_TABLE",
    "HEAVY_TAIL_INDEX",
    "PPCReport", "plan_pullback", "FLOOR_RATIO", "D_HAT_CALIBRATION",
    "TABULATED_N", "TWO_SAMPLE_VS_ONE_SAMPLE", "floor_ratio",
    "floor_ratio_measured", "nominal_to_measured", "measured_to_nominal",
    "vacuity_margin", "certification_limit",
    "EvaluationReport", "evaluate",
    "data", "__version__",
]
