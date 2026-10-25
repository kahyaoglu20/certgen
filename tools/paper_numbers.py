"""Every measured number of the manuscript, computed from results/.

Each function covers one part of paper/main.tex and returns the numbers it
quotes, formatted exactly as they are printed. tests/test_manuscript.py
requires every value to appear in the manuscript, and every numeric literal
of the manuscript body to be one of these values, a design parameter listed
in DESIGN_PARAMETERS, or a figure quoted from a cited work (CITED_FIGURES).

    python tools/paper_numbers.py            # print every number
"""
from __future__ import annotations

import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.stats import chi2, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return json.load(fh)


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve")


def word(k: int) -> str:
    """A count as an English word (up to twelve), otherwise as digits."""
    return WORDS[k] if 0 <= k < len(WORDS) else str(k)


def dash(x, fmt) -> str:
    """A formatted value, or a dash where it is undefined."""
    return fmt(x) if x is not None and np.isfinite(x) else "---"


def rnd(x, nd: int) -> str:
    """x rounded half-up on its decimal representation to twelve significant
    digits, so that 0.8575, also when computed as 0.8574999999999999,
    prints as 0.858."""
    q = Decimal(f"{float(x):.12g}").quantize(Decimal(1).scaleb(-nd),
                                             rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def f1(x):
    return rnd(x, 1)


def f2(x):
    return rnd(x, 2)


def f3(x):
    return rnd(x, 3)


def pct0(x):
    return rnd(100 * x, 0) + "\\%"


def pct1(x):
    return rnd(100 * x, 1) + "\\%"


def spct1(x):
    v = rnd(100 * x, 1)
    return (v if v.startswith("-") else "+" + v) + "\\%"


def thousands(n):
    """LaTeX thin-space thousands separator: 16384 -> 16\\,384."""
    return f"{int(n):,}".replace(",", "\\,")


def seconds(t):
    return (f"{t:.3f}" if t < 0.1 else f"{t:.2f}" if t < 1
            else f"{t:.1f}" if t < 10 else f"{t:.0f}")


# ----------------------------------------------------------------- Section 1
def introduction():
    a = load("resolution_atlas.json")
    c = next(x for x in a["cells"]
             if x["table"] == "adult:E1" and x["n_per_side"] == 250)
    excluded = set(a["summary"]["not_admissible"])
    at4000 = [x["dskill_T1_over_T1_p80_true"] for x in a["cells"]
              if x["n_per_side"] == 4000 and x["table"] not in excluded]
    return {"adult_span": f3(c["skill_span_T1"]),
            "adult_sd": f3(c["D_T1_sd_over_seeds"]),
            "adult_sd_over_span": f"{c['D_T1_sd_over_seeds'] / c['skill_span_T1']:.0f}",
            "rho4000_min_pct": pct1(min(at4000)),
            "rho4000_max": f1(max(at4000))}


# ----------------------------------------------------------------- Section 3
def floor_section():
    ex = load("copula_examples.json")
    law = ex["floor_law"]["law"]
    ub = ex["unbiased_statistics"]
    sl = ex["sliced_exponent"]
    sat = load("slope_estimator_bias.json")
    out = {"law_C": f"{law['C']:.4f}", "law_deff": f2(law["d_eff"]),
           "law_r2": f3(law["r2"]),
           "energy_zero_lo": f"{100 * ub['range_energy'][0]:.0f}",
           "energy_zero_hi": f"{100 * ub['range_energy'][1]:.0f}",
           "mmd_zero_lo": f"{100 * ub['range_mmd'][0]:.0f}",
           "mmd_zero_hi": f"{100 * ub['range_mmd'][1]:.0f}",
           "sliced_deff": f2(sl["d_eff_sliced"]),
           "sliced_underestimate": pct0(ex["sliced_underestimate"]["underestimate"]),
           "sat_sd_1": f2(sat["cells"][0]["sd"])}
    for c in sat["cells"]:
        out[f"sat_dhat_{c['nominal_d']}"] = f2(c["d_hat"])
        out[f"sat_bias_{c['nominal_d']}"] = f"{100 * c['rel_bias']:+.0f}\\%"
    cost = load("ot_cost.json")
    for c in cost["cells"]:
        out[f"cost_{c['d']}_{c['n']}"] = seconds(c["seconds_median"])
    out["cost_cpu"] = " ".join(re.sub(r"\((?:R|TM)\)", "", cost["cpu"]).split())
    exps = cost["time_exponent_in_n_(n>=1024)"]
    out["cost_exp_8"], out["cost_exp_32"] = f2(exps["8"]), f2(exps["32"])
    out["cost_exp_lo"], out["cost_exp_hi"] = (f1(min(exps.values())),
                                              f1(max(exps.values())))
    return out


# ----------------------------------------------------------------- Section 4
def design_section():
    ex = load("copula_examples.json")["floor_law"]
    out = {"resolvable_500": f3(ex["resolvable_effect"]["500"]),
           "resolvable_5000": f3(ex["resolvable_effect"]["5000"]),
           "resolvable_target": f3(ex["target_effect"]),
           "n_required": thousands(round(ex["n_required_for_target"], -3))}
    st = load("stress_tests.json")["goodness_of_fit_power"]["by_grid_length"]
    out["r2_power_4"], out["r2_power_6"], out["r2_power_8"] = (
        f2(st[0]["power_r2"]), f3(st[1]["power_r2"]), f3(st[2]["power_r2"]))
    out["fixed_cut_rejects"] = pct0(st[0]["rejection_fixed_cut_1.25_under_null"])
    size = load("curvature_test_size.json")
    for c in size["cells"]:
        d = c["nominal_d"]
        out[f"size_{d}"] = f3(c["size"])
        out[f"size_deff_{d}"] = f2(c["d_eff_median"])
        out[f"size_drift_{d}"] = f"{c['mean_log_drift']:+.3f}"
    trials = sorted(set(size["trials_per_cell"].values()))
    out["size_trials_lo"], out["size_trials_hi"] = str(trials[0]), str(trials[-1])
    power = load("curvature_test_power.json")
    for c in power["cells"]:
        if (c["grid_kind"], c["n_points"]) in (("densified", 6), ("densified", 12),
                                               ("range_extension", 8),
                                               ("range_extension", 12)):
            p = c["power"]
            out[f"power_{c['grid_kind']}_{c['n_points']}_{c['median_se']}"] = (
                f2(p) if round(p, 2) * 1000 == round(p * 1000) else f3(p))
    for k in (8, 12):
        out[f"power_nmax_{k}"] = thousands(max(
            c["n_max"] for c in power["cells"]
            if c["grid_kind"] == "range_extension" and c["n_points"] == k))
    bench = load("benchmark_floor_laws.json")["records"]
    se = [float(np.median(r["floor_se_log"])) for r in bench
          if r["single_exponent_ok"]]
    out["bench_se_lo"], out["bench_se_hi"] = pct1(min(se)), pct1(max(se))
    admissible = {r["dataset"] for r in bench if r["single_exponent_ok"]}
    sev = [t for t in load("severity.json")["tables"] if t["dataset"] in admissible]
    infl = [t["interval_inflation_vs_binomial_only"] for t in sev]
    out["sev_infl_lo"], out["sev_infl_hi"] = f2(min(infl)), f2(max(infl))
    out["sev_infl_med"] = f2(float(np.median(infl)))
    out["sev_between_larger"] = str(sum(t["sd_between_seed"] > t["sd_within_binomial"]
                                        for t in sev))
    out["sev_cells"] = str(len(sev))
    cov = load("interval_coverage.json")
    prim = ([c["coverage_n_interior"] for c in cov["cells"]]
            + [c["coverage_n_extrapolated"] for c in cov["cells"]])
    lc9 = cov["level_calibration"]["delivered_by_level"]["0.9"]
    out["cov_lo"] = f3(min(min(prim), lc9["min"]))
    out["cov_hi"] = f3(max(max(prim), lc9["max"]))
    out["cov_med"] = f3(float(np.median(prim)))
    cvs = sorted({c["replicate_cv"] for c in cov["level_calibration"]["cells"]})
    out["cov_cv_regime"] = ", ".join(f2(v) for v in cvs)
    out["cov_calibrated_level"] = (
        f"{cov['level_calibration']['inverse_map_requested_to_nominal']['0.9']:.4f}")
    inf = load("inference_checks.json")
    out["crn_var_ratio"] = f2(inf["crn_vs_independent"]["drift_statistic"]
                              ["var_ratio_crn_over_independent"])
    ratio = [r["d_W_sd_subsample"] / r["d_W_sd"] for r in bench
             if r["single_exponent_ok"]]
    out["subsample_ratio_lo"], out["subsample_ratio_hi"] = f1(min(ratio)), f1(max(ratio))
    out["subsample_ratio_med"] = f1(float(np.median(ratio)))
    return out


# ----------------------------------------------------------------- Section 5
def reporting_section():
    rep = load("reporting_endpoints.json")
    sk = rep["skill_T1"]
    out = {f"skill_{tag}": f"{sk[k]['median']:+.2f}" for k, tag in
           (("fresh draw", "fresh"), ("T2 sample", "t2"), ("T1 sample", "t1"),
            ("T0 sample", "t0"))}
    out["skill_fresh_min"] = f2(sk["fresh draw"]["min"])
    out["skill_fresh_max"] = f2(sk["fresh draw"]["max"])
    out["self_disjoint"] = f2(rep["t_self_over_floor"]["disjoint_remainder"]["median"])
    out["self_whole"] = f2(rep["t_self_over_floor"]["whole_sample"]["median"])
    out["endpoint_seeds"] = str(rep["config"]["R"])
    mem = load("stress_tests.json")["memorisation"]
    out["copy_skill_t2"] = f"{mem['exact_copy']['skill_T2']:+.2f}"
    out["fresh_skill_t2"] = f"{mem['fresh_draw']['skill_T2']:+.2f}"
    s1 = load("copula_examples.json")["skill_above_one"]
    out["p_above_one_4"] = f2(s1["4"]["p_skill_T2_above_1"])
    out["p_above_one_8"] = f2(s1["8"]["p_skill_T2_above_1"])
    out["p_above_one_trials"] = str(s1["8"]["trials"])
    t2 = [s1[d]["p_V_below_minus_005"]["T2_gaussian"] for d in ("4", "8")]
    out["v_rule_lo"], out["v_rule_hi"] = (f"{100 * min(t2):.0f}",
                                          f"{100 * max(t2):.0f}")
    lk = load("leakage_screen.json")
    out["leak_rows"] = thousands(max(r["N_real"] for r in lk["by_table_size"]
                                     if r["flagged"]))
    out["leak_escape"] = f2(lk["by_perturbation"]["escape_rel"])
    out["leak_discrete_cases"] = str(len(lk["discrete_laws"]))
    inf = load("inference_checks.json")["power_two_sd_rule"]
    out["two_sd_power"] = f2(inf["measured_power"])
    out["two_sd_trials"] = str(inf["setting"]["R"])
    out["two_sd_analytic"] = f3(inf["analytic"])
    mp = load("stress_tests.json")["mdd_proxy"]["cells"]
    lv = [c["ratio_level_scaled"] for c in mp]
    out["proxy_copula_lo"], out["proxy_copula_hi"] = f2(min(lv)), f2(max(lv))
    out["unscaled_near_floor"] = f2(mp[0]["ratio_floor_level"])
    out["unscaled_at_t1"] = f2(mp[-1]["ratio_floor_level"])
    out["proxy_replicates"] = str(load("stress_tests.json")["mdd_proxy"]["replicates"])
    sh = load("copula_examples.json")["shift_spread"]
    out["shift_sd_ratio"] = f1(sh["sd_ratio"])
    eq = load("equivalence_test.json")
    out["tost_eta"] = f"{eq['config']['delta']:.4f}"
    out["tost_boundary"] = f1(max(eq["rates"]["boundary +delta"]["equiv_rate"],
                                  eq["rates"]["boundary -delta"]["equiv_rate"]))
    out["tost_fresh"] = f1(eq["rates"]["honest (fresh draws)"]["equiv_rate"])
    out["tost_frac"] = f3(eq["p_hi_uniformity"]["frac_le_05"])
    out["tost_median"] = f3(eq["p_hi_uniformity"]["median"])
    out["tost_n"] = str(eq["p_hi_uniformity"]["n"])
    return out


# ----------------------------------------------------------------- Section 6
def certificate_section():
    ref = load("reference_tables.json")
    out = {}
    for c in ref["r_cells"]:
        out[f"r_{c['N']}_{c['d']}"] = f3(c["r_mean"])
    for c in ref["dhat_cells"]:
        out[f"dhat_{c['N']}_{c['d']}"] = f1(c["d_hat_mean"])
    for c in ref["crossings"]:
        N = c["N"]
        out[f"cross_nom_{N}"] = f1(c["d_star_limit_nominal"])
        out[f"cross_meas_{N}"] = f1(c["d_star_limit_measured"])
        out[f"cross_meas_ci_{N}"] = (f"[{f1(c['measured_ci90'][0])}, "
                                     f"{f1(c['measured_ci90'][1])}]")
    sens = [c["dlogd_dlogr_at_crossing"] for c in ref["crossings"]]
    out["sens_lo"], out["sens_hi"] = f1(min(sens)), f1(max(sens))
    out["r_se_max_d4"] = f"{np.ceil(max(c['r_se'] for c in ref['r_cells'] if c['d'] >= 4) * 1e4) / 1e4:.4f}"
    out["r_se_max"] = f"{np.ceil(max(c['r_se'] for c in ref['r_cells']) * 1e3) / 1e3:.3f}"
    ratio = [c["d_star_limit_nominal"] / c["d_star_limit_measured"]
             for c in ref["crossings"]]
    out["headroom_lo"], out["headroom_hi"] = f2(min(ratio)), f2(max(ratio))
    Ns = sorted(c["N"] for c in ref["crossings"])
    out["top_n"] = thousands(Ns[-1])
    out["n_rep_tables"] = str(ref["n_rep_r"])
    br = load("two_sample_vs_one_sample.json")
    v = [c["ratio_mean"] for c in br["ratio_by_d_N"].values()]
    out["bridge_lo"], out["bridge_hi"] = f3(min(v)), f2(max(v))
    out["bridge_ref_m"] = thousands(br["cells"][0]["M"])
    tr = load("transfer_smooth_laws.json")
    out["transfer_dhat_lo"] = f1(min(c["d_hat"] for c in tr["cells"]))
    out["transfer_dhat_hi"] = f1(max(c["d_hat"] for c in tr["cells"]))
    b = tr["band_residual_summary"]
    out["transfer_median"] = spct1(b["median_residual"])
    out["transfer_lo"], out["transfer_hi"] = (spct1(b["min_residual"]),
                                              spct1(b["max_residual"]))
    iso = [c["residual_vs_table_at_dhat"] for c in tr["isotropic_reference_cells"]]
    out["iso_lo"], out["iso_hi"] = spct1(min(iso)), spct1(max(iso))
    pr = load("transfer_probe.json")
    fam = pr["families"]
    out["lattice_lo"] = spct1(fam["lattice"]["band_range"][0])
    out["lattice_hi"] = spct1(fam["lattice"]["band_range"][1])
    out["mixed_lo"] = spct1(fam["mixed_type"]["band_range"][0])
    out["mixed_hi"] = spct1(fam["mixed_type"]["band_range"][1])
    out["t3_residual"] = spct1(max(fam["heavy_tailed"]["all_range"]))
    for c in pr["cells"]:
        if c["law"] == "student_t_nu5":
            key = "t5_band" if c["in_band"] else "t5_out"
            out[key] = spct1(c["residual_vs_table_at_dhat"])
    lap = [abs(c["residual_vs_table_at_dhat"]) for c in pr["cells"]
           if c["law"] == "laplace_rot"]
    out["laplace_max"] = f"{np.ceil(100 * max(lap)):.0f}"
    ti = load("tail_index.json")
    for r in ti["reference_laws"]:
        out[f"tail_ref_{r['law']}"] = f1(r["median"])
    out["tail_lo"], out["tail_hi"] = f2(ti["summary"]["min"]), f2(ti["summary"]["max"])
    out["tail_flagged"] = str(len(ti["summary"]["below_threshold"]))
    out["tail_tables"] = str(len(ti["tables"]))
    return out


# ----------------------------------------------------------------- Section 7
def pullback_section():
    a = load("plan_pullback_vae.json")
    s = a["summary"]
    return {"ppc_lo": f2(s["ppc_over_exact"]["min"]),
            "ppc_hi": f2(s["ppc_over_exact"]["max"]),
            "ppc_med": f2(s["ppc_over_exact"]["median"]),
            "lip_lo": f"{s['lipschitz_over_exact']['min']:.0f}",
            "lip_hi": f"{s['lipschitz_over_exact']['max']:.0f}",
            "lip_med": f"{s['lipschitz_over_exact']['median']:.0f}",
            "loc_lo": f1(s["local_lipschitz_over_exact"]["min"]),
            "loc_hi": f1(s["local_lipschitz_over_exact"]["max"]),
            "loc_med": f1(s["local_lipschitz_over_exact"]["median"]),
            "vae_eval": str(a["config"]["n_eval"]),
            "vae_epochs": str(a["config"]["epochs"])}


# ----------------------------------------------------------------- Section 8
CELL_ORDER = ("adult:E1", "adult:E2", "bank-marketing:E1", "bank-marketing:E2",
              "census-income:E1", "covertype:E1=E2", "higgs:E1", "letter:E1",
              "MagicTelescope:E1", "pendigits:E1", "shuttle:E1")


def benchmark_section():
    recs = {r["dataset"]: r for r in load("benchmark_floor_laws.json")["records"]}
    rng = {t["dataset"]: t for t in load("half_fit_ranges.json")["tables"]}
    sev = {t["dataset"]: t for t in load("severity.json")["tables"]}
    out = {}
    for k in CELL_ORDER:
        r = recs[k]
        out[f"floor_row_{k}"] = (
            f"{f2(r['d_eff'])} & {f3(r['r2'])} & {dash(r['drift_p'], f3)} & "
            f"{'yes' if r['single_exponent_ok'] else 'no'} & "
            f"${f2(r['d_wasserstein'])} \\pm {f2(r['d_W_sd'])}$ & "
            f"{f2(r['spread'])}$\\times$")
        s = sev[k]
        if s["drift_50_median"] is not None and np.isfinite(s["drift_50_median"]):
            lo, hi = s["drift_50_ci90_combined"]
            d50 = f"{f2(s['drift_50_median'])} [{f2(lo)}, {f2(hi)}]"
            obs = s["drift_observed"]
            inside = "below" if obs < lo else "above" if obs > hi else "inside"
        else:
            d50, inside = "---", "---"
        ratio = rng[k]["worst_ratio_wls"]
        times = f"{f2(ratio)}$\\times$" if np.isfinite(ratio) else "---"
        out[f"scope_row_{k}"] = f"{times} & {dash(r['drift_p'], f3)} & {d50} & {inside}"
    adm = [recs[k] for k in CELL_ORDER if recs[k]["single_exponent_ok"]]
    out["admissible"] = str(len(adm))
    out["r2_min_adm"] = f3(min(r["r2"] for r in adm))
    out["p_lo_adm"] = f3(min(r["drift_p"] for r in adm))
    out["p_hi_adm"] = f3(max(r["drift_p"] for r in adm))
    out["deff_lo_adm"] = f2(min(r["d_eff"] for r in adm))
    out["deff_hi_adm"] = f2(max(r["d_eff"] for r in adm))
    sh = recs["shuttle:E1"]
    out["shuttle_r2"] = f3(sh["r2"])
    out["shuttle_cv125"] = f2(sh["floor_cv"][0])
    out["shuttle_floors"] = "/".join(f2(v) for v in sh["floors"])
    cov = {t["dataset"]: t for t in load("interval_coverage.json")["real_tables"]}
    w = cov["shuttle:E1"]["targets"]
    out["shuttle_width_10k"] = f1(w["floor_at_n=10000"]["width_ratio"])
    out["shuttle_width_40k"] = f"{w['floor_at_n=40000']['width_ratio']:.0f}"
    out["shuttle_spread"] = f2(sh["spread"])
    out["shuttle_twonn"], out["shuttle_mle"] = f2(sh["d_twonn"]), f2(sh["d_mle"])
    out["shuttle_dhat"] = f2(sh["d_wasserstein"])
    sp = [r["spread"] for r in adm]
    out["spread_lo_adm"], out["spread_hi_adm"] = f2(min(sp)), f2(max(sp))
    d50 = [sev[r["dataset"]]["drift_50_median"] for r in adm]
    out["d50_max"], out["d50_min"] = f2(max(d50)), f2(min(d50))
    def position(k):
        t = sev[k]
        lo, hi = t["drift_50_ci90_combined"]
        return ("below" if t["drift_observed"] < lo else
                "above" if t["drift_observed"] > hi else "inside")
    rejected = [k for k in CELL_ORDER if np.isfinite(recs[k]["drift_p"])
                and recs[k]["drift_p"] < 0.05]
    out["rejected_p"] = " and ".join(f3(recs[k]["drift_p"]) for k in rejected)
    out["n_rejected"] = str(len(rejected))
    out["admissible_below"] = str(sum(position(r["dataset"]) == "below" for r in adm))
    out["rejected_above"] = str(sum(position(k) == "above" for k in rejected))
    wr = sorted(rng[r["dataset"]]["worst_ratio_wls"] for r in adm)
    out["halffit_lo"], out["halffit_hi"] = f2(wr[0]), f2(wr[-1])
    out["halffit_med"] = f2(float(np.median(wr)))
    widths = {10000: [], 40000: []}
    for r in adm:
        for n in widths:
            widths[n].append(cov[r["dataset"]]["targets"][f"floor_at_n={n}"]["width_ratio"])
    out["width10k_lo"], out["width10k_hi"] = f2(min(widths[10000])), f2(max(widths[10000]))
    out["width40k_lo"], out["width40k_hi"] = f2(min(widths[40000])), f2(max(widths[40000]))
    grid_max = max(sh["n_grid"])
    out["beyond_10k"] = f1(10000 / grid_max)
    out["beyond_40k"] = f"{40000 / grid_max:.0f}"
    out["bench_minutes"] = f"{sum(r['wall_s'] for r in recs.values()) / 60:.0f}"
    diag = load("benchmark_diagnostics.json")
    el = diag["exclusions"]["electricity:E1"]["w2_medians_by_side"]
    out["electricity_medians"] = "/".join(f3(el[s]) for s in ("1000", "2000", "4000"))
    eeg = diag["exclusions"]["eeg-eye-state:E1"]
    out["eeg_max_z"] = f1(eeg["max_abs_z"])
    eg = dict(zip(eeg["floor_law"]["n_grid"], eeg["floor_law"]["floors"]))
    out["eeg_floor_1000"], out["eeg_floor_2000"] = f3(eg[1000]), f2(eg[2000])
    out["miniboone_r2"] = f2(diag["exclusions"]["MiniBooNE:E1"]["floor_law"]["r2"])
    co = diag["correlations"]
    out["adult_corr_max"] = f3(co["max_pairwise_abs_corr"])
    out["adult_corr_med"] = f3(co["median_pairwise_abs_corr"])
    return out


def atlas_section():
    a = load("resolution_atlas.json")
    s = a["summary"]
    e = s["extended_true_p80"]
    out = {"atlas_cells": str(e["cells"]), "atlas_min": f3(e["min"]),
           "atlas_med": f3(e["median"]), "atlas_max": f3(e["max"]),
           "atlas_frac": pct0(e["frac_ge_1"])}
    p = s["proxy_over_true_sd"]
    out["proxy_lo"], out["proxy_med"], out["proxy_hi"] = (f2(p["min"]),
                                                          f2(p["median"]),
                                                          f2(p["max"]))
    out["proxy_cells"] = str(p["cells"])
    out["noise_cells"] = str(len(s["cells_where_T1_span_inside_baseline_noise"]))
    tables = []
    for c in a["cells"]:
        if c["table"] not in tables:
            tables.append(c["table"])
    for t in tables:
        cs = sorted((c for c in a["cells"] if c["table"] == t),
                    key=lambda c: c["n_per_side"])
        top = cs[-1]
        lo, hi = top["dskill_T1_over_T1_p80_true_ci90"]
        out[f"row_{t}"] = (
            f"{f3(top['dskill_T1_over_T1_p80'])} & "
            f"{f3(top['dskill_T1_over_T1_p80_true'])} & [{f3(lo)}, {f3(hi)}] & "
            f"{f3(float(np.median([c['dskill_T1_over_T1_p80_true'] for c in cs])))} & "
            f"{f2(float(np.median([c['proxy_over_true_sd'] for c in cs])))} & "
            f"{sum(not c['span_T1_exceeds_baseline_noise'] for c in cs)}")
    best = [100 * c["dskill_T1_over_T1_p80_true"] for c in a["cells"]
            if c["n_per_side"] == 4000
            and c["table"] in ("letter:E1", "pendigits:E1", "MagicTelescope:E1")]
    out["best_pct_lo"], out["best_pct_hi"] = f1(min(best)), f1(max(best))
    worst = [c["dskill_T1_over_T1_p80_true"] for c in a["cells"]
             if c["table"] in ("adult:E1", "bank-marketing:E1")]
    out["worst_min"] = f"{np.floor(10 * min(worst)) / 10:.1f}"
    adm = [c for c in a["cells"] if c["table"] not in set(s["not_admissible"])]
    rel = np.array([(c["D_T1"] - c["floor"]) / c["floor"] for c in adm])
    rho = np.array([c["dskill_T1_over_T1_p80_true"] for c in adm])
    out["spearman"] = f2(float(spearmanr(rel, rho).statistic))
    out["atlas_cpu_h"] = f1(sum(c["seconds"] for c in a["cells"]) / 3600)
    r = a["r_pair"]
    out["sd_lo_pct"] = f"{100 * (1 - np.sqrt((r - 1) / chi2.ppf(0.95, r - 1))):.0f}"
    out["sd_hi_pct"] = f"{100 * (np.sqrt((r - 1) / chi2.ppf(0.05, r - 1)) - 1):.0f}"
    out["chi2_dof"] = str(r - 1)
    out["r_pair"] = str(r)
    out["max_rows"] = thousands(a["max_rows"])
    out["atlas_n_rep"] = str(a["n_rep"])
    return out


def margin_section():
    a = load("certification_margins.json")
    h = a["heavy_tail_sensitivity"]
    out = {"heavy_divisor": f"{1 + h['worst_heavy_tail_residual']:.3f}",
           "n_mc": thousands(a["n_mc"])}
    for r in a["per_table"]:
        t = r["dataset"]
        out[f"row_{t}"] = (f"{f2(r['d_hat'])} & ${f2(r['margin_q05'])}\\times$ & "
                           f"${f2(r['margin_r_units_q05'])}\\times$ & "
                           f"${f2(h['margin_r_units_q05_if_applied'][t])}\\times$")
    rq = [r["margin_r_units_q05"] for r in a["per_table"]]
    out["rq05_lo"], out["rq05_hi"] = f2(min(rq)), f2(max(rq))
    hv = sorted(h["margin_r_units_q05_if_applied"].values())
    out["heavy_lo_above1"] = f2(min(v for v in hv if v >= 1))
    out["heavy_hi"] = f2(max(hv))
    out["heavy_below1"] = f2(min(hv))
    out["heavy_above1_count"] = word(sum(v >= 1 for v in hv)).capitalize()
    out["dhat_max"] = f2(max(r["d_hat"] for r in a["per_table"]))
    ti = {r["table"]: r for r in load("tail_index.json")["tables"]}
    out["tail_higgs"] = f2(ti["higgs:E1"]["tail_index_hill"])
    return out


# ------------------------------------------------- Sections 9 to 10, appendix
def implementation_section():
    out = {}
    tied = load("stress_tests.json")["tied_observations"]
    out["tied_spread_replaced"] = f2(tied[-1]["spread_zero_distance_replaced"])
    q = load("inference_checks.json")["exponent_ci_quantile"]
    out["cover_z"], out["cover_t"] = f3(q["simulated_cover_z"]), f3(q["simulated_cover_t"])
    out["lean_pinned"] = str(load("lean_build.json")["pinned_results"])
    ref = load("reference_tables.json")
    small = [c for c in ref["r_cells"] + ref["dhat_cells"]
             if c.get("source") != "large_n"]
    out["tables_cpu_h"] = f1(sum(c["seconds"] for c in small) / 3600)
    large = load("reference_tables_large_n.json")
    for key, t in large["timing"].items():
        out[f"large_median_s_{key}"] = f"{t['median_s']:.0f}"
        out[f"large_peak_gb_{key}"] = f1(t["peak_rss_gb_max"])
    out["large_cpu_h"] = f"{sum(t['total_cpu_h'] for t in large['timing'].values()):.0f}"
    cal = large["calibration"]
    out["calib_n"] = str(len(cal))
    out["calib_r_identical"] = str(sum(c["rel_diff"] == 0.0 for c in cal
                                       if c["kind"] == "r"))
    out["calib_dhat_n"] = str(sum(c["kind"] == "dhat" for c in cal))
    out["calib_identical"] = word(sum(c["rel_diff"] == 0.0 for c in cal))
    top = max(c["N"] for c in ref["crossings"])
    out["next_n"] = thousands(2 * top)
    gb = 8.0 * (2 * top) ** 2 / 1e9
    out["next_matrix_gb"] = f"{gb:.1f}" if gb < 10 else f"{gb:.0f}"
    return out


SECTIONS = (introduction, floor_section, design_section, reporting_section,
            certificate_section, pullback_section, benchmark_section,
            atlas_section, margin_section, implementation_section)


def all_numbers() -> dict:
    """{section.key: formatted value} for every number the manuscript quotes."""
    out = {}
    for fn in SECTIONS:
        out.update({f"{fn.__name__}.{k}": v for k, v in fn().items()})
    return out


# Design parameters and conventions stated in the manuscript itself.
DESIGN_PARAMETERS = {
    "0.05": "significance level; admissibility threshold of the curvature p-value",
    "0.80": "power", "80": "power in percent", "0.90": "interval level", "0.95": "R^2 threshold; interval quantile",
    "0.975": "normal quantile index", "0.10": "p-value band", "0.5": "lattice step",
    "0.8": "copula correlation", "0.6": "copula marginal", "0.2": "null exponent 1/5",
    "0.35": "two-scale amplitude", "0.25": "factor noise; equivalence margin factor",
    "0.85": "anisotropy decay", "0.1": "illustrative distance",
    "0.09": "illustrative floor", "0.001": "illustrative floor",
    "1.25": "fixed drift threshold; standard-error constant sqrt(pi/2)",
    "0.500": "analytical power 1/2", "0.0": "measured frequency", "1.0": "measured frequency",
    "12": "replicates", "19": "minimum replicates", "20": "replicates",
    "30": "paired replicates", "10": "base of 10^k", "16": "dimension",
    "24": "latent dimension", "28": "factors; band edge", "32": "dimension",
    "64": "dimension", "128": "dimension; width", "256": "size; width",
    "512": "size", "1024": "size", "2048": "size", "4096": "size", "8192": "size",
    "16384": "size", "32768": "size", "125": "grid", "250": "grid", "500": "grid",
    "1000": "grid", "2000": "grid", "4000": "grid; rows", "5000": "size",
    "8000": "rows; Monte-Carlo draws", "10000": "target", "40000": "target",
    "1300": "training rows", "1797": "rows of digits", "50": "ambient dimension",
    "100": "percent", "2026": "year", "90": "interval level",
    "95": "Monte-Carlo interval level", "15": "theorem numbering",
    "3.3": "software version",
}

# Figures quoted from cited works, with the citation key.
CITED_FIGURES = {
    "24.7": "chen2026", "58.6": "chen2026", "3.2": "dufour2026",
    "75": "card2020", "2000": "card2020",
}


if __name__ == "__main__":
    for k, v in all_numbers().items():
        print(f"{k:<50} {v}")
