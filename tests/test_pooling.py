"""noilai.stats.pooling (the hierarchical model-in-family pool of PREREG 8.1 / DD 1, 8.1; k of N; TOST) and
noilai.stats.agreement.weighted_precision (DD 10.1, PREREG 8.12) on synthetic data with known structure.
Family labels and model names below are abstract (F1.., m1_1..), never panel identifiers."""
import pathlib
import random

import numpy as np
import pytest
from scipy import stats as sps

from noilai import constants
from noilai import validation as VA
from noilai.stats import agreement, bootstrap, pooling
from noilai.stats import tests as st
from noilai.stats.tests import PairedTest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _simulate(rng, sizes=(3, 2, 1, 1, 1), mu=0.05, tau_between=0.03, tau_within=0.015, se=0.02):
    """Per-model estimates theta_m = mu + u_f + e_m + noise, u_f ~ N(0, tau_between^2), e_m ~ N(0, tau_within^2),
    noise ~ N(0, se^2) with the known variance se^2 attached, models nested in families F1..Fk."""
    fams, ests = {}, []
    for fi, n in enumerate(sizes):
        fam = f"F{fi + 1}"
        u = rng.normal(0.0, tau_between)
        for j in range(n):
            m = f"m{fi + 1}_{j + 1}"
            fams[m] = fam
            ests.append({"model": m, "estimate": mu + u + rng.normal(0.0, tau_within) + rng.normal(0.0, se),
                         "variance": se ** 2})
    return ests, fams


def _paired_data(n_pairs=300, items_per_pair=4, p_a=0.6, p_b=0.7, icc=0.3, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_pairs), items_per_pair)
    u = rng.normal(size=n_pairs)[clusters] * np.sqrt(icc)
    e_a = rng.normal(size=len(clusters)) * np.sqrt(1 - icc)
    e_b = rng.normal(size=len(clusters)) * np.sqrt(1 - icc)
    a = (u + e_a) < sps.norm.ppf(p_a)
    b = (u + e_b) < sps.norm.ppf(p_b)
    return a, b, clusters


# ----------------------------------------------------------------------------- the family map is required
def test_pool_requires_a_family_map_and_refuses_unmapped_models():
    ests = [{"model": "m1", "estimate": 0.1, "variance": 1e-4}, {"model": "m2", "estimate": 0.0, "variance": 1e-4}]
    with pytest.raises(ValueError, match="flat pool"):
        pooling.pool(ests, families={})
    with pytest.raises(ValueError, match="m2"):
        pooling.pool(ests, families={"m1": "F1"})
    with pytest.raises(ValueError, match="twice"):
        pooling.pool([*ests, ests[0]], families={"m1": "F1", "m2": "F2"})
    with pytest.raises(ValueError, match="method"):
        pooling.pool(ests, families={"m1": "F1", "m2": "F2"}, method="plain")
    with pytest.raises(ValueError):
        pooling.pool(ests, families={"m1": "F1", "m2": "F2"}, direction=2)


def test_source_has_no_one_stage_pool_over_models():
    src = (ROOT / "noilai" / "stats" / "pooling.py").read_text()
    assert "DerSimonian" not in src
    for literal in ('"dl"', "'dl'", '"chi2"', "'chi2'"):
        assert literal not in src
    assert "combine_effects" not in src and "method_re" not in src      # statsmodels' pooler only in tests


# ----------------------------------------------------------------------------- recovery and the family weight
def test_recovers_a_family_level_effect_with_between_family_heterogeneity():
    rng = np.random.default_rng(11)
    ests, fams = _simulate(rng, sizes=(7, 6, 1, 1, 1, 1, 1), mu=0.05, tau_between=0.03)   # the DD 8.1 entry counts
    res = pooling.pool(ests, families=fams, direction=+1, label="H-test")
    d = res.as_dict()
    assert res.method == "reml_hk" and res.pooled and not res.unreliable
    assert res.k_informative_families == 7 and res.n_panel_families == 7 and res.df == 6
    assert res.k_of_n == "7 of 7 tokenizer families" and res.phrase == "across the panel"
    assert res.n_models_pooled == 18 and res.n_panel_models == 18
    assert res.lo <= 0.05 <= res.hi and abs(res.estimate - 0.05) < 0.03
    assert res.tau2_between is not None and res.tau2_between >= 0.0
    assert set(res.tau2_within) == {f"F{i}" for i in range(1, 8)}
    assert res.tau2_within["F3"] is None                      # one-model family: pass-through
    assert res.alternative["method"] == "family_bootstrap" and res.alternative["lo"] <= 0.05 <= res.alternative["hi"]
    assert res.methods_agree in (True, False) and d["label"] == "H-test"
    assert len(d["per_family"]) == 7 and all(f["informative"] for f in d["per_family"]) and d["excluded"] == []
    assert res.on_predicted_side is True and res.all_informative_families_on_side in (True, False)
    import json
    json.dumps(d)                                             # the record is JSON-serialisable as written


def test_equal_family_weight_a_seven_entry_family_does_not_outvote():
    fams = {f"big{i}": "FA" for i in range(7)}
    fams.update({"s1": "FB", "s2": "FC", "s3": "FD", "s4": "FE"})
    ests = [{"model": f"big{i}", "estimate": 0.10, "variance": 1e-4} for i in range(7)]
    ests += [{"model": s, "estimate": 0.0, "variance": 1e-4} for s in ("s1", "s2", "s3", "s4")]
    res = pooling.pool(ests, families=fams)
    flat = np.mean([e["estimate"] for e in ests])                   # a one-stage pool over models: 0.0636
    assert res.alternative["estimate"] == pytest.approx(0.02)          # mean of the five family means
    assert res.estimate < 0.05 < flat
    assert res.k_informative_families == 5 and res.n_models_pooled == 11
    fa = next(f for f in res.per_family if f["family"] == "FA")
    assert fa["n_models"] == 7 and fa["estimate"] == pytest.approx(0.10) and fa["tau2_within"] == pytest.approx(0.0)
    assert fa["lo_hk"] is not None and fa["se"] == pytest.approx(np.sqrt(1e-4 / 7))


# ----------------------------------------------------------------------------- Hartung-Knapp
def test_hk_interval_is_wider_than_the_fixed_scale_z_interval_for_small_k():
    # five families whose estimates disagree more than their variances allow: q > 1, HK widens; t_4 > z
    fams = {f"m{i}": f"F{i}" for i in range(1, 6)}
    ests = [{"model": f"m{i}", "estimate": th, "variance": 1e-4} for i, th in enumerate((0.00, 0.03, 0.06, 0.09, 0.12), 1)]
    res = pooling.pool(ests, families=fams)
    z_lo, z_hi = res.ci_z
    assert (res.hi - res.lo) > (z_hi - z_lo)
    assert res.estimate == pytest.approx(0.06)
    assert (res.hi - res.estimate) / res.se == pytest.approx(sps.t.ppf(0.975, 4))
    assert res.df == 4 and any("truncated at 1" in n for n in res.notes)


def test_hk_interval_never_collapses_when_family_estimates_coincide():
    # five families with identical estimates: q_raw = 0; without the truncation the HK interval would have zero width
    fams = {f"m{i}": f"F{i}" for i in range(1, 6)}
    ests = [{"model": f"m{i}", "estimate": 0.05, "variance": 1e-4} for i in range(1, 6)]
    res = pooling.pool(ests, families=fams)
    z_lo, z_hi = res.ci_z
    assert res.hi > res.lo and res.lo < 0.05 < res.hi
    assert (res.hi - res.lo) >= (z_hi - z_lo) - 1e-12                # the truncated scale gives t_4 x the fixed-effect SE
    assert any("q_raw = 0.000 < 1" in n for n in res.notes)


def test_hk_coverage_is_near_nominal_at_k5_while_z_undercovers():
    rng = np.random.default_rng(2026)
    cover_hk = cover_z = 0
    reps = 200
    for _ in range(reps):
        ests, fams = _simulate(rng, sizes=(3, 2, 2, 1, 1), mu=0.04, tau_between=0.03, tau_within=0.01, se=0.02)
        res = pooling.pool(ests, families=fams, n_boot=50)
        cover_hk += res.lo <= 0.04 <= res.hi
        cover_z += res.ci_z[0] <= 0.04 <= res.ci_z[1]
    assert cover_hk / reps >= 0.92
    assert cover_z / reps < cover_hk / reps - 0.02


def test_reml_hk_arithmetic_matches_statsmodels_hksj_at_the_same_tau2():
    """Cross-check of the HKSJ arithmetic only: statsmodels estimates tau2 by Paule-Mandel ("iterated"), so its
    tau2 is fed into our stage-2 formulas and the variance and interval must coincide."""
    from statsmodels.stats.meta_analysis import combine_effects

    theta = np.array([0.05, 0.08, 0.02, 0.06, 0.04])
    v = np.array([4e-4, 3e-4, 5e-4, 2e-4, 4e-4])
    sm = combine_effects(theta, v, method_re="iterated", use_t=True)
    mu, _var_fixed, w = pooling._random_effects_mean(theta, v, float(sm.tau2))
    hk = pooling._hksj(theta, w, mu, 0.95)
    assert mu == pytest.approx(float(sm.mean_effect_re))
    assert hk["se"] ** 2 == pytest.approx(float(sm.var_hksj_re))
    ci_re_hksj = sm.conf_int(use_t=True)[3]
    assert (hk["lo"], hk["hi"]) == pytest.approx(tuple(float(x) for x in ci_re_hksj))
    # our REML tau2 maximises the REML likelihood (no competing value does better on a grid)
    tau2 = pooling.reml_tau2(theta, v)
    best = max(pooling._reml_loglik(t, theta, v) for t in np.linspace(0, 0.01, 2001))
    assert pooling._reml_loglik(tau2, theta, v) >= best - 1e-9


# ----------------------------------------------------------------------------- edge cases
def test_single_family_two_families_and_the_five_family_rule():
    one = pooling.pool([{"model": "a", "estimate": 0.1, "variance": 1e-4}, {"model": "b", "estimate": 0.12, "variance": 1e-4}],
                       families={"a": "F1", "b": "F1"}, direction=+1)
    assert one.method == "single_family" and not one.pooled and one.unreliable
    assert one.on_predicted_side is None and one.k_of_n == "1 of 1 tokenizer families"
    assert one.alternative["pooled"] is False and one.alternative["lo"] == one.alternative["hi"]
    two = pooling.pool([{"model": "a", "estimate": 0.1, "variance": 1e-4}, {"model": "b", "estimate": 0.12, "variance": 1e-4}],
                       families={"a": "F1", "b": "F2"}, direction=+1)
    assert two.pooled and two.unreliable and "tau2 not estimable" in two.unreliable_reason and two.df == 1
    assert two.alternative["pooled"] and "unreliable" in " ".join(two.alternative["notes"])
    four = pooling.pool([{"model": f"m{i}", "estimate": 0.1, "variance": 1e-4} for i in range(4)],
                        families={f"m{i}": f"F{i}" for i in range(4)}, direction=+1)
    assert four.unreliable and "fewer than five informative families" in four.unreliable_reason
    assert four.all_informative_families_on_side is True            # the per-family decision H3 falls back on
    four_h4 = pooling.pool([{"model": f"m{i}", "estimate": 0.1, "variance": 1e-4} for i in range(4)],
                           families={f"m{i}": f"F{i}" for i in range(4)}, min_reliable_families=3)
    assert not four_h4.unreliable and four_h4.min_reliable_families == 3
    # N from the panel, k from the data: two of the panel's seven families report
    part = pooling.pool([{"model": "a", "estimate": 0.1, "variance": 1e-4}, {"model": "b", "estimate": 0.0, "variance": 1e-4}],
                        families={"a": "F1", "b": "F2"}, panel_families=[f"F{i}" for i in range(1, 8)])
    assert part.k_of_n == "2 of 7 tokenizer families" and part.n_panel_families == 7
    assert sum(not f["informative"] for f in part.per_family) == 5
    assert pooling.pool([{"model": "a", "estimate": 0.1, "variance": 1e-4}], families={"a": "F1"},
                        panel_families=7).n_panel_families == 7


def test_k_of_n_never_exceeds_n():
    """A record whose family is outside the panel's labels (or more families than the panel count) is a disagreement
    between the family map and the panel, refused rather than printed as '3 of 2 tokenizer families'."""
    ests = [{"model": f"m{i}", "estimate": 0.1, "variance": 1e-4} for i in range(3)]
    fams = {"m0": "F1", "m1": "F2", "m2": "F3"}
    with pytest.raises(ValueError, match="cannot exceed N"):
        pooling.pool(ests, families=fams, panel_families=["F1", "F2"])
    with pytest.raises(ValueError, match="cannot exceed N"):
        pooling.pool(ests, families=fams, panel_families=2)
    assert pooling.pool(ests, families=fams, panel_families=["F1", "F2", "F3", "F4"]).k_of_n == "3 of 4 tokenizer families"
    assert pooling.pool(ests, families=fams, panel_families=3).k_of_n == "3 of 3 tokenizer families"


def test_census_verdicts_undefined_cells_and_zero_variance_are_excluded_and_reported():
    fams = {"n1": "F1", "c1": "F2", "u1": "F3", "z1": "F4", "ok1": "F5", "ok2": "F6", "ok3": "F6"}
    ests = [{"model": "n1", "estimate": 0.0, "variance": 1e-4, "verdict": "normalizes"},
            {"model": "c1", "estimate": 0.2, "variance": 1e-4, "verdict": "corrupts"},
            {"model": "u1", "estimate": float("nan"), "variance": 1e-4},
            {"model": "z1", "estimate": 0.05, "variance": 0.0},
            {"model": "ok1", "estimate": 0.04, "variance": 1e-4, "verdict": "passes_through"},
            {"model": "ok2", "estimate": 0.06, "variance": 1e-4},
            {"model": "ok3", "estimate": 0.05, "variance": 1e-4}]
    res = pooling.pool(ests, families=fams, direction=+1)
    assert res.k_informative_families == 2 and res.n_models_pooled == 3 and res.k_of_n == "2 of 6 tokenizer families"
    reasons = {e["model"]: e for e in res.excluded}
    assert set(reasons) == {"n1", "c1", "u1", "z1"}
    assert "not a test" in reasons["n1"]["reason"] and not reasons["n1"]["kept_in_family_bootstrap"]
    assert "excluded and reported" in reasons["c1"]["reason"]
    assert reasons["u1"]["reason"] == "undefined cell"
    assert reasons["z1"]["reason"] == "zero variance" and reasons["z1"]["kept_in_family_bootstrap"]
    assert res.alternative["k"] == 3                                   # the zero-variance row stays in B
    assert any("alternative estimator pools 3 families" in n for n in res.notes)
    f6 = next(f for f in res.per_family if f["family"] == "F6")
    assert f6["n_models"] == 2 and f6["tau2_flag"] == "unstable (2 models)"
    f1 = next(f for f in res.per_family if f["family"] == "F1")
    assert not f1["informative"] and "every model excluded" in f1["reason"]
    with pytest.raises(ValueError, match="no informative family"):
        pooling.pool(ests[:2], families=fams)
    assert pooling.pool(ests, families=fams, method="family_bootstrap").k_informative_families == 3


def test_item_level_records_use_the_cluster_bootstrap_replicate_variance():
    a, b, cl = _paired_data(n_pairs=120, p_a=0.6, p_b=0.7, icc=0.4, seed=3)
    strata = np.where(cl % 2 == 0, "lexicon", "pseudo")
    d = b.astype(float) - a.astype(float)
    rv = pooling.replicate_variance(d, cl, strata=strata, n_boot=300, seed=0)
    assert rv["estimate"] == pytest.approx(d.mean()) and rv["n_clusters"] == 120 and rv["n_items"] == 480
    # the same draws as bootstrap.cluster_bootstrap: reproduce them with its helpers and the same seed
    inv, k = bootstrap._cluster_index(cl)
    groups = bootstrap._strata_of_clusters(inv, k, strata)
    rng = np.random.default_rng(0)
    boots = np.array([np.average(d, weights=bootstrap._resample_weights(inv, k, rng, groups)) for _ in range(300)])
    assert rv["variance"] == pytest.approx(boots.var(ddof=1))
    # close to the paired-t variance of base-pair means (the two routes estimate the same SE)
    pt = bootstrap.paired_t_base_pairs(a, b, cl)
    assert abs(np.sqrt(rv["variance"]) - pt.se) / pt.se < 0.25
    fams = {"x": "F1", "y": "F2", "z": "F3"}
    recs = [{"model": "x", "values": d, "clusters": cl, "strata": strata},
            {"model": "y", "estimate": pt.mean_diff, "se": pt.se},
            {"model": "z", "estimate": pt.mean_diff, "lo": pt.mean_diff - 1.96 * pt.se, "hi": pt.mean_diff + 1.96 * pt.se}]
    res = pooling.pool(recs, families=fams, n_boot=300)
    src = {f["family"]: f["variance_sources"] for f in res.per_family}
    assert src == {"F1": ["bootstrap_replicates"], "F2": ["se"], "F3": ["ci_width"]}
    assert pooling.variance_from_ci(-1.959964, 1.959964) == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError, match="needs"):
        pooling.pool([{"model": "x", "estimate": 0.1}], families=fams)


def test_pooled_result_does_not_depend_on_input_order():
    rng = np.random.default_rng(5)
    ests, fams = _simulate(rng, sizes=(3, 3, 2, 1, 1, 1))
    ref = pooling.pool(ests, families=fams, direction=-1).as_dict()
    shuffled = list(ests)
    random.Random(9).shuffle(shuffled)
    other = pooling.pool(shuffled, families=fams, direction=-1).as_dict()
    assert other == ref


def test_family_map_helpers():
    cfg = {"groups": {"g_panel": {"panel": True}, "g_variant": {"panel": False}},
           "models": [{"name": "a", "family": "fx", "group": "g_panel"}, {"name": "b", "family": "fy", "group": "g_panel"},
                      {"name": "a--bf16", "family": "fx", "group": "g_variant"}]}
    assert pooling.family_map_from_models_yaml(cfg) == {"a": "fx", "b": "fy"}
    assert pooling.family_map_from_models_yaml(cfg, panel_only=False) == {"a": "fx", "b": "fy", "a--bf16": "fx"}
    assert pooling.tokenizer_family_map({"tokenizer_families": {"a": "T1"}}) == {"a": "T1"}
    assert pooling.tokenizer_family_map({"families": {"by_model": {"a": "T1"}}}) == {"a": "T1"}
    with pytest.raises(ValueError, match="tokenizer_families"):
        pooling.tokenizer_family_map({"families": {"by_model": {}}})
    assert "NOT the tokenizer family" in pooling.family_map_from_models_yaml.__doc__


# ----------------------------------------------------------------------------- k of N models, TOST (H4)
def _pt(name, diff, p, n_clusters=250):
    return PairedTest(name=name, n=1000, acc_a=0.5, acc_b=0.5 + diff, diff=diff, n_only_a=0, n_only_b=0, p_mcnemar=p,
                      p_bootstrap=p, ci_lo=diff - 0.02, ci_hi=diff + 0.02, p_paired_t=p, t_paired=0.0, n_clusters=n_clusters)


def test_k_of_n_models_counts_holm_significant_effects_on_the_predicted_side():
    tests = [_pt("m1", -0.08, 0.0001), _pt("m2", -0.05, 0.004), _pt("m3", +0.06, 0.001), _pt("m4", -0.01, 0.4),
             _pt("m5", -0.03, 0.03)]
    with pytest.raises(ValueError, match="correct_family"):
        pooling.k_of_n_models(tests, direction=-1)
    st.correct_family(tests)
    out = pooling.k_of_n_models(tests, direction=-1)
    assert out["k"] == 2 and out["n"] == 5 and out["k_of_n"] == "2 of 5 models" and out["models"] == ["m1", "m2"]
    assert out["significant_wrong_side"] == ["m3"] and out["direction"] == -1 and out["alpha"] == constants.ALPHA
    assert out["p_source"] == ["paired_t"]
    assert pooling.k_of_n_models(tests, direction=None)["k"] == 3
    assert not pooling.tost_applies(tests[0]) and pooling.tost_applies(tests[3])


def test_tost_declares_equivalence_only_within_two_points():
    a, _, cl = _paired_data(n_pairs=400, p_a=0.6, icc=0.3, seed=7)
    rng = np.random.default_rng(1)
    b = a.copy()
    down = rng.choice(np.flatnonzero(a), 24, replace=False)      # 24 flips each way: d = 0 exactly, se > 0
    up = rng.choice(np.flatnonzero(~a), 24, replace=False)
    b[down], b[up] = False, True
    res = pooling.tost_paired(a, b, cl)
    assert res["mean_diff"] == pytest.approx(0.0) and res["se"] > 0
    assert res["margin"] == constants.TOST_EQUIVALENCE_MARGIN_POINTS / 100 and res["df"] == 399 and res["primary"]
    assert res["equivalent"] is True and res["p_tost"] < constants.ALPHA
    assert -0.02 < res["ci90"][0] and res["ci90"][1] < 0.02 and res["ci_level"] == pytest.approx(0.90)
    assert res["p_tost"] == max(res["p_lower"], res["p_upper"])
    _, b_far, _ = _paired_data(n_pairs=400, p_a=0.6, p_b=0.7, icc=0.3, seed=7)
    far = pooling.tost_paired(a, b_far, cl)
    assert far["equivalent"] is False and far["p_tost"] > 0.5
    few = pooling.tost_paired(a[:400], b[:400], cl[:400])      # 100 base pairs: below the paired-t floor
    assert few["n_clusters"] == 100 and not few["primary"]
    none = pooling.tost_paired([1, 0], [1, 0], [0, 0])
    assert none["equivalent"] is None and np.isnan(none["p_tost"])
    with pytest.raises(ValueError):
        pooling.tost_paired(a, b, cl, margin=0.0)


# ----------------------------------------------------------------------------- weighted precision (DD 10.1)
def _fixture_rows():
    # cell T1-V1: lexical stratum (N=900) 10/10 correct, pseudo stratum (N=100) 5/10 correct -> 0.95 (tests/test_validation.py)
    key, final = {}, {}
    for i in range(20):
        lex = i < 10
        rid = f"B-{i}"
        key[rid] = {"control": False, "cell": "T1-V1", "stratum": "T1|V1|" + ("lexicon" if lex else "pseudo"),
                    "stratum_population": 900 if lex else 100, "weight": 90.0 if lex else 10.0}
        final[rid] = "yes" if lex or i < 15 else "no"
    key["B-c"] = {"control": True, "cell": "T1-V1", "stratum": "T1|V1|lexicon", "stratum_population": None, "weight": None}
    final["B-c"] = "no"
    key["B-u"] = {"control": False, "cell": "T1-V1", "stratum": "T1|V1|pseudo", "stratum_population": 100, "weight": 10.0}
    final["B-u"] = None
    return final, key


def test_weighted_precision_reproduces_stratified_precision_and_adds_the_weighted_interval():
    final, key = _fixture_rows()
    ref = VA.stratified_precision(final, key)
    out = agreement.weighted_precision_from_rows(final, key)
    cell, pooled = out["T1-V1"], out["pooled"]
    assert cell["estimate"] == pytest.approx(ref["T1-V1"]["weighted_precision"]) == pytest.approx(0.95)
    assert pooled["estimate"] == pytest.approx(ref["pooled"]["weighted_precision"])
    assert cell["n_sampled"] == 20 and cell["n_correct"] == 15 and cell["n_population"] == 1000 and cell["n_strata"] == 2
    assert cell["n_unresolved"] == 1 and pooled["n_unresolved"] == 1
    # Var = W_pseudo^2 p(1-p)/n (1 - n/N) = 0.1^2 * 0.25 / 10 * 0.9 = 2.25e-4 (the lexical stratum is at 100 %)
    assert cell["se"] == pytest.approx(np.sqrt(2.25e-4))
    assert cell["lo"] < 0.95 < cell["hi"] and cell["method"] == "stratified_wilson_neff(fpc)"
    assert cell["n_eff"] == pytest.approx(0.95 * 0.05 / 2.25e-4)
    assert cell["ci_normal"][0] == pytest.approx(0.95 - 1.959964 * 0.015, abs=1e-6)
    # the weighted interval is tighter than the unweighted Wilson on 15/20 (the big stratum is exact)
    assert (cell["hi"] - cell["lo"]) < (ref["T1-V1"]["wilson_unweighted"][1] - ref["T1-V1"]["wilson_unweighted"][0])
    assert cell["coverage"] == 1.0 and cell["empty_strata"] == [] and cell["singleton_strata"] == []
    no_fpc = agreement.weighted_precision_from_rows(final, key, fpc=False)["T1-V1"]
    assert no_fpc["se"] > cell["se"] and no_fpc["method"] == "stratified_wilson_neff(no fpc)"


def test_weighted_precision_recovers_a_known_stratified_precision_with_near_nominal_coverage():
    rng = np.random.default_rng(42)
    pops = {"T1|V1|lexicon": (900, 0.97), "T1|V1|pseudo": (100, 0.70)}
    truth = sum(n * p for n, p in pops.values()) / sum(n for n, _ in pops.values())
    cover, ests = 0, []
    reps = 300
    for _ in range(reps):
        strata = {}
        for k, (n_pop, p) in pops.items():
            n_s = 27 if n_pop == 900 else 3                        # proportional allocation of 30
            n_c = int(rng.binomial(n_s, p))
            strata[k] = {"n_population": n_pop, "n_sampled": n_s, "n_correct": n_c}
        out = agreement.weighted_precision(strata)["T1-V1"]
        ests.append(out["estimate"])
        cover += out["lo"] <= truth <= out["hi"]
    assert abs(np.mean(ests) - truth) < 0.01                        # unbiased for the population precision
    assert cover / reps >= 0.90                                      # n_eff Wilson: close to nominal with n = 30


def test_weighted_precision_edge_cases():
    # every stratum at 100 %: Var = 0 -> Wilson on the raw n, the upper bound pinned at 1
    full = agreement.weighted_precision({"T1|V1|lexicon": {"n_population": 900, "n_sampled": 10, "n_correct": 10},
                                         "T1|V1|pseudo": {"n_population": 100, "n_sampled": 5, "n_correct": 5}})["T1-V1"]
    assert full["estimate"] == 1.0 and full["hi"] == 1.0 and full["lo"] < 1.0 and full["method"].startswith("wilson_raw_n")
    zero = agreement.weighted_precision({"T2|V2|lexicon": {"n_population": 50, "n_sampled": 5, "n_correct": 0}})["T2-V2"]
    assert zero["estimate"] == 0.0 and zero["lo"] == 0.0 and zero["hi"] > 0.0
    # a singleton stratum is flagged; an empty stratum is listed and lowers the coverage; the pool spans cells
    out = agreement.weighted_precision({"T1|V1|lexicon": {"n_population": 90, "n_sampled": 9, "n_correct": 8},
                                        "T1|V1|pseudo": {"n_population": 10, "n_sampled": 1, "n_correct": 1},
                                        "T1|V2|lexicon": {"n_population": 80, "n_sampled": 8, "n_correct": 6},
                                        "T1|V2|pseudo": {"n_population": 20, "n_sampled": 0, "n_correct": 0}})
    assert out["T1-V1"]["singleton_strata"] == ["T1|V1|pseudo"] and out["T1-V1"]["coverage"] == 1.0
    assert out["T1-V2"]["empty_strata"] == ["T1|V2|pseudo"] and out["T1-V2"]["coverage"] == pytest.approx(0.8)
    assert out["T1-V2"]["estimate"] == pytest.approx(0.75) and out["T1-V2"]["n_population"] == 100
    assert out["pooled"]["n_sampled"] == 18 and out["pooled"]["n_strata"] == 3 and out["pooled"]["coverage"] == pytest.approx(0.9)
    assert out["pooled"]["estimate"] == pytest.approx((90 * 8 / 9 + 10 * 1.0 + 80 * 0.75) / 180)
    assert set(out) == {"T1-V1", "T1-V2", "pooled"}
    # an explicit cell label overrides the key; a weight that is not N_h / n_h is refused; n > N is refused
    cell = agreement.weighted_precision({"s": {"n_population": 40, "n_sampled": 4, "n_correct": 4, "cell": "X", "weight": 10.0}})
    assert "X" in cell
    with pytest.raises(ValueError, match="weight"):
        agreement.weighted_precision({"s": {"n_population": 40, "n_sampled": 4, "n_correct": 4, "weight": 5.0}})
    with pytest.raises(ValueError, match="n_population"):
        agreement.weighted_precision({"s": {"n_population": 3, "n_sampled": 4, "n_correct": 1}})
    with pytest.raises(ValueError, match="level"):
        agreement.weighted_precision({"s": {"n_population": 30, "n_sampled": 4, "n_correct": 1}}, level=1.5)
