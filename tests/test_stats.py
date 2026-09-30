"""Statistics tests on synthetic data with known structure (design 8.2–8.6, 10.1)."""
import numpy as np
import pandas as pd
import pytest

from noilai import constants
from noilai.stats import agreement, bootstrap, dose_response, mediation, mixed, power, tables
from noilai.stats import tests as st


def _paired_data(n_pairs=300, items_per_pair=4, p_a=0.6, p_b=0.7, icc=0.3, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_pairs), items_per_pair)
    u = rng.normal(size=n_pairs)[clusters] * np.sqrt(icc)
    e_a = rng.normal(size=len(clusters)) * np.sqrt(1 - icc)
    e_b = rng.normal(size=len(clusters)) * np.sqrt(1 - icc)
    from scipy.stats import norm
    a = (u + e_a) < norm.ppf(p_a)
    b = (u + e_b) < norm.ppf(p_b)
    return a, b, clusters


# ----------------------------------------------------------------------------- bootstrap (8.2)
def test_cluster_bootstrap_ci_covers_the_true_value_and_is_wider_than_iid():
    a, _, cl = _paired_data(p_a=0.6, icc=0.6)
    ci = bootstrap.accuracy_ci(a, cl, n_boot=500)
    # the probit-threshold generator's marginal is exactly p_a = 0.6
    assert ci.lo <= 0.6 <= ci.hi and ci.lo <= ci.estimate <= ci.hi
    assert ci.n_clusters == 300 and ci.n_items == 1200 and ci.method == "bca" and not ci.small_cell
    ci_iid = bootstrap.accuracy_ci(a, np.arange(len(a)), n_boot=500)
    assert (ci.hi - ci.lo) > (ci_iid.hi - ci_iid.lo)     # clustering inflates the interval


def test_bca_and_percentile_coincide_on_a_symmetric_case_and_bca_is_the_default_from_50_clusters():
    a, _, cl = _paired_data(p_a=0.6, icc=0.3)
    ci_bca = bootstrap.accuracy_ci(a, cl, n_boot=1000)
    ci_pct = bootstrap.accuracy_ci(a, cl, n_boot=1000, bca=False)
    assert ci_bca.method == "bca" and ci_pct.method == "percentile"
    assert abs(ci_bca.lo - ci_pct.lo) < 0.01 and abs(ci_bca.hi - ci_pct.hi) < 0.01
    few, _, cl_few = _paired_data(n_pairs=constants.BCA_MIN_CLUSTERS - 1, icc=0.3)
    assert bootstrap.accuracy_ci(few, cl_few, n_boot=200).method == "percentile"


def test_small_cells_get_wilson_on_n_over_deff_and_are_flagged():
    ci = bootstrap.accuracy_ci([1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1], n_boot=200)
    assert ci.small_cell and ci.method == "wilson_deff" and ci.lo < ci.estimate < ci.hi
    assert ci.hi - ci.lo > 0.3                      # not the degenerate [66.7, 66.7] of a 2-cluster percentile bootstrap
    # a 0 % / 100 % cell is a small cell whatever its size
    ones = bootstrap.accuracy_ci(np.ones(400), np.repeat(np.arange(100), 4), n_boot=100)
    assert ones.small_cell and ones.method == "wilson_deff" and ones.lo < 1.0 == ones.hi
    # a paired difference (values in {-1, 0, 1}) in a small cell keeps the percentile interval but carries the flag
    d = bootstrap.paired_difference_ci([1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], n_boot=200)
    assert d.small_cell and d.method == "percentile_small_cell"
    assert "small_cell" in d.as_dict() and "method" in d.as_dict()


def test_stratified_resampling_keeps_every_stratum_s_cluster_count():
    vals = np.array([1.0] * 40 + [0.0] * 60)          # 1 = lexical base pair, one item per pair
    cls = np.arange(100)
    strata = ["lex"] * 40 + ["pseudo"] * 60
    count = lambda x, w: float(np.sum(x * w))          # number of lexical pairs in the replicate
    ci_s = bootstrap.cluster_bootstrap(vals, cls, stat=count, n_boot=100, strata=strata, small_cell_rule=False, bca=False)
    ci_u = bootstrap.cluster_bootstrap(vals, cls, stat=count, n_boot=100, small_cell_rule=False, bca=False)
    assert ci_s.lo == ci_s.hi == 40.0                  # every replicate draws exactly 40 lexical pairs
    assert ci_u.lo < 40.0 < ci_u.hi                    # unstratified: the split varies
    with pytest.raises(ValueError):                    # a cluster cannot lie in two strata
        bootstrap.cluster_bootstrap([1.0, 0.0], [7, 7], strata=["lex", "pseudo"], n_boot=10)


def test_paired_difference_pvalue_and_paired_t_agree_on_effect_and_null():
    a, b, cl = _paired_data(p_a=0.6, p_b=0.75)
    ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=500)
    assert 0.05 < ci.estimate < 0.25 and ci.lo > 0
    p = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=500)
    pt = bootstrap.paired_t_base_pairs(a, b, cl)
    assert p < 0.01 and pt.p < 0.01 and pt.n_clusters == 300 and pt.primary and pt.mean_diff > 0.05
    a2, b2, cl2 = _paired_data(p_a=0.6, p_b=0.6, seed=3)
    assert bootstrap.paired_bootstrap_pvalue(a2, b2, cl2, n_boot=500) > 0.05
    assert bootstrap.paired_t_base_pairs(a2, b2, cl2).p > 0.05
    # the paired t accepts paraphrase-aggregated proportions (McNemar does not)
    pp = bootstrap.paired_t_base_pairs([0, 1 / 3, 1, 1], [1, 2 / 3, 0, 1], [0, 0, 1, 1])
    assert pp.n_clusters == 2 and not pp.primary and np.isfinite(pp.t)


# ----------------------------------------------------------------------------- paired tests (8.2, 8.3)
def test_mcnemar_exact_and_midp():
    assert st.mcnemar_exact(0, 0) == 1.0
    assert st.mcnemar_exact(10, 10) > 0.9
    assert st.mcnemar_exact(30, 5) < 0.001
    assert st.mcnemar_exact(8, 2, mid_p=False) >= st.mcnemar_exact(8, 2, mid_p=True)


def test_paired_test_refuses_aggregated_proportions_and_accepts_binary_inputs():
    with pytest.raises(ValueError, match="binary"):
        st.paired_test([0, 1 / 3, 1], [1, 2 / 3, 0], [0, 1, 2], bootstrap=False)
    with pytest.raises(ValueError, match="binary"):
        st.paired_test([0, 1, 1], [1, 0.5, 0], [0, 1, 2], bootstrap=False)
    t = st.paired_test([0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], bootstrap=False)
    assert t.n_only_a == 1 and t.n_only_b == 1
    t_int = st.paired_test(np.array([0, 1, 1, 0]), np.array([1, 1, 0, 0]), bootstrap=False)
    t_bool = st.paired_test([False, True, True, False], [True, True, False, False], bootstrap=False)
    assert t_int.acc_a == t_bool.acc_a == 0.5


def test_holm_monotone_and_controls():
    adj, rej = st.holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(np.diff(adj[np.argsort([0.01, 0.04, 0.03, 0.5])]) >= 0)
    assert adj[0] == pytest.approx(0.04) and rej[0] and not rej[3]
    adj2, _ = st.holm([])
    assert len(adj2) == 0


def test_paired_test_carries_the_primary_p_and_the_family_correction_uses_it():
    a, b, cl = _paired_data(p_a=0.6, p_b=0.72)
    t1 = st.paired_test(a, b, cl, name="nfd", n_boot=300)
    assert t1.n_clusters == 300 >= constants.PAIRED_T_MIN_BASE_PAIRS and t1.p_primary_name == "paired_t"
    assert t1.p_primary == t1.p_paired_t and t1.ci_method == "bca"
    a2, b2, _ = _paired_data(p_a=0.6, p_b=0.6, seed=9)
    t2 = st.paired_test(a2, b2, cl, name="placement_new", n_boot=300)
    st.correct_family([t1, t2])
    assert t1.significant and t1.p_adj >= t1.p_paired_t and t1.p_primary_source == "paired_t"
    assert t2.p_adj is not None and t1.diff > 0.05 and abs(t2.diff) < 0.05 and not t2.significant
    # below the base-pair floor the bootstrap p is primary; McNemar stays available as the labelled secondary
    a3, b3, cl3 = _paired_data(n_pairs=60, p_a=0.6, p_b=0.75, seed=2)
    t3 = st.paired_test(a3, b3, cl3, name="small", n_boot=300)
    assert t3.p_primary_name == "bootstrap" and t3.p_primary == t3.p_bootstrap
    st.correct_family([t3], use="mcnemar")
    assert t3.p_primary_source == "mcnemar" and t3.p_adj == pytest.approx(t3.p_mcnemar)
    d = t1.as_dict()
    assert d["n_only_b"] > d["n_only_a"] and "p_paired_t" in d


# ----------------------------------------------------------------------------- dose–response (8.6)
def test_delta_token_dose_response_and_zero_dose_contrast_recover_planted_structure():
    rng = np.random.default_rng(1)
    n = 2000
    cl = rng.integers(0, 500, size=n)
    dt = rng.choice([0, 0, 1, 2], size=n).astype(float)
    base = rng.random(n) < 0.8
    # the arm hurts only through token count: P(flip to wrong) = 0.25 per extra token
    flip = rng.random(n) < 0.25 * dt
    arm = base & ~flip
    dec = dose_response.decompose(base, arm, dt, cl, n_boot=300)
    assert dec.total_effect.estimate < -0.1
    assert abs(dec.form_only_effect.estimate) < 0.03          # no effect at Δtok = 0
    assert dec.token_associated_part.estimate < -0.1
    assert dec.effect_delta_zero is not None and abs(dec.effect_delta_zero.estimate) < 0.03
    assert dec.effect_delta_pos.estimate < -0.15
    # the forbidden estimand does not exist (design 8.6, 12.7)
    assert not hasattr(dec, "share_token_associated") and "share" not in " ".join(dec.as_dict())
    assert "associational" in dec.label
    # pure form effect: independent of token count
    arm2 = base & ~(rng.random(n) < 0.2)
    dec2 = dose_response.decompose(base, arm2, dt, cl, n_boot=300)
    assert dec2.form_only_effect.estimate < -0.08 and abs(dec2.slope_per_token.estimate) < 0.05
    assert isinstance(dec2.as_dict()["total_effect"], dict)
    assert mediation.decompose is dose_response.decompose         # the old module name only re-exports


# ----------------------------------------------------------------------------- power (8.5)
def test_power_arithmetic_matches_plan_figures():
    # the plan: 1,000 items at 70% -> about ±2.8 points; 94% power for a 5-point paired difference with 20% discordant
    assert power.ci_half_width(1000, 0.7) * 100 == pytest.approx(2.84, abs=0.02)
    assert 0.90 < power.mcnemar_power(1000, 0.05, 0.20) < 0.97
    assert power.n_for_half_width(0.05, 0.5) == 385
    assert power.design_effect(4, 0.25) == 1.75
    # DD 8.5 / PREREG §9 quote the conditional-on-discordant figures; the unconditional ones differ
    rates = (0.15, 0.20, 0.30, 0.40, 0.46)
    assert [power.n_for_mcnemar_power_conditional(0.10, r) for r in rates] == [100, 145, 227, 308, 356]
    assert [power.n_for_mcnemar_power(0.10, r) for r in rates] == [116, 155, 234, 312, 359]
    rows = power.power_table()
    assert len(rows) >= 8
    quoted = [r for r in rows if "n_conditional_on_discordant (quoted in DD 8.5 / PREREG 9)" in r]
    assert [r["n_conditional_on_discordant (quoted in DD 8.5 / PREREG 9)"] for r in quoted] == [100, 145, 227, 308, 356]


# ----------------------------------------------------------------------------- agreement (10.1)
def test_krippendorff_alpha_known_values():
    # perfect agreement
    r = [(i, c, i % 2) for i in range(50) for c in ("A", "B", "C")]
    assert agreement.krippendorff_alpha_nominal(r) == pytest.approx(1.0)
    # Krippendorff (2011) nominal example: alpha = 0.743 for the 4-coder, 12-unit data
    data = {
        "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    }
    rs = [(u, c, v) for c, vals in data.items() for u, v in enumerate(vals) if v is not None]
    assert agreement.krippendorff_alpha_nominal(rs) == pytest.approx(0.743, abs=0.01)
    est, lo, hi = agreement.alpha_bootstrap_ci(rs, n_boot=100)
    assert lo <= est <= hi
    assert 0 < agreement.percent_agreement(rs) <= 1


def test_gwet_ac1_hand_computed_and_robust_to_prevalence():
    # two coders, 10 items: 8 both yes, 1 A-yes/B-no, 1 both no
    #   p_a = 0.9; pi_yes = (8 + 0.5) / 10 = 0.85; p_e = 2 * 0.85 * 0.15 = 0.255; AC1 = (0.9 - 0.255) / 0.745 = 0.8658
    r = [(i, c, "yes") for i in range(8) for c in "AB"] + [(8, "A", "yes"), (8, "B", "no"), (9, "A", "no"), (9, "B", "no")]
    assert agreement.gwet_ac1(r) == pytest.approx(0.8658, abs=1e-3)
    assert agreement.percent_agreement(r) == pytest.approx(0.9)
    assert agreement.marginal_distribution(r) == {"no": pytest.approx(0.15), "yes": pytest.approx(0.85)}
    est, lo, hi = agreement.ac1_bootstrap_ci(r, n_boot=50)
    assert lo <= est <= hi
    # 96 % prevalence, two coders with 2 % independent error: alpha drops, AC1 tracks the 96 % raw agreement (design 10.1)
    rng = np.random.default_rng(0)
    rs = []
    for i in range(2000):
        truth = rng.random() < 0.96
        for c in "AB":
            rs.append((i, c, truth if rng.random() > 0.02 else (not truth)))
    assert agreement.krippendorff_alpha_nominal(rs) < 0.75 and agreement.gwet_ac1(rs) > 0.9
    assert agreement.gwet_ac1([(0, "A", 1)]) != agreement.gwet_ac1([(0, "A", 1), (0, "B", 1)])   # units need >= 2 labels


# ----------------------------------------------------------------------------- secondary fits (8.4)
def test_mixed_models_fit_the_design_8_4_coding_on_synthetic_data():
    from noilai.stats import e2

    rng = np.random.default_rng(2)
    n = 1200
    df = pd.DataFrame({
        "base_pair_id": rng.integers(0, 300, size=n).astype(str),
        "model": rng.choice(["m1", "m2", "m3"], size=n),
        "tokens_per_syllable": rng.choice([1.0, 1.5, 2.0, 3.0], size=n),
        "input_lexical": rng.random(n) < 0.5,
        "output_lexical": rng.random(n) < 0.2,
        "log_freq": rng.normal(size=n),
        "variant": rng.choice(["V1", "V2", "V3", "V4"], size=n),
        "task": rng.choice(["T1", "T2"], size=n),
    })
    df["align_among_split"] = np.where(df["tokens_per_syllable"] > 1, rng.random(n), np.nan)
    prep = e2.prepare_covariates(df)
    logit = 1.0 - 1.2 * prep["tps_w"] + 1.5 * prep["align_w"] + 0.5 * prep["input_lexical"]
    df["correct"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    assert "tps_w" in mixed.DEFAULT_FIXED and "align_w" in mixed.DEFAULT_FIXED and "boundary_alignment" not in mixed.DEFAULT_FIXED
    eff, _ = mixed.fit_gee(df)
    tab = mixed.effects_table(eff)
    assert tab.loc["tps_w", "estimate"] < -0.6 and tab.loc["align_w", "estimate"] > 0.5
    assert "whole_syllable_share" not in tab.index and "tokens_per_syllable" not in tab.index   # absent column dropped, raw count not fitted
    eff_vb, _ = mixed.fit_bayes_mixed(df, vb_iter=100)
    tab_vb = mixed.effects_table(eff_vb)
    assert tab_vb.loc["tps_w", "estimate"] < -0.4
    with pytest.raises(KeyError):
        mixed.fit_gee(df.drop(columns=["align_among_split"]))


# ----------------------------------------------------------------------------- tables (8.3)
def _scores(rng, models=(("A", 0.7), ("B", 0.5)), arms=(("nfc", 0.0), ("nfd", -0.15), ("placement_new", 0.0)), tasks=("T1",)):
    rows = []
    for model, p in models:
        for task in tasks:
            for arm, shift in arms:
                for i in range(400):
                    rows.append({"model": model, "task": task, "variant": "V1", "arm": arm, "prompt_id": "p0",
                                 "item_id": f"{task}-V1-{i:06d}", "base_pair_id": f"bp-{i//3}", "source": "lexical" if (i // 3) % 2 else "pseudo",
                                 "correct": bool(rng.random() < p + shift), "error_class": "correct"})
    return pd.DataFrame(rows)


def test_tables_from_scores_use_the_table3_family_and_runner_arm_names():
    rng = np.random.default_rng(4)
    df = _scores(rng)
    acc = tables.accuracy_table(df, n_boot=200)
    assert set(acc["model"]) == {"A", "B"} and acc.loc[acc["model"] == "A", "acc"].iloc[0] > acc.loc[acc["model"] == "B", "acc"].iloc[0]
    assert "ci_method" in acc and "small_cell" in acc
    # defaults: baseline nfc, the 14-cell Table 3 family (constants.HOLM_FAMILY_TABLE3) in the runner's arm names
    assert tables.DEFAULT_ARMS == ("nfd", "win1258", "placement_new", "strip_tones", "strip_all")
    fam = tables.table3_family()
    assert len(fam) == constants.HOLM_FAMILY_TABLE3_CELLS == 14 and ("win1258", "XCOPA", "main") in fam and ("placement_new", "T3", "c2") in fam
    inter = tables.intervention_table(df, n_boot=300)
    # only the cells present for this file are realized: nfd x T1 and placement_new x T1 per model row
    assert set(inter["arm"]) == {"nfd", "placement_new"} and (inter["family_size"] == 2).all() and (inter["family_defined"] == 14).all()
    nfd_A = inter[(inter["model"] == "A") & (inter["arm"] == "nfd")].iloc[0]
    assert nfd_A["effect"] < -0.05 and nfd_A["significant"] and nfd_A["p_source"] == "bootstrap"   # 134 base pairs < 200
    assert nfd_A["n_base_pairs"] == 134 and "p_mcnemar" in inter and "p_paired_t" in inter
    # the legacy arms= argument builds the family as arms x tasks present
    inter2 = tables.intervention_table(df, arms=("nfd",), n_boot=200)
    assert set(inter2["arm"]) == {"nfd"} and (inter2["family_size"] == 1).all()
    tex = tables.to_latex_accuracy(acc, "Accuracy", "tab:acc")
    assert "\\begin{tabular}" in tex and "A &" in tex
    tex2 = tables.to_latex_interventions(inter, "Effects", "tab:int", normalizes={"A": False, "B": True})
    assert "Normalizes NFD" in tex2 and "$^{*}$" in tex2 and "Family" in tex2
    assert "within each model row" in tex2 and "McNemar" in tex2 and "model $\\times$ task family" not in tex2
    eb = tables.error_breakdown(df)
    assert eb.loc[("A", "T1"), "correct"] == 1.0


def test_intervention_table_grows_the_family_across_tasks_and_uses_the_paired_t_from_200_base_pairs():
    rng = np.random.default_rng(5)
    df = _scores(rng, models=(("A", 0.7),), tasks=("T1", "T3"))
    df["base_pair_id"] = df["item_id"].str.replace(r"^T\d-V1-", "bp-", regex=True)     # 400 base pairs per task
    inter = tables.intervention_table(df, n_boot=200)
    assert set(zip(inter["arm"], inter["task"])) == {("nfd", "T1"), ("nfd", "T3"), ("placement_new", "T1"), ("placement_new", "T3")}
    assert (inter["family_size"] == 4).all() and (inter["p_source"] == "paired_t").all() and (inter["n_base_pairs"] == 400).all()
