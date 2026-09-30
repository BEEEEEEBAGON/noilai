"""Statistics tests on synthetic data with known structure."""
import numpy as np
import pandas as pd
import pytest

from noilai.stats import agreement, bootstrap, mediation, mixed, power, tables
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


def test_cluster_bootstrap_ci_covers_and_is_wider_than_iid():
    a, _, cl = _paired_data(p_a=0.6, icc=0.6)
    ci = bootstrap.accuracy_ci(a, cl, n_boot=500)
    assert ci.lo <= ci.estimate <= ci.hi and ci.n_clusters == 300 and ci.n_items == 1200
    ci_iid = bootstrap.accuracy_ci(a, np.arange(len(a)), n_boot=500)
    assert (ci.hi - ci.lo) > (ci_iid.hi - ci_iid.lo)     # clustering inflates the interval


def test_paired_difference_and_pvalue_detect_effect():
    a, b, cl = _paired_data(p_a=0.6, p_b=0.75)
    ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=500)
    assert 0.05 < ci.estimate < 0.25 and ci.lo > 0
    p = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=500)
    assert p < 0.01
    a2, b2, cl2 = _paired_data(p_a=0.6, p_b=0.6, seed=3)
    assert bootstrap.paired_bootstrap_pvalue(a2, b2, cl2, n_boot=500) > 0.05


def test_mcnemar_exact_and_midp():
    assert st.mcnemar_exact(0, 0) == 1.0
    assert st.mcnemar_exact(10, 10) > 0.9
    assert st.mcnemar_exact(30, 5) < 0.001
    assert st.mcnemar_exact(8, 2, mid_p=False) >= st.mcnemar_exact(8, 2, mid_p=True)


def test_holm_monotone_and_controls():
    adj, rej = st.holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(np.diff(adj[np.argsort([0.01, 0.04, 0.03, 0.5])]) >= 0)
    assert adj[0] == pytest.approx(0.04) and rej[0] and not rej[3]
    adj2, _ = st.holm([])
    assert len(adj2) == 0


def test_paired_test_and_family_correction():
    a, b, cl = _paired_data(p_a=0.6, p_b=0.72)
    t1 = st.paired_test(a, b, cl, name="nfd", n_boot=300)
    a2, b2, _ = _paired_data(p_a=0.6, p_b=0.6, seed=9)
    t2 = st.paired_test(a2, b2, cl, name="placement_old", n_boot=300)
    st.correct_family([t1, t2])
    assert t1.significant and t1.p_adj >= t1.p_mcnemar
    assert t2.p_adj is not None and t1.diff > 0.05 and abs(t2.diff) < 0.05
    d = t1.as_dict()
    assert d["n_only_b"] > d["n_only_a"]


def test_mediation_decomposition_recovers_planted_structure():
    rng = np.random.default_rng(1)
    n = 2000
    cl = rng.integers(0, 500, size=n)
    dt = rng.choice([0, 0, 1, 2], size=n).astype(float)
    base = rng.random(n) < 0.8
    # the arm hurts only through token count: P(flip to wrong) = 0.25 per extra token
    flip = rng.random(n) < 0.25 * dt
    arm = base & ~flip
    dec = mediation.decompose(base, arm, dt, cl, n_boot=300)
    assert dec.total_effect.estimate < -0.1
    assert abs(dec.form_only_effect.estimate) < 0.03          # no effect at Δtok = 0
    assert dec.share_token_associated.estimate > 0.8
    assert dec.effect_delta_zero is not None and abs(dec.effect_delta_zero.estimate) < 0.03
    assert dec.effect_delta_pos.estimate < -0.15
    # pure form effect: independent of token count
    arm2 = base & ~(rng.random(n) < 0.2)
    dec2 = mediation.decompose(base, arm2, dt, cl, n_boot=300)
    assert dec2.form_only_effect.estimate < -0.08 and abs(dec2.slope_per_token.estimate) < 0.05
    assert isinstance(dec2.as_dict()["total_effect"], dict)


def test_power_arithmetic_matches_plan_figures():
    # the plan: 1,000 items at 70% -> about ±2.8 points; 94% power for a 5-point paired difference with 20% discordant
    assert power.ci_half_width(1000, 0.7) * 100 == pytest.approx(2.84, abs=0.02)
    assert 0.90 < power.mcnemar_power(1000, 0.05, 0.20) < 0.97
    assert power.n_for_mcnemar_power(0.05, 0.20) < 1000
    assert power.n_for_half_width(0.05, 0.5) == 385
    assert power.design_effect(4, 0.25) == 1.75
    assert len(power.power_table()) >= 8


def test_krippendorff_alpha_known_values():
    # perfect agreement
    r = [(i, c, i % 2) for i in range(50) for c in ("A", "B", "C")]
    assert agreement.krippendorff_alpha_nominal(r) == pytest.approx(1.0)
    # Krippendorff (2011) nominal example: alpha = 0.691 for the 4-coder, 12-unit data
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


def test_mixed_models_fit_on_synthetic_data():
    rng = np.random.default_rng(2)
    n = 1200
    df = pd.DataFrame({
        "base_pair_id": rng.integers(0, 300, size=n).astype(str),
        "model": rng.choice(["m1", "m2", "m3"], size=n),
        "tokens_per_syllable": rng.choice([1.0, 1.5, 2.0, 3.0], size=n),
        "boundary_alignment": rng.random(n),
        "input_lexical": rng.random(n) < 0.5,
        "output_lexical": rng.random(n) < 0.2,
        "log_freq": rng.normal(size=n),
        "variant": rng.choice(["V1", "V2", "V3", "V4"], size=n),
        "task": rng.choice(["T1", "T2"], size=n),
    })
    logit = 1.0 - 1.2 * df["tokens_per_syllable"] + 1.5 * df["boundary_alignment"] + 0.5 * df["input_lexical"]
    df["correct"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    eff, _ = mixed.fit_gee(df)
    tab = mixed.effects_table(eff)
    assert tab.loc["tokens_per_syllable", "estimate"] < -0.6
    assert tab.loc["boundary_alignment", "estimate"] > 0.5
    eff_vb, _ = mixed.fit_bayes_mixed(df, vb_iter=100)
    tab_vb = mixed.effects_table(eff_vb)
    assert tab_vb.loc["tokens_per_syllable", "estimate"] < -0.4


def test_tables_from_scores():
    rng = np.random.default_rng(4)
    rows = []
    for model, p in (("A", 0.7), ("B", 0.5)):
        for arm, shift in (("nfc", 0.0), ("nfd", -0.15), ("placement_old", 0.0)):
            for i in range(400):
                rows.append({"model": model, "task": "T1", "variant": "V1", "arm": arm, "prompt_id": "p0",
                             "item_id": f"T1-V1-{i:06d}", "base_pair_id": f"bp-{i//3}", "correct": bool(rng.random() < p + shift),
                             "error_class": "correct"})
    df = pd.DataFrame(rows)
    acc = tables.accuracy_table(df, n_boot=200)
    assert set(acc["model"]) == {"A", "B"} and acc.loc[acc["model"] == "A", "acc"].iloc[0] > acc.loc[acc["model"] == "B", "acc"].iloc[0]
    inter = tables.intervention_table(df, arms=("nfd", "placement_old"), n_boot=300)
    nfd_A = inter[(inter["model"] == "A") & (inter["arm"] == "nfd")].iloc[0]
    assert nfd_A["effect"] < -0.05 and nfd_A["significant"]
    tex = tables.to_latex_accuracy(acc, "Accuracy", "tab:acc")
    assert "\\begin{tabular}" in tex and "A &" in tex
    tex2 = tables.to_latex_interventions(inter, "Effects", "tab:int", normalizes={"A": False, "B": True})
    assert "Normalizes NFD" in tex2 and "$^{*}$" in tex2
    eb = tables.error_breakdown(df)
    assert eb.loc[("A", "T1"), "correct"] == 1.0
