"""E2 (design 8.4, items 23/28): model fixed effects + model x fragmentation interactions;
primary covariance = base-pair cluster bootstrap; H1 decided by the nested LR/ΔAIC test;
two-way clustered SEs reported as a sensitivity only."""
import numpy as np
import pandas as pd
import pytest

from noilai.stats import e2


def _synthetic(n_bp=400, models=("m1", "m2", "m3"), seed=0, slope_tps=None, beta_align=1.0):
    rng = np.random.default_rng(seed)
    slope_tps = slope_tps or {"m1": -0.8, "m2": -1.2, "m3": -1.0}
    rows = []
    for b in range(n_bp):
        tps_item = rng.choice([1.0, 1.0, 1.5, 2.0, 3.0])
        align_item = rng.random() if tps_item > 1 else np.nan
        lex_in = rng.random() < 0.5
        u_bp = rng.normal(scale=0.5)
        for m in models:
            for para in range(3):
                x_align = align_item if tps_item > 1 else 0.0
                logit = 1.5 + slope_tps[m] * (tps_item - 1.7) + beta_align * x_align + 0.3 * lex_in + u_bp + {"m1": 0, "m2": 0.3, "m3": -0.3}[m]
                rows.append({"base_pair_id": f"bp{b}", "model": m, "tokens_per_syllable": tps_item,
                             "align_among_split": align_item, "input_lexical": lex_in, "output_lexical": False,
                             "log_freq": rng.normal(), "variant": rng.choice(["V1", "V2"]), "task": "T1",
                             "correct": rng.random() < 1 / (1 + np.exp(-logit))})
    return pd.DataFrame(rows)


def test_prepare_covariates_codes_split_and_within_model_centering():
    df = e2.prepare_covariates(_synthetic(n_bp=50))
    assert set(df["split"].unique()) <= {0, 1}
    assert abs(df.groupby("model")["tps_w"].mean()).max() < 1e-9
    assert (df.loc[df["split"] == 0, "align_w"] == 0).all()
    assert df["align_w"].between(0, 1).all()
    # the legacy all-syllable alignment (single-token syllables scored 1.0) is refused as align_w's source
    with pytest.raises(ValueError, match="boundary_alignment_mean"):
        e2.prepare_covariates(_synthetic(n_bp=5).rename(columns={"align_among_split": "boundary_alignment_mean"}),
                              align_col="boundary_alignment_mean")


def test_effective_formula_drops_absent_and_constant_terms():
    df = e2.prepare_covariates(_synthetic(n_bp=20))
    f = e2._effective_formula(df, e2.PRIMARY_FORMULA)
    assert "C(task)" not in f and "C(i_y_variant)" not in f and "C(model):tps_w" in f and "align_w" in f
    df["i_y_variant"] = np.where(np.arange(len(df)) % 2 == 0, "i", "y")
    assert "C(i_y_variant)" in e2._effective_formula(df, e2.PRIMARY_FORMULA)


def test_fixed_effects_fit_recovers_negative_fragmentation_slopes_and_labels_the_sandwich_as_sensitivity():
    df = e2.prepare_covariates(_synthetic())
    fe = e2.fit_fixed_effects_clustered(df)
    assert fe.n_models == 3 and fe.n_bp == 400
    assert fe.mean_slope_tps < -0.4 and all(fe.slopes_tps < 0)
    assert fe.mean_slope_align > 0.3
    # two-way covariance is PSD; it is the reported SENSITIVITY, named as such
    w = np.linalg.eigvalsh(fe.cov_twoway.values)
    assert w.min() >= -1e-8
    assert fe.se_mean_slope_tps_twoway > 0 and fe.se_mean_slope_align_twoway > 0
    assert "sensitivity" in fe.covariance_label and not hasattr(fe, "se_mean_slope_tps")
    fast = e2.fit_fixed_effects_clustered(df, sandwich=False)
    assert fast.cov_twoway is None and fast.mean_slope_tps == pytest.approx(fe.mean_slope_tps)
    d = e2.standardized_slope_difference(fe, df)
    assert "descriptive" in d["label"] and "not the H1 decision" in e2.standardized_slope_difference.__doc__


def test_bootstrap_mean_slopes_is_the_primary_covariance_and_covers_the_planted_mean_slope():
    # planted per-model slopes -0.8 / -1.2 / -1.0 (mean -1.0): slope heterogeneity across models, as in design 8.4's simulation
    df = e2.prepare_covariates(_synthetic(n_bp=200, seed=1))
    out = e2.bootstrap_mean_slopes(df, n_boot=30, seed=1)
    assert out["n_ok"] >= 20 and len(out["ci_mean_slope_tps"]) == 2 and "primary" in out["label"]
    lo, hi = out["ci_mean_slope_tps"]
    assert lo < -1.0 < hi and hi - lo > 0 and out["se_mean_slope_tps"] > 0
    assert out["ci_mean_slope_align"][0] < 1.0 < out["ci_mean_slope_align"][1]


def test_nested_h1_test_rejects_with_the_planted_alignment_effect_and_not_without_it():
    df = e2.prepare_covariates(_synthetic())
    r = e2.nested_h1_test(df)
    assert r["df"] >= 3 and r["p"] < 0.05 and r["delta_aic"] > 2 and r["h1_supported"]
    assert r["mean_slope_align"] > 0 and r["mean_slope_tps"] < 0 and "split" in r["formula_m1"] and "align_w" not in r["formula_m0"]
    # no alignment effect planted: the nested test does not favour M1 (ΔAIC < 2 or p > 0.05) on most seeds
    verdicts = []
    for seed in range(3):
        r0 = e2.nested_h1_test(e2.prepare_covariates(_synthetic(seed=seed, beta_align=0.0)))
        verdicts.append(r0["p"] > 0.05 or r0["delta_aic"] < 2)
        assert not r0["h1_supported"] or r0["p"] >= 0.01     # never a confident false positive on 3 seeds
    assert sum(verdicts) >= 2


def test_identifiability_floor_counts_misaligned_split_syllables_per_tokenizer():
    rows = [{"tokenizer": "g3", "encoding": "nfc", "input": {"n_misaligned_split": 2}} for _ in range(200)]
    rows += [{"tokenizer": "q3", "encoding": "nfc", "n_misaligned_split": 1} for _ in range(100)]
    rows += [{"tokenizer": "g3", "encoding": "nfd", "input": {"n_misaligned_split": 5}} for _ in range(50)]   # other encoding: ignored
    out = e2.identifiability_floor(rows, families={"g3": "gemma", "q3": "qwen"})
    assert out["per_tokenizer"]["g3"]["n_misaligned_split"] == 400 and out["per_tokenizer"]["g3"]["clears_floor"]
    assert out["per_tokenizer"]["q3"]["n_misaligned_split"] == 100 and not out["per_tokenizer"]["q3"]["clears_floor"]
    assert out["tokenizers_clearing"] == ["g3"] and out["families_clearing"] == ["gemma"] and not out["h1_identified"]
    assert out["floor"] == e2.H1_MISALIGNED_SPLIT_FLOOR == 300
    with pytest.raises(KeyError):
        e2.identifiability_floor([{"tokenizer": "x", "input": {"tokens_mean": 1.0}}])
    assert e2.identifiability_floor(pd.DataFrame(rows[:200]))["per_tokenizer"]["g3"]["n_misaligned_split"] == 400
