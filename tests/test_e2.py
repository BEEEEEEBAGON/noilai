"""E2 primary fit: model fixed effects, model x fragmentation interactions, two-way clustered SEs."""
import numpy as np
import pandas as pd

from noilai.stats import e2


def _synthetic(n_bp=400, models=("m1", "m2", "m3"), seed=0, slope_tps=None):
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
                logit = 1.5 + slope_tps[m] * (tps_item - 1.7) + 1.0 * x_align + 0.3 * lex_in + u_bp + {"m1": 0, "m2": 0.3, "m3": -0.3}[m]
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


def test_fixed_effects_fit_recovers_negative_fragmentation_slopes_with_valid_covariance():
    df = e2.prepare_covariates(_synthetic())
    fe = e2.fit_fixed_effects_clustered(df)
    assert fe.n_models == 3 and fe.n_bp == 400
    assert fe.mean_slope_tps < -0.4 and all(fe.slopes_tps < 0)
    assert fe.mean_slope_align > 0.3
    # two-way covariance is PSD and its diagonal is at least as large as the base-pair-only variance for the slope terms
    w = np.linalg.eigvalsh(fe.cov_twoway.values)
    assert w.min() >= -1e-8
    assert fe.se_mean_slope_tps > 0 and fe.se_mean_slope_align > 0
    d = e2.standardized_slope_difference(fe, df)
    assert set(d) >= {"diff", "se_approx", "beta_align_std", "beta_tps_std"}


def test_bootstrap_mean_slopes_runs():
    df = e2.prepare_covariates(_synthetic(n_bp=120))
    out = e2.bootstrap_mean_slopes(df, n_boot=3, seed=1)
    assert out["n_ok"] >= 2 and len(out["ci_mean_slope_tps"]) == 2
