"""E2 primary analysis (design 8.4): what predicts failure, with model fixed effects and
model x fragmentation interactions, two-way cluster-robust standard errors.

Covariate coding (item x tokenizer quantities):
  split    = 1[tokens per syllable > 1]
  tps_w    = tokens per syllable minus the model's mean (within-model centering; Mundlak)
  tps_b    = the model's mean (between)
  align_w  = share of aligned boundaries among SPLIT syllables (0 when split = 0); this is
             the alignment coding that is not collinear with the count.
Outcome: correctness aggregated over paraphrases to binomial counts per item x model, or
one row per (item, paraphrase) with the same clustering.

Primary fit: logit  correct ~ C(model) + split + tps_w + align_w + lexical_in + lexical_out
                             + logfreq + C(variant) + C(task) + C(model):tps_w + C(model):align_w
with cluster-robust covariance clustered two-way by base pair and by model
(Cameron-Gelbach-Miller: V = V_bp + V_model - V_(bp x model), negative eigenvalues set to 0).
The paper reports the mean of the per-model fragmentation slopes with a base-pair cluster
bootstrap CI (refits; keep n_boot modest) and, as the generalization check, a random-slope
model fitted elsewhere (bambi/PyMC or lme4), see docs/DESIGN_DECISIONS.md 8.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def prepare_covariates(df: pd.DataFrame, tps_col: str = "tokens_per_syllable", align_col: str = "align_among_split",
                       model_col: str = "model") -> pd.DataFrame:
    """Add split, tps_w, tps_b, align_w columns. `align_col` must be the alignment share among
    split syllables (from scripts/audit_items.py); when the item has no split syllable the
    column may be NaN and becomes 0."""
    d = df.copy()
    d["split"] = (d[tps_col] > 1).astype(int)
    means = d.groupby(model_col)[tps_col].transform("mean")
    d["tps_w"] = d[tps_col] - means
    d["tps_b"] = means
    d["align_w"] = np.where(d["split"] == 1, d[align_col].fillna(0.0), 0.0)
    for c in ("input_lexical", "output_lexical", "correct"):
        if c in d:
            d[c] = d[c].astype(int)
    if "log_freq" not in d and "input_freq_min" in d:
        d["log_freq"] = np.log1p(d["input_freq_min"])
    return d


PRIMARY_FORMULA = ("correct ~ C(model) + split + tps_w + align_w + input_lexical + output_lexical + log_freq"
                   " + C(variant) + C(task) + C(model):tps_w + C(model):align_w")


@dataclass
class FEResult:
    params: pd.Series
    cov_twoway: pd.DataFrame
    cov_bp: pd.DataFrame
    slopes_tps: pd.Series          # per-model slope of tps_w
    slopes_align: pd.Series        # per-model slope of align_w
    mean_slope_tps: float
    mean_slope_align: float
    se_mean_slope_tps: float
    se_mean_slope_align: float
    n_obs: int
    n_bp: int
    n_models: int
    result: object


def _effective_formula(d: pd.DataFrame, formula: str) -> str:
    """Drop C(col) terms (and interactions with them) whose column has a single level; a
    constant categorical is collinear with the intercept and makes the Hessian singular
    (e.g. a T1-only run has no task contrast)."""
    import re
    lhs, rhs = formula.split("~")
    terms = [t.strip() for t in rhs.split("+")]
    keep = []
    for t in terms:
        cols = re.findall(r"C\((\w+)\)", t)
        if any(d[c].nunique() < 2 for c in cols if c in d):
            continue
        plain = [c for c in re.findall(r"\b(\w+)\b", t) if c in d and c not in cols]
        if any(d[c].nunique() < 2 for c in plain):
            continue
        keep.append(t)
    return f"{lhs.strip()} ~ " + " + ".join(keep)


def _psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2)
    w = np.clip(w, 0, None)
    return (v * w) @ v.T


def fit_fixed_effects_clustered(df: pd.DataFrame, formula: str = PRIMARY_FORMULA, bp_col: str = "base_pair_id",
                                model_col: str = "model", maxiter: int = 200) -> FEResult:
    import statsmodels.formula.api as smf

    d = df.copy()
    d[bp_col] = d[bp_col].astype(str)
    d[model_col] = d[model_col].astype(str)
    formula = _effective_formula(d, formula)
    mod = smf.logit(formula, data=d)
    res = mod.fit(disp=0, maxiter=maxiter, method="bfgs")
    # one-way cluster-robust covariances at the same point estimate
    covs = {}
    for name, groups in (("bp", d[bp_col]), ("model", d[model_col]), ("both", d[bp_col] + "|" + d[model_col])):
        codes = pd.factorize(groups)[0]
        r = mod.fit(disp=0, maxiter=maxiter, method="bfgs", start_params=res.params.values,
                    cov_type="cluster", cov_kwds={"groups": codes})
        covs[name] = np.asarray(r.cov_params())
    V = _psd(covs["bp"] + covs["model"] - covs["both"])
    names = list(res.params.index)
    cov_twoway = pd.DataFrame(V, index=names, columns=names)
    cov_bp = pd.DataFrame(covs["bp"], index=names, columns=names)
    models = sorted(d[model_col].unique())
    ref = models[0]                       # the reference level absorbed into the main effect

    def per_model(term: str) -> tuple[pd.Series, float, float]:
        weights = []
        vals = {}
        for m in models:
            w = pd.Series(0.0, index=names)
            w[term] = 1.0
            inter = f"C(model)[T.{m}]:{term}"
            if m != ref and inter in w.index:
                w[inter] = 1.0
            vals[m] = float(w @ res.params)
            weights.append(w.values)
        Wm = np.mean(np.vstack(weights), axis=0)
        mean = float(Wm @ res.params.values)
        se = float(np.sqrt(Wm @ V @ Wm))
        return pd.Series(vals), mean, se

    s_tps, m_tps, se_tps = per_model("tps_w")
    s_al, m_al, se_al = per_model("align_w")
    return FEResult(params=res.params, cov_twoway=cov_twoway, cov_bp=cov_bp, slopes_tps=s_tps, slopes_align=s_al,
                    mean_slope_tps=m_tps, mean_slope_align=m_al, se_mean_slope_tps=se_tps, se_mean_slope_align=se_al,
                    n_obs=int(res.nobs), n_bp=int(d[bp_col].nunique()), n_models=len(models), result=res)


def standardized_slope_difference(fe: FEResult, df: pd.DataFrame) -> dict:
    """H1's quantity: |beta(align_w)| * sd(align_w) - |beta(tps_w)| * sd(tps_w), on the mean
    per-model slopes, with a delta-method SE from the two-way covariance of the means."""
    sd_a = float(df["align_w"].std())
    sd_t = float(df["tps_w"].std())
    diff = abs(fe.mean_slope_align) * sd_a - abs(fe.mean_slope_tps) * sd_t
    var = (sd_a * fe.se_mean_slope_align) ** 2 + (sd_t * fe.se_mean_slope_tps) ** 2   # ignores the covariance term
    return {"diff": diff, "se_approx": float(np.sqrt(var)), "beta_align_std": fe.mean_slope_align * sd_a,
            "beta_tps_std": fe.mean_slope_tps * sd_t}


def bootstrap_mean_slopes(df: pd.DataFrame, n_boot: int = 200, seed: int = 0, bp_col: str = "base_pair_id",
                          formula: str = PRIMARY_FORMULA, model_col: str = "model") -> dict:
    """Base-pair cluster bootstrap of the mean per-model slopes (refits the primary model)."""
    rng = np.random.default_rng(seed)
    bps = df[bp_col].unique()
    tps, al = [], []
    for _ in range(n_boot):
        pick = rng.choice(bps, size=len(bps), replace=True)
        parts = [df[df[bp_col] == b] for b in pick]
        boot = pd.concat(parts, ignore_index=True)
        try:
            fe = fit_fixed_effects_clustered(boot, formula=formula, bp_col=bp_col, model_col=model_col)
        except Exception:  # noqa: BLE001 - a singular replicate is skipped and counted
            continue
        tps.append(fe.mean_slope_tps)
        al.append(fe.mean_slope_align)
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if a else [np.nan, np.nan]  # noqa: E731
    return {"n_ok": len(tps), "ci_mean_slope_tps": q(tps), "ci_mean_slope_align": q(al)}
