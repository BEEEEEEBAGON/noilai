"""Item-level regression: what predicts failure (E2).

The pre-registered model is a logistic regression of item correctness on tokenization
covariates with random intercepts for base pair and for model:

  correct ~ tokens_per_syllable + boundary_alignment + input_lexical + output_lexical
            + log_freq + C(variant) + C(task)  +  (1 | base_pair_id) + (1 | model)

statsmodels has no frequentist GLMM; we fit it with BinomialBayesMixedGLM (variational
Bayes, Gaussian random effects) and report posterior means and SDs of the fixed effects.
As a robustness check we fit the same fixed effects with GEE (exchangeable working
correlation, clustered by base pair) per model, which gives sandwich standard errors
without distributional assumptions on the random effects. Both are reported in the
appendix; the sign and magnitude of the alignment and token-count coefficients are the
quantities H1 speaks about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_FIXED = "tokens_per_syllable + boundary_alignment + input_lexical + output_lexical + log_freq + C(variant) + C(task)"


@dataclass
class FixedEffect:
    term: str
    estimate: float
    sd: float
    z: float
    p: Optional[float]
    odds_ratio: float


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ("input_lexical", "output_lexical", "correct"):
        if col in d:
            d[col] = d[col].astype(int)
    if "log_freq" not in d and "input_freq_min" in d:
        d["log_freq"] = np.log1p(d["input_freq_min"])
    return d


def fit_bayes_mixed(df: pd.DataFrame, fixed: str = DEFAULT_FIXED, random_effects: Sequence[str] = ("base_pair_id", "model"),
                    vb_iter: int = 500) -> tuple[list[FixedEffect], object]:
    """Variational-Bayes logistic mixed model with one random intercept per grouping factor."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = _prepare(df)
    vc = {name: f"0 + C({name})" for name in random_effects}
    model = BinomialBayesMixedGLM.from_formula(f"correct ~ {fixed}", vc, d)
    res = model.fit_vb(verbose=False)
    names = list(model.exog_names)
    means = np.asarray(res.fe_mean)
    sds = np.asarray(res.fe_sd)
    out = []
    for n, m, s in zip(names, means, sds):
        z = m / s if s > 0 else np.nan
        from scipy import stats as sps
        p = float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else None
        out.append(FixedEffect(n, float(m), float(s), float(z), p, float(np.exp(m))))
    return out, res


def fit_gee(df: pd.DataFrame, fixed: str = DEFAULT_FIXED, cluster: str = "base_pair_id") -> tuple[list[FixedEffect], object]:
    """GEE logistic regression clustered by base pair (robust sandwich SEs). Fit per model or
    with C(model) in the fixed part."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = _prepare(df)
    model = smf.gee(f"correct ~ {fixed}", groups=d[cluster], data=d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    out = []
    for n in res.params.index:
        m, s = float(res.params[n]), float(res.bse[n])
        z = m / s if s > 0 else np.nan
        out.append(FixedEffect(n, m, s, float(z), float(res.pvalues[n]), float(np.exp(m))))
    return out, res


def effects_table(effects: Sequence[FixedEffect]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in effects]).set_index("term")
