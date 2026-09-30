"""Item-level regression, secondary fits of the E2 specification (design 8.4).

The PRIMARY E2 fit is `noilai.stats.e2` (model fixed effects + model x fragmentation
interactions, base-pair cluster bootstrap, nested LR/ΔAIC for H1). This module holds the two
secondary fits of the SAME covariate coding (`split`, within-model-centred `tps_w`, `align_w`
= alignment among split syllables; `e2.prepare_covariates`):

  * `fit_gee`: GEE logistic regression clustered by base pair (exchangeable working
    correlation, sandwich SEs) — the ROBUSTNESS fit;
  * `fit_bayes_mixed`: `BinomialBayesMixedGLM.fit_vb` with random intercepts for base pair
    and model — demoted to POINT ESTIMATES and variance-component screening only (its
    mean-field posterior SDs are not reported as uncertainty; design 8.4, item 53).
`whole_syllable_share` (a per-tokenizer covariate; H2) enters here, where model is a random
intercept, and is absorbed by the fixed effects of the primary fit. Terms whose columns are
absent or constant are dropped before fitting (`e2._effective_formula`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FIXED = ("split + tps_w + align_w + input_lexical + output_lexical + log_freq + whole_syllable_share"
                 " + C(variant) + C(task) + C(i_y_variant)")


@dataclass
class FixedEffect:
    term: str
    estimate: float
    sd: float
    z: float
    p: float | None
    odds_ratio: float


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """The design 8.4 coding: when the frame carries the raw per-syllable quantities but not
    `tps_w`, derive split / tps_w / align_w with `e2.prepare_covariates` (alignment among
    split syllables from `align_among_split`; the legacy `boundary_alignment_mean` is refused
    there)."""
    from .e2 import prepare_covariates      # local import: e2 is the primary module

    d = df.copy()
    if "tps_w" not in d and "tokens_per_syllable" in d:
        align_col = "align_among_split" if "align_among_split" in d else "boundary_alignment"
        if align_col not in d:
            raise KeyError("align_among_split (or a per-syllable boundary_alignment) is required for align_w (design 8.4)")
        d = prepare_covariates(d, align_col=align_col)
    for col in ("input_lexical", "output_lexical", "correct"):
        if col in d:
            d[col] = d[col].astype(int)
    if "log_freq" not in d and "input_freq_min" in d:
        d["log_freq"] = np.log1p(d["input_freq_min"])
    return d


def _fixed_for(d: pd.DataFrame, fixed: str) -> str:
    from .e2 import _effective_formula

    return _effective_formula(d, f"correct ~ {fixed}").split("~", 1)[1].strip()


def fit_bayes_mixed(df: pd.DataFrame, fixed: str = DEFAULT_FIXED, random_effects: Sequence[str] = ("base_pair_id", "model"),
                    vb_iter: int = 500) -> tuple[list[FixedEffect], object]:
    """Variational-Bayes logistic mixed model with one random intercept per grouping factor.
    Point estimates and variance-component screening only (design 8.4)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = _prepare(df)
    fixed = _fixed_for(d, fixed)
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
    """GEE logistic regression clustered by base pair (robust sandwich SEs): the robustness
    fit of the design 8.4 specification. Fit per model or with C(model) in the fixed part."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = _prepare(df)
    fixed = _fixed_for(d, fixed)
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
