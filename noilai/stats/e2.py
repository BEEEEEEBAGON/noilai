"""E2 primary analysis (design 8.4, amended by items 23 and 28): what predicts failure.

Covariate coding (item x tokenizer quantities, `prepare_covariates`):
  split    = 1[tokens per syllable > 1]
  tps_w    = tokens per syllable minus the model's mean (within-model centering; Mundlak)
  tps_b    = the model's mean (between)
  align_w  = share of aligned boundaries among SPLIT syllables (0 when split = 0) — the
             alignment coding that is not collinear with the count; the source column is
             `align_among_split` written by scripts/audit_items.py (never the legacy
             `boundary_alignment_mean`, which scores single-token syllables 1.0).
  i_y_variant / whole_syllable_share join the covariates when the columns exist (item 53);
  `whole_syllable_share` is a per-tokenizer quantity and is absorbed by the model fixed
  effects of the primary fit, so it enters the random-slope generalization check and the
  descriptive H2 table, not PRIMARY_FORMULA.
Outcome: one row per (item, paraphrase) with the same base-pair clustering, or paraphrase
means; `correct` is 0/1 here (the logit needs a binary response).

Primary fit (`fit_fixed_effects_clustered`): logit with model fixed effects and model x
fragmentation interactions,
    correct ~ C(model) + split + tps_w + align_w + input_lexical + output_lexical + log_freq
              + C(variant) + C(task) + C(model):tps_w + C(model):align_w
Standard errors and CIs of the reported quantities (the mean of the per-model slopes)
come from the BASE-PAIR CLUSTER BOOTSTRAP (`bootstrap_mean_slopes`, refits): that is the
primary covariance. The two-way (base pair x model) cluster-robust sandwich is computed and
labelled `*_twoway` as a REPORTED SENSITIVITY ONLY (item 28: 21 model clusters are too few
for sandwich asymptotics and redundant once the model fixed effects and interactions are in
the model).

H1 is decided by `nested_h1_test`: the likelihood-ratio test / ΔAIC of
    M0 = C(model) + tps_w + controls + C(model):tps_w
    M1 = M0 + split + align_w + C(model):align_w
(design section 1; falsified when p > 0.05 or ΔAIC < 2 or a slope sign is reversed).
`standardized_slope_difference` is descriptive and takes no part in the H1 decision (item 34).
`identifiability_floor` computes, from the item audit, which tokenizers clear the floor of
300 misaligned split syllables in the main sample and whether >= 4 families do (section 1).
The generalization check (random slopes; bambi/PyMC or lme4) is fitted elsewhere.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import constants

H1_MISALIGNED_SPLIT_FLOOR = 300     # design section 1 (item 23): per tokenizer, in the main sample
H1_MIN_FAMILIES = 4


def prepare_covariates(df: pd.DataFrame, tps_col: str = "tokens_per_syllable", align_col: str = "align_among_split",
                       model_col: str = "model") -> pd.DataFrame:
    """Add split, tps_w, tps_b, align_w columns. `align_col` must be the alignment share among
    split syllables (from scripts/audit_items.py); when the item has no split syllable the
    column may be NaN and becomes 0. The legacy `boundary_alignment_mean` is refused because
    it scores single-token syllables 1.0 and is collinear with the count (design 8.4)."""
    if align_col == "boundary_alignment_mean":
        raise ValueError("align_col='boundary_alignment_mean' averages over single-token syllables (scored 1.0) and is "
                         "collinear with the token count; use 'align_among_split' from scripts/audit_items.py (design 8.4)")
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


CONTROLS = "input_lexical + output_lexical + log_freq + C(variant) + C(task) + C(i_y_variant)"
PRIMARY_FORMULA = f"correct ~ C(model) + split + tps_w + align_w + {CONTROLS} + C(model):tps_w + C(model):align_w"
H1_M0_FORMULA = f"correct ~ C(model) + tps_w + {CONTROLS} + C(model):tps_w"
H1_M1_FORMULA = PRIMARY_FORMULA


@dataclass
class FEResult:
    params: pd.Series
    cov_twoway: pd.DataFrame | None       # SENSITIVITY: two-way (base pair x model) sandwich, PSD-projected
    cov_bp: pd.DataFrame | None           # one-way base-pair sandwich (diagnostic)
    slopes_tps: pd.Series                 # per-model slope of tps_w
    slopes_align: pd.Series               # per-model slope of align_w
    mean_slope_tps: float
    mean_slope_align: float
    se_mean_slope_tps_twoway: float       # sensitivity only; the reported SE/CI is bootstrap_mean_slopes'
    se_mean_slope_align_twoway: float
    n_obs: int
    n_bp: int
    n_models: int
    result: object
    formula: str = ""
    covariance_label: str = "two-way sandwich = reported sensitivity; primary CI = base-pair cluster bootstrap (design 8.4)"


def _effective_formula(d: pd.DataFrame, formula: str) -> str:
    """Drop terms whose columns are absent from the data, and C(col) terms (and interactions
    with them) whose column has a single level; a constant categorical is collinear with the
    intercept and makes the Hessian singular (e.g. a T1-only run has no task contrast)."""
    lhs, rhs = formula.split("~")
    terms = [t.strip() for t in rhs.split("+")]
    keep = []
    for t in terms:
        cols = re.findall(r"C\((\w+)\)", t)
        plain = [c for c in re.findall(r"\b([A-Za-z_]\w*)\b", t) if c not in cols and c != "C"]
        if any(c not in d for c in cols + plain):
            continue
        if any(d[c].nunique() < 2 for c in cols + plain):
            continue
        keep.append(t)
    return f"{lhs.strip()} ~ " + " + ".join(keep)


def _psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2)
    w = np.clip(w, 0, None)
    return (v * w) @ v.T


def _fit_logit(d: pd.DataFrame, formula: str, maxiter: int):
    import statsmodels.formula.api as smf

    mod = smf.logit(formula, data=d)
    res = mod.fit(disp=0, maxiter=maxiter, method="bfgs")
    return mod, res


def fit_fixed_effects_clustered(df: pd.DataFrame, formula: str = PRIMARY_FORMULA, bp_col: str = "base_pair_id",
                                model_col: str = "model", maxiter: int = 200, sandwich: bool = True) -> FEResult:
    """The primary point estimates (model fixed effects + interactions). With `sandwich`, the
    two-way sandwich covariance is added as the labelled sensitivity; `sandwich=False` is the
    fast path the bootstrap uses (the bootstrap IS the primary covariance)."""
    d = df.copy()
    d[bp_col] = d[bp_col].astype(str)
    d[model_col] = d[model_col].astype(str)
    formula = _effective_formula(d, formula)
    mod, res = _fit_logit(d, formula, maxiter)
    names = list(res.params.index)
    cov_twoway = cov_bp = None
    V = None
    if sandwich:
        covs = {}
        for name, groups in (("bp", d[bp_col]), ("model", d[model_col]), ("both", d[bp_col] + "|" + d[model_col])):
            codes = pd.factorize(groups)[0]
            r = mod.fit(disp=0, maxiter=maxiter, method="bfgs", start_params=res.params.values,
                        cov_type="cluster", cov_kwds={"groups": codes})
            covs[name] = np.asarray(r.cov_params())
        V = _psd(covs["bp"] + covs["model"] - covs["both"])
        cov_twoway = pd.DataFrame(V, index=names, columns=names)
        cov_bp = pd.DataFrame(covs["bp"], index=names, columns=names)
    models = sorted(d[model_col].unique())
    ref = models[0]                       # the reference level absorbed into the main effect

    def per_model(term: str) -> tuple[pd.Series, float, float]:
        weights = []
        vals = {}
        for m in models:
            w = pd.Series(0.0, index=names)
            if term in w.index:
                w[term] = 1.0
            inter = f"C(model)[T.{m}]:{term}"
            if m != ref and inter in w.index:
                w[inter] = 1.0
            vals[m] = float(w @ res.params)
            weights.append(w.values)
        Wm = np.mean(np.vstack(weights), axis=0)
        mean = float(Wm @ res.params.values)
        se = float(np.sqrt(Wm @ V @ Wm)) if V is not None else float("nan")
        return pd.Series(vals), mean, se

    s_tps, m_tps, se_tps = per_model("tps_w")
    s_al, m_al, se_al = per_model("align_w")
    return FEResult(params=res.params, cov_twoway=cov_twoway, cov_bp=cov_bp, slopes_tps=s_tps, slopes_align=s_al,
                    mean_slope_tps=m_tps, mean_slope_align=m_al, se_mean_slope_tps_twoway=se_tps,
                    se_mean_slope_align_twoway=se_al, n_obs=int(res.nobs), n_bp=int(d[bp_col].nunique()),
                    n_models=len(models), result=res, formula=formula)


def standardized_slope_difference(fe: FEResult, df: pd.DataFrame) -> dict:
    """DESCRIPTIVE ONLY; not the H1 decision (design section 1, item 34: `align_w` is
    structurally 0 when split = 0, so "which matters more" against a continuous covariate is
    not a meaningful contrast). |beta(align_w)| * sd(align_w) - |beta(tps_w)| * sd(tps_w) on the
    mean per-model slopes, with a delta-method SE from the two-way (sensitivity) covariance."""
    sd_a = float(df["align_w"].std())
    sd_t = float(df["tps_w"].std())
    diff = abs(fe.mean_slope_align) * sd_a - abs(fe.mean_slope_tps) * sd_t
    var = (sd_a * fe.se_mean_slope_align_twoway) ** 2 + (sd_t * fe.se_mean_slope_tps_twoway) ** 2   # ignores the covariance term
    return {"diff": diff, "se_approx": float(np.sqrt(var)), "beta_align_std": fe.mean_slope_align * sd_a,
            "beta_tps_std": fe.mean_slope_tps * sd_t, "label": "descriptive; not the H1 decision (item 34)"}


def bootstrap_mean_slopes(df: pd.DataFrame, n_boot: int = 200, seed: int = 0, bp_col: str = "base_pair_id",
                          formula: str = PRIMARY_FORMULA, model_col: str = "model", level: float = 0.95) -> dict:
    """PRIMARY covariance (design 8.4): base-pair cluster bootstrap of the mean per-model
    slopes, refitting the primary model on every replicate. Returns the percentile CIs, the
    bootstrap SEs and the replicate count."""
    rng = np.random.default_rng(seed)
    bps = df[bp_col].unique()
    groups = {b: g for b, g in df.groupby(bp_col, sort=False)}
    tps, al = [], []
    for _ in range(n_boot):
        pick = rng.choice(bps, size=len(bps), replace=True)
        parts = []
        for j, b in enumerate(pick):
            g = groups[b].copy()
            g[bp_col] = f"{b}#{j}"       # a pair drawn twice is two clusters in the replicate
            parts.append(g)
        boot = pd.concat(parts, ignore_index=True)
        try:
            fe = fit_fixed_effects_clustered(boot, formula=formula, bp_col=bp_col, model_col=model_col, sandwich=False)
        except Exception:  # noqa: BLE001, S112 - a replicate whose logit does not converge is dropped and counted in n_ok
            continue
        tps.append(fe.mean_slope_tps)
        al.append(fe.mean_slope_align)
    a = (1 - level) / 2

    def q(x):
        return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))] if x else [np.nan, np.nan]

    def se(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")

    return {"n_ok": len(tps), "n_boot": n_boot, "level": level, "ci_mean_slope_tps": q(tps), "ci_mean_slope_align": q(al),
            "se_mean_slope_tps": se(tps), "se_mean_slope_align": se(al),
            "label": "primary: base-pair cluster bootstrap of the mean per-model slopes (design 8.4)"}


def nested_h1_test(df: pd.DataFrame, bp_col: str = "base_pair_id", model_col: str = "model", maxiter: int = 200,
                   m0: str = H1_M0_FORMULA, m1: str = H1_M1_FORMULA) -> dict:
    """H1's decision (design section 1): nested likelihood-ratio test and ΔAIC of
    M0 = model FE + tps_w (+ controls, + C(model):tps_w) against M1 = M0 + split + align_w
    (+ C(model):align_w), fitted on the same rows. Returns lr, df, p, delta_aic (AIC(M0) −
    AIC(M1), positive favours M1), the main-effect and mean per-model slopes of tps_w and
    align_w under M1, and `h1_supported` = p < alpha and delta_aic >= 2 and beta(align_w) > 0
    and beta(tps_w) < 0."""
    d = df.copy()
    d[bp_col] = d[bp_col].astype(str)
    d[model_col] = d[model_col].astype(str)
    f0 = _effective_formula(d, m0)
    f1 = _effective_formula(d, m1)
    _, r0 = _fit_logit(d, f0, maxiter)
    _, r1 = _fit_logit(d, f1, maxiter)
    lr = float(2 * (r1.llf - r0.llf))
    k = int(r1.df_model - r0.df_model)
    p = float(sps.chi2.sf(lr, k)) if k > 0 else float("nan")
    delta_aic = float(r0.aic - r1.aic)
    fe1 = fit_fixed_effects_clustered(d, formula=m1, bp_col=bp_col, model_col=model_col, maxiter=maxiter, sandwich=False)
    beta_align = float(r1.params.get("align_w", np.nan))
    beta_tps = float(r1.params.get("tps_w", np.nan))
    supported = bool(np.isfinite(p) and p < constants.ALPHA and delta_aic >= 2.0
                     and fe1.mean_slope_align > 0 and fe1.mean_slope_tps < 0)
    return {"lr": lr, "df": k, "p": p, "delta_aic": delta_aic, "llf_m0": float(r0.llf), "llf_m1": float(r1.llf),
            "aic_m0": float(r0.aic), "aic_m1": float(r1.aic), "beta_align": beta_align, "beta_tps": beta_tps,
            "mean_slope_align": fe1.mean_slope_align, "mean_slope_tps": fe1.mean_slope_tps, "n_obs": int(r1.nobs),
            "formula_m0": f0, "formula_m1": f1, "h1_supported": supported,
            "rule": "H1 falsified when p > 0.05 or ΔAIC < 2 or a slope sign is reversed (design section 1)"}


def misaligned_split_count(row: Mapping) -> int:
    """Number of misaligned split syllables in one scripts/audit_items.py row (flat
    `n_misaligned_split`, or nested under `input`)."""
    if "n_misaligned_split" in row and row["n_misaligned_split"] is not None:
        return int(row["n_misaligned_split"])
    inp = row.get("input")
    if isinstance(inp, Mapping) and inp.get("n_misaligned_split") is not None:
        return int(inp["n_misaligned_split"])
    raise KeyError("n_misaligned_split (regenerate the item audit with scripts/audit_items.py)")


def identifiability_floor(items_audit: pd.DataFrame | Sequence[Mapping], floor: int = H1_MISALIGNED_SPLIT_FLOOR,
                          families: Mapping[str, str] | None = None, encoding: str = "nfc",
                          tokenizer_col: str = "tokenizer", min_families: int = H1_MIN_FAMILIES) -> dict:
    """Per tokenizer, the number of misaligned split syllables in the audited item file (main
    sample, baseline encoding), whether it clears the floor, and — with a tokenizer -> family
    map — how many families clear it (design section 1: a tokenizer below the floor
    contributes descriptively only; fewer than `min_families` families clearing demotes H1)."""
    rows = items_audit.to_dict("records") if isinstance(items_audit, pd.DataFrame) else list(items_audit)
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("encoding", encoding) != encoding:
            continue
        tok = str(r[tokenizer_col])
        counts[tok] = counts.get(tok, 0) + misaligned_split_count(r)
    per = {t: {"n_misaligned_split": n, "clears_floor": n >= floor, "family": (families or {}).get(t)} for t, n in sorted(counts.items())}
    fam_clear = sorted({v["family"] for v in per.values() if v["clears_floor"] and v["family"] is not None})
    out = {"floor": floor, "encoding": encoding, "per_tokenizer": per,
           "tokenizers_clearing": sorted(t for t, v in per.items() if v["clears_floor"])}
    if families is not None:
        out["families_clearing"] = fam_clear
        out["h1_identified"] = len(fam_clear) >= min_families
        out["min_families"] = min_families
    return out
