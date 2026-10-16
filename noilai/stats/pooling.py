"""Hierarchical "pooled" estimates: models nested in tokenizer families (PREREG 8.1, DD 1, DD 8.1).

What the binding documents fix (quoted; docs/PREREGISTRATION.md = PREREG, docs/DESIGN_DECISIONS.md = DD):
  * DD:20 / PREREG:19 -- "Pooled" means a hierarchical random-effects estimate with models nested in
    tokenizer families (REML with the Hartung-Knapp small-k adjustment, or a family-level cluster
    bootstrap), reported with k of N families beside it and phrased "across the panel", never "in
    models generally", never a count of significant models, and never a plain one-stage pool over 21
    non-independent models.
  * PREREG:83 = DD:386 (8.1) -- the families: Gemma SentencePiece, Qwen BPE, Llama 3.1, PhoGPT,
    Vistral/Mistral, o200k, Gemini; "pooled estimates are hierarchical (model in family)".
  * PREREG:25 (H3) -- "the per-family table is primary, and with fewer than five informative families
    the pooled CI is labelled unreliable and H3 is decided by the per-family CIs"; a normalizing
    tokenizer is "a sanity assertion (delta = 0 exactly), not a test" and a corrupting path is
    "excluded and reported".
  * PREREG:27 (H4) -- "report k of N with adjusted CI excluding 0 plus the pooled estimate; for models
    whose CI includes 0, TOST equivalence with delta = 2 points. Never 'in every model'."; PREREG:29
    (H6) and PREREG:93 (8.3) -- "per-model tests, Holm across the panel; report k of N plus pooled".
  * DD:400 (8.3) -- "every paper sentence rests on a pooled (family-level) estimate, never on which
    cells turned significant."

Two estimators, both always computed, one named primary (`method`), the other reported in `alternative`:
  A. `reml_hk`: stage 1 within each family, a REML random-effects mean over the family's models; stage 2
     over the k family estimates, REML between-family tau^2 with the Hartung-Knapp-Sidik-Jonkman variance
     and a t_{k-1} quantile (standard HKSJ as metafor `test="knha"`; the ad-hoc max(1, q) truncation of the
     scale is NOT applied, see `Pooled.notes`). The fixed-scale z interval is reported beside it (`ci_z`).
  B. `family_bootstrap`: the k families resampled with replacement B = constants.BOOTSTRAP_B times with a
     fixed seed; the statistic is the unweighted mean over the drawn families of each family's unweighted
     mean over its models (equal family weight: a seven-entry family does not outvote a one-entry family);
     percentile interval. Per-model variances are not used, so zero-variance rows stay in B.

The family map is REQUIRED: a flat pool over models does not exist here. The map comes from
configs/analysis.yaml (`tokenizer_family_map`); `family_map_from_models_yaml` is the documented fallback
and is NOT the tokenizer family (the `family` field of configs/models.yaml has eleven values over the panel
and one of them straddles two tokenizers; see the pooling handoff, section 9 item 2).

Also here, because they are the panel-level reporting of PREREG:27/:29: `k_of_n_models` (after
`tests.correct_family`) and `tost_paired` (two one-sided t tests on base-pair-level mean differences,
delta = constants.TOST_EQUIVALENCE_MARGIN_POINTS).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy import stats as sps

from .. import constants
from .bootstrap import _cluster_index, _resample_weights, _strata_of_clusters, paired_t_base_pairs
from .tests import PairedTest

METHODS = ("reml_hk", "family_bootstrap")
PHRASE = "across the panel"
VERDICT_REASONS = {
    "normalizes": "normalizing tokenizer: delta = 0 exactly by construction, a sanity assertion, not a test (PREREG:25)",
    "corrupts": "corrupting tokenizer path: excluded and reported (PREREG:25)",
}
_UNRELIABLE_K2 = "fewer than three families: tau2 not estimable, t_1 quantile"


# ----------------------------------------------------------------------------- variances per model
def variance_from_ci(lo: float, hi: float, level: float = 0.95) -> float:
    """Approximate variance from a symmetric-normal reading of a CI: ((hi - lo) / (2 z_{1-alpha/2}))^2.
    Only when no replicates and no paired-t SE exist; records built from it are flagged `ci_width`."""
    z = sps.norm.ppf(1 - (1 - level) / 2)
    return float(((hi - lo) / (2 * z)) ** 2)


def replicate_variance(values: Sequence[float], clusters: Sequence, strata: Sequence | None = None,
                       n_boot: int = constants.BOOTSTRAP_B, seed: int = 0) -> dict:
    """Mean of `values` and the variance of its B stratified base-pair cluster-bootstrap replicates, drawn
    exactly as bootstrap.cluster_bootstrap draws them (same helpers, same seed -> same draws). For a paired
    contrast pass values = b - a (the item-level difference), as bootstrap.paired_difference_ci does."""
    v = np.asarray(values, dtype=float)
    inv, k = _cluster_index(clusters)
    groups = _strata_of_clusters(inv, k, strata)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = _resample_weights(inv, k, rng, groups)
        boots[b] = float(np.average(v, weights=w)) if w.sum() else float("nan")
    est = float(v.mean()) if len(v) else float("nan")
    var = float(np.nanvar(boots, ddof=1)) if n_boot > 1 else float("nan")
    return {"estimate": est, "variance": var, "n_items": len(v), "n_clusters": int(k), "n_boot": int(n_boot)}


@dataclass
class _Record:
    model: str
    family: str
    estimate: float
    variance: float
    variance_source: str
    lo: float | None = None
    hi: float | None = None
    n_items: int | None = None
    n_clusters: int | None = None
    excluded_reason: str | None = None      # excluded from both estimators (undefined cell, census verdict)
    zero_variance: bool = False             # excluded from A only (inverse-variance weight undefined)

    @property
    def in_a(self) -> bool:
        return self.excluded_reason is None and not self.zero_variance

    @property
    def in_b(self) -> bool:
        return self.excluded_reason is None


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _normalize_record(rec: Mapping, families: Mapping[str, str], n_boot: int, seed: int, level: float) -> _Record:
    model = rec.get("model")
    if model is None:
        raise ValueError("every estimate record needs a 'model'")
    if model not in families:
        raise ValueError(f"model {model!r} has no tokenizer family in `families`; a model outside the family map "
                         "cannot be pooled (PREREG 8.1: models nested in tokenizer families)")
    family = str(families[model])
    reason = rec.get("reason")
    verdict = rec.get("verdict")
    if verdict in VERDICT_REASONS:
        reason = VERDICT_REASONS[verdict]
    lo = hi = n_items = n_clusters = None
    if "values" in rec:
        rv = replicate_variance(rec["values"], rec["clusters"], strata=rec.get("strata"), n_boot=n_boot, seed=seed)
        est, var, source = rv["estimate"], rv["variance"], "bootstrap_replicates"
        n_items, n_clusters = rv["n_items"], rv["n_clusters"]
    else:
        if "estimate" not in rec:
            raise ValueError(f"record for {model!r} needs 'estimate' (with 'variance', 'se' or 'lo'/'hi') or 'values'")
        est = float(rec["estimate"]) if rec["estimate"] is not None else float("nan")
        lo, hi = rec.get("lo"), rec.get("hi")
        if rec.get("variance") is not None:
            var, source = float(rec["variance"]), "variance"
        elif rec.get("se") is not None:
            var, source = float(rec["se"]) ** 2, "se"
        elif lo is not None and hi is not None:
            var, source = variance_from_ci(float(lo), float(hi), float(rec.get("level", level))), "ci_width"
        else:
            raise ValueError(f"record for {model!r} needs 'variance', 'se', 'lo'/'hi' or item-level 'values'")
        n_items, n_clusters = rec.get("n_items"), rec.get("n_clusters")
    out = _Record(str(model), family, est, float(var), source, lo, hi, n_items, n_clusters, reason)
    if out.excluded_reason is None and not _finite(est):
        out.excluded_reason = "undefined cell"
    if out.excluded_reason is None and not (_finite(var) and var > 0):
        out.zero_variance = True
    return out


# ----------------------------------------------------------------------------- REML and HKSJ
def _reml_loglik(tau2: float, theta: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / (v + tau2)
    mu = np.sum(w * theta) / np.sum(w)
    return float(-0.5 * np.sum(np.log(v + tau2)) - 0.5 * np.log(np.sum(w)) - 0.5 * np.sum(w * (theta - mu) ** 2))


def reml_tau2(theta: Sequence[float], variance: Sequence[float]) -> float | None:
    """REML between-unit variance tau^2 >= 0 for estimates `theta` with within-unit variances `variance`:
    the maximizer of l_R(tau2) = -1/2 sum log(v + tau2) - 1/2 log sum 1/(v + tau2)
    - 1/2 sum (theta - mu(tau2))^2 / (v + tau2) over [0, 10 max(max v, var theta)] (a grid scan, then
    scipy.optimize.minimize_scalar bounded in the best bracket). None with fewer than two units."""
    th = np.asarray(theta, dtype=float)
    vv = np.asarray(variance, dtype=float)
    k = len(th)
    if k < 2:
        return None
    upper = 10.0 * max(float(np.max(vv)), float(np.var(th, ddof=1)), 1e-12)
    grid = np.concatenate([[0.0], np.geomspace(upper * 1e-8, upper, 80)])
    ll = np.array([_reml_loglik(t, th, vv) for t in grid])
    i = int(np.argmax(ll))
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best, best_ll = float(grid[i]), float(ll[i])
    if hi_b > lo_b:
        res = optimize.minimize_scalar(lambda t: -_reml_loglik(t, th, vv), bounds=(lo_b, hi_b), method="bounded",
                                       options={"xatol": max(1e-12, 1e-6 * upper)})
        if res.success and -res.fun >= best_ll:
            best, best_ll = float(res.x), float(-res.fun)
    if best < 1e-10 * upper:
        best = 0.0
    return best


def _random_effects_mean(theta: np.ndarray, v: np.ndarray, tau2: float) -> tuple[float, float, np.ndarray]:
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * theta) / np.sum(w))
    return mu, float(1.0 / np.sum(w)), w


def _hksj(theta: np.ndarray, w: np.ndarray, mu: float, level: float) -> dict:
    """Hartung-Knapp-Sidik-Jonkman: var_HK = [max(1, q) sum W (theta - mu)^2 / (k - 1)] / sum W, t_{k-1} quantile.
    The scale q is truncated at 1 (the modified HKSJ): without it the interval is narrower than the fixed-scale z
    interval whenever q < 1 and collapses to zero width when the family estimates coincide; `q_raw` keeps the
    untruncated value and `truncated` says whether the floor applied."""
    k = len(theta)
    q_raw = float(np.sum(w * (theta - mu) ** 2) / (k - 1))
    q = max(1.0, q_raw)
    var_hk = q / float(np.sum(w))
    se = float(np.sqrt(var_hk))
    t = float(sps.t.ppf(1 - (1 - level) / 2, k - 1))
    return {"se": se, "lo": mu - t * se, "hi": mu + t * se, "df": k - 1, "q": q, "q_raw": q_raw,
            "truncated": q_raw < 1.0, "t_quantile": t}


def _z_interval(mu: float, var: float, level: float) -> tuple[float, float]:
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    se = float(np.sqrt(var))
    return mu - z * se, mu + z * se


# ----------------------------------------------------------------------------- stage 1: within family
def _family_stage1(family: str, recs: list[_Record], level: float) -> dict:
    """REML random-effects mean of a family's informative models (one model: pass-through of its estimate,
    variance and own CI). z interval when k_f >= 2; HK interval beside it when k_f >= 3; k_f = 2 flags tau2
    as unstable."""
    models = [r.model for r in recs]
    out = {"family": family, "n_models": len(recs), "models": models, "informative": bool(recs),
           "reason": None, "tau2_within": None, "tau2_flag": None, "lo_hk": None, "hi_hk": None,
           "ci_method": None, "variance_sources": sorted({r.variance_source for r in recs})}
    if not recs:
        out.update(estimate=float("nan"), se=float("nan"), lo=float("nan"), hi=float("nan"), variance=float("nan"))
        return out
    th = np.array([r.estimate for r in recs], dtype=float)
    vv = np.array([r.variance for r in recs], dtype=float)
    if len(recs) == 1:
        r = recs[0]
        lo, hi = (float(r.lo), float(r.hi)) if (r.lo is not None and r.hi is not None) else _z_interval(r.estimate, r.variance, level)
        out.update(estimate=r.estimate, variance=r.variance, se=float(np.sqrt(r.variance)), lo=lo, hi=hi,
                   ci_method="single model: own CI" if r.lo is not None else "single model: z on its variance")
        return out
    tau2 = reml_tau2(th, vv)
    mu, var_f, w = _random_effects_mean(th, vv, tau2)
    lo, hi = _z_interval(mu, var_f, level)
    out.update(estimate=mu, variance=var_f, se=float(np.sqrt(var_f)), lo=lo, hi=hi, tau2_within=tau2,
               ci_method="reml within family, z")
    if len(recs) == 2:
        out["tau2_flag"] = "unstable (2 models)"
    if len(recs) >= 3:
        hk = _hksj(th, w, mu, level)
        out["lo_hk"], out["hi_hk"] = hk["lo"], hk["hi"]
    return out


# ----------------------------------------------------------------------------- stage 2: across families
def _stage2_reml_hk(fams: list[dict], level: float) -> dict:
    """Estimator A over the informative families' stage-1 estimates."""
    k = len(fams)
    res = {"method": "reml_hk", "k": k, "n_models": int(sum(f["n_models"] for f in fams)), "pooled": k >= 2,
           "tau2_between": None, "df": None, "ci_z": None, "q": None, "notes": []}
    if k == 0:
        res.update(estimate=float("nan"), lo=float("nan"), hi=float("nan"), se=float("nan"))
        return res
    if k == 1:
        f = fams[0]
        res.update(method="single_family", estimate=f["estimate"], lo=f["lo"], hi=f["hi"], se=f["se"])
        res["notes"].append("single family: no pooled estimate, no claim")
        return res
    th = np.array([f["estimate"] for f in fams], dtype=float)
    vv = np.array([f["variance"] for f in fams], dtype=float)
    tau2 = reml_tau2(th, vv)
    mu, var_fixed, w = _random_effects_mean(th, vv, tau2)
    hk = _hksj(th, w, mu, level)
    res.update(estimate=mu, lo=hk["lo"], hi=hk["hi"], se=hk["se"], df=hk["df"], tau2_between=tau2, q=hk["q"],
               ci_z=list(_z_interval(mu, var_fixed, level)))
    res["notes"].append("HKSJ variance with t_{k-1}; scale truncated at 1 (modified HKSJ; raw q reported as q_raw)")
    if hk["truncated"]:
        res["notes"].append(f"q_raw = {hk['q_raw']:.3f} < 1: scale truncated to 1 (the untruncated HK interval would be "
                            "narrower than the fixed-scale z interval ci_z)")
    if k == 2:
        res["notes"].append(_UNRELIABLE_K2)
    return res


def _family_bootstrap(fams_b: list[dict], n_boot: int, seed: int, level: float) -> dict:
    """Estimator B over the families' unweighted model means (equal family weight), percentile interval."""
    k = len(fams_b)
    res = {"method": "family_bootstrap", "k": k, "n_models": int(sum(f["n_models"] for f in fams_b)), "pooled": k >= 2,
           "df": None, "n_boot": int(n_boot), "seed": int(seed), "notes": []}
    if k == 0:
        res.update(estimate=float("nan"), lo=float("nan"), hi=float("nan"), se=float("nan"))
        return res
    means = np.array([f["mean_unweighted"] for f in fams_b], dtype=float)
    est = float(means.mean())
    if k == 1:
        res.update(estimate=est, lo=est, hi=est, se=float("nan"))
        res["notes"].append("single family: no interval")
        return res
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = means[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    res.update(estimate=est, lo=float(lo), hi=float(hi), se=float(boots.std(ddof=1)))
    if k == 2:
        res["notes"].append("two families: at most 4 distinct replicate values; unreliable")
    return res


def _side(lo: float, hi: float, direction: int | None) -> bool | None:
    if direction is None or not (_finite(lo) and _finite(hi)):
        return None
    return bool(lo > 0) if direction > 0 else bool(hi < 0)


# ----------------------------------------------------------------------------- the result
@dataclass
class Pooled:
    estimate: float
    lo: float
    hi: float
    se: float
    level: float
    df: int | None
    method: str
    pooled: bool
    tau2_between: float | None
    tau2_within: dict
    k_informative_families: int
    n_panel_families: int
    n_models_pooled: int
    n_panel_models: int
    k_of_n: str
    phrase: str
    per_family: list
    excluded: list
    unreliable: bool
    unreliable_reason: str | None
    on_predicted_side: bool | None
    all_informative_families_on_side: bool | None
    direction: int | None
    ci_z: list | None
    alternative: dict
    methods_agree: bool | None
    label: str
    min_reliable_families: int
    n_boot: int
    seed: int
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return _jsonable(self.__dict__.copy())

    def __str__(self) -> str:  # pragma: no cover
        flag = " UNRELIABLE" if self.unreliable else ""
        return (f"{self.estimate:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] {self.phrase} ({self.k_of_n}; "
                f"{self.n_models_pooled} models; {self.method}){flag}")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None            # strict JSON: no NaN / inf tokens
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def pool(estimates: Sequence[Mapping], *, families: Mapping[str, str], method: str = "reml_hk", level: float = 0.95,
         n_boot: int = constants.BOOTSTRAP_B, seed: int = 0, direction: int | None = None,
         min_reliable_families: int = 5, panel_families: int | Sequence[str] | None = None, label: str = "") -> Pooled:
    """The pooled estimate across the panel (PREREG 8.1 / DD 8.1): models nested in tokenizer families.

    `estimates`: one record per model, either {model, estimate, variance} (variance = SE^2; for tau_m(a)
    use bootstrap.paired_t_base_pairs(...).se ** 2), {model, estimate, se}, {model, estimate, lo, hi[, level]}
    (variance from the CI width, flagged `ci_width`) or {model, values, clusters[, strata]} item-level
    vectors (mean and the replicate variance of the stratified base-pair cluster bootstrap; for a paired
    contrast values = b - a). A record may carry `verdict` (the census verdict of the arm the cell depends
    on, noilai.eval.run.CENSUS_ARM_VERDICTS: "normalizes" -> not informative, "corrupts" -> excluded, both
    reported) or an explicit `reason` to exclude it.
    `families`: Mapping model -> tokenizer family label, REQUIRED (ValueError when empty or when a model is
    missing); `panel_families`: N, as a count or the panel's labels (default: the map's distinct labels).
    `direction`: +1 / -1 / None, the pre-registered sign; drives `on_predicted_side` only.
    `min_reliable_families`: 5 is H3's rule (PREREG:25); H4 / H6 have none pre-registered (pass 3, labelled).
    k = 0 informative families raises ValueError; k = 1 returns the single family (`pooled=False`, no claim);
    k = 2 is computed and `unreliable`."""
    if not families:
        raise ValueError("`families` (model -> tokenizer family) is required: a flat pool over models is never "
                         "computed (PREREG 8.1, DD 1)")
    if method not in METHODS:
        raise ValueError(f"unknown pooling method {method!r}; one of {METHODS}")
    if direction not in (None, 1, -1):
        raise ValueError("direction must be +1, -1 or None")
    recs = [_normalize_record(r, families, n_boot, seed, level) for r in estimates]
    seen: set[str] = set()
    for r in recs:
        if r.model in seen:
            raise ValueError(f"model {r.model!r} appears twice in `estimates`")
        seen.add(r.model)
    recs.sort(key=lambda r: (r.family, r.model))                 # order invariance (the family bootstrap draws indices)

    panel_labels = sorted({str(f) for f in families.values()})
    if panel_families is None:
        n_panel = len(panel_labels)
        listed = panel_labels
    elif isinstance(panel_families, int):
        n_panel = int(panel_families)
        listed = panel_labels
    else:
        listed = sorted({str(f) for f in panel_families})
        n_panel = len(listed)
    data_families = {r.family for r in recs}
    if isinstance(panel_families, int) and len(data_families) > n_panel:
        raise ValueError(f"the records span {len(data_families)} families but panel_families = {n_panel}: k of N cannot exceed N")
    if panel_families is not None and not isinstance(panel_families, int) and data_families - set(listed):
        raise ValueError(f"families {sorted(data_families - set(listed))} of the records are not in `panel_families` {listed}: "
                         "the family map and the panel's labels disagree (k of N cannot exceed N)")
    fam_labels = sorted(set(listed) | data_families)

    per_family, fams_a, fams_b, excluded = [], [], [], []
    tau2_within: dict = {}
    tau2_flags: dict = {}
    for fam in fam_labels:
        in_fam = [r for r in recs if r.family == fam]
        a_recs = [r for r in in_fam if r.in_a]
        b_recs = [r for r in in_fam if r.in_b]
        st1 = _family_stage1(fam, a_recs, level)
        st1["mean_unweighted"] = float(np.mean([r.estimate for r in b_recs])) if b_recs else float("nan")
        st1["n_models_in_bootstrap"] = len(b_recs)
        st1["n_models_with_estimate"] = len(in_fam)
        if not in_fam:
            st1["reason"] = "no estimate for this family"
        elif not a_recs:
            st1["reason"] = "every model excluded: " + "; ".join(sorted({r.excluded_reason or "zero variance" for r in in_fam}))
        per_family.append(st1)
        if a_recs:
            fams_a.append(st1)
            tau2_within[fam] = st1["tau2_within"]
            if st1["tau2_flag"]:
                tau2_flags[fam] = st1["tau2_flag"]
        if b_recs:
            fams_b.append(st1)
        for r in in_fam:
            if not r.in_a:
                excluded.append({"model": r.model, "family": fam, "reason": r.excluded_reason or "zero variance",
                                 "kept_in_family_bootstrap": r.in_b})

    res_a = _stage2_reml_hk(fams_a, level)
    res_b = _family_bootstrap(fams_b, n_boot, seed, level)
    primary, other = (res_a, res_b) if method == "reml_hk" else (res_b, res_a)
    k = primary["k"]
    if k == 0:
        raise ValueError("no informative family: nothing to pool (excluded: "
                         + "; ".join(f"{e['model']} ({e['reason']})" for e in excluded) + ")")

    reasons = []
    if k == 1:
        reasons.append("single family: no pooled claim")
    elif k == 2:
        reasons.append(_UNRELIABLE_K2)
    if k < min_reliable_families:
        word = {3: "three", 4: "four", 5: "five"}.get(min_reliable_families, str(min_reliable_families))
        reasons.append(f"fewer than {word} informative families: decided by the per-family CIs")
    unreliable = bool(reasons)

    informative = [f for f in per_family if f["informative"]]
    side_fam = [_side(f["lo"], f["hi"], direction) for f in informative]
    all_on_side = None if direction is None or not informative else all(bool(s) for s in side_fam)
    on_side = _side(primary["lo"], primary["hi"], direction) if primary["pooled"] else None
    on_side_other = _side(other["lo"], other["hi"], direction) if other["pooled"] else None
    if primary["pooled"] and other["pooled"]:
        if direction is None:
            excl_p = _finite(primary["lo"]) and (primary["lo"] > 0 or primary["hi"] < 0)
            excl_o = _finite(other["lo"]) and (other["lo"] > 0 or other["hi"] < 0)
            agree = bool(excl_p == excl_o and (not excl_p or np.sign(primary["estimate"]) == np.sign(other["estimate"])))
        else:
            agree = bool(on_side == on_side_other)
    else:
        agree = None

    alternative = {kk: vv for kk, vv in other.items() if kk != "notes"}
    alternative["on_predicted_side"] = on_side_other
    alternative["notes"] = list(other["notes"])
    notes = list(primary["notes"])
    if tau2_flags:
        notes.append("within-family tau2 flagged: " + ", ".join(f"{f}: {t}" for f, t in sorted(tau2_flags.items())))
    if primary["k"] != other["k"]:
        notes.append(f"the alternative estimator pools {other['k']} families (zero-variance rows stay in the family bootstrap)")

    return Pooled(estimate=float(primary["estimate"]), lo=float(primary["lo"]), hi=float(primary["hi"]), se=float(primary["se"]),
                  level=level, df=primary.get("df"), method=primary["method"], pooled=bool(primary["pooled"]),
                  tau2_between=res_a.get("tau2_between") if method == "reml_hk" else None, tau2_within=tau2_within,
                  k_informative_families=k, n_panel_families=n_panel, n_models_pooled=int(primary["n_models"]),
                  n_panel_models=len(families), k_of_n=f"{k} of {n_panel} tokenizer families", phrase=PHRASE,
                  per_family=per_family, excluded=excluded, unreliable=unreliable,
                  unreliable_reason="; ".join(reasons) if reasons else None, on_predicted_side=on_side,
                  all_informative_families_on_side=all_on_side, direction=direction, ci_z=res_a.get("ci_z") if method == "reml_hk" else None,
                  alternative=alternative, methods_agree=agree, label=label, min_reliable_families=min_reliable_families,
                  n_boot=n_boot, seed=seed, notes=notes)


# ----------------------------------------------------------------------------- family maps
def tokenizer_family_map(analysis_cfg: Mapping) -> dict[str, str]:
    """The model -> tokenizer family map of configs/analysis.yaml (`tokenizer_families: {model: label}`, or
    `families: {by_model: {...}}`). Empty -> ValueError: the author confirms the labels of PREREG 8.1 before any
    pooled number is reported; the map is never guessed from configs/models.yaml silently."""
    m = analysis_cfg.get("tokenizer_families")
    if not m and isinstance(analysis_cfg.get("families"), Mapping):
        m = analysis_cfg["families"].get("by_model")
    if not m:
        raise ValueError("configs/analysis.yaml carries no tokenizer_families map (model -> family label of PREREG 8.1); "
                         "pooling refuses to run without one (family_map_from_models_yaml is the documented fallback, "
                         "and it is NOT the tokenizer family)")
    return {str(k): str(v) for k, v in m.items()}


def family_map_from_models_yaml(models_cfg: Mapping, key: str = "family", panel_only: bool = True) -> dict[str, str]:
    """Fallback map model name -> the `family` field of configs/models.yaml, panel entries only (entries whose
    `group` has `panel: true`; never the bf16_reference / reasoning_substudy serving variants). This field is
    NOT the tokenizer family of PREREG 8.1: it has eleven values over the 21 panel entries and one label covers
    two tokenizers (handoff section 9, item 2). Use it only to exercise code, never for a paper number."""
    groups = models_cfg.get("groups") or {}
    panel_groups = {g for g, spec in groups.items() if isinstance(spec, Mapping) and spec.get("panel")}
    out: dict[str, str] = {}
    for entry in models_cfg.get("models") or []:
        if panel_only and entry.get("group") not in panel_groups:
            continue
        if entry.get(key) is None:
            raise ValueError(f"model entry {entry.get('name')!r} has no {key!r}")
        out[str(entry["name"])] = str(entry[key])
    if not out:
        raise ValueError("no panel entries found in the models configuration")
    return out


# ----------------------------------------------------------------------------- panel-level helpers (H4 / H6)
def k_of_n_models(tests: Sequence[PairedTest], direction: int | None) -> dict:
    """k of N models after tests.correct_family (PREREG:27, :29, :93): k = models whose Holm-adjusted p is
    below alpha AND whose effect lies on the predicted side; N = models tested. Returns the names, never a
    sentence like "in every model"."""
    if direction not in (None, 1, -1):
        raise ValueError("direction must be +1, -1 or None")
    if any(t.significant is None for t in tests):
        raise ValueError("k_of_n_models needs Holm-corrected tests: call tests.correct_family first")
    on_side, wrong_side = [], []
    for t in tests:
        if not t.significant:
            continue
        side_ok = direction is None or (t.diff > 0 if direction > 0 else t.diff < 0)
        (on_side if side_ok else wrong_side).append(t.name)
    n = len(tests)
    return {"k": len(on_side), "n": n, "k_of_n": f"{len(on_side)} of {n} models", "models": on_side,
            "significant_wrong_side": wrong_side, "direction": direction, "alpha": constants.ALPHA,
            "p_source": sorted({t.p_primary_source for t in tests if t.p_primary_source}), "phrase": PHRASE}


def tost_applies(test: PairedTest) -> bool:
    """PREREG:27: TOST only for models whose (unadjusted) CI includes 0."""
    return test.ci_lo is not None and test.ci_hi is not None and test.ci_lo <= 0.0 <= test.ci_hi


def tost_paired(correct_a: Sequence[float], correct_b: Sequence[float], clusters: Sequence,
                margin: float = constants.TOST_EQUIVALENCE_MARGIN_POINTS / 100.0, alpha: float = constants.ALPHA) -> dict:
    """Two one-sided t tests of equivalence within +-margin (delta = 2 points, PREREG:27) on the base-pair-level
    mean differences d = b - a (bootstrap.paired_t_base_pairs gives mean, se and k): p_tost = max(P(T >= (d + delta)
    / se), P(T <= (d - delta) / se)) with k - 1 df; equivalent at alpha iff the (1 - 2 alpha) CI lies within
    +-delta. `primary` is False below constants.PAIRED_T_MIN_BASE_PAIRS base pairs, as the paired t itself."""
    if margin <= 0:
        raise ValueError("the equivalence margin must be positive")
    pt = paired_t_base_pairs(correct_a, correct_b, clusters)
    k, d, se = pt.n_clusters, pt.mean_diff, pt.se
    out = {"mean_diff": d, "se": se, "n_clusters": k, "df": k - 1, "margin": float(margin), "alpha": float(alpha),
           "p_lower": float("nan"), "p_upper": float("nan"), "p_tost": float("nan"), "equivalent": None,
           "ci90": [float("nan"), float("nan")], "ci_level": 1 - 2 * alpha, "primary": pt.primary}
    if k < 2 or not (_finite(se) and se > 0):
        return out
    df = k - 1
    p_lower = float(sps.t.sf((d + margin) / se, df))      # H0: d <= -delta
    p_upper = float(sps.t.cdf((d - margin) / se, df))     # H0: d >= +delta
    t = float(sps.t.ppf(1 - alpha, df))
    lo, hi = d - t * se, d + t * se
    out.update(p_lower=p_lower, p_upper=p_upper, p_tost=max(p_lower, p_upper), ci90=[lo, hi],
               equivalent=bool(-margin < lo and hi < margin))
    return out
