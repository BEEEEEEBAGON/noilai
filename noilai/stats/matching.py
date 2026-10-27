"""Matching, matched contrasts and the item-audit join: the helpers PREREGISTRATION 8.4, 8.7 and 8.8 name.

* H6 (PREREG 8.8 / DD 8.7): every exact attested item is matched to `k_min`..`k_max` generated items from base pairs
  outside the attested set, EXACTLY on variant, task, output lexicality, spelling-trigger presence and tone-pair
  class, NEAREST (Mahalanobis on the pooled covariance) on the log frequency of each input / output syllable and
  the syllable lengths, with a 0.2-SD caliper (on the root-mean-square standardized Mahalanobis distance, i.e. the
  distance in per-covariate SD units); standardized mean differences after matching are reported (< 0.1 is the
  pre-registered balance); the estimate is the mean over attested items of
  (acc_attested - mean acc of its controls) with the base-pair cluster bootstrap. `nearest_neighbour_match` and
  `matched_contrast` implement the matching and the contrast; `h6_covariates` the covariates of an item.
* H3b (PREREG 8.7 / DD 8.6): the tone-isolation difference-in-differences. Items whose NFD tokenization isolates
  the tone mark (ISO, the item audit's `tone_isolated` under NFD) are matched (1:1 nearest neighbour, with
  replacement, the same caliper) to NON-ISO items nearest on syllable frequency, NFC token count and NFD token count
  within base-pair strata (lexical / pseudo); the contrast is the
  per-item NFD effect y(NFD) - y(NFC) of the ISO item minus the mean of its controls', Δ > 0 on V3 and ≈ 0 on V1
  (`h3b_did`). Fewer than 500 items in a cell is reported as `underpowered` (PREREG 8.7).
* H1 (PREREG 8.4): `read_items_audit` reads `data/audit/items_<tokenizer>__<key>.jsonl` (scripts/audit_items.py) into
  the per-item covariates and `e2_frame` joins them to the score rows in the shape `noilai.stats.e2.prepare_covariates`
  consumes (`tokens_per_syllable`, `align_among_split`, `log_freq`, the lexicality flags).

Nothing here reads a run directory or decides a hypothesis: noilai.stats.analyze calls these on the frame it
built and applies the Holm families and the decision rules of the pre-registration.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants
from . import bootstrap
from .tests import mcnemar_exact

SMD_BALANCE = 0.1                   # PREREG 8.8: standardized mean differences after matching below 0.1
H6_K_MIN, H6_K_MAX = 3, 5           # PREREG 8.8: 3-5 generated controls per attested item
H3B_K_MIN, H3B_K_MAX = 1, 1         # PREREG 8.7: nearest neighbour
CALIPER_SD = 0.2                    # PREREG 8.8: the 0.2-SD caliper, on the RMS standardized Mahalanobis distance
H3B_MIN_ITEMS = 500                 # PREREG 8.7: ">= 500 items per cell" for the matched tone-isolation contrast
H6_EXACT = ("variant", "task_class", "output_lexical", "has_spelling_trigger", "tone_pair")
H6_NEAREST = ("log_freq_in_1", "log_freq_in_2", "log_freq_out_1", "log_freq_out_2", "len_in_1", "len_in_2", "len_out_1", "len_out_2")
H3B_EXACT = ("source",)
H3B_NEAREST = ("log_freq_in_1", "log_freq_in_2", "tokens_nfc", "tokens_nfd")


# ============================================================================ balance and matching
def standardized_mean_difference(x_t: Sequence[float], x_c: Sequence[float], w_c: Sequence[float] | None = None) -> float:
    """(mean_t - mean_c) / sqrt((var_t + var_c) / 2) with the control side optionally weighted (matching weights);
    0.0 when both sides are constant and equal, inf when they are constant and differ, NaN when a side is empty."""
    a = np.asarray(x_t, dtype=float)
    b = np.asarray(x_c, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if w_c is None:
        w = np.ones(len(b))
    else:
        w = np.asarray(w_c, dtype=float)
    w = w / w.sum()
    mb = float(np.sum(w * b))
    vb = float(np.sum(w * (b - mb) ** 2)) * (len(b) / max(len(b) - 1, 1))
    va = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    sd = math.sqrt((va + vb) / 2)
    diff = float(a.mean() - mb)
    if sd == 0:
        return 0.0 if diff == 0 else float("inf")
    return diff / sd


@dataclass
class MatchResult:
    matches: pd.DataFrame                       # treated_id, control_id, distance, rank, stratum
    n_treated: int
    n_matched: int
    unmatched: dict[str, int]                   # reason -> count
    k_min: int
    k_max: int
    caliper_sd: float | None
    exact: list[str]
    nearest: list[str]
    smd_before: dict[str, float]                # treated vs the whole pool, per nearest covariate
    smd_after: dict[str, float]                 # treated vs matched controls (weights 1/k), per nearest covariate
    n_pool: int = 0
    n_controls_distinct: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def balanced(self) -> bool | None:
        vals = [v for v in self.smd_after.values() if not math.isnan(v)]
        if not vals:
            return None
        return all(abs(v) < SMD_BALANCE for v in vals)

    @property
    def max_abs_smd_after(self) -> float | None:
        vals = [abs(v) for v in self.smd_after.values() if not math.isnan(v)]
        return max(vals) if vals else None

    def controls_of(self) -> dict:
        out: dict = {}
        for tid, g in self.matches.groupby("treated_id", sort=False):
            out[tid] = list(g.sort_values("rank")["control_id"])
        return out

    def as_dict(self) -> dict:
        return {"n_treated": self.n_treated, "n_matched": self.n_matched, "n_unmatched": sum(self.unmatched.values()),
                "unmatched": dict(self.unmatched), "n_pool": self.n_pool, "n_controls_distinct": self.n_controls_distinct,
                "k_min": self.k_min, "k_max": self.k_max, "caliper_sd": self.caliper_sd, "exact": list(self.exact),
                "nearest": list(self.nearest), "smd_before": {k: _f(v) for k, v in self.smd_before.items()},
                "smd_after": {k: _f(v) for k, v in self.smd_after.items()}, "balanced": self.balanced,
                "max_abs_smd_after": _f(self.max_abs_smd_after), "smd_balance_threshold": SMD_BALANCE,
                "mean_k": _f(self.matches.groupby("treated_id").size().mean()) if len(self.matches) else None, "notes": list(self.notes)}


def _f(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


def _mahalanobis_matrix(x: np.ndarray) -> np.ndarray:
    """The inverse of the pooled covariance (pseudo-inverse; a constant covariate contributes nothing)."""
    if x.shape[0] < 2:
        return np.eye(x.shape[1])
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    return np.linalg.pinv(cov)


def nearest_neighbour_match(treated: pd.DataFrame, pool: pd.DataFrame, *, exact: Sequence[str], nearest: Sequence[str],
                            id_col: str = "item_id", k_min: int = H6_K_MIN, k_max: int = H6_K_MAX,
                            caliper_sd: float | None = CALIPER_SD) -> MatchResult:
    """Match every treated row to its `k_max` nearest controls (at least `k_min`) within the exact-matching stratum,
    with replacement across treated rows, by Mahalanobis distance on the `nearest` covariates (pooled covariance over
    treated and pool, pseudo-inverse). A control is eligible for a treated row when the distance in per-covariate SD
    units, d_M / sqrt(p) (the root-mean-square standardized difference; equal to |z| for one covariate), is at most
    `caliper_sd` (None: no caliper). Treated rows with no control in their stratum or fewer than `k_min` eligible
    controls are left unmatched and counted by reason. Deterministic: ties are broken by the control id."""
    exact, nearest = list(exact), list(nearest)
    if k_min < 1 or k_max < k_min:
        raise ValueError("need 1 <= k_min <= k_max")
    for c in (id_col, *exact, *nearest):
        if c not in treated or c not in pool:
            raise KeyError(f"column {c!r} missing from treated or pool")
    t = treated.drop_duplicates(id_col).copy()
    p = pool.drop_duplicates(id_col).copy()
    p = p[~p[id_col].isin(set(t[id_col]))]
    both = pd.concat([t[nearest], p[nearest]], ignore_index=True).astype(float)
    if both.isna().any().any():
        raise ValueError("nearest covariates carry NaN; fill or drop them before matching")
    vinv = _mahalanobis_matrix(both.values)
    rows: list[dict] = []
    unmatched: dict[str, int] = {}

    def _stratum(r) -> tuple:
        return tuple(_key(r[c]) for c in exact)

    pool_groups: dict[tuple, pd.DataFrame] = {}
    for key_, g in p.groupby([*exact], sort=False, dropna=False) if exact else [((), p)]:
        key_ = key_ if isinstance(key_, tuple) else (key_,)
        pool_groups[tuple(_key(k) for k in key_)] = g
    for _, tr in t.iterrows():
        g = pool_groups.get(_stratum(tr))
        if g is None or len(g) == 0:
            unmatched["no_control_in_exact_stratum"] = unmatched.get("no_control_in_exact_stratum", 0) + 1
            continue
        xt = tr[nearest].to_numpy(dtype=float)
        xc = g[nearest].to_numpy(dtype=float)
        diff = xc - xt
        d2 = np.einsum("ij,jk,ik->i", diff, vinv, diff)
        dist = np.sqrt(np.clip(d2, 0, None))
        if caliper_sd is not None:
            ok = dist / math.sqrt(len(nearest)) <= caliper_sd + 1e-12
        else:
            ok = np.ones(len(g), dtype=bool)
        if ok.sum() < k_min:
            unmatched["fewer_than_k_min_within_caliper"] = unmatched.get("fewer_than_k_min_within_caliper", 0) + 1
            continue
        cand = pd.DataFrame({"control_id": g[id_col].values, "distance": dist})[ok]
        cand = cand.sort_values(["distance", "control_id"], kind="mergesort").head(k_max)
        for rank, (cid, dd) in enumerate(zip(cand["control_id"], cand["distance"]), start=1):
            rows.append({"treated_id": tr[id_col], "control_id": cid, "distance": float(dd), "rank": rank, "stratum": "|".join(map(str, _stratum(tr)))})
    matches = pd.DataFrame(rows, columns=["treated_id", "control_id", "distance", "rank", "stratum"])
    matched_ids = set(matches["treated_id"]) if len(matches) else set()
    tm = t[t[id_col].isin(matched_ids)]
    smd_before = {c: standardized_mean_difference(t[c], p[c]) for c in nearest}
    smd_after: dict[str, float] = {}
    if len(matches):
        k_of = matches.groupby("treated_id").size()
        w = matches["treated_id"].map(lambda i: 1.0 / k_of[i]).values
        ctrl = p.set_index(id_col).loc[matches["control_id"].values]
        smd_after = {c: standardized_mean_difference(tm[c], ctrl[c].values, w) for c in nearest}
    else:
        smd_after = dict.fromkeys(nearest, float("nan"))
    return MatchResult(matches=matches, n_treated=len(t), n_matched=len(matched_ids), unmatched=unmatched, k_min=k_min, k_max=k_max,
                       caliper_sd=caliper_sd, exact=exact, nearest=nearest, smd_before=smd_before, smd_after=smd_after, n_pool=len(p),
                       n_controls_distinct=int(matches["control_id"].nunique()) if len(matches) else 0)


def _key(v):
    if isinstance(v, (list, tuple)):
        return tuple(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.bool_, np.integer)):
        return v.item()
    return v


# ============================================================================ the matched contrast
@dataclass
class MatchedContrast:
    estimate: float
    lo: float
    hi: float
    ci_method: str
    small_cell: bool
    p_bootstrap: float | None
    p_sign_mcnemar: float            # McNemar mid-p on the signs of the per-item contributions (ignores clustering)
    n_treated: int
    n_base_pairs: int
    n_controls_distinct: int
    mean_k: float
    level: float
    n_boot: int
    per_treated: pd.DataFrame = field(repr=False)       # treated_id, y_treated, y_controls_mean, k, contribution, cluster

    def as_dict(self) -> dict:
        return {"estimate": _f(self.estimate), "lo": _f(self.lo), "hi": _f(self.hi), "ci_method": self.ci_method, "small_cell": self.small_cell,
                "p_bootstrap": _f(self.p_bootstrap), "p_sign_mcnemar": _f(self.p_sign_mcnemar),
                "p_sign_label": "McNemar mid-p on the signs of the per-item contributions; ignores clustering and the matching weights",
                "n_treated": self.n_treated, "n_base_pairs": self.n_base_pairs, "n_controls_distinct": self.n_controls_distinct,
                "mean_k": _f(self.mean_k), "level": self.level, "n_boot": self.n_boot,
                "estimand": "mean over treated items of (y_treated - mean y of its matched controls); CI by the base-pair cluster bootstrap over the treated items"}


def matched_contrast(y_treated: Mapping, y_control: Mapping, match: MatchResult, clusters: Mapping, *, strata: Mapping | None = None,
                     n_boot: int = constants.BOOTSTRAP_B, seed: int = 0, level: float = 0.95) -> MatchedContrast:
    """mean_i [y_treated_i - mean_j y_control_j] over the matched treated items (PREREG 8.8), the base-pair cluster
    bootstrap CI and two-sided p (bootstrap.paired_difference_ci / paired_bootstrap_pvalue over the treated items,
    clusters = their base pairs, stratified when `strata` is given) and the McNemar mid-p on the contribution signs.
    `y_*` map an item id to its outcome (0/1 accuracy, or a per-item effect such as y(NFD) - y(NFC) for H3b)."""
    controls = match.controls_of()
    recs = []
    for tid, cids in controls.items():
        if tid not in y_treated:
            continue
        ys = [y_control[c] for c in cids if c in y_control]
        if not ys:
            continue
        recs.append({"treated_id": tid, "y_treated": float(y_treated[tid]), "y_controls_mean": float(np.mean(ys)), "k": len(ys),
                     "cluster": clusters.get(tid, tid), "stratum": (strata or {}).get(tid)})
    per = pd.DataFrame(recs, columns=["treated_id", "y_treated", "y_controls_mean", "k", "cluster", "stratum"])
    if len(per) == 0:
        raise ValueError("no matched treated item carries an outcome")
    per["contribution"] = per["y_treated"] - per["y_controls_mean"]
    a, b, cl = per["y_controls_mean"].values, per["y_treated"].values, per["cluster"].values
    st = per["stratum"].values if strata is not None and per["stratum"].notna().all() else None
    try:
        ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed, level=level, strata=st)
        p_bs = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed, strata=st)
    except ValueError:
        ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed, level=level)
        p_bs = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed)
    n_pos = int((per["contribution"] > 0).sum())
    n_neg = int((per["contribution"] < 0).sum())
    return MatchedContrast(estimate=float(ci.estimate), lo=float(ci.lo), hi=float(ci.hi), ci_method=ci.method, small_cell=bool(ci.small_cell),
                           p_bootstrap=float(p_bs), p_sign_mcnemar=float(mcnemar_exact(n_neg, n_pos)), n_treated=len(per),
                           n_base_pairs=int(ci.n_clusters), n_controls_distinct=int(sum(len(v) for v in controls.values()) and len({c for v in controls.values() for c in v})),
                           mean_k=float(per["k"].mean()), level=level, n_boot=n_boot, per_treated=per)


# ============================================================================ item covariates (H6)
def _log1p(x) -> float:
    try:
        return float(np.log1p(max(float(x), 0.0)))
    except (TypeError, ValueError):
        return float("nan")


def _letters(word: str) -> int:
    import unicodedata
    return len(unicodedata.normalize("NFC", word))


def h6_covariates(input_text: str, output_text: str, *, variant: str, output_lexical: bool, spelling_triggers: Sequence | None,
                  tone_pair: str | None, input_freq: Sequence | None, output_freq: Sequence | None, task_class: str = "T1") -> dict:
    """The PREREG 8.8 matching covariates of one two-syllable item: exact on variant, task class, output lexicality,
    spelling-trigger presence and tone-pair class; nearest on the log frequency of each input / output syllable and
    the letter count of each syllable. `tone_pair` and `spelling_triggers` are the generator's strata (recomputed by
    the caller for attested rows from the parsed syllables)."""
    ins = str(input_text).split()
    outs = str(output_text).split()
    if len(ins) != 2 or len(outs) != 2:
        raise ValueError("h6_covariates needs two-syllable input and output phrases")
    fi = list(input_freq or [0, 0])
    fo = list(output_freq or [0, 0])
    return {"variant": variant, "task_class": task_class, "output_lexical": bool(output_lexical),
            "has_spelling_trigger": bool(spelling_triggers), "tone_pair": tone_pair,
            "log_freq_in_1": _log1p(fi[0]), "log_freq_in_2": _log1p(fi[1]), "log_freq_out_1": _log1p(fo[0]), "log_freq_out_2": _log1p(fo[1]),
            "len_in_1": _letters(ins[0]), "len_in_2": _letters(ins[1]), "len_out_1": _letters(outs[0]), "len_out_2": _letters(outs[1])}


def tone_pair_of(syllables: Sequence[Mapping]) -> str:
    """The generator's `tone_pair` stratum from two parsed syllables (`tone` 0..5)."""
    from ..vi import unicode as U
    return f"{U.TONE_NAMES[int(syllables[0]['tone'])]}-{U.TONE_NAMES[int(syllables[1]['tone'])]}"


# ============================================================================ the item audit (H1, H3b)
AUDIT_FIELDS = ("tokens_per_syllable", "align_among_split", "n_split", "n_misaligned_split", "tone_isolated", "byte_fallback",
                "any_single_token", "all_single_token", "tokens_total")


def flatten_audit_row(row: Mapping, field_: str = "input") -> dict:
    """One scripts/audit_items.py row -> the flat analysis row: `tokens_per_syllable` = the phrase's mean tokens per
    syllable, `align_among_split` (None when no syllable is split), the identifiability count, the NFD flags and
    `delta_tokens_vs_nfc` (noilai.stats.e2.prepare_covariates consumes the first two)."""
    st = row[field_]
    return {"item_id": row["item_id"], "tokenizer": row.get("tokenizer"), "encoding": row.get("encoding"),
            "tokens_per_syllable": st.get("tokens_mean"), "align_among_split": st.get("align_among_split"),
            "n_split": st.get("n_split"), "n_misaligned_split": st.get("n_misaligned_split"),
            "tone_isolated": st.get("tone_isolated"), "byte_fallback": st.get("byte_fallback"),
            "any_single_token": st.get("any_single_token"), "all_single_token": st.get("all_single_token"),
            "tokens_total": st.get("tokens_total"), "delta_tokens_vs_nfc": row.get("delta_tokens_vs_nfc")}


def read_items_audit(path: Path | str, field_: str = "input") -> pd.DataFrame:
    """`data/audit/items_<tokenizer>__<key>.jsonl` as a frame of flat rows (one per item x encoding)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(flatten_audit_row(json.loads(ln), field_))
    return pd.DataFrame(rows)


def items_audit_candidates(audit_dir: Path | str, *, hf_id: str | None, family: str | None, key: str | None) -> list[Path]:
    """Where scripts/kaggle_cpu_jobs.py writes a model's item audit: items_<hf_id with '/' -> '__'>__<key>.jsonl, the
    local SentencePiece names (<family>, e.g. gemma3) and the key-less forms; in that order, existing or not."""
    audit_dir = Path(audit_dir)
    names = []
    tok = hf_id.replace("/", "__") if hf_id else None
    for base in (tok, family):
        if base:
            if key:
                names.append(f"items_{base}__{key}.jsonl")
            names.append(f"items_{base}.jsonl")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(audit_dir / n)
    return out


def find_items_audit(audit_dir: Path | str, *, hf_id: str | None, family: str | None, key: str | None) -> Path | None:
    return next((p for p in items_audit_candidates(audit_dir, hf_id=hf_id, family=family, key=key) if p.exists()), None)


def e2_frame(scores: pd.DataFrame, audit: pd.DataFrame, *, encoding: str = "nfc", model_col: str = "model") -> pd.DataFrame:
    """Join score rows (item_id, model, correct, base_pair_id, variant, task, input_lexical, output_lexical, input_freq)
    to the item audit of their model's tokenizer (one encoding) and add `log_freq` = log1p(min input syllable frequency)
    when the scores carry `input_freq`; the result is what noilai.stats.e2.prepare_covariates takes. Rows without an
    audit row are dropped (the count is the caller's to report: len(scores) - len(result))."""
    a = audit[audit["encoding"] == encoding] if "encoding" in audit else audit
    a = a.drop_duplicates("item_id")[["item_id", "tokens_per_syllable", "align_among_split", "n_split", "n_misaligned_split"]]
    d = scores.merge(a, on="item_id", how="inner")
    if "log_freq" not in d and "input_freq" in d:
        d["log_freq"] = d["input_freq"].map(lambda v: _log1p(min(v)) if isinstance(v, (list, tuple)) and len(v) else float("nan"))
        d["log_freq"] = d["log_freq"].fillna(0.0)
    for c in ("input_lexical", "output_lexical", "correct"):
        if c in d:
            d[c] = d[c].fillna(False).astype(bool).astype(int)
    d["base_pair_id"] = d["base_pair_id"].astype(str)
    d[model_col] = d[model_col].astype(str)
    return d


# ============================================================================ H3b
def h3b_did(rows: pd.DataFrame, audit: pd.DataFrame, *, variant: str, baseline_arm: str = "nfc", arm: str = "nfd",
            exact: Sequence[str] = H3B_EXACT, nearest: Sequence[str] = H3B_NEAREST, k_min: int = H3B_K_MIN, k_max: int = H3B_K_MAX,
            caliper_sd: float | None = CALIPER_SD, min_items: int = H3B_MIN_ITEMS, n_boot: int = constants.BOOTSTRAP_B, seed: int = 0,
            level: float = 0.95) -> dict:
    """The tone-isolation DiD of one model and one T1 variant (PREREG 8.7 / DD 1 H3b): Δ = mean over ISO items of
    [(y(NFD) - y(NFC))_ISO - mean of the matched NON-ISO items' (y(NFD) - y(NFC))], 1:1 nearest neighbour with replacement
    by default (k_min / k_max widen it). `rows`: the model's T1 rows of the
    variant under both arms (one paraphrase), with item_id, base_pair_id, source, correct, input_freq; `audit`: the
    model's tokenizer item audit (read_items_audit) whose NFD rows give `tone_isolated` (ISO), `tokens_total` (NFD
    count) and `delta_tokens_vs_nfc`. Returns a JSON-safe dict: the contrast, the matching summary and
    `underpowered` (fewer than `min_items` items in the cell)."""
    d = rows[rows["variant"] == variant] if "variant" in rows else rows
    d = d[d["arm"].isin([baseline_arm, arm])]
    out: dict = {"variant": variant, "baseline_arm": baseline_arm, "arm": arm, "min_items": min_items}
    if len(d) == 0:
        return {**out, "computable": False, "reason": f"no {variant} rows under {baseline_arm} / {arm}"}
    piv = d.drop_duplicates(["item_id", "arm"]).pivot(index="item_id", columns="arm", values="correct")
    if baseline_arm not in piv or arm not in piv:
        return {**out, "computable": False, "reason": f"rows of only one arm ({list(piv.columns)})"}
    piv = piv.dropna()
    eff = (piv[arm].astype(float) - piv[baseline_arm].astype(float)).rename("effect").reset_index()
    first = d.drop_duplicates("item_id").set_index("item_id")
    eff["base_pair_id"] = eff["item_id"].map(first["base_pair_id"]).astype(str)
    eff["source"] = eff["item_id"].map(first["source"]) if "source" in first else "all"
    fr = first["input_freq"] if "input_freq" in first else pd.Series(dtype=object)
    eff["log_freq_in_1"] = eff["item_id"].map(lambda i: _log1p(fr.get(i, [0, 0])[0]) if isinstance(fr.get(i), (list, tuple)) else 0.0)
    eff["log_freq_in_2"] = eff["item_id"].map(lambda i: _log1p(fr.get(i, [0, 0])[1]) if isinstance(fr.get(i), (list, tuple)) and len(fr.get(i)) > 1 else 0.0)
    nfd = audit[audit["encoding"] == "nfd"].drop_duplicates("item_id").set_index("item_id") if "encoding" in audit else audit.set_index("item_id")
    eff = eff[eff["item_id"].isin(nfd.index)]
    n_items = len(eff)
    out.update(n_items=n_items, underpowered=n_items < min_items)
    if n_items == 0:
        return {**out, "computable": False, "reason": "no item of the cell has an NFD row in the item audit"}
    eff["tone_isolated"] = eff["item_id"].map(nfd["tone_isolated"]).fillna(False).astype(bool)
    tot = eff["item_id"].map(nfd["tokens_total"]).astype(float)
    delta = eff["item_id"].map(nfd["delta_tokens_vs_nfc"]).astype(float).fillna(0.0)
    eff["tokens_nfd"] = tot
    eff["tokens_nfc"] = tot - delta
    eff = eff.dropna(subset=[*nearest])
    iso, non = eff[eff["tone_isolated"]], eff[~eff["tone_isolated"]]
    out.update(n_iso=len(iso), n_non_iso=len(non))
    if len(iso) == 0 or len(non) == 0:
        return {**out, "computable": False, "reason": f"ISO {len(iso)} / NON-ISO {len(non)} items: both groups are needed"}
    match = nearest_neighbour_match(iso, non, exact=exact, nearest=nearest, k_min=k_min, k_max=k_max, caliper_sd=caliper_sd)
    out["matching"] = match.as_dict()
    if match.n_matched == 0:
        why = ", ".join(f"{k}: {v}" for k, v in sorted(match.unmatched.items())) or "no ISO item"
        return {**out, "computable": False, "reason": f"no ISO item matched ({why}; exact on {list(exact)}, caliper {caliper_sd} SD)"}
    y = dict(zip(eff["item_id"], eff["effect"]))
    cl = dict(zip(eff["item_id"], eff["base_pair_id"]))
    st = dict(zip(eff["item_id"], eff["source"]))
    mc = matched_contrast(y, y, match, cl, strata=st, n_boot=n_boot, seed=seed, level=level)
    out.update(computable=True, **mc.as_dict(), effect_iso=_f(iso["effect"].mean()), effect_non_iso=_f(non["effect"].mean()))
    out["estimand"] = "Δ = mean over ISO items of [(y(NFD) - y(NFC))_ISO - mean over matched NON-ISO items of (y(NFD) - y(NFC))]"
    return out
