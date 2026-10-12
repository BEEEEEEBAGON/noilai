"""Agreement statistics for the native-validator judgments (design 10.1, 12.21; PREREG 8.12).

Krippendorff's alpha (nominal) with a bootstrap CI, raw (pairwise percent) agreement, the
marginal label distribution and Gwet's AC1 (Gwet 2008) — AC1 is reported beside alpha
because at the ~96 % prevalence the validation anticipates, chance-corrected alpha falls
even when two coders agree on 96 % of items, whereas AC1's chance term is small.

Input: a table of (item, coder, label) with missing cells allowed (each item is judged by
two of three validators, a 200-item overlap by all three). Alpha is implemented directly
from the coincidence-matrix definition (Krippendorff 2011) so that no extra dependency is
needed; `nltk.metrics.agreement` gives the same value on complete data.

`weighted_precision` is not an agreement statistic: it is the generator-precision estimate per cell
"with a CI from the weighted probability sample" (PREREG 8.12, docs/PREREGISTRATION.md:129; DD 10.1
:505 names `stats.agreement.weighted_precision`, red-team row 66 :696), which is why it lives here.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps


def krippendorff_alpha_nominal(ratings: Iterable[tuple[Hashable, Hashable, Hashable]]) -> float:
    """ratings: iterable of (item_id, coder_id, label). Items with fewer than two labels are ignored."""
    by_item: dict = defaultdict(list)
    for item, _coder, label in ratings:
        by_item[item].append(label)
    units = [labels for labels in by_item.values() if len(labels) >= 2]
    if not units:
        return float("nan")
    values = sorted({v for labels in units for v in labels}, key=str)
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for labels in units:
        m = len(labels)
        for a in labels:
            for b in labels:
                if a is b and labels.count(a) == 1:
                    continue
        # coincidence matrix contribution: pairs of values within the unit, weighted 1/(m-1)
        counts = np.zeros(k)
        for v in labels:
            counts[idx[v]] += 1
        for c in range(k):
            for d in range(k):
                if c == d:
                    o[c, d] += counts[c] * (counts[c] - 1) / (m - 1)
                else:
                    o[c, d] += counts[c] * counts[d] / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d_o = (n - np.trace(o))
    d_e = (n_c.sum() ** 2 - np.sum(n_c ** 2)) / (n - 1)
    if d_e == 0:
        return 1.0
    return float(1 - d_o / d_e)


def _resample_items(ratings: Sequence[tuple[Hashable, Hashable, Hashable]], statistic: Callable, n_boot: int,
                    seed: int, level: float) -> tuple[float, float, float]:
    """Item (unit) bootstrap of any agreement statistic over (item, coder, label) triples."""
    by_item: dict = defaultdict(list)
    for item, coder, label in ratings:
        by_item[item].append((coder, label))
    items = list(by_item)
    est = statistic(ratings)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(items), size=len(items))
        rs = []
        for j, i in enumerate(pick):
            for coder, label in by_item[items[i]]:
                rs.append((f"{items[i]}#{j}", coder, label))
        boots.append(statistic(rs))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    return est, float(lo), float(hi)


def alpha_bootstrap_ci(ratings: Sequence[tuple[Hashable, Hashable, Hashable]], n_boot: int = 1000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float, float]:
    """Resample items with replacement."""
    return _resample_items(ratings, krippendorff_alpha_nominal, n_boot, seed, level)


def ac1_bootstrap_ci(ratings: Sequence[tuple[Hashable, Hashable, Hashable]], n_boot: int = 1000, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float, float]:
    return _resample_items(ratings, gwet_ac1, n_boot, seed, level)


def _units(ratings: Iterable[tuple[Hashable, Hashable, Hashable]]) -> list[list]:
    by_item: dict = defaultdict(list)
    for item, _coder, label in ratings:
        by_item[item].append(label)
    return [labels for labels in by_item.values() if len(labels) >= 2]


def marginal_distribution(ratings: Iterable[tuple[Hashable, Hashable, Hashable]]) -> dict:
    """Share of each label over all judgments of units with >= 2 labels."""
    units = _units(ratings)
    counts: dict = defaultdict(int)
    for labels in units:
        for v in labels:
            counts[v] += 1
    n = sum(counts.values())
    return {str(k): v / n for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))} if n else {}


def gwet_ac1(ratings: Iterable[tuple[Hashable, Hashable, Hashable]]) -> float:
    """Gwet's AC1 (nominal) over units with >= 2 labels, for any number of coders per unit:
        p_a = mean over units of sum_q r_q (r_q - 1) / (r (r - 1)),
        pi_q = mean over units of r_q / r,
        p_e = sum_q pi_q (1 - pi_q) / (Q - 1),
        AC1 = (p_a - p_e) / (1 - p_e),
    with r the unit's number of labels, r_q the count of category q and Q the number of
    categories observed. 1.0 when only one category ever occurs and every unit agrees."""
    units = _units(ratings)
    if not units:
        return float("nan")
    values = sorted({v for labels in units for v in labels}, key=str)
    Q = len(values)
    pa_terms = []
    pi = np.zeros(Q)
    for labels in units:
        r = len(labels)
        counts = np.array([labels.count(v) for v in values], dtype=float)
        pa_terms.append(float(np.sum(counts * (counts - 1)) / (r * (r - 1))))
        pi += counts / r
    pi /= len(units)
    p_a = float(np.mean(pa_terms))
    if Q < 2:
        return 1.0 if p_a == 1.0 else float("nan")
    p_e = float(np.sum(pi * (1 - pi)) / (Q - 1))
    if p_e >= 1.0:
        return float("nan")
    return float((p_a - p_e) / (1 - p_e))


def percent_agreement(ratings: Iterable[tuple[Hashable, Hashable, Hashable]]) -> float:
    by_item: dict = defaultdict(list)
    for item, _c, label in ratings:
        by_item[item].append(label)
    agree = tot = 0
    for labels in by_item.values():
        if len(labels) < 2:
            continue
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                tot += 1
                agree += labels[i] == labels[j]
    return agree / tot if tot else float("nan")


def krippendorff_alpha(ratings: Iterable[tuple[Hashable, Hashable, Hashable]], distance: Callable[[Hashable, Hashable], float]) -> float:
    """Krippendorff's alpha for any distance function (e.g. MASI or Jaccard over set-valued labels, the T2
    gold-set judgments of DESIGN_DECISIONS 10.1): alpha = 1 - D_o / D_e, with D_o the mean distance between
    pairable values within units (each unit's pairs weighted 1 / (m_u - 1)) and D_e the mean distance between
    all pairable values. Equals `krippendorff_alpha_nominal` when distance is 0/1 inequality."""
    by_item: dict = defaultdict(list)
    for item, _coder, label in ratings:
        by_item[item].append(label)
    units = [labels for labels in by_item.values() if len(labels) >= 2]
    if not units:
        return float("nan")
    values = [v for labels in units for v in labels]
    n = len(values)
    d_o = 0.0
    for labels in units:
        m = len(labels)
        s = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    s += distance(labels[i], labels[j])
        d_o += s / (m - 1)
    d_o /= n
    d_e = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d_e += distance(values[i], values[j])
    d_e /= n * (n - 1)
    if d_e == 0:
        return 1.0
    return float(1 - d_o / d_e)


# --------------------------------------------------------------------------- generator precision (DD 10.1)
def _cell_of_stratum(key: str, rec: Mapping) -> str:
    """The cell a stratum belongs to: an explicit `cell`, else the first two `|`-fields of the key
    (`<task>|<variant>|<lexicon|pseudo>` -> `<task>-<variant>`, as validation.stratified_precision)."""
    if rec.get("cell"):
        return str(rec["cell"])
    return "-".join(str(key).split("|")[:2])


def _check_stratum(key: str, rec: Mapping) -> tuple[int, int, int]:
    try:
        n_pop, n_s, n_c = int(rec["n_population"]), int(rec["n_sampled"]), int(rec["n_correct"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"stratum {key!r} needs integer n_population, n_sampled and n_correct") from e
    if n_pop < 0 or n_s < 0 or n_c < 0 or n_c > n_s or n_s > n_pop:
        raise ValueError(f"stratum {key!r}: need 0 <= n_correct <= n_sampled <= n_population "
                         f"(got {n_c}, {n_s}, {n_pop})")
    w = rec.get("weight")
    if w is not None and n_s > 0 and abs(float(w) - n_pop / n_s) > 1e-6 * max(1.0, n_pop / n_s):
        raise ValueError(f"stratum {key!r}: weight {w} is not n_population / n_sampled = {n_pop / n_s}")
    return n_pop, n_s, n_c


def _weighted_cell(rows: list[tuple[str, int, int, int]], level: float, fpc: bool) -> dict:
    """One cell (or the pool): p = sum_h W_h p_h with W_h = N_h / N over the sampled strata;
    Var = sum_h W_h^2 p_h (1 - p_h) / n_h (1 - n_h / N_h); Wilson on n_eff = p (1 - p) / Var."""
    from statsmodels.stats.proportion import proportion_confint

    sampled = [r for r in rows if r[2] > 0]
    empty = [r[0] for r in rows if r[2] == 0]
    n_pop_all = sum(r[1] for r in rows)
    n_pop = sum(r[1] for r in sampled)
    n_s = sum(r[2] for r in sampled)
    n_c = sum(r[3] for r in sampled)
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    strata_out, singleton = [], []
    est = var = 0.0
    for key, N_h, n_h, c_h in sampled:
        p_h = c_h / n_h
        W_h = N_h / n_pop
        f_h = (1.0 - n_h / N_h) if fpc else 1.0
        contrib = W_h ** 2 * p_h * (1 - p_h) / n_h * f_h       # n_h = 1: p_h (1 - p_h), flagged below
        if n_h == 1:
            singleton.append(key)
        est += W_h * p_h
        var += contrib
        strata_out.append({"stratum": key, "n_population": N_h, "n_sampled": n_h, "n_correct": c_h, "precision": p_h,
                           "weight_share": W_h, "weight": N_h / n_h, "variance_contribution": contrib,
                           "singleton": n_h == 1})
    fpc_label = "fpc" if fpc else "no fpc"
    if not sampled:
        return {"estimate": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": level,
                "method": f"stratified_wilson_neff({fpc_label})", "n_sampled": 0, "n_population": n_pop_all,
                "n_correct": 0, "n_strata": 0, "n_eff": float("nan"), "strata": [], "singleton_strata": [],
                "empty_strata": empty, "coverage": 0.0, "ci_normal": [float("nan"), float("nan")]}
    se = float(np.sqrt(var))
    if var > 0 and 0.0 < est < 1.0:
        n_eff = est * (1 - est) / var
        lo, hi = proportion_confint(est * n_eff, n_eff, alpha=1 - level, method="wilson")
        method = f"stratified_wilson_neff({fpc_label})"
    else:                                   # every stratum at 0 or 1: Wilson on the raw n (conservative)
        n_eff = float(n_s)
        lo, hi = proportion_confint(est * n_s, n_s, alpha=1 - level, method="wilson")
        method = f"wilson_raw_n(var=0, {fpc_label})"
    lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
    if est in (0.0, 1.0):                   # the bound at the observed edge is exact (bootstrap.wilson_deff_ci)
        lo, hi = (0.0, hi) if est == 0.0 else (lo, 1.0)
    ci_normal = [float(np.clip(est - z * se, 0.0, 1.0)), float(np.clip(est + z * se, 0.0, 1.0))]
    return {"estimate": float(est), "se": se, "lo": lo, "hi": hi, "level": level, "method": method,
            "n_sampled": n_s, "n_population": n_pop_all, "n_correct": n_c, "n_strata": len(sampled),
            "n_eff": float(n_eff), "strata": strata_out, "singleton_strata": singleton, "empty_strata": empty,
            "coverage": (n_pop / n_pop_all) if n_pop_all else float("nan"), "ci_normal": ci_normal}


def weighted_precision(strata: Mapping[str, Mapping], *, level: float = 0.95, fpc: bool = True) -> dict:
    """Generator precision per cell and pooled from the stratified probability sample (DD 10.1 :505,
    PREREG 8.12 :129; the sample of validation.probability_sample, strata = task x variant x lexicon/pseudo
    with n_population, n_sampled and weight recorded per stratum). Each stratum record carries
    {n_population, n_sampled, n_correct[, cell, weight]}; `weight` is checked against n_population / n_sampled.

    Per cell (strata grouped by `cell`, default the first two `|`-fields of the key) and `pooled` (all strata):
      estimate p = sum_h W_h p_h, W_h = N_h / N over the sampled strata, p_h = n_correct_h / n_sampled_h;
      Var(p) = sum_h W_h^2 p_h (1 - p_h) / n_h (1 - n_h / N_h) (the finite population correction when `fpc`;
      the release stratum is finite and small strata may be sampled heavily); a stratum with n_h = 1 has no
      unbiased variance, contributes W_h^2 p_h (1 - p_h) and is flagged under `singleton_strata`;
      interval: Wilson on the effective sample size n_eff = p (1 - p) / Var (the device of
      bootstrap.wilson_deff_ci), clipped to [0, 1]; when Var = 0 Wilson on the raw n, and p in {0, 1} pins the
      bound at the observed edge; `ci_normal` is the plain normal interval for the appendix.
    Strata with n_sampled = 0 are listed under `empty_strata`, carry weight 0 and leave `coverage` < 1.
    Controls and unresolved rows never reach this function (weighted_precision_from_rows drops them, as
    validation.stratified_precision does; PREREG 5.2 :60, DD 10.1 :503)."""
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    cells: dict[str, list[tuple[str, int, int, int]]] = {}
    for key, rec in strata.items():
        n_pop, n_s, n_c = _check_stratum(str(key), rec)
        cells.setdefault(_cell_of_stratum(str(key), rec), []).append((str(key), n_pop, n_s, n_c))
    out = {cell: _weighted_cell(rows, level, fpc) for cell, rows in sorted(cells.items())}
    out["pooled"] = _weighted_cell([r for rows in cells.values() for r in rows], level, fpc)
    return out


def weighted_precision_from_rows(final: Mapping[str, str | None], key: Mapping[str, Mapping], correct_label: str = "yes",
                                 **kw) -> dict:
    """weighted_precision from the adjudicated B rows: `final[row_id]` is the resolved label (None =
    unresolved: unsure or awaiting the author, excluded and counted per cell as `n_unresolved`), `key[row_id]`
    the B_key record (cell, stratum, stratum_population, control). Controls are excluded (DD 10.1 :503). The
    estimate equals validation.stratified_precision(final, key)[cell]["weighted_precision"]."""
    strata: dict[str, dict] = {}
    unresolved: dict[str, int] = {}
    for rid, k in key.items():
        if k.get("control"):
            continue
        cell = k.get("cell") or "-".join(str(k["stratum"]).split("|")[:2])
        lab = final.get(rid)
        if lab is None:
            unresolved[cell] = unresolved.get(cell, 0) + 1
            continue
        s = strata.setdefault(str(k["stratum"]), {"n_population": int(k["stratum_population"]), "n_sampled": 0,
                                                  "n_correct": 0, "cell": cell})
        if int(k["stratum_population"]) != s["n_population"]:
            raise ValueError(f"stratum {k['stratum']!r} carries two populations")
        s["n_sampled"] += 1
        s["n_correct"] += int(lab == correct_label)
    out = weighted_precision(strata, **kw)
    for cell in out:
        out[cell]["n_unresolved"] = sum(unresolved.values()) if cell == "pooled" else unresolved.get(cell, 0)
    return out
