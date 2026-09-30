"""Agreement statistics for the native-validator judgments (design 10.1, 12.21; PREREG 8.12).

Krippendorff's alpha (nominal) with a bootstrap CI, raw (pairwise percent) agreement, the
marginal label distribution and Gwet's AC1 (Gwet 2008) — AC1 is reported beside alpha
because at the ~96 % prevalence the validation anticipates, chance-corrected alpha falls
even when two coders agree on 96 % of items, whereas AC1's chance term is small.

Input: a table of (item, coder, label) with missing cells allowed (each item is judged by
two of three validators, a 200-item overlap by all three). Alpha is implemented directly
from the coincidence-matrix definition (Krippendorff 2011) so that no extra dependency is
needed; `nltk.metrics.agreement` gives the same value on complete data.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence

import numpy as np


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
