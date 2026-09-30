"""Krippendorff's alpha (nominal) for the native-validator judgments, with a bootstrap CI.

Input: a table of (item, coder, label) with missing cells allowed (each item is judged by
two of three validators, a 200-item overlap by all three). Implemented directly from the
coincidence-matrix definition (Krippendorff 2011) so that no extra dependency is needed;
`nltk.metrics.agreement` gives the same value on complete data.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Sequence

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


def alpha_bootstrap_ci(ratings: Sequence[tuple[Hashable, Hashable, Hashable]], n_boot: int = 1000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float, float]:
    """Resample items with replacement."""
    by_item: dict = defaultdict(list)
    for item, coder, label in ratings:
        by_item[item].append((coder, label))
    items = list(by_item)
    est = krippendorff_alpha_nominal(ratings)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(items), size=len(items))
        rs = []
        for j, i in enumerate(pick):
            for coder, label in by_item[items[i]]:
                rs.append((f"{items[i]}#{j}", coder, label))
        boots.append(krippendorff_alpha_nominal(rs))
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    return est, float(lo), float(hi)


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
