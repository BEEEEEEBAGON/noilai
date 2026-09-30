"""Clustered bootstrap for accuracies and paired differences.

Items nest in base pairs (every T1/T2/T3 item built from the same underlying pair shares
a `base_pair_id`), and a model's answers to items from one pair are correlated. Every
interval here therefore resamples *base pairs* with replacement (Miller 2024, "Adding
error bars to evals"), not items. Paired differences (two conditions on the same items,
e.g. NFC vs NFD, or model A vs model B) resample the same pairs for both conditions.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    estimate: float
    lo: float
    hi: float
    n_items: int
    n_clusters: int
    level: float = 0.95
    n_boot: int = 0

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "lo": self.lo, "hi": self.hi, "n_items": self.n_items,
                "n_clusters": self.n_clusters, "level": self.level, "n_boot": self.n_boot}

    def __str__(self) -> str:  # pragma: no cover
        return f"{100*self.estimate:.1f} [{100*self.lo:.1f}, {100*self.hi:.1f}] (n={self.n_items}, clusters={self.n_clusters})"


def _cluster_index(clusters: Sequence) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def _resample_weights(inv: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """How many times each item appears when clusters are drawn with replacement."""
    draws = rng.integers(0, n_clusters, size=n_clusters)
    counts = np.bincount(draws, minlength=n_clusters)
    return counts[inv]


def cluster_bootstrap(values: Sequence[float], clusters: Sequence, stat: Callable[[np.ndarray, np.ndarray], float] | None = None,
                      n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> CI:
    """Percentile bootstrap CI of a weighted statistic (default: the mean) over items,
    resampling clusters. `stat(values, weights)` must accept per-item weights."""
    v = np.asarray(values, dtype=float)
    inv, k = _cluster_index(clusters)
    if stat is None:
        stat = lambda x, w: float(np.average(x, weights=w)) if w.sum() else float("nan")
    est = stat(v, np.ones_like(v))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = _resample_weights(inv, k, rng)
        boots[b] = stat(v, w)
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return CI(float(est), float(lo), float(hi), len(v), k, level, n_boot)


def accuracy_ci(correct: Sequence[bool], clusters: Sequence, **kw) -> CI:
    return cluster_bootstrap(np.asarray(correct, dtype=float), clusters, **kw)


def paired_difference_ci(correct_a: Sequence[bool], correct_b: Sequence[bool], clusters: Sequence, **kw) -> CI:
    """CI of mean(b) - mean(a) on the same items (e.g. arm minus baseline)."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired conditions must have the same items in the same order")
    return cluster_bootstrap(b - a, clusters, **kw)


def paired_bootstrap_pvalue(correct_a: Sequence[bool], correct_b: Sequence[bool], clusters: Sequence,
                            n_boot: int = 4000, seed: int = 0) -> float:
    """Two-sided p-value for H0: mean(b) == mean(a) from the clustered bootstrap
    distribution of the difference, centred at the observed difference (Efron & Tibshirani)."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    d = b - a
    inv, k = _cluster_index(clusters)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        w = _resample_weights(inv, k, rng)
        boots[i] = np.average(d, weights=w) if w.sum() else np.nan
    centred = boots - obs
    p = (np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1)
    return float(min(1.0, p))


def stratified_accuracy(df, by: Sequence[str], correct_col: str = "correct", cluster_col: str = "base_pair_id", **kw) -> list[dict]:
    """Accuracy CI per stratum of a pandas DataFrame (e.g. by task and variant)."""
    out = []
    for keys, g in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ci = accuracy_ci(g[correct_col].astype(bool).values, g[cluster_col].values, **kw)
        out.append({**dict(zip(by, keys)), **ci.as_dict()})
    return out
