"""Clustered bootstrap for accuracies and paired differences (design 8.2).

Items nest in base pairs (every T1/T2/T3 item built from the same underlying pair shares
a `base_pair_id`), and a model's answers to items from one pair are correlated. Every
interval here therefore resamples *base pairs* with replacement (Miller 2024, "Adding
error bars to evals"), not items. Paired differences (two conditions on the same items,
e.g. NFC vs NFD, or model A vs model B) resample the same pairs for both conditions.

What design 8.2 fixes and this module implements:
  * B = constants.BOOTSTRAP_B, fixed seed, resampling of base pairs, optionally STRATIFIED
    by base-pair type (`strata`: clusters are redrawn within their stratum, so every
    replicate keeps the lexical / pseudo split of the cell);
  * percentile intervals, BCa when the cell has >= constants.BCA_MIN_CLUSTERS base pairs
    (bias z0 from the bootstrap distribution, acceleration from a leave-one-cluster-out
    jackknife of the statistic);
  * cells with < constants.SMALL_CELL_MAX_BASE_PAIRS base pairs, or at 0 % / 100 %:
    a Wilson interval on n / DEFF instead of the bootstrap (DEFF = 1 + (m - 1) ICC with the
    ANOVA ICC over clusters), flagged `small_cell=True` in the CI; a non-binary statistic
    (a paired difference) in such a cell keeps the percentile interval and the flag;
  * the paired t on base-pair-level mean differences (`paired_t_base_pairs`), the PRIMARY
    inference for every paired contrast from constants.PAIRED_T_MIN_BASE_PAIRS base pairs.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .. import constants


@dataclass(frozen=True)
class CI:
    estimate: float
    lo: float
    hi: float
    n_items: int
    n_clusters: int
    level: float = 0.95
    n_boot: int = 0
    method: str = "percentile"     # percentile | bca | wilson_deff | percentile_small_cell
    small_cell: bool = False       # fewer than SMALL_CELL_MAX_BASE_PAIRS clusters, or a 0 % / 100 % cell

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "lo": self.lo, "hi": self.hi, "n_items": self.n_items,
                "n_clusters": self.n_clusters, "level": self.level, "n_boot": self.n_boot, "method": self.method,
                "small_cell": self.small_cell}

    def __str__(self) -> str:  # pragma: no cover
        flag = " SMALL CELL" if self.small_cell else ""
        return (f"{100*self.estimate:.1f} [{100*self.lo:.1f}, {100*self.hi:.1f}] (n={self.n_items}, "
                f"clusters={self.n_clusters}, {self.method}){flag}")


def _cluster_index(clusters: Sequence) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def _strata_of_clusters(inv: np.ndarray, n_clusters: int, strata: Sequence | None) -> list[np.ndarray] | None:
    """Cluster ids grouped by stratum (None when unstratified). Every cluster must lie in one
    stratum (a base pair has one type), otherwise the stratification is ill-defined."""
    if strata is None:
        return None
    st = np.asarray(strata)
    if len(st) != len(inv):
        raise ValueError("strata must be given per item, aligned with values and clusters")
    stratum_of: dict[int, object] = {}
    for c, s in zip(inv.tolist(), st.tolist()):
        if c in stratum_of and stratum_of[c] != s:
            raise ValueError(f"cluster {c} appears in two strata ({stratum_of[c]!r}, {s!r}); a base pair has one type")
        stratum_of[c] = s
    groups: dict[object, list[int]] = {}
    for c in range(n_clusters):
        groups.setdefault(stratum_of[c], []).append(c)
    return [np.asarray(v) for v in groups.values()]


def _resample_weights(inv: np.ndarray, n_clusters: int, rng: np.random.Generator,
                      strata_groups: list[np.ndarray] | None = None) -> np.ndarray:
    """How many times each item appears when clusters are drawn with replacement (within
    each stratum when `strata_groups` is given: a replicate keeps every stratum's cluster count)."""
    if strata_groups is None:
        draws = rng.integers(0, n_clusters, size=n_clusters)
    else:
        draws = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in strata_groups])
    counts = np.bincount(draws, minlength=n_clusters)
    return counts[inv]


def _mean_stat(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w)) if w.sum() else float("nan")


def _is_binary(v: np.ndarray) -> bool:
    return bool(np.all(np.isin(v, (0.0, 1.0))))


def anova_icc(values: np.ndarray, inv: np.ndarray, n_clusters: int) -> float:
    """One-way ANOVA intraclass correlation of `values` over clusters (Fleiss), clipped to
    [0, 1]; 0 when it is not estimable (fewer than two clusters, or singleton clusters only)."""
    n = len(values)
    if n_clusters < 2 or n <= n_clusters:
        return 0.0
    sizes = np.bincount(inv, minlength=n_clusters).astype(float)
    means = np.bincount(inv, weights=values, minlength=n_clusters) / sizes
    grand = values.mean()
    ms_between = float(np.sum(sizes * (means - grand) ** 2) / (n_clusters - 1))
    ms_within = float(np.sum((values - means[inv]) ** 2) / (n - n_clusters))
    m0 = (n - np.sum(sizes ** 2) / n) / (n_clusters - 1)
    denom = ms_between + (m0 - 1) * ms_within
    if denom <= 0:
        return 0.0
    return float(np.clip((ms_between - ms_within) / denom, 0.0, 1.0))


def design_effect_from_clusters(values: np.ndarray, inv: np.ndarray, n_clusters: int) -> float:
    """DEFF = 1 + (m - 1) ICC with m the mean cluster size (design 8.5)."""
    if n_clusters == 0:
        return 1.0
    m = len(values) / n_clusters
    return 1.0 + (m - 1.0) * anova_icc(values, inv, n_clusters)


def wilson_deff_ci(values: np.ndarray, inv: np.ndarray, n_clusters: int, level: float = 0.95) -> CI:
    """Wilson interval for a binary cell on the design-effect-adjusted sample size n / DEFF."""
    from statsmodels.stats.proportion import proportion_confint

    n = len(values)
    p = float(values.mean()) if n else float("nan")
    n_eff = max(1.0, n / design_effect_from_clusters(values, inv, n_clusters))
    lo, hi = proportion_confint(p * n_eff, n_eff, alpha=1 - level, method="wilson")
    lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
    if p in (0.0, 1.0):      # a 0 % / 100 % cell: the bound at the observed edge is exact
        lo, hi = (0.0, hi) if p == 0.0 else (lo, 1.0)
    return CI(p, lo, hi, n, n_clusters, level, 0, method="wilson_deff", small_cell=True)


def _bca_bounds(boots: np.ndarray, est: float, jack: np.ndarray, level: float) -> tuple[float, float] | None:
    """BCa percentile positions (Efron 1987) from the bootstrap replicates, the estimate and
    the leave-one-cluster-out jackknife values; None when they are degenerate."""
    b = boots[np.isfinite(boots)]
    if len(b) == 0:
        return None
    frac = float(np.mean(b < est))
    if frac <= 0.0 or frac >= 1.0:
        return None
    z0 = sps.norm.ppf(frac)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * np.sum((jm - jack) ** 2) ** 1.5
    a = float(num / den) if den > 0 else 0.0
    alpha = (1 - level) / 2
    out = []
    for z in (sps.norm.ppf(alpha), sps.norm.ppf(1 - alpha)):
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(sps.norm.cdf(adj)))
    lo_q, hi_q = out
    if not (0.0 < lo_q < hi_q < 1.0):
        return None
    return lo_q, hi_q


def cluster_bootstrap(values: Sequence[float], clusters: Sequence, stat: Callable[[np.ndarray, np.ndarray], float] | None = None,
                      n_boot: int = constants.BOOTSTRAP_B, level: float = 0.95, seed: int = 0,
                      strata: Sequence | None = None, bca: bool | None = None, small_cell_rule: bool = True) -> CI:
    """Bootstrap CI of a weighted statistic (default: the mean) over items, resampling
    clusters (within `strata` when given). `stat(values, weights)` must accept per-item
    weights. `bca=None` follows design 8.2: BCa from constants.BCA_MIN_CLUSTERS clusters,
    percentile below; `bca=False` forces the percentile interval. With `small_cell_rule` a
    binary cell with fewer than constants.SMALL_CELL_MAX_BASE_PAIRS clusters or at 0 % / 100 %
    gets the Wilson interval on n / DEFF instead (flagged); a non-binary statistic in such a
    cell keeps the percentile interval, flagged `small_cell=True`."""
    v = np.asarray(values, dtype=float)
    inv, k = _cluster_index(clusters)
    groups = _strata_of_clusters(inv, k, strata)
    if stat is None:
        stat = _mean_stat
    est = stat(v, np.ones_like(v))
    binary = _is_binary(v) and stat is _mean_stat
    small = k < constants.SMALL_CELL_MAX_BASE_PAIRS or (binary and est in (0.0, 1.0))
    if small_cell_rule and small and binary:
        return wilson_deff_ci(v, inv, k, level)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = _resample_weights(inv, k, rng, groups)
        boots[b] = stat(v, w)
    alpha = (1 - level) / 2
    use_bca = (k >= constants.BCA_MIN_CLUSTERS) if bca is None else bool(bca)
    method = "percentile"
    lo_q, hi_q = alpha, 1 - alpha
    if use_bca and k >= 2:
        jack = np.empty(k)
        ones = np.ones_like(v)
        for c in range(k):
            w = ones.copy()
            w[inv == c] = 0.0
            jack[c] = stat(v, w)
        bounds = _bca_bounds(boots, est, jack, level)
        if bounds is not None:
            lo_q, hi_q = bounds
            method = "bca"
    lo, hi = np.nanquantile(boots, [lo_q, hi_q])
    if small_cell_rule and small:
        method = "percentile_small_cell"
    return CI(float(est), float(lo), float(hi), len(v), k, level, n_boot, method=method, small_cell=bool(small_cell_rule and small))


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
                            n_boot: int = 4000, seed: int = 0, strata: Sequence | None = None) -> float:
    """Two-sided p-value for H0: mean(b) == mean(a) from the clustered bootstrap
    distribution of the difference, centred at the observed difference (Efron & Tibshirani)."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    d = b - a
    inv, k = _cluster_index(clusters)
    groups = _strata_of_clusters(inv, k, strata)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        w = _resample_weights(inv, k, rng, groups)
        boots[i] = np.average(d, weights=w) if w.sum() else np.nan
    centred = boots - obs
    p = (np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1)
    return float(min(1.0, p))


@dataclass(frozen=True)
class PairedT:
    t: float
    p: float
    n_clusters: int
    mean_diff: float          # mean over base pairs of the per-pair mean difference (b - a)
    se: float
    primary: bool             # n_clusters >= constants.PAIRED_T_MIN_BASE_PAIRS (design 8.2)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def paired_t_base_pairs(correct_a: Sequence[float], correct_b: Sequence[float], clusters: Sequence) -> PairedT:
    """Paired t on base-pair-level mean differences: d_i = b_i - a_i per item, averaged within
    each base pair, then a one-sample t against 0 over base pairs (design 8.2, the primary
    inference for every paired contrast from PAIRED_T_MIN_BASE_PAIRS base pairs). Accepts
    binary rows or paraphrase-aggregated proportions alike."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired conditions must have the same items in the same order")
    d = b - a
    inv, k = _cluster_index(clusters)
    if k < 2:
        return PairedT(float("nan"), float("nan"), k, float(d.mean()) if len(d) else float("nan"), float("nan"), False)
    counts = np.bincount(inv, minlength=k).astype(float)
    means = np.bincount(inv, weights=d, minlength=k) / counts
    res = sps.ttest_1samp(means, 0.0)
    se = float(means.std(ddof=1) / np.sqrt(k))
    return PairedT(float(res.statistic), float(res.pvalue), k, float(means.mean()), se,
                   k >= constants.PAIRED_T_MIN_BASE_PAIRS)


def stratified_accuracy(df, by: Sequence[str], correct_col: str = "correct", cluster_col: str = "base_pair_id", **kw) -> list[dict]:
    """Accuracy CI per stratum of a pandas DataFrame (e.g. by task and variant)."""
    out = []
    for keys, g in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ci = accuracy_ci(g[correct_col].astype(bool).values, g[cluster_col].values, **kw)
        out.append({**dict(zip(by, keys)), **ci.as_dict()})
    return out
