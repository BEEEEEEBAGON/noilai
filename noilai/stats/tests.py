"""Paired significance tests and multiple-comparison control (design 8.2, 8.3).

Primary inference for a paired contrast (model vs model, arm vs base, H3b): the paired t on
base-pair-level mean differences when the cell has >= constants.PAIRED_T_MIN_BASE_PAIRS base
pairs (bootstrap.paired_t_base_pairs), otherwise the two-sided p from the same base-pair
cluster bootstrap that gives the CI. McNemar's test (exact mid-p; Dror et al. 2018) is the
SECONDARY p0 item-level statistic for single-model binary contrasts, labelled as ignoring
clustering; it needs binary paired outcomes and `paired_test` refuses aggregated paraphrase
proportions (items 28, 52). Holm's step-down correction is applied WITHIN a declared family
(Dror et al. 2017): the Table 3 family is one model row over the cells enumerated in
constants.HOLM_FAMILY_TABLE3.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .. import constants
from .bootstrap import paired_bootstrap_pvalue, paired_difference_ci, paired_t_base_pairs


@dataclass
class PairedTest:
    name: str
    n: int
    acc_a: float
    acc_b: float
    diff: float
    n_only_a: int          # a correct, b wrong
    n_only_b: int          # b correct, a wrong
    p_mcnemar: float       # secondary: item-level, ignores clustering (design 8.2)
    p_bootstrap: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_adj: float | None = None
    significant: bool | None = None
    p_paired_t: float | None = None      # paired t on base-pair mean differences (primary from PAIRED_T_MIN_BASE_PAIRS)
    t_paired: float | None = None
    n_clusters: int | None = None
    p_primary_source: str | None = None  # which p the family correction used: paired_t | bootstrap | mcnemar
    ci_method: str | None = None
    small_cell: bool | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def p_primary(self) -> float:
        """The pre-registered primary p: paired t when the cell has enough base pairs, else the
        bootstrap p, else (no clustering information) McNemar."""
        if self.p_paired_t is not None and self.n_clusters is not None and np.isfinite(self.p_paired_t) \
                and self.n_clusters >= constants.PAIRED_T_MIN_BASE_PAIRS:
            return self.p_paired_t
        if self.p_bootstrap is not None:
            return self.p_bootstrap
        return self.p_mcnemar

    @property
    def p_primary_name(self) -> str:
        if self.p_paired_t is not None and self.n_clusters is not None and np.isfinite(self.p_paired_t) \
                and self.n_clusters >= constants.PAIRED_T_MIN_BASE_PAIRS:
            return "paired_t"
        return "bootstrap" if self.p_bootstrap is not None else "mcnemar"


def mcnemar_exact(n_only_a: int, n_only_b: int, mid_p: bool = True) -> float:
    """Exact two-sided McNemar p-value from the discordant counts (binomial with p=1/2).
    mid_p halves the probability of the observed count (Fagerland et al. 2013), which is
    less conservative than the plain exact test and recommended for small discordant counts."""
    n = n_only_a + n_only_b
    if n == 0:
        return 1.0
    k = min(n_only_a, n_only_b)
    p_two = 2 * sps.binom.cdf(k, n, 0.5)
    if mid_p:
        p_two -= sps.binom.pmf(k, n, 0.5)
    return float(min(1.0, max(0.0, p_two)))


def _require_binary(name: str, x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == bool:
        return arr
    vals = np.unique(np.asarray(arr, dtype=float))
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise ValueError(f"paired_test needs binary outcomes ({name} has values {vals[:5]}); aggregated paraphrase "
                         "proportions go to bootstrap.paired_difference_ci / bootstrap.paired_t_base_pairs on "
                         "base-pair means (design 8.2, items 28, 52)")
    return arr.astype(bool)


def paired_test(correct_a: Sequence[bool], correct_b: Sequence[bool], clusters: Sequence | None = None,
                name: str = "", bootstrap: bool = True, n_boot: int = 4000, seed: int = 0,
                strata: Sequence | None = None) -> PairedTest:
    """Paired contrast b vs a on the same items. Binary outcomes only (McNemar is undefined on
    proportions). With `clusters` (base pairs) the paired t on base-pair means and the
    clustered bootstrap CI / p are computed; `strata` stratifies the bootstrap by base-pair type."""
    a = _require_binary("correct_a", correct_a)
    b = _require_binary("correct_b", correct_b)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must be aligned")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    p_mc = mcnemar_exact(only_a, only_b)
    p_bs = ci_lo = ci_hi = p_t = t_stat = n_cl = ci_method = small = None
    if bootstrap:
        cl = clusters if clusters is not None else np.arange(len(a))
        p_bs = paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed, strata=strata)
        ci = paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed, strata=strata)
        ci_lo, ci_hi, ci_method, small = ci.lo, ci.hi, ci.method, ci.small_cell
    if clusters is not None:
        pt = paired_t_base_pairs(a, b, clusters)
        p_t, t_stat, n_cl = pt.p, pt.t, pt.n_clusters
    return PairedTest(name=name, n=len(a), acc_a=float(a.mean()), acc_b=float(b.mean()), diff=float(b.mean() - a.mean()),
                      n_only_a=only_a, n_only_b=only_b, p_mcnemar=p_mc, p_bootstrap=p_bs, ci_lo=ci_lo, ci_hi=ci_hi,
                      p_paired_t=p_t, t_paired=t_stat, n_clusters=n_cl, ci_method=ci_method, small_cell=small)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjusted p-values and rejection flags (controls FWER within a family)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p, np.zeros(0, dtype=bool)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)      # enforce monotonicity
        adj[idx] = running
    return adj, adj <= alpha


def correct_family(tests: Sequence[PairedTest], alpha: float = constants.ALPHA, use: str = "primary") -> list[PairedTest]:
    """Apply Holm within one family, in place. `use`: 'primary' (design 8.2: paired t from
    PAIRED_T_MIN_BASE_PAIRS base pairs, else the bootstrap p), 'bootstrap', or 'mcnemar' (the
    labelled secondary that ignores clustering)."""
    if use == "primary":
        ps = [t.p_primary for t in tests]
        names = [t.p_primary_name for t in tests]
    elif use == "bootstrap":
        ps = [t.p_bootstrap if t.p_bootstrap is not None else t.p_mcnemar for t in tests]
        names = ["bootstrap" if t.p_bootstrap is not None else "mcnemar" for t in tests]
    elif use == "mcnemar":
        ps = [t.p_mcnemar for t in tests]
        names = ["mcnemar"] * len(tests)
    else:
        raise ValueError(f"unknown p-value source {use!r}")
    adj, rej = holm(ps, alpha)
    for t, a, r, nm in zip(tests, adj, rej, names):
        t.p_adj = float(a)
        t.significant = bool(r)
        t.p_primary_source = nm
    return list(tests)


def families_by(records: Sequence[dict], family_keys: Sequence[str]) -> dict[tuple, list[dict]]:
    """Group comparison records into families by the given keys (e.g. ('model',) for the
    'arms vs baseline within a model' family)."""
    out: dict[tuple, list[dict]] = {}
    for r in records:
        out.setdefault(tuple(r[k] for k in family_keys), []).append(r)
    return out
