"""Paired significance tests and multiple-comparison control.

McNemar's test (exact binomial and mid-p) for two paired binary outcomes on the same items
(Dror et al. 2018), the clustered paired bootstrap (bootstrap.py), and Holm's step-down
correction applied WITHIN a declared family (Dror et al. 2017). A family is a set of
comparisons that share a claim, e.g. "all C1–C3 arms against the NFC baseline for one
model" or "every model against the best model on T1".
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .bootstrap import paired_bootstrap_pvalue, paired_difference_ci


@dataclass
class PairedTest:
    name: str
    n: int
    acc_a: float
    acc_b: float
    diff: float
    n_only_a: int          # a correct, b wrong
    n_only_b: int          # b correct, a wrong
    p_mcnemar: float
    p_bootstrap: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_adj: float | None = None
    significant: bool | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


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


def paired_test(correct_a: Sequence[bool], correct_b: Sequence[bool], clusters: Sequence | None = None,
                name: str = "", bootstrap: bool = True, n_boot: int = 4000, seed: int = 0) -> PairedTest:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must be aligned")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    p_mc = mcnemar_exact(only_a, only_b)
    p_bs = ci_lo = ci_hi = None
    if bootstrap:
        cl = clusters if clusters is not None else np.arange(len(a))
        p_bs = paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed)
        ci = paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed)
        ci_lo, ci_hi = ci.lo, ci.hi
    return PairedTest(name=name, n=len(a), acc_a=float(a.mean()), acc_b=float(b.mean()), diff=float(b.mean() - a.mean()),
                      n_only_a=only_a, n_only_b=only_b, p_mcnemar=p_mc, p_bootstrap=p_bs, ci_lo=ci_lo, ci_hi=ci_hi)


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


def correct_family(tests: Sequence[PairedTest], alpha: float = 0.05, use: str = "mcnemar") -> list[PairedTest]:
    """Apply Holm within one family, in place, using the McNemar or the bootstrap p-values."""
    ps = [t.p_mcnemar if use == "mcnemar" else (t.p_bootstrap if t.p_bootstrap is not None else t.p_mcnemar) for t in tests]
    adj, rej = holm(ps, alpha)
    for t, a, r in zip(tests, adj, rej):
        t.p_adj = float(a)
        t.significant = bool(r)
    return list(tests)


def families_by(records: Sequence[dict], family_keys: Sequence[str]) -> dict[tuple, list[dict]]:
    """Group comparison records into families by the given keys (e.g. ('model',) for the
    'arms vs baseline within a model' family)."""
    out: dict[tuple, list[dict]] = {}
    for r in records:
        out.setdefault(tuple(r[k] for k in family_keys), []).append(r)
    return out
