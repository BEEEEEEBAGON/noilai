"""How much of a re-encoding effect is associated with the change in token count?

The founding plan asked for "the share of each intervention effect that runs through
token count". Within one tokenizer, re-encoding an item changes its token sequence
deterministically, so token count is a function of the intervention, not an independent
mediator, and a classical mediation decomposition is not identified without further
assumptions. We therefore report two well-defined quantities instead (pre-registered):

1. Dose–response decomposition. For a model and an arm, let d_i = y_i(arm) - y_i(base) be
   the item-level change in correctness and Δtok_i the change in the item's token count
   under the arm. Fit the linear projection d_i = α + β Δtok_i + ε_i with a clustered
   bootstrap. α is the effect among items whose token count does not change ("form-only"
   effect), β·mean(Δtok) is the part of the average effect that moves with token count,
   and the reported share is β·mean(Δtok) / mean(d). This is a descriptive decomposition
   under linearity; item features that predict both Δtok and d (e.g. rare rimes) can
   confound β, and the paper says so.
2. Matched contrast. The average effect among items with Δtok = 0 versus items with
   Δtok > 0, with clustered CIs. If the arm has an effect even when the token count does
   not change, the effect is not (only) a token-count effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .bootstrap import CI, _cluster_index, _resample_weights, cluster_bootstrap


@dataclass
class Decomposition:
    total_effect: CI
    form_only_effect: CI          # α: effect at Δtok = 0
    slope_per_token: CI           # β
    token_associated_part: CI     # β · mean(Δtok)
    share_token_associated: CI    # (β · mean Δtok) / mean(d); undefined when mean(d) ≈ 0
    effect_delta_zero: Optional[CI]   # matched contrast: items with Δtok == 0
    effect_delta_pos: Optional[CI]    # items with Δtok > 0
    n_delta_zero: int
    n_delta_pos: int

    def as_dict(self) -> dict:
        return {k: (v.as_dict() if isinstance(v, CI) else v) for k, v in self.__dict__.items()}


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares of y on [1, x]; returns (alpha, beta)."""
    W = w.astype(float)
    if W.sum() == 0:
        return np.nan, np.nan
    xm = np.average(x, weights=W)
    ym = np.average(y, weights=W)
    var = np.average((x - xm) ** 2, weights=W)
    if var <= 0:
        return ym, 0.0
    cov = np.average((x - xm) * (y - ym), weights=W)
    beta = cov / var
    return ym - beta * xm, beta


def decompose(correct_base: Sequence[bool], correct_arm: Sequence[bool], delta_tokens: Sequence[float],
              clusters: Sequence, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> Decomposition:
    yb = np.asarray(correct_base, dtype=float)
    ya = np.asarray(correct_arm, dtype=float)
    dt = np.asarray(delta_tokens, dtype=float)
    d = ya - yb
    if not (len(yb) == len(ya) == len(dt)):
        raise ValueError("aligned inputs required")
    inv, k = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    mean_dt = dt.mean()

    def stats_for(w: np.ndarray) -> tuple[float, float, float, float, float]:
        total = np.average(d, weights=w)
        a, b = _wls(dt, d, w)
        part = b * np.average(dt, weights=w)
        share = part / total if abs(total) > 1e-9 else np.nan
        return total, a, b, part, share

    obs = stats_for(np.ones_like(d))
    boots = np.empty((n_boot, 5))
    for i in range(n_boot):
        w = _resample_weights(inv, k, rng)
        boots[i] = stats_for(w)
    alpha = (1 - level) / 2

    def ci(j: int) -> CI:
        lo, hi = np.nanquantile(boots[:, j], [alpha, 1 - alpha])
        return CI(float(obs[j]), float(lo), float(hi), len(d), k, level, n_boot)

    zero = dt == 0
    pos = dt > 0
    e0 = cluster_bootstrap(d[zero], np.asarray(clusters)[zero], n_boot=n_boot, seed=seed) if zero.sum() >= 5 else None
    e1 = cluster_bootstrap(d[pos], np.asarray(clusters)[pos], n_boot=n_boot, seed=seed) if pos.sum() >= 5 else None
    return Decomposition(total_effect=ci(0), form_only_effect=ci(1), slope_per_token=ci(2), token_associated_part=ci(3),
                         share_token_associated=ci(4), effect_delta_zero=e0, effect_delta_pos=e1,
                         n_delta_zero=int(zero.sum()), n_delta_pos=int(pos.sum()))
