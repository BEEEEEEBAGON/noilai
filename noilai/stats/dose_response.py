"""Δtokens dose–response and zero-dose contrast for a re-encoding arm (design 8.6, items 2–3).

"Share of the intervention effect mediated by token count" is NOT estimated (design 8.6,
12.7): within one tokenizer the arm determines the token sequence deterministically, token
count is a coarsening of it, and no mediation estimand exists. This module holds the two
pre-registered replacement quantities, reported under these labels only:

(2) Effect modification by Δk — ASSOCIATIONAL, descriptive. For a model and an arm, with
    d_i = y_i(arm) − y_i(base) the item-level change in correctness and Δtok_i the change in
    the item's input token count under the arm, the linear projection d_i = α + β Δtok_i + ε_i
    with a base-pair cluster bootstrap. α is the effect among items whose token count does
    not change, β the slope per extra token, β·mean(Δtok) the token-associated part of the
    average effect (a descriptive quantity, not a share of anything). Items whose features
    predict both Δtok and d (rare rimes, many diacritics) confound β; the paper says
    "the effect grows with the number of diacritics" and nothing causal.
(3) Zero-dose contrast — the average effect among items with Δtok = 0 versus items with
    Δtok > 0, with clustered CIs, reported only where the Δtok = 0 stratum is large enough
    (design 8.6: ≥ 100 items per model; `MIN_STRATUM` here is the computational floor).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bootstrap import CI, _cluster_index, _resample_weights, cluster_bootstrap

MIN_STRATUM = 5


@dataclass
class Decomposition:
    total_effect: CI               # τ_m(arm): mean d
    form_only_effect: CI           # α: effect at Δtok = 0 (projection)
    slope_per_token: CI            # β
    token_associated_part: CI      # β · mean(Δtok); descriptive, not a share
    effect_delta_zero: CI | None   # zero-dose contrast: items with Δtok == 0
    effect_delta_pos: CI | None    # items with Δtok > 0
    n_delta_zero: int
    n_delta_pos: int
    label: str = "dose-response (associational); zero-dose contrast — design 8.6 items 2-3"

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

    def stats_for(w: np.ndarray) -> tuple[float, float, float, float]:
        total = np.average(d, weights=w)
        a, b = _wls(dt, d, w)
        part = b * np.average(dt, weights=w)
        return total, a, b, part

    obs = stats_for(np.ones_like(d))
    boots = np.empty((n_boot, 4))
    for i in range(n_boot):
        w = _resample_weights(inv, k, rng)
        boots[i] = stats_for(w)
    alpha = (1 - level) / 2

    def ci(j: int) -> CI:
        lo, hi = np.nanquantile(boots[:, j], [alpha, 1 - alpha])
        return CI(float(obs[j]), float(lo), float(hi), len(d), k, level, n_boot)

    zero = dt == 0
    pos = dt > 0
    cl = np.asarray(clusters)
    e0 = cluster_bootstrap(d[zero], cl[zero], n_boot=n_boot, seed=seed, bca=False, small_cell_rule=False) if zero.sum() >= MIN_STRATUM else None
    e1 = cluster_bootstrap(d[pos], cl[pos], n_boot=n_boot, seed=seed, bca=False, small_cell_rule=False) if pos.sum() >= MIN_STRATUM else None
    return Decomposition(total_effect=ci(0), form_only_effect=ci(1), slope_per_token=ci(2), token_associated_part=ci(3),
                         effect_delta_zero=e0, effect_delta_pos=e1, n_delta_zero=int(zero.sum()), n_delta_pos=int(pos.sum()))
