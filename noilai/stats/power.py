"""Power and precision arithmetic for the pre-registration.

Normal approximations; the clustered structure inflates variance by the design effect
1 + (m - 1)·ICC, where m is the mean number of items per base pair and ICC the
within-pair correlation of correctness. The pre-registration reports the unclustered
figures (as the founding plan did) and the design-effect-adjusted ones side by side.
"""
from __future__ import annotations

from math import ceil, sqrt

from scipy import stats as sps


def ci_half_width(n: int, acc: float, level: float = 0.95, design_effect: float = 1.0) -> float:
    z = sps.norm.ppf(1 - (1 - level) / 2)
    return z * sqrt(acc * (1 - acc) * design_effect / n)


def n_for_half_width(half_width: float, acc: float = 0.5, level: float = 0.95, design_effect: float = 1.0) -> int:
    z = sps.norm.ppf(1 - (1 - level) / 2)
    return ceil(design_effect * acc * (1 - acc) * (z / half_width) ** 2)


def mcnemar_power(n: int, diff: float, discordant_rate: float, alpha: float = 0.05) -> float:
    """Power of McNemar's test for a paired difference `diff` in accuracy when a fraction
    `discordant_rate` of items are discordant. Under H1 the discordant items split as
    p10 = (r + diff)/2, p01 = (r - diff)/2 (Connor 1987 normal approximation)."""
    r = discordant_rate
    if abs(diff) > r:
        raise ValueError("|diff| cannot exceed the discordant rate")
    p10 = (r + diff) / 2
    p01 = (r - diff) / 2
    z_a = sps.norm.ppf(1 - alpha / 2)
    num = sqrt(n) * abs(p10 - p01) - z_a * sqrt(p10 + p01)
    den = sqrt(p10 + p01 - (p10 - p01) ** 2)
    if den <= 0:
        return 1.0
    return float(sps.norm.cdf(num / den))


def n_for_mcnemar_power(diff: float, discordant_rate: float, power: float = 0.8, alpha: float = 0.05) -> int:
    lo, hi = 10, 1_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, diff, discordant_rate, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def n_for_mcnemar_power_conditional(diff: float, discordant_rate: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Items per cell for a paired contrast, CONDITIONAL on the discordant count (Miettinen
    1968 / Connor 1987 conditional form): among discordant items the H1 share of "b right, a
    wrong" is q = (r + diff) / (2r); the discordant count needed is
        n_d = (z_a/2 + z_b sqrt(q(1 - q)))^2 / (q - 1/2)^2
    and n = n_d / r, rounded to the nearest integer. This is the formula behind the figures
    quoted in docs/DESIGN_DECISIONS.md 8.5 and docs/PREREGISTRATION.md §9 (100 / 145 / 227 /
    308 / 356 items for a 10-point contrast at 15 / 20 / 30 / 40 / 46 % discordance); the
    unconditional `n_for_mcnemar_power` gives 116 / 155 / 234 / 312 / 359 on the same inputs."""
    r = discordant_rate
    if not 0 < abs(diff) <= r:
        raise ValueError("0 < |diff| <= discordant rate required")
    q = (r + abs(diff)) / (2 * r)
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    n_d = (z_a * 0.5 + z_b * sqrt(q * (1 - q))) ** 2 / (q - 0.5) ** 2
    return round(n_d / r)


def design_effect(items_per_cluster: float, icc: float) -> float:
    return 1 + (items_per_cluster - 1) * icc


def power_table() -> list[dict]:
    """The figures quoted in the pre-registration. The items-per-cell rows carry both the
    conditional-on-discordant figure (the one the documents quote) and the unconditional one."""
    rows = []
    for n, acc in ((1000, 0.7), (1500, 0.7), (125, 0.7), (500, 0.5), (3000, 0.7)):
        rows.append({"quantity": "95% CI half-width (points)", "n": n, "acc": acc,
                     "unclustered": round(100 * ci_half_width(n, acc), 2),
                     "design_effect_1.5": round(100 * ci_half_width(n, acc, design_effect=1.5), 2)})
    for n, diff, r in ((1000, 0.05, 0.20), (1000, 0.03, 0.20), (125, 0.10, 0.30), (500, 0.05, 0.20)):
        rows.append({"quantity": "McNemar power", "n": n, "diff": diff, "discordant": r,
                     "power": round(mcnemar_power(n, diff, r), 3)})
    for r in (0.15, 0.20, 0.30, 0.40, 0.46):
        rows.append({"quantity": "n per cell for a 10-point paired contrast at 80% power", "discordant": r,
                     "n_conditional_on_discordant (quoted in DD 8.5 / PREREG 9)": n_for_mcnemar_power_conditional(0.10, r),
                     "n_unconditional": n_for_mcnemar_power(0.10, r)})
    rows.append({"quantity": "n per cell for a 5-point paired contrast at 80% power, 20% discordant",
                 "n_conditional_on_discordant": n_for_mcnemar_power_conditional(0.05, 0.20),
                 "n_unconditional": n_for_mcnemar_power(0.05, 0.20)})
    return rows
