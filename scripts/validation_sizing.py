#!/usr/bin/env python
"""Size the native-validation sample by simulation (Gate 1 packet; DESIGN_DECISIONS 10.1 amendment of 1 Oct 2026).

Nothing here is a result about models or about the data: every number is a property of a
simulated design under stated assumptions. The script answers three questions for each
candidate design (items per task x variant cell, planted control items per cell, how many
validators, how many items all validators see):

  1. agreement: the expected Krippendorff's alpha and Gwet's AC1 on the binary `correct`
     judgment and the expected width of their 95% item-bootstrap intervals;
  2. generator precision: the Wilson 95% interval for one cell and for the pooled sample at
     an observed precision, and the probability that a rule bug affecting a share b of a
     cell's items shows up in at least one sampled item of that cell;
  3. workload: items and hours per validator at a stated time per item.

Assumptions (all stated in the output file): a generated item is truly wrong with probability
`gen_error`; a planted control item (a deliberately corrupted gold, DESIGN_DECISIONS 10.1
amendment) is always wrong; a validator says "no" to a correct item with probability
`false_alarm` and says "yes" to a wrong item with probability `miss`, independently across
validators and items. The statistics are the repository's own (`noilai.stats.agreement`); the
simulation uses a vectorized binary copy of the same formulas, checked against them on the
first replicate of every design.

    python scripts/validation_sizing.py --out data/audit/validation_sizing.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai import constants as C
from noilai.stats.agreement import gwet_ac1, krippendorff_alpha_nominal

N_CELLS = 12            # T1/T2/T3 x V1-V4


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def assignment(n_items: int, n_validators: int, overlap: int) -> np.ndarray:
    """Boolean matrix items x validators: the first `overlap` items go to every validator, the
    rest to two validators in rotating pairs (A-B, B-C, C-A, ...), the design of
    scripts/make_validation_forms.py sample."""
    m = np.zeros((n_items, n_validators), dtype=bool)
    if n_validators <= 2:
        m[:, :] = True
        return m
    m[:overlap, :] = True
    for j in range(overlap, n_items):
        r = (j - overlap) % n_validators
        m[j, r] = True
        m[j, (r + 1) % n_validators] = True
    return m


def binary_stats(yes: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """alpha and AC1 from per-unit counts of 'yes' labels (yes) out of n labels, units with n >= 2.
    Same formulas as noilai.stats.agreement for a two-category nominal judgment."""
    keep = n >= 2
    yes, n = yes[keep].astype(float), n[keep].astype(float)
    no = n - yes
    # alpha: coincidence matrix o_cd = sum_u count_c count_d / (m_u - 1), diagonal count_c (count_c - 1)/(m_u - 1)
    w = 1.0 / (n - 1)
    o_yy = np.sum(yes * (yes - 1) * w)
    o_nn = np.sum(no * (no - 1) * w)
    o_yn = np.sum(yes * no * w)
    n_y = o_yy + o_yn
    n_n = o_nn + o_yn
    tot = n_y + n_n
    d_o = 2 * o_yn
    d_e = (tot ** 2 - n_y ** 2 - n_n ** 2) / (tot - 1)
    alpha = 1.0 if d_e == 0 else 1 - d_o / d_e
    # AC1
    pa = np.mean((yes * (yes - 1) + no * (no - 1)) / (n * (n - 1)))
    pi_y = np.mean(yes / n)
    pe = 2 * pi_y * (1 - pi_y)          # sum_q pi_q (1 - pi_q) / (Q - 1) with Q = 2
    if pi_y in (0.0, 1.0):
        ac1 = 1.0 if pa == 1.0 else float("nan")
    else:
        ac1 = (pa - pe) / (1 - pe)
    return float(alpha), float(ac1)


def simulate(design: dict, assume: dict, reps: int, n_boot: int, rng: np.random.Generator, check: bool) -> dict:
    per_cell, controls, n_val, overlap = design["per_cell"], design["controls_per_cell"], design["validators"], design["overlap"]
    n_gen = per_cell * N_CELLS
    n_ctl = controls * N_CELLS
    n_items = n_gen + n_ctl
    assign = assignment(n_items, n_val, min(overlap, n_items))
    alphas, ac1s, a_w, c_w = [], [], [], []
    for rep in range(reps):
        truth = np.ones(n_items, dtype=bool)
        truth[:n_gen] = rng.random(n_gen) >= assume["gen_error"]
        truth[n_gen:] = False
        perm = rng.permutation(n_items)            # controls are spread over the sheet, not at its end
        truth = truth[perm]
        u = rng.random((n_items, n_val))
        says_yes = np.where(truth[:, None], u >= assume["false_alarm"], u < assume["miss"])
        says_yes &= assign
        yes = says_yes.sum(axis=1)
        n = assign.sum(axis=1)
        a, c = binary_stats(yes, n)
        if check and rep == 0:
            triples = [(i, v, "yes" if says_yes[i, v] else "no") for i in range(n_items) for v in range(n_val) if assign[i, v]]
            a_ref, c_ref = krippendorff_alpha_nominal(triples), gwet_ac1(triples)
            assert abs(a - a_ref) < 1e-9 and abs(c - c_ref) < 1e-9, (a, a_ref, c, c_ref)
        alphas.append(a)
        ac1s.append(c)
        boots_a, boots_c = [], []
        for _ in range(n_boot):
            pick = rng.integers(0, n_items, size=n_items)
            ba, bc = binary_stats(yes[pick], n[pick])
            boots_a.append(ba)
            boots_c.append(bc)
        lo, hi = np.nanquantile(boots_a, [0.025, 0.975])
        a_w.append(hi - lo)
        lo, hi = np.nanquantile(boots_c, [0.025, 0.975])
        c_w.append(hi - lo)
    per_val = assign.sum(axis=0)
    secs = assume["seconds_per_item"]
    return {
        **design,
        "n_items": int(n_items), "n_generated": int(n_gen), "n_controls": int(n_ctl),
        "items_per_validator": [int(x) for x in per_val],
        "hours_per_validator_generated_sheet": round(float(per_val.max()) * secs / 3600, 2),
        "alpha_mean": round(float(np.mean(alphas)), 3), "alpha_ci_width_mean": round(float(np.mean(a_w)), 3),
        "ac1_mean": round(float(np.mean(ac1s)), 3), "ac1_ci_width_mean": round(float(np.mean(c_w)), 3),
        "cell_wilson_at_100pct": [round(x, 3) for x in wilson(per_cell, per_cell)],
        "cell_wilson_at_97pct": [round(x, 3) for x in wilson(round(0.97 * per_cell), per_cell)],
        "pooled_wilson_at_99pct": [round(x, 3) for x in wilson(round(0.99 * n_gen), n_gen)],
        "p_detect_bug_in_cell": {str(b): round(1 - (1 - b) ** per_cell, 3) for b in (0.03, 0.05, 0.10, 0.20)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/audit/validation_sizing.json")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = {
        "expected": {"gen_error": 0.02, "false_alarm": 0.02, "miss": 0.10, "seconds_per_item": 35},
        "pessimistic": {"gen_error": 0.02, "false_alarm": 0.05, "miss": 0.25, "seconds_per_item": 40},
    }
    designs = []
    for per_cell in (20, 25, 30, 40, 84):
        for controls in (0, 3, 4):
            for n_val, overlap in ((2, 0), (3, 0), (3, 60), (3, 120)):
                designs.append({"per_cell": per_cell, "controls_per_cell": controls, "validators": n_val, "overlap": overlap})
    out = {"_note": "simulation of candidate designs; no data and no model output were used; see the module docstring",
           "seed": args.seed, "reps": args.reps, "n_boot": args.n_boot, "scenarios": scenarios,
           "chosen": {"per_cell": C.VALIDATION_PER_CELL, "controls_per_cell": C.VALIDATION_CONTROLS_PER_CELL,
                      "overlap": C.VALIDATION_OVERLAP},
           "results": {}}
    for name, assume in scenarios.items():
        out["results"][name] = [simulate(d, assume, args.reps, args.n_boot, rng, check=True) for d in designs]
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chosen = [r for r in out["results"]["expected"] if r["per_cell"] == C.VALIDATION_PER_CELL
              and r["controls_per_cell"] == C.VALIDATION_CONTROLS_PER_CELL and r["overlap"] == C.VALIDATION_OVERLAP]
    print(json.dumps({"written": str(path.relative_to(ROOT)), "chosen_expected": chosen}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
