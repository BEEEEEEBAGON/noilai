"""Tables and figure data from scored runs.

Input: a DataFrame with one row per scored output (scores.jsonl of one or more runs,
concatenated) holding at least: model, task, variant, arm, prompt_id, item_id,
base_pair_id, correct, error_class. Output: pandas tables and LaTeX fragments written
into paper/tables/. No number is typed by hand anywhere in the paper: every table in
the manuscript is produced here from scores files whose manifests name the model runs.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .bootstrap import accuracy_ci
from .tests import PairedTest, correct_family, paired_test


def load_scores(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_json(p, lines=True) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def accuracy_table(df: pd.DataFrame, by: Sequence[str] = ("model", "task"), arm: str = "nfc",
                   n_boot: int = 2000) -> pd.DataFrame:
    d = df[df["arm"] == arm] if "arm" in df else df
    rows = []
    for keys, g in d.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ci = accuracy_ci(g["correct"].astype(bool).values, g["base_pair_id"].values, n_boot=n_boot)
        rows.append({**dict(zip(by, keys)), "acc": ci.estimate, "lo": ci.lo, "hi": ci.hi, "n": ci.n_items, "clusters": ci.n_clusters})
    return pd.DataFrame(rows)


def intervention_table(df: pd.DataFrame, baseline_arm: str = "nfc", arms: Sequence[str] = ("nfd", "placement_old", "strip_tones"),
                       by: Sequence[str] = ("model", "task"), n_boot: int = 4000) -> pd.DataFrame:
    """Paired effect of every arm vs the baseline, Holm-corrected within each `by` group
    (the family = all arms for one model × task)."""
    rows = []
    key_cols = ["item_id", "prompt_id"]
    for keys, g in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = g[g["arm"] == baseline_arm].set_index(key_cols)
        tests: list[PairedTest] = []
        for arm in arms:
            other = g[g["arm"] == arm].set_index(key_cols)
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            a = base.loc[common, "correct"].astype(bool).values
            b = other.loc[common, "correct"].astype(bool).values
            cl = base.loc[common, "base_pair_id"].values
            tests.append(paired_test(a, b, cl, name=arm, n_boot=n_boot))
        correct_family(tests)
        for t in tests:
            rows.append({**dict(zip(by, keys)), "arm": t.name, "acc_base": t.acc_a, "acc_arm": t.acc_b, "effect": t.diff,
                         "ci_lo": t.ci_lo, "ci_hi": t.ci_hi, "p_mcnemar": t.p_mcnemar, "p_holm": t.p_adj,
                         "significant": t.significant, "n": t.n})
    return pd.DataFrame(rows)


def error_breakdown(df: pd.DataFrame, by: Sequence[str] = ("model", "task")) -> pd.DataFrame:
    d = df.copy()
    tab = d.groupby(list(by) + ["error_class"]).size().unstack(fill_value=0)
    return tab.div(tab.sum(axis=1), axis=0)


def to_latex_accuracy(tab: pd.DataFrame, caption: str, label: str, col: str = "task") -> str:
    """Model rows × task columns, 'acc [lo, hi]' cells, in percent."""
    piv = tab.pivot(index="model", columns=col, values=["acc", "lo", "hi"])
    cols = sorted({c for _, c in piv.columns})
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Model & " + " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    for model in piv.index:
        cells = []
        for c in cols:
            a, lo, hi = piv.loc[model, ("acc", c)], piv.loc[model, ("lo", c)], piv.loc[model, ("hi", c)]
            cells.append("--" if pd.isna(a) else f"{100*a:.1f} {{\\scriptsize[{100*lo:.1f}, {100*hi:.1f}]}}")
        lines.append(f"{_tex(model)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def to_latex_interventions(tab: pd.DataFrame, caption: str, label: str, normalizes: dict | None = None) -> str:
    arms = list(dict.fromkeys(tab["arm"]))
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{ll" + "c" * len(arms) + ("c" if normalizes else "") + "}", "\\toprule",
             "Model & Task & " + " & ".join(_tex(a) for a in arms) + (" & Normalizes NFD" if normalizes else "") + " \\\\", "\\midrule"]
    for (model, task), g in tab.groupby(["model", "task"], sort=True):
        cells = []
        for a in arms:
            r = g[g["arm"] == a]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            star = "$^{*}$" if r["significant"] else ""
            cells.append(f"{100*r['effect']:+.1f}{star} {{\\scriptsize[{100*r['ci_lo']:+.1f}, {100*r['ci_hi']:+.1f}]}}")
        extra = f" & {'yes' if normalizes.get(model) else 'no'}" if normalizes else ""
        lines.append(f"{_tex(model)} & {task} & " + " & ".join(cells) + extra + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{caption} Effects are accuracy differences (points) against the NFC baseline on the same items, with 95\\% clustered-bootstrap intervals; $^{{*}}$ Holm-adjusted McNemar $p<0.05$ within each model $\\times$ task family.}}",
              f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def _tex(s) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_table(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
