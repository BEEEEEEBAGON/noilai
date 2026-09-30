"""Tables and figure data from scored runs.

Input: a DataFrame with one row per scored output (scores.jsonl of one or more runs,
concatenated) holding at least: model, task, variant, arm, prompt_id, item_id,
base_pair_id, correct, error_class. Output: pandas tables and LaTeX fragments written
into paper/tables/. No number is typed by hand anywhere in the paper: every table in
the manuscript is produced here from scores files whose manifests name the model runs.

Arm names are the runner's (noilai.vi.reencode: nfc, nfd, win1258, placement_new,
strip_tones, strip_all); the docs/DATA_FORMAT.md names of constants.ARMS map onto them
through noilai.eval.prompts.ARM_ALIASES (base -> nfc, pc -> win1258).

Table 3 (design 8.3, item 65): ONE Holm family per model row, enumerated exactly by
constants.HOLM_FAMILY_TABLE3 (14 cells: {nfd, pc, strip_tones, strip_all} x {T1, T3, XCOPA}
on the main sample plus {placement_new} x {T1, T3} on the C2-enriched file), minus the
cells undefined for that model; the realized family size is written into every row and
into the table footer. The p-value corrected is the primary one of design 8.2 (paired t on
base-pair mean differences from PAIRED_T_MIN_BASE_PAIRS base pairs, else the clustered
bootstrap p); McNemar is carried as the labelled secondary column.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .. import constants
from .bootstrap import accuracy_ci
from .tests import PairedTest, correct_family, paired_test

_ARM_ALIASES_FALLBACK = {"base": "nfc", "pc": "win1258"}


def runner_arm_name(arm: str) -> str:
    """docs/DATA_FORMAT.md arm name -> the runner's (noilai.vi.reencode) name."""
    try:
        from ..eval.prompts import ARM_ALIASES
    except ImportError:  # pragma: no cover - the eval package needs yaml templates
        ARM_ALIASES = _ARM_ALIASES_FALLBACK
    return ARM_ALIASES.get(arm, arm)


def table3_family(cells: Sequence[tuple[str, str, str]] = constants.HOLM_FAMILY_TABLE3) -> list[tuple[str, str, str | None]]:
    """The Table 3 family as (runner arm name, task, item file) cells."""
    return [(runner_arm_name(arm), task, file) for arm, task, file in cells]


DEFAULT_ARMS: tuple[str, ...] = tuple(_ARM_ALIASES_FALLBACK.get(a, a) for a in constants.ARMS if a != "base")


def load_scores(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_json(p, lines=True) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def accuracy_table(df: pd.DataFrame, by: Sequence[str] = ("model", "task"), arm: str = "nfc",
                   n_boot: int = constants.BOOTSTRAP_B, strata_col: str | None = None) -> pd.DataFrame:
    d = df[df["arm"] == arm] if "arm" in df else df
    strata_col = strata_col if strata_col is not None else ("source" if "source" in d else None)
    rows = []
    for keys, g in d.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ci = accuracy_ci(g["correct"].astype(bool).values, g["base_pair_id"].values, n_boot=n_boot,
                         strata=g[strata_col].values if strata_col else None)
        rows.append({**dict(zip(by, keys)), "acc": ci.estimate, "lo": ci.lo, "hi": ci.hi, "n": ci.n_items,
                     "clusters": ci.n_clusters, "ci_method": ci.method, "small_cell": ci.small_cell})
    return pd.DataFrame(rows)


def _cell_rows(g: pd.DataFrame, task: str, file: str | None) -> pd.DataFrame:
    sub = g[g["task"] == task] if "task" in g else g
    if file is not None and "c2_enriched" in sub:
        flag = sub["c2_enriched"].fillna(False).astype(bool)
        sub = sub[flag] if file == "c2" else sub[~flag]
    return sub


def intervention_table(df: pd.DataFrame, baseline_arm: str = "nfc", family: Sequence[tuple[str, str, str | None]] | None = None,
                       arms: Sequence[str] | None = None, n_boot: int = 4000, use: str = "primary",
                       strata_col: str | None = None) -> pd.DataFrame:
    """Paired effect of every arm vs the baseline, Holm-corrected within ONE family per model
    row. `family` is a sequence of (arm, task, file) cells; the default is the Table 3 family
    of constants.HOLM_FAMILY_TABLE3 (design 8.3), and an analysis config may pass its own.
    `arms` (legacy) builds the family as arms x every task present. Cells with no paired rows
    for a model are undefined for it and leave the family. `file` is 'main' / 'c2' / None and
    is applied through the `c2_enriched` column when the scores carry it."""
    if family is None:
        if arms is not None:
            tasks = sorted(df["task"].unique()) if "task" in df else [None]
            family = [(a, t, None) for a in arms for t in tasks]
        else:
            family = table3_family()
    strata_col = strata_col if strata_col is not None else ("source" if "source" in df else None)
    rows = []
    key_cols = ["item_id", "prompt_id"]
    for model, g in df.groupby("model", sort=True):
        tests: list[PairedTest] = []
        meta: list[tuple] = []
        for arm, task, file in family:
            cell = _cell_rows(g, task, file) if task is not None else g
            base = cell[cell["arm"] == baseline_arm].set_index(key_cols)
            other = cell[cell["arm"] == arm].set_index(key_cols)
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            a = base.loc[common, "correct"].astype(bool).values
            b = other.loc[common, "correct"].astype(bool).values
            cl = base.loc[common, "base_pair_id"].values
            st = base.loc[common, strata_col].values if strata_col else None
            tests.append(paired_test(a, b, cl, name=arm, n_boot=n_boot, strata=st))
            meta.append((task, file, len(set(cl))))
        correct_family(tests, use=use)
        for t, (task, file, n_bp) in zip(tests, meta):
            rows.append({"model": model, "task": task, "arm": t.name, "item_file": file, "acc_base": t.acc_a, "acc_arm": t.acc_b,
                         "effect": t.diff, "ci_lo": t.ci_lo, "ci_hi": t.ci_hi, "ci_method": t.ci_method,
                         "small_cell": t.small_cell, "p_primary": t.p_primary, "p_source": t.p_primary_source,
                         "p_paired_t": t.p_paired_t, "p_bootstrap": t.p_bootstrap, "p_mcnemar": t.p_mcnemar,
                         "p_holm": t.p_adj, "significant": t.significant, "n": t.n, "n_base_pairs": n_bp,
                         "family_size": len(tests), "family_defined": len(family)})
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
    fam = tab.groupby("model")["family_size"].first() if "family_size" in tab else None
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{ll" + "c" * len(arms) + ("c" if normalizes else "") + "c}", "\\toprule",
             "Model & Task & " + " & ".join(_tex(a) for a in arms) + (" & Normalizes NFD" if normalizes else "")
             + " & Family \\\\", "\\midrule"]
    for (model, task), g in tab.groupby(["model", "task"], sort=True):
        cells = []
        for a in arms:
            r = g[g["arm"] == a]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            star = "$^{*}$" if r["significant"] else ""
            dagger = "$^{\\dagger}$" if bool(r.get("small_cell", False)) else ""
            cells.append(f"{100*r['effect']:+.1f}{star}{dagger} {{\\scriptsize[{100*r['ci_lo']:+.1f}, {100*r['ci_hi']:+.1f}]}}")
        extra = f" & {'yes' if normalizes.get(model) else 'no'}" if normalizes else ""
        size = f" & {int(fam[model])}" if fam is not None else " & --"
        lines.append(f"{_tex(model)} & {task} & " + " & ".join(cells) + extra + size + " \\\\")
    footer = (f"\\caption{{{caption} Effects are accuracy differences (points) against the NFC baseline on the same "
              "items, with 95\\% base-pair cluster-bootstrap intervals; $^{*}$ Holm-adjusted $p<0.05$ within each model "
              "row over its defined cells (\\emph{Family} = realized family size out of "
              f"{constants.HOLM_FAMILY_TABLE3_CELLS}); $p$ from the base-pair paired $t$ (cells with $\\geq$ "
              f"{constants.PAIRED_T_MIN_BASE_PAIRS} base pairs) or the clustered bootstrap; $^{{\\dagger}}$ small cell "
              f"(fewer than {constants.SMALL_CELL_MAX_BASE_PAIRS} base pairs). McNemar (ignores clustering) is in the appendix.}}")
    lines += ["\\bottomrule", "\\end{tabular}", footer, f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def _tex(s) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_table(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
