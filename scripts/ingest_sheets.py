#!/usr/bin/env python
"""Ingest returned human sheets: check them, score them, write the human lane's reports (DESIGN_DECISIONS 10.1-10.2).

    python scripts/ingest_sheets.py --items data/release/v0.3/noilai_test.jsonl data/release/v0.3/noilai_dev.jsonl \\
        --human-items data/human/human_items.json \\
        --validation-returned 'data/validation/returned/*.csv' --baseline-returned 'data/human/returned/*.csv'
    python scripts/ingest_sheets.py --baseline-returned 'data/human/returned/*.csv' --out experiments/human --n-boot 2000
    python scripts/ingest_sheets.py --template > data/human/returned/baseline_form_03.csv    # an empty long-format sheet
    make ingest-sheets                                                                      # both globs, the plan's release

Two kinds of sheet come back (both are CSV; Google Sheets / Google Forms exports are read as is, a UTF-8 BOM
is tolerated):

VALIDATION SHEETS (design 10.1; built by `make_validation_forms.py sample` under data/validation/): one file per
validator named `validation_form_<coder>.csv` (the coder letter is read from the file name, or from a `coder` /
`validator` column when the sheet has one), one row per item with the columns `item_id, task, variant, input,
candidate, question_vi, correct, spelling, lexical, offensive, comment`. A judgment is `yes` / `no` / `unsure`
(`có` / `không` accepted); a blank cell is "not rated". Checks: unknown item ids (the row is dropped), duplicate
ratings of an item by the same coder (the first is kept), labels outside the five accepted ones (the cell is
blanked), and the count of unrated cells per judgment. The cleaned sheets are then scored by
`make_validation_forms.cmd_score` (imported, not re-implemented: alpha with its bootstrap CI, raw agreement,
Gwet's AC1, the marginals and the generator precision; its own copy of the report goes to --validation-out,
data/validation/ by default) and the report is copied to <out>/validation_report.json with the coder list
(`coders`, read by `ledger.py ingest`) and the check results.

BASELINE SHEETS (design 10.2; the 20 forms built by `make_validation_forms.py baseline` under data/human/). Two
layouts are accepted and told apart by the header:
  long   columns `form_id, respondent, item_id, answer` (`--template` prints it): one row per answered item;
         `form_id` may be blank when the file name carries the form number (`baseline_form_03.csv`, any
         `form<nn>`); `respondent` is optional (a letter or a number; anything else, a name for instance, is
         replaced by the form number and reported). The respondent-level fields are rows whose `item_id` is
         `tool_use` (yes / no: "did you use a dictionary, a search engine or an AI assistant?"),
         `instruction_check` (the answer to the instruction-check item) and `region` (Bắc / Trung / Nam /
         ngoài Việt Nam); the same three may instead be columns repeated on every row. The sheet the builder
         wrote (`item_id, task, variant, prompt_vi, answer`), filled in, is a long sheet.
  wide   a Google Forms export: one row per respondent and one column per question, the item questions
         recognized by the item id in their header (`[T1-V1-000012] Nói lái kiểu V1 của ...`), the tool-use
         question by its header (`tool_use`, or wording that names a dictionary / search engine / AI
         assistant), the instruction check by a header containing --instruction-check-id, the region by a header
         naming it. The form number comes from a `form_id` column or the file name. A second response row on
         a form is a duplicate (see below).
Columns whose header names a person (e-mail, name, họ tên, phone, address) or whose values look like e-mail
addresses are DROPPED on reading and only their headers are listed in the report; an e-mail typed into an
answer cell is redacted before anything is written. Nothing here is ever written under paper/ and no
timestamp, name or address reaches <out> (CLAUDE.md: experiments/human/ holds scored sheets, no names or
e-mails).

Exclusions (PREREGISTRATION section 5 item 7 / DESIGN_DECISIONS 10.2 / docs/HUMAN_BASELINE_FORM.md section 4,
every count reported): `too_few_answers` (fewer than --min-answered of the form's items answered, 15 of 30 by
default), `tool_use` (the respondent reports a dictionary, a search engine or an AI assistant), `failed_
instruction_check` (the check answer is not --instruction-check-answer, or, when the check id is a release item,
not correct under the scorer; a missing check is counted, not excluded) and `duplicate_form_response` (a form
is claimed by its first included response; later responses on it are set aside).

Scoring: every kept answer becomes an output row `{item_id, arm: "base", prompt_id: "human-form-<nn>", raw}`
and goes through `noilai.eval.score.score_outputs`, the function that scores the models (HUMAN_BASELINE_FORM.md
section 4: the extractor reads the cell as it would a completion; strict T1 in the named order, T2 membership
in the gold set, T3 Có/Không; a blank answer is unparseable, hence wrong). Three correctness columns per row
(DESIGN_DECISIONS 10.2, item 32): `strict`, `lenient` (the gold syllables in the other order) and `tolerant`
(lenient, plus the `doublet` and `homophone` bins the scorer keeps apart, 5.4 items 8-9).

Outputs under <out> (experiments/human/ by default):
  baseline_scores.jsonl    one row per kept respondent x item: form, respondent, item_id, task, variant, answer
                           (as typed), extracted, strict, lenient, tolerant, error_class, region
  baseline_summary.json    mean-human accuracy (the mean over items of the item's mean judgment; PREREGISTRATION
                           8.11) with a two-way person x item bootstrap CI (persons and items resampled
                           independently, B = --n-boot, fixed seed), the any-human ceiling, overall / per task /
                           per task x variant and for the three correctness columns; the exclusion counts by
                           reason; the instruction-check and tool-use tallies; raters per item; nominal
                           Krippendorff's alpha between the raters on strict correctness (bootstrap CI) and on the
                           produced answer (point estimate) over the double-judged items, with raw agreement and
                           AC1 (noilai.stats.agreement); the error classes per task; respondents per region
  validation_report.json   cmd_score's report plus `coders`, the files and the checks
  ingest_report.json       every file with its layout, its problems and its dropped columns, and the exclusions
The exit status is 0 unless a file could not be read at all, or --fail-on-problems is given and a check failed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import glob
import importlib
import io
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # the package
sys.path.insert(0, str(ROOT / "scripts"))           # scripts/make_validation_forms.py (cmd_score scores the validation sheets)

from noilai import constants as C
from noilai.eval.score import score_outputs
from noilai.gen.generate import load_items
from noilai.stats.agreement import alpha_bootstrap_ci, gwet_ac1, krippendorff_alpha_nominal, percent_agreement
from noilai.vi.reencode import canonical_text

MVF = importlib.import_module("make_validation_forms")      # imported after the path insert, as kaggle_cpu_jobs.py does

OUT_DIR = ROOT / "experiments" / "human"                    # CLAUDE.md: scored sheets, no names or e-mails
VALIDATION_DIR = ROOT / "data" / "validation"               # git-ignored (DD 11.2); cmd_score's own copy of its report
PAPER_DIR = ROOT / "paper"                                  # CLAUDE.md: this workstream never writes under paper/
JUDGMENTS = MVF.JUDGMENTS
VALID_LABELS = ("yes", "no", "unsure", "có", "không")       # the labels cmd_score accepts (design 10.1)
# fewer than 15 of the 30 items answered excludes the form: PREREGISTRATION section 5 item 7, DD 10.2,
# docs/HUMAN_BASELINE_FORM.md section 4
MIN_ANSWERED = C.HUMAN_BASELINE_ITEMS_PER_PERSON // 2
INSTRUCTION_CHECK_ID = "instruction_check"
TOOL_USE_ID = "tool_use"
REGION_ID = "region"
RESPONDENT_ID = "respondent"
FORM_ID_COLUMNS = ("form_id", "form")
ITEM_ID_SHAPE = re.compile(r"T[123]-V[1-6]-\d+")            # the generated item ids (noilai.gen.generate); flags unknown ids
ID_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
# a respondent id is a letter or a number (an optional letter or two-character suffix after the number), never a name
SAFE_RESPONDENT = re.compile(r"^(?:[A-Za-z]|\d{1,4}[A-Za-z]?|\d{1,4}-[A-Za-z0-9]{1,2})$")
PERSONAL_HEADER = re.compile(r"e-?mail|họ\s*(?:và)?\s*tên|\btên\b|\bname\b|username|phone|điện thoại|\baddress\b|địa chỉ",
                             re.IGNORECASE)
EMAIL = re.compile(r"[^\s@,;<>()]+@[^\s@,;<>()]+\.[A-Za-z]{2,}")
TOOL_USE_HEADER = re.compile(r"tool[_ ]?use|từ điển|dictionary|search engine|công cụ tìm kiếm|trợ lý ai|ai assistant",
                             re.IGNORECASE)
REGION_HEADER = re.compile(r"\bregion\b|\bvùng\b|\bmiền\b", re.IGNORECASE)
# coarse region bands of docs/DATA_STATEMENT.md; "ngoài Việt Nam" is tested before "nam", which it contains
REGION_BANDS = (("ngoài", "ngoài Việt Nam"), ("ngoai", "ngoài Việt Nam"), ("abroad", "ngoài Việt Nam"),
                ("outside", "ngoài Việt Nam"), ("bắc", "Bắc"), ("bac", "Bắc"), ("north", "Bắc"), ("trung", "Trung"),
                ("central", "Trung"), ("nam", "Nam"), ("south", "Nam"))
YES_WORDS = frozenset({"yes", "có", "co", "y", "true", "1"})
NO_WORDS = frozenset({"no", "không", "khong", "n", "false", "0"})
REDACTED = "[redacted]"
ARM = "base"                                                # HUMAN_BASELINE_FORM.md section 4: arm "base", prompt "human-form-<nn>"
ERROR_SEVERITY = {"unreadable_file", "unrecognized_layout", "unknown_item_id", "duplicate_rating", "invalid_label",
                  "duplicate_item_column", "duplicate_answer", "unknown_item_column"}


# ------------------------------------------------------------------ small helpers
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header and body of a CSV (UTF-8, BOM tolerated); blank lines dropped; short rows padded."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        return [], []
    header = [h.strip() for h in rows[0]]
    body = [[c.strip() for c in r] + [""] * (len(header) - len(r)) for r in rows[1:]]
    return header, body


def form_id_from_name(path: Path) -> str:
    """`baseline_form_03.csv`, `Form 3 (Responses).csv` -> '03'; a stem without a number is the form id itself."""
    m = re.search(r"form[ _-]*(\d+)", path.stem, re.IGNORECASE) or re.search(r"(\d+)", path.stem)
    return f"{int(m.group(1)):02d}" if m else path.stem


def normalize_form_id(text: str, fallback: str) -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    m = re.search(r"(\d+)", t)
    return f"{int(m.group(1)):02d}" if m else t


def parse_yesno(text: str | None) -> bool | None:
    t = (text or "").strip().lower().rstrip(".!")
    if t in YES_WORDS:
        return True
    if t in NO_WORDS:
        return False
    return None


def region_band(text: str | None) -> str | None:
    t = (text or "").strip().lower()
    for key, band in REGION_BANDS:
        if key in t:
            return band
    return None


def column_index(header: list[str], *names: str) -> int | None:
    low = [h.lower() for h in header]
    for n in names:
        if n.lower() in low:
            return low.index(n.lower())
    return None


def personal_columns(header: list[str], body: list[list[str]]) -> list[int]:
    """Columns dropped on reading: a header naming a person, or values that look like e-mail addresses."""
    out = []
    for i, h in enumerate(header):
        if PERSONAL_HEADER.search(h) or any(EMAIL.search(r[i]) for r in body if i < len(r)):
            out.append(i)
    return out


def redact(text: str, problems: list[dict], where: str) -> str:
    if EMAIL.search(text):
        problems.append({"check": "email_in_cell_redacted", "where": where, "severity": "warning"})
        return REDACTED
    return text


def refuse_paper(path: Path) -> None:
    if PAPER_DIR in path.resolve().parents or path.resolve() == PAPER_DIR:
        raise SystemExit(f"refusing to write under paper/ ({path}); CLAUDE.md: this workstream never edits paper/")


# ------------------------------------------------------------------ baseline sheets
@dataclass
class Respondent:
    form: str
    respondent: str
    source: str
    answers: dict[str, str] = field(default_factory=dict)      # item_id -> the answer as typed ('' = blank)
    tool_use: bool | None = None
    instruction_check: str | None = None
    region: str | None = None
    unknown_items: list[str] = field(default_factory=list)

    @property
    def n_answered(self) -> int:
        return sum(1 for a in self.answers.values() if a.strip())


def safe_respondent(text: str, fallback: str, problems: list[dict], where: str) -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    if SAFE_RESPONDENT.match(t):
        return t
    problems.append({"check": "respondent_id_replaced", "where": where, "severity": "warning",
                     "detail": "a respondent id must be a letter or a number; replaced by the form number"})
    return fallback


def classify_headers(header: list[str], known_ids: set[str], check_id: str, tool_id: str) -> list[tuple[str, str | None]]:
    """Role of every wide-sheet column: item (payload = item id), unknown_item (an id-shaped token not in the
    release), check, tool, region, form, respondent, other."""
    roles: list[tuple[str, str | None]] = []
    for h in header:
        low = h.lower()
        tokens = ID_TOKEN.findall(h)
        if check_id.lower() in low or (check_id in known_ids and check_id in tokens):
            roles.append(("check", None))
        elif any(t in known_ids for t in tokens):
            roles.append(("item", max((t for t in tokens if t in known_ids), key=len)))
        elif any(ITEM_ID_SHAPE.fullmatch(t) for t in tokens):
            roles.append(("unknown_item", next(t for t in tokens if ITEM_ID_SHAPE.fullmatch(t))))
        elif low in FORM_ID_COLUMNS:
            roles.append(("form", None))
        elif low == RESPONDENT_ID:
            roles.append(("respondent", None))
        elif tool_id.lower() in low or TOOL_USE_HEADER.search(h):
            roles.append(("tool", None))
        elif low == REGION_ID or REGION_HEADER.search(h):
            roles.append(("region", None))
        else:
            roles.append(("other", None))
    return roles


def load_baseline_wide(path: Path, header: list[str], body: list[list[str]], known_ids: set[str], check_id: str,
                       tool_id: str, problems: list[dict]) -> tuple[list[Respondent], list[str]]:
    roles = classify_headers(header, known_ids, check_id, tool_id)
    dropped_idx = set(personal_columns(header, body))
    dropped = [header[i] for i in sorted(dropped_idx)]
    for i, (role, payload) in enumerate(roles):
        if role == "unknown_item" and i not in dropped_idx:
            problems.append({"check": "unknown_item_column", "where": f"{path.name}: column {header[i]!r}",
                             "severity": "error", "detail": f"{payload} is not in the release"})
    if not any(role == "item" for role, _ in roles):
        problems.append({"check": "unrecognized_layout", "where": path.name, "severity": "error",
                         "detail": "no `item_id` column and no column header carrying a release item id"})
        return [], dropped
    form_default = form_id_from_name(path)
    i_form = next((i for i, (r, _) in enumerate(roles) if r == "form"), None)
    i_resp = next((i for i, (r, _) in enumerate(roles) if r == "respondent"), None)
    out = []
    for k, row in enumerate(body, 1):
        form = normalize_form_id(row[i_form], form_default) if i_form is not None else form_default
        fallback = form if len(body) == 1 else f"{form}-{k}"
        where = f"{path.name}: response row {k}"
        resp = safe_respondent(row[i_resp], fallback, problems, where) if i_resp is not None else fallback
        R = Respondent(form, resp, path.name)
        for i, (role, payload) in enumerate(roles):
            if i in dropped_idx:
                continue
            v = row[i]
            if role == "item":
                if payload in R.answers:
                    problems.append({"check": "duplicate_item_column", "where": f"{where}, {payload}", "severity": "error",
                                     "detail": "the item appears in two columns; the first is kept"})
                    continue
                R.answers[payload] = redact(v, problems, f"{where}, {payload}")
            elif role == "check":
                R.instruction_check = redact(v, problems, where)
            elif role == "tool":
                R.tool_use = parse_yesno(v)
                if v and R.tool_use is None:
                    problems.append({"check": "tool_use_unparsed", "where": where, "severity": "warning", "detail": v[:40]})
            elif role == "region":
                R.region = region_band(v)
            elif role == "unknown_item":
                R.unknown_items.append(payload)
        out.append(R)
    return out, dropped


def load_baseline_long(path: Path, header: list[str], body: list[list[str]], known_ids: set[str], check_id: str,
                       tool_id: str, problems: list[dict]) -> tuple[list[Respondent], list[str]]:
    i_item = column_index(header, "item_id")
    i_ans = column_index(header, "answer")
    i_form = column_index(header, *FORM_ID_COLUMNS)
    i_resp = column_index(header, RESPONDENT_ID)
    i_tool = column_index(header, tool_id)
    i_check = column_index(header, check_id)
    i_region = column_index(header, REGION_ID)
    dropped_idx = set(personal_columns(header, body)) - {i_item, i_ans}
    dropped = [header[i] for i in sorted(dropped_idx)]
    form_default = form_id_from_name(path)
    groups: dict[tuple[str, str], Respondent] = {}
    resolved: dict[tuple[str, str], str] = {}                # (form, respondent cell) -> safe id, reported once
    for k, row in enumerate(body, 2):
        where = f"{path.name}: line {k}"
        form = normalize_form_id(row[i_form], form_default) if i_form is not None else form_default
        resp = form
        if i_resp is not None:
            if (form, row[i_resp]) not in resolved:
                resolved[(form, row[i_resp])] = safe_respondent(row[i_resp], form, problems, where)
            resp = resolved[(form, row[i_resp])]
        R = groups.setdefault((form, resp), Respondent(form, resp, path.name))
        item, ans = row[i_item], redact(row[i_ans], problems, where)
        low = item.lower()
        if low == tool_id.lower():
            R.tool_use = parse_yesno(ans)
            if ans and R.tool_use is None:
                problems.append({"check": "tool_use_unparsed", "where": where, "severity": "warning", "detail": ans[:40]})
            continue
        if low == check_id.lower() and check_id not in known_ids:
            R.instruction_check = ans
            continue
        if low == REGION_ID:
            R.region = region_band(ans)
            continue
        for idx, attr in ((i_tool, "tool_use"), (i_check, "instruction_check"), (i_region, "region")):
            if idx is not None and idx not in dropped_idx and row[idx] and getattr(R, attr) is None:
                v = row[idx]
                setattr(R, attr, parse_yesno(v) if attr == "tool_use" else (region_band(v) if attr == "region" else v))
        if item == check_id:                                 # the check is a release item: its row is the check, not a judgment
            R.instruction_check = ans
            continue
        if not item:
            continue
        if item not in known_ids:
            problems.append({"check": "unknown_item_id", "where": where, "severity": "error", "detail": item})
            R.unknown_items.append(item)
            continue
        if item in R.answers:
            problems.append({"check": "duplicate_answer", "where": where, "severity": "error",
                             "detail": f"{item} answered twice on form {form}; the first is kept"})
            continue
        R.answers[item] = ans
    return list(groups.values()), dropped


def load_baseline_file(path: Path, known_ids: set[str], check_id: str, tool_id: str) -> dict:
    """{'file', 'layout', 'respondents', 'problems', 'dropped_columns'} for one returned baseline sheet."""
    problems: list[dict] = []
    try:
        header, body = read_table(path)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        problems.append({"check": "unreadable_file", "where": path.name, "severity": "error", "detail": str(e)})
        return {"file": path.name, "layout": None, "respondents": [], "problems": problems, "dropped_columns": []}
    if column_index(header, "item_id") is not None and column_index(header, "answer") is not None:
        layout = "long"
        resps, dropped = load_baseline_long(path, header, body, known_ids, check_id, tool_id, problems)
    else:
        layout = "wide"
        resps, dropped = load_baseline_wide(path, header, body, known_ids, check_id, tool_id, problems)
    return {"file": path.name, "layout": layout, "respondents": resps, "problems": problems, "dropped_columns": dropped}


def check_passes(answer: str | None, expected: str | None, item: dict | None) -> bool | None:
    """None when the check cannot be judged (no answer recorded, or nothing to compare with)."""
    if answer is None:
        return None
    if expected is not None:
        if parse_yesno(expected) is not None:
            return parse_yesno(answer) == parse_yesno(expected)
        return canonical_text(answer) == canonical_text(expected)
    if item is not None:                                     # the check is a release item: the scorer decides
        return bool(score_outputs([item], [{"item_id": item["item_id"], "arm": ARM, "prompt_id": "human-check",
                                            "raw": answer or None}])[0]["correct"])
    return None


def apply_exclusions(resps: list[Respondent], min_answered: int, check_expected: str | None,
                     check_item: dict | None) -> tuple[list[Respondent], list[dict], dict]:
    """PREREGISTRATION section 5 item 7: the kept respondents, the excluded ones with their reasons, the tallies."""
    kept, excluded = [], []
    claimed: set[str] = set()
    tally = {"instruction_check": Counter(), "tool_use": Counter()}
    for R in resps:
        reasons = []
        if R.form in claimed:
            reasons.append("duplicate_form_response")
        if R.tool_use is True:
            reasons.append("tool_use")
        tally["tool_use"]["yes" if R.tool_use is True else ("no" if R.tool_use is False else "unreported")] += 1
        passed = check_passes(R.instruction_check, check_expected, check_item)
        if R.instruction_check is None:
            tally["instruction_check"]["missing"] += 1
        elif passed is None:
            tally["instruction_check"]["unjudged"] += 1       # answered, but no expected answer and no release item to score it
        elif passed:
            tally["instruction_check"]["passed"] += 1
        else:
            tally["instruction_check"]["failed"] += 1
            reasons.append("failed_instruction_check")
        if R.n_answered < min_answered:
            reasons.append("too_few_answers")
        if reasons:
            excluded.append({"form": R.form, "respondent": R.respondent, "reasons": reasons, "n_answered": R.n_answered,
                             "n_items": len(R.answers)})
        else:
            kept.append(R)
            claimed.add(R.form)
    return kept, excluded, {k: dict(v) for k, v in tally.items()}


def score_respondents(kept: list[Respondent], items_by_id: dict[str, dict]) -> list[dict]:
    """One row per kept respondent x item through the models' scorer (HUMAN_BASELINE_FORM.md section 4)."""
    outputs, owners = [], []
    for R in kept:
        for item_id, ans in R.answers.items():
            outputs.append({"item_id": item_id, "arm": ARM, "prompt_id": f"human-form-{R.form}", "raw": ans or None})
            owners.append(R)
    if not outputs:
        return []
    needed = {o["item_id"] for o in outputs}
    scored = score_outputs([items_by_id[i] for i in needed], outputs)
    rows = []
    for R, out, s in zip(owners, outputs, scored):
        strict, lenient = bool(s["correct"]), bool(s["correct_lenient"])
        # DD 10.2 (item 32): tolerant also accepts the ay/ây doublet and regional-homophone bins (5.4 items 8-9)
        tolerant = lenient or s["error_class"] in ("doublet", "homophone")
        rows.append({"form": R.form, "respondent": R.respondent, "item_id": s["item_id"], "task": s["task"],
                     "variant": s["variant"], "answer": out["raw"] or "", "extracted": s["answer"], "strict": strict,
                     "lenient": lenient, "tolerant": tolerant, "error_class": s["error_class"], "region": R.region})
    return rows


# ------------------------------------------------------------------ statistics
def _index(values: list) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def two_way_bootstrap(correct: list[bool], persons: list[str], items: list[str], n_boot: int, seed: int = 0,
                      level: float = 0.95, stat: str = "mean") -> dict:
    """Mean-human accuracy (stat 'mean': the mean over items of the item's mean judgment, PREREGISTRATION 8.11)
    or the any-human ceiling (stat 'any': the share of items some rater got right) with a two-way (person, item)
    bootstrap: persons and items are resampled with replacement independently, a judgment's weight is its
    person's draw count, an item's weight its own; percentile interval."""
    c = np.asarray(correct, dtype=float)
    p_inv, n_p = _index(persons)
    i_inv, n_i = _index(items)

    def statistic(wp: np.ndarray, wi: np.ndarray) -> float:
        w = wp[p_inv]
        num = np.bincount(i_inv, weights=w * c, minlength=n_i)
        den = np.bincount(i_inv, weights=w, minlength=n_i)
        ok = den > 0
        if not ok.any() or wi[ok].sum() == 0:
            return float("nan")
        vals = (num[ok] / den[ok]) if stat == "mean" else (num[ok] > 0).astype(float)
        return float(np.average(vals, weights=wi[ok]))

    est = statistic(np.ones(n_p), np.ones(n_i))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wp = np.bincount(rng.integers(0, n_p, size=n_p), minlength=n_p)
        wi = np.bincount(rng.integers(0, n_i, size=n_i), minlength=n_i)
        boots[b] = statistic(wp, wi)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a]) if n_boot else (float("nan"), float("nan"))
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "n_items": n_i, "n_persons": n_p, "n_judgments": len(c),
            "n_boot": n_boot, "level": level, "method": f"two_way_person_item_bootstrap_{stat}"}


def accuracy_cell(rows: list[dict], n_boot: int, seed: int) -> dict:
    persons = [r["respondent"] for r in rows]
    items = [r["item_id"] for r in rows]
    cell = {"n_judgments": len(rows), "n_items": len(set(items)), "n_persons": len(set(persons))}
    for key in ("strict", "lenient", "tolerant"):
        vals = [r[key] for r in rows]
        cell[key] = {"mean_human": two_way_bootstrap(vals, persons, items, n_boot, seed),
                     "any_human": two_way_bootstrap(vals, persons, items, n_boot, seed, stat="any")}
    return cell


def agreement_block(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Nominal alpha between the raters on strict correctness (item bootstrap CI) and on the produced answer
    (canonical text, blank = ''; point estimate only, its label set is large) over the double-judged items
    (exactly two raters), plus raw agreement and AC1; the same on correctness over every item with >= 2 raters."""
    per_item = Counter(r["item_id"] for r in rows)
    double = [r for r in rows if per_item[r["item_id"]] == 2]
    multi = [r for r in rows if per_item[r["item_id"]] >= 2]

    def on_correct(sub: list[dict]) -> dict | None:
        rs = [(r["item_id"], r["respondent"], "right" if r["strict"] else "wrong") for r in sub]
        if not rs:
            return None
        est, lo, hi = alpha_bootstrap_ci(rs, n_boot=n_boot, seed=seed)
        return {"alpha": est, "alpha_ci": [lo, hi], "percent_agreement": percent_agreement(rs), "ac1": gwet_ac1(rs),
                "n_items": len({r[0] for r in rs}), "n_ratings": len(rs)}

    def on_answer(sub: list[dict]) -> dict | None:
        rs = [(r["item_id"], r["respondent"], canonical_text(r["answer"]) if r["answer"] else "") for r in sub]
        if not rs:
            return None
        return {"alpha": krippendorff_alpha_nominal(rs), "percent_agreement": percent_agreement(rs),
                "n_items": len({r[0] for r in rs}), "n_ratings": len(rs)}

    return {"scope": "double_judged = items with exactly two raters (DD 10.2: the 240 double-judged items); "
                     "all_multi = every item with two or more raters (the anchors included)",
            "double_judged": {"correct_strict": on_correct(double), "answer": on_answer(double)},
            "all_multi": {"correct_strict": on_correct(multi)}}


def summarize_baseline(rows: list[dict], files: list[dict], excluded: list[dict], tally: dict, n_received: int,
                       design_ids: set[str] | None, n_boot: int, seed: int) -> dict:
    by_reason = Counter(reason for e in excluded for reason in e["reasons"])
    per_item = Counter(r["item_id"] for r in rows)
    summary = {
        "design": "DESIGN_DECISIONS 10.2 / PREREGISTRATION 8.11: mean-human accuracy (mean over items of the item's mean "
                  "judgment) with a two-way (person, item) bootstrap CI; any-human is a ceiling reported separately; "
                  "strict, lenient and doublet/homophone-tolerant all reported (item 32); no superhuman claim",
        "generated_utc": utc_now(), "n_boot": n_boot, "seed": seed,
        "files": [{"file": f["file"], "layout": f["layout"], "n_respondents": len(f["respondents"])} for f in files],
        "n_respondents_received": n_received, "n_respondents_included": len({r["respondent"] + "/" + r["form"] for r in rows}),
        "forms_included": sorted({r["form"] for r in rows}),
        "excluded": {"n": len(excluded), "by_reason": dict(sorted(by_reason.items())), "respondents": excluded},
        "instruction_check": tally["instruction_check"], "tool_use": tally["tool_use"],
        "items": {"n_items": len(per_item), "n_judgments": len(rows),
                  "raters_per_item": {str(k): v for k, v in sorted(Counter(per_item.values()).items())},
                  "n_double_judged": sum(1 for v in per_item.values() if v == 2),
                  "n_single_rated": sum(1 for v in per_item.values() if v == 1),
                  "not_in_human_design": sorted(i for i in per_item if design_ids is not None and i not in design_ids),
                  "design_items_unanswered": (len(design_ids - set(per_item)) if design_ids is not None else None)},
        "accuracy": {}, "agreement": None, "error_classes": {}, "by_region": {},
    }
    if not rows:
        return summary
    acc = {"overall": accuracy_cell(rows, n_boot, seed), "by_task": {}, "by_task_variant": {}}
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        acc["by_task"][task] = accuracy_cell(sub, n_boot, seed)
        for v in sorted({r["variant"] for r in sub}):
            acc["by_task_variant"][f"{task}-{v}"] = accuracy_cell([r for r in sub if r["variant"] == v], n_boot, seed)
        summary["error_classes"][task] = dict(sorted(Counter(r["error_class"] for r in sub).items()))
    summary["accuracy"] = acc
    summary["agreement"] = agreement_block(rows, n_boot, seed)
    for region in sorted({r["region"] or "unreported" for r in rows}):
        sub = [r for r in rows if (r["region"] or "unreported") == region]
        summary["by_region"][region] = {"n_respondents": len({(r["form"], r["respondent"]) for r in sub}),
                                        "n_judgments": len(sub), "strict": float(np.mean([r["strict"] for r in sub]))}
    return summary


# ------------------------------------------------------------------ validation sheets
def load_validation_file(path: Path, known_ids: set[str]) -> dict:
    """{'file', 'rows': {coder: [clean rows]}, 'problems', 'dropped_columns', 'stats'} for one returned sheet."""
    problems: list[dict] = []
    empty = {"file": path.name, "rows": {}, "problems": problems, "dropped_columns": [], "stats": {}}
    try:
        header, body = read_table(path)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        problems.append({"check": "unreadable_file", "where": path.name, "severity": "error", "detail": str(e)})
        return empty
    i_item = column_index(header, "item_id")
    if i_item is None or column_index(header, "correct") is None:
        problems.append({"check": "unrecognized_layout", "where": path.name, "severity": "error",
                         "detail": "a validation sheet has an `item_id` column and the judgment columns"})
        return empty
    i_coder = column_index(header, "coder", "validator")
    coder_default = path.stem.split("_")[-1]                 # cmd_score's convention: validation_form_<coder>.csv
    idx = {j: column_index(header, j) for j in JUDGMENTS}
    for j, i in idx.items():
        if i is None:
            problems.append({"check": "missing_judgment_column", "where": path.name, "severity": "warning", "detail": j})
    i_task, i_variant = column_index(header, "task"), column_index(header, "variant")
    dropped_idx = set(personal_columns(header, body)) - {i_item}
    rows: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    rated: Counter = Counter()
    unrated: Counter = Counter()
    for k, row in enumerate(body, 2):
        where = f"{path.name}: line {k}"
        coder = (row[i_coder] if i_coder is not None and row[i_coder] else coder_default)
        item = row[i_item]
        if not item:
            continue
        if item not in known_ids:
            problems.append({"check": "unknown_item_id", "where": where, "severity": "error", "detail": item})
            continue
        if (coder, item) in seen:
            problems.append({"check": "duplicate_rating", "where": where, "severity": "error",
                             "detail": f"coder {coder} rated {item} twice; the first is kept"})
            continue
        seen.add((coder, item))
        clean = {"item_id": item, "task": row[i_task] if i_task is not None else "",
                 "variant": row[i_variant] if i_variant is not None else ""}
        for j, i in idx.items():
            lab = row[i].lower() if i is not None else ""
            if lab and lab not in VALID_LABELS:
                problems.append({"check": "invalid_label", "where": f"{where}, {j}", "severity": "error",
                                 "detail": f"{lab[:40]!r} is not yes / no / unsure / có / không; the cell is blanked"})
                lab = ""
            (rated if lab else unrated)[j] += 1
            clean[j] = lab
        rows[coder].append(clean)
    return {"file": path.name, "rows": dict(rows), "problems": problems,
            "dropped_columns": [header[i] for i in sorted(dropped_idx)],
            "stats": {"coders": sorted(rows), "n_rows": sum(len(v) for v in rows.values()),
                      "n_rated": dict(sorted(rated.items())), "n_unrated": dict(sorted(unrated.items()))}}


def score_validation(loaded: list[dict], validation_out: Path, out: Path) -> dict | None:
    """Clean sheets -> make_validation_forms.cmd_score (the existing scorer) -> <out>/validation_report.json."""
    by_coder: dict[str, list[dict]] = defaultdict(list)
    for f in loaded:
        for coder, rows in f["rows"].items():
            by_coder[coder].extend(rows)
    if not by_coder:
        return None
    validation_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ingest_sheets_") as tmp:
        paths = []
        for coder, rows in sorted(by_coder.items()):
            p = Path(tmp) / f"validation_form_{coder}.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["item_id", "task", "variant", *JUDGMENTS])
                w.writeheader()
                w.writerows(rows)
            paths.append(str(p))
        with contextlib.redirect_stdout(io.StringIO()):     # cmd_score prints its report; the copy below is the record
            MVF.cmd_score(argparse.Namespace(out=str(validation_out), returned=paths))
    report = json.loads((validation_out / "validation_report.json").read_text(encoding="utf-8"))
    report.update({"coders": sorted(by_coder), "n_ratings_per_coder": {c: len(r) for c, r in sorted(by_coder.items())},
                   "files": [{"file": f["file"], **f["stats"], "dropped_columns": f["dropped_columns"],
                              "n_problems": len(f["problems"])} for f in loaded],
                   "problems": [p for f in loaded for p in f["problems"]], "generated_utc": utc_now(),
                   "scored_by": "scripts/make_validation_forms.py cmd_score"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


# ------------------------------------------------------------------ driver
def expand(patterns: list[str]) -> list[Path]:
    return sorted({Path(p) for pat in patterns for p in glob.glob(pat)})


def print_template() -> None:
    """An empty long-format baseline sheet: the header and the three respondent-level rows."""
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["form_id", RESPONDENT_ID, "item_id", "answer"])
    for special in (TOOL_USE_ID, INSTRUCTION_CHECK_ID, REGION_ID):
        w.writerow(["", "", special, ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ingest returned validation and human-baseline sheets")
    ap.add_argument("--items", nargs="+", default=[str(ROOT / "data" / "release" / "v0.3" / "noilai_test.jsonl"),
                                                   str(ROOT / "data" / "release" / "v0.3" / "noilai_dev.jsonl")],
                    help="release item files (gold); the main sample is inside the test file")
    ap.add_argument("--human-items", default=None, help="data/human/human_items.json: the 246 ids of the design (answers to other ids are flagged)")
    ap.add_argument("--validation-returned", nargs="*", default=[], help="glob(s) of returned validation sheets")
    ap.add_argument("--baseline-returned", nargs="*", default=[], help="glob(s) of returned baseline sheets")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--validation-out", default=str(VALIDATION_DIR), help="where cmd_score writes its own report copy")
    ap.add_argument("--min-answered", type=int, default=MIN_ANSWERED, help=f"PREREGISTRATION 5.7: fewer answers excludes (default {MIN_ANSWERED})")
    ap.add_argument("--instruction-check-id", default=INSTRUCTION_CHECK_ID, help="item id / header key of the instruction-check item")
    ap.add_argument("--instruction-check-answer", default=None, help="its expected answer (else a release item is scored)")
    ap.add_argument("--tool-use-id", default=TOOL_USE_ID, help="item id / header key of the self-reported tool-use question")
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", action="store_true", help="print an empty long-format baseline sheet and exit")
    ap.add_argument("--fail-on-problems", action="store_true", help="exit 1 when any check failed")
    args = ap.parse_args(argv)
    if args.template:
        print_template()
        return 0
    out, validation_out = Path(args.out), Path(args.validation_out)
    refuse_paper(out)
    refuse_paper(validation_out)
    val_files, base_files = expand(args.validation_returned), expand(args.baseline_returned)
    if not val_files and not base_files:
        print("ingest_sheets: no returned sheet matches; nothing to ingest")
        return 0
    missing = [p for p in args.items if not Path(p).exists()]
    if missing:
        print(f"ingest_sheets: item file(s) not found: {missing}; pass --items", file=sys.stderr)
        return 1
    items_by_id = {it["item_id"]: it for p in args.items for it in load_items(Path(p))}
    known_ids = set(items_by_id)
    design_ids = None
    if args.human_items:
        if Path(args.human_items).exists():
            design_ids = set(json.loads(Path(args.human_items).read_text(encoding="utf-8")))
        else:
            print(f"ingest_sheets: --human-items {args.human_items} not found; design membership not checked", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    report = {"generated_utc": utc_now(), "items": args.items, "validation": [], "baseline": [], "exclusions": None}

    if val_files:
        loaded = [load_validation_file(p, known_ids) for p in val_files]
        report["validation"] = [{k: v for k, v in f.items() if k != "rows"} for f in loaded]
        vr = score_validation(loaded, validation_out, out)
        if vr is not None:
            print(f"validation: {len(val_files)} sheet(s), coders {vr['coders']}, "
                  f"{sum(vr['n_ratings_per_coder'].values())} rated rows -> {out / 'validation_report.json'}")

    if base_files:
        loaded = [load_baseline_file(p, known_ids, args.instruction_check_id, args.tool_use_id) for p in base_files]
        resps = [R for f in loaded for R in f["respondents"]]
        kept, excluded, tally = apply_exclusions(resps, args.min_answered, args.instruction_check_answer,
                                                 items_by_id.get(args.instruction_check_id))
        rows = score_respondents(kept, items_by_id)
        with open(out / "baseline_scores.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        summary = summarize_baseline(rows, loaded, excluded, tally, len(resps), design_ids, args.n_boot, args.seed)
        (out / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        report["baseline"] = [{k: v for k, v in f.items() if k != "respondents"} | {"n_respondents": len(f["respondents"])}
                              for f in loaded]
        report["exclusions"] = summary["excluded"]
        overall = summary["accuracy"].get("overall", {}).get("strict", {}).get("mean_human")
        print(f"baseline: {len(base_files)} sheet(s), {len(resps)} respondent(s), {len(kept)} kept, "
              f"{len(excluded)} excluded {summary['excluded']['by_reason']}, {len(rows)} scored rows -> "
              f"{out / 'baseline_scores.jsonl'}")
        if overall:
            print(f"mean-human strict accuracy {overall['estimate']:.3f} [{overall['lo']:.3f}, {overall['hi']:.3f}] "
                  f"({overall['n_items']} items, {overall['n_persons']} persons, two-way bootstrap B={overall['n_boot']})")

    problems = [p for kind in ("validation", "baseline") for f in report[kind] for p in f["problems"]]
    for p in problems:
        print(f"  {p['severity']}: {p['check']} at {p['where']}" + (f" ({p['detail']})" if p.get("detail") else ""))
    report["n_problems"] = {"error": sum(1 for p in problems if p["severity"] == "error"),
                            "warning": sum(1 for p in problems if p["severity"] == "warning")}
    (out / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "validation_files": len(val_files), "baseline_files": len(base_files),
                      "problems": report["n_problems"]}))
    if any(p["severity"] == "error" and p["check"] in ("unreadable_file", "unrecognized_layout") for p in problems):
        return 1
    return 1 if args.fail_on_problems and problems else 0


if __name__ == "__main__":
    sys.exit(main())
