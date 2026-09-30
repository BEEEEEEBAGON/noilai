"""Scoring: correctness, error taxonomy, aggregation, scores.jsonl.

T1  correct iff canonical_text(answer) == canonical_text(gold). canonical_text (from
    noilai.vi.reencode) normalizes encoding, tone-mark placement, case and the lí/lý
    alternation but deliberately does NOT repair misspellings, so "mài céo" is wrong.
    Error classes, decided in this order:
      unparseable    no answer extracted, not two syllables, or a syllable the parser rejects
      correct
      copy           the answer is the input
      spelling       the syllables parse (non-strict) to exactly the gold structures but at
                     least one violates a spelling rule (céo for kéo, nge for nghe)
      illegal        a syllable fails Inventory.is_legal(level='onset_rime')
      order          the gold syllables in the wrong order
      wrong_variant  the output of another variant on the same input (either order)
      component      anything else; `component_errors` names the wrong components per
                     syllable (onset / rime / tone, via noilai.gen.variants.diff)
    A spelling violation on an otherwise wrong structure is recorded in component_errors
    as 'spelling' while the class comes from the structure.
T2  correct iff the canonical answer is one of the gold readings. The variant the model
    implicitly used is identified from the structures (identify_any_order); a variant the
    model NAMES in its completion (V1..V4 or the Vietnamese/English variant names) is
    recorded separately.
T3  Có/Không (yes/no) mapped to yes/no. When the run recorded log-probabilities,
    forced-choice accuracy (P(Có) vs P(Không) for the same prompt) and BLiMP-style paired
    accuracy (log P(correct candidate) > log P(twin candidate) under the same context, over
    the yes/no pair sharing pair_item_id) are added.
XCOPA  generated 1/2 against the label; log-probability choice when recorded.

`score_outputs(items, outputs)` returns one row per output; `aggregate(rows)` the per-run
summary; `write_scores` the scores.jsonl of docs/DATA_FORMAT.md.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..gen import variants as V
from ..vi import unicode as U
from ..vi.reencode import canonical_text
from ..vi.syllable import Inventory, Syllable, spell, try_parse
from .extract import extract_answer, t3_label

ERROR_CLASSES = ("correct", "copy", "spelling", "illegal", "order", "wrong_variant", "component", "unparseable")
COMPONENTS = ("onset", "rime", "tone")
STRATA_KEYS = ("input_lexical", "output_lexical", "output_syllables_attested", "has_glide", "has_zero_onset",
               "has_stop_coda", "spelling_triggers", "tone_pair", "same_tone", "same_rime", "input_freq",
               "output_freq", "n_readings")

_VARIANT_TOKEN = re.compile(r"(?<![A-Za-z0-9])V([1-4])(?![0-9])")


def _syl(d: dict) -> Syllable:
    return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])


def _inventory(inv: Optional[Inventory]) -> Inventory:
    if inv is not None:
        return inv
    from ..vi import lexicon as L
    return L.load_inventory()


def parse_phrase(text: str) -> Optional[tuple[list[Syllable], list[bool]]]:
    """Two syllables (non-strict) and, per syllable, whether the STRICT parse also succeeds.
    None when the text is not two parseable syllables."""
    words = canonical_text(text).split()
    if len(words) != 2:
        return None
    syls, strict_ok = [], []
    for w in words:
        p = try_parse(w, strict=False)
        if p is None:
            return None
        syls.append(p.syllable)
        strict_ok.append(try_parse(w, strict=True) is not None and p.i_y_variant != "nonstandard")
    return syls, strict_ok


def _component_diff(ans: list[Syllable], gold: list[Syllable]) -> tuple[list[list[str]], dict[str, list[bool]]]:
    detail: list[list[str]] = []
    correct = {c: [] for c in COMPONENTS}
    for x, g in zip(ans, gold):
        d = V.diff(x, g)
        wrong = []
        if d.onset:
            wrong.append("onset")
        if d.rime:
            wrong.append("rime")
        if d.tone:
            wrong.append("tone")
        detail.append(wrong)
        for c in COMPONENTS:
            correct[c].append(c not in wrong)
    return detail, correct


def _base(item: dict, answer: Optional[str], method: str) -> dict:
    return {"item_id": item["item_id"], "task": item["task"], "variant": item["variant"], "answer": answer,
            "extraction_method": method, "correct": False, "error_class": "unparseable",
            "component_errors": [], "component_detail": None, "component_correct": None,
            "identified_variants": [], "named_variant": None}


# ------------------------------------------------------------------ T1
def score_t1(item: dict, answer: Optional[str], method: str = "marker", inv: Optional[Inventory] = None) -> dict:
    row = _base(item, answer, method)
    gold = item["gold"][0]
    row["gold"] = gold
    if answer is None:
        return row
    ca, cg = canonical_text(answer), canonical_text(gold)
    inp = [_syl(d) for d in item["input_syllables"]]
    gold_syls = [_syl(d) for d in item["gold_syllables"]] if item.get("gold_syllables") else list(
        V.apply(item["variant"], *inp))
    parsed = parse_phrase(answer)
    if parsed is not None:
        # per-component view of every parseable answer (correct and copy included), so that
        # per-component accuracy is defined over all structural answers
        row["component_detail"], row["component_correct"] = _component_diff(parsed[0], gold_syls)
    if ca == cg:
        row.update(correct=True, error_class="correct")
        return row
    if ca == canonical_text(item["input"]):
        row["error_class"] = "copy"
        return row
    if parsed is None:
        return row
    syls, strict_ok = parsed
    spelling_bad = not all(strict_ok)
    errs = sorted({c for wrong in row["component_detail"] for c in wrong}, key=COMPONENTS.index)
    if spelling_bad:
        errs.append("spelling")
    row["component_errors"] = errs
    if syls == gold_syls:
        row["error_class"] = "spelling"          # right structure, wrong spelling
        return row
    inv = _inventory(inv)
    if not all(inv.is_legal(s, "onset_rime") for s in syls):
        row["error_class"] = "illegal"
        return row
    if syls == gold_syls[::-1]:
        row["error_class"] = "order"
        row["component_errors"] = ["order"] + (["spelling"] if spelling_bad else [])
        return row
    ident = V.identify_any_order(inp[0], inp[1], syls[0], syls[1])
    row["identified_variants"] = [v for v, _ in ident]
    if any(v != item["variant"] for v, _ in ident):
        row["error_class"] = "wrong_variant"
        return row
    row["error_class"] = "component"
    return row


# ------------------------------------------------------------------ T2
def named_variant(raw: Optional[str], templates: Optional[dict] = None) -> Optional[str]:
    """A variant the model names in its completion: 'V3', or a variant name from the templates."""
    if not raw:
        return None
    m = _VARIANT_TOKEN.findall(raw)
    if m:
        return f"V{m[-1]}"
    low = U.nfc(raw).lower()
    if templates is None:
        from .prompts import load_templates
        templates = load_templates()
    hits = []
    for v in V.VARIANTS:
        spec = templates["variants"][v]
        for name in (spec["name_vi"], spec["name_en"]):
            k = low.rfind(name.lower())
            if k >= 0:
                hits.append((k, v))
    return max(hits)[1] if hits else None


def score_t2(item: dict, answer: Optional[str], method: str = "marker", raw: Optional[str] = None,
             inv: Optional[Inventory] = None) -> dict:
    row = _base(item, answer, method)
    golds = {canonical_text(g["output"]): g for g in item["gold"]}
    row["gold"] = [g["output"] for g in item["gold"]]
    row["gold_variants"] = sorted({g["variant"] for g in item["gold"]})
    row["named_variant"] = named_variant(raw)
    row["named_variant_correct"] = None
    if answer is None:
        return row
    ca = canonical_text(answer)
    inp = [_syl(d) for d in item["input_syllables"]]
    parsed = parse_phrase(answer)
    if ca in golds:
        g = golds[ca]
        row.update(correct=True, error_class="correct", identified_variants=[g["variant"]])
        if row["named_variant"] is not None:
            row["named_variant_correct"] = row["named_variant"] == g["variant"]
        gp = parse_phrase(g["output"])
        if parsed is not None and gp is not None:
            row["component_detail"], row["component_correct"] = _component_diff(parsed[0], gp[0])
        return row
    if ca == canonical_text(item["input"]):
        row["error_class"] = "copy"
        return row
    if parsed is None:
        return row
    syls, strict_ok = parsed
    spelling_bad = not all(strict_ok)
    ident = V.identify_any_order(inp[0], inp[1], syls[0], syls[1])
    row["identified_variants"] = [v for v, _ in ident]
    if row["named_variant"] is not None:
        row["named_variant_correct"] = row["named_variant"] in row["gold_variants"]
    # right structure of some gold reading but misspelled
    for g in item["gold"]:
        gp = parse_phrase(g["output"])
        if gp and gp[0] == syls:
            row["error_class"] = "spelling"
            row["component_errors"] = ["spelling"]
            return row
    if spelling_bad:
        row["component_errors"] = ["spelling"]
    inv = _inventory(inv)
    if not all(inv.is_legal(s, "onset_rime") for s in syls):
        row["error_class"] = "illegal"
        return row
    if ident:
        row["error_class"] = "wrong_variant"      # a legal nói lái of the input, not a lexical reading
        return row
    # closest gold reading for the component diff
    best = None
    for g in item["gold"]:
        gp = parse_phrase(g["output"])
        if not gp:
            continue
        detail, correct = _component_diff(syls, gp[0])
        n = sum(len(w) for w in detail)
        if best is None or n < best[0]:
            best = (n, detail, correct)
    if best:
        row["component_detail"], row["component_correct"] = best[1], best[2]
        row["component_errors"] = sorted({c for wrong in best[1] for c in wrong}, key=COMPONENTS.index) + (
            ["spelling"] if spelling_bad else [])
    row["error_class"] = "component"
    return row


# ------------------------------------------------------------------ T3
def score_t3(item: dict, answer: Optional[str], method: str = "marker", logprobs: Optional[dict] = None) -> dict:
    row = _base(item, answer, method)
    row["gold"] = item["gold"]
    pred = t3_label(answer)
    row["pred"] = pred
    if pred is None:
        row["error_class"] = "unparseable"
    else:
        row["correct"] = pred == item["gold"]
        row["error_class"] = "correct" if row["correct"] else "wrong"
    row["forced_choice_pred"] = None
    row["forced_choice_correct"] = None
    row["candidate_logprob"] = None
    if logprobs:
        lp_yes, lp_no = logprobs.get("Có"), logprobs.get("Không")
        if lp_yes is not None and lp_no is not None:
            row["forced_choice_pred"] = "yes" if lp_yes > lp_no else "no"
            row["forced_choice_correct"] = row["forced_choice_pred"] == item["gold"]
        row["candidate_logprob"] = logprobs.get("candidate")
    row["twin_type"] = item.get("twin_type")
    row["pair_item_id"] = item.get("pair_item_id")
    return row


def paired_t3(rows: list[dict]) -> list[dict]:
    """Add `paired_correct` (BLiMP-style: the correct candidate's string log-probability beats
    the twin's, same context, same arm and prompt) and `pair_both_correct` (both generated
    answers right) to every T3 row that has a partner in the same run."""
    by_key = {}
    for r in rows:
        if r["task"] == "T3":
            by_key[(r["item_id"], r.get("arm"), r.get("prompt_id"))] = r
    for r in rows:
        if r["task"] != "T3":
            continue
        r.setdefault("paired_correct", None)
        r.setdefault("pair_both_correct", None)
        mate = by_key.get((r.get("pair_item_id"), r.get("arm"), r.get("prompt_id")))
        if mate is None:
            continue
        r["pair_both_correct"] = bool(r["correct"] and mate["correct"])
        yes, no = (r, mate) if r["gold"] == "yes" else (mate, r)
        if yes.get("candidate_logprob") is not None and no.get("candidate_logprob") is not None:
            r["paired_correct"] = yes["candidate_logprob"] > no["candidate_logprob"]
    return rows


# ------------------------------------------------------------------ XCOPA
def score_xcopa(item: dict, answer: Optional[str], method: str = "marker", logprobs: Optional[dict] = None) -> dict:
    row = _base(item, answer, method)
    gold = str(item["gold"])
    row["gold"] = gold
    pred = answer if answer in ("1", "2") else None
    row["pred"] = pred
    if pred is None:
        row["error_class"] = "unparseable"
    else:
        row["correct"] = pred == gold
        row["error_class"] = "correct" if row["correct"] else "wrong"
    row["logprob_pred"] = None
    row["logprob_correct"] = None
    if logprobs and logprobs.get("1") is not None and logprobs.get("2") is not None:
        row["logprob_pred"] = "1" if logprobs["1"] > logprobs["2"] else "2"
        row["logprob_correct"] = row["logprob_pred"] == gold
    return row


# ------------------------------------------------------------------ dispatch
def score_output(item: dict, out: dict, inv: Optional[Inventory] = None) -> dict:
    """Score one outputs.jsonl row against its item. Re-extracts the answer from `raw` when
    the row has no `answer` key (older runs)."""
    task = item["task"]
    if "answer" in out and "extraction_method" in out:
        answer, method = out["answer"], out["extraction_method"]
    else:
        answer, method = extract_answer(out.get("raw"), task)
    if task == "T1":
        row = score_t1(item, answer, method, inv)
    elif task == "T2":
        row = score_t2(item, answer, method, out.get("raw"), inv)
    elif task == "T3":
        row = score_t3(item, answer, method, out.get("logprobs"))
    elif task == "XCOPA":
        row = score_xcopa(item, answer, method, out.get("logprobs"))
    else:
        raise ValueError(f"unknown task {task!r}")
    for k in ("arm", "prompt_id", "prompt_hash", "n_prompt_tokens", "n_output_tokens", "latency_s"):
        row[k] = out.get(k)
    row["raw_len"] = len(out.get("raw") or "")
    for k in ("base_pair_id", "source", "split", "in_core", "twin_type", "pair_item_id"):
        if k in item:
            row[k] = item[k]
    strata = item.get("strata") or {}
    for k in STRATA_KEYS:
        if k in strata:
            row[k] = strata[k]
    row["n_input_tokens_syll"] = None
    return row


def score_outputs(items: Iterable[dict], outputs: Iterable[dict], inv: Optional[Inventory] = None,
                  audit_rows: Optional[dict] = None) -> list[dict]:
    by_id = {it["item_id"]: it for it in items}
    rows = []
    for out in outputs:
        it = by_id.get(out["item_id"])
        if it is None:
            raise KeyError(f"output for unknown item {out['item_id']!r}")
        row = score_output(it, out, inv)
        if audit_rows is not None:
            row["n_input_tokens_syll"] = input_token_counts(it, out.get("arm") or "nfc", audit_rows)
        rows.append(row)
    return paired_t3(rows)


# ------------------------------------------------------------------ tokenizer audit join
def load_audit_rows(path: Path) -> dict[tuple[str, str], int]:
    """{(syllable, encoding): n_tokens} from a data/audit/<name>_rows.csv."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(U.nfc(r["syllable"]), r["encoding"])] = int(r["n_tokens"])
    return out


def input_token_counts(item: dict, arm: str, audit_rows: dict) -> Optional[list[Optional[int]]]:
    enc = "nfd" if arm == "nfd" else "nfc"
    if arm not in ("nfc", "nfd"):
        return None   # placement/strip arms change the spelling; the audit has no row for them
    return [audit_rows.get((U.nfc(w), enc)) for w in item["input"].split()]


# ------------------------------------------------------------------ aggregation
def _rate(rows: list[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def aggregate(rows: list[dict]) -> dict:
    """Per-run summary: accuracy per task x variant, per arm, per prompt; error-class
    distribution; copy and unparseable rates; T3 forced-choice and paired accuracy; XCOPA
    accuracy by arm."""
    def cell(sub: list[dict]) -> dict:
        d = {"n": len(sub), "accuracy": _rate(sub, "correct"),
             "error_classes": dict(Counter(r["error_class"] for r in sub)),
             "copy_rate": sum(1 for r in sub if r["error_class"] == "copy") / len(sub) if sub else None,
             "unparseable_rate": sum(1 for r in sub if r["error_class"] == "unparseable") / len(sub) if sub else None}
        comp = Counter(c for r in sub for c in (r.get("component_errors") or []))
        d["component_errors"] = dict(comp)
        cc = defaultdict(list)
        for r in sub:
            for c, vals in (r.get("component_correct") or {}).items():
                cc[c].extend(vals)
        d["component_accuracy"] = {c: sum(v) / len(v) for c, v in cc.items() if v}
        if any(r["task"] == "T3" for r in sub):
            d["forced_choice_accuracy"] = _rate(sub, "forced_choice_correct")
            d["paired_accuracy"] = _rate(sub, "paired_correct")
            d["pair_both_correct_rate"] = _rate(sub, "pair_both_correct")
        if any(r["task"] == "XCOPA" for r in sub):
            d["logprob_accuracy"] = _rate(sub, "logprob_correct")
        return d

    groups: dict[str, dict] = {"by_task": {}, "by_task_variant": {}, "by_task_arm": {}, "by_task_prompt": {},
                               "by_task_twin_type": {}}
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        groups["by_task"][task] = cell(sub)
        for v in sorted({r["variant"] for r in sub}):
            groups["by_task_variant"][f"{task}-{v}"] = cell([r for r in sub if r["variant"] == v])
        for a in sorted({str(r.get("arm")) for r in sub}):
            groups["by_task_arm"][f"{task}-{a}"] = cell([r for r in sub if str(r.get("arm")) == a])
        for p in sorted({str(r.get("prompt_id")) for r in sub}):
            groups["by_task_prompt"][f"{task}-{p}"] = cell([r for r in sub if str(r.get("prompt_id")) == p])
        if task == "T3":
            for tt in sorted({str(r.get("twin_type")) for r in sub}):
                groups["by_task_twin_type"][f"{task}-{tt}"] = cell([r for r in sub if str(r.get("twin_type")) == tt])
    groups["n_rows"] = len(rows)
    groups["overall"] = cell(rows) if rows else {"n": 0}
    return groups


def summary_table(agg: dict) -> str:
    """A plain-text table of accuracy per task x variant (and per arm)."""
    lines = [f"{'cell':<22}{'n':>6}{'acc':>8}{'copy':>7}{'unpars':>8}  error classes"]
    for key in ("by_task_variant", "by_task_arm"):
        for name, c in agg.get(key, {}).items():
            acc = f"{c['accuracy']:.3f}" if c["accuracy"] is not None else "-"
            copy = f"{c['copy_rate']:.2f}" if c["copy_rate"] is not None else "-"
            unp = f"{c['unparseable_rate']:.2f}" if c["unparseable_rate"] is not None else "-"
            extra = ""
            if "forced_choice_accuracy" in c and c["forced_choice_accuracy"] is not None:
                extra += f" fc={c['forced_choice_accuracy']:.3f}"
            if "paired_accuracy" in c and c["paired_accuracy"] is not None:
                extra += f" paired={c['paired_accuracy']:.3f}"
            if "logprob_accuracy" in c and c["logprob_accuracy"] is not None:
                extra += f" lp={c['logprob_accuracy']:.3f}"
            ec = " ".join(f"{k}={v}" for k, v in sorted(c["error_classes"].items()))
            lines.append(f"{name:<22}{c['n']:>6}{acc:>8}{copy:>7}{unp:>8}  {ec}{extra}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ------------------------------------------------------------------ files
def write_scores(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def spell_pair(syls: Iterable[Syllable]) -> str:
    return " ".join(spell(s) for s in syls)
