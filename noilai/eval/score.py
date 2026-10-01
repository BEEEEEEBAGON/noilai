"""Scoring: correctness, error taxonomy, aggregation, scores.jsonl (DESIGN_DECISIONS 5.1-5.4).

T1  `correct` (strict) iff canonical_text(answer) == canonical_text(gold) for one of the listed
    golds, in the named order; `correct_lenient` (secondary) also accepts the two syllables in
    the other order. canonical_text (noilai.vi.reencode) normalizes encoding, tone-mark
    placement, case and the lí/lý alternation but deliberately does NOT repair misspellings,
    so "mài céo" is wrong. Attested items (task "attested") are scored here; three-syllable
    attested phrases get only correct / copy / unparseable / wrong.
    Error classes (`error_class`, one per output), decided in this precedence (5.4):
      unparseable       no answer extracted, not two syllables, or a syllable the parser rejects
      correct           strict match
      lenient_only      unordered match (the gold syllables in the other order)
      copy              the answer is the input
      reversal          the answer is the plain reversal of the input
      wrong_variant     the output of another of the six variants on the same input (either
                        order; `wrong_variant_labels` records which)
      spelling          the syllables parse (non-strict) to exactly the gold structures but at
                        least one violates a spelling rule (céo for kéo, nge for nghe)
      homophone         differs from the gold only by a regional homophone spelling
                        (HOMOPHONE_ONSET_PAIRS / HOMOPHONE_STRING_PAIRS, <= 12 entries, NV)
      doublet           differs from the gold only by an ay/ây lexical doublet (DOUBLETS, NV)
      illegal           a syllable fails Inventory.is_legal(level='onset_rime')
      component         anything else; `component_errors` names the wrong components per
                        syllable (onset / rime / tone, via noilai.gen.variants.diff)
    A spelling violation on an otherwise wrong structure is recorded in component_errors as
    'spelling' while the class comes from the structure; a misspelled answer whose structure
    is the gold in the other order is therefore `wrong_variant` (the reverse variant's output,
    6 before 7) unless that variant is one of the item's own labels (then `spelling` with
    'order' listed). `placement_variant` records the tone-mark convention the answer uses
    (old / new / mixed / same; 5.4 item 7).
T2  correct iff the canonical answer is one of the gold readings IN EITHER ORDER (3.2: gold
    stores the attested order and accepts either; `gold_order` records which matched). The
    variants that produce the answer are identified from the structures (identify_any_order,
    all six) and mapped to the three unordered classes {V1/V6}, {V2/V3}, {V4/V5}
    (`identified_classes`); a variant the model NAMES in its completion is recorded as
    `named_variant` (last mention by position) and scored by class (`named_variant_correct`).
    A legal, parseable answer that is a variant reading of the input but not in the gold set
    is `plausible_nongold` (5.2). Other classes as T1.
T3  Có/Không (yes/no) mapped with negation precedence (extract.t3_label). When the run
    recorded log-probabilities, forced-choice accuracy (P(Có) vs P(Không) for the same prompt)
    and BLiMP-style paired accuracy (log P(correct candidate) > log P(twin candidate) under the
    same context, over the yes/no pair sharing pair_item_id; raw sum and per-token mean,
    `paired_correct` / `paired_correct_norm`) are added. A prefix-property violation is recorded
    per continuation (`prefix_property_violations`: Có / Không / candidate) and nulls only the
    prediction that continuation enters -- the forced choice needs both answer tokens, the pair
    comparison both candidates (7.3); a log-probability block carrying only the row-level
    `prefix_property_violation` flag (an older run) nulls every prediction. Cells report
    balanced accuracy and d' next to accuracy (generated T3 has a "yes" bias; an unparseable
    row is the wrong label, so it lowers tpr or tnr) and the headline excludes spelling twins
    (`t3_headline`).
XCOPA  generated 1/2 against the label; log-probability choice when recorded, on the summed
    log-probability (`logprob_pred`, primary) and on the per-token mean (`logprob_pred_norm`,
    secondary, 5.7); `delta_n_tokens` = n_tokens_1 - n_tokens_2 is the 8.6 covariate.

`score_outputs(items, outputs)` returns one row per output (header records in `items` are
ignored); `aggregate(rows)` the per-run summary; `write_scores` the scores.jsonl of
docs/DATA_FORMAT.md (every `strata` key is copied so that the statistics module reads only
that file).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import NormalDist

from ..gen import variants as V
from ..vi import reencode as R
from ..vi import unicode as U
from ..vi.reencode import canonical_text
from ..vi.syllable import Inventory, Syllable, spell, try_parse
from .extract import extract_answer, t3_label

ERROR_CLASSES = ("unparseable", "correct", "lenient_only", "copy", "reversal", "wrong_variant", "spelling",
                 "homophone", "doublet", "illegal", "component", "plausible_nongold",
                 "wrong")   # "wrong": T3/XCOPA answers of the wrong label; attested phrases of != 2 syllables
COMPONENTS = ("onset", "rime", "tone")
# the documented minimum; score_output copies EVERY key of item["strata"] (c2_affected, same_onset,
# variant_labels, ... included), this tuple is kept for readers of the format
STRATA_KEYS = ("input_lexical", "output_lexical", "output_syllables_attested", "has_glide", "has_zero_onset",
               "has_stop_coda", "spelling_triggers", "tone_pair", "same_tone", "same_onset", "same_rime",
               "variant_labels", "c2_affected", "input_freq", "output_freq", "n_readings")
# DESIGN_DECISIONS 5.4 item 8: regional homophone spellings, canonical onsets (<= 12 entries).
# NATIVE-CHECK: the list and its regional attribution (d/gi and r/d/gi: Northern; ch/tr, s/x: Northern;
# quốc/cuốc: the /k/ + glide spelling).
HOMOPHONE_ONSET_PAIRS = frozenset({frozenset({"d", "gi"}), frozenset({"ch", "tr"}), frozenset({"s", "x"}),
                                   frozenset({"r", "d"}), frozenset({"r", "gi"})})
HOMOPHONE_STRING_PAIRS = frozenset({frozenset({"quốc", "cuốc"})})
# DESIGN_DECISIONS 5.4 item 9: ay/ây lexical doublets (<= 10 entries; high, NV). Scored wrong, binned.
# NATIVE-CHECK: completeness of the list.
DOUBLETS = frozenset({frozenset({"giày", "giầy"}), frozenset({"dày", "dầy"}), frozenset({"chày", "chầy"}),
                      frozenset({"tày", "tầy"})})

_VARIANT_TOKEN = re.compile(r"(?<![A-Za-z0-9])V([1-6])(?![0-9])")
_KIND_TOKEN = re.compile(r"(?<![a-z])(?:kiểu|kieu|type|variant)\s*([1-6])(?![0-9])", re.IGNORECASE)


def _syl(d: dict) -> Syllable:
    return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])


def _inventory(inv: Inventory | None) -> Inventory:
    if inv is not None:
        return inv
    from ..vi import lexicon as L
    return L.load_inventory()


def variant_class(v: str | None) -> str | None:
    """The unordered class of a variant label (V1/V6, V2/V3, V4/V5)."""
    return V.UNORDERED_CLASS.get(v) if v else None


def _classes(labels: Iterable[str]) -> list[str]:
    return sorted({V.UNORDERED_CLASS[v] for v in labels if v in V.UNORDERED_CLASS})


def _swap(text: str) -> str:
    words = text.split()
    return " ".join(reversed(words)) if len(words) == 2 else text


def parse_phrase(text: str) -> tuple[list[Syllable], list[bool]] | None:
    """Two syllables (non-strict) and, per syllable, whether the STRICT parse also succeeds.
    None when the text is not two parseable syllables."""
    return parse_words(text, 2)


def parse_words(text: str, n: int) -> tuple[list[Syllable], list[bool]] | None:
    """`n` syllables (non-strict) and, per syllable, whether the strict parse also succeeds."""
    words = canonical_text(text).split()
    if len(words) != n:
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


def is_homophone_of(ans: list[Syllable], gold: list[Syllable]) -> bool:
    """The answer differs from the gold only by a listed homophone spelling (in order)."""
    if len(ans) != len(gold):
        return False
    any_diff = False
    for x, g in zip(ans, gold):
        if x == g:
            continue
        any_diff = True
        same_rest = (x.glide, x.nucleus, x.coda, x.tone) == (g.glide, g.nucleus, g.coda, g.tone)
        if same_rest and frozenset({x.onset, g.onset}) in HOMOPHONE_ONSET_PAIRS:
            continue
        if frozenset({spell(x), spell(g)}) in HOMOPHONE_STRING_PAIRS:
            continue
        return False
    return any_diff


def is_doublet_of(ans: list[Syllable], gold: list[Syllable]) -> bool:
    """The answer differs from the gold only by a listed ay/ây doublet (in order)."""
    if len(ans) != len(gold):
        return False
    any_diff = False
    for x, g in zip(ans, gold):
        if x == g:
            continue
        any_diff = True
        if (x.onset, x.glide, x.coda, x.tone) != (g.onset, g.glide, g.coda, g.tone) or x.coda != "j":
            return False
        if {x.nucleus, g.nucleus} != {"ă", "â"} or frozenset({spell(x), spell(g)}) not in DOUBLETS:
            return False
    return any_diff


def _base(item: dict, answer: str | None, method: str) -> dict:
    return {"item_id": item["item_id"], "task": item["task"], "variant": item["variant"], "answer": answer,
            "extraction_method": method, "correct": False, "correct_lenient": False, "error_class": "unparseable",
            "component_errors": [], "component_detail": None, "component_correct": None,
            "identified_variants": [], "identified_classes": [], "wrong_variant_labels": [], "named_variant": None,
            "placement_variant": None}


def placement_variant(text: str | None) -> str | None:
    """The tone-mark placement convention the answer actually uses (DESIGN_DECISIONS 5.4 item 7 /
    item 64): 'old', 'new', 'mixed' (both across its syllables), 'same' (no syllable
    distinguishes the conventions), or None when a syllable does not parse. Read from the text
    as written (canonical_text would normalize the placement away)."""
    if not text:
        return None
    seen = set()
    for w in U.nfc(text).lower().split():
        p = try_parse(w, strict=False)
        if p is None:
            return None
        if p.placement in ("old", "new"):
            seen.add(p.placement)
    if not seen:
        return "same"
    return seen.pop() if len(seen) == 1 else "mixed"


def _item_labels(item: dict) -> set[str]:
    labels = set(item.get("variant_labels") or [])
    labels.add(item["variant"])
    return labels


# ------------------------------------------------------------------ T1
def score_t1(item: dict, answer: str | None, method: str = "marker", inv: Inventory | None = None) -> dict:
    """T1 and attested items. `gold` may list several accepted forms (attested examples whose
    folk form bends the rule output); the answer is correct if it matches ANY of them, and the
    error taxonomy is computed against the reference form (`rule_output` when present, else the
    first gold). Phrases of other than two syllables (three-syllable attested examples) get the
    classes correct / copy / unparseable / wrong only."""
    row = _base(item, answer, method)
    golds = list(item["gold"]) if isinstance(item["gold"], list) else [item["gold"]]
    ref = item.get("rule_output") or golds[0]
    row["gold"] = golds[0] if len(golds) == 1 else golds
    if answer is None:
        return row
    ca = canonical_text(answer)
    n = len(item["input"].split())
    inp = [_syl(d) for d in item["input_syllables"]]
    if item.get("gold_syllables"):
        gold_syls = [_syl(d) for d in item["gold_syllables"]]
    elif n == 2:
        gp = parse_words(ref, 2)
        gold_syls = gp[0] if gp else (list(V.apply(item["variant"], *inp)) if item["variant"] in V.ALL_VARIANTS
                                      else None)
    else:
        gold_syls = None
    parsed = parse_words(answer, n)
    row["placement_variant"] = placement_variant(answer) if parsed is not None else None
    if parsed is not None and gold_syls is not None:
        # per-component view of every parseable answer (correct and copy included), so that
        # per-component accuracy is defined over all structural answers
        row["component_detail"], row["component_correct"] = _component_diff(parsed[0], gold_syls)
    gold_canon = {canonical_text(g) for g in golds}
    if ca in gold_canon:
        row.update(correct=True, correct_lenient=True, error_class="correct")
        return row
    if n == 2 and _swap(ca) in gold_canon:
        row.update(correct_lenient=True, error_class="lenient_only", component_errors=["order"])
        return row
    inp_canon = canonical_text(item["input"])
    if ca == inp_canon:
        row["error_class"] = "copy"
        return row
    if n == 2 and ca == _swap(inp_canon):
        row["error_class"] = "reversal"
        return row
    if parsed is None:
        return row
    if gold_syls is None or n != 2:
        row["error_class"] = "wrong"
        return row
    syls, strict_ok = parsed
    spelling_bad = not all(strict_ok)
    errs = sorted({c for wrong in row["component_detail"] for c in wrong}, key=COMPONENTS.index)
    if spelling_bad:
        errs.append("spelling")
    row["component_errors"] = errs
    ident = V.identify_any_order(inp[0], inp[1], syls[0], syls[1])
    row["identified_variants"] = [v for v, _ in ident]
    row["identified_classes"] = _classes(row["identified_variants"])
    own = _item_labels(item)
    other = [(v, r) for v, r in ident if v not in own]
    if other and syls != gold_syls:
        # 5.4 precedence: wrong_variant (6) before spelling (7). A misspelled answer whose structure
        # is the gold in the other order is another variant's output (the reverse variant) and is
        # binned here with its labels; only when that reverse variant is among the item's own labels
        # does it fall through to spelling + order below.
        row["error_class"] = "wrong_variant"
        row["wrong_variant_labels"] = [v + ("r" if r else "") for v, r in other]
        return row
    if syls == gold_syls[::-1]:
        # the gold structure in the other order but misspelled (exact spellings were lenient above)
        row["error_class"] = "spelling"
        row["component_errors"] = ["order", "spelling"]
        return row
    if syls == gold_syls:
        row["error_class"] = "spelling"          # right structure, wrong spelling
        return row
    if is_homophone_of(syls, gold_syls):
        row["error_class"] = "homophone"
        return row
    if is_doublet_of(syls, gold_syls):
        row["error_class"] = "doublet"
        return row
    inv = _inventory(inv)
    if not all(inv.is_legal(s, "onset_rime") for s in syls):
        row["error_class"] = "illegal"
        return row
    row["error_class"] = "component"
    return row


# ------------------------------------------------------------------ T2
def named_variant(raw: str | None, templates: dict | None = None) -> str | None:
    """The variant the model names LAST in its completion (by position): a 'V3' / 'kiểu 3'
    token or a variant name from the prompt templates (V1..V6; 'thanh điệu' and 'thanh' are
    folded together so that both phrasings match)."""
    if not raw:
        return None
    from .prompts import variant_names

    text = U.nfc(raw)
    low = _fold_names(text)
    hits: list[tuple[int, str]] = []
    for m in _VARIANT_TOKEN.finditer(text):
        hits.append((m.start(), f"V{m.group(1)}"))
    for m in _KIND_TOKEN.finditer(text):
        hits.append((m.start(), f"V{m.group(1)}"))
    for v, names in variant_names(templates).items():
        for name in names:
            fn = _fold_names(name)
            k = low.rfind(fn)
            if k >= 0:
                hits.append((k, v))
    return max(hits)[1] if hits else None


def _fold_names(text: str) -> str:
    t = U.nfc(text).lower()
    t = t.replace("thanh điệu", "thanh")
    return re.sub(r"\s+", " ", t)


def score_t2(item: dict, answer: str | None, method: str = "marker", raw: str | None = None,
             inv: Inventory | None = None) -> dict:
    row = _base(item, answer, method)
    golds = {canonical_text(g["output"]): g for g in item["gold"]}
    golds_rev = {_swap(canonical_text(g["output"])): g for g in item["gold"]}
    row["gold"] = [g["output"] for g in item["gold"]]
    row["gold_variants"] = sorted({g["variant"] for g in item["gold"]})
    gold_labels = set()
    for g in item["gold"]:
        gold_labels.add(g["variant"])
        gold_labels.update(lbl.rstrip("r") for lbl in (g.get("variant_labels") or []))
    row["gold_classes"] = _classes(gold_labels)
    row["gold_order"] = None
    row["named_variant"] = named_variant(raw)
    row["named_class"] = variant_class(row["named_variant"])
    row["named_variant_correct"] = None if row["named_class"] is None else (row["named_class"] in row["gold_classes"])
    if answer is None:
        return row
    ca = canonical_text(answer)
    inp = [_syl(d) for d in item["input_syllables"]]
    parsed = parse_phrase(answer)
    row["placement_variant"] = placement_variant(answer) if parsed is not None else None
    if parsed is not None:
        ident = V.identify_any_order(inp[0], inp[1], parsed[0][0], parsed[0][1])
        row["identified_variants"] = [v for v, _ in ident]
        row["identified_classes"] = _classes(row["identified_variants"])
    g = golds.get(ca) or golds_rev.get(ca)
    if g is not None:
        row.update(correct=True, correct_lenient=True, error_class="correct",
                   gold_order="attested" if ca in golds else "reversed")
        gp = parse_phrase(g["output"])
        if parsed is not None and gp is not None:
            ref = gp[0] if ca in golds else gp[0][::-1]
            row["component_detail"], row["component_correct"] = _component_diff(parsed[0], ref)
        return row
    inp_canon = canonical_text(item["input"])
    if ca == inp_canon:
        row["error_class"] = "copy"
        return row
    if ca == _swap(inp_canon):
        row["error_class"] = "reversal"
        return row
    if parsed is None:
        return row
    syls, strict_ok = parsed
    spelling_bad = not all(strict_ok)
    gold_structs = []
    for g in item["gold"]:
        gp = parse_phrase(g["output"])
        if gp:
            gold_structs.append(gp[0])
            gold_structs.append(gp[0][::-1])
    # right structure of some gold reading (either order) but misspelled
    if any(gs == syls for gs in gold_structs):
        row["error_class"] = "spelling"
        row["component_errors"] = ["spelling"]
        return row
    if spelling_bad:
        row["component_errors"] = ["spelling"]
    if any(is_homophone_of(syls, gs) for gs in gold_structs):
        row["error_class"] = "homophone"
        return row
    if any(is_doublet_of(syls, gs) for gs in gold_structs):
        row["error_class"] = "doublet"
        return row
    inv = _inventory(inv)
    if not all(inv.is_legal(s, "onset_rime") for s in syls):
        row["error_class"] = "illegal"
        return row
    if row["identified_variants"]:
        row["error_class"] = "plausible_nongold"   # a legal nói lái reading of the input, not in the gold set
        return row
    # closest gold reading for the component diff
    best = None
    for gs in gold_structs:
        detail, correct = _component_diff(syls, gs)
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
def violated(logprobs: dict | None, continuation: str) -> bool:
    """Whether `continuation`'s prefix property was violated (DESIGN_DECISIONS 7.3). Reads the
    per-continuation `prefix_property_violations` when the block carries it; a block with only the
    row-level `prefix_property_violation` flag (older runs) counts every continuation as violated."""
    if not logprobs:
        return False
    per = logprobs.get("prefix_property_violations")
    if isinstance(per, dict) and continuation in per:
        return bool(per[continuation])
    return bool(logprobs.get("prefix_property_violation"))


def score_t3(item: dict, answer: str | None, method: str = "marker", logprobs: dict | None = None) -> dict:
    row = _base(item, answer, method)
    row["gold"] = item["gold"]
    pred = t3_label(answer)
    row["pred"] = pred
    if pred is None:
        row["error_class"] = "unparseable"
    else:
        row["correct"] = pred == item["gold"]
        row["correct_lenient"] = row["correct"]
        row["error_class"] = "correct" if row["correct"] else "wrong"
    row["forced_choice_pred"] = None
    row["forced_choice_correct"] = None
    row["candidate_logprob"] = None
    row["candidate_n_tokens"] = None
    row["prefix_property_violation"] = None
    row["prefix_property_violations"] = None
    if logprobs:
        lp_yes, lp_no = logprobs.get("Có"), logprobs.get("Không")
        # a violated continuation is excluded from the log-probability metric it enters (DESIGN_DECISIONS
        # 7.3, item 55): the forced choice needs both answer tokens unviolated; the candidate's violation
        # nulls the pair comparison (paired_t3), not the forced choice. Log-probabilities are recorded as
        # returned; the predictions are what is nulled.
        if not violated(logprobs, "Có") and not violated(logprobs, "Không") and lp_yes is not None and lp_no is not None:
            row["forced_choice_pred"] = "yes" if lp_yes > lp_no else "no"
            row["forced_choice_correct"] = row["forced_choice_pred"] == item["gold"]
        row["candidate_logprob"] = logprobs.get("candidate")
        row["candidate_n_tokens"] = logprobs.get("candidate_n_tokens")
        row["prefix_property_violation"] = logprobs.get("prefix_property_violation")
        per = logprobs.get("prefix_property_violations")
        row["prefix_property_violations"] = dict(per) if isinstance(per, dict) else None
    row["twin_type"] = item.get("twin_type")
    row["pair_item_id"] = item.get("pair_item_id")
    return row


def paired_t3(rows: list[dict]) -> list[dict]:
    """Add `paired_correct` (BLiMP-style: the correct candidate's string log-probability beats
    the twin's, same context, same arm and prompt), `paired_correct_norm` (the same on the
    per-token mean, DESIGN_DECISIONS 5.3 item 40: raw and length-normalized are both reported)
    and `pair_both_correct` (both generated answers right) to every T3 row that has a partner
    in the same run. A pair whose candidate continuation was violated on either side (7.3: the
    violated continuation's prediction is nulled, an answer-token violation does not null the
    pair) or with a missing candidate log-probability gets None; the normalized field also
    needs both token counts."""
    by_key = {}
    for r in rows:
        if r["task"] == "T3":
            by_key[(r["item_id"], r.get("arm"), r.get("prompt_id"))] = r
    for r in rows:
        if r["task"] != "T3":
            continue
        r["paired_correct"] = None          # recomputed, never inherited from an earlier pass over the same rows
        r["paired_correct_norm"] = None
        r["pair_both_correct"] = None
        mate = by_key.get((r.get("pair_item_id"), r.get("arm"), r.get("prompt_id")))
        if mate is None:
            continue
        r["pair_both_correct"] = bool(r["correct"] and mate["correct"])
        yes, no = (r, mate) if r["gold"] == "yes" else (mate, r)
        if (yes.get("candidate_logprob") is not None and no.get("candidate_logprob") is not None
                and not _row_violated(yes, "candidate") and not _row_violated(no, "candidate")):
            r["paired_correct"] = yes["candidate_logprob"] > no["candidate_logprob"]
            ny, nn = yes.get("candidate_n_tokens"), no.get("candidate_n_tokens")
            if ny and nn:
                r["paired_correct_norm"] = yes["candidate_logprob"] / ny > no["candidate_logprob"] / nn
    return rows


def _row_violated(row: dict, continuation: str) -> bool:
    """`violated` read from a SCORED row (its `prefix_property_violations` / `prefix_property_violation`)."""
    return violated({"prefix_property_violations": row.get("prefix_property_violations"),
                     "prefix_property_violation": row.get("prefix_property_violation")}, continuation)


# ------------------------------------------------------------------ XCOPA
def score_xcopa(item: dict, answer: str | None, method: str = "marker", logprobs: dict | None = None) -> dict:
    row = _base(item, answer, method)
    gold = str(item["gold"])
    row["gold"] = gold
    pred = answer if answer in ("1", "2") else None
    row["pred"] = pred
    if pred is None:
        row["error_class"] = "unparseable"
    else:
        row["correct"] = pred == gold
        row["correct_lenient"] = row["correct"]
        row["error_class"] = "correct" if row["correct"] else "wrong"
    row["logprob_pred"] = None
    row["logprob_correct"] = None
    row["logprob_pred_norm"] = None          # per-token mean, the pre-registered secondary (5.7, item 68)
    row["logprob_correct_norm"] = None
    row["n_tokens_1"] = row["n_tokens_2"] = None
    row["delta_n_tokens"] = None             # n_tokens_1 - n_tokens_2: the 8.6 covariate
    row["prefix_property_violation"] = (logprobs or {}).get("prefix_property_violation")
    per = (logprobs or {}).get("prefix_property_violations")
    row["prefix_property_violations"] = dict(per) if isinstance(per, dict) else None
    # the choice needs both continuations unviolated (DESIGN_DECISIONS 7.3, item 55)
    if logprobs and not violated(logprobs, "1") and not violated(logprobs, "2") and logprobs.get("1") is not None \
            and logprobs.get("2") is not None:
        row["logprob_pred"] = "1" if logprobs["1"] > logprobs["2"] else "2"
        row["logprob_correct"] = row["logprob_pred"] == gold
        n1, n2 = logprobs.get("n_tokens_1"), logprobs.get("n_tokens_2")
        row["n_tokens_1"], row["n_tokens_2"] = n1, n2
        if n1 and n2:
            row["delta_n_tokens"] = n1 - n2
            row["logprob_pred_norm"] = "1" if logprobs["1"] / n1 > logprobs["2"] / n2 else "2"
            row["logprob_correct_norm"] = row["logprob_pred_norm"] == gold
    return row


# ------------------------------------------------------------------ dispatch
_COPIED_OUTPUT_KEYS = ("arm", "arm_scope", "prompt_id", "prompt_hash", "templated_prompt_hash", "n_prompt_tokens",
                       "n_output_tokens", "n_prompt_tokens_base", "delta_prompt_tokens", "latency_s", "finish_reason",
                       "truncated", "n_thinking_chars", "thinking_unclosed", "n_placement_changes",
                       "demo_syllable_overlap", "hedged")
_COPIED_ITEM_KEYS = ("base_pair_id", "source", "split", "in_core", "twin_type", "pair_item_id", "variant_labels",
                     "gold_validated", "vulgar", "exactness", "exact", "eligible_h6", "n_syllables")


def score_output(item: dict, out: dict, inv: Inventory | None = None) -> dict:
    """Score one outputs.jsonl row against its item. Re-extracts the answer from `raw` when
    the row has no `answer` key (older runs)."""
    task = item["task"]
    if "answer" in out and "extraction_method" in out:
        answer, method = out["answer"], out["extraction_method"]
    else:
        answer, method = extract_answer(out.get("raw"), task)
    if task in ("T1", "attested"):
        row = score_t1(item, answer, method, inv)
    elif task == "T2":
        row = score_t2(item, answer, method, out.get("raw"), inv)
    elif task == "T3":
        row = score_t3(item, answer, method, out.get("logprobs"))
    elif task == "XCOPA":
        row = score_xcopa(item, answer, method, out.get("logprobs"))
    else:
        raise ValueError(f"unknown task {task!r}")
    for k in _COPIED_OUTPUT_KEYS:
        row[k] = out.get(k)
    row["raw_len"] = len(out.get("raw") or "")
    for k in _COPIED_ITEM_KEYS:
        if k in item:
            row[k] = item[k]
    for k, v in (item.get("strata") or {}).items():
        row[k] = v
    row["n_input_tokens_syll"] = None
    return row


def score_outputs(items: Iterable[dict], outputs: Iterable[dict], inv: Inventory | None = None,
                  audit_rows: dict | None = None) -> list[dict]:
    by_id = {it["item_id"]: it for it in items if "item_id" in it}      # a header record has no item_id
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


def _spellings(word: str, arm: str) -> list[str]:
    """Lookup keys for a syllable: as written under the arm's placement, then the canonical
    new-style and old-style spellings (the audit stores one convention)."""
    w = U.nfc(word).lower()
    keys = []
    if arm in ("placement_old", "placement_new"):
        keys.append(R.convert_placement(w, "old" if arm == "placement_old" else "new"))
    keys.append(w)
    p = try_parse(w, strict=False)
    if p is not None:
        keys.append(spell(p.syllable, style="new"))
        keys.append(spell(p.syllable, style="old"))
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def input_token_counts(item: dict, arm: str, audit_rows: dict) -> list[int | None] | None:
    """Tokens per input syllable under the arm's encoding, from the audit rows. Placement
    arms are looked up by their spelling and fall back to the canonical spellings; strip arms
    have no audit rows and return None."""
    from .prompts import normalize_arm

    arm = normalize_arm(arm)
    if arm in ("strip_tones", "strip_all"):
        return None
    enc = {"nfd": "nfd", "win1258": "win1258"}.get(arm, "nfc")
    out: list[int | None] = []
    for w in item["input"].split():
        val = None
        for k in _spellings(w, arm):
            if (k, enc) in audit_rows:
                val = audit_rows[(k, enc)]
                break
        out.append(val)
    return out


# ------------------------------------------------------------------ aggregation
def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def balanced_stats(rows: list[dict], pred_key: str = "pred") -> dict:
    """Balanced accuracy, d' and yes-rate of a yes/no cell. Unparseable rows count as wrong
    (5.6): a missing prediction is scored as the wrong label. d' uses the log-linear
    correction (0.5 added to every count) so that 0% / 100% cells stay finite."""
    yes_rows = [r for r in rows if r.get("gold") == "yes"]
    no_rows = [r for r in rows if r.get("gold") == "no"]
    n_pred_all = sum(1 for r in rows if r.get(pred_key) is not None)
    if not yes_rows or not no_rows:
        return {"balanced_accuracy": None, "d_prime": None,
                "yes_rate": (sum(1 for r in rows if r.get(pred_key) == "yes") / n_pred_all) if n_pred_all else None}
    hits = sum(1 for r in yes_rows if r.get(pred_key) == "yes")
    # a missing prediction on a 'no' row is scored as the wrong label 'yes' (a false alarm), exactly as
    # a missing prediction on a 'yes' row is a miss: unparseable is wrong, never a correct rejection (5.6)
    fas = sum(1 for r in no_rows if r.get(pred_key) != "no")
    tpr, tnr = hits / len(yes_rows), 1 - fas / len(no_rows)
    z = NormalDist().inv_cdf
    h = (hits + 0.5) / (len(yes_rows) + 1)
    f = (fas + 0.5) / (len(no_rows) + 1)
    n_pred = sum(1 for r in rows if r.get(pred_key) is not None)
    return {"balanced_accuracy": (tpr + tnr) / 2, "d_prime": z(h) - z(f),
            "yes_rate": (sum(1 for r in rows if r.get(pred_key) == "yes") / n_pred) if n_pred else None,
            "tpr": tpr, "tnr": tnr}


def _spelling_pair_ids(rows: list[dict]) -> set[str]:
    ids = set()
    for r in rows:
        if r.get("task") == "T3" and r.get("twin_type") == "spelling":
            ids.add(r["item_id"])
            if r.get("pair_item_id"):
                ids.add(r["pair_item_id"])
    return ids


def aggregate(rows: list[dict]) -> dict:
    """Per-run summary: accuracy (strict and lenient) per task x variant, per arm, per prompt;
    error-class distribution; copy and unparseable rates; T3 balanced accuracy / d',
    forced-choice and paired accuracy, and the spelling-twin-excluded headline; XCOPA
    accuracy by arm."""
    def cell(sub: list[dict]) -> dict:
        d = {"n": len(sub), "accuracy": _rate(sub, "correct"),
             "error_classes": dict(Counter(r["error_class"] for r in sub)),
             "copy_rate": sum(1 for r in sub if r["error_class"] == "copy") / len(sub) if sub else None,
             "unparseable_rate": sum(1 for r in sub if r["error_class"] == "unparseable") / len(sub) if sub else None}
        if any(r["task"] in ("T1", "attested", "T2") for r in sub):
            d["accuracy_lenient"] = _rate(sub, "correct_lenient")
        comp = Counter(c for r in sub for c in (r.get("component_errors") or []))
        d["component_errors"] = dict(comp)
        cc = defaultdict(list)
        for r in sub:
            for c, vals in (r.get("component_correct") or {}).items():
                cc[c].extend(vals)
        d["component_accuracy"] = {c: sum(v) / len(v) for c, v in cc.items() if v}
        if any(r["task"] == "T3" for r in sub):
            t3 = [r for r in sub if r["task"] == "T3"]
            d.update(balanced_stats(t3, "pred"))
            d["forced_choice_accuracy"] = _rate(sub, "forced_choice_correct")
            fc = [r for r in t3 if r.get("forced_choice_pred") is not None]
            if fc:
                d["forced_choice_balanced_accuracy"] = balanced_stats(fc, "forced_choice_pred")["balanced_accuracy"]
            d["paired_accuracy"] = _rate(sub, "paired_correct")
            d["paired_accuracy_norm"] = _rate(sub, "paired_correct_norm")
            d["pair_both_correct_rate"] = _rate(sub, "pair_both_correct")
        if any(r["task"] == "T2" for r in sub):
            d["named_variant_class_accuracy"] = _rate(sub, "named_variant_correct")
            d["gold_order_reversed_rate"] = (sum(1 for r in sub if r.get("gold_order") == "reversed") / len(sub)
                                             if sub else None)
        if any(r["task"] == "XCOPA" for r in sub):
            d["logprob_accuracy"] = _rate(sub, "logprob_correct")
            d["logprob_accuracy_norm"] = _rate(sub, "logprob_correct_norm")
        return d

    groups: dict[str, dict] = {"by_task": {}, "by_task_variant": {}, "by_task_arm": {}, "by_task_prompt": {},
                               "by_task_twin_type": {}, "t3_headline": {}}
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
            spelling_ids = _spelling_pair_ids(sub)
            excl = [r for r in sub if r["item_id"] not in spelling_ids]
            only = [r for r in sub if r["item_id"] in spelling_ids]
            groups["t3_headline"]["excl_spelling"] = cell(excl)
            groups["t3_headline"]["spelling_pairs"] = cell(only)
            for v in sorted({r["variant"] for r in excl}):
                groups["t3_headline"][f"excl_spelling-{v}"] = cell([r for r in excl if r["variant"] == v])
    groups["n_rows"] = len(rows)
    groups["overall"] = cell(rows) if rows else {"n": 0}
    return groups


def summary_table(agg: dict) -> str:
    """A plain-text table of accuracy per task x variant (and per arm)."""
    lines = [f"{'cell':<22}{'n':>6}{'acc':>8}{'lenient':>9}{'copy':>7}{'unpars':>8}  error classes"]
    for key in ("by_task_variant", "by_task_arm", "t3_headline"):
        for name, c in agg.get(key, {}).items():
            acc = f"{c['accuracy']:.3f}" if c["accuracy"] is not None else "-"
            len_ = f"{c['accuracy_lenient']:.3f}" if c.get("accuracy_lenient") is not None else "-"
            copy = f"{c['copy_rate']:.2f}" if c["copy_rate"] is not None else "-"
            unp = f"{c['unparseable_rate']:.2f}" if c["unparseable_rate"] is not None else "-"
            extra = ""
            if c.get("balanced_accuracy") is not None:
                extra += f" bal={c['balanced_accuracy']:.3f} d'={c['d_prime']:.2f}"
            if c.get("forced_choice_accuracy") is not None:
                extra += f" fc={c['forced_choice_accuracy']:.3f}"
            if c.get("paired_accuracy") is not None:
                extra += f" paired={c['paired_accuracy']:.3f}"
            if c.get("paired_accuracy_norm") is not None:
                extra += f" paired_norm={c['paired_accuracy_norm']:.3f}"
            if c.get("logprob_accuracy") is not None:
                extra += f" lp={c['logprob_accuracy']:.3f}"
            if c.get("logprob_accuracy_norm") is not None:
                extra += f" lp_norm={c['logprob_accuracy_norm']:.3f}"
            ec = " ".join(f"{k}={v}" for k, v in sorted(c["error_classes"].items()))
            lines.append(f"{name:<22}{c['n']:>6}{acc:>8}{len_:>9}{copy:>7}{unp:>8}  {ec}{extra}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ------------------------------------------------------------------ files
def write_scores(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip() and "_header" not in json.loads(ln)]


def spell_pair(syls: Iterable[Syllable]) -> str:
    return " ".join(spell(s) for s in syls)
