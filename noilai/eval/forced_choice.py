"""T1 multi-distractor forced choice: an EXPLORATORY analysis, pre-specified on 1 October 2026 before any model run
(docs/FORCED_CHOICE_EXPLORATORY.md; docs/DEVIATIONS.md row "exploratory T1 forced choice"). Not a pre-registered
hypothesis test and never compared with generated accuracy (DESIGN_DECISIONS 5.3 keeps scoring modes apart).

Why: when strict generated T1 accuracy is at or near zero (the floor risk of the Gate 1 pilot), it carries no graded
information. Forced choice asks a weaker question of the same prompt: does the model put more probability on the
rule's answer than on near-miss alternatives built from the same two syllables? It is the T1 analogue of the
pre-registered `t3_pair_lp` (DD 5.3), with the candidate set fixed by rule, not by the model.

Candidate set for a T1 item (input a b, named variant v), deduplicated after canonicalization:
  gold            the rule output in the named order;
  wrong_variant:V the outputs of the other five swap kinds (V1-V6), except any equal to the gold or to the gold in
                  the other order (lenient-correct, DD 3.2);
  copy            the input itself;
  reversal        the input in the other order;
  spelling        the gold misspelled by the c/k, g/gh or ng/ngh rule when the gold has a trigger (reported in its
                  own column, excluded from the headline set, like the T3 spelling twins).
Scoring context and boundary are those of `t3_pair_lp` (DD 7.3 amendment): the rendered T1 prompt + the answer
marker, every candidate a space-led continuation; a continuation that violates the prefix property scores None.
Metrics per item: fc_correct (the gold has the highest summed log-probability among the headline set),
fc_correct_mean (the same with the per-token mean), fc_rank, fc_n (set size) and fc_chance = 1 / fc_n.
"""
from __future__ import annotations

from noilai.gen import variants as V
from noilai.gen.generate import syl_from_dict
from noilai.vi import lexicon as L
from noilai.vi import reencode as R
from noilai.vi.reencode import canonical_text

HEADLINE_EXCLUDES = ("spelling",)
_MISSPELL = (("k", "c"), ("gh", "g"), ("ngh", "ng"))


def _spell(pair) -> str:
    return L.emit_phrase(pair, style="old")


def _misspelled(text: str) -> str | None:
    words = text.split()
    for i, w in enumerate(words):
        for good, bad in _MISSPELL:
            if w.startswith(good) and not (good == "k" and w.startswith("kh")):
                words[i] = bad + w[len(good):]
                return " ".join(words)
    return None


def t1_candidates(item: dict) -> list[dict]:
    """[{label, text}] with the gold first; strings in the release's placement style (old)."""
    a, b = (syl_from_dict(d) for d in item["input_syllables"])
    gold_text = item["gold"][0]
    gold = canonical_text(gold_text)
    gold_set = set((gold or "").split())
    out = [{"label": "gold", "text": gold_text}]
    seen = {gold}
    for v in V.ALL_VARIANTS:
        if v == item["variant"]:
            continue
        text = _spell(V.apply(v, a, b))
        c = canonical_text(text)
        if c is None or c in seen or set(c.split()) == gold_set:
            continue
        seen.add(c)
        out.append({"label": f"wrong_variant:{v}", "text": text})
    for label, text in (("copy", item["input"]), ("reversal", " ".join(reversed(item["input"].split())))):
        c = canonical_text(text)
        if c is not None and c not in seen and set(c.split()) != gold_set:
            seen.add(c)
            out.append({"label": label, "text": text})
    bad = _misspelled(gold_text)
    if bad is not None and bad != gold_text:
        out.append({"label": "spelling", "text": bad})
    return out


def score_candidates(backend, context: str, cands: list[dict], arm: str, encode: bool) -> dict:
    conts = [" " + (R.reencode(c["text"], arm) if encode else c["text"]) for c in cands]
    lp = backend.logprobs(context, conts)
    viol = list(getattr(backend, "last_prefix_violations", []) or [None] * len(conts))
    ntok = list(getattr(backend, "last_n_tokens", []) or [None] * len(conts))
    if len(viol) != len(conts):
        viol = [None] * len(conts)
    if len(ntok) != len(conts):
        ntok = [None] * len(conts)
    vals = [None if (v or x is None) else float(x) for x, v in zip(lp, viol)]
    return {"labels": [c["label"] for c in cands], "texts": [c["text"] for c in cands], "lp": vals,
            "n_tokens": ntok, "prefix_property_violations": viol,
            "prompt_ids_sha256": getattr(backend, "last_prompt_ids_sha256", None)}


def fc_metrics(fc: dict | None) -> dict:
    """Per-item metrics over the headline set (spelling excluded); None when the gold or every distractor is unscored."""
    empty = {"fc_correct": None, "fc_correct_mean": None, "fc_rank": None, "fc_n": None, "fc_chance": None,
             "fc_spelling_win": None}
    if not fc:
        return empty
    rows = [(lab, lp, n) for lab, lp, n in zip(fc["labels"], fc["lp"], fc["n_tokens"]) if lab not in HEADLINE_EXCLUDES]
    gold = next((r for r in rows if r[0] == "gold"), None)
    others = [r for r in rows if r[0] != "gold" and r[1] is not None]
    if gold is None or gold[1] is None or not others:
        return empty
    rank = 1 + sum(o[1] > gold[1] for o in others)
    per_tok = [(lab, lp / n) for lab, lp, n in rows if lp is not None and n]
    g_mean = next((v for lab, v in per_tok if lab == "gold"), None)
    out = {"fc_correct": rank == 1 and all(o[1] < gold[1] for o in others), "fc_rank": rank, "fc_n": 1 + len(others),
           "fc_chance": 1 / (1 + len(others)),
           "fc_correct_mean": (all(v < g_mean for lab, v in per_tok if lab != "gold") if g_mean is not None else None)}
    sp = [lp for lab, lp in zip(fc["labels"], fc["lp"]) if lab == "spelling" and lp is not None]
    out["fc_spelling_win"] = (gold[1] > sp[0]) if sp else None
    return out
