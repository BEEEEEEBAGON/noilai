"""Native validation and human-baseline material (DESIGN_DECISIONS 10.1 / 10.2 as amended on 1 October 2026).

Pure functions behind `scripts/make_validation_forms.py`: the probability sample with recorded
stratum weights, planted control items, the validator assignment, the keyed calibration round,
the attested sheet, the supplementary sheets (variant production, the qu- check, spelling
conventions), the T2 gold-set sheet, the full-prompt human-baseline rows, and the scoring of
returned sheets (agreement, adjudication, weighted generator precision, control catch rates,
attested verification, T2 gold sets).

Design (sizes in `noilai.constants`, chosen with `scripts/validation_sizing.py`):

* Part A, calibration: `VALIDATION_CALIBRATION_ITEMS` keyed items computed by the rule engine from
  lexical pairs that are not in the release, with planted errors; never used in any estimate.
* Part B, generated items: `VALIDATION_PER_CELL` items per task x variant cell drawn as a stratified
  probability sample (strata = cell x lexical/pseudo source, proportional allocation, weights
  N_h / n_h recorded), plus `VALIDATION_CONTROLS_PER_CELL` planted control items per cell (a
  deliberately wrong candidate or verdict, built from items outside the sample); every item judged
  by two validators, `VALIDATION_OVERLAP` items by all three when three take part. Rows carry
  opaque ids (`B-0001`); the key that maps them to items and marks the controls stays with the
  author.
* Part C, attested rows: every row of the attested seed plus the web-sourced candidates, judged by
  every validator.
* Part D, supplementary: the 20-pair "which kind do you produce?" item, the 20-pair qu- check and
  the spelling-convention checks.
* Part E, T2 gold sets of the core: each core T2 item's dictionary readings, accepted or rejected,
  split across validators with an overlap for agreement.

Nothing in this module calls a model or sends anything anywhere.
"""
from __future__ import annotations

import math
import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from noilai import constants as C
from noilai.gen import variants as V
from noilai.gen.generate import syl_from_dict
from noilai.vi import lexicon as L
from noilai.vi.reencode import canonical_text
from noilai.vi.syllable import Syllable, replace, try_parse

CELLS = tuple((t, v) for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4"))
JUDGMENTS_B = ("correct", "spelling", "lexical_input", "lexical_candidate", "offensive")
JUDGMENTS_C = ("known", "valid", "spelling_ok", "offensive")
LABELS = ("yes", "no", "unsure")
# The O3 merger table of DESIGN_DECISIONS 2.1, as the validators' `dialect` dropdown. `none` = my
# judgment does not depend on a regional pronunciation.
DIALECT_CHOICES = ("none", "d=gi", "d/gi=r", "ch=tr", "s=x", "v=d/gi", "final n=ng", "final t=c", "hỏi=ngã", "other")
VARIANT_GLOSS_VI = {"V1": "đổi vần, giữ phụ âm đầu và thanh", "V2": "đổi chỗ hai âm tiết, giữ thanh",
                    "V3": "đổi thanh, giữ phụ âm đầu và vần", "V4": "đổi vần và thanh, giữ phụ âm đầu"}
VARIANT_GLOSS_EN = {"V1": "swap rimes, keep onsets and tones", "V2": "swap whole syllables, keep tones in place",
                    "V3": "swap tones, keep onsets and rimes", "V4": "swap rimes with their tones, keep onsets"}
_YES = {"yes", "y", "có", "co", "c", "1", "true", "đúng", "dung"}
_NO = {"no", "n", "không", "khong", "k", "0", "false", "sai"}
_UNSURE = {"unsure", "u", "?", "không chắc", "khong chac", "kc", "chưa chắc", "chua chac"}


# --------------------------------------------------------------------------- labels
def norm_label(value) -> str | None:
    """Map a returned cell to yes / no / unsure; None for an empty cell (missing, not a label)."""
    if value is None:
        return None
    s = unicodedata.normalize("NFC", str(value)).strip().lower()
    if not s:
        return None
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    if s in _UNSURE:
        return "unsure"
    raise ValueError(f"unrecognized label {value!r}: use yes / no / unsure (Có / Không / Không chắc)")


def norm_dialect(value) -> str:
    s = unicodedata.normalize("NFC", str(value or "")).strip().lower()
    if not s:
        return "none"
    for c in DIALECT_CHOICES:
        if s == c.lower():
            return c
    return "other"


# --------------------------------------------------------------------------- engine helpers
def phrase_syllables(text: str) -> list[Syllable] | None:
    out = []
    for w in unicodedata.normalize("NFC", text).split():
        p = try_parse(w, strict=False)
        if p is None:
            return None
        out.append(p.syllable)
    return out


def spell_pair(pair: Sequence[Syllable]) -> str:
    return L.emit_phrase(pair, style="old")


def phrase_key(text: str) -> frozenset | None:
    """Order-free canonical key of a phrase (nói lái is an involution and items are kept in either order)."""
    c = canonical_text(text)
    if c is None:
        return None
    return frozenset(c.split())


def release_phrase_keys(items: Iterable[dict]) -> set[frozenset]:
    """Every input / gold / candidate phrase of a release, order-free: calibration material must avoid them."""
    keys = set()
    for it in items:
        texts = [it.get("input"), it.get("candidate")]
        g = it.get("gold")
        if isinstance(g, list):
            texts += [x["output"] if isinstance(x, dict) else x for x in g]
        for t in texts:
            if t:
                k = phrase_key(t)
                if k:
                    keys.add(k)
    return keys


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- Part B: sample, controls, assignment
def stratum_key(it: dict) -> tuple[str, str, str]:
    return (it["task"], it["variant"], it.get("source") or "unknown")


def _largest_remainder(sizes: dict, n: int) -> dict:
    total = sum(sizes.values())
    if total == 0:
        return {k: 0 for k in sizes}
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], math.floor(r)) for k, r in raw.items()}
    left = n - sum(alloc.values())
    for k in sorted(raw, key=lambda k: (raw[k] - math.floor(raw[k]), k), reverse=True):
        if left <= 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            left -= 1
    return alloc


def probability_sample(items: Iterable[dict], per_cell: int, rng: random.Random,
                       exclude_keys: set | None = None) -> tuple[list[dict], dict]:
    """A stratified probability sample: `per_cell` items in each task x variant cell, allocated to the
    cell's lexical / pseudo strata in proportion to their population (largest remainder), drawn at
    random within a stratum; never both members of a T3 twin pair; never an item whose phrase is in
    `exclude_keys` (the calibration material). Returns (items, strata) with strata[key] =
    {n_population, n_sampled, weight = n_population / n_sampled}."""
    exclude_keys = exclude_keys or set()
    pop: dict = defaultdict(list)
    for it in items:
        if it.get("task") not in ("T1", "T2", "T3"):
            continue
        if phrase_key(it["input"]) in exclude_keys:
            continue
        pop[stratum_key(it)].append(it)
    chosen, strata = [], {}
    for task, variant in CELLS:
        keys = sorted(k for k in pop if k[:2] == (task, variant))
        alloc = _largest_remainder({k: len(pop[k]) for k in keys}, per_cell)
        taken_pairs = set()
        for k in keys:
            cand = sorted(pop[k], key=lambda x: x["item_id"])
            rng.shuffle(cand)
            got = []
            for it in cand:
                if len(got) >= alloc[k]:
                    break
                if it.get("pair_item_id") and it["pair_item_id"] in taken_pairs:
                    continue
                got.append(it)
                taken_pairs.add(it["item_id"])
            strata["|".join(k)] = {"n_population": len(pop[k]), "n_sampled": len(got),
                                   "weight": (len(pop[k]) / len(got)) if got else None}
            chosen += got
    return chosen, strata


def _wrong_tone(s: Syllable) -> Syllable:
    """Another tone for the syllable that keeps it phonotactically possible (stop codas: sắc <-> nặng)."""
    if s.coda in ("p", "t", "c", "ch"):
        return replace(s, tone=5 if s.tone == 2 else 2)
    return replace(s, tone=(s.tone + 2) % 6 if (s.tone + 2) % 6 != s.tone else (s.tone + 1) % 6)


def make_control(it: dict, rng: random.Random) -> dict | None:
    """A planted control built from an item outside the sample: the shown candidate (T1, T2) or the
    shown system verdict (T3) is deliberately wrong, so the expected `correct` judgment is "no".
    Returns the display fields plus `control_kind`; None when no safe corruption exists."""
    task, variant = it["task"], it["variant"]
    inp = [syl_from_dict(d) for d in it["input_syllables"]]
    a, b = inp
    if task == "T1":
        gold = canonical_text(it["gold"][0])
        options = []
        for v in V.ALL_VARIANTS:
            if v == variant:
                continue
            out = V.apply(v, a, b)
            if out in ((a, b), (b, a)):
                continue
            cand = spell_pair(out)
            cc = canonical_text(cand)
            if cc is None or cc == gold or set(cc.split()) == set(gold.split()):
                continue                      # the gold itself or the gold in the other order (lenient-correct): not a safe control
            options.append(("wrong_variant:" + v, cand))
        g = [syl_from_dict(d) for d in it["gold_syllables"]]
        cand = spell_pair([_wrong_tone(g[0]), g[1]])
        if canonical_text(cand) != gold:
            options.append(("wrong_tone", cand))
        if not options:
            return None
        kind, cand = options[rng.randrange(len(options))]
        return {"candidate": cand, "system_verdict": "", "control_kind": kind}
    if task == "T2":
        readings = {canonical_text(g["output"]) for g in it["gold"]}
        first = [syl_from_dict(d) for d in it["input_syllables"]]
        base = phrase_syllables(it["gold"][0]["output"])
        if base is None:
            return None
        for cand_pair in ([_wrong_tone(base[0]), base[1]], [base[0], _wrong_tone(base[1])]):
            cand = spell_pair(cand_pair)
            cs = phrase_syllables(cand)
            if canonical_text(cand) in readings or cs is None:
                continue
            if V.identify_any_order(cs[0], cs[1], first[0], first[1]):
                continue                      # still a valid reading under some variant: not a safe control
            return {"candidate": cand, "system_verdict": "", "control_kind": "not_a_reading"}
        return None
    # T3: the candidate is shown with the opposite system verdict
    return {"candidate": it["candidate"], "system_verdict": "no" if it["gold"] == "yes" else "yes",
            "control_kind": "flipped_verdict"}


def assign(n_items: int, validators: Sequence[str], overlap: int) -> list[list[str]]:
    """Validators per item: with two validators every item to both; with three or more, the first
    `overlap` items to everyone and the rest to two validators in rotating pairs (A-B, B-C, C-A, ...)."""
    k = len(validators)
    if k < 2:
        raise ValueError("native validation needs at least two validators (agreement cannot be estimated with one)")
    if k == 2:
        return [list(validators) for _ in range(n_items)]
    out = []
    for j in range(n_items):
        if j < overlap:
            out.append(list(validators))
        else:
            r = (j - overlap) % k
            out.append([validators[r], validators[(r + 1) % k]])
    return out


def question_b(task: str, variant: str, inp: str, cand: str, verdict: str) -> tuple[str, str]:
    """[NATIVE-CHECK] the per-row question, Vietnamese and English."""
    if task == "T1":
        return (f"Nói lái kiểu {variant} ({VARIANT_GLOSS_VI[variant]}) của “{inp}” là “{cand}”?",
                f"Is “{cand}” the {variant} nói lái ({VARIANT_GLOSS_EN[variant]}) of “{inp}”?")
    if task == "T2":
        return (f"“{inp}” là cách nói lái (một kiểu bất kỳ) của “{cand}”?",
                f"Is “{inp}” a nói lái (of any kind) of “{cand}”?")
    return ((f"“{cand}” có phải là nói lái kiểu {variant} ({VARIANT_GLOSS_VI[variant]}) của “{inp}” không? "
             f"Hệ thống trả lời: {'Có' if verdict == 'yes' else 'Không'}."),
            f"Is “{cand}” the {variant} nói lái ({VARIANT_GLOSS_EN[variant]}) of “{inp}”? The system answers: {verdict}.")


def display_b(it: dict, override: dict | None = None) -> dict:
    """The fields a validator sees for one Part B row (no item id, no cell code beyond the kind name)."""
    task, variant = it["task"], it["variant"]
    if task == "T1":
        cand, verdict = it["gold"][0], ""
    elif task == "T2":
        cand, verdict = it["gold"][0]["output"], ""
    else:
        cand, verdict = it["candidate"], it["gold"]
    if override:
        cand = override.get("candidate", cand)
        verdict = override.get("system_verdict") or verdict
    q_vi, q_en = question_b(task, variant, it["input"], cand, verdict)
    return {"task": task, "kind": variant if task != "T2" else "", "input": it["input"], "candidate": cand,
            "system_verdict": verdict, "question_vi": q_vi, "question_en": q_en}


def build_part_b(items: list[dict], validators: Sequence[str], seed: int, per_cell: int = C.VALIDATION_PER_CELL,
                 controls_per_cell: int = C.VALIDATION_CONTROLS_PER_CELL, overlap: int = C.VALIDATION_OVERLAP,
                 exclude_keys: set | None = None) -> dict:
    """Sample + controls + assignment. Returns {'rows': [...], 'key': {...}, 'strata': {...}, 'sheets': {letter: [row ids]}}."""
    rng = random.Random(seed)
    sample, strata = probability_sample(items, per_cell, rng, exclude_keys)
    in_sample = {it["item_id"] for it in sample} | {it.get("pair_item_id") for it in sample if it.get("pair_item_id")}
    controls = []
    for task, variant in CELLS:
        pool = sorted((it for it in items if (it.get("task"), it.get("variant")) == (task, variant)
                       and it["item_id"] not in in_sample and not it.get("vulgar")
                       and phrase_key(it["input"]) not in (exclude_keys or set())), key=lambda x: x["item_id"])
        rng.shuffle(pool)
        got = 0
        for it in pool:
            if got >= controls_per_cell:
                break
            ctl = make_control(it, rng)
            if ctl is None:
                continue
            controls.append((it, ctl))
            in_sample.add(it["item_id"])
            got += 1
        if got < controls_per_cell:
            raise ValueError(f"cell {task}-{variant}: only {got} of {controls_per_cell} control items could be built")
    entries = [(it, None) for it in sample] + controls
    rng.shuffle(entries)
    who = assign(len(entries), validators, overlap)
    rows, key, sheets = [], {}, defaultdict(list)
    for j, ((it, ctl), vs) in enumerate(zip(entries, who)):
        rid = f"B-{j + 1:04d}"
        d = display_b(it, ctl)
        rows.append({"row_id": rid, **d})
        stratum = "|".join(stratum_key(it))
        key[rid] = {"item_id": it["item_id"], "cell": f"{it['task']}-{it['variant']}", "stratum": stratum,
                    "base_pair_id": it.get("base_pair_id"), "split": it.get("split"),
                    "control": ctl is not None, "control_kind": ctl["control_kind"] if ctl else None,
                    "expected_correct": "no" if ctl else "yes", "validators": vs,
                    "weight": None if ctl else strata[stratum]["weight"],
                    "stratum_population": None if ctl else strata[stratum]["n_population"]}
        for v in vs:
            sheets[v].append(rid)
    return {"rows": rows, "key": key, "strata": strata, "sheets": dict(sheets),
            "n_sample": len(sample), "n_controls": len(controls)}


# --------------------------------------------------------------------------- Part A: calibration
# (input phrase(s) tried in order, task, variant, manipulation). The first input that is legal for the
# manipulation and absent from the release is used. Inputs are common lexical pairs, never attested
# rows (Part C) or demonstration pairs. [NATIVE-CHECK] every phrase.
CALIBRATION_SPECS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("bàn ghế", "cây cối", "sách vở"), "T1", "V1", "none"),
    (("nhà cửa", "ruộng vườn", "quần áo"), "T1", "V2", "none"),
    (("bánh mì", "tôm cá", "nồi niêu"), "T1", "V3", "none"),
    (("cơm nước", "giường tủ", "rau dưa"), "T1", "V4", "none"),
    (("bàn ghế", "cây cối", "sách vở"), "T1", "V1", "wrong_variant"),
    (("nhà cửa", "ruộng vườn", "quần áo"), "T1", "V2", "wrong_tone"),
    (("cá mè", "cá chép", "bò bía"), "T1", "V1", "misspell_k"),
    (("hoa lá", "hoa quả", "hoa hồng"), "T1", "V3", "placement_new"),
    (("mì tôm", "lí do", "kì lạ"), "T1", "V4", "iy_variant"),
    (("đá đeo",), "T1", "V1", "vulgar"),
    (("kỹ sư", "bữa cơm", "sữa chua"), "T1", "V3", "dialect_hoi_nga"),
    (("bàn ghế", "cây cối", "sách vở"), "T2", "V1", "none"),
    (("cơm nước", "giường tủ", "rau dưa"), "T2", "V4", "wrong_reading"),
    (("nhà cửa", "ruộng vườn", "quần áo"), "T3", "V3", "none"),
    (("cá mè", "cá chép", "bò bía"), "T3", "V1", "spelling_twin"),
    (("cơm nước", "giường tủ", "rau dưa"), "T3", "V2", "flipped_verdict"),
)
assert len(CALIBRATION_SPECS) == C.VALIDATION_CALIBRATION_ITEMS

_EXPLAIN = {
    "none": ("Kết quả đúng quy tắc, đúng chính tả.", "The rule output, correctly spelled."),
    "wrong_variant": ("Đây là kết quả của một kiểu khác, không phải kiểu đã nêu → `correct` = Không.",
                      "This is the output of another kind, not the one named → `correct` = No."),
    "wrong_tone": ("Sai thanh điệu ở âm tiết đầu → `correct` = Không.", "Wrong tone on the first syllable → `correct` = No."),
    "misspell_k": ("Âm /k/ trước e, ê, i, y phải viết là k → sai chính tả: `correct` = Không, `spelling` = Không.",
                   "/k/ before e, ê, i, y is written k → misspelled: `correct` = No, `spelling` = No."),
    "placement_new": ("Dấu đặt theo kiểu mới (hoá) cũng đúng như kiểu cũ (hóa) → `correct` = Có, `spelling` = Có.",
                      "New-style mark placement (hoá) is as correct as old style (hóa) → `correct` = Yes, `spelling` = Yes."),
    "iy_variant": ("Viết i hay y sau phụ âm (tì/tỳ, lí/lý) đều được → `correct` = Có, `spelling` = Có.",
                   "i or y after a consonant (tì/tỳ, lí/lý) are both accepted → `correct` = Yes, `spelling` = Yes."),
    "vulgar": ("Đúng quy tắc nhưng kết quả có cách hiểu thô tục → `correct` = Có, `offensive` = Có.",
               "Rule-correct, but the output has a vulgar reading → `correct` = Yes, `offensive` = Yes."),
    "dialect_hoi_nga": (("Đúng quy tắc chính tả. Người nói phương ngữ không phân biệt hỏi/ngã có thể ghi `dialect` = hỏi=ngã; "
                         "`correct` vẫn chấm theo chữ viết chuẩn."),
                        ("Rule-correct in standard spelling. A speaker who merges hỏi/ngã may set `dialect` = hỏi=ngã; "
                         "`correct` is still judged on the standard spelling.")),
    "wrong_reading": ("Cụm được đề xuất không phải là cụm gốc theo kiểu nào cả → `correct` = Không.",
                      "The proposed original is not a reading under any kind → `correct` = No."),
    "spelling_twin": ("Ứng viên sai chính tả nên hệ thống trả lời Không là đúng → `correct` = Có, `spelling` = Không.",
                      "The candidate is misspelled, so the system's No is right → `correct` = Yes, `spelling` = No."),
    "flipped_verdict": ("Ứng viên đúng là kết quả, nhưng hệ thống trả lời Không → hệ thống sai: `correct` = Không.",
                        "The candidate is the right output but the system says No → the system is wrong: `correct` = No."),
}


def _misspell_k(text: str) -> str | None:
    words = text.split()
    for i, w in enumerate(words):
        if w.startswith("k") and not w.startswith("kh"):
            words[i] = "c" + w[1:]
            return " ".join(words)
    return None


def _to_new_placement(text: str) -> str:
    from noilai.vi.reencode import convert_placement
    return convert_placement(text, "new")


def _y_form(text: str) -> str | None:
    from noilai.vi import unicode as U
    words = text.split()
    for i, w in enumerate(words):
        p = try_parse(w, strict=False)
        if p and p.syllable.nucleus == "i" and not p.syllable.glide and not p.syllable.coda and p.syllable.onset not in ("", "s", "v"):
            base = U.strip_tones(w)
            if base.endswith("i"):
                words[i] = w[:-1] + U.compose_letter("y", "", p.syllable.tone)
                return " ".join(words)
    return None


def calibration_item(inp: str, task: str, variant: str, manip: str) -> dict | None:
    """One keyed calibration row computed by the rule engine, or None if the manipulation does not apply."""
    sy = phrase_syllables(inp)
    if sy is None or len(sy) != 2:
        return None
    a, b = sy
    out = V.apply(variant, a, b)
    if out in ((a, b), (b, a)):
        return None
    inv = L.load_inventory()
    if not all(inv.is_legal(s) for s in out):
        return None
    rule = spell_pair(out)
    key = {"correct": "yes", "spelling": "yes", "offensive": "no"}
    cand, verdict, shown_input = rule, "", inp
    if manip == "wrong_variant":
        others = [spell_pair(V.apply(v, a, b)) for v in ("V4", "V2", "V3", "V1") if v != variant]
        others = [o for o in others if canonical_text(o) != canonical_text(rule)]
        if not others:
            return None
        cand, key["correct"] = others[0], "no"
    elif manip == "wrong_tone":
        cand, key["correct"] = spell_pair([_wrong_tone(out[0]), out[1]]), "no"
    elif manip == "misspell_k":
        bad = _misspell_k(rule)
        if bad is None:
            return None
        cand, key["correct"], key["spelling"] = bad, "no", "no"
    elif manip == "placement_new":
        cand = _to_new_placement(rule)
        if cand == rule:
            return None
    elif manip == "iy_variant":
        alt = _y_form(rule)
        if alt is None:
            return None
        cand = alt
    elif manip == "vulgar":
        key["offensive"] = "yes"
    elif manip == "dialect_hoi_nga":
        if not any(s.tone in (3, 4) for s in out):
            return None
    if task == "T2":
        shown_input, cand = rule, inp                       # the nói lái form is shown, the original is the candidate
        if manip == "wrong_reading":
            cand, key["correct"] = spell_pair([_wrong_tone(a), b]), "no"
    elif task == "T3":
        verdict = "yes"
        if manip == "spelling_twin":
            bad = _misspell_k(rule)
            if bad is None:
                return None
            cand, verdict, key["spelling"] = bad, "no", "no"
        elif manip == "flipped_verdict":
            verdict, key["correct"] = "no", "no"
    q_vi, q_en = question_b(task, variant, shown_input, cand, verdict)
    return {"task": task, "kind": variant if task != "T2" else "", "input": shown_input, "candidate": cand,
            "system_verdict": verdict, "question_vi": q_vi, "question_en": q_en, "manipulation": manip,
            "base_phrase": inp, "key": key, "explanation_vi": _EXPLAIN[manip][0], "explanation_en": _EXPLAIN[manip][1]}


def build_calibration(release_keys: set | None = None, exclude_inputs: set | None = None, prefix: str = "A",
                      skip_unavailable: bool = False) -> list[dict]:
    """The calibration round: for each spec the first input that the manipulation fits and that is not a
    release phrase (in either order). Raises if a spec has no usable input (extend its list).

    The second calibration set (VALIDATION_PROTOCOL §5, for a validator below VALIDATION_CALIBRATION_PASS):
    `exclude_inputs` = the first round's inputs, `prefix` "A2", `skip_unavailable` True -- a spec with no other
    usable input (the single vulgar spec) is left out rather than repeated, and the row ids keep the spec number."""
    release_keys = release_keys or set()
    exclude = {phrase_key(x) for x in (exclude_inputs or ()) if phrase_key(x)}
    rows = []
    for j, (inputs, task, variant, manip) in enumerate(CALIBRATION_SPECS):
        for inp in inputs:
            if phrase_key(inp) in release_keys or phrase_key(inp) in exclude:
                continue
            row = calibration_item(inp, task, variant, manip)
            if row is None:
                continue
            if any(phrase_key(t) in release_keys for t in (row["input"], row["candidate"]) if phrase_key(t)):
                continue
            rows.append({"row_id": f"{prefix}-{j + 1:02d}", **row})
            break
        else:
            if skip_unavailable:
                continue
            raise ValueError(f"calibration spec {j + 1} ({task} {variant} {manip}): no usable input among {inputs}")
    return rows


# --------------------------------------------------------------------------- Part D: supplementary sheets
# D1: "which kind do you produce?" -- pairs with distinct onsets, rimes and tones so that the produced
# form identifies the kind (V1 and V4 differ only when tones differ, DESIGN_DECISIONS 3.5). [NATIVE-CHECK]
PRODUCTION_PAIRS = ("bánh chưng", "mặt trời", "sân bóng", "học trò", "bàn tay", "cửa sổ", "nước mắm", "giấy bút",
                    "quả táo", "đồng hồ", "lá cờ", "cái bàn", "con mèo", "bóng đèn", "ngủ trưa", "tiền lương",
                    "vườn rau", "mũ len", "đường phố", "thể thao", "điện thoại", "bạn bè", "nhà máy", "bếp lửa",
                    "trái cây", "xe đạp", "bút chì", "cơm tấm", "phở bò", "bánh xèo", "sữa chua", "mưa rào", "gió mùa",
                    "chợ búa", "cây đàn", "lửa trại", "cánh đồng", "núi rừng", "sông núi", "trường học")
# D2: qu- pairs (DESIGN_DECISIONS 2.1 O5 (b)): shown under the glide-in-rime analysis (the parser's) and
# the qu-as-onset analysis (school grammar). [NATIVE-CHECK]
QU_PAIRS = ("quả hồng", "quả cam", "cái quần", "quán cơm", "quốc gia", "quê nhà", "quyển sách", "quạt máy",
            "quà bánh", "quân đội", "quét nhà", "quên mất", "quý giá", "quyết định", "quanh năm", "quầy hàng",
            "quặng sắt", "quai nón", "quận huyện", "quỳ gối", "đá quý", "bánh quy")
CONVENTION_ITEMS_VI = (
    ("placement", "Bạn thường viết “hòa” hay “hoà”? (cả hai đều được chấp nhận trong bộ dữ liệu)", ("hòa", "hoà", "cả hai")),
    ("placement", "“khỏe” hay “khoẻ”?", ("khỏe", "khoẻ", "cả hai")),
    ("placement", "“thúy” hay “thuý”?", ("thúy", "thuý", "cả hai")),
    ("placement", "“họa sĩ” hay “hoạ sĩ”?", ("họa sĩ", "hoạ sĩ", "cả hai")),
    ("placement", "“lũy tre” hay “luỹ tre”?", ("lũy tre", "luỹ tre", "cả hai")),
    ("i/y", "“lí do” hay “lý do”?", ("lí do", "lý do", "cả hai")),
    ("i/y", "“kĩ sư” hay “kỹ sư”?", ("kĩ sư", "kỹ sư", "cả hai")),
    ("i/y", "“Mĩ” hay “Mỹ”?", ("Mĩ", "Mỹ", "cả hai")),
    ("i/y", "“tỉ lệ” hay “tỷ lệ”?", ("tỉ lệ", "tỷ lệ", "cả hai")),
    ("i/y", "“hi vọng” hay “hy vọng”?", ("hi vọng", "hy vọng", "cả hai")),
    ("i/y", "“bác sĩ” hay “bác sỹ”?", ("bác sĩ", "bác sỹ", "cả hai")),
    ("ay/ây", "“đôi giày” hay “đôi giầy”?", ("giày", "giầy", "cả hai")),
    ("oao", "Các chữ sau có phải là từ (hoặc tiếng) bạn từng gặp không: khoào, ngoao, ngoáo, quao, quào, quáo? Ghi từng chữ.", ()),
    ("gi", "“giặt gịa” hay “giặt giạ”? Và “giê”, “giề” có phải là tiếng bạn từng gặp không?", ()),
)


def qu_onset_analysis(variant: str, a: Syllable, b: Syllable) -> tuple[Syllable, Syllable]:
    """The output under the school analysis (qu = a 27th onset, its u not part of the rime). A qu-syllable is
    (c, glide=True, ...) in the parser; under the onset analysis its glide stays with the onset when rimes move."""
    def onset_part(s):
        return (s.onset, s.glide) if (s.onset == "c" and s.glide) else (s.onset, None)

    def rime_part(s):
        return (s.nucleus, s.coda) if (s.onset == "c" and s.glide) else (s.glide, s.nucleus, s.coda)

    def build(onset, rime, tone):
        o, g_onset = onset
        if len(rime) == 2:                        # a qu-rime moving: no glide of its own
            nuc, coda = rime
            g = False
        else:
            g, nuc, coda = rime
        if g_onset:                               # qu onset: the glide is written by the onset
            g = True
        return Syllable(o, g, nuc, coda, tone)

    moves = V.MOVES[variant]
    oa, ob = onset_part(a), onset_part(b)
    ra, rb = rime_part(a), rime_part(b)
    ta, tb = a.tone, b.tone
    if "onset" in moves:
        oa, ob = ob, oa
    if "rime" in moves:
        ra, rb = rb, ra
    if "tone" in moves:
        ta, tb = tb, ta
    return build(oa, ra, ta), build(ob, rb, tb)


def build_supplementary() -> dict:
    """D1 production pairs (verified discriminable by the engine), D2 qu- rows (both analyses), D3 conventions."""
    inv = L.load_inventory()
    d1 = []
    for ph in PRODUCTION_PAIRS:
        sy = phrase_syllables(ph)
        if not sy or len(sy) != 2:
            continue
        a, b = sy
        if a.onset == b.onset or a.rime == b.rime or a.tone == b.tone or (a.onset == "c" and a.glide) or (b.onset == "c" and b.glide):
            continue
        outs = {v: V.apply(v, a, b) for v in ("V1", "V2", "V4")}
        if len({canonical_text(spell_pair(o)) for o in outs.values()}) < 3:
            continue
        d1.append({"row_id": f"D1-{len(d1) + 1:02d}", "phrase": ph, "your_noilai": "", "comment": "",
                   "_engine": {v: spell_pair(o) for v, o in outs.items()}})
        if len(d1) == 20:
            break
    d2 = []
    for ph in QU_PAIRS:
        sy = phrase_syllables(ph)
        if not sy or len(sy) != 2:
            continue
        a, b = sy
        for variant in ("V1", "V4"):
            glide = V.apply(variant, a, b)
            onset = qu_onset_analysis(variant, a, b)
            sa, sb = spell_pair(glide), spell_pair(onset)
            if canonical_text(sa) == canonical_text(sb) or glide in ((a, b), (b, a)):
                continue
            d2.append({"row_id": f"D2-{len(d2) + 1:02d}", "phrase": ph, "kind": variant,
                       "option_A": sa, "option_B": sb, "choice": "", "your_form": "", "comment": "",
                       "_legal": {"A": all(inv.is_legal(s) for s in glide)}})
            break
        if len(d2) == 20:
            break
    d3 = [{"row_id": f"D3-{j + 1:02d}", "topic": topic, "question_vi": q, "options": " / ".join(opts), "answer": "", "comment": ""}
          for j, (topic, q, opts) in enumerate(CONVENTION_ITEMS_VI)]
    return {"D1": d1, "D2": d2, "D3": d3}


# --------------------------------------------------------------------------- Part C: attested rows
def build_part_c(attested_rows: Iterable[dict]) -> list[dict]:
    """One row per attested pair (seed rows and web-sourced candidates). Input keys: input, output, source."""
    out = []
    seen = set()
    for r in attested_rows:
        k = (canonical_text(r["input"]) or r["input"], canonical_text(r["output"]) or r["output"])
        if k in seen:
            continue
        seen.add(k)
        out.append({"row_id": f"C-{len(out) + 1:03d}", "input": r["input"], "output": r["output"],
                    "source": r.get("source", ""), "known": "", "valid": "", "spelling_ok": "", "your_form": "",
                    "offensive": "", "dialect": "", "comment": ""})
    return out


# --------------------------------------------------------------------------- Part E: T2 gold sets of the core
def build_part_e(core_items: Iterable[dict], validators: Sequence[str], seed: int, overlap: int = 50) -> dict:
    t2 = sorted((it for it in core_items if it.get("task") == "T2"), key=lambda x: x["item_id"])
    rng = random.Random(seed)
    rng.shuffle(t2)
    rows, key, sheets = [], {}, defaultdict(list)
    k = len(validators)
    for j, it in enumerate(t2):
        rid = f"E-{j + 1:04d}"
        readings = [g["output"] for g in it["gold"]]
        rows.append({"row_id": rid, "noilai_form": it["input"],
                     **{f"reading_{i + 1}": (readings[i] if i < len(readings) else "") for i in range(3)},
                     **{f"accept_{i + 1}": "" for i in range(3)}, "missing_reading": "", "comment": ""})
        who = list(validators) if j < overlap else [validators[(j - overlap) % k]]
        key[rid] = {"item_id": it["item_id"], "readings": readings, "validators": who}
        for v in who:
            sheets[v].append(rid)
    return {"rows": rows, "key": key, "sheets": dict(sheets)}


# --------------------------------------------------------------------------- scoring helpers
def triples(returned: dict[str, list[dict]], column: str, keep: set | None = None,
            unsure_as_missing: bool = True) -> list[tuple[str, str, str]]:
    """(row_id, validator, label) for one judgment column over the returned sheets."""
    out = []
    for v, rows in returned.items():
        for r in rows:
            if keep is not None and r["row_id"] not in keep:
                continue
            lab = norm_label(r.get(column))
            if lab is None or (unsure_as_missing and lab == "unsure"):
                continue
            out.append((r["row_id"], v, lab))
    return out


def resolve(labels: dict[str, str | None], author: str | None = None) -> tuple[str | None, str]:
    """The adjudication rule for one row's `correct` judgment.

    labels: {validator: yes/no/unsure/None}. The non-unsure labels decide: unanimous -> that label
    ('unanimous'); a strict majority of three or more -> majority ('majority'); otherwise the author's
    decision from the adjudication log if given ('author'), else unresolved (None, 'needs_author' when
    the validators split yes/no, 'unsure' when fewer than two definite labels exist)."""
    definite = [x for x in labels.values() if x in ("yes", "no")]
    c = Counter(definite)
    if len(definite) >= 2 and len(c) == 1:
        return definite[0], "unanimous"
    if len(definite) >= 3 and c.most_common(1)[0][1] > len(definite) / 2:
        return c.most_common(1)[0][0], "majority"
    if author in ("yes", "no"):                    # only where the validators did not decide: never overrides them
        return author, "author"
    if len(c) == 2:
        return None, "needs_author"
    return None, "unsure"


def stratified_precision(final: dict[str, str | None], key: dict) -> dict:
    """Generator precision per cell and pooled from the weighted probability sample (controls excluded,
    unresolved rows excluded and counted): p = sum_h W_h p_h with W_h the stratum's population share among
    the resolved strata of the cell; unweighted Wilson intervals beside the weighted estimate."""
    per_stratum: dict = defaultdict(lambda: [0, 0])
    unresolved = Counter()
    pop = {}
    for rid, k in key.items():
        if k["control"]:
            continue
        lab = final.get(rid)
        if lab is None:
            unresolved[k["cell"]] += 1
            continue
        s = per_stratum[k["stratum"]]
        s[0] += lab == "yes"
        s[1] += 1
        pop[k["stratum"]] = k["stratum_population"]
    cells: dict = defaultdict(dict)
    for st, (yes, n) in per_stratum.items():
        cell = "-".join(st.split("|")[:2])
        cells[cell][st] = (yes, n, pop[st])                 # N_h: the stratum's population in the release
    out, tot_yes, tot_n, tot_pop, tot_wsum = {}, 0, 0, 0.0, 0.0
    for cell in sorted(cells):
        sts = cells[cell]
        n_pop = sum(v[2] for v in sts.values())
        est = sum(v[2] / n_pop * (v[0] / v[1]) for v in sts.values())
        yes = sum(v[0] for v in sts.values())
        n = sum(v[1] for v in sts.values())
        lo, hi = wilson(yes, n)
        out[cell] = {"weighted_precision": est, "n": n, "n_yes": yes, "wilson_unweighted": [lo, hi],
                     "n_unresolved": unresolved.get(cell, 0)}
        tot_yes, tot_n = tot_yes + yes, tot_n + n
        tot_pop += n_pop
        tot_wsum += est * n_pop
    lo, hi = wilson(tot_yes, tot_n)
    out["pooled"] = {"weighted_precision": (tot_wsum / tot_pop) if tot_pop else float("nan"), "n": tot_n, "n_yes": tot_yes,
                     "wilson_unweighted": [lo, hi], "n_unresolved": sum(unresolved.values())}
    return out


def masi_distance(a: frozenset, b: frozenset) -> float:
    """Passonneau's MASI distance between two sets (1 - Jaccard x monotonicity)."""
    if not a and not b:
        return 0.0
    inter, union = len(a & b), len(a | b)
    jac = inter / union
    if a == b:
        m = 1.0
    elif a <= b or b <= a:
        m = 2 / 3
    elif inter:
        m = 1 / 3
    else:
        m = 0.0
    return 1 - jac * m


def jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1 - len(a & b) / len(a | b)


# --------------------------------------------------------------------------- scoring returned sheets
def agreement_block(rs_definite: list, rs_all: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """alpha (yes/no, unsure as missing; and with unsure as a category), AC1, raw agreement, marginals, CIs."""
    from noilai.stats.agreement import (
        ac1_bootstrap_ci,
        alpha_bootstrap_ci,
        gwet_ac1,
        krippendorff_alpha_nominal,
        marginal_distribution,
        percent_agreement,
    )
    if not rs_definite:
        return {"n_ratings": 0}
    if n_boot > 0:
        a, alo, ahi = alpha_bootstrap_ci(rs_definite, n_boot=n_boot, seed=seed)
        _, clo, chi = ac1_bootstrap_ci(rs_definite, n_boot=n_boot, seed=seed)
    else:                                                     # point estimates only (the adjudication-sheet pass)
        a, alo, ahi, clo, chi = krippendorff_alpha_nominal(rs_definite), None, None, None, None
    return {"alpha": a, "alpha_ci": [alo, ahi], "alpha_with_unsure_category": krippendorff_alpha_nominal(rs_all),
            "ac1": gwet_ac1(rs_definite), "ac1_ci": [clo, chi], "percent_agreement": percent_agreement(rs_definite),
            "marginals": marginal_distribution(rs_definite), "marginals_with_unsure": marginal_distribution(rs_all),
            "n_ratings": len(rs_definite), "n_ratings_with_unsure": len(rs_all)}


def score_part_b(returned: dict[str, list[dict]], key: dict, author_decisions: dict | None = None,
                 regions: dict | None = None, n_boot: int = 1000) -> dict:
    """Everything the protocol reports from the generated sheet. `returned` = {validator: rows};
    `author_decisions` = {row_id: yes/no} from the adjudication log; `regions` = {validator: N/C/S/abroad}."""
    author_decisions = author_decisions or {}
    regions = regions or {}
    by_row: dict = defaultdict(dict)
    for v, rows in returned.items():
        for r in rows:
            if r["row_id"] not in key:
                raise KeyError(f"sheet of {v}: unknown row {r['row_id']!r}")
            by_row[r["row_id"]][v] = r
    samples = {rid for rid, k in key.items() if not k["control"]}
    # primary agreement: every row, planted controls included (they put true negatives into the sheet, so alpha has
    # variance to measure); secondary: the probability sample alone (prevalence near 100%: read AC1 there)
    rep: dict = {"n_rows_returned": len(by_row), "n_rows_in_key": len(key), "agreement": {}, "agreement_samples_only": {}}
    for j in JUDGMENTS_B:
        rep["agreement"][j] = agreement_block(triples(returned, j), triples(returned, j, None, False), n_boot)
        rep["agreement_samples_only"][j] = agreement_block(triples(returned, j, samples), triples(returned, j, samples, False), n_boot)
    # controls: per-validator sensitivity (control rows answered `correct` = no) and the yes-rate on samples
    rep["validators"] = {}
    for v, rows in returned.items():
        ctl = [norm_label(r.get("correct")) for r in rows if key[r["row_id"]]["control"]]
        smp = [norm_label(r.get("correct")) for r in rows if not key[r["row_id"]]["control"]]
        caught = sum(x == "no" for x in ctl)
        answered = sum(x in ("yes", "no") for x in ctl)
        rep["validators"][v] = {"region": regions.get(v), "n_rows": len(rows),
                                "n_blank": sum(norm_label(r.get("correct")) is None for r in rows),
                                "controls_caught": caught, "controls_answered": answered,
                                "control_catch_rate": caught / answered if answered else None,
                                "control_catch_wilson": list(wilson(caught, answered)) if answered else None,
                                "sample_yes_rate": (sum(x == "yes" for x in smp) / max(1, sum(x in ("yes", "no") for x in smp))),
                                "unsure_rate": sum(x == "unsure" for x in smp + ctl) / max(1, len(rows))}
    # adjudication and final labels on `correct`
    final, how, adjud = {}, Counter(), []
    flags = {}
    for rid, k in key.items():
        # every validator who returned the row counts: the assigned two (or three) and, after the adjudication
        # round, the third validator who judged it blind (scripts/make_validation_forms.py adjudication-sheet)
        who = list(dict.fromkeys(list(k["validators"]) + sorted(by_row.get(rid, {}))))
        labs = {v: norm_label(by_row.get(rid, {}).get(v, {}).get("correct")) for v in who}
        lab, why = resolve(labs, author_decisions.get(rid))
        final[rid] = lab
        how[why] += 1
        if why in ("needs_author", "author"):
            comments = {v: by_row.get(rid, {}).get(v, {}).get("comment", "") for v in who}
            adjud.append({"row_id": rid, "item_id": k["item_id"], "cell": k["cell"], "control": k["control"],
                          "judgments": labs, "comments": comments, "decision": author_decisions.get(rid, ""),
                          "rule": why})
        rows_here = by_row.get(rid, {})
        offensive = any(norm_label(r.get("offensive")) == "yes" for r in rows_here.values())
        dialects = sorted({norm_dialect(r.get("dialect")) for r in rows_here.values()} - {"none"})
        rule_flag = any("rule?" in str(r.get("comment", "")).lower() for r in rows_here.values())
        flags[rid] = {"item_id": k["item_id"], "offensive_any": offensive, "dialect": dialects, "rule_flag": rule_flag}
    rep["adjudication"] = {"counts": dict(how), "rows": adjud}
    rep["generator_precision"] = stratified_precision(final, key)
    # how well the panel catches errors: final label on the controls
    ctl_final = [final[r] for r, k in key.items() if k["control"]]
    rep["controls_final"] = {"n": len(ctl_final), "labelled_no": sum(x == "no" for x in ctl_final),
                             "unresolved": sum(x is None for x in ctl_final)}
    rep["flags"] = {"offensive_rows": sorted(r for r, f in flags.items() if f["offensive_any"]),
                    "dialect_rows": {r: f["dialect"] for r, f in flags.items() if f["dialect"]},
                    "rule_flag_rows": sorted(r for r, f in flags.items() if f["rule_flag"])}
    rep["final"] = final
    rep["item_flags"] = flags
    return rep


def score_part_c(returned: dict[str, list[dict]], rows: list[dict], n_boot: int = 1000) -> dict:
    """Attested rows: per row the validators' `valid`, `known`, `spelling_ok`, `offensive`, `dialect`; a row
    is native-verified when at least two validators answer `valid` = yes and none answers no."""
    by_row: dict = defaultdict(dict)
    for v, rs in returned.items():
        for r in rs:
            by_row[r["row_id"]][v] = r
    out_rows = []
    for row in rows:
        got = by_row.get(row["row_id"], {})
        valid = {v: norm_label(r.get("valid")) for v, r in got.items()}
        yes = sum(x == "yes" for x in valid.values())
        no = sum(x == "no" for x in valid.values())
        out_rows.append({**{k: row[k] for k in ("row_id", "input", "output", "source")},
                         "valid": valid, "verified": yes >= 2 and no == 0,
                         "known_any": any(norm_label(r.get("known")) == "yes" for r in got.values()),
                         "spelling_disputed": any(norm_label(r.get("spelling_ok")) == "no" for r in got.values()),
                         "your_forms": sorted({r.get("your_form", "").strip() for r in got.values()} - {""}),
                         "offensive_any": any(norm_label(r.get("offensive")) == "yes" for r in got.values()),
                         "dialect": sorted({norm_dialect(r.get("dialect")) for r in got.values()} - {"none"}),
                         "validators": sorted(got)})
    keep = {r["row_id"] for r in rows}
    return {"agreement": {j: agreement_block(triples(returned, j, keep), triples(returned, j, keep, False), n_boot)
                          for j in JUDGMENTS_C},
            "n_rows": len(rows), "n_verified": sum(r["verified"] for r in out_rows), "rows": out_rows}


def score_part_e(returned: dict[str, list[dict]], key: dict, n_boot: int = 1000) -> dict:
    """T2 gold sets: each validator's accepted set per item (listed readings marked yes, plus a missing reading
    they added); MASI- and Jaccard-distance alpha over the overlap items; the validated gold set per item = the
    readings accepted by every validator who saw it, plus added readings (listed for the author to check)."""
    from noilai.stats.agreement import krippendorff_alpha
    sets: dict = defaultdict(dict)
    added: dict = defaultdict(set)
    for v, rs in returned.items():
        for r in rs:
            k = key.get(r["row_id"])
            if k is None:
                raise KeyError(f"sheet of {v}: unknown row {r['row_id']!r}")
            acc = set()
            answered = False
            for i, reading in enumerate(k["readings"][:3]):
                lab = norm_label(r.get(f"accept_{i + 1}"))
                answered |= lab is not None
                if lab == "yes":
                    acc.add(canonical_text(reading) or reading)
            extra = (r.get("missing_reading") or "").strip()
            if extra:
                c = canonical_text(extra) or extra
                acc.add(c)
                added[r["row_id"]].add(c)
            if answered or extra:
                sets[r["row_id"]][v] = frozenset(acc)
    ratings = [(rid, v, s) for rid, d in sets.items() for v, s in d.items()]
    validated = {}
    for rid, d in sets.items():
        if not d:
            continue
        inter = frozenset.intersection(*d.values())
        validated[key[rid]["item_id"]] = {"accepted_by_all": sorted(inter - added[rid]), "added": sorted(added[rid]),
                                          "n_validators": len(d)}
    return {"alpha_masi": krippendorff_alpha(ratings, masi_distance), "alpha_jaccard": krippendorff_alpha(ratings, jaccard_distance),
            "n_items_returned": len(sets), "n_items_multi_coded": sum(len(d) >= 2 for d in sets.values()),
            "validated_gold": validated}


def score_calibration(returned_rows: list[dict], calib: list[dict]) -> dict:
    """One validator's calibration sheet against the key: matches per judgment and the rows to discuss."""
    by_id = {r["row_id"]: r for r in calib}
    res = {"correct": [0, 0], "spelling": [0, 0], "offensive": [0, 0]}
    discuss = []
    for r in returned_rows:
        c = by_id.get(r["row_id"])
        if c is None:
            continue
        miss = []
        for j, counts in res.items():
            lab = norm_label(r.get(j))
            if lab is None:
                continue
            counts[1] += 1
            if lab == c["key"][j]:
                counts[0] += 1
            else:
                miss.append(j)
        if miss:
            discuss.append({"row_id": r["row_id"], "judgments": miss, "key": c["key"], "explanation_vi": c["explanation_vi"],
                            "explanation_en": c["explanation_en"]})
    return {"matches": {j: {"agree": a, "answered": n} for j, (a, n) in res.items()}, "to_discuss": discuss,
            "passes": res["correct"][1] > 0 and res["correct"][0] >= math.ceil(C.VALIDATION_CALIBRATION_PASS * res["correct"][1])}
