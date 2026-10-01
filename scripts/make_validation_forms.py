#!/usr/bin/env python
"""Sample the native-validation set and the human-baseline forms; score returned sheets.

Validation (plan 2.3): about 1,000 items stratified over task × variant × source × split,
each judged by two of three validators, with a 200-item overlap judged by all three
(the three-coder convention for Krippendorff's alpha). Four judgments per item:
  correct   the gold/candidate is the right nói lái of the input under the variant
  spelling  the output is spelled correctly (standard orthography)
  lexical   the output is a real Vietnamese word/phrase
  offensive the output has an offensive or vulgar reading
Judgments are 'yes' / 'no' / 'unsure'.

    python scripts/make_validation_forms.py sample --items data/release/v0.3/noilai_test.jsonl \
        --dev data/release/v0.3/noilai_dev.jsonl --out data/validation --n 1000 --overlap 200 --validators A B C
    python scripts/make_validation_forms.py score --out data/validation --returned data/validation/returned/*.csv

Human baseline (design 10.2, item 66): 20 respondents x 30 items = 6 ANCHOR items on every
form (one per T1 variant, one T2, one T3) + 240 items each on exactly two forms (20 per task x
variant cell; T3 half yes / half no, never both members of a twin pair, never the twin of an
anchor), drawn from the open-model main sample with `vulgar = false` and no spelling twins.
Assignment: cyclic double coverage -- the shuffled pool is read in blocks of n_forms, the first
copy of item k goes to form k % n_forms and the second to (k % n_forms + offset) % n_forms with
a block-specific offset 1 + (k // n_forms) % (n_forms - 1), so no form sees an item twice and
every pair of forms shares at least one item (the rater graph is connected: 140 pairs share
one item, 50 share two, with the defaults). The design generalizes: pool = n_forms x
(per_form - anchors) / 2 items, per_cell = pool / 12, both required to be whole numbers.

    python scripts/make_validation_forms.py baseline --items data/release/v0.3/noilai_main.jsonl --out data/human

Writes one CSV per form, `human_items.json` (the 246 ids every model is scored on) and
`baseline_manifest.json` (coverage: appearances per item, items shared per pair of forms).
The seed is a PUBLIC sampling seed (default 20261102), never the withheld build seed (4.6).

Forms are CSV files that import into Google Sheets/Forms; the sheets come back as CSV
with the same item_id column. No personal data is stored: validators are letters, and
the baseline form records only the coarse demographics of docs/DATA_STATEMENT.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai import constants as C
from noilai.gen.generate import load_items
from noilai.stats.agreement import (
    ac1_bootstrap_ci,
    alpha_bootstrap_ci,
    gwet_ac1,
    marginal_distribution,
    percent_agreement,
)
from noilai.stats.bootstrap import accuracy_ci

JUDGMENTS = ("correct", "spelling", "lexical", "offensive")
BASELINE_SEED = 20261102            # public sampling seed (docs/HUMAN_BASELINE_FORM.md); never the build seed (design 4.6)
BASELINE_CELLS = [(t, v) for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4")]
ANCHOR_CELLS = [("T1", "V1"), ("T1", "V2"), ("T1", "V3"), ("T1", "V4"), ("T2", "V1"), ("T3", "V1")]   # design 10.2: 6 anchors


def _display(it: dict) -> dict:
    if it["task"] == "T1":
        cand = it["gold"][0]
        q = f"Nói lái kiểu {it['variant']} của “{it['input']}” là “{cand}”?"
    elif it["task"] == "T2":
        cand = it["gold"][0]["output"]
        q = f"“{it['input']}” là cách nói lái của “{cand}”?"
    else:
        cand = it["candidate"]
        q = f"“{cand}” có phải là nói lái kiểu {it['variant']} của “{it['input']}” không? (đáp án hệ thống: {it['gold']})"
    return {"item_id": it["item_id"], "task": it["task"], "variant": it["variant"], "input": it["input"],
            "candidate": cand, "question_vi": q}


def cmd_sample(args) -> int:
    rng = random.Random(args.seed)
    items = load_items(Path(args.items)) + (load_items(Path(args.dev)) if args.dev else [])
    strata = defaultdict(list)
    for it in items:
        strata[(it["task"], it["variant"], it["source"], it["split"])].append(it)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    chosen = []
    i = 0
    while len(chosen) < args.n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            chosen.append(strata[k].pop())
        i += 1
    rng.shuffle(chosen)
    overlap = chosen[: args.overlap]
    rest = chosen[args.overlap:]
    vals = args.validators
    assign = defaultdict(list)
    for it in overlap:
        for v in vals:
            assign[v].append(it)
    for j, it in enumerate(rest):
        pair = [vals[j % len(vals)], vals[(j + 1) % len(vals)]]
        for v in pair:
            assign[v].append(it)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v, its in assign.items():
        rng.shuffle(its)
        with open(out / f"validation_form_{v}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "task", "variant", "input", "candidate", "question_vi", *JUDGMENTS, "comment"])
            w.writeheader()
            for it in its:
                w.writerow({**_display(it), **{j: "" for j in JUDGMENTS}, "comment": ""})
    meta = {"n_items": len(chosen), "overlap": len(overlap), "validators": vals, "per_validator": {v: len(a) for v, a in assign.items()},
            "strata": Counter(f"{k[0]}-{k[1]}-{k[2]}-{k[3]}" for k in ((it["task"], it["variant"], it["source"], it["split"]) for it in chosen)),
            "seed": args.seed}
    (out / "validation_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "strata"}, ensure_ascii=False))
    return 0


def cmd_score(args) -> int:
    ratings = defaultdict(list)     # judgment -> [(item, coder, label)]
    rows_by_item = defaultdict(dict)
    for path in args.returned:
        coder = Path(path).stem.split("_")[-1]
        with open(path, encoding="utf-8") as fh:
            rows_ = list(csv.DictReader(fh))
        for r in rows_:
            for j in JUDGMENTS:
                lab = (r.get(j) or "").strip().lower()
                if lab in ("yes", "no", "unsure", "có", "không"):
                    lab = {"có": "yes", "không": "no"}.get(lab, lab)
                    ratings[j].append((r["item_id"], coder, lab))
                    rows_by_item[r["item_id"]][(coder, j)] = lab
    report = {}
    for j in JUDGMENTS:
        rs = ratings[j]
        if not rs:
            continue
        est, lo, hi = alpha_bootstrap_ci(rs, n_boot=500)
        ac1_lo, ac1_hi = ac1_bootstrap_ci(rs, n_boot=500)[1:]
        # design 10.1 / 12.21: alpha beside raw agreement, Gwet's AC1 and the label marginals (at ~96% prevalence
        # alpha falls while two coders agree on 96% of items; AC1's chance term is small)
        report[j] = {"alpha": est, "alpha_ci": [lo, hi], "percent_agreement": percent_agreement(rs), "n_ratings": len(rs),
                     "ac1": gwet_ac1(rs), "ac1_ci": [ac1_lo, ac1_hi], "marginals": marginal_distribution(rs)}
    # generator precision: item counted correct when the majority of its validators say yes
    item_correct = []
    item_ids = []
    for item, labs in rows_by_item.items():
        votes = [v for (c, j), v in labs.items() if j == "correct"]
        if votes:
            yes = votes.count("yes")
            no = votes.count("no")
            if yes != no:
                item_correct.append(yes > no)
                item_ids.append(item)
    if item_correct:
        ci = accuracy_ci(item_correct, item_ids, n_boot=1000)
        report["generator_precision"] = ci.as_dict()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def baseline_prompt(it: dict) -> str:
    """[NATIVE-CHECK] the form's question per task (the wording of docs/HUMAN_BASELINE_FORM.md §1)."""
    if it["task"] == "T1":
        return f"Nói lái kiểu {it['variant']} của “{it['input']}” là gì?"
    if it["task"] == "T2":
        return f"“{it['input']}” là cách nói lái của cụm từ nào?"
    return f"“{it['candidate']}” có phải là nói lái kiểu {it['variant']} của “{it['input']}” không? (Có/Không)"


def baseline_design(items: list[dict], n_forms: int, per_form: int, seed: int, n_anchors: int = len(ANCHOR_CELLS),
                    per_cell: int | None = None) -> dict:
    """The design 10.2 assignment. Returns {'anchors', 'pool', 'forms' (lists of items, anchors first),
    'per_cell'}; raises ValueError when the sizes do not fit or a cell is short."""
    if n_forms < 2:
        raise ValueError("the double-coverage design needs at least two forms")
    if n_anchors != len(ANCHOR_CELLS):
        raise ValueError(f"design 10.2 fixes {len(ANCHOR_CELLS)} anchors (one per T1 variant, one T2, one T3)")
    slots = n_forms * (per_form - n_anchors)
    if slots <= 0 or slots % 2:
        raise ValueError(f"{n_forms} forms x ({per_form} - {n_anchors} anchors) = {slots} slots must be a positive even number")
    pool_size = slots // 2
    if per_cell is None:
        if pool_size % len(BASELINE_CELLS):
            raise ValueError(f"the {pool_size} double-judged items do not divide over the {len(BASELINE_CELLS)} task x variant cells")
        per_cell = pool_size // len(BASELINE_CELLS)
    elif per_cell * len(BASELINE_CELLS) != pool_size:
        raise ValueError(f"--per-cell {per_cell} x {len(BASELINE_CELLS)} cells != {pool_size} double-judged items")
    rng = random.Random(seed)
    items = [it for it in items if it.get("task") in ("T1", "T2", "T3") and not it.get("vulgar") and it.get("twin_type") != "spelling"]
    by_cell = defaultdict(list)
    for it in items:
        by_cell[(it["task"], it["variant"])].append(it)
    anchors, pool = [], []
    for cell in BASELINE_CELLS:
        c = sorted(by_cell.get(cell, []), key=lambda x: x["item_id"])
        rng.shuffle(c)
        if cell in ANCHOR_CELLS:
            if not c:
                raise ValueError(f"cell {cell} has no eligible item for its anchor")
            anchors.append(c.pop())
            c = [x for x in c if x.get("pair_item_id") != anchors[-1]["item_id"]]     # never the twin of an anchor
        if cell[0] == "T3":                                                            # half yes, half no, never both twins
            n_yes = (per_cell + 1) // 2
            yes = [x for x in c if x["gold"] == "yes"][:n_yes]
            taken = {y["item_id"] for y in yes}
            no = [x for x in c if x["gold"] == "no" and x.get("pair_item_id") not in taken][: per_cell - n_yes]
            picked = yes + no
        else:
            picked = c[:per_cell]
        if len(picked) < per_cell:
            raise ValueError(f"cell {cell} has {len(picked)} eligible items, {per_cell} needed")
        pool += picked
    rng.shuffle(pool)
    forms = [list(anchors) for _ in range(n_forms)]
    for k, it in enumerate(pool):                       # first copy: round robin
        forms[k % n_forms].append(it)
    for k, it in enumerate(pool):                       # second copy: block-specific cyclic offset, never 0
        offset = 1 + (k // n_forms) % (n_forms - 1)
        forms[(k % n_forms + offset) % n_forms].append(it)
    for fm in forms:
        assert len(fm) == per_form and len({x["item_id"] for x in fm}) == per_form
    return {"anchors": anchors, "pool": pool, "forms": forms, "per_cell": per_cell}


def cmd_baseline(args) -> int:
    rng = random.Random(args.seed)
    design = baseline_design(load_items(Path(args.items)), args.n_forms, args.per_form, args.seed, args.anchors, args.per_cell)
    anchors, pool, forms = design["anchors"], design["pool"], design["forms"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fi, its in enumerate(forms):
        its = list(its)
        rng.shuffle(its)                                # seeded random order per form (T1, T2, T3 interleaved)
        with open(out / f"baseline_form_{fi+1:02d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "task", "variant", "prompt_vi", "answer"])
            w.writeheader()
            for it in its:
                w.writerow({"item_id": it["item_id"], "task": it["task"], "variant": it["variant"],
                            "prompt_vi": baseline_prompt(it), "answer": ""})
    cover = Counter(it["item_id"] for fm in forms for it in fm)
    anchor_ids = {a["item_id"] for a in anchors}
    seen = [{x["item_id"] for x in fm} - anchor_ids for fm in forms]
    shared = Counter(len(seen[i] & seen[j]) for i in range(len(forms)) for j in range(i + 1, len(forms)))
    info = {"forms": args.n_forms, "per_form": args.per_form, "anchors": len(anchors), "per_cell": design["per_cell"],
            "double_judged": len(pool), "distinct_items": len(cover),
            "min_appearances": min(cover.values()), "max_appearances": max(cover.values()),
            "anchors_seen_by": min(cover[a] for a in anchor_ids),
            "others": sorted({v for k, v in cover.items() if k not in anchor_ids}),
            "shared_items_per_form_pair": {str(k): v for k, v in sorted(shared.items())},
            "rater_graph_connected": 0 not in shared, "seed": args.seed, "items_file": str(args.items),
            "design": "DESIGN_DECISIONS 10.2: anchors on every form, every other item on exactly two forms, cyclic double coverage"}
    (out / "human_items.json").write_text(json.dumps(sorted(cover), ensure_ascii=False), encoding="utf-8")
    (out / "baseline_manifest.json").write_text(json.dumps({**info, "anchor_ids": sorted(anchor_ids),
                                                            "appearances": dict(sorted(cover.items()))}, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--items", required=True)
    s.add_argument("--dev", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--overlap", type=int, default=200)
    s.add_argument("--validators", nargs="+", default=["A", "B", "C"])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("score")
    c.add_argument("--out", required=True)
    c.add_argument("--returned", nargs="+", required=True)
    c.set_defaults(func=cmd_score)
    b = sub.add_parser("baseline")
    b.add_argument("--items", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-forms", type=int, default=C.HUMAN_BASELINE_PEOPLE)
    b.add_argument("--per-form", type=int, default=C.HUMAN_BASELINE_ITEMS_PER_PERSON)
    b.add_argument("--anchors", type=int, default=C.HUMAN_BASELINE_ANCHORS)
    b.add_argument("--per-cell", type=int, default=None, help="double-judged items per task x variant cell (derived when omitted)")
    b.add_argument("--seed", type=int, default=BASELINE_SEED)
    b.set_defaults(func=cmd_baseline)
    args = ap.parse_args(argv)
    args.returned = [p for pat in getattr(args, "returned", []) for p in glob.glob(pat)] if hasattr(args, "returned") else None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
