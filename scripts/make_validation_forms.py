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

    python scripts/make_validation_forms.py sample --items data/release/v0.1/noilai_test.jsonl \
        --dev data/release/v0.1/noilai_dev.jsonl --out data/validation --n 1000 --overlap 200 --validators A B C
    python scripts/make_validation_forms.py score --out data/validation --returned data/validation/returned/*.csv

Human baseline: about 20 forms of 40 T1/T2/T3 items each (20 minutes), balanced over
variant, drawn from the core set; every item appears on at least two forms.

    python scripts/make_validation_forms.py baseline --items data/release/v0.1/noilai_core.jsonl --out data/human --n-forms 20 --per-form 40

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

from noilai.gen.generate import load_items  # noqa: E402
from noilai.stats.agreement import alpha_bootstrap_ci, krippendorff_alpha_nominal, percent_agreement  # noqa: E402
from noilai.stats.bootstrap import accuracy_ci  # noqa: E402

JUDGMENTS = ("correct", "spelling", "lexical", "offensive")


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
    items = [it for it in items if not (it["task"] == "T3" and it["gold"] == "no" and it.get("twin_type") == "spelling" and False)]
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
            "strata": Counter(f"{k[0]}-{k[1]}-{k[2]}-{k[3]}" for k in (tuple((it["task"], it["variant"], it["source"], it["split"])) for it in chosen)),
            "seed": args.seed}
    (out / "validation_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "strata"}, ensure_ascii=False))
    return 0


def cmd_score(args) -> int:
    ratings = defaultdict(list)     # judgment -> [(item, coder, label)]
    rows_by_item = defaultdict(dict)
    for path in args.returned:
        coder = Path(path).stem.split("_")[-1]
        for r in csv.DictReader(open(path, encoding="utf-8")):
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
        report[j] = {"alpha": est, "alpha_ci": [lo, hi], "percent_agreement": percent_agreement(rs), "n_ratings": len(rs)}
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


def cmd_baseline(args) -> int:
    rng = random.Random(args.seed)
    items = [it for it in load_items(Path(args.items)) if it["task"] in ("T1", "T2", "T3")]
    by_cell = defaultdict(list)
    for it in items:
        by_cell[(it["task"], it["variant"])].append(it)
    for v in by_cell.values():
        rng.shuffle(v)
    forms = [[] for _ in range(args.n_forms)]
    cells = sorted(by_cell)
    ptr = {c: 0 for c in cells}
    for fi in range(args.n_forms):
        for k in range(args.per_form):
            c = cells[k % len(cells)]
            pool = by_cell[c]
            forms[fi].append(pool[ptr[c] % len(pool)])
            ptr[c] += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fi, its in enumerate(forms):
        with open(out / f"baseline_form_{fi+1:02d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "task", "variant", "prompt_vi", "answer"])
            w.writeheader()
            for it in its:
                if it["task"] == "T1":
                    q = f"Nói lái kiểu {it['variant']} của “{it['input']}” là gì?"
                elif it["task"] == "T2":
                    q = f"“{it['input']}” là cách nói lái của cụm từ nào?"
                else:
                    q = f"“{it['candidate']}” có phải là nói lái kiểu {it['variant']} của “{it['input']}” không? (Có/Không)"
                w.writerow({"item_id": it["item_id"], "task": it["task"], "variant": it["variant"], "prompt_vi": q, "answer": ""})
    cover = Counter(it["item_id"] for fm in forms for it in fm)
    print(json.dumps({"forms": args.n_forms, "per_form": args.per_form, "distinct_items": len(cover),
                      "min_appearances": min(cover.values()), "max_appearances": max(cover.values())}))
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
    b.add_argument("--n-forms", type=int, default=20)
    b.add_argument("--per-form", type=int, default=40)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_baseline)
    args = ap.parse_args(argv)
    args.returned = [p for pat in getattr(args, "returned", []) for p in glob.glob(pat)] if hasattr(args, "returned") else None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
