#!/usr/bin/env python
"""Turn data/attested_seed.tsv into the release file attested.jsonl.

For every seed row the rule engine is applied to the input with the stated variant; the
release row records the attested output, the rule's output, whether they agree after
canonicalization, the parsed structures, glosses, the vulgar flag and the verification
status. Rows whose input or output does not parse are kept with `parse_ok = false` so
that nothing is silently dropped; they are excluded from prompts until fixed.

    python scripts/build_attested.py --out data/release/v0.1/attested.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.gen import variants as V
from noilai.gen.generate import syl_dict
from noilai.vi import lexicon as L
from noilai.vi.reencode import canonical_text
from noilai.vi.syllable import spell, try_parse


def candidate_outputs(inp_sylls, variant):
    """All rule outputs: two syllables -> the variant; three syllables -> the variant applied to
    one pair of positions with the third fixed, for each of the three position pairs (outer
    pair first, the folk default), each also in swapped order. Returns [(label, syllables)]."""
    outs = []
    if len(inp_sylls) == 2:
        x, y = V.apply(variant, inp_sylls[0], inp_sylls[1])
        outs.append(("0-1", (x, y)))
        outs.append(("0-1 reversed", (y, x)))
    elif len(inp_sylls) == 3:
        for i, j in ((0, 2), (1, 2), (0, 1)):
            x, y = V.apply(variant, inp_sylls[i], inp_sylls[j])
            o = list(inp_sylls)
            o[i], o[j] = x, y
            outs.append((f"{i}-{j}", tuple(o)))
            o2 = list(inp_sylls)
            o2[i], o2[j] = y, x
            outs.append((f"{i}-{j} reversed", tuple(o2)))
    return outs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", type=Path, default=ROOT / "data" / "attested_seed.tsv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    inv = L.load_inventory()
    rows = list(csv.DictReader(open(args.seed_file, encoding="utf-8"), delimiter="\t"))
    rows = [{k: (v or "") for k, v in r.items() if k} for r in rows]   # a trailing empty column reads as None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_match = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            inp = [try_parse(w, strict=False) for w in r["input"].split()]
            out = [try_parse(w, strict=False) for w in r["output"].split()]
            parse_ok = all(inp) and all(out) and len(inp) == len(out)
            rule_out = None
            rule_matches = None
            matched_positions = None
            if parse_ok:
                sy = [p.syllable for p in inp]
                cands = candidate_outputs(sy, r["variant"])
                if cands:
                    rule_out = " ".join(spell(s) for s in cands[0][1])       # the default (outer pair, given order)
                    target = canonical_text(r["output"])
                    for label, o in cands:
                        if canonical_text(" ".join(spell(s) for s in o)) == target:
                            matched_positions = label
                            break
                    rule_matches = matched_positions is not None
                n_ok += 1
                n_match += bool(rule_matches)
            row = {
                "item_id": f"ATT-{i+1:04d}", "task": "attested", "variant": r["variant"],
                "input": canonical_text(r["input"]), "attested_output": canonical_text(r["output"]),
                "rule_output": rule_out, "rule_matches_attested": rule_matches, "matched_positions": matched_positions,
                "gold": sorted({canonical_text(r["output"])} | ({canonical_text(rule_out)} if rule_out else set())),
                "input_syllables": [syl_dict(p.syllable) for p in inp] if all(inp) else None,
                "gloss_input": r["gloss_input"], "gloss_output": r["gloss_output"], "note": r["note"],
                "confidence": r["confidence"], "vulgar": r["vulgar"].strip().lower() == "yes",
                "verified_by": r["verified_by"].strip() or None, "parse_ok": parse_ok,
                "source": "attested", "base_pair_id": f"att-{canonical_text(r['input']).replace(' ', '_')}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} seed rows, {n_ok} parse, {n_match} where the rule reproduces the attested output -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
