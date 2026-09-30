#!/usr/bin/env python
"""Turn data/attested_seed.tsv into the release file attested.jsonl (design 4.7).

Seed columns: input, output, variant (V1-V6), positions ("0-1", "1-2", "0-2", optionally
"reversed"), exactness (exact | approx(<merger>)), region_tag (N/C/S or empty),
gloss_input, gloss_output, note, confidence, vulgar, source, verified_by.

For every row the rule engine is applied to the declared positions; the release row
records the attested output, the engine's output, whether they agree after
canonicalization, every position pair and order that reproduces the attested form (for
the record), the parsed structures, and the flags that decide where the row may be used:
only `exact` two-syllable rows enter T1/T3-style scoring and H6; every row enters T2 with
the attested gold; vulgar rows stay in the gated split. Rows whose input or output does
not parse are kept with `parse_ok = false` so that nothing is silently dropped.

    python scripts/build_attested.py --out data/release/v0.2/attested.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.gen import variants as V  # noqa: E402
from noilai.gen.generate import STYLE, syl_dict  # noqa: E402
from noilai.vi import lexicon as L  # noqa: E402
from noilai.vi.reencode import canonical_text  # noqa: E402
from noilai.vi.syllable import try_parse  # noqa: E402

POSITION_PAIRS_3 = ((0, 2), (1, 2), (0, 1))


def apply_at(sylls, variant: str, i: int, j: int, reverse: bool):
    x, y = V.apply(variant, sylls[i], sylls[j])
    o = list(sylls)
    if reverse:
        o[i], o[j] = y, x
    else:
        o[i], o[j] = x, y
    return tuple(o)


def parse_positions(spec: str, n: int) -> tuple[int, int, bool]:
    spec = (spec or "").strip()
    reverse = "reversed" in spec
    core = spec.replace("reversed", "").strip()
    if core:
        i, j = (int(x) for x in core.split("-"))
    else:
        i, j = (0, 1) if n == 2 else (0, 2)
    return i, j, reverse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", type=Path, default=ROOT / "data" / "attested_seed.tsv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    inv = L.load_inventory()
    with open(args.seed_file, encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in r.items() if k} for r in csv.DictReader(f, delimiter="\t")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_match = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            inp = [try_parse(w, strict=False) for w in r["input"].split()]
            out = [try_parse(w, strict=False) for w in r["output"].split()]
            parse_ok = all(inp) and all(out) and len(inp) == len(out) and len(inp) in (2, 3)
            rule_out = None
            rule_matches = None
            matches = []
            pi = pj = None
            reverse = False
            if parse_ok:
                sy = [p.syllable for p in inp]
                pi, pj, reverse = parse_positions(r.get("positions", ""), len(sy))
                rule_out = L.emit_phrase(apply_at(sy, r["variant"], pi, pj, reverse), STYLE, inv)
                target = canonical_text(r["output"])
                rule_matches = canonical_text(rule_out) == target
                pairs = POSITION_PAIRS_3 if len(sy) == 3 else ((0, 1),)
                for (a, b) in pairs:
                    for v in V.ALL_VARIANTS:
                        for rev in (False, True):
                            if canonical_text(L.emit_phrase(apply_at(sy, v, a, b, rev), STYLE, inv)) == target:
                                matches.append(f"{v}@{a}-{b}{' reversed' if rev else ''}")
                n_ok += 1
                n_match += bool(rule_matches)
            exactness = r.get("exactness", "") or ("exact" if rule_matches else "approx(unspecified)")
            row = {
                "item_id": f"ATT-{i+1:04d}", "task": "attested", "declared_variant": r["variant"],
                "variant": (sorted(m.split("@")[0] for m in matches if "reversed" not in m)
                            or sorted(m.split("@")[0] for m in matches) or [r["variant"]])[0],
                "variant_labels": sorted({m.split("@")[0] for m in matches}),
                "declared_matches_engine": bool(matches) and any(m.startswith(r["variant"] + "@") for m in matches),
                "positions": f"{pi}-{pj}{' reversed' if reverse else ''}" if pi is not None else None,
                "input": canonical_text(r["input"]), "attested_output": canonical_text(r["output"]),
                "rule_output": canonical_text(rule_out) if rule_out else None, "rule_matches_attested": rule_matches,
                "reproducing_labels": matches,
                "exactness": exactness, "exact": exactness == "exact",
                "eligible_h6": exactness == "exact" and len(inp) == 2 and bool(rule_matches),
                "gold": sorted({canonical_text(r["output"])} | ({canonical_text(rule_out)} if rule_out else set())),
                "input_syllables": [syl_dict(p.syllable) for p in inp] if all(inp) else None,
                "n_syllables": len(inp),
                "region_tag": r.get("region_tag", "") or None,
                "gloss_input": r["gloss_input"], "gloss_output": r["gloss_output"], "note": r["note"],
                "confidence": r["confidence"], "vulgar": r["vulgar"].strip().lower() == "yes",
                "source": r.get("source", ""), "verified_by": r["verified_by"].strip() or None, "parse_ok": parse_ok,
                "base_pair_id": f"att-{canonical_text(r['input']).replace(' ', '_')}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} seed rows, {n_ok} parse, {n_match} where the rule reproduces the attested output -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
