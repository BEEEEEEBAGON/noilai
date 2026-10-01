#!/usr/bin/env python
"""Audit tokenizers on the Vietnamese syllable inventory and a phrase sample.

    python scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 --out data/audit
    python scripts/audit_tokenizers.py --hf Qwen/Qwen2.5-7B-Instruct google/gemma-3-4b-it --out data/audit

Writes <out>/<name>.json (summary + the three-valued normalization census of design 6.2 on
the fixed 1,000-string probe set: 500 multi-syllable words + 500 NóiLái item inputs from
`--items`, seeded) and <out>/<name>_rows.csv. Hugging Face tokenizers need hub access
(Kaggle/Colab).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import (
    HFAdapter,
    SentencePieceAdapter,
    audit_inventory,
    census_probe_set,
    normalization_census,
)
from noilai.vi import lexicon as L

DEFAULT_ITEMS = ROOT / "data" / "release" / "v0.2" / "noilai_main.jsonl"


def run(adapter, out: Path, syllables, phrases, probe_set_source: dict | None = None):
    out.mkdir(parents=True, exist_ok=True)
    res = audit_inventory(adapter, syllables)
    res["normalization_census"] = normalization_census(adapter, phrases)
    res["normalization_census"]["probe_set"] = probe_set_source or {"n": len(phrases)}
    rows = res.pop("rows")
    with open(out / f"{adapter.name}_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["tokens"] = "|".join(r["tokens"])
            r["boundaries"] = "|".join(str(b) for b in r["boundaries"])
            w.writerow(r)
    with open(out / f"{adapter.name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({"tokenizer": adapter.name, **res["summary"], "census": res["normalization_census"]}, ensure_ascii=False, indent=1))


def _relative(path: Path | None) -> str | None:
    """The probe set's item file as a repository-relative path (the JSON is committed; an absolute
    path would name the build machine)."""
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm", nargs="*", default=[], help="path[:name] of SentencePiece models")
    ap.add_argument("--hf", nargs="*", default=[], help="Hugging Face tokenizer ids")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit")
    ap.add_argument("--max-syllables", type=int, default=0)
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS if DEFAULT_ITEMS.exists() else None,
                    help="release item file whose inputs give the 500 NóiLái strings of the census probe set (design 6.2)")
    ap.add_argument("--census-seed", type=int, default=0)
    args = ap.parse_args(argv)
    syllables = sorted({s for s in L.load_hunspell_syllables("new")})
    if args.max_syllables:
        syllables = syllables[: args.max_syllables]
    words = L.load_words()
    item_texts: list[str] = []
    if args.items is not None:
        from noilai.gen.generate import load_items

        item_texts = [it["input"] for it in load_items(args.items)]
    phrases = census_probe_set(words, item_texts, seed=args.census_seed)
    n_words = min(500, len({w for w in words if " " in w}))
    source = {"n": len(phrases), "n_words": n_words, "n_items": len(phrases) - n_words,
              "items_file": _relative(args.items), "seed": args.census_seed,
              "rule": "500 multi-syllable words + 500 NóiLái item inputs, seeded (design 6.2)"}
    for spec in args.spm:
        path, _, name = spec.partition(":")
        run(SentencePieceAdapter(path, name or None), args.out, syllables, phrases, source)
    for name in args.hf:
        run(HFAdapter(name), args.out, syllables, phrases, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
