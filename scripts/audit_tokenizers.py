#!/usr/bin/env python
"""Audit tokenizers on the Vietnamese syllable inventory and a phrase sample.

    python scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 --out data/audit
    python scripts/audit_tokenizers.py --hf Qwen/Qwen2.5-7B-Instruct google/gemma-3-4b-it --out data/audit

Writes <out>/<name>.json (summary + normalization census + per-syllable rows) and
<out>/<name>_rows.csv. Hugging Face tokenizers need hub access (Kaggle/Colab).
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
    normalization_census,
)
from noilai.vi import lexicon as L


def run(adapter, out: Path, syllables, phrases):
    out.mkdir(parents=True, exist_ok=True)
    res = audit_inventory(adapter, syllables)
    res["normalization_census"] = normalization_census(adapter, phrases)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm", nargs="*", default=[], help="path[:name] of SentencePiece models")
    ap.add_argument("--hf", nargs="*", default=[], help="Hugging Face tokenizer ids")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit")
    ap.add_argument("--max-syllables", type=int, default=0)
    args = ap.parse_args(argv)
    syllables = sorted({s for s in L.load_hunspell_syllables("new")})
    if args.max_syllables:
        syllables = syllables[: args.max_syllables]
    words = L.load_words()
    phrases = [w for w in words if " " in w][:2000:4]  # 500 multi-syllable words for the census
    for spec in args.spm:
        path, _, name = spec.partition(":")
        run(SentencePieceAdapter(path, name or None), args.out, syllables, phrases)
    for name in args.hf:
        run(HFAdapter(name), args.out, syllables, phrases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
