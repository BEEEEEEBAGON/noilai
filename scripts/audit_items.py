#!/usr/bin/env python
"""Annotate an item file with per-syllable tokenization covariates for one tokenizer.

    python scripts/audit_items.py --items data/release/v0.1/noilai_test.jsonl \
        --spm data/external/gemma3_tokenizer.model:gemma3 --out data/audit/items_gemma3.jsonl
    python scripts/audit_items.py --items ... --hf Qwen/Qwen2.5-7B-Instruct --out ...

For every item and every encoding arm the E2 regression needs: tokens per syllable of the
input (mean and per syllable), boundary alignment, whether any input syllable is a single
token, the token count of the gold answer (T1) and of the candidate (T3), and the change
in input token count between NFC and NFD (Δtok for the decomposition in
noilai.stats.mediation). The output has one row per (item_id, encoding) and is joined to
scores by item_id at analysis time; nothing here depends on any model output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import HFAdapter, SentencePieceAdapter, audit_syllable  # noqa: E402
from noilai.gen.generate import load_items  # noqa: E402


def phrase_stats(adapter, phrase: str, encoding: str) -> dict:
    per = []
    align = []
    for w in phrase.split():
        a = audit_syllable(adapter, w, encoding)
        if a is None:
            per.append(None)
            continue
        per.append(a.n_tokens)
        align.append(a.boundary_alignment)
    known = [x for x in per if x is not None]
    return {"tokens_per_syllable": per, "tokens_total": sum(known), "tokens_mean": mean(known) if known else None,
            "boundary_alignment_mean": mean(align) if align else None, "any_single_token": any(x == 1 for x in known),
            "all_single_token": bool(known) and all(x == 1 for x in known)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--spm", default=None, help="path[:name]")
    ap.add_argument("--hf", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encodings", nargs="*", default=["nfc", "nfd"])
    args = ap.parse_args(argv)
    if args.spm:
        path, _, name = args.spm.partition(":")
        adapter = SentencePieceAdapter(path, name or None)
    elif args.hf:
        adapter = HFAdapter(args.hf)
    else:
        ap.error("--spm or --hf required")
    items = load_items(args.items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            rows = {}
            for enc in args.encodings:
                row = {"item_id": it["item_id"], "tokenizer": adapter.name, "encoding": enc,
                       "input": phrase_stats(adapter, it["input"], enc)}
                if it["task"] == "T1":
                    row["gold"] = phrase_stats(adapter, it["gold"][0], enc)
                elif it["task"] == "T3":
                    row["candidate"] = phrase_stats(adapter, it["candidate"], enc)
                rows[enc] = row
            if "nfc" in rows and "nfd" in rows:
                rows["nfd"]["delta_tokens_vs_nfc"] = rows["nfd"]["input"]["tokens_total"] - rows["nfc"]["input"]["tokens_total"]
                rows["nfc"]["delta_tokens_vs_nfc"] = 0
            for row in rows.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} rows for {len(items)} items with tokenizer {adapter.name} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
