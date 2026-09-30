#!/usr/bin/env python
"""Annotate an item file with per-syllable tokenization covariates for one tokenizer.

    python scripts/audit_items.py --items data/release/v0.2/noilai_test.jsonl \
        --spm data/external/gemma3_tokenizer.model:gemma3 --out data/audit/items_gemma3.jsonl
    python scripts/audit_items.py --items ... --hf Qwen/Qwen2.5-7B-Instruct --out ...

For every item and every encoding arm the E2 regression needs (design 4.3, 8.4): tokens
per syllable of the input (mean and per syllable), `align_among_split` (share of aligned
boundaries among SPLIT syllables, None when no syllable is split — the E2 `align_w` source;
the legacy `boundary_alignment_mean` averages over all syllables with single-token
syllables scored 1.0 and is kept for readers only), the per-syllable alignment and the
count of misaligned split syllables (the H1 identifiability floor, design section 1),
`tone_isolated` under NFD and `byte_fallback`, whether any input syllable is a single
token, the token count of the gold answer (T1) and of the candidate (T3), and the change in
input token count between NFC and NFD (Δtok for the dose–response / zero-dose contrast of
design 8.6, `noilai.stats.dose_response`). The output has one row per (item_id, encoding)
and is joined to scores by item_id at analysis time (`flatten_for_e2` gives the analysis
row `noilai.stats.e2.prepare_covariates` consumes); nothing here depends on any model output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import HFAdapter, SentencePieceAdapter, audit_syllable
from noilai.gen.generate import load_items


def phrase_stats(adapter, phrase: str, encoding: str) -> dict:
    per = []
    align = []
    audits = []
    for w in phrase.split():
        a = audit_syllable(adapter, w, encoding)
        if a is None:
            per.append(None)
            align.append(None)
            continue
        audits.append(a)
        per.append(a.n_tokens)
        align.append(a.boundary_alignment)
    known = [x for x in per if x is not None]
    known_align = [x for x in align if x is not None]
    split_align = [a.boundary_alignment for a in audits if a.n_tokens > 1]
    return {"tokens_per_syllable": per, "tokens_total": sum(known), "tokens_mean": mean(known) if known else None,
            "align_among_split": mean(split_align) if split_align else None,     # design 8.4: align_w's source; None -> 0
            "alignment_per_syllable": align,
            "n_split": len(split_align),
            "n_misaligned_split": sum(1 for x in split_align if x < 1.0),        # design section 1: identifiability floor
            "boundary_alignment_mean": mean(known_align) if known_align else None,   # legacy: single-token syllables scored 1.0
            "tone_isolated": any(a.tone_isolated for a in audits),
            "byte_fallback": any(a.byte_fallback for a in audits),
            "any_single_token": any(x == 1 for x in known),
            "all_single_token": bool(known) and all(x == 1 for x in known)}


def flatten_for_e2(row: dict, field: str = "input") -> dict:
    """The analysis row for one (item, encoding) audit row, in the shape
    `noilai.stats.e2.prepare_covariates` consumes: `tokens_per_syllable` = the item's mean
    tokens per syllable, `align_among_split` (None when no syllable is split), the
    identifiability count and the arm-level covariates."""
    st = row[field]
    return {"item_id": row["item_id"], "tokenizer": row["tokenizer"], "encoding": row["encoding"],
            "tokens_per_syllable": st["tokens_mean"], "align_among_split": st["align_among_split"],
            "n_split": st["n_split"], "n_misaligned_split": st["n_misaligned_split"],
            "tone_isolated": st["tone_isolated"], "byte_fallback": st["byte_fallback"],
            "any_single_token": st["any_single_token"], "all_single_token": st["all_single_token"],
            "delta_tokens_vs_nfc": row.get("delta_tokens_vs_nfc")}


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
