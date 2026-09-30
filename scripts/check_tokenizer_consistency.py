#!/usr/bin/env python
"""Check that a model's Hugging Face tokenizer agrees with a SentencePiece model file.

The tokenizer audit in data/audit was computed from the Gemma tokenizer FILES fetched
from google/gemma_pytorch. Before any model result is joined to those statistics, run
this on the machine that has hub access:

    python scripts/check_tokenizer_consistency.py --hf google/gemma-3-1b-it --spm data/external/gemma3_tokenizer.model

It tokenizes the syllable inventory (both placement conventions, so the 69 syllable types
whose spelling differs are probed in the release's old style too) and a phrase sample with
both, asserts the TOKEN IDS equal string by string (design 6.2, item 67; HFAdapter encodes
without special tokens, so no BOS offset exists), and writes a JSON report next to the
audit. Exit code 1 on ANY disagreement; the piece sequences are kept in the disagreement
records as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import HFAdapter, SentencePieceAdapter
from noilai.vi import lexicon as L
from noilai.vi import unicode as U


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit")
    ap.add_argument("--max", type=int, default=3000)
    args = ap.parse_args(argv)
    hf = HFAdapter(args.hf)
    sp = SentencePieceAdapter(args.spm)
    strings = probe_strings(args.max)
    report = compare_adapters(hf, sp, strings)
    report.update({"hf": args.hf, "spm": args.spm, "hf_normalizer": hf.normalizer_info(), "spm_normalizer": sp.normalizer_info()})
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.hf.replace("/", "__")
    (args.out / f"consistency_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, ensure_ascii=False, indent=1))
    return 0 if report["ids_equal"] else 1


def probe_strings(max_syllables: int = 3000) -> list[str]:
    """Syllables in both placement conventions (deduplicated), 500 multi-syllable words, and
    the NFD form of the first 300 strings."""
    syl = list(dict.fromkeys(L.load_hunspell_syllables("old")[:max_syllables] + L.load_hunspell_syllables("new")[:max_syllables]))
    strings = [" " + s for s in syl]
    strings += [" " + w for w in L.load_words() if " " in w][:500]
    strings += [U.nfd(s) for s in strings[:300]]
    return strings


def compare_adapters(hf, sp, strings, max_records: int = 50) -> dict:
    """Token-id equality string by string (design 6.2). `ids_equal` is True only when EVERY
    string agrees; `agreement` is the fraction, for the report."""
    agree = 0
    disagreements = []
    for s in strings:
        ta, tb = hf.encode(s), sp.encode(s)
        if [t.id for t in ta] == [t.id for t in tb]:
            agree += 1
        elif len(disagreements) < max_records:
            disagreements.append({"text": s, "hf_ids": [t.id for t in ta], "spm_ids": [t.id for t in tb],
                                  "hf_pieces": [t.text for t in ta], "spm_pieces": [t.text for t in tb]})
    n = len(strings)
    return {"n": n, "n_agree": agree, "agreement": agree / n if n else None, "ids_equal": agree == n and n > 0,
            "rule": "token ids asserted equal on every string (design 6.2, item 67)", "disagreements": disagreements}


if __name__ == "__main__":
    sys.exit(main())
