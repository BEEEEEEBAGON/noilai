#!/usr/bin/env python
"""Check that a model's Hugging Face tokenizer agrees with a SentencePiece model file.

The tokenizer audit in data/audit was computed from the Gemma tokenizer FILES fetched
from google/gemma_pytorch. Before any model result is joined to those statistics, run
this on the machine that has hub access:

    python scripts/check_tokenizer_consistency.py --hf google/gemma-3-1b-it --spm data/external/gemma3_tokenizer.model

It tokenizes the syllable inventory and a phrase sample with both, compares the piece
sequences (ids may be offset by special tokens; pieces must match), and writes a JSON
report next to the audit. Exit code 1 when fewer than 99.9% of strings agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import HFAdapter, SentencePieceAdapter  # noqa: E402
from noilai.vi import lexicon as L  # noqa: E402
from noilai.vi import unicode as U  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit")
    ap.add_argument("--max", type=int, default=3000)
    args = ap.parse_args(argv)
    hf = HFAdapter(args.hf)
    sp = SentencePieceAdapter(args.spm)
    strings = [" " + s for s in L.load_hunspell_syllables("new")[: args.max]]
    strings += [" " + w for w in L.load_words() if " " in w][:500]
    strings += [U.nfd(s) for s in strings[:300]]
    agree = 0
    disagreements = []
    for s in strings:
        a = [t.text for t in hf.encode(s)]
        b = [t.text for t in sp.encode(s)]
        if a == b:
            agree += 1
        elif len(disagreements) < 50:
            disagreements.append({"text": s, "hf": a, "spm": b})
    frac = agree / len(strings)
    report = {"hf": args.hf, "spm": args.spm, "n": len(strings), "agreement": frac, "disagreements": disagreements,
              "hf_normalizer": hf.normalizer_info(), "spm_normalizer": sp.normalizer_info()}
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.hf.replace("/", "__")
    (args.out / f"consistency_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, ensure_ascii=False, indent=1))
    return 0 if frac >= 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
