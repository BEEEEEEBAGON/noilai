#!/usr/bin/env python
"""One source for every count quoted about the resources (red-team item 60).

    python scripts/reconcile_counts.py --out data/audit/counts.json

Reports, from the files in data/external and data/audit: Hunspell entries / lowercase /
parsable / rejected / standard; inventory structures and extension; distinct rimes; onset,
tone and coda distributions; the 69 placement-differing syllables; word-list two-syllable
pairs; Gemma 3 audit summary numbers. Every number in docs/DESIGN_DECISIONS.md section 0
and docs/RESULTS_LOG.md is to be re-read from this file, never retyped.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.vi import lexicon as L  # noqa: E402
from noilai.vi import unicode as U  # noqa: E402
from noilai.vi.syllable import Inventory, spell, try_parse  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit" / "counts.json")
    args = ap.parse_args(argv)
    raw_new = [ln.rstrip("\n") for ln in open(L.EXTERNAL / "vi-DauMoi.dic", encoding="utf-8")][1:]
    raw_old = [ln.rstrip("\n") for ln in open(L.EXTERNAL / "vi-DauCu.dic", encoding="utf-8")][1:]
    new = L.load_hunspell_syllables("new")
    old = L.load_hunspell_syllables("old")
    base = Inventory(new + old)
    ext = L.load_inventory()
    parsed = [try_parse(w, strict=False) for w in new]
    n_parsable = sum(1 for p in parsed if p)
    standard = [w for w, p in zip(new, parsed) if p and p.i_y_variant is None]
    onsets = Counter(p.syllable.onset for p in parsed if p)
    tones = Counter(U.TONE_NAMES[p.syllable.tone] for p in parsed if p)
    codas = Counter(p.syllable.coda or "open" for p in parsed if p)
    differing = sorted(set(new) - set(old))
    from noilai.vi.syllable import _onset_spelling
    onset_spellings = Counter(_onset_spelling(p.syllable) for p in parsed if p)
    audit = {}
    for name in ("gemma3", "gemma2"):
        p = ROOT / "data" / "audit" / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            audit[name] = {"summary": d["summary"], "normalizes_nfd": d["normalization_census"]["normalizes_nfd"],
                           "normalizer": d["normalization_census"].get("normalizer")}
    idx = L.wordlist_index()
    counts = {
        "hunspell": {
            "entries_new_file": len(raw_new), "entries_old_file": len(raw_old),
            "lowercase_letter_entries": len(new), "parsable": n_parsable, "unparsed": sorted(set(base.unparsed)),
            "rejected_phonotactics": sorted(set(base.rejected)), "standard_spelling_parsable": len(standard),
            "iy_variants": sorted(w for w, p in zip(new, parsed) if p and p.i_y_variant in ("y", "i")),
            "nonstandard": sorted(w for w, p in zip(new, parsed) if p and p.i_y_variant == "nonstandard"),
            "placement_differing_syllables": len(differing), "placement_differing_list": differing,
        },
        "inventory": {
            "base_structures": len(base.structures), "base_toneless": len(base.toneless), "base_rimes": len(base.rimes),
            "extended_structures": len(ext.structures), "extension_added": ext.n_extended, "extended_rimes": len(ext.rimes),
            "rimes": sorted(ext.rimes),
        },
        "distributions_new_file": {
            "onset_canonical": dict(onsets.most_common()), "onset_spelling": dict(onset_spellings.most_common()),
            "tone": dict(tones.most_common()), "coda": dict(codas.most_common()),
        },
        "wordlist": {"entries": sum(1 for _ in L.load_words()), "two_syllable_pairs": len(idx["two_syllable_pairs"]),
                     "distinct_canonical_syllables": len(idx["canonical_counts"]),
                     "iy_table_y_forms": sorted(k for k, v in L.iy_table().items() if v == "y")},
        "tokenizer_audit": audit,
        "resource_sha256": json.loads((ROOT / "data" / "HASHES.json").read_text()) if (ROOT / "data" / "HASHES.json").exists() else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {"hunspell_lowercase": len(new), "parsable": n_parsable, "standard": len(standard), "rejected": len(set(base.rejected)),
             "unparsed": len(set(base.unparsed)), "base_structures": len(base.structures), "extended": len(ext.structures),
             "extension_added": ext.n_extended, "rimes": len(ext.rimes), "placement_differing": len(differing),
             "two_syllable_pairs": len(idx["two_syllable_pairs"]),
             "gemma3_nfc_tps": audit.get("gemma3", {}).get("summary", {}).get("nfc", {}).get("tokens_per_syllable_mean"),
             "gemma3_nfd_tps": audit.get("gemma3", {}).get("summary", {}).get("nfd", {}).get("tokens_per_syllable_mean")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
