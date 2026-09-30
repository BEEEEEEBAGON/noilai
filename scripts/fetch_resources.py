#!/usr/bin/env python
"""Fetch the third-party resources into data/external/ and verify their hashes.

Nothing under data/external/ is committed. Each resource records its origin,
license and the SHA-256 observed on 30 September 2026; a hash mismatch is
reported (the upstream file changed) and the file is kept with a .UNVERIFIED
suffix so that no build silently uses a different resource than the paper's.

Usage: python scripts/fetch_resources.py [--only NAME ...] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"

RESOURCES = {
    "vi-DauMoi.dic": {
        "url": "https://raw.githubusercontent.com/1ec5/hunspell-vi/main/dictionaries/vi-DauMoi.dic",
        "sha256": None,  # filled by --record; see data/external/HASHES.json
        "license": "GPLv2 (Hồ Ngọc Đức's word list, converted by László Németh; maintained by 1ec5/hunspell-vi)",
        "role": "attested syllable inventory, new tone-mark placement",
    },
    "vi-DauCu.dic": {
        "url": "https://raw.githubusercontent.com/1ec5/hunspell-vi/main/dictionaries/vi-DauCu.dic",
        "sha256": None,
        "license": "GPLv2",
        "role": "the same inventory in old tone-mark placement (validation of the placement converter)",
    },
    "Viet74K.txt": {
        "url": "https://raw.githubusercontent.com/duyet/vietnamese-wordlist/master/Viet74K.txt",
        "sha256": None,
        "license": "GPL (Hồ Ngọc Đức, via duyet/vietnamese-wordlist)",
        "role": "two-syllable words as lexical base pairs; syllable frequency proxy; lexicality lookup",
    },
    "xcopa_test_vi.jsonl": {
        "url": "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/vi/test.vi.jsonl",
        "sha256": None,
        "license": "CC BY 4.0 (XCOPA, Ponti et al. 2020)",
        "role": "500 Vietnamese test items for the re-encoding arms (E3)",
    },
    "xcopa_val_vi.jsonl": {
        "url": "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/vi/val.vi.jsonl",
        "sha256": None,
        "license": "CC BY 4.0",
        "role": "100 Vietnamese validation items (prompt development only)",
    },
    "gemma3_tokenizer.model": {
        "url": "https://raw.githubusercontent.com/google/gemma_pytorch/main/tokenizer/gemma3_cleaned_262144_v2.spiece.model",
        "sha256": None,
        "license": "Apache-2.0 repository (google/gemma_pytorch); the tokenizer is shared by all Gemma 3 checkpoints",
        "role": "Gemma 3 SentencePiece model for the tokenizer audit without Hugging Face access",
    },
    "gemma2_tokenizer.model": {
        "url": "https://raw.githubusercontent.com/google/gemma_pytorch/main/tokenizer/tokenizer.model",
        "sha256": None,
        "license": "Apache-2.0 repository (google/gemma_pytorch)",
        "role": "Gemma 1/2 SentencePiece model (256k) for comparison",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict:
    p = EXTERNAL / "HASHES.json"
    if p.exists():
        import json
        return json.loads(p.read_text())
    p2 = ROOT / "data" / "HASHES.json"
    if p2.exists():
        import json
        return json.loads(p2.read_text())
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--record", action="store_true", help="write observed hashes to data/HASHES.json (committed)")
    args = ap.parse_args(argv)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    known = load_hashes()
    rc = 0
    observed = {}
    for name, meta in RESOURCES.items():
        if args.only and name not in args.only:
            continue
        dest = EXTERNAL / name
        if dest.exists() and not args.force:
            print(f"[keep] {name}")
        else:
            print(f"[get ] {name} <- {meta['url']}")
            try:
                with urllib.request.urlopen(meta["url"], timeout=120) as r, open(dest, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                print(f"[FAIL] {name}: {e}")
                rc = 1
                continue
        h = sha256(dest)
        observed[name] = h
        expected = known.get(name) or meta.get("sha256")
        if expected and h != expected:
            print(f"[HASH MISMATCH] {name}: expected {expected[:16]}..., got {h[:16]}...")
            dest.rename(dest.with_suffix(dest.suffix + ".UNVERIFIED"))
            rc = 1
        else:
            print(f"[ok  ] {name} sha256={h[:16]}... ({meta['license']})")
    if args.record:
        import json
        out = ROOT / "data" / "HASHES.json"
        merged = {**known, **observed}
        out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"[rec ] wrote {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
