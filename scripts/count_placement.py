#!/usr/bin/env python
"""Count the two tone-mark placement conventions in a text corpus (design 6.3).

    python scripts/count_placement.py data/external/xcopa_test_vi.jsonl data/external/xcopa_val_vi.jsonl
    python scripts/count_placement.py corpus.txt --limit-lines 2000000

Reads plain text (one document per line) or JSONL (every string value is text), finds the
syllables whose placement differs between the conventions (toned open oa/oe/uy after a
non-qu onset) and reports how many are written old-style (hòa) versus new-style (hoà), by
onset/rime type and overall. The pre-registration fixes the baseline convention from this
count before Gate 1; the result goes into docs/RESULTS_LOG.md with the corpus named.
The Wikipedia dump and the news corpora are not reachable from the build machine, so the
author runs this on a downloaded corpus.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.vi import unicode as U
from noilai.vi.syllable import try_parse

WORD = re.compile(r"[A-Za-zĐđÀ-ɏḀ-ỿ]+")


def texts_from(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("{"):
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    yield ln
                    continue
                for v in d.values():
                    if isinstance(v, str):
                        yield v
            else:
                yield ln


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    counts = Counter()
    by_rime = Counter()
    n_lines = n_tokens = 0
    for path in args.paths:
        for i, t in enumerate(texts_from(path)):
            if args.limit_lines and i >= args.limit_lines:
                break
            n_lines += 1
            for w in WORD.findall(U.nfc(t)):
                n_tokens += 1
                p = try_parse(w, strict=False)
                if p is None or p.placement not in ("old", "new"):
                    continue
                counts[p.placement] += 1
                by_rime[(p.rime_spelling, p.placement)] += 1
    total = counts["old"] + counts["new"]
    report = {"paths": [str(p) for p in args.paths], "n_texts": n_lines, "n_word_tokens": n_tokens,
              "affected_tokens": total, "old": counts["old"], "new": counts["new"],
              "old_share": (counts["old"] / total) if total else None,
              "by_rime": {f"{r}|{pl}": c for (r, pl), c in sorted(by_rime.items())},
              "majority": ("old" if counts["old"] >= counts["new"] else "new") if total else None}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
