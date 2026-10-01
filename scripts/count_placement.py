#!/usr/bin/env python
"""Count the two tone-mark placement conventions in a text corpus (design 6.3).

    python scripts/count_placement.py data/external/xcopa_test_vi.jsonl data/external/xcopa_val_vi.jsonl
    python scripts/count_placement.py corpus.txt --limit-lines 2000000

Reads plain text (one document per line) or JSONL (every string value is text), finds the
syllables whose placement differs between the conventions (toned open oa/oe/uy after a
non-qu onset) and reports how many are written old-style (hòa) versus new-style (hoà), by
onset/rime type and overall, plus a `per_file` block (design 6.3 / section 0): per input
file `n_items` (lines), `n_texts` (string fields read), `n_word_tokens` (regex word tokens),
`n_syllable_tokens` (word tokens that parse as a Vietnamese syllable), `affected_tokens`
(tokens whose placement differs between the conventions, i.e. the tokens C2 changes) and
`affected_items` (lines with at least one such token). The pre-registration fixes the
baseline convention from this count before Gate 1; the result goes into docs/RESULTS_LOG.md
with the corpus named.
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
    """Yields (item_number, text): the 1-based count of non-empty lines and each string field of it."""
    with open(path, encoding="utf-8", errors="replace") as f:
        n = 0
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n += 1
            if ln.startswith("{"):
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    yield n, ln
                    continue
                for v in d.values():
                    if isinstance(v, str):
                        yield n, v
            else:
                yield n, ln


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    counts = Counter()
    by_rime = Counter()
    n_lines = n_tokens = 0
    per_file: dict[str, dict] = {}
    for path in args.paths:
        f = {"n_items": 0, "n_texts": 0, "n_word_tokens": 0, "n_syllable_tokens": 0, "affected_tokens": 0, "affected_items": 0,
             "old": 0, "new": 0}
        for i, (line_no, t) in enumerate(texts_from(path)):
            if args.limit_lines and i >= args.limit_lines:
                break
            n_lines += 1
            f["n_texts"] += 1
            f["n_items"] = max(f["n_items"], line_no)
            hit = False
            for w in WORD.findall(U.nfc(t)):
                n_tokens += 1
                f["n_word_tokens"] += 1
                p = try_parse(w, strict=False)
                if p is None:
                    continue
                f["n_syllable_tokens"] += 1
                if p.placement not in ("old", "new"):
                    continue
                counts[p.placement] += 1
                f[p.placement] += 1
                f["affected_tokens"] += 1
                hit = True
                by_rime[(p.rime_spelling, p.placement)] += 1
            if hit:
                f.setdefault("_hit_lines", set()).add(line_no)
        f["affected_items"] = len(f.pop("_hit_lines", set()))
        per_file[_relative(path)] = f
    total = counts["old"] + counts["new"]
    report = {"paths": [_relative(p) for p in args.paths], "n_texts": n_lines, "n_word_tokens": n_tokens,
              "n_syllable_tokens": sum(f["n_syllable_tokens"] for f in per_file.values()),
              "n_items": sum(f["n_items"] for f in per_file.values()),
              "affected_tokens": total, "affected_items": sum(f["affected_items"] for f in per_file.values()),
              "old": counts["old"], "new": counts["new"],
              "old_share": (counts["old"] / total) if total else None,
              "by_rime": {f"{r}|{pl}": c for (r, pl), c in sorted(by_rime.items())},
              "majority": ("old" if counts["old"] >= counts["new"] else "new") if total else None,
              "per_file": per_file,
              "definitions": {"n_items": "non-empty lines of the file (one item / document per line)",
                              "n_texts": "string fields read (every string value of a JSONL record; the line for plain text)",
                              "n_word_tokens": "regex word tokens over the NFC text", "n_syllable_tokens": "word tokens that parse as a Vietnamese syllable",
                              "affected_tokens": "syllable tokens whose tone-mark placement differs between the conventions (what C2 changes)",
                              "affected_items": "lines with at least one affected token"}}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
