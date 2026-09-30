#!/usr/bin/env python
"""Build a NóiLái release: items, splits, core set, manifest.

    python scripts/build_data.py --out data/release/v0.1 --seed 20261004
    python scripts/build_data.py --out data/release/sealed --seed 777 --sealed   # never sent to any API

The manifest records the seed, item counts per cell, resource hashes, the git
commit of the generator and the canary string. A sealed build uses a different
seed and a different canary and is meant to be regenerated at release time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.gen.generate import Generator, write_release  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--", str(ROOT)], cwd=ROOT, text=True).strip())
    except Exception:  # noqa: BLE001
        return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--n-lexicon", type=int, default=1500)
    ap.add_argument("--n-pseudo", type=int, default=1000)
    ap.add_argument("--per-cell-t1", type=int, default=1000)
    ap.add_argument("--per-cell-t2", type=int, default=500)
    ap.add_argument("--per-cell-t3", type=int, default=500, help="yes/no PAIRS per variant cell")
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--core-per-cell", type=int, default=125)
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args(argv)

    t0 = dt.datetime.now(dt.timezone.utc)
    g = Generator(seed=args.seed)
    build = g.build(n_lexicon=args.n_lexicon, n_pseudo=args.n_pseudo, per_cell_t1=args.per_cell_t1,
                    per_cell_t2=args.per_cell_t2, per_cell_t3=args.per_cell_t3, dev_frac=args.dev_frac,
                    core_per_cell=args.core_per_cell)
    manifest = write_release(build, args.out, manifest_extra={
        "seed": args.seed, "sealed": args.sealed, "generator_args": vars(args) | {"out": str(args.out)},
        "git_commit": git_commit(), "git_dirty": git_dirty(),
        "started_utc": t0.isoformat(), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_lexical_pairs_available": len(g.lex_pairs), "n_attested_syllables": len(g.attested),
    })
    print(json.dumps({k: manifest[k] for k in ("n_items", "n_base_pairs", "counts", "core_counts", "canary", "git_commit", "git_dirty")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
