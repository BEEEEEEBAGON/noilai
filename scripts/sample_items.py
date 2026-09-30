#!/usr/bin/env python
"""Seeded sub-samples of a release (design 4.5, red-team items 46 and 56).

    python scripts/sample_items.py main --release data/release/v0.2 --per-cell 350 --seed 20261004
    python scripts/sample_items.py c2   --release data/release/v0.2 --n 500 --seed 20261005

`main` writes noilai_main.jsonl: the open-model main sample, 350 items per task × variant
cell drawn from the test split, always containing the core, stratified as far as the pool
allows over source (lexicon/pseudo) and output lexicality; T3 items are drawn as yes/no
pairs (175 pairs = 350 items). The manifest of the release gains a `samples` entry with
the file's SHA-256 and per-cell counts; `--limit` in the runner is for smoke tests only.

`c2` writes noilai_c2.jsonl: the C2-enriched set for H4, built from an INDEPENDENT pool of
base pairs (a Generator with its own seed whose base syllables are restricted to toned open
oa/oe/uy rimes after a non-qu onset), so that every item is C2-affected and none shares a
base pair or an input with the release; it is excluded from E1/E2 by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.gen import variants as V  # noqa: E402
from noilai.gen.generate import Generator, c2_affected, load_items, read_header  # noqa: E402
from noilai.vi.reencode import canonical_text  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, header: dict | None, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _balanced(rng, pool, n, key):
    groups = defaultdict(list)
    for it in pool:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=str)
    out = []
    i = 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
    return out


def cmd_main(args) -> int:
    rel = Path(args.release)
    test = load_items(rel / "noilai_test.jsonl")
    header = read_header(rel / "noilai_test.jsonl")
    rng = random.Random(args.seed)
    by_id = {it["item_id"]: it for it in test}
    chosen: dict[str, dict] = {}
    counts = Counter()
    for task in ("T1", "T2"):
        for v in V.VARIANTS:
            pool = [it for it in test if it["task"] == task and it["variant"] == v and not it["vulgar"]]
            core = [it for it in pool if it["in_core"]]
            rest = [it for it in pool if not it["in_core"]]
            need = max(args.per_cell - len(core), 0)
            sel = core + _balanced(rng, rest, need, key=lambda it: (it["source"], it["strata"]["output_lexical"]))
            for it in sel:
                chosen[it["item_id"]] = it
            counts[f"{task}-{v}"] = len(sel)
    for v in V.VARIANTS:
        pool = [it for it in test if it["task"] == "T3" and it["variant"] == v and it["gold"] == "yes" and not it["vulgar"]
                and it.get("pair_item_id") in by_id]
        core = [it for it in pool if it["in_core"]]
        rest = [it for it in pool if not it["in_core"]]
        need = max(args.per_cell // 2 - len(core), 0)
        sel = core + _balanced(rng, rest, need, key=lambda it: (it["source"], by_id[it["pair_item_id"]]["twin_type"]))
        for it in sel:
            chosen[it["item_id"]] = it
            mate = by_id[it["pair_item_id"]]
            chosen[mate["item_id"]] = mate
        counts[f"T3-{v}"] = 2 * len(sel)
    items = sorted(chosen.values(), key=lambda it: it["item_id"])
    out = rel / "noilai_main.jsonl"
    write_jsonl(out, header, items)
    _record(rel, "main", out, {"seed": args.seed, "per_cell": args.per_cell, "counts": dict(sorted(counts.items())), "n_items": len(items)})
    print(json.dumps({"n_items": len(items), "counts": dict(sorted(counts.items())), "sha256": sha256(out)}, ensure_ascii=False))
    return 0


def cmd_c2(args) -> int:
    rel = Path(args.release)
    header = read_header(rel / "noilai_test.jsonl")
    release_items = load_items(rel / "noilai_dev.jsonl") + load_items(rel / "noilai_test.jsonl")
    used_bp = {it["base_pair_id"] for it in release_items}
    used_inputs = {canonical_text(it["input"]) for it in release_items}
    g = Generator(seed=args.seed)
    # restrict the pseudo pool to syllables that make an item C2-affected
    g.base_syllables = [s for s in g.base_syllables if c2_affected([s])]
    if len(g.base_syllables) < 20:
        raise SystemExit("too few C2-affecting base syllables")
    rng = random.Random(args.seed)
    items: list[dict] = []
    pairs = g.base_pairs(n_lexicon=0, n_pseudo=args.n * 3)
    for bp in pairs:
        if bp.base_pair_id in used_bp:
            continue
        for it in g.t1_items(bp):
            if it["vulgar"] or canonical_text(it["input"]) in used_inputs or not it["strata"]["c2_affected"]:
                continue
            items.append(it)
    rng.shuffle(items)
    items = items[: args.n]
    for k, it in enumerate(items):
        it["item_id"] = f"C2-{it['variant']}-{k+1:06d}"
        it["split"] = "test"
        it["in_core"] = False
        if header:
            it["canary"] = header["canary"]
            it["do_not_train"] = True
            it["evaluation_only"] = True
    items.sort(key=lambda it: it["item_id"])
    out = rel / "noilai_c2.jsonl"
    write_jsonl(out, header, items)
    _record(rel, "c2", out, {"seed": args.seed, "n_items": len(items), "counts": dict(Counter(it["variant"] for it in items)),
                             "all_c2_affected": all(it["strata"]["c2_affected"] for it in items),
                             "disjoint_from_release": not ({it["base_pair_id"] for it in items} & used_bp)})
    print(json.dumps({"n_items": len(items), "counts": dict(Counter(it["variant"] for it in items)), "sha256": sha256(out)}, ensure_ascii=False))
    return 0


def _record(rel: Path, name: str, path: Path, meta: dict) -> None:
    mpath = rel / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    m.setdefault("samples", {})[name] = {"file": path.name, "sha256": sha256(path), **meta}
    mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("main")
    a.add_argument("--release", required=True)
    a.add_argument("--per-cell", type=int, default=350)
    a.add_argument("--seed", type=int, default=20261004)
    a.set_defaults(func=cmd_main)
    c = sub.add_parser("c2")
    c.add_argument("--release", required=True)
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--seed", type=int, default=20261005)
    c.set_defaults(func=cmd_c2)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
