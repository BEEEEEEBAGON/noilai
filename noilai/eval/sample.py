"""Seeded, stratified sub-samples of an item file (DESIGN_DECISIONS 4.5, 12.32).

`RunOptions.limit` is a head truncation over items sorted by item_id and is therefore
smoke-only: on the v0.2 test file the first 3,000 rows are T1 only. Every real sub-sample
is drawn here, per task x variant cell, with the core forced in, T3 items kept in their
yes/no pairs, vulgar-flagged items excluded, and lexical/pseudo base pairs allocated in
proportion to the cell's pool. The draw is a pure function of (items, n, seed): the file
order does not matter (units are sorted by id before shuffling) and the sampled ids, their
SHA-256, the seed and the per-cell counts go into the run manifest (`run.py`) or into a
sample file (`write_sample_file`, for `scripts/sample_items.py`).

    s = stratified_sample(items, 4200, seed=DEFAULT_SEED)
    s.per_cell           -> {"T1-V1": 350, ..., "T3-V4": 350}
    s.items              -> the sampled items, sorted by item_id
    write_sample_file(s, "data/release/v0.3/noilai_main4200.jsonl", header=read_header(src))

Quotas: n is split over the cells as evenly as possible (the first n mod k cells get one
more); a T3 quota is rounded down to an even number of items (pairs). A cell whose pool is
smaller than its quota is taken whole and the shortfall is recorded. The core is forced in
even when it exceeds the quota ("contains the core"), which is recorded too.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..gen import variants as V

CELL_TASKS = ("T1", "T2", "T3")
DEFAULT_SEED = 20261205   # public --sample seed (smoke/pilot only); never the withheld build seed, DD 4.6


def is_vulgar(item: dict) -> bool:
    v = item.get("vulgar")
    return v is True or (isinstance(v, str) and v.strip().lower() in ("yes", "true", "1"))


@dataclass
class Sample:
    items: list[dict]
    seed: int
    n_requested: int
    per_cell: dict[str, int] = field(default_factory=dict)
    per_cell_quota: dict[str, int] = field(default_factory=dict)
    per_cell_core: dict[str, int] = field(default_factory=dict)
    per_cell_shortfall: dict[str, int] = field(default_factory=dict)
    per_cell_strata: dict[str, dict[str, int]] = field(default_factory=dict)
    n_dropped_vulgar: int = 0
    n_broken_pairs: int = 0
    n_core_forced: int = 0
    strata_key: str = "source"
    exclude_vulgar: bool = True

    @property
    def item_ids(self) -> list[str]:
        return [it["item_id"] for it in self.items]

    def ids_sha256(self) -> str:
        return hashlib.sha256("\n".join(self.item_ids).encode("utf-8")).hexdigest()

    def describe(self) -> dict:
        """The manifest block: everything but the items themselves."""
        return {"n": len(self.items), "n_requested": self.n_requested, "seed": self.seed,
                "ids_sha256": self.ids_sha256(), "per_cell": dict(self.per_cell), "per_cell_quota": dict(self.per_cell_quota),
                "per_cell_core": dict(self.per_cell_core), "per_cell_shortfall": dict(self.per_cell_shortfall),
                "per_cell_strata": {k: dict(v) for k, v in self.per_cell_strata.items()},
                "n_dropped_vulgar": self.n_dropped_vulgar, "n_broken_pairs": self.n_broken_pairs,
                "n_core_forced": self.n_core_forced, "strata_key": self.strata_key, "exclude_vulgar": self.exclude_vulgar}


def cell_quotas(n: int, cells: Sequence[str], unit_sizes: dict[str, int] | None = None) -> dict[str, int]:
    """Split n items over the cells as evenly as possible; a cell with unit size 2 (T3 pairs)
    gets an even quota."""
    unit_sizes = unit_sizes or {}
    k = len(cells)
    base, extra = divmod(n, k)
    out = {}
    for i, c in enumerate(cells):
        q = base + (1 if i < extra else 0)
        u = unit_sizes.get(c, 1)
        out[c] = q - (q % u)
    return out


def _units(items: list[dict], task: str) -> tuple[list[tuple[str, list[dict]]], int]:
    """Sampling units of a cell: single items, or complete yes/no pairs for T3. Returns
    (units sorted by key, n_broken_pairs)."""
    if task != "T3":
        return sorted(((it["item_id"], [it]) for it in items), key=lambda u: u[0]), 0
    by_id = {it["item_id"]: it for it in items}
    seen: set[str] = set()
    units: list[tuple[str, list[dict]]] = []
    broken = 0
    for it in sorted(items, key=lambda x: x["item_id"]):
        if it["item_id"] in seen:
            continue
        mate = by_id.get(it.get("pair_item_id"))
        if mate is None or mate is it:
            broken += 1
            seen.add(it["item_id"])
            continue
        key = "|".join(sorted((it["item_id"], mate["item_id"])))
        units.append((key, sorted([it, mate], key=lambda x: x["item_id"])))
        seen.update({it["item_id"], mate["item_id"]})
    return sorted(units, key=lambda u: u[0]), broken


def _allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest remainders (never more than available)."""
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in counts}
    n = min(n, total)
    raw = {k: n * c / total for k, c in counts.items()}
    alloc = {k: min(counts[k], int(raw[k])) for k in counts}
    left = n - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - int(raw[k])), k))
    i = 0
    while left > 0 and i < 10 * len(order) + 1:
        k = order[i % len(order)]
        if alloc[k] < counts[k]:
            alloc[k] += 1
            left -= 1
        i += 1
    return alloc


def stratified_sample(items: Iterable[dict], n: int, seed: int = DEFAULT_SEED, tasks: Sequence[str] = CELL_TASKS,
                      variants: Sequence[str] = V.VARIANTS, force_core: bool = True, exclude_vulgar: bool = True,
                      strata_key: str = "source") -> Sample:
    """Draw n items: per task x variant cell, core forced in, T3 yes/no pairs kept together,
    vulgar excluded, `strata_key` (an item key or a `strata` key) allocated proportionally."""
    items = list(items)
    cells = [f"{t}-{v}" for t in tasks for v in variants]
    unit_sizes = {c: (2 if c.startswith("T3-") else 1) for c in cells}
    quotas = cell_quotas(int(n), cells, unit_sizes)
    out = Sample(items=[], seed=int(seed), n_requested=int(n), strata_key=strata_key, exclude_vulgar=exclude_vulgar)
    pools: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if "item_id" not in it or it.get("task") not in tasks or it.get("variant") not in variants:
            continue
        if exclude_vulgar and is_vulgar(it):
            out.n_dropped_vulgar += 1
            continue
        pools[f"{it['task']}-{it['variant']}"].append(it)

    def stratum(unit: list[dict]) -> str:
        it = unit[0]
        val = it.get(strata_key, (it.get("strata") or {}).get(strata_key))
        return str(val)

    chosen: list[dict] = []
    for cell in cells:
        task = cell.split("-")[0]
        units, broken = _units(pools.get(cell, []), task)
        out.n_broken_pairs += broken
        quota_units = quotas[cell] // unit_sizes[cell]
        core = [u for u in units if force_core and any(x.get("in_core") for x in u[1])]
        rest = [u for u in units if u not in core]
        if len(core) > quota_units:
            out.n_core_forced += (len(core) - quota_units) * unit_sizes[cell]
        need = max(0, quota_units - len(core))
        by_stratum: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
        for u in rest:
            by_stratum[stratum(u[1])].append(u)
        alloc = _allocate({k: len(v) for k, v in by_stratum.items()}, need)
        picked: list[tuple[str, list[dict]]] = list(core)
        for k in sorted(by_stratum):
            rng = random.Random(f"{seed}:{cell}:{k}")
            lst = list(by_stratum[k])
            rng.shuffle(lst)
            picked.extend(lst[: alloc[k]])
        cell_items = [x for _, u in picked for x in u]
        cell_items.sort(key=lambda x: x["item_id"])
        chosen.extend(cell_items)
        out.per_cell[cell] = len(cell_items)
        out.per_cell_quota[cell] = quotas[cell]
        out.per_cell_core[cell] = sum(1 for x in cell_items if x.get("in_core"))
        out.per_cell_shortfall[cell] = max(0, quotas[cell] - len(cell_items))
        out.per_cell_strata[cell] = dict(Counter(stratum([x]) for x in cell_items))
    out.items = sorted(chosen, key=lambda x: x["item_id"])
    return out


def write_sample_file(sample: Sample, path: Path, header: dict | None = None, source: Path | None = None) -> dict:
    """Write the sampled items as a release-style jsonl (header record kept when given) and a
    sidecar `<name>.sample.json` with `describe()`; returns the sidecar dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in sample.items)
    side = sample.describe()
    side["file"] = path.name
    side["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    side["source"] = str(source) if source else None
    with open(path.with_suffix(path.suffix + ".sample.json"), "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    return side


def head_sample(items: Iterable[dict], limit: int) -> list[dict]:
    """The smoke-only `--limit`: the first `limit` items in the given order. Documented as
    'first N', never a sample."""
    out = []
    for it in items:
        if len(out) >= limit:
            break
        out.append(it)
    return out
