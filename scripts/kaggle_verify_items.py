#!/usr/bin/env python
"""Verify an item file before a run: hash, canary, row count.

Every run manifest records the SHA-256 of the item file it read; a run on a file whose hash
differs from the one recorded in configs/run_plan.yaml at the data freeze is not a result
of the paper's benchmark. This script is the gate the notebooks call after fetching the
data and before the first model loads:

    python scripts/kaggle_verify_items.py --key noilai_test              # path + expected hash from run_plan.yaml
    python scripts/kaggle_verify_items.py --items path/to/items.jsonl     # hash printed, nothing to compare against
    python scripts/kaggle_verify_items.py --key noilai_core --expect-canary

Checks: (1) the file exists and every line is a JSON object with `item_id` and `task`;
(2) if run_plan.yaml records a sha256 for the key, the observed hash equals it; (3) when the
item file requires a canary (test/core/sealed), every row carries the same
`NOILAI-CANARY-<uuid>` and it equals the canary in the release manifest.json next to the
file when that manifest exists. Exit status 1 on any failure; the summary is JSON so the
notebook can keep it in the run's manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RUN_PLAN = ROOT / "configs" / "run_plan.yaml"
CANARY_PREFIX = "NOILAI-CANARY-"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_item_file_spec(key: str, plan_path: Path = RUN_PLAN) -> dict:
    import yaml

    plan = yaml.safe_load(open(plan_path, encoding="utf-8"))
    files = plan.get("item_files", {})
    if key not in files:
        raise KeyError(f"{key!r} is not an item file key in {plan_path} (have {sorted(files)})")
    return files[key]


def verify_items(path: Path, expected_sha256: Optional[str] = None, canary_required: bool = False,
                 manifest_canary: Optional[str] = None) -> dict:
    """Return a summary dict with an `ok` flag and a list of `problems` (empty when ok)."""
    path = Path(path)
    problems: list[str] = []
    out: dict = {"path": str(path), "sha256": None, "sha256_expected": expected_sha256, "sha256_ok": None,
                 "n_items": 0, "tasks": {}, "canary": None, "canary_ok": None, "problems": problems, "ok": False}
    if not path.exists():
        problems.append(f"missing file {path}")
        return out
    out["sha256"] = sha256_file(path)
    if expected_sha256:
        out["sha256_ok"] = out["sha256"] == expected_sha256
        if not out["sha256_ok"]:
            problems.append(f"sha256 mismatch: expected {expected_sha256[:16]}..., got {out['sha256'][:16]}...")
    tasks: Counter = Counter()
    canaries: set = set()
    n = 0
    with open(path, encoding="utf-8") as f:
        for ln_no, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                it = json.loads(ln)
            except json.JSONDecodeError as e:
                problems.append(f"line {ln_no}: not JSON ({e.msg})")
                continue
            if not isinstance(it, dict) or "item_id" not in it or "task" not in it:
                problems.append(f"line {ln_no}: not an item (needs item_id and task)")
                continue
            n += 1
            tasks[it["task"]] += 1
            if "canary" in it:
                canaries.add(it["canary"])
            elif canary_required:
                problems.append(f"line {ln_no}: {it['item_id']} has no canary")
    out["n_items"] = n
    out["tasks"] = dict(sorted(tasks.items()))
    if n == 0:
        problems.append("no items")
    if canary_required:
        if len(canaries) == 1:
            c = next(iter(canaries))
            out["canary"] = c
            ok = c.startswith(CANARY_PREFIX) and (manifest_canary is None or c == manifest_canary)
            out["canary_ok"] = ok
            if not ok:
                problems.append("canary malformed or different from the release manifest")
        elif len(canaries) > 1:
            out["canary_ok"] = False
            problems.append(f"{len(canaries)} different canary strings in one file")
        else:
            out["canary_ok"] = False
            if n:
                problems.append("canary required but absent")
    out["ok"] = not problems
    return out


def manifest_canary_for(path: Path) -> Optional[str]:
    m = Path(path).parent / "manifest.json"
    if m.exists():
        try:
            return json.loads(m.read_text(encoding="utf-8")).get("canary")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="item file key in configs/run_plan.yaml")
    g.add_argument("--items", type=Path, help="explicit item file path")
    ap.add_argument("--plan", type=Path, default=RUN_PLAN)
    ap.add_argument("--root", type=Path, default=ROOT, help="project root that relative plan paths refer to")
    ap.add_argument("--expect-canary", action="store_true", help="require a canary even if the plan does not")
    ap.add_argument("--sha256", default=None, help="expected hash (overrides the plan)")
    args = ap.parse_args(argv)

    expected = args.sha256
    canary_required = args.expect_canary
    if args.key:
        spec = load_item_file_spec(args.key, args.plan)
        path = Path(spec["path"])
        if not path.is_absolute():
            path = args.root / path
        expected = expected or spec.get("sha256")
        canary_required = canary_required or bool(spec.get("canary_required"))
    else:
        path = args.items
    res = verify_items(path, expected_sha256=expected, canary_required=canary_required,
                       manifest_canary=manifest_canary_for(path))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["ok"] and expected is None:
        print("[note] no sha256 recorded for this file yet: record the hash above in configs/run_plan.yaml at the data freeze",
              file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
