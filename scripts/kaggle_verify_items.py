#!/usr/bin/env python
"""Verify an item file before a run: hash, header record, canary, row count, cell counts.

Every run manifest records the SHA-256 of the item file it read; a run on a file whose hash
differs from the one recorded in configs/run_plan.yaml at the data freeze is not a result
of the paper's benchmark. This script is the gate the notebooks call after fetching the
data and before the first model loads:

    python scripts/kaggle_verify_items.py --key noilai_main               # path + expected hash from run_plan.yaml
    python scripts/kaggle_verify_items.py --key noilai_api_para300 --materialize   # derive the seeded file first if absent
    python scripts/kaggle_verify_items.py --items path/to/items.jsonl      # hash printed, nothing to compare against
    python scripts/kaggle_verify_items.py --key noilai_core --expect-canary

Checks (DESIGN_DECISIONS 4.6 and 4.5):
  1. the file exists; every non-header line is a JSON object with `item_id` and `task` (a
     NóiLái file) or `premise`/`choice1`/`choice2`/`question`/`label`/`idx` (an XCOPA file);
  2. the BIG-bench style HEADER RECORD: a first line `{"_header": "... canary GUID <uuid>",
     "canary": "NOILAI-CANARY-<uuid>", "do_not_train": true, "evaluation_only": true}` is
     skipped for the item checks; it is REQUIRED on every file that requires a canary
     (test, core, sealed and the files derived from them), its canary must equal the rows'
     canary and the release manifest's, and a `_header` record anywhere but on the first
     line is an error;
  3. if run_plan.yaml records a sha256 for the key, the observed hash equals it; for a file
     written by scripts/sample_items.py (`manifest_sample` in the plan) the hash recorded
     under `samples.<name>.sha256` in the release manifest must equal it as well;
  4. when the file requires a canary, every row carries the same `NOILAI-CANARY-<uuid>`;
  5. `expected_counts` (`n_items`, `per_cell`) from the plan, when given.

Exit status 1 on any failure; the summary is JSON so the notebook can keep it in the run's
manifest. `--materialize` derives a seeded sub-sample (plan `derive` block) through
scripts/kaggle_run_plan.py when the file does not exist yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_PLAN = ROOT / "configs" / "run_plan.yaml"
CANARY_PREFIX = "NOILAI-CANARY-"
HEADER_KEY = "_header"
XCOPA_KEYS = ("premise", "choice1", "choice2", "question", "label", "idx")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: str, plan: dict) -> str:
    """Replace `{release}` in a plan path with the plan's `release` directory."""
    return str(path).replace("{release}", str(plan.get("release", "")).rstrip("/"))


def load_plan(plan_path: Path = RUN_PLAN) -> dict:
    import yaml

    with open(plan_path, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    for spec in plan.get("item_files", {}).values():
        spec["path_template"] = spec["path"]
        spec["path"] = resolve_path(spec["path"], plan)
        if spec.get("produced_by"):
            spec["produced_by"] = resolve_path(spec["produced_by"], plan)
    if plan.get("release_manifest"):
        plan["release_manifest"] = resolve_path(plan["release_manifest"], plan)
    return plan


def load_item_file_spec(key: str, plan_path: Path = RUN_PLAN) -> dict:
    plan = load_plan(plan_path)
    files = plan.get("item_files", {})
    if key not in files:
        raise KeyError(f"{key!r} is not an item file key in {plan_path} (have {sorted(files)})")
    return files[key]


def is_header(d) -> bool:
    return isinstance(d, dict) and HEADER_KEY in d


def _is_item(d, kind: str) -> bool:
    if not isinstance(d, dict):
        return False
    if kind == "xcopa":
        return all(k in d for k in XCOPA_KEYS)
    return "item_id" in d and "task" in d


def verify_items(path: Path, expected_sha256: str | None = None, canary_required: bool = False,
                 manifest_canary: str | None = None, header_required: bool | None = None,
                 kind: str = "noilai", expected_counts: dict | None = None,
                 sample_sha256: str | None = None) -> dict:
    """Return a summary dict with an `ok` flag and a list of `problems` (empty when ok).

    `header_required` defaults to `canary_required` (DD 4.6: every file with a canary begins
    with the header record). `sample_sha256` is the hash the release manifest recorded for a
    sample file (scripts/sample_items.py); `expected_counts` may hold `n_items` and/or
    `per_cell` (items per task x variant cell for a NóiLái file).
    """
    path = Path(path)
    if header_required is None:
        header_required = canary_required
    problems: list[str] = []
    out: dict = {"path": str(path), "kind": kind, "sha256": None, "sha256_expected": expected_sha256, "sha256_ok": None,
                 "sample_sha256_ok": None, "header": None, "header_ok": None, "n_items": 0, "tasks": {}, "cells": {},
                 "canary": None, "canary_ok": None, "problems": problems, "ok": False}
    if not path.exists():
        problems.append(f"missing file {path}")
        return out
    out["sha256"] = sha256_file(path)
    if expected_sha256:
        out["sha256_ok"] = out["sha256"] == expected_sha256
        if not out["sha256_ok"]:
            problems.append(f"sha256 mismatch: expected {expected_sha256[:16]}..., got {out['sha256'][:16]}...")
    if sample_sha256:
        out["sample_sha256_ok"] = out["sha256"] == sample_sha256
        if not out["sample_sha256_ok"]:
            problems.append(f"file differs from the sample recorded in the release manifest ({sample_sha256[:16]}...)")
    tasks: Counter = Counter()
    cells: Counter = Counter()
    canaries: set = set()
    header: dict | None = None
    n = 0
    first_record = True
    with open(path, encoding="utf-8") as f:
        for ln_no, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                it = json.loads(ln)
            except json.JSONDecodeError as e:
                problems.append(f"line {ln_no}: not JSON ({e.msg})")
                first_record = False
                continue
            if is_header(it):
                if first_record:
                    header = it
                    out["header"] = {k: it.get(k) for k in (HEADER_KEY, "canary", "do_not_train", "evaluation_only")}
                else:
                    problems.append(f"line {ln_no}: a header record is allowed on the first line only")
                first_record = False
                continue
            first_record = False
            if not _is_item(it, kind):
                need = "premise, choice1, choice2, question, label, idx" if kind == "xcopa" else "item_id and task"
                problems.append(f"line {ln_no}: not an item (needs {need})")
                continue
            n += 1
            if kind == "xcopa":
                tasks["XCOPA"] += 1
            else:
                tasks[it["task"]] += 1
                cells[f"{it['task']}-{it.get('variant')}"] += 1
            if "canary" in it:
                canaries.add(it["canary"])
            elif canary_required:
                problems.append(f"line {ln_no}: {it.get('item_id') or it.get('idx')} has no canary")
    out["n_items"] = n
    out["tasks"] = dict(sorted(tasks.items()))
    out["cells"] = dict(sorted(cells.items()))
    if n == 0:
        problems.append("no items")
    # header record (DD 4.6)
    if header is not None:
        hc = header.get("canary")
        hok = isinstance(hc, str) and hc.startswith(CANARY_PREFIX) and header.get("do_not_train") is True \
            and header.get("evaluation_only") is True and hc.split(CANARY_PREFIX, 1)[-1] in str(header.get(HEADER_KEY, ""))
        if manifest_canary is not None and hc != manifest_canary:
            hok = False
            problems.append("header canary differs from the release manifest")
        if canaries and (len(canaries) != 1 or hc not in canaries):
            hok = False
            problems.append("header canary differs from the rows' canary")
        out["header_ok"] = hok
        if not hok and not any("header canary" in p for p in problems):
            problems.append("header record malformed (needs _header with the GUID, canary, do_not_train: true, evaluation_only: true)")
    elif header_required:
        out["header_ok"] = False
        problems.append("header record missing: a file with a canary must begin with the JSON header record (DD 4.6)")
    # rows' canary
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
    # counts
    if expected_counts:
        want_n = expected_counts.get("n_items")
        if want_n is not None and n != int(want_n):
            problems.append(f"expected {want_n} items, found {n}")
        per_cell = expected_counts.get("per_cell")
        if per_cell is not None and cells:
            bad = {k: v for k, v in cells.items() if v != int(per_cell)}
            if bad:
                problems.append(f"cells not at {per_cell} items: {bad}")
    out["ok"] = not problems
    return out


def manifest_canary_for(path: Path) -> str | None:
    m = Path(path).parent / "manifest.json"
    if m.exists():
        try:
            return json.loads(m.read_text(encoding="utf-8")).get("canary")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def sample_sha256_for(path: Path, sample_name: str | None) -> str | None:
    """The hash scripts/sample_items.py recorded under `samples.<name>` in the release manifest."""
    if not sample_name:
        return None
    m = Path(path).parent / "manifest.json"
    if not m.exists():
        return None
    try:
        return (json.loads(m.read_text(encoding="utf-8")).get("samples") or {}).get(sample_name, {}).get("sha256")
    except (OSError, json.JSONDecodeError):
        return None


def verify_spec(spec: dict, root: Path = ROOT, expected_sha256: str | None = None, expect_canary: bool = False) -> dict:
    """verify_items() driven by an item-file spec of run_plan.yaml (path already resolved)."""
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(root) / path
    expected = expected_sha256 or spec.get("sha256")
    canary_required = expect_canary or bool(spec.get("canary_required"))
    return verify_items(path, expected_sha256=expected, canary_required=canary_required,
                        manifest_canary=manifest_canary_for(path) if canary_required else None,
                        kind=spec.get("kind", "noilai"), expected_counts=spec.get("expected_counts"),
                        sample_sha256=sample_sha256_for(path, spec.get("manifest_sample")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="item file key in configs/run_plan.yaml")
    g.add_argument("--items", type=Path, help="explicit item file path")
    ap.add_argument("--plan", type=Path, default=RUN_PLAN)
    ap.add_argument("--root", type=Path, default=ROOT, help="project root that relative plan paths refer to")
    ap.add_argument("--expect-canary", action="store_true", help="require a canary (and the header record) even if the plan does not")
    ap.add_argument("--sha256", default=None, help="expected hash (overrides the plan)")
    ap.add_argument("--kind", default=None, choices=["noilai", "xcopa"], help="row schema (default: from the plan, else noilai)")
    ap.add_argument("--materialize", action="store_true",
                    help="with --key: derive the seeded file from its plan `derive` block when it does not exist yet")
    args = ap.parse_args(argv)

    if args.key:
        spec = load_item_file_spec(args.key, args.plan)
        if args.kind:
            spec = dict(spec, kind=args.kind)
        if args.materialize and spec.get("derive"):
            import kaggle_run_plan as KRP

            plan = KRP.load_plan(args.plan)
            info = KRP.derive_item_file(args.key, plan, project_root=args.root, force=False)
            print(f"[derive] {args.key}: {info['status']} -> {info['path']} ({info['n_items']} items, sha256 {info['sha256'][:16]}...)",
                  file=sys.stderr)
        res = verify_spec(spec, root=args.root, expected_sha256=args.sha256, expect_canary=args.expect_canary)
        expected = args.sha256 or spec.get("sha256")
    else:
        path = args.items
        kind = args.kind or "noilai"
        res = verify_items(path, expected_sha256=args.sha256, canary_required=args.expect_canary,
                           manifest_canary=manifest_canary_for(path) if args.expect_canary else None, kind=kind)
        expected = args.sha256
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["ok"] and expected is None:
        print("[note] no sha256 recorded for this file yet: record the hash above in configs/run_plan.yaml at the data freeze",
              file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
