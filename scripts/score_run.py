#!/usr/bin/env python
"""Score a run directory: scores.jsonl + summary.json, and print the summary table.

    python scripts/score_run.py --run data/runs/<run_id> [--items PATH] [--audit data/audit/gemma3_rows.csv]

The item file defaults to the one recorded in the run's manifest. A run made on another
machine (Kaggle: /kaggle/working/...) records an absolute path that does not exist here, so
the file is resolved in this order: the path as recorded; the path relative to the project
root; a file under data/release/*/ with the manifest's item sha256 (then content sha256);
a file under data/release/*/ or data/external/ with the same basename. --audit joins the
tokenizer audit (tokens per input syllable) into every row as `n_input_tokens_syll`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.eval import score as S
from noilai.eval.run import item_content_sha256, load_item_file, read_manifest, read_outputs, sha256_file


def resolve_items_path(manifest: dict, root: Path = ROOT) -> Path:
    """The item file of a run, found as documented above; raises FileNotFoundError with the
    candidates tried."""
    info = manifest.get("item_file") or {}
    recorded = Path(str(info.get("path") or (manifest.get("data") or {}).get("item_file") or ""))
    tried = []
    if str(recorded):
        for cand in (recorded, root / recorded):
            tried.append(cand)
            if cand.exists():
                return cand
        # the recorded path was absolute on another machine: try its tail relative to `data/`
        parts = recorded.parts
        if "data" in parts:
            cand = root / Path(*parts[parts.index("data"):])
            tried.append(cand)
            if cand.exists():
                return cand
    pool = sorted(list((root / "data" / "release").glob("*/*.jsonl")) + list((root / "data" / "external").glob("*.jsonl")))
    want = info.get("sha256")
    if want:
        for cand in pool:
            if sha256_file(cand) == want:
                return cand
    want_c = info.get("content_sha256")
    if want_c:
        for cand in pool:
            try:
                items, _ = load_item_file(cand)
            except (OSError, ValueError) as e:      # not a readable jsonl item file: not a candidate
                print(f"skipping {cand}: {e}", file=sys.stderr)
                continue
            if item_content_sha256(items) == want_c:
                return cand
    if recorded.name:
        same = [c for c in pool if c.name == recorded.name]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            raise FileNotFoundError(f"several files named {recorded.name} under data/: {[str(c) for c in same]}; "
                                    f"pass --items")
    raise FileNotFoundError(f"cannot find the run's item file; tried {[str(t) for t in tried]} and "
                            f"{len(pool)} files under data/release, data/external (sha256 {want}); pass --items")


def score_run_dir(run_dir: Path, items_path: Path | None = None, audit: Path | None = None,
                  quiet: bool = False) -> dict:
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    if items_path is None:
        items_path = resolve_items_path(manifest)
        if not quiet:
            print(f"items: {items_path}")
    items, _ = load_item_file(items_path)
    outputs = read_outputs(run_dir)
    audit_rows = S.load_audit_rows(audit) if audit else None
    rows = S.score_outputs(items, outputs, audit_rows=audit_rows)
    S.write_scores(rows, run_dir / "scores.jsonl")
    agg = S.aggregate(rows)
    agg["run_id"] = manifest.get("run_id")
    agg["model"] = (manifest.get("model") or {}).get("name")
    agg["item_file_sha256"] = (manifest.get("item_file") or {}).get("sha256")
    agg["items_path"] = str(items_path)
    agg["n_unchanged"] = manifest.get("n_unchanged")
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    if not quiet:
        print(f"run {agg['run_id']}  model {agg['model']}  rows {agg['n_rows']}")
        print(S.summary_table(agg))
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=None)
    ap.add_argument("--audit", type=Path, default=None)
    args = ap.parse_args(argv)
    score_run_dir(args.run, args.items, args.audit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
