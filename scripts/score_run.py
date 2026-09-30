#!/usr/bin/env python
"""Score a run directory: scores.jsonl + summary.json, and print the summary table.

    python scripts/score_run.py --run data/runs/<run_id> [--items PATH] [--audit data/audit/gemma3_rows.csv]

The item file defaults to the one recorded in the run's manifest. --audit joins the
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
from noilai.eval.run import load_item_file, read_manifest, read_outputs


def score_run_dir(run_dir: Path, items_path: Path | None = None, audit: Path | None = None,
                  quiet: bool = False) -> dict:
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    if items_path is None:
        items_path = Path(manifest["item_file"]["path"])
        if not items_path.is_absolute() and not items_path.exists():
            items_path = ROOT / items_path
    items, _ = load_item_file(items_path)
    outputs = read_outputs(run_dir)
    audit_rows = S.load_audit_rows(audit) if audit else None
    rows = S.score_outputs(items, outputs, audit_rows=audit_rows)
    S.write_scores(rows, run_dir / "scores.jsonl")
    agg = S.aggregate(rows)
    agg["run_id"] = manifest.get("run_id")
    agg["model"] = (manifest.get("model") or {}).get("name")
    agg["item_file_sha256"] = (manifest.get("item_file") or {}).get("sha256")
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
