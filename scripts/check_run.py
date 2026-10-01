#!/usr/bin/env python
"""End-to-end check of one scored run directory: item gate -> rescoring -> statistics -> hashed results.

The first real run (and every pilot run) goes through this before anyone reads a number from it:

    python scripts/check_run.py --run data/runs/<run_id>                 # check, compute stats, write results_hashes.json
    python scripts/check_run.py --run data/runs/<run_id> --verify        # recompute every hash and compare

1. **item gate**: the item file the manifest names (resolved as scripts/score_run.py does) hashes to the manifest's
   `item_file_sha256`, and to the run plan's recorded hash when `--plan-key` is given (scripts/kaggle_verify_items.py);
2. **rescoring is deterministic**: scoring outputs.jsonl again gives byte-identical scores (canonical JSON);
3. **statistics** (`stats.json`): per arm, strict T1 accuracy with the base-pair cluster-bootstrap CI (DD 8.2), the
   copy-baseline gain the Gate 1 criterion uses (PREREG §10: accuracy - 0, CI lower bound > 0), the paired arm
   contrast against `nfc` (noilai.stats.tests.paired_test: bootstrap CI, paired t, McNemar mid-p), and, when present,
   the EXPLORATORY T1 forced-choice accuracy against its chance level (docs/FORCED_CHOICE_EXPLORATORY.md);
4. **hashes** (`results_hashes.json`): SHA-256 of outputs.jsonl, scores.jsonl, summary.json, manifest.json, stats.json,
   the item file, and one `results_sha256` over them; `--verify` fails on any difference.
Nothing here is a result about a model unless the run itself is a registered run; a stand-in or smoke run says so in
its manifest (`notes.smoke`) and this script copies that flag into stats.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from noilai.eval.run import sha256_file
from noilai.eval.score import score_outputs
from noilai.gen.generate import load_items
from noilai.stats.bootstrap import accuracy_ci
from noilai.stats.tests import paired_test

HASHED = ("outputs.jsonl", "scores.jsonl", "summary.json", "manifest.json", "stats.json")


def canonical_rows(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) for r in rows)


def resolve_items(run: Path, manifest: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    import score_run
    return Path(score_run.resolve_items_path(manifest))


def stats_for(scores: list[dict], n_boot: int) -> dict:
    out: dict = {"by_task_arm": {}, "paired_vs_nfc": {}, "t1_forced_choice": {}}
    by = defaultdict(list)
    for r in scores:
        by[(r["task"], r.get("arm"))].append(r)
    for (task, arm), rows in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        ok = [bool(r.get("correct")) for r in rows]
        cl = [r.get("base_pair_id") or r["item_id"] for r in rows]
        ci = accuracy_ci(ok, cl, n_boot=n_boot)
        out["by_task_arm"][f"{task}|{arm}"] = {"n": len(rows), "n_clusters": ci.n_clusters, "accuracy": ci.estimate,
                                               "ci": [ci.lo, ci.hi], "ci_method": ci.method,
                                               "gain_over_copy_baseline": ci.estimate,      # the copy baseline is 0 on T1
                                               "gain_ci_lower_bound_above_0": ci.lo > 0}
        fc = [r for r in rows if r.get("fc_correct") is not None]
        if fc:
            fci = accuracy_ci([bool(r["fc_correct"]) for r in fc], [r.get("base_pair_id") or r["item_id"] for r in fc], n_boot=n_boot)
            chance = sum(r["fc_chance"] for r in fc) / len(fc)
            out["t1_forced_choice"][f"{task}|{arm}"] = {"n": len(fc), "accuracy": fci.estimate, "ci": [fci.lo, fci.hi],
                                                        "chance": chance, "ci_lower_bound_above_chance": fci.lo > chance,
                                                        "label": "EXPLORATORY"}
    for task in sorted({t for t, _a in by}):
        base = {(r["item_id"], r.get("prompt_id")): r for r in by.get((task, "nfc"), [])}
        for (t, arm), rows in by.items():
            if t != task or arm == "nfc" or not base:
                continue
            pairs = [(base[(r["item_id"], r.get("prompt_id"))], r) for r in rows if (r["item_id"], r.get("prompt_id")) in base]
            if not pairs:
                continue
            a = [bool(x.get("correct")) for x, _ in pairs]
            b = [bool(y.get("correct")) for _, y in pairs]
            cl = [x.get("base_pair_id") or x["item_id"] for x, _ in pairs]
            pt = paired_test(a, b, cl, name=f"{task}: {arm} - nfc", n_boot=n_boot)
            out["paired_vs_nfc"][f"{task}|{arm}"] = {k: v for k, v in pt.__dict__.items()}
    return out


def hashes(run: Path, item_file: Path) -> dict:
    h = {name: sha256_file(run / name) for name in HASHED if (run / name).exists()}
    h["item_file"] = sha256_file(item_file) if item_file.exists() else None
    h["results_sha256"] = hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--items", default=None)
    ap.add_argument("--plan-key", default=None, help="the run plan's item_files key whose recorded SHA-256 must match")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--allow-thinking", action="store_true", help="the reasoning sub-study's runs (scripts/score_run.py)")
    args = ap.parse_args(argv)
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    item_file = resolve_items(run, manifest, args.items)
    report: dict = {"run": str(run), "checks": {}}
    if args.verify:
        rec = json.loads((run / "results_hashes.json").read_text(encoding="utf-8"))
        now = hashes(run, item_file)
        bad = {k: (rec.get(k), now.get(k)) for k in set(rec) | set(now) if rec.get(k) != now.get(k)}
        print(json.dumps({"verified": not bad, "differences": bad}, ensure_ascii=False))
        return 1 if bad else 0
    if not (run / "scores.jsonl").exists() or (run / "scores.jsonl").stat().st_mtime < (run / "outputs.jsonl").stat().st_mtime:
        import score_run  # score (or rescore after new rows, e.g. a resumed or parked API run)
        score_run.score_run_dir(run, item_file, quiet=True, allow_thinking=args.allow_thinking)
        report["scored_here"] = True
    # 1. item gate
    observed = sha256_file(item_file)
    recorded = (manifest.get("data") or {}).get("item_file_sha256")
    report["checks"]["item_file_matches_manifest"] = observed == recorded
    if args.plan_key:
        import kaggle_verify_items as KV
        rc = KV.main(["--key", args.plan_key])
        report["checks"]["item_file_matches_run_plan"] = rc == 0
    # 2. deterministic rescoring
    items = load_items(item_file)
    outs = [json.loads(line) for line in (run / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rescored = score_outputs(items, outs)
    stored = [json.loads(line) for line in (run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report["checks"]["rescoring_identical"] = canonical_rows(rescored) == canonical_rows(stored)
    # 3. statistics
    st = stats_for(stored, args.n_boot)
    st["smoke_or_standin"] = bool((manifest.get("notes") or manifest.get("options", {}).get("notes") or {}).get("smoke"))
    st["model"] = (manifest.get("model") or {}).get("name") or manifest.get("model")
    (run / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    # 4. hashes
    h = hashes(run, item_file)
    (run / "results_hashes.json").write_text(json.dumps(h, indent=1) + "\n", encoding="utf-8")
    report["results_sha256"] = h["results_sha256"]
    ok = all(v for v in report["checks"].values())
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
