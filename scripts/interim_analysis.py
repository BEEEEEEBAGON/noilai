"""Interim analysis on COMPLETED units only: per-cell accuracy with base-pair cluster-bootstrap CIs.

    python scripts/interim_analysis.py                                  # experiments/ledger.csv + data/runs -> experiments/interim/
    python scripts/interim_analysis.py --runs <dir> --out <dir>         # no ledger: every finished run directory under <dir>
    python scripts/interim_analysis.py --ledger experiments/ledger.csv --runs data/runs --out experiments/interim --date 2026-10-07

Step 4 of the session recipe (CLAUDE.md), run after `ledger.py ingest`. A unit is COMPLETED when
its ledger row has status `done` (scripts/ledger.py: the run finished and every expected row is
present). With a ledger (--ledger, or the default experiments/ledger.csv when neither --ledger nor
--runs is given) only the done units' run directories are analysed, and only their arms. Without
a ledger (--runs alone, or a --ledger path that does not exist) every directory under --runs whose
manifest.json has status `finished` counts; unfinished runs are listed as skipped.

For each completed run the driver reads manifest.json (identity.model_key, data.arms,
item_file.path / sha256), then scores.jsonl as written by scripts/score_run.py. When scores.jsonl is
absent and the item file resolves (score_run.resolve_items_path), the run is scored here through
score_run.score_run_dir; when the item file cannot be found the run is reported as UNSCORABLE and
skipped, never a crash. Per (task, variant, arm, prompt_id) it computes strict accuracy with the
base-pair cluster bootstrap of noilai.stats.bootstrap (DESIGN_DECISIONS 8.2: clusters =
`base_pair_id`, carried in the score rows or recovered from the item file by item_id; B =
constants.BOOTSTRAP_B with a fixed seed; stratified by base-pair type, lexical vs pseudo; BCa from
constants.BCA_MIN_CLUSTERS base pairs; Wilson on n / DEFF for cells with fewer than
constants.SMALL_CELL_MAX_BASE_PAIRS base pairs or at 0 % / 100 %), the number of items and of base
pairs, the realized design effect and the unparseable rate (DESIGN_DECISIONS 8.9).

Outputs: <out>/<UTC date>_interim.md and <out>/<UTC date>_interim.json with the same numbers. The
title and every table are labelled INTERIM with n (items, base pairs, models), and both files carry
the sentence "interim numbers are for monitoring only; no design or analysis change without a
docs/DEVIATIONS.md entry; no paper prose". Nothing is ever written under paper/: --out there is refused.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                 # the project root: noilai
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))     # scripts/score_run.py

import score_run as SR

from noilai import constants
from noilai.eval.run import STATUS_FINISHED, STATUS_THINKING, load_item_file, read_manifest
from noilai.eval.score import read_jsonl
from noilai.stats import bootstrap

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "ledger.csv"
RUNS = ROOT / "data" / "runs"
OUT_DIR = ROOT / "experiments" / "interim"
PAPER_DIR = ROOT / "paper"                         # CLAUDE.md: this workstream never writes under paper/
LABEL = "INTERIM"
NOTICE = ("interim numbers are for monitoring only; no design or analysis change without a "
          "docs/DEVIATIONS.md entry; no paper prose")
COMPLETED_LEDGER_STATUS = "done"                   # scripts/ledger.py ingest: finished and every expected row present
COMPLETED_MANIFEST_STATUSES = (STATUS_FINISHED, STATUS_THINKING)   # a thinking-present main run is finished but unscorable
CELL_KEYS = ("task", "variant", "arm", "prompt_id")
CLUSTER_KEY = constants.BOOTSTRAP_CLUSTER          # base_pair_id: the sampled unit and the cluster (DESIGN_DECISIONS 8.1)
STRATUM_KEY = "source"                             # lexicon / pseudo base pair: the DESIGN_DECISIONS 8.2 stratification
BOOTSTRAP_SEED = 0                                 # DESIGN_DECISIONS 8.2: a fixed seed
CI_LEVEL = 0.95                                    # DESIGN_DECISIONS 8.9: the 95 % cluster-bootstrap interval per cell
SHA_SHORT = 12                                     # characters of a sha256 shown in the Markdown tables


class Unscorable(RuntimeError):
    """A completed run whose numbers cannot be computed here (reported, never a crash)."""


@dataclass
class RunRef:
    """One completed run directory and the ledger units that point at it."""
    run_dir: Path
    unit_ids: list[str] = field(default_factory=list)
    arms: set[str] | None = None                   # ledger mode: the arms whose units are done; None = every arm
    hash_verified: str = "n/a (no ledger)"         # the ledger's item-file hash gate (DESIGN_DECISIONS 4.6)


# ------------------------------------------------------------------ which runs are completed
def read_ledger(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def locate_run_dir(row: dict, runs_root: Path) -> Path | None:
    """The run directory of a done ledger row: the parent of its scores_file / result_file (as
    recorded, relative to the project root, or by name under --runs), else <runs>/<run_id>__<model>
    (the ingest convention of scripts/ledger.py). None when no candidate holds a manifest.json."""
    candidates: list[Path] = []
    for key in ("scores_file", "result_file"):
        recorded = (row.get(key) or "").strip()
        if recorded:
            parent = Path(recorded).parent
            candidates += [parent, ROOT / parent, runs_root / parent.name]
    if row.get("run_id") and row.get("model"):
        candidates.append(runs_root / f"{row['run_id']}__{row['model']}")
    for cand in candidates:
        if (cand / "manifest.json").exists():
            return cand
    return None


def completed_from_ledger(ledger: Path, runs_root: Path, skipped: list[dict]) -> list[RunRef]:
    """The run directories of the ledger's done units, one RunRef per directory with the done arms."""
    refs: dict[Path, RunRef] = {}
    for row in read_ledger(ledger):
        if row.get("status") != COMPLETED_LEDGER_STATUS:
            continue
        run_dir = locate_run_dir(row, runs_root)
        if run_dir is None:
            skipped.append({"unit_id": row.get("unit_id"), "run_dir": None,
                            "reason": "done but no run directory with a manifest.json "
                                      f"(result_file {row.get('result_file') or '-'}): a human or probe deliverable, "
                                      "or a run not mirrored on this machine"})
            continue
        key = run_dir.resolve()
        ref = refs.get(key)
        if ref is None:
            ref = refs[key] = RunRef(key, arms=set(), hash_verified=row.get("hash_verified") or "n/a")
        elif ref.hash_verified != (row.get("hash_verified") or "n/a"):
            ref.hash_verified = "mixed"
        ref.unit_ids.append(row.get("unit_id") or "")
        ref.arms.add(row.get("arm") or "nfc")
    return [refs[k] for k in sorted(refs)]


def completed_from_runs(runs_root: Path, skipped: list[dict]) -> list[RunRef]:
    """Every run directory under `runs_root` whose manifest says the run finished."""
    refs: list[RunRef] = []
    if not runs_root.exists():
        return refs
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not (d / "manifest.json").exists():
            continue
        try:
            status = read_manifest(d).get("status")
        except (OSError, json.JSONDecodeError) as e:
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": f"unreadable manifest.json: {e}"})
            continue
        if status not in COMPLETED_MANIFEST_STATUSES:
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": f"not completed (manifest status {status!r})"})
            continue
        refs.append(RunRef(d.resolve()))
    return refs


# ------------------------------------------------------------------ scores and clusters
def load_scores(run_dir: Path, rescore: bool = True) -> tuple[list[dict], bool]:
    """The run's score rows and whether they were scored just now. Without scores.jsonl the run is
    scored through score_run.score_run_dir when `rescore` (it needs the item file); raises
    Unscorable with the reason otherwise."""
    path = run_dir / "scores.jsonl"
    scored_now = False
    if not path.exists():
        if not rescore:
            raise Unscorable("no scores.jsonl (run scripts/score_run.py, or drop --no-rescore)")
        try:
            SR.score_run_dir(run_dir, quiet=True)
        except FileNotFoundError as e:                       # the item file is not on this machine
            raise Unscorable(f"no scores.jsonl and the item file is not resolvable: {e}") from e
        except SR.ThinkingPresentError as e:
            raise Unscorable(str(e)) from e
        except (KeyError, ValueError, OSError) as e:         # outputs for unknown items, unreadable files
            raise Unscorable(f"scoring failed: {type(e).__name__}: {e}") from e
        scored_now = True
    rows = read_jsonl(path)
    if not rows:
        raise Unscorable("scores.jsonl is empty")
    return rows, scored_now


def attach_clusters(rows: list[dict], manifest: dict) -> str | None:
    """Make sure every row carries CLUSTER_KEY: scores.jsonl copies base_pair_id from the item, older
    scores are completed from the item file by item_id. Returns a note, or raises Unscorable."""
    missing = [r for r in rows if not r.get(CLUSTER_KEY)]
    if not missing:
        return None
    try:
        items_path = SR.resolve_items_path(manifest)
        items, _ = load_item_file(items_path)
    except FileNotFoundError as e:
        raise Unscorable(f"{len(missing)} score rows carry no {CLUSTER_KEY} and the item file is not "
                         f"resolvable: {e}") from e
    by_id = {it["item_id"]: it for it in items}
    unresolved = 0
    for r in missing:
        it = by_id.get(r.get("item_id"))
        if it and it.get(CLUSTER_KEY):
            r[CLUSTER_KEY] = it[CLUSTER_KEY]
            if r.get(STRATUM_KEY) is None and it.get(STRATUM_KEY) is not None:
                r[STRATUM_KEY] = it[STRATUM_KEY]
        else:
            unresolved += 1
    if unresolved:
        raise Unscorable(f"{unresolved} score rows have no {CLUSTER_KEY} in the scores or in the item file")
    return f"{len(missing)} rows took {CLUSTER_KEY} from the item file {items_path.name}"


# ------------------------------------------------------------------ the numbers
def cell_ci(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Strict accuracy of one cell with its base-pair cluster-bootstrap CI (DESIGN_DECISIONS 8.2),
    the counts of rows, items and base pairs, the realized design effect and the unparseable rate."""
    correct = [bool(r.get("correct")) for r in rows]
    clusters = [r[CLUSTER_KEY] for r in rows]
    strata = [r.get(STRATUM_KEY) for r in rows]
    stratified = all(s is not None for s in strata)
    kw = {"n_boot": n_boot, "seed": seed, "level": CI_LEVEL}
    try:
        ci = bootstrap.accuracy_ci(correct, clusters, strata=strata if stratified else None, **kw)
    except ValueError:              # a base pair in two strata is not a base-pair stratification: unstratified
        stratified = False
        ci = bootstrap.accuracy_ci(correct, clusters, **kw)
    values = np.asarray(correct, dtype=float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    deff = bootstrap.design_effect_from_clusters(values, inv.ravel(), ci.n_clusters)
    n_unparseable = sum(1 for r in rows if r.get("error_class") == "unparseable")
    return {**ci.as_dict(), "n_rows": len(rows), "n_items": len({r.get("item_id") for r in rows}),
            "n_base_pairs": ci.n_clusters, "stratified": stratified, "deff": round(float(deff), 3),
            "unparseable_rate": n_unparseable / len(rows)}


def analyse_run(ref: RunRef, n_boot: int, seed: int, rescore: bool) -> tuple[dict, list[dict], set, set]:
    """One completed run: its summary row, its cells, and the item and base-pair ids it covers."""
    manifest = read_manifest(ref.run_dir)
    identity = manifest.get("identity") or {}
    info = manifest.get("item_file") if isinstance(manifest.get("item_file"), dict) else {}
    data = manifest.get("data") or {}
    run_id = identity.get("run_id") or manifest.get("run_id") or ref.run_dir.name
    model = identity.get("model_key") or (manifest.get("model") or {}).get("name") or "?"
    rows, scored_now = load_scores(ref.run_dir, rescore)
    note = attach_clusters(rows, manifest)
    if ref.arms is not None:
        rows = [r for r in rows if (r.get("arm") or "nfc") in ref.arms]
        if not rows:
            raise Unscorable(f"no score rows for the completed arms {sorted(ref.arms)}")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k)) for k in CELL_KEYS)].append(r)
    cells = [{"model": model, "run_id": run_id, **dict(zip(CELL_KEYS, key, strict=True)),
              **cell_ci(groups[key], n_boot, seed)} for key in sorted(groups)]
    item_ids = {r.get("item_id") for r in rows}
    base_pairs = {r[CLUSTER_KEY] for r in rows}
    run = {"run_id": run_id, "model": model, "run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids),
           "hash_verified": ref.hash_verified, "manifest_status": manifest.get("status"),
           "item_file": info.get("path") or data.get("item_file"),
           "item_file_sha256": info.get("sha256") or data.get("item_file_sha256"),
           "arms": sorted({r.get("arm") or "nfc" for r in rows}),
           "prompt_ids": sorted({str(r.get("prompt_id")) for r in rows}),
           "n_rows": len(rows), "n_items": len(item_ids), "n_base_pairs": len(base_pairs),
           "scored_now": scored_now, "cluster_note": note}
    return run, cells, item_ids, base_pairs


def analyse(refs: list[RunRef], n_boot: int, seed: int, rescore: bool, source: dict, skipped: list[dict],
            date: str) -> dict:
    """The report: every completed run that could be scored, its cells, the unscorable runs, n."""
    runs, cells, unscorable = [], [], []
    all_items: set = set()
    all_pairs: set = set()
    for ref in refs:
        try:
            run, run_cells, item_ids, base_pairs = analyse_run(ref, n_boot, seed, rescore)
        except Unscorable as e:
            unscorable.append({"run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids), "reason": str(e)})
            continue
        except (OSError, json.JSONDecodeError) as e:
            unscorable.append({"run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids),
                               "reason": f"unreadable run directory: {e}"})
            continue
        runs.append(run)
        cells.extend(run_cells)
        all_items |= item_ids
        all_pairs |= base_pairs
    n = {"items": len(all_items), "base_pairs": len(all_pairs), "models": len({r["model"] for r in runs}),
         "runs": len(runs), "cells": len(cells), "unscorable": len(unscorable), "skipped": len(skipped)}
    return {"label": LABEL, "notice": NOTICE, "date": date,
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"), "source": source,
            "n": n, "bootstrap": {"n_boot": n_boot, "seed": seed, "cluster": CLUSTER_KEY, "strata": STRATUM_KEY,
                                  "level": CI_LEVEL, "bca_min_clusters": constants.BCA_MIN_CLUSTERS,
                                  "small_cell_max_base_pairs": constants.SMALL_CELL_MAX_BASE_PAIRS,
                                  "design": "DESIGN_DECISIONS 8.2"},
            "runs": runs, "cells": cells, "unscorable": unscorable, "skipped": skipped}


# ------------------------------------------------------------------ rendering and writing
def n_label(n: dict) -> str:
    return f"n = {n['items']:,} items, {n['base_pairs']:,} base pairs, {n['models']} models"


def pct(x: float | None) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def _short_sha(sha: str | None) -> str:
    return (sha or "-")[:SHA_SHORT]


def render_markdown(rep: dict) -> str:
    n, b, src = rep["n"], rep["bootstrap"], rep["source"]
    where = f"`{src['ledger']}` (done units) and `{src['runs']}`" if src["mode"] == "ledger" else f"`{src['runs']}` (no ledger)"
    provenance = (f"_Generated by `scripts/interim_analysis.py` on {rep['generated_utc']} from {where}: {n['runs']} "
                  f"completed runs analysed, {n['unscorable']} unscorable, {n['skipped']} skipped. Intervals: base-pair "
                  f"cluster bootstrap, B = {b['n_boot']}, seed {b['seed']}, clusters = `{b['cluster']}`, stratified by "
                  f"`{b['strata']}`, {int(CI_LEVEL * 100)} % level; `wilson_deff` for cells with fewer than "
                  f"{b['small_cell_max_base_pairs']} base pairs or at 0 % / 100 %, `bca` from {b['bca_min_clusters']} "
                  f"base pairs, `percentile` otherwise ({b['design']})._")
    L = [f"# {LABEL} analysis {rep['date']} ({n_label(n)})", "", f"**{rep['notice']}**", "", provenance, ""]
    L += [f"## {LABEL}: completed runs ({n_label(n)})", "",
          "| run | model | item file | sha256 | hash gate | arms | prompts | rows | items | base pairs | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        notes = [x for x in (("scored here" if r["scored_now"] else None), r["cluster_note"]) if x]
        L.append(f"| `{r['run_id']}` | {r['model']} | {Path(r['item_file'] or '-').name} | `{_short_sha(r['item_file_sha256'])}` "
                 f"| {r['hash_verified']} | {', '.join(r['arms'])} | {', '.join(r['prompt_ids'])} | {r['n_rows']} "
                 f"| {r['n_items']} | {r['n_base_pairs']} | {'; '.join(notes) or ''} |")
    if not rep["runs"]:
        L.append("| (no completed run) | | | | | | | | | | |")
    L += ["", f"## {LABEL}: strict accuracy per cell ({n_label(n)})", "",
          ("| model | run | task | variant | arm | prompt | accuracy % [95 % CI] | method | items | base pairs "
           "| DEFF | unparseable % | flag |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep["cells"]:
        flags = [x for x in (("SMALL CELL" if c["small_cell"] else None), (None if c["stratified"] else "unstratified")) if x]
        L.append(f"| {c['model']} | `{c['run_id']}` | {c['task']} | {c['variant']} | {c['arm']} | {c['prompt_id']} "
                 f"| {pct(c['estimate'])} [{pct(c['lo'])}, {pct(c['hi'])}] | {c['method']} | {c['n_items']} "
                 f"| {c['n_base_pairs']} | {c['deff']:.2f} | {pct(c['unparseable_rate'])} | {', '.join(flags)} |")
    if not rep["cells"]:
        L.append("| (no cell) | | | | | | | | | | | | |")
    L += ["", f"## {LABEL}: unscorable runs ({n['unscorable']} of {n['unscorable'] + n['runs']} completed; {n_label(n)})", "",
          "| run directory | units | reason |", "|---|---|---|"]
    for u in rep["unscorable"]:
        L.append(f"| `{Path(u['run_dir']).name}` | {', '.join(u['unit_ids']) or '-'} | {u['reason']} |")
    if not rep["unscorable"]:
        L.append("| (none) | | |")
    if rep["skipped"]:
        L += ["", f"## {LABEL}: skipped units and directories ({n['skipped']}; {n_label(n)})", "",
              "| unit / directory | reason |", "|---|---|"]
        L += [f"| `{s['unit_id'] or Path(s['run_dir']).name}` | {s['reason']} |" for s in rep["skipped"]]
    L += ["", rep["notice"], ""]
    return "\n".join(L)


def refuse_paper(path: Path) -> None:
    """CLAUDE.md: nothing of this workstream is written under paper/."""
    if path.resolve().is_relative_to(PAPER_DIR.resolve()):
        raise ValueError(f"refusing to write under paper/ ({path}): the experiments workstream never writes paper files")


def write_report(rep: dict, out_dir: Path) -> tuple[Path, Path]:
    refuse_paper(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{rep['date']}_interim.md"
    js = out_dir / f"{rep['date']}_interim.json"
    md.write_text(render_markdown(rep), encoding="utf-8")
    js.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return md, js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=None,
                    help=f"the unit ledger (default {LEDGER.relative_to(ROOT)} unless --runs alone is given); "
                         "a path that does not exist falls back to scanning --runs")
    ap.add_argument("--runs", type=Path, default=None, help=f"run directories (default {RUNS.relative_to(ROOT)})")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help=f"report directory (default {OUT_DIR.relative_to(ROOT)})")
    ap.add_argument("--date", default=None, help="UTC date of the report files, YYYY-MM-DD (default: today)")
    ap.add_argument("--n-boot", type=int, default=constants.BOOTSTRAP_B, help="bootstrap replicates (DD 8.2)")
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="bootstrap seed (DD 8.2: fixed)")
    ap.add_argument("--no-rescore", action="store_true", help="never score here: a run without scores.jsonl is unscorable")
    ap.add_argument("--quiet", action="store_true", help="write the files without printing the Markdown")
    args = ap.parse_args(argv)
    try:
        refuse_paper(args.out)
    except ValueError as e:
        ap.error(str(e))
    if args.date is None:
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    else:
        try:
            date = dt.date.fromisoformat(args.date).isoformat()
        except ValueError:
            ap.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    ledger = args.ledger or LEDGER
    runs_root = args.runs or RUNS
    use_ledger = ledger.exists() and (args.ledger is not None or args.runs is None)
    skipped: list[dict] = []
    if use_ledger:
        refs = completed_from_ledger(ledger, runs_root, skipped)
        source = {"mode": "ledger", "ledger": str(ledger), "runs": str(runs_root)}
    else:
        if args.ledger is not None:
            print(f"ledger {ledger} does not exist: scanning {runs_root} for finished runs", file=sys.stderr)
        refs = completed_from_runs(runs_root, skipped)
        source = {"mode": "runs", "ledger": None, "runs": str(runs_root)}
    rep = analyse(refs, args.n_boot, args.seed, not args.no_rescore, source, skipped, date)
    md, js = write_report(rep, args.out)
    if not args.quiet:
        print(render_markdown(rep))
    print(f"wrote {md} and {js}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
