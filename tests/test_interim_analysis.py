"""scripts/interim_analysis.py, the per-session interim analysis on completed units only (CLAUDE.md step 4),
exercised on local echo runs of a NON-frozen build: the INTERIM label and n on the title and every table,
the base-pair cluster-bootstrap cells, rescoring through score_run when scores.jsonl is absent, a run whose
item file is missing reported as unscorable instead of a crash, the ledger's done-only selection, the
fallback to a runs directory without a ledger, and the refusal to write under paper/."""
import csv
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
IA = importlib.import_module("interim_analysis")

PY = sys.executable
SMOKE_DEV = Path("/tmp/claude-0/-home-user-noilai/7ed5d88b-f91c-57ec-b5a3-07ddcf2f1a1c/scratchpad/smoke_build/v0.3-smoke/noilai_dev.jsonl")
DATE = "2026-10-07"
N_BOOT = 300          # enough for the interval checks below; the driver's default is constants.BOOTSTRAP_B


def _run(*args, check=True):
    return subprocess.run([PY, *args], cwd=ROOT, text=True, capture_output=True, check=check)


def _driver(*args):
    return _run("scripts/interim_analysis.py", *args, check=False)


@pytest.fixture(scope="module")
def item_file(tmp_path_factory) -> Path:
    """The local NON-frozen smoke build's dev file when present, else a small seeded build in a temporary
    directory (never under data/release/)."""
    if SMOKE_DEV.exists():
        return SMOKE_DEV
    out = tmp_path_factory.mktemp("rel")
    _run("scripts/build_data.py", "--out", str(out), "--seed", "3", "--n-lexicon", "120", "--n-pseudo", "60",
         "--per-cell-t1", "40", "--per-cell-t2", "20", "--per-cell-t3", "15", "--core-per-cell", "8")
    return out / "noilai_dev.jsonl"


def _set_manifest(run_dir: Path, run_id=None, model_key=None, item_path=None, sha=None, status=None) -> None:
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if run_id:
        m["run_id"] = m["identity"]["run_id"] = run_id
    if model_key:
        m["identity"]["model_key"] = m["model"]["name"] = model_key
    if item_path:
        m["item_file"]["path"] = m["data"]["item_file"] = item_path
    if sha:
        m["item_file"]["sha256"] = m["data"]["item_file_sha256"] = sha
        m["item_file"]["content_sha256"] = m["data"]["item_content_sha256"] = sha[::-1]
    if status:
        m["status"] = status
    (run_dir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def _copy_run(src: Path, dst: Path, scored: bool = True, **manifest) -> Path:
    shutil.copytree(src, dst)
    if not scored:
        (dst / "scores.jsonl").unlink()
        (dst / "summary.json").unlink()
    _set_manifest(dst, **manifest)
    return dst


@pytest.fixture(scope="module")
def runs(tmp_path_factory, item_file) -> Path:
    """A runs directory with: the echo run of the task (scored), a model whose answers alternate right / wrong,
    a run without scores whose item file resolves, a run without scores whose item file is gone, an unfinished
    run, and a directory that is not a run."""
    runs = tmp_path_factory.mktemp("runs")
    _run("scripts/run_eval.py", "--items", str(item_file), "--backend", "echo", "--smoke", "--limit", "40", "--score",
         "--run-id", "interim_test", "--out-root", str(runs))
    base = runs / "interim_test"
    assert (base / "scores.jsonl").exists() and (base / "manifest.json").exists()
    mixed = _copy_run(base, runs / "interim_mixed", run_id="interim_mixed", model_key="echo-mixed")
    rows = [json.loads(ln) for ln in (mixed / "scores.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, r in enumerate(rows):
        r["correct"] = i % 2 == 0
        r["error_class"] = "correct" if r["correct"] else "component"
    (mixed / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    _copy_run(base, runs / "interim_rescore", scored=False, run_id="interim_rescore", model_key="echo-rescore")
    _copy_run(base, runs / "interim_missing_items", scored=False, run_id="interim_missing_items", model_key="echo-gone",
              item_path=str(runs / "nowhere" / "items_gone_from_this_machine.jsonl"), sha="0" * 64)
    _copy_run(base, runs / "interim_running", run_id="interim_running", status="running")
    (runs / "not_a_run").mkdir()
    (runs / "not_a_run" / "notes.txt").write_text("no manifest here\n", encoding="utf-8")
    return runs


def test_runs_mode_writes_interim_markdown_and_json_with_n_and_reports_the_unscorable_run(runs, tmp_path):
    out = tmp_path / "interim"
    r = _driver("--runs", str(runs), "--out", str(out), "--date", DATE, "--n-boot", str(N_BOOT), "--quiet")
    assert r.returncode == 0, r.stderr
    md, js = out / f"{DATE}_interim.md", out / f"{DATE}_interim.json"
    assert md.exists() and js.exists()
    text = md.read_text(encoding="utf-8")
    rep = json.loads(js.read_text(encoding="utf-8"))
    # the label, the n and the monitoring-only sentence: on the title, on every table, in both files
    assert text.startswith("# INTERIM analysis " + DATE) and rep["label"] == "INTERIM" and rep["date"] == DATE
    assert IA.NOTICE in text and rep["notice"] == IA.NOTICE and "monitoring only" in IA.NOTICE
    headings = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(headings) >= 4
    for h in headings:
        assert "INTERIM" in h and re.search(r"n = [\d,]+ items, [\d,]+ base pairs, \d+ models", h), h
    n = rep["n"]
    assert n["runs"] == 3 and n["models"] == 3 and n["items"] >= 1 and n["base_pairs"] >= 1 and n["unscorable"] == 1
    assert IA.n_label(n) in headings[0]
    assert rep["source"]["mode"] == "runs" and rep["bootstrap"]["n_boot"] == N_BOOT
    assert rep["bootstrap"]["cluster"] == "base_pair_id" and rep["bootstrap"]["strata"] == "source"
    # completed runs: the scored echo run, the alternating copy, and the copy scored here through score_run
    by_run = {x["run_id"]: x for x in rep["runs"]}
    assert set(by_run) == {"interim_test", "interim_mixed", "interim_rescore"}
    assert by_run["interim_test"]["scored_now"] is False and by_run["interim_rescore"]["scored_now"] is True
    assert (runs / "interim_rescore" / "scores.jsonl").exists() and "scored here" in text
    assert by_run["interim_test"]["n_rows"] == 40 and by_run["interim_test"]["hash_verified"] == "n/a (no ledger)"
    assert by_run["interim_test"]["item_file_sha256"] and by_run["interim_test"]["arms"] == ["nfc"]
    # the echo returns the prompt: every row unparseable, a 0 % cell -> Wilson on n / DEFF (DESIGN_DECISIONS 8.2)
    cells = [c for c in rep["cells"] if c["run_id"] == "interim_test"]
    assert cells and all(set(IA.CELL_KEYS) <= set(c) for c in cells)
    assert sum(c["n_rows"] for c in cells) == 40
    for c in cells:
        assert c["estimate"] == 0.0 and c["lo"] == 0.0 and c["hi"] > 0.0 and c["method"] == "wilson_deff" and c["small_cell"]
        assert c["unparseable_rate"] == 1.0 and 1 <= c["n_base_pairs"] <= c["n_items"] <= c["n_rows"] and c["deff"] >= 1.0
        assert c["model"] == "echo" and c["level"] == 0.95
    # alternating answers: 20 correct rows in all, each estimate strictly inside its interval
    mixed = [c for c in rep["cells"] if c["run_id"] == "interim_mixed"]
    assert sum(c["estimate"] * c["n_rows"] for c in mixed) == pytest.approx(20)
    for c in mixed:
        assert c["lo"] < c["estimate"] < c["hi"] and c["n_boot"] in (0, N_BOOT)
        if c["n_base_pairs"] >= IA.constants.SMALL_CELL_MAX_BASE_PAIRS:
            assert c["method"] in ("percentile", "bca") and not c["small_cell"] and c["n_boot"] == N_BOOT
    # the Markdown carries the same numbers as the JSON
    big = max(mixed, key=lambda c: c["n_rows"])
    assert f"| {IA.pct(big['estimate'])} [{IA.pct(big['lo'])}, {IA.pct(big['hi'])}] | {big['method']} |" in text
    assert "| echo-mixed |" in text and "`interim_test`" in text
    # the run whose item file is gone: reported, not scored, not a crash; the unfinished run is skipped
    uns = {Path(u["run_dir"]).name: u for u in rep["unscorable"]}
    assert set(uns) == {"interim_missing_items"} and "item file" in uns["interim_missing_items"]["reason"]
    assert not (runs / "interim_missing_items" / "scores.jsonl").exists()
    assert "interim_missing_items" in text and "unscorable" in text
    skipped = {Path(s["run_dir"]).name: s["reason"] for s in rep["skipped"]}
    assert set(skipped) == {"interim_running"} and "running" in skipped["interim_running"]
    assert "Traceback" not in r.stderr


def test_ledger_mode_analyses_done_units_only_and_carries_the_hash_gate(runs, tmp_path):
    runs2 = tmp_path / "runs"
    runs2.mkdir()
    shutil.copytree(runs / "interim_test", runs2 / "by_path__echo")            # found through result_file
    shutil.copytree(runs / "interim_mixed", runs2 / "by_name__echo-mixed")      # found as <runs>/<run_id>__<model>
    shutil.copytree(runs / "interim_test", runs2 / "arm_line__echo")           # done unit is an arm the run has no rows for
    cols = ["unit_id", "run_id", "model", "arm", "status", "hash_verified", "result_file", "scores_file", "notes"]
    rows = [
        {"unit_id": "by_path__echo__nfc", "run_id": "by_path", "model": "echo", "arm": "nfc", "status": "done",
         "hash_verified": "no", "result_file": str(runs2 / "by_path__echo" / "outputs.jsonl")},
        {"unit_id": "by_name__echo-mixed__nfc", "run_id": "by_name", "model": "echo-mixed", "arm": "nfc", "status": "done",
         "hash_verified": "yes"},
        {"unit_id": "arm_line__echo__nfd", "run_id": "arm_line", "model": "echo", "arm": "nfd", "status": "done",
         "hash_verified": "yes"},
        {"unit_id": "arm_line__echo__nfc", "run_id": "arm_line", "model": "echo", "arm": "nfc", "status": "planned"},
        {"unit_id": "planned__echo__nfc", "run_id": "planned", "model": "echo", "arm": "nfc", "status": "planned",
         "result_file": str(runs2 / "by_path__echo" / "outputs.jsonl")},
        {"unit_id": "validation__A", "run_id": "validation", "model": "", "arm": "", "status": "done",
         "result_file": str(tmp_path / "validation_report.json")},
    ]
    ledger = tmp_path / "ledger.csv"
    with open(ledger, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    out = tmp_path / "interim"
    r = _driver("--ledger", str(ledger), "--runs", str(runs2), "--out", str(out), "--date", DATE, "--n-boot", str(N_BOOT),
                "--quiet")
    assert r.returncode == 0, r.stderr
    rep = json.loads((out / f"{DATE}_interim.json").read_text(encoding="utf-8"))
    text = (out / f"{DATE}_interim.md").read_text(encoding="utf-8")
    assert rep["source"]["mode"] == "ledger" and rep["source"]["ledger"] == str(ledger)
    by_run = {x["run_id"]: x for x in rep["runs"]}
    assert set(by_run) == {"interim_test", "interim_mixed"}                    # the manifests' run ids
    assert by_run["interim_test"]["unit_ids"] == ["by_path__echo__nfc"] and by_run["interim_test"]["hash_verified"] == "no"
    assert by_run["interim_mixed"]["unit_ids"] == ["by_name__echo-mixed__nfc"] and by_run["interim_mixed"]["hash_verified"] == "yes"
    assert "| no |" in text and rep["n"]["runs"] == 2 and rep["n"]["models"] == 2
    uns = {Path(u["run_dir"]).name: u for u in rep["unscorable"]}
    assert set(uns) == {"arm_line__echo"} and "nfd" in uns["arm_line__echo"]["reason"]
    assert uns["arm_line__echo"]["unit_ids"] == ["arm_line__echo__nfd"]
    skipped = {s["unit_id"]: s["reason"] for s in rep["skipped"]}
    assert set(skipped) == {"validation__A"} and "no run directory" in skipped["validation__A"]
    assert "validation__A" in text
    # the same ledger through the function API, with the real ledger's status vocabulary
    refs = IA.completed_from_ledger(ledger, runs2, [])
    assert sorted(ref.run_dir.name for ref in refs) == ["arm_line__echo", "by_name__echo-mixed", "by_path__echo"]
    assert next(ref for ref in refs if ref.run_dir.name == "arm_line__echo").arms == {"nfd"}


def test_a_missing_ledger_falls_back_to_the_runs_directory_and_empty_runs_still_write_a_labelled_report(runs, tmp_path):
    out = tmp_path / "interim"
    r = _driver("--ledger", str(tmp_path / "no_ledger.csv"), "--runs", str(runs), "--out", str(out), "--date", DATE,
                "--n-boot", str(N_BOOT), "--quiet")
    assert r.returncode == 0 and "does not exist" in r.stderr, r.stderr
    rep = json.loads((out / f"{DATE}_interim.json").read_text(encoding="utf-8"))
    assert rep["source"]["mode"] == "runs" and rep["n"]["runs"] == 3
    empty = tmp_path / "empty_runs"
    empty.mkdir()
    out2 = tmp_path / "interim_empty"
    r = _driver("--runs", str(empty), "--out", str(out2), "--date", DATE, "--quiet")
    assert r.returncode == 0, r.stderr
    text = (out2 / f"{DATE}_interim.md").read_text(encoding="utf-8")
    assert text.startswith("# INTERIM analysis") and "n = 0 items, 0 base pairs, 0 models" in text and IA.NOTICE in text
    rep = json.loads((out2 / f"{DATE}_interim.json").read_text(encoding="utf-8"))
    assert rep["n"] == {"items": 0, "base_pairs": 0, "models": 0, "runs": 0, "cells": 0, "unscorable": 0, "skipped": 0}
    r = _driver("--runs", str(empty), "--out", str(tmp_path / "x"), "--date", "7 Oct 2026")
    assert r.returncode == 2 and "YYYY-MM-DD" in r.stderr


def test_refuses_to_write_under_paper(runs):
    target = ROOT / "paper" / "interim_must_not_exist"
    r = _driver("--runs", str(runs), "--out", str(target), "--date", DATE)
    assert r.returncode == 2 and "paper/" in r.stderr and not target.exists()
    with pytest.raises(ValueError, match="paper/"):
        IA.refuse_paper(ROOT / "paper" / "figures")
    IA.refuse_paper(ROOT / "experiments" / "interim")                          # the default is fine


def test_cell_ci_recovers_clusters_from_the_item_file_and_uses_the_base_pair_bootstrap(runs, item_file):
    rows = [json.loads(ln) for ln in (runs / "interim_test" / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    stripped = [{k: v for k, v in r.items() if k not in ("base_pair_id", "source")} for r in rows]
    manifest = json.loads((runs / "interim_test" / "manifest.json").read_text(encoding="utf-8"))
    note = IA.attach_clusters(stripped, manifest)
    assert note and "base_pair_id" in note and all(r["base_pair_id"] == o["base_pair_id"] for r, o in zip(stripped, rows, strict=True))
    assert all(r["source"] == o["source"] for r, o in zip(stripped, rows, strict=True))
    gone = dict(manifest, item_file={"path": str(item_file.parent / "nope.jsonl"), "sha256": "f" * 64}, data={})
    with pytest.raises(IA.Unscorable, match="not resolvable"):
        IA.attach_clusters([{k: v for k, v in rows[0].items() if k != "base_pair_id"}], gone)
    # eight rows per synthetic base pair, every base pair of one type: stratified by `source` (DESIGN_DECISIONS 8.2),
    # and with 10 base pairs the small-cell rule gives Wilson on n / DEFF with a design effect above 1
    paired = []
    for i, r in enumerate(rows):
        for j in range(2):
            paired.append(dict(r, item_id=f"{r['item_id']}-{j}", base_pair_id=f"bp-{i // 4}",
                               source="lexicon" if (i // 4) % 2 == 0 else "pseudo", correct=(i // 4 + j) % 3 == 0))
    cell = IA.cell_ci(paired, n_boot=N_BOOT, seed=0)
    assert cell["n_rows"] == 80 and cell["n_items"] == 80 and cell["n_base_pairs"] == 10 and cell["stratified"]
    assert cell["lo"] < cell["estimate"] < cell["hi"] and cell["method"] == "wilson_deff" and cell["small_cell"]
    assert cell["deff"] > 1.0 and 0 <= cell["unparseable_rate"] <= 1
    # 40 base pairs of two rows each, correctness varying BETWEEN base pairs: the percentile bootstrap (fewer than
    # BCA_MIN_CLUSTERS base pairs) with a non-degenerate interval inside (0, 1)
    forty = [dict(r, item_id=f"{r['item_id']}-{j}", correct=i % 2 == 0) for i, r in enumerate(rows) for j in range(2)]
    cell = IA.cell_ci(forty, n_boot=N_BOOT, seed=0)
    assert cell["n_base_pairs"] == 40 and cell["estimate"] == 0.5 and cell["method"] == "percentile" and cell["stratified"]
    assert 0.0 < cell["lo"] < 0.5 < cell["hi"] < 1.0 and cell["n_boot"] == N_BOOT and not cell["small_cell"]
    # a base pair of two types is no base-pair stratification: the cell falls back to the unstratified bootstrap
    for k, r in enumerate(forty):
        r["source"] = "lexicon" if k % 2 == 0 else "pseudo"
    cell = IA.cell_ci(forty, n_boot=N_BOOT, seed=0)
    assert not cell["stratified"] and cell["method"] == "percentile" and cell["lo"] < 0.5 < cell["hi"]
