"""noilai.stats.analyze, the pre-registered analysis entry point (PREREGISTRATION 8.13, DESIGN_DECISIONS 8.8): the
configuration mirrors noilai.constants (the Table 3 Holm family, alpha, the bootstrap settings); statsmodels' Holm equals
noilai.stats.tests.holm; the CLI on an empty runs root writes every labelled empty table and exits 0; on copies of
the local echo run (skipped without it) it produces Table 2 cells, pools a paraphrase split, emits Table 3 rows with
Holm within the model row and `0 by construction` rows for a normalizing census; --check compares bodies ignoring the
stamp lines; writing under paper/ is refused; FINAL is refused while units of the plan are missing or the gates are
unrecorded; the ledger mode analyses done, hash-verified units only. Model names in the fixtures are abstract (echo)."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from noilai import constants
from noilai.stats import analyze as AN
from noilai.stats import tests as st

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "analysis.yaml"
PY = sys.executable
DATE = "2026-10-12"
N_BOOT = 200
# the scored echo run of the task (a NON-frozen smoke build; never under data/release/): $NOILAI_ECHO_RUN when set, else the
# git-ignored data/smoke_build/runs_echo/smoke_echo_20__echo; the echo-run tests skip when it is absent (CI has neither)
ECHO_RUN = Path(os.environ.get("NOILAI_ECHO_RUN", "").strip()
                or str(Path(__file__).resolve().parents[1] / "data" / "smoke_build" / "runs_echo" / "smoke_echo_20__echo"))


def _cli(*args, check=False):
    return subprocess.run([PY, "-m", "noilai.stats.analyze", "--config", str(CONFIG), *map(str, args)], cwd=ROOT, text=True,
                          capture_output=True, check=check)


# ------------------------------------------------------------------ configuration and Holm
def test_analysis_yaml_mirrors_the_constants_and_the_holm_family():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert AN.config_table3_family(cfg) == constants.HOLM_FAMILY_TABLE3
    assert len(cfg["holm"]["table3_family"]) == constants.HOLM_FAMILY_TABLE3_CELLS
    assert cfg["holm"]["alpha"] == constants.ALPHA and cfg["bootstrap"]["B"] == constants.BOOTSTRAP_B
    assert cfg["bootstrap"]["cluster"] == constants.BOOTSTRAP_CLUSTER
    assert cfg["arms"]["aliases"] == {"base": "nfc", "pc": "win1258"}
    loaded = AN.load_config(CONFIG)                              # validates without raising
    assert loaded["labels"]["final_requires"]
    fam = AN.tokenizer_family_map(loaded)
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    panel_groups = {g for g, v in models["groups"].items() if v.get("panel")}
    panel = {e["name"] for e in models["models"] if e.get("group") in panel_groups}
    assert set(fam) == panel and len(panel) == 21                 # every panel entry has a tokenizer family
    assert len(set(fam.values())) == 7                           # the seven families of PREREG 8.1
    # a disagreement with the constants is refused
    bad = dict(loaded, holm=dict(loaded["holm"], alpha=0.1))
    with pytest.raises(ValueError, match="alpha"):
        AN.validate_config(bad)
    bad = dict(loaded, holm=dict(loaded["holm"], table3_family=loaded["holm"]["table3_family"][:-1]))
    with pytest.raises(ValueError, match="HOLM_FAMILY_TABLE3"):
        AN.validate_config(bad)


def test_statsmodels_holm_equals_the_in_house_holm_on_random_inputs():
    rng = np.random.default_rng(7)
    for _ in range(100):
        m = int(rng.integers(1, 15))
        p = rng.uniform(0, 1, size=m) ** rng.uniform(0.5, 3)
        reject, adj = AN.holm_adjust(p)
        adj2, rej2 = st.holm(p, constants.ALPHA)
        assert np.allclose(adj, adj2) and np.array_equal(reject, rej2)
    reject, adj = AN.holm_adjust([])
    assert len(reject) == 0 and len(adj) == 0


def test_run_id_and_prompt_id_parsing_follow_the_conventions():
    assert AN.split_run_id("E1_main__gemma-x__p2") == ("E1_main", "gemma-x", "p2")
    assert AN.split_run_id("E3_noilai__m") == ("E3_noilai", "m", "") and AN.split_run_id("plain") == ("plain", "", "")
    assert AN.paraphrase_of("vi-p1-s3-explained-raw") == "p1" and AN.paraphrase_of("xcopa-p0") == "p0"
    assert AN.paraphrase_of("odd") is None
    assert AN.condition_of("vi-p2-s3-explained-components") == "vi-*-s3-explained-components"
    assert AN.item_file_class("noilai_c2", None) == "c2" and AN.item_file_class(None, "/x/noilai_main.jsonl") == "main"
    assert AN.item_file_class("xcopa_vi_test", None) == "xcopa" and AN.item_file_class("noilai_core", None) == "core"


# ------------------------------------------------------------------ the CLI on an empty runs root
def test_cli_on_an_empty_runs_root_writes_labelled_empty_tables_and_exits_0(tmp_path):
    empty = tmp_path / "runs"
    empty.mkdir()
    out = tmp_path / "out"
    r = _cli("--runs", empty, "--out", out, "--date", DATE, "--quiet")
    assert r.returncode == 0, r.stderr
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    names = [n for n in cfg["outputs"] if n not in ("analysis_summary", "results_log_entry")]
    for name in names:
        tex, js = out / f"{name}.tex", out / f"{name}.json"
        assert tex.exists() and js.exists(), name
        lines = tex.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("% GENERATED by noilai.stats.analyze from 0 run directories") and "do not edit by hand" in lines[0]
        assert lines[1].startswith("% generator commit ") and "content sha256 " in lines[1] and f"generated {DATE}" in lines[1]
        assert lines[2].startswith("% INTERIM (n = 0 items, 0 base pairs, 0 models)") and AN.NOTICE in lines[2]
        assert "(no completed run)" in tex.read_text(encoding="utf-8")
        rep = json.loads(js.read_text(encoding="utf-8"))
        assert rep["label"] == "INTERIM" and rep["rows"] == [] and rep["inputs"] == [] and rep["notice"] == AN.NOTICE
        assert rep["n"] == {"items": 0, "base_pairs": 0, "models": 0, "runs": 0, "lines": 0, "cells": 0, "unscorable": 0, "skipped": 0}
        assert rep["computable"] is False and rep["reason"]
    summ = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summ["label"] == "INTERIM" and summ["n"]["items"] == 0
    assert summ["exact_call"].startswith("python -m noilai.stats.analyze --config") and f"--date {DATE}" in summ["exact_call"]
    assert summ["config"]["sha256"] == AN.sha256_file(CONFIG) and summ["git"]["commit"]
    assert set(summ["analyses"]) == set(AN.ANALYSES) and all(not a["computable"] and a["reason"] for a in summ["analyses"].values())
    assert summ["final_blockers"] and any("registration" in b for b in summ["final_blockers"])
    entry = (out / "results_log_entry.md").read_text(encoding="utf-8")
    assert entry.startswith(f"## {DATE} — Pre-registered analysis run, INTERIM (no model outputs) [RL-{DATE}-NN]")
    assert "\nSource: `noilai.stats.analyze`" in entry and summ["exact_call"] in entry and AN.NOTICE in entry
    report = (out / "analysis_report.md").read_text(encoding="utf-8")
    assert report.startswith("# INTERIM (n = 0 items, 0 base pairs, 0 models) analysis " + DATE) and AN.NOTICE in report
    for h in (ln for ln in report.splitlines() if ln.startswith("#")):
        assert "INTERIM (n = 0 items" in h, h
    # --check: current, then stale after a body edit (the stamp lines are ignored)
    r = _cli("--runs", empty, "--out", out, "--date", DATE, "--check")
    assert r.returncode == 0, r.stderr
    tex = out / "table2_main.tex"
    text = tex.read_text(encoding="utf-8").replace(f"generated {DATE}", "generated 2099-01-01 00:00Z")
    tex.write_text(text, encoding="utf-8")
    assert _cli("--runs", empty, "--out", out, "--date", DATE, "--check").returncode == 0
    tex.write_text(text.replace("(no completed run)", "(edited)"), encoding="utf-8")
    r = _cli("--runs", empty, "--out", out, "--date", DATE, "--check")
    assert r.returncode == 1 and "stale" in r.stdout + r.stderr
    r = _cli("--runs", empty, "--out", tmp_path / "x", "--date", "12 Oct 2026")
    assert r.returncode == 2 and "YYYY-MM-DD" in r.stderr


def test_refuses_to_write_under_paper(tmp_path):
    empty = tmp_path / "runs"
    empty.mkdir()
    target = ROOT / "paper" / "analysis_must_not_exist"
    r = _cli("--runs", empty, "--out", target, "--date", DATE)
    assert r.returncode == 2 and "paper/" in r.stderr and not target.exists()
    with pytest.raises(ValueError, match="paper/"):
        AN.refuse_paper(ROOT / "paper" / "tables")
    AN.refuse_paper(ROOT / "experiments" / "analysis")


# ------------------------------------------------------------------ the echo run
def _copy_run(dst: Path, run_id: str, model: str | None = None, rows_fn=None, census: dict | None = None, status: str | None = None) -> Path:
    shutil.copytree(ECHO_RUN, dst)
    m = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    m["run_id"] = m["identity"]["run_id"] = run_id
    if model:
        m["identity"]["model_key"] = m["model"]["name"] = model
    if census is not None:
        m["data"]["normalization_census"] = census
    if status:
        m["status"] = status
    (dst / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [json.loads(ln) for ln in (dst / "scores.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if rows_fn:
        rows = rows_fn(rows)
    (dst / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return dst


def _p2_half_right(rows):
    out = []
    for i, r in enumerate(rows):
        r = dict(r, prompt_id="vi-p2-s3-explained-raw", correct=i % 2 == 0)
        r["error_class"] = "correct" if r["correct"] else "wrong"
        out.append(r)
    return out


def _with_arms(rows):
    """nfc rows (all wrong) plus nfd rows right every second item and win1258 rows right every third item."""
    out = []
    for i, r in enumerate(rows):
        out.append(dict(r))
        out.append(dict(r, arm="nfd", correct=i % 2 == 0, error_class="correct" if i % 2 == 0 else "wrong", delta_prompt_tokens=40 + i))
        out.append(dict(r, arm="win1258", correct=i % 3 == 0, error_class="correct" if i % 3 == 0 else "wrong"))
    return out


@pytest.fixture(scope="module")
def echo_runs(tmp_path_factory) -> Path:
    if not ECHO_RUN.exists():
        pytest.skip(f"no scored echo run at {ECHO_RUN}")
    runs = tmp_path_factory.mktemp("runs")
    _copy_run(runs / "E1_main__echo", "E1_main__echo")                                   # 20 T1-V1 p0 rows, all unparseable
    _copy_run(runs / "E1_main__echo__p2", "E1_main__echo__p2", rows_fn=_p2_half_right)   # the p2 split of the same line
    _copy_run(runs / "E3_noilai__echo", "E3_noilai__echo", rows_fn=_with_arms)
    _copy_run(runs / "E3_noilai__echo-norm", "E3_noilai__echo-norm", model="echo-norm", rows_fn=_with_arms,
              census={"status": "censused", "verdict_nfd": "normalizes", "verdict_pc": "normalizes", "engine": "fake"})
    _copy_run(runs / "E1_main__echo-running", "E1_main__echo-running", model="echo-running", status="running")
    return runs


def test_echo_run_produces_table2_cells_table3_rows_and_zero_by_construction_rows(echo_runs, tmp_path):
    out = tmp_path / "out"
    r = _cli("--runs", echo_runs, "--out", out, "--date", DATE, "--n-boot", N_BOOT, "--include-smoke", "--quiet")
    assert r.returncode == 0, r.stderr
    t2 = json.loads((out / "table2_main.json").read_text(encoding="utf-8"))
    assert t2["label"] == "INTERIM" and t2["n"]["items"] == 20 and t2["n"]["base_pairs"] == 20 and t2["n"]["models"] == 2
    assert t2["n"]["runs"] == 4 and t2["n"]["cells"] == 1 and t2["computable"]
    [cell] = t2["rows"]
    assert cell["model"] == "echo" and cell["task"] == "T1" and cell["variant"] == "V1" and cell["arm"] == "nfc"
    assert cell["paraphrases"] == ["p0", "p2"] and cell["n_paraphrases"] == 2                 # the p0p1 + p2 split pooled
    prim = cell["primary"]
    assert prim["paraphrase"] == "p0" and prim["estimate"] == 0.0 and prim["lo"] == 0.0 and prim["hi"] > 0.0
    assert prim["ci_method"] == "wilson_deff" and prim["small_cell"] and prim["n_items"] == 20 and prim["n_base_pairs"] == 20
    assert prim["unparseable_rate"] == 1.0 and prim["deff"] >= 1.0 and not prim["claim_eligible"]
    assert cell["per_paraphrase"]["p2"]["estimate"] == 0.5 and cell["mean_over_paraphrases"] == 0.25
    assert cell["range_over_paraphrases"] == [0.0, 0.5]
    pm = t2["primary_metric_per_model"]["echo"]
    assert pm["acc_mean"] == 0.25 and pm["lo"] <= 0.25 <= pm["hi"] and pm["n_paraphrases"] == 2 and "fewer than three" in pm["label"]
    tex = (out / "table2_main.tex").read_text(encoding="utf-8")
    assert "% INTERIM (n = 20 items, 20 base pairs, 2 models)" in tex and "echo & T1 & V1 & 0.0 [0.0," in tex
    # Table 3: Holm within the model row over the realized family (2 cells of 14), the normalizing model 0 by construction
    t3 = json.loads((out / "table3_interventions.json").read_text(encoding="utf-8"))
    rows = {(x["model"], x["task"], x["arm"]): x for x in t3["rows"]}
    assert set(rows) == {("echo", "T1", "nfd"), ("echo", "T1", "win1258")}
    nfd = rows[("echo", "T1", "nfd")]
    assert nfd["effect"] == pytest.approx(0.5) and nfd["family_size"] == 2 and nfd["family_defined"] == constants.HOLM_FAMILY_TABLE3_CELLS
    assert nfd["p_holm"] >= nfd["p_primary"] and nfd["p_source"] in ("bootstrap", "paired_t") and nfd["n"] == 20 and nfd["n_base_pairs"] == 20
    assert t3["holm_cross_check_multipletests_equals_tests_holm"] is True and t3["family"] == [list(c) for c in constants.HOLM_FAMILY_TABLE3]
    zero = {(z["model"], z["task"], z["arm"]) for z in t3["zero_by_construction"]}
    assert zero == {("echo-norm", t, a) for t in ("T1", "T3", "XCOPA") for a in ("nfd", "win1258")}
    assert t3["normalizes"] == {"echo": False, "echo-norm": True}
    t3tex = (out / "table3_interventions.tex").read_text(encoding="utf-8")
    assert "\\label{tab:arms}" in t3tex and "% 0 by construction: echo-norm nfd T1" in t3tex and "echo & T1 & +50.0" in t3tex
    mc = json.loads((out / "table3_mcnemar_appendix.json").read_text(encoding="utf-8"))
    assert all("ignores clustering" in x["label"] for x in mc["rows"]) and len(mc["rows"]) == 2
    # census and runs tables, the dose-response data, and the report
    census = {x["model"]: x for x in json.loads((out / "table_census.json").read_text(encoding="utf-8"))["rows"]}
    assert census["echo-norm"]["verdict_nfd"] == "normalizes" and census["echo-norm"]["zero_by_construction_arms"] == ["nfd", "win1258"]
    assert census["echo"]["verdict_nfd"] is None
    runs = json.loads((out / "table_runs.json").read_text(encoding="utf-8"))
    assert {x["run_id"] for x in runs["rows"]} == {"E1_main__echo", "E1_main__echo__p2", "E3_noilai__echo", "E3_noilai__echo-norm"}
    assert all(x["smoke"] for x in runs["rows"]) and {x["tag"] for x in runs["rows"]} == {"", "p2"}
    dt = json.loads((out / "fig_deltatok.json").read_text(encoding="utf-8"))
    assert dt["computable"] and {x["arm"] for x in dt["rows"]} == {"nfd", "win1258"}
    assert all(x["total_effect"]["estimate"] == pytest.approx(0.5) for x in dt["rows"] if x["arm"] == "nfd" and x["model"] == "echo")
    summ = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summ["analyses"]["Table 2 (E1 estimation per done model)"]["computable"] is True
    assert summ["analyses"]["H3 pooled crossover (E3), PREREG 8.5"]["computable"] is False       # V3 rows absent
    skipped = {Path(s["run_dir"]).name: s["reason"] for s in summ["skipped"]}
    assert set(skipped) == {"E1_main__echo-running"} and "running" in skipped["E1_main__echo-running"]
    assert any("smoke" in b for b in summ["final_blockers"])
    assert len([i for i in summ["inputs"] if i["kind"] == "scores"]) == 4 and all(len(i["sha256"]) == 64 for i in summ["inputs"])
    report = (out / "analysis_report.md").read_text(encoding="utf-8")
    assert "| echo | T1 | V1 | 0.0 [0.0," in report and "n = 20 items, 20 base pairs, 2 models" in report
    # FINAL is refused on this material (smoke runs, no ledger, gates unrecorded)
    r = _cli("--runs", echo_runs, "--out", tmp_path / "final", "--date", DATE, "--n-boot", N_BOOT, "--include-smoke", "--label", "FINAL")
    assert r.returncode == 3 and "FINAL refused" in r.stderr and "smoke" in r.stderr and not (tmp_path / "final").exists()
    # without --include-smoke the smoke manifests are skipped and nothing is analysed
    r = _cli("--runs", echo_runs, "--out", tmp_path / "nosmoke", "--date", DATE, "--n-boot", N_BOOT, "--quiet")
    assert r.returncode == 0, r.stderr
    summ = json.loads((tmp_path / "nosmoke" / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summ["n"]["runs"] == 0 and sum("smoke" in s["reason"] for s in summ["skipped"]) == 4


def _write_ledger(path: Path, rows: list[dict]) -> None:
    cols = ["unit_id", "experiment", "run_id", "model", "arm", "split", "item_key", "kind", "tag", "status", "hash_verified", "result_file"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def test_ledger_mode_analyses_done_hash_verified_units_only_and_final_is_refused_while_units_are_missing(echo_runs, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(echo_runs / "E1_main__echo", runs / "E1_main__echo")
    shutil.copytree(echo_runs / "E3_noilai__echo", runs / "E3_noilai__echo")
    shutil.copytree(echo_runs / "E1_main__echo__p2", runs / "tpu_main__echo-tpu__p2")       # the alias: pooled under E1_main
    m = json.loads((runs / "tpu_main__echo-tpu__p2" / "manifest.json").read_text(encoding="utf-8"))
    m["run_id"] = m["identity"]["run_id"] = "tpu_main__echo-tpu__p2"
    m["identity"]["model_key"] = m["model"]["name"] = "echo-tpu"
    (runs / "tpu_main__echo-tpu__p2" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    ledger = tmp_path / "ledger.csv"
    _write_ledger(ledger, [
        {"unit_id": "E1_main__echo__nfc", "run_id": "E1_main", "model": "echo", "arm": "nfc", "split": "test", "item_key": "noilai_main",
         "kind": "confirmatory", "status": "done", "hash_verified": "yes"},
        {"unit_id": "E3_noilai__echo__nfc", "run_id": "E3_noilai", "model": "echo", "arm": "nfc", "split": "test", "item_key": "noilai_main",
         "kind": "confirmatory", "status": "done", "hash_verified": "no"},                    # DD 4.6: skipped
        {"unit_id": "E3_noilai__echo__nfd", "run_id": "E3_noilai", "model": "echo", "arm": "nfd", "split": "test", "item_key": "noilai_main",
         "kind": "confirmatory", "status": "planned", "hash_verified": "no"},
        {"unit_id": "tpu_main__echo-tpu__nfc__p2", "run_id": "tpu_main", "model": "echo-tpu", "arm": "nfc", "tag": "p2", "split": "test",
         "item_key": "noilai_main", "kind": "confirmatory", "status": "done", "hash_verified": "n/a"},
        {"unit_id": "E3_xcopa__echo__nfc", "run_id": "E3_xcopa", "model": "echo", "arm": "nfc", "split": "xcopa", "item_key": "xcopa_vi_test",
         "kind": "confirmatory", "status": "cut", "hash_verified": "no"},                     # non-live: not required for FINAL
        {"unit_id": "smoke_20__echo__nfc", "run_id": "smoke_20", "model": "echo", "arm": "nfc", "split": "dev", "item_key": "noilai_dev",
         "kind": "prerequisite", "status": "done", "hash_verified": "yes"},                   # never a table row
    ])
    out = tmp_path / "out"
    r = _cli("--ledger", ledger, "--runs", runs, "--out", out, "--date", DATE, "--n-boot", N_BOOT, "--quiet")
    assert r.returncode == 0, r.stderr
    summ = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    by_run = {x["run_id"]: x for x in summ["runs"]}
    assert set(by_run) == {"E1_main__echo", "tpu_main__echo-tpu__p2"}
    assert by_run["tpu_main__echo-tpu__p2"]["line"] == "E1_main" and by_run["tpu_main__echo-tpu__p2"]["raw_line"] == "tpu_main"
    assert by_run["tpu_main__echo-tpu__p2"]["tag"] == "p2" and by_run["tpu_main__echo-tpu__p2"]["hash_verified"] == "n/a"
    assert by_run["E1_main__echo"]["unit_ids"] == ["E1_main__echo__nfc"] and by_run["E1_main__echo"]["hash_verified"] == "yes"
    skipped = {s["unit_id"]: s["reason"] for s in summ["skipped"]}
    assert "E3_noilai__echo__nfc" in skipped and "DD 4.6" in skipped["E3_noilai__echo__nfc"]
    assert "prerequisite" in skipped["smoke_20__echo__nfc"]
    assert any(i["kind"] == "ledger" for i in summ["inputs"])
    t2 = json.loads((out / "table2_main.json").read_text(encoding="utf-8"))
    assert {(x["model"], x["paraphrases"][0]) for x in t2["rows"]} == {("echo", "p0"), ("echo-tpu", "p2")}
    tpu = next(x for x in t2["rows"] if x["model"] == "echo-tpu")
    assert tpu["primary"] is None and "p0 absent" in tpu["note"] and tpu["per_paraphrase"]["p2"]["estimate"] == 0.5
    assert not (out / "table3_interventions.json").exists() or not json.loads((out / "table3_interventions.json").read_text(encoding="utf-8"))["rows"]
    # FINAL: refused, naming the missing live unit and the unrecorded gates
    r = _cli("--ledger", ledger, "--runs", runs, "--out", tmp_path / "final", "--date", DATE, "--n-boot", N_BOOT, "--label", "FINAL")
    assert r.returncode == 3 and "FINAL refused" in r.stderr
    assert "2 live Table 2 / Table 3 units are not done" in r.stderr                 # the planned nfd unit and the hash-failed nfc unit
    assert "E3_noilai__echo__nfd" in r.stderr and "E3_noilai__echo__nfc" in r.stderr
    assert "registration" in r.stderr and "scoring_rule_freeze" in r.stderr and not (tmp_path / "final").exists()
    blockers = summ["final_blockers"]
    assert any("not done" in b for b in blockers) and not any("smoke" in b for b in blockers)
    # the function API agrees with the CLI
    cfg = AN.load_config(CONFIG)
    refs = AN.completed_from_ledger(ledger, runs, cfg, [])
    assert sorted(ref.run_dir.name for ref in refs) == ["E1_main__echo", "tpu_main__echo-tpu__p2"]
    assert AN.table_lines(cfg) >= {"E1_main", "tpu_main", "E3_noilai", "E3_xcopa", "E3_c2_enriched", "E3_api_core", "E1_api_core"}


# ------------------------------------------------------------------ API models: the p0-only metric (item 52) and H4 on C2-affected rows
def _frame_rows(model: str, line: str, items: list[dict], arm: str, prompt_id: str, correct_fn, is_api: bool) -> list[dict]:
    rows = []
    for it in items:
        rows.append({**it, "arm": arm, "arm_scope": constants.ARM_SCOPE_PRIMARY, "prompt_id": prompt_id, "correct": bool(correct_fn(it)),
                     "correct_lenient": None, "error_class": "wrong", "pair_item_id": None, "twin_type": None, "model": model,
                     "run_id": f"{line}__{model}", "line": line, "raw_line": line, "tag": "", "unit": "", "paraphrase": AN.paraphrase_of(prompt_id),
                     "condition": AN.condition_of(prompt_id), "census_nfd": None, "census_pc": None, "item_class": "core",
                     "c2_enriched": False, "is_api": is_api, "smoke": False})
    return rows


def _items(n_pairs: int, variants=("V1", "V2")) -> list[dict]:
    return [{"item_id": f"T1-{v}-{bp:04d}", "task": "T1", "variant": v, "base_pair_id": f"bp-{bp:04d}",
             "source": "lexicon" if bp % 2 else "pseudo", "in_core": True, "c2_affected": bp % 3 == 0}
            for bp in range(n_pairs) for v in variants]


def test_api_models_report_p0_only_as_their_metric_and_h4_pairs_the_c2_affected_core_rows_only():
    cfg = AN.load_config(CONFIG)
    cfg = dict(cfg, families=dict(cfg["families"], by_model={"api-x": "FA", "open-a": "FB"}))
    items = _items(60)
    sub = items[:30]                                        # the paraphrase subset: p1 / p2 on 15 of the 60 base pairs
    p0 = "vi-p0-s3-explained-raw"
    rows = _frame_rows("api-x", "E1_api_core", items, "nfc", p0, lambda it: it["base_pair_id"] < "bp-0040", True)
    for pid in ("vi-p1-s3-explained-raw", "vi-p2-s3-explained-raw"):
        rows += _frame_rows("api-x", "E1_api_paraphrase", sub, "nfc", pid, lambda it: True, True)      # 100 % on the subset
    rows += _frame_rows("api-x", "E3_api_core", items, "placement_new", p0, lambda it: it["base_pair_id"] < "bp-0010", True)
    for pid in (p0, "vi-p1-s3-explained-raw", "vi-p2-s3-explained-raw"):
        rows += _frame_rows("open-a", "E1_main", items, "nfc", pid, lambda it: it["base_pair_id"] < "bp-0030", False)
    df = pd.DataFrame(rows)
    ctx = AN.Context(cfg=cfg, df=df, runs=[], unscorable=[], skipped=[], n_boot=N_BOOT, paired_boot=N_BOOT, seed=0, level=0.95, midx={},
                     ledger_rows=None, paths={})
    t2 = AN.table2(ctx)
    pm = t2.extra["primary_metric_per_model"]
    api, open_ = pm["api-x"], pm["open-a"]
    # the API metric is p0 on the whole core (2/3 right), never the mean with the 100 % p1 / p2 cells of the subset
    assert api["paraphrases"] == ["p0"] and api["n_paraphrases"] == 1 and api["acc_mean"] == pytest.approx(40 / 60)
    assert "API model" in api["label"] and "item 52" in api["label"] and api["n_items"] == 120
    assert open_["paraphrases"] == ["p0", "p1", "p2"] and open_["acc_mean"] == pytest.approx(0.5) and "API" not in open_["label"]
    cell = next(r for r in t2.rows if r["model"] == "api-x" and r["variant"] == "V1")
    assert cell["is_api"] and "restricted" in cell["note"] and cell["primary"]["estimate"] == pytest.approx(40 / 60)
    # H4 for the API model: the placement rows pair with the E1_api_core baseline on the C2-affected items only (20 of 60 pairs)
    h4 = AN.table_h4(ctx)
    [row] = h4.rows
    assert row["model"] == "api-x" and row["line"] == "E3_api_core" and row["n"] == 40 and row["n_base_pairs"] == 20
    assert h4.extra["c2_affected_only"] is True and h4.extra["expected_direction"] == (-1 if constants.BASELINE_PLACEMENT == "old" else 1)
    assert row["effect"] == pytest.approx((4 - 14) / 20)            # affected pairs 0,3,...,57: 14 right under nfc, 4 under placement_new
    assert h4.extra["k_of_n_models"]["n"] == 1 and h4.extra["pooled"]["computable"] and not h4.extra["pooled"]["pooled"]


# ------------------------------------------------------------------ H1 / H3b / H6 on synthetic runs with an item audit and item files
def _syllables_by_tone() -> dict[int, list[str]]:
    """Real, strictly parseable Hunspell syllables by tone (the H6 covariates need real syllables and the Viet74K frequencies)."""
    if not (ROOT / "data" / "external" / "vi-DauMoi.dic").exists() or not (ROOT / "data" / "external" / "Viet74K.txt").exists():
        pytest.skip("third-party resources not fetched (python scripts/fetch_resources.py)")
    from noilai.vi import lexicon as L
    from noilai.vi.syllable import try_parse
    by: dict[int, list[str]] = {t: [] for t in range(6)}
    for w in L.load_hunspell_syllables():
        p = try_parse(w, strict=True)
        if p is not None and p.syllable.onset not in ("", "p", "c") and not p.syllable.glide and 2 <= len(w) <= 5:
            by[p.syllable.tone].append(w)
    return by


def _manifest(run_id: str, model: str, item_path: Path, census: dict | None = None) -> dict:
    return {"run_id": run_id, "status": "finished", "n_thinking_chars_total": 0,
            "identity": {"run_id": run_id, "model_key": model, "model_id": None, "revision": None},
            "model": {"name": model, "provider": None, "hardware_entry": "t4", "config_entry": {"name": model, "backend": "vllm"}},
            "options": {"notes": {"smoke": False}, "arms": ["nfc", "nfd"]},
            "item_file": {"path": str(item_path), "sha256": "0" * 64, "kind": "noilai"},
            "data": {"item_file": str(item_path), "item_file_sha256": "0" * 64,
                     "normalization_census": census or {"status": "not_censused", "classification": None}}}


def _write(root: Path, run_id: str, model: str, rows: list[dict], item_path: Path, census: dict | None = None) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(_manifest(run_id, model, item_path, census)), encoding="utf-8")
    (d / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _row(item: dict, arm: str, correct: bool, **extra) -> dict:
    r = {k: v for k, v in item.items() if k not in ("gold", "input", "input_syllables")}
    r.update({"gold": item["gold"], "arm": arm, "arm_scope": "whole_prompt", "prompt_id": "vi-p0-s3-explained-raw", "correct": bool(correct),
              "correct_lenient": bool(correct), "error_class": "correct" if correct else "wrong", "answer": "x"})
    r.update(extra)
    return r


@pytest.fixture(scope="module")
def matching_runs(tmp_path_factory) -> dict:
    """One abstract model `alpha`: E1_main T1 rows (4 variants x 300 base pairs) whose correctness follows the planted E2
    structure (fewer tokens per syllable and aligned boundaries help), E3_noilai V3 / V1 rows under nfc and nfd where NFD helps the
    tone-isolated V3 items by +0.35 and nothing else, E1_attested rows on 60 exact attested items at 85 % against a 50 % pool,
    the item files (input / gold text, parsed input syllables) and the item audit of `alpha`'s tokenizer."""
    from noilai.gen.generate import syl_dict
    from noilai.vi import lexicon as L
    from noilai.vi import unicode as U
    from noilai.vi.syllable import try_parse
    rng = np.random.default_rng(5)
    by = _syllables_by_tone()
    freqs = L.syllable_frequencies()
    root = tmp_path_factory.mktemp("mruns")
    items_dir, audit_dir = root / "items", root / "audit"
    items_dir.mkdir()
    audit_dir.mkdir()
    tone_pairs = [(0, 1), (1, 2), (2, 0), (0, 0)]
    n_pairs = 300
    main_items, main_rows, e3_rows, audit_rows = [], [], [], []
    for bp in range(n_pairs):
        ta, tb = tone_pairs[bp % 4]
        a, b = str(rng.choice(by[ta])), str(rng.choice(by[tb]))
        inp, out = f"{a} {b}", f"{b} {a}"
        in_freq = [int(freqs.get(a, 0)), int(freqs.get(b, 0))]
        tone_pair = f"{U.TONE_NAMES[ta]}-{U.TONE_NAMES[tb]}"
        iso = bp % 2 == 0                                                   # the NFD tokenization isolates a tone mark
        tps_nfc = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
        align = float(rng.random()) if tps_nfc > 1 else None
        for v in ("V1", "V2", "V3", "V4"):
            iid = f"T1-{v}-{bp:06d}"
            item = {"item_id": iid, "task": "T1", "variant": v, "input": inp, "gold": [out], "base_pair_id": f"bp-{bp:05d}",
                    "source": "lexicon" if bp % 3 else "pseudo", "in_core": False, "split": "test", "input_lexical": True,
                    "output_lexical": True, "attested_overlap": False, "spelling_triggers": [], "tone_pair": tone_pair,
                    "input_freq": in_freq, "output_freq": list(reversed(in_freq))}
            main_items.append(item)
            logit = 0.8 - 1.2 * (tps_nfc - 1.75) + 1.0 * (align or 0.0)      # the planted E2 structure (H1)
            main_rows.append(_row(item, "nfc", rng.random() < 1 / (1 + np.exp(-logit))))
            for enc, total in (("nfc", int(tps_nfc * 2)), ("nfd", int(tps_nfc * 2) + 2)):
                audit_rows.append({"item_id": iid, "tokenizer": "alpha", "encoding": enc, "delta_tokens_vs_nfc": 2 if enc == "nfd" else 0,
                                   "input": {"tokens_mean": total / 2, "tokens_total": total, "align_among_split": align if total > 2 else None,
                                             "n_split": 2 if total > 2 else 0, "n_misaligned_split": 1 if (total > 2 and (align or 0) < 1) else 0,
                                             "tone_isolated": iso and enc == "nfd", "byte_fallback": False, "any_single_token": total == 2,
                                             "all_single_token": total == 2}})
            if v in ("V3", "V1"):
                base = rng.random() < 0.4
                gain = 0.35 if (v == "V3" and iso) else 0.0
                e3_rows.append(_row(item, "nfc", base))
                e3_rows.append(_row(item, "nfd", rng.random() < 0.4 + gain, delta_prompt_tokens=40))
    (items_dir / "noilai_main.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in main_items), encoding="utf-8")
    (audit_dir / "items_alpha__noilai_main.jsonl").write_text("".join(json.dumps(r) + "\n" for r in audit_rows), encoding="utf-8")
    att_items, att_rows = [], []
    for i in range(60):
        ta, tb = tone_pairs[i % 4]
        a, b = str(rng.choice(by[ta])), str(rng.choice(by[tb]))
        inp, out = f"{a} {b}", f"{b} {a}"
        parsed = [try_parse(w, strict=False) for w in (a, b)]
        item = {"item_id": f"ATT-{i + 1:04d}", "task": "attested", "variant": "V1", "input": inp, "gold": [out], "rule_output": out,
                "input_syllables": [syl_dict(p.syllable, w) for p, w in zip(parsed, (a, b))], "exact": True, "eligible_h6": True, "n_syllables": 2,
                "verified_by": "A" if i < 40 else None, "source": "collection one" if i % 2 else "collection two", "vulgar": False,
                "base_pair_id": f"att-{a}_{b}", "split": "test", "in_core": False}
        att_items.append(item)
        att_rows.append(_row(item, "nfc", rng.random() < 0.85))
    (items_dir / "attested.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in att_items), encoding="utf-8")
    census = {"status": "censused", "source": "synthetic", "classification": "passes_through", "legacy_census": False,
              "verdict_nfd": "passes_through", "verdict_pc": "passes_through", "engine": "vllm"}
    _write(root, "E1_main__alpha", "alpha", main_rows, items_dir / "noilai_main.jsonl", census)
    _write(root, "E3_noilai__alpha", "alpha", e3_rows, items_dir / "noilai_main.jsonl", census)
    _write(root, "E1_attested__alpha", "alpha", att_rows, items_dir / "attested.jsonl", census)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["lines"]["h3b"]["models"] = ["alpha"]
    # the pre-registered H6 parameters (3-5 controls, 0.2-SD caliper) stay in configs/analysis.yaml; the synthetic pool has 75
    # candidates per exact stratum, so the mechanics are exercised with a looser caliper and one control at least
    assert cfg["lines"]["attested"]["k_min"] == 3 and cfg["lines"]["attested"]["k_max"] == 5 and cfg["lines"]["attested"]["caliper_sd"] == 0.2
    cfg["lines"]["attested"].update(k_min=1, k_max=5, caliper_sd=0.6)
    cfg_path = root / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"root": root, "audit": audit_dir, "config": cfg_path}


def test_h1_join_h3b_did_and_h6_matching_run_end_to_end_on_synthetic_runs(matching_runs, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([PY, "-m", "noilai.stats.analyze", "--config", str(matching_runs["config"]), "--runs", str(matching_runs["root"]), "--out", str(out),
                        "--date", DATE, "--n-boot", "100", "--audit-dir", str(matching_runs["audit"]), "--quiet"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    # H1: the join, the floor and the nested test ran; the planted alignment effect is found; the family map does not know `alpha`,
    # so the identifiability floor (>= 4 families) and the minimum viable panel are not met and H1 stays descriptive
    h1 = json.loads((out / "table_h1_e2.json").read_text(encoding="utf-8"))
    assert h1["computable"], h1["reason"]
    [row] = h1["rows"]
    assert row["model"] == "alpha" and row["in_fit"] and row["clears_floor"] and row["n_misaligned_split"] >= 300 and row["n_joined"] == 1200
    assert row["slope_tps"] < 0 and row["slope_align"] > 0 and row["audit"].endswith("items_alpha__noilai_main.jsonl")
    assert h1["nested"]["h1_supported"] and h1["nested"]["p"] < 0.05 and h1["nested"]["delta_aic"] >= 2 and h1["n_fit_rows"] == 1200
    assert h1["bootstrap"]["n_ok"] >= 80 and h1["bootstrap"]["ci_mean_slope_tps"][1] < 0 and "primary" in h1["bootstrap"]["label"]
    assert h1["decision"]["h1_supported_by_test"] and not h1["decision"]["identified"] and h1["decision"]["h1"].startswith("descriptive")
    assert "items_alpha__noilai_main.jsonl" in h1["audit_files"]
    # H3b: the four contrasts' Holm family holds the configured model's V3 and V1; V3 recovers the planted +0.35, V1 includes 0
    h3b = json.loads((out / "table_h3b.json").read_text(encoding="utf-8"))
    assert h3b["computable"], h3b["reason"]
    by_v = {x["variant"]: x for x in h3b["rows"] if x.get("computable")}
    v3, v1 = by_v["V3"], by_v["V1"]
    assert 0.15 < v3["estimate"] < 0.6 and v3["lo"] > 0 and v3["p_holm"] < 0.05 and v3["significant"] and v3["holm_family_size"] == 2
    assert v1["lo"] <= 0 <= v1["hi"] and v1["n_items"] == 300 and v1["underpowered"] and v3["configured"]
    assert v3["matching"]["k_max"] == 1 and v3["matching"]["n_matched"] >= 60 and v3["matching"]["exact"] == ["source"]
    assert h3b["decisions"]["alpha"]["supported"] and h3b["decisions"]["alpha"]["underpowered"]
    tex = (out / "table_h3b.tex").read_text(encoding="utf-8")
    assert "alpha & V3 & +" in tex and "underpowered" in tex
    # H6: exact attested items matched 3-5 : 1 to generated items with lexical output; the floor is not met (40 verified rows, no
    # expansion unit) so the matched contrast is descriptive (no p, no pool) and still recovers the planted +0.35
    h6 = json.loads((out / "table_h6.json").read_text(encoding="utf-8"))
    assert h6["computable"], h6["reason"]
    [row] = h6["rows"]
    m = row["matching"]
    assert m["n_treated"] == 60 and m["n_matched"] >= 40 and 1 <= m["mean_k"] <= 5 and m["exact"] == list(AN.MT.H6_EXACT)
    assert m["k_min"] == 1 and m["k_max"] == 5 and m["caliper_sd"] == 0.6 and m["nearest"] == list(AN.MT.H6_NEAREST)
    assert m["max_abs_smd_after"] is not None and m["smd_after"]["log_freq_in_1"] == m["smd_after"]["log_freq_in_1"]
    assert 0.15 < row["effect"] < 0.55 and row["ci_lo"] < row["effect"] < row["ci_hi"] and "p_holm" not in row
    assert not h6["floor_met"] and h6["floor_counts"] == {"n_exact": 60, "n_eligible_h6": 60, "n_verified_exact": 40, "n_collections": 2}
    assert h6["test"] == "not run" and h6["pooled"] is None and row["status"].startswith("descriptive")
    assert h6["descriptive_accuracy"][0]["estimate"] > 0.7
    tex = (out / "table_h6.tex").read_text(encoding="utf-8")
    assert "alpha & ? & +" in tex and "descriptive" in tex
    summ = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    for name in ("H1 nested LR test (E2), PREREG 8.4", "H3b tone-isolation DiD", "H6 attested vs matched (needs the 100-row floor)"):
        assert summ["analyses"][name]["computable"] is True, name
    kinds = {i["kind"] for i in summ["inputs"]}
    assert {"items_audit", "items"} <= kinds
    # --check sees the outputs as current
    r = subprocess.run([PY, "-m", "noilai.stats.analyze", "--config", str(matching_runs["config"]), "--runs", str(matching_runs["root"]), "--out", str(out),
                        "--date", DATE, "--n-boot", "100", "--audit-dir", str(matching_runs["audit"]), "--check"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_without_an_item_audit_or_item_files_the_three_analyses_say_why(echo_runs, tmp_path):
    out = tmp_path / "out"
    r = _cli("--runs", echo_runs, "--out", out, "--date", DATE, "--n-boot", N_BOOT, "--include-smoke", "--quiet", "--audit-dir", tmp_path / "no_audit")
    assert r.returncode == 0, r.stderr
    h1 = json.loads((out / "table_h1_e2.json").read_text(encoding="utf-8"))
    assert not h1["computable"] and "no item audit for echo" in h1["reason"] and h1["rows"][0]["in_fit"] is False
    h3b = json.loads((out / "table_h3b.json").read_text(encoding="utf-8"))
    status = [str(x.get("status")) for x in h3b["rows"]]
    assert not h3b["computable"] and any("not informative" in x or "no item audit" in x for x in status), status
    h6 = json.loads((out / "table_h6.json").read_text(encoding="utf-8"))
    assert not h6["computable"] and "no completed E1_attested rows" in h6["reason"]
