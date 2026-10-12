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
