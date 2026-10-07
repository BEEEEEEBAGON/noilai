"""experiments/ledger.csv and scripts/progress.py: units, cost split, ingest and the meter."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
L = importlib.import_module("ledger")        # scripts/ledger.py (imported after the path insert)
P = importlib.import_module("progress")      # scripts/progress.py

SMOKE_DEV = Path("/tmp/claude-0/-home-user-noilai/7ed5d88b-f91c-57ec-b5a3-07ddcf2f1a1c/scratchpad/smoke_build/v0.3-smoke/noilai_dev.jsonl")


def test_ledger_units_cover_every_plan_line_and_costs_sum_to_the_plan_midpoints():
    rows = L.build_rows()
    plan = L.load_plan(L.PLAN)
    by_run = {}
    for r in rows:
        if r["lane"] != "human":
            by_run.setdefault(r["run_id"], 0.0)
            by_run[r["run_id"]] += float(r["est_raw"])
    for run in plan["runs"]:
        est = run["estimate"]
        mid = (est["low"] + est["high"]) / 2
        if run["id"] == "tpu_main":
            assert "tpu_main" not in by_run           # alias of E1_main; its hours are booked on E1_main's TPU units
            continue
        got = by_run[run["id"]]
        if run["id"] == "E1_main":                    # GPU midpoint + tpu_main's TPU midpoint
            t = next(r for r in plan["runs"] if r["id"] == "tpu_main")["estimate"]
            mid += (t["low"] + t["high"]) / 2
        assert got == pytest.approx(mid, rel=0.02), (run["id"], got, mid)
    ids = {r["unit_id"] for r in rows}
    assert len(ids) == len(rows)
    assert "E1_main__gemma-3-1b-it__nfc" in ids and "E3_noilai__gemma-3-1b-it__nfd" in ids and "E4_probe__gemma-3-4b-it__nfc+nfd" in ids
    assert {r["section"] for r in rows} >= {"Results", "Counterfactuals", "Tone", "Human", "Prerequisite"}


def test_cut_order_puts_prerequisites_first_and_the_dd_cut_first_models_last_within_a_line():
    rows = L.build_rows()
    first = [r for r in rows if r["lane"] != "human"][:5]
    assert all(r["experiment"] == "smoke" for r in first)
    e1 = [r for r in rows if r["run_id"] == "E1_main" and r["arm"] == "nfc"]
    order = [r["model"] for r in sorted(e1, key=lambda r: int(r["cut_rank"]))]
    assert order[-3:] == ["gemma-3-12b-it", "qwen3.5-0.8b", "gemma-4-12b"]     # DESIGN_DECISIONS 7.1 cut order, last
    assert order[0] == "gemma-3-1b-it"
    nfd = {r["model"]: int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_noilai" and r["arm"] == "nfd"}
    assert nfd["qwen3.5-2b"] > nfd["llama-3.1-8b-instruct"]                   # normalizing tokenizer de-prioritized on nfd
    c2 = min(int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_c2_enriched")
    xc = min(int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_xcopa")
    assert c2 < xc


def test_init_refuses_to_overwrite_and_carries_statuses_over_with_force(tmp_path):
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    rows = L.read_ledger(led)
    rows[0]["status"] = "done"
    L.write_ledger(rows, led)
    assert L.main(["--ledger", str(led), "init"]) == 1                          # refuses
    assert L.main(["--ledger", str(led), "init", "--force"]) == 0
    again = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert again[rows[0]["unit_id"]]["status"] == "done"


@pytest.mark.skipif(not SMOKE_DEV.exists(), reason="local smoke build absent")
def test_ingest_marks_a_finished_run_done_and_enforces_the_item_hash_gate(tmp_path):
    runs = tmp_path / "runs"
    out = runs / "smoke_20__gemma-3-1b-it"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_eval.py"), "--items", str(SMOKE_DEV), "--backend", "echo", "--smoke",
                    "--limit", "20", "--tasks", "T1", "T3", "--variants", "V1", "--run-id", "smoke_20__gemma-3-1b-it",
                    "--out-root", str(runs)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "score_run.py"), "--run", str(out), "--items", str(SMOKE_DEV)],
                   check=True, capture_output=True)
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    # the plan records a hash for noilai_dev that this NON-frozen smoke build does not have -> mismatch, exit 1
    rc = L.main(["--ledger", str(led), "--exploratory", str(tmp_path / "none.yaml"), "ingest", "--runs", str(runs), "--human", str(tmp_path / "nohuman"), "--aggregates", str(tmp_path / "agg")])
    row = {r["unit_id"]: r for r in L.read_ledger(led)}["smoke_20__gemma-3-1b-it__nfc"]
    assert rc == 1 and row["hash_verified"] == "no" and "HASH MISMATCH" in row["notes"] and row["status"] == "done"
    assert int(row["n_rows"]) == 20 and row["result_sha256"] and row["scores_sha256"]
    assert (tmp_path / "agg" / "smoke_20__gemma-3-1b-it" / "summary.json").exists()
    assert not any((tmp_path / "agg").rglob("outputs.jsonl"))                 # aggregates only, never item-level rows
    # with the plan's reference hash equal to the file the run read, the gate passes
    plan = yaml.safe_load((ROOT / "configs" / "run_plan.yaml").read_text(encoding="utf-8"))
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    plan["item_files"]["noilai_dev"]["sha256"] = m["data"]["item_file_sha256"]
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    rc = L.main(["--ledger", str(led), "--plan", str(plan_path), "--exploratory", str(tmp_path / "none.yaml"), "ingest", "--runs", str(runs), "--human", str(tmp_path / "nohuman"),
                 "--aggregates", str(tmp_path / "agg")])
    row = {r["unit_id"]: r for r in L.read_ledger(led)}["smoke_20__gemma-3-1b-it__nfc"]
    assert rc == 0 and row["hash_verified"] == "yes"
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    assert d["per_experiment"]["smoke"][0] > 0 and d["overall"][0] > 0
    assert d["frontier"]["primary_not_at_band"] == []


def test_progress_meter_counts_only_done_and_hash_verified_units(tmp_path):
    led = tmp_path / "ledger.csv"
    L.main(["--ledger", str(led), "init"])
    rows = L.read_ledger(led)
    by = {r["unit_id"]: r for r in rows}
    by["E1_main__gemma-3-1b-it__nfc"].update(status="done", hash_verified="yes")
    by["E1_main__llama-3.1-8b-instruct__nfc"].update(status="done", hash_verified="no")   # must NOT count
    by["E1_main__gemma-3-12b-it__nfc"].update(status="cut")                                 # leaves the denominator
    L.write_ledger(rows, led)
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    done, tot = d["per_experiment"]["E1"][1], d["per_experiment"]["E1"][2]
    assert done == pytest.approx(float(by["E1_main__gemma-3-1b-it__nfc"]["est_cost"]))
    full = sum(float(r["est_cost"]) for r in rows if r["experiment"] == "E1")
    assert tot == pytest.approx(full - float(by["E1_main__gemma-3-12b-it__nfc"]["est_cost"]))
    assert d["analyses"]["Table 2 (E1 estimation per done model)"]["computable"] is True
    assert d["analyses"]["H1 nested LR test (E2), PREREG 8.4"]["computable"] is False
    block = P.render(d)
    assert P.BEGIN in block and "Explicit-input gap" in block
    status = tmp_path / "STATUS.md"
    status.write_text("# STATUS\n\nhand text\n\n" + P.BEGIN + "\nold\n" + P.END + "\n\n## after\n", encoding="utf-8")
    P.write_status(block, status)
    t = status.read_text(encoding="utf-8")
    assert "hand text" in t and "## after" in t and "old" not in t and block in t
