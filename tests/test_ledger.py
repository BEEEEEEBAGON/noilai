"""experiments/ledger.csv and scripts/progress.py: the chunk units of configs/compute_chunks.yaml, their cost split,
ingest (tagged run directories, unmatched directories, the human reports, re-pricing) and the per-chunk meter.
The echo runs use the local NON-frozen smoke build when present, else a small seeded build in a temporary directory."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
L = importlib.import_module("ledger")             # scripts/ledger.py (imported after the path insert)
P = importlib.import_module("progress")           # scripts/progress.py
PC = importlib.import_module("plan_chunks")       # scripts/plan_chunks.py: the chunk generator
KRP = importlib.import_module("kaggle_run_plan")
CL = importlib.import_module("compute_log")

SMOKE_DEV = Path("/tmp/claude-0/-home-user-noilai/7ed5d88b-f91c-57ec-b5a3-07ddcf2f1a1c/scratchpad/smoke_build/v0.3-smoke/noilai_dev.jsonl")
PY = sys.executable
CHUNK_COLUMNS = ("chunk_id", "chunk_order", "tier", "queue", "window", "tag", "chunk_hours_low", "chunk_hours_high", "notebook")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return L.build_rows()


@pytest.fixture(scope="module")
def by_unit(rows) -> dict[str, dict]:
    return {r["unit_id"]: r for r in rows}


@pytest.fixture(scope="module")
def spec() -> dict:
    return PC.build_chunks()


@pytest.fixture(scope="module")
def plan() -> dict:
    return L.load_plan(L.PLAN)


@pytest.fixture(scope="module")
def item_file(tmp_path_factory) -> Path:
    """The local NON-frozen smoke build's dev file when present, else a small seeded build in a temporary directory
    (never under data/release/)."""
    if SMOKE_DEV.exists():
        return SMOKE_DEV
    out = tmp_path_factory.mktemp("rel")
    subprocess.run([PY, str(ROOT / "scripts" / "build_data.py"), "--out", str(out), "--seed", "3", "--n-lexicon", "120",
                    "--n-pseudo", "60", "--per-cell-t1", "40", "--per-cell-t2", "20", "--per-cell-t3", "15", "--core-per-cell", "8"],
                   check=True, capture_output=True, cwd=ROOT)
    return out / "noilai_dev.jsonl"


def _echo_run(runs: Path, run_id: str, item_file: Path, check: bool = False) -> Path:
    out = runs / run_id
    subprocess.run([PY, str(ROOT / "scripts" / "run_eval.py"), "--items", str(item_file), "--backend", "echo", "--smoke",
                    "--limit", "20", "--tasks", "T1", "T3", "--variants", "V1", "--run-id", run_id, "--out-root", str(runs)],
                   check=True, capture_output=True, cwd=ROOT)
    subprocess.run([PY, str(ROOT / "scripts" / "score_run.py"), "--run", str(out), "--items", str(item_file)],
                   check=True, capture_output=True, cwd=ROOT)
    if check:                                                 # stats.json + results_hashes.json as the chunk notebooks write them
        subprocess.run([PY, str(ROOT / "scripts" / "check_run.py"), "--run", str(out), "--items", str(item_file), "--n-boot", "20"],
                       check=True, capture_output=True, cwd=ROOT)
    return out


def _ledger_ingest(led: Path, runs: Path, tmp_path: Path, capsys, plan_path: Path | None = None) -> tuple[int, dict]:
    argv = ["--ledger", str(led), "--exploratory", str(tmp_path / "none.yaml")]
    if plan_path is not None:
        argv += ["--plan", str(plan_path)]
    argv += ["ingest", "--runs", str(runs), "--human", str(tmp_path / "nohuman"), "--validation", str(tmp_path / "novalidation"),
             "--aggregates", str(tmp_path / "agg")]
    capsys.readouterr()
    rc = L.main(argv)
    out = capsys.readouterr().out.strip().splitlines()
    return rc, json.loads(out[-1])


# ------------------------------------------------------------------ the units
def test_every_chunk_job_model_arm_has_exactly_one_unit_and_the_columns_carry_the_chunk(rows, by_unit, spec, plan):
    assert L.COLUMNS.index("lane") + 1 == L.COLUMNS.index("chunk_id")            # the chunk columns follow `lane`
    assert all(c in L.COLUMNS for c in (*CHUNK_COLUMNS, "results_sha256"))
    assert len(spec["chunks"]) == 31
    counts: Counter = Counter()
    for ch in spec["chunks"]:
        for job in ch["jobs"]:
            run = KRP.find_run(plan, job["run"])
            narrowed = KRP.narrow_run(run, job.get("overrides"), job.get("tag"))
            for model in job["models"]:
                for arm in narrowed["arms"]:
                    uid = L.unit_id_of(run["id"], model, arm, job.get("tag") or "")
                    counts[uid] += 1
                    r = by_unit[uid]
                    if counts[uid] == 1:
                        assert r["chunk_id"] == ch["id"] and int(r["chunk_order"]) == ch["order"] and int(r["tier"]) == ch["tier"]
                        assert r["queue"] == ch["queue"] and r["window"] == ch["window"] and r["notebook"] == ch["notebook"]
                        assert r["tag"] == (job.get("tag") or "") and r["paraphrases"] == "|".join(narrowed.get("paraphrases") or [])
                        assert r["lane"] == L.LANE_OF_QUEUE[ch["queue"]] and r["est_unit"] == L.UNIT_OF_QUEUE[ch["queue"]]
    assert [u for u, c in counts.items() if c > 1] == ["smoke_20__gemma-3-1b-it__nfc__cpu"]   # c01's smoke, re-entered by c02
    assert by_unit["smoke_20__gemma-3-1b-it__nfc__cpu"]["chunk_id"] == "c01_cpu_t0"
    assert "re-entered by c02_cpu_t0" in by_unit["smoke_20__gemma-3-1b-it__nfc__cpu"]["notes"]
    assert sum(1 for r in rows if r["chunk_id"]) == len(counts) == 397
    assert len({r["chunk_id"] for r in rows if r["chunk_id"]}) == 31
    ids = [r["unit_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for uid in ("E1_main__gemma-3-1b-it__nfc", "tpu_main__gemma-3-12b-it__nfc__p0p1", "smoke_20__gemma-3-1b-it__nfc__cpu",
                "E3_noilai__gemma-3-1b-it__nfd", "E4_probe__gemma-3-4b-it__nfc+nfd", "E1_api_core__gpt-oss-20b__nfc"):
        assert uid in by_unit, uid
    assert {r["section"] for r in rows} >= {"Results", "Counterfactuals", "Tone", "Human", "Prerequisite", "Exploratory"}
    assert set(L.NON_LIVE_STATUSES) <= set(L.STATUSES) and {r["status"] for r in rows} <= set(L.STATUSES)
    # a paraphrase split is two units whose paraphrases and hours add up to the line's share (tpu_main: [4, 8] over 3 models)
    a, b = by_unit["tpu_main__gemma-3-12b-it__nfc__p0p1"], by_unit["tpu_main__gemma-3-12b-it__nfc__p2"]
    assert a["paraphrases"] == "p0|p1" and b["paraphrases"] == "p2" and int(a["tier"]) == 4 and int(b["tier"]) == 5
    assert float(a["chunk_hours_low"]) + float(b["chunk_hours_low"]) == pytest.approx(4 / 3, abs=1e-3)
    assert float(a["chunk_hours_high"]) + float(b["chunk_hours_high"]) == pytest.approx(8 / 3, abs=1e-3)
    assert a["run_id"] == "tpu_main" and a["experiment"] == "E1" and "alias of E1_main" in a["notes"]
    assert "E1_main__gemma-3-12b-it__nfc" not in by_unit                             # the TPU trio runs as tpu_main
    # the CPU pilot units are apart from the T4 ones, which are the alternative route (memo N11)
    cpu, t4 = by_unit["pilot_t1_200__qwen3.5-2b__nfc__cpu"], by_unit["pilot_t1_200__qwen3.5-2b__nfc"]
    assert cpu["status"] == "planned" and cpu["queue"] == "cpu" and cpu["lane"] == "kaggle_cpu" and cpu["est_unit"] == "cpu_hours"
    assert t4["status"] == "alternative" and t4["chunk_id"] == "c05_gpu_t0" and "memo N11" in t4["notes"]
    smoke_t4 = by_unit["smoke_20__qwen3.5-2b__nfc"]
    assert smoke_t4["status"] == "planned" and smoke_t4["chunk_id"] == "c06_gpu_t0"   # the CPU-vs-T4 smoke comparison: both kept
    # lines no chunk runs
    api = by_unit["E1_api_core__gpt-oss-20b__nfc"]
    assert api["chunk_id"] == "" and api["queue"] == "api" and api["lane"] == "api" and api["notebook"] == "notebooks/api_runs.ipynb"
    assert api["est_unit"] == "api_calls" and float(api["est_raw"]) == pytest.approx(6000 / 4) and api["gate"] == "prereg+qu+iy+gemini"
    e4 = by_unit["E4_probe__gemma-3-4b-it__nfc+nfd"]
    assert e4["chunk_id"] == "" and "23 Nov-6 Dec" in e4["window"] and e4["notebook"] == "notebooks/colab_probe_gemma3.ipynb"
    uns = by_unit["bf16_drift_200__phogpt-4b-chat--bf16__nfc"]
    assert uns["status"] == "unscheduled" and "memo N9" in uns["notes"] and uns["lane"] == "modal_l4" and uns["est_unit"] == "l4_hours"
    xp = [r for r in rows if r["run_id"] in ("floor_pilot_dev", "throughput_dev")]
    assert len(xp) == 10 and all(r["status"] == "unscheduled" and r["notes"] == L.EXPLORATORY_NOTE and r["kind"] == "exploratory"
                                 and r["notebook"] == L.EXPLORATORY_NOTEBOOK for r in xp)
    # the human lane, re-sized to the 1 Oct packet
    assert {r["unit_id"] for r in rows if r["experiment"] == "human_validation"} == {"validation__N", "validation__C", "validation__S"}
    assert all(float(r["est_raw"]) == L.VALIDATION_HOURS == 5.4 for r in rows if r["experiment"] == "human_validation")
    forms = [r for r in rows if r["experiment"] == "human_baseline"]
    assert len(forms) == 20 and all(float(r["est_raw"]) == 0.6 for r in forms)
    assert float(by_unit["attested__expansion"]["est_raw"]) == 12.0
    # sorted by chunk order, then cut rank; the non-chunked units after the chunks
    orders = [int(r["chunk_order"]) for r in rows if r["chunk_id"]]
    assert orders == sorted(orders) and rows[0]["chunk_id"] == "c01_cpu_t0"
    assert all(not r["chunk_id"] for r in rows[len(orders):])


def test_unit_hours_sum_to_the_chunk_hours_and_fall_inside_the_plan_line_estimates(rows, spec, plan):
    for ch in spec["chunks"]:
        units = [r for r in rows if r["chunk_id"] == ch["id"]]
        lo = sum(float(r["chunk_hours_low"]) for r in units)
        hi = sum(float(r["chunk_hours_high"]) for r in units)
        if ch["id"] == "c02_cpu_t0":                                   # minus the smoke share booked on c01 (re-entered)
            cpu_est = PC.cpu_estimates()
            lo += cpu_est["smoke_20"]
            hi += cpu_est["smoke_20"]
        assert lo == pytest.approx(ch["hours"]["low"], abs=0.02) and hi == pytest.approx(ch["hours"]["high"], abs=0.02), ch["id"]
        for r in units:
            assert float(r["est_raw"]) == pytest.approx((float(r["chunk_hours_low"]) + float(r["chunk_hours_high"])) / 2, abs=1e-3)
    for run in plan["runs"]:
        est = run["estimate"]
        sub = [r for r in rows if r["run_id"] == run["id"] and r["est_unit"] == est["unit"] and r["status"] != "unscheduled"]
        got = sum(float(r["est_raw"]) for r in sub)
        assert est["low"] - 1e-3 <= got <= est["high"] + 1e-3, (run["id"], got, est)
        if run["id"] in ("tpu_main", "smoke_20_tpu", "reasoning_500", "reasoning_500_tpu", "E3_scope_item", "E3_engine_hf"):
            assert got == pytest.approx((est["low"] + est["high"]) / 2, abs=1e-2)   # a line booked whole in its own unit: the midpoint
    t = L.totals(rows)
    assert t["units"] == len(rows) and t["chunks"] == 31 and t["chunked_units"] == 397
    live = [r for r in rows if r["lane"] != "human" and r["status"] not in L.NON_LIVE_STATUSES]
    assert t["compute_cost_units"] == pytest.approx(sum(float(r["est_cost"]) for r in live), abs=0.01)
    assert t["cpu_hours"] == pytest.approx(sum(float(r["est_raw"]) for r in rows if r["est_unit"] == "cpu_hours"), abs=0.01)
    assert all(float(r["est_cost"]) == 0.0 for r in rows if r["est_unit"] == "cpu_hours")   # quota.yaml: null -> weight 0


def test_the_cpu_weight_follows_quota_yaml(tmp_path):
    assert L.cpu_quota_weight(L.QUOTA) == 0.0 and L.cpu_quota_weight(tmp_path / "absent.yaml") == 0.0
    q = tmp_path / "quota.yaml"
    q.write_text("gpu_hours_per_week: 30\ncpu_sessions_count_against_gpu_quota: true\n", encoding="utf-8")
    assert L.cpu_quota_weight(q) == 1.0
    cpu = [r for r in L.build_rows(quota_path=q) if r["est_unit"] == "cpu_hours"]
    assert cpu and all(float(r["est_cost"]) == float(r["est_raw"]) > 0 for r in cpu)


def test_cut_order_puts_prerequisites_first_and_the_dd_cut_first_models_last_within_a_line(rows):
    compute = sorted([r for r in rows if r["lane"] != "human"], key=lambda r: (int(r["cut_rank"]), r["unit_id"]))
    assert all(r["experiment"] == "smoke" for r in compute[:5])
    e1 = [r for r in rows if r["run_id"] in ("E1_main", "tpu_main") and r["arm"] == "nfc" and r["tag"] != "p2"]
    order = [r["model"] for r in sorted(e1, key=lambda r: int(r["cut_rank"]))]
    assert order[-3:] == ["gemma-3-12b-it", "qwen3.5-0.8b", "gemma-4-12b"]     # DESIGN_DECISIONS 7.1 cut order, last
    assert order[0] == "gemma-3-1b-it" and len(order) == 17
    nfd = {r["model"]: int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_noilai" and r["arm"] == "nfd"}
    assert nfd["qwen3.5-2b"] > nfd["llama-3.1-8b-instruct"]                   # normalizing tokenizer de-prioritized on nfd
    c2 = min(int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_c2_enriched")
    xc = min(int(r["cut_rank"]) for r in rows if r["run_id"] == "E3_xcopa")
    assert c2 < xc
    # the launch order is the chunk order: tier-major, tier 0 = c01-c07, never decreasing
    tiers = [int(r["tier"]) for r in rows if r["chunk_id"]]
    tier0 = {f"c0{i}_{q}_t0" for i, q in zip(range(1, 8), ["cpu", "cpu", "cpu", "cpu", "gpu", "gpu", "tpu"], strict=True)}
    assert tiers == sorted(tiers) and {r["chunk_id"] for r in rows if str(r["tier"]) == "0"} == tier0


def test_init_refuses_to_overwrite_and_carries_observed_statuses_over_with_force(tmp_path, capsys):
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["chunks"] == 31 and summary["units"] == summary["compute_units"] + 24
    rows = L.read_ledger(led)
    by = {r["unit_id"]: r for r in rows}
    by["E1_main__gemma-3-1b-it__nfc"].update(status="done", hash_verified="yes", results_sha256="ab" * 32)
    by["E3_noilai__gemma-3-4b-it__nfc"].update(status="cut", notes="the author cut it")
    by["pilot_t1_200__qwen3.5-2b__nfc"]["status"] = "planned"                 # a plan-derived status (alternative) edited away
    by["floor_pilot_dev__gemma-3-1b-it__nfc"]["status"] = "planned"           # idem (unscheduled)
    L.write_ledger(rows, led)
    assert L.main(["--ledger", str(led), "init"]) == 1                          # refuses
    assert L.main(["--ledger", str(led), "init", "--force"]) == 0
    again = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert again["E1_main__gemma-3-1b-it__nfc"]["status"] == "done" and again["E1_main__gemma-3-1b-it__nfc"]["results_sha256"] == "ab" * 32
    assert again["E3_noilai__gemma-3-4b-it__nfc"]["status"] == "cut" and "the author cut it" in again["E3_noilai__gemma-3-4b-it__nfc"]["notes"]
    assert again["pilot_t1_200__qwen3.5-2b__nfc"]["status"] == "alternative"   # plan-derived statuses are re-derived
    assert again["floor_pilot_dev__gemma-3-1b-it__nfc"]["status"] == "unscheduled"
    assert again["E1_main__gemma-3-1b-it__nfc"]["chunk_id"] == "c08_gpu_t1"


# ------------------------------------------------------------------ ingest
def test_ingest_marks_finished_runs_done_locates_tagged_directories_and_reports_unmatched_ones(tmp_path, item_file, capsys):
    runs = tmp_path / "runs"
    t4 = _echo_run(runs, "smoke_20__gemma-3-1b-it", item_file)                    # the c06 unit (untagged)
    cpu = _echo_run(runs, "smoke_20__gemma-3-1b-it__cpu", item_file, check=True)  # the c01 unit (tag cpu) with check_run's files
    for name in ("nonsense__gemma-3-1b-it", "noseparator"):                      # directories that name no unit
        (runs / name).mkdir()
        m = json.loads((t4 / "manifest.json").read_text(encoding="utf-8"))
        m["run_id"] = m["identity"]["run_id"] = name
        (runs / name / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    # the plan records a hash for noilai_dev that this NON-frozen build does not have -> mismatch, exit 1
    rc, summary = _ledger_ingest(led, runs, tmp_path, capsys)
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    m = json.loads((cpu / "manifest.json").read_text(encoding="utf-8"))
    expected = int(m["item_file"]["n_selected"]) * len(m["options"]["paraphrases"])
    for uid in ("smoke_20__gemma-3-1b-it__nfc", "smoke_20__gemma-3-1b-it__nfc__cpu"):
        row = by[uid]
        assert row["status"] == "done" and row["hash_verified"] == "no" and "HASH MISMATCH" in row["notes"], uid
        assert int(row["n_rows"]) == int(row["n_expected"]) == expected > 0 and row["result_sha256"] and row["scores_sha256"]
    assert rc == 1 and set(summary["hash_mismatch"]) == {"smoke_20__gemma-3-1b-it__nfc", "smoke_20__gemma-3-1b-it__nfc__cpu"}
    hashes = json.loads((cpu / "results_hashes.json").read_text(encoding="utf-8"))
    assert by["smoke_20__gemma-3-1b-it__nfc__cpu"]["results_sha256"] == hashes["results_sha256"] and len(hashes["results_sha256"]) == 64
    assert by["smoke_20__gemma-3-1b-it__nfc"]["results_sha256"] == ""           # no check_run on the T4 copy
    assert summary["touched"] == 3 and summary["done"] == 2            # + attested__expansion, re-read from the seed every time
    unmatched = {u["dir"]: u for u in summary["unmatched"]}
    assert set(unmatched) == {"nonsense__gemma-3-1b-it", "noseparator"}
    assert unmatched["nonsense__gemma-3-1b-it"]["units"] == ["nonsense__gemma-3-1b-it__nfc"] and "ledger" in unmatched["nonsense__gemma-3-1b-it"]["reason"]
    assert unmatched["noseparator"]["units"] == [] and "<run>__<model>" in unmatched["noseparator"]["reason"]
    agg = tmp_path / "agg"
    assert {p.name for p in (agg / "smoke_20__gemma-3-1b-it__cpu").iterdir()} == {"summary.json", "manifest_excerpt.json", "stats.json", "results_hashes.json"}
    assert {p.name for p in (agg / "smoke_20__gemma-3-1b-it").iterdir()} == {"summary.json", "manifest_excerpt.json"}
    assert not any(agg.rglob("outputs.jsonl")) and not any(agg.rglob("scores.jsonl"))   # aggregates only, never item-level rows
    assert not (agg / "nonsense__gemma-3-1b-it").exists()
    # with the plan's reference hash equal to the file the run read, the gate passes
    plan = yaml.safe_load((ROOT / "configs" / "run_plan.yaml").read_text(encoding="utf-8"))
    plan["item_files"]["noilai_dev"]["sha256"] = m["data"]["item_file_sha256"]
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    rc, summary = _ledger_ingest(led, runs, tmp_path, capsys, plan_path)
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert rc == 0 and by["smoke_20__gemma-3-1b-it__nfc__cpu"]["hash_verified"] == "yes" and by["smoke_20__gemma-3-1b-it__nfc"]["hash_verified"] == "yes"
    assert "HASH MISMATCH" not in by["smoke_20__gemma-3-1b-it__nfc__cpu"]["notes"]      # the stale note went with the verification
    assert summary["conflicts"] == []
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    assert d["per_experiment"]["smoke"][0] > 0 and d["overall"][0] > 0
    assert d["frontier"]["primary_not_at_band"] == []
    chunks = {c["chunk_id"]: c for c in d["chunks"]}
    assert chunks["c01_cpu_t0"]["status"] == "done" and chunks["c01_cpu_t0"]["units_done"] == chunks["c01_cpu_t0"]["units_total"] == 1
    assert chunks["c06_gpu_t0"]["status"] == "partial" and (chunks["c06_gpu_t0"]["units_done"], chunks["c06_gpu_t0"]["units_total"]) == (1, 14)
    assert d["next_chunk"]["cpu"]["chunk_id"] == "c02_cpu_t0" and d["next_chunk"]["gpu"]["chunk_id"] == "c06_gpu_t0"
    assert d["cpu"]["units_done"] == 1 and d["cpu"]["done_hours"] > 0 and d["running"] == []


def test_a_finished_cpu_pilot_route_cuts_the_t4_alternative_and_thinking_counts_only_for_reasoning(tmp_path, capsys):
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    rows = L.read_ledger(led)
    by = {r["unit_id"]: r for r in rows}
    by["pilot_t1_200__gemma-3-1b-it__nfc__cpu"].update(status="done", hash_verified="yes")
    by["pilot_xcopa_200__phogpt-4b-chat__nfd"].update(status="done", hash_verified="yes")       # the T4 route finished instead
    by["pilot_xcopa_200__qwen3.5-2b__nfc"].update(status="partial")                             # started on the T4: left alone
    by["pilot_xcopa_200__qwen3.5-2b__nfc__cpu"].update(status="done", hash_verified="yes")
    L.write_ledger(rows, led)
    rc, summary = _ledger_ingest(led, tmp_path / "noruns", tmp_path, capsys)
    assert rc == 0 and set(summary["alternative_cut"]) == {"pilot_t1_200__gemma-3-1b-it__nfc", "pilot_xcopa_200__phogpt-4b-chat__nfd__cpu"}
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert by["pilot_t1_200__gemma-3-1b-it__nfc"]["status"] == "cut" and "memo N11" in by["pilot_t1_200__gemma-3-1b-it__nfc"]["notes"]
    assert by["pilot_xcopa_200__phogpt-4b-chat__nfd__cpu"]["status"] == "cut"
    assert by["pilot_xcopa_200__qwen3.5-2b__nfc"]["status"] == "partial"
    assert by["pilot_t1_200__gemma-3-1b-it__nfd"]["status"] == "alternative"               # its CPU twin is not done yet
    # `finished_thinking_present` finishes a reasoning unit, not a main one (DD 7.3)
    plans = [L.load_plan(L.PLAN)]
    for exp, status in (("reasoning", "done"), ("E1", "partial")):
        d = tmp_path / f"dir_{exp}"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"run_id": "x__m", "status": "finished_thinking_present",
                                                     "options": {"arms": ["nfc"], "paraphrases": ["p0"]},
                                                     "item_file": {"n_selected": 2}, "data": {}}), encoding="utf-8")
        (d / "outputs.jsonl").write_text('{"arm": "nfc"}\n{"arm": "nfc"}\n', encoding="utf-8")
        row = {"unit_id": "x__m__nfc", "experiment": exp, "item_key": "nope", "status": "planned", "notes": ""}
        touched, unmatched = L.ingest_run_dir(d, {"x__m__nfc": row}, plans, tmp_path / "agg2")
        assert touched == ["x__m__nfc"] and unmatched is None and row["status"] == status, exp


def test_ingest_reports_two_directories_that_name_the_same_unit(tmp_path, capsys):
    runs = tmp_path / "runs"
    for name in ("copy_of_the_run", "smoke_20__gemma-3-1b-it__cpu"):           # the same manifest run_id in two folders
        (runs / name).mkdir(parents=True)
        (runs / name / "manifest.json").write_text(json.dumps({"identity": {"run_id": "smoke_20__gemma-3-1b-it__cpu"}, "status": "finished",
                                                              "options": {"arms": ["nfc"], "paraphrases": ["p0"]},
                                                              "item_file": {"n_selected": 1}, "data": {}}), encoding="utf-8")
        (runs / name / "outputs.jsonl").write_text('{"arm": "nfc"}\n', encoding="utf-8")
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    _rc, summary = _ledger_ingest(led, runs, tmp_path, capsys)
    assert summary["conflicts"] == [{"unit": "smoke_20__gemma-3-1b-it__nfc__cpu", "dirs": ["copy_of_the_run", "smoke_20__gemma-3-1b-it__cpu"],
                                     "reason": summary["conflicts"][0]["reason"]}] and "more than one run directory" in summary["conflicts"][0]["reason"]
    assert summary["unmatched"] == [] and summary["touched"] == 2                 # the unit (once) and attested__expansion


def test_ingest_human_reads_the_packet_report_and_the_baseline_report(tmp_path):
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    rows = L.read_ledger(led)
    by = {r["unit_id"]: r for r in rows}
    val = tmp_path / "validation"
    (val / "report").mkdir(parents=True)
    (val / "validators.json").write_text(json.dumps({"A": {"region": "N"}, "B": {"region": "S"}, "C": {"region": "C"}}), encoding="utf-8")
    (val / "validation_manifest.json").write_text(json.dumps({"per_validator": {"A": {"B": 100, "hours_estimate": 5.4},
                                                                                  "B": {"B": 100, "hours_estimate": 5.4},
                                                                                  "C": {"B": 100, "hours_estimate": 5.4}}}), encoding="utf-8")
    report = val / "report" / "validation_report.json"
    report.write_text(json.dumps({"regions": {"A": "N", "B": "S", "C": "C"}, "validators_returned": ["A", "B"],
                                  "rows_returned": {"A_calibration": {"A": 10, "B": 10}, "B_items": {"A": 100, "B": 40}}}), encoding="utf-8")
    human = tmp_path / "human"
    human.mkdir()
    item_level = tmp_path / "human_scores.jsonl"
    item_level.write_text('{"form": "01"}\n', encoding="utf-8")
    (human / "human_baseline_report.json").write_text(json.dumps({
        "forms_returned": 3, "forms_included": ["01", "03"], "forms_excluded": {"baseline_form_02_r1.csv": "tool_use"},
        "per_form": {"01": {"file": "baseline_form_01_r1.csv", "response": 1, "n_main": 30, "answered": 28},
                     "03": {"file": "baseline_form_03_r1.csv", "response": 1, "n_main": 30, "answered": 30}},
        "item_level": {"path": str(item_level), "n_rows": 58, "sha256": "cd" * 32}}), encoding="utf-8")
    touched, unmatched = L.ingest_human(by, human, val)
    assert unmatched == [] and set(touched) == {"validation__N", "validation__S", "validation__C", "baseline__form01", "baseline__form03", "baseline__form02"}
    n, s, c = by["validation__N"], by["validation__S"], by["validation__C"]
    assert n["status"] == "done" and (n["n_rows"], n["n_expected"]) == (100, 100) and n["result_sha256"] == L.sha256_file(report)
    assert s["status"] == "partial" and (s["n_rows"], s["n_expected"]) == (40, 100)
    assert c["status"] == "planned" and c["n_rows"] == 0 and not c["result_file"]
    assert all(float(x["est_raw"]) == 5.4 for x in (n, s, c))
    f1, f2, f3 = by["baseline__form01"], by["baseline__form02"], by["baseline__form03"]
    assert f1["status"] == f3["status"] == "done" and f1["result_sha256"] == "cd" * 32 and f1["result_file"] == str(item_level)
    assert (f1["n_rows"], f1["n_expected"]) == (28, 30) and f2["status"] == "excluded" and "excluded: tool_use" in f2["notes"]
    assert by["baseline__form04"]["status"] == "planned"
    # letters that map to no region are reported, not guessed; two validators leave the third region unscheduled (memo 7)
    report.write_text(json.dumps({"regions": {"A": "N", "B": "S", "D": "Mars"}, "validators_returned": ["A", "D"],
                                  "rows_returned": {"B_items": {"A": 100, "D": 5}}}), encoding="utf-8")
    (val / "validation_manifest.json").write_text(json.dumps({"per_validator": {"A": {"B": 100, "hours_estimate": 6.9},
                                                                                  "B": {"B": 100, "hours_estimate": 6.9}}}), encoding="utf-8")
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    touched, unmatched = L.ingest_human(by, tmp_path / "nohuman", val)
    assert [u["validator"] for u in unmatched] == ["D"] and "region" in unmatched[0]["reason"]
    assert by["validation__N"]["status"] == "done" and float(by["validation__N"]["est_raw"]) == 6.9
    assert by["validation__C"]["status"] == "unscheduled" and "two validators" in by["validation__C"]["notes"]
    # the meter: person-hours on their own lines, an excluded form stays in the denominator
    L.write_ledger(list(by.values()), led)
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    assert d["human"]["human_validation"][2] == pytest.approx(6.9 + 6.9) and d["human"]["human_baseline"][2] == pytest.approx(20 * 0.6)


# ------------------------------------------------------------------ the meter
def test_progress_meter_counts_only_done_and_hash_verified_units_per_chunk_and_names_the_next_chunk(tmp_path, capsys):
    led = tmp_path / "ledger.csv"
    L.main(["--ledger", str(led), "init"])
    rows = L.read_ledger(led)
    by = {r["unit_id"]: r for r in rows}
    by["E1_main__gemma-3-1b-it__nfc"].update(status="done", hash_verified="yes")
    by["E1_main__llama-3.1-8b-instruct__nfc"].update(status="done", hash_verified="no")   # must NOT count
    by["tpu_main__gemma-3-12b-it__nfc__p0p1"].update(status="cut")                          # leaves the denominator
    by["tpu_main__qwen3.8-27b__nfc__p2"].update(status="done", hash_verified="yes")         # carries no p0: no E1 model
    by["E3_noilai__gemma-3-1b-it__nfc"].update(status="partial", n_rows="10", n_expected="4200")
    by["reasoning_500__qwen3.5-4b--thinking__nfc"].update(status="done", hash_verified="no")   # c18's only unit ran on the wrong file
    L.write_ledger(rows, led)
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    done, tot = d["per_experiment"]["E1"][1], d["per_experiment"]["E1"][2]
    assert done == pytest.approx(float(by["E1_main__gemma-3-1b-it__nfc"]["est_cost"]) + float(by["tpu_main__qwen3.8-27b__nfc__p2"]["est_cost"]))
    full = sum(float(r["est_cost"]) for r in rows if r["experiment"] == "E1" and r["status"] not in L.NON_LIVE_STATUSES)
    assert tot == pytest.approx(full)
    assert d["analyses"]["Table 2 (E1 estimation per done model)"] == {"computable": True, "primary": False, "n": "1 models"}
    assert d["analyses"]["H1 nested LR test (E2), PREREG 8.4"]["computable"] is False
    chunks = {c["chunk_id"]: c for c in d["chunks"]}
    assert [c["order"] for c in d["chunks"]] == list(range(1, 32))
    assert chunks["c08_gpu_t1"]["status"] == "partial" and (chunks["c08_gpu_t1"]["units_done"], chunks["c08_gpu_t1"]["units_total"]) == (1, 19)
    assert chunks["c29_tpu_t4"]["units_total"] == 20 and chunks["c29_tpu_t4"]["units_all"] == 21 and chunks["c29_tpu_t4"]["status"] == "planned"
    assert chunks["c31_tpu_t5"]["status"] == "partial" and chunks["c05_gpu_t0"]["status"] == "alternative" and chunks["c05_gpu_t0"]["units_total"] == 0
    assert chunks["c08_gpu_t1"]["quota"]["x2"] == [round(2 * chunks["c08_gpu_t1"]["hours"][0], 2), round(2 * chunks["c08_gpu_t1"]["hours"][1], 2)]
    assert chunks["c16_tpu_t1"]["quota"] == {"x1": chunks["c16_tpu_t1"]["hours"]} and chunks["c01_cpu_t0"]["quota"]["x1"] == [0.0, 0.0]
    assert chunks["c08_gpu_t1"]["in_flight"] == ["E3_noilai__gemma-3-1b-it__nfc"]
    assert d["per_tier"]["1"]["chunks_total"] == 9 and d["per_queue"]["gpu"]["chunks_total"] == 19 and d["per_queue"]["cpu"]["chunks_total"] == 4
    assert d["next_chunk"] == {"cpu": {**{k: chunks["c01_cpu_t0"][k] for k in d["next_chunk"]["cpu"]}},
                               "gpu": {**{k: chunks["c06_gpu_t0"][k] for k in d["next_chunk"]["gpu"]}},
                               "tpu": {**{k: chunks["c07_tpu_t0"][k] for k in d["next_chunk"]["tpu"]}}}
    assert d["running"] == [("c08_gpu_t1", "E3_noilai__gemma-3-1b-it__nfc", "partial", "10", "4200")]
    assert d["hash_mismatch"] == [("c15_gpu_t1", "E1_main__llama-3.1-8b-instruct__nfc"), ("c18_gpu_t2", "reasoning_500__qwen3.5-4b--thinking__nfc")]
    assert chunks["c18_gpu_t2"]["status"] == "partial" and chunks["c18_gpu_t2"]["units_done"] == 0   # ran, counts nothing, not planned
    assert d["frontier_chunks"]["total"]["gpu_x1"] == d["per_queue"]["gpu"]["hours"] and d["frontier_chunks"]["total"]["tpu"] == d["per_queue"]["tpu"]["hours"]
    fc = d["frontier_chunks"]["first_computable"]
    assert fc["Table 2 (E1 estimation per done model)"]["chunk_id"] == "c08_gpu_t1" and fc["H1 nested LR test (E2), PREREG 8.4"]["chunk_id"] == "c15_gpu_t1"
    assert fc["H1 nested LR test (E2), PREREG 8.4"]["gpu_x2"] == pytest.approx([2 * x for x in fc["H1 nested LR test (E2), PREREG 8.4"]["gpu_x1"]], abs=0.011)
    assert "H5 probes and patching (E4)" in d["frontier_chunks"]["outside_chunk_walk"]
    # every chunk of c01 done -> the chunk is done and the next CPU chunk is c02; a unit with hash `no` keeps a chunk from done
    by["smoke_20__gemma-3-1b-it__nfc__cpu"].update(status="done", hash_verified="yes")
    for r in rows:
        if r["chunk_id"] == "c02_cpu_t0":
            r.update(status="done", hash_verified="yes")
    by["pilot_xcopa_200__gemma-3-1b-it__nfd__cpu"]["hash_verified"] = "no"
    L.write_ledger(rows, led)
    d = P.compute(led, tmp_path / "nogates.yaml", ROOT / "experiments" / "quota.yaml", tmp_path / "nolog.csv")
    chunks = {c["chunk_id"]: c for c in d["chunks"]}
    assert chunks["c01_cpu_t0"]["status"] == "done" and chunks["c02_cpu_t0"]["status"] == "partial" and chunks["c02_cpu_t0"]["units_done"] == 3
    assert d["next_chunk"]["cpu"]["chunk_id"] == "c02_cpu_t0" and d["per_tier"]["0"]["chunks_done"] == 1 and d["per_queue"]["cpu"]["chunks_done"] == 1
    assert d["cpu"]["units_done"] == 4 and d["overall"][2] == pytest.approx(tot + sum(v[2] for k, v in d["per_experiment"].items() if k != "E1"))
    block = P.render(d)
    assert P.BEGIN in block and "Explicit-input gap" in block and "### Chunks" in block and "Next chunk per queue" in block
    assert "| `c01_cpu_t0` | 1 | 0 | cpu |" in block and "- cpu: `c02_cpu_t0`" in block and "1 CPU-h = 0.0" in block
    assert "`c08_gpu_t1` / `E3_noilai__gemma-3-1b-it__nfc`: partial (10/4200 rows)" in block
    assert "Done but NOT counted" in block and "`c18_gpu_t2` / `reasoning_500__qwen3.5-4b--thinking__nfc`" in block
    # --status PATH rewrites the block of that file only (a temporary copy of STATUS.md here), keeping the hand text
    status = tmp_path / "STATUS.md"
    status.write_text("# STATUS\n\nhand text\n\n" + P.BEGIN + "\nold\n" + P.END + "\n\n## after\n", encoding="utf-8")
    capsys.readouterr()
    assert P.main(["--ledger", str(led), "--status", str(status), "--gates", str(tmp_path / "nogates.yaml"),
                   "--compute-log", str(tmp_path / "nolog.csv"), "--json", str(tmp_path / "meter.json")]) == 0
    t = status.read_text(encoding="utf-8")
    assert "hand text" in t and "## after" in t and "old" not in t and "### Chunks" in t and t.count(P.BEGIN) == 1
    assert json.loads((tmp_path / "meter.json").read_text(encoding="utf-8"))["n_chunks"] == 31


def test_kaggle_hours_this_week_keep_cpu_rows_apart_and_read_x2_from_n_gpus(tmp_path):
    log = tmp_path / "compute_log.csv"
    today = CL.today()
    for gpu_type, n, hours, rid in (("t4", 2, 1.5, "E1_main__gemma-3-4b-it__p0p1"), ("t4", 2, 0.0, "E1_main__gemma-3-4b-it__p0p1"),
                                    ("tpu-v5e-8", 1, 2.0, "tpu_main__qwen3.8-27b__p0p1"), ("cpu", 0, 3.0, "smoke_20__gemma-3-1b-it__cpu"),
                                    ("api", 0, 0.5, "E1_api_core__gpt-oss-20b")):
        CL.append_entry(log, CL.Entry(today, "kaggle", gpu_type, n, hours, rid, "test"))
    quota = yaml.safe_load((ROOT / "experiments" / "quota.yaml").read_text(encoding="utf-8"))
    k = P.kaggle_hours_this_week(quota, log)
    assert (k["gpu_used"], k["gpu_used_x2"], k["tpu_used"], k["cpu_used"]) == (1.5, 3.0, 2.0, 3.0)
    assert k["gpu_left"] == pytest.approx(30 - 1.5) and k["gpu_left_x2"] == pytest.approx(30 - 3.0) and k["tpu_left"] == 18.0


def test_reprice_uses_measured_hours_per_run_directory_and_skips_cpu_rows(tmp_path, capsys):
    led = tmp_path / "ledger.csv"
    assert L.main(["--ledger", str(led), "init"]) == 0
    log = tmp_path / "compute_log.csv"
    today = CL.today()
    for gpu_type, n, hours, rid in (("t4", 2, 0.0, "E1_main__gemma-3-1b-it"), ("t4", 2, 1.5, "E1_main__gemma-3-1b-it"),
                                    ("tpu-v5e-8", 1, 2.0, "tpu_main__qwen3.8-27b__p0p1"), ("cpu", 0, 4.0, "smoke_20__gemma-3-1b-it__cpu"),
                                    ("t4", 2, 0.3, "session-c08_gpu_t1")):
        CL.append_entry(log, CL.Entry(today, "kaggle", gpu_type, n, hours, rid, "test"))
    capsys.readouterr()
    assert L.main(["--ledger", str(led), "reprice", "--compute-log", str(log)]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["repriced_units"] == 2 and summary["cpu_rows_skipped"] == 1 and summary["measured_runs"] == 3
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert float(by["E1_main__gemma-3-1b-it__nfc"]["est_raw"]) == 1.5 and float(by["E1_main__gemma-3-1b-it__nfc"]["est_cost"]) == 1.5
    assert float(by["tpu_main__qwen3.8-27b__nfc__p0p1"]["est_raw"]) == 2.0 and "repriced from measured 2.00 h" in by["tpu_main__qwen3.8-27b__nfc__p0p1"]["notes"]
    assert "repriced" not in by["smoke_20__gemma-3-1b-it__nfc__cpu"]["notes"]
    rep = {x["run_dir"]: x for x in summary["measured_vs_chunk"]}
    assert set(rep) == {"E1_main__gemma-3-1b-it", "tpu_main__qwen3.8-27b__p0p1"}
    assert rep["E1_main__gemma-3-1b-it"]["chunk_id"] == "c08_gpu_t1" and rep["E1_main__gemma-3-1b-it"]["measured_hours"] == 1.5
    assert rep["tpu_main__qwen3.8-27b__p0p1"]["chunk_estimate_hours"] == [pytest.approx(4 / 3 * 2 / 3, abs=1e-2), pytest.approx(8 / 3 * 2 / 3, abs=1e-2)]
    assert "plan_chunks.py --write" in summary["next"] and "init --force" in summary["next"]
    capsys.readouterr()
    assert L.main(["--ledger", str(led), "init", "--force"]) == 0                       # measured hours are observations: kept
    by = {r["unit_id"]: r for r in L.read_ledger(led)}
    assert float(by["E1_main__gemma-3-1b-it__nfc"]["est_raw"]) == 1.5 and float(by["E1_main__gemma-3-1b-it__nfc"]["est_cost"]) == 1.5
    assert "repriced from measured 1.50 h" in by["E1_main__gemma-3-1b-it__nfc"]["notes"]
    other = by["E1_main__phogpt-4b-chat__nfc"]                                            # not repriced: the plan estimate
    assert float(other["est_raw"]) == pytest.approx((float(other["chunk_hours_low"]) + float(other["chunk_hours_high"])) / 2, abs=1e-3)


def test_scoring_rule_freeze_records_in_place_and_keeps_comments(tmp_path):
    F = importlib.import_module("freeze_scoring_rule")
    gates = tmp_path / "gates.yaml"
    gates.write_text("# header comment\npreregistration:\n  registration_url: null\n\nscoring_rule_freeze:\n  # kept\n  commit: null\n  files_sha256: null\n  frozen_on: null\n", encoding="utf-8")
    assert F.main(["--gates", str(gates), "--check"]) == 1                     # nothing recorded yet
    assert F.main(["--gates", str(gates), "--record"]) == 0
    text = gates.read_text(encoding="utf-8")
    assert "# header comment" in text and "# kept" in text and "registration_url: null" in text
    rec = yaml.safe_load(text)["scoring_rule_freeze"]
    assert rec["files_sha256"] == F.scoring_rule_sha256()[0] and rec["frozen_on"]
    assert F.main(["--gates", str(gates), "--check"]) == 0
    empty = tmp_path / "new.yaml"
    assert F.main(["--gates", str(empty), "--record"]) == 0 and yaml.safe_load(empty.read_text())["scoring_rule_freeze"]["files_sha256"]
