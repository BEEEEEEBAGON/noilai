"""The pre-registration gate: experiments/gates.yaml and its enforcement in scripts/kaggle_run_plan.py.

CLAUDE.md "Binding documents and gates" (DESIGN_DECISIONS 8.8, 13.3, 13.5, 13.18): no confirmatory run --
a line whose item file carries the canary (test split or core), or any line that is neither smoke, pilot
nor exploratory -- starts until the registration link and date and the author's `qu` and `i/y` decisions
are recorded; an API-served model also waits for the Gemini route. A project root without the file is
never gated, which is what keeps the temporary project roots of tests/test_cloud.py running.

Everything here runs offline through the stub run_eval.py of tests/test_cloud.py (STUB_RUN_EVAL, with its
pinned-panel fixture pattern) in temporary project roots; nothing is written under data/release/.
"""
import copy
import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

KRP = importlib.import_module("kaggle_run_plan")
CL = importlib.import_module("compute_log")
TC = importlib.import_module("test_cloud")          # STUB_RUN_EVAL (the stub run_eval.py), PINNED_REVISION

GATES = ROOT / "experiments" / "gates.yaml"
NULL_GATES = yaml.safe_load(GATES.read_text(encoding="utf-8"))
# the four entries every confirmatory line waits for (CLAUDE.md) and the one an API model waits for on top
REGISTRATION = {"registration_url": "https://example.org/prereg/noilai", "registration_date": "2026-11-08"}
DECISIONS = {"qu_convention": "qu- inputs enter as a stratum", "iy_emission": "y-preference confirmed"}
GEMINI_ROUTE = "paid_key_opt_out"
SELF_HOSTED, API_GROQ, API_GEMINI = "gemma-3-1b-it", "gpt-oss-20b", "gemini-flash"


@pytest.fixture(scope="module")
def models_cfg():
    return KRP.load_models()


@pytest.fixture(scope="module")
def plan():
    """The pre-registered plan as loaded; the stub projects hold no item files, so every run passes
    verify=False (the one that does not shows the gate deciding before the item-file verification)."""
    return KRP.load_plan()


@pytest.fixture(scope="module")
def pinned_cfg(models_cfg):
    """models.yaml with every self-hosted `revision` pinned (DD 7.1), as tests/test_cloud.py does, so that
    a lifted gate shows as a run and not as the revision skip."""
    cfg = copy.deepcopy({k: v for k, v in models_cfg.items() if k != "by_name"})
    for m in cfg["models"]:
        if not KRP.is_api(m):
            m["revision"] = TC.PINNED_REVISION
    cfg["by_name"] = {m["name"]: m for m in cfg["models"]}
    return cfg


def _filled(**overrides) -> dict:
    """The repository's gates file with the registration, the two decisions and (optionally) the Gemini route
    recorded; `overrides` drop or change single fields (registration_url=None, gemini_route="drop", ...)."""
    g = copy.deepcopy(NULL_GATES)
    g["preregistration"].update(REGISTRATION)
    for key, value in DECISIONS.items():
        g["decisions"][key]["value"] = value
        g["decisions"][key]["decided_on"] = "2026-10-20"
    for key, value in overrides.items():
        if key in g["preregistration"]:
            g["preregistration"][key] = value
        else:
            g["decisions"][key]["value"] = value
    return g


def _project(tmp_path: Path, gates: dict | None) -> Path:
    """A project root holding the stub run_eval.py of tests/test_cloud.py and the given gates file (none when
    `gates` is None); no item files, no compute log, no API ledger."""
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    (proj / "scripts" / "run_eval.py").write_text(TC.STUB_RUN_EVAL, encoding="utf-8")
    if gates is not None:
        (proj / "experiments").mkdir()
        (proj / "experiments" / "gates.yaml").write_text(yaml.safe_dump(gates, sort_keys=False), encoding="utf-8")
    return proj


def _ran(proj: Path, run_id: str, model: str) -> bool:
    return (proj / "data" / "runs" / f"{run_id}__{model}" / "outputs.jsonl").exists()


# ------------------------------------------------------------------ the file
def test_the_repository_gates_file_has_the_documented_shape_and_every_reader_accepts_it():
    g = NULL_GATES
    assert set(g) == {"preregistration", "decisions", "scoring_rule_freeze"}
    assert set(g["preregistration"]) == {"stage1_commit", "stage2_commit", "registration_url", "registration_date"}
    assert set(g["decisions"]) == {"qu_convention", "iy_emission", "gemini_route"}
    assert set(g["decisions"]["qu_convention"]) == {"value", "decided_on", "affects"}
    assert g["decisions"]["qu_convention"]["affects"] == "items, prompts, gold answers (DD 13.3)"
    assert set(g["decisions"]["iy_emission"]) == {"value", "decided_on", "affects"}
    assert g["decisions"]["iy_emission"]["affects"] == "gold answers (DD 13.5)"
    assert set(g["decisions"]["gemini_route"]) == {"value", "decided_on", "allowed"}
    assert g["decisions"]["gemini_route"]["allowed"] == ["paid_key_opt_out", "dev_derived_set", "drop"]
    route = g["decisions"]["gemini_route"]["value"]
    assert route is None or route in g["decisions"]["gemini_route"]["allowed"]
    assert {"commit", "files_sha256", "frozen_on"} <= set(g["scoring_rule_freeze"])    # --record adds `files`
    # every gate value is null (not yet recorded) or a recorded scalar, never a structure
    for keys, _label in (*KRP.CONFIRMATORY_REQUIREMENTS, *KRP.API_REQUIREMENTS):
        value = KRP.gate_value(g, keys)
        assert value is None or isinstance(value, str) or hasattr(value, "isoformat"), keys
    # the three readers of the file agree on its layout
    assert KRP.gate_status(ROOT)["present"] is True and KRP.gate_status(ROOT)["decisions"] == g["decisions"]
    progress = importlib.import_module("progress")
    summary = progress.gates_summary(GATES)
    assert summary["present"] and set(summary) > {"registration_url", "qu_convention", "iy_emission", "gemini_route"}
    ledger = importlib.import_module("ledger")
    assert KRP.UNGATED_EXPERIMENTS == ledger.EXPLORATORY_EXPERIMENTS | ledger.PREREQUISITE_EXPERIMENTS


def test_gate_status_reports_an_absent_file_or_returns_its_contents(tmp_path):
    assert KRP.gate_status(tmp_path) == {"present": False}
    (tmp_path / "experiments").mkdir()
    path = tmp_path / "experiments" / "gates.yaml"
    path.write_text(yaml.safe_dump(NULL_GATES, sort_keys=False), encoding="utf-8")
    status = KRP.gate_status(tmp_path)
    assert status["present"] is True and status["preregistration"] == NULL_GATES["preregistration"]
    path.write_text("", encoding="utf-8")                   # an empty file: present, nothing recorded
    assert KRP.gate_status(tmp_path) == {"present": True}
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        KRP.gate_status(tmp_path)


# ------------------------------------------------------------------ the rule
def test_block_reason_follows_the_item_file_the_experiment_and_the_backend(plan, models_cfg):
    by = models_cfg["by_name"]
    files = plan["item_files"]
    e1_main, e3_xcopa = KRP.find_run(plan, "E1_main"), KRP.find_run(plan, "E3_xcopa")
    smoke, pilot = KRP.find_run(plan, "smoke_20"), KRP.find_run(plan, "pilot_t1_200")
    nulls = {"present": True, **NULL_GATES}
    # a canary-bearing (test-split) file: every missing entry is named
    reason = KRP.confirmatory_block_reason(e1_main, by[SELF_HOSTED], files[e1_main["items"]], nulls)
    assert reason.startswith("item file 'noilai_main' carries the canary")
    for label in ("pre-registration link", "pre-registration date", "qu convention (DD 13.3)", "i/y emission rule (DD 13.5)"):
        assert label in reason
    assert "Gemini" not in reason and "experiments/gates.yaml" in reason
    # a confirmatory experiment on a public file (E3 on XCOPA) is held by the experiment rule
    assert files[e3_xcopa["items"]].get("canary_required") is False
    assert KRP.confirmatory_block_reason(e3_xcopa, by[SELF_HOSTED], files[e3_xcopa["items"]], nulls).startswith(
        "experiment 'E3' is confirmatory")
    # smoke, the Gate 1 pilots and a line flagged exploratory on a public file are never held
    assert KRP.confirmatory_block_reason(smoke, by[SELF_HOSTED], files[smoke["items"]], nulls) is None
    assert KRP.confirmatory_block_reason(pilot, by[SELF_HOSTED], files[pilot["items"]], nulls) is None
    exploratory = {**e3_xcopa, "exploratory": True}
    assert KRP.confirmatory_block_reason(exploratory, by[SELF_HOSTED], files[e3_xcopa["items"]], nulls) is None
    # ... but the canary rule wins over the flag: a test-split file is confirmatory whatever the line says
    assert KRP.confirmatory_block_reason({**e1_main, "exploratory": True}, by[SELF_HOSTED], files[e1_main["items"]], nulls)
    assert KRP.is_confirmatory({"experiment": "smoke"}, {"canary_required": True})
    assert not KRP.is_confirmatory({"experiment": "smoke"}, {})
    # no file: never a reason
    assert KRP.confirmatory_block_reason(e1_main, by[SELF_HOSTED], files[e1_main["items"]], {"present": False}) is None
    # the four entries lift the gate for a self-hosted model; an API model waits for the Gemini route
    filled = {"present": True, **_filled()}
    assert KRP.confirmatory_block_reason(e1_main, by[SELF_HOSTED], files[e1_main["items"]], filled) is None
    e1_api = KRP.find_run(plan, "E1_api_core")
    for api in (API_GROQ, API_GEMINI):
        reason = KRP.confirmatory_block_reason(e1_api, by[api], files[e1_api["items"]], filled)
        assert reason.endswith("waits for Gemini route (DD 13.18) in experiments/gates.yaml"), reason
    for route in NULL_GATES["decisions"]["gemini_route"]["allowed"]:
        assert KRP.confirmatory_block_reason(e1_api, by[API_GROQ], files[e1_api["items"]], {"present": True, **_filled(gemini_route=route)}) is None
    reason = KRP.confirmatory_block_reason(e1_api, by[API_GROQ], files[e1_api["items"]], {"present": True, **_filled(gemini_route="maybe")})
    assert "'maybe' is not one of ['paid_key_opt_out', 'dev_derived_set', 'drop']" in reason
    # a blank string is "not recorded"; a partial fill names only what is still missing
    partial = {"present": True, **_filled(registration_date="", iy_emission=None)}
    reason = KRP.confirmatory_block_reason(e1_main, by[SELF_HOSTED], files[e1_main["items"]], partial)
    assert reason.endswith("waits for pre-registration date, i/y emission rule (DD 13.5) in experiments/gates.yaml")
    # a file without the sections (an empty gates.yaml) holds everything
    assert KRP.confirmatory_block_reason(e1_main, by[SELF_HOSTED], files[e1_main["items"]], {"present": True})


# ------------------------------------------------------------------ execute()
def test_execute_holds_every_model_of_a_confirmatory_line_and_launches_nothing(tmp_path, plan, pinned_cfg, models_cfg, capsys):
    proj = _project(tmp_path, NULL_GATES)
    log = proj / "log.csv"
    res = KRP.execute("E1_main", models=[SELF_HOSTED, "gemma-3-4b-it"], plan=plan, models_cfg=pinned_cfg, project_root=proj,
                      log_path=log, verify=False)
    assert [r["model"] for r in res] == [SELF_HOSTED, "gemma-3-4b-it"]
    assert all(r["status"].startswith("skipped: gate: item file 'noilai_main' carries the canary") for r in res)
    assert all(r["prereg_gate"] and r["cmd"] for r in res), "the command is built (and printed) before the gate holds it"
    assert not (proj / "data" / "runs").exists() and not log.exists(), "no command ran, nothing was logged"
    # an API model on the core, a non-canary confirmatory experiment: held alike; the API ledger stays untouched
    res = KRP.execute("E1_api_core", models=[API_GROQ], platform="api", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                      log_path=log, verify=False)
    assert res[0]["status"].startswith("skipped: gate:") and not KRP.ledger_path(plan, proj).exists()
    res = KRP.execute("E3_xcopa", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"].startswith("skipped: gate: experiment 'E3' is confirmatory")
    # the gate comes before the revision check and before the item-file verification: the reason is the gate's
    res = KRP.execute("E1_main", models=[SELF_HOSTED], plan=plan, models_cfg=models_cfg, project_root=proj, log_path=log)
    assert res[0]["status"].startswith("skipped: gate:") and "verify" not in res[0]
    assert not _ran(proj, "E1_main", SELF_HOSTED) and not log.exists()
    # a dry run prints the command and the reason, and stays a dry run
    capsys.readouterr()
    res = KRP.execute("E1_main", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, dry_run=True)
    out = capsys.readouterr().out
    assert res[0]["status"] == "dry-run" and res[0]["prereg_gate"].startswith("item file 'noilai_main' carries the canary")
    assert "[cmd  ]" in out and f"[gate ] {SELF_HOSTED}: item file 'noilai_main' carries the canary" in out


def test_smoke_pilot_and_exploratory_lines_run_under_closed_gates(tmp_path, plan, pinned_cfg):
    proj = _project(tmp_path, NULL_GATES)
    log = proj / "log.csv"
    res = KRP.execute("smoke_20", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"] == "ok" and "prereg_gate" not in res[0] and _ran(proj, "smoke_20", SELF_HOSTED)
    res = KRP.execute("pilot_t1_200", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"] == "ok" and _ran(proj, "pilot_t1_200", SELF_HOSTED)
    # a dev-only line flagged `exploratory` (configs/run_plan_exploratory.yaml's marker) on a public file runs too
    xplan = copy.deepcopy(plan)
    smoke = KRP.find_run(xplan, "smoke_20")
    xplan["runs"].append({**{k: v for k, v in smoke.items() if k != "limit"}, "id": "floor_dev", "experiment": "E1_floor",
                          "exploratory": True})
    res = KRP.execute("floor_dev", models=[SELF_HOSTED], plan=xplan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"] == "ok" and _ran(proj, "floor_dev", SELF_HOSTED)
    assert len(CL.read_entries(log)) == 6, "two compute-log rows per executed model, none for a held one"


def test_recording_the_registration_and_the_decisions_lifts_the_gate_but_api_models_wait_for_gemini(tmp_path, plan, pinned_cfg):
    proj = _project(tmp_path, _filled())
    log = proj / "log.csv"
    res = KRP.execute("E1_main", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"] == "ok" and "prereg_gate" not in res[0] and _ran(proj, "E1_main", SELF_HOSTED)
    res = KRP.execute("E1_api_core", models=[API_GROQ], platform="api", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                      log_path=log, verify=False)
    assert res[0]["status"].endswith("waits for Gemini route (DD 13.18) in experiments/gates.yaml"), res[0]["status"]
    assert not _ran(proj, "E1_api_core", API_GROQ)
    # the Gemini decision recorded: the Groq model runs (Gemini itself is still refused on the core by DD 11.2's guard)
    (proj / "experiments" / "gates.yaml").write_text(yaml.safe_dump(_filled(gemini_route=GEMINI_ROUTE), sort_keys=False), encoding="utf-8")
    res = KRP.execute("E1_api_core", models=[API_GROQ, API_GEMINI], platform="api", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                      log_path=log, verify=False)
    assert res[0]["status"] == "ok" and _ran(proj, "E1_api_core", API_GROQ)
    assert res[1]["status"].startswith("refused:") and "trains on inputs" in res[1]["status"]
    # one entry missing again closes the gate for everyone
    (proj / "experiments" / "gates.yaml").write_text(yaml.safe_dump(_filled(qu_convention=None), sort_keys=False), encoding="utf-8")
    res = KRP.execute("E1_explicit_input", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"].endswith("waits for qu convention (DD 13.3) in experiments/gates.yaml")


def test_a_project_root_without_gates_yaml_is_never_gated(tmp_path, plan, pinned_cfg):
    proj = _project(tmp_path, None)
    log = proj / "log.csv"
    assert KRP.gate_status(proj) == {"present": False}
    res = KRP.execute("E1_main", models=[SELF_HOSTED], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log, verify=False)
    assert res[0]["status"] == "ok" and "prereg_gate" not in res[0] and _ran(proj, "E1_main", SELF_HOSTED)
    res = KRP.execute("E1_api_core", models=[API_GROQ], platform="api", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                      log_path=log, verify=False)
    assert res[0]["status"] == "ok" and _ran(proj, "E1_api_core", API_GROQ)


def test_the_cli_dry_run_on_the_repository_prints_the_gate_and_exits_zero(plan, models_cfg, capsys):
    """The repository root carries experiments/gates.yaml, so `--dry-run` on a confirmatory line shows the
    reason (while the gates are closed) and still prints the command; the exit code stays 0 (nothing failed)."""
    assert KRP.main(["--run", "E1_main", "--models", SELF_HOSTED, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"--model-config {SELF_HOSTED}" in out
    run = KRP.find_run(plan, "E1_main")
    reason = KRP.confirmatory_block_reason(run, models_cfg["by_name"][SELF_HOSTED], plan["item_files"][run["items"]], KRP.gate_status(ROOT))
    assert (f"[gate ] {SELF_HOSTED}: {reason}" in out) if reason else ("[gate ]" not in out)
    assert KRP.main(["--run", "smoke_20", "--models", SELF_HOSTED, "--dry-run"]) == 0
    assert "[gate ]" not in capsys.readouterr().out
