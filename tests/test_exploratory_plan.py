"""The DEV-ONLY exploratory run plan (configs/run_plan_exploratory.yaml) and the notebooks' PLAN_PATH parameter.

Everything here runs offline: the plan is loaded through the real driver (scripts/kaggle_run_plan.py),
its commands are checked against the real run_eval.py CLI, its two seeded dev files are derived from a
NON-frozen local dev split (the smoke build; copied into a temporary release directory, never into
data/release/), and the generated notebooks are parsed, never executed. The pre-registered plan
(configs/run_plan.yaml) is read only to check that the two Gate 1 pilot lines are verbatim copies.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import nbformat
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import kaggle_build_notebooks as KBN  # noqa: E402  (scripts/ is on sys.path only from here)
import kaggle_run_plan as KRP  # noqa: E402

EXPLORATORY = ROOT / "configs" / "run_plan_exploratory.yaml"
PREREGISTERED = ROOT / "configs" / "run_plan.yaml"
EXPLORATORY_RUN_IDS = {"floor_pilot_dev", "throughput_dev", "pilot_t1_200", "pilot_xcopa_200"}
PILOT_RUN_IDS = {"pilot_t1_200", "pilot_xcopa_200"}
NEW_DERIVED = {"pilot_t1t3_dev", "throughput_t1_300"}
# the keys of the pre-registered plan that read the test split, the core or the sealed split: none may appear here
TEST_SPLIT_KEYS = {"noilai_test", "noilai_main", "noilai_core", "noilai_c2", "noilai_api_para300", "noilai_reasoning500",
                   "noilai_bf16_200", "noilai_scope500", "sealed"}
FLOOR_MODELS = ["gemma-3-1b-it", "qwen3.5-0.8b", "qwen3.5-2b", "gemma-4-e2b", "gemma-sea-lion-v4.5-e2b-it"]
THROUGHPUT_MODELS = ["gemma-3-1b-it", "qwen3.5-2b", "phogpt-4b-chat", "llama-3.1-8b-instruct", "gemma-3-4b-it"]
# a NON-frozen local dev split for exercising the derivation (the frozen release files are absent from this
# checkout); NOILAI_DEV_SOURCE overrides the location. The real public dev split is used when it is built.
SMOKE_BUILD = Path("/tmp/claude-0/-home-user-noilai/7ed5d88b-f91c-57ec-b5a3-07ddcf2f1a1c/scratchpad/smoke_build/v0.3-smoke")


def _dev_source() -> Path | None:
    """noilai_dev.jsonl to derive from: the built release's, else the smoke build's, else None (skip)."""
    cands = [ROOT / KRP.load_plan(EXPLORATORY)["release"] / "noilai_dev.jsonl"]
    env = os.environ.get("NOILAI_DEV_SOURCE")
    if env:
        cands.insert(0, Path(env))
    cands.append(SMOKE_BUILD / "noilai_dev.jsonl")
    return next((p for p in cands if p.exists()), None)


@pytest.fixture(scope="module")
def plan():
    return KRP.load_plan(EXPLORATORY)


@pytest.fixture(scope="module")
def prereg():
    return KRP.load_plan(PREREGISTERED)


@pytest.fixture(scope="module")
def models_cfg():
    return KRP.load_models()


def _source_chain(plan: dict, key: str) -> str:
    """Follow `derive.source` links to the item file a derived file ultimately comes from."""
    seen = []
    while plan["item_files"][key].get("derive"):
        seen.append(key)
        key = plan["item_files"][key]["derive"]["source"]
        assert key not in seen, f"derive cycle at {key}"
    return key


# ------------------------------------------------------------------ the plan is dev-only and exploratory
def test_every_run_is_exploratory_dev_only_and_reads_no_test_split_core_or_sealed_file(plan, models_cfg):
    assert {r["id"] for r in plan["runs"]} == EXPLORATORY_RUN_IDS
    assert plan.get("preregistered") is False and plan["preregistered_plan"] == "configs/run_plan.yaml"
    assert plan["release"] == "data/release/v0.3" and plan["runs_root"] == "data/runs"
    assert set(plan["item_files"]).isdisjoint(TEST_SPLIT_KEYS)
    for key, spec in plan["item_files"].items():
        assert spec.get("canary_required") is False, (key, "a canary marks a test-split file (DD 4.6): none here")
        assert not spec.get("never_to_api"), key
        assert _source_chain(plan, key) in ("noilai_dev", "xcopa_vi_test"), key
        assert "{release}" not in spec["path"], key
    for run in plan["runs"]:
        assert run.get("exploratory") is True, run["id"]
        assert run.get("dev_only") is True, run["id"]
        assert run.get("in_core_only") is False, run["id"]
        assert not run.get("limit") and not run.get("not_a_run_eval_line"), (run["id"], "no run-time sampling (DD 12.32)")
        assert run["items"] in plan["item_files"], run["id"]
        assert run["experiment"] in plan["experiments"], run["id"]
        assert run["counts_toward"] in plan["plan_lines"], run["id"]
        assert run["platform"] == "kaggle" and run["hardware"] != "api", run["id"]
        for key in ("tasks", "arms", "paraphrases", "estimate", "purpose", "shots", "week"):
            assert key in run, (run["id"], key)
        for name in KRP.expand_models(run, plan, models_cfg):
            assert not KRP.is_api(models_cfg["by_name"][name]), (run["id"], name, "no API model sees a dev-only line here")
    for name in plan["model_sets"]:
        assert KRP.expand_set(name, plan, models_cfg)


def test_the_two_new_lines_are_as_specified(plan, models_cfg):
    by_id = {r["id"]: r for r in plan["runs"]}
    floor, thr = by_id["floor_pilot_dev"], by_id["throughput_dev"]
    # floor pilot: the five smallest panel models (params_b), T1 exact match and T3 forced choice on a 50-per-cell dev file
    assert floor["experiment"] == "pilot_exploratory"
    assert KRP.expand_models(floor, plan, models_cfg) == FLOOR_MODELS
    panel_groups = {g for g, meta in models_cfg["groups"].items() if meta.get("panel")}
    panel = [m for m in models_cfg["models"] if m["group"] in panel_groups and not KRP.is_api(m)]
    smallest = sorted(panel, key=lambda m: (float(m["params_b"]), m["name"]))[:5]
    assert {m["name"] for m in smallest} == set(FLOOR_MODELS), "the floor pilot runs the smallest panel models"
    assert floor["items"] == "pilot_t1t3_dev" and floor["tasks"] == ["T1", "T3"] and floor["variants"] == ["V1", "V2", "V3", "V4"]
    assert floor["arms"] == ["nfc"] and floor["paraphrases"] == ["p0"] and floor["shots"] == 3 and floor["hardware"] == "t4"
    assert floor["estimate"] == {"unit": "gpu_hours", "low": 1, "high": 3}
    purpose = floor["purpose"].lower()
    for needle in ("exact match", "t1 strict", "forced choice", "t3_pair_lp", "near-miss twin", "exploratory", "dev split only",
                   "never a paper table"):
        assert needle in purpose, needle
    spec = plan["item_files"]["pilot_t1t3_dev"]
    assert spec["derive"] == {"source": "noilai_dev", "seed": 20261012, "tasks": ["T1", "T3"], "per_cell": {"T1": 50, "T3": 50}}
    assert spec["expected_counts"]["n_items"] == 400 and spec["path"] == "data/release/v0.3/pilot_t1t3_dev.jsonl"
    # throughput: one model per backend/precision class, 300 T1 dev items
    assert thr["experiment"] == "throughput"
    assert KRP.expand_models(thr, plan, models_cfg) == THROUGHPUT_MODELS
    by_name = models_cfg["by_name"]
    classes = {
        "fp32 vLLM": lambda m: m["backend"] == "vllm" and m["dtype"] == "float32" and not m.get("quantization") and m["hardware"] == "t4",
        "fp16 vLLM": lambda m: m["backend"] == "vllm" and m["dtype"] == "float16" and not m.get("quantization") and m["hardware"] == "t4",
        "HF transformers": lambda m: m["backend"] == "hf",
        "4-bit": lambda m: (m.get("quantization") or {}).get("bits") == 4,
        "2xT4": lambda m: m["hardware"] == "2xt4",
    }
    for label, pred in classes.items():
        assert any(pred(by_name[n]) for n in THROUGHPUT_MODELS), f"no model of the {label} class on the throughput line"
    assert thr["items"] == "throughput_t1_300" and thr["tasks"] == ["T1"] and thr["arms"] == ["nfc"]
    assert thr["paraphrases"] == ["p0"] and thr["shots"] == 3 and thr["hardware"] == "mixed"
    assert thr["estimate"] == {"unit": "gpu_hours", "low": 1, "high": 2}
    purpose = thr["purpose"].lower()
    for needle in ("prefill", "decode", "tokens per second", "items per gpu-hour", "batched inference", "re-priced",
                   "configs/run_plan.yaml", "dd 8.5", "item 30", "exploratory", "dev split only", "never a paper table"):
        assert needle in purpose, needle
    spec = plan["item_files"]["throughput_t1_300"]
    assert spec["derive"] == {"source": "noilai_dev", "seed": 20261013, "tasks": ["T1"], "per_cell": {"T1": 75}}
    assert spec["expected_counts"] == {"n_items": 300, "per_cell": 75}
    # derived counts agree with the per-cell counts; the compute lines hold their runs
    for key, spec in plan["item_files"].items():
        d = spec.get("derive")
        if d and "per_cell" in d:
            assert sum(KRP._per_cell(d, t) * 4 for t in d["tasks"]) == spec["expected_counts"]["n_items"], key
    sums: dict = {}
    for run in plan["runs"]:
        est, line = run["estimate"], plan["plan_lines"][run["counts_toward"]]
        assert est["unit"] == line["unit"] and est["low"] <= est["high"], run["id"]
        s = sums.setdefault(run["counts_toward"], [0, 0])
        s[0] += est["low"]
        s[1] += est["high"]
    for name, (lo, hi) in sums.items():
        assert lo <= plan["plan_lines"][name]["low"] and hi <= plan["plan_lines"][name]["high"], (name, lo, hi)


def test_seeds_are_distinct_from_each_other_and_from_every_pre_registered_seed(plan, prereg):
    mine = [spec["derive"]["seed"] for spec in plan["item_files"].values() if spec.get("derive")]
    assert len(mine) == len(set(mine))
    theirs = {spec["derive"]["seed"] for spec in prereg["item_files"].values() if spec.get("derive")}
    new = {plan["item_files"][k]["derive"]["seed"] for k in NEW_DERIVED}
    assert new == {20261012, 20261013} and new.isdisjoint(theirs), "an exploratory draw never coincides with a pre-registered one"
    assert set(plan["item_files"]).isdisjoint(set(prereg["item_files"]) - {"noilai_dev", "xcopa_vi_test", "pilot_t1_200", "pilot_xcopa_200"})
    assert {r["id"] for r in prereg["runs"]} & EXPLORATORY_RUN_IDS == PILOT_RUN_IDS, "only the two pilots exist in both plans"


def test_the_gate_one_pilots_and_their_files_are_verbatim_copies_of_the_pre_registered_plan(plan, prereg):
    """So that the Gate 1 pilot (PREREGISTRATION section 10) can be launched from either plan: same files, same
    seeds, same flags; `exploratory` / `dev_only` are the only keys this plan adds."""
    mine = {r["id"]: r for r in plan["runs"]}
    theirs = {r["id"]: r for r in prereg["runs"]}
    for rid in PILOT_RUN_IDS:
        stripped = {k: v for k, v in mine[rid].items() if k not in ("exploratory", "dev_only")}
        assert stripped == theirs[rid], rid
    for key in ("noilai_dev", "xcopa_vi_test", "pilot_t1_200", "pilot_xcopa_200"):
        assert plan["item_files"][key] == prereg["item_files"][key], key
    assert plan["model_sets"]["pilot"] == prereg["model_sets"]["pilot"]
    assert plan["experiments"]["pilot"] == prereg["experiments"]["pilot"]
    assert plan["plan_lines"]["pilots_and_smoke"]["low"] == prereg["plan_lines"]["pilots_and_smoke"]["low"]
    assert plan["plan_lines"]["pilots_and_smoke"]["high"] == prereg["plan_lines"]["pilots_and_smoke"]["high"]
    for key in ("release", "release_manifest", "runs_root", "compute_log", "run_eval_cli", "run_eval_out_style", "api"):
        assert plan[key] == prereg[key], key


# ------------------------------------------------------------------ the driver's commands
def test_every_command_of_every_line_is_accepted_by_the_real_run_eval_cli(plan, models_cfg):
    """As tests/test_cloud.py does for the pre-registered plan: every emitted flag exists in run_eval.py --help,
    and no (run, model) of this plan is refused by a build_command guard (no API model, no canary file)."""
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_eval.py"), "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-500:]
    accepted = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", r.stdout))
    for run in plan["runs"]:
        for model in KRP.expand_models(run, plan, models_cfg):
            cmd = KRP.build_command(run, model, plan, models_cfg)          # raises on a guarded combination
            emitted = {tok for tok in cmd if tok.startswith("--")}
            assert emitted <= accepted, (run["id"], model, emitted - accepted)
            assert "--smoke" not in cmd and "--limit" not in cmd and "--in-core-only" not in cmd, (run["id"], model)
            assert "--resume" in cmd and cmd[cmd.index("--run-id") + 1] == f"{run['id']}__{model}"
            assert cmd[cmd.index("--items") + 1] == plan["item_files"][run["items"]]["path"]
            assert Path(cmd[1]) == SCRIPTS / "run_eval.py"
    floor = KRP.find_run(plan, "floor_pilot_dev")
    cmd = KRP.build_command(floor, "gemma-3-1b-it", plan, models_cfg)
    assert cmd[cmd.index("--tasks"):cmd.index("--variants")] == ["--tasks", "T1", "T3"]
    assert "data/release/v0.3/pilot_t1t3_dev.jsonl" in cmd


def test_run_plan_cli_lists_and_dry_runs_the_exploratory_plan_through_its_plan_flag():
    cli = [sys.executable, str(SCRIPTS / "kaggle_run_plan.py"), "--plan", "configs/run_plan_exploratory.yaml"]
    r = subprocess.run([*cli, "--list"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    lines = {ln.split()[0]: ln for ln in r.stdout.splitlines() if ln.strip()}
    assert set(lines) == EXPLORATORY_RUN_IDS
    assert "pilot_exploratory" in lines["floor_pilot_dev"] and "5 models" in lines["floor_pilot_dev"]
    assert "items=throughput_t1_300" in lines["throughput_dev"] and "phogpt-4b-chat" in lines["throughput_dev"]
    assert "3 models" in lines["pilot_t1_200"] and "items=pilot_xcopa_200" in lines["pilot_xcopa_200"]
    r = subprocess.run([*cli, "--run", "floor_pilot_dev", "--dry-run", "--session-hardware", "2xt4"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.count("[cmd  ]") == 5 and "--tasks T1 T3" in r.stdout and "pilot_t1t3_dev.jsonl" in r.stdout
    assert "--smoke" not in r.stdout and "[summary] 5 commands, 0 failed, 0 parked, 0 refused" in r.stdout
    # the 2xT4 model of the throughput line is skipped in a single-T4 session, the four others run
    r = subprocess.run([*cli, "--run", "throughput_dev", "--dry-run", "--session-hardware", "t4"],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert "gemma-3-4b-it: hardware 2xt4 is not runnable in a t4 session" in r.stdout and r.stdout.count("[cmd  ]") == 5
    # the default plan is untouched by the new file: the pre-registered listing carries no exploratory line
    r = subprocess.run([sys.executable, str(SCRIPTS / "kaggle_run_plan.py"), "--list"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "E1_main" in r.stdout and "floor_pilot_dev" not in r.stdout and "throughput_dev" not in r.stdout


# ------------------------------------------------------------------ materialization from a local dev split
@pytest.mark.skipif(_dev_source() is None, reason="no noilai_dev.jsonl to derive from (release not built, smoke build absent)")
def test_materialize_derives_the_two_dev_files_with_the_expected_cells(tmp_path):
    """`--materialize` with a tmp copy of the plan whose `release` points at a copy of the local dev split: the two
    new files get their per-cell counts, T3 stays in yes/no pairs, nothing vulgar, no canary, and the derivation is
    idempotent. The copy keeps the shared smoke build untouched and never writes under data/release/."""
    from noilai.eval.run import RunOptions, select_items
    from noilai.gen.generate import load_items

    import run_eval as RE

    src = _dev_source()
    release = tmp_path / "release"
    release.mkdir()
    shutil.copy2(src, release / "noilai_dev.jsonl")
    if (src.parent / "manifest.json").exists():
        shutil.copy2(src.parent / "manifest.json", release / "manifest.json")
    raw = yaml.safe_load(EXPLORATORY.read_text(encoding="utf-8"))
    raw["release"] = str(release)
    for key in ("noilai_dev", *NEW_DERIVED):
        raw["item_files"][key]["sha256"] = None             # a non-frozen source: the frozen hash would rightly refuse it
    tmp_plan = tmp_path / "run_plan_exploratory.yaml"
    tmp_plan.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tree_before = set((ROOT / "data" / "release").rglob("*")) | set((ROOT / "data" / "external").rglob("*"))
    cli = [sys.executable, str(SCRIPTS / "kaggle_run_plan.py"), "--plan", str(tmp_plan)]
    r = subprocess.run([*cli, "--materialize", *sorted(NEW_DERIVED)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    infos = {json.loads(ln)["key"]: json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")}
    assert set(infos) == NEW_DERIVED and all(i["status"] == "written" for i in infos.values())
    assert infos["pilot_t1t3_dev"]["n_items"] == 400 and infos["throughput_t1_300"]["n_items"] == 300
    assert infos["pilot_t1t3_dev"]["counts"] == {f"{t}-V{v}": 50 for t in ("T1", "T3") for v in (1, 2, 3, 4)}
    assert infos["throughput_t1_300"]["counts"] == {f"T1-V{v}": 75 for v in (1, 2, 3, 4)}
    plan = KRP.load_plan(tmp_plan)
    dev_ids = {it["item_id"] for it in load_items(release / "noilai_dev.jsonl")}
    for key in NEW_DERIVED:
        path = KRP.item_path(plan, key, ROOT)
        assert path.parent == release and path.name == f"{key}.jsonl"
        items = load_items(path)
        assert {it["item_id"] for it in items} <= dev_ids and all(it["split"] == "dev" for it in items), key
        assert all(not it.get("vulgar") and "canary" not in it for it in items), key
        assert items == sorted(items, key=lambda it: it["item_id"]), key
    pilot = load_items(release / "pilot_t1t3_dev.jsonl")
    ids = {it["item_id"] for it in pilot}
    t3 = [it for it in pilot if it["task"] == "T3"]
    assert len(t3) == 200 and all(it["pair_item_id"] in ids for it in t3), "T3 is drawn as yes/no pairs"
    assert Counter((it["variant"], it["gold"]) for it in t3) == {(f"V{v}", g): 25 for v in (1, 2, 3, 4) for g in ("yes", "no")}
    # the real CLI parses the driver's command and the real select_items keeps every cell of the line
    cfg = KRP.load_models()
    for rid, cells in (("floor_pilot_dev", {(t, f"V{v}") for t in ("T1", "T3") for v in (1, 2, 3, 4)}),
                       ("throughput_dev", {("T1", f"V{v}") for v in (1, 2, 3, 4)})):
        run = KRP.find_run(plan, rid)
        argv = KRP.build_command(run, KRP.expand_models(run, plan, cfg)[0], plan, cfg, project_root=ROOT)[2:]
        args = RE.build_parser().parse_args(argv)
        opts = RunOptions(tasks=tuple(args.tasks), variants=tuple(args.variants), limit=args.limit, in_core_only=args.in_core_only)
        sel = select_items(load_items(Path(args.items)), opts, "noilai")
        assert {(it["task"], it["variant"]) for it in sel} == cells, rid
        assert len(sel) == plan["item_files"][run["items"]]["expected_counts"]["n_items"], rid
        gate = KRP.verify_run_items(run, plan, ROOT)
        assert gate["ok"], gate["problems"]
    # deterministic: a second materialization finds the files, --force rewrites the same bytes
    again = {i["key"]: i for i in KRP.materialize(plan, ROOT, keys=sorted(NEW_DERIVED))}
    forced = {i["key"]: i for i in KRP.materialize(plan, ROOT, keys=sorted(NEW_DERIVED), force=True)}
    for key in NEW_DERIVED:
        assert again[key]["status"] == "exists" and again[key]["sha256"] == infos[key]["sha256"] == forced[key]["sha256"], key
    # the pilot files were not asked for, and nothing was written under the repository's data/release or data/external
    assert not (release / "pilot_t1_200.jsonl").exists()
    assert set((ROOT / "data" / "release").rglob("*")) | set((ROOT / "data" / "external").rglob("*")) == tree_before


# ------------------------------------------------------------------ the notebooks' PLAN_PATH parameter
def test_notebooks_take_the_plan_from_plan_path_and_pass_it_to_the_driver():
    """Every notebook's parameters cell names the plan; the gate cell (d) and every run cell load it with
    kaggle_run_plan.load_plan and hand it to the driver, so the pre-registered plan is never edited to run an
    exploratory line. The cells stay plain Python (tests/test_cloud.py checks magics and drift)."""
    for name in KBN.NOTEBOOKS:
        nb = nbformat.read(str(ROOT / "notebooks" / f"{name}.ipynb"), as_version=4)
        cells = [c.source for c in nb.cells if c.cell_type == "code"]
        assert 'PLAN_PATH = "configs/run_plan.yaml"' in cells[0], (name, "the parameters cell names the plan")
        gate = next(c for c in cells if "kaggle_verify_items.py" in c)
        assert "PLAN = KRP.load_plan(Path(PLAN_PATH))" in gate and "item_keys_for_runs(PLAN," in gate, name
        assert '"--plan", PLAN_PATH, "--key", key, "--materialize"' in gate, (name, "the gate verifies against the same plan")
        assert "KRP.load_plan()" not in "\n".join(cells), (name, "no cell falls back to the default plan")
        run_cells = [c.source for c in nb.cells if c.cell_type == "code" and KBN.RUN_TAG in c.metadata.get("tags", [])]
        for src in run_cells:
            n_exec = src.count("KRP.execute(")
            assert n_exec and src.count("plan=PLAN") == n_exec, (name, "every execute call carries the plan")
            assert "PLAN = KRP.load_plan(Path(PLAN_PATH))" in src or "PLAN, MODELS_CFG = KRP.load_plan(Path(PLAN_PATH))" in "\n".join(cells), name
    t4 = nbformat.read(str(ROOT / "notebooks" / "kaggle_eval_t4.ipynb"), as_version=4)
    params = next(c.source for c in t4.cells if c.cell_type == "code")
    assert "run_plan_exploratory.yaml" in params and "SMOKE_RUN_IDS = []" in params, "the T4 parameters say how to run the exploratory plan"
    tpu = nbformat.read(str(ROOT / "notebooks" / "kaggle_eval_tpu.ipynb"), as_version=4)
    assert "run_plan_exploratory.yaml" in next(c.source for c in tpu.cells if c.cell_type == "code")
    assert KBN.check() == [], "notebooks/*.ipynb must be regenerated from the builder"


def test_the_gate_cell_runs_against_the_exploratory_plan_without_the_network(tmp_path, monkeypatch):
    """Cell (d) executed as plain Python with PLAN_PATH set to the exploratory plan: the keys derive from the run ids
    and every verify call carries `--plan`; subprocess calls are intercepted (no fetch, no verification here)."""
    nb = nbformat.read(str(ROOT / "notebooks" / "kaggle_eval_t4.ipynb"), as_version=4)
    gate = next(c.source for c in nb.cells if c.cell_type == "code" and "kaggle_verify_items.py" in c.source)
    calls = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)) or _Done())
    g = {"PLAN_PATH": str(EXPLORATORY), "ITEMS_DATASET_DIR": None, "VERIFY_ITEM_KEYS": None, "RUN_ID": "floor_pilot_dev",
         "SMOKE_RUN_IDS": [], "PROJECT": tmp_path, "subprocess": subprocess}
    monkeypatch.chdir(ROOT)
    exec(compile(gate, "gate_cell", "exec"), g)
    assert g["KEYS"] == ["pilot_t1t3_dev"] and g["PLAN"]["release"] == "data/release/v0.3"
    verify_calls = [c for c in calls if "scripts/kaggle_verify_items.py" in c]
    assert verify_calls == [[sys.executable, "scripts/kaggle_verify_items.py", "--plan", str(EXPLORATORY), "--key", "pilot_t1t3_dev", "--materialize"]]
    # two run ids: two keys, in order, no duplicates
    g2 = dict(g, RUN_ID="throughput_dev", SMOKE_RUN_IDS=["floor_pilot_dev"], KEYS=None)
    calls.clear()
    exec(compile(gate, "gate_cell", "exec"), g2)
    assert g2["KEYS"] == ["throughput_t1_300", "pilot_t1t3_dev"]
    # an id that is not a line of the plan stops the cell with the plan's run ids in the message
    g3 = dict(g, RUN_ID="E1_main")
    with pytest.raises(KeyError, match="floor_pilot_dev"):
        exec(compile(gate, "gate_cell", "exec"), g3)


def test_the_pre_registered_plan_does_not_know_the_exploratory_one():
    """The exploratory plan is a separate file the author never has to merge: configs/run_plan.yaml carries neither
    its name nor any of its run ids or derived keys, and the exploratory file says which plan it supplements."""
    text = PREREGISTERED.read_text(encoding="utf-8")
    for needle in ("run_plan_exploratory", "floor_pilot_dev", "throughput_dev", "pilot_t1t3_dev", "throughput_t1_300"):
        assert needle not in text, needle
    head = EXPLORATORY.read_text(encoding="utf-8").split("version:")[0]
    assert "DEV-ONLY" in head and "NOT the pre-registered run matrix" in head and "--plan configs/run_plan_exploratory.yaml" in head
