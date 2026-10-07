"""The experiment ledger: one row per planned unit, its cut-order rank, cost and verified result.

    python scripts/ledger.py init                     # build experiments/ledger.csv from the plans (refuses to overwrite)
    python scripts/ledger.py init --force             # rebuild (statuses of an existing ledger are carried over by unit_id)
    python scripts/ledger.py ingest --runs data/runs  # update statuses/hashes from run directories (and experiments/human/)
    python scripts/ledger.py show [--status done] [--experiment E3] [--lane kaggle_gpu]
    python scripts/ledger.py reprice --compute-log data/compute_log.csv   # re-price estimates from measured hours

A UNIT is one (run line, model, arm) of configs/run_plan.yaml (plus the dev-only lines of
configs/run_plan_exploratory.yaml when that file exists), one (model) of the E4 probe line, or one
human-lane deliverable (a validator's sheets, a human-baseline form, the attested-set expansion).
The split and seed are recorded per unit (the backend seed of configs/models.yaml `defaults.seed`
for model runs; the derived file's sampling seed where the item file is a seeded derivation).

Cut-order rank (`cut_rank`, lower = do first) follows DESIGN_DECISIONS 7.1 and 8.8: prerequisites
(smoke, Gate 1 pilots) first; then the confirmatory lines that decide the pre-registered hypotheses
in the order E1 main / explicit-input main result (RQ1-2), E3 on NóiLái, XCOPA and the C2 set (H3,
H4), the API core, E4 (H5; reported regardless of outcome), the attested run (H6), the ablations,
the reasoning and bf16 sub-studies, the scope/engine agreement checks. Within a line the models are
ordered so that the MINIMUM VIABLE PANEL (DD 7.1: >= 8 open models over >= 4 tokenizer families with
a pass-through family beyond Gemma) completes first, the second model of each family next, and the
pre-registered cut-first models (Gemma 3 12B, Qwen3.5-0.8B, Gemma 4 12B: tokenizer-redundant) last.
On E3's `nfd`/`pc` arms a tokenizer that is expected to NORMALIZE (DD 7.2: Qwen, Sailor2) is
de-prioritized: its cell is "0 by construction" once the census confirms it.

Cost model (the weights are also stated at the top of scripts/progress.py): a plan line's estimate
midpoint (low+high)/2 in its own unit is split over the line's models in proportion to
w = (1 + params_b / 8) x (2 if dtype == float32 else 1), then equally over the line's arms; TPU-hardware
models of a `mixed` line are booked in TPU hours (E1_main's three TPU models take the `tpu_main`
line's estimate); API lines split their call count equally over models and arms; E4 splits 1B : 4B
as 2 : 8 (DD 9.5). `reprice` replaces the plan midpoints by measured hours per model class once
data/compute_log.csv has rows (DD 8.5, item 30).

Statuses: planned, gated (waits on experiments/gates.yaml), blocked (an external blocker, see
BLOCKED.md), launched, running, partial, done, failed, refused, cut. `hash_verified` is yes / no / n/a
(no reference hash: exploratory dev-only files before they are materialized). Only `done` units with
hash_verified != no count in scripts/progress.py.

Raw item-level outputs stay where DESIGN_DECISIONS 7.6 / 11.1 keep them (the private Kaggle runs
dataset, mirrored under the git-ignored data/runs/); this ledger, the per-run aggregates it copies
into experiments/aggregates/ and the hashes are what the repository commits.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.csv"
AGGREGATES = ROOT / "experiments" / "aggregates"
PLAN = ROOT / "configs" / "run_plan.yaml"
PLAN_EXPLORATORY = ROOT / "configs" / "run_plan_exploratory.yaml"
MODELS = ROOT / "configs" / "models.yaml"
GATES = ROOT / "experiments" / "gates.yaml"

COLUMNS = ["unit_id", "experiment", "run_id", "model", "arm", "split", "item_key", "seed", "paraphrases", "section",
           "kind", "lane", "cut_rank", "est_cost", "est_unit", "est_raw", "status", "gate", "hash_verified",
           "result_file", "result_sha256", "scores_sha256", "n_rows", "n_expected", "updated_utc", "notes"]

# ---- cost weights (mirrored in scripts/progress.py) -------------------------------------------------------
COST_PER_GPU_HOUR = 1.0
COST_PER_TPU_HOUR = 1.0
COST_PER_L4_HOUR = 1.0
COST_PER_1000_API_CALLS = 0.25
FP32_FACTOR = 2.0          # DD 7.2: fp32 throughput is the Gemma 3 risk; a T4 has no bf16 units
PARAMS_SCALE_B = 8.0       # w = 1 + params_b / 8: an 8B model costs twice a tiny one per item
E4_SPLIT = {"gemma-3-1b-it": 0.2, "gemma-3-4b-it": 0.8}   # DD 9.5: 1.5-2 GPU-h (1B) + 3-8 GPU-h (4B)

# ---- cut order ---------------------------------------------------------------------------------------------
# DD 7.1: minimum viable panel first (one model per tokenizer family; pass-through families beyond Gemma early
# because E3's informative set needs them: Llama 3.1, PhoGPT, Vistral), the Gate 1 pilot models early, the second
# model of each family next, the pre-registered cut-first models (tokenizer-redundant) last, API models after
# the self-hosted panel (their lines are gated on the Gemini decision anyway).
MODEL_RANK = {
    "gemma-3-1b-it": 1, "llama-3.1-8b-instruct": 2, "phogpt-4b-chat": 3, "vistral-7b-chat": 4, "qwen3.5-2b": 5,
    "gemma-4-e2b": 6, "sailor2-8b-chat": 7, "gemma-sea-lion-v4.5-e2b-it": 8,
    "gemma-3-4b-it": 9, "qwen3.5-4b": 10, "gemma-4-e4b": 11, "qwen3.5-9b": 12, "qwen3.8-27b": 13,
    "qwen-sea-lion-v4.5-27b-it": 14,
    "gemma-3-12b-it": 15, "qwen3.5-0.8b": 16, "gemma-4-12b": 17,          # DD 7.1 cut-first, in the DD's order
    "gpt-oss-20b": 18, "gpt-oss-120b": 19, "gemini-flash-lite": 20, "gemini-flash": 21,
}
RUN_TIER = {
    "smoke_20": 0, "smoke_20_tpu": 0,
    "pilot_t1_200": 1, "pilot_xcopa_200": 1, "floor_pilot_dev": 1, "throughput_dev": 1,
    "E1_main": 10, "tpu_main": 10, "E1_explicit_input": 12,
    "E3_noilai": 20, "E3_c2_enriched": 21, "E3_xcopa": 22,    # the C2 line is cheap and decides H4: before XCOPA
    "E1_api_core": 30, "E1_api_paraphrase": 32, "E3_api_core": 33,
    "E4_probe": 40,
    "E1_attested": 45,
    "E1_abl_zero_shot": 50, "E1_abl_name_only": 51, "E1_abl_english": 52, "E1_abl_spaced": 53,
    "reasoning_500": 60, "reasoning_500_tpu": 60, "reasoning_500_api": 61, "bf16_drift_200": 62,
    "E3_scope_item": 64, "E3_engine_hf": 65,
}
SECTION_OF_EXPERIMENT = {"E1": "Results", "E1_ablation": "Results", "reasoning": "Results", "bf16_drift": "Results",
                         "E3": "Counterfactuals", "E4": "Tone", "pilot": "Prerequisite", "smoke": "Prerequisite",
                         "pilot_exploratory": "Exploratory", "throughput": "Exploratory"}
# DD 7.2 census expectations: which tokenizer families are expected to normalize NFD/PC to NFC (C1 = 0 by construction)
NORMALIZING_FAMILIES = {"qwen3.5", "qwen3.8", "sailor2"}
NORMALIZING_ARMS = {"nfd", "pc"}
PREREQUISITE_EXPERIMENTS = {"smoke", "pilot"}
EXPLORATORY_EXPERIMENTS = {"pilot", "pilot_exploratory", "throughput", "smoke"}

HUMAN_UNITS = [  # person-hour estimates: DD 10.1 (6-10 h per validator), 10.2 (~25 min per respondent), 13.6 (author sourcing)
    ("validation__A", "human_validation", "validator A: 1,000-item sample share + 200-item overlap", 8.0),
    ("validation__B", "human_validation", "validator B", 8.0),
    ("validation__C", "human_validation", "validator C", 8.0),
] + [(f"baseline__form{i:02d}", "human_baseline", f"human-baseline form {i} of 20 (30 items, ~25 min)", 0.42) for i in range(1, 21)] + [
    ("attested__expansion", "data", "attested set to >= 100 native-verified exact rows from >= 3 collections (H6 floor)", 12.0),
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(path: Path = MODELS) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {e["name"]: e for e in cfg["models"]}


def load_plan(path: Path) -> dict:
    import kaggle_verify_items as KVI
    return KVI.load_plan(path)


def expand_models(run: dict, plan: dict, models: dict) -> list[str]:
    import kaggle_run_plan as KRP
    cfg = {"by_name": models, "models": list(models.values())}
    return list(KRP.expand_models(run, plan, cfg))


def model_weight(entry: dict) -> float:
    params = float(entry.get("params_b") or 1.0)
    w = 1.0 + params / PARAMS_SCALE_B
    if (entry.get("dtype") or "") == "float32":
        w *= FP32_FACTOR
    return w


def split_of(item_key: str, spec: dict) -> str:
    if spec.get("kind") == "xcopa" or "xcopa" in item_key:
        return "xcopa"
    if item_key == "attested":
        return "attested"
    if item_key in ("noilai_dev",) or item_key.startswith("pilot_") or item_key.endswith("_dev"):
        return "dev"
    if item_key == "noilai_core" or (spec.get("derive") or {}).get("source") == "noilai_core" or item_key.startswith("noilai_api"):
        return "core"
    return "test"


def lane_of(entry: dict, run: dict) -> str:
    hw = entry.get("hardware")
    if run.get("platform") == "api" or hw == "api":
        return "api"
    if hw == "tpu":
        return "kaggle_tpu"
    if hw == "l4":
        return "modal_l4"
    return "kaggle_gpu"


def unit_rows_for_run(run: dict, plan: dict, models: dict, exploratory: bool, tpu_main: dict | None) -> list[dict]:
    rid = run["id"]
    exp = run["experiment"]
    if run.get("not_a_run_eval_line"):          # E4
        est = run["estimate"]
        mid = (est["low"] + est["high"]) / 2
        rows = []
        for name in run["models"]:
            share = E4_SPLIT.get(name, 1.0 / len(run["models"]))
            rows.append({"unit_id": f"{rid}__{name}__{'+'.join(run['arms'])}", "experiment": exp, "run_id": rid, "model": name,
                         "arm": "+".join(run["arms"]), "split": "inventory", "item_key": "syllable_inventory",
                         "seed": "syllable_seed=0;pair_seed=0", "paraphrases": "", "section": "Tone",
                         "kind": "confirmatory", "lane": "kaggle_gpu", "cut_rank": RUN_TIER[rid] * 100 + MODEL_RANK.get(name, 50),
                         "est_cost": round(mid * share * COST_PER_GPU_HOUR, 3), "est_unit": "gpu_hours",
                         "est_raw": round(mid * share, 3), "status": "planned", "gate": "prereg",
                         "hash_verified": "n/a", "notes": "DD 8.8: reported regardless of outcome; noilai.probe, not run_eval"})
        return rows
    if rid == "tpu_main":                        # alias of E1_main for the TPU trio (the plan says so)
        return []
    names = run["models"] if isinstance(run["models"], list) else expand_models(run, plan, models)
    item_key = run["items"]
    spec = plan["item_files"][item_key]
    arms = list(run.get("arms") or ["nfc"])
    est = run["estimate"]
    unit = est["unit"]
    mid = (est["low"] + est["high"]) / 2
    entries = {n: models[n] for n in names}
    # TPU models of a mixed line are booked in TPU hours; for E1_main the tpu_main line carries their estimate
    gpu_names = [n for n in names if entries[n].get("hardware") != "tpu"] if unit == "gpu_hours" else names
    tpu_names = [n for n in names if entries[n].get("hardware") == "tpu"] if unit == "gpu_hours" else []
    w_gpu = {n: model_weight(entries[n]) for n in gpu_names}
    w_sum = sum(w_gpu.values()) or 1.0
    tpu_mid = None
    if tpu_names:
        if rid == "E1_main" and tpu_main is not None:
            t = tpu_main["estimate"]
            tpu_mid = (t["low"] + t["high"]) / 2
        else:                                   # a mixed line: the TPU trio's share of its own estimate, in TPU hours
            w_all = {n: model_weight(entries[n]) for n in names}
            tpu_mid = mid * sum(w_all[n] for n in tpu_names) / sum(w_all.values())
            w_sum = sum(w_all[n] for n in gpu_names) or 1.0
            mid = mid * w_sum / sum(w_all.values())
    rows = []
    seed = f"backend_seed={models[names[0]].get('seed', 20261203) if names else 20261203}"
    if spec.get("derive"):
        seed += f";sample_seed={spec['derive'].get('seed')}"
    for name in names:
        entry = entries[name]
        lane = lane_of(entry, run)
        for arm in arms:
            if unit == "api_calls":
                calls = mid / len(names) / len(arms)
                est_raw, est_unit, cost = round(calls, 1), "api_calls", round(calls / 1000 * COST_PER_1000_API_CALLS, 4)
            elif name in tpu_names:
                w_t = {n: model_weight(entries[n]) for n in tpu_names}
                hours = tpu_mid * w_t[name] / sum(w_t.values()) / len(arms)
                est_raw, est_unit, cost = round(hours, 3), "tpu_hours", round(hours * COST_PER_TPU_HOUR, 4)
            elif unit == "tpu_hours":
                w_t = {n: model_weight(entries[n]) for n in names}
                hours = mid * w_t[name] / sum(w_t.values()) / len(arms)
                est_raw, est_unit, cost = round(hours, 3), "tpu_hours", round(hours * COST_PER_TPU_HOUR, 4)
            else:
                hours = mid * w_gpu[name] / w_sum / len(arms)
                est_raw, est_unit, cost = round(hours, 3), "gpu_hours", round(hours * COST_PER_GPU_HOUR, 4)
            rank = RUN_TIER.get(rid, 70) * 100 + MODEL_RANK.get(name, 50)
            notes = []
            if arm in NORMALIZING_ARMS and entry.get("family") in NORMALIZING_FAMILIES:
                rank += 50
                notes.append("DD 7.2: tokenizer expected to normalize; this cell is 0 by construction once the census confirms it")
            if rid == "E1_main" and name in tpu_names:
                notes.append("booked under tpu_main (identical command; --resume makes the other a no-op)")
            kind = "exploratory" if (exploratory or exp in EXPLORATORY_EXPERIMENTS or run.get("exploratory")) else "confirmatory"
            if exp in PREREQUISITE_EXPERIMENTS:
                kind = "prerequisite"
            gate = "none"
            if kind == "confirmatory":
                gate = "prereg+qu+iy" + ("+gemini" if lane == "api" else "")
            rows.append({"unit_id": f"{rid}__{name}__{arm}", "experiment": exp, "run_id": rid, "model": name, "arm": arm,
                         "split": split_of(item_key, spec), "item_key": item_key, "seed": seed,
                         "paraphrases": "|".join(run.get("paraphrases") or []), "section": SECTION_OF_EXPERIMENT.get(exp, "Other"),
                         "kind": kind, "lane": lane, "cut_rank": rank, "est_cost": cost, "est_unit": est_unit, "est_raw": est_raw,
                         "status": "planned", "gate": gate, "hash_verified": "n/a" if spec.get("sha256") is None else "no",
                         "notes": "; ".join(notes)})
    return rows


def human_rows() -> list[dict]:
    rows = []
    for uid, exp, note, hours in HUMAN_UNITS:
        item_key = "validation_sample" if exp == "human_validation" else ("noilai_main" if exp == "human_baseline" else "attested_seed")
        rows.append({"unit_id": uid, "experiment": exp, "run_id": uid.split("__")[0], "model": "", "arm": "", "split": "human",
                     "item_key": item_key, "seed": "", "paraphrases": "", "section": "Human", "kind": "confirmatory", "lane": "human",
                     "cut_rank": {"human_validation": 500, "human_baseline": 600, "data": 550}[exp],
                     "est_cost": hours, "est_unit": "person_hours", "est_raw": hours, "status": "planned", "gate": "none",
                     "hash_verified": "n/a", "notes": note})
    return rows


def build_rows(plan_path: Path = PLAN, exploratory_path: Path = PLAN_EXPLORATORY, models_path: Path = MODELS) -> list[dict]:
    models = load_models(models_path)
    plan = load_plan(plan_path)
    runs = {r["id"]: r for r in plan["runs"]}
    rows: list[dict] = []
    for run in plan["runs"]:
        rows += unit_rows_for_run(run, plan, models, exploratory=False, tpu_main=runs.get("tpu_main"))
    if exploratory_path.exists():
        xp = load_plan(exploratory_path)
        seen = {r["run_id"] for r in rows}
        for run in xp["runs"]:
            if run["id"] in seen:                 # pilot lines copied verbatim into the exploratory plan: one unit each
                continue
            rows += unit_rows_for_run(run, xp, models, exploratory=True, tpu_main=None)
    rows += human_rows()
    now = utc_now()
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")
        r["updated_utc"] = now
    rows.sort(key=lambda r: (int(r["cut_rank"]), r["unit_id"]))
    return rows


def read_ledger(path: Path = LEDGER) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_ledger(rows: list[dict], path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def cmd_init(args) -> int:
    path = Path(args.ledger)
    carry: dict[str, dict] = {}
    if path.exists():
        if not args.force:
            print(f"{path} exists; refusing to rebuild (use --force; statuses are carried over by unit_id)")
            return 1
        carry = {r["unit_id"]: r for r in read_ledger(path)}
    rows = build_rows(Path(args.plan), Path(args.exploratory), Path(args.models))
    carried = 0
    for r in rows:
        old = carry.get(r["unit_id"])
        if old:
            for c in ("status", "hash_verified", "result_file", "result_sha256", "scores_sha256", "n_rows", "n_expected", "updated_utc"):
                r[c] = old.get(c, r[c])
            if old.get("notes") and old["notes"] not in r["notes"]:
                r["notes"] = (r["notes"] + "; " if r["notes"] else "") + old["notes"]
            carried += 1
    write_ledger(rows, path)
    cost = sum(float(r["est_cost"]) for r in rows if r["lane"] != "human")
    person_hours = sum(float(r["est_raw"]) for r in rows if r["lane"] == "human")
    print(json.dumps({"ledger": str(path), "units": len(rows), "carried_over": carried,
                      "compute_cost_units": round(cost, 2), "human_person_hours": round(person_hours, 1)}))
    return 0


# ---- ingest -------------------------------------------------------------------------------------------------

def plan_sha_for(plan: dict, item_key: str) -> str | None:
    spec = (plan.get("item_files") or {}).get(item_key) or {}
    return spec.get("sha256")


def ingest_run_dir(run_dir: Path, rows_by_unit: dict[str, dict], plans: list[dict], aggregates: Path) -> list[str]:
    """Update the ledger rows of one run directory (data/runs/<run_id>__<model>/). Returns the unit ids touched."""
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return []
    m = json.loads(mpath.read_text(encoding="utf-8"))
    run_id = (m.get("identity") or {}).get("run_id") or m.get("run_id") or run_dir.name
    if "__" not in run_id:
        return []
    line_id, model = run_id.split("__", 1)
    opts = m.get("options") or {}
    arms = list(opts.get("arms") or ["nfc"])
    paraphrases = list(opts.get("paraphrases") or ["p0"])
    data = m.get("data") or {}
    item_file = m.get("item_file") if isinstance(m.get("item_file"), dict) else {}
    observed_sha = data.get("item_file_sha256") or item_file.get("sha256")
    n_selected = item_file.get("n_selected") or data.get("n_items") or 0
    rows_per_arm: dict[str, int] = dict.fromkeys(arms, 0)
    outputs = run_dir / "outputs.jsonl"
    if outputs.exists():
        with open(outputs, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows_per_arm[d.get("arm", "nfc")] = rows_per_arm.get(d.get("arm", "nfc"), 0) + 1
    unchanged_per_arm: dict[str, int] = {}
    if (run_dir / "unchanged.jsonl").exists():
        with open(run_dir / "unchanged.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                unchanged_per_arm[d.get("arm", "")] = unchanged_per_arm.get(d.get("arm", ""), 0) + 1
    out_sha = sha256_file(outputs) if outputs.exists() else ""
    scores = run_dir / "scores.jsonl"
    scores_sha = sha256_file(scores) if scores.exists() else ""
    finished = (m.get("status") == "finished")
    touched = []
    for arm in arms:
        uid = f"{line_id}__{model}__{arm}"
        row = rows_by_unit.get(uid)
        if row is None:
            continue
        expected = int(n_selected) * len(paraphrases) - unchanged_per_arm.get(arm, 0)
        n = rows_per_arm.get(arm, 0)
        ref = None                                   # the primary plan's hash wins; a later plan fills in only a missing one
        for p in plans:
            if ref is None:
                ref = plan_sha_for(p, row.get("item_key", ""))
        if ref is None:
            hv = "n/a"
        else:
            hv = "yes" if observed_sha == ref else "no"
        if finished and n >= expected > 0:
            status = "done"
        elif n > 0:
            status = "partial"
        else:
            status = "launched"
        if row.get("status") == "cut":
            status = "cut"
        row.update(status=status, hash_verified=hv, result_file=str(outputs.relative_to(ROOT)) if outputs.is_relative_to(ROOT) else str(outputs),
                   result_sha256=out_sha, scores_sha256=scores_sha, n_rows=n, n_expected=expected, updated_utc=utc_now())
        if hv == "no" and ref is not None:
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"ITEM FILE HASH MISMATCH: run read {observed_sha}, plan records {ref}"
        touched.append(uid)
    # aggregates only (never item-level rows) are mirrored into the repository
    summary = run_dir / "summary.json"
    if summary.exists() and touched:
        dest = aggregates / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_bytes(summary.read_bytes())
        keep = {k: m.get(k) for k in ("identity", "data", "options", "outcome", "status", "gpu_hours", "tpu_hours", "wall_s",
                                      "git_commit", "git_dirty", "code_sha256", "engine", "backend", "canary_check")}
        (dest / "manifest_excerpt.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    return touched


def ingest_probe_dirs(runs_root: Path, rows_by_unit: dict[str, dict]) -> list[str]:
    """E4: data/runs/<dir>/probes.csv + excess.csv (+ patching*.json) written by scripts/run_probe.py."""
    touched = []
    for d in runs_root.glob("*"):
        if not d.is_dir() or not (d / "probes.csv").exists():
            continue
        meta = {}
        if (d / "manifest.json").exists():
            try:
                meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        model = meta.get("model_key") or meta.get("model") or ""
        for uid, row in rows_by_unit.items():
            if row.get("experiment") != "E4" or (model and row.get("model") != model) or (not model and row["model"] not in d.name):
                continue
            patch = any(d.glob("patching*.json"))
            row.update(status="done" if patch else "partial", result_file=str(d / "probes.csv"),
                       result_sha256=sha256_file(d / "probes.csv"), updated_utc=utc_now(), hash_verified="n/a")
            touched.append(uid)
    return touched


def ingest_human(rows_by_unit: dict[str, dict], human_dir: Path) -> list[str]:
    touched = []
    rep = human_dir / "validation_report.json"
    if rep.exists():
        data = json.loads(rep.read_text(encoding="utf-8"))
        coders = set(data.get("coders") or [])
        for c in "ABC":
            uid = f"validation__{c}"
            if uid in rows_by_unit and (c in coders or data.get("coders") is None):
                rows_by_unit[uid].update(status="done", result_file=str(rep), result_sha256=sha256_file(rep), updated_utc=utc_now())
                touched.append(uid)
    bs = human_dir / "baseline_scores.jsonl"
    if bs.exists():
        forms = set()
        with open(bs, encoding="utf-8") as f:
            for line in f:
                try:
                    forms.add(str(json.loads(line).get("form")))
                except json.JSONDecodeError:
                    pass
        for i in range(1, 21):
            uid = f"baseline__form{i:02d}"
            if uid in rows_by_unit and (str(i) in forms or f"{i:02d}" in forms or f"baseline_form_{i:02d}" in forms):
                rows_by_unit[uid].update(status="done", result_file=str(bs), result_sha256=sha256_file(bs), updated_utc=utc_now())
                touched.append(uid)
    return touched


def ingest_attested(rows_by_unit: dict[str, dict], release: Path) -> list[str]:
    uid = "attested__expansion"
    row = rows_by_unit.get(uid)
    if row is None:
        return []
    seed = ROOT / "data" / "attested_seed.tsv"
    n_verified_exact = 0
    n = 0
    if seed.exists():
        with open(seed, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                n += 1
                if (r.get("exactness") or "").strip() == "exact" and (r.get("verified_by") or "").strip():
                    n_verified_exact += 1
    row.update(n_rows=n_verified_exact, n_expected=100, status="done" if n_verified_exact >= 100 else ("partial" if n_verified_exact else "planned"),
               updated_utc=utc_now(), notes=f"{row['notes'].split(' | ')[0]} | seed rows {n}, native-verified exact rows {n_verified_exact} (floor 100; PREREG section 4)")
    return [uid]


def cmd_ingest(args) -> int:
    path = Path(args.ledger)
    rows = read_ledger(path)
    by_unit = {r["unit_id"]: r for r in rows}
    plans = [load_plan(Path(args.plan))]
    if Path(args.exploratory).exists():
        plans.append(load_plan(Path(args.exploratory)))
    touched: list[str] = []
    runs_root = Path(args.runs)
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            if d.is_dir():
                touched += ingest_run_dir(d, by_unit, plans, Path(args.aggregates))
        touched += ingest_probe_dirs(runs_root, by_unit)
    touched += ingest_human(by_unit, Path(args.human))
    touched += ingest_attested(by_unit, ROOT / "data" / "release" / "v0.3")
    write_ledger(rows, path)
    done = [u for u in touched if by_unit[u]["status"] == "done"]
    bad = [u for u in touched if by_unit[u]["hash_verified"] == "no"]
    print(json.dumps({"touched": len(set(touched)), "done": len(set(done)), "hash_mismatch": sorted(set(bad)), "ledger": str(path)}))
    return 1 if bad else 0


def cmd_show(args) -> int:
    rows = read_ledger(Path(args.ledger))
    for r in rows:
        if args.status and r["status"] != args.status:
            continue
        if args.experiment and r["experiment"] != args.experiment:
            continue
        if args.lane and r["lane"] != args.lane:
            continue
        print(f"{r['cut_rank']:>5} {r['status']:<9} {r['hash_verified']:<4} {r['lane']:<11} {r['est_cost']:>7} {r['est_unit']:<12} {r['unit_id']}")
    return 0


def cmd_reprice(args) -> int:
    """Replace plan estimates by measured hours: per (run_id, model) from data/compute_log.csv (DD 8.5, item 30)."""
    import compute_log as CL
    rows = read_ledger(Path(args.ledger))
    entries = CL.read_entries(Path(args.compute_log)) if Path(args.compute_log).exists() else []
    measured: dict[str, float] = {}
    for e in entries:
        rid = getattr(e, "run_id", None) or (e.get("run_id") if isinstance(e, dict) else None)
        hours = float(getattr(e, "hours", 0) or (e.get("hours") if isinstance(e, dict) else 0) or 0)
        if rid and hours > 0:
            measured[rid] = measured.get(rid, 0.0) + hours
    n = 0
    for r in rows:
        key = f"{r['run_id']}__{r['model']}"
        if key in measured and r["est_unit"] in ("gpu_hours", "tpu_hours"):
            arms = max(1, sum(1 for x in rows if x["run_id"] == r["run_id"] and x["model"] == r["model"]))
            r["est_raw"] = round(measured[key] / arms, 3)
            r["est_cost"] = round(float(r["est_raw"]) * (COST_PER_GPU_HOUR if r["est_unit"] == "gpu_hours" else COST_PER_TPU_HOUR), 4)
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"repriced from measured {measured[key]:.2f} h on {dt.datetime.now(dt.timezone.utc).date().isoformat()}"
            n += 1
    write_ledger(rows, Path(args.ledger))
    print(json.dumps({"repriced_units": n, "measured_runs": len(measured)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--exploratory", default=str(PLAN_EXPLORATORY))
    ap.add_argument("--models", default=str(MODELS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)
    g = sub.add_parser("ingest")
    g.add_argument("--runs", default=str(ROOT / "data" / "runs"))
    g.add_argument("--human", default=str(ROOT / "experiments" / "human"))
    g.add_argument("--aggregates", default=str(AGGREGATES))
    g.set_defaults(func=cmd_ingest)
    s = sub.add_parser("show")
    s.add_argument("--status")
    s.add_argument("--experiment")
    s.add_argument("--lane")
    s.set_defaults(func=cmd_show)
    r = sub.add_parser("reprice")
    r.add_argument("--compute-log", default=str(ROOT / "data" / "compute_log.csv"))
    r.set_defaults(func=cmd_reprice)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
