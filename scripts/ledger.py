"""The experiment ledger: one row per planned unit, its compute chunk, cut-order rank, cost and verified result.

    python scripts/ledger.py init                     # build experiments/ledger.csv from the plans (refuses to overwrite)
    python scripts/ledger.py init --force             # rebuild (observed statuses of an existing ledger carry over by unit_id)
    python scripts/ledger.py ingest --runs data/runs  # statuses/hashes from run directories, the human reports, the attested seed
    python scripts/ledger.py show [--status done] [--experiment E3] [--lane kaggle_gpu] [--queue gpu] [--chunk c08_gpu_t1]
    python scripts/ledger.py reprice --compute-log data/compute_log.csv   # measured hours per run directory (DD 8.5)

THE RUN UNITS ARE THE 31 COMPUTE CHUNKS of configs/compute_chunks.yaml (scripts/plan_chunks.py, docs/COMPUTE_PLAN.md):
one chunk = one Kaggle session = an ordered list of jobs {run, models, overrides, tag}. A ledger UNIT is one
(run line, model, arm[, tag]) of a chunk job, unit_id `run__model__arm[__tag]`; the tag (`cpu`, `p0p1`, `p2`) is
the suffix of the run directory `data/runs/<run>__<model>[__<tag>]` (kaggle_run_plan.narrow_run). A paraphrase split
(DD 8.5) is two units whose `paraphrases` columns and n_expected add up to the line's; the TPU trio's E1 units carry
run_id `tpu_main` (the directory name) and are pooled under E1_main by the analyses; the CPU pilot units (`__cpu`) are
distinct from the T4 ones (a different engine, docs/COMPUTE_PLAN.md section 2). Lines no chunk runs are units too:
the API lines (notebooks/api_runs.ipynb, daily caps), E4 (notebooks/colab_probe_gemma3.ipynb, 23 Nov-6 Dec), the
unscheduled job (bf16_drift_200 x phogpt-4b-chat--bf16, memo N9), the dev-only exploratory lines of
configs/run_plan_exploratory.yaml (not in the chunk plan), and the human-lane deliverables (a validator's packet by
region, a human-baseline form, the attested-set expansion). Chunk ids renumber when the census moves E3 jobs
(plan_chunks.py), so rows are KEYED by unit_id, which stays stable; `init --force` re-derives chunk columns and carries
observed statuses over.

Order. `chunk_order` (the chunk's position, tier-major, the reverse of the pre-registered cut order) is the LAUNCH
order: the next chunk per queue is the lowest order with a unit still to do (scripts/progress.py). `cut_rank` stays
the ANALYSIS priority of DESIGN_DECISIONS 7.1 / 8.8 (prerequisites, then E1 main / explicit input, E3, API, E4,
attested, ablations, reasoning / bf16, scope / engine; within a line the minimum viable panel first, the second model
of each family next, the DD 7.1 cut-first models last; +50 on E3 nfd/pc arms of tokenizers expected to normalize).

Cost model (also stated at the top of scripts/progress.py): a unit's hours are its chunk job's hours (the plan line's
estimate split EQUALLY over the line's models, plan_chunks.share; the CPU chunks from the RL-2026-10-01-06 benchmark;
p0p1 : p2 = 2/3 : 1/3), midpoint, divided by the job's arms; est_cost = hours x COST_PER_{GPU,TPU,L4}_HOUR, and for
CPU hours x the weight experiments/quota.yaml `cpu_sessions_count_against_gpu_quota` gives (null or false -> 0: CPU
sessions cost no GPU quota and the meter shows CPU hours on their own line; true -> 1). API lines split their calls
equally over models and arms (0.25 per 1,000); E4 splits 1B : 4B as 2 : 8 (DD 9.5); human units in person-hours.
`reprice` replaces the estimates by measured hours per run directory once data/compute_log.csv has rows (DD 8.5).

Statuses: planned, gated, blocked, launched, running, partial, done, failed, refused, excluded (a human deliverable
returned but excluded by a pre-registered rule), cut (DD 7.1: named in the paper), zero_by_construction (DD 6.2: the
census makes the arm identical to nfc; never run), unscheduled (no route in the chunk plan), alternative (the T4
pilot chunk c05 duplicates the CPU pilot chunks: counted like cut until launched; whichever route finishes marks the
other cut). cut / zero_by_construction / unscheduled / alternative leave the meter's denominator. `hash_verified` is
yes / no / n/a. Only `done` units with hash_verified != no count in scripts/progress.py.

Raw item-level outputs stay where DESIGN_DECISIONS 7.6 / 11.1 keep them (the private Kaggle runs dataset, mirrored
under the git-ignored data/runs/); this ledger, the per-run aggregates it copies into experiments/aggregates/
(summary.json, manifest_excerpt.json, stats.json, results_hashes.json; never outputs or scores rows) and the hashes
are what the repository commits.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
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
QUOTA = ROOT / "experiments" / "quota.yaml"
HUMAN_DIR = ROOT / "experiments" / "human"                 # make ingest-sheets: human_baseline_report.json (aggregates)
VALIDATION_DIR = ROOT / "data" / "validation"             # the 1 Oct packet (git-ignored): report/validation_report.json
ATTESTED_SEED = ROOT / "data" / "attested_seed.tsv"

COLUMNS = ["unit_id", "experiment", "run_id", "model", "arm", "split", "item_key", "seed", "paraphrases", "section",
           "kind", "lane", "chunk_id", "chunk_order", "tier", "queue", "window", "tag", "chunk_hours_low",
           "chunk_hours_high", "notebook", "cut_rank", "est_cost", "est_unit", "est_raw", "status", "gate",
           "hash_verified", "result_file", "result_sha256", "scores_sha256", "results_sha256", "n_rows", "n_expected",
           "updated_utc", "notes"]
STATUSES = ("planned", "gated", "blocked", "launched", "running", "partial", "done", "failed", "refused", "excluded",
            "cut", "zero_by_construction", "unscheduled", "alternative")
NON_LIVE_STATUSES = frozenset({"cut", "zero_by_construction", "unscheduled", "alternative"})   # leave the denominator
PLAN_DERIVED_STATUSES = frozenset({"planned", "zero_by_construction", "unscheduled", "alternative"})  # re-derived by init
OBSERVED_COLUMNS = ("status", "hash_verified", "result_file", "result_sha256", "scores_sha256", "results_sha256",
                    "n_rows", "n_expected", "updated_utc")
REPRICED_MARK = "repriced from measured"      # the note `reprice` leaves; init --force then keeps est_raw / est_cost too

# ---- cost weights (mirrored in scripts/progress.py) -------------------------------------------------------
COST_PER_GPU_HOUR = 1.0
COST_PER_TPU_HOUR = 1.0
COST_PER_L4_HOUR = 1.0
COST_PER_1000_API_CALLS = 0.25
E4_SPLIT = {"gemma-3-1b-it": 0.2, "gemma-3-4b-it": 0.8}   # DD 9.5: 1.5-2 GPU-h (1B) + 3-8 GPU-h (4B)
UNIT_OF_QUEUE = {"cpu": "cpu_hours", "gpu": "gpu_hours", "tpu": "tpu_hours"}
LANE_OF_QUEUE = {"cpu": "kaggle_cpu", "gpu": "kaggle_gpu", "tpu": "kaggle_tpu", "api": "api", "human": "human"}
E4_WINDOW = "23 Nov-6 Dec (after the results freeze)"
EXPLORATORY_NOTE = ("not in the chunk plan: floor risk through c02-c04 forced choice (memo N4/N11), re-pricing through "
                    "the c06/c07 smoke week (DD 8.5)")
EXPLORATORY_NOTEBOOK = "notebooks/kaggle_eval_t4.ipynb (PLAN_PATH=configs/run_plan_exploratory.yaml)"

# ---- cut order (analysis priority) ---------------------------------------------------------------------------
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
RUN_ALIAS = {"tpu_main": "E1_main"}         # the TPU trio's E1 run directories; the analyses pool them under E1_main
THINKING_EXPERIMENTS = {"reasoning"}        # the only units for which `finished_thinking_present` counts as finished

# ---- human lane (the 1 Oct packet) ---------------------------------------------------------------------------
VALIDATION_REGIONS = ("N", "C", "S")        # memo row 7: one validator per region (North, Central, South)
VALIDATION_HOURS = 5.4                      # memo N1: ~5.4 person-hours each with three validators (6.9 with two)
N_BASELINE_FORMS = 20                       # docs/gate1/HUMAN_BASELINE_PROTOCOL.md: forms 01-20, 30 model items each
BASELINE_FORM_HOURS = 0.6                   # docs/HUMAN_BASELINE_FORM.md section 5: 30-45 minutes per respondent
ATTESTED_FLOOR = 100                        # PREREGISTRATION section 4 / memo N6: H6 tested from 100 verified exact rows
HUMAN_UNITS = [
    *[(f"validation__{reg}", "human_validation",
       (f"validator of region {reg}: the 1 Oct packet, Parts A-E (memo 7 / N1: ~{VALIDATION_HOURS} person-hours with three "
        "validators, 6.9 with two)"), VALIDATION_HOURS) for reg in VALIDATION_REGIONS],
    *[(f"baseline__form{i:02d}", "human_baseline",
       (f"human-baseline form {i:02d} of {N_BASELINE_FORMS} (30 model items + natural block; 30-45 min, HUMAN_BASELINE_FORM "
        "section 5)"), BASELINE_FORM_HOURS) for i in range(1, N_BASELINE_FORMS + 1)],
    ("attested__expansion", "data",
     f"attested set to >= {ATTESTED_FLOOR} native-verified exact rows from >= 3 collections (H6 floor; memo 6 / N5 / N6)", 12.0),
]
REGION_OF = {"n": "N", "north": "N", "northern": "N", "bac": "N", "bắc": "N",
             "c": "C", "central": "C", "centre": "C", "center": "C", "trung": "C",
             "s": "S", "south": "S", "southern": "S", "nam": "S"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(path: Path = MODELS) -> dict:
    """configs/models.yaml with its `by_name` index (kaggle_run_plan.load_models)."""
    import kaggle_run_plan as KRP
    return KRP.load_models(path)


def load_plan(path: Path) -> dict:
    import kaggle_verify_items as KVI
    return KVI.load_plan(path)


def expand_models(run: dict, plan: dict, mc: dict) -> list[str]:
    import kaggle_run_plan as KRP
    return list(KRP.expand_models(run, plan, mc))


def cpu_quota_weight(quota_path: Path = QUOTA) -> float:
    """experiments/quota.yaml `cpu_sessions_count_against_gpu_quota`: true -> a CPU hour weighs one GPU quota hour;
    null (not yet verified) or false -> 0 (CPU hours are shown on their own line of the meter)."""
    if not quota_path.exists():
        return 0.0
    q = yaml.safe_load(quota_path.read_text(encoding="utf-8")) or {}
    return 1.0 if q.get("cpu_sessions_count_against_gpu_quota") is True else 0.0


def cost_per_hour(est_unit: str, cpu_weight: float) -> float:
    return {"gpu_hours": COST_PER_GPU_HOUR, "tpu_hours": COST_PER_TPU_HOUR, "l4_hours": COST_PER_L4_HOUR,
            "cpu_hours": cpu_weight}[est_unit]


def split_of(item_key: str, spec: dict) -> str:
    if spec.get("kind") == "xcopa" or "xcopa" in item_key:
        return "xcopa"
    if item_key == "attested":
        return "attested"
    if item_key == "noilai_dev" or item_key.startswith("pilot_") or item_key.endswith("_dev"):
        return "dev"
    if item_key == "noilai_core" or (spec.get("derive") or {}).get("source") == "noilai_core" or item_key.startswith("noilai_api"):
        return "core"
    return "test"


def split_run_dir_name(name: str) -> tuple[str, str, str] | None:
    """`<run>__<model>[__<tag>]` -> (run, model, tag or ""); None when the name is not of that shape. Model names
    contain `--` (serving variants) and never `__`."""
    parts = name.split("__")
    if len(parts) not in (2, 3) or not all(parts):
        return None
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else "")


def unit_id_of(run_id: str, model: str, arm: str, tag: str = "") -> str:
    return f"{run_id}__{model}__{arm}" + (f"__{tag}" if tag else "")


def cut_rank_of(run_id: str, model: str) -> int:
    return RUN_TIER.get(run_id, 70) * 100 + MODEL_RANK.get(model, 50)


def kind_and_gate(run: dict, exp: str, lane: str, exploratory: bool) -> tuple[str, str]:
    kind = "exploratory" if (exploratory or exp in EXPLORATORY_EXPERIMENTS or run.get("exploratory")) else "confirmatory"
    if exp in PREREQUISITE_EXPERIMENTS:
        kind = "prerequisite"
    gate = "none"
    if kind == "confirmatory":
        gate = "prereg+qu+iy" + ("+gemini" if lane == "api" else "")
    return kind, gate


def seed_of(run: dict, spec: dict, mc: dict, names: list[str]) -> str:
    first = mc["by_name"].get(names[0], {}) if names else {}
    seed = f"backend_seed={first.get('seed', (mc.get('defaults') or {}).get('seed', 20261203))}"
    if spec.get("derive"):
        seed += f";sample_seed={spec['derive'].get('seed')}"
    return seed


def base_row(run: dict, plan: dict, model: str, arm: str, tag: str, lane: str, exploratory: bool) -> dict:
    """The plan-derived columns every run_eval unit shares (chunk columns and costs are filled by the caller)."""
    rid, exp = run["id"], run["experiment"]
    item_key = run["items"]
    spec = plan["item_files"][item_key]
    kind, gate = kind_and_gate(run, exp, lane, exploratory)
    return {"unit_id": unit_id_of(rid, model, arm, tag), "experiment": exp, "run_id": rid, "model": model, "arm": arm,
            "split": split_of(item_key, spec), "item_key": item_key, "paraphrases": "|".join(run.get("paraphrases") or []),
            "section": SECTION_OF_EXPERIMENT.get(exp, "Other"), "kind": kind, "lane": lane, "tag": tag,
            "chunk_id": "", "chunk_order": 0, "tier": "", "queue": "", "window": "", "notebook": "",
            "chunk_hours_low": "", "chunk_hours_high": "", "cut_rank": cut_rank_of(rid, model), "status": "planned",
            "gate": gate, "hash_verified": "n/a" if spec.get("sha256") is None else "no", "notes": []}


def normalizing_note(row: dict, entry: dict) -> None:
    if row["arm"] in NORMALIZING_ARMS and entry.get("family") in NORMALIZING_FAMILIES:
        row["cut_rank"] += 50
        row["notes"].append("DD 7.2: tokenizer expected to normalize; this cell is 0 by construction once the census confirms it")


# ---- units of the chunk plan -----------------------------------------------------------------------------------

def chunk_job_hours(ch: dict, job: dict, model: str, run: dict, mc: dict, job_hours: dict, cpu_est: dict) -> tuple[float, float]:
    """(low, high) hours of one (chunk job, model): tiers 1-5 from plan_chunks.make_jobs; tier-0 GPU/TPU chunks split the
    line estimate equally over the job's models; the CPU chunks take the RL-2026-10-01-06 benchmark per line x the
    params_b ratio, exactly as plan_chunks.tier0_chunks books the chunk."""
    import plan_chunks as PC
    if ch["tier"] >= 1:
        return tuple(job_hours[(job["run"], model, job.get("tag"))])
    if ch["queue"] == "cpu":
        ref = mc["by_name"][PC.CPU_REFERENCE_MODEL]["params_b"]
        s = round(mc["by_name"][model]["params_b"] / ref, 2)
        per_run = {"smoke_20": cpu_est["smoke_20"], "pilot_t1_200": cpu_est["pilot_t1_200"] + cpu_est["pilot_t1_200_forced_choice"],
                   "pilot_xcopa_200": cpu_est["pilot_xcopa_200"]}
        h = s * per_run[job["run"]]
        return h, h
    est = run["estimate"]
    n = len(job["models"])
    return est["low"] / n, est["high"] / n


def chunk_rows(spec: dict, plan: dict, mc: dict, cpu_weight: float) -> list[dict]:
    import kaggle_run_plan as KRP
    import plan_chunks as PC
    jobs, _unscheduled = PC.make_jobs(plan, mc)
    job_hours = {(j["run"], j["model"], j["tag"]): j["hours"] for j in jobs}
    cpu_est = PC.cpu_estimates()
    rows: dict[str, dict] = {}
    for ch in spec["chunks"]:
        est_unit = UNIT_OF_QUEUE[ch["queue"]]
        lane = LANE_OF_QUEUE[ch["queue"]]
        for job in ch["jobs"]:
            run = KRP.find_run(plan, job["run"])
            narrowed = KRP.narrow_run(run, job.get("overrides"), job.get("tag"))
            tag = job.get("tag") or ""
            arms = list(narrowed.get("arms") or ["nfc"])
            for model in job["models"]:
                entry = mc["by_name"][model]
                lo, hi = chunk_job_hours(ch, job, model, run, mc, job_hours, cpu_est)
                skipped = KRP.census_skipped_arms(narrowed, entry)
                for arm in arms:
                    uid = unit_id_of(run["id"], model, arm, tag)
                    if uid in rows:                       # c01's smoke_20 is re-entered by c02 (--resume makes it a no-op)
                        rows[uid]["notes"].append(f"re-entered by {ch['id']} (--resume no-op; its hours are booked on {rows[uid]['chunk_id']})")
                        continue
                    r = base_row(narrowed, plan, model, arm, tag, lane, exploratory=False)
                    r["seed"] = seed_of(run, plan["item_files"][run["items"]], mc, [model])
                    hours = (lo + hi) / 2 / len(arms)
                    r.update(chunk_id=ch["id"], chunk_order=ch["order"], tier=ch["tier"], queue=ch["queue"], window=ch["window"],
                             notebook=ch["notebook"], chunk_hours_low=round(lo / len(arms), 4), chunk_hours_high=round(hi / len(arms), 4),
                             est_unit=est_unit, est_raw=round(hours, 4), est_cost=round(hours * cost_per_hour(est_unit, cpu_weight), 4))
                    normalizing_note(r, entry)
                    if arm in skipped:
                        r["status"] = "zero_by_construction"
                        r["notes"].append(f"DD 6.2: {skipped[arm]}")
                    if run["id"] in RUN_ALIAS:
                        r["notes"].append(f"alias of {RUN_ALIAS[run['id']]} for the TPU trio (the analyses pool it under {RUN_ALIAS[run['id']]})")
                    if tag in ("p0p1", "p2"):
                        r["notes"].append("paraphrase split (DD 8.5): the p0p1 and p2 units of this (run, model, arm) add up to the line")
                    if tag == "cpu":
                        r["notes"].append("CPU engine (hf, float32): a unit apart from the T4 one (COMPUTE_PLAN section 2)")
                    rows[uid] = r
    # the T4 pilot chunk duplicates the CPU pilot chunks: an alternative route (memo N11 picks CPU), counted like cut until launched
    for r in rows.values():
        if r["tier"] == 0 and not r["tag"] and r["experiment"] == "pilot" and f"{r['unit_id']}__cpu" in rows:
            r["status"] = "alternative"
            r["notes"].append("T4 alternative to the CPU pilot chunks c02-c04 (memo N11): counted like cut until launched; "
                              "whichever route finishes marks the other cut (ledger.py ingest)")
    return list(rows.values())


def e4_rows(run: dict) -> list[dict]:
    rid, exp = run["id"], run["experiment"]
    est = run["estimate"]
    mid = (est["low"] + est["high"]) / 2
    rows = []
    for name in run["models"]:
        share = E4_SPLIT.get(name, 1.0 / len(run["models"]))
        rows.append({"unit_id": f"{rid}__{name}__{'+'.join(run['arms'])}", "experiment": exp, "run_id": rid, "model": name,
                     "arm": "+".join(run["arms"]), "split": "inventory", "item_key": "syllable_inventory",
                     "seed": "syllable_seed=0;pair_seed=0", "paraphrases": "", "section": "Tone",
                     "kind": "confirmatory", "lane": "kaggle_gpu", "queue": "gpu", "window": E4_WINDOW,
                     "notebook": "notebooks/colab_probe_gemma3.ipynb", "cut_rank": cut_rank_of(rid, name),
                     "est_cost": round(mid * share * COST_PER_GPU_HOUR, 4), "est_unit": "gpu_hours",
                     "est_raw": round(mid * share, 4), "status": "planned", "gate": "prereg", "hash_verified": "n/a",
                     "notes": ["DD 8.8: reported regardless of outcome; noilai.probe, not run_eval; not a chunk (plan_chunks NOT_CHUNKED)"]})
    return rows


def api_rows(run: dict, plan: dict, mc: dict) -> list[dict]:
    names = expand_models(run, plan, mc)
    arms = list(run.get("arms") or ["nfc"])
    est = run["estimate"]
    mid = (est["low"] + est["high"]) / 2
    rows = []
    for name in names:
        for arm in arms:
            r = base_row(run, plan, name, arm, "", "api", exploratory=False)
            r["seed"] = seed_of(run, plan["item_files"][run["items"]], mc, names)
            calls = mid / len(names) / len(arms)
            r.update(queue="api", window=str(run.get("week") or "").split(" (")[0], notebook="notebooks/api_runs.ipynb",
                     est_raw=round(calls, 1), est_unit="api_calls", est_cost=round(calls / 1000 * COST_PER_1000_API_CALLS, 4))
            r["notes"].append("not a chunk: bounded by the providers' daily caps, re-run daily (plan_chunks NOT_CHUNKED)")
            rows.append(r)
    return rows


def unscheduled_rows(spec: dict, plan: dict, mc: dict, cpu_weight: float) -> list[dict]:
    import kaggle_run_plan as KRP
    rows = []
    for u in spec.get("unscheduled") or []:
        run = KRP.find_run(plan, u["run"])
        entry = mc["by_name"][u["model"]]
        arms = list(run.get("arms") or ["nfc"])
        hw = entry.get("hardware")
        est_unit = {"l4": "l4_hours", "tpu": "tpu_hours"}.get(hw, "gpu_hours")
        lane = {"l4": "modal_l4", "tpu": "kaggle_tpu"}.get(hw, "kaggle_gpu")
        lo, hi = u["hours"]
        for arm in arms:
            r = base_row(run, plan, u["model"], arm, "", lane, exploratory=False)
            r["seed"] = seed_of(run, plan["item_files"][run["items"]], mc, [u["model"]])
            hours = (lo + hi) / 2 / len(arms)
            r.update(status="unscheduled", est_unit=est_unit, est_raw=round(hours, 4),
                     est_cost=round(hours * cost_per_hour(est_unit, cpu_weight), 4))
            r["notes"].append(f"unscheduled (memo N9: the Modal fallback is dropped, no money): {u['reason']}")
            normalizing_note(r, entry)
            rows.append(r)
    return rows


def exploratory_rows(xp: dict, mc: dict, seen_runs: set[str]) -> list[dict]:
    rows = []
    for run in xp["runs"]:
        if run["id"] in seen_runs:                     # pilot lines copied verbatim into the exploratory plan: one unit each
            continue
        names = expand_models(run, xp, mc)
        arms = list(run.get("arms") or ["nfc"])
        est = run["estimate"]
        lo, hi = est["low"] / len(names), est["high"] / len(names)
        for name in names:
            for arm in arms:
                r = base_row(run, xp, name, arm, "", "kaggle_gpu", exploratory=True)
                r["seed"] = seed_of(run, xp["item_files"][run["items"]], mc, names)
                hours = (lo + hi) / 2 / len(arms)
                r.update(status="unscheduled", queue="gpu", window=str(run.get("week") or "").split(" (")[0],
                         notebook=EXPLORATORY_NOTEBOOK, est_unit="gpu_hours", est_raw=round(hours, 4),
                         est_cost=round(hours * COST_PER_GPU_HOUR, 4))
                r["notes"].append(EXPLORATORY_NOTE)
                rows.append(r)
    return rows


def human_rows() -> list[dict]:
    rows = []
    for uid, exp, note, hours in HUMAN_UNITS:
        item_key = "validation_packet" if exp == "human_validation" else ("noilai_main" if exp == "human_baseline" else "attested_seed")
        rows.append({"unit_id": uid, "experiment": exp, "run_id": uid.split("__")[0], "model": "", "arm": "", "split": "human",
                     "item_key": item_key, "seed": "", "paraphrases": "", "section": "Human", "kind": "confirmatory", "lane": "human",
                     "queue": "human", "cut_rank": {"human_validation": 500, "human_baseline": 600, "data": 550}[exp],
                     "est_cost": hours, "est_unit": "person_hours", "est_raw": hours, "status": "planned", "gate": "none",
                     "hash_verified": "n/a", "notes": [note]})
    return rows


def sort_key(r: dict) -> tuple:
    order = int(r.get("chunk_order") or 0)
    return (order if order > 0 else 10**6, int(r["cut_rank"]), r["unit_id"])


def build_rows(plan_path: Path = PLAN, exploratory_path: Path = PLAN_EXPLORATORY, models_path: Path = MODELS,
               quota_path: Path = QUOTA) -> list[dict]:
    """Every unit of the chunk plan (configs/compute_chunks.yaml as its generator derives it from the plan and the panel),
    then the lines no chunk runs (API, E4), the unscheduled job, the exploratory dev lines and the human lane."""
    import plan_chunks as PC
    mc = load_models(models_path)
    plan = load_plan(plan_path)
    spec = PC.build_chunks(plan, mc)
    if plan_path == PLAN and models_path == MODELS and PC.CHUNKS_YAML.exists() and \
            PC.CHUNKS_YAML.read_text(encoding="utf-8") != PC.render_yaml(spec):
        print("warning: configs/compute_chunks.yaml differs from its generator; run `python scripts/plan_chunks.py --write` "
              "(the ledger follows the generator)", file=sys.stderr)
    cpu_weight = cpu_quota_weight(quota_path)
    rows = chunk_rows(spec, plan, mc, cpu_weight)
    chunked = {r["run_id"] for r in rows} | {u["run"] for u in spec.get("unscheduled") or []}
    for run in plan["runs"]:
        if run["id"] in chunked:
            continue
        if run.get("not_a_run_eval_line"):
            rows += e4_rows(run)
        elif run.get("platform") == "api":
            rows += api_rows(run, plan, mc)
        else:
            raise ValueError(f"run {run['id']!r} is in no chunk, is not an API line and is not E4: plan_chunks.py and the ledger disagree")
    rows += unscheduled_rows(spec, plan, mc, cpu_weight)
    if exploratory_path.exists():
        rows += exploratory_rows(load_plan(exploratory_path), mc, {r["run_id"] for r in rows})
    rows += human_rows()
    now = utc_now()
    for r in rows:
        r["notes"] = "; ".join(x for x in r["notes"] if x) if isinstance(r["notes"], list) else r["notes"]
        for c in COLUMNS:
            r.setdefault(c, "")
        r["updated_utc"] = now
    ids = [r["unit_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = sorted({u for u in ids if ids.count(u) > 1})
        raise ValueError(f"duplicate unit ids: {dup}")
    rows.sort(key=sort_key)
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


def totals(rows: list[dict]) -> dict:
    """Cost totals the meter and the hand-off quote (nothing typed by hand)."""
    live = [r for r in rows if r["status"] not in NON_LIVE_STATUSES]
    hours = {u: round(sum(float(r["est_raw"]) for r in live if r["est_unit"] == u), 2) for u in ("gpu_hours", "tpu_hours", "cpu_hours", "l4_hours")}
    return {"units": len(rows), "compute_units": sum(1 for r in rows if r["lane"] != "human"),
            "chunked_units": sum(1 for r in rows if r["chunk_id"]), "chunks": len({r["chunk_id"] for r in rows if r["chunk_id"]}),
            "status_counts": dict(sorted({s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}}.items())),
            "compute_cost_units": round(sum(float(r["est_cost"]) for r in live if r["lane"] != "human"), 2),
            **hours, "api_calls": round(sum(float(r["est_raw"]) for r in live if r["est_unit"] == "api_calls"), 1),
            "human_person_hours": round(sum(float(r["est_raw"]) for r in live if r["lane"] == "human"), 1)}


def cmd_init(args) -> int:
    path = Path(args.ledger)
    carry: dict[str, dict] = {}
    if path.exists():
        if not args.force:
            print(f"{path} exists; refusing to rebuild (use --force; observed statuses are carried over by unit_id)")
            return 1
        carry = {r["unit_id"]: r for r in read_ledger(path)}
    rows = build_rows(Path(args.plan), Path(args.exploratory), Path(args.models), Path(args.quota))
    carried = 0
    for r in rows:
        old = carry.get(r["unit_id"])
        if not old:
            continue
        if old.get("status") not in PLAN_DERIVED_STATUSES:           # an observation (or the author's `cut`): keep it
            for c in OBSERVED_COLUMNS:
                r[c] = old.get(c, r[c])
            carried += 1
        if REPRICED_MARK in (old.get("notes") or "") and old.get("est_raw"):   # measured hours (reprice) are observations too
            r["est_raw"], r["est_cost"] = old["est_raw"], old.get("est_cost") or r["est_cost"]
        if old.get("notes") and old["notes"] not in r["notes"]:
            extra = [x for x in old["notes"].split("; ") if x and x not in r["notes"]]
            if extra:
                r["notes"] = (r["notes"] + "; " if r["notes"] else "") + "; ".join(extra)
    write_ledger(rows, path)
    print(json.dumps({"ledger": str(path), "carried_over": carried, **totals(rows)}))
    return 0


# ---- ingest -------------------------------------------------------------------------------------------------

def plan_sha_for(plan: dict, item_key: str) -> str | None:
    spec = (plan.get("item_files") or {}).get(item_key) or {}
    return spec.get("sha256")


def _count_jsonl(path: Path, key: str, default: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = d.get(key, default)
                out[k] = out.get(k, 0) + 1
    return out


def mirror_aggregates(run_dir: Path, m: dict, dest: Path) -> list[str]:
    """Aggregates only (never item-level rows): summary.json, stats.json, results_hashes.json and a manifest excerpt."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("summary.json", "stats.json", "results_hashes.json"):
        if (run_dir / name).exists():
            (dest / name).write_bytes((run_dir / name).read_bytes())
            copied.append(name)
    keep = {k: m.get(k) for k in ("identity", "data", "options", "outcome", "status", "gpu_hours", "tpu_hours", "wall_s",
                                  "git_commit", "git_dirty", "code_sha256", "engine", "backend", "canary_check")}
    (dest / "manifest_excerpt.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    return [*copied, "manifest_excerpt.json"]


def ingest_run_dir(run_dir: Path, rows_by_unit: dict[str, dict], plans: list[dict], aggregates: Path) -> tuple[list[str], dict | None]:
    """Update the ledger rows of one run directory data/runs/<run>__<model>[__<tag>]/. Returns (unit ids touched,
    an `unmatched` record when the directory names no unit of the ledger, else None)."""
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return [], None
    m = json.loads(mpath.read_text(encoding="utf-8"))
    run_id = (m.get("identity") or {}).get("run_id") or m.get("run_id") or run_dir.name
    parts = split_run_dir_name(run_id)
    if parts is None:
        return [], {"dir": run_dir.name, "run_id": run_id, "reason": "not <run>__<model>[__<tag>]", "units": []}
    line_id, model, tag = parts
    opts = m.get("options") or {}
    arms = list(opts.get("arms") or ["nfc"])
    paraphrases = list(opts.get("paraphrases") or ["p0"])
    data = m.get("data") or {}
    item_file = m.get("item_file") if isinstance(m.get("item_file"), dict) else {}
    observed_sha = data.get("item_file_sha256") or item_file.get("sha256")
    n_selected = item_file.get("n_selected") or data.get("n_items") or 0
    rows_per_arm = _count_jsonl(run_dir / "outputs.jsonl", "arm", "nfc")
    unchanged_per_arm = _count_jsonl(run_dir / "unchanged.jsonl", "arm", "")
    outputs = run_dir / "outputs.jsonl"
    out_sha = sha256_file(outputs) if outputs.exists() else ""
    scores = run_dir / "scores.jsonl"
    scores_sha = sha256_file(scores) if scores.exists() else ""
    results_sha = ""
    if (run_dir / "results_hashes.json").exists():
        try:
            results_sha = json.loads((run_dir / "results_hashes.json").read_text(encoding="utf-8")).get("results_sha256") or ""
        except json.JSONDecodeError:
            results_sha = ""
    touched, unknown = [], []
    for arm in arms:
        uid = unit_id_of(line_id, model, arm, tag)
        row = rows_by_unit.get(uid)
        if row is None:
            unknown.append(uid)
            continue
        finished = m.get("status") == "finished" or (m.get("status") == "finished_thinking_present"
                                                     and row.get("experiment") in THINKING_EXPERIMENTS)
        expected = int(n_selected) * len(paraphrases) - unchanged_per_arm.get(arm, 0)
        n = rows_per_arm.get(arm, 0)
        ref = None                                   # the primary plan's hash wins; a later plan fills in only a missing one
        for p in plans:
            if ref is None:
                ref = plan_sha_for(p, row.get("item_key", ""))
        hv = "n/a" if ref is None else ("yes" if observed_sha == ref else "no")
        if finished and n >= expected > 0:
            status = "done"
        elif n > 0:
            status = "partial"
        else:
            status = "launched"
        if row.get("status") == "cut":
            status = "cut"
        row.update(status=status, hash_verified=hv, result_file=str(outputs.relative_to(ROOT)) if outputs.is_relative_to(ROOT) else str(outputs),
                   result_sha256=out_sha, scores_sha256=scores_sha, results_sha256=results_sha, n_rows=n, n_expected=expected,
                   updated_utc=utc_now())
        if hv == "no" and ref is not None and "ITEM FILE HASH MISMATCH" not in (row.get("notes") or ""):
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"ITEM FILE HASH MISMATCH: run read {observed_sha}, plan records {ref}"
        elif hv != "no" and "ITEM FILE HASH MISMATCH" in (row.get("notes") or ""):   # verified since: the stale note goes
            row["notes"] = "; ".join(x for x in row["notes"].split("; ") if not x.startswith("ITEM FILE HASH MISMATCH"))
        touched.append(uid)
    if touched:
        mirror_aggregates(run_dir, m, aggregates / run_dir.name)
    unmatched = None
    if unknown:
        unmatched = {"dir": run_dir.name, "run_id": run_id, "units": unknown,
                     "reason": "no such unit in the ledger" + (" (tag)" if tag else "") + "; run `ledger.py init --force` if the plan changed"}
    return touched, unmatched


def reconcile_alternatives(rows_by_unit: dict[str, dict]) -> list[str]:
    """The CPU pilot chunks (c02-c04, tag cpu) and the T4 pilot chunk (c05) are alternative routes to the same Gate 1
    numbers (memo N11): once one route's unit is done, the other's is cut unless it has started. Returns the units cut."""
    cut = []
    for uid, r in list(rows_by_unit.items()):
        if r.get("tag") != "cpu" or r.get("experiment") != "pilot":
            continue
        twin = rows_by_unit.get(uid[: -len("__cpu")])
        if twin is None:
            continue
        for done, other in ((r, twin), (twin, r)):
            if done.get("status") == "done" and other.get("status") in ("planned", "alternative"):
                other.update(status="cut", updated_utc=utc_now())
                other["notes"] = (other["notes"] + "; " if other["notes"] else "") + \
                    f"cut: the other route ({done['unit_id']}) finished (memo N11; whichever route finishes marks the other cut)"
                cut.append(other["unit_id"])
    return cut


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


def region_of(value) -> str | None:
    s = str(value or "").strip()
    if not s:
        return None
    return REGION_OF.get(s.lower()) or REGION_OF.get(s[0].lower())


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def ingest_validation(rows_by_unit: dict[str, dict], validation_dir: Path) -> tuple[list[str], list[dict]]:
    """validation__<region> from the packet scorer's data/validation/report/validation_report.json
    (make_validation_forms.py score): a validator letter maps to a region through the report's `regions`
    (data/validation/validators.json); done when its Part B rows returned reach its allocation
    (validation_manifest.json per_validator[V].B, else the rows of B_key.json that name V)."""
    rep_path = validation_dir / "report" / "validation_report.json"
    rep = _json(rep_path)
    if not rep:
        return [], []
    manifest = _json(validation_dir / "validation_manifest.json")
    per_v = manifest.get("per_validator") or {}
    regions = {k: region_of(v) for k, v in (rep.get("regions") or {}).items()}
    if not regions:
        regions = {k: region_of(v.get("region") if isinstance(v, dict) else v) for k, v in _json(validation_dir / "validators.json").items()}
    b_rows = (rep.get("rows_returned") or {}).get("B_items") or {}
    b_key = _json(validation_dir / "B_key.json")
    key_rows = b_key.get("rows") if isinstance(b_key, dict) else (b_key if isinstance(b_key, list) else [])
    touched, unmatched = [], []
    letters = set(rep.get("validators_returned") or []) | set(per_v) | set(regions)
    for v in sorted(letters):
        reg = regions.get(v)
        uid = f"validation__{reg}" if reg else None
        row = rows_by_unit.get(uid) if uid else None
        if row is None:
            unmatched.append({"validator": v, "reason": "no region for this validator letter (data/validation/validators.json)" if not reg
                              else f"no unit {uid}"})
            continue
        expected = (per_v.get(v) or {}).get("B")
        if expected is None and key_rows:
            expected = sum(1 for r in key_rows if isinstance(r, dict) and v in (r.get("validators") or []))
        n = int(b_rows.get(v) or 0)
        returned = v in set(rep.get("validators_returned") or [])
        if returned:
            status = "done" if expected and n >= expected else "partial"
        else:
            status = row["status"]
        upd = {"status": status, "n_rows": n, "n_expected": expected if expected is not None else "", "updated_utc": utc_now()}
        if returned:
            upd.update(result_file=str(rep_path.relative_to(ROOT)) if rep_path.is_relative_to(ROOT) else str(rep_path),
                       result_sha256=sha256_file(rep_path))
        hours = (per_v.get(v) or {}).get("hours_estimate")
        if hours:
            upd.update(est_raw=float(hours), est_cost=float(hours))
        row.update(upd)
        touched.append(uid)
    if per_v and regions:
        covered = {regions.get(v) for v in per_v}
        for reg in VALIDATION_REGIONS:
            uid = f"validation__{reg}"
            if reg not in covered and uid in rows_by_unit and rows_by_unit[uid]["status"] == "planned":
                rows_by_unit[uid].update(status="unscheduled", updated_utc=utc_now())
                rows_by_unit[uid]["notes"] += "; the packet has no validator of this region (memo 7: two validators at 6.9 h each)"
                touched.append(uid)
    return touched, unmatched


def ingest_baseline(rows_by_unit: dict[str, dict], human_dir: Path) -> list[str]:
    """baseline__formNN from experiments/human/human_baseline_report.json (make_validation_forms.py score-baseline):
    `forms_included` -> done; `forms_excluded` {file: first reason} -> excluded (delivered, contributes nothing)."""
    rep_path = human_dir / "human_baseline_report.json"
    rep = _json(rep_path)
    if not rep:
        return []
    touched = []
    item_level = rep.get("item_level") or {}
    per_form = rep.get("per_form") or {}
    for nn in rep.get("forms_included") or []:
        uid = f"baseline__form{int(nn):02d}"
        row = rows_by_unit.get(uid)
        if row is None:
            continue
        pf = per_form.get(str(nn)) or per_form.get(f"{int(nn):02d}") or {}
        row.update(status="done", result_file=item_level.get("path") or str(rep_path), result_sha256=item_level.get("sha256") or sha256_file(rep_path),
                   n_rows=pf.get("answered", ""), n_expected=pf.get("n_main", ""), updated_utc=utc_now())
        touched.append(uid)
    for fname, reason in (rep.get("forms_excluded") or {}).items():
        mt = re.search(r"baseline_form_(\d+)", str(fname))
        if not mt:
            continue
        uid = f"baseline__form{int(mt.group(1)):02d}"
        row = rows_by_unit.get(uid)
        if row is None or row["status"] == "done":
            continue
        row.update(status="excluded", updated_utc=utc_now())
        note = f"excluded: {reason} ({Path(str(fname)).name})"
        if note not in row["notes"]:
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + note
        touched.append(uid)
    return touched


def ingest_human(rows_by_unit: dict[str, dict], human_dir: Path, validation_dir: Path = VALIDATION_DIR) -> tuple[list[str], list[dict]]:
    touched, unmatched = ingest_validation(rows_by_unit, validation_dir)
    touched += ingest_baseline(rows_by_unit, human_dir)
    return touched, unmatched


def ingest_attested(rows_by_unit: dict[str, dict], seed: Path = ATTESTED_SEED) -> list[str]:
    uid = "attested__expansion"
    row = rows_by_unit.get(uid)
    if row is None:
        return []
    n_verified_exact = 0
    n = 0
    if seed.exists():
        with open(seed, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                n += 1
                if (r.get("exactness") or "").strip() == "exact" and (r.get("verified_by") or "").strip():
                    n_verified_exact += 1
    row.update(n_rows=n_verified_exact, n_expected=ATTESTED_FLOOR,
               status="done" if n_verified_exact >= ATTESTED_FLOOR else ("partial" if n_verified_exact else "planned"),
               updated_utc=utc_now(),
               notes=f"{row['notes'].split(' | ')[0]} | seed rows {n}, native-verified exact rows {n_verified_exact} (floor {ATTESTED_FLOOR}; PREREG section 4)")
    return [uid]


def cmd_ingest(args) -> int:
    path = Path(args.ledger)
    rows = read_ledger(path)
    by_unit = {r["unit_id"]: r for r in rows}
    plans = [load_plan(Path(args.plan))]
    if Path(args.exploratory).exists():
        plans.append(load_plan(Path(args.exploratory)))
    touched: list[str] = []
    unmatched: list[dict] = []
    claims: dict[str, list[str]] = {}                 # unit -> the run directories whose manifest named it
    runs_root = Path(args.runs)
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            if d.is_dir():
                t, u = ingest_run_dir(d, by_unit, plans, Path(args.aggregates))
                touched += t
                for uid in t:
                    claims.setdefault(uid, []).append(d.name)
                if u:
                    unmatched.append(u)
        touched += ingest_probe_dirs(runs_root, by_unit)
    conflicts = [{"unit": uid, "dirs": dirs, "reason": "more than one run directory names this unit (manifest run_id); "
                  "the last in name order wrote the row: keep one directory"} for uid, dirs in claims.items() if len(dirs) > 1]
    cut = reconcile_alternatives(by_unit)
    t, u = ingest_human(by_unit, Path(args.human), Path(args.validation))
    touched += t
    unmatched += u
    touched += ingest_attested(by_unit)
    write_ledger(rows, path)
    done = [u for u in touched if by_unit[u]["status"] == "done"]
    bad = [u for u in touched if by_unit[u]["hash_verified"] == "no"]
    print(json.dumps({"touched": len(set(touched)), "done": len(set(done)), "hash_mismatch": sorted(set(bad)),
                      "alternative_cut": cut, "unmatched": unmatched, "conflicts": conflicts, "ledger": str(path)}, ensure_ascii=False))
    return 1 if bad else 0


def cmd_show(args) -> int:
    """One line per unit in ledger order (chunk order): chunk, order, cut rank, status, hash gate, queue, estimate, unit."""
    rows = read_ledger(Path(args.ledger))
    filters = (("status", args.status), ("experiment", args.experiment), ("lane", args.lane), ("queue", args.queue), ("chunk_id", args.chunk))
    try:
        for r in rows:
            if any(want and r[col] != want for col, want in filters):
                continue
            print(f"{r['chunk_id'] or '-':<11} {r['chunk_order'] or '-':>3} {r['cut_rank']:>5} {r['status']:<20} {r['hash_verified']:<4} "
                  f"{r['queue'] or '-':<6} {r['est_raw']:>8} {r['est_unit']:<12} {r['unit_id']}")
    except BrokenPipeError:                                   # `show | head`
        return 0
    return 0


def run_dir_name(r: dict) -> str:
    return f"{r['run_id']}__{r['model']}" + (f"__{r['tag']}" if r.get("tag") else "")


def cmd_reprice(args) -> int:
    """Replace the plan estimates by measured hours per run directory (`run__model[__tag]`, the compute log's run_id;
    DD 8.5 item 30). CPU rows (gpu_type cpu / n_gpus 0) are skipped: CPU hours re-price nothing on the GPU or TPU.
    Prints measured vs chunk hours so the author can re-run `plan_chunks.py --write` and `ledger.py init --force`."""
    import compute_log as CL
    rows = read_ledger(Path(args.ledger))
    entries = CL.read_entries(Path(args.compute_log)) if Path(args.compute_log).exists() else []
    measured: dict[str, float] = {}
    skipped_cpu = 0
    for e in entries:
        if str(e.gpu_type).lower() == "cpu" or int(e.n_gpus) == 0:
            skipped_cpu += 1
            continue
        if e.run_id and float(e.hours) > 0:
            measured[e.run_id] = measured.get(e.run_id, 0.0) + float(e.hours)
    n = 0
    report = []
    by_dir: dict[str, list[dict]] = {}
    for r in rows:
        if r["est_unit"] in ("gpu_hours", "tpu_hours"):
            by_dir.setdefault(run_dir_name(r), []).append(r)
    cpu_weight = cpu_quota_weight(Path(args.quota))
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for key, units in by_dir.items():
        if key not in measured:
            continue
        est_lo = sum(float(x["chunk_hours_low"] or 0) for x in units)
        est_hi = sum(float(x["chunk_hours_high"] or 0) for x in units)
        for r in units:
            r["est_raw"] = round(measured[key] / len(units), 4)
            r["est_cost"] = round(float(r["est_raw"]) * cost_per_hour(r["est_unit"], cpu_weight), 4)
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"{REPRICED_MARK} {measured[key]:.2f} h on {today}"
            n += 1
        report.append({"run_dir": key, "chunk_id": units[0]["chunk_id"], "measured_hours": round(measured[key], 3),
                       "chunk_estimate_hours": [round(est_lo, 3), round(est_hi, 3)], "units": len(units)})
    write_ledger(rows, Path(args.ledger))
    print(json.dumps({"repriced_units": n, "measured_runs": len(measured), "cpu_rows_skipped": skipped_cpu, "measured_vs_chunk": report,
                      "next": "when measured hours leave the chunk estimates, re-run `python scripts/plan_chunks.py --write` (DD 8.5 re-pricing) "
                              "and `python scripts/ledger.py init --force` (statuses carry over by unit_id)"}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--exploratory", default=str(PLAN_EXPLORATORY))
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--quota", default=str(QUOTA), help="experiments/quota.yaml (the CPU weight of the quota meter)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)
    g = sub.add_parser("ingest")
    g.add_argument("--runs", default=str(ROOT / "data" / "runs"))
    g.add_argument("--human", default=str(HUMAN_DIR), help="human-baseline aggregates (human_baseline_report.json)")
    g.add_argument("--validation", default=str(VALIDATION_DIR), help="the validation packet directory (report/validation_report.json)")
    g.add_argument("--aggregates", default=str(AGGREGATES))
    g.set_defaults(func=cmd_ingest)
    s = sub.add_parser("show")
    s.add_argument("--status")
    s.add_argument("--experiment")
    s.add_argument("--lane")
    s.add_argument("--queue")
    s.add_argument("--chunk")
    s.set_defaults(func=cmd_show)
    r = sub.add_parser("reprice")
    r.add_argument("--compute-log", default=str(ROOT / "data" / "compute_log.csv"))
    r.set_defaults(func=cmd_reprice)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
