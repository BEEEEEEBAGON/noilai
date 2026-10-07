"""The progress meter: cost-weighted share of planned units that are done and hash-verified, per compute chunk.

    python scripts/progress.py                 # print the meter and rewrite the block in STATUS.md
    python scripts/progress.py --status PATH   # rewrite the block of another file (a copy, a test)
    python scripts/progress.py --no-write      # print only
    python scripts/progress.py --json out.json # also dump the numbers

WEIGHTS (every number below is read from experiments/ledger.csv `est_cost`, which scripts/ledger.py
computes with these same constants; nothing in STATUS.md is typed by hand):
    1 Kaggle GPU-hour        = 1.00 cost unit
    1 Kaggle TPU-hour        = 1.00 cost unit
    1 Modal L4-hour          = 1.00 cost unit (the one unscheduled job; leaves the denominator)
    1 Kaggle CPU-hour        = experiments/quota.yaml `cpu_sessions_count_against_gpu_quota`: true -> 1.00,
                               null (unverified) or false -> 0: CPU units leave the quota meter and their
                               hours are shown on a line of their own
    1,000 API calls          = 0.25 cost units
    human person-hours       -> their own line (validation, baseline, attested expansion), never in the compute meter
A unit counts as DONE only when its ledger status is `done` and its item-file hash check is not `no`
(DESIGN_DECISIONS 4.6: a run on a file whose hash differs from the one recorded at the data freeze is
not a result of the paper's benchmark). Units with status `cut`, `zero_by_construction`, `unscheduled` or
`alternative` leave the denominator (DD 7.1: cut models are named in the paper, not silently dropped;
DD 6.2: an arm the census makes identical to nfc is never run; the T4 pilot route is counted only if launched).

THE RUN UNITS ARE THE CHUNKS of configs/compute_chunks.yaml (one chunk = one Kaggle session). The meter is
reported overall, per experiment, per paper section (Results = E1, E1_ablation, reasoning, bf16_drift;
Counterfactuals = E3; Tone = E4; prerequisites and dev-only exploratory lines apart), then PER CHUNK in chunk
order (units done / total, status planned / launched / partial / done, hours low-high, quota x1 / x2), rolled up
per tier and per queue, and as the NEXT CHUNK PER QUEUE: the lowest chunk order with a unit still to do. That
is the launch order (CLAUDE.md step 3). The FRONTIER walks the chunks in order and reports, per pre-registered
analysis, the first chunk at which it becomes computable and the GPU quota hours (x1, and x2 = two quota hours
per 2xT4 session hour, the second reading of docs/COMPUTE_PLAN.md) and TPU hours spent by then; the cut-rank
band line (the 55-60% share reached in DD 7.1 cut order) is kept beside it. `tpu_main` units count as E1_main,
and a (run, model, arm) counts for an analysis when a done unit of it carries paraphrase p0.

Kaggle hours this week are summed from data/compute_log.csv (platform kaggle; `hours` is wall time, which is
what the weekly quota counts: x1; x2 = hours x n_gpus for the GPU rows; TPU rows apart; CPU rows, gpu_type cpu
or n_gpus 0, on their own line) inside the quota week defined in experiments/quota.yaml, whose figures are
assumptions until read from the account.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LG = importlib.import_module("ledger")        # scripts/ledger.py: the status vocabulary and the CPU weight
LEDGER = ROOT / "experiments" / "ledger.csv"
STATUS = ROOT / "STATUS.md"
GATES = ROOT / "experiments" / "gates.yaml"
QUOTA = ROOT / "experiments" / "quota.yaml"
COMPUTE_LOG = ROOT / "data" / "compute_log.csv"
BEGIN, END = "<!-- progress:begin -->", "<!-- progress:end -->"
TARGET_BAND = (0.55, 0.60)
SECTIONS = ("Results", "Counterfactuals", "Tone")
NON_LIVE = LG.NON_LIVE_STATUSES
RUN_ALIAS = LG.RUN_ALIAS                        # tpu_main -> E1_main
QUEUES = ("cpu", "gpu", "tpu")
X2_FACTOR = 2                                   # docs/COMPUTE_PLAN.md: the x2 reading of a 2xT4 session hour
IN_FLIGHT = ("launched", "running", "partial")
# tokenizer families per model (DD 7.2 / PREREGISTRATION 8.1) for the minimum-viable-panel rule
FAMILY = {"gemma-3-1b-it": "gemma3", "gemma-3-4b-it": "gemma3", "gemma-3-12b-it": "gemma3", "qwen3.5-0.8b": "qwen",
          "qwen3.5-2b": "qwen", "qwen3.5-4b": "qwen", "qwen3.5-9b": "qwen", "gemma-4-e2b": "gemma4", "gemma-4-e4b": "gemma4",
          "gemma-4-12b": "gemma4", "gemma-sea-lion-v4.5-e2b-it": "gemma4", "sailor2-8b-chat": "qwen", "phogpt-4b-chat": "phogpt",
          "vistral-7b-chat": "vistral", "llama-3.1-8b-instruct": "llama3.1", "qwen3.8-27b": "qwen", "qwen-sea-lion-v4.5-27b-it": "qwen",
          "gpt-oss-120b": "o200k", "gpt-oss-20b": "o200k", "gemini-flash": "gemini", "gemini-flash-lite": "gemini"}
PASS_THROUGH_EXPECTED = {"gemma3", "gemma4", "llama3.1", "phogpt", "vistral", "o200k"}   # DD 7.2 expectations, census pending


def is_done(r: dict) -> bool:
    return r["status"] == "done" and r["hash_verified"] != "no"


def is_live(r: dict) -> bool:
    return r["status"] not in NON_LIVE


def share(rows: list[dict]) -> tuple[float, float, float]:
    live = [r for r in rows if is_live(r)]
    tot = sum(float(r["est_cost"]) for r in live)
    done = sum(float(r["est_cost"]) for r in live if is_done(r))
    return (done / tot if tot else 0.0), done, tot


def carries_p0(r: dict) -> bool:
    p = (r.get("paraphrases") or "").split("|")
    return not any(p) or "p0" in p


def analyses_computable(done_units: set[str], rows_by_unit: dict[str, dict], attested_ok: bool, human_done: int) -> dict[str, dict]:
    """Which pre-registered analyses can run on the completed units (rules from PREREGISTRATION 8). `tpu_main` units
    count as E1_main; a (run, model, arm) counts when a done unit of it carries p0 (a p2-only split unit does not)."""
    def done(prefix: str, arm: str | None = None):
        return {r["model"] for u, r in rows_by_unit.items() if u in done_units and RUN_ALIAS.get(r["run_id"], r["run_id"]) == prefix
                and (arm is None or r["arm"] == arm) and carries_p0(r)}
    e1 = done("E1_main", "nfc")
    e1_fams = {FAMILY.get(m, m) for m in e1}
    explicit = done("E1_explicit_input", "nfc") & e1
    e3_nfd = done("E3_noilai", "nfd") & done("E3_noilai", "nfc")
    x_nfd = done("E3_xcopa", "nfd") & done("E3_xcopa", "nfc")
    h3_models = {m for m in (e3_nfd & x_nfd) if FAMILY.get(m) in PASS_THROUGH_EXPECTED}
    h3_fams = {FAMILY.get(m) for m in h3_models}
    c2 = done("E3_c2_enriched", "placement_new") & done("E3_c2_enriched", "nfc")
    e4 = done("E4_probe")
    att = done("E1_attested", "nfc")
    mvp = len(e1) >= 8 and len(e1_fams) >= 4 and bool((e1_fams & PASS_THROUGH_EXPECTED) - {"gemma3", "gemma4"})
    return {
        "Table 2 (E1 estimation per done model)": {"computable": len(e1) > 0, "primary": False, "n": f"{len(e1)} models"},
        "Explicit-input gap, DD 12.17 main result": {"computable": len(explicit) > 0, "primary": True, "n": f"{len(explicit)} models paired"},
        "H1 nested LR test (E2), PREREG 8.4": {"computable": mvp, "primary": True,
                                               "n": f"{len(e1)} models / {len(e1_fams)} families (needs >= 8 over >= 4 incl. a non-Gemma pass-through family; item audits data/audit/items_*.jsonl)"},
        "H3 pooled crossover (E3), PREREG 8.5": {"computable": len(h3_models) >= 1, "primary": True,
                                                 "n": f"{len(h3_models)} pass-through models / {len(h3_fams)} families (pooled CI reliable from 5 families)"},
        "H3b tone-isolation DiD": {"computable": len(h3_models) >= 1, "primary": False, "n": f"{len(h3_models)} models"},
        "H4 placement (C2 set)": {"computable": len(c2) >= 1, "primary": False, "n": f"{len(c2)} models"},
        "H5 probes and patching (E4)": {"computable": len(e4) >= 1, "primary": False, "n": f"{len(e4)} models"},
        "H6 attested vs matched (needs the 100-row floor)": {"computable": len(att) >= 1 and attested_ok, "primary": False,
                                                            "n": f"{len(att)} models; floor {'met' if attested_ok else 'NOT met'}"},
        "Human-baseline comparison (mean-human band)": {"computable": human_done >= 20 and len(e1) > 0, "primary": False, "n": f"{human_done}/20 forms"},
    }


# ------------------------------------------------------------------ chunks
def chunk_status(units: list[dict]) -> str:
    live = [u for u in units if is_live(u)]
    if not live:
        return "alternative" if any(u["status"] == "alternative" for u in units) else "cut"
    if all(is_done(u) for u in live):
        return "done"
    if any(u["status"] in ("done", "partial", "running") for u in live):   # a done unit with hash `no` ran too: not planned
        return "partial"
    if any(u["status"] == "launched" for u in live):
        return "launched"
    return "planned"


def quota_hours(queue: str, lo: float, hi: float, cpu_weight: float) -> dict:
    lo, hi = round(lo, 2), round(hi, 2)                                 # as configs/compute_chunks.yaml rounds: hours first, then x2
    if queue == "gpu":
        return {"x1": [lo, hi], "x2": [round(X2_FACTOR * lo, 2), round(X2_FACTOR * hi, 2)]}
    if queue == "tpu":
        return {"x1": [lo, hi]}
    return {"x1": [round(cpu_weight * lo, 2), round(cpu_weight * hi, 2)]}


def chunk_table(rows: list[dict], cpu_weight: float) -> list[dict]:
    """One entry per chunk in chunk order: units done / total (live units), status, hours low-high, quota x1 / x2."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("chunk_id"):
            by[r["chunk_id"]].append(r)
    out = []
    for cid, units in by.items():
        live = [u for u in units if is_live(u)]
        booked = [u for u in units if u["status"] not in ("cut", "zero_by_construction")]   # an alternative chunk shows its hours
        lo = sum(float(u["chunk_hours_low"] or 0) for u in booked)
        hi = sum(float(u["chunk_hours_high"] or 0) for u in booked)
        first = units[0]
        out.append({"chunk_id": cid, "order": int(first["chunk_order"]), "tier": int(first["tier"]), "queue": first["queue"],
                    "window": first["window"], "notebook": first["notebook"], "units_done": sum(1 for u in live if is_done(u)),
                    "units_total": len(live), "units_all": len(units), "status": chunk_status(units),
                    "hours": [round(lo, 2), round(hi, 2)], "quota": quota_hours(first["queue"], lo, hi, cpu_weight),
                    "in_flight": sorted(u["unit_id"] for u in units if u["status"] in IN_FLIGHT)})
    out.sort(key=lambda c: c["order"])
    return out


def rollup(chunks: list[dict], key: str) -> dict:
    out = {}
    for k in sorted({c[key] for c in chunks}, key=str):
        cs = [c for c in chunks if c[key] == k]
        done = [c for c in cs if c["status"] == "done"]
        live = [c for c in cs if c["status"] not in ("cut", "alternative")]
        out[str(k)] = {"chunks_done": len(done), "chunks_total": len(live), "chunks_all": len(cs),
                       "hours": [round(sum(c["hours"][0] for c in live), 2), round(sum(c["hours"][1] for c in live), 2)],
                       "hours_done": [round(sum(c["hours"][0] for c in done), 2), round(sum(c["hours"][1] for c in done), 2)],
                       "quota_x1": [round(sum(c["quota"]["x1"][i] for c in live), 2) for i in (0, 1)],
                       "quota_x2": [round(sum(c["quota"].get("x2", c["quota"]["x1"])[i] for c in live), 2) for i in (0, 1)]}
    return out


def next_chunks(chunks: list[dict]) -> dict:
    """Per queue, the lowest chunk order with a live unit not yet done (this is the launch order)."""
    out = {}
    for q in QUEUES:
        nxt = next((c for c in chunks if c["queue"] == q and c["status"] in ("planned", "launched", "partial")), None)
        out[q] = None if nxt is None else {k: nxt[k] for k in ("chunk_id", "order", "tier", "status", "window", "notebook", "hours", "quota")}
    return out


def frontier_chunks(rows: list[dict], rows_by_unit: dict[str, dict], attested_ok: bool, human_done: int) -> dict:
    """Walk the chunks in chunk order as if each were done in turn: the first chunk at which every analysis becomes
    computable and the cumulative GPU (x1, x2) and TPU hours (low-high) by then. Analyses that only API or E4 units
    (not chunks) would make computable are listed as outside the chunk walk."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("chunk_id") and is_live(r):
            by[r["chunk_id"]].append(r)
    reached: set[str] = set()
    cum = {"gpu_x1": [0.0, 0.0], "gpu_x2": [0.0, 0.0], "tpu": [0.0, 0.0]}
    first: dict[str, dict] = {}
    for cid, units in sorted(by.items(), key=lambda kv: int(kv[1][0]["chunk_order"])):
        lo = round(sum(float(u["chunk_hours_low"] or 0) for u in units), 2)    # per-chunk rounding as in the chunk table,
        hi = round(sum(float(u["chunk_hours_high"] or 0) for u in units), 2)   # so the totals agree with the rollups
        q = units[0]["queue"]
        if q == "gpu":
            cum["gpu_x1"] = [cum["gpu_x1"][0] + lo, cum["gpu_x1"][1] + hi]
            cum["gpu_x2"] = [cum["gpu_x2"][0] + X2_FACTOR * lo, cum["gpu_x2"][1] + X2_FACTOR * hi]
        elif q == "tpu":
            cum["tpu"] = [cum["tpu"][0] + lo, cum["tpu"][1] + hi]
        reached |= {u["unit_id"] for u in units}
        for name, info in analyses_computable(reached, rows_by_unit, attested_ok, human_done).items():
            if info["computable"] and name not in first:
                first[name] = {"chunk_id": cid, "order": int(units[0]["chunk_order"]), "tier": int(units[0]["tier"]),
                               **{k: [round(v[0], 2), round(v[1], 2)] for k, v in cum.items()}}
    names = list(analyses_computable(set(), rows_by_unit, attested_ok, human_done))
    return {"first_computable": first, "outside_chunk_walk": [n for n in names if n not in first],
            "total": {k: [round(v[0], 2), round(v[1], 2)] for k, v in cum.items()}}


def frontier(rows: list[dict], rows_by_unit: dict[str, dict], attested_ok: bool, human_done: int) -> dict:
    """The cut-rank band: walk the live compute units in cut-order rank; the units inside the 55-60% band and the
    analyses computable there (the pre-registered priority, kept beside the chunk walk)."""
    live = sorted([r for r in rows if r["lane"] != "human" and is_live(r)], key=lambda r: (int(r["cut_rank"]), r["unit_id"]))
    tot = sum(float(r["est_cost"]) for r in live) or 1.0
    cum, band_units, reached = 0.0, [], set()
    first_computable: dict[str, float] = {}
    for r in live:
        cum += float(r["est_cost"])
        reached.add(r["unit_id"])
        if cum / tot <= TARGET_BAND[1]:
            band_units.append(r["unit_id"])
        comp = analyses_computable(reached, rows_by_unit, attested_ok, human_done)
        for name, info in comp.items():
            if info["computable"] and name not in first_computable:
                first_computable[name] = round(cum / tot, 3)
    at_band = set(band_units)
    comp_band = analyses_computable(at_band, rows_by_unit, attested_ok, human_done)
    return {"band_units": band_units, "analyses_at_band": {k: v["computable"] for k, v in comp_band.items()},
            "first_computable_at_share": first_computable,
            "primary_not_at_band": [k for k, v in comp_band.items() if v["primary"] and not v["computable"]]}


# ------------------------------------------------------------------ the quota week
def kaggle_hours_this_week(quota: dict, log: Path) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    weekday = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}[str(quota.get("week_resets_on", "saturday")).lower()]
    days_since = (now.weekday() - weekday) % 7
    start = (now - dt.timedelta(days=days_since)).replace(hour=int(quota.get("week_reset_hour_utc", 0)), minute=0, second=0, microsecond=0)
    if start > now:
        start -= dt.timedelta(days=7)
    gpu_x1 = gpu_x2 = tpu = cpu = 0.0
    if log.exists():
        with open(log, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    d = dt.datetime.fromisoformat(r["date"]).replace(tzinfo=dt.timezone.utc)
                except ValueError:
                    continue
                if d < start or r.get("platform") != "kaggle":
                    continue
                h = float(r.get("hours") or 0)
                gpu_type = str(r.get("gpu_type", "")).lower()
                try:
                    n_gpus = int(float(r.get("n_gpus") or 0))
                except ValueError:
                    n_gpus = 0
                if gpu_type.startswith("tpu"):
                    tpu += h
                elif gpu_type == "cpu" or n_gpus == 0:
                    if gpu_type not in ("api", "none"):
                        cpu += h
                else:
                    gpu_x1 += h
                    gpu_x2 += h * max(1, n_gpus)
    gq, tq = float(quota.get("gpu_hours_per_week") or 0), float(quota.get("tpu_hours_per_week") or 0)
    return {"week_start_utc": start.strftime("%Y-%m-%d %H:%MZ"), "gpu_used": round(gpu_x1, 2), "gpu_used_x2": round(gpu_x2, 2),
            "tpu_used": round(tpu, 2), "cpu_used": round(cpu, 2),
            "gpu_quota": quota.get("gpu_hours_per_week"), "tpu_quota": quota.get("tpu_hours_per_week"),
            "gpu_left": round(gq - gpu_x1, 2), "gpu_left_x2": round(gq - gpu_x2, 2), "tpu_left": round(tq - tpu, 2)}


def gates_summary(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    g = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pre = g.get("preregistration") or {}
    dec = g.get("decisions") or {}
    return {"present": True, "registration_url": pre.get("registration_url"), "registration_date": pre.get("registration_date"),
            "stage1_commit": pre.get("stage1_commit"), "stage2_commit": pre.get("stage2_commit"),
            "qu_convention": (dec.get("qu_convention") or {}).get("value"), "iy_emission": (dec.get("iy_emission") or {}).get("value"),
            "gemini_route": (dec.get("gemini_route") or {}).get("value"),
            "scoring_rule_commit": (g.get("scoring_rule_freeze") or {}).get("commit")}


# ------------------------------------------------------------------ compute
def compute(ledger: Path = LEDGER, gates: Path = GATES, quota_path: Path = QUOTA, log: Path = COMPUTE_LOG) -> dict:
    with open(ledger, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:                                        # older ledgers without the chunk columns still render
        for c in ("chunk_id", "chunk_order", "tier", "queue", "window", "tag", "chunk_hours_low", "chunk_hours_high", "notebook"):
            r.setdefault(c, "")
    quota = yaml.safe_load(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else {}
    cpu_weight = LG.cpu_quota_weight(quota_path)
    by_unit = {r["unit_id"]: r for r in rows}
    compute_rows = [r for r in rows if r["lane"] != "human"]
    quota_rows = [r for r in compute_rows if not (r["est_unit"] == "cpu_hours" and cpu_weight == 0)]
    overall = share(quota_rows)
    per_exp = {}
    for exp in sorted({r["experiment"] for r in quota_rows}):
        per_exp[exp] = share([r for r in quota_rows if r["experiment"] == exp])
    per_sec = {}
    for sec in (*SECTIONS, "Prerequisite", "Exploratory"):
        sub = [r for r in quota_rows if r["section"] == sec]
        if sub:
            per_sec[sec] = share(sub)
    conf = share([r for r in quota_rows if r["kind"] == "confirmatory"])
    cpu_rows = [r for r in compute_rows if r["est_unit"] == "cpu_hours" and is_live(r)]
    cpu = {"weight": cpu_weight, "planned_hours": round(sum(float(r["est_raw"]) for r in cpu_rows), 2),
           "done_hours": round(sum(float(r["est_raw"]) for r in cpu_rows if is_done(r)), 2),
           "units_done": sum(1 for r in cpu_rows if is_done(r)), "units_total": len(cpu_rows)}
    human = {}
    for exp in ("human_validation", "human_baseline", "data"):
        sub = [r for r in rows if r["experiment"] == exp]
        if sub:
            human[exp] = share(sub)
    att = by_unit.get("attested__expansion", {})
    attested_ok = att.get("status") == "done"
    human_done = sum(1 for r in rows if r["experiment"] == "human_baseline" and is_done(r))
    done_units = {u for u, r in by_unit.items() if is_done(r)}
    analyses = analyses_computable(done_units, by_unit, attested_ok, human_done)
    chunks = chunk_table(rows, cpu_weight)
    counts = defaultdict(int)
    for r in compute_rows:
        counts[r["status"]] += 1
    running = [r for r in rows if r["status"] in IN_FLIGHT]
    mismatch = [(r["chunk_id"], r["unit_id"]) for r in rows if r["status"] == "done" and r["hash_verified"] == "no"]
    return {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
            "ledger": str(ledger.relative_to(ROOT)) if ledger.is_relative_to(ROOT) else str(ledger),
            "n_units": len(rows), "n_compute_units": len(compute_rows), "n_chunks": len(chunks), "status_counts": dict(sorted(counts.items())),
            "weights": {"gpu_hour": LG.COST_PER_GPU_HOUR, "tpu_hour": LG.COST_PER_TPU_HOUR, "l4_hour": LG.COST_PER_L4_HOUR,
                        "cpu_hour": cpu_weight, "api_1000_calls": LG.COST_PER_1000_API_CALLS,
                        "cpu_flag": quota.get("cpu_sessions_count_against_gpu_quota")},
            "overall": overall, "confirmatory": conf, "per_experiment": per_exp, "per_section": per_sec, "cpu": cpu, "human": human,
            "chunks": chunks, "per_tier": rollup(chunks, "tier"), "per_queue": rollup(chunks, "queue"), "next_chunk": next_chunks(chunks),
            "analyses": analyses, "frontier_chunks": frontier_chunks(rows, by_unit, attested_ok, human_done),
            "frontier": frontier(rows, by_unit, attested_ok, human_done),
            "running": [(r["chunk_id"], r["unit_id"], r["status"], r["n_rows"], r["n_expected"]) for r in running],
            "hash_mismatch": mismatch,
            "kaggle_week": kaggle_hours_this_week(quota, log), "quota_assumed": quota, "gates": gates_summary(gates)}


# ------------------------------------------------------------------ render
def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _rng(v: list) -> str:
    return f"{v[0]}-{v[1]}"


def render(d: dict) -> str:
    w = d["weights"]
    header = (f"_Generated by `scripts/progress.py` on {d['generated_utc']} from `{d['ledger']}` "
              f"({d['n_compute_units']} compute units in {d['n_chunks']} chunks + non-chunked lines, {d['n_units'] - d['n_compute_units']} human units). "
              f"Weights: 1 GPU-h = {w['gpu_hour']}; 1 TPU-h = {w['tpu_hour']}; 1 L4-h = {w['l4_hour']}; "
              f"1 CPU-h = {w['cpu_hour']} (`experiments/quota.yaml` cpu_sessions_count_against_gpu_quota: {w['cpu_flag']}); "
              f"1,000 API calls = {w['api_1000_calls']}; human person-hours on their own lines. A unit counts only when `done` and "
              "hash-verified; cut / zero_by_construction / unscheduled / alternative units leave the denominator._")
    L = [BEGIN, header, ""]
    o = d["overall"]
    L.append("### Meter")
    L.append("")
    L.append("| scope | done / planned (cost units) | share |")
    L.append("|---|---|---|")
    L.append(f"| **Overall (quota meter)** | {o[1]:.1f} / {o[2]:.1f} | **{pct(o[0])}** |")
    c = d["confirmatory"]
    L.append(f"| confirmatory units only | {c[1]:.1f} / {c[2]:.1f} | {pct(c[0])} |")
    for sec, v in d["per_section"].items():
        L.append(f"| section: {sec} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    for exp, v in d["per_experiment"].items():
        L.append(f"| experiment: {exp} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    cpu = d["cpu"]
    L.append(f"| CPU sessions (hours, weight {cpu['weight']}) | {cpu['done_hours']:.1f} / {cpu['planned_hours']:.1f} "
             f"({cpu['units_done']}/{cpu['units_total']} units) | {pct(cpu['done_hours'] / cpu['planned_hours'] if cpu['planned_hours'] else 0.0)} |")
    for exp, v in d["human"].items():
        label = {"human_validation": "human validation (person-h)", "human_baseline": "human baseline (person-h)", "data": "attested expansion (person-h)"}[exp]
        L.append(f"| {label} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    L.append("")
    L.append(f"Status counts (compute units): {json.dumps(d['status_counts'])}")
    L.append("")
    L.append("### Chunks (`configs/compute_chunks.yaml`, in run order; a chunk is done when every live unit is done and hash-verified)")
    L.append("")
    L.append("| chunk | order | tier | queue | window | units done/total | status | hours low-high | quota x1 | quota x2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ch in d["chunks"]:
        q = ch["quota"]
        L.append(f"| `{ch['chunk_id']}` | {ch['order']} | {ch['tier']} | {ch['queue']} | {ch['window']} | {ch['units_done']}/{ch['units_total']} "
                 f"| {ch['status']} | {_rng(ch['hours'])} | {_rng(q['x1'])} | {_rng(q.get('x2', q['x1']))} |")
    L.append("")
    L.append("| rollup | chunks done/total | hours low-high (planned) | hours done | quota x1 | quota x2 |")
    L.append("|---|---|---|---|---|---|")
    for t, v in d["per_tier"].items():
        L.append(f"| tier {t} | {v['chunks_done']}/{v['chunks_total']} | {_rng(v['hours'])} | {_rng(v['hours_done'])} | {_rng(v['quota_x1'])} | {_rng(v['quota_x2'])} |")
    for q, v in d["per_queue"].items():
        L.append(f"| queue {q} | {v['chunks_done']}/{v['chunks_total']} | {_rng(v['hours'])} | {_rng(v['hours_done'])} | {_rng(v['quota_x1'])} | {_rng(v['quota_x2'])} |")
    L.append("")
    L.append("### Next chunk per queue (the launch order: lowest chunk order with a unit still to do)")
    L.append("")
    for q in QUEUES:
        nx = d["next_chunk"].get(q)
        if nx:
            L.append(f"- {q}: `{nx['chunk_id']}` (order {nx['order']}, tier {nx['tier']}, {nx['status']}; `{nx['notebook']}`; "
                     f"{_rng(nx['hours'])} h; window {nx['window']})")
        else:
            L.append(f"- {q}: nothing left to launch")
    L.append("")
    L.append("### Pre-registered analyses computable on the completed units")
    L.append("")
    L.append("| analysis | computable now | n | primary |")
    L.append("|---|---|---|---|")
    for name, info in d["analyses"].items():
        L.append(f"| {name} | {'yes' if info['computable'] else 'no'} | {info['n']} | {'yes' if info['primary'] else ''} |")
    fc = d["frontier_chunks"]
    L.append("")
    L.append("### Frontier (chunk walk: the first chunk at which each analysis becomes computable, and the quota spent by then)")
    L.append("")
    L.append("| analysis | first chunk | order | tier | GPU h x1 by then | GPU h x2 by then | TPU h by then |")
    L.append("|---|---|---|---|---|---|---|")
    for name, f in fc["first_computable"].items():
        L.append(f"| {name} | `{f['chunk_id']}` | {f['order']} | {f['tier']} | {_rng(f['gpu_x1'])} | {_rng(f['gpu_x2'])} | {_rng(f['tpu'])} |")
    if fc["outside_chunk_walk"]:
        L.append("")
        L.append("Not reached by the chunk walk (needs API, E4 or human units, which are not chunks): " + "; ".join(fc["outside_chunk_walk"]) + ".")
    t = fc["total"]
    L.append("")
    L.append(f"All chunks: GPU {_rng(t['gpu_x1'])} h at x1 / {_rng(t['gpu_x2'])} h at x2, TPU {_rng(t['tpu'])} h (live units; plan estimates, not measurements).")
    fr = d["frontier"]
    L.append("")
    L.append(f"Cut-rank band (DD 7.1 priority, {int(TARGET_BAND[0]*100)}-{int(TARGET_BAND[1]*100)}% of the cost): "
             f"{len(fr['band_units'])} units (first: `{fr['band_units'][0] if fr['band_units'] else '-'}`, last: `{fr['band_units'][-1] if fr['band_units'] else '-'}`); "
             "share at which each analysis first becomes computable: " + ", ".join(f"{k} at {int(v*100)}%" for k, v in fr["first_computable_at_share"].items()) + ".")
    if fr["primary_not_at_band"]:
        L.append(f"**Primary analyses NOT computable at the band: {', '.join(fr['primary_not_at_band'])}.**")
    else:
        L.append("All three primary analyses are computable at the band.")
    L.append("")
    k = d["kaggle_week"]
    L.append("### Kaggle hours this quota week")
    L.append("")
    L.append(f"Week starting {k['week_start_utc']} (reset day per `experiments/quota.yaml`, unverified): GPU {k['gpu_used']} used at x1 "
             f"({k['gpu_used_x2']} at x2 = hours x n_gpus), {k['gpu_left']} left of {k['gpu_quota']} assumed at x1 ({k['gpu_left_x2']} at x2); "
             f"TPU {k['tpu_used']} used, {k['tpu_left']} left of {k['tpu_quota']} assumed; CPU sessions {k['cpu_used']} h (no GPU quota unless "
             "`cpu_sessions_count_against_gpu_quota` is true). Source: `data/compute_log.csv` (absent = nothing logged).")
    L.append("")
    g = d["gates"]
    L.append("### Gates (`experiments/gates.yaml`)")
    L.append("")
    if not g.get("present"):
        L.append("gates.yaml missing.")
    else:
        L.append(f"- pre-registration: url `{g['registration_url']}`, date `{g['registration_date']}`, stage-1 `{g['stage1_commit']}`, stage-2 `{g['stage2_commit']}`")
        L.append(f"- qu convention `{g['qu_convention']}`; i/y emission `{g['iy_emission']}`; Gemini route `{g['gemini_route']}`; scoring rule frozen at `{g['scoring_rule_commit']}`")
        blocked = [x for x in ("registration_url", "registration_date", "qu_convention", "iy_emission") if not g.get(x)]
        L.append("- confirmatory runs (every tier >= 1 chunk): **gated** (missing: " + ", ".join(blocked) + ")" if blocked else "- confirmatory runs: open")
        L.append("- API runs: **gated** (Gemini route undecided)" if not g.get("gemini_route") else "- API runs: open")
    L.append("")
    L.append("### Running now")
    L.append("")
    if d["running"]:
        for cid, u, s, n, e in d["running"]:
            L.append(f"- `{cid or 'not chunked'}` / `{u}`: {s} ({n or 0}/{e or '?'} rows)")
    else:
        L.append("Nothing is running (no unit is launched, running or partial).")
    if d.get("hash_mismatch"):
        L.append("")
        L.append("**Done but NOT counted (item-file hash mismatch, DD 4.6; `ledger.py ingest` exits 1):** "
                 + ", ".join(f"`{c or 'not chunked'}` / `{u}`" for c, u in d["hash_mismatch"]) + ".")
    L.append(END)
    return "\n".join(L)


def write_status(block: str, path: Path = STATUS) -> None:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
        else:
            text = text.rstrip("\n") + "\n\n" + block + "\n"
    else:
        text = "# STATUS\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--status", default=str(STATUS), help="the file whose progress block is rewritten (default STATUS.md)")
    ap.add_argument("--gates", default=str(GATES))
    ap.add_argument("--quota", default=str(QUOTA))
    ap.add_argument("--compute-log", default=str(COMPUTE_LOG))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    d = compute(Path(args.ledger), Path(args.gates), Path(args.quota), Path(args.compute_log))
    block = render(d)
    print(block)
    if args.json:
        Path(args.json).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    if not args.no_write:
        write_status(block, Path(args.status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
