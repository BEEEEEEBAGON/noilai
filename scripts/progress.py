"""The progress meter: cost-weighted share of planned units that are done and hash-verified.

    python scripts/progress.py                 # print the meter and rewrite the block in STATUS.md
    python scripts/progress.py --no-write      # print only
    python scripts/progress.py --json out.json # also dump the numbers

WEIGHTS (every number below is read from experiments/ledger.csv `est_cost`, which scripts/ledger.py
computes with these same constants; nothing in STATUS.md is typed by hand):
    1 Kaggle GPU-hour        = 1.00 cost unit
    1 Kaggle TPU-hour        = 1.00 cost unit
    1 Modal L4-hour          = 1.00 cost unit
    1,000 API calls          = 0.25 cost units
    human person-hours       -> their own line (validation, baseline, attested expansion), never in the compute meter
A unit counts as DONE only when its ledger status is `done` and its item-file hash check is not `no`
(DESIGN_DECISIONS 4.6: a run on a file whose hash differs from the one recorded at the data freeze is
not a result of the paper's benchmark). Units with status `cut` leave the denominator (DD 7.1: cut
models are named in the paper, not silently dropped).

The meter is reported overall, per experiment, per paper section (Results = E1, E1_ablation,
reasoning, bf16_drift; Counterfactuals = E3; Tone = E4; prerequisites and dev-only exploratory
lines are shown separately), and as a PRIORITY FRONTIER: walking the ledger in cut-order rank, the
cumulative cost share at which each pre-registered analysis first becomes computable, and whether
the 55-60% band reached in cut order makes the primary analyses computable (PREREGISTRATION 8.4
H1 nested LR test, 8.5 H3 pooled estimate, DD 12.17 the explicit-input main result).

Kaggle hours this week are summed from data/compute_log.csv (platform kaggle; `hours` is wall time,
which is what the weekly quota counts; TPU rows separately) inside the quota week defined in
experiments/quota.yaml, whose figures are assumptions until read from the account.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LEDGER = ROOT / "experiments" / "ledger.csv"
STATUS = ROOT / "STATUS.md"
GATES = ROOT / "experiments" / "gates.yaml"
QUOTA = ROOT / "experiments" / "quota.yaml"
COMPUTE_LOG = ROOT / "data" / "compute_log.csv"
BEGIN, END = "<!-- progress:begin -->", "<!-- progress:end -->"
TARGET_BAND = (0.55, 0.60)
SECTIONS = ("Results", "Counterfactuals", "Tone")
# tokenizer families per model (DD 7.2 / PREREGISTRATION 8.1) for the minimum-viable-panel rule
FAMILY = {"gemma-3-1b-it": "gemma3", "gemma-3-4b-it": "gemma3", "gemma-3-12b-it": "gemma3", "qwen3.5-0.8b": "qwen",
          "qwen3.5-2b": "qwen", "qwen3.5-4b": "qwen", "qwen3.5-9b": "qwen", "gemma-4-e2b": "gemma4", "gemma-4-e4b": "gemma4",
          "gemma-4-12b": "gemma4", "gemma-sea-lion-v4.5-e2b-it": "gemma4", "sailor2-8b-chat": "qwen", "phogpt-4b-chat": "phogpt",
          "vistral-7b-chat": "vistral", "llama-3.1-8b-instruct": "llama3.1", "qwen3.8-27b": "qwen", "qwen-sea-lion-v4.5-27b-it": "qwen",
          "gpt-oss-120b": "o200k", "gpt-oss-20b": "o200k", "gemini-flash": "gemini", "gemini-flash-lite": "gemini"}
PASS_THROUGH_EXPECTED = {"gemma3", "gemma4", "llama3.1", "phogpt", "vistral", "o200k"}   # DD 7.2 expectations, census pending


def is_done(r: dict) -> bool:
    return r["status"] == "done" and r["hash_verified"] != "no"


def share(rows: list[dict]) -> tuple[float, float, float]:
    live = [r for r in rows if r["status"] != "cut"]
    tot = sum(float(r["est_cost"]) for r in live)
    done = sum(float(r["est_cost"]) for r in live if is_done(r))
    return (done / tot if tot else 0.0), done, tot


def analyses_computable(done_units: set[str], rows_by_unit: dict[str, dict], attested_ok: bool, human_done: int) -> dict[str, dict]:
    """Which pre-registered analyses can run on the completed units (rules from PREREGISTRATION 8)."""
    def done(prefix: str, arm: str | None = None):
        return {r["model"] for u, r in rows_by_unit.items() if u in done_units and r["run_id"] == prefix and (arm is None or r["arm"] == arm)}
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
    out = {
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
    return out


def frontier(rows: list[dict], rows_by_unit: dict[str, dict], attested_ok: bool, human_done: int) -> dict:
    """Walk the compute units in cut-order rank; report the units inside the 55-60% band and the analyses computable there."""
    live = sorted([r for r in rows if r["lane"] != "human" and r["status"] != "cut"], key=lambda r: (int(r["cut_rank"]), r["unit_id"]))
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
    at_band = {u for u in band_units}
    comp_band = analyses_computable(at_band, rows_by_unit, attested_ok, human_done)
    return {"band_units": band_units, "analyses_at_band": {k: v["computable"] for k, v in comp_band.items()},
            "first_computable_at_share": first_computable,
            "primary_not_at_band": [k for k, v in comp_band.items() if v["primary"] and not v["computable"]]}


def kaggle_hours_this_week(quota: dict, log: Path) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    weekday = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}[str(quota.get("week_resets_on", "saturday")).lower()]
    days_since = (now.weekday() - weekday) % 7
    start = (now - dt.timedelta(days=days_since)).replace(hour=int(quota.get("week_reset_hour_utc", 0)), minute=0, second=0, microsecond=0)
    if start > now:
        start -= dt.timedelta(days=7)
    gpu = tpu = 0.0
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
                if str(r.get("gpu_type", "")).startswith("tpu"):
                    tpu += h
                else:
                    gpu += h
    return {"week_start_utc": start.strftime("%Y-%m-%d %H:%MZ"), "gpu_used": round(gpu, 2), "tpu_used": round(tpu, 2),
            "gpu_quota": quota.get("gpu_hours_per_week"), "tpu_quota": quota.get("tpu_hours_per_week"),
            "gpu_left": round(float(quota.get("gpu_hours_per_week") or 0) - gpu, 2), "tpu_left": round(float(quota.get("tpu_hours_per_week") or 0) - tpu, 2)}


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


def compute(ledger: Path = LEDGER, gates: Path = GATES, quota_path: Path = QUOTA, log: Path = COMPUTE_LOG) -> dict:
    with open(ledger, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_unit = {r["unit_id"]: r for r in rows}
    compute_rows = [r for r in rows if r["lane"] != "human"]
    overall = share(compute_rows)
    per_exp = {}
    for exp in sorted({r["experiment"] for r in compute_rows}):
        per_exp[exp] = share([r for r in compute_rows if r["experiment"] == exp])
    per_sec = {}
    for sec in SECTIONS + ("Prerequisite", "Exploratory"):
        sub = [r for r in compute_rows if r["section"] == sec]
        if sub:
            per_sec[sec] = share(sub)
    conf = share([r for r in compute_rows if r["kind"] == "confirmatory"])
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
    fr = frontier(rows, by_unit, attested_ok, human_done)
    counts = defaultdict(int)
    for r in compute_rows:
        counts[r["status"]] += 1
    running = [r for r in rows if r["status"] in ("launched", "running", "partial")]
    quota = yaml.safe_load(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else {}
    return {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"), "ledger": str(ledger.relative_to(ROOT)) if ledger.is_relative_to(ROOT) else str(ledger),
            "n_units": len(rows), "n_compute_units": len(compute_rows), "status_counts": dict(counts),
            "overall": overall, "confirmatory": conf, "per_experiment": per_exp, "per_section": per_sec, "human": human,
            "analyses": analyses, "frontier": fr, "running": [(r["unit_id"], r["status"], r["n_rows"], r["n_expected"]) for r in running],
            "kaggle_week": kaggle_hours_this_week(quota, log), "quota_assumed": quota, "gates": gates_summary(gates)}


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def render(d: dict) -> str:
    L = [BEGIN, f"_Generated by `scripts/progress.py` on {d['generated_utc']} from `{d['ledger']}` "
         f"({d['n_compute_units']} compute units, {d['n_units'] - d['n_compute_units']} human units). Weights: 1 GPU-h = 1 TPU-h = 1 L4-h = 1.0; "
         "1,000 API calls = 0.25; human person-hours on their own lines. A unit counts only when `done` and hash-verified._", ""]
    o = d["overall"]
    L.append("### Meter")
    L.append("")
    L.append("| scope | done / planned (cost units) | share |")
    L.append("|---|---|---|")
    L.append(f"| **Overall (compute)** | {o[1]:.1f} / {o[2]:.1f} | **{pct(o[0])}** |")
    c = d["confirmatory"]
    L.append(f"| confirmatory units only | {c[1]:.1f} / {c[2]:.1f} | {pct(c[0])} |")
    for sec, v in d["per_section"].items():
        L.append(f"| section: {sec} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    for exp, v in d["per_experiment"].items():
        L.append(f"| experiment: {exp} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    for exp, v in d["human"].items():
        label = {"human_validation": "human validation (person-h)", "human_baseline": "human baseline (person-h)", "data": "attested expansion (person-h)"}[exp]
        L.append(f"| {label} | {v[1]:.1f} / {v[2]:.1f} | {pct(v[0])} |")
    L.append("")
    L.append(f"Status counts (compute units): {json.dumps(d['status_counts'])}")
    L.append("")
    L.append("### Pre-registered analyses computable on the completed units")
    L.append("")
    L.append("| analysis | computable now | n | primary |")
    L.append("|---|---|---|---|")
    for name, info in d["analyses"].items():
        L.append(f"| {name} | {'yes' if info['computable'] else 'no'} | {info['n']} | {'yes' if info['primary'] else ''} |")
    fr = d["frontier"]
    L.append("")
    L.append(f"### Priority frontier (cut-order walk to the {int(TARGET_BAND[0]*100)}-{int(TARGET_BAND[1]*100)}% band)")
    L.append("")
    L.append(f"Units inside the band when done in cut order: {len(fr['band_units'])} (first: `{fr['band_units'][0] if fr['band_units'] else '-'}`, last: `{fr['band_units'][-1] if fr['band_units'] else '-'}`).")
    L.append("Share at which each analysis first becomes computable: " + ", ".join(f"{k} at {int(v*100)}%" for k, v in fr["first_computable_at_share"].items()))
    L.append("Analyses computable at the band: " + ", ".join(k for k, v in fr["analyses_at_band"].items() if v) + ".")
    if fr["primary_not_at_band"]:
        L.append(f"**Primary analyses NOT computable at the band: {', '.join(fr['primary_not_at_band'])}.**")
    else:
        L.append("All three primary analyses are computable at the band.")
    L.append("")
    k = d["kaggle_week"]
    L.append("### Kaggle hours this quota week")
    L.append("")
    L.append(f"Week starting {k['week_start_utc']} (reset day per `experiments/quota.yaml`, unverified): GPU {k['gpu_used']} used, {k['gpu_left']} left of {k['gpu_quota']} assumed; "
             f"TPU {k['tpu_used']} used, {k['tpu_left']} left of {k['tpu_quota']} assumed. Source: `data/compute_log.csv` (absent = nothing logged).")
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
        L.append("- confirmatory runs: **gated** (missing: " + ", ".join(blocked) + ")" if blocked else "- confirmatory runs: open")
        L.append("- API runs: **gated** (Gemini route undecided)" if not g.get("gemini_route") else "- API runs: open")
    L.append("")
    L.append("### Running now")
    L.append("")
    if d["running"]:
        for u, s, n, e in d["running"]:
            L.append(f"- `{u}`: {s} ({n or 0}/{e or '?'} rows)")
    else:
        L.append("Nothing is running (no unit is launched, running or partial).")
    L.append(END)
    return "\n".join(L)


def write_status(block: str, path: Path = STATUS) -> None:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        else:
            text = text.rstrip("\n") + "\n\n" + block + "\n"
    else:
        text = "# STATUS\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--status", default=str(STATUS))
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
