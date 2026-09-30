#!/usr/bin/env python
"""Turn configs/run_plan.yaml into `scripts/run_eval.py` commands, run them, and log the time.

The notebooks' RUN cells call this so that the flag mapping lives in one tested place:

    python scripts/kaggle_run_plan.py --list
    python scripts/kaggle_run_plan.py --run E1_main --dry-run                      # print every command
    python scripts/kaggle_run_plan.py --run E1_main --models gemma-3-1b-it qwen3.5-2b \
        --platform kaggle --continue-on-error                                       # execute, log hours
    python scripts/kaggle_run_plan.py --run E1_api_core --platform api               # quota-aware API loop

For each (run, model) the command is

    <python> scripts/run_eval.py --model-config <name> --items <path> --tasks ... --variants ...
        --paraphrases ... --shots N --arms ... [--limit N] [--in-core-only] --resume
        --run-id <run_id>__<name> --out-root data/runs [extra args]

The output flags follow `run_eval_out_style` in run_plan.yaml: "run_id_root" (default; the CLI
as landed writes data/runs/<run-id>/) or "out" (the first specification's `--out <dir>`).
The run directory is data/runs/<run_id>__<name> either way.

Guards (raise before anything runs): an API-served model must run with --in-core-only, and an
item file flagged `never_to_api` never reaches an API backend (plan 2.3: APIs see only the
core; the sealed split reaches no API). A run flagged `not_a_run_eval_line` (E4) is refused.

Accounting: every executed command appends a compute-log row (platform, device type and
count from models.yaml, wall hours, run dir, run id) through scripts/compute_log.py, and API
models keep a daily ledger (data/runs/api_ledger.json: calls per model per UTC day, counted
from new rows in outputs.jsonl) so that a model whose free-tier daily request budget is spent
is skipped until tomorrow. --resume makes re-running a run id a no-op for finished items.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compute_log as CL  # noqa: E402

CONFIGS = ROOT / "configs"
API_BACKENDS = {"openai_compat", "gemini"}
HARDWARE_DEVICES = {"t4": ("t4", 1), "2xt4": ("t4", 2), "tpu": ("tpu-v5e-8", 1), "api": ("api", 0),
                    "p100": ("p100", 1), "mixed": (None, None)}


# ------------------------------------------------------------------ loading
def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plan(path: Path = CONFIGS / "run_plan.yaml") -> dict:
    return load_yaml(path)


def load_models(path: Path = CONFIGS / "models.yaml") -> dict:
    """models.yaml with an added `by_name` index."""
    cfg = load_yaml(path)
    cfg["by_name"] = {m["name"]: m for m in cfg["models"]}
    return cfg


def find_run(plan: dict, run_id: str) -> dict:
    for r in plan["runs"]:
        if r["id"] == run_id:
            return r
    raise KeyError(f"no run {run_id!r}; have {[r['id'] for r in plan['runs']]}")


# ------------------------------------------------------------------ model sets
def expand_set(name: str, plan: dict, models_cfg: dict, _seen: Optional[set] = None) -> list[str]:
    """A model set is a list of names, {union: [sets]} or {group: <models.yaml group>}."""
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"model_sets cycle at {name!r}")
    _seen.add(name)
    sets = plan.get("model_sets", {})
    if name not in sets:
        raise KeyError(f"unknown model set {name!r}")
    spec = sets[name]
    if isinstance(spec, list):
        out = list(spec)
    elif "union" in spec:
        out = []
        for s in spec["union"]:
            out.extend(x for x in expand_set(s, plan, models_cfg, _seen) if x not in out)
    elif "group" in spec:
        out = [m["name"] for m in models_cfg["models"] if m.get("group") == spec["group"]]
    else:
        raise ValueError(f"model set {name!r}: expected a list, {{union: [...]}} or {{group: ...}}")
    unknown = [m for m in out if m not in models_cfg["by_name"]]
    if unknown:
        raise KeyError(f"model set {name!r} names unknown models {unknown}")
    return out


def expand_models(run: dict, plan: dict, models_cfg: dict) -> list[str]:
    spec = run["models"]
    if isinstance(spec, dict) and "set" in spec:
        return expand_set(spec["set"], plan, models_cfg)
    if isinstance(spec, list):
        unknown = [m for m in spec if m not in models_cfg["by_name"]]
        if unknown:
            raise KeyError(f"run {run['id']!r} names unknown models {unknown}")
        return list(spec)
    raise ValueError(f"run {run['id']!r}: models must be a list or {{set: name}}")


# ------------------------------------------------------------------ commands
def run_dir(plan: dict, run: dict, model: str) -> str:
    return f"{plan.get('runs_root', 'data/runs')}/{run['id']}__{model}"


def is_api(entry: dict) -> bool:
    return entry.get("backend") in API_BACKENDS or entry.get("hardware") == "api"


def build_command(run: dict, model: str, plan: dict, models_cfg: dict, project_root: Path = ROOT,
                  python: str = sys.executable, extra: Sequence[str] = ()) -> list[str]:
    """The run_eval.py argv for one (run, model). Raises on a guarded combination."""
    if run.get("not_a_run_eval_line"):
        raise ValueError(f"run {run['id']!r} is not a run_eval line (see its purpose)")
    entry = models_cfg["by_name"].get(model)
    if entry is None:
        raise KeyError(f"unknown model {model!r}")
    item_spec = plan["item_files"][run["items"]]
    api = is_api(entry)
    if api and item_spec.get("never_to_api"):
        raise ValueError(f"item file {run['items']!r} must never reach an API ({model})")
    if api and not run.get("in_core_only"):
        raise ValueError(f"API model {model!r} may only run with in_core_only: true (run {run['id']!r})")
    cli = plan.get("run_eval_cli", "scripts/run_eval.py")
    cmd = [python, str(Path(project_root) / cli), "--model-config", model, "--items", item_spec["path"],
           "--tasks", *run["tasks"]]
    if run.get("variants"):
        cmd += ["--variants", *run["variants"]]
    if run.get("paraphrases"):
        cmd += ["--paraphrases", *run["paraphrases"]]
    cmd += ["--shots", str(run.get("shots", 0)), "--arms", *run["arms"]]
    if run.get("limit"):
        cmd += ["--limit", str(run["limit"])]
    if run.get("in_core_only"):
        cmd.append("--in-core-only")
    cmd.append("--resume")
    rd = run_dir(plan, run, model)
    style = plan.get("run_eval_out_style", "run_id_root")
    if style == "run_id_root":          # the CLI as landed: --run-id <dir name> --out-root <runs root>
        cmd += ["--run-id", Path(rd).name, "--out-root", plan.get("runs_root", "data/runs")]
    elif style == "out":                # the CLI as first specified: --out <dir>
        cmd += ["--out", rd]
    else:
        raise ValueError(f"run_eval_out_style must be 'run_id_root' or 'out', not {style!r}")
    cmd += list(extra)
    return cmd


def device_for(entry: dict) -> tuple[str, int]:
    """(device type, count) for the compute log, from the model's hardware assignment."""
    hw = entry.get("hardware", "mixed")
    dev, n = HARDWARE_DEVICES.get(hw, (hw, 1))
    return (dev or hw, 1 if n is None else n)


# ------------------------------------------------------------------ API ledger
def ledger_path(plan: dict, project_root: Path = ROOT) -> Path:
    return Path(project_root) / plan.get("runs_root", "data/runs") / "api_ledger.json"


def ledger_load(path: Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def ledger_save(path: Path, ledger: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_day() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def ledger_add(ledger: dict, model: str, n_calls: int, day: Optional[str] = None) -> dict:
    day = day or utc_day()
    ledger.setdefault(model, {})
    ledger[model][day] = int(ledger[model].get(day, 0)) + int(n_calls)
    return ledger


def ledger_used(ledger: dict, model: str, day: Optional[str] = None) -> int:
    return int(ledger.get(model, {}).get(day or utc_day(), 0))


def daily_budget(entry: dict, models_cfg: dict) -> Optional[int]:
    """Free-tier requests/day for an API entry (None when unknown or not an API model)."""
    prov = entry.get("provider")
    if not prov:
        return None
    tier = models_cfg.get("providers", {}).get(prov, {}).get("free_tier", {})
    rpd = tier.get("requests_per_day")
    return int(rpd) if rpd else None


def count_outputs(run_directory: Path) -> int:
    p = Path(run_directory) / "outputs.jsonl"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


# ------------------------------------------------------------------ execution
def execute(run_id: str, models: Optional[Sequence[str]] = None, platform: str = "kaggle", dry_run: bool = False,
            continue_on_error: bool = True, extra: Sequence[str] = (), plan: Optional[dict] = None,
            models_cfg: Optional[dict] = None, project_root: Path = ROOT, log_path: Optional[Path] = None,
            python: str = sys.executable) -> list[dict]:
    """Run every (run, model) command in order; return one result dict per model."""
    plan = plan or load_plan()
    models_cfg = models_cfg or load_models()
    run = find_run(plan, run_id)
    names = list(models) if models else expand_models(run, plan, models_cfg)
    log_path = Path(log_path) if log_path else Path(project_root) / plan.get("compute_log", "data/compute_log.csv")
    lpath = ledger_path(plan, project_root)
    ledger = ledger_load(lpath)
    results: list[dict] = []
    for name in names:
        entry = models_cfg["by_name"][name]
        cmd = build_command(run, name, plan, models_cfg, project_root=project_root, python=python, extra=extra)
        out_dir = Path(project_root) / run_dir(plan, run, name)
        res = {"run": run_id, "model": name, "cmd": cmd, "out": str(out_dir), "status": "dry-run"}
        print(f"\n[plan ] {run_id} / {name}\n[cmd  ] {shlex.join(cmd)}")
        budget = daily_budget(entry, models_cfg) if is_api(entry) else None
        if budget is not None:
            used = ledger_used(ledger, name)
            res["api_budget_today"] = {"requests_per_day": budget, "used": used}
            if used >= budget:
                res["status"] = "skipped: daily request budget spent"
                print(f"[quota] {name}: {used}/{budget} requests used today; skipping until tomorrow (UTC)")
                results.append(res)
                continue
            print(f"[quota] {name}: {used}/{budget} requests used today")
        if dry_run:
            results.append(res)
            continue
        before = count_outputs(out_dir)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(project_root), check=False)
        hours = CL.hours_since(t0)
        after = count_outputs(out_dir)
        res.update({"returncode": proc.returncode, "hours": round(hours, 4), "new_outputs": after - before,
                    "status": "ok" if proc.returncode == 0 else f"failed ({proc.returncode})"})
        dev, n = device_for(entry)
        CL.append_entry(log_path, CL.Entry(date=CL.today(), platform=platform, gpu_type=dev, n_gpus=n,
                                           hours=hours, run_id=out_dir.name, purpose=run_id))
        if budget is not None:
            ledger_add(ledger, name, max(0, after - before))
            ledger_save(lpath, ledger)
        print(f"[done ] {name}: {res['status']}, {hours:.3f} h, {after - before} new outputs")
        results.append(res)
        if proc.returncode != 0 and not continue_on_error:
            break
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=CONFIGS / "run_plan.yaml")
    ap.add_argument("--models-config", type=Path, default=CONFIGS / "models.yaml")
    ap.add_argument("--list", action="store_true", help="list the runs and their expanded models")
    ap.add_argument("--run", help="run id from run_plan.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="subset of the run's models (default: all)")
    ap.add_argument("--platform", default="kaggle", choices=CL.PLATFORMS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--extra", default="", help="extra run_eval.py args, one shell-quoted string")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log", type=Path, default=None, help="compute log path (default from the plan)")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    models_cfg = load_models(args.models_config)
    if args.list:
        for r in plan["runs"]:
            try:
                names = expand_models(r, plan, models_cfg)
            except KeyError as e:
                names = [f"ERROR {e}"]
            est = r.get("estimate", {})
            print(f"{r['id']:<22} {r['experiment']:<10} {est.get('low')}-{est.get('high')} {est.get('unit')}"
                  f"   {len(names)} models: {', '.join(names)}")
        return 0
    if not args.run:
        ap.error("--run or --list is required")
    results = execute(args.run, models=args.models, platform=args.platform, dry_run=args.dry_run,
                      continue_on_error=not args.stop_on_error, extra=shlex.split(args.extra), plan=plan,
                      models_cfg=models_cfg, log_path=args.log, python=args.python)
    failed = [r for r in results if str(r["status"]).startswith("failed")]
    print(f"\n[summary] {len(results)} commands, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
