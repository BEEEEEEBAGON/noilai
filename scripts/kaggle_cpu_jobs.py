#!/usr/bin/env python
"""CPU-only Kaggle driver for the work that needs Hugging Face hub access but no GPU: the
tokenizer audits of the whole panel, the per-item tokenization covariates, and the panel pin.

    python scripts/kaggle_cpu_jobs.py audit-tokenizers                     # data/audit/<name>.json + <name>_rows.csv
    python scripts/kaggle_cpu_jobs.py audit-items --items-keys noilai_main noilai_core noilai_dev
    python scripts/kaggle_cpu_jobs.py pin                                  # experiments/panel_pins.json
    python scripts/kaggle_cpu_jobs.py all --report data/audit/cpu_jobs_report.json
    python scripts/kaggle_cpu_jobs.py all --dry-run                        # print the commands; report says "planned"

Jobs. `audit-tokenizers` runs scripts/audit_tokenizers.py once per distinct tokenizer of
configs/models.yaml: the two local SentencePiece files (gemma3, gemma2 from data/external,
the Makefile's `audit` target) and every distinct `hf_id` (self-hosted entries and the
open-weight API entries alike; serving variants share their reference's id, so the hub
tokenizers are fetched once each). `audit-items` runs scripts/audit_items.py for every
tokenizer on every item file named by --items-keys (keys of configs/run_plan.yaml
`item_files`, `{release}` resolved like kaggle_run_plan does), writing
data/audit/items_<tokenizer>__<key>.jsonl. `pin` runs scripts/pin_panel.py (hashes plus the
tokenizer-file SHA-256s into experiments/panel_pins.json; the apply step stays offline).
`all` is the three in that order. The scripts are called as subprocesses, never imported.

Inputs are checked before anything runs: a SentencePiece file that scripts/fetch_resources.py
has not fetched, or an item file the release does not have here, skips that command with a
message naming the path; the hub itself cannot be checked offline, so a hub failure shows up
as a failed command (return code and stderr tail in the report). The exit code is non-zero
only when a requested job could not run at all (every one of its commands was skipped);
individual failures are recorded and the driver goes on.

The report (--report, default data/audit/cpu_jobs_report.json) records every command with its
status (planned / ok / failed / skipped), return code, wall time and the SHA-256 and size of
every output file it produced, so a Kaggle session's artefacts can be matched to the files
later committed or attached as a dataset.

Kaggle CPU session: accelerator None, Internet ON, HF_TOKEN as a Kaggle secret exported into
the environment (see scripts/pin_panel.py), the code bundle cloned and `pip install -e .`
plus `transformers sentencepiece huggingface_hub` installed; the release item files restored
into data/release/<version>/ (or point --release-dir at the attached dataset's tree), then
`python scripts/fetch_resources.py` and this script. Everything under data/audit and
experiments/ is then saved from /kaggle/working.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"            # the scripts run beside this driver; --project-root only moves data/
sys.path.insert(0, str(SCRIPTS))
KVI = importlib.import_module("kaggle_verify_items")   # scripts/kaggle_verify_items.py, imported after the path insert

if TYPE_CHECKING:
    from collections.abc import Callable

CONFIGS = ROOT / "configs"
JOBS = ("audit-tokenizers", "audit-items", "pin")
DEFAULT_ITEMS_KEYS = ("noilai_main", "noilai_core", "noilai_dev")
DEFAULT_REPORT = ROOT / "data" / "audit" / "cpu_jobs_report.json"
# the local SentencePiece files of the Makefile's `audit` target (fetched by scripts/fetch_resources.py)
SPM_FILES = (("gemma3_tokenizer.model", "gemma3"), ("gemma2_tokenizer.model", "gemma2"))
# the item file whose inputs feed the normalization census probe set of scripts/audit_tokenizers.py (design 6.2)
CENSUS_ITEMS_KEY = "noilai_main"
STATUS_PLANNED, STATUS_OK, STATUS_FAILED, STATUS_SKIPPED = "planned", "ok", "failed", "skipped"
_TAIL = 2000                      # characters of stdout/stderr kept per command in the report


@dataclass
class Context:
    root: Path = ROOT
    python: str = sys.executable
    models_config: Path = CONFIGS / "models.yaml"
    plan_path: Path = CONFIGS / "run_plan.yaml"
    audit_dir: Path | None = None            # default: <root>/data/audit
    external_dir: Path | None = None         # default: <root>/data/external
    release_dir: Path | None = None          # overrides the plan's `release` directory
    pin_out: Path | None = None              # default: <root>/experiments/panel_pins.json
    include_api: bool = True                 # resolve API entries with an open-weight hf_id in the pin record

    def __post_init__(self):
        self.root = Path(self.root)
        self.audit_dir = Path(self.audit_dir) if self.audit_dir else self.root / "data" / "audit"
        self.external_dir = Path(self.external_dir) if self.external_dir else self.root / "data" / "external"
        self.pin_out = Path(self.pin_out) if self.pin_out else self.root / "experiments" / "panel_pins.json"


@dataclass
class Command:
    job: str
    name: str
    argv: list[str]
    outputs: list[Path]
    skip_reason: str | None = None
    note: str | None = None
    status: str = STATUS_PLANNED
    returncode: int | None = None
    wall_s: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    produced: dict = field(default_factory=dict)

    def as_record(self, root: Path) -> dict:
        return {"job": self.job, "name": self.name, "status": self.status, "command": shlex.join(self.argv),
                "argv": self.argv, "skip_reason": self.skip_reason, "note": self.note, "returncode": self.returncode,
                "wall_s": self.wall_s, "outputs": [_relative(p, root) for p in self.outputs], "produced": self.produced,
                "stdout_tail": self.stdout_tail, "stderr_tail": self.stderr_tail}


# ------------------------------------------------------------------ loading
def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plan(ctx: Context) -> dict:
    """run_plan.yaml with `{release}` resolved; --release-dir replaces the plan's `release`."""
    plan = KVI.load_plan(ctx.plan_path)
    if ctx.release_dir is not None:
        plan["release"] = str(ctx.release_dir)
        for spec in plan.get("item_files", {}).values():
            spec["path"] = KVI.resolve_path(spec["path_template"], plan)
    return plan


def item_path(plan: dict, key: str, ctx: Context) -> Path:
    files = plan.get("item_files", {})
    if key not in files:
        raise KeyError(f"{key!r} is not an item file key of {ctx.plan_path} (have {sorted(files)})")
    p = Path(files[key]["path"])
    return p if p.is_absolute() else ctx.root / p


def tokenizers(models_cfg: dict, ctx: Context) -> list[dict]:
    """Every distinct tokenizer to audit: the local SentencePiece files, then every distinct hf_id of
    models.yaml in order of first appearance. Names follow the adapters (HFAdapter: hf_id with '/' -> '__')."""
    out = [{"name": name, "kind": "spm", "path": ctx.external_dir / fname} for fname, name in SPM_FILES]
    seen: set[str] = set()
    for m in models_cfg.get("models") or []:
        hf_id = m.get("hf_id")
        if hf_id and hf_id not in seen:
            seen.add(hf_id)
            out.append({"name": hf_id.replace("/", "__"), "kind": "hf", "hf_id": hf_id})
    return out


def _tokenizer_flags(tok: dict) -> list[str]:
    if tok["kind"] == "spm":
        return ["--spm", f"{tok['path']}:{tok['name']}"]
    return ["--hf", tok["hf_id"]]


def _spm_skip(tok: dict) -> str | None:
    if tok["kind"] == "spm" and not Path(tok["path"]).exists():
        return f"missing {tok['path']}: run scripts/fetch_resources.py first"
    return None


# ------------------------------------------------------------------ planning
def plan_audit_tokenizers(ctx: Context, models_cfg: dict, plan: dict) -> list[Command]:
    script = str(SCRIPTS / "audit_tokenizers.py")
    census = item_path(plan, CENSUS_ITEMS_KEY, ctx)
    items_flags = ["--items", str(census)] if census.exists() else []
    note = None if census.exists() else (f"{_relative(census, ctx.root)} is absent: the census probe set is "
                                         "words only (design 6.2 wants 500 words + 500 item inputs)")
    out = []
    for tok in tokenizers(models_cfg, ctx):
        argv = [ctx.python, script, *_tokenizer_flags(tok), "--out", str(ctx.audit_dir), *items_flags]
        outputs = [ctx.audit_dir / f"{tok['name']}.json", ctx.audit_dir / f"{tok['name']}_rows.csv"]
        out.append(Command("audit-tokenizers", tok["name"], argv, outputs, skip_reason=_spm_skip(tok), note=note))
    return out


def plan_audit_items(ctx: Context, models_cfg: dict, plan: dict, keys: tuple[str, ...] = DEFAULT_ITEMS_KEYS) -> list[Command]:
    script = str(SCRIPTS / "audit_items.py")
    out = []
    for key in keys:
        items = item_path(plan, key, ctx)
        missing = None if items.exists() else f"missing item file {items} (run_plan.yaml key {key!r})"
        for tok in tokenizers(models_cfg, ctx):
            target = ctx.audit_dir / f"items_{tok['name']}__{key}.jsonl"
            argv = [ctx.python, script, "--items", str(items), *_tokenizer_flags(tok), "--out", str(target)]
            out.append(Command("audit-items", f"{tok['name']}__{key}", argv, [target],
                               skip_reason=missing or _spm_skip(tok)))
    return out


def plan_pin(ctx: Context) -> list[Command]:
    script = SCRIPTS / "pin_panel.py"
    argv = [ctx.python, str(script), "--models-config", str(ctx.models_config), "--out", str(ctx.pin_out)]
    if ctx.include_api:
        argv.append("--include-api")
    skip = None
    if not script.exists():
        skip = f"missing {script}"
    elif not Path(ctx.models_config).exists():
        skip = f"missing {ctx.models_config}"
    return [Command("pin", "pin", argv, [Path(ctx.pin_out)], skip_reason=skip,
                    note="exit 1 means the resolution is incomplete; the JSON is written either way")]


def plan_jobs(jobs: list[str], ctx: Context, items_keys: tuple[str, ...] = DEFAULT_ITEMS_KEYS) -> list[Command]:
    """The commands of the requested jobs, in run order (`all` expands to the three)."""
    wanted = list(JOBS) if "all" in jobs else [j for j in JOBS if j in jobs]
    unknown = sorted(set(jobs) - set(JOBS) - {"all"})
    if unknown:
        raise ValueError(f"unknown jobs {unknown}; choose from {(*JOBS, 'all')}")
    models_cfg = load_yaml(ctx.models_config)
    plan = load_plan(ctx)
    out: list[Command] = []
    for job in wanted:
        if job == "audit-tokenizers":
            out.extend(plan_audit_tokenizers(ctx, models_cfg, plan))
        elif job == "audit-items":
            out.extend(plan_audit_items(ctx, models_cfg, plan, items_keys))
        elif job == "pin":
            out.extend(plan_pin(ctx))
    return out


# ------------------------------------------------------------------ execution
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _produced(cmd: Command, root: Path) -> dict:
    out = {}
    for p in cmd.outputs:
        if p.exists():
            out[_relative(p, root)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def execute(commands: list[Command], ctx: Context, dry_run: bool = False,
            runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[Command]:
    """Run every non-skipped command in order, recording status, return code, wall time and the
    hashes of the outputs it produced. `runner` is subprocess.run or a test double with its signature."""
    for cmd in commands:
        if cmd.skip_reason:
            cmd.status = STATUS_SKIPPED
            print(f"[{cmd.job}] {cmd.name}: skipped: {cmd.skip_reason}", flush=True)
            continue
        print(f"[{cmd.job}] {cmd.name}: {shlex.join(cmd.argv)}", flush=True)
        if dry_run:
            cmd.status = STATUS_PLANNED
            continue
        for p in cmd.outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = runner(cmd.argv, cwd=str(ctx.root), text=True, capture_output=True, check=False)
            cmd.returncode = r.returncode
            cmd.stdout_tail, cmd.stderr_tail = (r.stdout or "")[-_TAIL:], (r.stderr or "")[-_TAIL:]
        except Exception as e:                      # the interpreter itself could not be started
            cmd.returncode = -1
            cmd.stderr_tail = f"{type(e).__name__}: {e}"[-_TAIL:]
        cmd.wall_s = round(time.time() - t0, 3)
        cmd.produced = _produced(cmd, ctx.root)
        cmd.status = STATUS_OK if cmd.returncode == 0 else STATUS_FAILED
        print(f"[{cmd.job}] {cmd.name}: {cmd.status} (rc={cmd.returncode}, {cmd.wall_s}s, "
              f"{len(cmd.produced)}/{len(cmd.outputs)} outputs)", flush=True)
        if cmd.status == STATUS_FAILED and cmd.stderr_tail:
            print(cmd.stderr_tail[-600:], file=sys.stderr, flush=True)
    return commands


def job_summary(commands: list[Command], jobs: list[str]) -> dict:
    """Per requested job: how many commands ended in each status and whether the job ran at all
    (at least one command was not skipped)."""
    wanted = list(JOBS) if "all" in jobs else [j for j in JOBS if j in jobs]
    out = {}
    for job in wanted:
        mine = [c for c in commands if c.job == job]
        counts = {s: sum(1 for c in mine if c.status == s) for s in (STATUS_PLANNED, STATUS_OK, STATUS_FAILED, STATUS_SKIPPED)}
        out[job] = {"n_commands": len(mine), **counts, "ran": any(c.status != STATUS_SKIPPED for c in mine)}
    return out


def write_report(path: Path, commands: list[Command], jobs: list[str], ctx: Context, dry_run: bool,
                 started: str) -> dict:
    doc = {"started_utc": started, "finished_utc": _now(), "dry_run": dry_run, "python": ctx.python,
           "root": str(ctx.root), "models_config": _relative(ctx.models_config, ctx.root),
           "plan": _relative(ctx.plan_path, ctx.root), "git_commit": _git_commit(ctx.root),
           "jobs": job_summary(commands, jobs), "commands": [c.as_record(ctx.root) for c in commands]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return doc


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), text=True, capture_output=True, check=False)
        return r.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", choices=(*JOBS, "all"), metavar="JOB",
                    help=f"one or more of {', '.join(JOBS)}, or all")
    ap.add_argument("--items-keys", nargs="*", default=list(DEFAULT_ITEMS_KEYS), metavar="KEY",
                    help="run_plan.yaml item_files keys for audit-items")
    ap.add_argument("--models-config", type=Path, default=CONFIGS / "models.yaml")
    ap.add_argument("--plan", type=Path, default=CONFIGS / "run_plan.yaml")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="where data/ and experiments/ live (the scripts always run from this checkout)")
    ap.add_argument("--release-dir", type=Path, default=None, help="replace the plan's `release` directory")
    ap.add_argument("--audit-dir", type=Path, default=None, help="default <project-root>/data/audit")
    ap.add_argument("--pin-out", type=Path, default=None, help="default <project-root>/experiments/panel_pins.json")
    ap.add_argument("--no-api", action="store_true", help="pin: leave out API entries with an open-weight hf_id")
    ap.add_argument("--report", type=Path, default=None, help="default <project-root>/data/audit/cpu_jobs_report.json")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the commands; the report records them as planned")
    args = ap.parse_args(argv)

    ctx = Context(root=args.project_root, python=args.python, models_config=args.models_config, plan_path=args.plan,
                  audit_dir=args.audit_dir, release_dir=args.release_dir, pin_out=args.pin_out,
                  include_api=not args.no_api)
    report = args.report or ctx.audit_dir / DEFAULT_REPORT.name
    started = _now()
    commands = plan_jobs(args.jobs, ctx, tuple(args.items_keys))
    execute(commands, ctx, dry_run=args.dry_run)
    doc = write_report(report, commands, args.jobs, ctx, args.dry_run, started)
    print(json.dumps(doc["jobs"], ensure_ascii=False))
    print(f"report: {report}")
    not_run = [job for job, s in doc["jobs"].items() if not s["ran"]]
    if not_run:
        print(f"could not run: {not_run} (every command skipped; see the report)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
