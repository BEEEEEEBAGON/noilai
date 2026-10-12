#!/usr/bin/env python
"""Run the plan's self-hosted lines on a rented CUDA box (a marketplace GPU or an AWS instance) in one command.

    python scripts/cuda_box.py --bootstrap --session auto --platform rented \
        --sync-cmd "aws s3 sync data/runs s3://<bucket>/noilai/runs"             # the whole plan, cut order
    python scripts/cuda_box.py --runs pilot_t1_200 pilot_xcopa_200 --session rtx3090:24   # the Gate 1 pilot only
    python scripts/cuda_box.py --dry-run --session a100-80gb:80                           # print every command

The box is described by one session spec, `<gpu_type>:<vram_gb>[x<n_gpus>][:nobf16]`, or read from
nvidia-smi (`--session auto`). Every (run, model) goes through scripts/kaggle_run_plan.py's execute loop
with that session: a model runs when its estimated weight size fits (kaggle_run_plan.cuda_fit: params x
bytes per parameter x 1.1 + 1 GB against 90% of the VRAM) and its dtype is supported (bfloat16 needs
compute capability >= 8.0); a model that does not fit is skipped with the estimate in the message; the
weights are sharded over every card only when one card is too small (--tensor-parallel-size). The
compute log records the box's card and count under the platform given here (rented, aws, local).

Run order: the plan's own order (the cut order), minus API lines (they run from anywhere), minus E4
(not a run_eval line), minus a line whose command duplicates an earlier line's for a subset of its
models (`tpu_main` books TPU time for the three models that E1_main already covers on a box). `--runs`
overrides the list; `--models` restricts every run to those entries.

`--sync-cmd` runs from the project root after every executed model (kaggle_run_plan's after_each hook),
so a box that dies mid-plan keeps what it finished: an `aws s3 sync`, an `rclone copy`, a `git` push of
data/runs, whatever reaches durable storage. `--bootstrap` first installs the pinned vLLM (the version
pyproject.toml's `vllm` extra pins; DESIGN_DECISIONS 7.3: one engine version for the whole campaign)
and the project's eval extra, fetches the third-party resources and materializes the derived item
files. Secrets (HF_TOKEN for gated checkpoints) come from the environment and are never printed.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import colab_setup as CS
import compute_log as CL
import kaggle_run_plan as KRP

# the keys of a plan line that decide its run_eval command (kaggle_run_plan.build_command)
COMMAND_KEYS = ("items", "tasks", "variants", "arms", "paraphrases", "shots", "in_core_only", "limit",
                *(k for k, _ in KRP.ABLATION_FLAGS))


def vllm_pin(pyproject: Path) -> str:
    """The vLLM version pyproject.toml's `vllm` extra pins (`vllm==X`)."""
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{pyproject}: not a TOML file ({e})") from e
    extras = (data.get("project") or {}).get("optional-dependencies") or {}
    for spec in extras.get("vllm", []):
        if spec.startswith("vllm=="):
            return spec.split("==", 1)[1].strip()
    raise ValueError(f"{pyproject}: no `vllm==<version>` pin in the vllm extra")


def bootstrap_commands(project: Path, python: str = sys.executable, vllm_version: str | None = None) -> list[list[str]]:
    """Pinned vLLM first, then the project's eval extra, the resources, the derived item files."""
    version = vllm_version or vllm_pin(project / "pyproject.toml")
    cmds = CS.pip_install_command(project, extras=("eval",), pins={"vllm": version}, python=python)
    cmds.append([python, "scripts/fetch_resources.py"])
    cmds.append([python, "scripts/kaggle_run_plan.py", "--materialize"])
    return cmds


def _signature(run: dict) -> tuple:
    return tuple(json.dumps(run.get(k), sort_keys=True) for k in COMMAND_KEYS)


def duplicate_of(run: dict, earlier: Sequence[dict], plan: dict, models_cfg: dict) -> str | None:
    """The id of an earlier line whose command equals this one's for every model of this line, else None."""
    models = set(KRP.expand_models(run, plan, models_cfg))
    sig = _signature(run)
    for e in earlier:
        if _signature(e) == sig and models <= set(KRP.expand_models(e, plan, models_cfg)):
            return e["id"]
    return None


def default_run_ids(plan: dict, models_cfg: dict) -> list[str]:
    kept: list[dict] = []
    for r in plan["runs"]:
        if r.get("not_a_run_eval_line") or r.get("hardware") == "api":
            continue
        if duplicate_of(r, kept, plan, models_cfg):
            continue
        kept.append(r)
    return [r["id"] for r in kept]


def run_all(run_ids: Sequence[str], session: KRP.CudaSession, platform: str = "rented", plan: dict | None = None,
            models_cfg: dict | None = None, project_root: Path = ROOT, log_path: Path | None = None,
            models: Sequence[str] | None = None, sync_cmd: str | None = None, verify: bool = True, dry_run: bool = False,
            allow_unpinned_revision: bool = False, max_hours: float | None = None, extra: Sequence[str] = (),
            python: str = sys.executable) -> list[dict]:
    """Every line in order through kaggle_run_plan.execute on the session; `sync_cmd` after each executed model."""
    plan = plan or KRP.load_plan()
    models_cfg = models_cfg or KRP.load_models()
    t0 = time.time()

    def sync(res: dict) -> None:
        print(f"[sync ] {sync_cmd}")
        subprocess.run(shlex.split(sync_cmd), cwd=str(project_root), check=True)

    results: list[dict] = []
    for rid in run_ids:
        results += KRP.execute(rid, models=models, platform=platform, dry_run=dry_run, extra=extra, plan=plan,
                               models_cfg=models_cfg, project_root=project_root, log_path=log_path, python=python,
                               session_hardware=session, session_t0=t0, max_session_hours=max_hours,
                               after_each=sync if (sync_cmd and not dry_run) else None, verify=verify,
                               allow_unpinned_revision=allow_unpinned_revision)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="auto", metavar="SPEC",
                    help="<gpu_type>:<vram_gb>[x<n_gpus>][:nobf16], or auto (nvidia-smi); default auto")
    ap.add_argument("--platform", default="rented", choices=CL.PLATFORMS, help="compute-log platform (default rented)")
    ap.add_argument("--bootstrap", action="store_true", help="install the pinned vLLM and the project, fetch resources, materialize")
    ap.add_argument("--runs", nargs="*", default=None, metavar="RUN_ID", help="run ids in order (default: the plan's cut order)")
    ap.add_argument("--models", nargs="*", default=None, help="restrict every run to these models.yaml entries")
    ap.add_argument("--sync-cmd", default=None, help="run from the project root after every executed model")
    ap.add_argument("--max-hours", type=float, default=None, help="launch no new model past this budget")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the item-file gate (rehearsals only)")
    ap.add_argument("--allow-unpinned-revision", action="store_true", help="rehearsals only (DD 7.1)")
    ap.add_argument("--extra", default="", help="extra run_eval.py args, one shell-quoted string")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log", type=Path, default=None, help="compute log path (default from the plan)")
    args = ap.parse_args(argv)

    try:
        session = KRP.detect_cuda_session() if args.session == "auto" else KRP.parse_cuda_session(args.session)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        ap.error(f"cannot describe the CUDA session ({e!r}); pass --session <gpu_type>:<vram_gb>[x<n_gpus>][:nobf16] "
                 f"or make nvidia-smi available")
    print(f"[session] {session}")
    if args.bootstrap:
        for cmd in bootstrap_commands(ROOT, python=args.python):
            print(f"[boot ] {shlex.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(ROOT), check=True)
    plan = KRP.load_plan()
    models_cfg = KRP.load_models()
    run_ids = args.runs or default_run_ids(plan, models_cfg)
    print(f"[runs ] {' '.join(run_ids)}")
    results = run_all(run_ids, session, platform=args.platform, plan=plan, models_cfg=models_cfg, log_path=args.log,
                      models=args.models, sync_cmd=args.sync_cmd, verify=not args.no_verify, dry_run=args.dry_run,
                      allow_unpinned_revision=args.allow_unpinned_revision, max_hours=args.max_hours,
                      extra=shlex.split(args.extra), python=args.python)
    failed = [r for r in results if str(r["status"]).startswith("failed")]
    refused = [r for r in results if str(r["status"]).startswith(KRP.REFUSED)]
    skipped = [r for r in results if str(r["status"]).startswith("skipped")]
    print(f"\n[summary] {len(results)} commands, {len(failed)} failed, {len(refused)} refused, {len(skipped)} skipped")
    return 1 if failed or refused else 0


if __name__ == "__main__":
    sys.exit(main())
