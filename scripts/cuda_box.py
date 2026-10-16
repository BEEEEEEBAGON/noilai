#!/usr/bin/env python
"""Run the compute chunks of configs/compute_chunks.yaml on a rented CUDA box (a marketplace GPU or an AWS instance).

    python scripts/cuda_box.py --bootstrap --session auto --chunks tier0 --dry-run      # print every command
    python scripts/cuda_box.py --session auto --chunks tier0 --platform rented \
        --kaggle-dataset noilai-runs --create-dataset                                    # the pilot and the smoke week
    python scripts/cuda_box.py --session auto --chunks from:c08_gpu_t1 --lanes 2 --platform rented \
        --kaggle-dataset noilai-runs                                                     # both cards, after the gate

Chunk mode (the default, `--chunks all`): the run units are the GPU and TPU chunks of configs/compute_chunks.yaml
(c05..c31 in `order`, docs/COMPUTE_PLAN.md; the CPU chunks c01-c04 are the HF-on-CPU route and never run on a box).
Every job of a chunk goes through scripts/kaggle_run_plan.py's execute loop with the job's models, line overrides
and tag, so the run directories are exactly the ledger's units (`<run>__<model>[__<tag>]`, scripts/ledger.py
ingests them without a mapping). Selectors: `all`, `tier0` (c05 pilot, c06 smoke, c07 TPU-trio smoke),
`from:<id>` (every GPU/TPU chunk from that id onward), explicit ids; `--models` restricts every job.

The box is described by one session spec, `<gpu_type>:<vram_gb>[x<n_gpus>][:nobf16]`, or read from nvidia-smi
(`--session auto`; a lane started with CUDA_VISIBLE_DEVICES=k sees one card). A model runs when its estimated
weight size fits (kaggle_run_plan.cuda_fit: params x bytes per parameter x 1.1 + 1 GB against 90% of the VRAM)
and its dtype is supported (bfloat16 needs compute capability >= 8.0); a model that does not fit is skipped with
the estimate in the message; the weights are sharded over every card only when one card is too small
(--tensor-parallel-size). The compute log records the box's card and count under the platform given here.

Disk (a 200 GB box cannot hold the panel's ~270 GB of checkpoints): after a model's command the driver purges the
model's hub repository (`models--<org>--<name>` under the huggingface cache) unless a later command of the same
chunk or any command of the next selected chunk uses it (`--keep-weights` turns the purge off, `--keep-repos`
exempts repositories); while a chunk runs, one background process prefetches the next chunk's repositories
that this chunk does not use, at their pinned commits, when the estimates leave 20 GB of head-room
(`--disk-headroom-gb`; `--no-prefetch`), and never for a chunk the pre-registration gate holds whole or once the
`--max-hours` budget is spent (nothing would run after the download). After every executed model: `scripts/check_run.py --run <dir>` (`--no-check`), then the
`--sync-cmd` and the push to the private Kaggle dataset (`--kaggle-dataset SLUG` through scripts/kaggle_dataset.py,
`--create-dataset` for the first version; KAGGLE_USERNAME / KAGGLE_KEY from the environment), both in the
background with at most one in flight, so a box that dies mid-chunk keeps what it finished.

`--lanes N` (N >= 2 on a box with >= 2 cards): the parent schedules the box's cards itself and runs child
processes of this script. Tiers are barriers (the cut order, DESIGN_DECISIONS 7.1 / 8.5); inside a tier the
chunks whose every model fits one card are balanced over N lanes by estimated hours (one child per card,
CUDA_VISIBLE_DEVICES=k, repositories shared between lanes never purged, the prefetch head-room multiplied by the
lane count because a lane does not see the other lanes' downloads in flight), then the chunks with a model that
needs every card (the 27B entries) run one at a time with the full session. `--lanes` above the card count uses
one lane per card. Each child logs to data/runs/_box_<phase>_<lane>.log; `--dry-run` prints the schedule and every
child command.

Whole-line mode (`--runs [RUN_ID ...]`), the alternative: whole plan lines in cut order (minus API lines, E4 and a
line whose command duplicates an earlier line's), one directory `<run>__<model>` per line; the ledger counts such
a directory only for an untagged line (a split line's whole-line directory is mapped to its p0p1 / p2 units).
The confirmatory lines (chunks c08-c31) are printed and skipped as `skipped: gate: ...` until experiments/gates.yaml
is filled (CLAUDE.md "Binding documents and gates"); the `[summary]` line counts them apart as `gated`.

`--bootstrap` first installs the pinned vLLM (the version pyproject.toml's `vllm` extra pins; DESIGN_DECISIONS 7.3:
one engine version for the whole campaign), the project's eval extra and, with `--kaggle-dataset`, the `kaggle`
CLI, fetches the third-party resources and materializes the derived item files. Secrets (HF_TOKEN for gated
checkpoints, the Kaggle credentials) come from the environment and are never printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))              # the planner imports noilai lazily (census); a dry run precedes the install
import colab_setup as CS
import compute_log as CL
import kaggle_run_plan as KRP

CHUNKS_YAML = ROOT / "configs" / "compute_chunks.yaml"
BOX_QUEUES = ("gpu", "tpu")            # the CPU chunks (c01-c04) run the HF backend on a Kaggle CPU session, never on a box
CPU_REFUSAL = "CPU chunks run the HF backend on a CPU session; on a box run c05 (the T4 route of the pilot) instead"
GATED = "skipped: gate:"               # kaggle_run_plan.execute's status for a confirmatory line held by experiments/gates.yaml
DISK_HEADROOM_GB = 20.0                # a prefetch starts only when the estimates leave this much free
PREFETCH_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py")
PREFETCH_IGNORE = ("consolidated*", "original/*")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# the keys of a plan line that decide its run_eval command (kaggle_run_plan.build_command)
COMMAND_KEYS = ("items", "tasks", "variants", "arms", "paraphrases", "shots", "in_core_only", "limit",
                *(k for k, _ in KRP.ABLATION_FLAGS))


# ------------------------------------------------------------------ bootstrap
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


def bootstrap_commands(project: Path, python: str = sys.executable, vllm_version: str | None = None,
                       kaggle: bool = False) -> list[list[str]]:
    """Pinned vLLM first, then the project's eval extra (and the `kaggle` CLI when the dataset push is wanted), the
    resources, the derived item files."""
    version = vllm_version or vllm_pin(project / "pyproject.toml")
    cmds = CS.pip_install_command(project, extras=("eval",), pins={"vllm": version}, python=python)
    if kaggle:
        cmds.append([python, "-m", "pip", "install", "-q", "kaggle"])
    cmds.append([python, "scripts/fetch_resources.py"])
    cmds.append([python, "scripts/kaggle_run_plan.py", "--materialize"])
    return cmds


# ------------------------------------------------------------------ whole-line mode (the alternative)
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


# ------------------------------------------------------------------ chunk mode: the plan's units
def load_chunks(path: Path = CHUNKS_YAML) -> dict:
    """configs/compute_chunks.yaml (written by scripts/plan_chunks.py --write; docs/COMPUTE_PLAN.md)."""
    return KRP.load_yaml(path)


def _known_chunk(by_id: Mapping[str, dict], chunk_id: str) -> dict:
    if chunk_id not in by_id:
        raise ValueError(f"unknown chunk {chunk_id!r}; known: {' '.join(by_id)}")
    chunk = by_id[chunk_id]
    if chunk["queue"] not in BOX_QUEUES:
        raise ValueError(f"{chunk_id}: {CPU_REFUSAL}")
    return chunk


def select_chunks(spec: dict, selectors: Sequence[str]) -> list[dict]:
    """The chunks the selectors name, in chunk `order` (the reverse cut order, docs/COMPUTE_PLAN.md), each once:
    `all` (every gpu/tpu chunk), `tier<N>` (`tier0`: the pilot and the smoke week), `from:<id>` (every gpu/tpu chunk
    from that id onward), or a chunk id. An unknown id names the known ones, a tier no gpu/tpu chunk has the known tiers;
    a CPU chunk is refused."""
    ordered = sorted(spec["chunks"], key=lambda c: c["order"])
    box = [c for c in ordered if c["queue"] in BOX_QUEUES]
    by_id = {c["id"]: c for c in ordered}
    chosen: dict[str, dict] = {}
    for sel in selectors:
        if sel == "all":
            picked = box
        elif m := re.fullmatch(r"tier(\d+)", sel):
            picked = [c for c in box if c["tier"] == int(m.group(1))]
            if not picked:
                raise ValueError(f"{sel}: no gpu/tpu chunk has tier {m.group(1)}; tiers: {sorted({c['tier'] for c in box})}")
        elif sel.startswith("from:"):
            start = _known_chunk(by_id, sel[len("from:"):])
            picked = [c for c in box if c["order"] >= start["order"]]
        else:
            picked = [_known_chunk(by_id, sel)]
        for c in picked:
            chosen.setdefault(c["id"], c)
    return sorted(chosen.values(), key=lambda c: c["order"])


def job_models(job: dict, models: Sequence[str] | None = None) -> list[str]:
    """The job's models, intersected with `--models` when given (order of the job)."""
    if models is None:
        return list(job["models"])
    keep = set(models)
    return [m for m in job["models"] if m in keep]


def chunk_gated(chunk: dict, plan: dict, models_cfg: dict, gates: dict, models: Sequence[str] | None = None) -> bool:
    """Does the pre-registration gate (kaggle_run_plan.confirmatory_block_reason on `gates` = gate_status()) hold EVERY
    selected command of the chunk? False for a chunk without a selected command and when the gate file is absent."""
    if gates.get("present") is False:
        return False
    by = models_cfg["by_name"]
    n = 0
    for job in chunk["jobs"]:
        run = KRP.narrow_run(KRP.find_run(plan, job["run"]), job.get("overrides"), job.get("tag"))
        item_spec = plan["item_files"].get(run.get("items"), {})
        for m in job_models(job, models):
            n += 1
            if not KRP.confirmatory_block_reason(run, by[m], item_spec, gates):
                return False
    return n > 0


def repo_of(entry: dict) -> str | None:
    """The hub repository the runner loads (scripts/pin_panel.py `checkpoint_of`): `quantization.checkpoint` when the
    entry names a pre-quantized checkpoint, else `hf_id`; None for an API entry (no local weights)."""
    if KRP.is_api(entry):
        return None
    return (entry.get("quantization") or {}).get("checkpoint") or entry["hf_id"]


def command_plan(chunks: Sequence[dict], models_cfg: dict, models: Sequence[str] | None = None) -> list[dict]:
    """The ordered (chunk, run, model) commands of the selected chunks with the repository each loads."""
    by = models_cfg["by_name"]
    out: list[dict] = []
    for c in chunks:
        for job in c["jobs"]:
            for m in job_models(job, models):
                out.append({"chunk": c["id"], "run": job["run"], "model": m, "tag": job.get("tag"), "repo": repo_of(by[m])})
    return out


def purge_points(commands: Sequence[dict], keep: Sequence[str] = ()) -> list[str | None]:
    """The repository to purge after command i, or None. Chunk-scoped: a repository goes when no later command of the
    same chunk and no command of the NEXT chunk in the list uses it (so a smoke chunk purges each model right after
    its run, a four-line ablation chunk keeps its models until its last line, and a 27B model used again three
    chunks later is downloaded again: bounded disk beats cached weights). Repositories in `keep` never go (a lane's
    weights shared with another lane)."""
    chunk_order: list[str] = []
    uses: dict[str, set[str]] = {}
    for cmd in commands:
        if not chunk_order or chunk_order[-1] != cmd["chunk"]:
            chunk_order.append(cmd["chunk"])
        uses.setdefault(cmd["chunk"], set()).add(cmd["repo"])
    next_chunk = {cid: (chunk_order[i + 1] if i + 1 < len(chunk_order) else None) for i, cid in enumerate(chunk_order)}
    kept = set(keep)
    points: list[str | None] = []
    for i, cmd in enumerate(commands):
        repo = cmd["repo"]
        if repo is None or repo in kept:
            points.append(None)
            continue
        later = any(c["repo"] == repo for c in commands[i + 1:] if c["chunk"] == cmd["chunk"])
        nxt = next_chunk[cmd["chunk"]]
        points.append(None if later or (nxt is not None and repo in uses[nxt]) else repo)
    return points


# ------------------------------------------------------------------ the huggingface cache
def hf_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """The hub cache directory as huggingface_hub resolves it: HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE, else
    $HF_HOME/hub, else ~/.cache/huggingface/hub (XDG_CACHE_HOME replaces ~/.cache)."""
    env = os.environ if env is None else env
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return Path(env[key]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    cache = Path(env["XDG_CACHE_HOME"]).expanduser() if env.get("XDG_CACHE_HOME") else Path.home() / ".cache"
    return cache / "huggingface" / "hub"


def repo_folder(repo: str) -> str:
    """`models--<org>--<name>`: the cache folder of a repository."""
    return "models--" + repo.replace("/", "--")


def repo_cached(cache_dir: Path, repo: str) -> bool:
    return (Path(cache_dir) / repo_folder(repo)).is_dir()


def dir_size_gb(path: Path) -> float:
    """Bytes under `path` in GB; symlinks are not followed (the cache links snapshots to blobs)."""
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total / 1e9


def purge_repo(cache_dir: Path, repo: str) -> bool:
    """Remove the repository's cache folder (and its lock folder); True when something was removed."""
    folder = Path(cache_dir) / repo_folder(repo)
    if not folder.exists():
        return False
    gb = dir_size_gb(folder)
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print(f"[purge] {repo}: could not remove {folder} ({e})")
        return False
    locks = Path(cache_dir) / ".locks" / repo_folder(repo)
    if locks.exists():
        shutil.rmtree(locks, ignore_errors=True)
    print(f"[purge] {repo} ({gb:.1f} GB freed)")
    return True


def disk_free_gb(path: Path) -> float:
    """Free space of the file system holding `path` (its nearest existing parent when it does not exist yet)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


# ------------------------------------------------------------------ prefetch of the next chunk's weights
_PREFETCH_TEMPLATE = """\
import sys
from huggingface_hub import snapshot_download
allow, ignore = ALLOW, IGNORE
rc = 0
for target in sys.argv[1:]:
    repo, _, rev = target.partition("@")
    try:
        path = snapshot_download(repo, revision=rev or None, allow_patterns=allow, ignore_patterns=ignore)
        print("[prefetch] " + repo + " -> " + str(path), flush=True)
    except Exception as e:
        rc = 1
        print("[prefetch] " + repo + ": " + repr(e), flush=True)
sys.exit(rc)
"""
PREFETCH_SCRIPT = _PREFETCH_TEMPLATE.replace("ALLOW", repr(list(PREFETCH_PATTERNS))).replace("IGNORE", repr(list(PREFETCH_IGNORE)))


def prefetch_commands(repos: Sequence[str], revisions: Mapping[str, str | None], python: str = sys.executable) -> list[list[str]]:
    """One background process, sequential inside: `python -c` with huggingface_hub.snapshot_download (the same
    interpreter and cache as run_eval; stable across the hub library's CLI generations), each repository at its
    pinned commit (`<repo>@<40-hex>`) or the default branch, weights and tokenizer files only (PREFETCH_PATTERNS,
    minus PREFETCH_IGNORE). Empty when there is nothing to fetch."""
    targets = [f"{r}@{revisions[r]}" if _COMMIT.match(str(revisions.get(r) or "")) else r for r in repos]
    return [[python, "-c", PREFETCH_SCRIPT, *targets]] if targets else []


def prefetch_available(python: str = sys.executable) -> bool:
    """Can `python` import huggingface_hub (the prefetch runs snapshot_download there)?"""
    try:
        return subprocess.run([python, "-c", "import huggingface_hub"], check=False, capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _estimate_gb(entry: dict, session: KRP.CudaSession, models_cfg: dict) -> float:
    need = KRP.cuda_fit(entry, session, models_cfg)[1]
    return 0.0 if math.isnan(need) else need


def repo_estimate_gb(repo: str, commands: Sequence[dict], session: KRP.CudaSession, models_cfg: dict) -> float:
    """The weight estimate of a repository (kaggle_run_plan.cuda_fit: params x bytes x 1.1 + 1 GB; the largest over
    the entries of `commands` that load it, an over-estimate for an fp32 entry whose bf16 variant shares the folder)."""
    by = models_cfg["by_name"]
    return max((_estimate_gb(by[c["model"]], session, models_cfg) for c in commands if c["repo"] == repo), default=0.0)


def prefetch_plan(commands: Sequence[dict], chunk_id: str, next_id: str, models_cfg: dict, session: KRP.CudaSession,
                  cache_dir: Path, free_gb: float, headroom_gb: float = DISK_HEADROOM_GB) -> dict:
    """What a prefetch for `next_id` fetches while `chunk_id` runs: the next chunk's repositories that the current
    chunk does not use and the cache does not hold, each at its entry's pinned commit when there is one, with the
    disk guard: `need_gb` (the current chunk's repositories plus the fetched ones) must leave `headroom_gb`
    (DISK_HEADROOM_GB; a lane child gets it multiplied by the lane count) of `free_gb`, else `skip` says why."""
    by = models_cfg["by_name"]
    current = [c for c in commands if c["chunk"] == chunk_id]
    nxt = [c for c in commands if c["chunk"] == next_id]
    current_repos = {c["repo"] for c in current} - {None}
    repos: list[str] = []
    revisions: dict[str, str | None] = {}
    for c in nxt:
        repo = c["repo"]
        if repo is None or repo in current_repos or repo in repos or repo_cached(cache_dir, repo):
            continue
        repos.append(repo)
    for c in nxt:
        rev = by[c["model"]].get("revision")
        if c["repo"] in repos and c["repo"] not in revisions and _COMMIT.match(str(rev or "")):
            revisions[c["repo"]] = rev
    need = (sum(repo_estimate_gb(r, current, session, models_cfg) for r in sorted(current_repos))
            + sum(repo_estimate_gb(r, nxt, session, models_cfg) for r in repos))
    skip = None if need <= free_gb - headroom_gb else f"needs ~{need:.0f} GB, {free_gb:.0f} GB free"
    return {"repos": repos, "revisions": revisions, "need_gb": need, "free_gb": free_gb, "skip": skip}


# ------------------------------------------------------------------ the hooks after an executed model
class Background:
    """At most one subprocess in flight per slot (`sync`, `kaggle`, `prefetch`): starting a new one waits for the
    previous, a non-zero exit is printed and never fatal, `wait_all` catches the last ones at the end of the run.
    `repos` names the repositories the in-flight prefetch downloads (a purge of one of them waits for it first)."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.repos: set[str] = set()

    def start(self, slot: str, cmd: Sequence[str], cwd: Path, log: Path | None = None,
              env: Mapping[str, str] | None = None, label: str = "") -> subprocess.Popen:
        self.wait(slot)
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as fh:
                proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
        print(f"[{slot:<5}] started" + (f": {label}" if label else ""))
        self.procs[slot] = proc
        return proc

    def wait(self, slot: str) -> int | None:
        proc = self.procs.pop(slot, None)
        if proc is None:
            return None
        rc = proc.wait()
        print(f"[{slot:<5}] rc={rc}")
        if slot == "prefetch":
            self.repos = set()
        return rc

    def wait_all(self) -> None:
        for slot in list(self.procs):
            self.wait(slot)


def check_command(out_dir: str | Path, python: str = sys.executable) -> list[str]:
    """scripts/check_run.py on one run directory (item gate, deterministic rescoring, stats, results_hashes.json)."""
    return [python, "scripts/check_run.py", "--run", str(out_dir)]


def run_check(out_dir: str | Path, project_root: Path, python: str = sys.executable, tail: int = 5) -> bool:
    """check_run.py in the foreground; the last lines are printed, a failure is reported and never stops the loop."""
    proc = subprocess.run(check_command(out_dir, python), cwd=str(project_root), check=False, capture_output=True, text=True)
    for ln in (proc.stdout + proc.stderr).strip().splitlines()[-tail:]:
        print(f"[check] {ln}")
    if proc.returncode != 0:
        print(f"[check] FAILED {out_dir} (rc={proc.returncode})")
    return proc.returncode == 0


def kaggle_push_command(project_root: Path, staging: Path | None, slug: str, message: str, create: bool = False,
                        python: str = sys.executable) -> list[str]:
    """scripts/kaggle_dataset.py staging data/runs and data/compute_log.csv (relative to the project root, its cwd)
    and pushing a version of the private dataset `slug` (`--create` for the first version)."""
    staging = Path(staging) if staging else default_staging(project_root)
    cmd = [python, "scripts/kaggle_dataset.py", "--src", "data/runs", "--src", "data/compute_log.csv",
           "--dest", str(staging), "--slug", slug, "--message", message]
    return [*cmd, "--create"] if create else cmd


def default_staging(project_root: Path) -> Path:
    """The dataset staging directory: beside the repository, never inside it."""
    return Path(project_root).resolve().parent / "noilai_runs_dataset"


def kaggle_credentials(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("KAGGLE_USERNAME")) and bool(env.get("KAGGLE_KEY"))


def git_head(project_root: Path) -> str:
    """`git rev-parse HEAD` of the checkout, or `nogit`."""
    try:
        return subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def status_counts(results: Sequence[dict]) -> dict:
    """The tallies of the `[summary]` line; `gated` counts the commands experiments/gates.yaml holds, on a dry run too
    (kaggle_run_plan.execute leaves their status `dry-run` and records the reason under `prereg_gate`)."""
    status = [str(r["status"]) for r in results]
    return {"commands": len(status), "ok": sum(s == "ok" for s in status), "failed": sum(s.startswith("failed") for s in status),
            "refused": sum(s.startswith(KRP.REFUSED) for s in status), "skipped": sum(s.startswith("skipped") for s in status),
            "gated": sum(s.startswith(GATED) or (s == "dry-run" and bool(r.get("prereg_gate")))
                         for r, s in zip(results, status, strict=True))}


def summary_line(results: Sequence[dict], prefix: str = "[summary]") -> str:
    n = status_counts(results)
    return (f"{prefix} {n['commands']} commands, {n['failed']} failed, {n['refused']} refused, {n['skipped']} skipped, "
            f"{n['gated']} gated")


def run_chunks(chunks: Sequence[dict], session: KRP.CudaSession, platform: str = "rented", plan: dict | None = None,
               models_cfg: dict | None = None, project_root: Path = ROOT, log_path: Path | None = None,
               models: Sequence[str] | None = None, verify: bool = True, dry_run: bool = False,
               allow_unpinned_revision: bool = False, max_hours: float | None = None, extra: Sequence[str] = (),
               python: str = sys.executable, check: bool = True, sync_cmd: str | None = None,
               kaggle_dataset: str | None = None, create_dataset: bool = False, staging: Path | None = None,
               purge: bool = True, prefetch: bool = True, keep_repos: Sequence[str] = (), cache_dir: Path | None = None,
               free_gb: float | None = None, headroom_gb: float = DISK_HEADROOM_GB) -> list[dict]:
    """Every job of every chunk, in order, through kaggle_run_plan.execute with the job's models, overrides and tag
    (the ledger's units). After each executed model: check_run, the sync command, the Kaggle push (both in the
    background), then the chunk-scoped purge (`purge_points`); while a chunk runs, the next chunk's weights are
    prefetched (`prefetch_plan`, `headroom_gb` of free disk kept) unless the gate holds every command of that chunk
    (`chunk_gated`) or the `max_hours` budget is spent: nothing would run after the download. A dry run executes,
    purges and fetches nothing and prints what it would do. `free_gb` and `cache_dir` are injectable for tests; every
    result dict carries `chunk`."""
    plan = plan or KRP.load_plan()
    models_cfg = models_cfg or KRP.load_models()
    project_root = Path(project_root)
    cache_dir = Path(cache_dir) if cache_dir else hf_cache_dir()
    commands = command_plan(chunks, models_cfg, models)
    points = purge_points(commands, keep=keep_repos)
    handled = [False] * len(commands)
    bg = Background()
    t0 = time.time()
    state = {"create": create_dataset, "push": bool(kaggle_dataset), "check_said": False, "prefetch_said": False}
    if kaggle_dataset and not dry_run and not kaggle_credentials():
        print("[kaggle] KAGGLE_USERNAME / KAGGLE_KEY are not set: no push to the dataset (outputs stay under data/runs)")
        state["push"] = False
    head = git_head(project_root) if kaggle_dataset else "nogit"
    check_script = project_root / "scripts" / "check_run.py"
    gates = KRP.gate_status(project_root)                     # {"present": False} without experiments/gates.yaml

    def settle(i: int, res: dict) -> None:            # the purge point of command i (executed or not)
        handled[i] = True
        repo = points[i]
        if repo is None or not purge:
            return
        if dry_run:
            print(f"[purge] (dry-run) {repo}")
            return
        if repo in bg.repos:
            bg.wait("prefetch")
        if purge_repo(cache_dir, repo):
            res["purged"] = repo

    def after_each(i: int, res: dict) -> None:        # an executed model: check, sync, push, purge
        if check and res["status"] == "ok":
            if check_script.exists():
                res["checked"] = run_check(res["out"], project_root, python)
            else:
                res["checked"] = None
                if not state["check_said"]:
                    print(f"[check] {check_script} not found: check_run skipped")
                    state["check_said"] = True
        if sync_cmd:
            bg.start("sync", shlex.split(sync_cmd), cwd=project_root, label=sync_cmd)
        if state["push"]:
            message = f"{res['chunk']}/{res['run']}/{res['model']} {res['status']} @ {head[:8]}"
            bg.start("kaggle", kaggle_push_command(project_root, staging, kaggle_dataset, message, create=state["create"], python=python),
                     cwd=project_root, label=message)
            state["create"] = False
        settle(i, res)

    def start_prefetch(chunk: dict, nxt: dict) -> None:
        if max_hours is not None and CL.hours_since(t0) >= max_hours:
            print(f"[prefetch] skipped for {nxt['id']}: session budget of {max_hours} h reached")
            return
        if chunk_gated(nxt, plan, models_cfg, gates, models):
            print(f"[prefetch] skipped for {nxt['id']}: every command gated ({KRP.GATES_FILE})")
            return
        free = disk_free_gb(cache_dir) if free_gb is None else free_gb
        want = prefetch_plan(commands, chunk["id"], nxt["id"], models_cfg, session, cache_dir, free, headroom_gb)
        if not want["repos"]:
            return
        if want["skip"]:
            print(f"[prefetch] skipped for {nxt['id']}: {want['skip']}")
            return
        if dry_run:
            print(f"[prefetch] (dry-run) would fetch for {nxt['id']}: {' '.join(want['repos'])} (~{want['need_gb']:.0f} GB with this chunk)")
            return
        if not prefetch_available(python):
            if not state["prefetch_said"]:
                print(f"[prefetch] {python} cannot import huggingface_hub: no prefetch (the model load downloads what is missing)")
                state["prefetch_said"] = True
            return
        cmd = prefetch_commands(want["repos"], want["revisions"], python)[0]
        bg.repos = set(want["repos"])
        bg.start("prefetch", cmd, cwd=project_root, log=project_root / plan.get("runs_root", "data/runs") / "_prefetch.log",
                 label=f"{nxt['id']}: {' '.join(want['repos'])}")

    results: list[dict] = []
    cursor = 0
    for ci, chunk in enumerate(chunks):
        bg.wait("prefetch")                                # the previous prefetch lands before this chunk starts
        jobs = [(job, names) for job in chunk["jobs"] if (names := job_models(job, models))]
        print(f"\n[chunk] {chunk['id']} (order {chunk['order']}, tier {chunk['tier']}, queue {chunk['queue']}): {len(jobs)} jobs")
        if prefetch and ci + 1 < len(chunks):
            start_prefetch(chunk, chunks[ci + 1])
        chunk_results: list[dict] = []
        for job, names in jobs:
            base = cursor
            cursor += len(names)

            def hook(res: dict, base: int = base, names: Sequence[str] = names, chunk_id: str = chunk["id"]) -> None:
                res["chunk"] = chunk_id
                after_each(base + list(names).index(res["model"]), res)

            res = KRP.execute(job["run"], models=names, line_overrides=job.get("overrides"), chunk_tag=job.get("tag"),
                              session_hardware=session, platform=platform, after_each=None if dry_run else hook, verify=verify,
                              dry_run=dry_run, session_t0=t0, max_session_hours=max_hours,
                              allow_unpinned_revision=allow_unpinned_revision, extra=extra, plan=plan, models_cfg=models_cfg,
                              project_root=project_root, log_path=log_path, python=python)
            for k, r in enumerate(res):
                r["chunk"] = chunk["id"]
                if not handled[base + k]:                  # skipped, refused or dry-run: the purge point still holds
                    settle(base + k, r)
            chunk_results += res
        print(summary_line(chunk_results, prefix=f"[chunk] {chunk['id']} done:"))
        results += chunk_results
    bg.wait_all()
    return results


# ------------------------------------------------------------------ lanes: the driver schedules the cards
def one_card(session: KRP.CudaSession) -> KRP.CudaSession:
    return KRP.CudaSession(session.gpu_type, session.vram_gb, 1, session.bf16)


def chunk_is_big(chunk: dict, session: KRP.CudaSession, models_cfg: dict, models: Sequence[str] | None = None) -> bool:
    """Does any model of the chunk need more than one card (kaggle_run_plan.cuda_fit on a one-card session)?"""
    card = one_card(session)
    by = models_cfg["by_name"]
    return any(not KRP.cuda_fit(by[m], card, models_cfg)[0] for job in chunk["jobs"] for m in job_models(job, models))


def balance_lanes(chunks: Sequence[dict], n_lanes: int) -> list[list[dict]]:
    """Greedy by `hours.high`, largest first, onto the least-loaded lane; each lane back in chunk order; empty lanes dropped."""
    loads = [0.0] * n_lanes
    lanes: list[list[dict]] = [[] for _ in range(n_lanes)]
    for c in sorted(chunks, key=lambda c: -float(c["hours"]["high"])):
        k = min(range(n_lanes), key=lambda i: (loads[i], i))
        lanes[k].append(c)
        loads[k] += float(c["hours"]["high"])
    return [sorted(lane, key=lambda c: c["order"]) for lane in lanes if lane]


def shared_repos(lanes: Sequence[Sequence[dict]], models_cfg: dict, models: Sequence[str] | None = None) -> list[str]:
    """Repositories that chunks of more than one lane load: a lane never purges them (the other lane may be loading)."""
    by = models_cfg["by_name"]
    per_lane = [{repo for c in lane for job in c["jobs"] for m in job_models(job, models) if (repo := repo_of(by[m]))} for lane in lanes]
    counts = Counter(r for s in per_lane for r in s)
    return sorted(r for r, n in counts.items() if n > 1)


def schedule(chunks: Sequence[dict], session: KRP.CudaSession, models_cfg: dict, n_lanes: int,
             models: Sequence[str] | None = None) -> list[dict]:
    """Phases in run order. Tiers are barriers (a tier's phases finish before the next tier starts: the cut order
    of DESIGN_DECISIONS 7.1 / 8.5 holds and weights shared across tiers are harmless). Inside a tier, with N >= 2
    lanes on a >= 2-card session: the chunks whose every model fits one card are balanced over the lanes
    (`{"kind": "lanes", "tier", "lanes": [[ids], ...], "shared": [repos]}`), then the chunks with a model that needs
    every card run one at a time (`{"kind": "exclusive", "tier", "chunks": [ids]}`); either phase is omitted when
    empty. With one lane or one card: one exclusive phase per tier with every chunk."""
    tiers: list[int] = []
    for c in chunks:
        if c["tier"] not in tiers:
            tiers.append(c["tier"])
    phases: list[dict] = []
    for t in tiers:
        group = [c for c in chunks if c["tier"] == t]
        if n_lanes < 2 or session.n_gpus < 2:
            phases.append({"kind": "exclusive", "tier": t, "chunks": [c["id"] for c in group]})
            continue
        big = [c for c in group if chunk_is_big(c, session, models_cfg, models)]
        big_ids = {c["id"] for c in big}
        small = [c for c in group if c["id"] not in big_ids]
        if small:
            lanes = balance_lanes(small, n_lanes)
            phases.append({"kind": "lanes", "tier": t, "lanes": [[c["id"] for c in lane] for lane in lanes],
                           "shared": shared_repos(lanes, models_cfg, models)})
        if big:
            phases.append({"kind": "exclusive", "tier": t, "chunks": [c["id"] for c in big]})
    return phases


def lane_command(chunk_ids: Sequence[str], session: KRP.CudaSession, n_gpus: int, platform: str = "rented",
                 python: str = sys.executable, check: bool = True, keep_weights: bool = False, prefetch: bool = True,
                 verify: bool = True, allow_unpinned_revision: bool = False, extra: str = "",
                 models: Sequence[str] | None = None, log: Path | None = None, max_hours: float | None = None,
                 keep_repos: Sequence[str] = (), sync_cmd: str | None = None, kaggle_dataset: str | None = None,
                 create_dataset: bool = False, staging: Path | None = None, headroom_gb: float | None = None,
                 script: str = "scripts/cuda_box.py") -> list[str]:
    """The argv of one child of this script: a one-card session (`<type>:<vram>x1`, with CUDA_VISIBLE_DEVICES=k in its
    environment) for a lane, the full session for an exclusive phase; the parent's flags forwarded, `--lanes 1`;
    `headroom_gb` (the prefetch disk guard of a lane child) when given."""
    spec = f"{session.gpu_type}:{session.vram_gb:g}x{n_gpus}" + ("" if session.bf16 else ":nobf16")
    cmd = [python, script, "--session", spec, "--chunks", *chunk_ids, "--platform", platform, "--lanes", "1"]
    if not check:
        cmd.append("--no-check")
    if keep_weights:
        cmd.append("--keep-weights")
    if not prefetch:
        cmd.append("--no-prefetch")
    if not verify:
        cmd.append("--no-verify")
    if allow_unpinned_revision:
        cmd.append("--allow-unpinned-revision")
    if extra:
        cmd += ["--extra", extra]
    if models is not None:
        cmd += ["--models", *models]
    if log is not None:
        cmd += ["--log", str(log)]
    cmd += ["--python", python]
    if max_hours is not None:
        cmd += ["--max-hours", f"{max_hours:.3f}"]
    if keep_repos:
        cmd += ["--keep-repos", *keep_repos]
    if headroom_gb is not None:
        cmd += ["--disk-headroom-gb", f"{headroom_gb:g}"]
    if sync_cmd:
        cmd += ["--sync-cmd", sync_cmd]
    if kaggle_dataset:
        cmd += ["--kaggle-dataset", kaggle_dataset]
        if create_dataset:
            cmd.append("--create-dataset")
        if staging is not None:
            cmd += ["--staging", str(staging)]
    return cmd


def last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1] if lines else ""


def run_lanes(chunks: Sequence[dict], session: KRP.CudaSession, models_cfg: dict, n_lanes: int, platform: str = "rented",
              project_root: Path = ROOT, python: str = sys.executable, dry_run: bool = False, max_hours: float | None = None,
              models: Sequence[str] | None = None, kaggle_dataset: str | None = None, create_dataset: bool = False,
              staging: Path | None = None, sync_cmd: str | None = None, runs_root: str = "data/runs",
              keep_repos: Sequence[str] = (), headroom_gb: float = DISK_HEADROOM_GB, **forward) -> int:
    """The parent of `--lanes N`: runs `schedule`'s phases with child processes of this script (one per lane with
    CUDA_VISIBLE_DEVICES=k, or one exclusive child with the full session), waits for each phase, prints each child's
    last log line. `--kaggle-dataset` (and `--sync-cmd`) go to lane 0 of a lanes-phase and to every exclusive child
    (one pusher at a time; lane 0's pushes carry the whole data/runs), `--create-dataset` only to the first child.
    Every child keeps the operator's `keep_repos` (a lane child the phase's shared repositories on top); a lane child's
    prefetch head-room is `headroom_gb` times the lanes of its phase (the other lanes' downloads are not in its view of
    the free space). `forward` holds lane_command's remaining flags. Returns 1 when a child exited non-zero; the final
    Kaggle push is printed, never fatal."""
    phases = schedule(chunks, session, models_cfg, n_lanes, models)
    project_root = Path(project_root)
    log_dir = project_root / runs_root
    t0 = time.time()
    create = create_dataset
    worst = 0
    for phase in phases:
        tier = phase["tier"]
        if phase["kind"] == "lanes":
            desc = " ".join(f"{chr(65 + k)}=[{', '.join(ids)}]" for k, ids in enumerate(phase["lanes"]))
            print(f"\n[phase] tier {tier} lanes: {desc}" + (f" (shared: {', '.join(phase['shared'])})" if phase["shared"] else ""))
        else:
            print(f"\n[phase] tier {tier} exclusive: [{', '.join(phase['chunks'])}]")
        remaining = None if max_hours is None else max_hours - CL.hours_since(t0)
        if remaining is not None and remaining <= 0:
            print(f"[phase] session budget of {max_hours} h reached: no more phases")
            break
        children: list[tuple[str, str, list[str], dict | None]] = []
        if phase["kind"] == "lanes":
            keep = sorted({*phase["shared"], *keep_repos})
            for k, ids in enumerate(phase["lanes"]):
                pusher = k == 0
                cmd = lane_command(ids, session, 1, platform=platform, python=python, models=models, max_hours=remaining,
                                   keep_repos=keep, sync_cmd=sync_cmd if pusher else None,
                                   kaggle_dataset=kaggle_dataset if pusher else None, create_dataset=create and pusher,
                                   staging=staging, headroom_gb=headroom_gb * len(phase["lanes"]), **forward)
                children.append((f"t{tier}-lanes", str(k), cmd, {**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}))
        else:
            cmd = lane_command(phase["chunks"], session, session.n_gpus, platform=platform, python=python, models=models,
                               max_hours=remaining, keep_repos=keep_repos, sync_cmd=sync_cmd, kaggle_dataset=kaggle_dataset,
                               create_dataset=create, staging=staging, **forward)
            children.append((f"t{tier}-exclusive", "0", cmd, None))
        procs: list[tuple[str, subprocess.Popen, Path]] = []
        for label, lane, cmd, env in children:
            log = log_dir / f"_box_{label}_{lane}.log"
            print(f"[lane {lane}] {shlex.join(cmd)}\n[lane {lane}] log: {log}")
            if "--create-dataset" in cmd:
                create = False
            if dry_run:
                continue
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as fh:
                procs.append((lane, subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=fh, stderr=subprocess.STDOUT), log))
        for lane, proc, log in procs:
            rc = proc.wait()
            print(f"[lane {lane}] rc={rc}: {last_line(log)}")
            worst = worst or rc
    if kaggle_dataset and not dry_run and phases and kaggle_credentials():
        cmd = kaggle_push_command(project_root, staging, kaggle_dataset, f"box: every phase done @ {git_head(project_root)[:8]}", python=python)
        print(f"[kaggle] final push: {shlex.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(project_root), check=False).returncode
        print(f"[kaggle] final push rc={rc}")              # printed, never fatal: exit 1 means a child (a model) failed
    return 1 if worst else 0


# ------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="auto", metavar="SPEC",
                    help="<gpu_type>:<vram_gb>[x<n_gpus>][:nobf16], or auto (nvidia-smi, CUDA_VISIBLE_DEVICES-aware); default auto")
    ap.add_argument("--platform", default="rented", choices=CL.PLATFORMS, help="compute-log platform (default rented)")
    ap.add_argument("--bootstrap", action="store_true", help="install the pinned vLLM and the project, fetch resources, materialize")
    ap.add_argument("--chunks", nargs="+", default=None, metavar="ID",
                    help="compute chunks (configs/compute_chunks.yaml): ids, all (every gpu/tpu chunk in order; the default), "
                         "tier0, from:<id>")
    ap.add_argument("--runs", nargs="*", default=None, metavar="RUN_ID",
                    help="whole-line mode instead of chunks: run ids in order (no id: the plan's cut order)")
    ap.add_argument("--models", nargs="*", default=None, help="restrict every job / run to these models.yaml entries")
    ap.add_argument("--lanes", type=int, default=1, help="run the box's cards in parallel: N children, one per card (default 1)")
    ap.add_argument("--sync-cmd", default=None, help="run from the project root after every executed model (background)")
    ap.add_argument("--kaggle-dataset", default=None, metavar="SLUG",
                    help="push data/runs and the compute log to this private Kaggle dataset after every executed model")
    ap.add_argument("--create-dataset", action="store_true", help="with --kaggle-dataset: the first push creates the dataset")
    ap.add_argument("--staging", type=Path, default=None, help="dataset staging directory (default <root>/../noilai_runs_dataset)")
    ap.add_argument("--keep-weights", action="store_true", help="never purge the huggingface cache")
    ap.add_argument("--keep-repos", nargs="*", default=[], metavar="REPO", help="never purge these repositories (a lane's shared weights)")
    ap.add_argument("--no-prefetch", action="store_true", help="do not prefetch the next chunk's weights")
    ap.add_argument("--disk-headroom-gb", type=float, default=DISK_HEADROOM_GB, metavar="GB",
                    help=f"free disk a prefetch must leave (default {DISK_HEADROOM_GB:g}; a lane child gets it times the lanes)")
    ap.add_argument("--no-check", action="store_true", help="do not run scripts/check_run.py after each model")
    ap.add_argument("--max-hours", type=float, default=None, help="launch no new model past this budget")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the item-file gate (rehearsals only)")
    ap.add_argument("--allow-unpinned-revision", action="store_true", help="rehearsals only (DD 7.1)")
    ap.add_argument("--extra", default="", help="extra run_eval.py args, one shell-quoted string")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log", type=Path, default=None, help="compute log path (default from the plan)")
    args = ap.parse_args(argv)
    if args.runs is not None and args.chunks is not None:
        ap.error("--runs and --chunks are mutually exclusive")
    if args.lanes < 1:
        ap.error("--lanes must be >= 1")
    if args.models is not None and not args.models:
        ap.error("--models needs at least one entry")

    try:
        session = KRP.detect_cuda_session() if args.session == "auto" else KRP.parse_cuda_session(args.session)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        ap.error(f"cannot describe the CUDA session ({e!r}); pass --session <gpu_type>:<vram_gb>[x<n_gpus>][:nobf16] "
                 f"or make nvidia-smi available")
    print(f"[session] {session}")
    if args.bootstrap:
        for cmd in bootstrap_commands(ROOT, python=args.python, kaggle=bool(args.kaggle_dataset)):
            print(f"[boot ] {shlex.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(ROOT), check=True)
    plan = KRP.load_plan()
    models_cfg = KRP.load_models()

    if args.runs is not None:                     # whole-line mode: PR #5 semantics, the sync in the foreground
        for flag in ("kaggle_dataset", "no_check"):
            if getattr(args, flag):
                print(f"[note ] --{flag.replace('_', '-')} applies to chunk mode only; ignored with --runs")
        run_ids = args.runs or default_run_ids(plan, models_cfg)
        print(f"[runs ] {' '.join(run_ids)}")
        results = run_all(run_ids, session, platform=args.platform, plan=plan, models_cfg=models_cfg, log_path=args.log,
                          models=args.models, sync_cmd=args.sync_cmd, verify=not args.no_verify, dry_run=args.dry_run,
                          allow_unpinned_revision=args.allow_unpinned_revision, max_hours=args.max_hours,
                          extra=shlex.split(args.extra), python=args.python)
        print("\n" + summary_line(results))
        n = status_counts(results)
        return 1 if n["failed"] or n["refused"] else 0

    try:
        chunks = select_chunks(load_chunks(), args.chunks or ["all"])
    except ValueError as e:
        ap.error(str(e))
    if not chunks:
        ap.error("no chunk selected")
    print(f"[chunks] {' '.join(c['id'] for c in chunks)}")
    n_lanes = args.lanes
    if n_lanes >= 2 and session.n_gpus < 2:
        print(f"[note ] --lanes {args.lanes} needs >= 2 cards; {session} has one: one process")
    elif n_lanes > session.n_gpus:
        print(f"[note ] --lanes {args.lanes} > {session.n_gpus} cards: using {session.n_gpus} lanes")
        n_lanes = session.n_gpus
    if n_lanes >= 2 and session.n_gpus >= 2:
        return run_lanes(chunks, session, models_cfg, n_lanes, platform=args.platform, project_root=ROOT, python=args.python,
                         dry_run=args.dry_run, max_hours=args.max_hours, models=args.models, kaggle_dataset=args.kaggle_dataset,
                         create_dataset=args.create_dataset, staging=args.staging, sync_cmd=args.sync_cmd,
                         runs_root=plan.get("runs_root", "data/runs"), keep_repos=args.keep_repos, headroom_gb=args.disk_headroom_gb,
                         check=not args.no_check, keep_weights=args.keep_weights, prefetch=not args.no_prefetch,
                         verify=not args.no_verify, allow_unpinned_revision=args.allow_unpinned_revision, extra=args.extra, log=args.log)
    results = run_chunks(chunks, session, platform=args.platform, plan=plan, models_cfg=models_cfg, project_root=ROOT,
                         log_path=args.log, models=args.models, verify=not args.no_verify, dry_run=args.dry_run,
                         allow_unpinned_revision=args.allow_unpinned_revision, max_hours=args.max_hours,
                         extra=shlex.split(args.extra), python=args.python, check=not args.no_check, sync_cmd=args.sync_cmd,
                         kaggle_dataset=args.kaggle_dataset, create_dataset=args.create_dataset, staging=args.staging,
                         purge=not args.keep_weights, prefetch=not args.no_prefetch, keep_repos=args.keep_repos,
                         headroom_gb=args.disk_headroom_gb)
    print("\n" + summary_line(results))
    n = status_counts(results)
    return 1 if n["failed"] or n["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
