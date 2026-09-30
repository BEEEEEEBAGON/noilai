#!/usr/bin/env python
"""Build the four cloud notebooks under notebooks/ from one source of truth.

    python scripts/kaggle_build_notebooks.py --write     # (re)write notebooks/*.ipynb
    python scripts/kaggle_build_notebooks.py --check     # exit 1 if the .ipynb files drifted from this file

Why a generator: the Kaggle T4, Kaggle TPU, Colab probe and API notebooks share their
bootstrap (parameters, environment check, token-safe clone, pinned install, resource and
item-file verification, output hand-off, compute log). Defining each cell once keeps the
four in step and lets tests/test_cloud.py parse every code cell as Python: no IPython
magics, no input(), no secret literals. Notebook edits are code changes: edit THIS file,
run --write, commit both; the drift test refuses a hand-edited .ipynb.

Cell conventions
  * every code cell is plain Python (subprocess instead of `!`, os.chdir instead of `%cd`);
  * the first code cell is the PARAMETERS cell and the only one meant to be edited on the
    platform; nothing prompts (no input(), no getpass);
  * cells that launch model runs are tagged "run" and headed "RUN CELL"; they call
    scripts/kaggle_run_plan.py, which owns the run_eval.py flag mapping;
  * secrets are read from Kaggle Secrets / Colab userdata / the environment into os.environ
    and reported as found/not found, never printed;
  * every Vietnamese example a cell uses is COMPUTED by the rule engine from the syllables in
    E4_PATCH_PAIR (DESIGN_DECISIONS 9.3), never typed as a literal output; tests/test_cloud.py
    re-derives the pair with noilai.gen.variants and checks legality;
  * the inference engine is installed as DESIGN_DECISIONS 7.3 says (vLLM 0.30.0 pins torch 2.13
    while Kaggle ships torch 2.10: a venv under /kaggle/tmp, or kaggle-vllm), the version
    staying a flagged parameter until the smoke tests pin it; every run_eval.py process is
    launched with the engine's interpreter (RUN_PYTHON);
  * a notebook never rewrites a file under configs/: session overrides (GEMINI_RPD_OVERRIDE)
    are applied to the in-memory models_cfg the driver receives and recorded in the ledger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = ("kaggle_eval_t4", "kaggle_eval_tpu", "colab_probe_gemma3", "api_runs")

RUN_TAG = "run"
E4_TAG = "e4"

# The activation-patching pair of DESIGN_DECISIONS 9.3 (readout B): the varying syllable is the
# SECOND input syllable, so that under V3 (swap tones) the two gold answers diverge at their
# FIRST syllable (bí mà -> bì má vs bí mạ -> bị má; pseudo-phrases, medium, [NATIVE-CHECK]).
# The plan's bí mật / bí mất is NOT a legal V3 pair (equal tones make V3 the identity) and is the
# V4 pair (bí mật -> bật mí). The cell recomputes every output with noilai.gen.variants; the
# answers below are what the test expects the engine to produce.
E4_PATCH_PAIR = {"variant": "V3", "partner": "bí", "clean": "mà", "corrupt": "mạ",
                 "answer_clean": "bì má", "answer_corrupt": "bị má", "source": "DESIGN_DECISIONS 9.3"}


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(dedent(text).strip("\n"))


def code(text: str, tags: list[str] | None = None) -> nbformat.NotebookNode:
    cell = new_code_cell(dedent(text).strip("\n"))
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


# =============================================================================== shared cells
SECRET_HELPER = '''
# secrets: environment -> Colab userdata -> Kaggle Secrets. Values go into os.environ for the
# library that needs them and are reported as found / not found. Nothing here prints a value.
import os

def _secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        from google.colab import userdata          # Colab
        try:
            val = userdata.get(name)
        except Exception:
            val = None
        if val:
            return val
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient   # Kaggle
        try:
            val = UserSecretsClient().get_secret(name)
        except Exception:
            val = None
        if val:
            return val
    except ImportError:
        pass
    return None

def _export(name, required=False):
    val = _secret(name)
    if val:
        os.environ[name] = val
        print(f"secret {name}: found ({len(val)} chars, not shown)")
        return True
    print(f"secret {name}: not found")
    if required:
        raise SystemExit(f"{name} is required for this notebook")
    return False
'''

GPU_CHECK = '''
# (a) environment check: interpreter, GPU(s), CUDA, vLLM's compute-capability floor (7.5), disk
import json, shutil, subprocess, sys
print(sys.version.split()[0], "on", sys.platform)
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not found: is a GPU accelerator attached?")
ENV_CHECK = {"n_gpus": 0, "devices": [], "vllm_ok": False}
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        ENV_CHECK["devices"].append({"name": torch.cuda.get_device_name(i), "capability": f"{cap[0]}.{cap[1]}", "memory_gb": round(mem, 1)})
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}  capability {cap[0]}.{cap[1]}  {mem:.1f} GB")
    ENV_CHECK["n_gpus"] = torch.cuda.device_count()
    ENV_CHECK["vllm_ok"] = ENV_CHECK["n_gpus"] > 0 and all(
        tuple(int(x) for x in d["capability"].split(".")) >= (7, 5) for d in ENV_CHECK["devices"])
except ImportError:
    print("torch is not importable yet (installed in the install cell)")
if REQUIRE_GPU and ENV_CHECK["n_gpus"] == 0:
    raise SystemExit("no CUDA device: choose the GPU accelerator (Kaggle: 'GPU T4 x2') and restart")
if ENV_CHECK["n_gpus"] and not ENV_CHECK["vllm_ok"]:
    print("WARNING: a device is below compute capability 7.5 (P100?) -> vLLM cannot run; use backend hf / llama.cpp")
du = shutil.disk_usage(WORK_DIR)
print(f"disk at {WORK_DIR}: {du.free / 2**30:.1f} GB free of {du.total / 2**30:.1f} GB")
print(json.dumps(ENV_CHECK))
'''

TPU_CHECK = '''
# (a) environment check: TPU v5e-8 through JAX / tpu-info; the vLLM-TPU startup takes 6-22 minutes on top
import json, shutil, subprocess, sys
print(sys.version.split()[0], "on", sys.platform)
ENV_CHECK = {"tpu_devices": 0, "device_kind": None}
try:
    import jax
    devs = jax.devices()
    ENV_CHECK["tpu_devices"] = len(devs)
    ENV_CHECK["device_kind"] = getattr(devs[0], "device_kind", str(devs[0])) if devs else None
    print("jax", jax.__version__, "devices:", [str(d) for d in devs])
except Exception as e:  # jax missing or no TPU runtime
    print("jax/TPU not visible from here:", repr(e))
if shutil.which("tpu-info"):
    print(subprocess.run(["tpu-info"], capture_output=True, text=True).stdout)
if REQUIRE_TPU and ENV_CHECK["tpu_devices"] == 0:
    raise SystemExit("no TPU device: choose 'TPU VM v5e-8' in the Kaggle accelerator menu and restart")
du = shutil.disk_usage(WORK_DIR)
print(f"disk at {WORK_DIR}: {du.free / 2**30:.1f} GB free of {du.total / 2**30:.1f} GB")
print(json.dumps(ENV_CHECK))
'''

CLONE = '''
# (b) clone: from a git bundle in a private dataset / Drive when present (no token needed), else from
# GitHub with a token from the platform's secret store. The token reaches git only through GIT_ASKPASS,
# a two-line helper that echoes an environment variable: it is never in a URL, an argv or this output.
import os, stat, subprocess, sys
from pathlib import Path

clone_dir = Path(CLONE_DIR)
env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
bundle = Path(BUNDLE_PATH) if BUNDLE_PATH else None
if bundle is not None and not bundle.exists():
    print(f"bundle {bundle} not found; falling back to {REPO_URL}")
    bundle = None
if bundle is None:
    if "OWNER/REPO" in REPO_URL:
        raise SystemExit("set REPO_URL (or provide BUNDLE_PATH) in the parameters cell")
    if _export(GITHUB_TOKEN_SECRET):
        askpass = Path.home() / ".noilai_askpass.sh"
        askpass.write_text("#!/bin/sh\\necho \\"$" + GITHUB_TOKEN_SECRET + "\\"\\n")
        askpass.chmod(stat.S_IRWXU)
        env["GIT_ASKPASS"] = str(askpass)
source = str(bundle) if bundle else REPO_URL
if (clone_dir / ".git").exists():
    subprocess.run(["git", "-C", str(clone_dir), "fetch", "--tags", source, "+refs/heads/*:refs/remotes/origin/*"], check=True, env=env)
else:
    subprocess.run(["git", "clone", source, str(clone_dir)], check=True, env=env)
# a persisted clone may hold a STALE local branch named like REPO_REF: try the freshly fetched
# remote-tracking ref first and the literal ref last (commit hashes and tags have no remote form)
import re
cands = [REPO_REF] if re.fullmatch(r"[0-9a-f]{7,40}", REPO_REF) else [f"origin/{REPO_REF}", REPO_REF]
for cand in cands:
    if subprocess.run(["git", "-C", str(clone_dir), "checkout", "--detach", cand], env=env).returncode == 0:
        break
else:
    raise SystemExit(f"cannot check out {REPO_REF} (tried {cands})")
HEAD = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], check=True, env=env,
                      capture_output=True, text=True).stdout.strip()
PROJECT = clone_dir / REPO_SUBDIR if (clone_dir / REPO_SUBDIR / "pyproject.toml").exists() else clone_dir
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "scripts"))
print("HEAD", HEAD)
print("project", PROJECT)
'''

INSTALL_GPU = '''
# (c) pinned install per DESIGN_DECISIONS 7.3. vLLM 0.30.0 pins torch 2.13 while Kaggle ships torch
# 2.10 / CUDA 12.8, so the engine goes into its own venv under /kaggle/tmp (INSTALL_MODE "venv"),
# or the Kaggle-built wheel is used (INSTALL_MODE "kaggle_vllm"); "system" installs into the kernel's
# interpreter and is for a runtime that already matches. The project (eval extra) is installed into
# the same interpreter, RUN_PYTHON, which every run_eval.py process is launched with; pip freeze and
# the framework versions of THAT interpreter are recorded next to the outputs (checklist C4).
import subprocess, sys
from pathlib import Path
if INSTALL_MODE == "venv":
    venv = Path(VENV_DIR)
    if not (venv / "bin" / "python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    RUN_PYTHON = str(venv / "bin" / "python")
    subprocess.run([RUN_PYTHON, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    pins = [f"vllm=={VLLM_VERSION}"]
elif INSTALL_MODE == "kaggle_vllm":
    RUN_PYTHON = sys.executable
    pins = [f"kaggle-vllm=={KAGGLE_VLLM_VERSION}"]
elif INSTALL_MODE == "system":
    RUN_PYTHON = sys.executable
    pins = [f"vllm=={VLLM_VERSION}"]
else:
    raise SystemExit(f"INSTALL_MODE must be venv, kaggle_vllm or system, not {INSTALL_MODE!r}")
if TRANSFORMERS_VERSION:
    pins.append(f"transformers=={TRANSFORMERS_VERSION}")
subprocess.run([RUN_PYTHON, "-m", "pip", "install", "-q", *pins], check=True)      # the engine first: its requirements win
subprocess.run([RUN_PYTHON, "-m", "pip", "install", "-q", "-e", ".[eval]"], check=True)
if RUN_PYTHON != sys.executable:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)   # the driver imports the project too
os.environ["HF_HOME"] = HF_HOME
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
_export(HF_TOKEN_SECRET)          # gated checkpoints (Gemma, Llama, Vistral); exported as HF_TOKEN when the secret has that name
if HF_TOKEN_SECRET != "HF_TOKEN" and os.environ.get(HF_TOKEN_SECRET):
    os.environ["HF_TOKEN"] = os.environ[HF_TOKEN_SECRET]
import colab_setup as CS
ENV_RECORD = CS.record_environment(PROJECT / "data" / "runs" / "env", python=RUN_PYTHON)
print("engine interpreter:", RUN_PYTHON)
print("environment recorded at", ENV_RECORD)
'''

INSTALL_TPU = '''
# (c) pinned install for the TPU: the vLLM TPU build first, then the project with its eval extra;
# record pip freeze + versions next to the outputs (checklist C4).
import subprocess, sys
from pathlib import Path
spec = f"{VLLM_TPU_PACKAGE}=={VLLM_TPU_VERSION}" if VLLM_TPU_VERSION else VLLM_TPU_PACKAGE
subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval]"], check=True)
RUN_PYTHON = sys.executable          # the TPU image's interpreter carries the vLLM-TPU build [UNCERTAIN: verify]
os.environ["HF_HOME"] = HF_HOME
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
_export(HF_TOKEN_SECRET)
if HF_TOKEN_SECRET != "HF_TOKEN" and os.environ.get(HF_TOKEN_SECRET):
    os.environ["HF_TOKEN"] = os.environ[HF_TOKEN_SECRET]
import colab_setup as CS
ENV_RECORD = CS.record_environment(PROJECT / "data" / "runs" / "env")
print("environment recorded at", ENV_RECORD)
'''

INSTALL_API = '''
# (c) install: the project with its eval extra (API SDKs: openai, google-genai, groq); no inference engine.
# Record pip freeze + versions next to the outputs (checklist C4).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval]"], check=True)
RUN_PYTHON = sys.executable
import colab_setup as CS
ENV_RECORD = CS.record_environment(PROJECT / "data" / "runs" / "env")
print("environment recorded at", ENV_RECORD)
'''

FETCH_VERIFY = '''
# (d) third-party resources (SHA-256 against data/HASHES.json) and the item file(s) of the runs this
# notebook will execute: the keys are DERIVED from the run ids (plus the smoke run) unless
# VERIFY_ITEM_KEYS lists them; a seeded sub-sample the plan derives from another file is written first
# (--materialize). Each file is checked for its hash against configs/run_plan.yaml, the header record,
# the canary on every row and its row/cell counts. A failure stops the notebook here.
import shutil, subprocess, sys
from pathlib import Path
import kaggle_run_plan as KRP
r = subprocess.run([sys.executable, "scripts/fetch_resources.py"], check=False)
if r.returncode:
    raise SystemExit("resource fetch or hash check failed (see above)")
if ITEMS_DATASET_DIR:
    src = Path(ITEMS_DATASET_DIR)
    if not src.exists():
        raise SystemExit(f"ITEMS_DATASET_DIR {src} not found: attach the private dataset that holds data/release")
    rel = Path(KRP.load_plan()["release"])            # e.g. data/release/v0.2: the plan's `release` key decides where the files go
    dest = PROJECT / rel
    if (src / rel.name).exists():                      # the dataset holds the data/release tree (one directory per version)
        shutil.copytree(src / rel.name, dest, dirs_exist_ok=True)
    elif (src / "manifest.json").exists():             # the dataset holds ONE release directory
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        raise SystemExit(f"{src} holds neither {rel.name}/ nor a release manifest.json: attach the frozen release")
    print("release copied from", src, "to", dest)
_run_ids = list(globals().get("RUN_IDS") or [globals().get("RUN_ID")]) + list(globals().get("SMOKE_RUN_IDS") or [])
KEYS = VERIFY_ITEM_KEYS or KRP.item_keys_for_runs(KRP.load_plan(), [r for r in _run_ids if r])
print("item files to verify:", KEYS)
for key in KEYS:
    r = subprocess.run([sys.executable, "scripts/kaggle_verify_items.py", "--key", key, "--materialize"], check=False)
    if r.returncode:
        raise SystemExit(f"item file {key} failed verification; nothing runs on it")
'''

PUSH_HELPER = '''
# outputs hand-off used after EVERY model (DD 7.6: Kaggle keeps /kaggle/working only if the version
# completes, so a session that times out must already have pushed what it finished) and once more at
# the end: copy data/runs (+ compute log) into /kaggle/working and add a version to a private Kaggle
# dataset through scripts/kaggle_dataset.py (credentials from Secrets, never printed).
import os, shutil, subprocess, sys
from pathlib import Path

def push_outputs(label):
    out = Path(WORK_DIR) / "noilai_runs_out"
    shutil.copytree(PROJECT / "data" / "runs", out / "runs", dirs_exist_ok=True)
    if (PROJECT / "data" / "compute_log.csv").exists():
        shutil.copy2(PROJECT / "data" / "compute_log.csv", out / "compute_log.csv")
    print("outputs copied to", out)
    if not PUSH_DATASET:
        return
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        _export(name)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        cmd = [sys.executable, "scripts/kaggle_dataset.py", "--src", str(PROJECT / "data" / "runs"),
               "--src", str(PROJECT / "data" / "compute_log.csv"), "--dest", str(Path(WORK_DIR) / "noilai_runs_dataset"),
               "--slug", KAGGLE_DATASET_SLUG, "--message", f"{label} @ {HEAD[:8]}"]
        if CREATE_DATASET and not globals().get("_DATASET_CREATED"):
            cmd.append("--create")
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            globals()["_DATASET_CREATED"] = True
    else:
        print("Kaggle credentials not found: dataset push skipped (outputs are still under", out, ")")

def after_each_model(res):
    if PUSH_AFTER_EACH_MODEL:
        push_outputs(f"{res['run']}/{res['model']} {res['status']}")
'''

OUTPUTS_KAGGLE = '''
# (g) final hand-off: the same push as after each model, once more for the session as a whole.
push_outputs(RUN_LABEL)
'''

COMPUTE_LOG_TAIL = '''
# (h) GPU-hours log (checklist C1). kaggle_run_plan wrote one row per executed model run; this adds the
# session's remaining wall time (startup, install, model downloads) as "overhead" and prints the totals.
import shutil
import compute_log as CL
LOG = PROJECT / "data" / "compute_log.csv"
run_hours = sum((r.get("hours") or 0.0) for r in (globals().get("RESULTS", []) + globals().get("SMOKE_RESULTS", [])))
overhead = max(0.0, CL.hours_since(SESSION_T0) - run_hours)
CL.append_entry(LOG, CL.Entry(date=CL.today(), platform=PLATFORM, gpu_type=GPU_TYPE, n_gpus=N_GPUS,
                              hours=overhead, run_id=f"session-{RUN_LABEL}", purpose="overhead"))
print(CL.format_totals(CL.totals(CL.read_entries(LOG))))
out = Path(WORK_DIR) / "noilai_runs_out"
out.mkdir(parents=True, exist_ok=True)
shutil.copy2(LOG, out / "compute_log.csv")
print("log copied to", out / "compute_log.csv", "-> merge into the repository's data/compute_log.csv")
'''


# =============================================================================== kaggle T4
def build_kaggle_t4() -> nbformat.NotebookNode:
    cells = [
        md("""
        # NóiLái — Kaggle 2×T4 evaluation runs

        Runs one line of `configs/run_plan.yaml` (default `E1_main`) over a list of models from
        `configs/models.yaml` with `scripts/run_eval.py --resume`, on Kaggle's free 2×T4 session
        (12 h max, ~30 GPU-h/week; ≤ 9 h of work per session, outputs pushed after every model).
        Models configured for other hardware (the TPU trio, the Modal L4) are skipped by the
        session guard, never run on the wrong device. Every cell is non-interactive: set the
        **parameters** cell, then *Run all*. Attach before starting:

        * accelerator **GPU T4 ×2**; internet on;
        * Kaggle Secrets: `GITHUB_TOKEN` (repository read, only if no bundle), `HF_TOKEN`
          (gated Gemma/Llama/Vistral), `KAGGLE_USERNAME` + `KAGGLE_KEY` (dataset push);
        * a private dataset holding `noilai-main.bundle` (`git bundle create noilai-main.bundle main`)
          and one holding the frozen `data/release/` (never public: the test split is gated).

        Cells: (a) environment · (b) clone · (c) pinned install (vLLM in a venv, DD 7.3) · (d) resources +
        item-file gate · (e) smoke test · (f) **RUN** · (g) outputs → dataset · (h) GPU-hours log.
        The flag mapping to `run_eval.py` lives in `scripts/kaggle_run_plan.py`; adjust flags there
        or through `EXTRA_ARGS`. Kaggle's own environment (nvidia driver, CUDA) is recorded by (c).
        """),
        code('''
        # ---- parameters: edit this cell only; nothing below asks for input ------------------------------
        import time
        SESSION_T0 = time.time()

        REPO_URL = "https://github.com/OWNER/REPO.git"     # [UNCERTAIN: verify] the repository that holds noilai/ (or the split-out noilai repo)
        REPO_REF = "main"                                   # pin a commit hash for a paper run; recorded in every manifest
        REPO_SUBDIR = "noilai"                              # "" once noilai is a repository of its own
        BUNDLE_PATH = "/kaggle/input/noilai-bundle/noilai-main.bundle"   # git bundle in a private dataset; preferred over a token
        CLONE_DIR = "/kaggle/working/repo"
        WORK_DIR = "/kaggle/working"                        # ~20 GB, saved with the notebook
        HF_HOME = "/kaggle/tmp/hf"                          # model cache outside the working dir [UNCERTAIN: verify the quota of /kaggle/tmp]
        ITEMS_DATASET_DIR = "/kaggle/input/noilai-release"  # private dataset with the frozen data/release/<version>/ tree (or that tree's parent); None if the clone has it
        VERIFY_ITEM_KEYS = None          # None = derived from RUN_ID and the smoke run by (d); or list keys of configs/run_plan.yaml item_files

        INSTALL_MODE = "venv"            # "venv": vLLM into /kaggle/tmp (DD 7.3: vLLM 0.30 pins torch 2.13, Kaggle ships 2.10); "kaggle_vllm": the Kaggle-built wheel; "system"
        VENV_DIR = "/kaggle/tmp/vllm-venv"
        VLLM_VERSION = "0.30.0"          # DESIGN_DECISIONS 7.3 [UNCERTAIN: verify] pin to the version the smoke tests pass with; recorded by (c)
        KAGGLE_VLLM_VERSION = "0.2.0"    # INSTALL_MODE "kaggle_vllm" only [UNCERTAIN: verify]
        TRANSFORMERS_VERSION = None      # None = whatever the vllm wheel requires (pip freeze is recorded either way)

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        HF_TOKEN_SECRET = "HF_TOKEN"

        RUN_ID = "E1_main"               # a run id from configs/run_plan.yaml (kaggle_run_plan.py --list)
        MODELS = ["gemma-3-1b-it"]       # subset of the run's models; [] = every model of the run that this session's hardware can run
        SMOKE_RUN_IDS = ["smoke_20"]     # (e) the smoke line, 20 items on SMOKE_MODEL before the run
        SMOKE_MODEL = "gemma-3-1b-it"
        EXTRA_ARGS = ""                  # extra run_eval.py flags, one shell-quoted string, e.g. "--seed 7"
        DRY_RUN = False                  # True prints the commands and runs nothing

        PLATFORM, GPU_TYPE, N_GPUS = "kaggle", "t4", 2
        SESSION_HARDWARE = "2xt4"        # models.yaml hardware key of THIS session; entries for other hardware are skipped (t4 entries run here)
        ALLOW_HARDWARE_MISMATCH = False  # True runs e.g. a TPU-configured model here as the documented 2xT4 fallback (logged as this session's device)
        MAX_SESSION_HOURS = 9.0          # DD 7.6: launch no new model past 9 h of a 12-h session
        REQUIRE_GPU = True
        RUN_LABEL = RUN_ID
        PUSH_DATASET = True
        PUSH_AFTER_EACH_MODEL = True     # DD 7.6: push mid-run so a timed-out session keeps what it finished
        KAGGLE_DATASET_SLUG = "noilai-runs"
        CREATE_DATASET = False           # True the first time the dataset is published
        '''),
        code(SECRET_HELPER),
        code(GPU_CHECK),
        code(CLONE),
        code(INSTALL_GPU),
        code(FETCH_VERIFY),
        code(PUSH_HELPER),
        md("""
        ### RUN CELL (e) — smoke test
        One small model, 20 items (`smoke_20` in `run_plan.yaml`, the only kind of line allowed a
        `--limit`): load, generate, log-probs. Stops the notebook on failure so that no GPU hour is
        spent on a broken environment.
        """),
        code('''
        # RUN CELL (e): smoke test. Flags come from run_plan.yaml (smoke_20) via scripts/kaggle_run_plan.py;
        # the process runs on the engine's interpreter (RUN_PYTHON) and the session guards apply.
        import shlex
        import kaggle_run_plan as KRP
        SMOKE_RESULTS = []
        for smoke_id in SMOKE_RUN_IDS:
            SMOKE_RESULTS += KRP.execute(smoke_id, models=[SMOKE_MODEL], platform=PLATFORM, dry_run=DRY_RUN,
                                         continue_on_error=False, extra=shlex.split(EXTRA_ARGS), python=RUN_PYTHON,
                                         session_hardware=SESSION_HARDWARE, allow_hardware_mismatch=ALLOW_HARDWARE_MISMATCH,
                                         session_t0=SESSION_T0, max_session_hours=MAX_SESSION_HOURS)
        for r in SMOKE_RESULTS:
            print(r["run"], r["model"], "->", r["status"])
        if any(str(r["status"]).startswith(("failed", "skipped")) for r in SMOKE_RESULTS):
            raise SystemExit("smoke test failed or was skipped: fix the environment, the model entry or SESSION_HARDWARE before the run cell")
        ''', tags=[RUN_TAG]),
        md("""
        ### RUN CELL (f) — the run
        Loops over `MODELS` (or every model of `RUN_ID` that `SESSION_HARDWARE` can run) calling
        `scripts/run_eval.py --model-config <name> --items ... --tasks ... --arms ... --resume --run-id <RUN_ID>__<name> --out-root data/runs`
        on the engine's interpreter. `--resume` makes a re-run after a session timeout continue where it
        stopped; no new model starts past `MAX_SESSION_HOURS`; the outputs are pushed after every model.
        Each model's wall time is appended to `data/compute_log.csv` with THIS session's device type and
        count (2 × T4), whatever hardware the model is configured for.
        """),
        code('''
        # RUN CELL (f): the run matrix line RUN_ID over MODELS, with --resume. Adjust flags in
        # scripts/kaggle_run_plan.py (build_command) or through EXTRA_ARGS; the CLI is owned by noilai/eval.
        import json, shlex
        import kaggle_run_plan as KRP
        RESULTS = KRP.execute(RUN_ID, models=MODELS or None, platform=PLATFORM, dry_run=DRY_RUN, continue_on_error=True,
                              extra=shlex.split(EXTRA_ARGS), python=RUN_PYTHON, session_hardware=SESSION_HARDWARE,
                              allow_hardware_mismatch=ALLOW_HARDWARE_MISMATCH, session_t0=SESSION_T0,
                              max_session_hours=MAX_SESSION_HOURS, after_each=after_each_model)
        print(json.dumps([{"model": r["model"], "status": r["status"], "hours": r.get("hours"),
                           "new_outputs": r.get("new_outputs"), "logged_device": r.get("logged_device")} for r in RESULTS], indent=1))
        ''', tags=[RUN_TAG]),
        code(OUTPUTS_KAGGLE),
        code(COMPUTE_LOG_TAIL),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update(_metadata("kaggle", "GPU T4 x2"))
    return nb


# =============================================================================== kaggle TPU
def build_kaggle_tpu() -> nbformat.NotebookNode:
    cells = [
        md("""
        # NóiLái — Kaggle TPU v5e-8 runs (vLLM-TPU)

        The TPU variant of the T4 notebook, same structure: the three large models (`tpu_main`:
        Gemma 3 12B, Qwen3.8-27B, Qwen-SEA-LION-v4.5-27B-IT in bf16, tensor parallel over the 8
        chips), the bf16 drift check (`bf16_drift_200`: 200 core items per T4 model, unquantized;
        `phogpt-4b-chat--bf16` is configured for the Modal L4 and skipped here by the session guard)
        and the 27B reasoning sub-study (`reasoning_500_tpu`). Sessions last 9 h (~20 TPU-h/week),
        one at a time; the vLLM-TPU startup takes 6–22 minutes per model, so keep `MODELS` short
        per session and rely on `--resume`; outputs are pushed after every model. Attach:
        accelerator **TPU VM v5e-8**, internet on, the same secrets and private datasets as the
        T4 notebook.

        Every TPU-hosted model must load under vLLM-TPU and return prompt log-probabilities before its
        full run (plan §3): the smoke cell checks that.
        """),
        code('''
        # ---- parameters: edit this cell only; nothing below asks for input ------------------------------
        import time
        SESSION_T0 = time.time()

        REPO_URL = "https://github.com/OWNER/REPO.git"     # [UNCERTAIN: verify]
        REPO_REF = "main"
        REPO_SUBDIR = "noilai"
        BUNDLE_PATH = "/kaggle/input/noilai-bundle/noilai-main.bundle"
        CLONE_DIR = "/kaggle/working/repo"
        WORK_DIR = "/kaggle/working"
        HF_HOME = "/kaggle/tmp/hf"                          # 27B bf16 weights are ~54 GB: not in the 20 GB working dir [UNCERTAIN: verify]
        ITEMS_DATASET_DIR = "/kaggle/input/noilai-release"
        VERIFY_ITEM_KEYS = None          # None = derived from RUN_IDS and SMOKE_RUN_IDS by (d)

        VLLM_TPU_PACKAGE = "vllm-tpu"    # [UNCERTAIN: verify] the TPU build's package name for the pinned version (kaggle-tpu-lab documents the working recipe)
        VLLM_TPU_VERSION = None          # [UNCERTAIN: verify] pin once the smoke cell passes; None = latest (recorded by (c))

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        HF_TOKEN_SECRET = "HF_TOKEN"

        RUN_IDS = ["tpu_main"]           # any of: tpu_main, bf16_drift_200, reasoning_500_tpu
        MODELS = ["gemma-3-12b-it"]      # subset of each run's models; [] = every model the TPU session can run
        SMOKE_RUN_IDS = ["smoke_20_tpu"] # (e) the TPU smoke line on SMOKE_MODEL
        SMOKE_MODEL = "gemma-3-12b-it"
        EXTRA_ARGS = ""
        DRY_RUN = False

        PLATFORM, GPU_TYPE, N_GPUS = "kaggle", "tpu-v5e-8", 1
        SESSION_HARDWARE = "tpu"         # entries configured for t4 / 2xt4 / l4 are skipped here
        ALLOW_HARDWARE_MISMATCH = False
        MAX_SESSION_HOURS = 8.0          # 9-h TPU sessions: launch no new model past 8 h
        REQUIRE_TPU = True
        RUN_LABEL = "+".join(RUN_IDS)
        PUSH_DATASET = True
        PUSH_AFTER_EACH_MODEL = True
        KAGGLE_DATASET_SLUG = "noilai-runs"
        CREATE_DATASET = False
        '''),
        code(SECRET_HELPER),
        code(TPU_CHECK),
        code(CLONE),
        code(INSTALL_TPU),
        code(FETCH_VERIFY),
        code(PUSH_HELPER),
        md("""
        ### RUN CELL (e) — smoke test on the TPU
        `smoke_20_tpu` on `SMOKE_MODEL` (booked under TPU hours): the model must load under vLLM-TPU
        with tensor parallel 8, generate, and return prompt log-probabilities (the T3 forced-choice
        scoring needs them).
        """),
        code('''
        # RUN CELL (e): smoke test (smoke_20_tpu) on the TPU model.
        import shlex
        import kaggle_run_plan as KRP
        SMOKE_RESULTS = []
        for smoke_id in SMOKE_RUN_IDS:
            SMOKE_RESULTS += KRP.execute(smoke_id, models=[SMOKE_MODEL], platform=PLATFORM, dry_run=DRY_RUN,
                                         continue_on_error=False, extra=shlex.split(EXTRA_ARGS), python=RUN_PYTHON,
                                         session_hardware=SESSION_HARDWARE, allow_hardware_mismatch=ALLOW_HARDWARE_MISMATCH,
                                         session_t0=SESSION_T0, max_session_hours=MAX_SESSION_HOURS)
        for r in SMOKE_RESULTS:
            print(r["run"], r["model"], "->", r["status"])
        if any(str(r["status"]).startswith(("failed", "skipped")) for r in SMOKE_RESULTS):
            raise SystemExit("smoke test failed or was skipped: the TPU route for this model is not ready (fallback: 2xT4 4-bit, labelled quantized)")
        ''', tags=[RUN_TAG]),
        md("""
        ### RUN CELL (f) — the TPU runs
        One `run_eval.py --resume` command per (run id, model) that the TPU session can run; wall
        time per model is logged as TPU device-hours; outputs are pushed after every model; no new
        model starts past `MAX_SESSION_HOURS`.
        """),
        code('''
        # RUN CELL (f): each run id in RUN_IDS over MODELS (or all of the run's TPU-runnable models), --resume.
        import json, shlex
        import kaggle_run_plan as KRP
        RESULTS = []
        for run_id in RUN_IDS:
            RESULTS += KRP.execute(run_id, models=MODELS or None, platform=PLATFORM, dry_run=DRY_RUN, continue_on_error=True,
                                   extra=shlex.split(EXTRA_ARGS), python=RUN_PYTHON, session_hardware=SESSION_HARDWARE,
                                   allow_hardware_mismatch=ALLOW_HARDWARE_MISMATCH, session_t0=SESSION_T0,
                                   max_session_hours=MAX_SESSION_HOURS, after_each=after_each_model)
        print(json.dumps([{"run": r["run"], "model": r["model"], "status": r["status"], "hours": r.get("hours"),
                           "logged_device": r.get("logged_device")} for r in RESULTS], indent=1))
        ''', tags=[RUN_TAG]),
        code(OUTPUTS_KAGGLE),
        code(COMPUTE_LOG_TAIL),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update(_metadata("kaggle", "TPU VM v5e-8"))
    return nb


# =============================================================================== colab probe
def build_colab_probe() -> nbformat.NotebookNode:
    cells = [
        md("""
        # NóiLái — E4 skeleton: tone/onset/rime probes and patching in Gemma 3 (Colab / Kaggle 2×T4)

        Loads **Gemma 3 1B (IT)** in float32 on one T4 (fp32 ≈ 4 GB), or **Gemma 3 4B (IT)** as the
        text-only `Gemma3ForCausalLM` (skips the SigLIP tower) **sharded over 2×T4** with
        `device_map="auto"` (fp32 ≈ 15.5 GB does not fit one T4; DESIGN_DECISIONS 9.5). Gemma 3 must
        never run in fp16; bf16 only on a bf16-capable GPU (L4/A100) and then with the 200-pair drift
        check. The notebook extracts residual-stream states at each syllable's last sub-token under
        NFC and NFD with `noilai.probe.extract`, fits layer-wise probes with control tasks
        (`noilai.probe.probes`), and holds the patching skeleton (`noilai.probe.patching`) for the
        DESIGN_DECISIONS 9.3 pair *bí mà* (clean) vs *bí mạ* (corrupt) under V3 with E1's frozen
        prompt. Results go to Drive (Colab) or the working directory (Kaggle). Secrets: `HF_TOKEN`
        (gated Gemma), `GITHUB_TOKEN` if no bundle.

        The two cells tagged `e4` are the experiment and are meant to be edited as E4 takes shape
        (plan §2.6, weeks of Nov 23 – Dec 6; Gate 4 keeps probing only if patching is clean).
        """),
        code('''
        # ---- parameters: edit this cell only; nothing below asks for input ------------------------------
        import time
        SESSION_T0 = time.time()

        REPO_URL = "https://github.com/OWNER/REPO.git"     # [UNCERTAIN: verify]
        REPO_REF = "main"
        REPO_SUBDIR = "noilai"
        DRIVE_DIR = "/content/drive/MyDrive/noilai"         # bundle, release and outputs live here
        BUNDLE_PATH = DRIVE_DIR + "/noilai-main.bundle"
        CLONE_DIR = "/content/repo"
        WORK_DIR = "/content"
        HF_HOME = "/content/hf"
        ITEMS_DATASET_DIR = None                            # probes use the syllable inventory, not the item files
        VERIFY_ITEM_KEYS = []
        OUT_DIR = DRIVE_DIR + "/probe"

        VLLM_VERSION = None                                 # not used here: probes run through transformers
        TRANSFORMERS_VERSION = None
        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        HF_TOKEN_SECRET = "HF_TOKEN"

        MODEL_CONFIG = "gemma-3-1b-it"    # entry of configs/models.yaml (hf_id read from it); "gemma-3-4b-it" needs N_GPUS = 2 (Kaggle 2xT4, fp32 sharded)
        DTYPE = "float32"                 # "bfloat16" only on a bf16-capable GPU; never "float16" for Gemma 3
        TEXT_ONLY = True                  # multimodal checkpoints (4B/12B): load Gemma3ForCausalLM, the text tower only (DD 9.5)
        N_SYLLABLES = 600                 # tone-bearing syllables sampled from the inventory (train/test split by syllable)
        ENCODINGS = ["nfc", "nfd"]
        FEATURES = ["tone", "onset", "rime"]
        POSITION = "last"                 # "last" sub-token of the syllable or "after"
        SEED = 20261004
        BATCH_SIZE = 8

        PLATFORM, GPU_TYPE, N_GPUS = "colab", "t4", 1    # Kaggle 2xT4 for the 4B: PLATFORM "kaggle", N_GPUS 2, DRIVE_DIR -> /kaggle/working/noilai
        REQUIRE_GPU = True
        RUN_LABEL = "E4_probe"
        '''),
        code(SECRET_HELPER),
        code('''
        # Drive: results, the bundle and (optionally) the release live under DRIVE_DIR
        from pathlib import Path
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            print("Drive mounted; outputs ->", OUT_DIR)
        except ImportError:
            print("not on Colab: outputs stay local under", WORK_DIR)
            OUT_DIR = str(Path(WORK_DIR) / "probe")
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        '''),
        code(GPU_CHECK),
        code(CLONE),
        code('''
        # (c) install: transformers path only (eval extra for torch/transformers/accelerate, dev extra for scikit-learn probes)
        import subprocess, sys
        from pathlib import Path
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval,dev]"], check=True)
        os.environ["HF_HOME"] = HF_HOME
        Path(HF_HOME).mkdir(parents=True, exist_ok=True)
        _export(HF_TOKEN_SECRET)
        import colab_setup as CS
        ENV_RECORD = CS.record_environment(Path(OUT_DIR) / "env")
        print("environment recorded at", ENV_RECORD)
        '''),
        code(FETCH_VERIFY),
        code('''
        # load the model named by MODEL_CONFIG in DTYPE (fp32 by default: Gemma 3 overflows in fp16). The 4B/12B
        # checkpoints are multimodal: TEXT_ONLY loads Gemma3ForCausalLM (the text tower, no SigLIP), and N_GPUS > 1
        # shards it over the session's GPUs with device_map="auto" (accelerate); DESIGN_DECISIONS 9.5.
        import re, torch, yaml
        from transformers import AutoModelForCausalLM, AutoTokenizer
        entry = next(m for m in yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))["models"] if m["name"] == MODEL_CONFIG)
        is_gemma3 = entry.get("family") == "gemma3" or bool(re.match(r"google/gemma-3-", entry.get("hf_id") or ""))
        if is_gemma3 and DTYPE == "float16":
            raise SystemExit("Gemma 3 must not run in float16 (transformers PR #36832)")
        if DTYPE == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (8, 0):
            raise SystemExit("this GPU has no bfloat16 units; use float32 (or an L4/A100 runtime)")
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_visible < N_GPUS:
            raise SystemExit(f"N_GPUS = {N_GPUS} but {n_visible} CUDA device(s) are visible: the 4B in fp32 needs Kaggle's 2xT4")
        HF_ID = entry["hf_id"]
        REVISION = entry.get("revision")        # a full commit hash before a paper run (DD 7.1); None = the hub's default branch
        print("loading", HF_ID, "in", DTYPE, "| hf_id_status:", entry.get("hf_id_status"), "| revision:", REVISION, "| gpus:", N_GPUS)
        tok = AutoTokenizer.from_pretrained(HF_ID, revision=REVISION)
        device_map = "auto" if N_GPUS > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
        load_kw = {"dtype": getattr(torch, DTYPE), "device_map": device_map, "revision": REVISION}   # transformers 5.x: `dtype`, not torch_dtype
        multimodal = is_gemma3 and float(entry.get("params_b") or 0) > 1.5
        if multimodal and TEXT_ONLY:
            from transformers import Gemma3ForCausalLM
            model = Gemma3ForCausalLM.from_pretrained(HF_ID, **load_kw).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(HF_ID, **load_kw).eval()
        cfg = getattr(model.config, "text_config", model.config)
        print("class:", type(model).__name__, "| layers:", cfg.num_hidden_layers, "| hidden:", cfg.hidden_size,
              "| devices:", sorted({str(p.device) for p in model.parameters()}))
        '''),
        md("""
        ### E4 CELL — layer-wise probes with control tasks
        Tone-bearing syllables from the inventory are placed in carrier sentences (NFC and NFD), the
        residual stream at the syllable's last sub-token is extracted at every layer, and a probe per
        layer predicts tone / onset / rime with a syllable-disjoint split; selectivity = real − control.
        Edit freely as E4 develops; this is the skeleton called for by the plan.
        """),
        code('''
        # E4 CELL: probes. Uses noilai.probe as it exists in the clone (extract.make_examples,
        # extract.extract_hidden_states, probes.run_layer_probes, probes.results_table).
        import json, random
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from noilai.probe import extract, probes
        from noilai.vi import lexicon as L
        from noilai.vi.syllable import try_parse

        rng = random.Random(SEED)
        pool = [s for s in L.load_hunspell_syllables("new") if (p := try_parse(s)) is not None and p.syllable.tone != 0]
        rng.shuffle(pool)
        syllables = pool[:N_SYLLABLES]
        PROBE_TABLES = {}
        for enc in ENCODINGS:
            exs = extract.make_examples(syllables, encoding=enc)
            H = extract.extract_hidden_states(model, tok, exs, batch_size=BATCH_SIZE, positions=(POSITION,))[POSITION]
            groups = [e.syllable for e in exs]
            for feature in [f for f in FEATURES if f in exs[0].labels]:
                labels = [e.labels[feature] for e in exs]
                res = probes.run_layer_probes(H, labels, groups, feature=feature, seed=SEED)
                table = probes.results_table(res)
                PROBE_TABLES[(enc, feature)] = table
                best = probes.best_layer(res)
                print(f"{enc:>3} {feature:<6} best layer {best.layer}: {table.loc[table['layer'] == best.layer].to_dict('records')[0]}")
                Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
                table.to_csv(Path(OUT_DIR) / f"probe_{MODEL_CONFIG}_{enc}_{feature}_{POSITION}.csv", index=False)
            np.save(Path(OUT_DIR) / f"hidden_{MODEL_CONFIG}_{enc}_{POSITION}.npy", H.astype(np.float16))
        json.dump({"model": HF_ID, "dtype": DTYPE, "n_syllables": len(syllables), "encodings": ENCODINGS, "features": FEATURES,
                   "position": POSITION, "seed": SEED, "head": HEAD}, open(Path(OUT_DIR) / f"probe_{MODEL_CONFIG}_manifest.json", "w"), indent=2)
        ''', tags=[E4_TAG]),
        md("""
        ### E4 CELL — activation patching (DESIGN_DECISIONS 9.3 pair *bí mà* / *bí mạ*, readout B)
        The varying syllable is the SECOND input syllable, so under V3 (swap tones) the two gold
        answers diverge at their FIRST syllable: *bí mà* → *bì má* (clean) and *bí mạ* → *bị má*
        (corrupt), read at the pieces `▁bì` vs `▁bị`. Every output is recomputed by the rule engine
        (`noilai.gen.variants`) and checked for legality; the prompt is E1's frozen V3 template with
        its three rule-engine demonstrations and the IT chat template, the answer teacher-forced from
        the `Đáp án:` line. (The plan's *bí mật* / *bí mất* is not a legal V3 pair — equal tones make V3
        the identity — and belongs to V4 patching: *bí mật* → *bật mí*.) The cell caches the clean run,
        patches the corrupt run layer by layer at the varying syllable's positions and reports recovery
        of the logit difference; it refuses a pair whose token spans differ outside that syllable.
        """),
        code(f'''
        # E4 CELL: activation patching on the {E4_PATCH_PAIR["source"]} pair with E1's exact {E4_PATCH_PAIR["variant"]} prompt (frozen
        # template p0, three rule-engine demonstrations, the IT chat template): readout B. The pair, its outputs
        # and their legality come from the rule engine; nothing below is a typed-out example. [NATIVE-CHECK] the
        # three syllables are ordinary (bí "gourd/secret", mà "but", mạ "rice seedling"); the outputs are pseudo-phrases.
        import json, torch
        from pathlib import Path
        from noilai.eval import prompts as P
        from noilai.gen import variants as V
        from noilai.probe import patching
        from noilai.vi import lexicon as L
        from noilai.vi.syllable import parse, spell

        VARIANT = "{E4_PATCH_PAIR["variant"]}"
        PARTNER, CLEAN_SYL, CORRUPT_SYL = "{E4_PATCH_PAIR["partner"]}", "{E4_PATCH_PAIR["clean"]}", "{E4_PATCH_PAIR["corrupt"]}"   # the varying syllable is the SECOND one
        inv = L.load_inventory()

        def _pair(second):
            a, b = parse(PARTNER).syllable, parse(second).syllable
            if V.is_identity(VARIANT, a, b):
                raise SystemExit(f"{{PARTNER}} {{second}} is an identity under {{VARIANT}}: not a legal patching pair (DD 3.4)")
            out = V.apply(VARIANT, a, b)
            if not all(inv.is_legal(x, "onset_rime") for x in out):
                raise SystemExit(f"{{VARIANT}} on {{PARTNER}} {{second}} gives an illegal syllable")
            return f"{{PARTNER}} {{second}}", f"{{spell(out[0])}} {{spell(out[1])}}"

        CLEAN, ANSWER_CLEAN = _pair(CLEAN_SYL)
        CORRUPT, ANSWER_CORRUPT = _pair(CORRUPT_SYL)
        first_clean, first_corrupt = ANSWER_CLEAN.split()[0], ANSWER_CORRUPT.split()[0]
        assert first_clean != first_corrupt, "the two gold answers must diverge at their first syllable"
        print(f"clean {{CLEAN}} -> {{ANSWER_CLEAN}} | corrupt {{CORRUPT}} -> {{ANSWER_CORRUPT}} | readout {{first_clean}} vs {{first_corrupt}}")

        def _prompt(phrase):
            item = {{"task": "T1", "variant": VARIANT, "input": phrase, "item_id": "P-" + phrase.replace(" ", "_"), "gold": [""]}}
            msgs = P.render(item, paraphrase="p0", shots=3, arm="nfc")          # E1's frozen template + rule-engine demonstrations
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + P.answer_marker()   # teacher-forced answer line
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)          # the template carries <bos>: no double BOS
            start = text.rindex(phrase) + len(PARTNER) + 1                        # the ITEM's varying syllable (the demonstrations come earlier)
            end = start + len(phrase.split()[1])
            positions = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if b > start and a < end]
            return enc["input_ids"], positions, text

        clean_ids, clean_pos, clean_text = _prompt(CLEAN)
        corrupt_ids, corrupt_pos, _ = _prompt(CORRUPT)
        tok_clean = tok(" " + first_clean, add_special_tokens=False)["input_ids"]       # one word-initial piece each (DD 9.3)
        tok_corrupt = tok(" " + first_corrupt, add_special_tokens=False)["input_ids"]
        if clean_ids.shape != corrupt_ids.shape or clean_pos != corrupt_pos or len(tok_clean) != 1 or len(tok_corrupt) != 1:
            print("pair not aligned for this tokenizer: token spans or answer pieces differ; pick another DD 9.3 pair",
                  clean_ids.shape, corrupt_ids.shape, clean_pos, corrupt_pos, tok_clean, tok_corrupt)
        else:
            diff = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
            assert set(diff) <= set(clean_pos), f"clean/corrupt prompts differ outside the varying syllable: {{diff}} vs {{clean_pos}}"
            device = next(model.parameters()).device
            result = patching.run_patching(model, clean_ids.to(device), corrupt_ids.to(device), answer_pos=clean_ids.shape[1] - 1,
                                           tok_clean=tok_clean[0], tok_corrupt=tok_corrupt[0], position_groups=[clean_pos])
            print(result)
            json.dump({{"variant": VARIANT, "clean": CLEAN, "corrupt": CORRUPT, "answer_clean": ANSWER_CLEAN, "answer_corrupt": ANSWER_CORRUPT,
                       "positions": clean_pos, "prompt_hash": P.prompt_hash([{{"role": "user", "content": clean_text}}]),
                       "result": str(result), "head": HEAD, "model": HF_ID, "dtype": DTYPE}},
                      open(Path(OUT_DIR) / f"patching_{{MODEL_CONFIG}}_{{VARIANT}}_{{PARTNER}}_{{CLEAN_SYL}}_{{CORRUPT_SYL}}.json", "w"), ensure_ascii=False, indent=2)
        ''', tags=[E4_TAG]),
        code('''
        # GPU-hours log (checklist C1): this session as one E4 row; totals printed and copied to Drive
        import shutil
        import compute_log as CL
        LOG = PROJECT / "data" / "compute_log.csv"
        CL.append_entry(LOG, CL.Entry(date=CL.today(), platform=PLATFORM, gpu_type=GPU_TYPE, n_gpus=N_GPUS,
                                      hours=CL.hours_since(SESSION_T0), run_id=f"{RUN_LABEL}__{MODEL_CONFIG}", purpose="E4_probe"))
        print(CL.format_totals(CL.totals(CL.read_entries(LOG))))
        shutil.copy2(LOG, Path(OUT_DIR) / "compute_log.csv")
        '''),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update(_metadata("colab_or_kaggle", "T4 (1B, fp32) / 2xT4 (4B, fp32 sharded) / L4-A100 (bf16)"))
    return nb


# =============================================================================== API runs
def build_api_runs() -> nbformat.NotebookNode:
    cells = [
        md("""
        # NóiLái — API runs (Gemini and Groq free tiers), core set only

        Runs the API lines of `configs/run_plan.yaml` (`E1_api_core`, `E1_api_paraphrase`,
        `E3_api_core`, `reasoning_500_api`) on the four API-served models. Guards in
        `scripts/kaggle_run_plan.py` make every API command `--in-core-only` and refuse the sealed
        split. Keys come from the environment / Colab userdata / Kaggle Secrets (`GEMINI_API_KEY`,
        `GROQ_API_KEY`) and are never printed.

        **Daily-quota loop.** Groq's free plan allows 1,000 requests and 200K tokens a day; at
        ~500 tokens a call the TOKEN cap binds first (~400 calls), so the core takes about four days
        per model. The driver keeps `data/runs/api_ledger.json` (requests AND tokens per model per
        UTC day, counted from the new rows of `outputs.jsonl`), skips a model whose daily request or
        token budget is spent, and stops a running command (SIGINT, rows kept) the moment the day's
        cap is reached — reported as *parked*, not failed; the backend paces itself at the entry's
        `requests_per_minute`. Run this notebook once a day with `--resume` until every run is
        complete. Gemini's free-tier limits are shown only in AI Studio: read them and set
        `GEMINI_RPD_OVERRIDE`; the override is applied to the in-memory config and recorded in the
        ledger — no file under `configs/` is ever rewritten by a notebook.

        Terms: Gemini's free tier trains on inputs and requires users to be 18+; only the core reaches
        an API, never annotator data, never the sealed split (plan §2.3, §3).
        """),
        code('''
        # ---- parameters: edit this cell only; nothing below asks for input ------------------------------
        import time
        SESSION_T0 = time.time()

        REPO_URL = "https://github.com/OWNER/REPO.git"     # [UNCERTAIN: verify]
        REPO_REF = "main"
        REPO_SUBDIR = "noilai"
        BUNDLE_PATH = "/content/drive/MyDrive/noilai/noilai-main.bundle"   # Colab; on Kaggle use /kaggle/input/noilai-bundle/noilai-main.bundle
        CLONE_DIR = "/content/repo"
        WORK_DIR = "/content"
        ITEMS_DATASET_DIR = "/content/drive/MyDrive/noilai/release"        # the frozen data/release tree; None if the clone has it
        VERIFY_ITEM_KEYS = None          # None = derived from RUN_IDS by (d)

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        API_KEY_SECRETS = ["GEMINI_API_KEY", "GROQ_API_KEY"]   # names only; values stay in the secret store
        GEMINI_RPD_OVERRIDE = None       # requests/day read in AI Studio; None = no ledger cap for Gemini [UNCERTAIN: verify]; applied in memory, recorded in the ledger
        GEMINI_TPD_OVERRIDE = None       # tokens/day read in AI Studio, same handling [UNCERTAIN: verify]

        RUN_IDS = ["E1_api_core", "E1_api_paraphrase", "E3_api_core"]   # add "reasoning_500_api" for the cost curve
        MODELS = []                      # subset of the run's models; [] = all four
        EXTRA_ARGS = ""
        DRY_RUN = False

        PLATFORM, GPU_TYPE, N_GPUS = "api", "api", 0
        SESSION_HARDWARE = "api"
        MAX_SESSION_HOURS = None         # the API loop is bounded by the daily caps, not by a session
        REQUIRE_GPU = False
        RUN_LABEL = "+".join(RUN_IDS)
        MOUNT_DRIVE = True
        '''),
        code(SECRET_HELPER),
        code('''
        # Drive (Colab only): bundle, release and the outputs' backup
        from pathlib import Path
        if MOUNT_DRIVE:
            try:
                from google.colab import drive
                drive.mount("/content/drive")
                print("Drive mounted")
            except ImportError:
                print("not on Colab: no Drive; set BUNDLE_PATH / ITEMS_DATASET_DIR to local paths")
        '''),
        code(CLONE),
        code(INSTALL_API),
        code('''
        # API keys into the environment (names from configs/models.yaml providers.*.api_key_env); found / not found only.
        # Session overrides of the Gemini caps are applied to the IN-MEMORY config the driver receives (MODELS_CFG),
        # never written to configs/models.yaml (that would strip its comments and dirty every manifest's git state).
        import kaggle_run_plan as KRP
        PLAN, MODELS_CFG = KRP.load_plan(), KRP.load_models()
        wanted = sorted({p["api_key_env"] for p in MODELS_CFG["providers"].values()} & set(API_KEY_SECRETS))
        FOUND = {name: _export(name) for name in wanted}
        if not any(FOUND.values()):
            raise SystemExit("no API key found: add GEMINI_API_KEY / GROQ_API_KEY to the secret store")
        OVERRIDES = {}
        if GEMINI_RPD_OVERRIDE:
            MODELS_CFG["providers"]["gemini"]["free_tier"]["requests_per_day"] = int(GEMINI_RPD_OVERRIDE)
            OVERRIDES["gemini.requests_per_day"] = int(GEMINI_RPD_OVERRIDE)
        if GEMINI_TPD_OVERRIDE:
            MODELS_CFG["providers"]["gemini"]["free_tier"]["tokens_per_day"] = int(GEMINI_TPD_OVERRIDE)
            OVERRIDES["gemini.tokens_per_day"] = int(GEMINI_TPD_OVERRIDE)
        if OVERRIDES:
            lpath = KRP.ledger_path(PLAN)
            ledger = KRP.ledger_load(lpath)
            ledger.setdefault("_overrides", {})[KRP.utc_day()] = OVERRIDES
            KRP.ledger_save(lpath, ledger)
            print("session overrides (in memory, recorded in the ledger):", OVERRIDES)
        '''),
        code(FETCH_VERIFY),
        md("""
        ### RUN CELL — daily API loop
        For each run id and model: skip if today's ledger says the daily request OR token budget is
        spent, else `run_eval.py ... --in-core-only --resume` under the budget watchdog (the command
        is interrupted at the cap and reported as *parked*). Models whose key is missing are skipped
        with a message. Re-run tomorrow; `--resume` continues.
        """),
        code('''
        # RUN CELL: quota-aware loop over RUN_IDS x MODELS through scripts/kaggle_run_plan.py (in-core-only enforced there);
        # the in-memory MODELS_CFG carries the session overrides.
        import json, shlex
        import kaggle_run_plan as KRP
        RESULTS = []
        for run_id in RUN_IDS:
            run = KRP.find_run(PLAN, run_id)
            names = MODELS or KRP.expand_models(run, PLAN, MODELS_CFG)
            runnable = []
            for name in names:
                prov = MODELS_CFG["by_name"][name].get("provider")
                key_env = MODELS_CFG["providers"][prov]["api_key_env"] if prov else None
                if key_env and not os.environ.get(key_env):
                    print(f"skip {name}: {key_env} not set")
                else:
                    runnable.append(name)
            RESULTS += KRP.execute(run_id, models=runnable, platform=PLATFORM, dry_run=DRY_RUN, continue_on_error=True,
                                   extra=shlex.split(EXTRA_ARGS), plan=PLAN, models_cfg=MODELS_CFG, python=RUN_PYTHON,
                                   session_hardware=SESSION_HARDWARE)
        print(json.dumps([{"run": r["run"], "model": r["model"], "status": r["status"], "new_outputs": r.get("new_outputs"),
                           "new_tokens": r.get("new_tokens"), "budget": r.get("api_budget_today")} for r in RESULTS], indent=1))
        ledger = KRP.ledger_load(KRP.ledger_path(PLAN))
        print("ledger (requests and tokens per model per UTC day):", json.dumps(ledger, indent=1))
        ''', tags=[RUN_TAG]),
        code('''
        # keep the outputs: copy data/runs, the ledger and the compute log to Drive (Colab) or the working dir
        import shutil
        from pathlib import Path
        dest = Path("/content/drive/MyDrive/noilai/runs_api") if Path("/content/drive/MyDrive").exists() else Path(WORK_DIR) / "noilai_runs_out"
        shutil.copytree(PROJECT / "data" / "runs", dest / "runs", dirs_exist_ok=True)
        import compute_log as CL
        LOG = PROJECT / "data" / "compute_log.csv"
        CL.append_entry(LOG, CL.Entry(date=CL.today(), platform=PLATFORM, gpu_type=GPU_TYPE, n_gpus=N_GPUS,
                                      hours=CL.hours_since(SESSION_T0), run_id=f"session-{RUN_LABEL}", purpose="api_runs"))
        shutil.copy2(LOG, dest / "compute_log.csv")
        print(CL.format_totals(CL.totals(CL.read_entries(LOG))))
        print("outputs copied to", dest)
        '''),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update(_metadata("colab_or_kaggle", "none (API)"))
    return nb


# =============================================================================== assembly
def _metadata(platform: str, accelerator: str) -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "noilai": {"platform": platform, "accelerator": accelerator, "generator": "scripts/kaggle_build_notebooks.py",
                   "non_interactive": True, "plan": "docs/PLAN_2026-09-30.md"},
    }


BUILDERS = {
    "kaggle_eval_t4": build_kaggle_t4,
    "kaggle_eval_tpu": build_kaggle_tpu,
    "colab_probe_gemma3": build_colab_probe,
    "api_runs": build_api_runs,
}


def build_all() -> dict[str, nbformat.NotebookNode]:
    out = {}
    for name in NOTEBOOKS:
        nb = BUILDERS[name]()
        nbformat.validate(nb)
        out[name] = nb
    return out


def sources(nb: nbformat.NotebookNode) -> list[tuple[str, str, list[str]]]:
    """(cell_type, source, tags) per cell: what the drift check compares (outputs/ids are ignored)."""
    return [(c.cell_type, c.source, list(c.metadata.get("tags", []))) for c in nb.cells]


def write_all(out_dir: Path = NOTEBOOK_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, nb in build_all().items():
        p = out_dir / f"{name}.ipynb"
        nbformat.write(nb, str(p))
        paths.append(p)
    return paths


def check(out_dir: Path = NOTEBOOK_DIR) -> list[str]:
    """Names of notebooks whose cell sources differ from this generator (or are missing)."""
    drifted = []
    for name, nb in build_all().items():
        p = out_dir / f"{name}.ipynb"
        if not p.exists():
            drifted.append(f"{name}: missing")
            continue
        on_disk = nbformat.read(str(p), as_version=4)
        if sources(on_disk) != sources(nb):
            drifted.append(f"{name}: cell sources differ from scripts/kaggle_build_notebooks.py")
    return drifted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=NOTEBOOK_DIR)
    args = ap.parse_args(argv)
    if args.write:
        for p in write_all(args.out):
            print("[write]", p)
        return 0
    drift = check(args.out)
    for d in drift:
        print("[drift]", d)
    print("[ok   ] notebooks match the generator" if not drift else f"[FAIL ] {len(drift)} notebook(s) drifted; run --write")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
