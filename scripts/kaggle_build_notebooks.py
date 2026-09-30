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
    and reported as found/not found, never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = ("kaggle_eval_t4", "kaggle_eval_tpu", "colab_probe_gemma3", "api_runs")

RUN_TAG = "run"
E4_TAG = "e4"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(dedent(text).strip("\n"))


def code(text: str, tags: Optional[list[str]] = None) -> nbformat.NotebookNode:
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
for cand in (REPO_REF, f"origin/{REPO_REF}"):
    if subprocess.run(["git", "-C", str(clone_dir), "checkout", "--detach", cand], env=env).returncode == 0:
        break
else:
    raise SystemExit(f"cannot check out {REPO_REF}")
HEAD = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"], check=True, env=env,
                      capture_output=True, text=True).stdout.strip()
PROJECT = clone_dir / REPO_SUBDIR if (clone_dir / REPO_SUBDIR / "pyproject.toml").exists() else clone_dir
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "scripts"))
print("HEAD", HEAD)
print("project", PROJECT)
'''

INSTALL_GPU = '''
# (c) pinned install: the inference engine first so that its own requirements win, then the project
# with its eval extra; then record pip freeze + versions next to the outputs (checklist C4).
import subprocess, sys
from pathlib import Path
pins = [f"vllm=={VLLM_VERSION}"]
if TRANSFORMERS_VERSION:
    pins.append(f"transformers=={TRANSFORMERS_VERSION}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval]"], check=True)
os.environ["HF_HOME"] = HF_HOME
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
_export(HF_TOKEN_SECRET)          # gated checkpoints (Gemma, Llama, Vistral); exported as HF_TOKEN when the secret has that name
if HF_TOKEN_SECRET != "HF_TOKEN" and os.environ.get(HF_TOKEN_SECRET):
    os.environ["HF_TOKEN"] = os.environ[HF_TOKEN_SECRET]
import colab_setup as CS
ENV_RECORD = CS.record_environment(PROJECT / "data" / "runs" / "env")
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
import colab_setup as CS
ENV_RECORD = CS.record_environment(PROJECT / "data" / "runs" / "env")
print("environment recorded at", ENV_RECORD)
'''

FETCH_VERIFY = '''
# (d) third-party resources (SHA-256 against data/HASHES.json) and the item file(s): hash against
# configs/run_plan.yaml, canary on every row, row count. A failure stops the notebook here.
import shutil, subprocess, sys
from pathlib import Path
r = subprocess.run([sys.executable, "scripts/fetch_resources.py"], check=False)
if r.returncode:
    raise SystemExit("resource fetch or hash check failed (see above)")
if ITEMS_DATASET_DIR:
    src = Path(ITEMS_DATASET_DIR)
    if not src.exists():
        raise SystemExit(f"ITEMS_DATASET_DIR {src} not found: attach the private dataset that holds data/release")
    shutil.copytree(src, PROJECT / "data" / "release", dirs_exist_ok=True)
    print("release copied from", src)
for key in VERIFY_ITEM_KEYS:
    r = subprocess.run([sys.executable, "scripts/kaggle_verify_items.py", "--key", key], check=False)
    if r.returncode:
        raise SystemExit(f"item file {key} failed verification; nothing runs on it")
'''

OUTPUTS_KAGGLE = '''
# (g) keep the outputs: copy data/runs (+ compute log) into /kaggle/working (saved with the notebook) and
# add a version to a private Kaggle dataset through scripts/kaggle_dataset.py (credentials from Secrets).
import os, shutil, subprocess, sys
from pathlib import Path
out = Path(WORK_DIR) / "noilai_runs_out"
shutil.copytree(PROJECT / "data" / "runs", out / "runs", dirs_exist_ok=True)
if (PROJECT / "data" / "compute_log.csv").exists():
    shutil.copy2(PROJECT / "data" / "compute_log.csv", out / "compute_log.csv")
print("outputs copied to", out)
if PUSH_DATASET:
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        _export(name)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        cmd = [sys.executable, "scripts/kaggle_dataset.py", "--src", str(PROJECT / "data" / "runs"),
               "--src", str(PROJECT / "data" / "compute_log.csv"), "--dest", str(Path(WORK_DIR) / "noilai_runs_dataset"),
               "--slug", KAGGLE_DATASET_SLUG, "--message", f"{RUN_LABEL} @ {HEAD[:8]}"]
        if CREATE_DATASET:
            cmd.append("--create")
        subprocess.run(cmd, check=False)
    else:
        print("Kaggle credentials not found: dataset push skipped (outputs are still under", out, ")")
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
        (12 h max, ~30 GPU-h/week). Every cell is non-interactive: set the **parameters** cell,
        then *Run all*. Attach before starting:

        * accelerator **GPU T4 ×2**; internet on;
        * Kaggle Secrets: `GITHUB_TOKEN` (repository read, only if no bundle), `HF_TOKEN`
          (gated Gemma/Llama/Vistral), `KAGGLE_USERNAME` + `KAGGLE_KEY` (dataset push);
        * a private dataset holding `noilai-main.bundle` (`git bundle create noilai-main.bundle main`)
          and one holding the frozen `data/release/` (never public: the test split is gated).

        Cells: (a) environment · (b) clone · (c) pinned install · (d) resources + item hash ·
        (e) smoke test · (f) **RUN** · (g) outputs → dataset · (h) GPU-hours log.
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
        ITEMS_DATASET_DIR = "/kaggle/input/noilai-release"  # private dataset with the frozen data/release/<version>/ tree; None if the clone has it
        VERIFY_ITEM_KEYS = ["noilai_test", "noilai_dev"]    # keys of configs/run_plan.yaml item_files checked in (d)

        VLLM_VERSION = "0.11.0"          # [UNCERTAIN: verify] pin to the version the smoke tests pass with; recorded by (c)
        TRANSFORMERS_VERSION = None      # None = whatever the vllm wheel requires (pip freeze is recorded either way)

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        HF_TOKEN_SECRET = "HF_TOKEN"

        RUN_ID = "E1_main"               # a run id from configs/run_plan.yaml (kaggle_run_plan.py --list)
        MODELS = ["gemma-3-1b-it"]       # subset of the run's models; [] = every model of the run
        SMOKE_MODEL = "gemma-3-1b-it"    # (e) 20 items on this model before the run
        EXTRA_ARGS = ""                  # extra run_eval.py flags, one shell-quoted string, e.g. "--seed 7"
        DRY_RUN = False                  # True prints the commands and runs nothing

        PLATFORM, GPU_TYPE, N_GPUS = "kaggle", "t4", 2
        REQUIRE_GPU = True
        RUN_LABEL = RUN_ID
        PUSH_DATASET = True
        KAGGLE_DATASET_SLUG = "noilai-runs"
        CREATE_DATASET = False           # True the first time the dataset is published
        '''),
        code(SECRET_HELPER),
        code(GPU_CHECK),
        code(CLONE),
        code(INSTALL_GPU),
        code(FETCH_VERIFY),
        md("""
        ### RUN CELL (e) — smoke test
        One small model, 20 items (`smoke_20` in `run_plan.yaml`): load, generate, log-probs. Stops the
        notebook on failure so that no GPU hour is spent on a broken environment.
        """),
        code('''
        # RUN CELL (e): smoke test. Flags come from run_plan.yaml (smoke_20) via scripts/kaggle_run_plan.py.
        import shlex
        import kaggle_run_plan as KRP
        SMOKE_RESULTS = KRP.execute("smoke_20", models=[SMOKE_MODEL], platform=PLATFORM, dry_run=DRY_RUN,
                                    continue_on_error=False, extra=shlex.split(EXTRA_ARGS))
        for r in SMOKE_RESULTS:
            print(r["model"], "->", r["status"])
        if any(str(r["status"]).startswith("failed") for r in SMOKE_RESULTS):
            raise SystemExit("smoke test failed: fix the environment or the model entry before the run cell")
        ''', tags=[RUN_TAG]),
        md("""
        ### RUN CELL (f) — the run
        Loops over `MODELS` (or every model of `RUN_ID`) calling
        `scripts/run_eval.py --model-config <name> --items ... --tasks ... --arms ... --resume --run-id <RUN_ID>__<name> --out-root data/runs`.
        `--resume` makes a re-run after a session timeout continue where it stopped. Each model's wall
        time is appended to `data/compute_log.csv` with the device type from `models.yaml`.
        """),
        code('''
        # RUN CELL (f): the run matrix line RUN_ID over MODELS, with --resume. Adjust flags in
        # scripts/kaggle_run_plan.py (build_command) or through EXTRA_ARGS; the CLI is owned by noilai/eval.
        import json, shlex
        import kaggle_run_plan as KRP
        RESULTS = KRP.execute(RUN_ID, models=MODELS or None, platform=PLATFORM, dry_run=DRY_RUN,
                              continue_on_error=True, extra=shlex.split(EXTRA_ARGS))
        print(json.dumps([{"model": r["model"], "status": r["status"], "hours": r.get("hours"),
                           "new_outputs": r.get("new_outputs")} for r in RESULTS], indent=1))
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
        Gemma 3 12B, Qwen3.8-27B, Qwen-SEA-LION-v4.5-27B-IT in bf16), the bf16 drift check
        (`bf16_drift_200`: 200 core items per T4 model, unquantized) and the 27B reasoning
        sub-study (`reasoning_500_tpu`). Sessions last 9 h (~20 TPU-h/week), one at a time; the
        vLLM-TPU startup takes 6–22 minutes per model, so keep `MODELS` short per session and rely
        on `--resume`. Attach: accelerator **TPU VM v5e-8**, internet on, the same secrets and
        private datasets as the T4 notebook.

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
        VERIFY_ITEM_KEYS = ["noilai_test"]

        VLLM_TPU_PACKAGE = "vllm-tpu"    # [UNCERTAIN: verify] the TPU build's package name for the pinned version (kaggle-tpu-lab documents the working recipe)
        VLLM_TPU_VERSION = None          # [UNCERTAIN: verify] pin once the smoke cell passes; None = latest (recorded by (c))

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        HF_TOKEN_SECRET = "HF_TOKEN"

        RUN_IDS = ["tpu_main"]           # any of: tpu_main, bf16_drift_200, reasoning_500_tpu
        MODELS = ["gemma-3-12b-it"]      # subset of each run's models; [] = all
        SMOKE_MODEL = "gemma-3-12b-it"
        EXTRA_ARGS = ""
        DRY_RUN = False

        PLATFORM, GPU_TYPE, N_GPUS = "kaggle", "tpu-v5e-8", 1
        REQUIRE_TPU = True
        RUN_LABEL = "+".join(RUN_IDS)
        PUSH_DATASET = True
        KAGGLE_DATASET_SLUG = "noilai-runs"
        CREATE_DATASET = False
        '''),
        code(SECRET_HELPER),
        code(TPU_CHECK),
        code(CLONE),
        code(INSTALL_TPU),
        code(FETCH_VERIFY),
        md("""
        ### RUN CELL (e) — smoke test on the TPU
        `smoke_20` on `SMOKE_MODEL`: the model must load under vLLM-TPU, generate, and return prompt
        log-probabilities (the T3 forced-choice scoring needs them).
        """),
        code('''
        # RUN CELL (e): smoke test (smoke_20) on the TPU model.
        import shlex
        import kaggle_run_plan as KRP
        SMOKE_RESULTS = KRP.execute("smoke_20", models=[SMOKE_MODEL], platform=PLATFORM, dry_run=DRY_RUN,
                                    continue_on_error=False, extra=shlex.split(EXTRA_ARGS))
        for r in SMOKE_RESULTS:
            print(r["model"], "->", r["status"])
        if any(str(r["status"]).startswith("failed") for r in SMOKE_RESULTS):
            raise SystemExit("smoke test failed: the TPU route for this model is not ready (fallback: 2xT4 4-bit, labelled quantized)")
        ''', tags=[RUN_TAG]),
        md("""
        ### RUN CELL (f) — the TPU runs
        One `run_eval.py --resume` command per (run id, model); wall time per model is logged as
        TPU device-hours.
        """),
        code('''
        # RUN CELL (f): each run id in RUN_IDS over MODELS (or all of the run's models), --resume.
        import json, shlex
        import kaggle_run_plan as KRP
        RESULTS = []
        for run_id in RUN_IDS:
            RESULTS += KRP.execute(run_id, models=MODELS or None, platform=PLATFORM, dry_run=DRY_RUN,
                                   continue_on_error=True, extra=shlex.split(EXTRA_ARGS))
        print(json.dumps([{"run": r["run"], "model": r["model"], "status": r["status"], "hours": r.get("hours")} for r in RESULTS], indent=1))
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
        # NóiLái — E4 skeleton: tone/onset/rime probes and patching in Gemma 3 (Colab)

        Loads **Gemma 3 1B (IT)** in float32 (bf16 only on a bf16-capable GPU: L4/A100; a T4 has
        none, and Gemma 3 must never run in fp16), extracts residual-stream states at each
        syllable's last sub-token under NFC and NFD with `noilai.probe.extract`, fits layer-wise
        probes with control tasks (`noilai.probe.probes`), and holds a patching skeleton
        (`noilai.probe.patching`) for the P pair *bí mật* (clean) vs *bí mất* (corrupt). Results
        are written to Drive. Colab secrets: `HF_TOKEN` (gated Gemma), `GITHUB_TOKEN` if no bundle.

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

        MODEL_CONFIG = "gemma-3-1b-it"    # entry of configs/models.yaml (hf_id read from it); gemma-3-4b-it needs an L4/A100 for fp32
        DTYPE = "float32"                 # "bfloat16" only on a bf16-capable GPU; never "float16" for Gemma 3
        N_SYLLABLES = 600                 # tone-bearing syllables sampled from the inventory (train/test split by syllable)
        ENCODINGS = ["nfc", "nfd"]
        FEATURES = ["tone", "onset", "rime"]
        POSITION = "last"                 # "last" sub-token of the syllable or "after"
        SEED = 20261004
        BATCH_SIZE = 8

        PLATFORM, GPU_TYPE, N_GPUS = "colab", "t4", 1
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
        # load the model named by MODEL_CONFIG in DTYPE (fp32 by default: Gemma 3 overflows in fp16)
        import torch, yaml
        from transformers import AutoModelForCausalLM, AutoTokenizer
        entry = next(m for m in yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))["models"] if m["name"] == MODEL_CONFIG)
        if entry.get("family") == "gemma3" and DTYPE == "float16":
            raise SystemExit("Gemma 3 must not run in float16 (transformers PR #36832)")
        if DTYPE == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (8, 0):
            raise SystemExit("this GPU has no bfloat16 units; use float32 (or an L4/A100 runtime)")
        HF_ID = entry["hf_id"]
        print("loading", HF_ID, "in", DTYPE, "| hf_id_status:", entry.get("hf_id_status"))
        tok = AutoTokenizer.from_pretrained(HF_ID)
        model = AutoModelForCausalLM.from_pretrained(HF_ID, torch_dtype=getattr(torch, DTYPE), device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
        print("layers:", model.config.num_hidden_layers, "| hidden:", model.config.hidden_size)
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
        ### E4 CELL — activation patching skeleton (P pair *bí mật* / *bí mất*)
        V3 (swap tones) maps the clean input *bí mật* to *bị mất* and leaves the corrupt input *bí mất*
        unchanged, so the first answer token differs (*bị* vs *bí*). The cell caches the clean run,
        patches the corrupt run layer by layer at the second syllable's positions and reports recovery
        of the logit difference. The prompt below is a placeholder until the prompt file is frozen
        (NATIVE-CHECK its wording); token spans must match between clean and corrupt runs.
        """),
        code('''
        # E4 CELL: patching skeleton. Adjust PROMPT to the frozen prompt file; the answer position is the
        # last prompt token, the patched positions are the tokens of the second syllable.
        import torch
        from noilai.probe import patching

        # NATIVE-CHECK: instruction wording (variant V3 = swap the tones, keep onsets and rimes)
        PROMPT = "Nói lái kiểu đổi thanh, giữ phụ âm đầu và vần. Ví dụ: hiện đại -> hiền đậi. Câu: {phrase}. Đáp án:"
        CLEAN, CORRUPT = "bí mật", "bí mất"
        ANSWER_CLEAN, ANSWER_CORRUPT = "bị", "bí"

        def _encode(phrase):
            text = PROMPT.format(phrase=phrase)
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
            start = text.index(phrase) + len(phrase.split()[0]) + 1          # second syllable
            end = start + len(phrase.split()[1])
            positions = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()) if b > start and a < end]
            return enc["input_ids"], positions

        clean_ids, clean_pos = _encode(CLEAN)
        corrupt_ids, corrupt_pos = _encode(CORRUPT)
        if clean_ids.shape != corrupt_ids.shape or clean_pos != corrupt_pos:
            print("token spans differ between clean and corrupt runs; pick a pair whose second syllables tokenize alike",
                  clean_ids.shape, corrupt_ids.shape, clean_pos, corrupt_pos)
        else:
            tok_clean = tok(" " + ANSWER_CLEAN, add_special_tokens=False)["input_ids"][0]
            tok_corrupt = tok(" " + ANSWER_CORRUPT, add_special_tokens=False)["input_ids"][0]
            device = next(model.parameters()).device
            result = patching.run_patching(model, clean_ids.to(device), corrupt_ids.to(device), answer_pos=clean_ids.shape[1] - 1,
                                           tok_clean=tok_clean, tok_corrupt=tok_corrupt, position_groups=[clean_pos])
            print(result)
            import json
            json.dump({"clean": CLEAN, "corrupt": CORRUPT, "positions": clean_pos, "result": str(result), "head": HEAD},
                      open(Path(OUT_DIR) / f"patching_{MODEL_CONFIG}_bi_mat.json", "w"), ensure_ascii=False, indent=2)
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
    nb.metadata.update(_metadata("colab", "T4 (fp32) or L4/A100 (bf16)"))
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

        **Daily-quota loop.** Groq's free plan allows 1,000 requests and 200K tokens a day per model,
        so the 1,500-item core takes about four days per model. The driver keeps
        `data/runs/api_ledger.json` (calls per model per UTC day, counted from new rows in
        `outputs.jsonl`) and skips a model whose daily request budget is spent; run this notebook
        once a day with `--resume` until every run is complete. Gemini's free-tier limits are shown
        only in AI Studio: read them and set `GEMINI_RPD_OVERRIDE` if the run must stop earlier.

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
        VERIFY_ITEM_KEYS = ["noilai_test"]

        GITHUB_TOKEN_SECRET = "GITHUB_TOKEN"
        API_KEY_SECRETS = ["GEMINI_API_KEY", "GROQ_API_KEY"]   # names only; values stay in the secret store
        GEMINI_RPD_OVERRIDE = None       # requests/day read in AI Studio; None = no ledger cap for Gemini [UNCERTAIN: verify]

        RUN_IDS = ["E1_api_core", "E1_api_paraphrase", "E3_api_core"]   # add "reasoning_500_api" for the cost curve
        MODELS = []                      # subset of the run's models; [] = all four
        EXTRA_ARGS = ""
        DRY_RUN = False

        PLATFORM, GPU_TYPE, N_GPUS = "api", "api", 0
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
        # API keys into the environment (names from configs/models.yaml providers.*.api_key_env); found / not found only
        import yaml
        cfg = yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))
        wanted = sorted({p["api_key_env"] for p in cfg["providers"].values()} & set(API_KEY_SECRETS))
        FOUND = {name: _export(name) for name in wanted}
        if not any(FOUND.values()):
            raise SystemExit("no API key found: add GEMINI_API_KEY / GROQ_API_KEY to the secret store")
        if GEMINI_RPD_OVERRIDE:
            cfg["providers"]["gemini"]["free_tier"]["requests_per_day"] = int(GEMINI_RPD_OVERRIDE)
            yaml.safe_dump(cfg, open("configs/models.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
            print("Gemini requests/day set to", GEMINI_RPD_OVERRIDE, "for this session's ledger (local edit, not committed)")
        '''),
        code(FETCH_VERIFY),
        md("""
        ### RUN CELL — daily API loop
        For each run id and model: skip if today's ledger says the daily request budget is spent, else
        `run_eval.py ... --in-core-only --resume`. Models whose key is missing are skipped with a
        message. Re-run tomorrow; `--resume` continues.
        """),
        code('''
        # RUN CELL: quota-aware loop over RUN_IDS x MODELS through scripts/kaggle_run_plan.py (in-core-only enforced there).
        import json, shlex
        import kaggle_run_plan as KRP
        plan, models_cfg = KRP.load_plan(), KRP.load_models()
        RESULTS = []
        for run_id in RUN_IDS:
            run = KRP.find_run(plan, run_id)
            names = MODELS or KRP.expand_models(run, plan, models_cfg)
            runnable = []
            for name in names:
                prov = models_cfg["by_name"][name].get("provider")
                key_env = models_cfg["providers"][prov]["api_key_env"] if prov else None
                if key_env and not os.environ.get(key_env):
                    print(f"skip {name}: {key_env} not set")
                else:
                    runnable.append(name)
            RESULTS += KRP.execute(run_id, models=runnable, platform=PLATFORM, dry_run=DRY_RUN, continue_on_error=True,
                                   extra=shlex.split(EXTRA_ARGS), plan=plan, models_cfg=models_cfg)
        print(json.dumps([{"run": r["run"], "model": r["model"], "status": r["status"], "new_outputs": r.get("new_outputs"),
                           "budget": r.get("api_budget_today")} for r in RESULTS], indent=1))
        ledger = KRP.ledger_load(KRP.ledger_path(plan))
        print("ledger (calls per model per UTC day):", json.dumps(ledger, indent=1))
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


def main(argv: Optional[list[str]] = None) -> int:
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
