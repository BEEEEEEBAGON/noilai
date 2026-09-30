#!/usr/bin/env python
"""Bootstrap helpers shared by the Colab and Kaggle notebooks: Drive mount, token-safe clone,
project directory, pinned pip install, environment record, secrets lookup.

Design rules (docs/PLAN_2026-09-30.md, section 3; the notebooks' cells follow them):

* A token never appears in a command line, a URL or a log. Git receives it through
  GIT_ASKPASS (a two-line shell script that echoes an environment variable), so `ps`, the
  notebook output and the clone's .git/config stay clean. GIT_TERMINAL_PROMPT=0 makes a
  missing token fail fast instead of hanging a non-interactive cell.
* A git bundle in Drive / a Kaggle dataset is preferred to GitHub: private repositories are
  invisible to the Colab opener and a bundle needs no token at all. Cloning from a bundle,
  then checking out `ref`, gives the same commit as a network clone.
* The repository may hold the subproject in a subdirectory (`noilai/`); `project_dir`
  resolves either layout so that notebooks work before and after a repository split.
* Every notebook records `pip freeze` and the framework versions next to its outputs
  (checklist C4) and passes them to the run manifest.

Only the command construction runs in tests; the functions that call git, pip and Drive
are thin wrappers around those commands.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_TOKEN_ENV = "GITHUB_TOKEN"
DEFAULT_SUBDIR = "noilai"


# ------------------------------------------------------------------ secrets
def get_secret(name: str) -> str | None:
    """Look a secret up in the environment, then Colab's userdata, then Kaggle Secrets.

    Returns None when absent. Never prints the value; callers put it in os.environ for the
    library that needs it and print only whether it was found.
    """
    val = os.environ.get(name)
    if val:
        return val
    try:  # Colab
        from google.colab import userdata  # type: ignore

        try:
            val = userdata.get(name)
        except Exception:  # noqa: BLE001 - SecretNotFoundError / NotebookAccessError
            val = None
        if val:
            return val
    except ImportError:
        pass
    try:  # Kaggle
        from kaggle_secrets import UserSecretsClient  # type: ignore

        try:
            val = UserSecretsClient().get_secret(name)
        except Exception:  # noqa: BLE001
            val = None
        if val:
            return val
    except ImportError:
        pass
    return None


def export_secret(name: str, required: bool = False) -> bool:
    """Copy a secret into os.environ under the same name; report presence, not the value."""
    val = get_secret(name)
    if val:
        os.environ[name] = val
        print(f"[secret] {name}: found ({len(val)} chars)")
        return True
    msg = f"[secret] {name}: not found"
    if required:
        raise RuntimeError(msg)
    print(msg)
    return False


# ------------------------------------------------------------------ Drive
def mount_drive(mountpoint: str = "/content/drive") -> Path | None:
    """Mount Google Drive on Colab; return the mountpoint, or None outside Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        print("[drive] not on Colab; skipping the mount")
        return None
    drive.mount(mountpoint)
    return Path(mountpoint)


# ------------------------------------------------------------------ clone
def write_askpass(token_env: str = DEFAULT_TOKEN_ENV, path: Path | None = None) -> Path:
    """Write the GIT_ASKPASS helper: it prints the token from the environment when git asks."""
    path = Path(path or Path.home() / ".noilai_askpass.sh")
    path.write_text(f"#!/bin/sh\necho \"${token_env}\"\n", encoding="utf-8")
    path.chmod(stat.S_IRWXU)
    return path


def has_embedded_credentials(repo_url: str) -> bool:
    """True for an http(s) URL with a userinfo part (https://user:token@host/...). An SSH remote
    such as git@github.com:owner/repo.git carries no credential and is accepted."""
    parsed = urlparse(repo_url)
    if parsed.scheme in ("http", "https"):
        return bool(parsed.username or parsed.password)
    return False


def clone_command(repo_url: str, dest: Path, bundle_path: Path | None = None, depth: int | None = None) -> list[str]:
    """`git clone` argv. From a bundle when one is given, else from the URL; never a token."""
    if has_embedded_credentials(repo_url):
        raise ValueError("repo_url must not embed credentials; use GIT_ASKPASS")
    cmd = ["git", "clone"]
    if depth and not bundle_path:
        cmd += ["--depth", str(depth)]
    src = str(bundle_path) if bundle_path else repo_url
    return cmd + [src, str(dest)]


def fetch_command(dest: Path, bundle_path: Path | None = None) -> list[str]:
    """Refresh an existing clone: fetch from the bundle (as a remote path) or from origin."""
    if bundle_path:
        return ["git", "-C", str(dest), "fetch", str(bundle_path), "+refs/heads/*:refs/remotes/bundle/*"]
    return ["git", "-C", str(dest), "fetch", "--all", "--tags"]


_HEX_REF = re.compile(r"^[0-9a-f]{7,40}$")


def is_commit_hash(ref: str) -> bool:
    return bool(_HEX_REF.fullmatch(ref.strip().lower()))


def checkout_candidates(ref: str, remote: str = "origin") -> list[str]:
    """The refs to try, in order, when checking out `ref` in a clone that may hold a STALE local
    branch of the same name: the freshly fetched remote-tracking ref first (origin/main), the
    literal ref last (commit hashes and tags have no remote-tracking form). A persisted clone
    (Drive, a re-used Kaggle dataset) therefore runs the fetched code, not last week's branch."""
    ref = ref.strip()
    if is_commit_hash(ref):
        return [ref]
    return [f"{remote}/{ref}", ref]


def checkout_command(dest: Path, ref: str) -> list[str]:
    return ["git", "-C", str(dest), "checkout", "--detach", ref]


def git_env(token_env: str = DEFAULT_TOKEN_ENV, askpass: Path | None = None) -> dict:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    if env.get(token_env):
        env["GIT_ASKPASS"] = str(askpass or write_askpass(token_env))
    return env


def clone(repo_url: str, dest: Path, ref: str | None = None, bundle_path: Path | None = None,
          token_env: str = DEFAULT_TOKEN_ENV, depth: int | None = None) -> Path:
    """Clone (or refresh) the repository at `dest` and check out `ref` when given."""
    dest = Path(dest)
    if bundle_path is not None and not Path(bundle_path).exists():
        print(f"[clone] bundle {bundle_path} not found; falling back to {repo_url}")
        bundle_path = None
    env = git_env(token_env)
    existing = (dest / ".git").exists()
    if existing:
        cmd = fetch_command(dest, bundle_path)
    else:
        cmd = clone_command(repo_url, dest, bundle_path=bundle_path, depth=depth)
    print("[clone]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    if ref:
        remote = "bundle" if (existing and bundle_path) else "origin"
        for cand in checkout_candidates(ref, remote):
            if subprocess.run(checkout_command(dest, cand), env=env, check=False).returncode == 0:
                break
        else:
            raise RuntimeError(f"cannot check out {ref!r} (tried {checkout_candidates(ref, remote)})")
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, env=env,
                          capture_output=True, text=True).stdout.strip()
    print(f"[clone] HEAD {head}")
    return dest


def project_dir(clone_dir: Path, subdir: str = DEFAULT_SUBDIR) -> Path:
    """The subproject root: <clone>/<subdir> when it holds pyproject.toml, else the clone itself."""
    clone_dir = Path(clone_dir)
    cand = clone_dir / subdir
    if (cand / "pyproject.toml").exists():
        return cand
    if (clone_dir / "pyproject.toml").exists():
        return clone_dir
    raise FileNotFoundError(f"no pyproject.toml under {clone_dir} or {cand}")


def git_head(project: Path) -> dict:
    """Commit and dirty flag of the checkout the notebook runs from (for the manifests)."""
    try:
        commit = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(project), "status", "--porcelain", "--", "."], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_dirty": True}
    return {"git_commit": commit, "git_dirty": dirty}


# ------------------------------------------------------------------ install
def pip_install_command(project: Path, extras: Sequence[str] = ("eval",), pins: dict | None = None,
                        python: str = sys.executable) -> list[list[str]]:
    """Pinned packages first (so that e.g. vllm's own requirements win), then the project.

    `pins` maps package -> version; a None version installs the package unpinned.
    """
    cmds: list[list[str]] = []
    if pins:
        specs = [f"{k}=={v}" if v else k for k, v in pins.items()]
        cmds.append([python, "-m", "pip", "install", "-q", *specs])
    spec = str(project) + (f"[{','.join(extras)}]" if extras else "")
    cmds.append([python, "-m", "pip", "install", "-q", "-e", spec])
    return cmds


def pip_install(project: Path, extras: Sequence[str] = ("eval",), pins: dict | None = None) -> None:
    for cmd in pip_install_command(project, extras, pins):
        print("[pip  ]", " ".join(cmd))
        subprocess.run(cmd, check=True)


def record_environment(out_dir: Path, python: str = sys.executable) -> Path:
    """pip freeze + interpreter/torch/CUDA versions into <out_dir>/env_<utc>.json (checklist C4)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout
    info: dict = {"utc": stamp, "python": platform.python_version(), "platform": platform.platform(),
                  "pip_freeze": freeze.splitlines()}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["devices"] = [{"name": torch.cuda.get_device_name(i),
                                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                               for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"unavailable: {e}"
    for mod in ("transformers", "vllm", "sentencepiece"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None
    p = out_dir / f"env_{stamp}.json"
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / f"pip_freeze_{stamp}.txt").write_text(freeze, encoding="utf-8")
    return p


def cuda_summary() -> dict:
    """Devices, compute capability and whether vLLM's floor (7.5) is met; empty when no CUDA."""
    try:
        import torch  # type: ignore
    except ImportError:
        return {"torch": None, "cuda": False, "devices": []}
    out = {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "devices": []}
    if out["cuda"]:
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            out["devices"].append({"index": i, "name": torch.cuda.get_device_name(i),
                                   "capability": f"{cap[0]}.{cap[1]}", "vllm_ok": cap >= (7, 5),
                                   "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)})
    return out


if __name__ == "__main__":  # pragma: no cover - a quick self-description when run directly
    print(json.dumps(cuda_summary(), indent=2))
