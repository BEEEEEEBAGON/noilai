"""Run manifest for the E4 driver (design 7.5, 9.6): the commit and dirty flag, one hash over
the probe code, the hashes of the external resources (data/HASHES.json), the seeds, and the
per-filter retention counts of the patching stage (design 9.3: retention at every filter).
A result without a manifest is not a result."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HASHES_FILE = ROOT / "data" / "HASHES.json"
CODE_FILES = ("noilai/probe", "noilai/vi", "noilai/stats/bootstrap.py", "scripts/run_probe.py")


def git_info(root: Path = ROOT) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": (bool(status) if status is not None else None), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def code_sha256(root: Path = ROOT, files=CODE_FILES) -> str:
    """One hash over the probe code the run depends on, file names included, sorted."""
    h = hashlib.sha256()
    paths = []
    for f in files:
        p = root / f
        paths += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for p in sorted(set(paths)):
        if p.exists():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def resource_hashes() -> dict:
    if HASHES_FILE.exists():
        return json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    return {}


def base_manifest(**fields) -> dict:
    return {**git_info(), "code_sha256": code_sha256(), "resource_hashes": resource_hashes(),
            "python": platform.python_version(), "platform": platform.platform(),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}


def write_manifest(path: Path, manifest: dict) -> None:
    manifest = {**manifest, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
