#!/usr/bin/env python
"""Stage run outputs and push them to a Kaggle dataset with the `kaggle` CLI.

Kaggle's working directory holds about 20 GB and disappears with the session, so every
notebook copies data/runs (outputs.jsonl, scores.jsonl, manifest.json per run) plus the
compute log into a staging directory and publishes it as a versioned Kaggle dataset. The
author later drags the dataset's contents back under data/runs in the repository.

    python scripts/kaggle_dataset.py --src data/runs --src data/compute_log.csv \
        --dest /kaggle/working/noilai_runs --owner <kaggle-user> --slug noilai-runs --message "E1 batch 1"
    python scripts/kaggle_dataset.py ... --create        # first version of a new dataset
    python scripts/kaggle_dataset.py ... --dry-run       # print the CLI command, push nothing

The `kaggle` package is imported lazily and its absence is reported, not fatal, so this
module imports and its command construction is testable on a machine without it. API
credentials come from the environment (KAGGLE_USERNAME / KAGGLE_KEY, which Kaggle Secrets
set) or ~/.kaggle/kaggle.json; they are never read or printed here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git"}
DEFAULT_LICENSE = "CC-BY-NC-ND-4.0"   # the gated test split's license (plan 2.3); outputs quote test items


def kaggle_available() -> dict:
    """Which routes to the Kaggle API exist here (CLI binary, python package), without importing it."""
    return {"cli": shutil.which("kaggle") is not None,
            "package": importlib.util.find_spec("kaggle") is not None}


def stage(sources: Iterable[Path], dest: Path, clean: bool = False) -> list[Path]:
    """Copy each source (file or directory) under `dest`; return the copied top-level paths."""
    dest = Path(dest)
    if clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sources:
        src = Path(src)
        if not src.exists():
            print(f"[skip] {src} does not exist")
            continue
        target = dest / src.name
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True,
                            ignore=lambda d, names: [n for n in names if n in SKIP_DIR_NAMES])
        else:
            shutil.copy2(src, target)
        copied.append(target)
    return copied


def write_metadata(dest: Path, owner: str, slug: str, title: str, license_name: str = DEFAULT_LICENSE) -> Path:
    """dataset-metadata.json as the kaggle CLI expects it (id = owner/slug)."""
    if not owner or not slug:
        raise ValueError("owner and slug are required for the dataset id")
    meta = {"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": license_name}]}
    p = Path(dest) / "dataset-metadata.json"
    p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return p


def push_command(dest: Path, message: str, create: bool = False, public: bool = False) -> list[str]:
    """The kaggle CLI argv (pure; nothing is executed here)."""
    if create:
        cmd = ["kaggle", "datasets", "create", "-p", str(dest), "--dir-mode", "zip"]
        if public:
            cmd.append("--public")
        return cmd
    return ["kaggle", "datasets", "version", "-p", str(dest), "-m", message, "--dir-mode", "zip"]


def push(dest: Path, message: str, create: bool = False, public: bool = False, dry_run: bool = False) -> int:
    cmd = push_command(dest, message, create=create, public=public)
    print("[cmd ]", " ".join(cmd))
    if dry_run:
        return 0
    avail = kaggle_available()
    if not avail["cli"]:
        print("[FAIL] the `kaggle` CLI is not installed (pip install kaggle) or not on PATH", file=sys.stderr)
        return 2
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if var not in os.environ and not (Path.home() / ".kaggle" / "kaggle.json").exists():
            print(f"[FAIL] no Kaggle credentials: set {var} (Kaggle Secrets) or ~/.kaggle/kaggle.json", file=sys.stderr)
            return 2
    r = subprocess.run(cmd, check=False)
    return r.returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, action="append", default=None,
                    help="file or directory to stage (repeatable); default data/runs and data/compute_log.csv")
    ap.add_argument("--dest", type=Path, required=True, help="staging directory (becomes the dataset)")
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"), help="Kaggle user name (default $KAGGLE_USERNAME)")
    ap.add_argument("--slug", default="noilai-runs")
    ap.add_argument("--title", default="NoiLai model runs")
    ap.add_argument("--message", default="update")
    ap.add_argument("--license", default=DEFAULT_LICENSE)
    ap.add_argument("--create", action="store_true", help="create the dataset instead of adding a version")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--clean", action="store_true", help="empty the staging directory first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sources = args.src or [ROOT / "data" / "runs", ROOT / "data" / "compute_log.csv"]
    copied = stage(sources, args.dest, clean=args.clean)
    for c in copied:
        print(f"[copy] {c}")
    if not args.owner:
        print("[FAIL] --owner or KAGGLE_USERNAME is required", file=sys.stderr)
        return 2
    write_metadata(args.dest, args.owner, args.slug, args.title, args.license)
    return push(args.dest, args.message, create=args.create, public=args.public, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
