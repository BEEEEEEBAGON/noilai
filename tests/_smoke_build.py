"""Where the local NON-frozen smoke build lives, shared by test_exploratory_plan, test_interim_analysis and test_ledger
(not a test module: pytest collects tests/test_*.py only, and the README's test list names those).

The frozen release files are absent from CI and from most checkouts; a smoke build made outside data/release/
(`python scripts/build_data.py --out <dir> ...`) stands in for exercising code. NOILAI_SMOKE_BUILD names its directory
(the one holding noilai_dev.jsonl); unset, the git-ignored data/smoke_build/v0.3-smoke is tried; when neither exists
the callers keep their skip, or build a small seeded set in a temporary directory.
"""
import os
from pathlib import Path

ENV_VAR = "NOILAI_SMOKE_BUILD"
# the fallback when NOILAI_SMOKE_BUILD is unset: the git-ignored data/smoke_build/ (a copy or symlink of a local build)
DEFAULT_SMOKE_BUILD = Path(__file__).resolve().parents[1] / "data" / "smoke_build" / "v0.3-smoke"


def smoke_build_dir() -> Path:
    """The smoke build's directory: $NOILAI_SMOKE_BUILD when set, else DEFAULT_SMOKE_BUILD. It may not exist: callers
    test `.exists()` (their skip conditions) exactly as before."""
    env = os.environ.get(ENV_VAR, "").strip()
    return Path(env).expanduser() if env else DEFAULT_SMOKE_BUILD


def smoke_dev_file() -> Path:
    """`noilai_dev.jsonl` of the smoke build (may not exist)."""
    return smoke_build_dir() / "noilai_dev.jsonl"
