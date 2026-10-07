"""Record the scoring-rule freeze: a hash over the code that turns a model output into a score.

    python scripts/freeze_scoring_rule.py            # print the hash and the files it covers
    python scripts/freeze_scoring_rule.py --record   # write commit + hash + date into experiments/gates.yaml
    python scripts/freeze_scoring_rule.py --check    # exit 1 if the working tree's scoring code differs from the recorded hash

The gate of the experiments workstream: conventions that only affect SCORING (canonicalization,
the i/y and placement equivalences, the error taxonomy, answer extraction) do not block runs, but
the rule must be frozen before anyone looks at confirmatory results. The hash covers
noilai/eval/score.py, noilai/eval/extract.py, noilai/vi/syllable.py, noilai/vi/unicode.py,
noilai/vi/lexicon.py, noilai/vi/reencode.py, noilai/constants.py and the prompt files (the answer
marker the extractor looks for lives there); DESIGN_DECISIONS 5.1-5.5. `--check` is what the
ingest step runs before scoring a confirmatory unit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "experiments" / "gates.yaml"
FILES = ["noilai/eval/score.py", "noilai/eval/extract.py", "noilai/vi/syllable.py", "noilai/vi/unicode.py",
         "noilai/vi/lexicon.py", "noilai/vi/reencode.py", "noilai/constants.py", "prompts/noilai.yaml", "prompts/demos.yaml"]


def scoring_rule_sha256(root: Path = ROOT, files: list[str] = FILES) -> tuple[str, list[str]]:
    h = hashlib.sha256()
    covered = []
    for rel in files:
        p = root / rel
        if not p.exists():
            continue
        h.update(rel.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
        covered.append(rel)
    return h.hexdigest(), covered


def git_head(root: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", type=Path, default=GATES)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    digest, covered = scoring_rule_sha256(args.root)
    print(f"scoring rule sha256 {digest} over {len(covered)} files: {', '.join(covered)}")
    if args.check:
        g = yaml.safe_load(args.gates.read_text(encoding="utf-8")) if args.gates.exists() else {}
        rec = ((g or {}).get("scoring_rule_freeze") or {}).get("files_sha256")
        if not rec:
            print("no scoring-rule freeze recorded in", args.gates)
            return 1
        if rec != digest:
            print(f"SCORING RULE CHANGED since the freeze: recorded {rec}")
            return 1
        print("scoring rule unchanged since the freeze")
        return 0
    if args.record:
        record(args.gates, git_head(args.root), digest, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
        print("recorded in", args.gates)
    return 0


def record(gates: Path, commit: str | None, digest: str, frozen_on: str) -> None:
    """Write the three values of the `scoring_rule_freeze:` block IN PLACE, keeping every comment of the
    gates file (a YAML dump would drop them); the block is appended when the file lacks it."""
    gates.parent.mkdir(parents=True, exist_ok=True)
    text = gates.read_text(encoding="utf-8") if gates.exists() else ""
    values = {"commit": commit, "files_sha256": digest, "frozen_on": frozen_on}
    if re.search(r"(?m)^scoring_rule_freeze:\s*$", text):
        head, _, block = text.partition("scoring_rule_freeze:")
        lines = block.split("\n")
        seen = set()
        for i, line in enumerate(lines):
            m = re.match(r"^(\s+)(commit|files_sha256|frozen_on):\s*(.*)$", line)
            if m and m.group(2) not in seen:
                val = values[m.group(2)]
                lines[i] = f"{m.group(1)}{m.group(2)}: {'null' if val is None else val}"
                seen.add(m.group(2))
        for key in ("commit", "files_sha256", "frozen_on"):
            if key not in seen:
                lines.insert(1, f"  {key}: {'null' if values[key] is None else values[key]}")
        text = head + "scoring_rule_freeze:" + "\n".join(lines)
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + "scoring_rule_freeze:\n" + "".join(
            f"  {k}: {'null' if v is None else v}\n" for k, v in values.items())
    gates.write_text(text, encoding="utf-8")
    yaml.safe_load(text)                      # still valid YAML


if __name__ == "__main__":
    sys.exit(main())
