#!/usr/bin/env python
"""Release facts of the data statement, generated from the release manifest and attested file.

    python paper/gen_data_statement_facts.py --manifest data/release/v0.2/manifest.json \
        --attested data/release/v0.2/attested.jsonl --doc docs/DATA_STATEMENT.md
    python paper/gen_data_statement_facts.py ... --check      # exit 1 if the committed block is stale

docs/DATA_STATEMENT.md is released with the data, so its numbers must not be typed: this
script rewrites the block between the two marker comments

    <!-- BEGIN GENERATED release facts (paper/gen_data_statement_facts.py); do not edit by hand -->
    <!-- END GENERATED release facts -->

from `manifest.json` (`scripts/build_data.py`) and `attested.jsonl` (`scripts/build_attested.py`):
release path and build date, generator commit (never the seed: DESIGN_DECISIONS 4.6 withholds the
build seed because one seeded stream builds dev and test), item counts per split, the stored
tone-mark placement convention, vulgar-flagged counts, and the attested seed's counts (rows,
exact reproductions, rule-vs-folk mismatches with both forms, three-syllable rows, vulgar rows,
low-confidence rows). Nothing else in the statement is touched. tests/test_paper.py runs the
--check mode, so a rebuilt release that is not re-rendered here fails the suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN GENERATED release facts (paper/gen_data_statement_facts.py); do not edit by hand -->"
END = "<!-- END GENERATED release facts -->"
PLACEMENT_NAME = {"old": "old style (`hòa`, `khỏe`, `thủy`: mark on the first vowel letter of the open rimes oa, oe, uy)",
                  "new": "new style (`hoà`, `khoẻ`, `thuỷ`: mark on the nucleus letter)"}


def load_attested(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def _split_totals(counts: dict) -> dict[str, int]:
    out = {"dev": 0, "test": 0}
    for k, v in counts.items():
        for split in out:
            if k.endswith("-" + split):
                out[split] += int(v)
    return out


def _vi(s: str) -> str:
    return f"*{s}*"


def facts(manifest: dict, attested: list[dict], manifest_path: str) -> str:
    args = manifest.get("generator_args") or {}
    release = args.get("out") or str(Path(manifest_path).parent)
    built = str(manifest.get("started_utc", ""))[:10] or "unknown date"
    commit = str(manifest.get("git_commit", "unknown"))[:7]
    dirty = " (dirty tree)" if manifest.get("git_dirty") else ""
    splits = _split_totals(manifest.get("counts", {}))
    core = sum(int(v) for v in (manifest.get("core_counts") or {}).values())
    vulgar = manifest.get("vulgar_counts") or {}
    n_vulgar = sum(int(v) for v in vulgar.values())
    style = str(manifest.get("placement_style", "unknown"))
    release_line = (
        f"- **Release:** `{release}`, built {built}, generator commit `{commit}`{dirty}; the build seed is withheld "
        f"(one seeded stream builds dev and test, DESIGN_DECISIONS 4.6); "
        f"content SHA-256 `{manifest.get('content_sha256', 'n/a')}` (the item files without the canary fields). "
        f"{int(manifest.get('n_items', 0)):,} generated items from {int(manifest.get('n_base_pairs', 0)):,} base pairs: "
        f"dev {splits['dev']:,}, test {splits['test']:,}, of which core {core:,}. "
        f"*qu-* inputs excluded: {'yes' if manifest.get('exclude_qu') else 'no'}."
    )
    placement_line = f"- **Tone-mark placement as stored:** {PLACEMENT_NAME.get(style, style)} (`placement_style: {style}` in the manifest)."
    vulgar_cells = ", ".join(f"{k} {v}" for k, v in sorted(vulgar.items())) or "none"
    vulgar_line = (f"- **Vulgar-flagged generated items:** {n_vulgar} in {len(vulgar)} of 12 cells ({vulgar_cells}); "
                   "flagged items stay in the gated test file only.")
    lines = [BEGIN, release_line, placement_line, vulgar_line]
    n = len(attested)
    exact = [r for r in attested if r.get("exact")]
    mism = [r for r in attested if not r.get("rule_matches_attested")]
    three = [r for r in attested if int(r.get("n_syllables", 2)) == 3]
    vulg = [r for r in attested if r.get("vulgar")]
    low = [r for r in attested if str(r.get("confidence", "")).lower() == "low"]
    def _row(r: dict) -> str:
        return (_vi(r["input"]) + " → folk " + _vi(r["attested_output"]) + ", rule " + _vi(r.get("rule_output") or "—")
                + " (" + str(r.get("exactness", "")) + ")")

    mism_txt = "; ".join(_row(r) for r in mism) or "none"
    three_txt = ", ".join(_vi(r["input"]) for r in three) or "none"
    vulg_txt = ", ".join(_vi(r["input"]) + " → " + _vi(r["attested_output"]) for r in vulg) or "none"
    low_txt = ", ".join(_vi(r["input"]) for r in low) or "none"
    lines.append(f"- **Attested seed ({n} rows, `data/attested_seed.tsv` → `attested.jsonl`):** {len(exact)} exact reproductions by the engine; "
                 f"{len(mism)} rule-vs-folk mismatches, tagged by kind: {mism_txt}. "
                 f"{len(three)} three-syllable rows ({three_txt}). "
                 f"{len(vulg)} flagged vulgar ({vulg_txt}; gated split only). "
                 f"{len(low)} graded low confidence ({low_txt}).")
    lines.append(END)
    return "\n".join(lines)


def attested_macros(attested: list[dict], source: str) -> str:
    """LaTeX macros for the attested-seed counts the paper's prose quotes (paper/sec_benchmark.tex):
    rows, exact reproductions, approximate rows, H6-eligible rows and native-verified rows. Generated,
    never typed, so that Table 1 and the prose cannot disagree."""
    n = len(attested)
    exact = sum(1 for r in attested if r.get("exact"))
    approx = sum(1 for r in attested if not r.get("rule_matches_attested"))
    h6 = sum(1 for r in attested if r.get("eligible_h6"))
    verified = sum(1 for r in attested if str(r.get("verified_by") or "").strip())
    lines = ["% GENERATED by paper/gen_data_statement_facts.py -- do not edit by hand.",
             f"% source: {source}",
             f"\\newcommand{{\\nAttested}}{{{n}}}",
             f"\\newcommand{{\\nAttestedExact}}{{{exact}}}",
             f"\\newcommand{{\\nAttestedApprox}}{{{approx}}}",
             f"\\newcommand{{\\nAttestedHsix}}{{{h6}}}",
             f"\\newcommand{{\\nAttestedVerified}}{{{verified}}}"]
    return "\n".join(lines) + "\n"


def _strip_generated(tex: str) -> str:
    return "\n".join(ln for ln in tex.splitlines() if not ln.startswith("%"))


def splice(doc: str, block: str) -> str:
    i = doc.find(BEGIN)
    j = doc.find(END)
    if i < 0 or j < 0 or j < i:
        raise SystemExit("docs/DATA_STATEMENT.md lacks the BEGIN/END GENERATED release facts markers")
    return doc[:i] + block + doc[j + len(END):]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "release" / "v0.2" / "manifest.json")
    ap.add_argument("--attested", type=Path, default=None, help="release attested.jsonl (default: next to the manifest)")
    ap.add_argument("--doc", type=Path, default=ROOT / "docs" / "DATA_STATEMENT.md")
    ap.add_argument("--tex", type=Path, default=ROOT / "paper" / "tables" / "attested_facts.tex",
                    help="LaTeX macros for the attested-seed counts quoted in the paper's prose")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the committed block or the macros differ")
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    attested_path = args.attested or args.manifest.parent / "attested.jsonl"
    attested = load_attested(attested_path) if attested_path.exists() else []
    block = facts(manifest, attested, str(args.manifest))
    assert str(manifest.get("seed", "")) == "" or str(manifest.get("seed")) not in block, "the build seed must not be rendered"
    doc = args.doc.read_text(encoding="utf-8")
    new = splice(doc, block)
    tex = attested_macros(attested, str(attested_path.relative_to(ROOT)) if attested_path.is_relative_to(ROOT) else str(attested_path))
    old_tex = args.tex.read_text(encoding="utf-8") if args.tex.exists() else None
    if args.check:
        stale = new != doc or _strip_generated(old_tex or "") != _strip_generated(tex)
        print(json.dumps({"stale": stale, "doc": str(args.doc), "tex": str(args.tex)}))
        return 1 if stale else 0
    args.doc.write_text(new, encoding="utf-8")
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(tex, encoding="utf-8")
    print(json.dumps({"doc": str(args.doc), "tex": str(args.tex), "manifest": str(args.manifest), "n_attested": len(attested),
                      "changed": new != doc or old_tex != tex}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
