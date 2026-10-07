#!/usr/bin/env python
"""Release identity of the frozen data, generated as LaTeX macros and a hash table.

    python paper/gen_release_facts.py --manifest data/release/v0.3/manifest.json \
        --run-plan configs/run_plan.yaml --out paper/tables/release_facts.tex
    python paper/gen_release_facts.py ... --check      # exit 1 if the committed files are stale

The paper's "freeze and hashes" sentences (Section 3, the data-statement appendix) quote the
identity of the frozen release: the content SHA-256 of the item files (computed over the items
with the canary fields removed; the dataset's identity, DESIGN_DECISIONS 4.6), the SHA-256 of the
canary GUID (the paper prints only the digest, item 44), the SHA-256 of every seeded item file the
runs read (recorded in configs/run_plan.yaml at the data freeze and verified before every run),
the frozen attested-string list, the resource hashes, and the counts the manifest carries. None
of these may be typed into a .tex file: this script writes

    paper/tables/release_facts.tex   -- \\newcommand macros (\\releaseContentSha, \\releaseNItems, ...)
    paper/tables/release_hashes.tex  -- the appendix table of artefacts and their SHA-256s
    paper/tables/release_facts.json  -- the values, for the tests and the claim ledger

from the public manifest (never the private one: no seed, no GUID) and the run plan. The build
seed is withheld (DESIGN_DECISIONS 4.6) and this script asserts that it does not render it. The
generator commit printed is the one the manifest records (the original repository's hash, see
docs/MIGRATION.md); the pre-registration commit is a \\placeholder in the paper until it exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (macro name, manifest key, how to render)
COUNT_MACROS = (
    ("releaseNItems", "n_items"),
    ("releaseNBasePairs", "n_base_pairs"),
    ("releaseNLexicalBasePairs", "n_lexical_base_pairs"),
    ("releaseNLexicalPairsAvailable", "n_lexical_pairs_available"),
    ("releaseNAttestedSyllables", "n_attested_syllables"),
    ("releaseNMarginalRimes", "n_marginal_rimes"),
    ("releaseNReservedSyllables", "n_reserved_syllables"),
    ("releaseNReservedPairs", "n_reserved_pairs"),
    ("releaseNDevPairsMoved", "dev_base_pairs_moved_for_vulgar"),
)

RUN_PLAN_FILES = ("noilai_dev", "noilai_test", "noilai_core", "noilai_main", "noilai_c2")
RUN_PLAN_LABELS = {
    "noilai_dev": "public development split",
    "noilai_test": "gated test split (header record and canary on every item)",
    "noilai_core": "core (API-served models)",
    "noilai_main": "open-model main sample",
    "noilai_c2": "placement-enriched set (C2)",
}
RESOURCE_LABELS = {
    "vi-DauMoi.dic": "Hunspell vi\\_VN, new-style placement",
    "vi-DauCu.dic": "Hunspell vi\\_VN, old-style placement",
    "Viet74K.txt": "Viet74K word list",
    "vulgar_lexicon.tsv": "taboo-syllable blocklist",
    "attested_seed.tsv": "attested seed table",
}


def fmt_int(x) -> str:
    return f"{int(x):,}"


def split_totals(counts: dict) -> dict[str, int]:
    out = {"dev": 0, "test": 0}
    for k, v in counts.items():
        for split in out:
            if k.endswith("-" + split):
                out[split] += int(v)
    return out


def task_totals(counts: dict) -> dict[str, int]:
    out = {"T1": 0, "T2": 0, "T3": 0}
    for k, v in counts.items():
        out[k.split("-")[0]] += int(v)
    return out


def values(manifest: dict, plan: dict) -> dict:
    v = {}
    for macro, key in COUNT_MACROS:
        if key in manifest:
            v[macro] = fmt_int(manifest[key])
    splits = split_totals(manifest.get("counts", {}))
    tasks = task_totals(manifest.get("counts", {}))
    v["releaseNDev"] = fmt_int(splits["dev"])
    v["releaseNTest"] = fmt_int(splits["test"])
    v["releaseNCore"] = fmt_int(sum(int(x) for x in (manifest.get("core_counts") or {}).values()))
    v["releaseNTone"] = fmt_int(tasks["T1"])
    v["releaseNTtwo"] = fmt_int(tasks["T2"])
    v["releaseNTthree"] = fmt_int(tasks["T3"])
    v["releaseNVulgar"] = fmt_int(sum(int(x) for x in (manifest.get("vulgar_counts") or {}).values()))
    v["releaseNCtwoAffected"] = fmt_int(sum(int(x) for x in (manifest.get("c2_affected_counts") or {}).values()))
    drops = manifest.get("drops") or {}
    v["releaseDropIdentity"] = fmt_int(sum(n for k, n in drops.items() if k.endswith(":identity")))
    v["releaseDropReversal"] = fmt_int(sum(n for k, n in drops.items() if k.endswith(":plain_reversal")))
    v["releaseDropIllegal"] = fmt_int(sum(n for k, n in drops.items() if k.endswith(":illegal")))
    v["releaseDropVulgarInput"] = fmt_int(sum(n for k, n in drops.items() if k.startswith("base:vulgar_input")))
    samples = manifest.get("samples") or {}
    if "main" in samples:
        v["releaseNMain"] = fmt_int(samples["main"]["n_items"])
        v["releaseMainPerCell"] = fmt_int(samples["main"]["per_cell"])
        v["releaseMainSeed"] = str(samples["main"]["seed"])          # a public sampling seed, not the build seed
    if "c2" in samples:
        v["releaseNCtwo"] = fmt_int(samples["c2"]["n_items"])
        v["releaseCtwoSeed"] = str(samples["c2"]["seed"])
    gargs = manifest.get("generator_args") or {}
    if "n_pseudo" in gargs:
        v["releaseNPseudoBasePairs"] = fmt_int(gargs["n_pseudo"])
    if "n_lexicon" in gargs:
        v["releaseNLexicalDrawn"] = fmt_int(gargs["n_lexicon"])
    if "dev_frac" in gargs:
        v["releaseDevPercent"] = f"{round(100 * float(gargs['dev_frac']))}"
    v["releaseVersion"] = Path(str((manifest.get("generator_args") or {}).get("out", "data/release/unknown"))).name
    v["releaseBuiltOn"] = str(manifest.get("started_utc", ""))[:10]
    v["releaseGeneratorCommit"] = str(manifest.get("git_commit", "unknown"))[:7]
    v["releaseGeneratorCommitFull"] = str(manifest.get("git_commit", "unknown"))
    v["releaseGitDirty"] = "dirty" if manifest.get("git_dirty") else "clean"
    v["releasePlacementStyle"] = str(manifest.get("placement_style", "unknown"))
    v["releaseContentSha"] = str(manifest.get("content_sha256", ""))
    v["releaseContentShaShort"] = v["releaseContentSha"][:12]
    v["releaseCanarySha"] = str(manifest.get("canary_sha256", ""))
    v["releaseAttestedStringsSha"] = str(manifest.get("attested_strings_sha256", ""))
    for key in RUN_PLAN_FILES:
        entry = (plan.get("item_files") or {}).get(key) or {}
        v["sha_" + key] = str(entry.get("sha256", ""))
    for name, sha in (manifest.get("resource_sha256") or {}).items():
        v["sha_resource_" + name] = str(sha)
    return v


def check_no_seed(v: dict, manifest: dict) -> None:
    # the public manifest carries no seed; refuse to render one if a private manifest were passed by mistake
    if "seed" in manifest or "seed" in (manifest.get("generator_args") or {}) or "canary" in manifest:
        sys.exit("refusing to render a manifest that carries the build seed or the canary GUID (DESIGN_DECISIONS 4.6)")


def macros_tex(v: dict, src: str, stamp: str) -> str:
    lines = [f"% GENERATED by paper/gen_release_facts.py from {src} on {stamp}; do not edit by hand.",
             "% Release identity and counts quoted in the prose (DESIGN_DECISIONS 4.6, 12.43): never typed.",
             "% The build seed is withheld and is not among these macros; the sampling seeds below are public."]
    for name in sorted(k for k in v if not k.startswith("sha_")):
        lines.append(f"\\newcommand{{\\{name}}}{{{v[name]}}}")
    return "\n".join(lines) + "\n"


def hashes_tex(v: dict, src: str, stamp: str) -> str:
    rows = []
    rows.append(("Release content (item files, canary fields removed)", v["releaseContentSha"]))
    rows.append(("Canary GUID (digest only; the GUID lives in the gated release)", v["releaseCanarySha"]))
    rows.append(("Attested input/output strings, frozen before the build", v["releaseAttestedStringsSha"]))
    for key in RUN_PLAN_FILES:
        if v.get("sha_" + key):
            fname = key.replace("_", "\\_") + ".jsonl"
            rows.append((f"\\texttt{{{fname}}}: {RUN_PLAN_LABELS[key]}", v["sha_" + key]))
    for name, label in RESOURCE_LABELS.items():
        if v.get("sha_resource_" + name):
            rows.append((label, v["sha_resource_" + name]))
    out = [f"% GENERATED by paper/gen_release_facts.py from {src} on {stamp}; do not edit by hand.",
           "\\begin{table*}[t]\\centering\\footnotesize",
           "\\begin{tabular}{@{}p{0.50\\textwidth}p{0.44\\textwidth}@{}}",
           "\\toprule",
           "Artefact & SHA-256 \\\\",
           "\\midrule"]
    for label, sha in rows:
        half = len(sha) // 2                      # a 64-character digest never breaks; print it on two lines
        out.append(f"{label} & \\texttt{{{sha[:half]}}}\\newline\\texttt{{{sha[half:]}}} \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Identity of the frozen release \\releaseVersion{} (generator commit \\releaseGeneratorCommit{} as recorded in its manifest; the build seed is withheld, Section~\\ref{sec:sampling}). The item-file digests are recorded in the run plan at the data freeze and verified before every run; the resource digests pin the look-up lists the generator consulted.}",
            "\\label{tab:hashes}",
            "\\end{table*}"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/release/v0.3/manifest.json")
    ap.add_argument("--run-plan", default="configs/run_plan.yaml")
    ap.add_argument("--out", default="paper/tables/release_facts.tex")
    ap.add_argument("--check", action="store_true", help="compare with the committed files; exit 1 if stale")
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    import yaml  # noqa: PLC0415  (optional dependency, listed in requirements.txt)
    plan = yaml.safe_load(Path(a.run_plan).read_text(encoding="utf-8"))
    check_no_seed({}, manifest)
    v = values(manifest, plan)
    rel = Path(a.manifest)
    try:
        src = str(rel.resolve().relative_to(ROOT))
    except ValueError:
        src = str(rel)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    out = Path(a.out)
    files = {out: macros_tex(v, src, stamp),
             out.with_name("release_hashes.tex"): hashes_tex(v, src, stamp)}
    sidecar = json.dumps(v, ensure_ascii=False, indent=1, sort_keys=True) + "\n"

    def body(s: str) -> list[str]:
        return [ln for ln in s.splitlines() if not ln.startswith("% GENERATED")]

    if a.check:
        stale = []
        for p, text in files.items():
            if not p.exists() or body(p.read_text(encoding="utf-8")) != body(text):
                stale.append(str(p))
        js = out.with_suffix(".json")
        if not js.exists() or json.loads(js.read_text(encoding="utf-8")) != v:
            stale.append(str(js))
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        print("release facts current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    for p, text in files.items():
        p.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(sidecar, encoding="utf-8")
    print(json.dumps({"out": [str(p) for p in files] + [str(out.with_suffix('.json'))], "content_sha256": v["releaseContentSha"],
                      "n_items": v["releaseNItems"], "macros": len([k for k in v if not k.startswith("sha_")])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
