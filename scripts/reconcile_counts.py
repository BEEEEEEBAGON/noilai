#!/usr/bin/env python
"""One source for every count quoted about the resources (red-team item 60).

    python scripts/reconcile_counts.py --out data/audit/counts.json

    python scripts/reconcile_counts.py --out data/audit/counts.json --release data/release/v0.3

Reports, from the files in data/external and data/audit: Hunspell entries / lowercase /
parsable / rejected / standard; inventory structures and extension; distinct rimes (parser
rimes and the 162 orthographic spellings, `rimes_orthographic`, design 2.3); onset (canonical,
re-spelled, and WRITTEN `onset_written`, design 2.1), tone and coda distributions and the
coda x tone cross-tab `coda_tone` (design 2.3 C11); the UTF-8 `byte_length` of every toned
vowel letter by tone (design 2.4); the 69 placement-differing syllables; word-list two-syllable
entries and distinct canonical pairs (design 4.1); per tokenizer the audit summary, the
three-valued census verdicts (`verdict_nfd`, `verdict_pc`, design 6.2) and the single-token
share by tone from the audit rows (`single_token_by_tone`, design 9.1); and, when `--release`
names a built release directory, a `release_strata` block (per task x variant degenerate
counts, `output_lexical` share, C2-affected test items; design 3.4 / 4.5); and the
`reconciliations` block of design section 0: the two discrepancies the design document lists
(Gemma 3 NFD tokens per syllable 2.83 vs 2.90; XCOPA-vi syllable tokens vs word tokens), each
re-read from its source file with both quoted figures, their origin and the resolution. Every
number in docs/DESIGN_DECISIONS.md sections 0, 2, 4, 9 and docs/RESULTS_LOG.md is to be re-read
from this file, never retyped.

Without `--release` (the item files are private and absent from most checkouts) a computed
`release_strata` block already in the output file is carried over, marked `carried_over: true`
with the date of the file it came from, so a re-run elsewhere never erases the author's counts.

    python scripts/reconcile_counts.py --out data/audit/counts.json --log-entry -        # print the RESULTS_LOG entry
    python scripts/reconcile_counts.py --out data/audit/counts.json --append-log         # append it to docs/RESULTS_LOG.md

The entry (`--log-entry PATH`, `-` for stdout; `--append-log` appends to docs/RESULTS_LOG.md under
the next free id of the day, RL-<date>-NN) is generated from the counts, so no number in it is typed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent

from noilai.vi import lexicon as L
from noilai.vi import unicode as U
from noilai.vi.syllable import Inventory, try_parse

TASKS = ("T1", "T2", "T3")
VARIANTS = ("V1", "V2", "V3", "V4")


def orthographic_rime(p) -> str:
    """The toneless rime as WRITTEN after the onset (design 2.3): the `u` of `qu` counts as onset
    (`quốc` -> `ôc`, `quýnh` -> `ynh`) and the contracted `i` of `gi` is restored (`gìn` -> `in`,
    `giếng` -> `iêng`); 162 spellings over the 6,595 parsable entries."""
    low = p.surface.lower()
    if p.onset_spelling == "qu":
        return U.strip_tones(low[2:])
    rime = U.strip_tones(low[len(p.onset_spelling):])
    if p.onset_spelling == "gi" and (not rime or rime[0] not in "aeiouyăâêôơư" or rime.startswith("ê")):
        rime = "i" + rime
    return rime


def byte_length_table() -> dict:
    """Design 2.4: UTF-8 byte length of every toned vowel letter (NFC), by tone; the hỏi/nặng
    letters are 3 bytes on every vowel, the others 2 or 3 depending on the vowel block."""
    table = {}
    for tone, name in enumerate(U.TONE_NAMES):
        row = {letter: len(U.compose_letter(*U.decompose_letter(letter)[:2], tone).encode("utf-8")) for letter in U.VOWELS_NFC}
        table[name] = {"letters": row, "two_byte": [k for k, v in row.items() if v == 2], "three_byte": [k for k, v in row.items() if v == 3],
                       "one_byte": [k for k, v in row.items() if v == 1]}
    return table


def single_token_by_tone(rows_csv: Path) -> dict:
    """Design 9.1 / RESULTS_LOG RL-2026-09-30-06: share of single-token syllables by parsed tone,
    per encoding, over the audit rows whose syllable parses."""
    out: dict = {}
    by = defaultdict(list)
    with open(rows_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = try_parse(r["syllable"], strict=False)
            if p is None:
                continue
            by[(r["encoding"], U.TONE_NAMES[p.syllable.tone])].append(r["single_token"] in ("True", "true", "1"))
    for (enc, tone), vals in sorted(by.items()):
        out.setdefault(enc, {})[tone] = {"n": len(vals), "single_token_frac": sum(vals) / len(vals)}
    return out


def release_strata(release: Path | None) -> dict:
    """Design 3.4 / 4.5 / RISKS: per task x variant cell over the released dev + test items, the
    degenerate count (any of `same_tone`, `same_onset`, `same_rime`, or the `degenerate` covariate
    when the build carries it), the `output_lexical` share, and the C2-affected count on the test
    split; absent directory -> a note, never a guess."""
    if release is None:
        return {"status": "not_computed", "note": "pass --release <dir> to count the release strata"}
    release = Path(release)
    files = [release / f"noilai_{split}.jsonl" for split in ("dev", "test")]
    if not release.is_dir() or not all(f.exists() for f in files):
        return {"status": "not_computed", "note": f"release directory {release} or its dev/test files are absent"}
    from noilai.gen.generate import load_items

    items = [it for f in files for it in load_items(f)]
    cells: dict = {}
    for task in TASKS:
        for var in VARIANTS:
            sub = [it for it in items if it["task"] == task and it["variant"] == var]
            if not sub:
                continue
            st = [it["strata"] for it in sub]
            degenerate = [x.get("degenerate", bool(x.get("same_tone") or x.get("same_onset") or x.get("same_rime"))) for x in st]
            lexical = [x["output_lexical"] for x in st if "output_lexical" in x]
            cells[f"{task}-{var}"] = {
                "n_items": len(sub), "n_test": sum(1 for it in sub if it["split"] == "test"),
                "degenerate": sum(degenerate), "degenerate_share": sum(degenerate) / len(sub),
                "same_tone": sum(bool(x.get("same_tone")) for x in st), "same_onset": sum(bool(x.get("same_onset")) for x in st),
                "same_rime": sum(bool(x.get("same_rime")) for x in st),
                "output_lexical": sum(lexical), "output_lexical_share": (sum(lexical) / len(lexical)) if lexical else None,
                "c2_affected": sum(bool(x.get("c2_affected")) for x in st),
                "c2_affected_test": sum(1 for it in sub if it["split"] == "test" and it["strata"].get("c2_affected")),
            }
    manifest = {}
    mp = release / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        manifest = {k: m.get(k) for k in ("content_sha256", "git_commit", "n_items", "degenerate_counts", "c2_affected_counts") if k in m}
    try:
        rel = str(release.resolve().relative_to(ROOT))
    except ValueError:
        rel = str(release)
    return {"status": "computed", "release": rel, "manifest": manifest, "n_items": len(items),
            "degenerate_rule": "strata.degenerate when present, else same_tone or same_onset or same_rime (design 3.4)",
            "c2_affected_test_total": sum(1 for it in items if it["split"] == "test" and it["strata"].get("c2_affected")),
            "output_lexical_share_total": sum(1 for it in items if it["strata"].get("output_lexical")) / len(items),
            "cells": cells}


def _placement_counter():
    """scripts/count_placement.py as a module (the same WORD regex and parse rule as the committed placement counts)."""
    spec = importlib.util.spec_from_file_location("count_placement", SCRIPTS / "count_placement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recount_xcopa_tokens(path: Path) -> dict | None:
    """Live recount of one XCOPA-vi file with the current parser and count_placement's definitions: regex word
    tokens and the word tokens that parse as a Vietnamese syllable (non-strict); None when the file is absent."""
    if not path.exists():
        return None
    cp = _placement_counter()
    n_words = n_syl = 0
    n_items = 0
    for _, text in cp.texts_from(path):
        n_items = max(n_items, _)
        for tok in cp.WORD.findall(U.nfc(text)):
            n_words += 1
            if try_parse(tok.lower(), strict=False) is not None:
                n_syl += 1
    return {"n_items": n_items, "n_word_tokens": n_words, "n_syllable_tokens": n_syl}


def reconciliations(audit: dict, audit_dir: Path = ROOT / "data" / "audit", external: Path = L.EXTERNAL) -> dict:
    """Design section 0, 'still to reconcile by the same script': each discrepancy re-read from its source with the
    two quoted figures, where each came from, and the resolution (a sentence written from the values, never typed)."""
    out: dict = {}
    # (a) Gemma 3 NFD tokens per syllable: 2.83 (README, models.yaml note, RESULTS_LOG corrected row) vs 2.90 (the first
    #     draft of the RESULTS_LOG Gemma 3 row, a hand copy of Gemma 2's value)
    g3 = ((audit.get("gemma3") or {}).get("summary") or {}).get("nfd", {}).get("tokens_per_syllable_mean")
    g2 = ((audit.get("gemma2") or {}).get("summary") or {}).get("nfd", {}).get("tokens_per_syllable_mean")
    rec = {"quoted": {"2.83": ["README.md", "configs/models.yaml (gemma3 note)", "docs/RESULTS_LOG.md RL-2026-09-30-01 (corrected Gemma 3 NFD row)",
                               "docs/DESIGN_DECISIONS.md 1 (H3 row)"],
                      "2.90": ["docs/RESULTS_LOG.md RL-2026-09-30-01 (Gemma 2 NFD row; also the first draft of the Gemma 3 row)"]},
           "source": {"gemma3_nfd_tokens_per_syllable_mean": g3, "gemma2_nfd_tokens_per_syllable_mean": g2,
                      "files": ["data/audit/gemma3.json", "data/audit/gemma2.json"], "field": "summary.nfd.tokens_per_syllable_mean"}}
    if g3 is not None and g2 is not None:
        rec["gemma3_rounded"], rec["gemma2_rounded"] = round(g3, 2), round(g2, 2)
        rec["consistent"] = rec["gemma3_rounded"] == 2.83 and rec["gemma2_rounded"] == 2.90
        rec["resolution"] = (f"{rec['gemma3_rounded']:.2f} is Gemma 3 ({g3:.4f}) and {rec['gemma2_rounded']:.2f} is Gemma 2 ({g2:.4f}): the 2.90 once "
                             "printed in the Gemma 3 row was Gemma 2's value copied by hand (RL-2026-09-30-01, corrected the same day); "
                             "every document quotes the gemma3 value of this file.")
    else:
        rec["consistent"] = None
        rec["resolution"] = "data/audit/gemma3.json or gemma2.json absent: not reconciled here"
    out["gemma3_nfd_tokens_per_syllable"] = rec
    # (b) XCOPA-vi syllable tokens 9,628 vs 10,137: two definitions (word tokens parsing as syllables vs regex word tokens)
    #     and two parser versions (the 9,628 predates the 30 September parser corrections); the C2-affected counts are the same
    test_json = audit_dir / "placement_xcopa_test.json"
    both_json = audit_dir / "placement_xcopa.json"
    committed = json.loads(test_json.read_text(encoding="utf-8")) if test_json.exists() else None
    both = json.loads(both_json.read_text(encoding="utf-8")) if both_json.exists() else None
    live = recount_xcopa_tokens(external / "xcopa_test_vi.jsonl")
    xc = {"quoted": {"9,628": ["docs/DESIGN_DECISIONS.md 0 ('XCOPA-vi syllable tokens 9,628 vs 10,137')"],
                     "10,137": ["docs/DESIGN_DECISIONS.md 0", "data/audit/placement_xcopa_test.json n_word_tokens"]},
          "definitions": {"n_word_tokens": "regex word tokens over the NFC text (scripts/count_placement.py WORD)",
                          "n_syllable_tokens": "word tokens that parse as a Vietnamese syllable with the current parser (non-strict)"},
          "committed": {k: committed.get(k) for k in ("n_items", "n_texts", "n_word_tokens", "n_syllable_tokens", "affected_tokens", "affected_items")} if committed else None,
          "committed_both_files": {k: both.get(k) for k in ("n_items", "n_word_tokens", "n_syllable_tokens", "affected_tokens", "affected_items")} if both else None,
          "live_recount": live, "source_files": ["data/audit/placement_xcopa_test.json", "data/external/xcopa_test_vi.jsonl"]}
    if committed and live:
        same_words = committed.get("n_word_tokens") == live["n_word_tokens"]
        same_syl = committed.get("n_syllable_tokens") == live["n_syllable_tokens"]
        xc["consistent"] = bool(same_words and same_syl)
        xc["resolution"] = (f"The two figures are two definitions over the 500-item test file: {live['n_word_tokens']:,} regex word tokens, "
                            f"of which {live['n_syllable_tokens']:,} parse as Vietnamese syllables with the current parser (committed count "
                            f"{committed.get('n_syllable_tokens'):,}; the 9,628 of the design document predates the 30 September parser "
                            f"corrections and is superseded). The C2-affected count is unchanged: {committed.get('affected_tokens')} tokens in "
                            f"{committed.get('affected_items')} items. The paper quotes word tokens with the definition.")
        xc["superseded_figure_reproduced"] = live["n_syllable_tokens"] == 9628
    else:
        xc["consistent"] = None
        xc["resolution"] = "placement_xcopa_test.json or data/external/xcopa_test_vi.jsonl absent: not reconciled here"
    out["xcopa_syllable_tokens"] = xc
    return out


def log_entry(counts: dict, date: str, entry_id: str, out_path: Path) -> str:
    """The docs/RESULTS_LOG.md entry for this run of the script, written from `counts` (no number typed)."""
    rc = counts.get("reconciliations") or {}
    g = rc.get("gemma3_nfd_tokens_per_syllable") or {}
    x = rc.get("xcopa_syllable_tokens") or {}
    try:
        rel = out_path.resolve().relative_to(ROOT)
    except ValueError:
        rel = out_path
    source = (f"Source: `{rel}` `reconciliations` (`scripts/reconcile_counts.py`), read from `data/audit/gemma3.json`, "
              "`data/audit/gemma2.json`, `data/audit/placement_xcopa_test.json` and a live recount of `data/external/xcopa_test_vi.jsonl` "
              "with the current parser.")
    lines = [f"## {date} — Reconciliations of the section-0 discrepancies, re-read by the counts script (no model outputs) [{entry_id}]", "", source, ""]
    if g.get("resolution"):
        lines.append(f"- **Gemma 3 NFD tokens per syllable (2.83 vs 2.90).** {g['resolution']} (`consistent`: {g.get('consistent')}.)")
    if x.get("resolution"):
        lines.append(f"- **XCOPA-vi syllable tokens (9,628 vs 10,137).** {x['resolution']} (`consistent` with the committed count: {x.get('consistent')}.)")
    rs = counts.get("release_strata") or {}
    if rs.get("carried_over"):
        lines.append(f"- `release_strata` carried over from {rs.get('carried_over_from')}: the release item files are not on this machine.")
    h = counts.get("hunspell") or {}
    inventory = (f"Inventory used: `vi-DauMoi.dic` {h.get('entries_new_file'):,} entries, {h.get('lowercase_letter_entries'):,} lower-case, "
                 f"{h.get('parsable'):,} parsable, {h.get('n_rimes_orthographic')} orthographic rimes (base inventory for the counts; the extended "
                 f"Hunspell-plus-Viet74K inventory has {(counts.get('inventory') or {}).get('extended_structures'):,} structures).")
    lines += ["", inventory, ""]
    return "\n".join(lines)


def next_entry_id(log_text: str, date: str) -> str:
    used = {int(m.group(1)) for m in re.finditer(rf"\[RL-{re.escape(date)}-(\d+)\]", log_text)}
    n = 1
    while n in used:
        n += 1
    return f"RL-{date}-{n:02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "audit" / "counts.json")
    ap.add_argument("--release", type=Path, default=None,
                    help="a built release directory (dev/test files + manifest) for the release_strata block; absent -> the previous "
                         "computed block of --out is carried over (marked), else noted")
    ap.add_argument("--log-entry", default=None, help="write the docs/RESULTS_LOG.md entry for this run to PATH ('-' = stdout)")
    ap.add_argument("--append-log", action="store_true", help="append the entry to docs/RESULTS_LOG.md under the next free id of the day")
    ap.add_argument("--date", default=None, help="the entry's date, YYYY-MM-DD (default today, UTC)")
    args = ap.parse_args(argv)
    raw_new = (L.EXTERNAL / "vi-DauMoi.dic").read_text(encoding="utf-8").splitlines()[1:]
    raw_old = (L.EXTERNAL / "vi-DauCu.dic").read_text(encoding="utf-8").splitlines()[1:]
    new = L.load_hunspell_syllables("new")
    old = L.load_hunspell_syllables("old")
    base = Inventory(new + old)
    ext = L.load_inventory()
    parsed = [try_parse(w, strict=False) for w in new]
    n_parsable = sum(1 for p in parsed if p)
    standard = [w for w, p in zip(new, parsed) if p and p.i_y_variant is None]
    onsets = Counter(p.syllable.onset for p in parsed if p)
    tones = Counter(U.TONE_NAMES[p.syllable.tone] for p in parsed if p)
    codas = Counter(p.syllable.coda or "open" for p in parsed if p)
    differing = sorted(set(new) - set(old))
    from noilai.vi.syllable import _onset_spelling
    onset_spellings = Counter(_onset_spelling(p.syllable) for p in parsed if p)
    onset_written = Counter(p.onset_spelling for p in parsed if p)      # as WRITTEN in the dictionary (design 2.1 table)
    coda_tone = {}
    for p in parsed:
        if p:
            coda_tone.setdefault(p.syllable.coda or "open", dict.fromkeys(U.TONE_NAMES, 0))[U.TONE_NAMES[p.syllable.tone]] += 1
    rimes_ortho = Counter(orthographic_rime(p) for p in parsed if p)
    audit = {}
    for name in ("gemma3", "gemma2"):
        p = ROOT / "data" / "audit" / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            census = d["normalization_census"]
            rows_csv = ROOT / "data" / "audit" / f"{name}_rows.csv"
            audit[name] = {"summary": d["summary"], "normalizes_nfd": census.get("normalizes_nfd"),
                           "verdict_nfd": census.get("verdict_nfd"), "verdict_pc": census.get("verdict_pc"),
                           "census_n_samples": census.get("n_samples"), "normalizer": census.get("normalizer"),
                           "single_token_by_tone": single_token_by_tone(rows_csv) if rows_csv.exists() else None}
    idx = L.wordlist_index()
    pairs = idx["two_syllable_pairs"]
    counts = {
        "hunspell": {
            "entries_new_file": len(raw_new), "entries_old_file": len(raw_old),
            "lowercase_letter_entries": len(new), "parsable": n_parsable, "unparsed": sorted(set(base.unparsed)),
            "rejected_phonotactics": sorted(set(base.rejected)), "standard_spelling_parsable": len(standard),
            "iy_variants": sorted(w for w, p in zip(new, parsed) if p and p.i_y_variant in ("y", "i")),
            "nonstandard": sorted(w for w, p in zip(new, parsed) if p and p.i_y_variant == "nonstandard"),
            "placement_differing_syllables": len(differing), "placement_differing_list": differing,
            "onset_written": dict(sorted(onset_written.items())),
            "coda_tone": dict(sorted(coda_tone.items())),
            "coda_tone_columns": list(U.TONE_NAMES),
            "n_rimes_orthographic": len(rimes_ortho), "rimes_orthographic": sorted(rimes_ortho),
        },
        "byte_length": byte_length_table(),
        "inventory": {
            "base_structures": len(base.structures), "base_toneless": len(base.toneless), "base_rimes": len(base.rimes),
            "extended_structures": len(ext.structures), "extension_added": ext.n_extended, "extended_rimes": len(ext.rimes),
            "rimes": sorted(ext.rimes),
        },
        "distributions_new_file": {
            "onset_canonical": dict(onsets.most_common()), "onset_spelling": dict(onset_spellings.most_common()),
            "tone": dict(tones.most_common()), "coda": dict(codas.most_common()),
        },
        "wordlist": {"entries": sum(1 for _ in L.load_words()), "two_syllable_pairs": len(pairs),
                     "two_syllable_entries": len(pairs),                                  # design 4.1: entries of two syllables
                     "distinct_canonical_pairs": len({tuple(p) for p in pairs}),          # design 4.1: the lexical base-pair pool
                     "distinct_canonical_syllables": len(idx["canonical_counts"]),
                     "iy_table_y_forms": sorted(k for k, v in L.iy_table().items() if v == "y")},
        "tokenizer_audit": audit,
        "release_strata": release_strata(args.release),
        "reconciliations": reconciliations(audit),
        "resource_sha256": json.loads((ROOT / "data" / "HASHES.json").read_text()) if (ROOT / "data" / "HASHES.json").exists() else {},
    }
    if args.release is None and args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8")).get("release_strata") or {}
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("status") == "computed":                  # the author's counts over the private item files: never erased
            counts["release_strata"] = {**prev, "carried_over": True,
                                        "carried_over_from": prev.get("carried_over_from") or
                                        f"the previous {args.out.name} (computed over {prev.get('release', '?')} on the author's machine)"}
    counts["generated"] = {"script": "scripts/reconcile_counts.py", "date_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
    date = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if args.log_entry or args.append_log:
        log_path = ROOT / "docs" / "RESULTS_LOG.md"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        entry = log_entry(counts, date, next_entry_id(log_text, date), args.out)
        if args.log_entry:
            if args.log_entry == "-":
                print(entry)
            else:
                Path(args.log_entry).write_text(entry + "\n", encoding="utf-8")
        if args.append_log:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(("\n" if log_text.endswith("\n") else "\n\n") + entry + "\n")
            print(f"appended {entry.split('[')[-1].rstrip(']')} to {log_path}", file=sys.stderr)
    brief = {"hunspell_lowercase": len(new), "parsable": n_parsable, "standard": len(standard), "rejected": len(set(base.rejected)),
             "unparsed": len(set(base.unparsed)), "base_structures": len(base.structures), "extended": len(ext.structures),
             "extension_added": ext.n_extended, "rimes": len(ext.rimes), "placement_differing": len(differing),
             "two_syllable_pairs": len(pairs), "distinct_canonical_pairs": counts["wordlist"]["distinct_canonical_pairs"],
             "rimes_orthographic": len(rimes_ortho), "release_strata": counts["release_strata"]["status"],
             "release_strata_carried_over": bool(counts["release_strata"].get("carried_over")),
             "reconciliations_consistent": {k: v.get("consistent") for k, v in counts["reconciliations"].items()},
             "gemma3_verdict_nfd": audit.get("gemma3", {}).get("verdict_nfd"),
             "gemma3_nfc_tps": audit.get("gemma3", {}).get("summary", {}).get("nfc", {}).get("tokens_per_syllable_mean"),
             "gemma3_nfd_tps": audit.get("gemma3", {}).get("summary", {}).get("nfd", {}).get("tokens_per_syllable_mean")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
