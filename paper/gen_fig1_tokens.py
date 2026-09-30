#!/usr/bin/env python
"""Figure 1: "meo cai -> mai keo" with the REAL Gemma 3 token boundaries.

    python paper/gen_fig1_tokens.py --spm data/external/gemma3_tokenizer.model \
        --out paper/figures/fig1_tokens.tex --json paper/figures/fig1_tokens.json

Tokens come from noilai.audit.tokenizers.SentencePieceAdapter on the Gemma 3 SentencePiece
model file (google/gemma_pytorch; SHA-256 in data/HASHES.json), each syllable tokenized in
running-text position (preceded by a space, the audit's convention), under NFC and NFD.
The .tex file holds only the tabular body; sec_intro.tex wraps it in the figure and writes
the caption. A second file, fig1_tokens_note.tex, defines \\figtokensnote: the two caption
sentences that state what the numbers show (which syllables are cut where under NFC; the
alignment each syllable falls to under NFD and whether its tone mark becomes a token of its
own), composed from the same audit rows, so that the caption cannot contradict the figure.
The .json sidecar records the raw pieces, ids, offsets, aligned seams and the note text so
that the drawn boundaries can be checked against the audit rows.

Rendering: the SentencePiece space marker becomes \\tokspace{} (a visible space); an
isolated combining mark is drawn with the T5 accent macro over an empty group
(\\`{} \\'{} \\~{} \\h{} \\d{} \\^{} \\u{} \\horn{}), so that a token consisting of a bare tone
mark is visible as such. pdflatex cannot read combining code points directly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.audit.tokenizers import SentencePieceAdapter, audit_syllable, linguistic_boundaries  # noqa: E402
from noilai.gen import variants as V  # noqa: E402
from noilai.vi import unicode as U  # noqa: E402
from noilai.vi.syllable import spell, try_parse  # noqa: E402

INPUT = ("mèo", "cái")
VARIANT = "V1"
DEFAULT_SPM = ROOT / "data" / "external" / "gemma3_tokenizer.model"

COMBINING_TEX = {
    "̀": r"\`{}",     # grave  (huyền)
    "́": r"\'{}",     # acute  (sắc)
    "̃": r"\~{}",     # tilde  (ngã)
    "̉": r"\h{}",     # hook above (hỏi)   -- T5 (vntex)
    "̣": r"\d{}",     # dot below (nặng)
    "̂": r"\^{}",     # circumflex
    "̆": r"\u{}",     # breve
    "̛": r"\horn{}",  # horn -- T5 (vntex)
}
SPACE_MARKER = "▁"
SEAM_NAME = {"onset_rime": "onset$|$rime", "glide_nucleus": "glide$|$nucleus", "nucleus_coda": "nucleus$|$coda"}


def piece_to_tex(piece: str) -> str:
    """LaTeX for one SentencePiece piece: visible space marker, drawn combining marks,
    precomposed letters passed through (T5 encoding renders them)."""
    out = []
    for ch in piece:
        if ch == SPACE_MARKER:
            out.append(r"\tokspace{}")
        elif ch in COMBINING_TEX:
            out.append(COMBINING_TEX[ch])
        elif unicodedata.combining(ch):
            out.append(rf"\textsuperscript{{U+{ord(ch):04X}}}")
        elif ch in "\\{}$&#%_^~":
            out.append({"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                        "%": r"\%", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def phrase_tokens(adapter: SentencePieceAdapter, syllables: tuple[str, ...], encoding: str) -> list[dict]:
    rows = []
    for syl in syllables:
        a = audit_syllable(adapter, syl, encoding)
        if a is None:
            raise SystemExit(f"{syl!r} does not parse")
        # ids of the same pieces, for the record
        text = " " + (U.nfd(syl) if encoding == "nfd" else U.nfc(syl))
        ids = [t.id for t in adapter.encode(text) if t.end > 1]
        # which linguistic seams (offsets in the NFC surface) the internal boundaries hit
        ling = linguistic_boundaries(try_parse(syl, strict=False), U.nfc(syl))
        seams = [name for name, off in ling.items() if off is not None and off in a.boundaries]
        rows.append({"syllable": syl, "encoding": encoding, "pieces": a.tokens, "ids": ids, "n_tokens": a.n_tokens,
                     "boundaries": a.boundaries, "boundary_alignment": a.boundary_alignment,
                     "onset_rime_split": a.onset_rime_split, "tone_isolated": a.tone_isolated,
                     "aligned_seams": seams})
    return rows


def _join(parts: list[str]) -> str:
    parts = list(parts)
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _join_clauses(clauses: list[str]) -> str:
    if len(clauses) <= 1:
        return "".join(clauses)
    if len(clauses) == 2:
        return clauses[0] + ", and " + clauses[1]
    return "; ".join(clauses[:-1]) + "; and " + clauses[-1]


def note_sentences(rows: dict[str, dict[str, list[dict]]]) -> str:
    """The caption's two data sentences, composed from the audit rows (never typed).

    Sentence 1 (NFC): which syllables are single tokens and where the split ones are cut.
    Sentence 2 (NFD): the alignment each group of syllables falls to, whether its tone mark
    becomes a token of its own, and which seam the remaining aligned boundary sits on."""
    def vi(s: str) -> str:
        return rf"\vi{{{s}}}"

    nfc = rows["input"]["nfc"] + rows["output"]["nfc"]
    nfd = rows["input"]["nfd"] + rows["output"]["nfd"]
    parts = []
    singles = [r["syllable"] for r in nfc if r["n_tokens"] == 1]
    if singles:
        parts.append(f"{_join([vi(x) for x in singles])} {'is a single token' if len(singles) == 1 else 'are single tokens'}")
    for r in nfc:
        if r["n_tokens"] == 1:
            continue
        where = (f"the {_join([SEAM_NAME[x] for x in r['aligned_seams']])} seam" if r["aligned_seams"]
                 else "a boundary inside a component")
        parts.append(f"{vi(r['syllable'])} is cut at {where} (alignment {r['boundary_alignment']:.2f})")
    s1 = "Under NFC " + _join(parts) + "."
    groups: dict[tuple, list[str]] = {}
    for r in nfd:
        groups.setdefault((r["boundary_alignment"], r["tone_isolated"], tuple(r["aligned_seams"])), []).append(r["syllable"])
    clauses = []
    for (al, iso, seams), names in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        mark = ("whose tone mark becomes a token of its own" if iso
                else "whose tone mark stays attached to the following letter")
        rest = f", leaving one aligned boundary at the {_join([SEAM_NAME[x] for x in seams])} seam" if seams else ""
        clauses.append(f"to {al:.2f} for {_join([vi(x) for x in names])}, {mark}{rest}")
    s2 = ("Under NFD every syllable gains a boundary between a base letter and its combining tone mark, which is never "
          "linguistic, so alignment falls " + _join_clauses(clauses) + ".")
    return s1 + " " + s2


def render_note(note: str, now: str, spm: str) -> str:
    header = f"% GENERATED by paper/gen_fig1_tokens.py from {spm} on {now}; do not edit by hand."
    what = "% \\figtokensnote: the caption sentences of Figure 1 that state what the token table shows,"
    why = "% composed from the audit rows so that the caption cannot contradict the figure."
    return "\n".join([header, what, why, f"\\newcommand{{\\figtokensnote}}{{{note}}}"]) + "\n"


def render(input_syls: tuple[str, ...], output_syls: tuple[str, ...], rows: dict[str, dict[str, list[dict]]],
           info: dict, now: str, spm: str) -> str:
    lines = [f"% GENERATED by paper/gen_fig1_tokens.py from {spm} on {now}; do not edit by hand.",
             f"% tokenizer: {info.get('model_type')} {info.get('vocab_size'):,} pieces, normalizer {info.get('normalizer')}, "
             f"byte_fallback {info.get('byte_fallback')}; syllables tokenized in running-text position (leading space).",
             r"\begin{tabular}{@{}l" + "l" * len(input_syls) + "c" + "l" * len(output_syls) + "@{}}",
             r"\toprule",
             " & ".join([""] + [rf"\vi{{{s}}}" for s in input_syls] + [""] + [rf"\vi{{{s}}}" for s in output_syls]) + r" \\",
             " & ".join(["", rf"\multicolumn{{{len(input_syls)}}}{{l}}{{input (\vi{{{' '.join(input_syls)}}})}}",
                         r"$\xrightarrow{\ \var{1}\ }$",
                         rf"\multicolumn{{{len(output_syls)}}}{{l}}{{output (\vi{{{' '.join(output_syls)}}})}}"]) + r" \\",
             r"\midrule"]
    for enc, label in (("nfc", "NFC (precomposed)"), ("nfd", "NFD (combining)")):
        cells = [label]
        for r in rows["input"][enc]:
            cells.append("".join(rf"\tok{{{piece_to_tex(p)}}}" for p in r["pieces"]))
        cells.append("")
        for r in rows["output"][enc]:
            cells.append("".join(rf"\tok{{{piece_to_tex(p)}}}" for p in r["pieces"]))
        lines.append(" & ".join(cells) + r" \\")
        counts = ["\\quad tokens / alignment"]
        for r in rows["input"][enc]:
            counts.append(f"{r['n_tokens']} / {r['boundary_alignment']:.2f}")
        counts.append("")
        for r in rows["output"][enc]:
            counts.append(f"{r['n_tokens']} / {r['boundary_alignment']:.2f}")
        lines.append(" & ".join(counts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spm", type=Path, default=DEFAULT_SPM)
    ap.add_argument("--name", default="gemma3")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures" / "fig1_tokens.tex")
    ap.add_argument("--json", type=Path, default=None, help="sidecar with pieces, ids and offsets (default: next to --out)")
    ap.add_argument("--note", type=Path, default=None,
                    help="the \\figtokensnote definition (default: fig1_tokens_note.tex next to --out)")
    args = ap.parse_args(argv)
    if not args.spm.exists():
        raise SystemExit(f"tokenizer model not found: {args.spm} (run scripts/fetch_resources.py)")
    adapter = SentencePieceAdapter(args.spm, name=args.name)
    a, b = (try_parse(s).syllable for s in INPUT)
    x, y = V.apply(VARIANT, a, b)
    output = (spell(x), spell(y))
    rows = {"input": {}, "output": {}}
    for enc in ("nfc", "nfd"):
        rows["input"][enc] = phrase_tokens(adapter, INPUT, enc)
        rows["output"][enc] = phrase_tokens(adapter, output, enc)
    info = adapter.normalizer_info()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(INPUT, output, rows, info, now, str(args.spm)), encoding="utf-8")
    note = note_sentences(rows)
    note_path = args.note or args.out.parent / "fig1_tokens_note.tex"
    note_path.write_text(render_note(note, now, str(args.spm)), encoding="utf-8")
    side = args.json or args.out.with_suffix(".json")
    side.write_text(json.dumps({"tokenizer": args.name, "model_file": str(args.spm), "normalizer_info": info,
                                "variant": VARIANT, "input": " ".join(INPUT), "output": " ".join(output),
                                "rows": rows, "note": note, "generated": now}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(json.dumps({"out": str(args.out), "note": str(note_path), "json": str(side), "input": " ".join(INPUT),
                      "output": " ".join(output),
                      "nfc_tokens": [r["pieces"] for r in rows["input"]["nfc"] + rows["output"]["nfc"]]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
