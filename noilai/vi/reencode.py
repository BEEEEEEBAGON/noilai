"""Text-level re-encodings for the orthographic-counterfactual arms.

C1  nfc / nfd             precomposed vs combining diacritics (meaning preserved)
C2  placement old / new   tone-mark placement convention (meaning preserved; only the
                          open rimes oa, oe, uy after a non-q onset differ: hoà/hòa)
C3  strip_tones / strip_all  destroys information (contrast arm)
plus win1258, the partially-composed legacy form, for the encoding census.

All functions operate on arbitrary text: words are located with a regex over
Vietnamese letters, each word is re-spelled if it parses as a syllable, and
everything else (punctuation, digits, Latin words) is passed through unchanged.
Capitalization is preserved letter by letter.
"""
from __future__ import annotations

import re

from . import unicode as U
from .syllable import spell, try_parse

_WORD = re.compile(r"[A-Za-zĐđÀ-ɏḀ-ỿ̀-ͯ]+")

ARMS = ("nfc", "nfd", "win1258", "placement_old", "placement_new", "strip_tones", "strip_all")


def _apply_case(template: str, text: str) -> str:
    """Copy the capitalization pattern of `template` onto `text` (same letter count)."""
    tl = U.letters(template)
    xl = U.letters(text)
    if len(tl) != len(xl):
        # lengths can differ only for i/y alternations; fall back to first-letter case
        if template[:1].isupper():
            return text[:1].upper() + text[1:]
        return text
    out = []
    for t, x in zip(tl, xl):
        out.append(x.upper() if t.isupper() else x)
    return "".join(out)


def convert_placement(text: str, style: str) -> str:
    """Rewrite every parseable syllable with the tone mark placed per `style` ('old'|'new').

    Syllables that do not parse (loanwords, typos) are left untouched, so the
    transformation is idempotent and the count of changed words is reported by
    `count_placement_changes`.
    """
    if style not in ("old", "new"):
        raise ValueError(style)

    def repl(m: re.Match) -> str:
        w = m.group(0)
        p = try_parse(w, strict=False)
        if p is None or p.syllable.tone == 0:
            return w
        # keep non-standard i/y spellings (lý) as written; only move the mark
        if p.i_y_variant in ("y", "i", "nonstandard"):
            return w
        return _apply_case(w, spell(p.syllable, style=style))

    return _WORD.sub(repl, U.nfc(text))


def count_placement_changes(text: str) -> int:
    a = convert_placement(text, "old")
    b = convert_placement(text, "new")
    return sum(1 for x, y in zip(_WORD.findall(a), _WORD.findall(b)) if x != y)


def to_nfc(text: str) -> str:
    return U.nfc(text)


def to_nfd(text: str) -> str:
    return U.nfd(text)


def to_win1258(text: str) -> str:
    return U.to_nfd_partial_windows1258(text)


def strip_tones(text: str) -> str:
    return U.strip_tones(text)


def strip_all(text: str) -> str:
    return U.strip_diacritics(text, keep_d=False)


def reencode(text: str, arm: str) -> str:
    """Apply one named arm. 'nfc'/'nfd'/'win1258' change the encoding only; the
    placement arms first normalize to NFC (placement is defined on letters) and
    return NFC; the strip arms return NFC."""
    if arm == "nfc":
        return to_nfc(text)
    if arm == "nfd":
        return to_nfd(text)
    if arm == "win1258":
        return to_win1258(text)
    if arm == "placement_old":
        return convert_placement(text, "old")
    if arm == "placement_new":
        return convert_placement(text, "new")
    if arm == "strip_tones":
        return strip_tones(text)
    if arm == "strip_all":
        return strip_all(text)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


def meaning_preserving(arm: str) -> bool:
    return arm in ("nfc", "nfd", "win1258", "placement_old", "placement_new")


def canonical_text(text: str) -> str:
    """Scoring normalization: NFC, lowercase, new-style placement, standard i/y spelling,
    collapsed whitespace, no surrounding punctuation. Two answers that differ only in
    encoding, placement or a lý/lí alternation compare equal after this."""
    t = U.nfc(text).strip().lower()
    t = re.sub(r"[\s]+", " ", t)
    t = t.strip(" .,;:!?\"'“”‘’()[]{}«»-–—*_`")

    def repl(m: re.Match) -> str:
        w = m.group(0)
        # strict: a spelling-rule violation (céo for kéo) is NOT repaired, so that a
        # misspelled answer scores as wrong and the scorer can classify it as a spelling error
        p = try_parse(w, strict=True)
        if p is None:
            return w
        return spell(p.syllable, style="new")

    return _WORD.sub(repl, t)
