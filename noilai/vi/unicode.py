"""Unicode handling for Vietnamese: tone marks, quality diacritics, NFC/NFD, stripping.

Vietnamese letters carry up to two diacritics: a *quality* mark that changes the
vowel (breve on ă, circumflex on â ê ô, horn on ơ ư) and a *tone* mark (grave,
acute, hook above, tilde, dot below). đ/Đ is a separate letter with no
decomposition. Precomposed code points exist for every combination (Latin-1,
Latin Extended-A and Latin Extended Additional), so text can be written in NFC
(one code point per letter) or NFD (base letter + combining marks); both are
valid Vietnamese and both occur in the wild.

Everything here is pure functions on strings. Nothing knows about syllables.
"""
from __future__ import annotations

import unicodedata

# ---------------------------------------------------------------- tone marks
# Index order is the conventional order of the six tones.
TONE_NAMES = ("ngang", "huyen", "sac", "hoi", "nga", "nang")
TONE_NAMES_VI = ("ngang", "huyền", "sắc", "hỏi", "ngã", "nặng")
TONE_MARKS = ("", "̀", "́", "̉", "̃", "̣")
MARK_TO_TONE = {m: i for i, m in enumerate(TONE_MARKS) if m}
TONE_MARK_SET = frozenset(MARK_TO_TONE)

# --------------------------------------------------------- quality diacritics
BREVE = "̆"       # ă
CIRCUMFLEX = "̂"  # â ê ô
HORN = "̛"        # ơ ư
QUALITY_MARK_SET = frozenset({BREVE, CIRCUMFLEX, HORN})

# Canonical combining class: dot below is 220, the other four tone marks and the
# three quality marks are 230. NFD therefore orders dot-below BEFORE a quality
# mark (ậ -> a U+0323 U+0302) but keeps the source order among class-230 marks
# (ấ -> a U+0302 U+0301 from the precomposed form).

VOWEL_BASES = frozenset("aeiouy")
# All 12 toneless vowel letters as NFC strings.
VOWELS_NFC = ("a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y")
VOWELS_NFC_SET = frozenset(VOWELS_NFC)

CONSONANT_LETTERS = frozenset("bcdđghklmnpqrstvx")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def is_nfc(s: str) -> bool:
    return unicodedata.is_normalized("NFC", s)


def is_nfd(s: str) -> bool:
    return unicodedata.is_normalized("NFD", s)


def decompose_letter(ch: str) -> tuple[str, str, int]:
    """Split one (possibly precomposed) letter into (base, quality_mark, tone).

    `ch` may be an NFC letter or an NFD sequence (base + marks). Returns the
    base letter (lowercase preserved as given), the quality mark ('' or one of
    BREVE/CIRCUMFLEX/HORN) and the tone index 0..5. Raises ValueError if the
    sequence carries more than one tone mark or an unknown combining mark.
    """
    d = nfd(ch)
    base = d[0]
    quality = ""
    tone = 0
    seen_tone = False
    for m in d[1:]:
        if m in QUALITY_MARK_SET:
            if quality:
                raise ValueError(f"two quality marks in {ch!r}")
            quality = m
        elif m in MARK_TO_TONE:
            if seen_tone:
                raise ValueError(f"two tone marks in {ch!r}")
            tone = MARK_TO_TONE[m]
            seen_tone = True
        else:
            raise ValueError(f"unknown combining mark {m!r} in {ch!r}")
    return base, quality, tone


def compose_letter(base: str, quality: str = "", tone: int = 0) -> str:
    """Inverse of decompose_letter, returned in NFC."""
    return nfc(base + quality + TONE_MARKS[tone])


def letters(s: str) -> list[str]:
    """Split an NFC/NFD string into NFC letters (each letter = base + its marks).

    Combining marks attach to the preceding base character. Non-Vietnamese
    combining sequences are left as they are (they will simply not parse later).
    """
    out: list[str] = []
    for ch in nfd(s):
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return [nfc(x) for x in out]


def strip_tones(s: str) -> str:
    """Remove tone marks only; keep quality diacritics and đ (bàn -> ban, mượn -> mươn)."""
    return nfc("".join(ch for ch in nfd(s) if ch not in TONE_MARK_SET))


def strip_diacritics(s: str, keep_d: bool = False) -> str:
    """Remove every diacritic (tone and quality). đ/Đ becomes d/D unless keep_d.

    This is the C3 "accent stripping" arm: it destroys information (bàn, bán,
    bạn -> ban) and is the one re-encoding that does NOT preserve meaning.
    """
    out = []
    for ch in nfd(s):
        if unicodedata.combining(ch):
            continue
        if not keep_d:
            if ch == "đ":
                ch = "d"
            elif ch == "Đ":
                ch = "D"
        out.append(ch)
    return nfc("".join(out))


def to_nfd_partial_windows1258(s: str) -> str:
    """Windows-1258 style encoding: quality diacritics precomposed, tone marks combining.

    This "partially composed" form (e.g. 'ấ' as U+00E2 U+0301) is what legacy
    Vietnamese Windows text and some keyboard drivers produce; it is neither
    NFC nor NFD. Provided so the re-encoding census can include it as a third
    encoding condition; the paper's C1 arm compares NFC with NFD.
    """
    out = []
    for letter in letters(s):
        base, quality, tone = _safe_decompose(letter)
        if base is None:
            out.append(letter)
            continue
        out.append(nfc(base + quality) + TONE_MARKS[tone])
    return "".join(out)


def _safe_decompose(letter: str):
    try:
        return decompose_letter(letter)
    except ValueError:
        return None, "", 0


def tone_of_letter(letter: str) -> int:
    """Tone index of a single letter (0 if none or not a vowel)."""
    try:
        return decompose_letter(letter)[2]
    except ValueError:
        return 0


def count_tone_marks(s: str) -> int:
    return sum(1 for ch in nfd(s) if ch in TONE_MARK_SET)


def encoding_form(s: str) -> str:
    """Name the Unicode form of a string: 'nfc', 'nfd', 'mixed' or 'ascii'."""
    if s.isascii():
        return "ascii"
    if is_nfc(s):
        return "nfc"
    if is_nfd(s):
        return "nfd"
    return "mixed"
