"""Vietnamese syllable parser and speller.

A written Vietnamese syllable is  onset + rime + tone,  where the rime is
glide? + nucleus + coda?. This module parses an orthographic syllable into that
structure (canonical, phonemic-ish symbols) and spells a structure back into
standard orthography, applying the spelling rules:

* /k/ is written  q  before the glide,  k  before e ê i (iê, ia),  c  elsewhere;
* /ɣ/ is written  gh  before e ê i, else  g;   /ŋ/ is  ngh  before e ê i, else  ng;
* the onset  gi  drops the rime's initial  i  (gi + i = gì, gi + iêng = giêng);
* the glide is written  u  after q and before â ê ơ ô i ia iê, and  o  before a ă e;
* the nucleus  i  is written  y  after the glide and in the bare syllable  y/ý,
  ia -> ya and iê -> yê after the glide, iê -> yê with zero onset (yêu, yên);
* the short vowel ă is written  a  before the semivowel codas (ay, au);
* the semivowel coda /j/ is  y  after ă â and  i  elsewhere; /w/ is  o  after a e
  and  u  elsewhere;
* the tone mark sits on the nucleus letter (new style) or, in old style, on the
  first vowel letter of the open rimes oa oe uy (hòa/hoà, khỏe/khoẻ, thủy/thuỷ);
* syllables ending in a stop (p t c ch) take only the tones sắc and nặng.

Canonical symbols
-----------------
onset : '' b c ch d đ g gi h kh l m n ng nh p ph r s t th tr v x
        ('c' stands for /k/ however spelled; 'g' for /ɣ/ however spelled;
         'ng' for /ŋ/ however spelled; 'p' occurs only in loanwords)
glide : bool
nucleus: a ă â e ê i o ô ơ u ư ia iê ua uô ưa ươ oo
coda  : '' m n ng nh p t c ch j w      (j = /j/ written i/y, w = /w/ written o/u)
tone  : 0..5 (ngang huyền sắc hỏi ngã nặng)

Legality is decided by an attested inventory of syllables (the Hunspell vi_VN
list, 6,642 syllables) loaded by noilai.vi.lexicon: a structure is legal if its
(glide, nucleus, coda) rime is attested, its onset+rime combination is attested
(with any tone), and the tone respects the stop-coda constraint. The generator
never emits a syllable that fails `is_legal`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from . import unicode as U

# ------------------------------------------------------------------ onsets
# Surface spellings (longest first when matching) -> canonical onset.
ONSET_SPELLINGS = {
    "ngh": "ng", "ng": "ng", "nh": "nh", "kh": "kh", "ph": "ph", "th": "th",
    "tr": "tr", "ch": "ch", "gh": "g", "gi": "gi", "qu": "c",
    "b": "b", "c": "c", "d": "d", "đ": "đ", "g": "g", "h": "h", "k": "c",
    "l": "l", "m": "m", "n": "n", "p": "p", "r": "r", "s": "s", "t": "t",
    "v": "v", "x": "x",
}
CANONICAL_ONSETS = ("", "b", "c", "ch", "d", "đ", "g", "gi", "h", "kh", "l", "m",
                    "n", "ng", "nh", "p", "ph", "r", "s", "t", "th", "tr", "v", "x")
_ONSET_KEYS = sorted(ONSET_SPELLINGS, key=len, reverse=True)

NUCLEI = ("a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư",
          "ia", "iê", "ua", "uô", "ưa", "ươ", "oo")
CONSONANT_CODAS = ("m", "n", "ng", "nh", "p", "t", "c", "ch")
CODAS = ("",) + CONSONANT_CODAS + ("j", "w")
STOP_CODAS = frozenset({"p", "t", "c", "ch"})
# Diphthongs that occur only without a coda vs only with one.
OPEN_ONLY_NUCLEI = frozenset({"ia", "ua", "ưa"})
CLOSED_ONLY_NUCLEI = frozenset({"iê", "uô", "ươ", "ă", "â"})

# Front vowels that trigger k / gh / ngh.
_FRONT_FIRST = frozenset({"e", "ê", "i", "iê", "ia"})


@dataclass(frozen=True)
class Syllable:
    onset: str            # canonical
    glide: bool
    nucleus: str          # canonical
    coda: str             # canonical ('' j w or consonant)
    tone: int             # 0..5

    # ---- convenience views -------------------------------------------------
    @property
    def rime(self) -> str:
        """Canonical toneless rime key, e.g. 'w+a+j' for oai/quai."""
        return ("w+" if self.glide else "") + self.nucleus + ("+" + self.coda if self.coda else "")

    @property
    def rime_with_tone(self) -> str:
        return f"{self.rime}/{self.tone}"

    def with_tone(self, tone: int) -> Syllable:
        return replace(self, tone=tone)

    def with_onset(self, onset: str) -> Syllable:
        return replace(self, onset=onset)

    def with_rime_of(self, other: Syllable) -> Syllable:
        return replace(self, glide=other.glide, nucleus=other.nucleus, coda=other.coda)

    def spelled(self, style: str = "new") -> str:
        return spell(self, style=style)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return spell(self)


@dataclass(frozen=True)
class Parse:
    """A parsed surface syllable: the structure plus how it was written."""
    syllable: Syllable
    surface: str              # as given (NFC)
    onset_spelling: str       # e.g. 'k', 'qu', 'ngh'
    rime_spelling: str        # toneless rime letters as written, e.g. 'oa', 'uyên'
    tone_letter_index: int    # index (within the rime letters) of the letter bearing the tone, -1 if none
    placement: str            # 'new', 'old', 'same' (styles agree), 'none' (no tone), 'invalid'
    i_y_variant: str | None  # 'y' if a non-standard y was used for nucleus i (lý), 'i' if i used where y is standard (quí), else None
    capitalization: str       # 'lower', 'title', 'upper', 'mixed'


class ParseError(ValueError):
    pass


# ------------------------------------------------------------------ parsing
_GLIDE_O_NUCLEI = frozenset({"a", "ă", "e"})
_GLIDE_U_NEXT = frozenset({"â", "ê", "ơ", "y"})  # letters after which a written u is the glide (uâ uê uơ uy); uô ua ui are not


def parse(text: str, strict: bool = True) -> Parse:
    """Parse one orthographic syllable. Raises ParseError if it is not a syllable.

    Standard spellings and the attested i/y alternations (lý for lí, quí for quý,
    í for ý) are accepted in both modes and recorded in Parse.i_y_variant.
    `strict=False` additionally accepts any structurally parseable spelling that
    violates a spelling rule (gen for ghen, ka for ca) and tags it 'nonstandard';
    `strict=True` rejects those, and rejects a tone mark on the wrong letter.
    """
    if not isinstance(text, str) or not text:
        raise ParseError("empty")
    surface = U.nfc(text.strip())
    if " " in surface:
        raise ParseError(f"not a single syllable: {surface!r}")
    cap = _capitalization(surface)
    low = surface.lower()
    lets = U.letters(low)
    # Split letters into base+quality (toneless letter) and tone.
    toneless: list[str] = []
    tone = 0
    tone_idx = -1
    n_tone = 0
    for i, letter in enumerate(lets):
        try:
            base, quality, t = U.decompose_letter(letter)
        except ValueError as e:
            raise ParseError(f"bad letter {letter!r} in {surface!r}: {e}") from e
        if base not in U.VOWEL_BASES and base not in U.CONSONANT_LETTERS:
            raise ParseError(f"non-Vietnamese letter {letter!r} in {surface!r}")
        if quality and base not in "aeou":
            raise ParseError(f"quality mark on {base!r} in {surface!r}")
        if t:
            if base not in U.VOWEL_BASES:
                raise ParseError(f"tone mark on consonant in {surface!r}")
            n_tone += 1
            tone = t
            tone_idx = i
        toneless.append(U.compose_letter(base, quality, 0))
    if n_tone > 1:
        raise ParseError(f"more than one tone mark in {surface!r}")
    s = "".join(toneless)

    # --- onset
    onset_sp = ""
    for k in _ONSET_KEYS:
        if s.startswith(k):
            onset_sp = k
            break
    rest = s[len(onset_sp):]
    onset = ONSET_SPELLINGS.get(onset_sp, "")
    # 'qu' is /k/ + glide: put the u back as the glide letter.
    if onset_sp == "qu":
        rest = "u" + rest
    # 'gi' contraction: gì, gìn, giêng, giết -> the rime starts with i.
    if onset_sp == "gi":
        if rest == "" or rest[0] not in U.VOWELS_NFC_SET or rest[0] == "ê":
            rest = "i" + rest
    # 'g' before i/e/ê is spelled gh; a bare 'g'+front vowel is not a standard spelling
    # (gi handles /z/). 'ng' + front vowel must be 'ngh'. 'c' + front vowel must be 'k'.
    # These are checked after the rime is known (spelling round-trip).
    if not rest:
        raise ParseError(f"no rime in {surface!r}")
    if any(ch not in U.VOWELS_NFC_SET and ch not in U.CONSONANT_LETTERS for ch in rest):
        raise ParseError(f"bad rime letters in {surface!r}")

    # --- glide
    glide = False
    if onset_sp == "qu":
        glide = True
        rest = rest[1:]
        if not rest or rest[0] not in U.VOWELS_NFC_SET:
            raise ParseError(f"qu without a vowel in {surface!r}")
    elif len(rest) >= 2 and rest[0] == "o" and rest[1] in _GLIDE_O_NUCLEI or len(rest) >= 2 and rest[0] == "u" and rest[1] in _GLIDE_U_NEXT:
        glide = True
        rest = rest[1:]

    # --- nucleus (longest match), then coda
    nucleus, rest, i_y = _match_nucleus(rest, glide=glide, onset=onset)
    coda = _match_coda(rest, nucleus)
    if coda is None:
        raise ParseError(f"bad coda {rest!r} in {surface!r}")
    if nucleus == "a" and rest in ("y", "u"):
        nucleus = "ă"          # ay au (oay quay): short a before a semivowel coda is written a

    syl = Syllable(onset=onset, glide=glide, nucleus=nucleus, coda=coda, tone=tone)

    # --- structural constraints that hold regardless of the inventory
    if nucleus in OPEN_ONLY_NUCLEI and coda:
        raise ParseError(f"{nucleus} takes no coda: {surface!r}")
    if nucleus in CLOSED_ONLY_NUCLEI and not coda:
        raise ParseError(f"{nucleus} needs a coda: {surface!r}")
    if coda in STOP_CODAS and tone not in (0, 2, 5):
        # tone 0 is allowed here only because toneless input is a valid query
        # form for the parser; is_legal() enforces sắc/nặng for real syllables.
        raise ParseError(f"stop coda with tone {U.TONE_NAMES[tone]}: {surface!r}")

    # --- spelling round-trip: the standard spelling must reproduce the input
    std = spell(syl, style="new")
    std_old = spell(syl, style="old")
    variant = i_y
    if low != std and low != std_old:
        alt = _accepted_variants(syl)
        if low in alt:
            variant = alt[low]
        elif not strict:
            variant = variant or "nonstandard"
        else:
            raise ParseError(f"non-standard spelling {surface!r} (standard: {std!r})")

    # --- placement
    rime_letters = _rime_letters_of(low, onset_sp)
    tone_letter_index = -1
    placement = "none"
    if tone:
        pos_new = _tone_position(syl, style="new", spelled=std)
        pos_old = _tone_position(syl, style="old", spelled=std_old)
        # locate the letter carrying the tone among the rime letters of the input
        rl = U.letters(low)
        tone_letter_index = tone_idx - (len(rl) - len(rime_letters))
        if pos_new == pos_old:
            placement = "same" if tone_letter_index == pos_new else "invalid"
        elif tone_letter_index == pos_new:
            placement = "new"
        elif tone_letter_index == pos_old:
            placement = "old"
        else:
            placement = "invalid"
        if placement == "invalid" and strict:
            raise ParseError(f"tone mark on the wrong letter in {surface!r}")

    return Parse(syllable=syl, surface=surface, onset_spelling=onset_sp,
                 rime_spelling=U.strip_tones(low[len(onset_sp):]) if onset_sp != "qu" else "u" + U.strip_tones(low[2:]),
                 tone_letter_index=tone_letter_index, placement=placement,
                 i_y_variant=variant, capitalization=cap)


def try_parse(text: str, strict: bool = True) -> Parse | None:
    try:
        return parse(text, strict=strict)
    except ParseError:
        return None


def _capitalization(s: str) -> str:
    if s.islower() or not any(c.isalpha() for c in s):
        return "lower"
    if s.isupper() and len(s) > 1:
        return "upper"
    if s[0].isupper() and s[1:].islower():
        return "title"
    return "mixed"


def _match_nucleus(rest: str, glide: bool, onset: str) -> tuple[str, str, str | None]:
    """Return (canonical nucleus, remaining letters, i/y variant note)."""
    variant = None
    if glide:
        # after the glide: a ă e â ê ơ ô, i (written y), ia (ya), iê (yê)
        for cand, canon in (("yê", "iê"), ("ya", "ia"), ("iê", "iê"), ("ia", "ia"),
                            ("y", "i"), ("i", "i"), ("a", "a"), ("ă", "ă"), ("e", "e"),
                            ("â", "â"), ("ê", "ê"), ("ơ", "ơ"), ("ô", "ô")):
            if rest.startswith(cand):
                if cand in ("i", "iê", "ia"):
                    variant = "i"       # qui, quiên: non-standard for quy, quyên
                return canon, rest[len(cand):], variant
        raise ParseError(f"no nucleus after glide in {rest!r}")
    for cand in ("iê", "yê", "ia", "uô", "ua", "ươ", "ưa", "oo"):
        if rest.startswith(cand):
            if cand == "yê":
                if onset != "":
                    raise ParseError(f"yê after an onset in {rest!r}")
                return "iê", rest[2:], None
            if cand == "iê" and onset == "":
                variant = "i"       # 'iêu' for 'yêu': non-standard
            return cand, rest[len(cand):], variant
    if rest[0] in U.VOWELS_NFC_SET:
        ch = rest[0]
        if ch == "y":
            # nucleus i written y: standard only for the bare syllable (y, ý)
            after = rest[1:]
            if not (onset == "" and after == ""):
                variant = "y"
            return "i", after, variant
        if ch == "i" and onset == "" and rest[1:] == "":
            variant = "i"           # 'i' for standard 'y'
        return ch, rest[1:], variant
    raise ParseError(f"no nucleus in {rest!r}")


def _match_coda(rest: str, nucleus: str) -> str | None:
    if rest == "":
        return ""
    if rest in CONSONANT_CODAS:
        return rest
    if rest in ("i", "y"):
        return "j"
    if rest in ("o", "u"):
        return "w"
    return None


# ------------------------------------------------------------------ spelling
def spell(syl: Syllable, style: str = "new") -> str:
    """Spell a Syllable in standard orthography with the tone mark placed per `style`."""
    if style not in ("new", "old"):
        raise ValueError("style must be 'new' or 'old'")
    body = _spell_toneless(syl)
    if syl.tone == 0:
        return body
    pos = _tone_position(syl, style=style, spelled=body)
    lets = U.letters(body)
    onset_len = len(lets) - len(_rime_letters_of(body, _onset_spelling(syl)))
    idx = onset_len + pos
    base, quality, _ = U.decompose_letter(lets[idx])
    lets[idx] = U.compose_letter(base, quality, syl.tone)
    return U.nfc("".join(lets))


def _onset_spelling(syl: Syllable) -> str:
    o = syl.onset
    # k/gh/ngh are triggered by the first WRITTEN letter after the onset. With a
    # glide that letter is o or u (ngoe, nguy, ghe vs goe), so the trigger is off.
    front = (not syl.glide) and syl.nucleus in _FRONT_FIRST
    if o == "c":
        if syl.glide:
            return "qu"
        return "k" if front else "c"
    if o == "g":
        return "gh" if front else "g"
    if o == "ng":
        return "ngh" if front else "ng"
    return o


def _spell_toneless(syl: Syllable) -> str:
    onset_sp = _onset_spelling(syl)
    nuc = syl.nucleus
    glide = ""
    if syl.glide:
        if syl.onset == "c" or nuc in ("â", "ê", "ơ", "ô", "i", "ia", "iê"):
            glide = "u"
        else:
            glide = "o"
    # nucleus spelling
    if nuc == "i":
        nuc_sp = "y" if (syl.glide or (syl.onset == "" and syl.coda == "")) else "i"
    elif nuc == "ia":
        nuc_sp = "ya" if syl.glide else "ia"
    elif nuc == "iê":
        nuc_sp = "yê" if (syl.glide or syl.onset == "") else "iê"
    elif nuc == "ă" and syl.coda in ("j", "w"):
        nuc_sp = "a"
    else:
        nuc_sp = nuc
    # coda spelling
    if syl.coda == "j":
        coda_sp = "y" if nuc in ("ă", "â") else "i"
    elif syl.coda == "w":
        coda_sp = "o" if nuc in ("a", "e") else "u"
    else:
        coda_sp = syl.coda
    if onset_sp == "qu":
        # the glide letter is already the u of qu
        return "qu" + nuc_sp + coda_sp
    if onset_sp == "gi" and nuc_sp.startswith("i"):
        return "gi" + nuc_sp[1:] + coda_sp
    return onset_sp + glide + nuc_sp + coda_sp


def _rime_letters_of(spelled_lower: str, onset_sp: str) -> list[str]:
    """Letters of the rime as written, treating the u of 'qu' as an onset letter
    (it never carries the tone) and re-inserting the i that 'gi' contracts."""
    lets = U.letters(spelled_lower)
    n = len(U.letters(onset_sp))
    rime = lets[n:]
    if onset_sp == "gi" and (not rime or U.strip_tones(rime[0]) not in U.VOWELS_NFC_SET or U.strip_tones(rime[0]) == "ê"):
        rime = ["i"] + rime
    return rime


def _tone_position(syl: Syllable, style: str, spelled: str) -> int:
    """Index within the written rime letters of the letter that carries the tone."""
    onset_sp = _onset_spelling(syl)
    toneless = U.strip_tones(spelled)
    rime = _rime_letters_of(toneless, onset_sp)
    # the tone-bearing letter is a vowel letter of the nucleus
    vowel_idx = [i for i, ch in enumerate(rime) if ch in U.VOWELS_NFC_SET]
    # glide letter (written o or u) is the first vowel letter when glide and onset is not qu
    glide_written = syl.glide and onset_sp != "qu"
    nucleus_idx = vowel_idx[1:] if glide_written else vowel_idx
    if syl.coda in ("j", "w"):
        nucleus_idx = nucleus_idx[:-1]  # drop the semivowel coda letter
    if not nucleus_idx:
        raise ParseError(f"no tone-bearing letter in {spelled!r}")
    if syl.nucleus in ("iê", "uô", "ươ"):
        pos = nucleus_idx[-1]          # second letter of the diphthong (tiếng, muốn, mượn)
    elif syl.nucleus in ("ia", "ua", "ưa"):
        pos = nucleus_idx[0]           # first letter (mía, mùa, mưa)
    elif syl.nucleus == "oo":
        pos = nucleus_idx[-1]          # boóng, coóc, toòng: the second o carries the mark (Hunspell list)
    else:
        pos = nucleus_idx[0]
    if style == "old" and glide_written and syl.coda == "" and syl.nucleus in ("a", "e", "i"):
        # hòa, khỏe, thủy: old style puts the mark on the glide letter
        pos = vowel_idx[0]
    # Contracted gi (gì, gìn, giếng): _rime_letters_of re-inserts the virtual i, so the
    # index is relative to a rime that starts one letter before the written rime; the
    # caller subtracts the same offset, so no shift is needed here.
    return pos


def _accepted_variants(syl: Syllable) -> dict[str, str]:
    """Non-standard but attested spellings of a structure -> variant tag."""
    out: dict[str, str] = {}
    if syl.nucleus == "i" and not syl.glide and syl.onset != "" and syl.coda == "":
        # lí/lý, kĩ/kỹ, mĩ/mỹ, hi/hy, sĩ/sỹ, tỉ/tỷ, quí is handled via glide
        body = _spell_toneless(syl)
        alt = body[:-1] + "y"
        out[_place(alt, syl, len(U.letters(alt)) - 1)] = "y"
    if syl.glide and syl.onset == "c" and syl.nucleus in ("i", "iê"):
        body = _spell_toneless(syl)           # quy / quyên
        alt = body.replace("quy", "qui", 1)
        for st in ("new", "old"):
            out[_place_like(alt, syl, st)] = "i"
    if syl.onset == "" and syl.nucleus == "i" and syl.coda == "":
        out[U.compose_letter("i", "", syl.tone)] = "i"   # í for ý
    return out


def _place(body: str, syl: Syllable, letter_index: int) -> str:
    lets = U.letters(body)
    base, quality, _ = U.decompose_letter(lets[letter_index])
    lets[letter_index] = U.compose_letter(base, quality, syl.tone)
    return U.nfc("".join(lets))


def _place_like(alt_body: str, syl: Syllable, style: str) -> str:
    """Place the tone on alt_body at the same rime index the standard spelling uses."""
    std = _spell_toneless(syl)
    pos = _tone_position(syl, style=style, spelled=std)
    onset_len = len(U.letters(std)) - len(_rime_letters_of(std, _onset_spelling(syl)))
    return _place(alt_body, syl, onset_len + pos)


# ------------------------------------------------------------ legality
class Inventory:
    """Attested syllable inventory used to decide legality.

    Built from a list of surface syllables (see noilai.vi.lexicon.load_inventory).
    """

    def __init__(self, syllables: Iterable[str]):
        self.syllables: set[str] = set()
        self.structures: set[Syllable] = set()
        self.toneless: set[tuple] = set()      # (onset, glide, nucleus, coda)
        self.rimes: set[str] = set()           # canonical rime keys
        self.rime_tones: dict[str, set[int]] = {}
        self.unparsed: list[str] = []
        self.rejected: list[str] = []          # parseable but phonotactically impossible (gip, têt, hoc)
        for w in syllables:
            p = try_parse(w, strict=False)
            if p is None:
                self.unparsed.append(w)
                continue
            s = p.syllable
            if s.coda in STOP_CODAS and s.tone not in (2, 5):
                self.rejected.append(w)
                continue
            self.syllables.add(U.nfc(w.lower()))
            self.structures.add(s)
            self.toneless.add((s.onset, s.glide, s.nucleus, s.coda))
            self.rimes.add(s.rime)
            self.rime_tones.setdefault(s.rime, set()).add(s.tone)

    def is_legal(self, syl: Syllable, level: str = "onset_rime") -> bool:
        """level: 'attested' (exact syllable attested), 'onset_rime' (onset+rime attested
        with some tone, and tone allowed by phonotactics), 'rime' (rime attested only)."""
        if syl.coda in STOP_CODAS and syl.tone not in (2, 5):
            return False
        if syl.nucleus in OPEN_ONLY_NUCLEI and syl.coda:
            return False
        if syl.nucleus in CLOSED_ONLY_NUCLEI and not syl.coda:
            return False
        if level == "attested":
            return syl in self.structures
        if level == "onset_rime":
            return (syl.onset, syl.glide, syl.nucleus, syl.coda) in self.toneless
        if level == "rime":
            return syl.rime in self.rimes
        raise ValueError(level)

    def is_attested(self, syl: Syllable) -> bool:
        return syl in self.structures


def canonical(text: str) -> str | None:
    """Standard (new-style, NFC, lowercase) spelling of a syllable, or None if unparseable."""
    p = try_parse(text, strict=False)
    return None if p is None else spell(p.syllable, style="new")
