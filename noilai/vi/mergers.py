"""Regional mergers of Vietnamese onsets, codas and tones: the DESIGN_DECISIONS 2.1 O3 table, stated once.

Everything that names a merger derives from this module (DESIGN_DECISIONS 2.1 O3, red-team item 12):

* the scorer's `homophone` bin (DESIGN_DECISIONS 5.4 item 8; `noilai.eval.score.HOMOPHONE_ONSET_PAIRS`): the
  onset mergers that 5.4 lists as regional homophone SPELLINGS, `d<->gi`, `ch<->tr`, `s<->x`, `r<->d/gi`, come out
  of `homophone_onset_pairs()`; the `quoc/cuoc`-type /k/ spellings of the same list are a spelling class
  (O3's first sentence, O5), not a merger, and stay in the scorer;
* the attested `approx(merger: <name>)` tags (DESIGN_DECISIONS 3.7; `scripts/build_attested.py`): a name lists
  two or more values of ONE merger separated by `/`, `=` or an arrow (`hỏi/ngã`, `n/ng`, `d/gi`, `tr/ch`),
  several mergers joined by `+`; `resolve` parses one name, `parse_tag` a whole tag, `mergers_between` finds
  the mergers that explain a rule-vs-attested difference and `approximation_name` writes the tag;
* the region-stratified reporting (DESIGN_DECISIONS 10): `for_region` lists the mergers active in a region,
  `REGION_TAGS` maps the seed's `region_tag` letters (N / C / S) to the table's regions.

Spelling classes (`c/k/qu` -> canonical `c`, `g/gh` -> `g`, `ng/ngh` -> `ng`) are the parser's business
(`noilai.vi.syllable.ONSET_SPELLINGS`) and are listed in `SPELLING_CLASSES` for the record only. Every
merger remains a distinct onset, coda or tone in the parser; the table matters for the error taxonomy
(5.4), the attested tags (3.7) and the reporting (10). The parser's canonical symbols are used throughout
(`gi` is an onset of its own; tones are the integers 0..5 of `noilai.vi.unicode.TONE_NAMES`).

NATIVE-CHECK: the list and its regional attribution (high as phonology; the homophone list is NV).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from . import unicode as U

REGIONS = ("everywhere", "north", "south_centre")
# the seed's region_tag column (data/attested_seed.tsv; DESIGN_DECISIONS 10): N = North, C = Centre, S = South
REGION_TAGS = {"N": "north", "C": "south_centre", "S": "south_centre"}
COMPONENTS = ("onset", "coda", "tone")
# O3's first sentence: spelling classes, collapsed by the parser (noilai.vi.syllable.ONSET_SPELLINGS); not mergers
SPELLING_CLASSES = (("c", ("c", "k", "qu")), ("g", ("g", "gh")), ("ng", ("ng", "ngh")))

_TONE_BY_NAME = {n: i for i, n in enumerate(U.TONE_NAMES_VI)} | {n: i for i, n in enumerate(U.TONE_NAMES)}
_SPLIT_RE = re.compile(r"\s*(?:/|=|→|->|↔|<->)\s*")
_TAG_RE = re.compile(r"^approx\(\s*merger\s*:\s*(.+?)\s*\)$")


@dataclass(frozen=True)
class Merger:
    """One row of the O3 table: the parser values a merger identifies, in one region."""
    name: str                 # the canonical tag name (DESIGN_DECISIONS 3.7): the values joined by '/'
    region: str               # everywhere | north | south_centre
    component: str            # onset | coda | tone
    values: frozenset         # parser values (onset / coda strings; tones as integers 0..5)
    example: str              # the table's example pair
    homophone_bin: bool       # DESIGN_DECISIONS 5.4 item 8 lists it as a regional homophone SPELLING
    note: str = ""

    def covers(self, a, b) -> bool:
        """The merger identifies the two (distinct) values."""
        return a != b and a in self.values and b in self.values

    @property
    def labels(self) -> tuple[str, ...]:
        """The values as written in a tag (tones by their Vietnamese names), in the table's order."""
        return tuple(_label(v) for v in _ordered(self.values))

    def as_dict(self) -> dict:
        return {"name": self.name, "region": self.region, "component": self.component, "values": list(self.labels),
                "example": self.example, "homophone_bin": self.homophone_bin, "note": self.note}


_ONSET_ORDER = ("d", "gi", "r", "ch", "tr", "s", "x", "v")
_CODA_ORDER = ("n", "ng", "t", "c")


def _ordered(values: Iterable) -> list:
    vals = list(values)
    if all(isinstance(v, int) for v in vals):
        return sorted(vals)
    order = {k: i for i, k in enumerate(_ONSET_ORDER + _CODA_ORDER)}
    return sorted(vals, key=lambda v: (order.get(v, len(order)), str(v)))


def _label(v) -> str:
    return U.TONE_NAMES_VI[v] if isinstance(v, int) else str(v)


def _merger(name: str, region: str, component: str, values: Iterable, example: str, homophone_bin: bool, note: str = "") -> Merger:
    return Merger(name, region, component, frozenset(values), example, homophone_bin, note)


# DESIGN_DECISIONS 2.1 O3, by region (the table's order). `homophone_bin` follows 5.4 item 8 verbatim: the
# Northern onset mergers and the everywhere d = gi are listed there; the South/Centre v -> d/gi, the coda and
# the tone mergers are not (a change to the bin is a scoring change: docs/DEVIATIONS.md, the scoring-rule freeze).
MERGERS: tuple[Merger, ...] = (
    _merger("d/gi", "everywhere", "onset", ("d", "gi"), "dày/giày", True),
    _merger("d/gi/r", "north", "onset", ("d", "gi", "r"), "rượu/diệu", True, "d/gi = r; 5.4 writes it r<->d/gi"),
    _merger("ch/tr", "north", "onset", ("ch", "tr"), "tranh/chanh", True),
    _merger("s/x", "north", "onset", ("s", "x"), "xa/sa", True),
    _merger("v/d/gi", "south_centre", "onset", ("v", "d", "gi"), "vô/dô", False, "v -> d/gi (onset); a pronunciation merger, not in the 5.4 spelling list"),
    _merger("n/ng", "south_centre", "coda", ("n", "ng"), "đơn/đơng", False, "coda -n = -ng"),
    _merger("t/c", "south_centre", "coda", ("t", "c"), "mắt/mắc", False, "coda -t = -c"),
    _merger("hỏi/ngã", "south_centre", "tone", (3, 4), "củ/cũ", False, "tone hỏi = ngã"),
)
BY_NAME = {m.name: m for m in MERGERS}


def for_region(region: str) -> tuple[Merger, ...]:
    """The mergers active in a region (`everywhere` plus the region's own); a seed `region_tag` letter is accepted."""
    region = REGION_TAGS.get(region, region)
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}: one of {REGIONS} or a tag {sorted(REGION_TAGS)}")
    return tuple(m for m in MERGERS if m.region in ("everywhere", region))


def regions_of(merger: Merger | str) -> tuple[str, ...]:
    """The regions in which a merger holds (`everywhere` expands to every region)."""
    m = BY_NAME[merger] if isinstance(merger, str) else merger
    return REGIONS[1:] if m.region == "everywhere" else (m.region,)


def homophone_onset_pairs() -> frozenset[frozenset[str]]:
    """DESIGN_DECISIONS 5.4 item 8: every pair of onsets identified by a merger flagged `homophone_bin`
    (d/gi, d/r, gi/r, ch/tr, s/x) — the scorer's `homophone` bin, derived here, never typed there."""
    pairs = set()
    for m in MERGERS:
        if m.homophone_bin and m.component == "onset":
            pairs.update(frozenset(p) for p in combinations(sorted(m.values), 2))
    return frozenset(pairs)


def parse_values(name: str) -> frozenset | None:
    """The values a tag name lists (`hỏi/ngã`, `-n/-ng`, `gi→d`, `tr=ch`), as parser values; None when fewer than
    two values are named or tone names and letters are mixed."""
    parts = [p.strip().lstrip("-") for p in _SPLIT_RE.split(name.strip()) if p.strip()]
    if len(parts) < 2:
        return None
    if all(p in _TONE_BY_NAME for p in parts):
        return frozenset(_TONE_BY_NAME[p] for p in parts)
    if any(p in _TONE_BY_NAME for p in parts):
        return None
    return frozenset(parts)


def resolve(name: str) -> tuple[Merger, frozenset] | None:
    """The merger a tag name refers to and the values it names: the smallest merger whose values cover the
    named ones (`d/gi` -> the everywhere merger, `r/d` -> the Northern one, `v/d` -> the Southern one)."""
    vals = parse_values(name)
    if vals is None:
        return None
    covering = [m for m in MERGERS if vals <= m.values]
    if not covering:
        return None
    best = min(covering, key=lambda m: (len(m.values), MERGERS.index(m)))
    return best, vals


def parse_tag(exactness: str) -> list[tuple[Merger, frozenset]] | None:
    """Every (merger, named values) of an `approx(merger: a + b)` tag; None when the tag is not a merger tag
    or a name resolves to no merger."""
    m = _TAG_RE.match(exactness.strip())
    if m is None:
        return None
    out = []
    for part in m.group(1).split("+"):
        r = resolve(part)
        if r is None:
            return None
        out.append(r)
    return out


def mergers_between(rule, attested) -> list[Merger] | None:
    """The mergers that turn one parsed syllable into another: for every differing onset, coda or tone the
    merger covering the pair, in table order and without repeats. None when the two differ in the glide or the
    nucleus, when a difference is covered by no merger, or when they are equal."""
    if (rule.glide, rule.nucleus) != (attested.glide, attested.nucleus):
        return None
    found: list[Merger] = []
    for comp in COMPONENTS:
        a, b = getattr(rule, comp), getattr(attested, comp)
        if a == b:
            continue
        hit = [m for m in MERGERS if m.component == comp and m.covers(a, b)]
        if not hit:
            return None
        m = min(hit, key=lambda m: (len(m.values), MERGERS.index(m)))
        if m not in found:
            found.append(m)
    return found or None


def mergers_between_phrases(rule: Sequence, attested: Sequence) -> list[Merger] | None:
    """`mergers_between` over two phrases of equal length: the union of the mergers of every syllable, in
    table order; None when any syllable's difference is unexplained or the phrases are equal."""
    if len(rule) != len(attested):
        return None
    found: list[Merger] = []
    changed = False
    for r, a in zip(rule, attested):
        if r == a:
            continue
        changed = True
        ms = mergers_between(r, a)
        if ms is None:
            return None
        found += [m for m in ms if m not in found]
    return sorted(found, key=MERGERS.index) if changed else None


def approximation_name(rule: Sequence, attested: Sequence) -> str | None:
    """The DESIGN_DECISIONS 3.7 tag derived from the engine's output and the attested form:
    `approx(merger: n/ng + hỏi/ngã)` (every merger named once, table order); None when the forms are equal
    or the difference is not a merger (a substitution or an unparseable change)."""
    ms = mergers_between_phrases(rule, attested)
    if not ms:
        return None
    return "approx(merger: " + " + ".join(m.name for m in ms) + ")"


def same_mergers(tag: str, mergers: Iterable[Merger]) -> bool:
    """Whether a declared merger tag names exactly the given mergers (order and spelling aside)."""
    parsed = parse_tag(tag)
    if parsed is None:
        return False
    return {m.name for m, _ in parsed} == {m.name for m in mergers}


def table() -> list[dict]:
    """The O3 table as rows (for documents and the data statement), in the table's order."""
    return [m.as_dict() for m in MERGERS]
