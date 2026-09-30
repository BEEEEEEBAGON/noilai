"""Nói lái: the four swap variants on a pair of syllables.

Given two syllables A = (oA, rA, tA) and B = (oB, rB, tB) with onset o, rime r
(glide + nucleus + coda) and tone t:

  V1  swap rimes            (oA rB tA)(oB rA tB)     mèo cái -> mài kéo
  V2  swap onset+rime       (oB rB tA)(oA rA tB)     đầu tiên -> tiền đâu   (= swap syllables, keep tones in place)
  V3  swap tones            (oA rA tB)(oB rB tA)     hiện đại -> hiền đậi?  (tones move, segments stay)
  V4  swap rime+tone        (oA rB tB)(oB rA tA)     bí mật -> bật mí       (= swap everything but the onsets)

Every variant is an involution (applying it twice returns the input), and the
four together with the identity and 'swap whole syllables' form the group of
component permutations that keep each syllable's onset/rime/tone slots filled;
the two remaining permutations (swap onsets only; swap onsets and tones) are
V1 and V4 composed with a whole-syllable swap, so they are not separate
variants: swapping onsets only produces the same two syllables as V4 in the
other order, and 'swap onsets and tones' produces V1's syllables in the other
order. The taxonomy therefore covers every distinct output pair up to word
order; the generator records the order it emits.

The functions here are pure. Legality of the outputs (spelling constraints,
attested rimes, stop-coda tones) is decided by `Inventory.is_legal` in the
generator, which drops pairs that produce an illegal syllable or return the
input unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..vi.syllable import Syllable, replace

# The four generated T1 cells (the plan's V1-V4) and the two further members of the
# six-type tradition (docs/DESIGN_DECISIONS.md section 3): V5 swaps onsets only and V6
# swaps onsets and tones. Under order reversal reverse(V1) = V6, reverse(V2) = V3 and
# reverse(V4) = V5, so V5/V6 add no new unordered output pair; they enter the taxonomy,
# T2 gold, T3 twin material and the attested labels, but are not T1 cells.
VARIANTS = ("V1", "V2", "V3", "V4")
ALL_VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6")
UNORDERED_CLASS = {"V1": "V1/V6", "V6": "V1/V6", "V2": "V2/V3", "V3": "V2/V3", "V4": "V4/V5", "V5": "V4/V5"}
REVERSE_OF = {"V1": "V6", "V6": "V1", "V2": "V3", "V3": "V2", "V4": "V5", "V5": "V4"}
VARIANT_NAMES_VI = {
    "V1": "đổi vần, giữ phụ âm đầu và thanh",
    "V2": "đổi cả phụ âm đầu và vần, giữ thanh",
    "V3": "đổi thanh, giữ phụ âm đầu và vần",
    "V4": "đổi vần và thanh, giữ phụ âm đầu",
    "V5": "đổi phụ âm đầu, giữ vần và thanh",
    "V6": "đổi phụ âm đầu và thanh, giữ vần",
}
VARIANT_NAMES_EN = {
    "V1": "swap rimes; keep onsets and tones",
    "V2": "swap onsets and rimes; keep tones in place",
    "V3": "swap tones; keep onsets and rimes",
    "V4": "swap rimes and tones; keep onsets",
    "V5": "swap onsets; keep rimes and tones",
    "V6": "swap onsets and tones; keep rimes",
}
# Which components move under each variant (used for T3 twin construction and the error taxonomy).
MOVES = {
    "V1": {"rime"},
    "V2": {"onset", "rime"},
    "V3": {"tone"},
    "V4": {"rime", "tone"},
    "V5": {"onset"},
    "V6": {"onset", "tone"},
}


def apply(variant: str, a: Syllable, b: Syllable) -> tuple[Syllable, Syllable]:
    """Return the two output syllables (in order) for `variant` on the pair (a, b)."""
    if variant == "V1":
        return (replace(a, glide=b.glide, nucleus=b.nucleus, coda=b.coda),
                replace(b, glide=a.glide, nucleus=a.nucleus, coda=a.coda))
    if variant == "V2":
        return (replace(b, tone=a.tone), replace(a, tone=b.tone))
    if variant == "V3":
        return (replace(a, tone=b.tone), replace(b, tone=a.tone))
    if variant == "V4":
        return (replace(a, glide=b.glide, nucleus=b.nucleus, coda=b.coda, tone=b.tone),
                replace(b, glide=a.glide, nucleus=a.nucleus, coda=a.coda, tone=a.tone))
    if variant == "V5":
        return (replace(a, onset=b.onset), replace(b, onset=a.onset))
    if variant == "V6":
        return (replace(a, onset=b.onset, tone=b.tone), replace(b, onset=a.onset, tone=a.tone))
    raise ValueError(f"unknown variant {variant!r}")


def is_plain_reversal(variant: str, a: Syllable, b: Syllable) -> bool:
    """True when the variant's output is just the input in the other order."""
    return apply(variant, a, b) == (b, a)


def is_identity(variant: str, a: Syllable, b: Syllable) -> bool:
    """True when the variant returns the input pair unchanged (e.g. V3 on equal tones)."""
    out = apply(variant, a, b)
    return out == (a, b)


def identify(a: Syllable, b: Syllable, x: Syllable, y: Syllable) -> str | None:
    """Which variant maps (a, b) to (x, y) exactly in this order? None if none does."""
    for v in VARIANTS:
        if apply(v, a, b) == (x, y):
            return v
    return None


def identify_any_order(a: Syllable, b: Syllable, x: Syllable, y: Syllable) -> list[tuple[str, bool]]:
    """All (variant, reversed) pairs mapping (a, b) onto {x, y} in either order."""
    out = []
    for v in ALL_VARIANTS:
        o = apply(v, a, b)
        if o == (x, y):
            out.append((v, False))
        elif o == (y, x):
            out.append((v, True))
    return out


@dataclass(frozen=True)
class Diff:
    """Component-level difference between two syllables."""
    onset: bool
    glide: bool
    nucleus: bool
    coda: bool
    tone: bool

    @property
    def rime(self) -> bool:
        return self.glide or self.nucleus or self.coda

    @property
    def any(self) -> bool:
        return self.onset or self.rime or self.tone


def diff(x: Syllable, y: Syllable) -> Diff:
    return Diff(onset=x.onset != y.onset, glide=x.glide != y.glide, nucleus=x.nucleus != y.nucleus,
                coda=x.coda != y.coda, tone=x.tone != y.tone)
