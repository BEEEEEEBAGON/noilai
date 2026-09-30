"""Minimal pairs and prompts for activation patching (E4).

A patching pair is two prompts that are token-for-token identical except at the target
syllable, whose tone differs (clean 'bí mật' vs corrupt 'bị mật'; both attested), such
that the correct answer's first token differs between the two. We use the T1 prompt for
V3 (swap tones): the answer to 'bí mật' is 'bị mất' and to 'bị mật' is 'bí mất', so the
first answer token carries the tone that must have been read from the input.

Pairs are built from the attested inventory: same onset, glide, nucleus and coda,
different tone, both attested, tokenized to the same length in the prompt (so that
positions align), and with the two gold answers diverging at their first token.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..gen import variants as V
from ..vi import unicode as U
from ..vi.syllable import Inventory, Syllable, spell

# Minimal-pair SKELETON used to build and align candidate pairs cheaply (and by the paper's
# worked example, tests/test_paper.py). Readout B of design 9.3 does NOT run on it: the
# driver re-renders every retained pair into E1's exact V3 prompt (`rerender_pair`, with the
# IT chat template) before patching.   # NATIVE-CHECK
PATCH_PROMPT = "Nói lái kiểu đổi thanh điệu (giữ phụ âm đầu và vần, đổi chỗ hai thanh): {a} {b} →"


@dataclass
class PatchPair:
    clean: str                  # full prompt text, clean
    corrupt: str                # full prompt text, corrupt
    target_syllable_clean: str
    target_syllable_corrupt: str
    partner: str
    answer_clean: str           # gold answer for the clean prompt
    answer_corrupt: str
    target_char_span: tuple[int, int]   # char span of the target syllable in the prompt


def build_pairs(inv: Inventory, n: int, seed: int = 0, target_first: bool = False, prompt: str = PATCH_PROMPT,
                encoding: str = "nfc") -> list[PatchPair]:
    """Tone-minimal pairs in the V3 prompt. With target_first=False (design 9.3) the varying
    syllable is the SECOND input syllable, so the two gold answers diverge at their first
    token (the first output syllable takes the second input syllable's tone). With
    encoding='nfd' only the varying syllable is written in NFD and the span is recomputed."""
    rng = random.Random(seed)
    attested = sorted({s for s in inv.structures if inv.is_legal(s, "attested")}, key=str)
    by_segments: dict[tuple, list[Syllable]] = {}
    for s in attested:
        by_segments.setdefault((s.onset, s.glide, s.nucleus, s.coda), []).append(s)
    candidates = [v for v in by_segments.values() if len(v) >= 2]
    rng.shuffle(candidates)
    pairs: list[PatchPair] = []
    for group in candidates:
        if len(pairs) >= n:
            break
        a, b = rng.sample(group, 2)
        partner = rng.choice(attested)
        if partner.tone in (a.tone, b.tone) or (partner.onset, partner.glide, partner.nucleus, partner.coda) == (a.onset, a.glide, a.nucleus, a.coda):
            continue
        # V3 outputs must be legal for both
        outs = []
        ok = True
        for t in (a, b):
            pair = (t, partner) if target_first else (partner, t)
            o = V.apply("V3", *pair)
            if not all(inv.is_legal(s, "onset_rime") for s in o):
                ok = False
                break
            outs.append(f"{spell(o[0])} {spell(o[1])}")
        if not ok:
            continue
        sa, sb, sp = spell(a), spell(b), spell(partner)
        enc = U.nfd if encoding == "nfd" else U.nfc
        ea, eb = enc(sa), enc(sb)
        if target_first:
            clean, corrupt = prompt.format(a=ea, b=sp), prompt.format(a=eb, b=sp)
            start = prompt.index("{a}")
        else:
            clean, corrupt = prompt.format(a=sp, b=ea), prompt.format(a=sp, b=eb)
            start = prompt.index("{a}") + len(sp) + 1
        pairs.append(PatchPair(clean=clean, corrupt=corrupt, target_syllable_clean=sa, target_syllable_corrupt=sb,
                               partner=sp, answer_clean=outs[0], answer_corrupt=outs[1], target_char_span=(start, start + len(ea))))
    return pairs


def rerender_pair(pair: PatchPair, render_input: Callable[[str], str]) -> PatchPair:
    """The same pair inside a full prompt: `render_input(item_input)` returns the complete
    prompt text for the item whose input is `partner target` (target second, design 9.3),
    e.g. E1's V3 prompt with demonstrations and the chat template. The target span is
    recomputed on the rendered text (the item input is its LAST occurrence: the
    demonstrations precede it). The rendered text must contain the input verbatim."""
    inp_c = f"{pair.partner} {pair.clean[pair.target_char_span[0]:pair.target_char_span[1]]}"
    inp_k = f"{pair.partner} {pair.corrupt[pair.target_char_span[0]:pair.target_char_span[1]]}"
    clean, corrupt = render_input(inp_c), render_input(inp_k)
    at = clean.rfind(inp_c)
    if at < 0 or corrupt.rfind(inp_k) < 0:
        raise ValueError("the rendered prompt must contain the item input verbatim")
    start = at + len(pair.partner) + 1
    tgt_len = pair.target_char_span[1] - pair.target_char_span[0]
    return PatchPair(clean=clean, corrupt=corrupt, target_syllable_clean=pair.target_syllable_clean,
                     target_syllable_corrupt=pair.target_syllable_corrupt, partner=pair.partner,
                     answer_clean=pair.answer_clean, answer_corrupt=pair.answer_corrupt,
                     target_char_span=(start, start + tgt_len))


def positions_for_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> list[int]:
    """Token indices whose span overlaps [start, end) (a space-only token is excluded)."""
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]


def align_pair(tokenizer, pair: PatchPair, add_special_tokens: bool = True, answer_prefix: str = " ") -> dict | None:
    """Tokenize both prompts; keep the pair only if the token sequences have equal length and
    differ only inside the target span, and the two answers diverge. Answers are tokenized
    with a leading space (the model emits ' bị', one word-initial piece such as '▁bị', after
    the arrow), so the readout token is never a bare 'b' + 'ị'. Also reports whether the
    readout is tone-only: the two readout pieces are identical once tone marks are stripped
    (design 9.3 filter 2). Returns ids, differing positions, target positions, answer token
    ids, the forced shared prefix length, the readout pieces and the tone-only flag."""
    ec = tokenizer(pair.clean, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    ek = tokenizer(pair.corrupt, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    if len(ec["input_ids"]) != len(ek["input_ids"]):
        return None
    diff = [i for i, (x, y) in enumerate(zip(ec["input_ids"], ek["input_ids"])) if x != y]
    s, e = pair.target_char_span
    target_pos = [i for i, (a, b) in enumerate(ec["offset_mapping"]) if b > s and a < e and b > a]
    if not diff or any(d not in target_pos for d in diff):
        return None
    ans_c = tokenizer(answer_prefix + pair.answer_clean, add_special_tokens=False)["input_ids"]
    ans_k = tokenizer(answer_prefix + pair.answer_corrupt, add_special_tokens=False)["input_ids"]
    # the two answers may share a prefix (e.g. a character tokenizer: 'b' before 'ị'/'í');
    # teacher-force the shared prefix and read the logit difference at the FIRST DIVERGING token
    k = 0
    while k < min(len(ans_c), len(ans_k)) and ans_c[k] == ans_k[k]:
        k += 1
    if k >= len(ans_c) or k >= len(ans_k):
        return None
    prefix = ans_c[:k]
    clean_ids = list(ec["input_ids"]) + prefix
    corrupt_ids = list(ek["input_ids"]) + prefix
    piece_c = tokenizer.decode([ans_c[k]])
    piece_k = tokenizer.decode([ans_k[k]])
    tone_only = U.strip_tones(piece_c) == U.strip_tones(piece_k) and U.count_tone_marks(piece_c) <= 1 and U.count_tone_marks(piece_k) <= 1
    # position groups of design 9.3 that follow from the prompt's structure: the fixed
    # (partner) syllable G2, the context before the item G3, marker / suffix after the item G4
    partner_start = s - len(pair.partner) - 1
    partner_pos = positions_for_span(ec["offset_mapping"], partner_start, s - 1) if partner_start >= 0 else []
    item_start = min(partner_pos + target_pos) if partner_pos else target_pos[0]
    n_prompt = len(ec["input_ids"])
    context_pos = [i for i in range(item_start) if i not in partner_pos]
    suffix_pos = [i for i in range(max(target_pos) + 1, n_prompt + k - 1)]     # marker, newline, forced prefix; not the answer position
    return {"clean_ids": clean_ids, "corrupt_ids": corrupt_ids, "diff_positions": diff, "target_positions": target_pos,
            "answer_pos": len(clean_ids) - 1, "tok_clean": ans_c[k], "tok_corrupt": ans_k[k], "n_forced_prefix": k,
            "readout_pieces": [piece_c, piece_k], "readout_tone_only": tone_only,
            "partner_positions": partner_pos, "context_positions": context_pos, "suffix_positions": suffix_pos,
            "offsets": [tuple(o) for o in ec["offset_mapping"]]}
