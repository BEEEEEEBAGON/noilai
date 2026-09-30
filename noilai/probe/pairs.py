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
from dataclasses import dataclass
from typing import Optional, Sequence

from ..gen import variants as V
from ..vi import unicode as U
from ..vi.syllable import Inventory, Syllable, spell

# Minimal instruction so that the answer follows immediately; the model answers with
# the transformed phrase. The prompt text is kept short because every extra token is a
# position the patching sweep must cover.   # NATIVE-CHECK
PATCH_PROMPT = "Nói lái kiểu đổi thanh điệu (giữ phụ âm đầu và vần, đổi chỗ hai thanh): {a} {b} → "


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


def build_pairs(inv: Inventory, n: int, seed: int = 0, target_first: bool = True, prompt: str = PATCH_PROMPT) -> list[PatchPair]:
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
        if target_first:
            clean, corrupt = prompt.format(a=sa, b=sp), prompt.format(a=sb, b=sp)
            start = prompt.index("{a}")
        else:
            clean, corrupt = prompt.format(a=sp, b=sa), prompt.format(a=sp, b=sb)
            start = prompt.index("{a}") + len(sp) + 1
        pairs.append(PatchPair(clean=U.nfc(clean), corrupt=U.nfc(corrupt), target_syllable_clean=sa, target_syllable_corrupt=sb,
                               partner=sp, answer_clean=outs[0], answer_corrupt=outs[1], target_char_span=(start, start + len(sa))))
    return pairs


def align_pair(tokenizer, pair: PatchPair, add_special_tokens: bool = True) -> Optional[dict]:
    """Tokenize both prompts; keep the pair only if the token sequences have equal length and
    differ only inside the target span, and the two answers' first tokens differ.
    Returns ids, differing positions, target positions and the answer token ids."""
    ec = tokenizer(pair.clean, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    ek = tokenizer(pair.corrupt, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    if len(ec["input_ids"]) != len(ek["input_ids"]):
        return None
    diff = [i for i, (x, y) in enumerate(zip(ec["input_ids"], ek["input_ids"])) if x != y]
    s, e = pair.target_char_span
    target_pos = [i for i, (a, b) in enumerate(ec["offset_mapping"]) if b > s and a < e and b > a]
    if not diff or any(d not in target_pos for d in diff):
        return None
    ans_c = tokenizer(pair.answer_clean, add_special_tokens=False)["input_ids"]
    ans_k = tokenizer(pair.answer_corrupt, add_special_tokens=False)["input_ids"]
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
    return {"clean_ids": clean_ids, "corrupt_ids": corrupt_ids, "diff_positions": diff, "target_positions": target_pos,
            "answer_pos": len(clean_ids) - 1, "tok_clean": ans_c[k], "tok_corrupt": ans_k[k], "n_forced_prefix": k}
