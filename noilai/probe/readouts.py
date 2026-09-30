"""The two patching readouts of design 9.3 (items 19, 25, 62).

A (perception): "Chữ «bí» mang thanh gì? Đáp án: thanh" → the next token; the answers are
the TONE NAMES (never digits: Vietnamese has at least three conflicting tone numberings
and after "thanh " a digit is two Gemma 3 pieces), and the logit difference is read on the
FIRST piece of each name. On `data/external/gemma3_tokenizer.model` the first pieces of the
six names after a space are single and distinct (`▁ngang ▁huyền ▁sắc ▁hỏi ▁nặng` are whole
pieces and `▁ngã` = `▁ng` + `ã`, so ngã is read at `▁ng`); that fact is asserted at import
time when the model file is present (`check_gemma3_answer_pieces`). The frame stays NFC.

B (manipulation): E1's exact V3 prompt (noilai.eval.prompts.render, p0, three
demonstrations, baseline arm) with the IT chat template, the answer teacher-forced from
the model turn with a leading space (`▁bị`, not `b` + `ị`); built by `render_readout_b`.

IT vs PT handling: an instruction-tuned tokenizer carries a chat template and the prompt is
its user turn followed by the generation prompt; a pretrained checkpoint gets the raw text.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ..vi import unicode as U
from ..vi.syllable import try_parse
from .pairs import PatchPair

ROOT = Path(__file__).resolve().parents[2]
GEMMA3_MODEL = ROOT / "data" / "external" / "gemma3_tokenizer.model"

# [NATIVE-CHECK] "What tone does «{}» carry? Answer: tone" — design 9.3 readout A frame (high, NV)
READOUT_A_PROMPT = "Chữ «{}» mang thanh gì? Đáp án: thanh"
TONE_ANSWERS = U.TONE_NAMES_VI          # ("ngang", "huyền", "sắc", "hỏi", "ngã", "nặng"), index = tone number
ANSWER_PREFIX = " "                     # the answer follows "thanh" / the model turn with a leading space


def readout_a_prompt(syllable: str, encoding: str = "nfc") -> str:
    """The readout-A frame around the syllable (only the syllable is re-encoded; frame NFC)."""
    surf = U.nfd(syllable) if encoding == "nfd" else U.nfc(syllable)
    return READOUT_A_PROMPT.format(surf)


def first_answer_pieces(encode: Callable[[str], list[int]]) -> list[int]:
    """First token id of each tone name written after a space, in tone order 0..5."""
    return [encode(ANSWER_PREFIX + name)[0] for name in TONE_ANSWERS]


def check_answer_pieces(encode: Callable[[str], list[int]], decode: Callable[[list[int]], str] | None = None) -> dict:
    """Design 9.3: the six first pieces must be distinct (they are what LD is read on).
    Returns the ids and the decoded pieces; raises AssertionError when two names share a
    first piece (then readout A cannot separate those tones on this tokenizer)."""
    ids = first_answer_pieces(encode)
    pieces = [decode([i]) for i in ids] if decode is not None else [None] * len(ids)
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        clash = [n for n, i in zip(TONE_ANSWERS, ids) if i in dup]
        raise AssertionError(f"readout A: tone names {clash} share a first piece on this tokenizer; the LD readout "
                             "cannot separate them (design 9.3)")
    return {"ids": ids, "pieces": pieces, "distinct": True}


def check_gemma3_answer_pieces(model_path: Path = GEMMA3_MODEL) -> dict | None:
    """Import-time assertion on the pinned Gemma 3 SentencePiece file (skipped when absent or
    when NOILAI_SKIP_READOUT_CHECK is set)."""
    if os.environ.get("NOILAI_SKIP_READOUT_CHECK") or not model_path.exists():
        return None
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    info = check_answer_pieces(sp.encode, lambda ids: sp.id_to_piece(ids[0]))
    # the whole-piece names and ngã's first piece, as design 9.3 records them
    single = {name: len(sp.encode(ANSWER_PREFIX + name)) == 1 for name in TONE_ANSWERS}
    info["single_piece"] = single
    assert info["pieces"][4] == "▁ng" and not single["ngã"], "ngã is read at ▁ng on Gemma 3 (design 9.3)"
    return info


GEMMA3_ANSWER_PIECES = check_gemma3_answer_pieces()


def is_instruction_tuned(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def wrap_chat(tokenizer, user: str | list[dict]) -> str:
    """IT checkpoint: the message(s) through the tokenizer's chat template plus the generation
    prompt; PT checkpoint: the raw user text. `<bos>` is added by the tokenizer at encoding
    time, never pasted (design 9.3), so the template's own BOS is removed when it emits one."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else list(user)
    if not is_instruction_tuned(tokenizer):
        return "\n\n".join(m["content"] for m in messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def readout_a_pair(tokenizer, pair: PatchPair, encoding: str = "nfc", add_special_tokens: bool = True) -> dict | None:
    """Readout-A version of a tone-minimal pair: the two syllables in the readout-A frame, the
    answer tokens = first pieces of the two tone names. Same alignment rule as
    `pairs.align_pair` (equal length, differences inside the syllable's span); None when the
    tokenizer cannot separate the two names or the prompts do not align."""
    pc = try_parse(pair.target_syllable_clean, strict=False)
    pk = try_parse(pair.target_syllable_corrupt, strict=False)
    if pc is None or pk is None or pc.syllable.tone == pk.syllable.tone:
        return None
    text_c = wrap_chat(tokenizer, readout_a_prompt(pair.target_syllable_clean, encoding))
    text_k = wrap_chat(tokenizer, readout_a_prompt(pair.target_syllable_corrupt, encoding))
    ec = tokenizer(text_c, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    ek = tokenizer(text_k, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    if len(ec["input_ids"]) != len(ek["input_ids"]):
        return None
    diff = [i for i, (x, y) in enumerate(zip(ec["input_ids"], ek["input_ids"])) if x != y]
    surf_c = U.nfd(pair.target_syllable_clean) if encoding == "nfd" else U.nfc(pair.target_syllable_clean)
    at = text_c.find("«") + 1
    target = [i for i, (a, b) in enumerate(ec["offset_mapping"]) if b > at and a < at + len(surf_c) and b > a]
    if not diff or any(d not in target for d in diff):
        return None
    enc = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]
    ids = first_answer_pieces(enc)
    tok_c, tok_k = ids[pc.syllable.tone], ids[pk.syllable.tone]
    if tok_c == tok_k:
        return None
    return {"clean_ids": list(ec["input_ids"]), "corrupt_ids": list(ek["input_ids"]), "diff_positions": diff,
            "target_positions": target, "answer_pos": len(ec["input_ids"]) - 1, "tok_clean": tok_c, "tok_corrupt": tok_k,
            "readout": "A", "tone_clean": pc.syllable.tone, "tone_corrupt": pk.syllable.tone,
            "answer_names": [TONE_ANSWERS[pc.syllable.tone], TONE_ANSWERS[pk.syllable.tone]]}


def render_readout_b(tokenizer, item_input: str, variant: str = "V3", paraphrase: str = "p0", shots: int = 3,
                     arm: str = "nfc") -> str:
    """E1's exact V3 prompt for a two-syllable input (readout B, design 9.3): render through
    noilai.eval.prompts with the baseline arm and wrap it in the chat template for an IT
    checkpoint. The item carries no canary (it is a pseudo-item built from the pair)."""
    from ..eval import prompts as PR

    item = {"item_id": f"patch-{item_input.replace(' ', '_')}", "task": "T1", "variant": variant, "input": item_input,
            "gold": [], "canary": None}
    messages = PR.render(item, "T1", variant, paraphrase=paraphrase, shots=shots, arm=arm)
    return wrap_chat(tokenizer, messages)


def readout_b_answer_prefix() -> str:
    """What the model turn starts with before the transformed phrase: the answer marker of
    E1's prompt plus the leading space of the phrase's first piece (`Đáp án: ▁cổng`), so the
    forced prefix covers the marker and LD is read on the first diverging phrase token."""
    from ..eval import prompts as PR

    return PR.answer_marker() + ANSWER_PREFIX
