"""Hidden-state extraction at syllable positions for the E4 probes (design 9.1–9.2).

Given a Hugging Face causal LM and its (fast) tokenizer, place each target syllable in
a carrier sentence, locate the tokens that cover the syllable from the tokenizer's offset
mapping, run the model ONCE with the same forward hooks activation patching uses
(`patching.ResidualCache`: the residual stream AFTER each decoder block, pre-final-norm —
`output_hidden_states[-1]` is the post-norm vector in current transformers and is not
used) and keep the residual stream at every hidden index at the probed positions:
  first   the first token of the syllable's span (onset fragment of a split syllable)
  last    the LAST sub-token of the syllable (where a model that spells syllable by
          syllable must have assembled it)
  mark    the bare tone-mark token under NFD when the span has one (NaN row otherwise)
  after   the FIRST token after the syllable (the same string under NFC and NFD, the one
          position where the two encodings are compared on equal terms)
Hidden index 0 is the model's input embedding (Gemma's embedding module applies its
sqrt(d) scale itself); hidden index b + HIDDEN_INDEX_OFFSET is the output of decoder block b
— the one place this convention is pinned (design 9.2).

Under NFD the syllable's combining marks are separate code points and often separate
tokens, so the span is computed on the exact string that was tokenized.
Everything returns numpy arrays so that the probing code has no torch dependency.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..vi import unicode as U
from ..vi.syllable import try_parse

HIDDEN_INDEX_OFFSET = 1          # hidden index = decoder block index + 1; index 0 = embeddings (design 9.2)
POSITIONS = ("first", "last", "mark", "after")

# Carrier sentences: the slot {} receives the syllable (design 9.2: the slot is never
# sentence-initial, the token after the slot is one of the fixed tone-neutral words
# và / lên / trong / này, and one carrier quotes the slot). Neutral frames so that the
# syllable's tone is not predictable from context.   [NATIVE-CHECK]
CARRIERS = (
    "Từ tiếp theo là {} và chỉ có vậy.",
    "Người ta viết chữ {} lên bảng.",
    "Tôi vừa đọc thấy chữ {} trong sách.",
    "Âm tiết {} này đứng ở đầu câu.",          # [NATIVE-CHECK] replaces "... {} xuất hiện ..." (toned word after the slot)
    "Hãy nhìn vào chữ {} này.",
    "Người ta viết chữ «{}» lên bảng.",        # [NATIVE-CHECK] the quoted-slot carrier of design 9.2
)
TONE_NEUTRAL_AFTER_WORDS = ("và", "lên", "trong", "này")


def carrier_after_word(template: str) -> str:
    """The word (or quote mark) immediately after the slot."""
    suffix = template.split("{}")[1].strip()
    return suffix.split()[0].strip(".,;:") if suffix else ""


@dataclass
class ProbeExample:
    syllable: str          # NFC surface
    carrier_id: int
    text: str              # the full carrier text as tokenized (NFC or NFD)
    encoding: str          # 'nfc' | 'nfd'
    char_start: int
    char_end: int
    labels: dict = field(default_factory=dict)   # tone, onset, rime, nucleus, coda, glide, coda_class (+ extra)


def make_examples(syllables: Iterable[str], carriers: Sequence[str] = CARRIERS, encoding: str = "nfc",
                  carrier_ids: Sequence[int] | None = None, reencode_carrier: bool = False,
                  extra_labels: Mapping[str, Mapping] | None = None) -> list[ProbeExample]:
    """By default only the TARGET syllable is re-encoded (design 9.2) and the carrier stays
    NFC, so that the position after the slot is the same string under both encodings;
    `reencode_carrier=True` re-encodes the whole sentence (what the E3 C1 arm does).
    `extra_labels` adds per-syllable covariates (e.g. {'ma': {'attested': True}})."""
    out = []
    for syl in syllables:
        p = try_parse(syl, strict=False)
        if p is None:
            continue
        s = p.syllable
        labels = {"tone": s.tone, "onset": s.onset, "rime": s.rime, "nucleus": s.nucleus, "coda": s.coda, "glide": int(s.glide),
                  "coda_class": "stop" if s.coda in ("p", "t", "c", "ch") else ("nasal" if s.coda in ("m", "n", "ng", "nh") else "open")}
        if extra_labels and U.nfc(syl) in extra_labels:
            labels.update(extra_labels[U.nfc(syl)])
        ids = carrier_ids if carrier_ids is not None else range(len(carriers))
        for cid in ids:
            surf = U.nfc(syl) if encoding == "nfc" else U.nfd(syl)
            template = carriers[cid]
            prefix, suffix = template.split("{}")
            if reencode_carrier and encoding == "nfd":
                prefix, suffix = U.nfd(prefix), U.nfd(suffix)
            else:
                prefix, suffix = U.nfc(prefix), U.nfc(suffix)
            text = prefix + surf + suffix
            out.append(ProbeExample(syllable=U.nfc(syl), carrier_id=cid, text=text, encoding=encoding,
                                    char_start=len(prefix), char_end=len(prefix) + len(surf), labels=labels))
    return out


def token_span(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Indices [first, last] of the tokens overlapping [char_start, char_end). A token that
    only carries the preceding space is excluded. Raises if no token overlaps."""
    idx = [i for i, (s, e) in enumerate(offsets) if e > char_start and s < char_end and e > s]
    if not idx:
        raise ValueError("no token overlaps the syllable span")
    return idx[0], idx[-1]


def mark_token_index(text: str, offsets: Sequence[tuple[int, int]], first: int, last: int) -> int | None:
    """Within the span's tokens, the index of the token whose text (space markers removed)
    is non-empty and consists only of combining tone marks (the NFD bare-mark token, design
    9.1 position `mark`); None when there is none (NFC, or a tokenizer that keeps the mark
    with its base letter)."""
    for i in range(first, last + 1):
        s, e = offsets[i]
        piece = text[s:e].replace("▁", "").replace(" ", "")
        if piece and all(ch in U.TONE_MARK_SET for ch in piece):
            return i
    return None


def extract_hidden_states(model, tokenizer, examples: Sequence[ProbeExample], batch_size: int = 16,
                          positions: Sequence[str] = ("last", "after"), device: str | None = None,
                          add_special_tokens: bool = True, return_token_ids: bool = False):
    """Return {position: array [n_examples, n_layers+1, hidden]} (index 0 = embeddings; hidden
    index b + HIDDEN_INDEX_OFFSET = output of decoder block b, captured with the patching
    hooks). A `mark` position row is NaN when the span has no bare tone-mark token. With
    return_token_ids, also return the list of token-id spans covering each example's
    syllable (for the structural baseline)."""
    import torch

    from .patching import ResidualCache

    for p in positions:
        if p not in POSITIONS:
            raise ValueError(f"unknown position {p!r}; one of {POSITIONS}")
    device = device or next(model.parameters()).device
    model.eval()
    out = {p: [] for p in positions}
    spans: list[list[int]] = []
    embed = model.get_input_embeddings()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            enc = tokenizer([e.text for e in batch], return_offsets_mapping=True, return_tensors="pt", padding=True,
                            add_special_tokens=add_special_tokens)
            offsets = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with ResidualCache(model) as cache:
                model(**enc, use_cache=False)
            blocks = [cache.store[b] for b in range(len(cache.layers))]
            hs = torch.stack([embed(enc["input_ids"])] + blocks, dim=1)      # [B, L+1, T, d]
            attn = enc["attention_mask"]
            for b, ex in enumerate(batch):
                # padding side: find valid token indices
                valid = attn[b].nonzero().flatten().tolist()
                offs = [tuple(offsets[b][t]) for t in valid]
                first, last = token_span(offs, ex.char_start, ex.char_end)
                spans.append([int(enc["input_ids"][b, t]) for t in valid[first:last + 1]])
                for p in positions:
                    if p == "last":
                        t = valid[last]
                    elif p == "first":
                        t = valid[first]
                    elif p == "after":
                        t = valid[min(last + 1, len(valid) - 1)]
                    else:   # mark
                        m = mark_token_index(ex.text, offs, first, last)
                        if m is None:
                            out[p].append(np.full((hs.shape[1], hs.shape[3]), np.nan, dtype=np.float32))
                            continue
                        t = valid[m]
                    out[p].append(hs[b, :, t, :].float().cpu().numpy())
    arrays = {p: np.stack(v) for p, v in out.items()}
    return (arrays, spans) if return_token_ids else arrays


def syllable_token_count(tokenizer, example: ProbeExample, add_special_tokens: bool = True) -> int:
    enc = tokenizer(example.text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    first, last = token_span([tuple(o) for o in enc["offset_mapping"]], example.char_start, example.char_end)
    return last - first + 1
