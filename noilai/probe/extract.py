"""Hidden-state extraction at syllable positions for the E4 probes.

Given a Hugging Face causal LM and its (fast) tokenizer, place each target syllable in
a carrier sentence, locate the tokens that cover the syllable from the tokenizer's offset
mapping, run the model with output_hidden_states=True and keep the residual stream at
every layer at (a) the LAST sub-token of the syllable and (b) the FIRST token after it.
Position (a) is where a model that spells syllable by syllable must have assembled the
syllable; position (b) is where the syllable has certainly been read. Under NFD the
syllable's combining marks are separate code points and often separate tokens, so the
span is computed on the exact string that was tokenized.

Everything returns numpy arrays so that the probing code has no torch dependency.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..vi import unicode as U
from ..vi.syllable import try_parse

# Carrier sentences: the slot {} receives the syllable. Neutral frames so that the
# syllable's tone is not predictable from context.   # NATIVE-CHECK
CARRIERS = (
    "Từ tiếp theo là {} và chỉ có vậy.",
    "Người ta viết chữ {} lên bảng.",
    "Tôi vừa đọc thấy chữ {} trong sách.",
    "Âm tiết {} xuất hiện ở đầu câu.",
    "Hãy nhìn vào chữ {} này.",
)


@dataclass
class ProbeExample:
    syllable: str          # NFC surface
    carrier_id: int
    text: str              # the full carrier text as tokenized (NFC or NFD)
    encoding: str          # 'nfc' | 'nfd'
    char_start: int
    char_end: int
    labels: dict = field(default_factory=dict)   # tone, onset, rime, nucleus, coda, glide


def make_examples(syllables: Iterable[str], carriers: Sequence[str] = CARRIERS, encoding: str = "nfc",
                  carrier_ids: Sequence[int] | None = None) -> list[ProbeExample]:
    out = []
    for syl in syllables:
        p = try_parse(syl, strict=False)
        if p is None:
            continue
        s = p.syllable
        labels = {"tone": s.tone, "onset": s.onset, "rime": s.rime, "nucleus": s.nucleus, "coda": s.coda, "glide": int(s.glide)}
        ids = carrier_ids if carrier_ids is not None else range(len(carriers))
        for cid in ids:
            surf = U.nfc(syl) if encoding == "nfc" else U.nfd(syl)
            template = carriers[cid]
            prefix = template.split("{}")[0]
            prefix = U.nfc(prefix) if encoding == "nfc" else U.nfd(prefix)
            suffix = template.split("{}")[1]
            suffix = U.nfc(suffix) if encoding == "nfc" else U.nfd(suffix)
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


def extract_hidden_states(model, tokenizer, examples: Sequence[ProbeExample], batch_size: int = 16,
                          positions: Sequence[str] = ("last", "after"), device: str | None = None,
                          add_special_tokens: bool = True) -> dict[str, np.ndarray]:
    """Return {position: array [n_examples, n_layers+1, hidden]} (index 0 = embeddings)."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    out = {p: [] for p in positions}
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            enc = tokenizer([e.text for e in batch], return_offsets_mapping=True, return_tensors="pt", padding=True,
                            add_special_tokens=add_special_tokens)
            offsets = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            res = model(**enc, output_hidden_states=True)
            hs = torch.stack(res.hidden_states, dim=1)      # [B, L+1, T, d]
            attn = enc["attention_mask"]
            for b, ex in enumerate(batch):
                # padding side: find valid token indices
                valid = attn[b].nonzero().flatten().tolist()
                offs = [tuple(offsets[b][t]) for t in valid]
                first, last = token_span(offs, ex.char_start, ex.char_end)
                for p in positions:
                    if p == "last":
                        t = valid[last]
                    elif p == "first":
                        t = valid[first]
                    elif p == "after":
                        t = valid[min(last + 1, len(valid) - 1)]
                    else:
                        raise ValueError(p)
                    out[p].append(hs[b, :, t, :].float().cpu().numpy())
    return {p: np.stack(v) for p, v in out.items()}


def syllable_token_count(tokenizer, example: ProbeExample, add_special_tokens: bool = True) -> int:
    enc = tokenizer(example.text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    first, last = token_span([tuple(o) for o in enc["offset_mapping"]], example.char_start, example.char_end)
    return last - first + 1
