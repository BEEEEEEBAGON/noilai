"""Activation patching and difference-in-means steering on the residual stream.

Minimal pairs differ in one syllable's tone (clean: 'bí mật', corrupt: 'bị mật', both
legal). We cache the residual stream of the clean run at every decoder layer, then run
the corrupt input while overwriting the residual at layer ℓ and the syllable's token
positions with the clean values, and measure how much of the clean answer is recovered:

    recovery(ℓ, positions) = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt)

where LD is the logit difference between the clean and corrupt answer tokens at the
answer position. A layer/position where recovery jumps to ~1 is where the tone
information that the answer depends on is carried. Steering adds α·(μ_a − μ_b), the
difference of mean residuals between two tone classes at a layer, at the syllable
positions and measures how often the answer flips.

Works with any Hugging Face decoder-only model whose decoder blocks live in an
nn.ModuleList (found automatically); tested on a tiny random LLaMA.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def get_decoder_layers(model):
    """Find the ModuleList of decoder blocks (model.model.layers for LLaMA/Gemma/Qwen;
    language_model.model.layers for multimodal wrappers)."""
    import torch.nn as nn

    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2 and len({type(m) for m in mod}) == 1:
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise ValueError("no decoder layer list found")
    return best


def _output_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _replace_output(out, new):
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    if isinstance(out, list):
        return [new] + list(out[1:])
    return new


class ResidualCache:
    """Captures each decoder layer's output hidden states (the residual stream after the block)."""

    def __init__(self, model):
        self.layers = get_decoder_layers(model)
        self.store: dict[int, "object"] = {}
        self._handles = []

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        return self

    def _make_hook(self, i):
        def hook(_mod, _inp, out):
            self.store[i] = _output_tensor(out).detach().clone()
        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


@contextmanager
def patch_layer(model, layer_idx: int, positions: Sequence[int], values, batch_index: int = 0):
    """Overwrite hidden[:, positions, :] at layer `layer_idx`'s output with `values`
    (tensor [len(positions), d] or [B, len(positions), d])."""
    import torch

    layers = get_decoder_layers(model)
    pos = torch.as_tensor(list(positions), dtype=torch.long)

    def hook(_mod, _inp, out):
        h = _output_tensor(out)
        h = h.clone()
        v = values.to(h.device, h.dtype)
        if v.dim() == 2:
            h[:, pos, :] = v.unsqueeze(0)
        else:
            h[:, pos, :] = v
        return _replace_output(out, h)

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def add_direction(model, layer_idx: int, positions: Optional[Sequence[int]], direction, alpha: float):
    """Add alpha * direction to the residual at layer_idx (at `positions`, or everywhere if None)."""
    import torch

    layers = get_decoder_layers(model)

    def hook(_mod, _inp, out):
        h = _output_tensor(out).clone()
        d = torch.as_tensor(direction).to(h.device, h.dtype)
        if positions is None:
            h = h + alpha * d
        else:
            pos = torch.as_tensor(list(positions), dtype=torch.long)
            h[:, pos, :] = h[:, pos, :] + alpha * d
        return _replace_output(out, h)

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def logit_diff(logits, position: int, tok_a: int, tok_b: int, batch_index: int = 0) -> float:
    return float(logits[batch_index, position, tok_a] - logits[batch_index, position, tok_b])


@dataclass
class PatchResult:
    clean_ld: float
    corrupt_ld: float
    recovery: np.ndarray        # [n_layers, n_position_groups]
    position_groups: list[list[int]]
    layers: list[int]


def run_patching(model, clean_ids, corrupt_ids, answer_pos: int, tok_clean: int, tok_corrupt: int,
                 position_groups: Sequence[Sequence[int]], layers: Optional[Sequence[int]] = None) -> PatchResult:
    """clean_ids / corrupt_ids: LongTensor [1, T] of equal length (same tokenization outside
    the patched positions). position_groups: lists of token positions to patch together."""
    import torch

    model.eval()
    with torch.no_grad():
        with ResidualCache(model) as cache:
            clean_logits = model(input_ids=clean_ids).logits
        corrupt_logits = model(input_ids=corrupt_ids).logits
        ld_clean = logit_diff(clean_logits, answer_pos, tok_clean, tok_corrupt)
        ld_corr = logit_diff(corrupt_logits, answer_pos, tok_clean, tok_corrupt)
        n_layers = len(cache.layers)
        layers = list(layers) if layers is not None else list(range(n_layers))
        rec = np.zeros((len(layers), len(position_groups)))
        denom = ld_clean - ld_corr
        for li, L in enumerate(layers):
            for gi, pos in enumerate(position_groups):
                vals = cache.store[L][0, list(pos), :]
                with patch_layer(model, L, pos, vals):
                    logits = model(input_ids=corrupt_ids).logits
                ld = logit_diff(logits, answer_pos, tok_clean, tok_corrupt)
                rec[li, gi] = (ld - ld_corr) / denom if abs(denom) > 1e-8 else np.nan
    return PatchResult(ld_clean, ld_corr, rec, [list(p) for p in position_groups], layers)


def difference_in_means(H_a: np.ndarray, H_b: np.ndarray) -> np.ndarray:
    """Unit direction μ_a − μ_b from residuals [n, d] of two classes."""
    d = H_a.mean(0) - H_b.mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def steering_flip_rate(model, input_ids_list: Sequence, answer_pos_list: Sequence[int], tok_a: int, tok_b: int,
                       layer_idx: int, direction: np.ndarray, alpha: float, positions_list: Optional[Sequence[Sequence[int]]] = None) -> dict:
    """Share of inputs whose preferred answer (a vs b) flips under steering, plus mean LD shift."""
    import torch

    model.eval()
    flips = 0
    shift = []
    with torch.no_grad():
        for k, ids in enumerate(input_ids_list):
            base = model(input_ids=ids).logits
            ld0 = logit_diff(base, answer_pos_list[k], tok_a, tok_b)
            pos = positions_list[k] if positions_list is not None else None
            with add_direction(model, layer_idx, pos, direction, alpha):
                steered = model(input_ids=ids).logits
            ld1 = logit_diff(steered, answer_pos_list[k], tok_a, tok_b)
            flips += (ld0 > 0) != (ld1 > 0)
            shift.append(ld1 - ld0)
    return {"flip_rate": flips / len(input_ids_list), "mean_ld_shift": float(np.mean(shift)), "n": len(input_ids_list)}
