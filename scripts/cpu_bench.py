#!/usr/bin/env python
"""CPU cost of the pilot's operations for a Gemma-3-1B-SIZED model (random weights, the 1B's published dimensions):
generation (prompt ~520 tokens NFC / ~950 NFD, 24 new tokens, batch 1 and 8) and the shared-prefix log-probabilities
(one prompt pass, then a 3-token continuation on a copy of its cache).

Random weights cost the same arithmetic as trained ones, so the times are a fair proxy for gemma-3-1b-it on the same
CPUs (the build machine: 4 cores); Kaggle CPU sessions have other cores [UNCERTAIN: verify], so docs/COMPUTE_PLAN.md
scales these numbers with a stated factor. Medians of 3 repeats (5 for the continuation; 1 for generation).

    python scripts/cpu_bench.py [out.json]
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "audit" / "cpu_bench_gemma3_1b_shape.json"


def timed(f, n: int = 3) -> float:
    xs = []
    for _ in range(n):
        s = time.time()
        f()
        xs.append(time.time() - s)
    return round(statistics.median(xs), 2)


def main(argv=None) -> int:
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else DEFAULT_OUT
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    cfg = Gemma3TextConfig(vocab_size=262144, hidden_size=1152, intermediate_size=6912, num_hidden_layers=26,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=256, max_position_embeddings=32768,
                           sliding_window=512)
    m = Gemma3ForCausalLM(cfg).eval()
    res = {"model": "Gemma3ForCausalLM with gemma-3-1b-it's dimensions, random weights, float32", "cpus": os.cpu_count(),
           "torch": torch.__version__, "load_avg_start": os.getloadavg()}
    for n_prompt in (520, 950):
        ids = torch.randint(10, 200000, (1, n_prompt))
        with torch.no_grad():
            m(input_ids=ids, use_cache=True, logits_to_keep=1)
            res[f"prefill_{n_prompt}_s"] = timed(lambda ids=ids: m(input_ids=ids, use_cache=True, logits_to_keep=1))
            po = m(input_ids=ids, use_cache=True, logits_to_keep=1)
            c = torch.randint(10, 200000, (1, 3))
            res[f"continuation_3tok_on_cache_{n_prompt}_s"] = timed(
                lambda po=po, c=c: m(input_ids=c, past_key_values=copy.deepcopy(po.past_key_values), use_cache=True), 5)
        for bs in (1, 8):
            ids_b = torch.randint(10, 200000, (bs, n_prompt))
            with torch.no_grad():
                res[f"generate_bs{bs}_{n_prompt}p_24new_s"] = timed(
                    lambda ids_b=ids_b: m.generate(input_ids=ids_b, attention_mask=torch.ones_like(ids_b), max_new_tokens=24,
                                                   min_new_tokens=24, do_sample=False), 1)
        print(json.dumps(res), flush=True)
    res["load_avg_end"] = os.getloadavg()
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
