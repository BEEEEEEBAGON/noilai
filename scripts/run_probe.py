#!/usr/bin/env python
"""E4 driver: layer-wise probes, activation patching and steering on one model.

    python scripts/run_probe.py --model google/gemma-3-1b-it --dtype float32 --out data/runs/probe_gemma3_1b \
        --n-syllables 1500 --n-pairs 200 --encodings nfc nfd

Steps
 1. sample attested syllables (stratified by tone), build carrier examples in NFC and NFD;
 2. extract residual-stream states at the syllable's last sub-token and the token after it;
 3. layer-wise logistic probes for tone, onset, rime, coda with syllable-disjoint splits and
    control tasks (3 seeds); write probes.csv;
 4. activation patching over minimal tone pairs in the V3 prompt (layer × position group:
    target syllable, partner syllable, instruction, last token); write patching.npz/json;
 5. difference-in-means steering at the best probe layer; write steering.json;
 6. manifest.json with model revision, dtype, device, timings.
Runs on CPU for tiny models (tests) and on a T4 for Gemma 3 1B/4B (float32 or bfloat16;
never float16, which overflows in Gemma 3).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.probe import extract, patching, probes
from noilai.probe import pairs as P
from noilai.vi import lexicon as L
from noilai.vi.syllable import spell


def sample_syllables(inv, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_tone = defaultdict(list)
    for s in inv.structures:
        if inv.is_legal(s, "attested"):
            by_tone[s.tone].append(spell(s))
    out = []
    per = n // 6
    for t in range(6):
        pool = sorted(by_tone[t])
        rng.shuffle(pool)
        out.extend(pool[:per])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-syllables", type=int, default=1200)
    ap.add_argument("--n-pairs", type=int, default=150)
    ap.add_argument("--encodings", nargs="*", default=["nfc", "nfd"])
    ap.add_argument("--carriers", type=int, default=2)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-patching", action="store_true")
    ap.add_argument("--tiny-test", action="store_true", help="use a random tiny model (tests only)")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.tiny_test:
        from tests.test_probe import VOCAB_CHARS  # noqa
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2}
        for ch in dict.fromkeys(VOCAB_CHARS + list("→(),:")):
            vocab.setdefault(ch, len(vocab))
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>")
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)).eval()
        add_special = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
        add_special = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inv = L.load_inventory()
    syllables = sample_syllables(inv, args.n_syllables, seed=0)
    probe_rows = []
    best_layer = {}
    H_by_enc = {}
    for enc in args.encodings:
        exs = extract.make_examples(syllables, encoding=enc, carrier_ids=list(range(args.carriers)))
        H = extract.extract_hidden_states(model, tokenizer, exs, batch_size=args.batch_size, device=device, add_special_tokens=add_special)
        H_by_enc[enc] = (exs, H)
        groups = [e.syllable for e in exs]
        for pos in ("last", "after"):
            for feat in ("tone", "onset", "rime", "coda"):
                labels = [e.labels[feat] for e in exs]
                for seed in args.seeds:
                    for r in probes.run_layer_probes(H[pos], labels, groups, feature=feat, seed=seed):
                        probe_rows.append({"encoding": enc, "position": pos, "seed": seed, **r.__dict__})
        tone_last = [r for r in probe_rows if r["encoding"] == enc and r["position"] == "last" and r["feature"] == "tone"]
        agg = defaultdict(list)
        for r in tone_last:
            agg[r["layer"]].append(r["selectivity"])
        best_layer[enc] = max(agg, key=lambda k: np.mean(agg[k]))
    import pandas as pd
    pd.DataFrame(probe_rows).to_csv(args.out / "probes.csv", index=False)

    patch_summary = None
    if not args.skip_patching:
        cands = P.build_pairs(inv, args.n_pairs * 3, seed=0)
        recs = []
        used = []
        for pr in cands:
            al = P.align_pair(tokenizer, pr, add_special_tokens=add_special)
            if al is None:
                continue
            ids_c = torch.tensor([al["clean_ids"]], device=device)
            ids_k = torch.tensor([al["corrupt_ids"]], device=device)
            T = ids_c.shape[1]
            target = al["target_positions"]
            rest = [i for i in range(T) if i not in target and i != T - 1]
            groups_pos = [target, rest, [T - 1]]
            res = patching.run_patching(model, ids_c, ids_k, al["answer_pos"], al["tok_clean"], al["tok_corrupt"], groups_pos)
            recs.append(res.recovery)
            used.append({"clean": pr.clean, "corrupt": pr.corrupt, "ld_clean": res.clean_ld, "ld_corrupt": res.corrupt_ld})
            if len(recs) >= args.n_pairs:
                break
        if recs:
            R = np.stack(recs)              # [n_pairs, L, 3]
            np.savez(args.out / "patching.npz", recovery=R)
            patch_summary = {"n_pairs": len(recs), "position_groups": ["target", "rest", "last"],
                             "mean_recovery": np.nanmean(R, axis=0).tolist(), "pairs": used[:50]}
            (args.out / "patching.json").write_text(json.dumps(patch_summary, ensure_ascii=False, indent=1))

    steering = None
    if "nfc" in H_by_enc:
        exs, H = H_by_enc["nfc"]
        L_best = int(best_layer["nfc"])
        tones = np.array([e.labels["tone"] for e in exs])
        if (tones == 2).sum() > 5 and (tones == 5).sum() > 5:
            d = patching.difference_in_means(H["last"][tones == 2, L_best + 0], H["last"][tones == 5, L_best + 0])
            steering = {"layer_hidden_index": L_best, "direction_norm": float(np.linalg.norm(d)), "classes": ["sac", "nang"]}
            np.save(args.out / "steer_direction.npy", d)

    manifest = {"model": args.model, "revision": args.revision, "dtype": args.dtype, "device": str(device),
                "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "n_syllables": len(syllables), "encodings": args.encodings, "carriers": args.carriers, "seeds": args.seeds,
                "best_tone_layer_by_encoding": {k: int(v) for k, v in best_layer.items()},
                "patching": patch_summary and {k: v for k, v in patch_summary.items() if k != "pairs"}, "steering": steering,
                "wall_s": time.time() - t0, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "torch": torch.__version__}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
