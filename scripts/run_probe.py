#!/usr/bin/env python
"""E4 driver: layer-wise probes and activation patching on one model (design 9.1–9.3).

    python scripts/run_probe.py --model google/gemma-3-1b-it --dtype float32 --out data/runs/probe_gemma3_1b \
        --encodings nfc nfd

Steps
 1. sample LEGAL syllables balanced by tone (constants.PROBE_SYLLABLES_PER_TONE per tone,
    legal non-word syllables included, `attested` recorded as a covariate), build carrier
    examples (>= constants.PROBE_MIN_CARRIERS carriers) in NFC and NFD (target re-encoded only);
 2. extract the residual stream at the probed positions through the patching hooks
    (hidden index = block + 1; index 0 = embeddings);
 3. tune the probe's L2 strength ONCE on the dev fold at the middle hidden index, then the
    primary grid: tone × positions × encodings × PROBE_SPLIT_SEEDS split seeds ×
    PROBE_CONTROL_SEEDS control-label seeds (paired), plus the rime-holdout split, plus the
    structural-baseline excess with a syllable-clustered CI per hidden index; write probes.csv,
    excess.csv;
 4. activation patching: tone-minimal pairs (varying syllable second) filtered in order —
    (1) equal token length with differences inside the varying syllable's span on the FULL
    readout-B prompt (E1's V3 prompt with the chat template), (2) tone-only readout,
    (3) LD_clean − LD_corrupt >= PATCHING_MIN_GAP_NATS and greedy clean answer correct —
    retention recorded at every filter; readout B runs only when >= PATCHING_READOUT_B_MIN_RETENTION
    pairs survive (otherwise `readout_b: not_run` and E4 is a localization study); readout A
    (perception; tone names) runs on the pairs that pass its own filters; position groups
    G1 (varying syllable), G1_first / G1_last (its tokens), G2 (fixed syllable), G3 (context
    before the item: instruction + demonstrations), G4 (marker / suffix), G5 (final position),
    G6 (everything from the varying syllable onward); mean recovery with a syllable-clustered
    bootstrap CI; write patching.npz / patching.json;
 5. the difference-in-means tone direction at the best probe hidden index for the future-work
    steering study (design 9.4), with the decoder-block index it maps to;
 6. manifest.json: commit, dirty flag, code hash, resource hashes, seeds, retention, timings.
Runs on CPU for tiny models (tests, `--tiny-test`) and on a T4 for Gemma 3 1B/4B (float32 or
bfloat16; never float16, which overflows in Gemma 3).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai import constants
from noilai.probe import extract, patching, probes, readouts
from noilai.probe import manifest as MF
from noilai.probe import pairs as P
from noilai.stats.bootstrap import cluster_bootstrap
from noilai.vi import lexicon as L
from noilai.vi import unicode as U
from noilai.vi.syllable import Syllable, spell

PRIMARY_FEATURES = ("tone",)
SECONDARY_FEATURES = ("onset", "rime", "coda")
GROUP_NAMES = ("G1_target", "G1_first", "G1_last", "G2_partner", "G3_context", "G4_suffix", "G5_final", "G6_from_target")


def sample_syllables(inv, n: int, seed: int) -> tuple[list[str], dict[str, dict]]:
    """Legal syllables (onset-rime legality, design 9.2: non-word syllables included so that
    tone is not confounded with lexeme), n // 6 per tone, seeded; returns the spellings and
    a per-syllable {'attested': bool} covariate."""
    rng = random.Random(seed)
    by_tone = defaultdict(list)
    for onset, glide, nucleus, coda in inv.toneless:
        for tone in range(6):
            s = Syllable(onset=onset, glide=glide, nucleus=nucleus, coda=coda, tone=tone)
            if inv.is_legal(s, "onset_rime"):
                by_tone[tone].append(s)
    out: list[str] = []
    labels: dict[str, dict] = {}
    per = n // 6
    for t in range(6):
        pool = sorted(by_tone[t], key=str)
        rng.shuffle(pool)
        for s in pool[:per]:
            w = U.nfc(spell(s))
            out.append(w)
            labels[w] = {"attested": bool(inv.is_legal(s, "attested"))}
    return out, labels


def _position_groups(al: dict, T: int) -> list[list[int]]:
    tgt = list(al["target_positions"])
    return [tgt, [tgt[0]], [tgt[-1]], list(al.get("partner_positions", [])), list(al.get("context_positions", [])),
            list(al.get("suffix_positions", [])), [al["answer_pos"]], list(range(min(tgt), T))]


def _patch_one(model, al: dict, device) -> patching.PatchResult:
    import torch

    ids_c = torch.tensor([al["clean_ids"]], device=device)
    ids_k = torch.tensor([al["corrupt_ids"]], device=device)
    T = ids_c.shape[1]
    groups = _position_groups(al, T)
    live = [g for g in groups if g]
    res = patching.run_patching(model, ids_c, ids_k, al["answer_pos"], al["tok_clean"], al["tok_corrupt"], live)
    rec = np.full((len(res.layers), len(groups)), np.nan)
    raw = np.full_like(rec, np.nan)
    j = 0
    for gi, g in enumerate(groups):
        if g:
            rec[:, gi] = res.recovery[:, j]
            raw[:, gi] = res.raw_ld[:, j]
            j += 1
    res.recovery, res.raw_ld, res.position_groups = rec, raw, groups
    return res


def summarize_recovery(R: np.ndarray, clusters, n_boot: int, seed: int = 0) -> dict:
    """Mean recovery per (hidden index, group) with a syllable-clustered bootstrap CI (design
    9.3: the varying syllable's segmental structure is the cluster)."""
    n, nl, ng = R.shape
    mean = np.full((nl, ng), np.nan)
    lo = np.full_like(mean, np.nan)
    hi = np.full_like(mean, np.nan)
    cl = np.asarray(clusters)
    for li in range(nl):
        for gi in range(ng):
            v = R[:, li, gi]
            ok = np.isfinite(v)
            if ok.sum() == 0:
                continue
            ci = cluster_bootstrap(v[ok], cl[ok], n_boot=n_boot, seed=seed, small_cell_rule=False, bca=False)
            mean[li, gi], lo[li, gi], hi[li, gi] = ci.estimate, ci.lo, ci.hi
    return {"mean_recovery": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n_pairs": int(n),
            "n_clusters": len(set(cl.tolist())), "n_boot": n_boot}


def run_patching_stage(model, tokenizer, inv, args, device, add_special: bool) -> tuple[dict | None, dict]:
    """Filters 1–3 in order with retention, readout B on the retained pairs (or `not_run`
    below the retention floor), readout A on its own aligned pairs."""
    n_cand = args.n_pairs * 3
    cands = P.build_pairs(inv, n_cand, seed=args.pair_seed, target_first=False)
    retention = {"pair_candidates": len(cands), "f1_aligned_skeleton": 0, "f2_tone_only": 0, "f1_aligned_full_prompt": 0,
                 "f3_gap_ok": 0, "f3_clean_greedy_correct": 0, "used_b": 0, "a_aligned": 0, "a_gap_ok": 0, "used_a": 0,
                 "min_gap_nats": constants.PATCHING_MIN_GAP_NATS, "readout_b_min_retention": constants.PATCHING_READOUT_B_MIN_RETENTION}
    render = None
    if args.readout_b_prompt == "v3":
        render = lambda inp: readouts.render_readout_b(tokenizer, inp)
        answer_prefix = readouts.readout_b_answer_prefix()
    else:
        answer_prefix = " "
    retained_b = []
    used_b = []
    for pr in cands:
        al0 = P.align_pair(tokenizer, pr, add_special_tokens=add_special)
        if al0 is None:
            continue
        retention["f1_aligned_skeleton"] += 1
        if not al0["readout_tone_only"]:
            continue
        retention["f2_tone_only"] += 1
        full = P.rerender_pair(pr, render) if render is not None else pr
        al = P.align_pair(tokenizer, full, add_special_tokens=add_special, answer_prefix=answer_prefix) if render is not None else al0
        if al is None or not al["readout_tone_only"]:
            continue
        retention["f1_aligned_full_prompt"] += 1
        import torch
        gap = patching.pair_gap(model, torch.tensor([al["clean_ids"]], device=device), torch.tensor([al["corrupt_ids"]], device=device),
                                al["answer_pos"], al["tok_clean"], al["tok_corrupt"])
        if gap["gap"] >= constants.PATCHING_MIN_GAP_NATS:
            retention["f3_gap_ok"] += 1
        if gap["clean_greedy_correct"]:
            retention["f3_clean_greedy_correct"] += 1
        if not (gap["gap"] >= constants.PATCHING_MIN_GAP_NATS and gap["clean_greedy_correct"]):
            continue
        retained_b.append((pr, al, gap))
        if len(retained_b) >= args.n_pairs:
            break
    retention["retained_b"] = len(retained_b)
    summary: dict = {"position_groups": list(GROUP_NAMES), "readout_b_prompt": args.readout_b_prompt, "retention": retention}
    arrays = {}
    if len(retained_b) >= constants.PATCHING_READOUT_B_MIN_RETENTION or (args.tiny_test and retained_b):
        recs, clusters = [], []
        for pr, al, gap in retained_b:
            res = _patch_one(model, al, device)
            recs.append(res.recovery)
            clusters.append(U.strip_tones(pr.target_syllable_clean))
            used_b.append({"clean": pr.clean if args.readout_b_prompt != "v3" else pr.partner + " " + pr.target_syllable_clean,
                           "corrupt_target": pr.target_syllable_corrupt, "ld_clean": gap["ld_clean"], "ld_corrupt": gap["ld_corrupt"],
                           "readout_pieces": al["readout_pieces"]})
        R = np.stack(recs)
        arrays["recovery_b"] = R
        summary["readout_b"] = {"status": "run", **summarize_recovery(R, clusters, args.n_boot), "pairs": used_b[:50]}
        retention["used_b"] = len(recs)
    else:
        summary["readout_b"] = {"status": "not_run", "reason": f"retention {len(retained_b)} < {constants.PATCHING_READOUT_B_MIN_RETENTION} "
                                "(design 9.3, item 25): readout A only; E4 is a localization study"}
    # readout A (perception) on the tone-minimal syllables, its own alignment and gap filters
    recs_a, clusters_a = [], []
    for pr in cands:
        al = readouts.readout_a_pair(tokenizer, pr, add_special_tokens=add_special)
        if al is None:
            continue
        retention["a_aligned"] += 1
        import torch
        gap = patching.pair_gap(model, torch.tensor([al["clean_ids"]], device=device), torch.tensor([al["corrupt_ids"]], device=device),
                                al["answer_pos"], al["tok_clean"], al["tok_corrupt"])
        if not (gap["gap"] >= constants.PATCHING_MIN_GAP_NATS and gap["clean_greedy_correct"]):
            continue
        retention["a_gap_ok"] += 1
        al["partner_positions"], al["context_positions"] = [], list(range(min(al["target_positions"])))
        al["suffix_positions"] = list(range(max(al["target_positions"]) + 1, al["answer_pos"]))
        res = _patch_one(model, al, device)
        recs_a.append(res.recovery)
        clusters_a.append(U.strip_tones(pr.target_syllable_clean))
        if len(recs_a) >= args.n_pairs:
            break
    retention["used_a"] = len(recs_a)
    if recs_a:
        RA = np.stack(recs_a)
        arrays["recovery_a"] = RA
        summary["readout_a"] = {"status": "run", **summarize_recovery(RA, clusters_a, args.n_boot)}
    else:
        summary["readout_a"] = {"status": "not_run", "reason": "no pair passed readout A's alignment and gap filters"}
    summary["study_type"] = "dissociation (readouts A and B)" if summary["readout_b"]["status"] == "run" else "localization (readout A only)"
    if arrays:
        np.savez(args.out / "patching.npz", **arrays)
    (args.out / "patching.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary, retention


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-syllables", type=int, default=6 * constants.PROBE_SYLLABLES_PER_TONE)
    ap.add_argument("--syllable-seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=constants.PATCHING_MIN_CLEAN_PAIRS)
    ap.add_argument("--pair-seed", type=int, default=0)
    ap.add_argument("--encodings", nargs="*", default=["nfc", "nfd"])
    ap.add_argument("--positions", nargs="*", default=["last", "after"])
    ap.add_argument("--carriers", type=int, default=constants.PROBE_MIN_CARRIERS)
    ap.add_argument("--split-seeds", type=int, default=constants.PROBE_SPLIT_SEEDS)
    ap.add_argument("--control-seeds", type=int, default=constants.PROBE_CONTROL_SEEDS)
    ap.add_argument("--secondary", action="store_true", help="also probe onset/rime/coda (1B secondary pass, design 9.2)")
    ap.add_argument("--n-boot", type=int, default=constants.BOOTSTRAP_B)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--readout-b-prompt", default="v3", choices=["v3", "skeleton"],
                    help="v3 = E1's V3 prompt with the chat template (design 9.3); skeleton = PATCH_PROMPT (tests only)")
    ap.add_argument("--skip-patching", action="store_true")
    ap.add_argument("--tiny-test", action="store_true", help="use a random tiny model (tests only)")
    args = ap.parse_args(argv)
    if args.carriers < constants.PROBE_MIN_CARRIERS and not args.tiny_test:
        ap.error(f"--carriers must be >= {constants.PROBE_MIN_CARRIERS} (design 9.2)")
    if args.carriers > len(extract.CARRIERS):
        ap.error(f"only {len(extract.CARRIERS)} carriers exist")

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
        for ch in dict.fromkeys(VOCAB_CHARS + list("→(),:«»")):
            vocab.setdefault(ch, len(vocab))
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>")
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048)).eval()
        add_special = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
        add_special = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inv = L.load_inventory()
    syllables, syl_labels = sample_syllables(inv, args.n_syllables, seed=args.syllable_seed)
    probe_rows = []
    excess_rows = []
    best_layer = {}
    H_by_enc = {}
    tuned = None
    features = PRIMARY_FEATURES + (SECONDARY_FEATURES if args.secondary else ())
    for enc in args.encodings:
        exs = extract.make_examples(syllables, encoding=enc, carrier_ids=list(range(args.carriers)), extra_labels=syl_labels)
        H, spans = extract.extract_hidden_states(model, tokenizer, exs, batch_size=args.batch_size, device=device,
                                                 add_special_tokens=add_special, positions=args.positions, return_token_ids=True)
        H_by_enc[enc] = (exs, H)
        groups = [e.syllable for e in exs]
        rimes = [e.labels["rime"] for e in exs]
        coda_class = [e.labels["coda_class"] for e in exs]
        tones = [e.labels["tone"] for e in exs]
        if tuned is None:
            mid = H[args.positions[0]].shape[1] // 2
            tuned = probes.tune_C(H[args.positions[0]][:, mid, :], tones, groups, seed=0)
            tuned["hidden_index"] = mid
            tuned["encoding"], tuned["position"] = enc, args.positions[0]
        C = tuned["C"]
        for pos in args.positions:
            for feat in features:
                labels = [e.labels[feat] for e in exs]
                for split_seed in range(args.split_seeds):
                    n_ctrl = args.control_seeds if feat == "tone" else 1
                    for control_seed in range(n_ctrl):
                        for r in probes.run_layer_probes(H[pos], labels, groups, feature=feat, seed=split_seed, C=C, control_seed=control_seed):
                            probe_rows.append({"encoding": enc, "position": pos, **r.__dict__})
                    if feat == "tone":
                        for r in probes.run_layer_probes(H[pos], labels, groups, feature=feat, seed=split_seed, C=C, control_seed=0,
                                                         holdout_groups=rimes, split_key="rime"):
                            probe_rows.append({"encoding": enc, "position": pos, **r.__dict__})
            # condition (ii): excess over the structural baseline per hidden index (tone)
            for Lh in range(H[pos].shape[1]):
                ex = probes.excess_over_structural(H[pos][:, Lh, :], spans, coda_class, tones, groups, seed=0, C=C, n_boot=args.n_boot)
                excess_rows.append({"encoding": enc, "position": pos, "layer": Lh, **ex})
        tone_last = [r for r in probe_rows if r["encoding"] == enc and r["position"] == args.positions[0] and r["feature"] == "tone"
                     and r["split_key"] == "syllable"]
        agg = defaultdict(list)
        for r in tone_last:
            agg[r["layer"]].append(r["selectivity"])
        best_layer[enc] = max(agg, key=lambda k: np.mean(agg[k]))
    import pandas as pd
    pd.DataFrame(probe_rows).to_csv(args.out / "probes.csv", index=False)
    pd.DataFrame(excess_rows).to_csv(args.out / "excess.csv", index=False)

    patch_summary, retention = (None, None)
    if not args.skip_patching:
        patch_summary, retention = run_patching_stage(model, tokenizer, inv, args, device, add_special)

    steering = None
    if "nfc" in H_by_enc:
        exs, H = H_by_enc["nfc"]
        L_best = int(best_layer["nfc"])
        pos0 = args.positions[0]
        tones = np.array([e.labels["tone"] for e in exs])
        if (tones == 2).sum() > 5 and (tones == 5).sum() > 5:
            d = patching.difference_in_means(H[pos0][tones == 2, L_best], H[pos0][tones == 5, L_best])
            # hidden index L is the output of decoder block L - HIDDEN_INDEX_OFFSET; index 0 (embeddings) has no block
            block = L_best - extract.HIDDEN_INDEX_OFFSET if L_best >= extract.HIDDEN_INDEX_OFFSET else None
            steering = {"layer_hidden_index": L_best, "decoder_block_index": block, "position": pos0,
                        "direction_norm": float(np.linalg.norm(d)), "classes": ["sac", "nang"],
                        "note": "future work (design 9.4); direction saved only when a decoder block precedes the index"}
            if block is not None:
                np.save(args.out / "steer_direction.npy", d)

    manifest = MF.base_manifest(
        model=args.model, revision=args.revision, dtype=args.dtype, device=str(device),
        hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        n_syllables=len(syllables), n_attested=sum(v["attested"] for v in syl_labels.values()),
        syllable_seed=args.syllable_seed, pair_seed=args.pair_seed, encodings=args.encodings, positions=args.positions,
        carriers=[extract.CARRIERS[i] for i in range(args.carriers)], split_seeds=args.split_seeds,
        control_seeds=args.control_seeds, split_fracs=list(constants.PROBE_SPLIT_FRACS), l2_tuning=tuned,
        hidden_index_offset=extract.HIDDEN_INDEX_OFFSET, features=list(features), n_boot=args.n_boot,
        best_tone_layer_by_encoding={k: int(v) for k, v in best_layer.items()},
        patching=patch_summary and {k: v for k, v in patch_summary.items() if k not in ("readout_a", "readout_b")}
        | {"readout_b_status": patch_summary["readout_b"]["status"], "readout_a_status": patch_summary["readout_a"]["status"]},
        retention=retention, steering=steering, wall_s=time.time() - t0, torch=torch.__version__)
    MF.write_manifest(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
