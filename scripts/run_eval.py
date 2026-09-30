#!/usr/bin/env python
"""Run one model on one item file.

    python scripts/run_eval.py --items data/release/v0.2/noilai_core.jsonl --model-config gemma-3-1b-it \
        --tasks T1 T2 T3 --variants V1 V2 V3 V4 --paraphrases p0 p1 p2 --shots 3 --arms nfc nfd

    python scripts/run_eval.py --items data/release/v0.2/noilai_test.jsonl --model-config gemma-3-1b-it \
        --sample 4200 --sample-seed 20261004          # seeded stratified sample, ids in the manifest

    python scripts/run_eval.py --items data/release/v0.2/noilai_dev.jsonl --backend echo --smoke --limit 20 --score
    python scripts/run_eval.py --items data/external/xcopa_test_vi.jsonl --model-config gemma-3-1b-it \
        --arms nfc nfd placement_new strip_tones

--model-config names an entry of configs/models.yaml (or --models-file); --backend overrides
the entry's backend (e.g. its fallback). Outputs stream to data/runs/<run_id>/outputs.jsonl;
--resume continues a run under the same --run-id. Sub-samples: --sample N draws a seeded,
stratified sample (per task x variant cell, core forced in, T3 pairs together, vulgar
excluded; noilai.eval.sample) and records the ids and seed in the manifest; --limit N is a
head truncation ("first N", one task on a release file) and is refused unless --smoke is
given (DESIGN_DECISIONS 12.32). An API backend refuses non-core items unless
--allow-noncore-api and any file whose path contains validation / human / sealed; an item
equal to a demonstration phrase is refused unless --allow-demo-overlap; items sharing a
syllable with the demonstrations are flagged (--demo-overlap-policy flag|drop|refuse).
--arm-scope whole_prompt|item (default whole_prompt, DESIGN_DECISIONS 6.1). --score runs
scripts/score_run.py's scoring at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.eval import backends as B
from noilai.eval import prompts as P
from noilai.eval.run import (
    ATTESTED_POLICIES,
    DEMO_OVERLAP_POLICIES,
    RUNS_DIR,
    RunOptions,
    load_item_file,
    run,
)
from noilai.eval.sample import DEFAULT_SEED
from noilai.gen import variants as V


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="NóiLái jsonl or XCOPA jsonl")
    ap.add_argument("--model-config", help="entry name in the models file")
    ap.add_argument("--models-file", type=Path, default=B.MODELS_FILE)
    ap.add_argument("--backend", help="override the entry's backend: hf vllm openai_compat gemini llama_cpp echo")
    ap.add_argument("--model-id", help="model id when no --model-config is given")
    ap.add_argument("--dtype")
    ap.add_argument("--device", default=None, help="hf: cuda / cpu / auto")
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--variants", nargs="+", default=list(V.VARIANTS),
                    help="T1/T2/T3 cells (attested rows are selected by --attested-policy, not by this)")
    ap.add_argument("--paraphrases", nargs="+", default=["p0"])
    ap.add_argument("--shots", nargs="+", type=int, default=[3])
    ap.add_argument("--arms", nargs="+", default=["nfc"], help="nfc|base nfd win1258|pc placement_old placement_new "
                                                              "strip_tones strip_all")
    ap.add_argument("--arm-scope", default=P.DEFAULT_ARM_SCOPE, choices=P.ARM_SCOPES)
    ap.add_argument("--limit", type=int, default=0, help="SMOKE ONLY: the first N selected items (requires --smoke)")
    ap.add_argument("--smoke", action="store_true", help="a smoke test: allows --limit")
    ap.add_argument("--sample", type=int, default=0, help="seeded stratified sample of N items (recorded in the manifest)")
    ap.add_argument("--sample-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sample-keep-vulgar", action="store_true", help="keep vulgar-flagged items in the sample")
    ap.add_argument("--in-core-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--out-root", type=Path, default=RUNS_DIR)
    ap.add_argument("--allow-noncore-api", action="store_true")
    ap.add_argument("--allow-demo-overlap", action="store_true", help="override the phrase-level refusal (pilots)")
    ap.add_argument("--demo-overlap-policy", default="flag", choices=DEMO_OVERLAP_POLICIES,
                    help="items sharing a syllable with the demonstrations: flag per row (default), drop, or refuse")
    ap.add_argument("--attested-policy", default="exact2", choices=ATTESTED_POLICIES,
                    help="exact2: exact two-syllable attested rows only (default); all: every row")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-logprobs", action="store_true", help="skip T3/XCOPA log-probabilities")
    ap.add_argument("--input-format", default="raw", choices=P.INPUT_FORMATS)
    ap.add_argument("--instruction", default="explained", choices=P.INSTRUCTIONS)
    ap.add_argument("--language", default="vi", choices=P.LANGUAGES)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--n-accelerators", type=int, default=None, help="GPU-hour multiplier override (TPUs)")
    ap.add_argument("--account-holder", default=None, help="ROLE of the API/compute account holder (never a name)")
    ap.add_argument("--on-long-prompt", default=None, choices=["error", "truncate_left"],
                    help="hf: what to do with a prompt longer than max_model_len - max_new_tokens (default error)")
    ap.add_argument("--require-census", action="store_true", help="fail without a tokenizer normalization census")
    ap.add_argument("--score", action="store_true", help="score the run when it finishes")
    ap.add_argument("--audit", type=Path, default=None, help="tokenizer audit rows csv for n_input_tokens_syll")
    return ap


def resolve_entry(args) -> dict:
    if args.model_config:
        cfg = B.load_models_config(args.models_file)
        entry = B.get_model_entry(cfg, args.model_config)
    else:
        entry = {"name": args.model_id or args.backend or "model", "model_id": args.model_id, "hf_id": args.model_id,
                 "backend": args.backend, "seed": 20261004, "max_new_tokens": 64}
    if args.dtype:
        entry["dtype"] = args.dtype
    if args.device:
        entry["device"] = args.device
    if args.seed is not None:
        entry["seed"] = args.seed
    if args.on_long_prompt:
        entry["on_long_prompt"] = args.on_long_prompt
    entry["batch_size"] = args.batch_size
    return entry


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.limit and not args.smoke:
        ap.error("--limit is a head truncation (the first N items: one task on a release file) and is smoke-only; "
                 "pass --smoke for a smoke test, or --sample N --sample-seed S for a seeded stratified sample "
                 "(DESIGN_DECISIONS 4.5 / 12.32)")
    if args.limit and args.sample:
        ap.error("--limit and --sample are exclusive")
    entry = resolve_entry(args)
    backend = B.make_backend(entry, backend=args.backend)
    items, kind = load_item_file(args.items)
    opts = RunOptions(
        tasks=tuple(args.tasks), variants=tuple(args.variants), paraphrases=tuple(args.paraphrases),
        shots=tuple(args.shots), arms=tuple(args.arms), arm_scope=args.arm_scope, limit=args.limit,
        sample_n=args.sample, sample_seed=args.sample_seed, sample_exclude_vulgar=not args.sample_keep_vulgar,
        in_core_only=args.in_core_only, resume=args.resume, allow_noncore_api=args.allow_noncore_api,
        allow_demo_overlap=args.allow_demo_overlap, demo_overlap_policy=args.demo_overlap_policy,
        attested_policy=args.attested_policy,
        max_new_tokens=args.max_new_tokens or int(entry.get("max_new_tokens") or 64), batch_size=args.batch_size,
        logprobs=not args.no_logprobs, input_format=args.input_format, instruction=args.instruction,
        language=args.language, seed=int(entry.get("seed", 0)), run_id=args.run_id, out_root=args.out_root,
        system_prompt=args.system_prompt, n_accelerators=args.n_accelerators, account_holder=args.account_holder,
        require_census=args.require_census, notes={"smoke": bool(args.smoke)},
    )
    try:
        run_dir = run(backend, entry, items, args.items, opts, kind=kind)
    finally:
        backend.close()
    print(run_dir)
    if args.score:
        from score_run import score_run_dir

        score_run_dir(run_dir, items_path=args.items, audit=args.audit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
