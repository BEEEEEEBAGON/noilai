#!/usr/bin/env python
"""Run one model on one item file.

    python scripts/run_eval.py --items data/release/v0.1/noilai_core.jsonl --model-config gemma-3-1b-it \
        --tasks T1 T2 T3 --variants V1 V2 V3 V4 --paraphrases p0 p1 p2 --shots 3 --arms nfc nfd

    python scripts/run_eval.py --items data/release/v0.1/noilai_dev.jsonl --backend echo --limit 20 --score
    python scripts/run_eval.py --items data/external/xcopa_test_vi.jsonl --model-config gemma-3-1b-it \
        --arms nfc nfd placement_old strip_tones

--model-config names an entry of configs/models.yaml (or --models-file); --backend overrides
the entry's backend (e.g. its fallback). Outputs stream to data/runs/<run_id>/outputs.jsonl;
--resume continues a run under the same --run-id. An API backend refuses non-core items
unless --allow-noncore-api; an item file sharing syllables with the demonstrations is refused
unless --allow-demo-overlap. --score runs scripts/score_run.py's scoring at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.eval import backends as B
from noilai.eval import prompts as P
from noilai.eval.run import RUNS_DIR, RunOptions, load_item_file, run
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
    ap.add_argument("--variants", nargs="+", default=list(V.VARIANTS))
    ap.add_argument("--paraphrases", nargs="+", default=["p0"])
    ap.add_argument("--shots", nargs="+", type=int, default=[3])
    ap.add_argument("--arms", nargs="+", default=["nfc"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--in-core-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--out-root", type=Path, default=RUNS_DIR)
    ap.add_argument("--allow-noncore-api", action="store_true")
    ap.add_argument("--allow-demo-overlap", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-logprobs", action="store_true", help="skip T3/XCOPA log-probabilities")
    ap.add_argument("--input-format", default="raw", choices=P.INPUT_FORMATS)
    ap.add_argument("--instruction", default="explained", choices=P.INSTRUCTIONS)
    ap.add_argument("--language", default="vi", choices=P.LANGUAGES)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--n-accelerators", type=int, default=None, help="GPU-hour multiplier override (TPUs)")
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
    entry["batch_size"] = args.batch_size
    return entry


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    entry = resolve_entry(args)
    backend = B.make_backend(entry, backend=args.backend)
    items, kind = load_item_file(args.items)
    opts = RunOptions(
        tasks=tuple(args.tasks), variants=tuple(args.variants), paraphrases=tuple(args.paraphrases),
        shots=tuple(args.shots), arms=tuple(args.arms), limit=args.limit, in_core_only=args.in_core_only,
        resume=args.resume, allow_noncore_api=args.allow_noncore_api, allow_demo_overlap=args.allow_demo_overlap,
        max_new_tokens=args.max_new_tokens or int(entry.get("max_new_tokens") or 64), batch_size=args.batch_size,
        logprobs=not args.no_logprobs, input_format=args.input_format, instruction=args.instruction,
        language=args.language, seed=int(entry.get("seed", 0)), run_id=args.run_id, out_root=args.out_root,
        system_prompt=args.system_prompt, n_accelerators=args.n_accelerators,
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
