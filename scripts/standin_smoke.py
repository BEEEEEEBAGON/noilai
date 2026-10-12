#!/usr/bin/env python
"""Local, CPU-only end-to-end smoke of the harness with a RANDOM-WEIGHT stand-in model (no Hub access needed).

huggingface.co is unreachable from the build machine, so no panel model can be downloaded here. This script builds
the closest thing that exercises the real code paths: a 2-layer Gemma3ForCausalLM with random weights and the REAL
Gemma 3 SentencePiece tokenizer (data/external/gemma3_tokenizer.model, fetched from google/gemma_pytorch) and a
stand-in of the Gemma 3 chat template; then runs harness -> scoring -> statistics -> hash gate on ~20 items:

    python scripts/standin_smoke.py --items <an item file> --work <scratch dir>

What it exercises: chat-template rendering and single tokenization, the HF backend's generation and log-probability
paths (T3 forced choice, the exploratory T1 forced choice), NFD re-encoding with real token counts, max_model_len,
extraction, scoring, the cluster bootstrap and paired tests, the item and results hashes. What it does NOT show:
anything about any model's ability (the weights are random; outputs are noise) or the real chat template's exact
tokens. The first real model run is the Kaggle CPU notebook (notebooks/kaggle_cpu_pilot.ipynb).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDIN_TEMPLATE = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'assistant' %}{% set r = 'model' %}"
                    "{% else %}{% set r = m['role'] %}{% endif %}<start_of_turn>{{ r }}\n{{ m['content'] | trim }}"
                    "<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")


def build_standin(out: Path, spm: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig
    spmdir = out / "spm"
    spmdir.mkdir(parents=True, exist_ok=True)
    (spmdir / "tokenizer.model").write_bytes(spm.read_bytes())
    (spmdir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>",
                                                              "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
                                                              "add_bos_token": True, "add_eos_token": False}))
    tok = AutoTokenizer.from_pretrained(spmdir)
    tok.chat_template = STANDIN_TEMPLATE
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=2048,
                           sliding_window=64, bos_token_id=tok.bos_token_id, eos_token_id=[tok.eos_token_id, eot],
                           pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    model_dir = out / "standin_gemma3"
    Gemma3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return model_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="an item file with T1 items (a dev split; never the gated test split)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-variant", type=int, default=5)
    ap.add_argument("--spm", default=str(ROOT / "data" / "external" / "gemma3_tokenizer.model"))
    args = ap.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    model_dir = build_standin(work, Path(args.spm))
    models = work / "standin_models.yaml"
    models.write_text(
        "# stand-in for the local harness smoke: RANDOM weights, the real Gemma 3 tokenizer; not a panel model\n"
        "version: 1\ndefaults: {max_model_len: 2048, max_new_tokens: 64, temperature: 0.0, seed: 20261203, revision: null}\n"
        "models:\n  - {name: standin-gemma3-tiny-random, hf_id: " + json.dumps(str(model_dir)) + ", hf_id_status: standin,"
        " revision: null, family: gemma3, group: standin, backend: hf, dtype: float32, device: cpu, batch_size: 8,"
        " hardware: cpu, max_model_len: 2048, chat_template_kwargs: {}}\n", encoding="utf-8")
    rows = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    t1 = sorted((r for r in rows if r.get("task") == "T1" and not r.get("vulgar")), key=lambda r: r["item_id"])
    random.Random(20261001).shuffle(t1)
    pick = [r for v in ("V1", "V2", "V3", "V4") for r in [x for x in t1 if x["variant"] == v][: args.per_variant]]
    items = work / "standin_items_t1.jsonl"
    items.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pick) + "\n", encoding="utf-8")
    run_id = "standin_smoke_t1"
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "run_eval.py"), "--items", str(items), "--models-file", str(models),
                    "--model-config", "standin-gemma3-tiny-random", "--tasks", "T1", "--variants", "V1", "V2", "V3", "V4",
                    "--paraphrases", "p0", "--shots", "3", "--arms", "nfc", "nfd", "--smoke", "--t1-forced-choice",
                    "--run-id", run_id, "--out-root", str(work / "runs"), "--score"], check=True, cwd=ROOT)
    r = subprocess.run([py, str(ROOT / "scripts" / "check_run.py"), "--run", str(work / "runs" / run_id), "--n-boot", "500"],
                       check=True, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip())
    v = subprocess.run([py, str(ROOT / "scripts" / "check_run.py"), "--run", str(work / "runs" / run_id), "--verify"],
                       check=False, cwd=ROOT, capture_output=True, text=True)
    print(v.stdout.strip())
    return v.returncode


if __name__ == "__main__":
    sys.exit(main())
