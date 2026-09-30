# NóiLái — can subword LLMs reach the onset, rime and tone inside Vietnamese syllables?

Code, data generator and run kit for an ACL 2027 long paper (ARR deadline 4 January 2027,
23:59 AoE). The paper joins **NóiLái**, a rule-generated and rule-scored benchmark of about
10,000 *nói lái* wordplay items (three tasks over four documented swap variants, plus
200–400 attested folk and literary examples), with **orthographic counterfactuals** that
re-encode Vietnamese text with meaning held fixed (precomposed vs combining diacritics, old vs
new tone-mark placement, and an accent-stripping contrast arm), and a **tone-probing and
activation-patching** study in Gemma 3 1B/4B. Every primary metric is computed by code; no
LLM judge is used. The founding plan is `docs/PLAN_2026-09-30.md`; later decisions go to
`docs/DESIGN_DECISIONS.md`, which wins where the two disagree.

This directory is a self-contained Python subproject (`pyproject.toml`) that happens to live
inside an unrelated repository; nothing outside it is used or modified.

## Layout

```
noilai/
  vi/         syllable.py  parser/speller (onset + glide/nucleus/coda + tone; c/k/q, g/gh, ng/ngh, placement)
              unicode.py   NFC/NFD, tone marks, Windows-1258 partial composition, stripping
              lexicon.py   Hunspell syllable inventory and word lists (lookups only, never redistributed)
              reencode.py  the arms: nfc, nfd, win1258, placement_old/new, strip_tones, strip_all; scoring normalization
  gen/        variants.py  V1–V4 as pure functions on Syllable pairs;  generate.py  items, splits, core, canary, manifest
  audit/      tokenizers.py  tokens per syllable, boundary alignment (STAD-style), NFD tone isolation, normalization census
  stats/      clustered bootstrap, McNemar + Holm families, mixed models (E2), effect decomposition, power, agreement, tables
  probe/      hidden-state extraction at syllable positions, layer-wise probes with control tasks, patching, steering (E4)
  eval/       prompts.py (YAML templates -> chat messages, demos, arms, prompt hash), backends.py (hf | vllm |
              openai_compat | gemini | llama_cpp + echo/scripted test backends), extract.py (the "Đáp án:" line),
              score.py (T1/T2/T3 scoring, error taxonomy, scores.jsonl), xcopa.py (E3 on XCOPA vi), run.py (a run:
              requests, streaming outputs.jsonl, manifest.json, --resume, API and demo-overlap guards)
scripts/
  fetch_resources.py      third-party resources -> data/external/, SHA-256 checked against data/HASHES.json
  build_data.py           a release: data/release/<version>/{noilai_dev,noilai_test,noilai_core,attested}.jsonl + manifest
  audit_tokenizers.py     tokenizer audit -> data/audit/<name>.json (+ per-syllable rows)
  run_eval.py             one model x one item file x tasks/variants/paraphrases/shots/arms -> data/runs/<run-id>/
  score_run.py            score a run directory: scores.jsonl + summary.json (also run_eval.py --score)
  run_probe.py            E4 driver: layer-wise probes, activation patching and steering on one model
  build_attested.py       data/attested_seed.tsv -> the release file attested.jsonl
  audit_items.py          annotate an item file with per-syllable tokenization covariates for one tokenizer
  check_tokenizer_consistency.py  a model's HF tokenizer vs the SentencePiece model file used offline
  make_validation_forms.py  sample the native-validation set and the human-baseline forms; score returned sheets
  kaggle_run_plan.py      configs/run_plan.yaml -> run_eval.py commands; runs them, logs hours, keeps the API daily ledger
  kaggle_verify_items.py  item-file gate: SHA-256 against run_plan.yaml, canary on every row, row count
  kaggle_dataset.py       stage data/runs and push a versioned private Kaggle dataset (kaggle CLI, import guarded)
  kaggle_build_notebooks.py  generates notebooks/*.ipynb from one source; --check refuses hand-edited notebooks
  colab_setup.py          Drive mount, token-safe clone (GIT_ASKPASS), project dir, pinned pip, environment record, secrets
  compute_log.py          GPU-hours CSV for checklist C1 (append / totals / show)
configs/
  models.yaml   the 21-model panel + bf16-reference and thinking variants: ids, hardware, dtype, quantization, limits, risks
  run_plan.yaml the run matrix (pilot, smoke, E1, E3, reasoning, bf16 drift, E4) with the plan's compute estimates
notebooks/      kaggle_eval_t4, kaggle_eval_tpu, colab_probe_gemma3, api_runs  (generated; see below)
prompts/        noilai.yaml (Vietnamese templates p0–p2 and the ablation variants), demos.yaml (few-shot demonstrations built
                from syllables outside the test set), xcopa.yaml (the COPA framing); every string awaits a [NATIVE-CHECK]
data/           HASHES.json (resource hashes), attested_seed.tsv, audit/ (committed), external/ and runs/ (ignored), release/
docs/           PLAN_2026-09-30.md (founding plan), DATA_FORMAT.md (item, output, score schemas), DESIGN_DECISIONS.md
paper/          ACL 2027 LaTeX sources
tests/          pytest (test_vi, test_gen, test_audit, test_stats, test_probe, test_cloud)
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[eval,dev]"            # eval: torch/transformers/API SDKs; dev: pytest, nbformat, scikit-learn
python scripts/fetch_resources.py       # data/external/, hash-checked
python -m pytest -q                     # every test runs offline (models are stand-ins or absent)
```

`huggingface.co` is not needed for the tests or the data build: the Gemma tokenizer files come
from `google/gemma_pytorch` on GitHub and the Hunspell/word-list resources from GitHub raw.

## Build the data

```bash
python scripts/build_data.py --out data/release/v0.1 --seed 20261004
python scripts/build_data.py --out data/release/sealed --seed 777 --sealed      # regenerated at release; never sent to any API
```

Base pairs are real two-syllable words (Viet74K) and pseudo-pairs sampled from attested
syllables; every output syllable must pass `Inventory.is_legal` (attested onset+rime, tone
allowed by the coda); the split is by base pair (`base_pair_id`, the cluster the bootstrap
resamples); the 1,500-item **core** (125 per task × variant cell) is the only NóiLái data an API
sees; every test-split row carries the canary `NOILAI-CANARY-<uuid>`. Schemas are in
`docs/DATA_FORMAT.md`. Attested examples (`data/attested_seed.tsv`) are unverified until a native
validator's name is in `verified_by`; rows marked `vulgar = yes` never reach an API or the
human-baseline form.

## Audit tokenizers

```bash
python scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 --out data/audit
python scripts/audit_tokenizers.py --hf Qwen/Qwen2.5-7B-Instruct --out data/audit     # needs hub access (Kaggle/Colab)
```

`data/audit/gemma3.json` is real: over the 6,595 parseable Hunspell syllables, Gemma 3's
262,144-piece tokenizer spends 1.78 tokens per syllable in NFC (29.5% single-token, boundary
alignment 0.94, onset|rime split in 51.5%) and 2.83 in NFD (7.1% single-token, alignment 0.27,
the combining tone mark isolated in 41.6% of syllables); it does **not** normalize NFD to NFC
(identity normalizer; 3.4% of 500 phrases get the same ids), so the C1 arm can have an effect
on it by construction.

## Run the evaluations (cloud kit)

Compute is $0: Kaggle's free 2×T4 (~30 GPU-h/week, 12-h sessions) and TPU v5e-8 (~20 TPU-h/week,
9-h sessions), free Colab for the probes, and the Gemini and Groq free tiers for the four API
models. The plan budgets 70–150 GPU-hours and 10–20 TPU-hours for everything including a 100%
rerun buffer (`configs/run_plan.yaml`, `plan_lines`).

1. **Panel and settings**: `configs/models.yaml`. One entry per (model, serving configuration):
   `hf_id`, family, group, tokenizer type, backend, dtype, quantization, hardware (`t4`, `2xt4`,
   `tpu`, `api`), `max_model_len`, `chat_template_kwargs` (`enable_thinking: false` for the Qwen
   families in the main runs), provider `base_url` and the **name** of the API-key environment
   variable (never a key), free-tier limits with their source URL, and the per-model risks from
   the plan. Rules the tests enforce: a T4 entry is fp16/fp32/quantized (no bf16 units on a T4);
   **Gemma 3 never runs in fp16** (fp32 on T4s, bf16 on the TPU); TPU entries are bf16; API
   entries have no dtype. Ids the plan names that could not be resolved from the build machine
   (`Qwen3.5-*`, `Gemma 4 *`, `Qwen3.8-27B`, the SEA-LION v4.5 fine-tunes, the Gemini ids, the
   Groq catalogue ids) carry `hf_id_status: uncertain` and a `[UNCERTAIN: verify]` line; keep the
   plan's names, fix the ids when the panel is frozen (week of 19 October 2026).
2. **Run matrix**: `configs/run_plan.yaml`. One line per experiment cell (pilot, smoke, E1 main,
   E1 API core/paraphrase, E3 NóiLái/XCOPA/API, reasoning sub-study, TPU main, bf16 drift, E4)
   naming the item file, tasks, variants, arms, paraphrases, shots, limit, `in_core_only` and the
   plan's estimate. `python scripts/kaggle_run_plan.py --list` shows the expansion.
3. **The eval CLI** (`noilai/eval`, `scripts/run_eval.py`):
   `scripts/run_eval.py --model-config <name> --items <file> --tasks T1 T2 T3 --variants V1 V2 V3 V4
   --paraphrases p0 p1 p2 --shots 3 --arms nfc [--limit N] [--in-core-only] --resume --run-id <id>__<name> --out-root data/runs`
   (the run directory is `data/runs/<id>__<name>/`; `run_plan.yaml`'s `run_eval_out_style` can switch the
   driver back to a single `--out <dir>` flag). `kaggle_run_plan.py` builds exactly that command per
   (run, model) — a test checks every emitted flag against `run_eval.py --help` — refuses an API model
   without `--in-core-only` and refuses the sealed split for any API, then runs the commands,
   logs each model's wall time with its device type, and keeps `data/runs/api_ledger.json`
   (requests per model per UTC day) so that a Groq model at its 1,000-requests/day cap is
   skipped until tomorrow.
4. **Notebooks** (`notebooks/`, generated by `scripts/kaggle_build_notebooks.py`; edit the
   generator, run `--write`, commit both — `tests/test_cloud.py` refuses a hand-edited copy):
   * `kaggle_eval_t4.ipynb`: environment check → clone (git bundle from a private dataset, or
     GitHub with a token from Kaggle Secrets through GIT_ASKPASS) → pinned install (vLLM version
     as a parameter) → resources + item hash/canary gate → smoke test (20 items) → **RUN cell**
     over a model list with `--resume` → outputs to `/kaggle/working` and a private Kaggle dataset
     → GPU-hours log.
   * `kaggle_eval_tpu.ipynb`: the same on the TPU v5e-8 (vLLM-TPU) for Gemma 3 12B, the two 27B
     models, the bf16 drift check and the 27B reasoning run.
   * `colab_probe_gemma3.ipynb`: E4 skeleton — Gemma 3 1B in fp32, `noilai.probe.extract` +
     `probes` (tone/onset/rime, NFC vs NFD, syllable-disjoint split, control tasks) and a patching
     skeleton on *bí mật* / *bí mất*; the two `e4`-tagged cells are the ones to edit.
   * `api_runs.ipynb`: Gemini / Groq on the core only, keys from the environment / Colab userdata /
     Kaggle Secrets, daily-quota loop with `--resume`.
   Every cell is plain Python (no magics, no `input()`); the first code cell is the only one to
   edit. Kaggle Secrets to define: `GITHUB_TOKEN` (only without a bundle), `HF_TOKEN`,
   `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.
5. **Compute log** (checklist C1): `python scripts/compute_log.py totals` on
   `data/compute_log.csv` (`date, platform, gpu_type, n_gpus, hours, run_id, purpose`). The
   driver writes one row per executed model run; the notebooks add a session-overhead row; merge
   the copy that comes back from Kaggle into the repository's file.

Reproducibility: every run directory holds `outputs.jsonl`, `scores.jsonl` and a `manifest.json`
with model id and revision, backend and version, dtype, quantization, engine flags, seed, prompt
and item hashes, canary check, hardware, start/end, GPU-hours, resource hashes, git commit and
dirty flag (`docs/DATA_FORMAT.md`). Notebooks record `pip freeze` and the CUDA/TPU environment next
to the outputs (`data/runs/env/`, checklist C4).

## Statistics and probes

* `noilai.stats` (present): `bootstrap` (clustered by base pair; paired differences),
  `tests` (exact/mid-p McNemar, Holm within declared families), `mixed` (Bayesian mixed GLM +
  GEE for E2), `mediation` (dose–response decomposition of an arm's effect by token-count change),
  `power`, `agreement` (Krippendorff's α with bootstrap CI), `tables` (LaTeX fragments; no number
  in the paper is typed by hand). `scripts/score_run.py` produces the `scores.jsonl` these read; a
  `scripts/run_stats.py` driver over `data/runs/*/scores.jsonl` is **planned**.
* `noilai.probe` (present): `extract` (residual stream at the syllable's last sub-token / the token
  after it, NFC and NFD), `probes` (layer-wise logistic probes, syllable-disjoint split, control
  labels, selectivity), `patching` (residual cache, layer/position patching, recovery of the logit
  difference, difference-in-means steering). `scripts/run_probe.py` is the batch driver; the Colab
  notebook `colab_probe_gemma3.ipynb` is the interactive skeleton for the same calls.

## Resources and licensing

Nothing under `data/external/` is committed; `scripts/fetch_resources.py` downloads each file and
verifies it against `data/HASHES.json` (the SHA-256 observed on 30 September 2026; a changed
upstream file is kept with a `.UNVERIFIED` suffix and the build stops).

| file | origin | license | role |
|---|---|---|---|
| `vi-DauMoi.dic`, `vi-DauCu.dic` | 1ec5/hunspell-vi (Hồ Ngọc Đức's list, László Németh) | GPLv2 | attested syllable inventory, new and old placement |
| `Viet74K.txt` | duyet/vietnamese-wordlist (Hồ Ngọc Đức) | GPL | two-syllable base pairs, frequency proxy, lexicality |
| `xcopa_test_vi.jsonl`, `xcopa_val_vi.jsonl` | cambridgeltl/xcopa | CC BY 4.0 | E3 re-encoding arms (test), prompt development (val) |
| `gemma3_tokenizer.model`, `gemma2_tokenizer.model` | google/gemma_pytorch | Apache-2.0 repository | tokenizer audit without hub access |

The GPL word lists are used as look-ups at generation time and never redistributed; the released
items are produced by the rule engine. Release licenses: the dev split under CC BY 4.0, the gated
test split under CC BY-NC-ND 4.0 with a canary string; a sealed, regenerated split never reaches
any API (Gemini's free tier trains on inputs and requires users to be 18 or older; Groq's and
Gemini's terms are quoted in `configs/models.yaml`). Model outputs quote test items, so the Kaggle
dataset that carries `data/runs` is private and labelled CC BY-NC-ND 4.0.

## Vietnamese text

Prompts, instructions and examples in this repository are marked for native validation:
`# NATIVE-CHECK` in code and `[NATIVE-CHECK]` in documents. No example is treated as verified
before a native validator's name is recorded (`verified_by` in `data/attested_seed.tsv`).

## AI-assistance disclosure

Code in this subproject was written with the help of an AI coding assistant (Claude, Anthropic)
under the author's direction and review; the author is responsible for its correctness. Benchmark
items come from the rule engine, not from a language model. This use is disclosed in the
Responsible NLP Checklist (E1) and will be stated in the paper's Acknowledgements at camera-ready,
per the ARR policy on generative assistance.

## Change log

* **2026-09-30** — Kick-off: syllable parser/speller, unicode and re-encoding arms, Hunspell/word-list
  lexicon, V1–V4 generator with T1/T2/T3 items, splits, core and canary; tokenizer audit with the
  real Gemma 3 numbers; statistics and probe modules; the cloud run kit (`configs/models.yaml`
  with the 21-model panel and its variants, `configs/run_plan.yaml`, the four generated notebooks,
  `kaggle_run_plan.py`, `kaggle_verify_items.py`, `kaggle_dataset.py`, `colab_setup.py`,
  `compute_log.py`, `tests/test_cloud.py`) and this README. The evaluation harness (`noilai/eval`,
  `scripts/run_eval.py`, `scripts/score_run.py`), the prompt templates under `prompts/` and the probe
  driver `scripts/run_probe.py` landed the same day; the run kit's driver follows the CLI as landed
  (`--run-id`/`--out-root`) and a test keeps the two in step. No model has been run yet; every
  Vietnamese string awaits native validation.
