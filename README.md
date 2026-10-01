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
              lexicon.py   Hunspell syllable inventory and word lists (consulted at build time; Viet74K two-syllable entries
                           appear verbatim as inputs/gold of the released lexical items, see "Resources and licensing")
              reencode.py  the arms: nfc, nfd, win1258, placement_old/new, strip_tones, strip_all; scoring normalization
  gen/        variants.py  V1–V4 as pure functions on Syllable pairs;  generate.py  items, splits, core, canary, manifest
  audit/      tokenizers.py  tokens per syllable, boundary alignment (STAD-style), NFD tone isolation, normalization census
  stats/      clustered bootstrap, McNemar + Holm families, mixed models (E2), effect decomposition, power, agreement, tables
  probe/      hidden-state extraction at syllable positions, layer-wise probes with control tasks, patching; steering code
              stays but steering is future work (DD 9.4 / 12.36)
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
  run_probe.py            E4 driver: layer-wise probes and activation patching on one model (steering: future work, DD 9.4)
  build_attested.py       data/attested_seed.tsv -> the release file attested.jsonl
  audit_items.py          annotate an item file with per-syllable tokenization covariates for one tokenizer
  check_tokenizer_consistency.py  a model's HF tokenizer vs the SentencePiece model file used offline
  make_validation_forms.py  sample the native-validation set and the human-baseline forms; score returned sheets
  sample_items.py         seeded sub-samples of a release (DD 4.5): noilai_main.jsonl (4,200, 350 per cell) and noilai_c2.jsonl
  kaggle_run_plan.py      configs/run_plan.yaml -> run_eval.py commands; derives the seeded core subsets (--materialize), gates every
                          run on its item file, runs the commands under the session guards, logs hours, keeps the API request+token ledger
  kaggle_verify_items.py  item-file gate: SHA-256 against run_plan.yaml, the BIG-bench header record, canary on every row, row/cell counts
  kaggle_dataset.py       stage data/runs and push a versioned private Kaggle dataset (kaggle CLI, import guarded)
  kaggle_build_notebooks.py  generates notebooks/*.ipynb from one source; --check refuses hand-edited notebooks
  colab_setup.py          Drive mount, token-safe clone (GIT_ASKPASS), project dir, pinned pip, environment record, secrets
  compute_log.py          GPU-hours CSV for checklist C1 (append / totals / show)
configs/
  models.yaml   the 21-model panel + bf16-reference and thinking variants: ids, revision, hardware, dtype, quantization, limits, risks
  run_plan.yaml the run matrix (pilot, smoke, E1 + explicit-input main result + ablations, E3, reasoning, bf16 drift, E4) with the
                plan's compute estimates; `release:` names the release version once, every item path derives from it
notebooks/      kaggle_eval_t4, kaggle_eval_tpu, colab_probe_gemma3, api_runs  (generated; see below)
prompts/        noilai.yaml (Vietnamese templates p0–p2 and the ablation variants), demos.yaml (few-shot demonstrations built
                from syllables outside the test set), xcopa.yaml (the COPA framing); every string awaits a [NATIVE-CHECK]
data/           HASHES.json (resource hashes), attested_seed.tsv, audit/ (committed), external/ and runs/ (ignored), release/;
                validation/ and human/ hold validator and human-baseline material (forms, returned sheets): git-ignored,
                never in the bundle a notebook clones, never sent anywhere (DD 11.2)
docs/           PLAN_2026-09-30.md (founding plan), DATA_FORMAT.md (item, output, score schemas), DESIGN_DECISIONS.md (binding)
paper/          ACL 2027 LaTeX sources
tests/          pytest (test_vi, test_gen, test_audit, test_stats, test_constants, test_probe, test_eval, test_scripts,
                test_release, test_e2, test_paper, test_cloud)
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
python scripts/build_data.py --out data/release/v0.3 --seed <private>
python scripts/build_data.py --out data/release/sealed --seed <private> --sealed   # regenerated at release; never sent to any API
make data SEED=<private>                                                           # build + attested.jsonl + the sampled files
```

The build seed is not published (DD 4.6, item 51): dev and test are drawn, shuffled and split from
one `random.Random(seed)` stream, so publishing it would regenerate the gated test split; it lives
only in `data/release/<version>/manifest_private.json` (git-ignored), while the committed
`manifest.json` carries the counts, the content hash and the SHA-256 of the canary. `--seed` is
required, `make data` fails when `SEED` is unset, and the sampling seeds of `scripts/sample_items.py`
(20261201 for `main`, 20261202 for `c2`) are public because they select from the built test split
and regenerate nothing.

Base pairs are real two-syllable words (Viet74K) and pseudo-pairs sampled from attested
syllables; every output syllable must pass `Inventory.is_legal` (attested onset+rime, tone
allowed by the coda); the split is by base pair (`base_pair_id`, the cluster the bootstrap
resamples); the **core** of about 1,500 items (125 per T1/T2 cell, 62 yes/no pairs = 124 items
per T3 cell; 1,496 in v0.3) is the only NóiLái test data an API may see, and only a provider with
no-training terms (Groq); a provider whose tier trains on inputs (Gemini unpaid) never receives it
(DD 11.2: a paid key with a verified opt-out, or the dev-derived API set, reported separately —
the run driver refuses the combination until the author decides, DD 13.18); every test-split row carries
the canary `NOILAI-CANARY-<uuid>` and every test/core file begins with the BIG-bench header
record (`{"_header": ..., "canary": ..., "do_not_train": true, "evaluation_only": true}`) that
loaders skip by key. The current working build is `data/release/v0.3`; DD 12.30 supersedes it by
v0.3 at the generator freeze (bump the run plan's `release:` key then and re-record the hashes).
Sub-samples are seeded files, never drawn at run time (DD 4.5 / 12.32): `scripts/sample_items.py
main` writes the 4,200-item open-model sample (`noilai_main.jsonl`, 350 per cell, contains the
core) and `c2` the C2-enriched set; the smaller core subsets the run plan needs (296-item API
paraphrase set, 500-item reasoning set, 200-item bf16 set, the two 200-item pilot files) are
derived deterministically by `scripts/kaggle_run_plan.py --materialize` from the `derive` blocks
of `configs/run_plan.yaml`. Schemas are in `docs/DATA_FORMAT.md`. Attested examples
(`data/attested_seed.tsv`) are unverified until a native validator's name is in `verified_by`;
rows marked `vulgar = yes` never reach an API or the human-baseline form.

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
   `hf_id`, `revision` (null until the panel freeze; DD 7.1: `run_eval.py` refuses an unpinned local
   backend without `--smoke`, and the run driver skips an unpinned self-hosted model on every
   non-smoke line unless `--allow-unpinned-revision`), family, group, tokenizer
   type, backend, dtype, quantization, hardware (`t4`, `2xt4`, `tpu` with tensor parallel 8, `l4`
   as the Modal bf16 fallback, `api`), `max_model_len` (1,024 per DD 7.3), `chat_template_kwargs`
   (`enable_thinking: false` for the Qwen families in the main runs), the DD 7.3 engine defaults
   (`gpu_memory_utilization 0.85`, `engine_kwargs`: TRITON_ATTN, zero multimodal limits, prefix
   caching, 64 seqs; the TPU entries drop the attention backend), provider `base_url` and the
   **name** of the API-key environment variable (never a key), free-tier limits with their source
   URL and a client-side `requests_per_minute` per API entry, and the per-model risks from the
   plan. PhoGPT runs on HF transformers (llama.cpp fallback): MPT is not runnable on vLLM 0.30
   (DD 7.2 / 12.25). Every provider block carries `terms.trains_on_inputs` as a boolean (Gemini
   unpaid: true; Groq: false; OpenRouter: false only with `data_collection: deny`), the key the
   core guards read. Rules the tests enforce: a T4 entry is fp16/fp32/quantized (no bf16 units on
   a T4); **Gemma 3 never runs in fp16** (keyed on `google/gemma-3-*` and the family label; fp32 on
   T4s, bf16 on the TPU); TPU/L4 entries are bf16; API entries have no dtype; every hardware block
   defines `tensor_parallel_size`; every self-hosted entry carries `revision`. Ids the plan names
   that could not be resolved from the build machine (`Qwen3.5-*`, `Gemma 4 *`, `Qwen3.8-27B`, the
   SEA-LION v4.5 fine-tunes, the Gemini ids, the Groq catalogue ids) carry `hf_id_status:
   uncertain` and a `[UNCERTAIN: verify]` line; keep the plan's names, fix the ids when the panel
   is frozen (week of 19 October 2026). The Groq daily caps are recorded under both readings
   (per model, as the console lists them; per org, as DD 7.2 writes) with `scope` to flip.
2. **Run matrix**: `configs/run_plan.yaml`. `release:` names the release version once and every
   item path derives from it (`{release}`); one line per experiment cell: the two Gate 1 pilots
   (T1 and XCOPA, 200 seeded items each), smoke tests (T4 and TPU; the only lines allowed a
   `--limit`, DD 12.32), E1 main on the 4,200-item sample (DD 12.11), the explicit onset–rime–tone
   input **main result** (DD 12.17) and the four prompt ablations (zero-shot, name-only, English
   instruction, character-spaced input) on the core, E1 attested, E1 API core/paraphrase, E3
   NóiLái (the DD 8.3 family `nfd`, `pc`, `strip_tones`, `strip_all` on the main sample, T1 and T3),
   E3 C2-enriched (`placement_new`, the only C2 line: the release is stored old-style, DD 6.3),
   E3 XCOPA (same four arms; C2 on XCOPA is not a tested cell, DD 12.36), E3 API, the DD 6.1/6.2
   scope and engine agreement checks on a 500-item Gemma 3 1B subset (`arm_scope: item`,
   `backend: hf`), the reasoning sub-study, TPU main, bf16 drift, E4. Every E3 test-split line is
   scheduled after 8 November (DD 8.8); E4 is reported regardless of outcome. Each line names the
   item file, tasks, variants, arms, paraphrases, shots, `input_format` / `instruction` /
   `language` / `arm_scope` / `backend` when it is an ablation or a check, `in_core_only` and the
   plan's estimate; the compute table is re-budgeted within the plan's 70–150 GPU-h (the buffer
   pays for the larger sample, the ablations and the E3 family on the main sample) and is
   re-priced from the pilot's measured tokens/s. `python scripts/kaggle_run_plan.py --list` shows
   the expansion.
3. **The eval CLI** (`noilai/eval`, `scripts/run_eval.py`):
   `scripts/run_eval.py --model-config <name> --items <file> --tasks T1 T2 T3 --variants V1 V2 V3 V4
   --paraphrases p0 p1 p2 --shots 3 --arms nfc [--input-format F] [--instruction I] [--language L]
   [--limit N] [--in-core-only] --resume --run-id <id>__<name> --out-root data/runs`
   (the run directory is `data/runs/<id>__<name>/`; `run_plan.yaml`'s `run_eval_out_style` can switch the
   driver back to a single `--out <dir>` flag). `kaggle_run_plan.py` builds exactly that command per
   (run, model) — a test checks every emitted flag against `run_eval.py --help` — refuses an API model
   without `--in-core-only`, the sealed split for any API, a canary-bearing (test-split) file for a
   provider whose `terms.trains_on_inputs` is not false (DD 11.2; a recorded paid-key opt-out adds
   `--core-to-training-provider-opt-out` instead) and a `--limit` outside a smoke line (a refused
   model is reported as *refused* and the loop goes on); skips a self-hosted model whose `revision`
   is unpinned on a non-smoke line (DD 7.1); gates every run on its item file
   (`kaggle_verify_items`: hash, header record, canary, counts; derived files are materialized
   first); skips models whose configured hardware the session cannot run (`--session-hardware`)
   unless `--allow-hardware-mismatch`, and logs the SESSION's device type and count; launches no
   new model past `--max-session-hours` (DD 7.6); writes a provisional 0-hour compute-log row at
   each model's start and the final row at its end; calls an `after_each` hook (the notebooks
   push the outputs to the private dataset after every model, and a thread pushes them every
   `PUSH_EVERY_MINUTES` while a model runs); passes `--account-holder <role>` from the notebook's
   `ACCOUNT_HOLDER_ROLE` (DD 11.2); and keeps `data/runs/api_ledger.json` (requests AND tokens per
   model per UTC day, from the new rows' `n_prompt_tokens + n_output_tokens`; a thinking variant
   shares its base entry's allowance) so that a Groq model at its 1,000-requests or
   200K-tokens/day cap is skipped until tomorrow — a running command is interrupted by a watchdog
   the moment the cap is reached and reported as *parked*, not failed.
4. **Notebooks** (`notebooks/`, generated by `scripts/kaggle_build_notebooks.py`; edit the
   generator, run `--write`, commit both — `tests/test_cloud.py` refuses a hand-edited copy):
   * `kaggle_eval_t4.ipynb`: environment check → clone (git bundle from a private dataset, or
     GitHub with a token from Kaggle Secrets through GIT_ASKPASS; the fetched `origin/<ref>` is
     checked out before any stale local branch) → **restore** the previous session's `data/runs`
     (outputs, manifests, API ledger, compute log) from the private `noilai-runs` dataset so that
     `--resume` continues instead of restarting at item 0 → pinned install per DD 7.3 (vLLM 0.30.0
     into a venv under `/kaggle/tmp`, or `kaggle-vllm`; the version is a flagged parameter until the
     smoke tests pin it; `pyproject.toml`'s `vllm` extra carries the same pin) → resources +
     item-file gate (keys derived from the run ids) → smoke test → **RUN cell** under the session
     guards with `--resume` → outputs to `/kaggle/working` and a private Kaggle dataset every 30
     minutes and after every model → GPU-hours log.
   * `kaggle_eval_tpu.ipynb`: the same on the TPU v5e-8 (vLLM-TPU, tensor parallel 8) for Gemma 3
     12B, the two 27B models, the bf16 drift check and the 27B reasoning run; its smoke line is
     `smoke_20_tpu`, booked under TPU hours.
   * `colab_probe_gemma3.ipynb`: E4 skeleton — Gemma 3 1B in fp32 on one T4, or the 4B as the
     text-only `Gemma3ForCausalLM` sharded over Kaggle's 2×T4 (`N_GPUS = 2`; DD 9.5),
     `noilai.probe.extract` + `probes` (tone/onset/rime, NFC vs NFD, syllable-disjoint split,
     control tasks) and the patching cell on the DD 9.3-protocol pair *công tử* / *công tự* under V3
     with E1's frozen prompt (the pair the paper uses: it passes alignment filter (1) under Gemma 3,
     where DD 9.3's example *bí mà* / *bí mạ* does not — *mạ* tokenizes to two pieces; every output
     computed by the rule engine, never typed); the two `e4`-tagged cells are the ones to edit.
   * `api_runs.ipynb`: the API lines (core-derived files) with the keys from the environment /
     Colab userdata / Kaggle Secrets; the Groq models run, the Gemini models are *refused* on every
     canary-bearing file until the author records a paid-key opt-out or adds the dev-derived set
     (DD 11.2 / 13.18); daily-quota loop with `--resume` over the runs tree restored from Drive;
     `GEMINI_RPD_OVERRIDE` / `GEMINI_TPD_OVERRIDE` are applied to the in-memory config and recorded
     in the ledger, and a provider with no cap and no override is refused unless
     `ALLOW_UNCAPPED_API` (no notebook rewrites `configs/`).
   Every cell is plain Python (no magics, no `input()`); the first code cell is the only one to
   edit. Kaggle Secrets to define: `GITHUB_TOKEN` (only without a bundle), `HF_TOKEN`,
   `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.
5. **Compute log** (checklist C1): `python scripts/compute_log.py totals` on
   `data/compute_log.csv` (`date, platform, gpu_type, n_gpus, hours, run_id, purpose`; the path is
   the plan's `compute_log` key and awaits the author's confirmation). The driver writes one row
   per executed model run with the session's device count; the notebooks add a session-overhead
   row; merge the copy that comes back from Kaggle into the repository's file.

Reproducibility: every run directory holds `outputs.jsonl`, `scores.jsonl` and a `manifest.json`
with model id and revision, backend and version, dtype, quantization, engine flags, seed, prompt
and item hashes, canary check, hardware, start/end, GPU-hours, resource hashes, git commit and
dirty flag (`docs/DATA_FORMAT.md`). Notebooks record `pip freeze` and the CUDA/TPU environment next
to the outputs (`data/runs/env/`, checklist C4).

## Statistics and probes

* `noilai.stats` (present): `bootstrap` (clustered by base pair; paired differences),
  `tests` (exact/mid-p McNemar, Holm within declared families), `mixed` (Bayesian mixed GLM +
  GEE for E2), `mediation` (the Δtokens dose–response decomposition and the zero-dose contrast,
  labelled descriptive per DD 8.6 (2)–(3); no "share mediated by token count" is estimated or
  reported, DD 12.7 — the module keeps its name because the pre-registration refers to it),
  `power`, `agreement` (Krippendorff's α with bootstrap CI), `tables` (LaTeX fragments; no number
  in the paper is typed by hand). `scripts/score_run.py` produces the `scores.jsonl` these read; a
  `scripts/run_stats.py` driver over `data/runs/*/scores.jsonl` is **planned**.
* `noilai.probe` (present): `extract` (residual stream at the syllable's last sub-token / the token
  after it, NFC and NFD), `probes` (layer-wise logistic probes, syllable-disjoint split, control
  labels, selectivity), `patching` (residual cache, layer/position patching, recovery of the logit
  difference; the difference-in-means steering code stays but steering is future work, DD 9.4).
  `scripts/run_probe.py` is the batch driver; the Colab notebook `colab_probe_gemma3.ipynb` is the
  interactive skeleton for the same calls.

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

The GPL lists are consulted at build time (legality, lexicality, frequency proxies) and the items
are produced by the rule engine, but the lexical base pairs ARE Viet74K entries: 1,183 two-syllable
Viet74K entries appear verbatim as inputs or gold of the released lexical items in v0.3 (6,012 items
across dev and test; counted from the release files, `tests/test_cloud.py` keeps this number
current), and are released under the terms `docs/DATA_STATEMENT.md` §H and DD 4.1 / 11.1 / 12.22
state: written permission from the lists' authors (requested 30 September 2026) or, failing a reply
by the data freeze, GPLv2 with attribution for the item files. The Hunspell syllable inventory is
not redistributed. Release licenses: the dev split under CC BY 4.0, the gated test split under
CC BY-NC-ND 4.0 with a canary string; a sealed, regenerated split never reaches any API (Gemini's
free tier trains on inputs and requires users to be 18 or older, so it never receives the core;
Groq's and Gemini's terms are quoted in `configs/models.yaml`). Model outputs quote test items, so the Kaggle
dataset that carries `data/runs` is private; its licence label (CC BY-NC-ND 4.0 is the current
default in `scripts/kaggle_dataset.py`) awaits the author's confirmation together with the
release story of DD 13.19.

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
* **2026-09-30 (cloud kit review)** — The item-file gate accepts and checks the BIG-bench header
  record (DD 4.6); the run plan points at `data/release/v0.3` through one `release:` key, replaces
  every `--limit` sample by a seeded file (`noilai_main` from `scripts/sample_items.py`; derived
  core subsets and the two pilot files from `derive` blocks), adds the explicit-input main result
  (DD 12.17), the four prompt ablations, the E3 C2-enriched line and the TPU smoke line, and
  re-budgets the compute table within 70–150 GPU-h; `models.yaml` gains `tensor_parallel_size: 8`
  on the TPU, the DD 7.3 engine defaults, `revision` on every self-hosted entry, PhoGPT on HF
  transformers, per-entry API pacing and both readings of the Groq scope; the driver gains the
  token ledger with a budget watchdog, the session hardware and time guards, per-model pushes and
  the pre-run item gate; the notebooks install vLLM per DD 7.3, derive their verify keys from the
  run ids, check out the fetched remote ref first, keep config overrides in memory and rebuild the
  E4 patching cell from the rule engine on the DD 9.3 pair. Still no model run.
* **2026-09-30 (cloud-kit review round)** — The run plan's C2 arm is `placement_new` on every line
  (the release is stored old-style; `placement_old` changed no prompt and the runner refused it),
  E3 NóiLái runs the DD 8.3 family (`nfd`, `pc`, `strip_tones`, `strip_all`; T1 and T3) on the
  main sample, C2 only on the enriched file, no C2 on XCOPA, every E3 test-split line after
  8 November, E4 reported regardless of outcome, two DD 6.1/6.2 agreement lines
  (`E3_scope_item`, `E3_engine_hf`) on a derived 500-item Gemma 3 1B file, the counterfactuals
  line re-budgeted 17–41 from the buffer, the API token total corrected to 6.9M; the driver refuses
  a canary-bearing file for a provider that trains on inputs (Gemini unpaid) and reports it as
  *refused*, skips unpinned revisions (DD 7.1), pools a thinking variant's ledger with its base
  entry, maps `arm_scope` / `backend`, writes a provisional compute-log row per model start;
  `models.yaml` gains Groq/OpenRouter `terms`, a Gemini opt-out flag and DD 11.2 wording; the
  notebooks restore the persisted runs tree before `--resume`, build git's environment after the
  token export (the GIT_ASKPASS fallback was unrunnable), take `ACCOUNT_HOLDER_ROLE`,
  `ALLOW_UNPINNED_REVISION`, `ALLOW_UNCAPPED_API` and push every 30 minutes; the E4 pair is the
  aligned *công tử* / *công tự*; `record_environment` probes the engine interpreter; `pyproject`
  pins vLLM 0.30.0 and llama-cpp-python 0.3.35; the Makefile builds v0.3 with the sampled files and
  refuses the superseded 20 × 40 baseline forms; `data/validation/` and `data/human/` are ignored.
  Still no model run.

## Origin

This repository was split out of a branch of `BEEEEEEBAGON/ntcf` (an unrelated paper) on 1 October 2026
with its full history; `docs/MIGRATION.md` records the original head (`83c9bb6`) and the commit map.
Documents and manifests that cite commit hashes cite the original ones.
