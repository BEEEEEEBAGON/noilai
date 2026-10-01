# Zero-budget compute plan (1 October 2026)

Everything here costs nothing: Kaggle's free CPU, GPU (2×T4) and TPU sessions on one account, free API tiers with
daily caps, no card on file anywhere. The run matrix is `configs/run_plan.yaml` (unchanged). This document says how it
is cut into sessions (`scripts/plan_chunks.py` → `configs/compute_chunks.yaml` + `notebooks/chunks/*.ipynb`), in what
order, and what the free tier can and cannot deliver before the results freeze (Gate 3, 22 November). **No number in
this file is a model result.** Hours are the plan's line estimates (author's estimates, re-priced after the smoke week
per DESIGN_DECISIONS 8.5 item 30) or derived from the CPU benchmark RL-2026-10-01-06 (random weights; an engineering
measurement).

## 1. Quotas: what is verified and what is not

Kaggle's pages could not be opened from the build machine on 1 October 2026 (`www.kaggle.com` is blocked by its
network proxy), and the web-search budget of this session was spent on the attested-example sweep. **None of the Kaggle
figures below is verified**; they come from `configs/run_plan.yaml` `quota`, which cites dated secondary sources (plan
section 3). Reading them takes one minute on the account's settings page (BLOCKED.md).

| figure | value used | source | status |
|---|---|---|---|
| GPU quota | 30 GPU-h a week | AIMultiple (plan §3) | [UNCERTAIN: verify] |
| 2×T4 accounting | one quota hour per session hour (x1) **or** two (x2) | — | [UNCERTAIN: verify]; both readings are carried in every table |
| GPU session limit | 12 h (≤ 9 h of work, DD 7.6) | plan §3 | [UNCERTAIN: verify] |
| TPU quota / session | 20 TPU-h a week, one v5e-8 session at a time, 9 h | Medium, kaggle-tpu-lab (plan §3) | [UNCERTAIN: verify] |
| quota reset | weekly; weekday unknown | — | [UNCERTAIN: verify] |
| CPU session | 12 h; ~4 cores, ~30 GB RAM | secondary | [UNCERTAIN: verify]; the CPU notebook checks RAM before loading |
| vLLM-TPU startup | 6–22 min per model | kaggle-tpu-lab (plan §3) | secondary |

## 2. The CPU pilot: can it test the floor risk?

**Yes, for the three pilot models, and only for them.** The Gate 1 criterion (PREREGISTRATION §10) is computed on the
pilot models (`gemma-3-1b-it`, `qwen3.5-2b`, `phogpt-4b-chat`), all small enough for float32 on a CPU session. If all
three score at the floor on generated T1 in both arms, the CPU run shows it as well as a GPU run would; the exploratory
forced choice (`docs/FORCED_CHOICE_EXPLORATORY.md`) then says whether there is graded signal under the floor. It cannot
say anything about the 7–27B models (fp32 weights exceed the CPU session's RAM), and it gives no GPU tokens/s, which DD
8.5 needs to re-price the GPU lines; the smoke week does that.

**Cost** (`scripts/plan_chunks.py` `cpu_estimates`, from RL-2026-10-01-06 on 4 cores, float32): decode
≈ 0.11 s/token; a worst-case request (64 new tokens) ≈ 10.0 s after a 520-token NFC prompt and ≈ 12.0 s after a
950-token NFD prompt; forced choice ≈ 4.2 s (NFC) and 6.7 s (NFD) per item with 8 candidates on the shared prompt cache.
Per pilot model: smoke_20 + pilot_t1_200 (both arms, generation and forced choice) + pilot_xcopa_200 (bounded by a
520-token prompt):

| model | scale (≈ parameters / gemma-3-1b-it) | worst-case hours | sessions |
|---|---|---|---|
| gemma-3-1b-it | 1.0 | 2.3 | 1 |
| qwen3.5-2b | 2.0 [UNCERTAIN: size unverified] | 4.6 | 1 |
| phogpt-4b-chat | 3.7 | 8.5 | 1 (≤ 10 h guard); `FORCED_CHOICE = False` saves ~2.4 h |

Caveats, all pointing the same way (the table is an upper bound on the arithmetic, not a promise): answers usually stop
well before 64 tokens; PhoGPT's Vietnamese tokenizer makes its prompts shorter; Kaggle's CPUs may be slower or faster
than the build machine's [UNCERTAIN]. Running the pilot lines on CPU through the HF backend is a logged deviation
(DEVIATIONS, "pilot lines on CPU (proposed)"): the engine differs from the T4/vLLM lines, so the first GPU smoke re-runs
the same 20 smoke items on the T4 and compares outputs row by row; a disagreement is reported, not averaged away.

**What CPU decides and what it does not:**

| question | CPU pilot | needs a GPU |
|---|---|---|
| does the harness run end to end on real weights (outputs → scores → stats → hashes)? | yes (`c01`) | — |
| are the pilot models at the floor on T1 (Gate 1)? | yes | — |
| is there graded signal under the floor (exploratory forced choice)? | yes, pilot models | larger models |
| does vLLM give the same outputs as HF on the T4? | — | first smoke (20 items, row by row) |
| GPU tokens/s for re-pricing (DD 8.5) | — | smoke week |
| anything about 7–27B models | — | E1 |

## 3. Forced choice as a logged exploratory analysis

The DD has no multi-distractor T1 forced choice (it has `t3_pair_lp`). It is added as an **exploratory** analysis,
pre-specified on 1 October before any model output (`docs/FORCED_CHOICE_EXPLORATORY.md`; DEVIATIONS row "exploratory T1
forced choice"; memo row N4 asks you to list it in PREREG §8.14 before the stage-2 commit). It never enters a
hypothesis test or the Gate 1 decision. It is on in the CPU pilot chunks; on the GPU it is a flag (`--t1-forced-choice`)
you may add to the E1 chunks through `EXTRA_ARGS` (cost ≈ one prompt pass + 8 short continuations per T1 item).

## 4. Chunks: one resumable notebook per Kaggle session

`python scripts/plan_chunks.py --write` cuts every GPU/TPU line of the run plan into sessions and writes one notebook per
chunk: `notebooks/kaggle_eval_t4.ipynb` (or the TPU / CPU notebook) with its parameters cell preset to an ordered job
list `JOBS = [{run, models, overrides, tag}]`. Each chunk notebook is resumable: `run_eval.py --resume`, outputs
pushed to the private runs dataset every 30 minutes and after each model, `check_run.py` hashes every run directory
(`results_hashes.json`). Re-run the same notebook until every job reports `ok`; then go to the next chunk.
`tests/test_cloud.py` checks that the chunk notebooks and this file's generated block match the generator.

**Order = the reverse of the pre-registered cut order**, so that if the quota runs out, what is left undone is what
the pre-registration says to cut first (DD 7.1: "tokenizer-redundant models first (Gemma 3 12B, Qwen3.5-0.8B, Gemma 4
12B), then the second model of any family, then seeds/paraphrases per 8.5"; DD 8.5: above 120 GPU-h, "E3's full arms
are capped to the C1-informative families and paraphrases drop to two for the TPU/2×T4 models before any model is
cut"):

| tier | what | when cut |
|---|---|---|
| 0 | first run, pilot, smoke week (dev split only) | never; the only lines allowed before the stage-2 commit |
| 1 | the first model of each family on every main line (E1 main, explicit input, attested; E3 NóiLái, XCOPA, C2; the DD 6.1/6.2 checks) | never in the cut order |
| 2 | ablations, the reasoning sub-study, bf16 drift of tier-1 models | not in the cut order (see question Q-C3) |
| 3 | the second model of a family | DD 7.1, second |
| 4 | Gemma 3 12B, Qwen3.5-0.8B, Gemma 4 12B | DD 7.1, first |
| 5 | p2 of the TPU / 2×T4 models; E3 of families the census marks normalizing | DD 8.5, before any model |

Rules the generator applies (each a scheduling choice, listed for your review):

- **First model of a family** = the pilot model of the family if it has one, else the first entry of
  `configs/models.yaml` (`gemma-3-1b-it`, `qwen3.5-2b`, `gemma-4-e2b`, `gemma-sea-lion-v4.5-e2b-it`, `sailor2-8b-chat`,
  `phogpt-4b-chat`, `vistral-7b-chat`, `llama-3.1-8b-instruct`, `qwen3.8-27b`): nine open models over at least four
  tokenizer families, which meets the minimum viable panel of DD 7.1. The DD does not say which model of a family is
  "second"; `qwen-sea-lion-v4.5-27b-it` shares the `sea-lion-v4.5` family label with the E2B model although its
  tokenizer is Qwen's (question Q-C2).
- **Estimates:** a line's estimate split equally over its models; E1_main's split excludes the TPU trio, which
  `tpu_main` books. A TPU model's share of a gpu_hours line runs on the TPU and is booked on no plan line (the
  "unbooked" total below; question Q-C4).
- **p2 split:** for the TPU / 2×T4 models, the lines with three paraphrases run p0+p1 in their tier and p2 in tier 5,
  each in its own run directory (`<run>__<model>__p0p1`, `__p2`); the analysis concatenates them.
- **Normalizing families:** when `data/audit/<family>.json` marks a tokenizer as normalizing NFD, its E3 jobs move to
  tier 5, and the runner already drops the arms that are 0 by construction (DD 6.2, `census_skipped_arms`). Today only
  Gemma 3 is censused, so nothing moves; re-run `--write` after `audit_tokenizers.py` has run on the pinned tokenizers.
- **Packing:** tier-pure, model-major, at most 8 h (GPU), 7 h (TPU) or 10 h (CPU) of high estimate per chunk; a session
  overhead of 0.5 h (install, loads, pushes) is assumed in the capacity table [UNCERTAIN: measure in the smoke week].
  TPU chunks that load many models (bf16 drift) can spend 1–3 h on vLLM-TPU startups alone (6–22 min each); the 8-h
  guard stops launching models and the next session resumes.
- **Not chunked:** the API lines (bounded by daily caps; `notebooks/api_runs.ipynb` re-run daily; `docs/API_ROUTES.md`)
  and E4 (window 23 Nov–6 Dec, after the results freeze; the probe notebook is a skeleton without resume and the E4
  protocol is a stage-2 item; BLOCKED.md).

<!-- plan_chunks:begin (generated by scripts/plan_chunks.py; do not edit by hand) -->
**Chunks, in run order** (estimates: plan line estimates split equally per model; not measurements):

| # | chunk | queue | tier | what | est. h (low–high) | quota h x1 | quota h x2 |
|---|---|---|---|---|---|---|---|
| 1 | `c01_cpu_t0` | cpu | 0 | first real run on CPU (smoke_20, 20 dev items) | 0.11–0.11 | 0–0 | 0–0 |
| 2 | `c02_cpu_t0` | cpu | 0 | Gate 1 pilot on CPU: gemma-3-1b-it | 2.3–2.3 | 0–0 | 0–0 |
| 3 | `c03_cpu_t0` | cpu | 0 | Gate 1 pilot on CPU: qwen3.5-2b | 4.61–4.61 | 0–0 | 0–0 |
| 4 | `c04_cpu_t0` | cpu | 0 | Gate 1 pilot on CPU: phogpt-4b-chat | 8.52–8.52 | 0–0 | 0–0 |
| 5 | `c05_gpu_t0` | gpu | 0 | Gate 1 pilot on the T4 (the alternative to the CPU chunks; free quota before the smoke week) | 2–3 | 2–3 | 4–6 |
| 6 | `c06_gpu_t0` | gpu | 0 | smoke week: every T4 / 2xT4 model on 20 dev items (panel freeze input) | 2–2 | 2–2 | 4–4 |
| 7 | `c07_tpu_t0` | tpu | 0 | smoke week: the TPU trio on 20 dev items | 1–1 | 1–1 | 1–1 |
| 8 | `c08_gpu_t1` | gpu | 1 | models: gemma-3-1b-it | 4.11–6.62 | 4.11–6.62 | 8.22–13.24 |
| 9 | `c09_gpu_t1` | gpu | 1 | models: qwen3.5-2b | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 10 | `c10_gpu_t1` | gpu | 1 | models: gemma-4-e2b | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 11 | `c11_gpu_t1` | gpu | 1 | models: gemma-sea-lion-v4.5-e2b-it | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 12 | `c12_gpu_t1` | gpu | 1 | models: sailor2-8b-chat | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 13 | `c13_gpu_t1` | gpu | 1 | models: phogpt-4b-chat | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 14 | `c14_gpu_t1` | gpu | 1 | models: vistral-7b-chat | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 15 | `c15_gpu_t1` | gpu | 1 | models: llama-3.1-8b-instruct | 2.11–4.62 | 2.11–4.62 | 4.22–9.24 |
| 16 | `c16_tpu_t1` | tpu | 1 | models: qwen3.8-27b | 2.05–4.5 | 2.05–4.5 | 2.05–4.5 |
| 17 | `c17_gpu_t2` | gpu | 2 | models: gemma-3-1b-it, qwen3.5-2b, gemma-4-e2b, gemma-sea-lion-v4.5-e2b-it, sailor2-8b-chat, phogpt-4b-chat, vistral-7b-chat, llama-3.1-8b-instruct | 1.88–3.76 | 1.88–3.76 | 3.76–7.52 |
| 18 | `c18_gpu_t2` | gpu | 2 | models: qwen3.5-4b--thinking | 2.5–5.0 | 2.5–5.0 | 5.0–10.0 |
| 19 | `c19_gpu_t2` | gpu | 2 | models: qwen3.5-9b--thinking | 2.5–5.0 | 2.5–5.0 | 5.0–10.0 |
| 20 | `c20_tpu_t2` | tpu | 2 | models: qwen3.8-27b, gemma-3-1b-it--bf16, qwen3.5-2b--bf16, gemma-4-e2b--bf16, gemma-sea-lion-v4.5-e2b-it--bf16, sailor2-8b-chat--bf16, vistral-7b-chat--bf16, llama-3.1-8b-instruct--bf16 | 1.74–3.97 | 1.74–3.97 | 1.74–3.97 |
| 21 | `c21_tpu_t2` | tpu | 2 | models: qwen3.8-27b--thinking | 2.0–4.0 | 2.0–4.0 | 2.0–4.0 |
| 22 | `c22_gpu_t3` | gpu | 3 | models: gemma-3-4b-it | 2.01–4.43 | 2.01–4.43 | 4.02–8.86 |
| 23 | `c23_gpu_t3` | gpu | 3 | models: qwen3.5-4b | 2.34–5.09 | 2.34–5.09 | 4.68–10.18 |
| 24 | `c24_gpu_t3` | gpu | 3 | models: qwen3.5-9b | 2.34–5.09 | 2.34–5.09 | 4.68–10.18 |
| 25 | `c25_gpu_t3` | gpu | 3 | models: gemma-4-e4b | 2.34–5.09 | 2.34–5.09 | 4.68–10.18 |
| 26 | `c26_tpu_t3` | tpu | 3 | models: qwen-sea-lion-v4.5-27b-it, gemma-3-4b-it--bf16, qwen3.5-4b--bf16, qwen3.5-9b--bf16, gemma-4-e4b--bf16 | 3.14–6.97 | 3.14–6.97 | 3.14–6.97 |
| 27 | `c27_gpu_t4` | gpu | 4 | models: qwen3.5-0.8b | 2.34–5.09 | 2.34–5.09 | 4.68–10.18 |
| 28 | `c28_gpu_t4` | gpu | 4 | models: gemma-4-12b | 2.01–4.43 | 2.01–4.43 | 4.02–8.86 |
| 29 | `c29_tpu_t4` | tpu | 4 | models: gemma-3-12b-it, qwen3.5-0.8b--bf16, gemma-4-12b--bf16 | 2.71–5.97 | 2.71–5.97 | 2.71–5.97 |
| 30 | `c30_gpu_t5` | gpu | 5 | models: gemma-3-4b-it, gemma-4-12b | 0.66–1.32 | 0.66–1.32 | 1.32–2.64 |
| 31 | `c31_tpu_t5` | tpu | 5 | models: gemma-3-12b-it, qwen3.8-27b, qwen-sea-lion-v4.5-27b-it | 1.39–2.78 | 1.39–2.78 | 1.39–2.78 |

**Totals by queue and tier:**

| queue | tier | chunks | est. h low | est. h high | + session overhead (assumed) |
|---|---|---|---|---|---|
| cpu | 0 | 4 | 15.5 | 15.5 | 17.5 |
| gpu | 0 | 2 | 4.0 | 5.0 | 6.0 |
| gpu | 1 | 8 | 18.9 | 39.0 | 43.0 |
| gpu | 2 | 3 | 6.9 | 13.8 | 15.3 |
| gpu | 3 | 4 | 9.0 | 19.7 | 21.7 |
| gpu | 4 | 2 | 4.3 | 9.5 | 10.5 |
| gpu | 5 | 1 | 0.7 | 1.3 | 1.8 |
| tpu | 0 | 1 | 1.0 | 1.0 | 1.5 |
| tpu | 1 | 1 | 2.0 | 4.5 | 5.0 |
| tpu | 2 | 2 | 3.7 | 8.0 | 9.0 |
| tpu | 3 | 1 | 3.1 | 7.0 | 7.5 |
| tpu | 4 | 1 | 2.7 | 6.0 | 6.5 |
| tpu | 5 | 1 | 1.4 | 2.8 | 3.3 |

**Capacity between the stage-2 commit and the results freeze** (quota 30 GPU-h and 20 TPU-h a week, `configs/run_plan.yaml` `quota`, secondary sources [UNCERTAIN: verify in Kaggle's settings]; one account):

| stage-2 commit | test-split weeks to 22 Nov | GPU quota h (x1) | GPU clock h (x2) | TPU h | GPU tiers 1-5 fit at x1 (high + overhead) | at x2 |
|---|---|---|---|---|---|---|
| 8 Nov (DD 8.8 latest) | 2 | 60 | 30 | 40 | through order 19 (tier 2) | through order 12 (tier 1) |
| 1 Nov | 3 | 90 | 45 | 60 | through order 27 (tier 4) | through order 15 (tier 1) |
| 25 Oct (right after the panel freeze: the earliest) | 4 | 120 | 60 | 80 | all (through tier 5) | through order 19 (tier 2) |

**Unscheduled (no zero-cost route):** `bf16_drift_200` × `phogpt-4b-chat--bf16` (0.21–0.5 h)

**Unbooked TPU time:** the TPU models' equal shares of gpu_hours lines, 4.2–9.7 TPU-h, are booked on no line of `plan_lines` (the plan's `tpu_models_and_bf16` line covers only tpu_main, the TPU smoke, reasoning and bf16 drift).

<!-- plan_chunks:end -->

## 5. The schedule is bound by Gate 2, not by the GPU

DD 8.8 allows no test-split run before the stage-2 commit (latest 8 November) and none before the validators' generator
fixes are in; the results freeze is 22 November (Gate 3). With the stage-2 commit on 8 November, all test-split GPU work
has two weeks: at 30 GPU-h a week that is 60 quota hours under the x1 reading and **30 session hours under x2**, against
the GPU tier totals of section 4 (high estimate plus assumed overhead). The capacity table above shows the consequence: at x2 and
8 November, the run stops inside tier 1. The TPU (20 h a week) is not binding.

Three levers, none of which spends money:

1. **Commit stage 2 earlier** (memo row N13): right after the panel freeze (25 October) and the validators' fixes, the
   window is four weeks and everything fits at x1, tiers 1–2 at x2. The cost: stage 2 is committed before the
   co-author has edited it unless one has joined by then (DD 8.8 commits it on 8 November "anyway" if nobody has).
2. **Use the weeks before stage 2 for everything that is not test-split:** tier 0 (CPU pilot, GPU pilot, smoke), the
   tokenizer census, `pin_panel.py` on Kaggle, the bf16 / thinking serving variants' smoke, and E3 on dev (DD 8.8 allows
   it in the week of 26 October).
3. **Colab's free T4** as extra capacity for the lines that are not tied to Kaggle datasets (E4 already supports Colab;
   any chunk notebook runs there with the Drive paths of `notebooks/api_runs.ipynb`). Colab's free quota is not
   published [UNCERTAIN]; it is a bonus, not a plan line.

E4 (5–15 GPU-h on 2×T4, plan estimate) has its own window, 23 November–6 December, two quota weeks with nothing else
queued.

## 6. What changed against the run plan

- **Modal L4 fallback dropped.** `configs/run_plan.yaml` keeps a Modal L4 fallback for bf16 drift ("card on file").
  This project enters no payment details, so the route does not exist; `phogpt-4b-chat--bf16` (MPT, not runnable on
  vLLM-TPU) is therefore unscheduled. Options (question Q-C1): (a) drop PhoGPT from the bf16 drift appendix and name
  it; (b) an fp32 reference on a Kaggle CPU session through HF (unquantized, a different precision: a logged
  deviation); (c) bf16 through HF on the T4, which has no native bf16 (slow, emulated) [UNCERTAIN]. DEVIATIONS row of
  1 October.
- **Chunk run directories.** A job that narrows its line (only the p2 split today) writes `<run>__<model>__<tag>`
  (`kaggle_run_plan.narrow_run`); the check cell follows the driver's output directory. DEVIATIONS row of 1 October.
- Nothing in `configs/run_plan.yaml`, the DD or the pre-registration is edited by this plan.

## 7. Running a chunk (checklist)

1. Kaggle: attach `noilai-bundle`, `noilai-release` (private), and from the second session on `noilai-runs`; Secrets
   `HF_TOKEN` (Gemma, Llama, Vistral licences accepted on that HF account), `KAGGLE_USERNAME`, `KAGGLE_KEY`.
2. Open `notebooks/chunks/<chunk>.ipynb`, set `ACCOUNT_HOLDER_ROLE`, accelerator as the chunk says (None / GPU T4 ×2 /
   TPU v5e-8), *Save & Run All* (commit mode runs without the browser open).
3. When it ends: if any job is not `ok`, run the same notebook again (it resumes). Then `python scripts/compute_log.py`
   for the hours actually used; after the first chunk of each tier, re-run `plan_chunks.py --write` if the measured
   hours change the packing (DD 8.5 re-pricing).
