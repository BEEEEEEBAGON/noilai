# STATUS — experiments and validation workstreams

Hand-maintained sections first (dated **2026-10-16**); the meter block at the end is written by `scripts/progress.py`
and is never edited by hand. Blockers, in deadline order, are in `BLOCKED.md`; the author's rulings are in
`docs/gate1/DD13_DECISION_MEMO.md` (reply with the row numbers you accept).

## Session log (newest first)

- **2026-10-17 (Vistral: proceed without it).** Hugging Face access as of 17 Oct: Gemma accepted, Llama 3.1 pending
  with Meta (the plan loads the ungated AWQ checkpoint, so nothing waits on it), `Viet-Mistral/Vistral-7B-Chat` pending
  with the owner (manual, no date). The author decided to proceed without Vistral (DEVIATIONS 2026-10-17; the
  `vistral-7b-chat` note in `configs/models.yaml`; `BLOCKED.md` 2, 22): the pin leaves it unpinned, the driver skips
  or fails it and goes on; dropped from the panel at the freeze (25 Oct) if still pending; the analysis then has six
  tokenizer families (above every floor). No run code changed. The v0.4 release and the box kit were re-sent to the
  author as two zips. Still no model run.
- **2026-10-16, later (the release rebuilt as v0.4; the first paid run lost).** The author rented 2 h on the 2x A800
  box and ran `scripts/box/laptop_start_nogit.sh`; the frozen item files were on no machine the author could reach,
  so the box ran only what needs none, and the rental ended before the hand-back ran: nothing came back (no result,
  no pin, no audit). The frozen v0.3 release and its private manifest (seed, canary GUID) could not be found anywhere;
  the history's pre-redaction seed (the September Makefile's default) does not reproduce v0.3's content hash at the
  freeze commit, so the freeze had drawn its own seed. Rebuilt as **v0.4** (DEVIATIONS row 2026-10-16): same generator and arguments, a new
  private seed, 10,000 items / 2,500 base pairs / the same core counts; `configs/run_plan.yaml` `release` → v0.4 with
  every item-file hash re-recorded (six base files, five derived files; `pilot_xcopa_200` unchanged);
  `data/release/v0.4/manifest.json` committed; the eight files sent to the author (two private copies to keep;
  `BLOCKED.md` 3). The next rental runs tier 0 whole once the files are on the author's Mac. Not done: the PREREG /
  RESULTS_LOG / paper references to v0.3 (author's edits), the `noilai-release` dataset.
- **2026-10-16 (box-kit session, adoring-planck; no Kaggle or Hugging Face access).** Merged before this session:
  PR #5 (`036f603`), the rented-CUDA-box kit (`scripts/cuda_box.py`, `kaggle_run_plan.CudaSession`, `cuda_fit`,
  `tensor_parallel_for`; DEVIATIONS row of 2026-10-12). The author bought about 24 h on a 2x NVIDIA A800-SXM4-40GB
  box (80 GB VRAM, 48 GB RAM, 200 GB disk, $1.33/h, a $33 credit). This session makes the box run the CHUNK PLAN
  instead of whole plan lines (DEVIATIONS row of 2026-10-16):
  - `scripts/cuda_box.py --chunks` (the default `all` = every GPU / TPU chunk `c05`–`c31` in plan order; `tier0`;
    `from:<id>`; the CPU chunks `c01`–`c04` are refused) runs each chunk's jobs through `kaggle_run_plan.execute`
    with the job's tag, so the run directories are the ledger's units; after every executed model: `check_run.py`
    (`--no-check`), the push of `data/runs` + `data/compute_log.csv` to the private Kaggle dataset
    (`--kaggle-dataset noilai-runs`, `--create-dataset` once; `--sync-cmd` in the background) and the chunk-scoped
    purge of the model's weights (`--keep-weights` turns it off), with the next chunk's weights prefetched meanwhile
    (`--no-prefetch`); `--lanes 2` schedules the two cards tier by tier (one-card chunks in parallel under
    `CUDA_VISIBLE_DEVICES`, the 27B chunks `c16`, `c20`, `c21`, `c26`, `c31` on both cards); the summary counts the
    gated models apart (`skipped: gate: ...`).
  - `scripts/ledger.py ingest` maps an untagged whole-line directory of a split line (`E1_main__gemma-3-4b-it`,
    `E1_main__qwen3.8-27b`, `tpu_main__<model>`: the `--runs` mode) to its `__p0p1` / `__p2` units with a note.
  - Documents: `RUNBOOK.md` §6 (the box, click by click; the never-list is now §7), the README box paragraph, the
    DEVIATIONS row, this file, `BLOCKED.md` 14 / 19 / 33 (the box as the primary route of the pilot and the smoke
    week; the lanes command after the gate), `CLAUDE.md` step 3.
  - What did NOT happen: still no model run, nothing ingested, the meter not re-run (nothing to ingest);
    `www.kaggle.com` and `huggingface.co` unreachable from this environment; the frozen item files absent here; the
    gate file all null, so on the box only tier 0 runs today. Fit on the box (`cuda_fit`): a 27B model in bf16
    ≈ 60.4 GB → both cards (tensor parallel 2); `gemma-3-12b-it` ≈ 27.8 GB → one card; every other entry one card.
    Before the box runs anything: `RUNBOOK.md` §6 steps 1–4 (the frozen files `scp`'d in, the pin with `--partial`
    on the box, the audits), then `--chunks tier0` (step 5).
- **2026-10-12 (analysis-code session, adoring-planck; no Kaggle or Hugging Face access).** Nothing ingested, nothing
  launched: `www.kaggle.com` and `huggingface.co` are still unreachable from this environment, no token exists, the
  frozen item files are still absent (`data/release/v0.3/` holds `manifest.json` only); the prerequisites table is
  unchanged; the meter was not re-run (nothing to ingest). Built the analysis code of `BLOCKED.md` 34, each module
  through an independent verifier:
  - `noilai/stats/pooling.py` (PREREG 8.1 / DD 1, 8.1): the model-in-tokenizer-family pool, REML with the
    Hartung–Knapp adjustment and the family-level bootstrap always beside it, k of N families, `k_of_n_models`
    (Holm-significant AND on the predicted side) and `tost_paired` (δ = 2 points) for H4; it refuses to run without
    the tokenizer-family map.
  - `noilai/stats/aggregate.py` (PREREG 8.1, 8.2, 8.5, 8.13; DD 5.1, 5.3): run discovery over
    `data/runs/<run>__<model>[__<tag>]` (the `p0p1` + `p2` directories of one line pooled, `cpu` apart), paraphrase
    means over the paraphrases present (never filled), cell tables with the stratified base-pair bootstrap and the
    small-cell rule, the primary metric, τ_m(a) per model and arm with the Table 3 family filtered per model (census
    "normalizes" → 0 by construction, "corrupts" → refused, API → no XCOPA), the explicit-input gap (CIs only), the
    T3 headline without the spelling twins.
  - `noilai/stats/analyze.py` + `configs/analysis.yaml` (PREREG 8.13 / DD 8.8): `python -m noilai.stats.analyze
    --config configs/analysis.yaml --runs data/runs [--ledger experiments/ledger.csv] [--out experiments/analysis]`
    writes 21 .tex + .json artefacts with their provenance, `analysis_summary.json`, `results_log_entry.md` and
    `analysis_report.md`, labelled INTERIM with n; `--label FINAL` is refused (exit 3) until the registration and the
    scoring-rule freeze are recorded and every live Table 2 / Table 3 unit is done and hash-verified; `--out` under
    `paper/` is refused; `--check` refuses stale outputs; the config is refused when it disagrees with
    `noilai/constants.py`. Exercised on the local echo run (`--include-smoke`) and on the real ledger (0 runs; the
    blockers name the 242 live units not done).
  - `noilai.stats.agreement.weighted_precision` / `weighted_precision_from_rows` (DD 10.1): the stratified estimate
    with a Wilson interval on the effective n (finite-population correction); `validation.stratified_precision` still
    reports `wilson_unweighted` and should call it.
  - Not implemented, written by `analyze` as `not computable` with the reason: the H1/E2 item-audit join and floor
    gating, the H3b difference-in-differences and H6 matching helpers, the E4 retention table.
  - Two defects fixed: `noilai/eval/backends.py` recognises the quantization methods as `configs/models.yaml` spells
    them (`bnb_bits`: `bitsandbytes_nf4`, `bnb*`, `int8` → a bitsandbytes load-time request; awq/gptq left to the
    engine; vLLM gets `quantization="bitsandbytes"` instead of the entry's own string), and the tests that read the
    local smoke build take its directory from `NOILAI_SMOKE_BUILD` (`tests/_smoke_build.py`) and skip without it.
    Consequence for the author: `qat_int4` entries are now quantized at load by the backends while
    `scripts/pin_panel.py` treats them as checkpoint-carried (`BLOCKED.md` 22).
  - Author rulings recorded in the handoffs, none edited into PREREG or DD: the seven tokenizer-family labels for
    the 21 panel entries in `configs/analysis.yaml` (one entry's placement is a guess, marked in the file); REML+HK
    as the primary pool with the family bootstrap beside it; standard HKSJ without the max(1, q) truncation; three
    informative families as the H4/H6 floor (five for H3); H4's "adjusted CI excluding 0" read as Holm-significant
    AND on the predicted side; the API models' C2 cell and the never-realizable (placement_new, T3, c2) cell
    (`family_defined` stays 14).
  - Documents: the `docs/DESIGN_DECISIONS.md` parentheticals "(to write)" for the four items now read "implemented
    2026-10-12"; `docs/PREREGISTRATION.md` §2 ("the pooling helper is to write") and §8.13 ("analysis entry point ...
    **planned**") keep their wording, the author's file — the stage-2 edit updates them. DD 10.1 still says
    `cmd_sample` does round-robin allocation: superseded by `make_validation_forms.py packet` (1 Oct), not edited here.
- **2026-10-07 (integration session).** Branch `claude/integrate-2026-10-07` at merge commit `6ee5bb6` =
  `origin/main` + the 1 Oct branch (PR #1, "lucid") + the PR #3 branch ("adoring"), with the paper ports of PR #2
  ("eloquent"). Merged from each, one line each:
  - lucid: the Gate 1 packet and protocols (`docs/gate1/*`, consent form, recruitment messages, validator
    instructions, baseline form + Apps Script, `make validation` / `make baseline`), the decision memo, the 31-chunk
    compute plan (`configs/compute_chunks.yaml`, `notebooks/chunks/c01…c31.ipynb`, `scripts/plan_chunks.py`,
    `docs/COMPUTE_PLAN.md`), chunk narrowing in `scripts/kaggle_run_plan.py`, the CPU pilot notebook, the attested
    candidates sweep, `docs/API_ROUTES.md`, the corpus-count kit, the shared-prompt-cache HF backend, `max_model_len`
    2048, the 1 Oct DEVIATIONS rows.
  - adoring: the reviewed paper (`paper/*`, canonical), `experiments/ledger.csv` + `scripts/progress.py` (the meter),
    `experiments/gates.yaml`, `experiments/quota.yaml`, `RUNBOOK.md`, `CLAUDE.md`, `scripts/pin_panel.py` (CLI) +
    `scripts/kaggle_cpu_jobs.py`, `scripts/interim_analysis.py`, `scripts/freeze_scoring_rule.py`, the
    pre-registration gate in `scripts/kaggle_run_plan.py`, `configs/run_plan_exploratory.yaml` + `PLAN_PATH`.
  - eloquent: paper ports only (each listed in `paper/claims.md` §12 "Ports of 2026-10-07").
  - Resolutions (one working version per thing): paper = adoring + the listed ports; one pin kit
    (`scripts/pin_panel.py` → `experiments/panel_pins.json`, via `scripts/kaggle_cpu_jobs.py pin`; lucid's record
    fields ported; `configs/panel_manifest.json` removed — memo row 9 still names it); one sheet path
    (`scripts/make_validation_forms.py score` / `import-responses` / `score-baseline` behind `make ingest-sheets`;
    `scripts/ingest_sheets.py` removed; item-level rows in the git-ignored `data/human/report/human_scores.jsonl`,
    aggregates in `experiments/human/human_baseline_report.json`); the notebook generator = lucid's + `PLAN_PATH`
    everywhere, PIN_CELL re-targeted, all 36 notebooks regenerated; `RUNBOOK.md` rewritten around the chunks; the
    ledger and meter keyed by chunk (the 31 chunks are THE run units; the dev-only exploratory lines are unscheduled
    alternatives); quota numbers live in `experiments/quota.yaml` only; this file and `BLOCKED.md` rebuilt.
  - Not done here: no unit has run; Kaggle, Hugging Face and the frozen item files are unreachable from this
    environment; CI green on both parents' heads before the merge; the hash gate is not exercisable here.
- **2026-10-07 (experiments session, adoring; no Kaggle access).** Built the bookkeeping: the ledger (one row per
  planned unit, cut-order ranks per DD 7.1/8.8, cost model in `scripts/ledger.py`), the meter, the gate file, the
  quota assumptions, `RUNBOOK.md`, `CLAUDE.md`, the dev-only exploratory plan, the pin resolver and CPU job driver,
  the interim-analysis driver, the pre-registration gate in the run driver. Exercised the pipeline end to end on a
  local NON-frozen smoke build (outside `data/release/`): `run_eval.py --backend echo --smoke --limit 20` →
  `score_run.py` → `ledger.py ingest` → the meter. Every builder's output went through an independent verifier.
- **2026-10-01 (packet / plan / memo session, lucid).** Gate 1 packet sized by simulation (30 + 4 planted controls
  per cell, Parts A–E, calibration, adjudication rule), the recruitment and consent texts, the human-baseline
  protocol and exact-prompt forms, the decision memo (38 rows), the compute plan cut into 31 resumable chunks in the
  reverse of the cut order, the first end-to-end CPU run on a random-weight stand-in with the real tokenizer
  (RL-2026-10-01-06), 102 web-sourced attested candidates, the API routes under zero budget.

## Prerequisites (DD 7.1, 4.6)

| prerequisite | state on 2026-10-07 |
|---|---|
| pinned panel (`configs/models.yaml` revisions) | **missing**: `experiments/panel_pins.json` not yet produced; every self-hosted `revision` null (34 entries; 35 `revision: null` lines counting the defaults block), 24 ids `hf_id_status: uncertain` (21 self-hosted, 3 API); the kit is ready (`scripts/kaggle_cpu_jobs.py pin` on a Kaggle CPU session with `HF_TOKEN`, then `scripts/pin_panel.py --apply --from`; or on the rented box `scripts/pin_panel.py --apply --partial --out experiments/panel_pins.json`, `RUNBOOK.md` §6 step 4; a full apply is refused until the five panel `quantization.checkpoint` values are chosen) — `BLOCKED.md` 19, 22 |
| passing ~20-item end-to-end run | echo backend **passes** on the local non-frozen smoke build (20/20 rows scored and ingested); the first real run, chunk `c01_cpu_t0`, has **not run** — `BLOCKED.md` 5 |
| frozen item files present and passing the hash gate | **absent here** (`data/release/v0.3/` holds `manifest.json` only; the files live on the author's machine and in the private Kaggle dataset); the gate code (`scripts/kaggle_verify_items.py`, `ledger.py ingest`) is in place and tested against a mismatching file |

## Gates (`experiments/gates.yaml`; copied into the meter below)

Every value is null: `stage1_commit`, `stage2_commit`, `registration_url`, `registration_date`, `qu_convention`,
`iy_emission`, `gemini_route`, `scoring_rule_freeze.*`. `docs/PREREGISTRATION.md` still reads `<fill>` for both
commit fields. → every confirmatory unit (chunks `c08`–`c31`) and the whole API lane are **gated**; only the author
records a value (`BLOCKED.md` 4, 20, 31); the assistant never runs `scripts/freeze_scoring_rule.py --record`.

## Next chunk per queue (`configs/compute_chunks.yaml`; recipes in `RUNBOOK.md` §1, the rented box in §6)

| queue | next | then | waits on |
|---|---|---|---|
| cpu | `c01_cpu_t0` — first real run, `smoke_20` on 20 dev items (memo N10) | `c02`–`c04`, the Gate 1 pilot, one model per session (memo N11) | `BLOCKED.md` 1–3, 5 |
| gpu (T4 x2) | `c06_gpu_t0` — smoke week (19–25 Oct), `smoke_20` on every T4/2xT4 model, unpinned allowed; take the x1/x2 quota reading | `c08_gpu_t1` after the stage-2 commit; `c05_gpu_t0` only as the alternative to `c02`–`c04` if the CPU route will not finish by 18 Oct (needs its three models pinned; the ledger counts it as `alternative`) | `BLOCKED.md` 14, 19, 31 |
| tpu (v5e-8) | `c07_tpu_t0` — smoke week (19–25 Oct), `smoke_20_tpu` on the TPU trio (a `limit` line: unpinned allowed; the whole-panel pin runs the same week) | `c16_tpu_t1` after the stage-2 commit | `BLOCKED.md` 19, 31 |
| box (2x A800-40GB, paid) | `tier0` through `RUNBOOK.md` §6 (`python scripts/cuda_box.py --session auto --chunks tier0 --platform rented`, after the pin on the box): `c05_gpu_t0` as the pilot route instead of `c02`–`c04` (the ledger counts `c05` `alternative` until launched, memo N11), then `c06_gpu_t0` and `c07_tpu_t0` (the two 27B models on both cards) | `c08`–`c31` after the gate: `--chunks from:c08_gpu_t1 --lanes 2` (`RUNBOOK.md` §6 step 6); every one of them `skipped: gate: ...` until then | `BLOCKED.md` 2, 3 (the frozen release files, `scp`'d to the box), 4 / 31, 14 and 19 (the box steps) |
| api | nothing: gated on the Gemini route and the registration | the four API lines daily from the stage-2 commit | `BLOCKED.md` 20, 31, 32 |
| human | validator invitations 8 Oct; packet 12 Oct; ~20 respondents 12 Oct | calibration 13–17 Oct; Parts B–E 19 Oct–1 Nov; baseline 2–8 Nov | `BLOCKED.md` 9–13, 18, 26 |

Nothing is running. The chunk plan is the schedule; the dev-only `floor_pilot_dev` and `throughput_dev` lines of
`configs/run_plan_exploratory.yaml` are unscheduled alternatives outside the chunk order (`RUNBOOK.md` §2). After
each hand-back: `python scripts/ledger.py ingest --runs data/runs`, `make ingest-sheets`, `python scripts/progress.py`.

## Open questions for the author (only where the memo gives no answer)

Everything the memo already answers is not asked again: the stage-1 commit (row 15), `qu` and `i/y` (rows 3, 5),
Gemini (rows 8, 18, 10), accounts and age clauses (row 17), people (rows 7, 6), the H6 floor (N6), validation and
baseline sizes (N1, N2), the first-run route (N10, N11), the stage-2 date (N13: 25 Oct without a co-author, 8 Nov
with one; `BLOCKED.md` 21 decides it on 25 Oct). Reply to the memo with the row numbers you accept; then:

1. **Kaggle access for this environment.** Allow `www.kaggle.com`, `huggingface.co`, `cdn-lfs.huggingface.co` in the
   cloud environment's network settings and provide `KAGGLE_API_TOKEN` (or `~/.kaggle/access_token`)? Both hosts are
   unreachable here today and no token exists; otherwise every session goes through `RUNBOOK.md` by hand.
2. **Dataset slugs.** Confirm `noilai-bundle`, `noilai-release`, `noilai-runs` (the names every notebook and the
   runbook assume); `noilai-runs` does not exist yet since nothing has run.
3. **Real quota numbers.** GPU and TPU hours per week, the reset day and hour, whether CPU sessions count, and the
   2xT4 accounting (x1 or x2) read during the first 2xT4 session → `experiments/quota.yaml` only (memo N8 says how,
   not the numbers).
4. **Groq cap scope.** Per model or per organisation on the console (9.25 vs 18.5 days; `docs/API_ROUTES.md` §2).
5. **Registration venue and date.** Where the stage-1 (and later stage-2) pre-registration is made public
   (tagged-commit URL, OSF or similar) and when → `experiments/gates.yaml` `registration_url` / `registration_date`.
6. **The canary digest mismatch** between `docs/RESULTS_LOG.md` RL-2026-10-01-01 and `data/release/v0.3/manifest.json`
   (`BLOCKED.md` 16): which record is right; the assistant edits neither.

7. **Analysis rulings before any pooled number is reported** (`configs/analysis.yaml`, `noilai/stats/pooling.py`,
   `noilai/stats/analyze.py`; built 2026-10-12, `BLOCKED.md` 34): the seven tokenizer-family labels of the 21 panel
   entries (Sailor2 is placed under the Qwen BPE family); REML + Hartung–Knapp as the primary pool with the family
   bootstrap beside it; the Hartung–Knapp variance is truncated at the fixed-effect variance (q = max(1, q), the raw
   q reported) so the interval never collapses when family estimates coincide; the family floor for H4/H6 pooling (3)
   against H3's 5; how an API model's placement cell enters its Table 3 family (the C2-enriched file is never run on
   API models; `analyze` drops the cell, `aggregate.intervention_effects` pairs the core's C2-affected items); and
   whether the `qat_int4` entries of `configs/models.yaml` are quantize-at-load requests (today the backends and
   `scripts/pin_panel.py` both treat them as checkpoint-carried and unchosen).

<!-- progress:begin -->
_Generated by `scripts/progress.py` on 2026-10-07 22:06Z from `experiments/ledger.csv` (429 compute units in 31 chunks + non-chunked lines, 24 human units). Weights: 1 GPU-h = 1.0; 1 TPU-h = 1.0; 1 L4-h = 1.0; 1 CPU-h = 0.0 (`experiments/quota.yaml` cpu_sessions_count_against_gpu_quota: None); 1,000 API calls = 0.25; human person-hours on their own lines. A unit counts only when `done` and hash-verified; cut / zero_by_construction / unscheduled / alternative units leave the denominator._

### Meter

| scope | done / planned (cost units) | share |
|---|---|---|
| **Overall (quota meter)** | 0.0 / 98.2 | **  0.0%** |
| confirmatory units only | 0.0 / 95.2 |   0.0% |
| section: Results | 0.0 / 53.6 |   0.0% |
| section: Counterfactuals | 0.0 / 31.6 |   0.0% |
| section: Tone | 0.0 / 10.0 |   0.0% |
| section: Prerequisite | 0.0 / 3.0 |   0.0% |
| section: Exploratory | 0.0 / 0.0 |   0.0% |
| experiment: E1 | 0.0 / 32.1 |   0.0% |
| experiment: E1_ablation | 0.0 / 6.0 |   0.0% |
| experiment: E3 | 0.0 / 31.6 |   0.0% |
| experiment: E4 | 0.0 / 10.0 |   0.0% |
| experiment: bf16_drift | 0.0 / 4.6 |   0.0% |
| experiment: pilot | 0.0 / 0.0 |   0.0% |
| experiment: pilot_exploratory | 0.0 / 0.0 |   0.0% |
| experiment: reasoning | 0.0 / 10.9 |   0.0% |
| experiment: smoke | 0.0 / 3.0 |   0.0% |
| experiment: throughput | 0.0 / 0.0 |   0.0% |
| CPU sessions (hours, weight 0.0) | 0.0 / 15.4 (0/15 units) |   0.0% |
| human validation (person-h) | 0.0 / 16.2 |   0.0% |
| human baseline (person-h) | 0.0 / 12.0 |   0.0% |
| attested expansion (person-h) | 0.0 / 12.0 |   0.0% |

Status counts (compute units): {"alternative": 12, "planned": 406, "unscheduled": 11}

### Chunks (`configs/compute_chunks.yaml`, in run order; a chunk is done when every live unit is done and hash-verified)

| chunk | order | tier | queue | window | units done/total | status | hours low-high | quota x1 | quota x2 |
|---|---|---|---|---|---|---|---|---|---|
| `c01_cpu_t0` | 1 | 0 | cpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/1 | planned | 0.11-0.11 | 0.0-0.0 | 0.0-0.0 |
| `c02_cpu_t0` | 2 | 0 | cpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/4 | planned | 2.19-2.19 | 0.0-0.0 | 0.0-0.0 |
| `c03_cpu_t0` | 3 | 0 | cpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/5 | planned | 4.61-4.61 | 0.0-0.0 | 0.0-0.0 |
| `c04_cpu_t0` | 4 | 0 | cpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/5 | planned | 8.52-8.52 | 0.0-0.0 | 0.0-0.0 |
| `c05_gpu_t0` | 5 | 0 | gpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/0 | alternative | 2.0-3.0 | 2.0-3.0 | 4.0-6.0 |
| `c06_gpu_t0` | 6 | 0 | gpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/14 | planned | 2.0-2.0 | 2.0-2.0 | 4.0-4.0 |
| `c07_tpu_t0` | 7 | 0 | tpu | now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct) | 0/3 | planned | 1.0-1.0 | 1.0-1.0 | 1.0-1.0 |
| `c08_gpu_t1` | 8 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 4.11-6.62 | 4.11-6.62 | 8.22-13.24 |
| `c09_gpu_t1` | 9 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c10_gpu_t1` | 10 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c11_gpu_t1` | 11 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c12_gpu_t1` | 12 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c13_gpu_t1` | 13 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c14_gpu_t1` | 14 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c15_gpu_t1` | 15 | 1 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.11-4.62 | 2.11-4.62 | 4.22-9.24 |
| `c16_tpu_t1` | 16 | 1 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/15 | planned | 2.05-4.5 | 2.05-4.5 | 2.05-4.5 |
| `c17_gpu_t2` | 17 | 2 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/32 | planned | 1.88-3.76 | 1.88-3.76 | 3.76-7.52 |
| `c18_gpu_t2` | 18 | 2 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/1 | planned | 2.5-5.0 | 2.5-5.0 | 5.0-10.0 |
| `c19_gpu_t2` | 19 | 2 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/1 | planned | 2.5-5.0 | 2.5-5.0 | 5.0-10.0 |
| `c20_tpu_t2` | 20 | 2 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/11 | planned | 1.74-3.97 | 1.74-3.97 | 1.74-3.97 |
| `c21_tpu_t2` | 21 | 2 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/1 | planned | 2.0-4.0 | 2.0-4.0 | 2.0-4.0 |
| `c22_gpu_t3` | 22 | 3 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.01-4.43 | 2.01-4.43 | 4.02-8.86 |
| `c23_gpu_t3` | 23 | 3 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.34-5.09 | 2.34-5.09 | 4.68-10.18 |
| `c24_gpu_t3` | 24 | 3 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.34-5.09 | 2.34-5.09 | 4.68-10.18 |
| `c25_gpu_t3` | 25 | 3 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.34-5.09 | 2.34-5.09 | 4.68-10.18 |
| `c26_tpu_t3` | 26 | 3 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/23 | planned | 3.14-6.97 | 3.14-6.97 | 3.14-6.97 |
| `c27_gpu_t4` | 27 | 4 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.34-5.09 | 2.34-5.09 | 4.68-10.18 |
| `c28_gpu_t4` | 28 | 4 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/19 | planned | 2.01-4.43 | 2.01-4.43 | 4.02-8.86 |
| `c29_tpu_t4` | 29 | 4 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/21 | planned | 2.71-5.97 | 2.71-5.97 | 2.71-5.97 |
| `c30_gpu_t5` | 30 | 5 | gpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/4 | planned | 0.66-1.32 | 0.66-1.32 | 1.32-2.64 |
| `c31_tpu_t5` | 31 | 5 | tpu | after the stage-2 commit (DD 8.8) - 22 Nov (Gate 3, results freeze) | 0/6 | planned | 1.39-2.78 | 1.39-2.78 | 1.39-2.78 |

| rollup | chunks done/total | hours low-high (planned) | hours done | quota x1 | quota x2 |
|---|---|---|---|---|---|
| tier 0 | 0/6 | 18.43-18.43 | 0-0 | 3.0-3.0 | 5.0-5.0 |
| tier 1 | 0/9 | 20.93-43.46 | 0-0 | 20.93-43.46 | 39.81-82.42 |
| tier 2 | 0/5 | 10.62-21.73 | 0-0 | 10.62-21.73 | 17.5-35.49 |
| tier 3 | 0/5 | 12.17-26.67 | 0-0 | 12.17-26.67 | 21.2-46.37 |
| tier 4 | 0/3 | 7.06-15.49 | 0-0 | 7.06-15.49 | 11.41-25.01 |
| tier 5 | 0/2 | 2.05-4.1 | 0-0 | 2.05-4.1 | 2.71-5.42 |
| queue cpu | 0/4 | 15.43-15.43 | 0-0 | 0.0-0.0 | 0.0-0.0 |
| queue gpu | 0/19 | 41.8-85.26 | 0-0 | 41.8-85.26 | 83.6-170.52 |
| queue tpu | 0/7 | 14.03-29.19 | 0-0 | 14.03-29.19 | 14.03-29.19 |

### Next chunk per queue (the launch order: lowest chunk order with a unit still to do)

- cpu: `c01_cpu_t0` (order 1, tier 0, planned; `notebooks/chunks/c01_cpu_t0.ipynb`; 0.11-0.11 h; window now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct))
- gpu: `c06_gpu_t0` (order 6, tier 0, planned; `notebooks/chunks/c06_gpu_t0.ipynb`; 2.0-2.0 h; window now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct))
- tpu: `c07_tpu_t0` (order 7, tier 0, planned; `notebooks/chunks/c07_tpu_t0.ipynb`; 1.0-1.0 h; window now - 25 Oct (Gate 1 on 18 Oct; panel freeze 25 Oct))

### Pre-registered analyses computable on the completed units

| analysis | computable now | n | primary |
|---|---|---|---|
| Table 2 (E1 estimation per done model) | no | 0 models |  |
| Explicit-input gap, DD 12.17 main result | no | 0 models paired | yes |
| H1 nested LR test (E2), PREREG 8.4 | no | 0 models / 0 families (needs >= 8 over >= 4 incl. a non-Gemma pass-through family; item audits data/audit/items_*.jsonl) | yes |
| H3 pooled crossover (E3), PREREG 8.5 | no | 0 pass-through models / 0 families (pooled CI reliable from 5 families) | yes |
| H3b tone-isolation DiD | no | 0 models |  |
| H4 placement (C2 set) | no | 0 models |  |
| H5 probes and patching (E4) | no | 0 models |  |
| H6 attested vs matched (needs the 100-row floor) | no | 0 models; floor NOT met |  |
| Human-baseline comparison (mean-human band) | no | 0/20 forms |  |

### Frontier (chunk walk: the first chunk at which each analysis becomes computable, and the quota spent by then)

| analysis | first chunk | order | tier | GPU h x1 by then | GPU h x2 by then | TPU h by then |
|---|---|---|---|---|---|---|
| Table 2 (E1 estimation per done model) | `c08_gpu_t1` | 8 | 1 | 6.11-8.62 | 12.22-17.24 | 1.0-1.0 |
| Explicit-input gap, DD 12.17 main result | `c08_gpu_t1` | 8 | 1 | 6.11-8.62 | 12.22-17.24 | 1.0-1.0 |
| H3 pooled crossover (E3), PREREG 8.5 | `c08_gpu_t1` | 8 | 1 | 6.11-8.62 | 12.22-17.24 | 1.0-1.0 |
| H3b tone-isolation DiD | `c08_gpu_t1` | 8 | 1 | 6.11-8.62 | 12.22-17.24 | 1.0-1.0 |
| H4 placement (C2 set) | `c08_gpu_t1` | 8 | 1 | 6.11-8.62 | 12.22-17.24 | 1.0-1.0 |
| H1 nested LR test (E2), PREREG 8.4 | `c15_gpu_t1` | 15 | 1 | 20.88-40.96 | 41.76-81.92 | 1.0-1.0 |

Not reached by the chunk walk (needs API, E4 or human units, which are not chunks): H5 probes and patching (E4); H6 attested vs matched (needs the 100-row floor); Human-baseline comparison (mean-human band).

All chunks: GPU 41.8-85.26 h at x1 / 83.6-170.52 h at x2, TPU 14.03-29.19 h (live units; plan estimates, not measurements).

Cut-rank band (DD 7.1 priority, 55-60% of the cost): 255 units (first: `smoke_20__gemma-3-1b-it__nfc`, last: `E3_xcopa__gemma-3-12b-it__strip_tones`); share at which each analysis first becomes computable: Table 2 (E1 estimation per done model) at 4%, H1 nested LR test (E2), PREREG 8.4 at 14%, Explicit-input gap, DD 12.17 main result at 29%, H4 placement (C2 set) at 52%, H3 pooled crossover (E3), PREREG 8.5 at 54%, H3b tone-isolation DiD at 54%, H5 probes and patching (E4) at 66%.
All three primary analyses are computable at the band.

### Kaggle hours this quota week

Week starting 2026-10-03 00:00Z (reset day per `experiments/quota.yaml`, unverified): GPU 0.0 used at x1 (0.0 at x2 = hours x n_gpus), 30.0 left of 30 assumed at x1 (30.0 at x2); TPU 0.0 used, 20.0 left of 20 assumed; CPU sessions 0.0 h (no GPU quota unless `cpu_sessions_count_against_gpu_quota` is true). Source: `data/compute_log.csv` (absent = nothing logged).

### Gates (`experiments/gates.yaml`)

- pre-registration: url `None`, date `None`, stage-1 `None`, stage-2 `None`
- qu convention `None`; i/y emission `None`; Gemini route `None`; scoring rule frozen at `None`
- confirmatory runs (every tier >= 1 chunk): **gated** (missing: registration_url, registration_date, qu_convention, iy_emission)
- API runs: **gated** (Gemini route undecided)

### Running now

Nothing is running (no unit is launched, running or partial).
<!-- progress:end -->
