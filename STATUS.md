# STATUS — experiments and validation workstreams

Hand-maintained sections first (dated **2026-10-07**); the meter block at the end is written by `scripts/progress.py`
and is never edited by hand. Blockers, in deadline order, are in `BLOCKED.md`; the author's rulings are in
`docs/gate1/DD13_DECISION_MEMO.md` (reply with the row numbers you accept).

## Session log (newest first)

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
| pinned panel (`configs/models.yaml` revisions) | **missing**: `experiments/panel_pins.json` not yet produced; every self-hosted `revision` null (34 entries; 35 `revision: null` lines counting the defaults block), 24 ids `hf_id_status: uncertain` (21 self-hosted, 3 API); the kit is ready (`scripts/kaggle_cpu_jobs.py pin` on a Kaggle CPU session with `HF_TOKEN`, then `scripts/pin_panel.py --apply --from`; a full apply is refused until the five panel `quantization.checkpoint` values are chosen) — `BLOCKED.md` 19, 22 |
| passing ~20-item end-to-end run | echo backend **passes** on the local non-frozen smoke build (20/20 rows scored and ingested); the first real run, chunk `c01_cpu_t0`, has **not run** — `BLOCKED.md` 5 |
| frozen item files present and passing the hash gate | **absent here** (`data/release/v0.3/` holds `manifest.json` only; the files live on the author's machine and in the private Kaggle dataset); the gate code (`scripts/kaggle_verify_items.py`, `ledger.py ingest`) is in place and tested against a mismatching file |

## Gates (`experiments/gates.yaml`; copied into the meter below)

Every value is null: `stage1_commit`, `stage2_commit`, `registration_url`, `registration_date`, `qu_convention`,
`iy_emission`, `gemini_route`, `scoring_rule_freeze.*`. `docs/PREREGISTRATION.md` still reads `<fill>` for both
commit fields. → every confirmatory unit (chunks `c08`–`c31`) and the whole API lane are **gated**; only the author
records a value (`BLOCKED.md` 4, 20, 31); the assistant never runs `scripts/freeze_scoring_rule.py --record`.

## Next chunk per queue (`configs/compute_chunks.yaml`; recipes in `RUNBOOK.md` §1)

| queue | next | then | waits on |
|---|---|---|---|
| cpu | `c01_cpu_t0` — first real run, `smoke_20` on 20 dev items (memo N10) | `c02`–`c04`, the Gate 1 pilot, one model per session (memo N11) | `BLOCKED.md` 1–3, 5 |
| gpu (T4 x2) | `c06_gpu_t0` — smoke week (19–25 Oct), `smoke_20` on every T4/2xT4 model, unpinned allowed; take the x1/x2 quota reading | `c08_gpu_t1` after the stage-2 commit; `c05_gpu_t0` only as the alternative to `c02`–`c04` if the CPU route will not finish by 18 Oct (needs its three models pinned; the ledger counts it as `alternative`) | `BLOCKED.md` 14, 19, 31 |
| tpu (v5e-8) | `c07_tpu_t0` — smoke week (19–25 Oct), `smoke_20_tpu` on the TPU trio (a `limit` line: unpinned allowed; the whole-panel pin runs the same week) | `c16_tpu_t1` after the stage-2 commit | `BLOCKED.md` 19, 31 |
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
