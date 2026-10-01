# BLOCKED — what only the author (or a GPU, or a person) can do

Every entry: what is needed, the exact steps, the expected time. **Ordered by urgency: the date it blocks.** Nothing here
needs money; nothing may be done with a misstated age or on someone else's account (an adult collaborator's own
account is fine, named by role only, DD 11.2). Decisions behind the entries: `docs/gate1/DD13_DECISION_MEMO.md` (rows
cited as "memo N"). Updated 1 October 2026.

## This week (by 5 October): accounts and the first real run

### 1. Kaggle account: phone verification, quota page, 2×T4 accounting — today
- **Needed:** the real quotas; this machine cannot open kaggle.com, and every quota in `docs/COMPUTE_PLAN.md` §1 is unverified.
- **Steps:** kaggle.com → Settings → verify the phone number (needed for GPU/TPU and internet in notebooks). Note from the
  quota box: GPU hours a week, TPU hours a week, the reset day and time, the session limits; once a 2×T4 session has run,
  whether one hour of it took 1 or 2 hours off the quota (memo N8). Put the numbers in `configs/run_plan.yaml` `quota` and
  a `docs/RESULTS_LOG.md` entry; re-run `python scripts/plan_chunks.py --write` (it re-computes the capacity table).
- **Time:** 10 minutes (the 2×T4 reading: during the first GPU session).

### 2. Hugging Face: licences and a read token — today (approvals can take days)
- **Steps:** on huggingface.co (the account holder must meet HF's and each licence's age terms: memo 17), accept the
  licences of `google/gemma-3-1b-it` (and 4b, 12b), `google/gemma-4-*`, `meta-llama/Llama-3.1-8B-Instruct`,
  `Viet-Mistral/Vistral-7B-Chat`; create a **read** token; add it to Kaggle → notebook → Add-ons → Secrets as `HF_TOKEN`.
- **Time:** 15 minutes; Llama and Vistral approvals are manual (hours to days).

### 3. Kaggle datasets for the notebooks — today
- **Steps:** (a) `git bundle create noilai-main.bundle main` → private dataset `noilai-bundle`; (b) the frozen
  `data/release/v0.3/` tree (private files, never public) → private dataset `noilai-release`; (c) Kaggle Secrets
  `KAGGLE_USERNAME`, `KAGGLE_KEY` (Settings → API → create token); the first notebook run with `CREATE_DATASET = True`
  creates the private `noilai-runs` dataset, attach it as input from the second session on.
- **Time:** 20 minutes.

### 4. Fill the stage-1 commit in the pre-registration — today (memo 15)
- **Steps:** in `docs/PREREGISTRATION.md`'s header set `PREREG_COMMIT_STAGE1` to the v0.3 freeze commit (`f94f655…` in
  the original history = `62f63ae…` in this repository, `docs/MIGRATION.md`; the release manifest's `git_commit`). This
  session does not edit the pre-registration.
- **Time:** 5 minutes.

### 5. First real run: chunk c01 on a Kaggle CPU session — by 5 October
- **Steps:** Kaggle → New notebook → File → Import `notebooks/chunks/c01_cpu_t0.ipynb`; accelerator **None**; internet on;
  attach `noilai-bundle`, `noilai-release`; Secrets as in 2–3; set `ACCOUNT_HOLDER_ROLE` (a role, never a name);
  *Save & Run All*. It pins gemma-3-1b-it's revision (`pin_panel.py --apply`), runs the 20-item smoke through HF on CPU,
  then `check_run.py` (item gate, rescoring, stats, hashes) and pushes `data/runs/smoke_20__gemma-3-1b-it__cpu`.
- **Then:** copy the run directory back (`kaggle datasets download`), commit `configs/models.yaml`'s pinned revision and
  the manifest, add an RL entry. If anything breaks, the error and the cell go to the next coding session.
- **Expected runtime:** ~0.1 h of compute (RL-2026-10-01-06, worst case) plus install and a ~2 GB download: under 30 min.

## Before Gate 1 (18 October)

### 6. Minimum-age clauses — by 12 October (memo 17)
- **Steps:** read and record the minimum age in the terms of Kaggle, Google (Colab, AI Studio, Forms), Hugging Face, Groq,
  OpenRouter; put each in `configs/models.yaml` `providers.*.terms.minimum_age` and `docs/RISKS.md`. Where the author does
  not meet one, that service runs on an adult collaborator's own account (`ACCOUNT_HOLDER_ROLE = "adult collaborator"`).
- **Time:** 30 minutes.

### 7. Recruit 2–3 native validators (N, C, S) — invitations by 8 October, packet out 12 October (memo 7)
- **Steps:** first fill the consent form's placeholders: `[NAME]`, `[EMAIL]`, the retention date in §5
  (`[RETENTION — author decides]`) and, if the author is under 18, the adult second contact in §9; have the
  `[NATIVE-CHECK]` passages read by a native speaker and remove the markers. Then send `docs/gate1/RECRUITMENT.md`
  (VI/EN) to adult native speakers by personal message only; record region and availability in
  `data/validation/validators.json` (git-ignored; ids A/B/C and region only); each returns the ticked
  `docs/CONSENT_FORM.md`. Validators and human-baseline respondents must be different people.
- **Time:** the message is ready; ~5.4 h of work per validator with three, ~6.9 h with two (`docs/gate1/VALIDATION_PROTOCOL.md`).

### 8. Build and send the validation packet — by 12 October
- **Needed:** the private v0.3 item files (`data/release/v0.3/noilai_{test,dev,core,main}.jsonl`, `attested.jsonl`), which
  exist only on the author's machine (git-ignored; the build seed is withheld).
- **Steps:**
  1. `pip install -r requirements.txt` (adds `openpyxl` and `krippendorff`).
  2. `make validation VALIDATORS="A B C"` (or `"A B"`) → `data/validation/validation_<V>.xlsx` per validator, CSV copies,
     and the author's keys (`B_key.json`, `A_calibration_key.json`, `D_engine.json`, `E_key.json`: never send these).
  3. Check `data/validation/validation_manifest.json`: `sizes.n_sample` 360, `n_controls` 48, hours per validator.
  4. Upload each workbook to Google Drive → open with Google Sheets (dropdowns survive), share each with its validator only.
  5. Calibration (13–15 Oct): `make validation-score` scores Part A; a validator below 80% gets the second set,
     `python scripts/make_validation_forms.py calibration2 --dir data/validation --validators <V> --release data/release/v0.3`.
  6. After each weekly return: `make validation-score` → `report/validation_report.json`, adjudication, and
     `data/audit/validator_flags.json`, which **must be committed** before any API run and before `make baseline`
     (the API screen and the baseline builder read it). Third-validator sheets once per letter (protocol §5).
  7. By 5 November (the consent form promises it): download each validator's Google Sheet as .xlsx into
     `data/validation/returned/`, delete the Sheet in Drive, empty the Drive trash, delete any e-mailed workbook.
- **Time:** ~2 minutes to build, ~20 minutes to upload and share; ~10 minutes per scoring round.

### 9. GPL permission e-mails — confirm this week (memo 1)
- **Steps:** confirm that the two permission requests for the Viet74K / Hunspell entries went out on 30 September (the
  README says "requested"); if not, send them now. No reply by 8 November → the GPLv2 fallback of DD 4.1 (memo 19).
- **Time:** 10 minutes.

### 10. The Gate 1 pilot on CPU: chunks c02–c04 — by 16 October (memo N11)
- **Steps:** as entry 5, one Kaggle CPU session per chunk (`c02` gemma-3-1b-it, `c03` qwen3.5-2b, `c04` phogpt-4b-chat;
  `c04` may need `TRANSFORMERS_OVERRIDE` if PhoGPT's remote code fails under transformers 5.x). Each runs smoke_20,
  pilot_t1_200 (nfc, nfd, with the exploratory forced choice) and pilot_xcopa_200, then `check_run.py`.
  The go/no-go is computed from these files by PREREG §10 over the three models. **Alternative:** chunk `c05` runs the same
  lines on a T4 in 2–3 GPU-h (free quota before the smoke week).
- **Expected runtime (worst case, RL-2026-10-01-06, 4 cores):** 2.3 h, 4.6 h, 8.5 h; the sessions can run on different days.

### 11. The C2 placement corpus count — by 18 October (DD 6.3; memo 2, N12)
- **Steps:** `bash scripts/corpus_count_kit.sh` on any machine with internet (a laptop or a Kaggle CPU session): streams the
  first 2M lines of the Vietnamese Wikipedia dump and of CC-100 vi, counts both conventions, writes
  `data/audit/placement_corpus/{viwiki,cc100_vi}.json` with the files' dates. If a corpus is under 10M syllable tokens, re-run
  with `LINES=4000000`. Then a RESULTS_LOG entry naming both corpora, their dates and the two `old_share` values.
- **Expected runtime:** ~7 minutes of counting per 10M syllable tokens (measured: 24,000 tokens/s) plus the download.

## Before the panel freeze (25 October)

### 12. Gemini: decide, and an adult's AI Studio key if kept — by 25 October (memo 8, 18)
- **Options under zero budget:** drop Gemini (panel 19 models), or dev-only with an adult's key (needs a `noilai_api_dev`
  item file, not yet in the run plan: `docs/API_ROUTES.md`). The paid key is excluded (money).

### 13. GPU smoke week: chunks c06 (T4) and c07 (TPU), then the panel freeze — 19–25 October
- **Steps:** `c06` (accelerator GPU T4 ×2) runs smoke_20 on every T4 / 2×T4 model; `c07` (TPU v5e-8) on the TPU trio. Before
  them, `scripts/pin_panel.py` on Kaggle (internet) resolves every `uncertain` id and writes `configs/panel_manifest.json`
  (`--apply`, then `--verify`). Compare the T4 smoke's gemma-3-1b-it rows with c01's CPU rows (same 20 items, row by row).
  Record tokens/s per family and the 2×T4 quota reading (entry 1). The panel-freeze commit (PREREG §7: ids, revisions,
  smoke results, census verdicts, provider limits, tokenizer SHA-256s) is the author's.
- **Expected runtime:** 2 GPU-h + 1 TPU-h (plan estimates); TPU startups of 6–22 min per model.

### 14. Human-baseline respondents — recruit by 1 November, forms after Gate 1
- **Steps:** ~20 adult native speakers who are **not** validators (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md`);
  `make validation-score` first (validator flags and verified attested rows), then `make baseline` → forms;
  `python scripts/make_validation_forms.py google-form --dir data/human --contact-email <address>` writes the Apps
  Script that creates the Google Forms in the account holder's own Google account (e-mail collection off, consent
  required, the form number in the confirmation); responses → `import-responses` (first submission per form) →
  `score-baseline`. Runbook: `docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §4.
- **Time:** 30–45 minutes per respondent.

## Before the stage-2 commit (25 October – 8 November)

### 15. Attested set: verify sources, books, Pham & Pham — by 8 November (memo 6, N5, N6)
- **Steps:** (a) for every row of `data/attested_candidates.tsv` that the Part C validators accept, open its URL and set
  `source_checked` (the evidence column is a search-summary quote, never checked on the live page); merge verified rows into
  `data/attested_seed.tsv` and rebuild `attested.jsonl` only (memo N5, a DEVIATIONS row). (b) The floor: at most 79
  H6-eligible rows even if every candidate verifies (floor 100); a manual pass over *Thú chơi chữ* and two other
  collections, or H6 becomes descriptive (memo N6). (c) Ask Pham & Pham (PACLIC 2018, "Building a Spoonerism Detection System
  for Vietnamese", aclanthology Y18-1063) for their sentence set; draft: *"Dear Dr [name], I am preparing a benchmark of
  Vietnamese nói lái for language models and would like to cite and, with your permission, use the attested examples behind
  your PACLIC 2018 paper. Could you share the data or tell me the terms under which it may be used? Every row would be cited
  to your paper. [name, role]"* (the corresponding address is on the paper's first page).
- **Time:** ~1 minute per URL; the books depend on access.

### 16. The stage-2 commit — 25 October to 8 November (memo N13)
- **Steps:** the author (and the co-author if one has joined) writes stage 2 (hypotheses with tests, analysis plan, E4
  protocol, pilot-informed constants with the pilot numbers, PREREG §8.14 exploratory list incl. the T1 forced choice,
  memo N4) and commits it; `PREREG_COMMIT_STAGE2` is that commit. No test-split run before it (DD 8.8).
- **Why early:** with 8 November, the test-split GPU work has two weeks; `docs/COMPUTE_PLAN.md` §5 shows what fits.

## After the stage-2 commit (to the results freeze, 22 November)

### 17. GPU and TPU chunks c08–c31, in order
- **Steps:** one Kaggle session per chunk (`notebooks/chunks/`), in the order of `configs/compute_chunks.yaml`; re-run a
  chunk until every job is `ok`; after the first chunk of each tier, `python scripts/compute_log.py` and, if the measured
  hours differ from the plan, `plan_chunks.py --write`. If the quota runs out, what is left is what the pre-registered cut
  order drops first; name every cut model in the paper.
- **Expected runtime:** tiers 1–5, GPU 83.3 h and TPU 28.3 h at the high estimate, plus session overhead (plan estimates, equal split;
  `docs/COMPUTE_PLAN.md` §4).

### 18. API runs on Groq — daily, from the stage-2 commit
- **Steps:** `notebooks/api_runs.ipynb` with `GROQ_API_KEY` (an account whose holder meets Groq's terms), `RUN_IDS` the four
  API lines, `MODELS = []` (Gemini is refused on core-derived files until memo 8 is decided); re-run once a day; the
  ledger parks a model at its cap. First, read the live caps and whether they are per model or per org (`docs/API_ROUTES.md`).
- **Expected runtime:** 9.25 days per model at 200 K tokens a day (18.5 days for both if the caps are per org).

### 19. PhoGPT's bf16 reference — decide (memo N9)
- Modal is excluded (card on file); recommended: drop it from the bf16 appendix and say so (`docs/COMPUTE_PLAN.md` §6).

## Later

### 20. E4 notebook resume and protocol — by 23 November
- The probe notebook is a skeleton without checkpoints, and the E4 protocol is a stage-2 item; once the protocol is
  committed, a coding session adds resume/checkpoints to `colab_probe_gemma3` (5–15 GPU-h on 2×T4, 23 Nov–6 Dec).

### 21. Web-search budget for the next coding session (memo N7)
- **Steps:** start the next session with `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` raised (e.g. 400) so quotas, terms,
  credit programs and attested sources can be verified; this session's 200 searches went to the attested sweep.

### 22. Optional: OpenAI Researcher Access application
- Draft in `docs/API_ROUTES.md` §3; the applicant checks eligibility and age first. Credits would arrive after the deadline
  (rebuttal / camera-ready runs).
