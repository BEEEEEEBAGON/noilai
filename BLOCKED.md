# BLOCKED — one list, ordered by the date each item blocks (experiments and validation workstreams)

Updated **2026-10-07**. One entry per thing that waits on a person, ordered by the date it blocks; **PAST DUE** = a
deadline before 7 October. Each entry: deadline, who (author / adult collaborator / validators / respondents /
assistant), exact steps, what it blocks, and the row of `docs/gate1/DD13_DECISION_MEMO.md` that already answers it
(the memo is the author's record, not edited here; reply to it with the row numbers you accept). Dates are the
protocols': invitations 8 Oct, packet 12 Oct, Gate 1 18 Oct, panel freeze 25 Oct, stage-2 commit 8 Nov at the
latest (25 Oct recommended, memo N13), results freeze 22 Nov, E4 23 Nov–6 Dec. The frozen release,
`docs/PREREGISTRATION.md`, `configs/run_plan.yaml`, recorded hashes and `experiments/gates.yaml` are the author's edits.

1. **[1 Oct → now] PAST DUE — Kaggle account: phone verification and the quota page.** Who: author (the adult
   collaborator if Kaggle's age clause requires it, entry 10). Steps: Settings → Phone Verification (no GPU/TPU and
   no Internet toggle without it, `RUNBOOK.md` §0.2); read the quota box (GPU h/week, TPU h/week, reset day and hour,
   session limits, whether CPU sessions count) into `experiments/quota.yaml` only (never `configs/run_plan.yaml`)
   plus a `docs/RESULTS_LOG.md` line; the 2xT4 x1/x2 reading of the first 2xT4 session (entry 19) goes into the same
   file as a dated comment line. Blocks: nothing hard; the meter's "hours left" line and the capacity table of
   `docs/COMPUTE_PLAN.md` §4 are assumptions. Memo N8, 11.
2. **[1 Oct → now] PAST DUE — Hugging Face licences and a read token.** Who: the account holder who meets HF's and
   each licence's age terms (entry 10). Steps: accept the licence of every `gated: true` entry in `configs/models.yaml`
   (`RUNBOOK.md` §0.3; Llama's is a form with manual approval, hours to days); Settings → Access Tokens → Read →
   Kaggle Add-ons → Secrets as `HF_TOKEN` (never in a cell or the repository). Blocks: c01 (entry 5), the pin run and
   the audits (entry 19), every GPU/TPU chunk. Memo 17.
3. **[1 Oct → now] PAST DUE — Kaggle datasets and secrets for the notebooks.** Who: author (the frozen item files
   exist only on the author's machine; here `data/release/v0.3/` holds `manifest.json` alone). Steps: `make bundle` →
   private dataset `noilai-bundle`; the frozen `data/release/v0.3/` tree without `manifest_private.json` and `sealed/`
   → private dataset `noilai-release` (`RUNBOOK.md` §0.4); Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`; c01 with
   `CREATE_DATASET = True` creates `noilai-runs`. Blocks: c01 and every chunk; every ingest (`data/runs`). Memo N10.
4. **[1 Oct → now] PAST DUE — Stage-1 pre-registration commit recorded.** Who: author ("your edit, not mine", memo
   15; the assistant never edits `docs/PREREGISTRATION.md`). Steps: `PREREG_COMMIT_STAGE1` = `62f63ae5a737…` (the
   v0.3 freeze commit of this repository, 2026-10-01 00:40Z; `f94f655…` of the original history, `docs/MIGRATION.md`)
   and its date in the PREREG header (both fields read `<fill>` today), `experiments/gates.yaml`
   `preregistration.stage1_commit`, `paper/appendix.tex`; choose where the registration is public (tagged-commit
   URL, OSF or similar) and record `registration_url` / `registration_date` (both null; the driver gates on them).
   Blocks: every confirmatory unit (c08–c31, the API lines). Memo 15.
5. **[5 Oct] PAST DUE — First real run: chunk `c01_cpu_t0`.** Who: author in the Kaggle UI; the assistant ingests.
   Steps (`RUNBOOK.md` §1, §1a): import `notebooks/chunks/c01_cpu_t0.ipynb`; Accelerator None; Internet on; inputs
   `noilai-bundle`, `noilai-release`; secrets of entries 2–3; set `ACCOUNT_HOLDER_ROLE` (a role, never a name) and
   `CREATE_DATASET = True`; Save & Run All. Cell (c') pins the session's model (`pin_panel.py --names <MODELS>
   --apply --partial --out /kaggle/working/noilai_runs_out/panel_pins.json`, saved with the version, not pushed to
   `noilai-runs`), then `smoke_20` on 20 dev items through the HF backend on CPU, then `check_run.py`; the directory
   `smoke_20__<model>__cpu` is pushed. Hand-back: `RUNBOOK.md` §5 (`ledger.py ingest --runs`, `compute_log.py`,
   commit). Expected under an hour (RL-2026-10-01-06). Blocks: proof of the harness on real weights, the CPU-vs-T4
   row comparison of the smoke week, Gate 1 evidence. Memo N10 (a).
6. **[5 Oct, "this week"] PAST DUE — GPL permission requests confirmed sent.** Who: author. Steps: confirm the two
   requests to the maintainers of the two GPL word lists (DD 4.1; `README.md` "Resources and licensing", line 342,
   reads "requested 30 September 2026") went out; if not, send them now. No reply by 8 Nov → entry 30. Blocks: the
   release licence story (`docs/DATA_STATEMENT.md` §H, checklist B2, the dataset cards); no run. Memo 1 (a)+(b), 19 (a).
7. **[by 8 Oct] Reply to the decision memo.** Who: author. Steps: answer `docs/gate1/DD13_DECISION_MEMO.md` with the
   row numbers accepted and any override; rows 7, 17, N1, N2 feed the invitations and the consent form (entry 9),
   row 15 entry 4, rows 3 and 5 the gate file (entry 31); until then nothing is recorded in `experiments/gates.yaml`
   (the assistant never guesses a value). Blocks: entries 4, 9, 20, 31. Memo header.
8. **[before the next assistant session; optional] Assistant-side enablers.** Who: author (environment settings).
   Steps: (a) raise `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` (e.g. 400) so quotas, terms, credit programmes and
   attested sources can be verified; (b) allow `www.kaggle.com`, `huggingface.co`, `cdn-lfs.huggingface.co` in the
   cloud environment's network settings and provide `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` (today both hosts
   are unreachable here and no token exists; the `kaggle` CLI is installed). Blocks: nothing hard; without (b) every
   session goes through `RUNBOOK.md` by hand (the default every entry assumes). Memo N7.
9. **[8 Oct] Validator invitations out; consent form finished.** Who: author (own account, personal messages only;
   `docs/gate1/RECRUITMENT.md` §1); a native speaker for every `[NATIVE-CHECK]` passage; the adult collaborator for
   any step the author's age excludes. Steps: (i) fill `docs/CONSENT_FORM.md` `[NAME]`, `[EMAIL]`, the retention
   date in §5 (HBP §9 decision 6), and §9 `[ADULT NAME]`/`[ADULT EMAIL]` if the author is under 18; (ii) native check
   and remove the markers (today: CONSENT_FORM 12, RECRUITMENT 15, VALIDATOR_INSTRUCTIONS 30, HUMAN_BASELINE_FORM 7,
   HUMAN_BASELINE_PROTOCOL 8, `make_validation_forms.py` 10, `noilai/validation.py` 4); (iii) send message (a) to
   adult native speakers, one each from N, C, S if possible; (iv) letter and region only in the git-ignored
   `data/validation/validators.json`, the letter-to-person key outside the repository; (v) each validator returns
   the ticked consent form. Validators and respondents are different people. Blocks: Gate 1, the packet (entry 11),
   the α row of Table 1, the generator-bug fixes before Gate 2. Memo 7 (3 validators; 2 supported at 6.9 h each
   instead of 5.4 h), N1, N2 (30–45 min stated in the consent text).
10. **[12 Oct] Minimum-age clauses and the account-holder role.** Who: author. Steps: read and record the minimum
    age in the terms of Kaggle, Google (Colab, AI Studio, Forms), Hugging Face, Groq, OpenRouter →
    `configs/models.yaml` `providers.*.terms.minimum_age` (groq and openrouter are `null  # [UNCERTAIN: verify]`)
    and `docs/RISKS.md`; decide the `ACCOUNT_HOLDER_ROLE` string ("author" or "adult collaborator", never a name;
    `RUNBOOK.md` §0.1); a service whose clause the author does not meet runs on the adult collaborator's own
    account (DD 11.2); no account is opened with a misstated age. Blocks: the `ACCOUNT_HOLDER_ROLE` parameter of
    every chunk (so entry 5 in practice), every manifest's `account_holder`, the key holders (entries 20, 32). Memo 17, 8.
11. **[12 Oct] Build and send the validation packet.** Who: author (needs the private v0.3 item files). Steps:
    `pip install -r requirements.txt` (openpyxl, krippendorff); `make validation VALIDATORS="A B C"` (or `"A B"`)
    → `data/validation/validation_<V>.xlsx` per validator, CSV copies, the author's keys (never sent); check
    `data/validation/validation_manifest.json` (`sizes.n_sample` 360, `n_controls` 48, hours per validator); upload
    each workbook to Drive → open as a Google Sheet → share with its validator only. Every command that reads
    `data/validation/` runs in a plain terminal, never inside an AI session. Blocks: calibration (entry 13), Gate 1.
    Memo N1 (30 + 4 planted controls per cell, 2 coders per row, 60 by all three), 7.
12. **[12 Oct] Recruit about 20 human-baseline respondents (not validators).** Who: author. Steps: message (b),
    personal invitations, adults only, nobody in a dependency relation; tell them the window 2–8 Nov; form numbers
    01–20 in the private key (off the repository); reserve volunteers for unreturned forms (HBP §9 decision 3).
    Blocks: the human lane (entry 26), the mean-human band of Table 2. Memo 7 (a third rater only > 20 volunteers), N2.
13. **[13–17 Oct] Calibration round (Part A) and debrief.** Who: validators (≈ 10 min), author. Steps:
    `make validation-score` (`calibration` block); send each validator the key with explanations of the rows they
    missed; under 80% on `correct` → a short call and the second set
    (`make_validation_forms.py calibration2 --dir data/validation --validators <V> --release data/release/v0.3`,
    scored as `calibration_round2`); nobody is excluded on calibration; any instruction change is versioned and
    logged (`docs/gate1/VALIDATION_PROTOCOL.md` §5). Blocks: Gate 1 ("packet sent, calibration done"). Memo N1.
14. **[16 Oct] Gate 1 pilot on Kaggle CPU: chunks `c02`–`c04` (alternative: `c05` on the T4, 2–3 GPU-h).** Who:
    author on Kaggle, one CPU session per chunk (c02 2.3 h, c03 4.6 h, c04 8.5 h worst case; c04 may need
    `TRANSFORMERS_OVERRIDE` if its remote code fails under transformers 5.x); the assistant ingests and computes
    PREREG §10. Steps: as entry 5 per chunk (`MODE = "pilot"`: `smoke_20`, `pilot_t1_200` nfc/nfd with the
    exploratory forced choice, `pilot_xcopa_200`, `check_run.py`); hand back; `ledger.py ingest`; the go/no-go of
    PREREG §10 (strict T1 ≥ 15 points over the copy baseline for ≥ 1 model, cluster-bootstrap CI excluding 0; E3
    scope |Δ| ≥ 5 with McNemar mid-p < 0.05 on ≥ 1 pass-through model; discordance > 35% → raise n or drop cell
    claims). `c05` runs only if the CPU route will not finish by 18 Oct; it needs its three models pinned (entry 19);
    the ledger marks the unused route `cut`. Blocks: Gate 1, the pilot constants of stage 2, the E3 scope decision.
    Memo N11 (a), N4 (forced choice exploratory), N10.
15. **[18 Oct] The C2 placement corpus count.** Who: author or adult collaborator, any machine with internet (a
    laptop or a Kaggle CPU session; no GPU). Steps: `bash scripts/corpus_count_kit.sh` streams the first 2M lines of
    the Vietnamese Wikipedia dump and of CC-100 vi, counts both conventions, writes
    `data/audit/placement_corpus/{viwiki,cc100_vi}.json`; under 10M syllable tokens → re-run with `LINES=4000000`;
    then a RESULTS_LOG entry (corpora, dates, download hashes, both `old_share` values). Blocks: H4's direction (new
    style winning in both registers flips the baseline: a DEVIATIONS row and the C2 arm, not the data), E2's
    `logfreq` covariates, stage 2. Memo 2 ((a) or (b), two registers, ≥ 10M each), N12.
16. **[18 Oct, before the registration link of entry 4 is recorded] Canary digest mismatch (author item).**
    Who: author. The `canary_sha256` in `docs/RESULTS_LOG.md` RL-2026-10-01-01 (`07fb9d79…8ca5`) differs from
    `data/release/v0.3/manifest.json` line 344 and `paper/tables/release_facts.json` (`dedff579…701c`). Steps: find
    which digest the frozen build produced and correct the other record with a dated note (no test covers it); the
    assistant never edits either file. Blocks: the release facts the paper and the registration cite. Memo: none
    (found 7 Oct by the paper review; `paper/claims.md` §12).
17. **[18 Oct] Gate 1 decision (milestone).** Who: author. Steps: the go/no-go of PREREG §10 from the three pilot
    files; the novelty verdict (hand-repeat the arXiv searches of `docs/NOVELTY_SWEEP_2026-09-30.md`); the scope
    decisions (saturation, E3 at full scale, sample sizes); record in RESULTS_LOG / DEVIATIONS. Blocks: every
    "after Gate 1" window; the Bundle B switch is open only here. Memo: none (PREREG §10 binds); N11 feeds it.
18. **[19 Oct – 1 Nov] Validators work Parts B–E; weekly returns scored.** Who: validators; author after each
    return. Steps: `make validation-score` → `data/validation/report/validation_report.json` (α, AC1, marginals,
    control catch rate; under 75% on controls → re-brief) and `data/audit/validator_flags.json`, COMMITTED before any
    API run (`check_api_safety` reads it) and before `make baseline`; every `RULE?` comment checked against the rule
    tables; a confirmed generator bug → fix and full regeneration before any model run (PREREG §5 rule 2). Blocks:
    entries 25, 26, 32; Gate 2. Memo N1, 21 (Part C judges the merger rows), 3 (D2 descriptive), 5 (D3 reported).
19. **[19–25 Oct] Smoke week: the whole-panel pin, the tokenizer and item audits, `c06` (T4) and `c07` (TPU).**
    Who: author on Kaggle (one CPU session with internet for `RUNBOOK.md` §3; GPU T4 x2 for c06; TPU v5e-8 for
    c07); the assistant applies the pins, re-prices, re-packs. Steps: (i) `python scripts/kaggle_cpu_jobs.py all`
    (`audit-tokenizers`, `audit-items`, `pin`) → `data/audit/<name>.json`, `data/audit/items_*.jsonl` (the H1
    identifiability floor), `experiments/panel_pins.json`; hand the files back; at home `python scripts/pin_panel.py
    --apply --from experiments/panel_pins.json` (a full apply is refused until the five panel entries with
    `quantization.checkpoint: null` and a non-bitsandbytes method get a checkpoint; `--partial` applies the rest);
    every `hf_id_status: uncertain` id (24: 21 of 34 self-hosted entries + 3 API; every `revision` null today) resolved
    or removed with a reason; census verdicts per family → `plan_chunks.py --write` (normalizing families' E3 jobs
    move to tier 5); (ii) `c06_gpu_t0`: `smoke_20` on every T4/2xT4 model, 20 dev items (a `limit` line, so unpinned
    is allowed); compare its CPU-model rows with c01's row by row (a disagreement is reported, not averaged away);
    record tokens/s per family (DD 8.5 re-pricing) and whether `max_model_len` 2048 fits (memo N3); read the 2xT4
    quota accounting during this session (entry 1); (iii) `c07_tpu_t0`: the TPU trio, 20 dev items (6–22 min
    vLLM-TPU start-ups). `compute_log.py` after each session. Blocks: the panel freeze (entry 22), every E1/E3 run,
    the census column, the H1 floor, E2's item audits, the tier-5 re-pack. Memo 9 (a), 11, N3, N8, N11.
20. **[25 Oct] Gemini route decided and recorded.** Who: author (+ an adult holding a Google AI Studio key if (b)).
    Steps: (a) drop Gemini (panel of 19; the paper names the drop) or (b) dev-only on the dev-derived API set with an
    adult's key, reported separately and never in Table 2 — (b) needs a `noilai_api_dev` `derive` block and a run
    line in `configs/run_plan.yaml` (DD 4.5; `docs/API_ROUTES.md` §2), Flash-Lite only; the paid key is excluded by
    the no-money rule; record `decisions.gemini_route` (`dev_derived_set` or `drop`, + `decided_on`) in
    `experiments/gates.yaml`. Blocks: the API lane, the Table 2 footnote, the panel count. Memo 8, 18, 10.
21. **[25 Oct] Stage-2 date chosen and the plan's weeks adjusted.** Who: author. Steps: (b) commit stage 2 right
    after the panel freeze and the validators' generator fixes if no co-author has joined (four test-split weeks:
    everything fits at x1, tiers 1–2 at x2); (a) 8 Nov if a co-author is editing (two weeks: through tier 2 at x1,
    inside tier 1 at x2; `docs/COMPUTE_PLAN.md` §4); a DEVIATIONS row and the chunk windows record the choice.
    Blocks: the windows of c08–c31, the Groq days (entry 32). Memo N13.
22. **[25 Oct] Panel-freeze commit (PREREG §7).** Who: author. Steps: one commit that (a) applies the pins of entry
    19 (`configs/models.yaml` revisions; the pin record is `experiments/panel_pins.json` — memo row 9 still names
    `configs/panel_manifest.json`, which no longer exists), (b) chooses the five missing `quantization.checkpoint`
    values or removes those entries, (c) records each entry's smoke result and census verdict per engine, (d) the
    free-tier limits read from the providers' consoles (entries 20, 32), (e) tokenizer SHA-256s (a tokenizer-fixed
    control pair needs byte-identical files, else the control claim is dropped); the 27B control pair is a case
    study, no new model is added; settle COMPUTE_PLAN Q-C2 (one `family` line + `plan_chunks.py --write`), Q-C3,
    Q-C4. Blocks: every E1/E3 run; stage 2 (panel, families); no model enters the main tables after it. Memo 9, 11.
23. **[25 Oct] Co-author outreach checkpoint (sent in week 1, 5–11 Oct).** Who: author. Steps: e-mail 5–10 PhD
    students/postdocs (`docs/CONTRIBUTOR_OUTREACH.md`; attach the tokenizer-audit result and the plan's abstract);
    offer E2 and E4 as the open sections; no reply by 25 Oct → widen (ACL mentorship Slack, SEACrowd, Cohere Labs)
    by Gate 2; "in principle" by 8 Nov; by 18 Dec lottery vs the ACL 2027 SRW. Also week 1: ORCID; OpenReview
    profile (activation can take two weeks). Blocks: the stage-2 date (entry 21), the ARR service-contributor
    requirement, the IRB route. Memo 16, 12, N13.
24. **[before 2 Nov] Native check of the baseline form's Vietnamese strings.** Who: a native speaker (the strings
    live in `scripts/make_validation_forms.py` and `noilai/validation.py`; HBP §3.3). Blocks: `make baseline` /
    `build_forms.gs` (entry 26). Memo N2.
25. **[2–5 Nov] Scoring and adjudication.** Who: author; the third validator (blind). Steps: final
    `make validation-score`; `make_validation_forms.py adjudication-sheet --to <V>` once per letter; the rule of
    `docs/gate1/VALIDATION_PROTOCOL.md` §6 (unanimous → majority → author decides against the rule tables, logged in
    `data/validation/adjudication.tsv`; never overriding agreeing validators); confirmed rule bugs → fix and full
    regeneration before any test-split run (by 8 Nov); D2 qu- and D3 i/y results reported descriptively; attested
    rows: `attested_verified.tsv` (≥ 2 `valid` = Có, none Không). Blocks: Gate 2, the stage-2 commit (entry 31), the
    pre-send gate of the baseline, every test-split run. Memo 3, 5, 21, N1.
26. **[2 Nov; window 2–8 Nov] Human baseline: build, send, import, score.** Who: author (own Google account or the
    adult collaborator's); respondents (30–45 min each, once). Steps: 0. pre-send gate: no `RULE?` flag open (else
    after adjudication closes on 5 Nov); `data/audit/validator_flags.json` committed; 1. `make baseline`
    (`--exclude-flags`, `--attested-verified`; HBP §9 decisions 2 and 5 are implemented since 1 Oct — confirm them
    with a DEVIATIONS note before the build); 2. `make_validation_forms.py google-form --dir data/human
    --contact-email <address>` → `data/human/build_forms.gs`, run in script.google.com; 3. one link per person, the
    key kept privately; 4. after the window: `responses_form_<nn>.csv` into `data/human/responses/` →
    `make_validation_forms.py import-responses` (first submission per form) → `make ingest-sheets`
    (= `validation-score` + `baseline-score`; the one sheet path: item-level rows in the git-ignored
    `data/human/report/human_scores.jsonl`, aggregates with the file's SHA-256 in
    `experiments/human/human_baseline_report.json`); 5. delete the 20 Forms and empty the Drive trash. A confirmed
    bug that changes any of the 246 items → rebuild and re-run before 22 Nov, first round reported as superseded.
    Blocks: the mean-human band (Table 2, open models only; HBP §9 decision 4), the two-rater α. Memo N2, 7.
27. **[5 Nov] Consent-form promise on the validators' Sheets.** Who: author. Steps: after each validator's last
    return, download the Sheet as .xlsx into `data/validation/returned/`, delete the Sheet, empty the Drive trash,
    delete any e-mailed workbook (`docs/CONSENT_FORM.md` §5). Blocks: nothing; a stated obligation. Memo: none.
28. **[6 Nov, optional] EACL 2027 SRW question to the chairs** (address in `docs/WEEK1_CHECKLIST.md`): whether a
    high-school first author is eligible and may still submit to ARR. Who: author. Blocks no run. Memo: none.
29. **[8 Nov] Attested set: page-level source checks, merge, books, the PACLIC 2018 sentence set.** Who: author
    (+ Part C validators). Steps: (a) for every candidate row Part C accepts, open its URL and set `source_checked`
    in `data/attested_candidates.tsv` (102 rows = 85 `candidate` + 17 `duplicate_of_seed`, `source_checked` empty on
    all); merge rows native-verified AND page-checked into `data/attested_seed.tsv` (34 rows, `verified_by` empty on
    all), rebuild `attested.jsonl` ONLY and re-record its hash in `configs/run_plan.yaml` with a DEVIATIONS row (the
    author's edit; DD 4.1 allows late rows); (b) the floor: at most 79 H6-eligible rows even if every candidate
    verifies (floor 100) → a manual pass over *Thú chơi chữ* and two other collections if the books can be borrowed,
    else H6 descriptive on 8 Nov; (c) e-mail the authors of the PACLIC 2018 paper (Y18-1063) for their sentence
    set. Blocks: whether H6 is tested (stage 2), `E1_attested`. Memo 6, N5, N6, 21.
30. **[8 Nov] GPL decision point.** Who: author. Steps: no reply to entry 6 → the GPLv2 fallback with release story
    (a): pseudo-pair dev set public under Apache-2.0/CC BY, lexical items only in the gated set, the gated file
    labelled GPLv2 and redistributable (DD 4.1); update DATA_STATEMENT §H, the dataset cards, checklist B2. Blocks:
    the data statement and dataset cards; no run. Memo 1, 19.
31. **[8 Nov at the latest; 25 Oct under memo N13 (b)] The stage-2 commit, the gate file, the scoring-rule
    freeze.** Who: author (and the co-author if one has joined); the assistant drafts nothing binding. Steps: write
    stage 2 — hypotheses with tests, the analysis plan, the E4 protocol, the pilot constants (entry 14), the PREREG
    §8.14 exploratory list incl. the T1 forced choice (memo N4), the corpus source (entry 15), the H6 decision (entry
    29), the panel and families (entry 22), the A3 boundary and A4 position-group rulings, the qu / i-y rulings
    (memo 3, 5: keep v0.3 as frozen) — and commit it; `PREREG_COMMIT_STAGE2` = that commit in the PREREG header and
    `paper/appendix.tex`; in `experiments/gates.yaml` record `stage2_commit`, `registration_url`, `registration_date`,
    `decisions.qu_convention`, `decisions.iy_emission` (value + `decided_on`); then the author runs `python
    scripts/freeze_scoring_rule.py --record` before anyone looks at a confirmatory result; `kaggle_run_plan.py`
    enforces all of it. No test-split run before this commit (DD 8.8). Blocks: every confirmatory unit (c08–c31),
    every API line, the whole "after the stage-2 commit" window. Memo 15, N13, N4, 3, 5, A3, A4, 2, 6, 11.
32. **[from the stage-2 commit; daily; done by 22 Nov] Groq API runs.** Who: the `GROQ_API_KEY` holder (an account
    meeting Groq's terms, entry 10); the assistant ingests. Steps: first read the live caps on the Groq console and
    whether they are per model or per organisation (`docs/API_ROUTES.md` §2: 9.25 days per model, 18.5 for both if
    per org — the per-org reading does not fit between an 8 Nov stage 2 and 22 Nov: memo N13 (b), or the thinking
    line after the freeze); `notebooks/api_runs.ipynb` with the four API lines, Gemini out unless entry 20 chose (b);
    once a day; the ledger parks a model at its cap; save raw outputs at once (catalogue risk). Blocks: the four API
    lines (`E1_api_core`, `E1_api_paraphrase`, `E3_api_core`, `reasoning_500_api`). Memo 8, 17, N13.
33. **[stage-2 commit → 22 Nov] GPU and TPU chunks `c08`–`c31` in order.** Who: author on Kaggle (or the assistant
    if entry 8 (b) is granted); the assistant ingests. Steps: one session per chunk in the order of
    `configs/compute_chunks.yaml` (tier 1 c08–c16 first; `p2` splits and normalizing-family E3 jobs in tier 5);
    re-run a chunk until every job is `ok`; `compute_log.py` after each; `plan_chunks.py --write` after the first
    chunk of a tier if measured hours differ; quota out → the pre-registered cut order decides what is dropped,
    every cut model named (GPU 83.3 h, TPU 28.3 h at the high estimate, COMPUTE_PLAN §4). The PhoGPT bf16
    reference stays unscheduled: drop it from the bf16 appendix and say so (the author edits `configs/run_plan.yaml`
    `bf16_drift_200.fallback`). Blocks: Gate 3. Memo 11, N9 (a), N13, Q-C1..Q-C4.
34. **[before the first confirmatory ingest; hard by 22 Nov] Analysis code the DD names "to write".** Who:
    assistant (after the pilot). Steps: `noilai/stats/pooling.py`, `aggregate.py`, `analyze.py` +
    `configs/analysis.yaml`, release encryption, weighted validation sampling. Blocks: the pooled estimates (H3/H4/H6),
    not any run. Memo: none.
35. **[22 Nov] Results freeze (Gate 3; milestone).** Who: author. Withdrawal possible until then (consent §4); after
    publication delete the form-number key and `data/human/responses/` (HBP §4 step 11). Memo: none.
36. **[23 Nov – 6 Dec] E4: notebook resume and the protocol.** Who: assistant (code), author on 2xT4 (5–15 GPU-h,
    two quota weeks). Steps: once the E4 protocol is in stage 2, add checkpoint/resume to
    `notebooks/colab_probe_gemma3.ipynb`; run after the results freeze; Gate 4 (6 Dec): keep probing only if
    patching is clean. Memo A4 (position groups (a)), 13 (Interpretability area only if readout-B retention ≥ 50).
37. **[Dec] Paper-stage author decisions.** Who: author. Steps: human-subjects route (a co-author's IRB exempt
    determination if one with an institution joins before December, else checklist D4 "No" with the justification);
    title and ARR area; Gate 5 18 Dec paper freeze and the service contributor (lottery vs SRW if nobody vouches);
    ARR 4 Jan 2027 23:59 AoE; OpenReview profiles complete within a week of it. Memo 12, 13, 20.
38. **[optional; December] OpenAI Researcher Access Program application** (draft in `docs/API_ROUTES.md` §3; the
    applicant checks eligibility and age first; credits would arrive after 4 Jan). Who: author. Memo: none.
