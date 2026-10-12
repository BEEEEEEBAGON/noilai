# Human-baseline protocol (runs after Gate 1, week of 2–8 November 2026)

Status: **run-ready draft, 1 October 2026.** It implements DESIGN_DECISIONS 10.2 as amended on 1 October 2026
(`docs/DEVIATIONS.md`, row "DESIGN_DECISIONS 10.2 (baseline forms carry the models' exact prompt)") and the
pre-registered exclusions and analysis (`docs/PREREGISTRATION.md` §5 rule 7, §8.11). It changes neither the
pre-registration nor the frozen data. No form has been built and no response exists; every result below is a
placeholder until the run fills it.

Who reads this: the **author** (runs every step). Respondents see only the recruitment message
(`docs/gate1/RECRUITMENT.md`, message (b)), the consent form (`docs/CONSENT_FORM.md`) and their Google Form. The form
design and its assignment arithmetic are in `docs/HUMAN_BASELINE_FORM.md`; the code is
`scripts/make_validation_forms.py` (`baseline`, `google-form`, `import-responses`, `score-baseline`) and the Makefile
target `make baseline`.

## 1. Purpose and what the number means

- **What it gives.** A human reference for Table 2: the accuracy of adult native speakers on the same items the models
  are scored on, shown in the same format, scored by the same function (`noilai.eval.score.score_outputs`).
- **A reference band, not a contest** (DD 10.2). The pre-registered precision of the human estimate (PREREG 8.11) is
  too coarse for cell-level claims, so the paper reports the human number as a band beside the models. **No
  "superhuman" claim and no per-cell human–model comparison** is pre-registered or made.
- **What it measures.** Most generated T1 golds are pseudo-phrases (DD 10.2), so on the generated items the baseline
  measures native speakers following explicit instructions (phụ âm đầu / vần / thanh điệu [NATIVE-CHECK]) on rule-generated component
  permutations. Folk practice bends outputs toward real words (the rule output may be *giầy* where the folk form is
  *giày*; DD 10.2, 5.4), so a strict score understates what speakers know. Therefore every human answer is scored three
  ways, and all three are reported:

  | scoring | an answer counts as correct when | source in the scorer's output |
  |---|---|---|
  | **strict** | it is the gold, in the named order (either tone-mark placement and *lí/lý* accepted; misspellings not repaired) | `correct` |
  | **lenient** | strict, or the gold's two syllables in the other order | `correct_lenient` |
  | **doublet/homophone-tolerant** | strict, or it differs from the gold only by a listed regional homophone spelling or an *ay/ây* doublet | `correct` or `error_class` ∈ {`homophone`, `doublet`} |

- **The cultural task.** A 10-item **natural-competence block** (T2 on 10 attested, non-vulgar two-syllable nói lái
  drawn at random from the attested file) gives the human reference for decoding real wordplay, which the generated
  items do not test (DD 10.2). The paper reserves the word "nói lái" for the attested set and calls the generated items
  rule-generated component permutations. The draw (`natural_block` in the script) keeps one row per original, skips
  rows whose source or note says `[UNCERTAIN]`, "same source" or "illustration" and rows a validator flagged offensive,
  and, once Part C is scored, draws only native-verified rows (`--attested-verified`; open decision 5 in §9, settled
  in the script on 1 October 2026).

## 2. Who takes part

- **About 20 adult native speakers** (18 or older), unpaid volunteers, recruited by personal invitation only
  (`docs/gate1/RECRUITMENT.md` §1 and message (b)); no one in a dependency relation with the author; written consent
  (`docs/CONSENT_FORM.md`, role "human-baseline respondent", plus the three required consent boxes on the form's first
  page).
- **Never the validators.** Validators have seen test items and the rule tables. They are never sent a baseline link,
  and the form's closing question `WAS_VALIDATOR` screens for it (a "Có" excludes the form, §5).
- **Region recorded.** The form asks the region the respondent grew up in (Bắc / Trung / Nam / Ngoài Việt Nam), plus age
  band (required), years lived in Vietnam and whether they were raised abroad. Nothing else is collected; e-mail
  collection is off. [NATIVE-CHECK] the region labels match the form.
- **Anonymous.** Each person gets one form number (01–20). The form-number-to-person key lives only in the author's
  private list, outside the repository, and is deleted after publication.

## 3. The items and the form: the same items and format as the models

### 3.1 Items

- **Source:** the open-model main sample `data/release/v0.3/noilai_main.jsonl` (the run plan's seeded stratification of
  the test split). Items flagged vulgar and spelling twins are left out by the builder.
- **6 anchors on every form:** one T1 item per generated kind (V1–V4), one T2 item and one T3 item. They measure
  between-person agreement on identical material.
- **240 double-judged items:** 20 per task × variant cell over the 12 cells (T1, T2, T3 × V1–V4); T3 half yes and half
  no, never both members of a twin pair and never an anchor's twin. Each is on exactly two forms by the **cyclic
  double-coverage** assignment of `docs/HUMAN_BASELINE_FORM.md` §1: every respondent judges 24 distinct items besides the
  anchors, and every pair of forms shares one or two items (140 pairs share one, 50 share two), so the rater graph is
  connected for α and for the two-way bootstrap.
- **6 + 240 = 246 distinct items**, listed in `data/human/human_items.json`. **Every open model is scored on exactly
  these 246 items**: they are drawn from the main sample, which every open model runs in full, so for open models the
  human–model comparison needs no extra run. API models run the core only (DD §4.5, `noilai_core.jsonl`), which need not contain
  all 246 items; whether they are also run on the 246 items, or the human row is compared with open models only, is
  open decision 4 in §9.
- `make baseline` tests five values of the builder's printout (`forms` = 20, `per_form` = 30, `distinct_items` = 246,
  `min_appearances` = 2, `max_appearances` = 20) and **deletes the forms and fails** if any of them differs. It does not
  test that the anchors are on all 20 forms and every other item on exactly 2 (`anchors_seen_by`, `others`), nor
  `rater_graph_connected` or `natural_block_items`; the author checks those by eye (§4 step 1). (Pending Makefile
  change: test all nine values and print JSON.)

### 3.2 Same format as the models

Each of the 30 model items on a form carries, as the **help text under its question**, the exact p0 user message the
models receive under the main condition: the task instruction (for T1 and T3 the kind is named and defined; for T2 it is
withheld), the three fixed demonstrations, the item, and the `Đáp án:` answer line. It is rendered by
`noilai.eval.prompts.render(item, paraphrase="p0", shots=3, arm="nfc")`, the function the harness uses, and its hash
(`noilai.eval.prompts.prompt_hash`) is stored in the form's `prompt_hash` column. The harness writes the same
`prompt_hash` on every model output, so the analysis can check, item by item, that humans and models saw the same
message (§6.2).

The question title above the help text is a short question for orientation (`baseline_prompt` in the script)
[NATIVE-CHECK]:

| task | title shown | answer box |
|---|---|---|
| T1 | `Nói lái kiểu <V> của “<input>” là gì?` | short text |
| T2 | `“<input>” là cách nói lái của cụm từ nào?` | short text |
| T3 | `“<candidate>” có phải là nói lái kiểu <V> của “<input>” không? (Có/Không)` | Có / Không |

Differences from the models, stated in the paper: humans see the short title as well as the prompt; humans type the
answer into a box, so no answer extraction is needed (`extraction_method` = `human`); a blank answer counts as wrong,
exactly like an empty model output.

### 3.3 One form, page by page

Built by the Apps Script that `google-form` writes (`data/human/build_forms.gs`); 43 questions per form, verified on a
synthetic 20 × 30 build:

| page | content | required |
|---|---|---|
| description | the form's introduction (`FORM_INTRO_VI` in the script; time about 30–45 minutes) | — |
| 1. consent | one question "Đồng ý tham gia" with three boxes: 18 or older; read the information sheet and agree (voluntary, unpaid, can stop at any time); agree to storage and aggregate, anonymous publication. The form cannot be submitted unless all three are ticked. The page carries no summary of the consent form (no contact, no withdrawal route, no statement that no ethics board reviewed the study), and the form shows Google's default message after submission (gap and proposed strings: `docs/HUMAN_BASELINE_FORM.md` §2.1) | yes |
| 2. demographics | age band 18–29 / 30–49 / 50+ (required); region grown up in; years lived in Vietnam (< 5 / 5–15 / > 15); raised abroad (Có / Không) | age band only |
| 3. questions, positions 1–31 | the 30 model items in a seeded random order (T1, T2, T3 interleaved) with the instruction-check row `CHECK-01` at a random position after the first five ("write exactly the two words *đã đọc*") | no |
| 3. questions, positions 32–41 | the natural-competence block: 10 attested, non-vulgar two-syllable nói lái drawn at random from the attested file (§1), the same 10 on every form, each asked as "“X” là cách nói lái của cụm từ nào?" with no model prompt | no |
| 3. questions, positions 42–43 | `TOOLS` (used a dictionary, search engine or AI assistant?) and `WAS_VALIDATOR` (were you a validator?), Có / Không | yes |
| end | optional free comment | no |

[NATIVE-CHECK] every Vietnamese string the form shows comes from `scripts/make_validation_forms.py` (`FORM_INTRO_VI`,
`CONSENT_BOXES_VI`, `DEMOGRAPHICS_VI`, `CHECK_ITEM`, `CLOSING_ITEMS`, `baseline_prompt`, `natural_block`, and the
titles set in `buildOne` inside `cmd_google_form`); the native check is done on those strings, never on the model
prompt, which must stay identical to what the models receive. `CONSENT_BOXES_VI` and `DEMOGRAPHICS_VI` carry no
in-string marker: check them by name. The same holds for the titles in `buildOne`: the form title `NóiLái - phiếu <nn>`,
the consent question `Đồng ý tham gia` and its summary `CONSENT_HELP_VI`, the confirmation message `CONFIRMATION_VI`,
the page titles `Thông tin chung (khái quát, không định danh)` and `Các câu hỏi`, the choices `Có` / `Không`, and the
closing `Góp ý (không bắt buộc)`. A `# [NATIVE-CHECK]` comment above `FORM_INTRO_VI` covers the whole block.

## 4. Runbook

Commands run from the repository root on the author's machine (the release files exist only there). Each step says
what it writes and what to check.

0. **Pre-send gate (before step 1).** Build and send only when no rule bug is open: every row listed under
   `B.flags.rule_flag_rows` in `data/validation/report/validation_report.json` (`rule_flag` = True in
   `data/validation/report/item_flags.tsv`) has been checked against the rule tables, with the outcome logged in
   `docs/DEVIATIONS.md` (confirmed bug) or `docs/RESULTS_LOG.md` (not a bug). The scorer does not scan `RULE?` notes in
   Parts A, C and D (`D1_production`, `D2_qu`, `D3_conventions`) or E, so read the `comment` columns of those sheets by
   hand. (`data/validation/adjudication.tsv` is not the place to look: it lists only Part B rows that need the author's
   decision on `correct`, and has no field for a rule bug.) If a bug is confirmed, PREREG §5 rule 2 requires a fix and
   a full regeneration before any model run; the main sample and therefore the 246 items may change. Then wait for the regenerated release, build the forms from it, and run the
   baseline after the regeneration and before the results freeze (22 November 2026), telling respondents the new dates.
   (The adjudication window, 2–5 November, overlaps the baseline window; see open decision 1 in §9.)
1. **Build the forms.**
   `make baseline`
   Writes `data/human/baseline_form_01.csv` … `baseline_form_20.csv`, `human_items.json` and `baseline_manifest.json`.
   The target prints the builder's line as JSON and checks nine values itself: `'forms': 20`,
   `'per_form': 30`, `'distinct_items': 246`, `'min_appearances': 2`, `'max_appearances': 20`,
   `'anchors_seen_by': 20`, `'others': [2]`, `'rater_graph_connected': true`; it fails and deletes the forms if any
   differs. Check `natural_block_items` (10) by eye. Keep the printout with the private notes.

   The builder leaves out items flagged `vulgar` at generation and, through `--exclude-flags
   data/audit/validator_flags.json` (written by `make validation-score`), every item a validator flagged offensive;
   the printout's `items_excluded_by_flags` gives the count and `validator_flags` the file read (null = the file was
   missing: score the validation first). As a second check, run from the repository root:

   ```bash
   python - <<'EOF'
   import csv, json
   from noilai.validation import canonical_text as k
   R = "data/validation/report/"
   t = lambda p: list(csv.DictReader(open(R + p, encoding="utf-8"), delimiter="\t"))
   f = {r["item_id"] for r in t("item_flags.tsv") if r["offensive_any"] == "True"}             # Part B, by item id
   o = {k(r["input"]) for r in t("attested_verified.tsv") if r["offensive_any"] == "True"}    # Part C, by phrase
   h = set(json.load(open("data/human/human_items.json", encoding="utf-8")))
   m = json.load(open("data/human/baseline_manifest.json", encoding="utf-8"))
   print(sorted(f & h), [n["expected"] for n in m["natural_block"] if k(n["expected"]) in o])
   EOF
   ```

   It must print `[] []`. Otherwise the listed items are on the forms: do not send; re-run `make validation-score`
   and `make baseline`, then this check. Part B items are matched by item id, Part C
   rows by their original phrase. (Tested on a simulated return with one planted Part C flag: it printed
   `[] ['đầu tiên']`.)
2. **Write the Apps Script.**
   `python scripts/make_validation_forms.py google-form --dir data/human`
   Writes `data/human/build_forms.gs`.
3. **Remove the review markers** (only after a native speaker has checked the strings listed in §3.3):
   `sed -i 's/ \[NATIVE-CHECK\]//g' data/human/build_forms.gs` then
   `grep -c "NATIVE-CHECK" data/human/build_forms.gs` must print `0`. (The markers are part of the script's strings and
   would otherwise appear on the live form. The consent boxes, the demographic labels and the titles in `buildOne`
   carry none, so a clean grep does not show that they were checked; §3.3.)
4. **Create the 20 Google Forms in the author's own Google account** (or an adult collaborator's; never an account
   opened with a misstated age). Open script.google.com → New project → delete the placeholder code → paste the whole of
   `build_forms.gs` → save → select `buildAll` → Run → authorize the script for that account. The execution log lists
   one line per form, written as each form is made: the form number, a tab, the form's link. Copy those lines into the
   private list, never into the repository. If the run stops part-way (an error, or Apps Script's per-run time limit),
   the log holds the forms made so far: run `buildRange(first, last)` for the missing numbers only (or delete the forms
   found under `NóiLái - phiếu` in Drive and run `buildAll` again).
5. **Check one form end to end before sending anything.** Open form 01 and check: the form cannot be submitted with a
   consent box unticked; the age band is required; each model item shows the full prompt as help text; titles are
   numbered 1–43; `TOOLS` and `WAS_VALIDATOR` are required; the consent question shows its summary (voluntary,
   unpaid, no ethics board, how to withdraw with the form number) and the confirmation after submitting states the
   form number (`docs/HUMAN_BASELINE_FORM.md` §2.1; `google-form --contact-email` fills the address). If you submit a test response, delete it afterwards (Responses tab → menu → Delete all responses) so that it
   is not imported.
6. **Send one link per person.** Personal message from the author's own account, with the consent form as a PDF
   (`docs/gate1/RECRUITMENT.md` §1 step 7). One form number per person; record the number against the person in the
   private list; state the form number in the message as well (it is also in the form's title and the confirmation
   message); ask them not to forward the link. Never send a link to a validator.
7. **Close and download.** At the end of the window turn off "Accepting responses" in each form. For each form:
   Responses tab → ⋮ → Download responses (.csv); do not link a response Sheet, which would keep a timestamped copy of
   the responses in Google Drive. If the download arrives as a .zip, unzip it. Save the file as
   `data/human/responses/responses_form_<nn>.csv` (`<nn>` = the form number, e.g. `responses_form_07.csv`).
8. **Import.**
   `python scripts/make_validation_forms.py import-responses --dir data/human`
   Writes one file per response, `data/human/returned/baseline_form_<nn>_r<k>.csv` (answers mapped back to item ids by
   the position number at the start of each question title), and `data/human/returned/demographics.csv` (form,
   response number and the four coarse fields; timestamps are not kept). The importer keeps only the first response of
   each form (open decision 2 in §9) and prints how many later ones it dropped unread (`extra_submissions_dropped`);
   that count goes to `docs/RESULTS_LOG.md`.
9. **Score.**
   `python scripts/make_validation_forms.py score-baseline --items data/release/v0.3/noilai_main.jsonl --dir data/human --returned 'data/human/returned/baseline_form_*.csv'`
   Writes `data/human/report/human_scores.jsonl` (one scored row per answer: `correct`, `correct_lenient`,
   `error_class`, ...) and `data/human/report/human_baseline_report.json` (§6.1). Record the exclusion counts in
   `docs/RESULTS_LOG.md`.
10. **Analyse** (§6.2) and fill the paper's placeholders from the analysis output only.
11. **Once the import (step 8) has been checked:** delete the 20 Google Forms (and any response Sheet linked by
    mistake) in Google Drive and empty the Drive trash. **After publication:** delete the form-number-to-person key and the downloaded
    response CSVs that still carry timestamps (`data/human/responses/`). Keep `returned/` and `report/` privately
    (git-ignored), only to recompute the published aggregates, until the retention date of `docs/CONSENT_FORM.md` §5
    (`[RETENTION — author decides]`; open decision 6 in §9), then delete them. They are never released: the release
    carries the aggregates only.

## 5. Exclusions (PREREG §5 rule 7, as implemented in `score-baseline`)

A returned form is excluded, with the first reason that applies recorded in `forms_excluded`:

1. fewer than **15 of the 30** model items answered (the check row, the natural block and the closing rows do not
   count);
2. the instruction-check row `CHECK-01` is not "đã đọc" after normalization (case, encoding and surrounding punctuation
   are ignored; an answer without diacritics fails) [NATIVE-CHECK];
3. `TOOLS` = Có (used a dictionary, a search engine or an AI assistant);
4. `WAS_VALIDATOR` = Có. This fourth reason is not in the text of PREREG §5 rule 7; it comes from the 1 October 2026
   amendment (`docs/DEVIATIONS.md`) and is reported as such.

**All counts are reported**: forms sent, forms returned, forms excluded by reason. Within a scored form a blank answer
counts as wrong; no answer, item or respondent is dropped after scoring.

## 6. Analysis (PREREG §8.11)

### 6.1 What `score-baseline` produces today

- `human_scores.jsonl`: every answer of every scored form, scored by `score_outputs` (the models' scorer) with its
  `error_class`, `correct` and `correct_lenient`, and the form it came from.
- `human_baseline_report.json`: forms scored, forms excluded with reasons, per form the number answered and the
  natural-block score (`natural_correct` of `natural_n`; an answer counts when its two syllables are the attested
  original's, in either order), and the **strict** mean-human accuracy per task as a point estimate. It does not compute
  intervals, α, the lenient and tolerant tables, or the lexicality and region splits.

### 6.2 The analysis step (code to write; not blocked on the author)

All of the following read `human_scores.jsonl`, `returned/demographics.csv` (joined on form and response number),
`human_items.json` and the item file:

- **Item-level human accuracy** = the mean of the item's judgments (two for a double-judged item; for an anchor, every
  scored respondent).
- **Mean-human accuracy** per task × variant and pooled = the mean of the item-level accuracies: **the comparator**. A
  model is one rater. "Any human correct" is a ceiling, reported separately.
- **95% interval** from a **two-way (person × item) cluster bootstrap**: resample respondents and items independently
  with replacement and recompute.
- Strict, lenient and doublet/homophone-tolerant scoring (§1), all three reported.
- **By output lexicality** (`strata.output_lexical` of the item) and **by region** of the respondent; hỏi/ngã V3 items
  and items with final -n/-ng reported separately for Southern respondents (DD 10.2).
- **Agreement:** nominal Krippendorff's α between the two raters over the 240 double-judged items, on correctness and on
  the produced answer (canonical text), with a bootstrap CI (`noilai.stats.agreement`).
- The human error taxonomy (`error_class` counts) beside the models'.
- The natural-competence block: per-respondent scores and their mean.
- **Prompt check:** for each of the 246 items, the form's `prompt_hash` equals the `prompt_hash` of the model outputs
  for the same item under p0, three demonstrations and the NFC arm; any mismatch is reported (it means the prompts
  changed between building the forms and the run).
- Table 2's human row: mean-human accuracy with the number of scored respondents and the 246 items; no cell-level
  human–model comparison.

## 7. Data handling

- `data/human/` is git-ignored: forms, response CSVs, returned files and reports never enter the repository and never
  enter the bundle a Kaggle notebook clones.
- Nothing in `data/human/` is ever sent to any AI service (this includes pasting answers or reports into a chat
  assistant). The paper and the data release carry aggregates only; the form-number-to-person key is never released.
- E-mail collection is off; the form asks no name. The form-number-to-person key is held in the author's private list,
  separately from the answers, and deleted after publication. (Respondents return no consent PDF: they tick the three
  boxes on the form.)
- Google keeps the responses inside each form until the form is deleted; the 20 forms are deleted once the import has
  been checked (§4 step 11). Responses are downloaded as CSV directly, never through a linked Sheet.
- `returned/` and `report/` are kept privately only to recompute the published aggregates, until the retention date of
  `docs/CONSENT_FORM.md` §5, and are never released.
- Withdrawal: a respondent who gives their form number before the results freeze (22 November 2026) has their response
  removed and the analysis re-run; after it, their raw data are still deleted on request (`docs/CONSENT_FORM.md`).
- The forms show no vulgar item: the builder excludes items flagged vulgar at generation, the natural block uses only
  attested rows not marked vulgar, and §4 step 1 checks that no item flagged offensive by a validator is on the forms.

## 8. Timeline

| date | step | who |
|---|---|---|
| by 12 Oct 2026 | recruit about 20 respondents (message (b)), different people from the validators; tell them the window | author |
| 13 Oct – 1 Nov | native check of the form strings in the script (§3.3); validators do Parts A–E | author, a native speaker |
| 18 Oct | Gate 1 (validation packet sent, calibration done) | |
| 2–5 Nov | validation scoring and adjudication; the pre-send gate (§4 step 0) | author |
| 2–8 Nov | **human baseline**: build (steps 1–5), send links (step 6), responses come in | author, respondents |
| by 8 Nov | confirmed rule bugs fixed and data regenerated before any test-split run (Gate 2); if this changes the items, the baseline moves after it (§4 step 0) | author |
| after the window | import, score, analyse (steps 7–10) | author |
| after the import | delete the 20 Google Forms and empty the Drive trash (step 11) | author |
| 22 Nov | results freeze; withdrawal possible until then | |
| after publication | delete the form-number-to-person key and `data/human/responses/` (step 11); `returned/` and `report/` at the retention date | author |

## 9. Open decisions for the author (recommendation first)

1. **Window squeeze.** Adjudication (2–5 Nov) and the baseline window (2–8 Nov) overlap, and a confirmed bug can change
   the items. *Recommended:* send links on 2 Nov if no `RULE?` flag is open in the returns received so far (the
   `rule_flag_rows` of Part B and the `comment` columns of Parts A, C, D and E, §4 step 0); if a bug
   is confirmed later and regeneration changes any of the 246 items, rebuild and re-run the baseline before 22 Nov and
   report the first round as superseded. *Alternative:* send links only after adjudication closes (5 Nov), leaving three
   days for responses.
2. **A form with more than one response** (a forwarded link). Not covered by the pre-registration; today the importer
   writes every response and each is scored as its own respondent. A second response comes from someone the author did
   not invite, whose age and non-validator status rest only on boxes they ticked themselves. *Recommended:* import only
   the first response of each form (`_r1`, the first row of the downloaded CSV); delete every later response unread, as
   the consent form does for a form returned with the 18-or-older box unticked, and report only their count. Log the
   rule as a `docs/DEVIATIONS.md` row before any response is read; it also keeps the double-coverage design intact.
   **Implemented** in `import-responses` on 1 October 2026 (DEVIATIONS row of that date): only the first response is
   written; the count of later ones is printed. Delete the later responses from the downloaded CSVs afterwards.
3. **Fewer than 20 respondents.** *Recommended:* do not rebuild with fewer forms (that would change the 246 items the
   models are scored on); re-send an unreturned form to a reserve volunteer within the window; report items with fewer
   than two judgments.
4. **API models and the 246 items.** API models run the core (`noilai_core.jsonl`), not the main sample, so only
   those of the 246 items that are also in the core have API output (count them with `human_items.json`). *Recommended:* compare the human row with open models only in Table 2 and say so
   in its caption; running API models on the 246 items would add item files and calls to the zero-budget API plan.
   *Alternative:* add the 246 items to the API runs, if the free-tier quota allows.
5. **The natural-competence block.** `natural_block` shuffles every two-syllable attested row not marked vulgar (seed
   20261102) and takes the first 10; it does not de-duplicate originals or select for familiarity or native
   verification. The build from the scratch release put three of the seed's six textbook illustrations of *thay đổi*
   (source `Thú chơi chữ 1990 [UNCERTAIN]`, note "as recalled; NV") into the block, so three of the ten decode to the
   same phrase and are not natural usage. **Implemented** on 1 October 2026 (DEVIATIONS row of that date): at most
   one row per original; rows whose `note` or `source` contains "same source", "illustration" or `[UNCERTAIN]` are
   skipped, and so are validator-flagged rows; after Gate 1, `make baseline` draws only rows with `verified` = True in
   `data/validation/report/attested_verified.tsv`. (Restricting further to `known_any` = True, rows a validator knew,
   is left to the author: it would make the block "familiar" but may leave fewer than ten rows.) The documents describe
   the block as collected nói lái, not as familiar.
6. **Retention of `returned/` and `report/`.** `docs/CONSENT_FORM.md` §5 still carries `[RETENTION — author decides]`.
   Set one date for both roles before the consent form is sent; §4 step 11 and §7 point to it.

## 10. Blocked on the author (to be added to `BLOCKED.md`)

| what | why only the author | when |
|---|---|---|
| recruit about 20 respondents, not validators | personal invitations from the author's own account | by 12 Oct |
| native check of the form's Vietnamese strings | needs a native speaker | before 2 Nov |
| `make baseline` | the v0.3 release files exist only on the author's machine | 2 Nov (after the gate) |
| run `build_forms.gs` in script.google.com | the author's own (or an adult collaborator's) Google account | 2 Nov |
| send one link per person; keep the private key | personal contact | 2–8 Nov |
| download `responses_form_<nn>.csv` into `data/human/responses/` | the author's Google account | after the window |
| delete the 20 Google Forms and empty the Drive trash (§4 step 11) | the author's Google account | after the import |
| set the retention date (open decision 6) | the author's decision | before the consent form is sent (by 12 Oct) |
| decide open decisions 2 and 5 and log them in `docs/DEVIATIONS.md` | the author's decision | before the build (2 Nov) |
