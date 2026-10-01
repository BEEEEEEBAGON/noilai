# Native-validation protocol (Gate 1, 18 October 2026)

Status: **draft for the author's approval** (1 October 2026). It amends DESIGN_DECISIONS 10.1 (sample size, controls) and
10.2 (baseline form format); both amendments are logged in `docs/DEVIATIONS.md` and marked in the design document, and the
author confirms or reverts them in `docs/gate1/DD13_DECISION_MEMO.md`. Nothing here changes the frozen data or the
pre-registration. No model output exists.

Code: `noilai/validation.py`, `scripts/make_validation_forms.py` (`make validation`, `make validation-score`). Sizes:
`noilai/constants.py` (`VALIDATION_*`), chosen with `scripts/validation_sizing.py` (`data/audit/validation_sizing.json`,
RESULTS_LOG RL-2026-10-01-04). Instructions to validators: `docs/VALIDATOR_INSTRUCTIONS.md` (Vietnamese and English).

## 1. What the validation must deliver

1. **Generator precision**: is the rule output (T1 gold, T2 gold reading, T3 verdict) right? Reported per task × variant
   cell and pooled, as a weighted estimate from a probability sample. This is the number the paper cites for "the
   generator is correct", and a confirmed rule bug triggers a fix and a full regeneration before any model run (PREREG
   §5 rule 2).
2. **Agreement**: Krippendorff's α with a bootstrap CI, reported with raw agreement, the label marginals and Gwet's AC1
   (DD 10.1, 12.21; PREREG §8.12).
3. **Offensive screen**: items with a vulgar or offensive reading (any validator's flag) leave every API prompt and the
   human forms (DD 11.5).
4. **Dialect dependence**: items whose validity depends on a regional pronunciation are flagged with the merger named,
   and every validator's region (North / Centre / South / grew up abroad) is recorded for region-stratified reporting.
5. **Attested set**: native verification of every attested row; the H6 floor (≥ 100 exact, verified, two-syllable rows
   with per-row citations from ≥ 3 collections, DD 1 H6 / 4.7) counts only rows verified here.
6. **Core T2 gold sets**: the validated reading sets that are the primary T2 gold for the core (DD 5.2, 12.12).
7. **Convention checks**: the qu- analysis (DD 2.1 O5), which nói lái kind speakers produce (DD 3.3, 3.5), tone-mark
   placement and i/y preferences (DD 2.5, R5), `oao` and `gi` spellings (DD 2.3 C10, O4).

## 2. Who

- **2 or 3 adult native speakers** (target: one Northern, one Central, one Southern), unpaid volunteers recruited by
  personal invitation (`docs/gate1/RECRUITMENT.md`), written consent (`docs/CONSENT_FORM.md`, role "validator").
- Validators are identified by a letter (A, B, C). The author records each letter's region of upbringing and coarse
  demographics in `data/validation/validators.json` (git-ignored), e.g. `{"A": {"region": "N"}, "B": {"region": "S"}}`.
- **Validators never take the human-baseline form** (they have seen test items and the rule tables); the baseline
  form screens for it (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md`).

## 3. The packet (one workbook per validator)

`make validation VALIDATORS="A B C"` writes `data/validation/validation_<V>.xlsx` (dropdowns on every judgment column;
imports into Google Sheets with the dropdowns) plus CSV copies, and the author's keys, which are never sent:
`A_calibration_key.json`, `B_key.json` (row → item id, cell, stratum weight, control flag), `D_engine.json`,
`E_key.json`, `validation_manifest.json`.

| Part | Content | Rows per validator (3 validators / 2) | Judged by | Time (planning) |
|---|---|---|---|---|
| **A** calibration | 16 keyed items computed by the rule engine from pairs that are not in the release: 4 plain correct T1 items (one per kind), 5 planted errors (wrong kind, wrong tone, c/k misspelling, a wrong T2 reading, a wrong T3 verdict), and the cases the instructions single out (new-style placement *hoá*, *y* after a consonant, a vulgar output, a hỏi/ngã output, a T2 item, a correct T3 item, a T3 spelling twin) | 16 / 16 | everyone | ≈ 10 min |
| **B** generated items | the probability sample (§4) + planted controls, opaque row ids `B-0001…` | 292 / 408 | 2 each; 60 rows by all 3 | ≈ 2.8 h / 4.0 h at 35 s |
| **C** attested rows | the 34 seed rows + the web-sourced candidates of `data/attested_candidates.tsv` (85 rows not already in the seed) | 119 / 119 | everyone | ≈ 50 min at 25 s |
| **D** supplementary | D1: 20 pairs "write the nói lái you produce first" (onsets, rimes and tones all differ, so the answer identifies the kind); D2: 20 qu- pairs shown under both analyses; D3: 14 spelling-convention questions | 54 / 54 | everyone | ≈ 30 min |
| **E** core T2 gold sets | each core T2 item's dictionary readings, accept/reject, add a missing reading | 200 / 275 | 1 each; 50 items by everyone | ≈ 1.1 h / 1.5 h at 20 s |

**Total**: ≈ 5.4 h per validator with three validators, ≈ 6.9 h with two (the manifest prints the figure for the actual
packet). **Priority if time runs short**: A → B → C → D → E. Parts A–D are the commitment (≈ 4.3 h / 5.4 h); E is asked
for but optional: an unvalidated core T2 item keeps its dictionary gold with `gold_validated = false` and is reported as
such (DD 5.2 already allows this for non-core items; for the core it is a deviation the paper states).

Columns of Part B (Part A is identical, Part C has its own):

| column | values | meaning |
|---|---|---|
| `correct` | Có / Không / Không chắc | T1: the candidate is the named kind's output of the input; T2: the input is a nói lái (any kind, either order) of the candidate original; T3: the system's yes/no verdict is right |
| `spelling` | Có / Không / Không chắc | the candidate is spelled correctly (both tone-mark conventions and i/y after a consonant are correct) |
| `lexical_input`, `lexical_candidate` | Có / Không / Không chắc | is the phrase a real Vietnamese word or phrase (the speaker's judgment, not dictionary membership) |
| `offensive` | Có / Không / Không chắc | the input or the candidate has a vulgar or offensive reading, including one reached by decoding |
| `dialect` | none, d=gi, d/gi=r, ch=tr, s=x, v=d/gi, final n=ng, final t=c, hỏi=ngã, other | does your `correct` answer depend on how your region pronounces something; if so, which merger (the DD 2.1 O3 table) |
| `comment` | free text | `RULE?` + a description when several items fail the same way |

## 4. The Part B sample

- **Population**: every item of the release (dev + test), excluding items whose phrase equals a calibration phrase.
- **Strata**: task × variant cell (12) × lexical/pseudo source; within a cell the 30 items are allocated to the two
  sources in proportion to their population (largest remainder), drawn at random; the manifest records N_h, n_h and
  the weight N_h / n_h of every stratum. T3 items are drawn singly, never both members of a twin pair.
- **Controls**: 4 per cell, built from items outside the sample: a T1 candidate replaced by another kind's output (never
  the gold in the other order, which is lenient-correct) or by the gold with one tone changed; a T2 candidate replaced by
  a phrase that is not a reading under any of the six kinds in either order (checked by the engine); a T3 item shown
  with the opposite system verdict. Expected answer `correct` = Không. Controls are excluded from generator precision.
  They serve two purposes: they put true negatives into the sheet, which α needs at ≈ 97% prevalence (§6), and they show
  whether a validator catches errors rather than agreeing with everything (the per-validator catch rate is reported).
  Validators are told that some rows are deliberately wrong.
- **Assignment**: rows shuffled; 60 rows to every validator, the rest to two validators in rotating pairs (A–B, B–C,
  C–A). With two validators every row goes to both.
- **Seed**: 20261018 (public sampling seed; selects from the built release and regenerates nothing; DD 4.6).

Why 30 + 4 per cell (simulation, `data/audit/validation_sizing.json`, assumptions stated there; expected scenario:
2% generator error, validator false alarm 2%, miss 10%):

| design | rows per validator (3 val.) | hours | α on `correct` (mean CI width) | AC1 (width) | P(detect a 10% bug in a cell) |
|---|---|---|---|---|---|
| 1,000 items, no controls (DD 10.1 before the amendment; 60-row triple overlap) | 692 | 6.7 | 0.41 (0.29) | 0.95 (0.03) | 1.00 |
| 30 + 0 controls per cell | 260 | 2.5 | 0.40 (0.45) | 0.95 (0.05) | 0.96 |
| **30 + 4 controls per cell (chosen)** | **292** | **2.8** | **0.76 (0.18)** | **0.93 (0.06)** | **0.96** |
| 40 + 4 controls per cell | 372 | 3.6 | 0.74 (0.18) | 0.93 (0.05) | 0.98 |

Per cell, 30 items judged correct give a Wilson 95% lower bound of 0.886; pooled over 360 items at 99% observed, the
interval is [0.972, 0.996]. Cell-level precision is therefore an appendix table; the paper's sentence rests on the pooled
estimate and on the absence of any confirmed rule bug.

## 5. Procedure and timeline

| date | step | who |
|---|---|---|
| by 12 Oct | recruit 2–3 validators and ~20 baseline respondents (different people); send consent forms | author |
| by 12 Oct | `make validation`; upload the workbooks to Google Sheets; share each with one validator | author (BLOCKED.md) |
| 13–15 Oct | **Part A, calibration** (≈ 10 min); the author scores it (`make validation-score`, `calibration` block), sends each validator the key with the explanations of the rows they missed | validators, author |
| 15–17 Oct | calibration debrief: if a validator matches the key on fewer than 80% of `correct` answers (`VALIDATION_CALIBRATION_PASS`), a short call or message and the second calibration set (the same specs, next inputs); nobody is excluded on calibration; any change to the instructions is versioned and logged here | author |
| **18 Oct** | **Gate 1**: packet sent, calibration done | |
| 19 Oct – 1 Nov | Parts B, C, D (and E); validators return what they have each week | validators |
| 2–5 Nov | score; adjudication (§6); third-validator sheet (`make_validation_forms.py adjudication-sheet --to C`); author decisions logged | author, third validator |
| by 8 Nov | confirmed rule bugs fixed and the data regenerated (PREREG §5 rule 2) — before any test-split run; attested rows merged only by the author's decision (`docs/DEVIATIONS.md`) | author |

No AI assistant and no discussion between validators before submission; a dictionary is allowed for the lexicality
question only.

## 6. Agreement, adjudication, outputs

**Agreement** (`make validation-score` → `data/validation/report/validation_report.json`): for each judgment, nominal α
on yes/no with "Không chắc" as missing (primary) and as a third category (sensitivity), Gwet's AC1, raw pairwise
agreement and the label marginals, each with a 1,000-replicate item bootstrap CI. Primary agreement is computed over all
Part B rows (controls included); the sample-only figures are reported beside it. Part C: the same for `valid`, `known`,
`spelling_ok`, `offensive`. Part E: MASI- and Jaccard-distance α over the overlap items (DD 10.1).

**Adjudication rule** (published with the data; `noilai.validation.resolve`):

1. All definite labels agree (≥ 2 of them) → that label (`unanimous`).
2. Two validators disagree → the third validator, who has not seen the row, judges it blind (`adjudication-sheet`);
   a strict majority of three → that label (`majority`).
3. Still split, or only two validators → the author decides against the rule tables (DD 2–3) and writes the decision
   and the reason into `data/validation/adjudication.tsv` (`author`); the count of author decisions is reported.
4. Fewer than two definite labels ("Không chắc") → unresolved; the item stays in the benchmark and out of the
   precision estimate (PREREG §5 rule 2); the count is reported.
5. The author never overrides validators who agree.

**Low control catch rate** (pre-specified): a validator who labels fewer than 75% of the planted controls Không (`VALIDATION_CONTROL_CATCH_MIN`) is re-briefed after the first weekly return and the rate is reported in the paper; no validator's labels are excluded or down-weighted on it.

**Offensive**: one validator's "Có" flags the item (union rule; conservative); flagged items join the blocklist screen
(DD 11.5). **Dialect**: the union of the mergers named. **`RULE?` comments**: every one is checked by the author against
the rule tables; a confirmed bug → fix + full regeneration before any model run.

**Generator precision** (`stratified_precision`): per cell p = Σ_h W_h p_h with W_h the stratum's population share; the
pooled estimate weights cells by population; unweighted Wilson intervals beside each estimate; unresolved rows counted.

**Attested rows** (`attested_verified.tsv`): a row is **native-verified** when at least two validators answer `valid` =
Có and none answers Không. A verified row enters `data/attested_seed.tsv` only by the author's decision, with its
citation checked on the live page (`source_checked` in `data/attested_candidates.tsv`), its `verified_by` set to the
validator letters, and a `docs/DEVIATIONS.md` row (DD 4.1: a row added after the build never triggers a regeneration;
overlapping generated items get `attested_overlap`).

**H6 floor arithmetic, stated plainly**: the seed has 24 H6-eligible rows (exact, two-syllable, 0 verified); the
candidates add at most 55 exact two-syllable rows from 24 collections. **Even if every one is verified, the total is 79,
below the floor of 100.** Clearing it needs ≈ 21 more exact two-syllable rows with citations, more realistically 40 given
rejections; see BLOCKED.md and the decision memo (the fallback is the pre-registered one: H6 descriptive).

## 7. Reporting (paper and data statement)

Committed wording for agreement (so that it is not chosen after the numbers exist): "α on `correct` is computed over the Part B sheet including its N planted control rows; the probability-sample-only α and AC1 are reported beside it."

Number of validators and their regions; hours; α / AC1 / raw agreement / marginals per judgment with CIs; per-validator
control catch rate; adjudication counts (unanimous / majority / author / unresolved); generator precision pooled (main
text) and per cell (appendix); offensive-flag rate with α; dialect-flag counts by merger and by validator region; the
D1 production table by region; the qu- and convention results; attested rows verified, by collection.
