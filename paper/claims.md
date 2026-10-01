# Claim ledger (1 October 2026)

Every sentence in the paper that asserts something must trace to a row here. Evidence is a design-document section
(DD = `docs/DESIGN_DECISIONS.md`), the pre-registration (PREREG = `docs/PREREGISTRATION.md`), a data file with its
SHA-256 (from `configs/run_plan.yaml`; first 16 hex digits shown), a citation verified in
`docs/RELATED_WORK_VERIFICATION.md`, or **pending E#** (no number exists yet; the prose carries a `\placeholder`).
Nothing here has a model result behind it: no model has been run.

Release v0.3 hashes: `noilai_test` 35aed39d14b03de3 · `noilai_main` 9e2165baf004c92a · `noilai_core` 4be46cc642158dac ·
`noilai_dev` f0052e688d43407c · `noilai_c2` 08d6c30ba6aade64 · `attested` 2a2c8ff516ee5510 · XCOPA-vi test 24cb28827066abb0.

## Contributions

| # | Claim | Evidence | Strongest allowed wording | Overclaim to avoid |
|---|---|---|---|---|
| C1 | NóiLái: generator, three tasks, attested subset, validation, human baseline | DD 3–5, 10; release v0.3 hashes above; validation and baseline **pending (Gate 1, 2 Nov)** | "a rule-verified benchmark of nói lái with three tasks over four generated variants, an attested subset and native validation (results pending)" | "the first nói lái benchmark" stated as fact: the novelty sweep (`docs/NOVELTY_SWEEP_2026-09-30.md`) found Pham & Pham 2018 (detection, not LLMs) and was cut short by the search budget → "to our knowledge, the first **LLM** evaluation of nói lái"; "validated" before Gate 1 |
| C2 | Syllable-alignment audit linked to accuracy | DD 5.4, 8.4; PREREG 8.4 (H1); Gemma 3 audit `data/audit/gemma3.json`; accuracy link **pending E2** | "an audit of how N production tokenizers split onset, rime and tone (Gemma 3: mean 1.78 → 2.83 tokens per syllable NFC → NFD, `data/audit/gemma3.json`)" | "alignment causes failure" (E2 is observational, DD 8.4) |
| C3 | Orthographic counterfactuals that change only the token sequence | DD 6.1–6.4, 8.3; PREREG 8.5–8.6; census implemented (DD 6.2) | "within-item interventions that hold the characters' meaning fixed and change the code points" | "tokenizer-caused" without the distribution-shift alternative (H3 row of DD 1); "C2 matters in natural text" (0.5% of XCOPA-vi tokens, DD 6.3) |
| C4 | Mechanistic localization of tone in Gemma 3 1B/4B | DD 9; PREREG 8.9–8.10; **pending E4** | "probes with control tasks and structural baselines, and patching with separate perception and manipulation readouts" | "the model uses tone at layer k" before E4; "generalizes beyond Gemma 3" |
| C5 | Reusable tooling | code in `noilai/vi/`, `noilai/gen/` (tests `tests/test_vi.py`, `test_gen.py`) | "NFC/NFD re-encoding and diacritic stripping apply to any diacritic-heavy Latin script; the parser, speller and placement arms are Vietnamese-specific" | "script-general parser" |

## Claims in the results-independent sections

| # | Section | Claim | Evidence | Allowed wording | Overclaim |
|---|---|---|---|---|---|
| B1 | Background | Vietnamese writes onset, rime and tone in the string; NFC and NFD encode the same text | DD 2; Unicode | stated as fact | "every syllable is one token" |
| B2 | Background | nói lái has a six-type tradition; four generated variants | DD 3.1–3.5; the variant source is a blog + one unconsulted book | "documented in popular sources as six types" | "the standard linguistic taxonomy" |
| B3 | Background | attested examples (e.g. *trò chơi → trời cho*) | `attested` 2a2c8ff516ee5510; seed rows 34, exact 28, **0 native-verified** (`paper/tables/attested_facts.tex`) | cite each example's source; say "attested in folk collections" | "verified examples" before Gate 1 |
| S1 | Benchmark | generator never emits an illegal syllable; every gold re-parsed | DD 2.4, 4.2; `tests/test_gen.py` | "by construction, checked by tests" | "error-free" (validation pending) |
| S2 | Benchmark | main sample 4,200 (350 per cell), core 1,496 | DD 4.5, 12.11; `noilai_main`, `noilai_core` hashes | as stated | — |
| S3 | Benchmark | 350 per cell resolves a 10-point contrast at 80% power | DD 8.5 (`noilai.stats.power`), design effect **pending pilot** | "under the design effect the pilot measures" | unconditional "powered for 10 points" |
| S4 | Benchmark | canary, gated test split, withheld seed, sealed split | DD 4.6, 11.1 | as stated | "uncontaminated" |
| S5 | Benchmark | validation: 30 + 4 controls per cell, 408 rows, α with controls | DD 10.1 amendment (pending the author's ruling, memo N1); `docs/gate1/VALIDATION_PROTOCOL.md`; **results pending Gate 1** | protocol in present tense, results `\placeholder` | any α or precision number |
| S6 | Benchmark | H6 tested only with ≥ 100 verified exact rows | DD 1 (H6), 4.7; current max 79 (`docs/gate1/VALIDATION_PROTOCOL.md` §6) | "tested only if the floor is met; otherwise descriptive" | implying the floor is met |
| D1 | Design | panel of 17 self-hosted + 4 API models; cut order | DD 7.1; PREREG 7; panel freeze **pending 25 Oct** | numbers as `\placeholder` until the freeze | naming models that may be cut as final |
| D2 | Design | arms: NFC base, NFD (C1), partially composed (C1′), placement (C2), strip tones / strip all (C3, contrasts not counterfactuals) | DD 6.1–6.4 | as stated | calling C3 a counterfactual |
| D3 | Design | primary metric strict T1 accuracy; analysis plan, Holm families, base-pair cluster bootstrap | DD 5, 8; PREREG 3, 8 | copied from the DD, not redesigned | "pre-registered" before the stage-2 commit — write "pre-registered before any test-split run" only once stage 2 is committed (PREREG header) |
| D4 | Design | exploratory T1 forced choice | `docs/FORCED_CHOICE_EXPLORATORY.md`; DEVIATIONS row 1 Oct | "exploratory, specified before any model output" | reporting it beside the generated accuracy or in a hypothesis decision |
| R1 | Related work | each cited paper's finding | verified rows of `docs/RELATED_WORK_VERIFICATION.md`; two figures unchecked at page level (Bean et al. 16.0%; Ghosh & Jyothi drops) | unchecked figures stay inside `\placeholder` | quoting an unchecked number |
| L1 | Limitations | free compute bounds the panel; one run per configuration | `docs/COMPUTE_PLAN.md` | as stated | — |
| E1 | Ethics | volunteers unpaid, adults, consent, no IRB | `docs/CONSENT_FORM.md`; DD 11 | protocol in present tense | past tense before anyone has taken part |

## Pending (fills Results, Discussion and the abstract)

E1 accuracies and baselines · E2 coefficients (H1) · E3 effects (H3, H3b, H4) · E4 probes and patching (H5) · H6 · validation α / precision · human baseline · panel freeze counts · compute totals.
