# Claim ledger — *Tones Hidden in Tokens* (ARR draft)

Every sentence of the manuscript that asserts something must trace to a row here. Columns: the claim; its
evidence (DD = `docs/DESIGN_DECISIONS.md` section, PR = `docs/PREREGISTRATION.md` section, a data file with
its SHA-256, a citation verified against the ACL Anthology XML, or **pending E#** for an experiment not run);
the strongest wording that evidence allows; the wording that would overclaim. Status on 1 October 2026: **no
model has been run**, nothing is registered (both PR commit fields read `<fill>`), and no attested row is
native-verified. Anything marked pending stays inside `\placeholder{}` in the `.tex` sources.

Hashes quoted below: release v0.3 content `92d332e58b0e4d5d3dbf637900cd574dd2b6aac83ffecc2865c7a274ea208e11`
(`data/release/v0.3/manifest.json`, generator commit `f94f655`, clean tree); main sample `noilai_main.jsonl`
`9e2165baf004c92aa0cf247cdeddd40ea7fe618144228d8a575d2f2dde9c9d53`; C2 file `noilai_c2.jsonl`
`08d6c30ba6aade640019351192a8b933a5e07fcdf70072ab4868a1f9274980a0`; Gemma 3 tokenizer
`1299c11d…2e79c`, `vi-DauMoi.dic` `a075773a…3143`, `vi-DauCu.dic` `50b23ff2…88`, `Viet74K.txt`
`b83d4930…fb90`, XCOPA-vi test `24cb2882…4170` (`data/HASHES.json`).

## A. Contributions (Introduction list)

| # | Claim | Evidence | Strongest allowed wording | Overclaim (forbidden) |
|---|---|---|---|---|
| C1 | NóiLái: a rule-verified generator of T1/T2/T3 items over four generated variants of a six-type tradition, plus an attested subset, validation and a human baseline | DD 3–5, 10; manifest v0.3 (10,000 items, 2,500 base pairs); validation and baseline **pending (protocol only)**; attested seed 34 rows, 0 verified (RL-2026-10-01-01) | "a benchmark … with a native-validation protocol and a human baseline" (protocol); counts from the manifest | "validated by native speakers" (nothing validated yet); "the first nói lái dataset" (Pham & Pham 2018 built a spoonerism detector; their data status unknown) |
| C2 | First LLM study of nói lái; first generative/decoding/validity benchmark of it | Novelty sweep `docs/NOVELTY_SWEEP_2026-09-30.md`; Pham & Pham 2018 (Y18-1063, title verified: detection system) | "to our knowledge the first LLM study of nói lái and the first generation, decoding and validity benchmark of it" | "the first computational study of nói lái"; "no prior work on Vietnamese spoonerism" |
| C3 | Alignment audit links tokenizer splits to item accuracy across the panel | Audit implemented (`data/audit/gemma3.json`); link to accuracy **pending E1/E2** | "an audit … that we relate to item-level accuracy" | "alignment predicts accuracy" (H1 untested) |
| C4 | Orthographic counterfactuals change only the token sequence on deployed models, with a three-valued census | DD 6.1–6.2; census for Gemma 2/3 only (RL-2026-10-01-02); effects **pending E3** | "interventions that hold meaning and characters' information fixed"; "separating tokenizer-caused from knowledge-caused errors" only as the design goal | "we show that tokenization causes …" ; "the effect is mediated by token count" (DD 8.6 forbids mediation) |
| C5 | Mechanistic localization of tone in Gemma 3 1B/4B | DD 9; **pending E4** | "a protocol for localizing …; results in Section 7" | "tone is represented at layer …"; "steering" (moved to future work, DD 9.4) |
| C6 | Reusable tooling; NFC/NFD re-encoding and stripping apply to any diacritic-heavy Latin script; parser/speller/placement are Vietnamese-only | `noilai/vi/unicode.py`, `reencode.py`; DD 14 item 35 | as stated, scoped as in the test `test_contribution_five_claims_script_generality_for_the_re_encoder_only` | "the toolkit applies to any language" |

## B. Background and benchmark facts (stated as facts; every one has a file or a citation)

| # | Claim | Evidence | Allowed wording | Overclaim |
|---|---|---|---|---|
| B1 | Gemma 3 splits 70% of the 6,595 parsable syllables under NFC; 51.5% at onset\|rime; 1.78 → 2.83 tokens/syllable NFC→NFD; tone mark isolated in 41.6% | `data/audit/gemma3.json` summary (`single_token_frac` 0.2946, `onset_rime_split_frac` 0.5152, 1.7826/2.8308, `tone_isolated_frac` 0.4164) | exact figures, one tokenizer named | "subword tokenizers split Vietnamese at random" (one tokenizer measured) |
| B2 | Alignment among split syllables 0.92 NFC / 0.21 NFD (0.94 / 0.27 over all syllables) | `gemma3.json` `boundary_alignment_among_split_mean`; RL-2026-10-01-03 | as stated, labelled "among split" vs "all" | quoting 0.94 as "among split" |
| B3 | Gemma 3 passes NFD and PC through: identical ids for 1.3% of the 1,000 census strings under NFD, NFD longer for 91.0% | `gemma3.json` `normalization_census` (`nfd_same_ids_frac` 0.013, `nfd_longer_frac` 0.91); RL-2026-10-01-02 | "passes through; 98.7% of the census strings receive different ids" | **"96.6% of NFD words with different ids"** (in the earlier Setup draft; not in any file — corrected) |
| B4 | 69 syllable types differ between the two placement conventions; old style is the majority in XCOPA-vi (60 of 62 differing tokens) | `tests/test_vi.py`; `data/audit/placement_xcopa.json`; DD 2.5, 6.3 | "69 of the 6,611 syllables of the standard list"; "in XCOPA-vi 60 of 62 tokens"; majority in running text **pending corpus count** | "the older convention is the majority in Vietnamese text" without the count |
| B5 | Six-type nói lái tradition; V1–V6 as subsets of {O,R,T}; group (ℤ/2)³; reverse pairs | DD 3.1–3.2 (engine-verified); six types from plo.vn/tuoitre and Lê & Hồ 1990 **[book not read; UNCERTAIN]** | "documented in popular and press sources" with the book cited as unverified | "the canonical six-type taxonomy of Vietnamese linguistics" |
| B6 | Regional labels of variants are inconsistent across sources | DD 3.5 (snippet-level sources) | "the sources we found attach regional labels inconsistently" | "Northern style = V3" |
| B7 | Release v0.3: 10,000 items (T1 4,000 / T2 2,000 / T3 4,000) from 2,500 base pairs (1,500 lexical); dev 2,000 / test 8,000; core 1,496; 47,535 lexical candidates; 143 vulgar-flagged; 20 marginal rimes excluded | manifest v0.3, content hash above; RL-2026-10-01-01 | numbers through `tables/table1_counts.tex` (generated) or quoted with the manifest | typing different numbers; calling v0.3 "final" before the stage-1 commit is filled |
| B8 | Main sample 4,200 (350/cell); C2 file 500 items, all affected, disjoint from the release | manifest `samples` (hashes above) | as stated | "the main sample is representative of Vietnamese" |
| B9 | Attested seed: 34 rows, 28 exact, 6 approximate, 24 H6-eligible, 0 native-verified | RL-2026-10-01-01; `tables/attested_facts.tex` (generated macros) | via the macros only; target 200–400 as a target | "200–400 attested examples" as a fact |
| B10 | Build seed withheld; only the canary GUID's SHA-256 is printed | DD 4.6; manifest `canary_sha256` `dedff579…701c` | as stated | printing the GUID. **Note:** `docs/RESULTS_LOG.md` RL-2026-10-01-01 quotes a different canary SHA-256 (`07fb9d79…`) from the committed manifest — reconcile outside `paper/` before release |
| B11 | Validation protocol: 1,000-item probability sample, 2 validators + 200-item overlap by 3; α with raw agreement and AC1 | DD 10.1; results **pending** | protocol in the present tense ("is judged") | "validators agreed", any α |
| B12 | Human baseline: 20 × 30, 246 items, connected double-coverage design | DD 10.2; PR 8.11; `tests/test_paper.py` assignment test | protocol | "humans score …" |

## C. Experimental design (from DD/PR, never redesigned)

| # | Claim | Evidence | Allowed wording | Overclaim |
|---|---|---|---|---|
| D1 | Panel 17 self-hosted + 4 API, frozen after smoke tests | DD 7.1–7.2; PR 7; freeze **pending (25 Oct)** | "planned panel", numbers in placeholders until the freeze | "we evaluate 21 models" |
| D2 | Arms: base, nfd, pc, placement_new, strip_tones, strip_all; meaning-preserving arms whole-prompt primary, item scope secondary on 500 Gemma 3 1B items | DD 6.1; PR 8.5; `noilai.eval.prompts` default | as stated | "item-only by default" |
| D3 | τ_m(a) within-item ATE is identified without assumptions beyond representativeness of the item sample | DD 6.4; PR 8.5 | as stated | "identifies the causal effect of tokenization on language understanding" |
| D4 | No mediated share is estimated; Δk analyses are associational | DD 8.6; PR 8.7 | as stated | any "mediated by" |
| D5 | Base-pair cluster bootstrap (B = 2,000) and paired t primary; McNemar secondary, labelled; Holm families incl. 14-cell Table 3 row | DD 8.2–8.3; PR 8.2–8.3 | as stated | "two-way clustered SEs" |
| D6 | H1 decided by nested LR/ΔAIC; 300-misaligned-split floor; ≥ 4 families | DD 1; PR 2 | as stated | "which matters more" contrast |
| D7 | H2 descriptive; H3 per-family primary with OOD default; H3b DiD; H4 leads E3; H5 two-condition decodability + readout B with < 50-pair fallback; H6 floor ≥ 100 verified exact rows | DD 1; PR 2 | as stated | H4 direction as a fact (set by a corpus count, pending) |
| D8 | Pre-registered before any test-split run, in two stages | PR header; DD 8.8; **commits not yet made** | "pre-registered in two dated commits before any test-split run (Appendix …)" with hashes as placeholders | "pre-registered before any model was run"; "were pre-registered" in the past tense |
| D9 | Power: 1,000 iid at 70% ±2.84; 350/cell for a 10-point contrast at 80%; API cells resolve 11–17 points | PR 9; `noilai.stats.power` (test-recomputed) | as stated | "adequately powered" without the discordance caveat |

## D. Related-work contrasts (each cited paper verified against Anthology XML; only the abstract was read)

| # | Claim | Evidence | Allowed wording | Overclaim |
|---|---|---|---|---|
| R1 | Subword embeddings encode character composition | 2022.naacl-main.179, 2022.naacl-main.373 (abstracts) | "encode character-level information / spell part of the vocabulary" | "fully" |
| R2 | CUTE: models know their tokens' spelling but fail to manipulate; EXECUTE extends to more languages and scripts | 2024.emnlp-main.177, 2025.findings-acl.95 (abstracts) | as stated | **"in languages that do not include Vietnamese"** (not in the abstract; page-level check pending) |
| R3 | Spelling-out: intermediate/higher layers reconstruct character knowledge | 2025.findings-emnlp.719 | as stated | — |
| R4 | STAD measures misalignment of tokenization with syllable boundaries; misalignment correlates with weaker phonological encoding | 2026.acl-long.634 | "our measure adapts it to boundaries inside a syllable" | "STAD is defined on English only" (language coverage not in the abstract) |
| R5 | SubTokenTest; Chai et al.: practical sub-token tasks, typo sensitivity | 2026.acl-long.915, 2024.findings-emnlp.86 | as in abstracts | — |
| R6 | Tokenizer quality/cost across languages; tokenization sensitive to variation | 2021.acl-long.243, 2023.emnlp-main.614, 2023.findings-acl.350, 2025.findings-acl.572 | as in abstracts | — |
| R7 | Causal tokenization bias via regression discontinuity on the vocabulary cutoff | 2025.acl-long.1374 | as stated | **"by training families of controlled models"** (earlier draft; the abstract describes an RD design) |
| R8 | Phonological benchmarks: PhonologyBench (English G2P, syllable counting, rhyme); Phun-Bench (Chinese; models recall pronunciations but struggle to use them) | 2024.knowllm-1.1, 2026.acl-long.1041 | "struggle to leverage" | **"cannot use them"** (earlier draft) |
| R9 | Puns/wordplay: SemEval-2017 Task 7; LLM pun understanding; robustness; Russian headline wordplay | S17-2005, 2024.emnlp-main.657, 2025.emnlp-main.1419, 2025.ranlp-1.15 | as in abstracts | **"spoonerisms appear in KoWit-24"** (not in the abstract) |
| R10 | Vietnamese NLP: PhoBERT, ViT5, VnCoreNLP, PhoGPT, a broad Vietnamese LLM evaluation, VMLU, dialect resources | 2020.findings-emnlp.92, 2022.naacl-srw.18, N18-5012, 2311.02945, 2024.findings-naacl.182, 2025.acl-long.563, 2023.findings-emnlp.925, 2024.emnlp-main.426 | as in abstracts | **"diacritic removal appears as one of several perturbations"** in 2024.findings-naacl.182 (not in the abstract; kept only inside a placeholder) |
| R11 | Pham & Pham 2018 built a spoonerism detection system for Vietnamese | Y18-1063 (title only; no abstract in the XML) | "a spoonerism detection system for Vietnamese" | **"with rules and an n-gram model, on classical text"** (unchecked) |
| R12 | Benchmark practice: preregistration in NLP; dynamic benchmarking; reporting; minimal pairs; contamination | 2021.naacl-main.51, 2021.naacl-main.324, D19-1224, 2020.tacl-1.25, 2023.emnlp-main.308 | as in abstracts | "the first pre-registered NLP benchmark" (not checked) |
| R13 | Probing: probing classifiers, control tasks, presence ≠ use | P18-1198, D19-1275, 2022.cl-1.7 | as stated | — |
| R14 | Unverified, kept only in the PLACEHOLDER bib and marked: PACUTE, TokSuite, Ghosh & Jyothi (numbers only inside `\placeholder`), UGTPhon, 2609.21362 (authors missing), 2606.15044 | `docs/RELATED_WORK_VERIFICATION.md` §§3–4 | cited with `\placeholder{verify}` beside the claim | any number from them outside a placeholder |

## E. Pending — Results, Discussion, abstract numbers (left for when numbers land)

E1 accuracy table (H6, human comparison) · E2 regression (H1, H2) · E3 arms (H3, H3b, H4) · E4 probes/patching
(H5) · validation α and generator precision · census of the full panel · compute totals · the corpus
placement count (fixes H4's direction) · the two pre-registration commit hashes.
