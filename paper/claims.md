# Claim ledger — *Tones Hidden in Tokens* (ACL 2027, ARR January 2027)

Written 1 October 2026 for the paper workstream. Every contribution and claim the manuscript makes
or will make is a row below, with the evidence that exists today, the strongest wording that
evidence allows, and the wording that would overclaim. The paper is drafted from this ledger:
a sentence that is not covered by a row is not written, and a row whose evidence column says
**pending** appears in the paper only as a `\placeholder{...}` or as protocol in the present tense.

Sources of authority, in order: `docs/DESIGN_DECISIONS.md` (DD, binding), `docs/PREREGISTRATION.md`
(PREREG, a draft until its two commits exist), the committed data files named by hash below,
`docs/RESULTS_LOG.md` (RL entries), `docs/RELATED_WORK_VERIFICATION.md` (RWV),
`docs/NOVELTY_SWEEP_2026-09-30.md` (NS), `docs/DEVIATIONS.md`. Where the DD and an earlier document
disagree, the DD wins (DD §0).

## 0. How to read this ledger

**Status codes.**

| code | meaning |
|---|---|
| **DATA** | computed from a committed file named by path and SHA-256; reproduced by `tests/test_paper.py` or a generator under `paper/` |
| **DD §x** | a design decision; a claim about what the study *does*, written in the present tense as protocol |
| **CITE-V** | a citation whose identifier was verified against the ACL Anthology XML (`acl-org/acl-anthology`, `data/xml/`, fetched 1 October 2026) or, for the entries the previous round verified, per RWV §1; title, authors, venue, pages and DOI come from the XML |
| **CITE-U** | a citation that exists only in `paper/references_PLACEHOLDER.bib`, marked `UNVERIFIED`; cited in the draft, must be checked by hand before submission |
| **NUM-U** | a number attributed to a source that was not checked at page level; appears only inside `\placeholder{}` (DD §14 item 45) |
| **pending E#** | depends on an experiment or procedure that has not run; see the registry below |

**Pending-work registry** (the `E#` labels the evidence column uses).

| label | what | when (plan) | paper sections it unlocks |
|---|---|---|---|
| P | Gate-1 pilot: 3 open models × 200 dev T1 items (NFC and NFD) + 200 XCOPA-vi items; fixes ICC, discordance, δ_sel, n per cell; go/no-go in PREREG §10 | 18 Oct 2026 | the power sentences' "pilot" placeholders; no table |
| F | panel freeze: smoke tests, checkpoint ids and revisions, tokenizer SHA-256s, census verdict per engine | 25 Oct 2026 | Table of models (now "as planned"), tokenizer-profile table, census column |
| CC | reference-corpus choice; placement count on two corpora (fixes H4's direction); frequency covariate | before Gate 1 | Background sentence on the majority convention; H4 direction |
| PR1 | stage-1 pre-registration commit (generator, data v0.3, scoring, extraction, census, constants) | at the v0.3 freeze; `PREREG_COMMIT_STAGE1 = <fill>` today | Appendix "Pre-registration": commit hash |
| PR2 | stage-2 pre-registration commit (hypotheses, analysis plan, E4 protocol, pilot-informed constants) | ≤ 8 Nov 2026, before any test-split run | Appendix: commit hash and date |
| V | native validation: 3 validators, 1,000-item probability sample, T2 gold sets, attested rows, qu- and variant-production checks | 19 Oct–8 Nov 2026 | Table 1 α row; §3 validation results; generator precision; attested `verified_by` |
| L | licence decision: permission from the list authors or the GPLv2 fallback | by Gate 2, 8 Nov 2026 | §3 Release; Ethics "Data and terms of use" |
| E1 | panel accuracy on T1/T2/T3 (main sample; core for APIs) | 9–22 Nov 2026 | §5 Results, Table 2, Fig. 3; abstract numbers |
| E2 | what predicts failure: tokenization covariates, nested LR test (H1); PhoGPT case study (H2) | with E1 | §5, Fig. 2 |
| E3 | re-encoding arms on NóiLái and XCOPA; census; H3, H3b, H4 | 9–22 Nov 2026 (test split) | §6, Table 3, Fig. 4 |
| E4 | probes and patching in Gemma 3 1B/4B; H5 | 23 Nov–6 Dec 2026, reported regardless of outcome | §7, Fig. 5 |
| HB | human baseline: 20 respondents × 30 items | with V | Table 2 human row; §3 |
| H6 | attested vs matched generated; contamination discriminators (floor: ≥ 100 verified exact rows) | with E1 | §5 "memorization" |
| R | hand check of every reference from an unblocked machine (arxiv.org and aclanthology.org are egress-blocked here) | before 18 Dec 2026 | bibliography; the CITE-U rows |
| NV | native-speaker check of every Vietnamese string in the paper and prompts | with V | every Vietnamese example |

**Evidence files and their SHA-256 (computed 1 October 2026 on the working tree).**

| file | SHA-256 | role |
|---|---|---|
| `data/release/v0.3/manifest.json` | `dfa9ce005ad8ce569148b297e139538cacbf162abfee91489a3acaa497fd0d84` | public release manifest (no seed, no GUID) |
| `data/audit/gemma3.json` | `6007e7c793bf7a2432cd4eb2359b37a48e16ccbf6747cbae437c6173a3030e2f` | Gemma 3 tokenizer audit + census |
| `data/audit/gemma2.json` | `f26dae169e730d0bbaf4ce330b2b59275b7de0eed9d334aab2c85dd87434b600` | Gemma 2 tokenizer audit + census |
| `data/audit/counts.json` | `5da27bf626be93366db261620b5346b35de3282d39dcd53a9d0d407979bef730` | reconciled counts (`scripts/reconcile_counts.py --release data/release/v0.3`) |
| `data/audit/placement_xcopa.json` | `10c491cc4f9107d8bb37d6ae99e5c2a3cd60d0419fc9c9864093bceeba318bb4` | placement count, XCOPA-vi test + val |
| `data/audit/placement_xcopa_test.json` | `011d2cf33801377bb8c480cf774925d7aecd8cd9d6743c438e29dc77cccd1ee0` | placement count, XCOPA-vi test only |
| `data/attested_seed.tsv` | `0e62d4c5c1f77103f0c72d8ea0cc9d030d2906d59eda55b736b2f148776ec2a4` | attested seed (34 rows); also `resource_sha256.attested_seed.tsv` in the manifest |
| `data/vulgar_lexicon.tsv` | `d81eafbb6bd3662129fdcd80c7049d3478cbb410ada81e87d4b6fa636c345ab2` | blocklist |
| `paper/figures/fig1_tokens.json` | `bc56926c3d54f68f001c17c37fd6b8d368c4569b012cb321e8b1103065ed8ebd` | Figure 1 token boundaries (Gemma 3) |
| `data/external/gemma3_tokenizer.model` | `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c` | `data/HASHES.json` |
| `data/external/gemma2_tokenizer.model` | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` | `data/HASHES.json` |
| `data/external/vi-DauMoi.dic` / `vi-DauCu.dic` | `a075773a27729713a1f5853404c7aafbb78e9056a12ccd729c19622953723143` / `50b23ff20ea83ad886e2ff166220d53b307784c9870fb447341e0f298f0e5588` | Hunspell vi_VN, new / old placement |
| `data/external/Viet74K.txt` | `b83d49308336f12ca46d06287d8db6e2cdc7a725d9a9dac56930051775fafb90` | word list |
| `data/external/xcopa_test_vi.jsonl` / `xcopa_val_vi.jsonl` | `24cb28827066abb00a2f4004d86c447c4a30a9edb17938b26b61fb2385f94170` / `6f804ec3c25b12a3d51f58e338b631bc1b45e7223b15481f8593923e0bdb26fa` | XCOPA-vi |

Release identity quoted in the paper (content SHA-256, canary digest, item-file digests, counts) is
emitted by `paper/gen_release_facts.py` into `paper/tables/release_facts.tex`, `release_hashes.tex`
and `release_facts.json` from the public manifest and `configs/run_plan.yaml`; the prose quotes the
macros and never types the values. The generator commit in the manifest (`f94f655`) is the ORIGINAL
repository's hash; its hash in this repository is `62f63ae` (`docs/MIGRATION.md`). The paper prints
the manifest's value, labelled "as recorded in its manifest", and leaves the pre-registration
commits as placeholders until PR1/PR2 exist.

---

## 1. Contributions

| id | claim | evidence | strongest wording allowed | overclaim (never write) | where |
|---|---|---|---|---|---|
| C1 | NóiLái: a rule-generated, rule-verified benchmark of nói lái with three tasks (transformation, decoding, validity) over the four generated variants of a six-type tradition, an attested subset, native validation and a human baseline | DATA: release v0.3 (manifest; `\releaseNItems{}` = 10,000 items from 2,500 base pairs; `\nAttested{}` = 34 attested rows); DD §3–5, §10; V and HB **pending** | "a benchmark generated and scored by a rule engine, with a small attested subset; native validation and a human baseline follow the protocol of §3.5" | "native-validated" (0 of 34 attested rows verified; the 1,000-item validation has not run); "200–400 attested examples" as achieved (it is the target; 34 rows exist); "human-validated gold" | §1 contribution 1, §3 |
| C1a | first LLM study of nói lái and first generative/decoding/validity benchmark of it | NS §3(a): 10 EN/VI queries found no LLM study; prior computational work = Pham & Pham 2018 (detection, rules + 3-gram LM; CITE-V `Y18-1063`); KoWit-24 lists spoonerism among wordplay types in Russian (CITE-V `2025-ranlp-1-15`) | "to our knowledge the first LLM study of nói lái and the first generative, decoding and validity benchmark of it; the one prior computational treatment detects attested nói lái in classical text" | "the first computational treatment of nói lái"; "nói lái is unstudied in NLP"; "the first spoonerism benchmark" (KoWit-24 covers spoonerism as a type) | §1, §2, §8 |
| C2 | a syllable-alignment audit of production tokenizers linked to item-level accuracy | DATA for the audit of two tokenizers (Gemma 3, Gemma 2; `gemma3.json`, `gemma2.json`, `counts.json`); the link to accuracy is **pending E1/E2**; the panel-wide audit is **pending F** | "an audit that measures, for every panel tokenizer, tokens per syllable, the split rate and the alignment of token boundaries with onset–rime, glide–nucleus and nucleus–coda seams; two tokenizers are audited today" | "accuracy tracks alignment" (H1, untested); "across 21 tokenizers" (two audited) | §1 contribution 2, §4 |
| C3 | orthographic counterfactuals: meaning- and information-preserving re-encodings (NFC→NFD, NFC→partially composed, old→new placement) on deployed models, with a census of which tokenizers pass through, normalize or corrupt | DD §6; census implemented (`noilai/audit/tokenizers.py`; Gemma 3 and Gemma 2 verdicts in `gemma3.json`/`gemma2.json`); effects **pending E3** | "re-encodings that change only the code points of the prompt, hence the token sequence, with meaning and information fixed; the within-item contrast is identified without assumptions beyond representativeness (DD §6.4); which tokenizers normalize, pass through or corrupt is a result of the census" | "causal effect of tokenization on model competence" in general; "mediation through token count" (DD §12.7: not estimated); "the first study of Unicode normalization in LLMs" (TokSuite, Ghosh & Jyothi, the Arabic-diacritics paper are near neighbours) | §1 contribution 3, §6 |
| C3a | first comparison of canonically equivalent encodings, and of tone-placement conventions, of a tonal Latin script on deployed LLMs | NS §3(b): no paper found; near neighbours TokSuite (CITE-U, arXiv 2512.20757), Ghosh & Jyothi 2026 (CITE-U, 2607.26831), Inoue et al. 2026 (CITE-V `2026-findings-eacl-22`), Gambardella et al. 2025 (CITE-V `2025-acl-short-75`), Gorman & Pinter 2025 (CITE-V `2025-naacl-short-25`) | "no study we know of compares canonically equivalent encodings, or tone-placement conventions, of a tonal Latin script on deployed models" with the neighbours cited in the same paragraph | "first Unicode study"; "first to show tokenizers are not normalization-invariant" | §2/§8 |
| C4 | mechanistic localization of tone in Gemma 3 1B/4B with control-validated probes, structural baselines and activation patching with two readouts | DD §9 protocol; code under `noilai/probe/`; results **pending E4** | protocol in the present tense; "E4 is reported regardless of outcome (DD §8.8)" | any claim about where tone lives; "decodable earlier under NFD" (DD §9.1 deletes it) | §1 contribution 4, §7 |
| C5 | reusable tooling: NFC/NFD/PC re-encoder and diacritic strippers (script-general), a Vietnamese syllable parser, speller and generator (Vietnamese-specific) | code exists (`noilai/vi/`, `noilai/gen/`); release terms **pending L** | "the re-encoding and stripping tools apply to any diacritic-heavy Latin script; the parser, speller and placement arms encode Vietnamese phonotactics and spelling and are not script-general" (DD §14 item 35; `tests/test_paper.py` checks this split) | "a toolkit for diacritic-heavy scripts" without the split | §1 contribution 5, Limitations |

## 2. Novelty and positioning (what the gap sentence may say)

| id | claim | evidence | strongest wording allowed | overclaim | where |
|---|---|---|---|---|---|
| N1 | Vietnamese writes onset, rime and tone in the string; a failure to manipulate them is attributable to access, not to missing knowledge, unlike Chinese or Filipino where phonology is latent behind the script | linguistic fact (DD §2); contrast with Phun-Bench (CITE-V `2026-acl-long-1041`: "LLMs excel at recalling correct pronunciations, they generally struggle to leverage phonological knowledge") and PACUTE (CITE-U) | "Vietnamese spells the units in the string, so the question is access to spelled units, not recall of unspelled ones" | "Vietnamese isolates tokenization from knowledge" (spelling rules are knowledge the model needs) | §1, §8 |
| N2 | EXECUTE does not include Vietnamese | RWV: snippets list 8 primary languages + Tamazight, Santali; PDF not read (NUM-U for a language count) | "in a set of languages that does not include Vietnamese" | "ten languages" | §8 |
| N3 | STAD is defined over English syllabification; our boundary-alignment metric adapts it to sub-syllabic boundaries | RWV mismatch 6 [UNCERTAIN: appendix of 2026.acl-long.634 not read] | "adapts" / "defined over English syllabification" (hedged as "as far as we can tell") | "we apply STAD" | §4, §8 |
| N4 | Pham & Pham 2018: rules + 3-gram LM detect nói lái in ~300 sentences, F1 95.47% | CITE-V `Y18-1063` (title, authors, venue); size and F1 from snippets (NUM-U) | "detect nói lái in classical text with rules and an n-gram model" | the F1 or the sentence count in prose | §2, §8 |
| N5 | "Beyond Atomic Tokens" factorizes Vietnamese syllables into onset, rime and tone for pretraining; complements rather than pre-empts a diagnosis of deployed tokenizers | `2609.21362` in `references.bib` with **authors not captured** ("FILL FROM ARXIV"); snippets only | cite as a preprint "that factorizes syllables before pretraining"; the entry must be completed before submission (R) | any detail of its results | §8 |
| N6 | Ghosh & Jyothi 2026: non-canonical segmentations lower accuracy across 27 languages | CITE-U; drops 9.9–23.7% relative are NUM-U | "lower accuracy across 27 languages" with the numbers inside a placeholder; Vietnamese coverage unconfirmed | "10–24 points" (they are relative drops); "including Vietnamese" | §8 |
| N7 | contemporaneous-work rule: papers appearing < 3 months before the deadline (after ~4 Oct 2026) need no detailed comparison | ARR CFP, "Citation and Comparison" (fetched 1 Oct 2026 from `acl-org/aclrollingreview` `cfp.md`) | cite and position; no obligation to compare in detail | — | §8 |

## 3. Linguistic and engine claims

| id | claim | evidence | strongest wording | overclaim | where |
|---|---|---|---|---|---|
| L1 | a written syllable is onset + rime + tone, rime = glide? + nucleus + coda?; 24 onsets (incl. ∅), 18 nuclei, 11 codas (incl. ∅), 6 tones | DD §2; `tables/rules_components.tex` generated from `noilai.vi.syllable`; anchors Thompson 1965, Đoàn 1977, Nguyễn 1997, Kirby 2011, Pham 2003 (all CITE-U) | state the parser's inventory as the parser's; cite the grammars for the model | "the standard analysis" without the anchors verified | §2 |
| L2 | spelling rules R1–R6 and the round trip R7; every rule checked against the dictionary with 0 exceptions unless stated (`gen`, `ka`) | DD §2.2; `tables/rules_spelling.tex`, `rules_onsets.tex` generated; `counts.json` `hunspell.nonstandard = [gen, ka]`; 3 level-toned stop-coda entries rejected (`gip, têt, xit`) | "checked against the two spelling lists; the exceptions are two loans and three onomatopoeic spellings, listed" | "exceptionless" | §2, §3.1, App. B |
| L3 | the two placement conventions differ on exactly 69 syllable types (oa, oe, uy after a non-qu onset) and reproduce both Hunspell lists 100% | `counts.json` `hunspell.placement_differing_syllables = 69`; `tests/test_vi.py`; `tables/rules_placement.tex` (69 pairs, test-asserted); DD §2.5 "verified" | "exactly 69 of the 6,611 lower-case entries differ" | "the rule is standard" without saying both lists reproduce | §2, App. B |
| L4 | Hunspell inventory: 6,642 entries, 6,611 lower-case, 16 unparsed (9 loans + 7 others), 6,595 parsable; 162 toneless rimes; 261 word-list additions; 6,830 extended structures | `counts.json` `hunspell.*`, `inventory.*` (base_structures 6,572, extended 6,830, extension_added 261, rimes 162); manifest `n_attested_syllables` 6,830 | quote with the file named | counts from the superseded parser (6,608; g 147 …) | §2, §3.1 |
| L5 | stop codas take only sắc and nặng; cross-tab rows c 0/0/197/0/0/145, ch 0/0/62/0/0/55, p 1/0/155/0/0/126, t 2/0/279/0/0/229 | `counts.json` `hunspell.coda_tone` | "take only sắc and nặng (the three level-toned entries are loans, excluded)" | — | §2 |
| L6 | the qu analysis (glide vs onset) gives different swaps; qu- inputs excluded pending a 20-pair native check | DD §2.1 O5; manifest `exclude_qu: true`; `t2:qu_input` drop 28 | as DD; "excluded, which removes the question rather than settling it" (Limitations) | "we adopt the correct analysis" | §3.1, Limitations |
| L7 | i/y emission per syllable from the corpus/word-list majority, never y after s or v; in v0.3 only `mĩ` takes y | DD §2.2 R5(c); `counts.json` `wordlist.iy_table_y_forms = ["mĩ"]`; `tables/rules_spelling.tex` row | "follows the reference count; today only one syllable takes y" | "the usage majority is y" (unverified shares) | §3.1 |
| L8 | the six-variant group (ℤ/2)³; reverse(V1)=V6, reverse(V2)=V3, reverse(V4)=V5; V2 and V3 produce the same unordered pair | algebra (DD §3.2); engine-verified; `tables/rules_variants.tex` | as DD | regional labels for variants (DD §3.5) | §2 |
| L9 | six-type tradition documented in Vietnamese sources (plo.vn; Lê Trung Hoa & Hồ Lê 1990; Nguyễn Văn Hiệp) | CITE-U for the book and the survey (book not read; from a reviewer's recollection and a news snippet) | "documented in newspaper and popular sources and in one book we could not consult directly" (Limitations) | "the standard six types" | §2, Limitations |
| L10 | degenerate pairs: equal tones make V3 the identity and V2 a plain reversal; equal onsets make V4 a plain reversal; equal rimes make V1 the identity | DD §3.4; manifest `degenerate_counts` (T1-V1 303, V2 35, V3 107; V4 none by construction) | as DD; "the non-degenerate subset is the primary basis for variant-vs-variant claims" | variant comparisons on all items without the stratum | §3.3, §5 |
| L11 | NFD puts the tone mark before the quality mark for ộ ậ ặ ệ and after the horn for ớ ứ; PC form = quality precomposed, tone combining | Unicode canonical ordering (DD §2.6); `noilai/vi/unicode.py` | as DD | "NFD makes the tone the last symbol" | §2 |
| L12 | byte-length asymmetry: hỏi and nặng letters are always 3 bytes; ý is 2 bytes | `counts.json` `byte_length` | stratify E2/E4 by tone class and vowel block | — | §4/App. |

## 4. Dataset claims (release v0.3)

| id | claim | evidence | strongest wording | overclaim | where |
|---|---|---|---|---|---|
| D1 | 10,000 generated items: T1 4,000, T2 2,000, T3 4,000 (2,000 yes/no pairs); 2,500 base pairs (1,500 lexical drawn, 1,000 pseudo); dev 2,000 / test 8,000 by base pair; core 1,496 | manifest `counts`, `core_counts`, `generator_args`; `tables/table1_counts.tex` (generated, `--release`); macros `\releaseN*` | quote the macros | "1,500 lexical base pairs in the release" (1,500 were drawn; 1,183 survive in the files per DATA_STATEMENT §H, counted at the freeze commit) | §3, Table 1 |
| D2 | main sample 4,200 items (350 per cell; public sampling seed 20261201; SHA-256 `9e2165…`); C2-enriched 500 items (seed 20261202; SHA `08d6c3…`; V1 114 / V2 155 / V3 150 / V4 81; all C2-affected; disjoint from the release) | manifest `samples`; `configs/run_plan.yaml` `item_files` | quote the macros and Table of hashes | sampling seeds described as "the seed" (the build seed is withheld) | §3.3, App. D |
| D3 | the build seed is not published: one seeded stream draws, shuffles and splits dev and test | DD §4.6 item 51; public manifest carries no `seed`; `tests/test_paper.py` scans every tracked file for it | "The build seed is not published" (phrase required by the test) | printing or promising the seed | §3.3 |
| D4 | content SHA-256 `92d332e5…` (items with canary fields removed) is the dataset's identity; canary GUID printed only as SHA-256 `dedff579…`; header record + `do_not_train` flags on every test item | manifest `content_sha256`, `canary_sha256`; DD §4.6 items 44, 50 | as DD; "the paper prints only its SHA-256" (test requires "SHA-256" in the same sentence as "GUID … print") | printing the GUID | §3.3, App. D |
| D5 | vulgar-flagged items 143 (12 of 12 cells), forced to the gated test split; 8 dev base pairs moved whole; input screen pair-only (0 pairs dropped) | manifest `vulgar_counts`, `dev_base_pairs_moved_for_vulgar`, `drops base:vulgar_input*` | quote the macros; "a per-syllable input screen would remove ordinary vocabulary" | "no vulgar item in the release" | §3.3, Ethics |
| D6 | filters and drop counts: identity 1,766, plain reversal 1,766 (mirror images across V1/V6, V2/V3, V4/V5), illegal 4,636 over 15,000 candidate outputs; 20 marginal rimes excluded (loan set + fewer than 4 dictionary types) | manifest `drops`, `marginal_rimes`, `marginal_rime_convention` | quote the macros | the DD's hand enumeration of marginal rimes (`oao` is kept by the code; DD §2.3 amendment) | §3.3 |
| D7 | pseudo pairs quota-sampled to the lexical base pairs' marginals on five strata, achieved shares within 0.01 of target, no shortfall | manifest `pseudo_quota` (`scope: base-pair pool`, shortfall 0 everywhere) | "quota-sampled … the achieved shares are in the manifest" (test requires "quota-sampled") | "per cell" (the quota is applied once over the pool; DD §4.2 amendment) | §3.3 |
| D8 | attested seed: 34 rows, 28 exact, 6 approximate (4 substitution, 2 merger), 24 H6-eligible, 5 three-syllable, 1 vulgar, 3 low-confidence, 0 native-verified | `tables/attested_facts.tex` macros (generated from `attested.jsonl`, private); RL-2026-10-01-01; DATA_STATEMENT §0 | quote the macros only (`\nAttested{}` …); "the target is 200–400" | typing 34/28 in prose; "verified" | §3.4 |
| D9 | three-valued exactness tag; every approx row reachable from the rule output by exactly the named change (test) | DD §3.7; `scripts/build_attested.py`; DATA_STATEMENT §0 lists the six mismatches | "three-valued tag" (phrase required) | "the folk forms are rule outputs" | §3.4 |
| D10 | attested strings frozen before the build (SHA `d3fe35…`); later rows never trigger regeneration (`attested_overlap`) | manifest `attested_strings_sha256`; DD §4.1 item 57 | as DD | — | §3.4 |
| D11 | lexical base pairs: 47,535 distinct canonical pairs from 49,103 two-syllable entries of a 73,901-entry list | `counts.json` `wordlist.*`; manifest `n_lexical_pairs_available` | quote | "the lexicon" as a corpus | §3.3 |
| D12 | release terms: code Apache-2.0; dev CC BY 4.0; test gated + encrypted CC BY-NC-ND 4.0; look-up lists GPLv2 consulted at build time; N Viet74K entries reproduced verbatim under permission or GPLv2 fallback | DD §4.1, §11.1; **pending L** | protocol with the fallback stated (`\placeholder`) | "released under CC BY" as a fact | §3.6, Ethics |
| D13 | XCOPA-vi: 500 test items, pure NFC, old-style placement majority (test: 51 old / 2 new of 53 affected tokens in 43 items; 9,594 syllable tokens; test+val: 60/2 of 62, 96.8%) | `placement_xcopa_test.json`, `placement_xcopa.json` (RL-2026-09-30-04) | "touches 43 of the 500 items (0.5% of syllable tokens)" | H4 on XCOPA | §6 |

## 5. Tokenizer audit and census (Gemma 3 unless stated)

| id | claim | evidence | strongest wording | overclaim | where |
|---|---|---|---|---|---|
| T1 | over the 6,595 standard syllables in running-text position: NFC 1.78 tokens/syllable, 29.5% single-token, alignment 0.94 over all syllables and 0.917 among the 4,652 split ones, onset–rime split in 51.5%; NFD 2.83, 7.1%, 0.27 / 0.215 among 6,130, tone mark isolated in 41.6%, onset–rime split 6.8% | `counts.json` `tokenizer_audit.gemma3.summary`; `gemma3_rows.csv`; RL-2026-09-30-01/06; `tests/test_paper.py` recomputes the among-split means | "splits 70% of syllables under NFC, half of all syllables exactly at the onset–rime seam" | "92% longer" (superseded); 2.90 for Gemma 3 (that is Gemma 2) | §1, §4 |
| T2 | Gemma 2: NFC 1.77 / 29.9% / 0.94 (0.912 among 4,624) / 50.7%; NFD 2.90 / 7.3% / 0.29 (0.234 among 6,115) / tone isolated 50.4% | `counts.json` `tokenizer_audit.gemma2` | appendix table | — | App. F |
| T3 | census on the fixed 1,000-string probe set (500 Viet74K multi-syllable words + 500 main-sample inputs, seed 0): Gemma 3 passes through NFD and PC (round trip exact 100%); NFD ids identical for 1.3% of strings, longer for 91.0%; PC identical 5.7%, longer 79.4%; Gemma 2 passes through, 1.5% identical, longer 91.7% | `gemma3.json` `normalization_census`; `gemma2.json`; RL-2026-10-01-02 | "the identity normalizer leaves NFD ids identical for 1.3% of the census strings" (replaces the stale "96.6% different" sentence of the earlier draft, a 500-string figure) | "3.4% identical ids"; "96.6%" | §4 |
| T4 | NFD tokens per syllable are 59% above NFC (2.83/1.78 − 1 = 58.8%) | `counts.json`; test `test_nfd_length_figures_in_the_h3_row_reproduce` asserts the exact phrase in `sec_counterfactuals.tex` | "longer for 91% of words, carries 59% more tokens per syllable" (exact phrase required) | "92% longer" | §6 |
| T5 | single-token share by tone (NFC): ngang 50.8%, sắc 28.3%, hỏi 23.2%, huyền 23.0%, nặng 22.4%, ngã 18.4% → token count predicts tone, hence the structural baseline | `counts.json` `single_token_by_tone`; RL-2026-09-30-06 | "under NFC token count itself predicts tone (51% vs 18–28%)" | — | §7 |
| T6 | 14 of 69 old-style placement forms are whole pieces in the Gemma 3 vocabulary against 2 new-style forms | DD §2.5 (reviewer computation, no RL id; not in `counts.json`) | do not quote in the paper until a script emits it | — | — |
| T7 | Figure 1: `mèo cái → mài kéo` under Gemma 3; NFC: mèo, cái, kéo single tokens, mài = ▁m + ài; NFD: every syllable gains a base–mark boundary; alignment 0.50 (mèo, kéo), 0.00 (cái, mài) | `fig1_tokens.json`; `fig1_tokens_note.tex` generated; test reproduces | the generated caption sentence `\figtokensnote{}` | "no internal boundary is linguistic" | Fig. 1 |
| T8 | HF `tokenizers` Precompiled normalizers can delete NFD tone marks (issue #2334); normalization is engine-dependent (arXiv 2609.20614) | DD §6.2; the arXiv item is NS-listed, not in any bib | state as the reason for the per-engine census, without citing the unverified preprint in the main text, or cite it as CITE-U | "HF tokenizers corrupt Vietnamese" as a fact | §4 |

## 6. Design claims (protocol; written in the present tense)

| id | claim | evidence | strongest wording | overclaim | where |
|---|---|---|---|---|---|
| S1 | primary metric: strict T1 accuracy, mean over three paraphrases, pooled over V1–V4 with equal weight, per model, default condition; API models: p0 core accuracy with the 300-item paraphrase range beside it | DD §5.1; PREREG §3 | as DD | mixing 1- and 3-trial means | §4 |
| S2 | T2 gold = every lexical reading under any of six variants in either order; validated gold for the core (pending V); `plausible_nongold` bin; variant class over three unordered classes | DD §5.2 | as DD | "dictionary gold" as final | §3.2 |
| S3 | T3 headline excludes spelling twins; balanced accuracy / d′; two log-probability quantities named `t3_yesno_lp`, `t3_pair_lp`, never compared with the generated answer; raw and per-token | DD §5.3 item 40, 63 (test requires both names and "length-normalized per token") | as DD | comparing forced-choice with generation | §4 |
| S4 | error taxonomy in fixed precedence: unparseable, correct, lenient_only, copy, reversal, wrong_variant, spelling, homophone, doublet, component, illegal (+ `plausible_nongold`, `wrong`) | DD §5.4; `noilai/eval/score.py` `ERROR_CLASSES` | list the eleven | — | §4 |
| S5 | extraction: NFC → strip thinking → last `đáp án:` marker (fallbacks) → task validation; unparseable counts as wrong, never excluded | DD §5.5–5.6 (rewritten from `extract.py`) | as DD | — | §4 |
| S6 | arms: base (NFC, old style, canonical i/y); C1 `nfd`; C1′ `pc`; C2 `placement_new`; C3a `strip_tones`; C3b `strip_all`; meaning-preserving arms applied to the whole prompt (primary), item-only as a secondary 500-item Gemma 3 1B check; strip arms item-only; a byte-identical arm pair is skipped and recorded `unchanged` | DD §6.1 item 48; `noilai/eval/prompts.py` `DEFAULT_ARM_SCOPE`; test requires "applied to the whole prompt" and "item text alone is a secondary condition" | as DD | "arms apply to the item text" | §4 |
| S7 | census three-valued per (tokenizer, engine): passes through / normalizes (0 by construction, asserted) / corrupts (arm refused); native vs HF ids asserted equal; C1 through two engines on a 500-item Gemma 3 1B subset | DD §6.2 | as DD | — | §4 |
| S8 | estimand τ_m(a) = within-item ATE, identified by construction (same item, prompt, paraphrase, decoding, engine); C2 is in-distribution both ways and leads the causal section | DD §6.4, §6.3 item 24; test requires H4 before H3 in `sec_counterfactuals.tex` | as DD | "mediation"; "share of the effect through token count" | §6 |
| S9 | what replaces mediation: per-model ATE; effect modification by Δtokens (associational); zero-dose contrast (≥ 100 items per model); three-arm ordered contrast base < pc < nfd; encoding × spacing 2×2; normalizing tokenizers as negative control | DD §8.6, §12.7; test forbids "mediat" outside the disclaimer sentences | as DD | — | §6 |
| S10 | statistics: base pair = cluster; paraphrases aggregated; bootstrap B = 2,000, percentile (BCa ≥ 50 pairs), stratified lexical/pseudo; paired t on base-pair means (≥ 200 pairs) primary; McNemar mid-p secondary, binary, single-model, "labelled as ignoring clustering"; Wilson for small cells (< 20 pairs) | DD §8.1–8.2; `noilai/constants.py` (BCA_MIN_CLUSTERS 50, PAIRED_T_MIN_BASE_PAIRS 200, SMALL_CELL_MAX_BASE_PAIRS 20) | as DD (phrases "base-pair cluster bootstrap" and "labelled as ignoring clustering" required) | two-way clustered SEs as primary; McNemar on proportions | §4 |
| S11 | Holm families: Table 3 = per model, {nfd, pc, strip_tones, strip_all} × {T1, T3, XCOPA} + {placement_new} × {T1, T3} on the C2 file = 14 cells minus undefined; H3 family of 2 per pass-through model; H3b 4 contrasts; H4/H6 per model across the panel; Table 2 estimation only | DD §8.3; `noilai.constants.HOLM_FAMILY_TABLE3` (phrase "fourteen cells" required) | as DD | "six intervention effects" | §4 |
| S12 | pooling: hierarchical, models nested in tokenizer families (Gemma SentencePiece, Qwen BPE, Llama 3.1, PhoGPT, Vistral/Mistral, o200k, Gemini), REML + Hartung–Knapp or family bootstrap, k of N families, "across the panel"; < 5 informative families → pooled CI labelled unreliable, per-family CIs decide (H3) | DD §8.1, H3 row (phrase "Hartung--Knapp" required) | as DD | "in models generally" / DerSimonian–Laird over 21 | §4, §6 |
| S13 | E2 model: `cbind(k, 3−k) ~ split + tps_w + align_w + lexical_in + lexical_out + logfreq_s1 + logfreq_s2 + variant + task + (1 + split + tps_w + align_w \| model) + (1 \| base_pair/item)`; primary = logit with model FE and model × fragmentation interactions, base-pair cluster-bootstrap CIs; H1 by nested LR / ΔAIC of M0 + tps_w vs + split + align_w; identifiability floor 300 misaligned split syllables per model, ≥ 4 families; with/without frequency; GEE robustness; random-slope generalization | DD §8.4 (phrases "nested likelihood-ratio test", "300 misaligned split syllables" required) | as DD | the standardized-coefficient comparison; "which matters more" | §4, §5 |
| S14 | power: 1,000 iid items at 70% → ±2.84; DEFF table (m, ρ); 10-point paired contrast at 80%: 100/145/227/308/356 items at 15/20/30/40/46% discordance (`n_for_mcnemar_power_conditional`); rule ≥ 300–350 per cell → 4,200 main sample; 125-item API cells resolve 11–17 points; XCOPA 500 resolves 2.7/3.9/5.5 at flip 5/10/20%; pilot ±6–7; the 94% figure is a best case | DD §8.5; PREREG §9; `noilai.stats.power` (test recomputes) | as DD | "94% power" unqualified | §4 |
| S15 | pre-registration two-stage; wording "before any test-split run", never "before any model was run"; pilot on dev only; nothing registered today (both commits `<fill>`) | DD §8.8; PREREG header; test forbids "were pre-registered in a dated commit" etc. and requires "before any test-split run" | present tense: "are fixed in a dated commit before any test-split run"; commits as `\placeholder` | past tense registration | §4, §5, App. E |
| S16 | exclusion rules (model outputs never excluded; items only on a validator-confirmed generator bug with full regeneration; vulgar by rule; demo/attested overlap at build; models only for engineering failure, named; bf16 drift > 5 points flags; human forms < 15/30, tools, failed check; probe layers never excluded) and stopping rules (fixed n; results freeze; no rerun because of a result; sample sizes may rise before the test run only) | PREREG §5–6; DD §8.8 | as PREREG, compactly, in §4 and App. E | — | §4, App. E |
| S17 | panel: 17 self-hosted + 4 API = 21 planned; frozen only after smoke tests with tokens/s; minimum viable panel ≥ 8 open models, ≥ 4 tokenizer families, one pass-through family beyond Gemma; cut order Gemma 3 12B, Qwen3.5-0.8B, Gemma 4 12B, then second models of a family; every cut model named | DD §7.1; `configs/models.yaml` (every `revision` null; 9 ids `uncertain`) | "as planned" with the table placeholder; the freeze rule and the minimum panel stated | "21 models" as achieved | §4, Table of models |
| S18 | decoding greedy, ≤ 64 new tokens, thinking off (template census, `n_thinking_chars` = 0), no system prompt, chat template rendered in the runner, three fixed demonstrations (`trung bình`, `làm chủ`, `vô hình`) in one user turn, three paraphrases p0–p2, Vietnamese instructions; explicit onset–rime–tone input is a main condition | DD §7.3–7.4; `prompts/noilai.yaml`, `demos.yaml`; `tables/prompt_examples.tex` generated | as DD; Vietnamese strings `[NATIVE-CHECK]` | — | §4, App. A |
| S19 | API policy: only the core to providers whose terms exclude training on inputs (Groq; OpenRouter with `data_collection: deny`); a provider whose tier trains on inputs (Gemini unpaid) never receives test items; paid key with opt-out or the dev-derived set, reported separately; adult account holder by role | DD §11.2; `configs/models.yaml` `terms`; `check_api_safety` | protocol (test requires "never receives test items") | past tense; "sent only the 1,500-item core" | §4, Ethics |
| S20 | E4: positions first/last/mark/after; stimuli 300 per tone incl. legal non-words, ≥ 4 carriers; 60/10/30 nested split + rime holdout; primary grid tone × {last, after} × 2 encodings × 5 split seeds × 3 control seeds; two-condition decodability (selectivity ≥ δ_sel = 0.15 provisional; excess over the structural baseline, syllable-clustered CI); patching pairs with the varying syllable second (`công tử`/`công tự` → `cổng tư`/`cộng tư`), filters (1)–(3), readout A = tone names, readout B = the V3 prompt on retained pairs (< 50 → A only, localization study); position groups G1–G6 (code: G1_first/G1_last, G3_context) | DD §9; `noilai/probe/*`; `noilai/constants.py` PROBE_*; test requires the pair strings and "reads the tone's name", "fewer than 50 pairs survive", "localization study", "No claim of earlier decodability under NFD" | as DD | digits as tone labels; steering or SAEs as part of the paper (future work only; "steering" may appear only in sec_discussion.tex) | §7 |
| S21 | tokenizer-fixed control pairs (SEA-LION E2B vs Gemma 4 E2B) require byte-identical tokenizer files; n = 1 "suggests, never isolates" | DD §7.2 | as DD | "isolates SEA training" | §4, §5 |

## 7. Hypotheses (all pending; wording fixed by DD §1 = PREREG §2, test-asserted identical)

| id | statement (short) | decided by | status | wording allowed now |
|---|---|---|---|---|
| H1 | alignment carries information beyond token count: β(align_w) > 0, β(tps_w) < 0, nested LR/ΔAIC improves fit | LR p ≤ 0.05 / ΔAIC ≥ 2, signs; floor 300 misaligned split syllables per model, ≥ 4 families | pending E2 | prediction + decision rule; "demoted to descriptive if fewer than four families clear the floor" |
| H2 | PhoGPT-4B-Chat's 20,480-piece vocabulary is the panel's extreme on whole-syllable share; NLL and T1 reported beside Qwen3.5-4B and Gemma 3 4B | descriptive case study, no inequality | pending E1/E2 | "descriptive case study, not a tested hypothesis" (phrase required) |
| H3 | crossover in pass-through tokenizers: τ(NFD, T1-V3) > 0, τ(NFD, XCOPA) < 0; distribution shift is the default explanation to beat | two paired tests per model, Holm in pair; both pooled CIs on the predicted side; per-family table primary; < 5 informative families → per-family CIs decide | pending E3 | prediction only; "a uniform drop is reported as OOD sensitivity" |
| H3b | tone-isolation DiD on V3 > 0 in Gemma 3 4B (replicated on 1B); V1 contrast ≈ 0 | matched items, cluster bootstrap, McNemar mid-p secondary, Holm over 4 | pending E3 | prediction only |
| H4 | the rarer placement convention (expected: new style; fixed by the corpus count CC) lowers accuracy on affected items in models generally | per-model paired tests on the C2-enriched set, Holm; k of N + pooled; TOST δ = 2 | pending CC, E3 | "expected: the newer one; the direction is fixed by a corpus count before any model run" |
| H5 | dissociation in Gemma 3: tone decodable at `after` by L/2 in a model with V3 accuracy < 25%, and perception recovers where manipulation does not | DD §9; readout B on retained pairs; < 50 pairs → A only, localization study | pending E4 | prediction + fallback; "a clean localization without the dissociation would be equally informative" |
| H6 | attested exact items outscore matched generated; guided-instruction contamination > 0 on attested, ≈ 0 on matched | matching DD §8.7; floor ≥ 100 verified exact rows from ≥ 3 collections, else descriptive | pending V, E1; today 24 eligible, 0 verified | prediction + floor ("at least 100 native-verified exact" phrase required) |

Secondary expectations (reported, not tested): V2 > V1 ≈ V4 > V3 on non-degenerate items; spelling errors concentrate on c>k, g>gh, ng>ngh triggers; copy and reversal errors in the smallest models (PREREG §2).

## 8. Validation, human baseline, ethics (protocol; nothing has happened)

| id | claim | evidence | strongest wording | overclaim | where |
|---|---|---|---|---|---|
| V1 | three adult native validators, one per region if possible; 1,000-item probability sample with stratum weights; two per item, 200 triple; four judgments + T2 readings + region tag; α with raw agreement, marginals and Gwet's AC1; MASI/Jaccard for sets; adjudication logged; confirmed bug → regeneration before any model run; generator precision per cell with CI | DD §10.1; `docs/VALIDATOR_INSTRUCTIONS.md`; `scripts/make_validation_forms.py` | protocol; results `\placeholder` | "validated" | §3.5, App. C |
| V2 | agreement illustration: two coders with 1% independent error each at 96% prevalence give raw 0.980, α 0.788, AC1 0.978; 2% error: 0.961 / 0.645 / 0.956 | RL-2026-09-30-06 (simulated with `noilai.stats.agreement`; test reproduces) | "high prevalence depresses α; we report raw agreement and AC1 beside it" | the earlier "α = 0.84 at 98%" | §3.5 |
| HB1 | 20 respondents × 30 items; 6 anchors; 240 items double-judged → 246 distinct, ≈ 20 per cell, from the main sample; cyclic block-offset design: every one of the 190 rater pairs shares one (140) or two (50) items; same p0 prompt and demos; instruction check; no dictionary/search/AI; mean-human comparator, "any human correct" as a ceiling; ±4–5 points → reference band; scored strict, lenient and doublet/homophone-tolerant; 10-item natural-competence block on attested items; no superhuman or per-cell claim | DD §10.2; `docs/HUMAN_BASELINE_FORM.md`; test checks the arithmetic | protocol | "models beat humans" | §3.5, App. C |
| X1 | participants: adult volunteers, bilingual written consent, unpaid, withdrawal until the results freeze, letters not names, coarse demographics (age band, region, years in Vietnam, raised abroad); no IRB (D4 "No" with justification); recruited by personal invitation, no dependency; vulgar items excluded from forms | DD §11.7; `docs/CONSENT_FORM.md`; `docs/CHECKLIST_DRAFT.md` | protocol (test requires "review board" and "18") | "took part", "were recruited" (past tense forbidden by the test) | Ethics |
| X2 | offensive content: blocklist at generation on outputs, readings and candidates per syllable and pair, inputs only when the pair itself is taboo; flagged items out of dev, APIs and forms; validators judge offensive readings; dual use: a fluent spoonerizer could evade keyword filters; the same engine is the detector; gated release | DD §11.5 | as DD (phrase "only when the input pair itself is a taboo phrase" required) | "to inputs, gold answers, readings and candidates" (superseded) | §3.3, Ethics |
| X3 | AI assistance: code, documentation and drafts produced with an AI coding assistant under the authors' direction; items and gold from the rule engine; no LLM judge; disclosed in the checklist now and the Acknowledgements at camera-ready | DD §11.6; `docs/AI_USE_LOG.md`; ARR CFP "AI Writing/Coding Assistance Policy" | as DD | — | Ethics |
| X4 | reproducibility: release contents; manifests (fields DD §7.5); compute totals **pending**; anonymized repository link only | DD §7.5, §11.1 | protocol; URL as `\placeholder{anonymized repository URL}` (`\anonrepo`) | a github.com URL (test forbids "github.com/") | Ethics |
| X5 | API account holder: an eligible adult by role if the author is under 18; every provider requires 18+ | DD §11.2 | one sentence, present tense | naming the holder | Ethics |

## 9. Related work: citation status

Keys in `paper/references.bib` are Anthology ids (`2024-findings-emnlp-86`) or arXiv ids; keys in
`paper/references_PLACEHOLDER.bib` are surname-year and every entry carries `note = {UNVERIFIED}`.
"Verified" means the identifier resolves to a paper with the stated title and authors (Anthology XML
read on 1 October 2026 from `raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/`), not
that any number attributed to it was checked. arXiv-only and non-ACL entries could not be opened from
this machine (arxiv.org, aclanthology.org, api.semanticscholar.org, dblp.org and crossref are
egress-blocked) and remain CITE-U unless the previous round verified them through a mirror (RWV §1).

**Verified this round from the Anthology XML (moved into `references.bib`, key = Anthology id).**

| topic | key | paper |
|---|---|---|
| tokenization effects, sub-word | `2024-findings-emnlp-86` | Chai, Fang, Peng, Li (2024) Tokenization Falling Short: On Subword Robustness in LLMs, Findings EMNLP |
| | `2023-acl-long-284` | Zouhar et al. (2023) Tokenization and the Noiseless Channel, ACL |
| | `2024-emnlp-main-40` | Schmidt et al. (2024) Tokenization Is More Than Compression, EMNLP |
| | `2024-findings-acl-134` | Goldman et al. (2024) Unpacking Tokenization, Findings ACL |
| | `2024-findings-naacl-247` | Ali et al. (2024) Tokenizer Choice for LLM Training, Findings NAACL |
| | `2024-acl-short-73` | Uzan et al. (2024) Greed is All You Need, ACL short |
| | `2020-findings-emnlp-414` | Bostrom & Durrett (2020) BPE is Suboptimal for LM Pretraining, Findings EMNLP |
| | `2021-acl-long-243` | Rust et al. (2021) How Good is Your Tokenizer?, ACL |
| | `2023-emnlp-main-614` | Ahia et al. (2023) Do All Languages Cost the Same?, EMNLP |
| | `2023-findings-acl-350` | Limisiewicz, Balhar, Mareček (2023) Tokenization Impacts Multilingual Language Modeling, Findings ACL |
| | `2022-acl-short-43` | Hofmann, Schütze, Pierrehumbert (2022) An Embarrassingly Simple Method to Mitigate … Tokenizers, ACL short |
| | `2024-naacl-long-284` | Truong et al. (2024) Revisiting subword tokenization: affixal negation in LLMs, NAACL |
| | `2025-findings-acl-572` | Wegmann, Nguyen, Jurgens (2025) Tokenization is Sensitive to Language Variation, Findings ACL |
| | `2025-emnlp-main-919` | Jang et al. (2025) Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers, EMNLP |
| | `2025-acl-long-1546` | Lotz et al. (2025) Beyond Text Compression: Evaluating Tokenizers Across Scales, ACL |
| | `2026-acl-long-342` | Foroutan et al. (2026) Parity-Aware BPE, ACL |
| | `2026-eacl-long-394` | Alqahtani et al. (2026) Stop Taking Tokenizers for Granted, EACL |
| | `P16-1162`, `D18-2012`, `P18-1007` | Sennrich et al. (2016) BPE; Kudo & Richardson (2018) SentencePiece; Kudo (2018) subword regularization |
| | `2022-tacl-1-17`, `2022-tacl-1-5` | ByT5 (Xue et al. 2022); CANINE (Clark et al. 2022) |
| character/sub-word knowledge and probing | `2022-naacl-main-179` | Kaushal & Mahowald (2022) What do tokens know about their characters, NAACL |
| | `2022-naacl-main-373` | Itzhak & Levy (2022) Models In a Spelling Bee, NAACL |
| | `2023-findings-acl-770` | Huang, Wu, Mahowald, Potts (2023) Inducing Character-level Structure … Interchange Intervention Training, Findings ACL |
| | `2025-emnlp-main-1434` | Cosma et al. (2025) The Strawberry Problem, EMNLP |
| | `2025-acl-long-194` | Xu et al. (2025) Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning, ACL |
| | `2026-eacl-long-282` | Sato & Sasano (2026) How Do Language Models Acquire Character-Level Information?, EACL |
| | `2026-acl-long-915` | Hou, Hu, Zhang (2026) SubTokenTest, ACL |
| | `2021-acl-long-144` | Finlayson et al. (2021) Causal Analysis of Syntactic Agreement Mechanisms, ACL (patching precedent in the Anthology) |
| | `2021-eacl-main-295`, `2020-acl-main-420`, `P18-1198` | Ravichander et al. (2021); Pimentel et al. (2020); Conneau et al. (2018) — probing methodology |
| wordplay and phonology for LLMs | `Y18-1063` | Pham & Pham (2018) Building a Spoonerism Detection System for Vietnamese, PACLIC 32 (no pages or DOI in the XML) |
| | `2026-acl-long-1041` | Yue, Shen, Lu (2026) Phun-Bench: Evaluating LLMs on Phonological Understanding in Chinese, ACL |
| | `2024-knowllm-1-1` | Suvarna, Khandelwal, Peng (2024) PhonologyBench, KnowLLM @ ACL |
| | `2025-emnlp-main-961`, `2025-findings-acl-1132` | PhonoThink (Ma et al. 2025); P-CoT phonological reasoning (Jang et al. 2025) |
| | `2025-ranlp-1-15` | Baranov et al. (2025) KoWit-24, RANLP |
| | `2024-emnlp-main-657` | Xu et al. (2024) "A good pun is its own reword", EMNLP |
| | `2025-emnlp-main-1419` | Zangari et al. (2025) Pun Unintended, EMNLP |
| | `S17-2005` | Miller, Hempelmann, Gurevych (2017) SemEval-2017 Task 7 puns |
| | `2025-conll-1-1` | Cheng et al. (2025) HKCanto-Eval, CoNLL |
| | `2026-acl-long-1872` | Miletić, Kallini, Shutova (2026) Phonemes to the Rescue, ACL |
| Vietnamese NLP | `2020-findings-emnlp-92` | Nguyen & Tuan Nguyen (2020) PhoBERT |
| | `2022-naacl-srw-18` | Phan et al. (2022) ViT5 |
| | `2023-emnlp-main-315` | Nguyen et al. (2023) ViSoBERT |
| | `2024-findings-naacl-261`, `2024-findings-naacl-15` | ViGLUE (Tran et al. 2024); VLUE (Do et al. 2024) |
| | `2026-acl-long-472` | Dinh et al. (2026) When Morphology Hides in Plain Sight: … Vietnamese, ACL |
| | `2024-emnlp-main-426`, `2023-findings-emnlp-925` | ViMD dialects (Dinh et al. 2024); Central–Northern dialect transfer (Le & Luu 2023) |
| | `2021-naacl-demos-1` | PhoNLP (Nguyen & Nguyen 2021) |
| diacritics and Unicode | `2025-naacl-short-25` | Gorman & Pinter (2025) Don't Touch My Diacritics, NAACL short |
| | `D19-1151` | Alqahtani, Mishra, Diab (2019) Efficient CNNs for Diacritic Restoration, EMNLP |
| | `2024-naacl-long-420` | Chen, Adebara, Abdul-Mageed (2024) Interplay of MT, Diacritics, and Diacritization, NAACL |
| | `2024-lrec-main-1479` | Ansary et al. (2024) Unicode Normalization and Grapheme Parsing of Indic Languages, LREC-COLING |
| | `2025-findings-acl-969` | Cooper, Blanco, Surdeanu (2025) The Lies Characters Tell: … Adversarial Unicode Perturbations, Findings ACL |
| | `2020-lrec-1-429` | Moon & Okazaki (2020) Jamo Pair Encoding, LREC |
| benchmark construction, pre-registration, evaluation practice | `2021-naacl-main-51` | van Miltenburg, van der Lee, Krahmer (2021) Preregistering NLP research, NAACL |
| | `2024-lrec-main-809` | Reuver, Verberne, Fokkens (2024) … A Preregistered Study, LREC-COLING |
| | `2021-naacl-main-385` | Bowman & Dahl (2021) What Will it Take to Fix Benchmarking in NLU?, NAACL |
| | `2020-emnlp-main-745` | Card et al. (2020) With Little Power Comes Great Responsibility, EMNLP |
| | `D19-1224` | Dodge et al. (2019) Show Your Work, EMNLP |
| | `2022-findings-emnlp-196` | Ulmer et al. (2022) Experimental Standards for Deep Learning in NLP Research, Findings EMNLP |
| | `P19-1267`, `2021-eacl-main-156` | Gorman & Bedrick (2019) standard splits; Søgaard et al. (2021) random splits |
| | `2022-acl-short-18`, `2023-findings-emnlp-722`, `2024-naacl-long-482` | data contamination: Magar & Schwartz (2022); Sainz et al. (2023); Deng et al. (2024) |
| | `2020-emnlp-main-393`, `2020-acl-main-442`, `2021-naacl-main-324` | Ethayarajh & Jurafsky (2020); CheckList (Ribeiro et al. 2020); Dynabench (Kiela et al. 2021) |
| | `2020-emnlp-main-185` | Ponti et al. (2020) XCOPA, EMNLP (moved from the placeholder file) |

**Verified by the previous round (RWV §1; already in `references.bib`)**: `2024-emnlp-main-177` (CUTE), `2025-findings-acl-95` (EXECUTE), `2025-findings-emnlp-719` (Spelling-out), `2026-acl-long-634` (STAD), `2025-acl-long-1374` (Lesci et al.), `2025-acl-short-75` (Gambardella et al.), `2026-findings-eacl-22` (Arabic diacritics), `2024-findings-naacl-182` (Crossing Linguistic Horizons), `2024-naacl-long-239` (tone in speech models), `2025-acl-long-563` (VMLU), `N18-5012` (VnCoreNLP), `2020-tacl-1-25` (BLiMP), `P19-1116` (Voita et al.), `D19-1275` (Hewitt & Liang), `2022-cl-1-7` (Belinkov), `2023-acl-long-230`, `2023-emnlp-main-306` (Hu et al.; Hu & Levy), `2023-emnlp-main-308` (Jacovi et al.), `J08-4004`, `L06-1392`, `P18-1128`, `Q17-1033`, `Q18-1041`, `2022-emnlp-main-731`, `2024-acl-long-620` (Aya), `2020-findings-emnlp-195` (Masakhane), `2025-findings-emnlp-587` (multilingual judges), `2026-tacl-1-10` (MultiBLiMP), `2025-acl-long-229`, `2025-naacl-long-312`, `kantamneni25a`; arXiv entries verified through mirrors or snippets only: `2308.08493`, `2310.11324`, `2311.02945`, `2411.00640`, `2511.04703` (author list incomplete), `2606.15044`, `2608.10414`, `2609.21362` (**authors missing**), `2609.26942`, `2609.19885`, `2608.09280`, `2406.07882`.

**Unverified (placeholder file; cited in the draft; R before submission)**: grammars and phonology (Thompson 1965; Đoàn Thiện Thuật 1977; Nguyễn Đình-Hoà 1997; Kirby 2011; Pham 2003), the Hunspell resource, the vlstudies blog (not cited in the paper), Lê Trung Hoa & Hồ Lê 1990, Nguyễn Văn Hiệp (year and venue unknown), Krippendorff 2011, McNemar 1947, Holm 1979, Connor 1987, Meng et al. 2022 (ROME), Heimersheim & Nanda 2024, Gemma 3 report, Gemma Scope (Lieberum et al. 2024), Gemma Scope 2 blog, BIG-bench (Srivastava et al. 2023), lm-evaluation-harness, Petrov et al. 2023 (NeurIPS), PACUTE (arXiv 2606.15144), TokSuite (2512.20757), Ghosh & Jyothi (2607.26831), UGTPhon (2609.27205).

**Numbers attributed to sources that may appear only inside `\placeholder{}`**: Bean et al. 16.0% of 445 benchmarks; Ghosh & Jyothi relative drops 9.9–23.7%; Pham & Pham ~300 sentences and F1 95.47%; Sclar et al. "76 points" (the paper says "tens of points"); EXECUTE's language count; SEA-HELM's diagnostic coverage; MultiBLiMP's exclusion of Vietnamese; KoWit-24's 2,700 headlines (now CITE-V via the abstract: "2,700 Russian news headlines" — may be typed); PhonologyBench's 17%/45% gaps (abstract-level, not needed).

## 10. ARR compliance (CFP wording fetched 1 October 2026 from `acl-org/aclrollingreview` `cfp.md`, `authors.md`, `authorchecklist.md`)

| rule (quoted) | what the draft does |
|---|---|
| Long papers: "up to eight (8) pages of content" plus "unlimited extra space after the conclusion for limitations (required) and optional section on ethical considerations … plus unlimited pages of references"; appendices "come after the references … do not count towards the page limit" and "must follow the official double-column format" | `main.tex`: 9 numbered sections budgeted to 8.0 pages (page-budget comment); **the compiled draft is over budget: see the page accounting below**; `\section*{Limitations}` and `\section*{Ethical Considerations}` after the conclusion, before `\bibliography`; `\appendix` after it; two-column throughout (`table*` floats only) |
| "a dedicated section titled 'Limitations'"; "should not introduce new methods, analysis, or results"; "Papers without a limitations section will be desk rejected" | `sec_limitations.tex`: methodological caveats only; a closing `\placeholder` forces a re-read against the final results, never new results |
| optional ethics section, "we recommend it to be titled 'Ethical considerations'" | titled "Ethical Considerations" (the test fixes this spelling) |
| anonymity: no names or affiliations; no identifying self-references; "Links to file hosting services that can track downloads, such as Dropbox, are not allowed"; supplementary software "properly anonymized (e.g., Anonymous GitHub)" | `\author{Anonymous ACL submission}`, `[review]` option; one `\anonrepo` macro = `\placeholder{anonymized repository URL}`; no github.com URL in any `.tex` (test); the AI-assistance disclosure defers the Acknowledgements to camera-ready |
| "must use the official ACL style template"; no modified style files | `acl.sty`, `acl_natbib.bst` from acl-org/acl-style-files kept unchanged; only `\usepackage[T1,T5]{fontenc}` and standard packages added in the preamble (allowed) |
| related work "placed exclusively in the appendices … desk rejection"; "main text … must be self-contained, and reviewers are not expected to read the supplementary materials" | a numbered Related Work section in the body; appendices hold prompts, rule tables, protocols, data statement, hashes, extra results only |
| Responsible NLP checklist: "incorrect, incomplete or misleading information … can result in desk rejection" | `docs/CHECKLIST_DRAFT.md` answers A–E; the paper's sections it points to exist |
| AI assistance: "use for writing or coding, as well as its scope, must be disclosed in the Responsible NLP Checklist" | Ethics "Generative assistance" + `docs/AI_USE_LOG.md`; Acknowledgements at camera-ready |
| "hallucitations" are a desk-rejection reason (author checklist) | every cited key resolves (test); verified vs unverified split in two bib files; R before submission |
| contemporaneous work: "less than 3 months before the submission deadline … not obliged to make detailed comparisons" | September 2026 preprints are cited and positioned, not compared in detail |

### Page accounting of the compiled draft (`pdflatex → bibtex → pdflatex ×2`, zero errors; measured by the position of each heading in the PDF text, so ±0.1 page)

<!-- pages:begin -->
| section | starts on page | measured length (pages) | budget (`main.tex`) |
|---|---|---|---|
| 1 Introduction | 1.7 | 0.76 | 1.00 |
| 2 Background | 2.5 | 1.13 | 0.75 |
| 3 Benchmark | 3.6 | 2.90 | 1.50 |
| 4 Experimental Design | 6.5 | 1.20 | 1.00 |
| 5 Results | 7.7 | 2.05 | 1.25 |
| 6 Counterfactuals | 9.7 | 0.98 | 1.00 |
| 7 Tone | 10.7 | 1.81 | 0.75 |
| 8 Related Work | 12.5 | 0.94 | 0.50 |
| 9 Discussion | 13.5 | 0.18 | 0.25 |
| **content (1–9)** | 1 | **12.7** | **8.00** |
| Limitations + Ethics | 13.7 | 3.01 | not counted |
| References | 16.7 | 5.32 | not counted |
| Appendices | 22.0 | 11.01 | not counted |

Sections 1-4 and 8: 6.93 pages (budget 4.75); Sections 5-7 and 9: 5.02 pages (budget 3.25); total pages in PDF: 32
<!-- pages:end -->

Sections 1–4 and 8 (this workstream) carry Figure 1, Table 1 and the model table (≈1.2 pages of floats) and ≈4,600 words of text after the condensation of 1 October 2026 (intro ≈660, background ≈760, benchmark ≈1,500, setup ≈1,140, related work ≈560 words; every protocol sentence they dropped is in Appendices A, C, D, E and H). Sections 5–7 and 9 are results-dependent and were not redrafted. The draft therefore does **not** fit the 8-page limit yet; the cut list for when the numbers land, in order of least damage to the self-contained main text:

1. Sections 5–7 and 9: 5.0 pages against 3.25 (results workstream; most of it is placeholder tables at full width).
2. §3 Validation and human baseline → three sentences in the body, the protocol in Appendix D (≈0.3 page).
3. §4 Tokenizer profile → the definitions stay, the Gemma 3 numbers move into the caption of the audit table (≈0.15 page; `tests/test_paper.py` pins two of the numbers to `sec_setup.tex`, so the sentence that carries them must stay or the test be amended).
4. The model table → Appendix H, with the panel described in one sentence (≈0.25 page).
5. §2 Nói lái: drop the six-way `thay đổi` illustration (unverified against Lê & Hồ 1990) and the three-syllable example (≈0.15 page).
6. §8: drop the second citation group of every sentence (≈0.2 page); the Anthology-verified entries stay in the `.bib`.

## 11. Provenance index of numbers typed in the prose

Every number that appears outside a `\placeholder{}` and is not a generated macro:

| number | where | source |
|---|---|---|
| 6,595 standard syllables; 6,611 lower-case entries; 16 unparsed; 3 stop-coda rejects | §1, §2, §3.1 | `counts.json` `hunspell.*` (`parsable`, `lowercase_letter_entries`, `unparsed`, `rejected_phonotactics`) |
| 69 placement pairs | §2, App. B | `counts.json` `hunspell.placement_differing_syllables`; `tests/test_vi.py`, `tests/test_paper.py` |
| 24 onsets, 18 nuclei, 11 codas, 6 tones | §2 | `noilai.vi.syllable` constants; `tables/rules_components.tex` |
| 162 rimes; 261 additions | §3.1 | `counts.json` `inventory.rimes`, `inventory.extension_added` |
| 70% split, 51.5% onset–rime, one extra token (1.78 → 2.83), 42% isolated | §1 | `counts.json` `tokenizer_audit.gemma3.summary` (1 − 0.2946; 0.5152; 1.78/2.83; 0.4164) |
| 1.78, 29.5%, 0.92 (0.94), 51.5%, 2.83, 7.1%, 0.21 (0.27), 41.6%, 1.3% identical ids, 91% longer | §4 | same; `gemma3.json` `normalization_census` (`nfd_same_ids_frac` 0.013, `nfd_longer_frac` 0.91) |
| 59% more tokens per syllable | §6 | `counts.json` (2.8308/1.7826 − 1); test-asserted phrase |
| 51% vs 18–28% single-token by tone | §7 | `counts.json` `single_token_by_tone` |
| 43 of 500 XCOPA items; 0.5% of syllable tokens; 51 old / 2 new of 53 | §6 | `placement_xcopa_test.json` |
| 47,535 candidate pairs | §3.3 | macro `\releaseNLexicalPairsAvailable` |
| 4,200 / 350 per cell; 1,496 core; 125 per cell / 62 pairs; 20% dev | §3.3, §4 | macros; DD §4.5–4.6 |
| 2,000 bootstrap replicates; 50 pairs (BCa); 200 pairs (paired t); 20 pairs (Wilson); 14 cells; 300 misaligned syllables; 4 families; δ_sel 0.15; 50 pairs (readout fallback); 25% (H5 condition); L/2; 2-point TOST | §4, §6, §7 | `noilai/constants.py`; DD §1, §8, §9 |
| ±2.8 points; 3.4–5 points; 94% / 65%; 350 per cell; 11–17 points; 100/145/227/308/356 | §4 | DD §8.5; `noilai.stats.power` |
| 20 respondents, 30 items, 6 anchors, 24 items, 246 items, 190 pairs (140 / 50) | §3.5, App. C | DD §10.2; test arithmetic |
| 3 validators, 1,000 items, 200 overlap, 20-pair checks | §3.5 | DD §10.1 |
| 17 + 4 = 21 models; ≥ 8 models, ≥ 4 families | §4 | `configs/models.yaml`; DD §7.1 |
| 3 demonstrations; 64 tokens; 3 paraphrases; 500-item scope check; 200-item bf16 drift; 5 points drift threshold | §4 | DD §7.3–7.4, §6.1, PREREG §5 |
| 300 per tone; ≥ 4 carriers; 60/10/30; 5 × 3 seeds; 600 candidate pairs → 200; 1 nat | §7 | DD §9.2–9.3; `noilai/constants.py` |
| 0.980 / 0.788 / 0.978 and 0.961 / 0.645 / 0.956 | §3.5 | RL-2026-09-30-06 (simulation; test reproduces) |

## 12. Drafting decisions taken in this round (1 October 2026) and open author decisions

Taken (all inside `paper/`; recorded here because `docs/` is outside this workstream):

1. A numbered **Related Work** section (`sec_related.tex`) replaces the related-work paragraph of Background, so that no related work lives only in an appendix (ARR); Background keeps the syllable, the three encodings and nói lái. Page budget re-cut to 8.0 pages (comment in `main.tex`).
2. **Freeze and hashes**: `paper/gen_release_facts.py` emits `tables/release_facts.tex` (macros) and `tables/release_hashes.tex` (appendix table) from the public manifest and the run plan; §3 and the data-statement appendix quote the macros.
3. **Bibliography**: in-entry `%` comments of `references.bib` (illegal in BibTeX; they truncated 15 entries) moved outside the entries; the stacked T5 accent forms of the placeholder file braced (`{\'{\^e}}`), because the unbraced form does not compile under vntex; verified entries (section 9) moved from the placeholder file into `references.bib` under Anthology-id keys and the `\cite` keys updated; new related work added only when verified.
4. **Stale figure corrected**: the Setup paragraph's "96.6% of NFD words with different ids" (a 500-string census figure) is replaced by the 1,000-string census values (identical ids 1.3%, longer 91%).
5. **Anonymized URL**: one macro `\anonrepo` for every code/data link.
6. **Compile check**: TeX Live + vntex installed on the build machine; `pdflatex → bibtex → pdflatex × 2` builds `main.pdf` with zero errors, zero undefined references or citations and zero BibTeX warnings; the page accounting is in section 10 and in the PR description.
7. **Condensation**: Sections 1–4 and 8 were cut to roughly 660/760/1,500/1,140/560 words; the build, filter, quota, canary and seed mechanics went to a new Appendix C (*Build, Filters and Sampling*), the error taxonomy, census mechanics, arm scope, bootstrap, Holm families, regression specification, intervention estimands and power to a new Appendix E (*Scoring, Census and Statistics*), the run conventions to Appendix H and the decoding, paraphrase and ablation details to Appendix A. Every phrase `tests/test_paper.py` asserts stayed in the file it is asserted in.
8. **Layout**: Figure 1, Table 1 and the onset and variant rule tables are full width (`figure*`/`table*`; they overflowed a column by 74–151 pt); the release-identity table prints each 64-character digest on two lines; the tokenizer audit table is `\footnotesize` with 4 pt column separation; `\tokspace` renders as `\ensuremath{\sqcup}` because the T5 Times fonts have no visible-space glyph; the model table and the placeholder results tables are `\footnotesize`. The compile log reports no overfull box in a file of this workstream (three remain in the placeholder tables of Sections 5–7).
9. **Review**: an adversarial review (five lenses: design-document fidelity, overclaiming against this ledger, citations, ARR compliance and numeric consistency, linguistic examples against the rule engine; each finding re-checked by an independent verifier) ran over the in-scope files before the commit; confirmed findings were fixed and are listed in the PR description.

Open (DD §13; they change wording, not evidence):

- title choice (DD §13.13; the current title is plan §2.1's first option);
- ARR area ("Phonology, Morphology and Word Segmentation" default, DD §12.45);
- GPL permission vs the GPLv2 fallback (§3.6, Ethics placeholders);
- Gemini: paid key with opt-out, dev-derived set, or drop (§4, Ethics);
- the qu convention, V5 as a fifth cell, the i/y emission rule (§3.1 placeholders);
- the pre-registration dates and the co-author's edits (App. E placeholders).
