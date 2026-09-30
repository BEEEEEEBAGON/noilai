# Risk register — NóiLái / Bundle A (30 September 2026)

Merges the founding plan's register (`docs/PLAN_2026-09-30.md` §4) with the risks raised by the eight lens reports. Likelihood and impact are judgments. "Early signal" is the observable that says the risk is materializing; "owner" is who acts (author = only the author can; implementer = the next code session; validators = the native validators; co-author = the ARR-qualified collaborator once recruited). Binding mitigations are cross-referenced to `docs/DESIGN_DECISIONS.md` (DD §).

## A. Procedure and venue (from the plan; unchanged unless noted)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| No ARR-qualified co-author or voucher by mid-December | Medium–high | Severe: lottery, then desk rejection if not drawn | No reply to week-1 emails by 25 Oct; no "in principle" by Gate 2 (8 Nov) | Outreach from week 1 with the tokenizer-audit result and pilot write-up (`docs/CONTRIBUTOR_OUTREACH.md`); widen to ACL mentorship Slack, SEACrowd, Cohere Labs by Gate 2; backup name; by 18 Dec choose lottery vs direct SRW | author |
| Contributor over-committed, misses the 48-hour form, or defaults on reviews | Low–medium | Severe | Contributor cannot confirm a free January slot | Confirm slot; skeleton submission when the site opens (~21 Dec); verify registration by 5 Jan; agree a substitute | author, co-author |
| Mechanical desk rejection (profiles, anonymity, Limitations, length, template, checklist, citations, quality bar) | Low if audited | Severe; citations even after acceptance | OpenReview profile not active by mid-Dec; any reference without a hand check | Accounts now (ORCID, OpenReview on a school email); every reference hand-checked against Anthology/arXiv (`docs/RELATED_WORK_VERIFICATION.md`); exact template; ARR author checklist a week early; outside read in December | author |
| Scoop | Medium | Medium–high | Gate 1 / December sweep finds an LLM nói lái or Vietnamese re-encoding paper; a UIT-group LLM follow-up to "Beyond Atomic Tokens"; an EXECUTE/PACUTE extension to Vietnamese | Novelty sweep now (`docs/NOVELTY_SWEEP_2026-09-30.md`), at Gate 1 from an unblocked machine, and mid-December; Scholar alerts set; work under three months old counts as contemporaneous; anonymized public version dates the work | author |
| Weak or uninteresting results | Medium | Medium | Pilot: no model ≥ 15 points over the copy baseline; E3 effects within CI of 0 everywhere | Pilot before scaling; negative results protected if the construct is sound; fall back to a focused short paper | author |
| Time crunch (13 weeks, 4 crunch weeks) | High | Medium–high | Gate 3 (22 Nov) results freeze slips | Content-complete by 18 Dec; cut the analysis layer before validation; prefer a later cycle to a weak paper | author |
| Ethics flag (no review board) | Low–medium | Medium | Reviewer raises D4 | Consent forms, justified unpaid status, reasoned D4 "No", no personal data, adult contact named beside the minor author (DD §11.7) | author, co-author's IRB if any |
| Registration due before waiver decisions | Medium | Paper dropped from proceedings | Fee pages live without a student definition covering high school | Budget ≈ $350; ask chairs early; funded co-author presents | author |
| SRW fallback restricts AI-written code | Low–medium | Medium | ACL 2027 SRW call copies EACL 2026's "refinement tasks" wording | Read the call when published; the AI-use log (DD §11.6) documents scope; be ready to state author verification per module | author |

## B. Construct validity and reviewer objections (new)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| Paper reads as "EXECUTE-vi" / a wordplay dataset plus a probe that finds what the input contains (ARR G1, R4) | High if the T1 leaderboard leads | High: Findings at best | Draft intro leads with Table 2 | Lead with the tokenization question and the re-encoding intervention; nói lái is the instrument; explicit onset–rime–tone input and T3 in the main results; cite PACUTE, Phun-Bench, EXECUTE prominently (DD §1, §12.17, §12.27) | author |
| Generated items measure formal permutation, not nói lái (Bean et al. construct validity) | Certain as a fact; medium as an objection | Medium | 14.5% of lexicon T1 outputs and 1.1% of V3 outputs are dictionary entries (v0.1) | State it: outputs are mostly pseudo-phrases by design; `output_lexical` stratum in every regression; attested set scored separately; H6 compares only pure rule outputs (DD §4.7, §8.7) | author |
| Attested "gold" is the rule's answer, not the folk answer (5 of 18 two-syllable seed rows) | Certain today | High for H6 | `rule_matches_attested = false` rows scored as rule outputs | Folk form is gold; `exact`/`approx.(merger)` tags; approx rows only in T2; attested inputs reserved out of the generated pool (DD §4.7, §12.16) | author, validators |
| Four-variant taxonomy vs the six-type tradition; order-sensitive T1 | High | Medium | A Vietnamese reviewer names `đay thổi`-type answers scored wrong | Six types documented with the group structure; strict + lenient T1; T2 classes unordered; V5 as attested label (DD §3) | implementer, author |
| Degenerate cells (equal tones → V2 = reversal, V1 ≡ V4; duplicate T2 readings; 86 spelling twins vs ~480 others) | Certain in v0.1 | Medium | `same_tone` rate 30% in V1; T3 spelling column with one-fifth the power | Drop reversal outputs, merge duplicates with `variant_labels`, de-duplicate T2 gold, report spelling twins separately (DD §3.4, §5.3) | implementer |
| T2 lexicality by Viet74K undercounts real phrases (`tiền đâu`, `trời cho`, `bị mất` absent) | Certain | Medium–high | Valid native readings scored wrong | Native-validated gold for core and attested items; `plausible_nongold` bin; corpus frequency (DD §5.2, §12.12) | validators, implementer |
| Dialect construct validity (hỏi/ngã, -n/-ng, d/gi mergers; V3 items inaudible for Southern speakers) | High | Medium | Southern validators reject V3 items Northern validators accept | Region-tagged validators and baseline; report by region and twin type; regional labels in Limitations (DD §3.5, §10) | author, validators |
| Generation vs recognition confounded in T1 (rare/non-word outputs need byte-fallback spelling) | Medium–high | Medium | T1 failures concentrated on outputs that are multi-token in the audit | T3 (no generation), explicit-input condition, per-item output-token covariate in E2 (DD §7.4, §8.4) | implementer |
| Regional labels of variants contradict across sources | Certain | Low–medium | — | Codes region-neutral; "which variant do you produce?" validator item (DD §3.5) | author |
| "First" claims overreach (Pham & Pham 2018; KoWit-24 spoonerism type) | Certain if unqualified | Medium (G5, H3 objection) | — | Narrowed claims (DD §12.27); cite both | author |
| STAD metric described as adopted while it is defined over English syllabification | Medium | Low–medium | — | Call it an adaptation and validate against the native-checked parser (DD §12.27) | author |
| Human baseline not comparable to models (different prompt, tiny n, single-rated) | High as planned | Medium | Form drafted with a different explanation than p0 | 246-item double-coverage design on model items, same p0 prompt and demos, mean-human comparator, ±4–5 pt band stated (DD §10.2) | author |
| Krippendorff's α looks low at 95%+ prevalence | High | Low–medium | Validation α < 0.8 with raw agreement > 95% | Report raw agreement and Gwet's AC1 beside α (DD §10.1) | implementer |

## C. Interventions and statistics (new)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| C1 is engine-dependent: HF `tokenizers` Precompiled normalizer deletes NFD tone marks (issue #2334) while native SentencePiece normalizes; llama.cpp/ollama ignore tokenizer.json normalization (arXiv 2609.20614) → C1 becomes C3 in disguise or flips sign | High for any `nmt_nfkc` model (Vistral?) | High: attributes a stripping effect to encoding | Native-vs-HF token-ID assertion fails; decoded IDs ≠ input | Three-valued census per (tokenizer, engine) as a pre-flight gate; effects attributed to the string actually received; C1 on two engines for a subset (DD §6.2) | implementer |
| 7 of 17 open models normalize to NFC → C1 = 0 by construction | Certain (Qwen lineage) | Medium (halves the informative set) | Census shows identical IDs | Skip the arm, emit a "0 by construction" row; the "normalizes" column is a first-class result (DD §6.2, §7.2) | implementer |
| C2 near-null on natural text (0.5% of XCOPA syllables; 43/500 items) and baseline = the rare form | Certain | High for H4 | XCOPA C2 discordance ≪ 20% | Old style as baseline; affected-items-only analysis; C2-enriched NóiLái set; H4 directional (DD §6.3, §12.5–6) | implementer, author (corpus count) |
| Mediation share undefined (deterministic mediator) | Certain | High if kept (R3 objection) | Fig. 4 titled "share mediated" | Replaced by ATE, dose–response (descriptive), three-arm contrast, tone-isolation DiD, 2×2 (DD §8.6) | implementer |
| Paraphrases counted as independent trials; McNemar on clustered items; base-pair-only SEs for cross-model slopes (7× understatement in simulation); VB GLMM SDs reported | Medium (easy to do by accident) | High (R5) | CI half-widths shrink by √3; slope SE < 0.05 | Aggregate paraphrases; cluster by base pair; two-way clustered SEs or random slopes for H1; VB point estimates only; `tests/test_stats.py` clustered simulation (DD §8) | implementer |
| Power overstated (94% is the best case; 125-item API cells resolve 11–17 points) | Certain as arithmetic | Medium | Pilot discordance > 35% | Sensitivity table in the paper; 350 items per cell; pilot fixes n before test runs (DD §8.5) | author |
| H4 "small nonzero shifts in every model" unfalsifiable / 21-way conjunction | Certain | Medium | — | Directional H4, k of N + pooled, TOST (DD §1) | author |
| Frequency covariate from Viet74K is dictionary productivity, confounded with BPE vocabulary | High | Medium for H1 | — | Independent reference corpus; Viet74K secondary (DD §8.4) | author (corpus), implementer |
| H6 conflates memorization with difficulty/salience | Certain | Medium | Attested > generated even after matching | Guided-instruction, recall, log-prob contrast and name-only tests carry the inference (DD §8.7) | implementer |
| Deterministic scoring but nondeterministic engines (vLLM batch composition, fp16 drift) | Medium | Low–medium | 200-item rerun disagrees > threshold | Reported reproducibility check, not optional; thresholds pre-registered (DD §8.8) | implementer |
| The `qu` convention gives different swap outputs under two analyses | Certain | Low (excluded) / medium if included wrongly | Native check disagrees with the parser | qu- inputs excluded pending the 20-pair check (DD §2.1 O5) | author, validators |

## D. Probing and patching (new)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| Tone probe trivial: tone is in the embedding at layer 0 for single-token syllables (≈ 30% of NFC slots) or is its own token under NFD (42%) | Certain if unstratified | High (R2/R3) | Layer-0 accuracy already high | Token-type stratification; layer-0 baseline; structural baseline (token IDs + coda); non-word stimuli; `after` position as the fair comparison (DD §9.1–9.2) | implementer |
| Control task vacuous under a disjoint-syllable split; real baselines are 25.4% majority and ~50% on stop codas | Certain | Medium | Selectivity ≈ accuracy − 1/6 | Two-condition definition; majority and coda-conditional baselines (DD §9.1) | implementer |
| Patching pairs misaligned (only 55% align in NFC; a third confounded by segmentation; NFD mark order depends on tone) | Certain | Medium | Retention < 200 pairs | ≥ 600 candidates per orientation; tone-only readout filter; plain nuclei for NFD; retention reported (DD §9.3) | implementer |
| Perception and manipulation not separated | High as planned | High for H5 | Only one readout implemented | Two readouts × two source groups; copy-type test with a second pair family (DD §9.3) | implementer |
| Gemma 3 1B near chance on V3 → LD is noise | Medium | Medium | Filter (3) removes most pairs | Report clean-run accuracy first; treat an undefined manipulation readout as a result | implementer |
| Existing E4 code defects (answer tokenization, no NFD path, steering off-by-one, dead branch) | Certain | Medium | — | Fix list DD §9.6 with tests | implementer |
| bf16 on T4 emulated/unsupported; fp32 4B does not fit one T4 | High | Medium | cuBLAS error or OOM | `Gemma3ForCausalLM` over 2×T4 in fp32; bf16 only as smoke-tested fallback; never fp16 (DD §9.5) | implementer |

## E. Engineering and compute (new)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| Gemma 3 in fp16 (vLLM raises; transformers produces inf/NaN silently) | Certain if attempted | High | `ValueError` or garbage outputs | `dtype="auto"` (fp32); nan/inf logits guard on the first batch of every run (DD §7.2) | implementer |
| PhoGPT not runnable on vLLM 0.30 (MPT removed in 0.28); remote code may fail under transformers 5 | Certain | Medium | Import error | HF transformers with a pinned 4.x venv or llama.cpp GGUF; corrected engine table (DD §7.2) | implementer |
| Qwen3.5 GDN kernels in fp16 on sm_75 unverified; Gemma 4 E2B kernel patch report | Medium | Medium | Smoke test fails or NaN | Smoke test every size; GGUF/TPU fallback; record engine per model | implementer |
| Uncertain HF IDs (Gemma 4 12B may not exist; Qwen3.5 suffixes; SEA-LION IDs; Gemini IDs) | Medium | Medium | 404 on gated access request | Verify before the Gate 1 freeze; names stay, IDs change (DD §7.1) | author |
| Tokenizer-fixed control pairs invalid (Qwen-SEA-LION-27B is a Qwen3.6 fine-tune; SEA-LION E2B tokenizer unverified) | Medium | Medium | SHA-256 mismatch of tokenizer files | Add Qwen3.6-27B or drop the claim; hash identity required (DD §7.2) | author, implementer |
| HF fast tokenizer ≠ the audited SentencePiece file | Low–medium | High for E2 covariates | `check_tokenizer_consistency.py` finds different IDs | Run the check before joining any model result to audit statistics | implementer |
| API quotas far below plan (Gemini Flash ≈ 20 RPD?, Groq 200K TPD, OpenRouter 50 RPD) | High | Medium | Live quota page | Read quotas live; resumable multi-day cache; Flash-Lite or drop Gemini (DD §7.3, §13.10) | author |
| Thinking not actually off (Qwen3.8 `reasoning_effort`, Gemma 4 `<|channel>thought`) | Medium | Medium | `n_thinking_chars > 0` | Template census; strip and count at run time (DD §7.3) | implementer |
| Chat templates alter prompts (Llama 3.1 date; Gemma trim; Qwen think blocks) | Certain | Low–medium | Prompt hash differs across days/engines | Render in the runner; no system prompt; pin `date_string`; hash test (DD §7.4) | implementer |
| Kaggle output lost at the 12-h wall; 20 GB working dir | High | Medium | Session killed before version completes | 500-item chunks with atomic renames; ≤ 9 h work per session; mid-run push; caches in scratch (DD §7.6) | implementer |
| Version drift (vLLM 0.30 pins torch 2.13 vs Kaggle torch 2.10) | Certain | Medium | pip resolver replaces the kernel torch | Venv under `/kaggle/tmp` or `kaggle-vllm`; lock file; `pip freeze` in every manifest | implementer |
| Byte-level tokenizers split combining marks into bytes → offsets inside code points break `audit_syllable` | High for Qwen/Llama/PhoGPT | Medium for E2 | Zero-width/overlapping offset spans | Byte-aware `tone_isolated`; test `_nfd_to_nfc_offset` with a byte-level tokenizer | implementer |
| Prefix property fails for prompt+candidate log-likelihood | Medium | Medium | Assertion fails | Assert; start candidates after a newline (DD §7.3) | implementer |
| Project environment lacks pytest and every optional dependency; `noilai/stats/` and eval changes uncommitted | Certain today | Medium | `pytest` not found; `git status` dirty | `requirements.txt` with statsmodels/scipy/pymc-bambi/krippendorff/nltk/pytest; commit before the next gate (author commits, agents never) | author |
| Compute budget exceeded by the raised 4,200-item sample and the two-engine C1 check | Medium | Medium | Pilot throughput below 150 output tok/s on 4-bit 8B | Pool variants for cell claims; cut models at Gate 3, never dates | author |

## F. Data, licensing, ethics (new)

| Risk | Likelihood | Impact | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|
| GPL word lists: released lexical items and audit CSVs may be derivative works; CC BY / CC BY-NC-ND relicensing incompatible with GPLv2 | Medium (legal reading uncertain) | High (T2 objection; takedown) | No permission reply by Gate 2 | Permission emails now; GPLv2 fallback; pseudo-only last resort; never publish `data/cache`, `data/external`, per-syllable CSVs (DD §4.1, §11.1) | author |
| Hunspell licence ambiguity (no root LICENSE; GPLv2 dic vs GPLv3+ packaging) | Certain | Low–medium | — | Cite as GPLv2, pinned by hash; ask the maintainer | author |
| NC-ND test licence contradicts the reuse story while the Apache generator regenerates the split | Certain | Low–medium | Reviewer asks why ND | State that ND protects the file, not the items; publish only the dev seed; "Enabling" lives in the generator (DD §4.6, §11.1) | author |
| Vulgar outputs minted from innocent inputs; only one manual flag exists today | Certain | Medium (B4) | Validators flag items the blocklist missed | Blocklist at generation for all items; validator feedback; gated-only for Hồ Xuân Hương rows (DD §11.5) | implementer, validators |
| Gemini free tier trains on inputs; all APIs require 18+ | Certain | Medium | — | Adult account holder recorded by role; core treated as potentially contaminated; sealed set; ZDR settings (DD §11.2) | author |
| Personal data of validators/participants (Vietnam PDPL 91/2025 in force since 1 Jan 2026 [scope UNCERTAIN]) | Low | Medium | Google Form collecting emails | Letters only; email collection off; key held separately and deleted; coarse bands (DD §10.1) | author |
| Referenced documents do not exist (`CONSENT_FORM.md`, `VALIDATOR_INSTRUCTIONS.md`, `DATA_STATEMENT.md`, `AI_USE_LOG.md`, `README` licence/NOTICE, `compute_log.csv`) | Certain today | Medium (validators recruited without consent text) | 19 Oct arrives without them | Write before 19 October (DD §10.1, §11.6) | author |
| Hallucinated or misattributed citations (plan numbers unchecked; 2609.21362 authors unknown; preprints with archival versions) | Medium | Severe after acceptance | Any `[UNCERTAIN]` left in the bib | Hand-check every entry from an unblocked machine; use archival versions (`docs/RELATED_WORK_VERIFICATION.md`) | author |
| Canary derivable from the build seed; plain-text test file crawlable | Certain as designed | Medium | — | GUID from `secrets`; BIG-bench header; encryption (DD §4.6) | implementer |
