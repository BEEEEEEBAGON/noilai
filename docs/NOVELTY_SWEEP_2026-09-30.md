# Novelty sweep — Bundle A (30 September 2026)

Edited from the novelty lens report. Access from the build machine: WebFetch is egress-blocked for aclanthology.org, semanticscholar.org, arxiv.org, openreview.net and every paper-mirror domain tried (mcml.ai, pith.science, alphaxiv.org, themoonlight.io, lacuna.tiptreesystems.com, en.papernotes.org, awesomepapers.io, bytez.com, paperswithcode.co); github.com and raw.githubusercontent.com fetch; `api.github.com` is refused. **Every paper-content claim below rests on WebSearch snippets or on GitHub READMEs and must be re-verified from an unblocked machine at Gate 1 (18 October 2026) and again in the week of 14 December 2026.** Cross-references: the bibliographer's verification of the plan's own citations is in `docs/RELATED_WORK_VERIFICATION.md`; framing consequences are decisions in `docs/DESIGN_DECISIONS.md` §12.27.

## 1. Queries run (73 WebSearch queries, grouped by round; hits in brief)

1. `"nói lái" LLM` → generic Vietnamese "LLM là gì" explainers; no research hit.
2. `Vietnamese spoonerism language model` → **Pham & Pham, "Building a Spoonerism Detection System for Vietnamese", PACLIC 32 (Dec 2018), aclanthology Y18-1063**; otherwise PhoBERT/BamiBERT/VinaLLaMA.
3. `"spoonerism" LLM benchmark` → generic benchmark listicles.
4. `tokenization NFC NFD LLM Unicode normalization` → HF normalizers docs; TokSuite (arXiv 2512.20757); "Inference-Engine Fingerprinting Attacks" (arXiv 2609.20614: HF-based engines apply tokenizer.json normalization, llama.cpp/ollama do not).
5. `"combining diacritics" tokenizer evaluation language model` → "Tokenizers Introduce Unfairness Between Languages" (2305.15425); TokEval (2608.18062); TokSuite; AraToken.
6. `Vietnamese tone mark placement NLP old style new style` → typography/learner pages, Wikidata Q10810166; FreeTxt-Vi (2603.05690); no LLM study.
7. `Vietnamese tokenization syllable LLM 2026` → Beyond Atomic Tokens (2609.21362); VialectBench (2608.10414); STAD repo liaodisen/Tokenization-Phonology.
8. `"EXECUTE" benchmark character understanding multilingual languages LLM` → EXECUTE (Findings ACL 2025; Edman, Schmid, Fraser).
9. `"CUTE" benchmark tokens characters LLM` → CUTE (EMNLP 2024); StochasTok (2506.01687).
10. `"Spelling-out is not straightforward" LLM` → Hiraoka & Inui (Findings EMNLP 2025; arXiv 2506.10641).
11. `STAD syllabification tokenization ACL 2026 phonological knowledge` → Liao & Shi, ACL 2026 long 634 (arXiv 2604.17105); "Phonemes to the Rescue" (ACL 2026 long 1872).
12. `"Beyond Atomic Tokens" Vietnamese PhonemicBERT` → confirmed: onset/rime/tone factorization via IPA; 256-entry Vietnamese vocabulary; PhonemicBERT-Vi/Zh; submitted 18 Sep 2026.
13. `"Equity with Efficiency" tokenizer Southeast Asian languages` → confirmed (2606.15044); 11 SEA languages; also SEA-LION v4.8 report (2609.18310).
14. `phonological knowledge LLM tokenization limits` → Liao & Shi; ICML 2025 workshop precursor (icml.cc/virtual/2025/47753).
15. `wordplay benchmark LLM pun understanding` → "Pun Unintended" (EMNLP 2025); PunGraph (2609.16557); audio puns (2603.18678); pun-generation survey (2507.04793). No Vietnamese.
16. `Vietnamese orthography perturbation robustness LLM diacritics` → VialectBench; Crossing Linguistic Horizons (Findings NAACL 2024); **Korean jamo-level typographical vulnerabilities (2608.30229, EMNLP 2026)**; proxy-model robustness (2506.07645).
17. `arXiv 2609.21362` → authors (Nghia Hieu Nguyen … Kiet Van Nguyen, Ngan Luu-Thuy Nguyen; UIT group), "under review" [UNCERTAIN: the bibliographer could not capture the author list from any reachable source].
18. `"nói lái" mô hình ngôn ngữ lớn ChatGPT` (VI) → nothing.
19. `Arabic diacritics tokenization LLM re-encoding causal EACL 2026` → Inoue, Alhafni, Habash, Baldwin, "Do Diacritics Matter?", Findings EACL 2026 (findings-eacl.22).
20. `Japanese tokenization inconsistency LLM ACL 2025 short` → Gambardella et al., ACL 2025 short 75.
21. `"Building a Spoonerism Detection System for Vietnamese" PACLIC 2018 Pham` → rules + 3-gram LM, F1 95.47% on poems/folk songs.
22. `Vietnamese spoonerism detection nói lái dataset GitHub` → no public repo found.
23. `EXECUTE … languages list …` → 8 primary: Amharic, Arabic, Chinese, English, Hindi, Japanese, Korean, Russian; no Vietnamese.
24. `"How Tokenization Limits Phonological Knowledge" … languages …` → abstract only; no language list in snippets.
25. `Korean jamo-level typographical vulnerabilities LLM sub-syllable` → Lee & Lee (Chung-Ang), EMNLP 2026: 5 jamo perturbations on KMMLU, 4 models, scaling does not help, TACoT.
26. `LLM spoonerism generation GPT-4 phonological manipulation "spoonerisms"` → KoWit-24 (spoonerism as a wordplay type); Beguš et al. "Large Linguistic Models"; blogs. No transformation study.
27. `"chơi chữ" OR "nói lái" Vietnamese wordplay dataset NLP` → VIVID idioms/proverbs benchmark (2608.03095); UIT dataset list; no wordplay dataset.
28. `Unicode normalization attack LLM robustness NFKC homoglyph …` → security blogs on token smuggling; TokSuite; not an evaluation study.
29. `Mandarin tone probing LLM text linear probe …` → speech-model tone probing only (2403.16865 covers Mandarin + Vietnamese speech SSL; 2604.07467).
30. `"onset" "rime" LLM syllable structure benchmark 2026` → PhonologyBench (2024) only.
31. `"TokSuite" tokenizer choice … Unicode diacritics` → 14 identical models differing only in tokenizer + ~5k-item benchmark; Turkish/Farsi diacritics, Unicode styling; XGLM NFKC vs Llama-3.2 no normalization.
32. `Vietnamese tone LLM evaluation "tone marks" removal benchmark …` → VMLU, ViLLM; nothing tone-specific.
33. `"nói lái" trí tuệ nhân tạo AI bài báo nghiên cứu` (VI) → nothing.
34. `rhyme benchmark LLM phonology "rhyming" character-level Pig Latin` → **Phun-Bench (ACL 2026 long 1041; arXiv 2606.07300)**; Greek rhyme (LaTeCH 2026); PhonologyBench.
35. `Thai syllable tokenization LLM sub-syllable diacritics 2026` → FastThaiG2P; "Type-Driven Tokenization for Brahmic Scripts" (2609.22125); nothing Thai-LLM-sub-syllable.
36. `"Phun-Bench" …` → homophony, rhyme, phonetic similarity; repo xing-stellus-yue/Phun-Bench.
37. `"TokSuite" Vietnamese OR NFC OR NFD …` → target languages Chinese, English, Farsi, Italian, Turkish; no Vietnamese.
38. `"Phonemes to the Rescue" …` → Miletić, Kallini, Shutova; IPA tokenizers, 24 languages, 14 scripts.
39. `"Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models" …` → Polish-centric [UNCERTAIN]; missing-diacritics perturbation.
40. `"Crossing Linguistic Horizons" … diacritics removal …` → four perturbations incl. Vietnamese-diacritic removal.
41. `ChatGPT "nói lái" thử nghiệm …` (VI) → nothing.
42. `"nói lái" học máy OR "xử lý ngôn ngữ tự nhiên" …` (VI) → nothing.
43. `Vietnamese Unicode normalization NFC NFD tokenizer PhoBERT …` → **VietNormalizer (arXiv 2603.04145, Mar 2026)**; underthesea does the same.
44. `"canonical equivalence" OR NFD … LLM evaluation 2026` → **Ghosh & Jyothi, "LMs are not Equally Robust to Non-Canonical Tokenization across Languages" (arXiv 2607.26831, 29 Jul 2026)**; a "canonical-equivalence" fact-checker paper (2607.16212) probably about logical forms [UNCERTAIN].
45. `character-level manipulation benchmark LLM 2026 …` → SubTokenTest (ACL 2026 long 915); CharBench (2508.02591); EXECUTE extras Tamazight, Santali.
46. `"Causal Estimation of Tokenisation Bias" ACL 2025 method` → Lesci et al.; regression discontinuity; trains own models.
47. `KoWit-24 wordplay dataset spoonerism …` → RANLP 2025; 2,700 Russian headlines; spoonerism among types; 5 LLMs.
48. `"Type-Driven Tokenization for Brahmic Scripts" …` → TyDe 2026 workshop; mentions Vietnamese NFD as base + ≤1 quality + ≤1 tone mark.
49. `Vietnamese tone probing text language model …` → speech only.
50. `"Beyond Atomic Tokens" PhonemicBERT evaluation tasks …` → 14 targets, best mean on 9/14; homophone-substitution robustness; baselines PhoBERT, WikiBERT, mBERT, DistilBERT, XLM-R.
51. `"How Tokenization Limits Phonological Knowledge" English CMU …` → CMU Pronouncing Dictionary / ARPAbet → English.
52. `"Language Models are not Equally Robust to Non-Canonical Tokenization" languages Vietnamese` → 27 languages, FLORES-200, 6 tasks; Llama-3.1-8B −23.7%, Qwen3-8B −11.4%, Gemma-3-12B −9.9%; Vietnamese not named.
53. `HKCanto-Eval …` → CoNLL 2025; 100 phonology questions.
54. `Vietnamese poetry LLM tone rules "bằng trắc" …` → lục-bát generation with rule validators (2401.01078).
55. `Vietnamese diacritic restoration LLM …` → Romanian LLM study (2511.13182); Vietnamese restoration is pre-LLM.
56. `Vietnamese tokenizer fertility analysis …` → TokLens (ACL 2026 SRW 18); STRR (2510.09947); Equity with Efficiency.
57. `spoonerism "language models" phonological awareness …` → PhonologyBench; KoBALT; "Emergence of a phonological bias in ChatGPT" (2305.15929).
58. `Vietnamese LLM interpretability activation patching …` → nothing.
59. `"SUBTOKENTEST" …` → 10 tasks, 4 domains; reasoning models mitigate at high token cost; English-centric [UNCERTAIN].
60. `Vietnamese character-level understanding LLM … CUTE-style …` → **PACUTE (arXiv 2606.15144, June 2026): Filipino CUTE extension**.
61. `"VietNormalizer" …` → confirmed: NFC + tone-mark-position correction; PyPI.
62. `"PACUTE" Filipino …` → 4,600 tasks; infixation, reduplication, syllabification; repo raileymontalan/pacute-bench.
63. `Ghosh "non-canonical tokenization" 27 languages …` → Indic-heavy list visible; Vietnamese unconfirmed.
64. `"TokLens" … 15 languages Vietnamese` → venue confirmed; language list not in snippets.
65. `"Phun-Bench" tone perturbed idioms …` → pinyin initial + final + tone; repo has `idioms_tone_*_perturbed.json`.
66. `Vietnamese spoonerism "nói lái" … 2023–2026 deep learning` → only "Understanding Tieq Viet with Deep Learning Models" (2207.00975).
67. `LLM sensitivity orthographic variants … British American spelling …` → "Tokenization is Sensitive to Language Variation" (Findings ACL 2025); "Which English Do LLMs Prefer?" (2604.04204); (2303.03457).
68. `"hoà" "hòa" tokenizer OR "language model" …` → nothing evaluates placement variants.
69. `Vietnamese LLM arXiv September 2026 tokenization OR diacritics …` → Beyond Atomic Tokens; "To Each Language Its Tokenizer" (2609.15528); Brahmic paper.
70. `"YOMI-Bench" OR "KoBALT" …` → YOMI-Bench (2607.00664); KoBALT phonology 31%.
71. `"TokSuite" perturbation categories …` → Input, Diacritics, Orth&Gram, Morph, Noise, LaTeX, STEM, Unicode styling, capitalization. No NFC/NFD arm named.
72. `"Understanding Tieq Viet with Deep Learning Models" …` → Nguyen Ha Thanh (NII Tokyo), July 2022.
73. `Thai-Hoang Pham spoonerism … dataset released code` → ~300 sentences from Hồ Xuân Hương poetry, folk tales, songs [UNCERTAIN]; no repository found.

Added by other lenses (not queried by the novelty lens; verify at Gate 1): **UGTPhon** (arXiv 2609.27205, 24 Sep 2026), "the first grapheme-to-phoneme benchmark for user-generated text in English, Vietnamese, and Korean" (bibliographer); Lê Trung Hoa & Hồ Lê, *Thú chơi chữ* (NXB Trẻ, 1990), six nói lái types (reviewer; [UNCERTAIN]); Nguyễn Văn Hiệp, "Nói lái trong ngôn ngữ và văn học Việt Nam" (linguist; snippet only).

## 2. Closest prior work

| Work | Venue / date | What it does | Overlap with Bundle A | URL |
|---|---|---|---|---|
| Pham & Pham, "Building a Spoonerism Detection System for Vietnamese" | PACLIC 32, Dec 2018 | Rules + 3-gram LM detect nói lái in ~300 sentences from Hồ Xuân Hương, folk tales and songs; F1 95.47% [size UNCERTAIN]; 9 primary + 8 dialect-exception rules | **partial — the only prior computational nói lái work**; detection only, pre-LLM, no generation/decoding/validity, no tokenization angle | https://aclanthology.org/Y18-1063/ |
| PACUTE (Layacan, Flores, Africa, Montalan et al.) | arXiv 2606.15144, Jun 2026 | 4,600-task CUTE-style diagnostic for Filipino incl. syllabification, infixation, reduplication; open models near chance on morpheme decomposition | **partial, closest SEA analog**: same access-vs-use logic; no tone, no wordplay, no re-encoding, no probing | https://arxiv.org/abs/2606.15144 ; https://github.com/raileymontalan/pacute-bench |
| EXECUTE (Edman, Schmid, Fraser) | Findings ACL 2025 | CUTE extended to Amharic, Arabic, Chinese, English, Hindi, Japanese, Korean, Russian (+Tamazight, Santali, cipher/byte controls); sub-character tasks | partial (methodological template; no Vietnamese) | https://aclanthology.org/2025.findings-acl.95/ |
| CUTE (Edman et al.) | EMNLP 2024 | English (+Russian) spelling/containment/manipulation tasks | partial | https://aclanthology.org/2024.emnlp-main.177/ |
| Liao & Shi, "How Tokenization Limits Phonological Knowledge Representation in LMs and How to Improve Them" | ACL 2026 long 634 (arXiv 2604.17105; ICML 2025 workshop precursor) | English (CMU/ARPAbet) probing of rhyme and syllabification in Llama-3.1-8B; STAD metric; IPA fine-tuning | partial (STAD borrowed and **adapted**; English only [UNCERTAIN: appendix]) | https://aclanthology.org/2026.acl-long.634/ ; https://github.com/liaodisen/Tokenization-Phonology |
| Hiraoka & Inui, "Spelling-out is not Straightforward" | Findings EMNLP 2025 | Layer-wise reconstruction of character identity inside tokens; breakthrough layer | partial (probe template for E4) | https://aclanthology.org/2025.findings-emnlp.719/ |
| Phun-Bench | ACL 2026 long 1041 (arXiv 2606.07300) | Chinese homophony, rhyme, phonetic similarity; pinyin initial/final/tone; tone-perturbed idiom sets in repo; LLMs recall pronunciations but cannot use them | partial (tonal-language phonology; no transformation task, no tokenization intervention; script hides phonology) | https://aclanthology.org/2026.acl-long.1041/ ; https://github.com/xing-stellus-yue/Phun-Bench |
| TokSuite | arXiv 2512.20757, Dec 2025 | 14 controlled models differing only in tokenizer; perturbations incl. optional diacritics (Farsi, Italian), homoglyphs, Unicode styling; Chinese/English/Farsi/Italian/Turkish | partial (causal tokenizer study with diacritic/Unicode arms; no Vietnamese; no NFC/NFD arm found [UNCERTAIN]) | https://arxiv.org/abs/2512.20757 |
| Ghosh & Jyothi, "LMs are not Equally Robust to Non-Canonical Tokenization across Languages" | arXiv 2607.26831, 29 Jul 2026 | 27 languages, FLORES-200, 6 tasks, Llama-3.1-8B / Qwen3-8B / Gemma-3-12B; alternative segmentations of the same string; drops 9.9–23.7% | partial (changes tokens with code points fixed; A changes code points with meaning fixed; same model families; Vietnamese [UNCERTAIN]) | https://arxiv.org/abs/2607.26831 |
| Inoue, Alhafni, Habash, Baldwin, "Do Diacritics Matter?" | Findings EACL 2026 | Adds/removes optional Arabic diacritics; full diacritization → fragmentation and degraded scores | partial (closest re-encoding design; Arabic diacritics add information, NFC/NFD does not) | https://aclanthology.org/2026.findings-eacl.22/ |
| Lee & Lee, "Quantifying and Mitigating Korean Jamo-Level Typographical Vulnerabilities in LLMs" | EMNLP 2026 (arXiv 2608.30229) | Five intra-syllable jamo perturbations on KMMLU; monotone drop; scale does not help | partial (sub-syllabic, meaning-destroying, like C3) | https://arxiv.org/abs/2608.30229 |
| Nguyen Ha Thanh, "Understanding Tieq Viet with Deep Learning Models" | arXiv 2207.00975, Jul 2022 | Seq2seq recovers standard Vietnamese from Bùi Hiền's reformed spelling | partial (lossy Vietnamese re-encoding, pre-LLM) | https://arxiv.org/abs/2207.00975 |
| Truong et al., "Crossing Linguistic Horizons" | Findings NAACL 2024 | Vietnamese LLM eval; diacritic removal as one perturbation | partial (C3 analog) | https://aclanthology.org/2024.findings-naacl.182/ |
| Gambardella et al., "Inconsistent Tokenizations Cause LMs to be Perplexed by Japanese Grammar" | ACL 2025 short 75 | Same surface form tokenized inconsistently hurts grammaticality | partial | https://aclanthology.org/2025.acl-short.75/ |
| Lesci et al., "Causal Estimation of Tokenisation Bias" | ACL 2025 long 1374 | Regression discontinuity at vocabulary cutoff; trains 57M–850M models | partial (causal framing; different intervention) | https://aclanthology.org/2025.acl-long.1374/ |
| "Tokenization is Sensitive to Language Variation" | Findings ACL 2025 | Spelling variants tokenize differently; task-dependent | partial (English analog of meaning-preserving variation) | https://aclanthology.org/2025.findings-acl.572/ |
| "Which English Do LLMs Prefer?" | arXiv 2604.04204, Apr 2026 | American spellings tokenize more compactly (2.85–5.42% gap) | partial | https://arxiv.org/abs/2604.04204 |
| Beyond Atomic Tokens: Factorizing Syllables for LM Pretraining (UIT group) | arXiv 2609.21362, 18 Sep 2026 | IPA-based onset/rime/tone Phonemic Tokenizer (256 Vietnamese entries); PhonemicBERT-Vi/Zh; homophone-substitution robustness | partial (same decomposition; builds new encoders; our explicit-input condition is its natural baseline) | https://arxiv.org/abs/2609.21362 |
| Equity with Efficiency (SEA tokenizers) | arXiv 2606.15044, Jun 2026 | Parity-aware BPE vs MYTE on 11 SEA languages; 1.5B models | partial (fertility numbers; trains models) | https://arxiv.org/abs/2606.15044 |
| Miletić, Kallini, Shutova, "Phonemes to the Rescue" | ACL 2026 long 1872 | IPA subword tokenizers across 24 languages, 14 scripts | partial/none | https://aclanthology.org/2026.acl-long.1872/ |
| KoWit-24 | RANLP 2025 (arXiv 2503.01510) | 2,700 Russian headlines with wordplay types incl. spoonerism; GPT-4o + 4 others on detection/interpretation | partial (only LLM evaluation touching spoonerisms; detection, not transformation) | https://aclanthology.org/2025.ranlp-1.15/ |
| Pun Unintended | EMNLP 2025 main 1419 | English pun detection collapses on PunBreak | none/partial | https://aclanthology.org/2025.emnlp-main.1419/ |
| PhonologyBench | KnowLLM @ ACL 2024 | English G2P, syllable counting, rhyme generation | partial | https://aclanthology.org/2024.knowllm-1.1/ |
| SubTokenTest | ACL 2026 long 915 (arXiv 2601.09089) | Ten sub-token tasks; reasoning models fix errors at token cost | partial (reasoning-cost precedent; English-centric [UNCERTAIN]) | https://aclanthology.org/2026.acl-long.915/ |
| HKCanto-Eval | CoNLL 2025 | Cantonese jyutping G2P, homophone and rhyme questions | partial/none | https://aclanthology.org/2025.conll-1.1/ |
| KoBALT | arXiv 2505.16125 | Korean linguistic benchmark; phonology 31% | none/partial | https://arxiv.org/abs/2505.16125 |
| VietNormalizer | arXiv 2603.04145, Mar 2026 | Vietnamese normalization library: NFC + tone-mark-position correction | tooling overlap; use as an independent check of the C2 rules | https://arxiv.org/abs/2603.04145 |
| Type-Driven Tokenization for Brahmic Scripts | TyDe 2026 (arXiv 2609.22125) | Formalizes valid token boundaries in decomposed scripts; notes Vietnamese NFD structure | formal support for why NFD tokens split a syllable's tone mark | https://arxiv.org/abs/2609.22125 |
| Inference-Engine Fingerprinting Attacks | arXiv 2609.20614, Sep 2026 | HF-tokenizer engines honor tokenizer.json normalization; llama.cpp/ollama do not | **validity threat to C1** (normalization is engine-dependent) | https://arxiv.org/abs/2609.20614 |
| Speech-SSL tone probing (Mandarin + Vietnamese) | arXiv 2403.16865 / NAACL 2024 | Linear tone probes on speech-model layers; BERT text baseline | partial (probe design; speech) | https://arxiv.org/abs/2403.16865 |
| UGTPhon (added by the bibliographer) | arXiv 2609.27205, 24 Sep 2026 | G2P benchmark for user-generated text in English, Vietnamese, Korean [title from an RSS mirror; authors not captured] | low–medium (Vietnamese phonology in LLMs; no nói lái, no re-encoding) | https://arxiv.org/abs/2609.27205 |

## 3. Verdict on each gap claim

**(a) "No LLM study of nói lái or Vietnamese spoonerism" — SUPPORTED as stated; the broader "unstudied in NLP" framing is REFUTED.** Ten English- and Vietnamese-language queries returned no LLM evaluation, generation or probing study of nói lái. But Pham & Pham (PACLIC 2018, Y18-1063) built a rule + 3-gram nói lái detector on ~300 attested sentences (F1 95.47%), and KoWit-24 (RANLP 2025) evaluated five LLMs on Russian wordplay whose type inventory includes spoonerism. Write: "the first LLM study of nói lái and the first generative/decoding/validity benchmark of it; no LLM study treats spoonerism as a controlled transformation task, and none in Vietnamese." Ask the 2018 authors for their data (attested set, external gold check); their rule description may replace the blog as the taxonomy citation [UNCERTAIN: paper not read].

**(b) "No NFC-vs-NFD or tone-placement re-encoding study on LLMs" — SUPPORTED, with three near-neighbours that must be cited.** No paper compares canonically equivalent encodings of the same Vietnamese text on LLMs and none evaluates placement variants. Neighbours: TokSuite (optional-diacritic and Unicode-styling perturbations on 14 controlled models; no Vietnamese; no named NFC/NFD arm [UNCERTAIN: read the perturbation appendix before Gate 1]); Ghosh & Jyothi (segmentation perturbed with code points fixed, same model families; check for Vietnamese); Inoue et al. (Arabic diacritics change information content). "Understanding Tieq Viet" is a lossy Vietnamese re-encoding on seq2seq. The claim stands if written as "a meaning- and information-identical re-encoding of a tonal Latin script on deployed LLMs". New validity risk from arXiv 2609.20614: log the engine per run and treat "normalizes to NFC" as a (tokenizer, engine) property (`docs/DESIGN_DECISIONS.md` §6.2).

**(c) "EXECUTE excludes Vietnamese" — SUPPORTED at snippet level.** Two snippets list the eight primary languages (Amharic, Arabic, Chinese, English, Hindi, Japanese, Korean, Russian) plus Tamazight and Santali and cipher/byte controls. Confirm against Table 1 of the PDF before citing (the plan says "ten languages").

**(d) STAD paper's language coverage — English only [UNCERTAIN at appendix level].** CMU Pronouncing Dictionary / ARPAbet; released probing data `arpabet_data_llama3_good/bad.csv`; probing targets Llama-3.1-8B; IPA fine-tuning on English tasks. Consequence: STAD is defined over English syllabification of multi-syllable words; applying it to Vietnamese onset–rime boundaries inside a space-delimited syllable is an **extension** the paper must state and validate against the native-checked parser. Cite the ACL 2026 long paper, not the workshop version.

## 4. Framing risks and positioning

- **Pham & Pham 2018** — cite in Background and in the attested-subset provenance; they detect attested nói lái in classical text with rules, we generate, decode and judge validity at scale and ask a tokenization question; report agreement between their rules and our verifier if their data can be obtained.
- **PACUTE** — biggest "clone of another language's benchmark" risk. Position: Vietnamese writes onset, rime and tone *in the string*, so the manipulation is orthographically observable and failure is attributable to tokenization; nói lái is a native practice with attested items; we add meaning-preserving re-encoding and mechanistic localization. Adopt PACUTE's reporting granularity and cite it prominently.
- **Phun-Bench** — in Chinese phonology is latent behind logographs, so failures mix knowledge and access; in Vietnamese it is spelled out, isolating access. Reuse "recall pronunciations but cannot use them" as the Chinese counterpart of H5.
- **TokSuite** — reviewers will ask why we do not train controlled models: TokSuite has no Vietnamese and asks a design question; we ask what deployed tokenizers do to a live script with meaning fixed. If it has a Unicode-normalization arm, say how C1 differs (canonical equivalence, not styling).
- **Ghosh & Jyothi** — same model families; they alter segmentation of identical code points, we alter code points users actually type; NFD changes the canonical tokenization itself. If Vietnamese is among their 27 languages, report their number next to ours.
- **Arabic diacritics (EACL 2026)** — closest re-encoding design; stress information-identity of our arms.
- **Korean jamo (EMNLP 2026), Tieq Viet (2022)** — information-destroying perturbations; place beside C3.
- **Beyond Atomic Tokens** — the fix to our diagnosis, from a group likely to publish an LLM follow-up; cite; use its factorization as the explicit-input baseline; author alerts.
- **KoWit-24** — phrase the gap as "no controlled spoonerism transformation task, none in Vietnamese".
- **Engine-dependent normalization (2609.20614)** — a validity threat; run C1 through two engines for a subset.
- **"Toy or niche"** — ARR H9/H10 protect single-language work; lead with the tokenization question. Introduction examples (all NV): `mèo cái → mài kéo` (high; attested), `bí mật → bị mất` (high; attested), `đầu tiên → tiền đâu` (high; attested), placement pairs `hòa/hoà`, `khỏe/khoẻ` (high).

## 5. Alerts to set (Google Scholar; phrases quoted, AND/OR supported)

1. `"nói lái"`
2. `"nói lái" AND ("language model" OR LLM OR GPT OR Gemma)`
3. `Vietnamese spoonerism`
4. `spoonerism AND ("large language model" OR LLM)`
5. `Vietnamese AND tokenization AND ("large language model" OR LLM) AND (syllable OR tone OR diacritic)`
6. `("NFC" OR "NFD" OR "Unicode normalization" OR "combining diacritics" OR "canonical equivalence") AND (tokenizer OR "language model")`
7. `"tone mark" AND Vietnamese AND ("language model" OR LLM)`
8. `("character-level" OR "sub-token" OR "token understanding") AND benchmark AND (Vietnamese OR "Southeast Asian")`
9. `("Phonemic Tokenizer" OR PhonemicBERT) AND Vietnamese`
10. `("syllabification-tokenization alignment" OR STAD) AND tokenization`
11. `Vietnamese AND ("activation patching" OR "linear probe" OR "sparse autoencoder") AND (tone OR diacritic)`
12. `(CUTE OR EXECUTE OR PACUTE) AND (Vietnamese OR "token understanding")`
13. `wordplay AND Vietnamese AND ("language model" OR LLM)`
14. `"chơi chữ" AND ("mô hình ngôn ngữ" OR LLM)`
15. `"non-canonical tokenization" OR "tokenization robustness" AND Vietnamese`
16. (added) `"grapheme-to-phoneme" AND Vietnamese AND ("language model" OR LLM)` — for UGTPhon-type follow-ups.

Author alerts (Scholar "follow"): Lukas Edman; Alexander Fraser; Disen Liao; Freda Shi; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen; Poulami Ghosh; Bashar Alhafni; Tatsuya Hiraoka; Railey Montalan.

Re-run queries 1–3, 5–7, 12 and the UGTPhon/Pham & Pham checks on arXiv listing pages and Semantic Scholar from an unblocked machine at Gate 1 (18 Oct 2026) and in the week of 14 Dec 2026; record the result in this file with a dated section.
