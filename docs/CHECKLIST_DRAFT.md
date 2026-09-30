# ARR Responsible NLP Checklist — first-pass answers for this design (draft, 30 September 2026)

The checklist is filled in the OpenReview submission form and appended to the paper by ARR; this file is the working draft so that every answer is decided before the form opens (about 21 December 2026). Item wordings below paraphrase the checklist as published at https://aclrollingreview.org/responsibleNLPresearch/ — **[UNCERTAIN: verify]** the live form's exact items and numbering at submission, especially Part E, whose wording changed in 2024–2025. A justified "No" is not grounds for rejection; an incorrect or incomplete checklist is. Section pointers refer to `paper/main.tex` and its inputs. Answers that depend on results are marked **[at submission]**.

## A. For every submission

| # | Item | Answer | Justification / where |
|---|---|---|---|
| A1 | Did you describe the limitations of your work? | **Yes** | Mandatory *Limitations* section (`paper/sec_limitations.tex`): generated items are not natural usage; the variant taxonomy rests on a secondary source; one author and three validators shape the gold; the attested subset is contaminated; cross-tokenizer comparisons are observational; API drift and data use; precision/hardware; two small models in the mechanistic study; per-cell power; prompt sensitivity; single language. |
| A2 | Did you discuss any potential risks of your work? | **Yes** | *Ethical Considerations* (`paper/sec_ethics.tex`): offensive readings in nói lái (flagged, excluded from APIs and the human form); dual use (fluent nói lái could evade keyword filters; the benchmark also helps detect it); free API tiers that train on inputs. |
| A3 | Do the abstract and introduction summarize the paper's main claims? | **Yes** **[at submission]** | Every result in the abstract is a `\placeholder` until the numbers exist; the final abstract is re-read against Tables 2–3 and Figure 5 before submission. |
| A4 | Did you use AI assistants (writing or coding) and disclose it? *(if the live form still lists this here rather than in Part E)* | **Yes** | See E1. |

## B. Did you use or create scientific artifacts?

| # | Item | Answer | Justification / where |
|---|---|---|---|
| B1 | Did you cite the creators of artifacts you used? | **Yes** | Hunspell vi_VN (Hồ Ngọc Đức; László Németh), Viet74K (Hồ Ngọc Đức), XCOPA (Ponti et al.), the Gemma 3 tokenizer files (google/gemma_pytorch), every model's release paper or model card, lm-evaluation-harness if used; `paper/references*.bib`. |
| B2 | Did you discuss the license or terms for use of the artifacts? | **Yes** | Section 3.6 "Release" and *Ethical Considerations*: the GPLv2 spelling and word lists are consulted at build time, and the release states under which terms the reproduced two-syllable entries appear (written permission requested from the lists' authors on 30 Sep 2026; GPLv2 with attribution as the fallback; DD §4.1) **[at submission]**; dev split CC BY 4.0; test split gated, encrypted, CC BY-NC-ND 4.0; XCOPA CC BY 4.0; model licenses (Gemma Terms, Apache-2.0 families, Llama community license, PhoGPT, Vistral) in the appendix panel table **[at submission]**; API terms (all three providers 18+; Gemini free tier trains on inputs) quoted in `configs/models.yaml`. |
| B3 | Did you discuss whether your use of existing artifacts was consistent with their intended use, and whether the derived artifacts' intended use is specified? | **Yes** | Spelling and word lists used as look-ups (their intended use); XCOPA used as an evaluation set; models evaluated, not trained on outputs (providers' terms permit evaluation). Derived artifact: NóiLái dev split CC BY 4.0, test split gated CC BY-NC-ND 4.0 with canary; intended for evaluation, not training. |
| B4 | Did you discuss steps taken to check whether the data contain personal information or offensive content, and how you handled it? | **Yes** | Generated items contain no personal information (no names; the attested seed contains one proper-name pun, *Vũ Như Cẩn*, which is public folk material). Offensive content: a blocklist of taboo syllables and pairs applied **at generation** to inputs, gold, T2 readings and T3 candidates in either order (`data/vulgar_lexicon.tsv`; counts per cell in the manifest); attested items screened by hand; validators judge every sampled item for an offensive reading (rate with α reported); flagged items excluded from the public dev split, every API prompt and every human form, kept flagged in the gated test file (Section 3.3, Ethics; DD §11.5). |
| B5 | Did you provide documentation of the artifacts (domains, languages, linguistic phenomena, demographic groups)? | **Yes** | Data statement (`docs/DATA_STATEMENT.md`, short form in Appendix D): Vietnamese `vi-VN`, wordplay, syllable-internal phenomena, coarse annotator demographics **[at submission]**. |
| B6 | Did you report relevant statistics like the number of examples and details of train/test/dev splits? | **Yes** | Table 1 (counts per task × variant × split, core, pool sizes, attested count, validation α) generated from the build manifest; no training split (evaluation only). |

## C. Did you run computational experiments?

| # | Item | Answer | Justification / where |
|---|---|---|---|
| C1 | Did you report the number of parameters, total computational budget (e.g. GPU hours) and computing infrastructure? | **Yes** **[at submission]** | Appendix F: per-run table with model size, hardware (Tesla T4 ×1/×2, TPU v5e-8), engine, precision, quantization, wall time; totals from `data/compute_log.csv` (`scripts/compute_log.py totals`), kept from the first GPU session; API call counts from `data/runs/api_ledger.json`. Plan budget: 70–150 GPU-hours, 10–20 TPU-hours, all free tiers. |
| C2 | Did you discuss the experimental setup, including hyperparameter search and best-found values? | **Yes** | Section 4: greedy decoding, 64 new tokens, three fixed demonstrations, three paraphrases, thinking off; no hyperparameter search (evaluation only); probe hyperparameters and seeds in Appendix; every run's manifest copies the model configuration verbatim. |
| C3 | Did you report descriptive statistics about your results (error bars, summary statistics from multiple runs)? | **Yes** | Every accuracy with a 95% clustered bootstrap interval; paired tests with Holm correction within declared families; mean and range over three paraphrases; probe results over 5 seeds; pre-registered analysis plan (`docs/PREREGISTRATION.md`). |
| C4 | If you used existing packages (e.g. for preprocessing, normalization, evaluation), did you report the implementation, model and parameter settings used? | **Yes** | Appendix F: package versions (`pip freeze` recorded per session under `data/runs/env/`), vLLM/transformers/llama.cpp versions, SentencePiece, statsmodels; scoring is our own released code, no LLM judge. |

## D. Did you use human annotators (e.g. crowdworkers) or research with human participants?

| # | Item | Answer | Justification / where |
|---|---|---|---|
| D1 | Did you report the full text of instructions given to participants, including screenshots, disclaimers of risks, and how the data would be used? | **Yes** | `docs/VALIDATOR_INSTRUCTIONS.md` and `docs/HUMAN_BASELINE_FORM.md` (released with the data; summarized in Appendix C); risks (vulgar readings) and data use stated in the consent form. |
| D2 | Did you report information about how you recruited (e.g. crowdsourcing platform, students) and paid participants, and discuss whether payment was adequate given the participants' demographic (e.g. country of residence)? | **Yes** | Volunteers recruited from the author's personal network and a Vietnamese maker community; **unpaid**, which the paper states plainly with the Aya and Masakhane volunteer precedents; validators' expected time (6–10 h) and respondents' (20 min) given; acknowledgement offered. An honest description of unpaid volunteering, not a claim that payment was adequate. |
| D3 | Did you discuss whether and how consent was obtained from people whose data you're using/curating? | **Yes** | Bilingual written consent (`docs/CONSENT_FORM.md`): purpose, task, time, voluntary and unpaid, withdrawal until the results freeze, data stored, no ethics review, contact. Attested folk material is public and anonymous; literary quotations are cited. |
| D4 | Was the data collection protocol approved (or determined exempt) by an ethics review board? | **No** | The authors have no access to an institutional review board or school-based board (**[at submission]**: if a collaborator's IRB or a school board reviews it before December, change to Yes and cite the approval). Justification stated in the paper: minimal-risk anonymous language judgments by adult volunteers, no personal data, written consent, right to withdraw, vulgar items excluded. |
| D5 | Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? | **Yes** **[at submission]** | Coarse fields only (age band, region grown up in, years in Vietnam, raised abroad) for validators and respondents; Appendix D / data statement. |

## E. Did you use AI assistants (e.g. ChatGPT, Copilot) in your research, coding, or writing?

| # | Item | Answer | Justification / where |
|---|---|---|---|
| E1 | Did you include information about your use of AI assistants? | **Yes** | Wording per `docs/DESIGN_DECISIONS.md` §11.6: "Yes — AI assistants (Claude Code, Anthropic; versions in the appendix) drafted and refactored code, documentation and planning documents and helped edit prose; every module was reviewed, tested and run by the authors, every reference verified by hand; research questions, design, rule tables and analyses are the authors'; no LLM generated or scored items." Running log: `docs/AI_USE_LOG.md`; stated in *Ethical Considerations* now and in the *Acknowledgements* at camera-ready (anonymity). |
| E2 | *(if present on the live form)* Did you disclose which parts (ideas, text, code) involved AI assistance and how they were verified? **[UNCERTAIN: verify the item's existence and wording]** | **Yes** | Ideas: the research design is the author's (founding plan of 30 Sep 2026); AI assistance was used for implementation, tests, drafting and editing. Verification: every line of code is covered by tests (`pytest`), every generated table comes from data, every reference is checked by hand against its source before submission (ACL 2026 desk-rejected papers for invented citations); the AI-use log records each session. |

## Open items for the author before the form opens

1. Confirm the live checklist's numbering and the wording of A4/E1/E2 **[UNCERTAIN: verify]**.
2. D4: pursue a school-based board or a collaborator's IRB; if obtained, update D4 and the Ethics section.
3. D2: decide whether small thank-you gifts (non-monetary) are offered; if so, state it.
4. C1: keep `data/compute_log.csv` current from the first GPU session; the total goes in Appendix F and here.
5. B2: collect every model's license string into the appendix panel table at the panel freeze.
