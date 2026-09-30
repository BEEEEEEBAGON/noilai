# AI-use log

Running record of generative-AI assistance in this project, kept for the ARR Responsible NLP checklist (item E1; E2 if present) and for the *Acknowledgements* of the final version. ARR's policy: AI tools do not qualify for authorship; their use for writing or coding, and its scope, must be disclosed in the checklist, with details in the Acknowledgements; the authors remain fully responsible for correctness. One entry per session or per distinct use; never edited after the fact, only appended. Grammar checkers and short-form input aids need no entry.

Format: `date | tool | what it produced | who directed and reviewed it | how it was verified`.

| Date | Tool | What was produced | Direction and review | Verification |
|---|---|---|---|---|
| 2026-09-30 | Claude Code (Anthropic; model Claude) | This repository's code (syllable parser and speller, Unicode and re-encoding arms, lexicon, V1–V4 generator, tokenizer audit, statistics, probing, evaluation harness, cloud run kit, tests), the configuration files, the prompt templates (all Vietnamese strings marked `[NATIVE-CHECK]`), the LaTeX paper skeleton (`paper/`: section drafts, hypothesis text, generated tables), and the study documents (`docs/PREREGISTRATION.md`, `DATA_STATEMENT.md`, `CONSENT_FORM.md`, `VALIDATOR_INSTRUCTIONS.md`, `HUMAN_BASELINE_FORM.md`, `CHECKLIST_DRAFT.md`, this file). The research design is the author's founding plan (`docs/PLAN_2026-09-30.md`); the assistant implemented and drafted under that plan. | The author directed each session, specified the plan sections to implement, and reviews every file; the author is responsible for correctness. | `pytest` (every module has tests; the suite must stay green); generated tables and figures are produced from data by scripts, never typed; every Vietnamese string awaits a native validator; every bibliography entry is marked UNVERIFIED until checked by hand against its source; **no result in the paper is written until a run's manifest exists**. To be disclosed in checklist E1 and in the Acknowledgements at camera-ready (not before, to preserve anonymity). |
| 2026-09-30 | Claude Code | The placeholder bibliography `paper/references_PLACEHOLDER.bib`, drafted from the plan's source list and from memory. | Author to replace with the hand-verified `paper/references.bib`. | Every entry carries `note = {UNVERIFIED}`; none may be cited in a submitted PDF before it is confirmed in `docs/RELATED_WORK_VERIFICATION.md`. |

## What AI assistance did **not** do

- It did not generate benchmark items: every item is the output of the deterministic rule engine, seeded, with a manifest.
- It did not produce any experimental result: no model has been run as of 2026-09-30; every result in the manuscript is a red `\placeholder`.
- It did not act as an LLM judge: no metric uses one.
- It did not choose the hypotheses: H1–H6 are the founding plan's, pre-registered in `docs/PREREGISTRATION.md`.

## Wording for the final version (Acknowledgements, camera-ready only)

> Code in the released repository and drafts of this manuscript were produced with the assistance of an AI coding assistant (Claude Code, Anthropic) under the authors' direction and review; the authors are responsible for every line and every claim. Benchmark items come from a rule engine, not from a language model.
