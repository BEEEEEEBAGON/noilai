# Week 1 checklist (5–11 October 2026) — things only the author can do

Everything below comes from the founding plan; the code side of week 1 (parser, generator,
verifier, tokenizer statistics, pre-registration draft) already exists in this repository.
Tick items in this file and commit; the dates are the plan's.

## Accounts and access (do first; some take two weeks)
- [ ] ORCID profile created (free).
- [ ] OpenReview profile started on an institutional or school email; select "High School Student" if applicable; parental consent form if aged 13–17. Activation on a public email can take two weeks.
- [ ] Kaggle account phone-verified (needed for GPU/TPU quota). Note the weekly quota shown in the account settings in `docs/RESULTS_LOG.md` (the plan's 30 GPU-h / 20 TPU-h figures are from secondary sources).
- [ ] Hugging Face account; request access to the gated checkpoints in `configs/models.yaml` (Gemma 3, Llama 3.1, Vistral); store the token in Kaggle Secrets as `HF_TOKEN`, never in the repository.
- [ ] Google AI Studio key (only if 18+; otherwise an adult collaborator's account, disclosed in the paper); read the real free-tier quotas in AI Studio and record them in `configs/models.yaml`.
- [ ] Groq key; confirm the current free models and per-day limits on the rate-limit page and record them.
- [ ] OpenAI Researcher Access Program form filed (credits, if granted, arrive after the deadline; still worth it for camera-ready experiments).

## People (the critical path)
- [ ] Email 5–10 PhD students / postdocs working on tokenization, multilingual NLP or Vietnamese NLP, offering genuine co-authorship. Use `docs/CONTRIBUTOR_OUTREACH.md`; attach the tokenizer-audit result and the plan's abstract.
- [ ] Recruit three adult native validators (ideally one Northern, one Central, one Southern speaker). They need about 6–10 hours each between 19 October and 8 November. Send `docs/CONSENT_FORM.md` and `docs/VALIDATOR_INSTRUCTIONS.md`.
- [ ] Line up about 20 native speakers for the 20-minute human-baseline form (week of 2 November).
- [ ] Ask the EACL 2027 SRW chairs (eacl2027-srw@googlegroups.com) in writing whether a high-school first author is eligible for the mentorship and whether a mentee may still submit to ARR (deadline 6 November).

## Novelty and literature
- [ ] Read `docs/NOVELTY_SWEEP_2026-09-30.md`; repeat the arXiv search by hand (arXiv is not reachable from the coding environment): "nói lái", "Vietnamese spoonerism", "NFD tokenization", "tone mark placement", "Vietnamese tokenization syllable", and the STAD paper (2026.acl-long.634) to confirm its language coverage.
- [ ] Set the Google Scholar alerts listed at the end of the novelty sweep.
- [ ] Hand-verify every entry of `paper/references.bib` against the ACL Anthology / arXiv page (ACL 2026 desk-rejected over 100 accepted papers for invented citations). Record the check in `docs/RELATED_WORK_VERIFICATION.md`.

## Data and pilot
- [ ] Native check of the parser's conventions that the linguist review flagged: the `qu` question (glide in the onset or in the rime?) with 20 qu- pairs; the `gi` contraction; the old/new placement table (`paper/tables/rules_placement.tex`).
- [ ] Native check of `data/attested_seed.tsv` (every row is unverified; fill `verified_by`), and add 200–400 attested examples from published collections (cite the source per row).
- [ ] Native check of the prompt templates (`prompts/*.yaml`, every line marked `[NATIVE-CHECK]`).
- [ ] Run the 200-item pilot on three small models on Kaggle (`notebooks/kaggle_eval_t4.ipynb`, run plan `pilot` in `configs/run_plan.yaml`) and record the result in `docs/RESULTS_LOG.md`. Gate 1 (18 October) needs the pilot's effect and the novelty verdict.

## Compute log
- [ ] From the first GPU session on, append every session to `data/compute_log.csv` (`scripts/compute_log.py add ...`); checklist item C1 asks for the total.
