# Contributor outreach: draft email and pilot write-up template

ARR guarantees review only to submissions that bring a qualified "service contributor"
(a PhD student with two major-venue papers and three overall, or a postdoc/faculty member
with two). The plan's judgment is that a co-author who genuinely shapes the work is the
route that works; a name added only to supply the service breaches the authorship rule.
The email below therefore offers real work and real credit, and attaches evidence that
the project exists.

## Email (adapt; keep it under 250 words)

Subject: Co-author invitation: do subword LLMs reach the tone inside Vietnamese syllables? (ACL 2027, Jan 4 ARR)

Dear Dr./Ms./Mr. <Name>,

I am a <final-year secondary-school student / ...> in <place>, a native Vietnamese speaker, and I am preparing an ACL 2027 long paper on whether subword LLMs can reach the onset, rime and tone inside Vietnamese syllables. The instrument is nói lái, the Vietnamese spoonerism that swaps rimes or tones between two syllables; a rule engine I have written generates and scores about 10,000 items with no LLM judge, and two training-free interventions (NFC vs NFD encoding of the same text, and the two tone-mark placement conventions) change only the tokens while holding meaning fixed. A first result: the Gemma 3 tokenizer splits 70% of Vietnamese syllables, half of them exactly at the onset–rime boundary, and does not normalize NFD, which costs about one extra token per syllable and isolates the tone mark in 42% of syllables.

Your work on <their paper> is the closest to the tokenization question, and I would like to invite you to join as a co-author with real responsibility for <the experimental design of the re-encoding interventions / the probing and patching analysis / the statistical analysis>, including the writing of that section. The code, data-generation pipeline, pre-registration and a draft manuscript are in a private repository I can share today; the compute is free-tier (Kaggle), and the ARR deadline is 4 January 2027.

I should say plainly that ARR's new service-contributor requirement means a submission needs a qualified reviewer among its authors; I cannot fill that role and am not asking for a name on a paper you do not shape. If the topic interests you, I would be glad to send the two-page pilot write-up and a 20-minute call slot.

With thanks and best regards,
<Name>, <contact>, <repository link on request>

## Pilot write-up (2 pages; produce after the Gate 1 pilot)

1. Question and why it is open (four sentences; cite EXECUTE's language coverage, "Spelling-out", the tokenizer-building preprints).
2. Instrument: nói lái in one figure — "mèo cái → mài kéo" with the Gemma 3 token boundaries (from `paper/figures/fig1_tokens.tex`).
3. What exists: generator (counts from `data/release/v0.3/manifest.json`; v0.3 at the freeze), scoring, validation plan, pre-registration (commit hash).
4. Pilot numbers: three models × 200 items, accuracy per variant with clustered 95% CIs (from `scripts/score_run.py`), copy rate, error classes; NFC vs NFD on the same items.
5. What the co-author would own, and the timeline to 4 January.
6. Honest risks: qu- convention, attested examples that bend the rule, API terms, no IRB.
