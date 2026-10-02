# paper/ — ARR manuscript sources

`main.tex` (review mode: `\usepackage[review]{acl}`) inputs nine numbered section files, the
mandatory unnumbered Limitations, the Ethical Considerations (with the anonymized Reproducibility
paragraph) and the appendices. Every result is a red `\placeholder{...}`; no count or hash is
typed by hand (see *Generated files*). `claims.md` is the claim ledger the prose is written from:
every contribution and claim, its evidence (design-document section, data file and SHA-256,
verified citation, or "pending E#"), the strongest wording that evidence allows and the wording
that would overclaim; its section 10 records the ARR call-for-papers rules and the measured page
accounting of the last compile.

## Build

A TeX Live 2024+ installation with the `vntex` package (T5 font encoding; `\usepackage[T1,T5]{fontenc}`
is what makes the Vietnamese letters, `\h{}` and `\horn{}` work) compiles the manuscript with

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Debian/Ubuntu, `apt-get install texlive-latex-extra texlive-fonts-recommended texlive-lang-other`
provides everything (the T5 encoding file `t5enc.def` ships with `texlive-lang-other`). The last compile
of this draft produced no LaTeX error, no undefined reference or citation and no BibTeX warning;
the Vietnamese font set (`utmr8v`) has no visible-space glyph, so `\tokspace` in `main.tex` renders the
SentencePiece word-initial space as `\ensuremath{\sqcup}`. Section files begin with a page-budget comment;
the budget is the plan, and the ledger's section 10 reports how far the compiled sections are from it.

Before every compile, `tests/test_paper.py` is the gate: inputs resolve, cite keys exist in the two
bibliography files, braces balance, Limitations is present and unnumbered, `TODO` only inside
`\placeholder`, every `\ref` has a `\label`, no identifying information, the design-document wording
the paper must carry (and the superseded wording it must not), and every generator script
reproduces its committed table.

## Generated files (never edit by hand)

| output | script | source |
|---|---|---|
| `tables/table1_counts.tex` | `gen_table1.py --manifest <build>/manifest.json [--attested <build>/attested.jsonl] [--release] [--note ...]` | a build manifest; a build whose generator arguments differ from the plan, or that is smaller than ~10,000 items without `--release`, is marked "smoke build; regenerate" in the comment **and** in the caption; the header records which rule decided; `--note` adds a red placeholder note to both |
| `figures/fig1_tokens.tex` (+ `fig1_tokens_note.tex`, `.json`) | `gen_fig1_tokens.py` | `data/external/gemma3_tokenizer.model` through `noilai.audit.tokenizers.SentencePieceAdapter`; the note file defines `\figtokensnote`, the caption's data sentences, from the same audit rows |
| `tables/rules_*.tex` | `gen_rule_tables.py` | `noilai.vi.syllable` constants, the engine's speller, the two Hunspell lists (69 placement pairs) |
| `tables/prompt_examples.tex` (+ `.json`) | `gen_prompt_appendix.py` | `prompts/noilai.yaml` + `prompts/demos.yaml` rendered by `noilai.eval.prompts.render` |
| `docs/DATA_STATEMENT.md`, the generated block of §0; `tables/attested_facts.tex` | `gen_data_statement_facts.py --manifest <release>/manifest.json [--check]` | the release manifest and `attested.jsonl`: counts per split, stored placement convention, vulgar-flag counts, the attested seed's exact/mismatch/three-syllable/vulgar/low-confidence rows; the `.tex` file defines `\nAttested`, `\nAttestedExact`, `\nAttestedApprox`, `\nAttestedHsix`, `\nAttestedVerified`, the counts `sec_benchmark.tex` quotes (never typed) |
| `tables/release_facts.tex`, `tables/release_hashes.tex` (+ `release_facts.json`) | `gen_release_facts.py [--manifest <release>/manifest.json] [--plan configs/run_plan.yaml] [--check]` | the **public** release manifest and the run plan: 39 `\release...` macros (item, base-pair, split, sample and drop counts, version, build date, generator commit, dirty flag, placement style, content and canary digests) and the release-identity table `tab:hashes` (content SHA-256, canary-GUID SHA-256, the frozen attested strings, every seeded item file of the run plan, the pinned look-up resources). It refuses a manifest that carries the build seed or the canary GUID, so the private manifest can never be passed by mistake |
| `tables/tokenizer_audit.tex` (+ `.json`) | `gen_tokenizer_audit.py [--audit data/audit] [--check]` | `data/audit/counts.json` and the per-tokenizer `gemma3.json` / `gemma2.json`: tokens per syllable, single-token share, alignment over all and over split syllables, onset|rime split rate, tone-isolation rate and the census verdict, as the table `tab:tokaudit` in the compute appendix |

The committed Table 1 comes from `data/release/v0.3/manifest.json` (10,000 items, plan defaults, built at
generator commit f94f655 of the original repository, which maps to 62f63ae here per `docs/MIGRATION.md`, on a clean tree) with `--release`, which the header records as "generator arguments
match the plan; declared the frozen release by --release". Neither the table header, the release-facts file
nor the data-statement block prints the build seed (DESIGN_DECISIONS 4.6); the canary GUID appears only as its
SHA-256.

Unnumbered sections (`\section*{Limitations}`, `\section*{Ethical Considerations}`) carry no
`\label`: a `\label` after a starred section resolves to the preceding numbered section, so they
are referred to by name, and `tests/test_paper.py` refuses a `\label` that follows a `\section*`.

## Bibliography

`main.tex` cites two files. `references.bib` holds the verified entries: the bibliographer's entries
(Anthology-id keys such as `2025-findings-acl-95`, arXiv ids such as `2609.21362`; see
`docs/RELATED_WORK_VERIFICATION.md`) and, under the header comment "Entries verified against the ACL
Anthology XML", the entries added for the related-work section, whose keys are the Anthology ids
(`2024-findings-emnlp-86`, `Y18-1063`, `D18-2012`, `P16-1162`, ...) and whose fields were copied from
the Anthology's own XML, never typed. `references_PLACEHOLDER.bib` holds only the entries the verified
file lacks (grammars, the Hunspell resource, the nói lái blog and book, model reports, statistics
classics, patching methods, the closest-work preprints `docs/DESIGN_DECISIONS.md` §12.27 says must be
cited). Every placeholder entry is marked `UNVERIFIED`; as each is confirmed by hand it moves into
`references.bib` and is deleted here. Stacked Vietnamese accents in the placeholder file are written with a
braced base (`{\'{\^e}}`, `{\d{\^o}}`): the unbraced forms (`\'\^e`) are a fatal error under T5.
`tests/test_paper.py` checks that every `\cite` key resolves, that no key occurs in both files
(BibTeX's "Repeated entry"), that every placeholder entry still carries its mark, and that the
Vietnamese accent macros of the placeholder file re-render to the intended NFC strings.
`claims.md` section 9 lists, per related-work thread, which keys are verified and which still carry
the mark, and which attributed numbers may appear only inside a `\placeholder`.
