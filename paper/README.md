# paper/ — ACL 2027 manuscript sources

`claims.md` is the claim ledger: every contribution and claim, its evidence (design-document section,
data file and hash, verified citation, or the pending experiment), the strongest allowed wording and the
wording that would overclaim. Edit the ledger first, then the prose.

`main.tex` (review mode: `\usepackage[review]{acl}`) inputs nine section files (Related Work is
`sec_related.tex`), the
mandatory Limitations, the Ethical Considerations (with the anonymized Reproducibility
paragraph) and the appendices. Every result is a red `\placeholder{...}`; no number is
typed by hand.

## Build

Needs TeX Live with the `vntex` package (T5 font encoding; `\usepackage[T1,T5]{fontenc}` is what
makes the Vietnamese letters, `\h{}` and `\horn{}` work). Verified on 1 October 2026 with TeX Live 2023
(Ubuntu 24.04: `texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-lang-other`):
no errors, no undefined citations or references, body ends on page 8, Limitations opens page 9.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`tests/test_paper.py` adds the checks a compile does not make: inputs resolve, cite keys
exist, braces balance, Limitations present, `TODO` only inside `\placeholder`, refs have labels,
no identifying information, and the generator scripts reproduce the committed tables.

## Generated files (never edit by hand)

| output | script | source |
|---|---|---|
| `tables/table1_counts.tex` | `gen_table1.py --manifest <build>/manifest.json [--attested <build>/attested.jsonl] [--release] [--note ...]` | a build manifest; a build whose generator arguments differ from the plan, or that is smaller than ~10,000 items without `--release`, is marked "smoke build; regenerate" in the comment **and** in the caption; the header records which rule decided; `--note` adds a red placeholder note to both |
| `figures/fig1_tokens.tex` (+ `fig1_tokens_note.tex`, `.json`) | `gen_fig1_tokens.py` | `data/external/gemma3_tokenizer.model` through `noilai.audit.tokenizers.SentencePieceAdapter`; the note file defines `\figtokensnote`, the caption's data sentences, from the same audit rows |
| `tables/rules_*.tex` | `gen_rule_tables.py` | `noilai.vi.syllable` constants, the engine's speller, the two Hunspell lists (69 placement pairs) |
| `tables/prompt_examples.tex` (+ `.json`) | `gen_prompt_appendix.py` | `prompts/noilai.yaml` + `prompts/demos.yaml` rendered by `noilai.eval.prompts.render` |
| `docs/DATA_STATEMENT.md`, the generated block of §0; `tables/attested_facts.tex` | `gen_data_statement_facts.py --manifest <release>/manifest.json [--check]` | the release manifest and `attested.jsonl`: counts per split, stored placement convention, vulgar-flag counts, the attested seed's exact/mismatch/three-syllable/vulgar/low-confidence rows; the `.tex` file defines `\nAttested`, `\nAttestedExact`, `\nAttestedApprox`, `\nAttestedHsix`, `\nAttestedVerified`, the counts `sec_benchmark.tex` quotes (never typed) |

The committed Table 1 comes from `data/release/v0.2/manifest.json` (10,000 items, plan defaults; built at
commit ee28792 on a dirty tree with the red-team generator corrections included) with `--note` marking it as
the v0.2 counts that the v0.3 rebuild at the stage-1 pre-registration commit replaces; regenerate it from the
frozen release with `--release` at the data freeze. Neither the table header nor the data-statement block
prints the build seed (DESIGN_DECISIONS 4.6).

Unnumbered sections (`\section*{Limitations}`, `\section*{Ethical Considerations}`) carry no
`\label`: a `\label` after a starred section resolves to the preceding numbered section, so they
are referred to by name, and `tests/test_paper.py` refuses a `\label` that follows a `\section*`.

## Bibliography

`main.tex` cites two files: `references.bib` (the bibliographer's verified entries, Anthology-id
keys such as `2025-findings-acl-95`, arXiv ids such as `2609.21362`; see
`docs/RELATED_WORK_VERIFICATION.md`) and `references_PLACEHOLDER.bib`, which holds only the
entries the verified file lacks (grammars, the Hunspell resource, the nói lái blog and book,
XCOPA, model reports, statistics classics, patching methods, and the closest-work items
`docs/DESIGN_DECISIONS.md` §12.27 says must be cited). Every placeholder entry is marked
`UNVERIFIED`; as each is confirmed by hand it moves into `references.bib` and is deleted here.
`tests/test_paper.py` checks that every `\cite` key resolves, that no key occurs in both files
(BibTeX's "Repeated entry"), that every placeholder entry still carries its mark, and that the
Vietnamese accent macros of the placeholder file re-render to the intended NFC strings.

On 1 October 2026 the related-work round added 24 entries verified against the ACL Anthology XML
(`acl-org/acl-anthology`, `data/xml/`, via raw.githubusercontent.com, since aclanthology.org is
egress-blocked here) and generated from it rather than typed; seven of them replace placeholder
entries (Kaushal & Mahowald, Itzhak & Levy, Rust et al., Ahia et al., Pham & Pham, Phun-Bench,
KoWit-24), whose cite keys moved to Anthology ids. "Verified" means the title, authors and venue
match the XML and the claim made in the paper matches the abstract; numbers inside the papers were
not checked. BibTeX has no comment syntax inside an entry, so notes are `comment = {...}` fields
(ignored by `acl_natbib.bst`), and stacked accents are braced (`{\`{\^{o}}}`) so that natbib can
format the citation labels.
