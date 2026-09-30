# paper/ — ACL 2027 manuscript sources

`main.tex` (review mode: `\usepackage[review]{acl}`) inputs eight section files, the
mandatory Limitations, the Ethical Considerations (with the anonymized Reproducibility
paragraph) and the appendices. Every result is a red `\placeholder{...}`; no number is
typed by hand.

## Build

pdflatex is not installed on the build machine. On a TeX Live 2024+ installation with the
`vntex` package (T5 font encoding; `\usepackage[T1,T5]{fontenc}` is what makes the Vietnamese
letters, `\h{}` and `\horn{}` work):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Until then, `tests/test_paper.py` stands in for the compile check: inputs resolve, cite keys
exist, braces balance, Limitations present, `TODO` only inside `\placeholder`, refs have labels,
no identifying information, and the generator scripts reproduce the committed tables.

## Generated files (never edit by hand)

| output | script | source |
|---|---|---|
| `tables/table1_counts.tex` | `gen_table1.py --manifest <build>/manifest.json [--attested <build>/attested.jsonl]` | a build manifest; a reduced build is marked "smoke build; regenerate" in the comment **and** in the caption |
| `figures/fig1_tokens.tex` (+ `.json`) | `gen_fig1_tokens.py` | `data/external/gemma3_tokenizer.model` through `noilai.audit.tokenizers.SentencePieceAdapter` |
| `tables/rules_*.tex` | `gen_rule_tables.py` | `noilai.vi.syllable` constants, the engine's speller, the two Hunspell lists (69 placement pairs) |
| `tables/prompt_examples.tex` (+ `.json`) | `gen_prompt_appendix.py` | `prompts/noilai.yaml` + `prompts/demos.yaml` rendered by `noilai.eval.prompts.render` |

The committed Table 1 comes from the smoke build
`scripts/build_data.py --n-lexicon 300 --n-pseudo 200 --per-cell-t1 150 --per-cell-t2 80 --per-cell-t3 60 --core-per-cell 20`
(1,400 items) and must be regenerated from the frozen release at the data freeze.

## Bibliography

`main.tex` cites two files: `references.bib` (the bibliographer's verified entries, Anthology-id
keys such as `2025-findings-acl-95`, arXiv ids such as `2609.21362`; see
`docs/RELATED_WORK_VERIFICATION.md`) and `references_PLACEHOLDER.bib`, which holds only the
entries the verified file lacks (grammars, the Hunspell resource, the nói lái blog and book,
XCOPA, model reports, statistics classics, patching methods, and the closest-work items
`docs/DESIGN_DECISIONS.md` §12.27 says must be cited). Every placeholder entry is marked
`UNVERIFIED`; as each is confirmed by hand it moves into `references.bib` and is deleted here.
`tests/test_paper.py` checks that every `\cite` key resolves and that every placeholder entry
still carries its mark.
