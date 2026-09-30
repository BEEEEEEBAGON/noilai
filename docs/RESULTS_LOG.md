# Results log

Every number that may end up in the paper is recorded here with its provenance (script,
input hashes, commit) the day it is produced. Nothing in this file is typed from memory;
each entry names the file the number was read from. Model-output results require a
run directory with a manifest; tokenizer-only results require the tokenizer file's hash.

## 2026-09-30 — Tokenizer audit, Gemma 3 and Gemma 2 SentencePiece models (no model outputs) [RL-2026-09-30-01]

Source: `data/audit/gemma3.json`, `data/audit/gemma2.json`, produced by
`scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 data/external/gemma2_tokenizer.model:gemma2`
at commit 699634c on the 6,595 standard syllables of the Hunspell vi_VN list (new-style
placement), each tokenized in running-text position (preceded by a space). Tokenizer
hashes: `data/HASHES.json` (gemma3 `1299c11d…`, gemma2 `61a7b147…`).

| tokenizer | encoding | tokens / syllable | single-token share | boundary alignment | onset–rime split | tone mark isolated |
|---|---|---|---|---|---|---|
| Gemma 3 (262,144 BPE, normalizer = identity, byte fallback) | NFC | 1.78 | 29.5% | 0.94 | 51.5% | 0% |
| Gemma 3 | NFD | 2.83 | 7.1% | 0.27 | 6.8% | 41.6% |
| Gemma 2 (256,000 BPE, normalizer = identity, byte fallback) | NFC | 1.77 | 29.9% | 0.94 | 50.7% | 0% |
| Gemma 2 | NFD | 2.90 | 7.3% | 0.29 | 4.4% | 50.4% |

Normalization census (500 multi-syllable words from the word list): NFD input produces
the same token ids as NFC input for 3.4% of words only (those without diacritics); the
NFD sequence is longer for 92–93%. Neither tokenizer normalizes Unicode, so the C1 arm
(NFC vs NFD) is a live intervention on Gemma models by construction. The SentencePiece
normalizer spec is `identity` in both model files.

Reading: under NFC, 70% of Vietnamese syllables are split by the Gemma tokenizers, and
when they are split the boundary is overwhelmingly at a linguistic boundary (alignment 0.92 among split syllables; 0.94 over all syllables, a single-token syllable counted as 1.0 — RL-2026-09-30-06), most often between onset and rime (51.5% of all syllables have an onset|rime
split). Under NFD, the split moves inside letters (a boundary between a base letter and
its combining mark is never a linguistic boundary; alignment 0.21 among split syllables, 0.27 over all) and the tone mark
becomes a token of its own in 42% of syllables. This is the pattern H3 relies on: NFD
makes the tone an explicit symbol at the cost of longer, misaligned sequences.

Correction (later the same day): the Gemma 3 NFD row first read 2.90 tokens per syllable and
7.3% single-token, which are Gemma 2's numbers copied by hand; the values above are re-read from
`data/audit/gemma3.json` by `scripts/reconcile_counts.py` (`data/audit/counts.json`), which is
now the only source for these figures.

Caveats: these are the Gemma 2/3 tokenizer FILES from google/gemma_pytorch; the Hugging
Face tokenizers used at inference must be checked against them (same ids on the same
strings) before any model result is joined to these statistics. Token counts here exclude
the special/BOS tokens and the leading space when it forms a token of its own.

## 2026-09-30 — Benchmark build v0.1 at the plan's default sizes (generator output, no model outputs) [RL-2026-09-30-02]

Source: `data/release/v0.1/manifest.json` (build seed withheld, DESIGN_DECISIONS 4.6; generator at commit b03c7d5 with
uncommitted docs at the first build; to be rebuilt at the commit that freezes the generator). The split sizes and
commit below are those of the first v0.1 build; the committed v0.1 manifest reads dev 1,968 / test 8,032 at
commit 21fbeec (dirty), see RL-2026-09-30-05. 2,500 base pairs
(1,500 lexical, 1,000 pseudo), 2,181 of them used; 10,000 items: T1 4,000, T2 2,000, T3 4,000
(2,000 yes/no pairs); dev 1,984 / test 8,016 by base pair; core 1,496 (125 per T1/T2 cell,
62 pairs per T3 cell). Pool sizes before capping: T1 1,465–2,145 per variant, T2 844–1,343.
Of the 4,000 T1 items, 288 have a lexical output (287 from lexical inputs). T3 twins: tone
494, rime 486, onset 476, other-variant 458, spelling 86 (spelling twins exist only when the
gold has a c/k, g/gh or ng/ngh trigger). T2 inputs have 1 reading in 728 cases, 2 in 939,
3+ in 333. Attested seed: 22 rows, 16 reproduced exactly by the engine.

## 2026-09-30 — Benchmark build v0.2 (six-variant generator; supersedes v0.1; rebuilt after the red-team round) [RL-2026-09-30-03]

Rebuilt after the red-team corrections (design 14, items 2, 8, 11, 46, 50, 56, 61): per-syllable
i/y emission from the word-list majority (only `mĩ -> mỹ` prefers y in Viet74K), zero-onset bare
/i/ excluded, marginal rimes by the N < 4 rule plus the loan list, pseudo pairs quota-sampled to the
lexical marginals (achieved shares within 0.02 of target in the manifest), `content_sha256`, and
the two seeded sub-samples `noilai_main.jsonl` (350 per cell, core included) and `noilai_c2.jsonl`
(500 C2-affected items from an independent pool).

**Superseded (first v0.2 build, not the committed release).** The counts in the next paragraph were read
from the FIRST v0.2 build's manifest and do not match the committed `data/release/v0.2/manifest.json`
(37 vs 52 vulgar-flagged, 364 vs 291 C2-affected, identity/plain reversal 1,789 vs 1,844, illegal 4,474 vs
4,304, attested 35 rows / 27 exact vs 34 / 28); the committed release's counts are RL-2026-09-30-05 below.

Source at the time: the first v0.2 build's manifest (superseded; build seed withheld, DESIGN_DECISIONS 4.6;
the manifest's `git_commit` names the generator code, rebuilt at the freeze commit). Design changes applied (docs/DESIGN_DECISIONS.md
section 12: items 4, 5, 15, 16, 23, 24): six-variant outputs with identity and plain-reversal
drops and merged `variant_labels`; old-style placement storage with attested i/y; vulgar screen
at generation; reserved attested and demonstration pairs; canary header. 2,500 base pairs;
10,000 items (T1 4,000, T2 2,000, T3 4,000); core 1,496; 37 items vulgar-flagged (forced to the
gated test split); 364 items C2-affected. Drop counts per filter are in the manifest: for the
2,500 pairs the six variants produced 15,000 candidate outputs, of which identity 1,789,
plain reversal 1,789 and illegal 4,474 were dropped (the identity/plain-reversal counts are
mirror images across V1/V6, V2/V3, V4/V5, as the algebra requires). Attested seed: 35 rows,
27 reproduced exactly by the engine at the declared positions (the six-way `thay đổi`
illustration reproduces under all six variants).

## 2026-09-30 — Tone-mark placement convention in XCOPA-vi (pre-registration input, no model outputs) [RL-2026-09-30-04]

Source: `data/audit/placement_xcopa.json`, from `scripts/count_placement.py` on the XCOPA-vi
test and validation files (600 items, 2,400 text fields, 12,333 word tokens). Of the 62 tokens
whose placement differs between the conventions, 60 are old style (hòa; 46 oa, 1 oe, 13 uy)
and 2 new style (hoà): old share 96.8%. This supports the design's baseline (old style stored;
C2 = old -> new). The count on a larger reference corpus (Vietnamese Wikipedia or a news
corpus, not reachable from the build machine) is the author's before Gate 1; if it flips the
majority, DEVIATIONS.md records the flip and H4's direction flips with it.

## 2026-09-30 — v0.2 release counts re-read from the committed files (generator output, no model outputs) [RL-2026-09-30-05]

Source: `data/release/v0.2/manifest.json` (generator commit ee28792, dirty tree; build seed withheld,
DESIGN_DECISIONS 4.6; content SHA-256 `95ba53ae9f3f4b1a45192dceeccff93de34f4c303f20025e3164e12ae18cb0cd`)
and `data/release/v0.2/attested.jsonl`; every number below is the sum or count over that file and
`tests/test_paper.py` recomputes each of them. 10,000 items (T1 4,000, T2 2,000, T3 4,000) from 2,500 base
pairs; dev 2,078 / test 7,922; core 1,496. Vulgar-flagged items 52 (`vulgar_counts`; T1 32, T2 24 flagged,
17 T3 twins dropped); C2-affected items 291 (`c2_affected_counts`); base pairs dropped because the input pair
itself is a taboo phrase 15 (`base:vulgar_input`). Six-variant outputs dropped over the 2,500 pairs: identity
1,844, plain reversal 1,844 (mirror images across V1/V6, V2/V3, V4/V5), illegal 4,304. Marginal rimes excluded
25 (`n_marginal_rimes`). Pseudo-pair quota over four strata (zero onset, glide, stop coda, tone class; the
spelling-trigger stratum joined the generator after this build). Attested seed: 34 rows, 28 reproduced
exactly, 6 approximate (4 substitution, 2 merger), 24 H6-eligible, 5 three-syllable, 1 vulgar-flagged,
0 native-verified. v0.1 correction: the committed `data/release/v0.1/manifest.json` reads dev 1,968 /
test 8,032 at commit 21fbeec (dirty); RL-2026-09-30-02 quotes the first build's 1,984 / 8,016.

## 2026-09-30 — Figures derived from the tokenizer audit rows and a simulation (no model outputs) [RL-2026-09-30-06]

Source: `data/audit/gemma3_rows.csv` (6,595 syllables × 2 encodings, written by `scripts/audit_tokenizers.py`,
RL-2026-09-30-01) and `data/audit/gemma3.json`; `tests/test_paper.py` recomputes every figure.

- **Alignment among split syllables** (mean `boundary_alignment` over rows with `single_token` false; a
  single-token syllable is scored 1.0 in the all-syllable mean): NFC 0.917 over 4,652 split syllables,
  NFD 0.215 over 6,130. The all-syllable means 0.94 / 0.27 of RL-2026-09-30-01 stand, labelled as such; the
  reading "when they are split the boundary is overwhelmingly at a linguistic boundary" refers to 0.92,
  not 0.94. Once `noilai.audit.tokenizers.audit_inventory` emits `boundary_alignment_among_split_mean`
  (`n_split`), the summary field replaces this recomputation.
- **NFD length**: the NFD sequence is longer for 92.2% of the 500 census words (`gemma3.json`
  `normalization_census.nfd_longer_frac`), and the NFD tokens per syllable are 58.8% above NFC
  (2.8308 / 1.7826 − 1). "92% longer" in earlier drafts conflated the two; the design document's H3 row now
  quotes both.
- **Single-token share by tone, NFC** (the 6,594 NFC rows whose syllable parses, grouped by parsed tone):
  ngang 50.8%, sắc 28.3%, hỏi 23.2%, huyền 23.0%, nặng 22.4%, ngã 18.4% — i.e. 51% vs 18–28%, the
  structural-baseline motivation of DESIGN_DECISIONS 9.1. Once `scripts/reconcile_counts.py` emits
  `single_token_by_tone`, that field replaces this recomputation.
- **Agreement illustration** (DESIGN_DECISIONS 10.1; simulated with `noilai.stats.agreement` on 200,000
  binary units at 96% prevalence, `numpy.random.default_rng(0)`, two coders with independent symmetric
  error): 1% error each → raw agreement 0.980, Krippendorff's α 0.788, Gwet's AC1 0.978; 2% error each → raw agreement 0.961,
  Krippendorff's α 0.645, Gwet's AC1 0.956. The earlier "α = 0.84 against 98% raw agreement at 2% error" did not
  reproduce and is replaced.
