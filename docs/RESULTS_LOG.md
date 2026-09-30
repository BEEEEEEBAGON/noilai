# Results log

Every number that may end up in the paper is recorded here with its provenance (script,
input hashes, commit) the day it is produced. Nothing in this file is typed from memory;
each entry names the file the number was read from. Model-output results require a
run directory with a manifest; tokenizer-only results require the tokenizer file's hash.

## 2026-09-30 — Tokenizer audit, Gemma 3 and Gemma 2 SentencePiece models (no model outputs)

Source: `data/audit/gemma3.json`, `data/audit/gemma2.json`, produced by
`scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 data/external/gemma2_tokenizer.model:gemma2`
at commit 699634c on the 6,595 standard syllables of the Hunspell vi_VN list (new-style
placement), each tokenized in running-text position (preceded by a space). Tokenizer
hashes: `data/HASHES.json` (gemma3 `1299c11d…`, gemma2 `61a7b147…`).

| tokenizer | encoding | tokens / syllable | single-token share | boundary alignment | onset–rime split | tone mark isolated |
|---|---|---|---|---|---|---|
| Gemma 3 (262,144 BPE, normalizer = identity, byte fallback) | NFC | 1.78 | 29.5% | 0.94 | 51.5% | 0% |
| Gemma 3 | NFD | 2.90 | 7.3% | 0.27 | 6.8% | 41.6% |
| Gemma 2 (256,000 BPE, normalizer = identity, byte fallback) | NFC | 1.77 | 29.9% | 0.94 | 50.7% | 0% |
| Gemma 2 | NFD | 2.90 | 7.3% | 0.29 | 4.4% | 50.4% |

Normalization census (500 multi-syllable words from the word list): NFD input produces
the same token ids as NFC input for 3.4% of words only (those without diacritics); the
NFD sequence is longer for 92–93%. Neither tokenizer normalizes Unicode, so the C1 arm
(NFC vs NFD) is a live intervention on Gemma models by construction. The SentencePiece
normalizer spec is `identity` in both model files.

Reading: under NFC, 70% of Vietnamese syllables are split by the Gemma tokenizers, and
when they are split the boundary is overwhelmingly at a linguistic boundary (alignment
0.94), most often between onset and rime (51.5% of all syllables have an onset|rime
split). Under NFD, the split moves inside letters (a boundary between a base letter and
its combining mark is never a linguistic boundary; alignment 0.27) and the tone mark
becomes a token of its own in 42% of syllables. This is the pattern H3 relies on: NFD
makes the tone an explicit symbol at the cost of longer, misaligned sequences.

Caveats: these are the Gemma 2/3 tokenizer FILES from google/gemma_pytorch; the Hugging
Face tokenizers used at inference must be checked against them (same ids on the same
strings) before any model result is joined to these statistics. Token counts here exclude
the special/BOS tokens and the leading space when it forms a token of its own.
