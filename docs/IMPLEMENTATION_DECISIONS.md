# Implementation decisions (30 September 2026)

Decisions taken while implementing the founding plan, each with the reason and the place in
the code that enforces it. `DESIGN_DECISIONS.md` (from the design review) is the binding
design document; where the two disagree, the disagreement is listed in its section 12 and
resolved there. Numbers in brackets are the plan's sections.

## Orthography and parsing (`noilai/vi/`)

1. **Canonical rime = glide + nucleus + coda; the glide belongs to the rime, not the onset.**
   `qu` is analysed as /k/ + glide (`quả` = c, w, a, hỏi), so `quả` and `hoa` share the rime
   w+a and V1 on `hoa quả` is the identity. School grammar treats `qu` as an onset and would
   give a different swap. Because the two analyses produce different outputs and no attested
   qu- example settles it, **qu- syllables are excluded from generated base pairs by default**
   (`Generator(exclude_qu=True)`) and will be added as a separate stratum only after the
   native check of 20 qu- pairs. Attested qu- examples are kept in the attested set.
2. **Short `a` before a semivowel coda is the nucleus `ă`** (`tay` = t, ă, j; `tai` = t, a, j;
   `tau` = t, ă, w; `tao` = t, a, w), spelled `a` again on output. This is the phonological
   analysis and it is what makes `thầy giáo → tháo giầy` come out with `ây`.
3. **Semivowel codas are canonical `j` and `w`**, spelled `y` after ă/â and `i` elsewhere,
   `o` after a/e and `u` elsewhere.
4. **`gi` is an onset that contracts a following `i`** (`gì` = gi + i, `gìn` = gi + in,
   `giếng` = gi + iêng). A rime that begins with `ê` after `gi` is read as `iê…`; `giê` alone
   does not parse (it is not a syllable).
5. **k/gh/ngh are triggered by the first written letter after the onset.** With a glide the
   trigger is off (`ngoe`, `nguy`, `Nguyễn`), never `nghoe`.
6. **i/y alternations**: standard output uses `i` after a consonant (`lí`, `kĩ`, `mĩ`), `y`
   after the glide and in the bare syllable (`quy`, `ý`, `yêu`); the attested variants `lý`,
   `quí`, `í` are accepted on input and recorded in `Parse.i_y_variant`. Scoring canonicalizes
   them (`lý` and `lí` compare equal).
7. **Tone-mark placement**: new style puts the mark on the nucleus letter (second letter of
   iê/yê/uô/ươ, first of ia/ua/ưa, second `o` of `oo`); old style differs only on the open rimes
   oa, oe, uy after a non-`qu` onset (hòa, khỏe, thủy). Verified against the two Hunspell lists:
   exactly the 69 syllables that differ between them are the ones the rule predicts.
8. **Placement conversion leaves non-standard i/y spellings untouched** (a `lý` in running
   text stays `lý`), so the C2 arm changes placement and nothing else.
9. **Stop codas (p, t, c, ch) take only sắc and nặng.** The parser accepts a toneless input
   (`bat`) as a query form; the inventory rejects entries with a level tone on a stop coda
   (`gip`, `têt`, `xit`, `hoc`) as phonotactically impossible loan spellings.
10. **Legality is attested, not generated.** A structure is legal at level `onset_rime` when its
    onset + glide + nucleus + coda occurs in the inventory with some tone and the tone respects
    rule 9. The inventory is the Hunspell vi_VN list (both placement styles) extended with
    word-list syllables that parse strictly, occur in at least two entries and use a rime the
    Hunspell base already attests (261 additions such as gẫy, nhếch, cược; no new rime types).
11. **Scoring normalization does not repair misspellings.** `canonical_text` parses in strict
    mode, so `mài céo` stays `mài céo` and scores as wrong; the scorer then classifies it as a
    spelling error (non-strict parse gives the right structure). Normalizing away c/k errors
    would erase the spelling-rule signal the benchmark exists to measure.

## Generator (`noilai/gen/`)

12. **Variants are the four component permutations that keep the syllables' slots filled;
    each is an involution**, tested on 500 random pairs. Swapping onsets only is V4 in the
    other order, and swapping onsets and tones is V1 in the other order, so they are not
    separate variants; T2 accepts either order of any variant's output.
13. **Base pairs** come from the two-syllable entries of the Viet74K word list (47,535 pairs;
    the list is a lookup, never redistributed) and from pseudo-pairs of two attested syllables.
    Lexical base pairs must have both syllables attested (drops word-list typos and loanwords).
14. **Items whose output equals the input (identity: equal tones under V3, equal rimes under
    V1) or contains an illegal syllable are dropped**, and the manifest records the pool size
    per cell before capping so that the drop rate is reportable.
15. **T2 gold = every lexical reading of the input under any variant in either order**; the
    base pair is always one of them (asserted). Inputs with several readings are kept and
    `n_readings` is a covariate.
16. **T3 items come in yes/no pairs with the same input**: the yes item's candidate is the gold,
    the no item's candidate is a twin of one type (`other_variant`, `onset`, `rime`, `tone`,
    `spelling`), balanced 50/50 by construction, with `pair_item_id` linking the two so that
    open models can also be scored by forced choice on log-probabilities. Spelling twins exist
    only when the gold contains a c/k, g/gh or ng/ngh trigger (a real misspelling that parses to
    the same structure); placement variants are never twins because both placements are valid.
17. **Splits are by base pair** (every item derived from one underlying pair is in one split);
    the core set is a balanced 125-per-cell subset of test (T3: 62 yes/no pairs); every test item
    carries the build's canary string; a build is deterministic given its seed.
18. **Attested examples are not all rule outputs.** Of the 22 seed rows, 16 are reproduced
    exactly by the engine; the others bend a vowel or a tone to reach a real word (`độc hại` for
    the rule's `đọc hại`; `cũ` for `củ`, the Southern hỏi/ngã merger) or swap a non-outer
    position pair in a three-syllable phrase. The release file records the engine's output,
    the matching position pair and both forms in `gold`; the paper's memorization analysis (H6)
    must compare attested items only with generated items that are pure rule outputs.

## Audit (`noilai/audit/`)

19. **Syllables are tokenized in running-text position** (preceded by a space), and a token
    that carries only the space is not counted. Boundary alignment is the share of token
    boundaries strictly inside the syllable that fall on onset|rime, glide|nucleus or
    nucleus|coda; under NFD a boundary between a base letter and its combining mark is never
    aligned. `qu` counts as onset `q` + rime `u…` for alignment; contracted `gi` as a 2-letter onset.
20. **The normalization census** compares token ids of NFC and NFD encodings of 500
    multi-syllable words; a tokenizer "normalizes NFD" only if every word tokenizes identically.

## Statistics (`noilai/stats/`)

21. **Every interval resamples base pairs**, never items; paired comparisons resample the same
    pairs for both conditions. McNemar uses the exact mid-p variant. Holm is applied within a
    declared family (all arms for one model × task; all models against the best model for one
    task).
22. **"Share of the effect mediated by token count" is replaced** by two well-defined
    quantities: the dose–response decomposition (form-only effect at Δtok = 0, slope per
    token, token-associated part and its share, with clustered CIs) and the matched contrast of
    items whose token count does versus does not change. The paper calls it a descriptive
    decomposition, not a causal mediation.
23. **The mixed model** is fitted by variational Bayes (`BinomialBayesMixedGLM`) with random
    intercepts for base pair and model, and checked with a GEE fit clustered by base pair.

## Probing (`noilai/probe/`)

24. **Probes are split by syllable identity** and reported with selectivity against a control
    task whose labels are random per syllable type; positions probed are the syllable's last
    sub-token and the first token after it; NFC and NFD are separate conditions.
25. **Patching pairs differ in one syllable's tone inside the V3 prompt** and are aligned so
    that the token sequences have equal length and differ only inside the target syllable; the
    logit difference is read at the first token where the two gold answers diverge (a shared
    answer prefix is teacher-forced). Recovery is normalized by the clean–corrupt gap.
