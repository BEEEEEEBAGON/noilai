# Exploratory analysis: T1 multi-distractor forced choice (pre-specified 1 October 2026, before any model run)

**Status: exploratory, not a pre-registered hypothesis test.** It is specified here, and committed, before any model
has produced a single output, so that its definition cannot follow the data. It is logged in `docs/DEVIATIONS.md`
(row "exploratory T1 forced choice"). `docs/PREREGISTRATION.md` is not edited: §8.14 already lists what is exploratory,
and this analysis is reported under that heading. It never enters a hypothesis decision, a Holm family, or a sentence
that compares it with generated accuracy (DESIGN_DECISIONS 5.3: scoring modes are reported in separate columns).

## Why

The Gate 1 go/no-go (PREREG §10) asks whether one pilot model's strict T1 accuracy exceeds the copy baseline by 15 points.
If small models score at or near 0% on generated T1, that number carries no graded information: "0% everywhere" cannot
separate a model that has no access to sub-syllabic structure from one that has some but cannot produce the string.
Forced choice asks a weaker question of the same prompt: given the rule's answer and the near misses a model would make,
does the model put the most probability on the rule's answer? It is the T1 analogue of the pre-registered `t3_pair_lp`
(DESIGN_DECISIONS 5.3) with a larger, rule-built candidate set.

## Definition (code: `noilai/eval/forced_choice.py`; enabled by `scripts/run_eval.py --t1-forced-choice`)

For a T1 item with input *a b* and named kind *v*, the candidate set is built by the rule engine, never by a model:

| label | candidate |
|---|---|
| `gold` | the rule output in the named order |
| `wrong_variant:Vk` | the output of each other swap kind (V1–V6), dropped when it equals the gold or the gold in the other order (that is the lenient answer, DESIGN_DECISIONS 3.2) |
| `copy` | the input |
| `reversal` | the input in the other order |
| `spelling` | the gold misspelled by the c/k, g/gh or ng/ngh rule, when the gold has such a trigger; scored but **excluded from the headline set** (like the T3 spelling twins, DD 5.3) |

Candidates are deduplicated after canonicalization (DD 2.7). The scoring context is the rendered T1 prompt (same
paraphrase, shots, arm and scope as the generation row) plus the answer marker `Đáp án:` (re-encoded with the arm under
the whole-prompt scope); every candidate is a space-led continuation, re-encoded with the arm. This is exactly the
boundary convention of `t3_pair_lp` (DESIGN_DECISIONS 7.3 amendment). A continuation that violates the prefix property
scores `None` and drops out of that item's comparison. Strip arms are not scored (stripped candidates collide).

Per item (`noilai.eval.forced_choice.fc_metrics`):

- `fc_correct`: the gold's summed log-probability is strictly higher than every other headline candidate's (a tie is
  not a win);
- `fc_correct_mean`: the same with the per-token mean log-probability (candidates differ in token count; the length
  confound is stated next to the numbers, DD 5.3 item 40);
- `fc_rank`, `fc_n` (headline set size), `fc_chance = 1 / fc_n`;
- `fc_spelling_win`: the gold beats its misspelling (own column).

Reported per model × arm × variant: forced-choice accuracy with the base-pair cluster-bootstrap CI (DD 8.2) beside the
mean chance level, and the summed and per-token versions side by side (`scripts/check_run.py` writes them to
`stats.json` with the label EXPLORATORY). Read as: "above chance" = the CI's lower bound exceeds the mean chance level.

## What it can and cannot say

- It can show graded sensitivity to sub-syllabic structure when generation is at floor, and whether the NFD arm moves
  that sensitivity (paired by item, the same way as generated accuracy, but reported separately).
- It cannot rescue the Gate 1 criterion: the go/no-go stays the pre-registered generated-accuracy rule. A model that
  fails Gate 1 but is above chance here is reported as such, in the exploratory section.
- Log-probability scoring needs open weights; API models have no forced-choice column.

## Cost

One extra forward pass per candidate (≈ 6–8 per item) on the prompt length of the item; on a GPU this is small next to
generation; on CPU it roughly triples the time per item (see `docs/COMPUTE_PLAN.md`).
