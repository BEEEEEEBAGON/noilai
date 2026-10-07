# Zero-cost routes for the API runs, and draft credit applications (1 October 2026)

**Status: nothing on this page was re-verified on 1 October 2026.** The provider pages (console.groq.com,
ai.google.dev, openrouter.ai, kaggle.com) are blocked by the build machine's network proxy, and this session's web-search
budget was spent on the attested-example sweep. Every limit and term below is the one recorded on 30 September in
`docs/PLAN_2026-09-30.md` §3 and `configs/models.yaml` `providers`, with that day's source; each carries
[UNCERTAIN: verify]. Reading them is a person's job (BLOCKED.md); the run notebook already refuses a provider whose daily
cap is not configured (`ALLOW_UNCAPPED_API = False`).

Rules that bind every route: no money, no card on file, no sign-up made by this session (the author or an adult
account holder signs up, named by role only, DD 11.2); test-split items only to providers that do not train on inputs
(DD 11.2: `terms.trains_on_inputs: false`, enforced by `kaggle_run_plan.build_command`); never the sealed split; never
annotators' personal data.

## 1. What the API lines need (from `configs/run_plan.yaml`; plan estimates)

| line | items | calls | tokens | models |
|---|---|---|---|---|
| `E1_api_core` | core, 1,496 | 6,000 | 3.0 M | gemini-flash, gemini-flash-lite, gpt-oss-120b, gpt-oss-20b |
| `E1_api_paraphrase` | para300, 296 × p1, p2 | 2,400 | 1.2 M | same four |
| `E3_api_core` | para300 × nfd, placement_new | 2,400 | 1.2 M | same four |
| `reasoning_500_api` | reasoning500 | 1,500 | 1.5 M | gpt-oss-120b/20b `--thinking`, gemini-flash `--thinking` |

Per Groq model (equal split of each line over its models): 1,500 + 600 + 600 + 500 = **3,200 calls and 1.85 M tokens**.
All four lines read test-split files, so they run only after the stage-2 commit (DD 8.8).

## 2. Routes

| route | models | free limits as recorded 30 Sep [UNCERTAIN: verify] | trains on inputs? | minimum age [UNCERTAIN] | may see test items? | verdict |
|---|---|---|---|---|---|---|
| **Groq free plan** | gpt-oss-120b, gpt-oss-20b (and thinking variants) | 1,000 requests and 200 K tokens a day, 30 RPM, 8 K TPM; per model (plan) or per org (DD 7.2) | no ("no training and no default retention", DD 11.2) | not recorded (memo row 17) | yes (core, para300, reasoning500) | **the core route** |
| Gemini API unpaid tier | Flash, Flash-Lite | shown only in AI Studio; a third-party 1,500 RPD is unverified | **yes** (prompts and responses improve Google's products; human review) | 18 (Gemini API terms) | **no** (DD 11.2) | dev-only, or drop (memo rows 8 / 18) |
| Gemini paid key with data-use opt-out | Flash, Flash-Lite | — | opt-out | 18 | yes, with the opt-out recorded | **excluded: costs money** |
| OpenRouter free models | `:free` variants | 50 requests a day; 1,000 after a one-time $10 purchase | per downstream provider; `data_collection: deny` required (DD 11.2) | not recorded | only with `deny` + ZDR recorded | 50/day is ~30 days for one model's core; the $10 tier is **excluded: costs money** |
| GitHub Models, Cohere trial keys, HF monthly credit | various | "too small to matter" (plan §3) | varies | varies | — | not used |
| Kaggle / Colab open-weight copies | gpt-oss-20b on the TPU [UNCERTAIN: fits and runs under vLLM-TPU] | the TPU quota | no (self-hosted) | Kaggle's | yes | **fallback** if Groq's catalogue changes; it becomes a self-hosted run, labelled as such |

**Days on Groq** (derived from the table above, at the 200 K-token daily cap; requests are not binding: 3,200 calls
< 4 days of 1,000): **9.25 days per model** if the caps are per model; **18.5 days for both** if they are per org.
Between a stage-2 commit on 8 November and the results freeze on 22 November there are 14 days: enough under the
per-model reading, **not enough under the per-org reading**, which then needs the earlier stage-2 commit of memo row
N13 or the thinking line moved after the freeze (it is an appendix line, DD 12.36). Catalogue risk: Groq retired Llama
from its free plan on 16 August 2026 (plan §3); pin the model ids and save raw outputs immediately (the notebook does).

**Gemini under zero budget.** With the paid key excluded, Gemini can see only dev-derived items. The design's
`noilai_api_dev` set (DD 4.5, "reported separately and never in Table 2") is **not yet an item file** in
`configs/run_plan.yaml`, so today the driver refuses every Gemini call on the API lines, which is the safe default. If
you choose the dev-only route (memo row 8 / 18), the set needs a `derive` block over `noilai_dev` and a run line; if
you drop Gemini, the panel has 19 models and the paper names the drop (PREREG: a model "may be dropped only under rule
5.5"; this is a pre-freeze panel decision, before 25 October).

## 3. Credit programs (none applied for by this session; eligibility not claimed)

From the plan's §3 sweep of 30 September (sources there):

| program | what | fit | timing | action |
|---|---|---|---|---|
| OpenAI Researcher Access Program | up to $1,000 API credits, 12 months | "limited financial and institutional resources"; "fairness and representation in language models" focus | reviews in March, June, September, December; decisions 4–6 weeks | **draft below**; credits most likely arrive after 4 Jan, so they pay for author-response / camera-ready runs, not the submission |
| Anthropic External Researcher Access | $1,000 | safety and alignment; student eligibility unstated | first Monday of each month (5 Oct, 2 Nov, 7 Dec) | **not recommended**: no truthful safety angle in this project |
| Google TPU Research Cloud | TPU time | bills for the VM and storage | — | not zero-cost |
| Google Cloud research credits | credits | faculty, postdocs, PhD students | — | not eligible unless a co-author applies |
| NAIRR pilot | allocations | US institution and institutional e-mail | — | not eligible |
| Modal academic credits | credits | graduate students and researchers; card on file | — | excluded (card) |
| Microsoft AFMR | — | ended | — | — |

### Draft: OpenAI Researcher Access Program (fill the brackets; the form's fields could not be read)

> **Project title.** NóiLái: can subword language models manipulate the onset, rime and tone inside Vietnamese
> syllables?
>
> **Summary (≈150 words).** We release a rule-generated, automatically verifiable benchmark of nói lái, a native
> Vietnamese wordplay that swaps rimes, onsets or tones between syllables, and use it to ask whether models can operate
> on units their tokenizers never expose. Because Vietnamese diacritics can be written precomposed or as combining
> characters, the same text can be re-tokenized without changing its meaning; the benchmark exploits this as a natural
> experiment that separates tokenization from knowledge. The design, analysis plan and compute plan are pre-registered
> in a public repository before any test-split run. [One sentence on the pilot, only once it exists.]
>
> **Why API access.** Open-weight models run on free notebooks; hosted frontier models are the reference point the
> paper needs and cannot be run there. Credits would add [model names] on the 1,496-item core set and its 296-item
> paraphrase subset: [N calls, N tokens, from `configs/run_plan.yaml`].
>
> **Data handling.** Test items are released gated with a canary string; they are sent only to endpoints whose terms
> exclude training on inputs; no personal data is sent.
>
> **Timeline.** Submission to ACL Rolling Review, 4 January 2027; credits would serve the response and camera-ready
> period. **Applicant.** [name, role, affiliation or "independent researcher", as true]. **Resources today.** [as
> true: e.g. "no institutional compute; free notebook tiers only"].

Nothing here states eligibility; the applicant checks the program's current criteria and age requirement before
filing.
