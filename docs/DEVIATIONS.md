# Deviations log

`docs/DESIGN_DECISIONS.md` is binding from 30 September 2026 and the pre-registration
(`docs/PREREGISTRATION.md`, to be frozen in a dated commit before Gate 1) fixes the
hypotheses, metrics, exclusion rules, stopping rules and analysis code path. After the
pre-registration commit nothing in those documents changes by silent edit: every change
is a dated row here, with the reason, what it replaces, and whether any model output had
been seen when the decision was taken. A row that post-dates a model run is a post hoc
change and the paper says so.

| date | section changed | before | after | reason | model outputs seen? | commit |
|---|---|---|---|---|---|---|
| 2026-09-30 | DESIGN_DECISIONS 12.5 / 6.3 (C2 baseline) | plan: new-style placement canonical | old style is the stored baseline; C2 = old -> new; direction to be confirmed by a corpus count before Gate 1 | old style is the majority in text and in the Gemma 3 vocabulary (design review) | no | 975be27 |
| 2026-09-30 | DESIGN_DECISIONS 12.7 (mediation) | plan: "share of the effect mediated by token count" | per-model ATE, Δtokens dose–response labelled descriptive, three-arm contrast, tone-isolation DiD, encoding × spacing 2×2 | the mediator is a deterministic coarsening of the treatment; the estimand is undefined | no | 975be27 |
| 2026-09-30 | DESIGN_DECISIONS 12.1–12.4 (variants) | plan: four variants from a blog, "V3 Northern" | six documented variants, four generated cells, plain reversals dropped, region-neutral codes | six-type tradition in the Vietnamese literature; V2/V3 are the same unordered pair | no | 975be27 |

Rules for a row: one row per decision; "before" and "after" quote the text; the commit is
the one that applies the change in code or documents; a row added after a run names the run.
