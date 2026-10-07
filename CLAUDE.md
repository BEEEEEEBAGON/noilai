# CLAUDE.md — session protocol for the experiments and validation workstreams

This file is read at the start of every session that works on experiments, compute, validation or
data. The paper workstream has its own rules in `paper/README.md` and `paper/claims.md`; sessions on
this workstream **never edit anything under `paper/`**.

## Resume recipe (every session, in this order)

1. Read `STATUS.md` (the meter, what is running, the quota week, the gates, the session log),
   `BLOCKED.md` (blockers in urgency order) and `experiments/ledger.csv` (one row per planned unit).
   If the ledger exists, do not rebuild it; `python scripts/ledger.py init --force` only when the
   plans changed, and it carries statuses over by `unit_id`.
2. Ingest: `python scripts/ledger.py ingest --runs data/runs` (finished outputs, hash gate, scores,
   aggregates copied into `experiments/aggregates/`), `make ingest-sheets` for returned human
   sheets, then `python scripts/progress.py` (rewrites the meter block of `STATUS.md`; never type a
   number into it by hand).
3. Launch the next units in `cut_rank` order per lane (`python scripts/ledger.py show --status planned`):
   Kaggle GPU / TPU through the notebooks (`RUNBOOK.md` when no Kaggle token is available; the
   `kaggle` CLI when `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` exists and `www.kaggle.com` is
   reachable), Kaggle CPU for hub-dependent jobs (`scripts/kaggle_cpu_jobs.py`), API lines only on
   providers whose terms do not train on inputs, the human lane through the forms.
4. Interim analysis on completed units only: `python scripts/interim_analysis.py` (labelled INTERIM
   with n; monitoring only; no design or analysis change without a `docs/DEVIATIONS.md` entry; no
   paper prose).
5. Update `STATUS.md` (hand sections), `BLOCKED.md`, commit `experiments/`, `STATUS.md`, `BLOCKED.md`,
   `data/compute_log.csv` and any script changes; never commit `data/runs/` or item-level outputs.

## Binding documents and gates

- `docs/DESIGN_DECISIONS.md` (experiment specs; the cut order and minimum viable panel in §7.1; the
  run order and the two-stage pre-registration in §8.8), `docs/PREREGISTRATION.md` (hypotheses,
  analysis plan, Gate 1 criterion in §10), `configs/run_plan.yaml` (the pre-registrable run matrix;
  do not edit it for exploratory work: dev-only exploratory lines live in
  `configs/run_plan_exploratory.yaml`), `configs/models.yaml` (the panel; `revision` must be pinned
  before any paper run), `docs/DESIGN_DECISIONS.md` §13 (the author-decision ledger).
- `experiments/gates.yaml`: no confirmatory run (any test-split or core item file, any non-smoke,
  non-pilot line) starts until the pre-registration's registration link and date and the author's
  decisions on the `qu` and `i/y` conventions are recorded; API runs also wait for the Gemini
  decision; the scoring rule is frozen (commit recorded) before anyone looks at confirmatory
  results. `scripts/kaggle_run_plan.py` enforces the gate when the file exists.
- Never send test or core items to an API tier whose terms allow training on inputs (Gemini free
  tier). Never use `--limit` outside a smoke line. Never rebuild the frozen release: the build seed
  is private; a non-frozen smoke build for exercising code lives outside `data/release/`.

## Where things live

- `experiments/ledger.csv` (units, cut rank, cost, status, result hashes), `experiments/aggregates/`
  (per-run `summary.json` and manifest excerpts, never item-level rows), `experiments/interim/`
  (interim analyses), `experiments/human/` (scored-sheet summaries, no names or e-mails; the item-level
  `baseline_scores.jsonl` goes to the git-ignored `data/human/scored/`), `experiments/quota.yaml`
  (quota assumptions until verified), `experiments/panel_pins.json` (resolved revisions).
- Raw item-level outputs: the private Kaggle dataset `noilai-runs`, mirrored under the git-ignored
  `data/runs/` on the machine that ingests them.
- `RUNBOOK.md`: click steps for running on Kaggle by hand and for handing results back.

## Lint and tests, as CI runs them

CI (`.github/workflows/tests.yml`) installs the newest `ruff` (0.16.10 on 7 Oct 2026, whose default
rule set includes C4, DTZ, ISC, FURB and RUF and excludes E402) and runs `python -m ruff check noilai
scripts tests`, then `python -m pytest -q -p no:cacheprovider -o addopts=""`. When the local ruff is
older, lint changed files with `ruff check --select ALL --ignore
D,ANN,T201,PLR,S,CPY,E501,FBT,PTH,TRY,EM,PLW,C901,PLC,ERA,N,ARG,B,UP,SIM,RET,INP,PERF,FURB,PT,PGH,A,DTZ005,G,TD,FIX,BLE001 <files>`
as well as with the default rules; import scripts in tests through `importlib.import_module` after the
`sys.path` insert (clean under both rule sets). `tests/test_cloud.py::test_readme_statements_are_true_against_the_tree`
requires every `tests/test_*.py` module to be named in the README's test list.

