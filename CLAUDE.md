# CLAUDE.md — session protocol for the experiments and validation workstreams

This file is read at the start of every session that works on experiments, compute, validation or
data. The paper workstream has its own rules in `paper/README.md` and `paper/claims.md`; sessions on
this workstream **never edit anything under `paper/`**. The run units are the 31 compute chunks of
`configs/compute_chunks.yaml` (one chunk = one Kaggle session, `docs/COMPUTE_PLAN.md`): one plan, one
ledger (`experiments/ledger.csv`), one runbook (`RUNBOOK.md`).

## Resume recipe (every session, in this order)

1. Read `STATUS.md` (the session log, the prerequisites, the gates, "Next chunk per queue", the open
   questions, then the generated meter block), `BLOCKED.md` (one list in deadline order, PAST DUE
   marked; what `docs/gate1/DD13_DECISION_MEMO.md` already answers points at its row) and
   `experiments/ledger.csv` (one row per planned unit, keyed by `unit_id` to its chunk). If the ledger
   exists, do not rebuild it; `python scripts/ledger.py init --force` only after the plan or the chunk
   packing changed (`plan_chunks.py --write`, a new census JSON, the `quota.yaml` CPU flag); statuses
   carry over by `unit_id`.
2. Ingest: `python scripts/check_run.py --run <dir> --verify` on every run directory that came back (a
   directory without `results_hashes.json` first gets a plain `--run`), then `python scripts/ledger.py
   ingest --runs data/runs` (tag-aware: a `<run>__<model>__<tag>` directory, tags `cpu`, `p0p1`, `p2`, maps
   to its tagged units; item-file hash gate, exit 1 on a mismatch; aggregates copied into
   `experiments/aggregates/`; every directory that matches no unit is reported as `unmatched`).
   Returned human sheets go ONE way: validators' sheets to `data/validation/returned/`, each Google
   Forms export through `python scripts/make_validation_forms.py import-responses --dir data/human`,
   then `make ingest-sheets` (= `validation-score` + `baseline-score`; both write nothing when nothing
   has come back; `ledger.py ingest` reads their two reports). Merge the returned `compute_log.csv` into
   `data/compute_log.csv` (`python scripts/compute_log.py totals`). Then `python scripts/progress.py`
   (rewrites the meter block of `STATUS.md`; never type a number into it).
3. Launch: the next chunk per queue is the "Next chunk per queue" table of `STATUS.md` (the lowest
   chunk order with a unit still to do; `python scripts/ledger.py show --status planned --queue <q>`
   lists its units), one Kaggle session per chunk through `notebooks/chunks/<id>.ipynb` (`RUNBOOK.md`
   §1): the CPU chunks `c01`–`c04` first (`c01` = the first real run, `c02`–`c04` the Gate 1 pilot;
   memo N10, N11), then `c06` / `c07` (the smoke week), then, after the stage-2 commit and with every
   gate recorded, `c08`–`c31` in chunk order; `c05` is the T4 alternative to `c02`–`c04`. On Kaggle only
   the account-holder settings of the parameters cell are edited (`ACCOUNT_HOLDER_ROLE`, `REPO_REF`,
   `CREATE_DATASET`, the `PUSH_*` settings); `JOBS`, `MODE`, `MODELS`, `RUN_ID(S)`, `RUN_LABEL`,
   `VERIFY_ITEM_KEYS`, `SMOKE_RUN_IDS`, `PLAN_PATH`, `FORCED_CHOICE` are the plan and change only through
   the generators (`scripts/plan_chunks.py`, `scripts/kaggle_build_notebooks.py`, `--write`, commit the
   notebooks). By hand through `RUNBOOK.md` when no Kaggle token is available; the `kaggle` CLI when
   `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` exists and `www.kaggle.com` is reachable. The hub
   jobs (tokenizer audits, per-item covariates, the whole-panel pin) run on a Kaggle CPU session through
   `scripts/kaggle_cpu_jobs.py` (`RUNBOOK.md` §3): `pin` → `experiments/panel_pins.json`; at home
   `python scripts/pin_panel.py --apply --from experiments/panel_pins.json` (`--dry-run` first;
   `--partial` while panel entries wait for a `quantization.checkpoint`), commit `configs/models.yaml`
   and the JSON together, `make bundle`; required before `c05` and before any non-smoke GPU / TPU line
   (panel freeze 25 Oct, memo row 9). API lines (`notebooks/api_runs.ipynb`, `RUNBOOK.md` §4) run once a
   day, only on providers whose terms do not train on inputs and only once the API gate is open; the
   human lane goes through the forms (`docs/gate1/`). The dev-only lines of
   `configs/run_plan_exploratory.yaml` are unscheduled alternatives outside the chunk order
   (`RUNBOOK.md` §2), run only when the author asks.
4. Interim analysis on completed units only: `python scripts/interim_analysis.py` (labelled INTERIM
   with n; monitoring only; no design or analysis change without a `docs/DEVIATIONS.md` entry; no
   paper prose).
5. Update `STATUS.md` (hand sections) and `BLOCKED.md`; commit `experiments/` (ledger, aggregates,
   human, `panel_pins.json`), `STATUS.md`, `BLOCKED.md`, `data/compute_log.csv`,
   `data/audit/validator_flags.json`, and any script change together with the notebooks it regenerates;
   never commit `data/runs/` or item-level outputs (`outputs.jsonl`, `scores.jsonl`, `items_*.jsonl`,
   `human_scores.jsonl`).

## Binding documents and gates

- `docs/DESIGN_DECISIONS.md` (experiment specs; the cut order and minimum viable panel in §7.1; the
  run order and the two-stage pre-registration in §8.8), `docs/PREREGISTRATION.md` (hypotheses,
  analysis plan, Gate 1 criterion in §10), `configs/run_plan.yaml` (the pre-registrable run matrix;
  do not edit it for exploratory work or quota readings: dev-only exploratory lines live in
  `configs/run_plan_exploratory.yaml`, quota numbers in `experiments/quota.yaml`), `configs/models.yaml`
  (the panel; `revision` must be pinned before any paper run), `docs/COMPUTE_PLAN.md` (the chunk plan:
  order, windows, hours; its tables are generated by `scripts/plan_chunks.py`), `docs/DESIGN_DECISIONS.md`
  §13 (the author-decision ledger) and `docs/gate1/DD13_DECISION_MEMO.md` (the author's record of the
  open decisions: cite its rows, never edit it).
- `experiments/gates.yaml`: no confirmatory run (any test-split or core item file, any non-smoke,
  non-pilot line; chunks `c08`–`c31` and the API lane) starts until the pre-registration's registration
  link and date and the author's decisions on the `qu` and `i/y` conventions are recorded; API runs
  also wait for the Gemini decision; the scoring rule is frozen (commit recorded) before anyone looks
  at confirmatory results. `scripts/kaggle_run_plan.py` enforces the gate when the file exists. Only
  the author fills the file; `scripts/freeze_scoring_rule.py --record` is the author's command, a
  session runs `--check` only.
- Never send test or core items to an API tier whose terms allow training on inputs (Gemini free
  tier). Never use `--limit` outside a smoke line. Never rebuild the frozen release: the build seed
  is private; a non-frozen smoke build for exercising code lives outside `data/release/`. Never edit
  `configs/compute_chunks.yaml`, a chunk notebook or any generated notebook by hand: edit the
  generator, `--write`, and run `python scripts/kaggle_build_notebooks.py --check` and
  `python scripts/plan_chunks.py --check` (`tests/test_cloud.py` refuses drift).

## Where things live

- `experiments/ledger.csv` (units with their chunk, order, tier, queue, cut rank, cost, status, result
  hashes), `experiments/aggregates/` (per run: `summary.json`, `stats.json`, `results_hashes.json`, a
  manifest excerpt; never item-level rows), `experiments/interim/` (interim analyses),
  `experiments/human/` (`human_baseline_report.json`: aggregates only, no names or e-mails),
  `experiments/quota.yaml` (quota assumptions until verified), `experiments/gates.yaml` (the gate),
  `experiments/panel_pins.json` (the whole-panel pin from `kaggle_cpu_jobs.py pin`; committed together
  with the `configs/models.yaml` it was applied to). `aggregates/`, `human/` and `interim/` are created
  by the first ingest, scoring and interim run.
- The chunk plan: `configs/compute_chunks.yaml`, `notebooks/chunks/c01_cpu_t0.ipynb` …
  `c31_tpu_t5.ipynb` and the generated block of `docs/COMPUTE_PLAN.md`, all written by
  `scripts/plan_chunks.py --write` from `configs/run_plan.yaml`, `configs/models.yaml` and the census
  JSONs under `data/audit/`; chunk ids renumber on a re-pack, so the ledger is keyed by `unit_id`.
- Raw item-level outputs: the private Kaggle dataset `noilai-runs`, mirrored under the git-ignored
  `data/runs/` on the machine that ingests them. Human material: `data/validation/` and `data/human/`
  (git-ignored; item-level `data/validation/report/validation_report.json` and
  `data/human/report/human_scores.jsonl`), with `data/audit/validator_flags.json` committed.
- `RUNBOOK.md`: click steps for running a chunk on Kaggle by hand (§1), the CPU hub jobs (§3), the API
  runs (§4) and for handing results back (§5).

## Lint and tests, as CI runs them

CI (`.github/workflows/tests.yml`) installs the newest `ruff` (0.16.10 on 7 Oct 2026, whose default
rule set includes C4, DTZ, ISC, FURB and RUF and excludes E402) and runs `python -m ruff check noilai
scripts tests`, then `python -m pytest -q -p no:cacheprovider -o addopts=""`. `python -m ruff` in this
environment is 0.16.10 now, so `python -m ruff check <files>` is the CI lint; lint changed files also
with `python -m ruff check --select E4,E7,E9,F,C4,DTZ,ISC,FURB,RUF,I,S102,EXE --ignore E402,E731 <files>`.
When the local ruff is older, fall back to `ruff check --select ALL --ignore
D,ANN,T201,PLR,S,CPY,E501,FBT,PTH,TRY,EM,PLW,C901,PLC,ERA,N,ARG,B,UP,SIM,RET,INP,PERF,FURB,PT,PGH,A,DTZ005,G,TD,FIX,BLE001 <files>`
as well as the default rules. Import scripts in tests through `importlib.import_module` after the
`sys.path` insert (clean under both rule sets). `tests/test_cloud.py::test_readme_statements_are_true_against_the_tree`
requires every `tests/test_*.py` module to be named in the README's test list. CI has no release item
files and no local smoke build: a test that reads either must skip cleanly without them.
