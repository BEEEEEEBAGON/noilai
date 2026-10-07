# BLOCKED — in urgency order (experiments and validation workstreams)

Updated 2026-10-07. Each item names who can unblock it and what it blocks.

1. **Frozen item files are not here.** `data/release/v0.3/` holds only `manifest.json`; the item files
   (`noilai_dev/test/core/main/c2/attested.jsonl`) are git-ignored and live in the author's private
   Kaggle dataset. Without them nothing can be verified against the plan's hashes here, no pilot file
   can be materialized, the validation sample and human-baseline forms cannot be generated, and no
   ingest can be re-scored locally. **Author:** either allow `www.kaggle.com` in this environment's
   network settings and provide a Kaggle API token (`KAGGLE_API_TOKEN` or `~/.kaggle/access_token`)
   so the assistant downloads the dataset, or run the Kaggle side by hand (`RUNBOOK.md`) and hand
   back aggregates. Blocks: every confirmatory unit, the pilot, validation sampling, the baseline forms.
2. **No Kaggle access from this machine.** `www.kaggle.com` is denied by the environment's network
   policy (proxy 403 on CONNECT) and no token exists. The `kaggle` CLI is installed. **Author:** edit
   the cloud environment (title-bar menu → Edit → Network access: add `www.kaggle.com`,
   `huggingface.co` and `cdn-lfs.huggingface.co` under Allowed domains, package managers left on) and
   add the token. Blocks: launching, polling and downloading notebooks from here; until then
   `RUNBOOK.md` is the launch path.
3. **Hugging Face is unreachable.** Only the Gemma 3 and Gemma 2 tokenizer files are local, so the
   tokenizer-only measurements for the other eleven panel tokenizers (Qwen 3.5/3.8, Gemma 4, SEA-LION,
   Sailor2, PhoGPT, Vistral, Llama 3.1, o200k) and the panel pinning (`scripts/pin_panel.py`) must run
   on Kaggle CPU (`scripts/kaggle_cpu_jobs.py all`, RUNBOOK §3). Blocks: E2's item audits
   (`data/audit/items_*.jsonl`, the H1 identifiability floor), the census, the pinned panel manifest.
4. **Pre-registration not registered.** `docs/PREREGISTRATION.md` has both commit fields `<fill>`;
   the stage-1 hash must be a commit of THIS repository (the manifest's `f94f655` is the original
   repository's hash; `docs/MIGRATION.md` maps it to `62f63ae`). **Author:** commit stage 1, record the
   link and date in `experiments/gates.yaml`. Blocks: every confirmatory run (gate).
5. **Author decisions that change items, prompts or gold:** the `qu` convention (DD 13.3) and the
   `i/y` emission rule (DD 13.5). **Author:** record both in `experiments/gates.yaml`. Blocks: every
   confirmatory run (gate). The Gemini route (DD 13.18) blocks the API lane.
6. **Panel not pinned.** Every self-hosted `revision` is null and 11 of 21 ids are `uncertain`;
   `run_eval.py` refuses a non-smoke run without a pinned revision. **Author on Kaggle CPU:**
   `python scripts/kaggle_cpu_jobs.py pin` then hand back `experiments/panel_pins.json`; the assistant
   applies it with `python scripts/pin_panel.py --apply --from experiments/panel_pins.json`.
   Blocks: the panel freeze (PREREG §7), every confirmatory run.
7. **Kaggle quota unverified.** `experiments/quota.yaml` assumes 30 GPU-h and 20 TPU-h per week with a
   Saturday reset and does not know whether CPU sessions count. **Author:** read the quota page once
   and edit the file. Blocks: nothing hard; the "hours left" line of STATUS.md is an assumption.
8. **Validators and respondents not recruited** (3 validators, ~20 respondents; DD 10.1–10.2); the
   attested set has 34 rows, 0 native-verified, against the 100-row H6 floor. **Author.** Blocks: the
   human lane, H6, the Table 1 α row.
9. **Analysis code the DD names as "to write"**: `noilai/stats/pooling.py`, `aggregate.py`,
   `analyze.py` + `configs/analysis.yaml`, release encryption, weighted validation sampling. Not a
   blocker for runs; a blocker for the pooled estimates. Assistant work, scheduled after the pilot.
