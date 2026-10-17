#!/usr/bin/env bash
# NóiLái, tier 0 on a rented CUDA box in one go (RUNBOOK.md §6 steps 3-5, automated; started by laptop_start.sh).
#   bash box_run.sh <minutes of rental left>
# Environment: HF_TOKEN (optional, gated checkpoints), HF_ENDPOINT (optional, a provider's hub mirror),
# BOXDIR (the data disk; default /workspace). Writes box_run.log beside itself; keeps 12 minutes of the
# rental free so that laptop_finish.sh can copy the results home.
set -uo pipefail
MINUTES_LEFT="${1:-90}"
BOXDIR="${BOXDIR:-/workspace}"
START=$(date +%s)
export HF_HOME="$BOXDIR/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export TOKENIZERS_PARALLELISM=false
cd "$BOXDIR/noilai" || { echo "[box] $BOXDIR/noilai is missing: run laptop_start.sh first"; exit 2; }
echo "[box] $(date -u +%FT%TZ) start; $MINUTES_LEFT min of rental left; hub endpoint $HF_ENDPOINT; token $([ -n "${HF_TOKEN:-}" ] && echo set || echo NOT set)"

# Python >= 3.11 (scripts/cuda_box.py reads pyproject.toml with tomllib; vLLM 0.30 needs >= 3.10)
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  python3 -m venv "$BOXDIR/venv"
else
  echo "[box] system python is $(python3 --version 2>&1): installing uv and Python 3.12"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 --seed "$BOXDIR/venv" >/dev/null
fi
# shellcheck disable=SC1091
source "$BOXDIR/venv/bin/activate"
echo "[box] python $(python --version 2>&1) at $(command -v python)"

REL=$(sed -n 's/^release: *\([^ #]*\).*/\1/p' configs/run_plan.yaml | head -1)   # data/release/v0.4
mkdir -p "$REL"
if [ -f "$BOXDIR/release/noilai_dev.jsonl" ]; then
  cp -rn "$BOXDIR/release/." "$REL/"       # the frozen files; manifest.json from the archive stays
  FROZEN=1
else
  echo "[box] NO frozen item files under $BOXDIR/release: only the lines that need none run (the XCOPA pilot); the others are skipped by the item gate"
  FROZEN=0
fi
python -m pip install -q --upgrade pip
python -m pip install -q vllm==0.30.0 || { echo "[box] vLLM install FAILED"; exit 3; }
python -m pip install -q -e ".[eval,dev]" || { echo "[box] project install FAILED"; exit 3; }
python scripts/fetch_resources.py || { echo "[box] fetch_resources FAILED"; exit 3; }
python scripts/kaggle_run_plan.py --materialize || echo "[box] materialize: some derived files were skipped (a missing source)"
GPUS=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv); echo "$GPUS"
# consumer cards without NVLink can hang on NCCL peer-to-peer transfers when a model spans several cards
if echo "$GPUS" | grep -q GeForce; then
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
  echo "[box] GeForce cards: NCCL P2P and IB disabled for the multi-card models"
fi
df -h "$BOXDIR" | tail -1

# the whole-panel pin against the real hub (a mirror may not answer the API); needs HF_TOKEN for the gated entries
if ! HF_ENDPOINT=https://huggingface.co python scripts/pin_panel.py --apply --partial --out experiments/panel_pins.json; then
  echo "[box] PIN FAILED or partial: unpinned models are skipped on the pilot lines (smoke lines run unpinned)"
fi
# the hub audits on the CPU while the GPU works (data/audit/*.json, *_rows.csv, items_*.jsonl)
AUDIT_JOBS="audit-tokenizers audit-items"; [ "$FROZEN" = 1 ] || AUDIT_JOBS="audit-tokenizers"
# shellcheck disable=SC2086
( python scripts/kaggle_cpu_jobs.py $AUDIT_JOBS > "$BOXDIR/audits.log" 2>&1; echo "[box] audits rc=$?" >> "$BOXDIR/audits.log" ) &
AUDITS=$!

ELAPSED=$(( ($(date +%s) - START) / 60 ))
BUDGET_MIN=$(( MINUTES_LEFT - ELAPSED - 12 ))
BUDGET_H=$(python -c "print(round(max(0.05, $BUDGET_MIN / 60), 2))")
echo "[box] setup took $ELAPSED min; launching no new model after $BUDGET_H h"
python scripts/cuda_box.py --session auto --chunks c05_gpu_t0 c06_gpu_t0 c07_tpu_t0 --platform rented \
    --max-hours "$BUDGET_H" 2>&1 | tee "$BOXDIR/cuda_box.log"

# pilot models the pin could not resolve: run them once more as a rehearsal (manifests record revision null)
UNPINNED=$(grep -oE '^\[skip \] [^:]+: skipped: revision not pinned' "$BOXDIR/cuda_box.log" | sed -E 's/^\[skip \] ([^:]+):.*/\1/' | sort -u | tr '\n' ' ')
ELAPSED=$(( ($(date +%s) - START) / 60 ))
if [ -n "$UNPINNED" ] && [ $(( MINUTES_LEFT - ELAPSED )) -gt 20 ]; then
  BUDGET_H=$(python -c "print(round(max(0.05, ($MINUTES_LEFT - $ELAPSED - 12) / 60), 2))")
  echo "[box] rehearsal of the pilot on the unpinned models: $UNPINNED (DD 7.1: not a paper run)"
  # shellcheck disable=SC2086
  python scripts/cuda_box.py --session auto --chunks c05_gpu_t0 --models $UNPINNED --platform rented \
      --allow-unpinned-revision --max-hours "$BUDGET_H" 2>&1 | tee -a "$BOXDIR/cuda_box.log"
fi
wait "$AUDITS" 2>/dev/null; tail -3 "$BOXDIR/audits.log"

# count what came back, here (the frozen files are here, so the hash gate works); the laptop copies experiments/ home
python scripts/ledger.py ingest --runs data/runs
python scripts/progress.py
python scripts/compute_log.py totals
echo "[box] DONE $(date -u +%FT%TZ): now run laptop_finish.sh on the laptop"
