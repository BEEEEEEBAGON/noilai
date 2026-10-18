#!/usr/bin/env bash
# NóiLái: the laptop as the box. Runs the plan's CPU chunks (c01-c04: the Gate 1 pilot through the Hugging Face
# backend in float32 on the CPU, exactly as notebooks/kaggle_cpu_pilot.ipynb does on a Kaggle CPU session;
# RUNBOOK.md §6 "the laptop as the box") on this Mac, from the folder that holds noilai.bundle. No SSH, no GPU.
#   caffeinate -dims bash mac_run.sh                                             (default: gemma-3-1b-it; fits 8 GB)
#   MODELS="gemma-3-1b-it qwen3.5-2b phogpt-4b-chat" caffeinate -dims bash mac_run.sh      (a Mac with 24 GB or more)
# Environment: MODELS (panel names, space-separated; default gemma-3-1b-it), MACDIR (work dir; default ~/noilai-mac),
# RELEASE_DIR (the frozen release folder; found with Spotlight when unset), HF_TOKEN or the file ~/.noilai_hf_token
# (gated Gemma; never printed), MAC_HOURS (start no new line past this many hours), BRANCH (the bundle's branch).
# A model whose float32 weights do not fit the memory (about 4 GB per billion parameters, plus 2 GB) is skipped.
# Writes $MACDIR/mac_run.log. Raw outputs stay in $MACDIR/noilai/data/runs (private; never upload them); the hand-back
# for the assistant is ~/noilai-handback-<date>.zip (aggregates, ledger, pins, audits, compute log, the log; no item rows).
set -uo pipefail
MODELS="${MODELS:-gemma-3-1b-it}"
MACDIR="${MACDIR:-$HOME/noilai-mac}"
BRANCH="${BRANCH:-claude/adoring-planck-t3b4wb}"
DATE="${DATE:-$(date +%F)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$HERE/noilai.bundle"
START=$(date +%s)
mkdir -p "$MACDIR"
LOG="$MACDIR/mac_run.log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[mac] $(date -u +%FT%TZ) $*"; }
test -f "$BUNDLE" || { say "noilai.bundle is not next to this script (looked in $HERE)"; exit 1; }
HF_TOKEN="${HF_TOKEN:-$( [ -f "$HOME/.noilai_hf_token" ] && tr -d '[:space:]' < "$HOME/.noilai_hf_token" || true )}"
export HF_TOKEN
export HF_HOME="$MACDIR/hf"
export TOKENIZERS_PARALLELISM=false
say "start; models: $MODELS; token $([ -n "$HF_TOKEN" ] && echo set || echo NOT set); work dir $MACDIR"

# the code: a clone of the bundle (a second run refreshes it; data/runs is git-ignored and survives)
command -v git >/dev/null 2>&1 || { say "git is missing: run 'xcode-select --install' once, then this script again"; exit 1; }
if [ -d "$MACDIR/noilai/.git" ]; then
  git -C "$MACDIR/noilai" fetch -q "$BUNDLE" "$BRANCH" && git -C "$MACDIR/noilai" reset -q --hard FETCH_HEAD
else
  git clone -q --branch "$BRANCH" "$BUNDLE" "$MACDIR/noilai"
fi
cd "$MACDIR/noilai" || { say "$MACDIR/noilai is missing"; exit 2; }
say "code: $(git rev-parse --short HEAD)"

# the frozen item files (the plan's `release:` directory; manifest_private.json never leaves its two copies)
REL=$(sed -n 's/^release: *\([^ #]*\).*/\1/p' configs/run_plan.yaml | head -1)   # data/release/v0.4
SRC="${RELEASE_DIR:-}"
if [ -z "$SRC" ] && command -v mdfind >/dev/null 2>&1; then
  HIT=$(mdfind -name noilai_dev.jsonl 2>/dev/null | grep -E '/release/v0\.[0-9]+/noilai_dev\.jsonl$' | sort | tail -1 || true)
  [ -n "$HIT" ] && SRC="$(dirname "$HIT")"
fi
if [ -n "$SRC" ] && [ -f "$SRC/noilai_dev.jsonl" ]; then
  mkdir -p "$REL"
  rsync -a --exclude manifest_private.json "$SRC/" "$REL/"
  say "frozen item files: $SRC -> $REL"
else
  say "NO frozen item files found on this Mac: run again with RELEASE_DIR=/full/path/to/release/v0.4 in front"
  exit 1
fi

# Python >= 3.11 (scripts read pyproject.toml with tomllib); the macOS system python is older, so uv brings 3.12
if [ ! -x "$MACDIR/venv/bin/python" ]; then
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    python3 -m venv "$MACDIR/venv"
  else
    say "system python is $(python3 --version 2>&1): installing uv and Python 3.12"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 --seed "$MACDIR/venv" >/dev/null
  fi
fi
# shellcheck disable=SC1091
source "$MACDIR/venv/bin/activate"
say "python $(python --version 2>&1) at $(command -v python)"
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[eval,dev]" || { say "project install FAILED"; exit 3; }
python scripts/fetch_resources.py || { say "fetch_resources FAILED"; exit 3; }
python scripts/kaggle_run_plan.py --materialize || say "materialize: some derived files were skipped (a missing source)"
# the whole-panel pin: the pilot lines refuse an unpinned model (DD 7.1); needs HF_TOKEN for the gated Gemma entry
if ! python scripts/pin_panel.py --apply --partial --out experiments/panel_pins.json; then
  say "PIN FAILED or partial: an unpinned model is skipped on the pilot lines (the smoke line runs unpinned)"
fi

# memory: float32 weights need about 4 GB per billion parameters, plus about 2 GB for the system and activations
MEM_GB="${MEM_GB:-$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))}"
BATCH=8; [ "$MEM_GB" -le 8 ] && BATCH=2
RUN_MODELS=""
for m in $MODELS; do
  NEED=$(python - "$m" <<'EOF'
import sys, yaml
entries = {e["name"]: e for e in yaml.safe_load(open("configs/models.yaml"))["models"]}
print(round(4 * float(entries[sys.argv[1]]["params_b"]) + 2, 1))
EOF
)
  if python -c "import sys; sys.exit(0 if $NEED <= $MEM_GB else 1)"; then
    RUN_MODELS="$RUN_MODELS $m"; say "$m: needs ~$NEED GB of $MEM_GB GB: runs (batch size $BATCH)"
  else
    say "$m: needs ~$NEED GB of $MEM_GB GB: SKIPPED (a Mac with more memory, or the T4 chunk c05 on Kaggle)"
  fi
done
[ -n "$RUN_MODELS" ] || { say "no model fits this Mac's memory: nothing to run"; exit 4; }

# the jobs of the CPU chunks (configs/compute_chunks.yaml, tag cpu) for these models, in chunk order, each once
# shellcheck disable=SC2086
JOBS=$(python - $RUN_MODELS <<'EOF'
import sys, yaml
want, seen = sys.argv[1:], set()
for ch in sorted(yaml.safe_load(open("configs/compute_chunks.yaml"))["chunks"], key=lambda c: c["order"]):
    forced = int(bool((ch.get("params") or {}).get("FORCED_CHOICE")))
    for job in ch["jobs"]:
        if job.get("tag") != "cpu":
            continue
        for m in job["models"]:
            if m in want and (job["run"], m) not in seen:
                seen.add((job["run"], m))
                print(job["run"], m, forced)
EOF
)
echo "$JOBS" | while read -r RUN MODEL FORCED; do
  [ -n "$RUN" ] || continue
  ELAPSED_H=$(python -c "print(round(($(date +%s) - $START) / 3600, 2))")
  if [ -n "${MAC_HOURS:-}" ] && python -c "import sys; sys.exit(0 if $ELAPSED_H >= $MAC_HOURS else 1)"; then
    say "MAC_HOURS=$MAC_HOURS reached after $ELAPSED_H h: not starting $RUN on $MODEL (run again later; finished rows resume)"
    break
  fi
  # the CPU route = the HF backend in float32 with no accelerator (the entries carry t4 hardware: the documented
  # mismatch flag; the compute log records device "cpu" x 0); the pilot's forced choice is the chunk's parameter
  EXTRA="--backend hf --device cpu --dtype float32 --batch-size $BATCH"
  case "$RUN" in pilot_t1*) [ "$FORCED" = 1 ] && EXTRA="$EXTRA --t1-forced-choice" ;; esac
  UNPINNED=""
  case "$RUN" in smoke*) UNPINNED="--allow-unpinned-revision" ;; esac
  say "run $RUN on $MODEL ($ELAPSED_H h elapsed)"
  # shellcheck disable=SC2086
  python scripts/kaggle_run_plan.py --run "$RUN" --models "$MODEL" --platform local --session-hardware cpu \
      --allow-hardware-mismatch --chunk-tag cpu $UNPINNED --extra "$EXTRA"
  DIR="data/runs/${RUN}__${MODEL}__cpu"
  if [ -d "$DIR" ]; then
    # a fresh directory first gets the plain check (writes results_hashes.json), then every hash is recomputed
    { python scripts/check_run.py --run "$DIR" && python scripts/check_run.py --run "$DIR" --verify; } \
      || say "check_run reported a problem in $DIR (see above)"
  else
    say "$DIR was not written (skipped or failed: see the driver's status lines above)"
  fi
done
if grep -q 'revision not pinned' "$LOG"; then
  say "a pilot line was skipped because its model is not pinned: put the Hugging Face token in ~/.noilai_hf_token and run again"
fi

# count what came back here (the frozen files are here, so the hash gate works) and pack the hand-back
python scripts/ledger.py ingest --runs data/runs
python scripts/progress.py
python scripts/compute_log.py totals
OUT="$HOME/noilai-runs-$DATE"; HB="$OUT/handback"
rm -rf "$HB"; mkdir -p "$HB/audit"
cp -R experiments "$HB/experiments"
cp configs/models.yaml "$HB/models.yaml"
cp data/compute_log.csv "$HB/compute_log.csv" 2>/dev/null || true
cp data/audit/*.json data/audit/*_rows.csv "$HB/audit/" 2>/dev/null || true
cp STATUS.md "$HB/STATUS.md"
cp "$LOG" "$HB/mac_run.log"
rm -f "$HOME/noilai-handback-$DATE.zip"
(cd "$OUT" && zip -qr "$HOME/noilai-handback-$DATE.zip" handback)
say "DONE after $(python -c "print(round(($(date +%s) - $START) / 3600, 2))") h"
say "raw outputs: $MACDIR/noilai/data/runs (private: keep them, never upload them)"
say "hand-back for the assistant: $HOME/noilai-handback-$DATE.zip (upload it to the chat)"
