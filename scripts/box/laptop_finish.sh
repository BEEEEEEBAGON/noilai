#!/usr/bin/env bash
# Run on the laptop, inside the repository folder, when box_run.log says "[box] DONE" (or 10 minutes before
# the rental ends): copies the outputs home (outside the repository), then the aggregates, pins, audits,
# compute log and STATUS meter into a branch box-run-<date> and pushes it. The raw outputs under
# ~/noilai-runs-<date>/runs/ are never committed (item-level data stays off the repository).
set -euo pipefail
BOX="${BOX:-root@js4.blockelite.cn}"
PORT="${PORT:-14024}"
DATE="${DATE:-$(date +%F)}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
cd "$(git rev-parse --show-toplevel)"
BOXDIR=$($SSH "$BOX" 'for d in /workspace /data /root; do if [ -d "$d/noilai" ]; then echo "$d"; break; fi; done')
OUT="$HOME/noilai-runs-$DATE"
mkdir -p "$OUT"
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/data/runs/" "$OUT/runs/"
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/data/compute_log.csv" "$BOX:$BOXDIR/box_run.log" "$BOX:$BOXDIR/cuda_box.log" "$BOX:$BOXDIR/audits.log" "$OUT/" || true
echo "raw outputs at $OUT (never commit them)"
git fetch origin main
git checkout -B "box-run-$DATE" origin/main
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/configs/models.yaml" configs/models.yaml
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/experiments/" experiments/
# the per-model audits come home; the normalization census of record (gemma2.json, gemma3.json: RESULTS_LOG
# RL-2026-09-30-06, quoted in docs/ and the paper) stays as committed until the paper workstream adopts a re-census
rsync -az --rsh="ssh -p $PORT" --exclude='gemma2.json' --exclude='gemma3.json' --include='*.json' --include='*_rows.csv' --exclude='*' "$BOX:$BOXDIR/noilai/data/audit/" data/audit/
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/STATUS.md" STATUS.md
cp "$OUT/compute_log.csv" data/compute_log.csv
mkdir -p data/box_logs && cp "$OUT"/box_run.log "$OUT"/cuda_box.log "$OUT"/audits.log data/box_logs/ 2>/dev/null || true
git add configs/models.yaml experiments data/compute_log.csv STATUS.md data/box_logs
git add data/audit/*.json data/audit/*_rows.csv 2>/dev/null || true
git commit -m "box run $DATE: tier 0 hand-back (aggregates, ledger, pins, audits, compute log, driver logs)" || echo "nothing new to commit"
git push -u origin "box-run-$DATE"
echo "pushed branch box-run-$DATE: tell the assistant"
