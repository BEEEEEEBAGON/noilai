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
RS="rsync -az -e ssh\ -p\ $PORT"
cd "$(git rev-parse --show-toplevel)"
BOXDIR=$($SSH "$BOX" 'for d in /workspace /data /root; do if [ -d "$d/noilai" ]; then echo "$d"; break; fi; done')
OUT="$HOME/noilai-runs-$DATE"
mkdir -p "$OUT"
$RS "$BOX:$BOXDIR/noilai/data/runs/" "$OUT/runs/"
$RS "$BOX:$BOXDIR/noilai/data/compute_log.csv" "$BOX:$BOXDIR/box_run.log" "$BOX:$BOXDIR/cuda_box.log" "$BOX:$BOXDIR/audits.log" "$OUT/" || true
echo "raw outputs at $OUT (never commit them)"
git fetch origin main
git checkout -B "box-run-$DATE" origin/main
$RS "$BOX:$BOXDIR/noilai/configs/models.yaml" configs/models.yaml
$RS "$BOX:$BOXDIR/noilai/experiments/" experiments/
$RS --include='*.json' --include='*_rows.csv' --exclude='*' "$BOX:$BOXDIR/noilai/data/audit/" data/audit/
$RS "$BOX:$BOXDIR/noilai/STATUS.md" STATUS.md
cp "$OUT/compute_log.csv" data/compute_log.csv
mkdir -p data/box_logs && cp "$OUT"/box_run.log "$OUT"/cuda_box.log "$OUT"/audits.log data/box_logs/ 2>/dev/null || true
git add configs/models.yaml experiments data/compute_log.csv STATUS.md data/box_logs
git add data/audit/*.json data/audit/*_rows.csv 2>/dev/null || true
git commit -m "box run $DATE: tier 0 hand-back (aggregates, ledger, pins, audits, compute log, driver logs)" || echo "nothing new to commit"
git push -u origin "box-run-$DATE"
echo "pushed branch box-run-$DATE: tell the assistant"
