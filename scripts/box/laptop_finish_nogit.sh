#!/usr/bin/env bash
# Run on the laptop (no repository clone needed) when box_run.log says "[box] DONE" or 10 minutes before the
# rental ends: copies the raw outputs to ~/noilai-runs-<date>/runs (private; never commit them) and zips the
# hand-back for the assistant (aggregates, ledger, pins, audits, compute log, logs; no item-level rows).
set -euo pipefail
BOX="${BOX:-root@js4.blockelite.cn}"
PORT="${PORT:-14024}"
DATE="${DATE:-$(date +%F)}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
BOXDIR=$($SSH "$BOX" 'for d in /workspace /data /root; do if [ -d "$d/noilai" ]; then echo "$d"; break; fi; done')
OUT="$HOME/noilai-runs-$DATE"
mkdir -p "$OUT/runs" "$OUT/handback/audit"
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/data/runs/" "$OUT/runs/" || true
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/experiments/" "$OUT/handback/experiments/" || true
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/configs/models.yaml" "$OUT/handback/models.yaml" || true
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/data/compute_log.csv" "$OUT/handback/compute_log.csv" || true
rsync -az --rsh="ssh -p $PORT" --include='*.json' --include='*_rows.csv' --exclude='*' "$BOX:$BOXDIR/noilai/data/audit/" "$OUT/handback/audit/" || true
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/noilai/STATUS.md" "$OUT/handback/STATUS.md" || true
rsync -az --rsh="ssh -p $PORT" "$BOX:$BOXDIR/box_run.log" "$BOX:$BOXDIR/cuda_box.log" "$BOX:$BOXDIR/audits.log" "$OUT/handback/" || true
rm -f "$HOME/noilai-handback-$DATE.zip"
(cd "$OUT" && zip -qr "$HOME/noilai-handback-$DATE.zip" handback)
echo "raw outputs: $OUT/runs  (private: keep them, never commit them)"
echo "hand-back for the assistant: $HOME/noilai-handback-$DATE.zip  (upload it to the chat, or paste handback/box_run.log)"
