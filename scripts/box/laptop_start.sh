#!/usr/bin/env bash
# Run on the laptop, inside the NóiLái repository folder that holds the frozen release files (the plan's `release:` directory):
#   bash laptop_start.sh <minutes of rental left>         (HF_TOKEN=hf_... in front of it if you have one)
# Copies the code of origin/main, the frozen item files and box_run.sh to the rented box, then starts
# box_run.sh there in the background. The box address lives in the three lines below.
set -euo pipefail
BOX="${BOX:-root@js4.blockelite.cn}"
PORT="${PORT:-14024}"
MINUTES_LEFT="${1:-90}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
cd "$(git rev-parse --show-toplevel)"
REL=$(sed -n 's/^release: *\([^ #]*\).*/\1/p' configs/run_plan.yaml | head -1)   # data/release/v0.4
test -f "$REL/noilai_dev.jsonl" || { echo "$REL/noilai_dev.jsonl is not here: run this inside the repository that holds the frozen files"; exit 1; }
git fetch origin main
BOXDIR=$($SSH "$BOX" 'for d in /workspace /data /root; do if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; break; fi; done')
echo "box work directory: $BOXDIR"
$SSH "$BOX" "df -h $BOXDIR | tail -1; mkdir -p $BOXDIR/noilai $BOXDIR/release"
git archive --format=tar origin/main | $SSH "$BOX" "tar -x -C $BOXDIR/noilai"
rsync -a -e "ssh -p $PORT" --exclude manifest_private.json "$REL/" "$BOX:$BOXDIR/release/"
scp -P "$PORT" "$HERE/box_run.sh" "$BOX:$BOXDIR/box_run.sh"
$SSH "$BOX" "cd $BOXDIR && HF_TOKEN='${HF_TOKEN:-}' HF_ENDPOINT='${HF_ENDPOINT:-}' BOXDIR=$BOXDIR nohup setsid bash $BOXDIR/box_run.sh $MINUTES_LEFT > $BOXDIR/box_run.log 2>&1 < /dev/null & sleep 2; head -3 $BOXDIR/box_run.log"
echo
echo "Started. Watch it with:   ssh -p $PORT $BOX tail -f $BOXDIR/box_run.log"
echo "When it prints '[box] DONE' (or 10 minutes before the rental ends):   bash $HERE/laptop_finish.sh"
