#!/usr/bin/env bash
# Run on the laptop from the folder that holds noilai.bundle (the code, sent by the assistant), e.g. ~/Downloads:
#   bash laptop_start_nogit.sh <minutes of rental left>      (HF_TOKEN=hf_... and HF_ENDPOINT=... in front if you have them)
# No repository clone is needed on the laptop: the box clones the bundle. The frozen item files are found with
# Spotlight (or given as RELEASE_DIR=/path/to/release/v0.3); without them the box runs only what needs none.
set -euo pipefail
BOX="${BOX:-root@js4.blockelite.cn}"
PORT="${PORT:-14024}"
MINUTES_LEFT="${1:-90}"
BRANCH="${BRANCH:-claude/adoring-planck-t3b4wb}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$HERE/noilai.bundle"
test -f "$BUNDLE" || { echo "noilai.bundle is not next to this script (looked in $HERE)"; exit 1; }
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
BOXDIR=$($SSH "$BOX" 'for d in /workspace /data /root; do if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; break; fi; done')
echo "box work directory: $BOXDIR"
$SSH "$BOX" "df -h $BOXDIR | tail -1; command -v git >/dev/null || dnf -y -q install git || apt-get -y -q install git"
scp -P "$PORT" "$BUNDLE" "$BOX:$BOXDIR/noilai.bundle"
$SSH "$BOX" "rm -rf $BOXDIR/noilai && git clone -q --branch $BRANCH $BOXDIR/noilai.bundle $BOXDIR/noilai && cp $BOXDIR/noilai/scripts/box/box_run.sh $BOXDIR/box_run.sh && echo code: \$(git -C $BOXDIR/noilai rev-parse --short HEAD)"
REL="${RELEASE_DIR:-}"
if [ -z "$REL" ] && command -v mdfind >/dev/null 2>&1; then
  HIT=$(mdfind -name noilai_dev.jsonl 2>/dev/null | grep '/release/v0.3/noilai_dev.jsonl$' | head -1 || true)
  [ -n "$HIT" ] && REL="$(dirname "$HIT")"
fi
if [ -n "$REL" ] && [ -f "$REL/noilai_dev.jsonl" ]; then
  echo "frozen item files: $REL"
  rsync -a -e "ssh -p $PORT" --exclude manifest_private.json "$REL/" "$BOX:$BOXDIR/release_v0.3/"
else
  echo "NO frozen item files found on this Mac: the box runs what needs none (the XCOPA pilot, the pin, the tokenizer audits)."
  echo "If you know where data/release/v0.3/ is, run again with RELEASE_DIR=/that/path in front."
fi
$SSH "$BOX" "cd $BOXDIR && HF_TOKEN='${HF_TOKEN:-}' HF_ENDPOINT='${HF_ENDPOINT:-}' BOXDIR=$BOXDIR nohup setsid bash $BOXDIR/box_run.sh $MINUTES_LEFT > $BOXDIR/box_run.log 2>&1 < /dev/null & sleep 3; head -5 $BOXDIR/box_run.log"
echo
echo "Started. Watch it with:   ssh -p $PORT $BOX tail -f $BOXDIR/box_run.log"
echo "When it prints '[box] DONE' (or 10 minutes before the rental ends):   bash $HERE/laptop_finish_nogit.sh"
