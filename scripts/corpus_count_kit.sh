#!/usr/bin/env bash
# The C2 tone-placement corpus count (DESIGN_DECISIONS 6.3; memo rows 2 and N12): before Gate 1, count the old (hòa)
# and new (hoà) placement conventions in two registers of >= 10M syllable tokens each. Wikipedia alone is not enough
# (DD 6.3: it measures a style guide), so the kit counts a Vietnamese Wikipedia dump AND the CC-100 Vietnamese web sample.
#
#   bash scripts/corpus_count_kit.sh                # both corpora, LINES lines of each
#   LINES=4000000 bash scripts/corpus_count_kit.sh  # more lines if a corpus stays under 10M syllable tokens
#
# Runs anywhere with the repository, Python and internet (a laptop or a Kaggle CPU session; no GPU). Only the first
# LINES lines are streamed: neither corpus is downloaded whole. Measured speed of scripts/count_placement.py on the
# build machine: ~24,000 syllable tokens a second on one core (1.92M tokens in 79 s, 1 October 2026), so ~7 minutes per
# 10M tokens plus the download. The source URLs are the standard ones [UNCERTAIN: verify they are still served]; the
# script records each file's Last-Modified date next to the counts. Afterwards: add a docs/RESULTS_LOG.md entry naming
# both corpora, their dates and the two `old_share` values (DD 6.3 fixes H4's direction from them).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/audit/placement_corpus}
LINES=${LINES:-2000000}
PY=${PY:-python}
WIKI_URL=${WIKI_URL:-https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2}
CC100_URL=${CC100_URL:-https://data.statmt.org/cc-100/vi.txt.xz}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

stream() {  # name url decompressor
  local name=$1 url=$2 dec=$3
  echo "[$name] $url (first $LINES lines)"
  curl -sIL "$url" | grep -i '^last-modified' | tail -1 > "$OUT/$name.last_modified.txt" || true
  # head closes the pipe after LINES lines; curl and the decompressor then stop on SIGPIPE, which is expected here
  curl -sL --fail "$url" | $dec | head -n "$LINES" > "$TMP/$name.txt" || true
  local n
  n=$(wc -l < "$TMP/$name.txt")
  if [ "$n" -eq 0 ]; then echo "[$name] nothing downloaded: check the URL / network"; return 1; fi
  "$PY" scripts/count_placement.py "$TMP/$name.txt" --out "$OUT/$name.json" > /dev/null
  "$PY" - "$OUT/$name.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
n = d["n_syllable_tokens"]
print(f"  syllable tokens {n:,}; affected {d['affected_tokens']:,}; old {d['old']:,}; new {d['new']:,}; "
      f"old_share {d['old_share']:.4f}" if d.get("old_share") is not None else f"  syllable tokens {n:,}; no affected token")
if n < 10_000_000:
    print("  WARNING: under the 10M syllable tokens of memo row 2: re-run with a larger LINES")
EOF
}

stream viwiki "$WIKI_URL" "bzcat"
stream cc100_vi "$CC100_URL" "xzcat"
echo "counts in $OUT/{viwiki,cc100_vi}.json; Last-Modified dates beside them"
