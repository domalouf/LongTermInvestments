#!/usr/bin/env bash
#
# Weekly: bring in whatever the SEC has published since the last run and rebuild
# the fundamentals from it, so the nightly list and the track record rank on
# current filings rather than on the day the server was set up.
#
#   lti update --force      new SEC data sets + the standardization pipeline (only
#                           the new quarter is processed; minutes, not hours)
#   lti refresh-tickers     newly listed companies, renamed tickers
#   lti build-fundamentals  fundamentals.parquet and quarterly.parquet (the TTM rows)
#   lti fetch-prices        full history for any ticker the cache hasn't seen
#
# A week with nothing new from the SEC rebuilds the same tables. The last good
# fundamentals stay beside the new ones as *.prev.parquet, and a rebuild that
# comes out more than 2% smaller is thrown away for them.
#
# Config via environment (all optional):
#   LTI_VENV   virtualenv dir (default: <repo>/venv)
#
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${LTI_VENV:-$repo/venv}"
derived="$repo/data/derived"

log() { printf '==> %s  %s\n' "$(date -u +%FT%TZ)" "$*"; }

# shellcheck disable=SC1091
source "$venv/bin/activate"
cd "$repo"

# shared with publish-undervalued.sh: the nightly job waits for this to finish
exec 9>"$repo/data/.lti.lock"
flock -w 3600 9 || { log "another lti job still holds data/.lti.lock — giving up"; exit 1; }

rows() {  # rows in a parquet file, 0 if it isn't there
  python -c 'import sys, os, pyarrow.parquet as pq; p = sys.argv[1]; print(pq.ParquetFile(p).metadata.num_rows if os.path.exists(p) else 0)' "$1"
}

log "checking the SEC for new data sets"
lti update --force

log "refreshing the ticker map"
lti refresh-tickers || log "ticker refresh failed — building on the cached map"

before="$(rows "$derived/fundamentals.parquet")"
for t in fundamentals quarterly; do
  [[ -f "$derived/$t.parquet" ]] && cp -p "$derived/$t.parquet" "$derived/$t.prev.parquet"
done

restore() {
  for t in fundamentals quarterly; do
    [[ -f "$derived/$t.prev.parquet" ]] && cp -p "$derived/$t.prev.parquet" "$derived/$t.parquet"
  done
}

log "rebuilding the fundamentals ($before rows before)"
if ! lti build-fundamentals; then
  log "build failed — putting the previous tables back"
  restore
  exit 1
fi
after="$(rows "$derived/fundamentals.parquet")"
if (( after * 100 < before * 98 )); then
  log "rebuild has $after rows against $before before — putting the previous tables back"
  restore
  exit 1
fi
log "fundamentals: $before -> $after rows"

log "fetching prices for tickers the cache hasn't seen"
lti fetch-prices || log "price fetch failed — the nightly refresh only tops up cached tickers; rerun \`lti fetch-prices\`"

log "done"
