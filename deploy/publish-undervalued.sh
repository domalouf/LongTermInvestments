#!/usr/bin/env bash
#
# Nightly: refresh the price cache, regenerate the public "undervalued today"
# snapshot, and push the static files to the Pi that serves domalouf.com.
#
# Runs on the always-on server (the laptop), NOT the Pi — the Pi is 32-bit and
# can't run the Python stack. See deploy/README.md for one-time setup.
#
# Config via environment (all optional):
#   LTI_VENV         virtualenv dir                (default: <repo>/venv)
#   LTI_STAGING      local dir for generated files (default: <repo>/build/invest)
#   LTI_PI_DEST      rsync destination             (default: pi:HealthBoard/piStuff/website/invest/)
#   LTI_TOP_N        rows to publish               (default: 40)
#   LTI_SKIP_PRICES  set to 1 to skip the price refresh
#   LTI_UNDERVALUED_ARGS  extra flags for `lti undervalued` (e.g. "--min-profit-years 5")
#
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${LTI_VENV:-$repo/venv}"
staging="${LTI_STAGING:-$repo/build/invest}"
dest="${LTI_PI_DEST:-pi:HealthBoard/piStuff/website/invest/}"
top_n="${LTI_TOP_N:-40}"

log() { printf '==> %s  %s\n' "$(date -u +%FT%TZ)" "$*"; }

# shellcheck disable=SC1091
source "$venv/bin/activate"
cd "$repo"

if [[ "${LTI_SKIP_PRICES:-0}" != "1" ]]; then
  log "refreshing price cache"
  lti refresh-prices
fi

log "generating snapshot -> $staging"
rm -rf "$staging"
mkdir -p "$staging"
# shellcheck disable=SC2086
lti undervalued --top "$top_n" ${LTI_UNDERVALUED_ARGS:-} --out "$staging"

log "publishing -> $dest"
rsync -az --delete "$staging/" "$dest"

log "done"
