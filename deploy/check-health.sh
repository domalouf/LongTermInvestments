#!/usr/bin/env bash
#
# Are the jobs alive and the data current? Alerts (deploy/alert.sh) and exits 1 if not.
#
# On the server (lti-health.timer): checks that neither job's last run failed,
# then `lti health` — prices, track record, fundamentals, rates, the published list.
#
# On another machine (lti-watch.timer), with LTI_HEALTH_HOST=lts: runs the same
# check over SSH and alerts *here* — a desktop notification — which also catches
# the one thing the server can't report: being off.
#
# Config via environment (all optional):
#   LTI_HEALTH_HOST   ssh host to check instead of this machine
#   LTI_REMOTE_REPO   the repo's path there      (default: Projects/LongTermInvestments)
#   LTI_VENV          virtualenv dir             (default: <repo>/venv)
#   LTI_DEST          the published web root     (default: ~/site/www/invest/)
#   --no-alert        print and exit 1, but leave the alerting to the caller
#
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
alert=1; [[ "${1:-}" == "--no-alert" ]] && alert=0

if [[ -n "${LTI_HEALTH_HOST:-}" ]]; then
  title="lti on $LTI_HEALTH_HOST needs a look"
  out="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$LTI_HEALTH_HOST" \
        "${LTI_REMOTE_REPO:-Projects/LongTermInvestments}/deploy/check-health.sh --no-alert" 2>&1)"
  rc=$?
  (( rc == 255 )) && out="could not reach $LTI_HEALTH_HOST over ssh — is it on?"$'\n'"$out"
else
  title="lti needs a look"
  out=""; rc=0
  for unit in lti-undervalued.service lti-fundamentals.service; do
    if systemctl --user is-failed -q "$unit" 2>/dev/null; then
      out+="FAIL  $unit: last run failed — journalctl --user -u $unit"$'\n'; rc=1
    fi
  done
  # shellcheck disable=SC1091
  source "${LTI_VENV:-$repo/venv}/bin/activate"
  dest="${LTI_DEST:-$HOME/site/www/invest/}"
  out+="$(cd "$repo" && lti health --published "${dest%/}/undervalued.json" 2>&1)" || rc=1
fi

echo "$out"
if (( rc != 0 )); then
  (( alert )) && "$repo/deploy/alert.sh" "$title" "$(grep -v '^ok ' <<<"$out" | head -12)"
  exit 1
fi
