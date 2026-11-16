#!/usr/bin/env bash
#
# Tell someone. Usage: alert.sh "title" ["body"]
#
# Always: the journal (stderr) and a line in ~/lti-alerts.log.
# If there is a desktop session here: a critical desktop notification.
# If LTI_ALERT_URL is set: an HTTP POST of the text to it — an ntfy topic URL
#   (https://ntfy.sh/<a-topic-name-you-keep-private>) sends it to a phone.
#
# LTI_ALERT_URL can live in ~/.config/lti/alert.env (KEY=value lines), so the
# units need no editing.
#
set -uo pipefail

title="${1:-lti alert}"
body="${2:-}"
[[ -f "$HOME/.config/lti/alert.env" ]] && { set -a; source "$HOME/.config/lti/alert.env"; set +a; }

host="$(hostname)"
printf '!! %s  [%s] %s\n%s\n' "$(date -u +%FT%TZ)" "$host" "$title" "$body" >&2
printf '%s  [%s] %s | %s\n' "$(date -u +%FT%TZ)" "$host" "$title" "$(tr '\n' ';' <<<"$body")" >> "$HOME/lti-alerts.log"

if command -v notify-send >/dev/null && [[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
  notify-send -u critical -a "Long-term investments" "$title" "$body" || true
fi
if [[ -n "${LTI_ALERT_URL:-}" ]]; then
  curl -fsS -m 20 -H "Title: $title ($host)" -H "Priority: high" -d "${body:-$title}" "$LTI_ALERT_URL" >/dev/null \
    || echo "alert: POST to LTI_ALERT_URL failed" >&2
fi
exit 0
