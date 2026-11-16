# Deployment

How the investing tools reach the web.

| Piece | Who sees it | Where it runs | How |
| --- | --- | --- | --- |
| **Undervalued today** — daily list | public | static files in the site's web root (`domalouf.com/invest/`) | `publish-undervalued.sh` + `lti-undervalued.timer` |
| **Fundamentals** — the SEC filings everything ranks on | — | the server | `update-fundamentals.sh` + `lti-fundamentals.timer`, weekly |
| **Track-record backup** | — | a second machine | `lti-track-pull.timer` |
| **Alerts** — a failed job, stale data, a server that's off | — | the server, and a second machine | `alert.sh`, `check-health.sh`, `lti-health.timer`, `lti-watch.timer` |
| **Full Streamlit GUI** (screener, backtest, …) | just you | the always-on server (`lts`) | `lti-streamlit.service`, behind the site's admin sign-in |

Everything runs on the **server** (`lts`), which also serves domalouf.com (the site's
own stack, from the [MyWebsite](https://github.com/domalouf/MyWebsite) repo), so the
snapshot is copied straight into the local web root.

---

## Nightly public snapshot

`publish-undervalued.sh` does five things:

1. `lti refresh-prices` — tops up the price cache with the last few days of bars
   and re-fetches in full any ticker that split or paid a dividend since the last
   run (cheap; `fetch-prices` is only needed after the universe grows, or once to
   backfill a cache built before `close.parquet` existed).
2. `lti fetch-rates` — re-downloads the 10-year Treasury and AAA corporate yields
   from FRED (two small files), which the published list discounts at. A failure is
   logged, not fatal: the list values at the cached rates, or a fixed 9% without any.
3. `lti track-record` — writes down what each tracked strategy holds today
   (`data/track/records/<date>.parquet`, append-only; see the Track record page).
   A failure is logged, not fatal. A copy then goes to `LTI_TRACK_BACKUP`
   (`~/backups/lti-track/` in the shipped unit) — see "Backing up the track record".
4. `lti undervalued --out build/invest` — regenerates `index.html`,
   `undervalued.json`, `undervalued.csv`.
5. `rsync` those into `~/site/www/invest/`, the site's web root.

### One-time setup — the web root

```bash
mkdir -p ~/site/www/invest
```

`~/site/www/` is the site's web root (nginx serves it; see MyWebsite's README), a
plain directory rather than a git checkout, so `https://domalouf.com/invest/` serves
`~/site/www/invest/index.html` with no nginx change.

Then point the "Investments" card on the landing page at `/invest/`. The landing
page lives in its own repo — `github.com/domalouf/MyWebsite` (`~/Projects/MyWebsite`),
deployed with that repo's `deploy/deploy.sh`.

### One-time setup — the job

The server is `ssh lts` (Debian 13, x86_64). From a bare install it also needs
`git rsync curl` from apt and **Python 3.12** — trixie's system Python is 3.13 and
`python3.12` isn't packaged, so install it with mise, which fetches a precompiled
build and needs no compiler (that box has no gcc):

```bash
curl https://mise.run | sh && mise use -g python@3.12
git clone https://github.com/domalouf/LongTermInvestments.git ~/Projects/LongTermInvestments
#   the repo is public, so over HTTPS the server needs no key of its own to pull
cd ~/Projects/LongTermInvestments
"$(mise which python --tool=python@3.12)" -m venv venv && source venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e ".[dev]"
```

Copy the data over rather than rebuilding it — `lti update` is a multi-hour pipeline:

```bash
# from the machine that already has it:
rsync -a data/derived data/prices data/track lts:Projects/LongTermInvestments/data/
rsync -a data/sec lts:Projects/LongTermInvestments/data/   # 12 GB, quarterly rebuilds only
```

Then:

```bash
cd ~/Projects/LongTermInvestments
git pull && pip install -e .            # picks up refresh-prices + the renderer

# smoke-test by hand first (writes to the live web root):
./deploy/publish-undervalued.sh
#   or dry-run locally:  LTI_DEST="$PWD/build/_test/" ./deploy/publish-undervalued.sh

# install the timer as a user service
mkdir -p ~/.config/systemd/user
cp deploy/lti-undervalued.{service,timer} deploy/lti-fundamentals.{service,timer} \
   deploy/lti-health.{service,timer} deploy/lti-alert@.service ~/.config/systemd/user/
#   edit ExecStart in each .service if the repo is not at ~/Projects/LongTermInvestments
systemctl --user daemon-reload
systemctl --user enable --now lti-undervalued.timer lti-fundamentals.timer lti-health.timer

# run the job even when you're logged out
sudo loginctl enable-linger "$USER"
```

Run the job on one machine only: each copy keeps its own append-only track record,
and two of them diverge.

### Weekly: the fundamentals

The nightly job refreshes prices and rates, not filings. Without this second job the
list and the track record go on ranking the filings the server was set up with —
which is what happened from September to November 2026, on filings through June 30.
`update-fundamentals.sh` (`lti-fundamentals.timer`, Saturdays 14:00 UTC):

1. `lti update --force` — downloads any SEC data set published since the last run
   and runs the standardization pipeline on it. Only the new quarter is processed:
   a few minutes, though the concat step peaks near 14 GB of memory.
2. `lti refresh-tickers` — the SEC's current CIK→ticker map.
3. `lti build-fundamentals` — `fundamentals.parquet` and `quarterly.parquet` (the
   trailing-twelve-month rows from 10-Qs). The raw-tag caches beside them rebuild
   themselves when a quarter has arrived since they were written. The previous
   tables are kept as `*.prev.parquet`, and put back if the build fails or comes
   out more than 2% smaller.
4. `lti fetch-prices` — full history for tickers the cache hasn't seen.

The SEC publishes a quarter at a time, a few weeks after it ends, so most weeks
change nothing and the tables still trail today by one to four months. The public
page says how far (*SEC filings through …*), `undervalued.json` carries it as
`filings_through`, and each track record's sidecar stores it with the number of TTM
rows — so the record itself dates any change in what the strategies could see.

Both jobs take `data/.lti.lock`, so the nightly one waits rather than rank on a
half-written table.

```bash
systemctl --user start lti-fundamentals.service     # run now
journalctl --user -u lti-fundamentals.service -n 50 -f
```

### Backing up the track record

`data/track/` is the one thing here that can't be rebuilt: a record is only worth
anything because it was written on the day. Two copies are kept:

- **On the server**, the nightly job rsyncs it to `~/backups/lti-track/`
  (`LTI_TRACK_BACKUP` in the unit) — outside the checkout, so it survives a deleted
  or re-cloned repo, but not the disk.
- **On another machine**, `lti-track-pull.timer` pulls it over SSH once a day (and
  at next boot if the machine was off) into `~/backups/lti-track/`. Install it on any
  machine that can `ssh lts`:

  ```bash
  cp deploy/lti-track-pull.{service,timer} ~/.config/systemd/user/
  systemctl --user daemon-reload && systemctl --user enable --now lti-track-pull.timer
  systemctl --user start lti-track-pull.service && ls ~/backups/lti-track/records | tail -2
  ```

Neither copy uses `--delete`. Both are in the same house: for a copy elsewhere, point
a second `rsync` (or a private git remote) at either backup directory.

### Alerts: when something stops

Three things can go wrong without anyone noticing, and each has its own tripwire:

| What | Caught by |
| --- | --- |
| A job exits with an error | `OnFailure=lti-alert@%n.service` in both job units |
| A job "succeeds" on stale data — Yahoo down for a week, a record not written, the weekly rebuild not running | `lti health`, daily, via `lti-health.timer` on the server |
| The server is off, or its timers are | `lti-watch.timer` on another machine, which runs the same check over SSH |

`lti health` reads staleness off the files themselves and exits 1 if any check fails:

```
ok    prices                latest close 2026-11-13, 1 trading day behind
ok    track record          latest record 2026-11-13, 1 trading day behind
ok    fundamentals rebuild  last built 2026-11-15, 0 days ago (weekly)
ok    filings               through 2026-09-30, 46 days ago
ok    10-Q rows             quarterly.parquet present
ok    rates                 latest reading 2026-11-12, 3 days ago
ok    published list        generated 2026-11-15 23:32 UTC, 9 hours ago
```

Daily series may trail by three trading days (a long weekend isn't a missed night), the
fundamentals by nine days since their last rebuild and 125 since their latest filing —
the SEC publishes a quarter at a time — and the rates by ten.

Every alert goes through `deploy/alert.sh`, which always writes to the journal and to
`~/lti-alerts.log`, raises a **desktop notification** where there is a desktop session,
and POSTs to `LTI_ALERT_URL` when that is set. So out of the box the alerts reach you
through the second machine's desktop (`lti-watch.timer`, which also pulls nothing and
changes nothing on the server):

```bash
# on a machine that can `ssh lts` and has this repo at ~/Projects/LongTermInvestments
cp deploy/lti-watch.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now lti-watch.timer
```

For a push to a phone, pick a hard-to-guess [ntfy](https://ntfy.sh) topic, subscribe to
it in the ntfy app, and put it on the server:

```bash
mkdir -p ~/.config/lti && echo 'LTI_ALERT_URL=https://ntfy.sh/<your-topic>' > ~/.config/lti/alert.env
~/Projects/LongTermInvestments/deploy/alert.sh "test" "hello from lts"
```

### Check it

```bash
systemctl --user list-timers lti-undervalued.timer
systemctl --user start lti-undervalued.service      # run now
journalctl --user -u lti-undervalued.service -n 50 -f
curl -sI https://domalouf.com/invest/ | head -1
```

### Knobs

Set these as `Environment=` lines in `~/.config/systemd/user/lti-undervalued.service`
(see the script header for the full list):

| Var | Default | |
| --- | --- | --- |
| `LTI_TOP_N` | `20` | rows published |
| `LTI_DEST` | `~/site/www/invest/` | rsync target |
| `LTI_UNDERVALUED_ARGS` | — | extra `lti undervalued` flags, e.g. `--min-models 4 --market-cap-min 2000` |
| `LTI_SKIP_PRICES` | — | `1` to skip the price refresh |
| `LTI_TRACK_BACKUP` | `~/backups/lti-track/` (set in the shipped unit) | where to copy `data/track/` after each record; unset, no copy is made |

Schedule lives in the `.timer` (`OnCalendar=*-*-* 07:30:00 UTC`, `Persistent=true`
so a missed night runs at next boot).

### Quarterly

When a new SEC quarter lands, refresh the fundamentals on the server:

```bash
lti update && lti build-fundamentals && lti refresh-tickers && lti fetch-prices
```

---

## Private Streamlit GUI

The full GUI (screener, backtest, factor analysis, stock detail, undervalued) at
`https://invest.domalouf.com`, reachable from anywhere but only by the site's admin.
The app itself has **no login** — the site does the auth, so nobody else's traffic
ever reaches Streamlit.

```
browser ──TLS──▶ Cloudflare edge ──▶ the domalouf.com tunnel (lts)
                                       │
                                       ▼
                         the site's nginx: signed in as admin?
                          │ no: to domalouf.com/admin/ to sign in, then back
                          ▼ yes
                        streamlit  127.0.0.1:8501  (lts)
```

Signed in on domalouf.com is signed in here (the site's admin cookies cover every
subdomain), and the site-wide admin bar shows on the GUI's pages too. The
`invest.domalouf.com` server block, the tunnel's public hostname and the sign-in are
all the site's (MyWebsite: `server/nginx/site.conf`). This repo only runs Streamlit
on loopback.

### One-time — the server

```bash
cp deploy/lti-streamlit.service ~/.config/systemd/user/
#   edit ExecStart/WorkingDirectory paths if the repo isn't at ~/Projects/LongTermInvestments
systemctl --user daemon-reload
systemctl --user enable --now lti-streamlit.service
sudo loginctl enable-linger "$USER"             # (already done if the timer is installed)
```

The Streamlit server config lives in the repo at `.streamlit/config.toml`
(loopback bind, headless, dark theme) — it applies to local `streamlit run` too.

Before this, the GUI had a tunnel of its own and Cloudflare Access in front. If
those are still set up, the move is in MyWebsite's README ("Moving the site out of
HealthBoard"): the DNS record and Access application go, and
`systemctl --user disable --now cloudflared-invest && cloudflared tunnel delete invest`.

### Check it

```bash
systemctl --user status lti-streamlit.service
curl -sf http://127.0.0.1:8501/_stcore/health && echo " streamlit ok"
curl -sI https://invest.domalouf.com/ | grep -i -e '^HTTP' -e '^location'   # signed out: 302 to sign in
# from a browser, signed in at domalouf.com/admin/: https://invest.domalouf.com -> the app
```

### Data on the server

The GUI reads `data/derived/fundamentals.parquet` and the three price artifacts in
`data/prices/` (`adj_close.parquet`, `close.parquet`, `splits.parquet`).
The nightly `lti-undervalued` job keeps prices fresh; rebuild fundamentals
quarterly (see above). The app degrades gracefully if a file is missing (the Home
page shows what's absent).

### Updating the app

```bash
cd ~/Projects/LongTermInvestments && git pull && pip install -e .
systemctl --user restart lti-streamlit.service
```
