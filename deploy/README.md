# Deployment

How the investing tools reach the web.

| Piece | Who sees it | Where it runs | How |
| --- | --- | --- | --- |
| **Undervalued today** — daily list | public | static files in the site's web root (`domalouf.com/invest/`) | `publish-undervalued.sh` + `lti-undervalued.timer` |
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
   A failure is logged, not fatal. Set `LTI_TRACK_BACKUP` to an rsync destination
   to keep a copy off this machine — the record can't be rebuilt after the fact.
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

```bash
cd ~/Projects/LongTermInvestments
git pull && pip install -e .            # picks up refresh-prices + the renderer

# smoke-test by hand first (writes to the live web root):
./deploy/publish-undervalued.sh
#   or dry-run locally:  LTI_DEST="$PWD/build/_test/" ./deploy/publish-undervalued.sh

# install the timer as a user service
mkdir -p ~/.config/systemd/user
cp deploy/lti-undervalued.{service,timer} ~/.config/systemd/user/
#   edit ExecStart in the .service if the repo is not at ~/Projects/LongTermInvestments
systemctl --user daemon-reload
systemctl --user enable --now lti-undervalued.timer

# run the job even when you're logged out
sudo loginctl enable-linger "$USER"
```

Run the job on one machine only: each copy keeps its own append-only track record,
and two of them diverge.

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
| `LTI_TOP_N` | `40` | rows published |
| `LTI_DEST` | `~/site/www/invest/` | rsync target |
| `LTI_UNDERVALUED_ARGS` | — | extra `lti undervalued` flags, e.g. `--min-models 4 --market-cap-min 2000` |
| `LTI_SKIP_PRICES` | — | `1` to skip the price refresh |
| `LTI_TRACK_BACKUP` | unset (commented out in the shipped unit) | where to copy `data/track/` after each record; unset, it stays on this machine only |

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
