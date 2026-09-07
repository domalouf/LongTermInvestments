# Deployment

How the investing tools reach the web.

| Piece | Who sees it | Where it runs | How |
| --- | --- | --- | --- |
| **Undervalued today** — daily list | public | static files on the Pi (`domalouf.com/invest/`) | `publish-undervalued.sh` + `lti-undervalued.timer` |
| **Full Streamlit GUI** (screener, backtest, …) | just you | the always-on server (laptop) | `lti-streamlit.service` + `cloudflared-invest.service` + Cloudflare Access |

The Pi is 32-bit ARM and cannot run the Python stack (Streamlit hard-depends on
`pyarrow`, no wheels), so all compute happens on the **server** (the laptop) and
only static HTML/JSON/CSV is copied to the Pi.

---

## Nightly public snapshot

`publish-undervalued.sh` does three things:

1. `lti refresh-prices` — tops up the price cache with the last few days of bars
   (cheap; `fetch-prices` is only needed after the universe grows).
2. `lti undervalued --out build/invest` — regenerates `index.html`,
   `undervalued.json`, `undervalued.csv`.
3. `rsync` those to the Pi.

### One-time setup — on the Pi

```bash
ssh pi 'mkdir -p ~/HealthBoard/piStuff/website/invest'
# keep `git pull --ff-only` deploys clean — the snapshot is not in git:
ssh pi 'cd ~/HealthBoard && grep -qxF "piStuff/website/invest/" .gitignore || echo "piStuff/website/invest/" >> .gitignore'
```

`piStuff/website/` is the nginx web root (bind-mounted to `/usr/share/nginx/html`
by `piStuff/docker-compose.yml`), so `https://domalouf.com/invest/` serves
`~/HealthBoard/piStuff/website/invest/index.html` with no nginx change. Add a
`location /invest/` block only if you want custom cache headers.

Then point the "Investments" card on the landing page at `/invest/`. The landing
page lives in its own repo — `github.com/domalouf/MyWebsite` (`~/Projects/MyWebsite`),
deployed with that repo's `deploy/deploy.sh`.

### One-time setup — on the server (laptop)

```bash
cd ~/Projects/LongTermInvestments
git pull && pip install -e .            # picks up refresh-prices + the renderer

# smoke-test by hand first (writes to the Pi):
./deploy/publish-undervalued.sh
#   or dry-run locally:  LTI_PI_DEST="$PWD/build/_test/" ./deploy/publish-undervalued.sh

# install the timer as a user service
mkdir -p ~/.config/systemd/user
cp deploy/lti-undervalued.{service,timer} ~/.config/systemd/user/
#   edit ExecStart in the .service if the repo is not at ~/Projects/LongTermInvestments
systemctl --user daemon-reload
systemctl --user enable --now lti-undervalued.timer

# run the job even when you're logged out
sudo loginctl enable-linger "$USER"
```

Requires `ssh pi` to work non-interactively for the laptop's user (key in
`~/.ssh/config`, same as the main PC — see the `pi-deployment` note).

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
| `LTI_PI_DEST` | `pi:HealthBoard/piStuff/website/invest/` | rsync target |
| `LTI_UNDERVALUED_ARGS` | — | extra `lti undervalued` flags, e.g. `--min-models 4 --market-cap-min 2000` |
| `LTI_SKIP_PRICES` | — | `1` to skip the price refresh |

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
`https://invest.domalouf.com`, reachable from anywhere but gated to your email by
Cloudflare Access. The app itself has **no login** — Cloudflare does the auth, so
unauthenticated traffic never reaches Streamlit.

```
browser ──TLS──▶ Cloudflare edge ──▶ Access policy (your email only)
                                       │
                                       ▼
                             cloudflared tunnel (laptop)
                                       │
                                       ▼
                        streamlit  127.0.0.1:8501  (laptop)
```

This is a **separate** Cloudflare tunnel from the Pi's — the Pi keeps serving
`domalouf.com`, and the public `/invest/` snapshot above is independent, so the
GUI can be down without affecting it.

### One-time — Cloudflare dashboard

1. **DNS / tunnel** is created by the CLI below (`cloudflared tunnel route dns`).
2. **Access application** — [one-time-PIN, no IdP needed]:
   Zero Trust dashboard → Access → Applications → *Add* → *Self-hosted*
   - Application domain: `invest.domalouf.com`
   - Session duration: e.g. 24h
   - Policy: *Allow*, Include → *Emails* → `malouf.dominic@gmail.com`
   - Leave the login method as the default one-time PIN (email code).

### One-time — laptop server

```bash
# 1. cloudflared  (Arch: `sudo pacman -S cloudflared`, or the official binary)
cloudflared tunnel login                       # pick the domalouf.com zone
cloudflared tunnel create invest               # note the UUID it prints
cloudflared tunnel route dns invest invest.domalouf.com

cp deploy/cloudflared-invest.yml ~/.cloudflared/config.yml
#   edit: set <TUNNEL-UUID> and <USER> (twice)

# 2. systemd user services
cp deploy/lti-streamlit.service deploy/cloudflared-invest.service ~/.config/systemd/user/
#   edit ExecStart/WorkingDirectory paths if the repo isn't at ~/Projects/LongTermInvestments
systemctl --user daemon-reload
systemctl --user enable --now lti-streamlit.service
systemctl --user enable --now cloudflared-invest.service
sudo loginctl enable-linger "$USER"             # (already done if the timer is installed)
```

The Streamlit server config lives in the repo at `.streamlit/config.toml`
(loopback bind, headless, dark theme) — it applies to local `streamlit run` too.

### Check it

```bash
systemctl --user status lti-streamlit.service cloudflared-invest.service
curl -sf http://127.0.0.1:8501/_stcore/health && echo " streamlit ok"
cloudflared tunnel info invest
# from a browser: https://invest.domalouf.com  -> Cloudflare email-code prompt -> app
```

### Data on the server

The GUI reads `data/derived/fundamentals.parquet` and `data/prices/adj_close.parquet`.
The nightly `lti-undervalued` job keeps prices fresh; rebuild fundamentals
quarterly (see above). The app degrades gracefully if a file is missing (the Home
page shows what's absent).

### Updating the app

```bash
cd ~/Projects/LongTermInvestments && git pull && pip install -e .
systemctl --user restart lti-streamlit.service
```
