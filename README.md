# LongTermInvestments

Analyze the **SEC Financial Statement Data Sets** to screen US stocks on value/quality
metrics and backtest simple long-term strategies (e.g. *"the 10 stocks with the lowest
average P/E and debt/equity, rebalanced annually"*), with a Streamlit GUI for browsing,
sorting and filtering the data.

- **Fundamentals:** [`secfsdstools`](https://github.com/HansjoergW/sec-fincancial-statement-data-set)
  (bulk SEC filings, 2009–present, 10-K filers).
- **Prices:** [`yfinance`](https://github.com/ranaroussi/yfinance) adjusted close, cached locally.
- **GUI:** Streamlit.

> ⚠️ **Survivorship bias.** The CIK→ticker map (`sec.gov/files/company_tickers.json`) only
> lists *currently listed* issuers, and Yahoo often lacks delisted names. Backtests can only
> trade the survivors, so real-world returns for a value screen are typically **worse** than
> shown. The app surfaces the affected counts on every run rather than hiding them.

## Setup

Requires **Python 3.12** (the `secfsdstools` stack pins `numpy<2`). This repo pins the
interpreter via `mise.toml`.

```bash
mise install python@3.12         # if not already present
python -m venv venv              # run from the repo dir so mise picks 3.12
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"          # installs the `lti` CLI + package (+ pytest)
```

Run the unit tests (pure logic, no network or SEC data needed):

```bash
pytest -q
```

Configuration for `secfsdstools` is generated automatically: importing `lti` renders
`.secfsdstools.cfg` from `.secfsdstools.cfg.template` (absolute `data/` paths, your SEC
`User-Agent` email) and points `SECFSDSTOOLS_CFG` at it. Override the email with
`LTI_USER_AGENT_EMAIL=...` if needed.

## Usage

```bash
# 1. One-time: download all SEC quarterly data + build the index.
#    Without LTI_SMOKE this also runs the multi-hour standardization pipeline.
lti update --force
#    (if the download already ran and you only need to re-run the pipeline: `lti pipeline`)

# 2. Build the flat fundamentals table (data/derived/fundamentals.parquet)
lti build-fundamentals
lti coverage

# 3. Map CIKs to tickers, then cache prices for the universe (resumable)
lti refresh-tickers
lti fetch-prices                 # thousands of tickers via yfinance — takes a while

# 4. Backtest a strategy
lti backtest --metrics pe,debt_to_equity --top-n 10 --start 2011-01-01

# 4b. See which metrics actually rank stocks by forward return
lti factor-ic --start 2012-01-01 --horizon 12 --step 12

# 5. GUI (see "GUI" section below)
streamlit run src/lti/app/Home.py
```

`lti progress` prints an ASCII progress bar for each pipeline stage (SEC data,
filter / standardize / concat, fundamentals, ticker map, prices) — handy for checking
on the multi-hour full build. `lti fetch-prices` shows a live `tqdm` bar while running.

## GUI

```bash
source venv/bin/activate
streamlit run src/lti/app/Home.py          # opens http://localhost:8501
```

Navigate the pages from the sidebar. Leave `LTI_SMOKE` unset to use the full
`fundamentals.parquet`.

### 🏠 Home
Data-health dashboard — which artifacts exist, row counts, latest SEC quarter,
fundamentals coverage, filings-per-year chart, price-cache summary. No controls; if
something is missing it names the `lti` command to run.

### 🔎 Screener — rank the universe as of a date
Sidebar: **As of** (point-in-time date — only filings filed on/before it are used),
**Min market cap ($M)**, **Rank by** (one or more of `pe`, `pb`, `peg`, `earnings_yield`,
`debt_to_equity`, `current_ratio`, `roe`, `net_margin`, `gross_margin`, `fcf_margin`,
`revenue_growth_1y`, `eps_growth_1y`), **Top N**, **Require positive EPS**.
Body: ranked table (raw metric values + a `<metric>_pctile` per input when ranking by
more than one, plus `composite_score` = mean percentile-rank, lower = better), CSV
download, a **"Ranked metric values"** bar chart per metric (each pick's value labelled,
universe median marked; the old histogram is in a per-metric expander), and a
**"Fair-value estimates"** table running the intrinsic-value models (below) on the picks
with adjustable discount rate / max growth.

### 🧪 Backtest — simulate a strategy vs SPY
Sidebar builds the strategy (**Rank by**, **Top N**, **Start/End**, **Rebalance month**,
**Min market cap**, **Initial capital**); hit **Run backtest**.
Body: equity curve vs SPY (log toggle), tiles (strategy/SPY CAGR, max drawdown, Sharpe),
full stats table, a **survivorship-bias callout** with per-run delisting counts, the
per-period summary, a holdings expander (every pick, every period, + CSV), and a warnings
expander. Results are cached per exact config.

### 📐 Factor analysis — which metrics predict returns
Sidebar: **Metrics**, **Start/End**, **Forward-return horizon** (months), **As-of spacing**
(months — set ≥ horizon for non-overlapping, honest t-stats), **Min market cap**,
**Require positive EPS**, **Quantile buckets**, **Correlation** (spearman / pearson);
hit **Run analysis**.

For a grid of historical as-of dates the page takes a point-in-time snapshot (same
no-look-ahead path as the backtest), computes every metric and each stock's forward
return, then measures the **cross-sectional** correlation between metric and forward
return on that date — the *Information Coefficient* (IC). Per-date ICs are aggregated
into `mean_ic`, `ic_ir` (mean/std), `t_stat`, `hit_rate` (share of periods with the
dominant sign), `q_spread` (top-minus-bottom quantile forward return) and `monotonicity`.
Body: the sorted summary table, a mean-IC bar chart, the per-period IC time series for a
chosen metric, and mean forward return by metric quantile. **A negative mean IC means
lower values of the metric went with higher returns** (expected for `pe`, `pb`,
`debt_to_equity`). Univariate IC ignores that metrics are correlated with each other, and
the universe is survivorship-biased — see the caveats on the page. Also on the CLI as
`lti factor-ic`.

### 🔬 Stock detail — one company over time
Sidebar: **Ticker** (matches the primary symbol *and* the full `tickers_all` list, so
`JPM` resolves), **Log price axis**, **Mark 10-K filing dates**, **Split-adjust EPS /
book value**.
Body: eight tabs — **Price** (adjusted close with filing-date markers), **Income**
(revenue → net income bars + EPS), **Margins & returns** (gross / net / FCF margin, ROE),
**Balance sheet** (assets / liabilities / equity + debt-to-equity), **Cash flow**
(CFO / capex / FCF), **Valuation** (trailing P/E and P/B time series with a median line),
**Fair value** (intrinsic-value models, below, with a per-company 5-year CAGR growth
input and adjustable discount rate / terminal growth / DCF window), and **Raw data** (the
annual table + CSV). The Valuation and Fair-value tabs carry each 10-K's EPS and book
value forward from its filing date and restate them onto today's share count using the
split history pulled from Yahoo — without that, ratios across a split are wrong.

### Intrinsic-value models (`lti.valuation`)
`add_valuation_models()` turns a fundamentals snapshot + price into a fair value per
share for each of: **two-stage DCF** (FCF/share grown at the estimated rate for N years
then a Gordon terminal value, discounted at the required return), **Peter Lynch** (fair
P/E = earnings-growth % + dividend yield %), **Graham number** (√(22.5·EPS·BVPS)),
**Graham revised** (EPS·(8.5+2g)·4.4/Y), **DDM** (Gordon growth on dividends, perpetual
growth capped at the terminal rate) and **EPV** (no-growth capitalised earnings, EPS/r).
`fair_value_est` is the median of the models that produced a number; `*_upside` is
`fair value ÷ price − 1`. Growth defaults to a one-year figure clipped to `[0, cap]` —
crude; pass a multi-year `historical_cagr()` for a real estimate. These are rough,
assumption-sensitive estimates, not investment advice.

### Smoke mode

For a fast end-to-end check on a small subset, set `LTI_SMOKE=1`. `lti update` then skips
the standardization pipeline, and the derived artifacts use `*.smoke.parquet` names.

```bash
export LTI_SMOKE=1
lti update --force               # still downloads the full ~2–3 GB of SEC zips (one-time)
lti smoke                        # build-fundamentals --smoke, refresh-tickers, ~15 tickers, a backtest
streamlit run src/lti/app/Home.py
```

## Layout

```
src/lti/
  config.py        paths + one-time secfsdstools config (import side-effect)
  sec_update.py    wrappers around secfsdstools update / automation pipeline
  tickers.py       CIK <-> ticker map from SEC company_tickers.json
  fundamentals.py  build/load the flat fundamentals.parquet + coverage report
  prices.py        yfinance adjusted-close cache (wide parquet panel, resumable)
  metrics.py       P/E, P/B, PEG, debt/equity, ROE, margins, growth, ...
  valuation.py     intrinsic-value models: DCF, Lynch, Graham, DDM, EPV
  pit.py           point-in-time snapshots (no look-ahead)
  ranking.py       ScreenSpec + composite percentile-rank selection
  backtest.py      annual-rebalance engine
  performance.py   CAGR / drawdown / Sharpe / hit rate / turnover
  progress.py      `lti progress` per-stage pipeline dashboard
  cli.py           `lti` command-line entry point
  factor.py        cross-sectional IC of each metric vs forward return
  stock.py         one company's annual fundamentals + valuation time series
  app/             Streamlit: Home, Screener, Backtest, Factor analysis, Stock detail
tests/             pure-logic unit tests (no network / SEC data)
```

`data/` (gitignored) holds everything generated: `data/sec/` (secfsdstools),
`data/derived/` (fundamentals, ticker map), `data/prices/` (price cache).

## Known limitations / v2 ideas

- Annual (10-K) only; no quarterly rebalancing yet.
- "Debt/equity" = total liabilities / equity (not just interest-bearing debt).
- No transaction costs, slippage or taxes.
- Survivorship bias (see above) — a proper point-in-time delisting map needs paid data.
