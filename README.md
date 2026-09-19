# LongTermInvestments

Analyze the **SEC Financial Statement Data Sets** to screen US stocks on value/quality
metrics and backtest simple long-term strategies (e.g. *"the 10 stocks with the lowest
average P/E and debt/equity, rebalanced annually"*), with a Streamlit GUI for browsing,
sorting and filtering the data.

- **Fundamentals:** [`secfsdstools`](https://github.com/HansjoergW/sec-fincancial-statement-data-set)
  (bulk SEC filings, 2009–present, 10-K filers).
- **Prices:** [`yfinance`](https://github.com/ranaroussi/yfinance) — total-return and split-adjusted
  close plus split history, cached locally.
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
lti fetch-prices                 # thousands of tickers via yfinance — takes about an hour
lti refresh-prices              # later: cheap daily top-up of already-cached tickers

# 4. Backtest a strategy (rebalanced each April; vs SPY and vs its own universe)
lti backtest --metrics pe,debt_to_equity --top-n 10 --start 2011-01-01
lti backtest --metrics pe,debt_to_equity --top-n 10 --all-months   # the same, once per rebalance month

# 4a. Greenblatt's Magic Formula (EBIT/EV + return on capital, no financials/utilities)
lti backtest --magic-formula --top-n 30 --start 2013-01-01

# 4b. See which metrics actually rank stocks by forward return
lti factor-ic --start 2012-01-01 --horizon 12 --step 12

# 4c. Today's most undervalued names by blended intrinsic value
lti undervalued --top 30 --market-cap-min 2000 --min-models 3
#     --format html|json|csv  prints that instead of a table
#     --out DIR               writes index.html + undervalued.{json,csv} (the public snapshot)

# 5. GUI (see "GUI" section below)
streamlit run src/lti/app/Home.py
```

`lti progress` prints an ASCII progress bar for each pipeline stage (SEC data,
filter / standardize / concat, fundamentals, ticker map, prices) — handy for checking
on the multi-hour full build. `lti fetch-prices` shows a live `tqdm` bar while running.

### Prices, splits and look-ahead

`data/prices/` holds three things, and each has one job:

- `adj_close.parquet` — close adjusted for splits **and dividends**: a total-return
  series. Used for **returns** only.
- `close.parquet` — close adjusted for **splits only**: the traded price on today's
  share basis. Used for **valuation** (P/E, P/B, market cap, EV, fair value).
- `splits.parquet` — every split (`ticker, date, ratio`), used to restate each 10-K's
  EPS and share count onto that same basis (`lti.pit.restate_per_share`).

That split matters. A 10-K reports per-share figures on the share count of the day it
was filed, while Yahoo back-adjusts prices for every split since. Pair the two as-is and
any company that split *later* looks cheaper by the split ratio — AAPL screened at a P/E
of 0.37 in January 2013 (really ~12), NVDA at 0.31, BKNG at 1.2. Companies split after
their stock has risen, so this is look-ahead bias that hands a cheapness screen the
future winners. On January rebalances, fixing it took the P/E + debt/equity top-10
backtest (2011–) from 16.0% to 6.9% a year, and the Magic Formula top-30 (2013–) from
12.7% to 9.4% — against 13.9% and 14.6% for SPY. The factor analysis deflated the same
way: ROIC's IC t-stat fell from 7.6 to 2.1, and P/B's from −4.2 to 0.0. Valuing on the
dividend-adjusted price would flatter past dividend payers too, which is why valuation
uses `close.parquet`.

### Share counts

A market cap needs a share count, and the standardized income statement only has one
when the filer shows its weighted-average shares on the face of the statement. The SEC
data sets carry nothing from the footnotes, and Alphabet, Procter & Gamble, Chevron,
Merck, Boeing and many more put it in the EPS footnote — which left 28% of companies with
$10B+ of revenue without a market cap, invisible to every priced screen. `lti.rawtags`
now also reads the balance-sheet count (tagged outstanding, else issued less treasury),
and `reconcile_shares` picks one count per filing from three sources: the reported
weighted average, the balance sheet, and net income ÷ EPS. That cuts the gap to 1.4%
(2.4% of companies with $1B+ of revenue); what's left is mostly filers with no ordinary
common equity (TVA, preferred-only listings) and Berkshire, whose EPS is per class.

Every source has errors — counts tagged in thousands or millions (Bruker's weighted
average is 146), placeholder zeros, EPS off by a million (Halliburton's 2,930,000) —
so a count is kept when a second source backs it up, a reported count that's the odd
one out is replaced, and where two counts sit a scale error apart with nothing to break
the tie, or EPS × shares is a power of 1000 away from net income, the values are left
NaN: a missing market cap drops a company from a screen, a wrong one sorts it to the
top. `shares_source` and `eps_source` record which rule applied; the standardized
values survive as `shares_reported` and `eps_reported`. Net income ÷ EPS only ever
confirms (net income is before preferred dividends), except for multi-class filers,
where it's the count in units of the share EPS is quoted for.

`lti fetch-prices` downloads full history for any ticker without a split-adjusted
series — so a cache from before `close.parquet` existed re-fetches every ticker once.
`lti refresh-prices` downloads a short recent window and compares the bars it shares
with the cache: Yahoo rewrites a ticker's whole history on a split (and the adjusted
close on every dividend), and splicing the new window onto the old history would put a
fake crash at the seam, so any ticker whose overlap doesn't match is re-fetched in full.

## GUI

```bash
source venv/bin/activate
streamlit run src/lti/app/Home.py          # opens http://localhost:8501
```

**Undervalued today is the landing page.** The sidebar groups the rest by what you're
there to do: *Find something to buy* (Undervalued, Stock detail), *Test an idea*
(Screener, Backtest, Factor analysis) and *Housekeeping* (Data health). Leave
`LTI_SMOKE` unset to use the full `fundamentals.parquet`.

`Home.py` is only the entry point — it sets the page config, applies the shared
stylesheet and defines the navigation. The views themselves live in
`src/lti/app/views/`, deliberately *not* in a `pages/` directory: Streamlit
auto-routes any `pages/` folder by filename, which serves a view directly and skips
the entry script, losing the grouped sidebar and the theme with it.

`src/lti/app/theme.py` holds the whole visual language — the categorical palette
(validated for colour-vision separation and contrast against the app's own dark
surface), the Plotly chrome, the page furniture. Charts should be built with
`theme.show()` rather than `st.plotly_chart` so they stay consistent.

To host it (private, email-gated at `invest.domalouf.com`) plus the public daily
list at `domalouf.com/invest/`, see [`deploy/README.md`](deploy/README.md).

### 🩺 Data health
Which artifacts exist, how far the fundamentals reach, per-field coverage of
everything the screens depend on, and how much of the universe has prices. No
controls; every gap names the `lti` command that fills it, and it states plainly
what share of the cached tickers are survivors.

### 🔎 Screener — rank the universe as of a date
Sidebar: **As of** (point-in-time date — only filings filed on/before it are used),
**Min market cap ($M)**, **Greenblatt Magic Formula** (a one-click preset, below),
**Rank by** (one or more of `pe`, `pb`, `peg`, `earnings_yield`, `ebit_ev`,
`debt_to_equity`, `current_ratio`, `roe`, `roic`, `net_margin`, `gross_margin`,
`fcf_margin`, `revenue_growth_1y`, `eps_growth_1y`), **Top N**,
**Require positive EPS**, and the **Exclusions** (operating companies only — no
commodity/crypto trusts or revenue-less shells — financials, utilities).
Body: ranked table (raw metric values + a `<metric>_pctile` per input when ranking by
more than one, plus `composite_score` = mean percentile-rank, lower = better), CSV
download, a **"Ranked metric values"** bar chart per metric (each pick's value labelled,
universe median marked; the old histogram is in a per-metric expander), and a
**"Fair-value estimates"** table running the intrinsic-value models (below) on the picks
with adjustable discount rate / max growth.

#### Greenblatt's Magic Formula (`ebit_ev` + `roic`)
The screen from *The Little Book that Beats the Market*: rank the universe on how cheap
a business is and how good it is, equally weighted, and buy the top of the combined list.

- **`ebit_ev`** — operating income ÷ enterprise value, where EV = market cap + total debt
  − cash. Unlike `pe` (and `earnings_yield`, which is just its reciprocal) this doesn't
  care how the company is financed, so a debt-laden company and a debt-free one with the
  same operating earnings are compared on the same basis.
- **`roic`** — operating income ÷ (net working capital + net PP&E). Working capital
  excludes cash and credits back short-term debt; a negative position is floored at zero;
  fixed assets exclude goodwill. Unlike `roe` it isn't inflated by leverage and doesn't
  break when buybacks push book equity negative.

Financials and utilities are excluded, as Greenblatt does — return on capital says nothing
about a bank, whose balance sheet *is* its product, and a regulated utility earns an
allowed return on a rate base, so ranking it on capital efficiency measures the regulator.

```bash
lti backtest --magic-formula --top-n 30 --start 2013-01-01
```

Both inputs come from fields the `secfsdstools` standardizers don't emit or don't get
right, so `lti.rawtags` reads them from the raw SEC files: SIC from `sub.txt`, and net
PP&E, interest-bearing debt, as-reported operating income and share counts from `num.txt`.

- **Debt** tagging is inconsistent across filers, so `total_debt` carries a `debt_source`
  column — `reported` (a debt tag was present), `assumed_zero` (no debt tag and negligible
  noncurrent liabilities on a *classified* balance sheet, so almost certainly no
  borrowings) or `unknown`. `unknown` leaves `total_debt` NaN and `ebit_ev` therefore drops
  the company, rather than quoting a yield that assumes the debt away. The classified-
  balance-sheet condition matters: homebuilders file every liability as current, which
  otherwise looks exactly like having no long-term debt.
- **EBIT** comes from `operating_income_reported` — the raw `OperatingIncomeLoss` tag —
  not from the standardized income statement. The standardizer always produces a number,
  but when the filer never tagged one it *derives* it, and the derivation lands near 100%
  of revenue for filers with an unusual income statement. Where the filer did tag it the
  standardizer agrees to within 1% on 99.5% of rows, so preferring the raw tag costs
  nothing and removes the artefacts that would otherwise sort straight to the top.

### 🧪 Backtest — simulate a strategy vs SPY and vs its own universe
Sidebar builds the strategy (**Rank by**, **Top N**, **Start/End**, **Rebalance month** —
April by default, when calendar-year 10-Ks are in; in January a screen ranks on
fundamentals a median of a year old — **Min market cap**, **Initial capital**); hit
**Run backtest**.

Two benchmarks: **SPY**, and the **universe** — every stock the screen ranked on each
rebalance date, equal-weighted, which is what picking at random from the same candidates
would have returned. The universe can only hold today's survivors too, so the gap
between it and the strategy is the honest measure of the ranking; the gap to SPY has the
survivorship bias baked in.

Body: equity curve vs SPY and the universe (log toggle), tiles (strategy / universe / SPY
CAGR, max drawdown, Sharpe), full stats table, a **survivorship-bias callout**, per-period
excess returns against either benchmark, **Does the rebalance month matter?** (the same
strategy run once per month — with a dozen annual rebalances, the month alone can decide
whether a screen beats its universe), the per-period summary, a holdings expander (every
pick with the metric values it was ranked on, + CSV), and a warnings expander. Results
are cached per exact config.

### 📐 Factor analysis — which metrics predict returns
Sidebar: **Metrics**, **Start/End** (as-of dates repeat on the start's day of year —
April by default), **Forward-return horizon** (months), **As-of spacing**
(months — set ≥ horizon for non-overlapping, honest t-stats), **Min market cap**,
**Require positive EPS**, **Operating companies only**, **Quantile buckets**,
**Correlation** (spearman / pearson); hit **Run analysis**.

For a grid of historical as-of dates the page takes a point-in-time snapshot (the same
split-correct, no-look-ahead path as the backtest), computes every metric and each stock's
forward return, then measures the **cross-sectional** correlation between metric and forward
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
value forward from its filing date, restate them onto today's share count using the
cached split history, and price them off the split-adjusted close — without that,
ratios across a split are wrong.

### 🎯 Undervalued today — the widest value-vs-price gaps
**The landing page.** Runs every intrinsic-value model across the whole point-in-time
universe and ranks by the gap between blended fair value and the current price.

Sidebar, in two groups: *Universe* (**As of**, **Min market cap ($M)**, **Require
positive EPS**, **Min ROE**, **Exclude financials** — Graham/DDM/EPV all assume an
operating business) and *Confidence* (**Models that must agree**, **Show top N**), with
the DCF assumptions behind an expander.

Body: four headline tiles; **the list** — a ranked table where upside is drawn as a bar
so the shape of the distribution is visible at a glance, alongside price, fair value,
how many models agreed, and `pe`/`roe`/`net_margin`/`debt_to_equity` + CSV; a **widest
gaps** bar chart of the top 15; a **cheap for a reason?** scatter of upside against ROE
for spotting value traps; and **do the models agree?** — every model's fair value for one
chosen company against its traded price, which is the honest way to read a blend, since
a tight cluster is worth far more than a high median.

Each 10-K's per-share figures are restated for any split since it was filed (a
company that split 10:1 after its last 10-K would otherwise show ten times its real EPS
against today's price); commodity/crypto trusts are excluded and >+500% upsides are
filtered as data noise, but a single year's earnings can still be a cyclical peak or a
one-off gain — the page says so. Also on the CLI as
`lti undervalued` — with `--out DIR` it writes a self-contained `index.html` +
`undervalued.{json,csv}`, which `deploy/` publishes nightly to `domalouf.com/invest/`
as the public daily list.

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
  tickers.py       CIK <-> ticker map (primary = most common-stock-like symbol)
  fundamentals.py  build/load the flat fundamentals.parquet + coverage report
  rawtags.py       SIC + debt / PP&E / goodwill straight from the raw SEC files
  sectors.py       SIC -> division, and the financials / utilities exclusions
  prices.py        yfinance cache: total-return + split-adjusted panels, split history (resumable)
  metrics.py       P/E, P/B, PEG, EBIT/EV, ROIC, debt/equity, ROE, margins, growth, ...
  valuation.py     intrinsic-value models (DCF, Lynch, Graham, DDM, EPV) + rank_undervalued
  report.py        render the undervalued list to static index.html / .json / .csv
  pit.py           point-in-time snapshots: split-correct, operating companies, priced
  ranking.py       ScreenSpec + composite percentile-rank selection
  backtest.py      annual-rebalance engine, universe benchmark, rebalance-month spread
  performance.py   CAGR / drawdown / Sharpe / hit rate / turnover
  progress.py      `lti progress` per-stage pipeline dashboard
  cli.py           `lti` command-line entry point
  factor.py        cross-sectional IC of each metric vs forward return
  stock.py         one company's annual fundamentals + valuation time series
  app/             Streamlit UI
    Home.py        entry point: page config, theme, navigation
    theme.py       palette + Plotly chrome + page furniture (import this, not raw styling)
    views/         Undervalued, Stock, Screener, Backtest, Factor analysis, Data health
deploy/            nightly publish of the public "undervalued today" snapshot (see deploy/README.md)
tests/             pure-logic unit tests (no network / SEC data)
```

`data/` (gitignored) holds everything generated: `data/sec/` (secfsdstools),
`data/derived/` (fundamentals, ticker map), `data/prices/` (price panels + split history).

## Known limitations / v2 ideas

- Annual (10-K) only; no quarterly rebalancing yet.
- "Debt/equity" = total liabilities / equity (not just interest-bearing debt).
- Share counts are the 10-K's (restated for splits, see above), so buybacks or issuance
  between the filing and the as-of date aren't in `market_cap` yet.
- "Operating companies only" means positive revenue and not a commodity pool, so it also
  drops pre-revenue companies (early-stage biotech, SPACs) — fine for value screens,
  worth knowing when reading a universe benchmark.
- EBIT is taken only from filings that actually tagged `OperatingIncomeLoss`
  (~76% of them), because the standardizer's *derived* value is badly wrong for
  filers whose income statement doesn't follow the usual shape. That's a real
  coverage cost, and it isn't random — homebuilders, PEOs and integrated oil are
  over-represented among the filers who don't tag it.
- `gross_profit` (and therefore `gross_margin`) is unreliable: the standardizer
  sets it equal to revenue whenever it can't find a cost-of-revenue line, which
  is ~27% of $1B+ revenue filings — Chevron, GM, JPMorgan and Berkshire all come
  through at a 100% gross margin. Don't screen on it without checking.
- No transaction costs, slippage or taxes.
- Survivorship bias (see above) — a proper point-in-time delisting map needs paid data.
