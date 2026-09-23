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

# 4. Backtest a strategy (rebalanced each April; vs SPY and vs its own universe),
#    after 10 bps a trade — see "Trading costs and taxes" below
lti backtest --metrics pe,debt_to_equity --top-n 10 --start 2011-01-01
lti backtest --metrics pe,debt_to_equity --top-n 10 --all-months   # the same, once per rebalance month
lti rolling-backtest --metrics pe,debt_to_equity --top-n 10 --windows 3,5   # over every 3- and 5-year window
#     --cost-bps 0            gross: no trading costs
#     --taxable               tax dividends and realized gains (--short-term-tax, --long-term-tax, --dividend-tax)
#     --hold-past-year        wait a year and a day between rebalances, so every gain is long-term
#     --sell-rank 60          with --top-n 30: keep a holding until it drops out of the top 60
#     --industry-cap 0.2      at most 20% of the picks in any one industry

# 4a. Greenblatt's Magic Formula (EBIT/EV + return on capital, no financials/utilities)
lti backtest --magic-formula --top-n 30 --start 2013-01-01

# 4b. See which metrics actually rank stocks by forward return
lti factor-ic --start 2012-01-01 --horizon 12 --step 12
lti factor-study                 # the pre-registered test of published factors (~5 min, see below)

# 4c. Today's most undervalued steady earners by blended intrinsic value (normalized earnings)
lti explain-models               # what each of the six equations does, and what it assumes
lti undervalued --top 30 --market-cap-min 2000 --min-profit-years 4
#     --include-financials    keep banks, insurers, REITs and BDCs
#     --format html|json|csv  prints that instead of a table
#     --out DIR               writes index.html + undervalued.{json,csv} (the public snapshot)

# 4d. Keep score going forward (see "Track record" below)
lti track-record                 # write down what each tracked strategy holds today (the nightly job does)
lti track-report                 # how the records have done since
lti journal-add AAPL buy --thesis "why" --change-my-mind "what would prove it wrong"
lti journal                      # every logged decision and how it has aged

# 5. GUI (see "GUI" section below)
streamlit run src/lti/app/Home.py
```

`lti progress` prints an ASCII progress bar for each pipeline stage (SEC data,
filter / standardize / concat, fundamentals, ticker map, prices) — handy for checking
on the multi-hour full build. `lti fetch-prices` shows a live `tqdm` bar while running.

### Prices, splits, dividends and look-ahead

`data/prices/` holds four things, and each has one job:

- `adj_close.parquet` — close adjusted for splits **and dividends**: a total-return
  series. Used for **returns** only.
- `close.parquet` — close adjusted for **splits only**: the traded price on today's
  share basis. Used for **valuation** (P/E, P/B, market cap, EV, fair value).
- `splits.parquet` — every split (`ticker, date, ratio`), used to restate each 10-K's
  EPS and share count onto that same basis (`lti.pit.restate_per_share`).
- `dividends.parquet` — every dividend (`ticker, date, amount`), the cash paid per
  share. Used for the **yield**, the payout ratio and the dividend-discount model.

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
series or a recorded dividend count — so a cache from before `close.parquet` or
`dividends.parquet` existed re-fetches every ticker once (about 15 minutes for ~4,200
tickers at `--batch-size 80`). `lti refresh-prices` downloads a short recent window and
compares the bars it shares with the cache: Yahoo rewrites a ticker's whole history on a
split (and the adjusted close on every dividend), and splicing the new window onto the
old history would put a fake crash at the seam, so any ticker whose overlap doesn't
match is re-fetched in full. A split or dividend inside the window forces that re-fetch
outright — a dividend re-bases the bars *before* its ex-date, so one landing on the
window's first bar leaves the overlap looking untouched while its event goes unrecorded.

### Dividends

The dividend events come from the same download as the prices, and Yahoo states the
amounts on today's share basis, exactly like `close.parquet` — a payment from before a
4:1 split comes back quartered. So a yield is one divided by the other, with nothing to
restate in between.

Every priced snapshot carries four columns off that table, all point-in-time (only
ex-dates on or before the as-of date count, so a backtest can't see a dividend declared
after its rebalance):

| column | what it is |
| --- | --- |
| `dps_ttm` | dividends per share over the last twelve months |
| `dividend_yield` | that, over the split-adjusted close — rankable |
| `dividend_growth_5y` | CAGR of the trailing-twelve-month payment over five years — rankable |
| `payout_ratio` | `dps_ttm` over EPS |

A company with no payments yields **0**, not NaN: the cache holds every fetched
ticker's whole history, so silence there means it paid nothing, and a screen should be
able to rank on that. `dividend_growth_5y` is the exception — it stays NaN unless the
company paid in both windows, since a starter has no rate yet and a cutter's isn't
finite.

This replaces the only dividend figure the project had before: `PaymentsOfDividends`
from the standardized cash-flow statement, which carries just **25%** of the latest
filings, lags by up to a year, and lumps preferred dividends in with common. The
valuation models (`ddm_value`, and the dividend yield in Peter Lynch's fair P/E) use the
payments when a frame has them and fall back to the tag when it doesn't. Gordon growth
also grows the dividend at its *own* rate where one is known rather than at the
revenue-side estimate — a company can grow revenue and hold its payout flat — though
both are capped at the terminal rate inside the model.

Note that this changes no return anywhere: `adj_close.parquet` has always been a
total-return series, so the backtests, the factor study and the forward track record
already counted every dividend. What is new is being able to *see* and *rank on* them.

### Free cash flow and stock-based pay

`free_cash_flow` is operating cash flow less capex **less stock-based compensation**. The
cash-flow statement adds stock pay back as a non-cash expense, but paying staff in shares is
still paying them: the cost lands on the owners as dilution, or as the buybacks spent
offsetting it. Left in, it flatters exactly the companies that pay most in stock, and
everything built on free cash flow inherits that — the DCF, `fcf_yield`, `fcf_yield_norm`,
`fcf_margin`, and cash conversion on the Undervalued page.

The figure is the `ShareBasedCompensation` line of the standardized cash-flow statement
(`stock_comp`). A filing without one is taken to have none, which leaves its free cash flow
as it was — the line is where material stock pay shows. A stock-comp figure too large by a
scale error can only push a company *down* the cash rankings and out of the DCF, never up.
`free_cash_flow_reported` keeps the textbook `cfo − |capex|`, and the Stock page's cash-flow
tab charts `stock_comp` beside the others. A fundamentals table built before this is
upgraded as it loads; no rebuild needed. The Undervalued page's backtest figures quoted
below predate the change.

Two things stay on the old definition on purpose, via `study.as_registered`: the
pre-registered factor study, and the track record's *quality + value* strategy, which tests
the study's composite. Changing a registered hypothesis's inputs after seeing its result is
what pre-registration exists to prevent.

### Trading costs and taxes

A gross backtest trades at the close for free and never pays tax, which flatters a strategy
that turns over every year against SPY, bought once and held. So every backtest — the
Backtest and Rolling pages, `lti backtest`, `lti rolling-backtest` — runs the strategy, its
universe and SPY a second time as a book of tax lots (`lti.frictions.Book`), and reports that:

- **Trading costs.** Every dollar traded pays `--cost-bps` of itself each way — the
  half-spread plus any commission. The default, 10 bps, is a middle estimate for the $500M+
  companies these screens buy: large caps trade tighter, small caps wider. It's charged on
  the actual trades — selling what dropped out, buying what came in, and trimming or topping
  up what stayed back to equal weight. `--cost-bps 0` gives the gross backtest.
- **Taxes** (`--taxable`, or the *Taxable account* toggle; off is an IRA or 401(k)).
  Dividends are taxed as they arrive and only the rest is reinvested — each holding's
  dividend being its total return less its price return, `adj_close` against `close`. Gains
  are taxed when a sale realizes them: at the short-term rate for a lot held a year or less,
  the long-term rate for one held longer, oldest lot first, with losses offsetting gains and
  the rest carried forward. The defaults — 24% short-term, 15% long-term and on dividends —
  are a middle federal bracket with no state tax: set your own.
- **The same rules for all three.** The universe pays them on its own, smaller, turnover,
  so the gap to it is what the ranking adds after paying for the trading it takes. SPY pays
  the cost once and the tax on its dividends — and, never being sold, nothing on its gains.
  *Sold at the end* (`*_cagr_liquidated`) puts all three on the same footing: the CAGR had
  everything been sold on the last day, paying the tax on every gain still unrealized.

**The one-year trap.** A gain is long-term only if the lot was held *more* than a year. The
annual rebalance falls on the first trading day of the month, which in most years is on or
before the anniversary of the last one — April 1, 2013 to April 1, 2014 is a year to the
day, so short-term. Left alone, most of an annual strategy's gains are taxed as income.
`--hold-past-year` (*Sell only after a full year*) waits until a year and a day have
passed, which drifts the rebalance a few days later each year. It's half of what Greenblatt
advises for running the Magic Formula in a taxable account; the other half, selling losers
just *before* the year, would need two trading dates a year.

**Trading less: a sell buffer.** Without one, a holding is sold the moment it slips out of
the top N — 30th to 31st is enough — and often bought back a year later, paying the cost
and the tax both ways for rank noise. `--sell-rank` (*Sell a holding once it drops out of
the top* …) keeps a holding until it falls out of a wider band, and refills only the places
its departures free up with the best-ranked names not already held: `--top-n 30
--sell-rank 60` buys the top 30 but sells only below 60th. The price is that the portfolio
holds some names the screen no longer ranks in its top N. Each period records
`n_held_over`, and each holding `held_over` and the `rank` it was bought or kept at, so the
trade-off shows in the turnover, the costs and the taxes against the same run without it.
Off by default (`sell_rank = None`).

Each backtest reports, beside the usual statistics (which are now net): `*_cagr_gross`,
`*_costs_pa` and `*_taxes_pa` (the average paid per rebalance period, as a share of the
portfolio), `*_cagr_liquidated`, and `port_short_term_share`; each period adds
`port_return_gross`, `turnover` (one-way), `costs`, `taxes` and the realized gains by
term. Left out: the $3,000 of losses a year that can offset ordinary income (a fixed sum,
meaningless at an arbitrary portfolio size), wash sales, and the delay to the following
April — tax is paid when the gain is realized, which is slightly conservative.

**Returns quoted elsewhere in this README were measured gross, before this existed** —
rerun with `--cost-bps 0` to reproduce them. The factor study's backtests stay gross, as
registered.

### Industry concentration, and a cap on it

A screen can rank well on every metric and still be one bet on one industry: the factor
study's final screen, below, came out as a portfolio of shrinking retailers and telecoms.
So every backtest now records the largest industry in the portfolio at each rebalance
(`top_industry`, `top_industry_share`; `port_top_industry_share` averages it), and
`--industry-cap` (*Most in one industry* on the Backtest and Rolling pages) limits it:
walking down the ranking, a name whose industry already fills its share is passed over for
the next one. With equal weights the cap is a count — `--top-n 30 --industry-cap 0.2` allows
6 names per industry — and never below one.

Industries are Fama and French's 12, from SIC codes (`lti.sectors.industry`): Consumer
Non-Durables, Consumer Durables, Manufacturing, Energy, Chemicals, Business Equipment,
Telecom, Utilities, Shops, Health, Finance and Other. The SIC *divisions* the Screener shows
as `sector` are too coarse for this — Manufacturing alone is about half the market, drugs,
chips, cars and food together — so a cap on them would mostly push a portfolio out of
manufacturing rather than off a theme. A company with no SIC code is never capped. Kept
holdings under a sell buffer count toward their industry but aren't sold to make room.
Each holding records its `industry`. Off by default (`industry_cap = None`).

## GUI

```bash
source venv/bin/activate
streamlit run src/lti/app/Home.py          # opens http://localhost:8501
```

**Undervalued today is the landing page.** The sidebar groups the rest by what you're
there to do: *Find something to buy* (Undervalued, Stock detail), *Test an idea*
(Screener, Backtest, Rolling backtest, Factor analysis), *Keep score* (Track record, Decision journal)
and *Housekeeping* (Data health). Leave
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
`fcf_margin`, `revenue_growth_1y`, `eps_growth_1y`, `dividend_yield`,
`dividend_growth_5y`, … — the menu is the four lists in `lti.metrics`), **Top N**,
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
fundamentals a median of a year old — **Min market cap**, **Initial capital**, and the sell
buffer: **Sell a holding once it drops out of the top** …, and **Most in one industry**)
and the
**Costs and taxes** (trading cost, taxable account, the three rates, *Sell only after a
full year* — see [Trading costs and taxes](#trading-costs-and-taxes)); hit **Run backtest**.

Two benchmarks: **SPY**, and the **universe** — every stock the screen ranked on each
rebalance date, equal-weighted, which is what picking at random from the same candidates
would have returned. The universe can only hold today's survivors too, so the gap
between it and the strategy is the honest measure of the ranking; the gap to SPY has the
survivorship bias baked in.

Body: equity curve vs SPY and the universe, plus the strategy before costs and taxes
(log toggle), tiles (strategy / universe / SPY CAGR after costs and taxes, max drawdown,
Sharpe), full stats table, **What trading and taxes took** (each portfolio's CAGR before
and after, costs and taxes a year, the CAGR if sold at the end, turnover and the share of
gains taxed short-term), a **survivorship-bias callout**, per-period
excess returns against either benchmark, **Was it one theme?** (the largest industry's
share of the portfolio at each rebalance, against the cap if one is set), **Does the
rebalance month matter?** (the same
strategy run once per month — with a dozen annual rebalances, the month alone can decide
whether a screen beats its universe), the per-period summary, a holdings expander (every
pick with the metric values it was ranked on, + CSV), and a warnings expander. Results
are cached per exact config.

### 🔁 Rolling backtest — the same strategy over every window
One backtest is one draw from history, and easy to fit to. This reruns it over every
window of each chosen length — every 3-year and every 5-year stretch of the price
history, say, starting a year apart — and asks how often the strategy beat its universe
and SPY. Sidebar: the strategy as on the Backtest page, plus **Window lengths**, **Step
between window starts** and an optional **Earliest start / Latest end** (blank = all the
price history), and the same **Costs and taxes**. Body: a summary per window length (median, worst and best CAGR; the
average gap to the universe and to SPY, and how often each was beaten; drawdown and
Sharpe), a box plot of each window's gap to the universe, every window + CSV, and a
warnings expander. Also on the CLI as `lti rolling-backtest`. Windows of one length
overlap heavily — at a 12-month step, neighbouring 5-year windows share four years — so
read the spread as illustrative rather than as independent samples; and every window
carries the backtest's survivorship bias, which is why the universe is the yardstick.

### 📐 Factor analysis — which metrics predict returns
Sidebar: **Metrics**, **Start/End** (as-of dates repeat on the start's day of year —
April by default), **Forward-return horizon** (months), **As-of spacing**
(months — set ≥ horizon for non-overlapping, honest t-stats), **Min market cap**,
**Require positive EPS**, **Operating companies only**, **Quantile buckets**,
**Correlation** (spearman / pearson); hit **Run analysis**. Besides the single-filing
metrics it covers the multi-year ones (`pe_norm`, `earnings_yield_norm`, `fcf_yield_norm`,
`profit_years`, `revenue_cagr`, `fcf_conversion`, `roic_median`) and the Undervalued page's
own ranking, `fair_value_upside` — plus `fair_value_upside_1y`, the same models on the
latest year alone, for comparison. The Screener and Backtest can rank on all of them.

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

### What works in this data — a pre-registered test (`lti factor-study`)
Choosing a strategy and judging it on the same data flatters it, so `lti.study` fixes
everything in code first: sixteen hypotheses, each with the direction its paper found
(profitability, cash profitability, accruals, share issuance, asset growth, shareholder
yield, 12-1 momentum, Piotroski F-score, Altman Z, four value ratios and three
composites); the universe ($500M+ operating companies, financials and BDCs out); the test
(monthly as-of dates, 12-month forward returns, rank IC, Newey-West t for the overlap); and
the split — April 2011 to March 2018 chooses, April 2019 on judges. The one decision made
from data: a final screen of the single factors that pointed the expected way in the first
half. Results as of September 2026 (IC signed so that positive = as the paper said):

| | 2011–18 IC (t) | 2019–25 IC (t) |
|---|---|---|
| **Held up in both halves** | | |
| share issuance (buybacks good) | +0.058 (4.3) | +0.102 (2.6) |
| shareholder yield | +0.049 (2.0) | +0.099 (2.1) |
| Piotroski F-score | +0.025 (1.9) | +0.052 (1.9) |
| **Worked only in 2019–25** | | |
| operating / cash profitability | −0.005 / −0.003 | +0.094 (2.9) / +0.083 (2.8) |
| quality composite · quality + value | −0.007 · 0.000 | +0.067 (3.0) · +0.092 (2.3) |
| **Weak both times** | | |
| earnings yield, EBIT/EV, FCF yield, book/market, value composite, momentum, asset growth | −0.030 to +0.028 | +0.011 to +0.087 (t ≤ 1.7) |
| **Wrong way both times** | | |
| accruals, Altman Z | −0.018, −0.020 | −0.013, −0.001 |

The final screen the first half chose — share issuance, asset growth, shareholder yield
and F-score — kept a positive IC out of sample (+0.097, t 2.0), and its top half of stocks
beat its bottom half by about two points a year. But a **top-30 portfolio of it trailed its
own universe by 3.6% a year in 2019–25**, ahead in 1 of 12 rebalance months (in-sample,
2011–18, it had beaten it by 5.0%). The names that score well on all four at once are
shrinking cash-returners — Macy's, Kohl's, Best Buy, Western Union, Lumen, Sirius — much of
it in industries in decline, so the portfolio rides one theme (`--industry-cap`, above, tests
whether spreading it helps). The signals are real across
the market; a concentrated screen on them isn't a way to collect them. Caveats: two
seven-year halves, a survivor-only universe (which flatters distressed stocks and so
works against Altman Z and quality in the first half), and published factors typically
lose much of their edge after publication. Two later changes are held off so these numbers
stay reproducible: the study measures free cash flow with stock-based pay still in it, and
its backtests are gross of trading costs.

### 🔬 Stock detail — one company over time
Sidebar: **Ticker** (matches the primary symbol *and* the full `tickers_all` list, so
`JPM` resolves), **Log price axis**, **Mark 10-K filing dates**, **Split-adjust EPS /
book value**.
Body: eight tabs — **Price** (adjusted close with filing-date markers), **Income**
(revenue → net income bars + EPS), **Margins & returns** (gross / net / FCF margin, ROE),
**Balance sheet** (assets / liabilities / equity + debt-to-equity), **Cash flow**
(CFO / capex / stock comp / FCF), **Valuation** (trailing P/E and P/B time series with a median line),
**Fair value** (intrinsic-value models, below, on normalized or latest-year earnings, with
a per-company 5-year CAGR growth input and adjustable discount rate / terminal growth /
DCF window), and **Raw data** (the annual table + CSV). The Valuation and Fair-value tabs carry each 10-K's EPS and book
value forward from its filing date, restate them onto today's share count using the
cached split history, and price them off the split-adjusted close — without that,
ratios across a split are wrong.

### 🎯 Undervalued today — the widest value-vs-price gaps among steady earners
**The landing page.** Runs every intrinsic-value model on each company's **normalized**
earnings and ranks by the gap between blended fair value and the current price.

One year's earnings is a poor guide: it can be a cyclical peak, a one-off or the one good
year of a chronic loss-maker, and every model multiplies it. On the latest year alone the
top of this list was Lyft (a tax-asset release on an operating loss), Novavax and PTC
(one good year in ten), Uniti (a merger gain) and Cal-Maine (an avian-flu egg-price
year) — plus Prudential and DTE priced off their $25 baby bonds, which the ticker map had
picked over the common stock. Now (`lti.history`) each company's last five fiscal years
are taken point in time — as filed by the as-of date — and the models run on the median
of their EPS and free cash flow, with growth from the five-year revenue trend.

Sidebar: *Universe* (**As of**, **Min market cap ($M)**, **Exclude financials** — banks,
insurers, REITs and business development companies — **Min ROE**), *Consistency*
(**Profitable in at least … of the last 5 years**, default 4; **Profitable now**; **Show
top N**) and the valuation assumptions behind an expander. A year counts as profitable
only if EPS, net income *and* operating income were all positive.

Body: four headline tiles; **the list** — upside drawn as a bar, price, fair value, P/E on
the latest and on normalized EPS, profitable years, *latest vs norm* (latest EPS ÷
normalized — a peak or a one-off shows far above 1, a trough far below), the revenue
trend, *cash conversion* (five years' free cash flow ÷ net income) and debt, + CSV; a
**widest gaps** chart; **cheap for a reason?** — upside against latest-vs-norm EPS, since
a name below the line is a bet on earnings recovering; and **how the fair value is built**
— each model's value for one company beside its earnings history and the normalized line.

**What the evidence says.** Backtested (April rebalance, 2012–, $1B+, financials out),
buying the 30 widest gaps each year returned 8.6% a year against 11.9% for all steady
earners, trailing them in every one of the 12 rebalance months; the old latest-year
version did 10.1% against 12.4%. The fair-value upside has no measurable IC either way
(t ≈ 0.5), while consistency does: `profit_years` has the strongest IC of any metric here
(0.08, t 2.4). So this is a list of candidates to research, not a buy list — and the page
says so. Also on the CLI as `lti undervalued`; with `--out DIR` it writes a self-contained
`index.html` + `undervalued.{json,csv}`, which `deploy/` publishes nightly to
`domalouf.com/invest/` as the public daily list.

### 📒 Track record — the only test free of hindsight
Every backtest here runs on a survivor-only universe, and every idea in this project was
chosen after looking at the same fifteen years. The future is the one clean test, so
`lti track-record` — run by the nightly job — writes down what each tracked strategy
holds that day, once, never revised: one parquet file per day in `data/track/records/`,
with a JSON sidecar recording the parameters and the git commit that made it. It won't
record a day more than a week old — the price cache holds only companies still trading,
so a backdated record would quietly drop the ones that failed in between. The page
(and `lti track-report`) then measures what each day's holdings returned over the next
1, 3, 6 and 12 months — bought at the next day's close, since the list is made after the
market shuts — against SPY and against the whole universe recorded the same day, plus a
monthly-rebalanced paper portfolio for each. Companies later acquired or
delisted stay in the record at their last price, so it has no survivorship bias. Returns
come from the current total-return prices, not prices stored at the time.

Tracked, fixed in `lti.track.STRATEGIES`: the **Undervalued list** (what's published);
the factor study's **final screen**, top 30 (does its 2019–25 shortfall persist?); two
ideas the study suggested but couldn't test — **no heavy diluters** (the universe without
its top tenth by share issuance) and **cash returners** (top fifth by shareholder yield);
**quality + value** (top fifth); and the **universe** itself as the yardstick
($1B+, operating companies, financials and BDCs out, equal-weighted). Give it a year
before reading anything into it — the page shows a Newey-West t for how far each gap is
from luck. Set `LTI_TRACK_BACKUP` for the nightly job to keep a copy off the machine: a
record can't be rebuilt after the fact.

### ✍️ Decision journal
Log each decision — buy, add, trim, sell, watch, pass — with why, what would change your
mind, a fair value and conviction if you have them, and a review date (default a year
on); the page has a form, the CLI `lti journal-add`. Every decision is scored against
SPY from its date: a buy is right so far if the stock has beaten the market since, a sell
or pass if it has lagged. Entries are append-only (`data/track/journal.jsonl`); a later
look is a `review` entry pointing at the original, and the page flags reviews that are due.

### Intrinsic-value models (`lti.valuation`)
`add_valuation_models()` turns a fundamentals snapshot + price into a fair value per share
from six equations, and blends them into `fair_value_est` — the **median** of the ones that
produced a number. `*_upside` is `fair value ÷ price − 1`.

Same explanations everywhere: each equation is walked through in its function's docstring,
carried as data in `MODEL_DOCS` for the Stock page, the Undervalued page and the published
`index.html` to render, and `explain(model, row)` prints the equation with one company's own
numbers in it (`EPS $3.05 / 9.0% = $33.89`). `add_valuation_models()` records the inputs it
actually used — `eps_used`, `bvps_used`, `fcf_ps_used`, `dps_used`, `est_growth` — so any fair
value can be checked against the numbers that produced it.

#### What goes in

| input | `basis="normalized"` (default) | `basis="latest"` |
| --- | --- | --- |
| EPS | median of the last 5 years' EPS (`eps_norm`) | the latest 10-K's EPS |
| FCF per share | median 5-year FCF ÷ today's share count | (CFO − capex − stock comp) ÷ shares, latest 10-K |
| book value per share | equity ÷ shares outstanding | same |
| dividend | last 12 months actually paid (`dps_ttm`), else the cash-flow tag | same |
| growth `g` | 5-year revenue CAGR (`revenue_cagr`) | one-year EPS change, else revenue's |

Per-share figures reach the models already restated onto today's share count for every split
since each filing, and priced off the split-adjusted close (`lti.pit.priced_snapshot`, upstream)
— without that, every ratio across a split is wrong. Growth is clipped to `[0, growth_cap]` — a 40% grower is
not a 40% grower for a decade, and a negative one would value a shrinking business at less than
zero. Pass an explicit `growth` Series (e.g. `historical_cagr(annual, "eps")`) to override the
estimate. The rest is `ValuationAssumptions`: `discount_rate` (9%), `terminal_growth` (2.5%,
forced at least a point below the discount rate), `dcf_years` (10), `bond_yield` (4.5%).

#### The six equations

| model | equation | multiplies | at the defaults | no value when |
| --- | --- | --- | --- | --- |
| Two-stage DCF | `Σ FCF·(1+g)ᵗ/(1+r)ᵗ + terminal/(1+r)ᴺ` | FCF/share | ≈19× FCF at 5% growth, 13× flat | FCF/share ≤ 0 |
| Peter Lynch | `EPS · (g% + dividend yield%)` | EPS | fair P/E of 14 at 12% growth + 2% yield | EPS ≤ 0 |
| Graham number | `√(22.5 · EPS · BVPS)` | EPS × book | 15× EPS at a 10% return on book | EPS or BVPS ≤ 0 |
| Graham revised | `EPS · (8.5 + 2g) · 4.4/Y` | EPS | 8.3× EPS flat, 18.1× at 5%, 37.6× at the cap | EPS ≤ 0 |
| Dividend discount | `D₀·(1+g) / (r − g)` | dividends paid | 15.8× the trailing dividend | no dividend |
| Earnings power | `EPS / r` | EPS | a flat 11.1× EPS | EPS ≤ 0 |

**Two-stage DCF.** A share is worth the cash the business will hand its owners, with cash
further out worth less. Stage one walks free cash flow per share forward for `dcf_years`,
growing it at `g` and discounting year *t* by `1/(1+r)ᵗ`. Stage two assumes the business then
settles into growing at `terminal_growth` forever and capitalises that with Gordon growth —
`FCF_N·(1+g_term)/(r − g_term)` — a lump sitting at year N, so it gets discounted back N years
too. *The weak point:* that terminal lump is most of the answer (≈57% of it at 5% growth), so
the value is largely a bet on the perpetuity, and it moves more on a point of the discount rate
than on the entire explicit window.

**Peter Lynch.** *One Up on Wall Street*'s rule of thumb: a growth company is fairly priced when
its P/E equals its growth rate (PEG = 1), plus the dividend yield, since a payout is return that
arrives whether or not the growth does. A 12% grower yielding 2% earns a fair P/E of 14; on $3
of EPS that is $42. *The weak point:* a heuristic, not a valuation — no discount rate, no
horizon, no balance sheet — and growth is the whole answer, making it the most sensitive of the
six to a growth number that here comes from a revenue trend, not the analyst forecasts Lynch was
reading. A profitable company with no growth and no dividend fairly values at $0.

**Graham number.** The *Intelligent Investor* asks a defensive buyer for two things at once: no
more than 15× earnings, and no more than 1.5× book. Multiply the limits and the constant is 22.5
— the equation is that pair of screens rearranged, so a company may be dearer on one where it is
cheaper on the other. Written as `√(15·EPS × 1.5·BVPS)` it is plainly the *geometric mean* of the
two ceilings. *The weak point:* it is a ceiling for a defensive buy, not an estimate of worth,
and half of it is book value — it understates asset-light businesses whose R&D and brands are
expensed rather than capitalised, and flatters ones carrying goodwill from acquisitions that
didn't work.

**Graham revised.** Graham's 1962 multiple table: `8.5` is the P/E for a company expected to grow
not at all, `2g` adds two turns of that multiple per point of annual growth (`g` in *percentage
points* — 5 for 5%), and `4.4/Y` rescales the lot for interest rates, 4.4% having been the AAA
corporate yield when he wrote it. `g` is clipped to 20 points and `Y` floored at 0.5% so a
runaway growth estimate or a zero yield can't produce an absurd multiple. *The weak point:*
Graham came to think it too crude to rely on; it is near-linear in `g`, and the rate term marks
every company up as bond yields fall.

**Dividend discount.** Gordon growth — the sum of a dividend growing at `g` forever, discounted
at `r`, closes to `D₀·(1+g)/(r − g)`. `D₀` is the trailing twelve months of payments actually
made (from the price cache, not the sparse cash-flow tag), grown at the dividend's *own* five-year
rate where there is one, since a company can grow revenue and hold its payout flat. Growth is
clipped to `terminal_growth`: as `g` approaches `r` the denominator goes to zero and the value to
infinity. *The weak point:* that cap values a genuine dividend grower as if it grew 2.5% a year,
so this is usually the lowest of the six, and buybacks and retained earnings are invisible to it —
a company returning its cash by repurchase is worth nothing here.

**Earnings power.** Greenwald's argument is that growth is the least knowable input, so the
sturdier question is what the business is worth if it never grows: a perpetuity of today's
earnings, `EPS / r`. *The weak point:* it is a single multiple, so it says as much about the
discount rate you chose as about the company, and the textbook version adjusts for maintenance
capex, excess cash and one-offs where this one capitalises reported EPS as it stands.

#### Blending them, and a worked example

A company with normalized EPS $3.05, book value per share $12.40, FCF per share $2.10, a $0.64
trailing dividend and 4.3% growth, trading at $30 — the arithmetic the Stock page prints:

| model | with its numbers in | fair value |
| --- | --- | --- |
| Two-stage DCF | 10 years of FCF/share $2.10 growing 4.3%, then 2.5% forever, discounted at 9.0% | **$37.92** |
| Peter Lynch | EPS $3.05 × (growth 4.3 + yield 2.1) | **$19.62** |
| Graham number | √(22.5 × EPS $3.05 × BVPS $12.40) | **$29.17** |
| Graham revised | EPS $3.05 × (8.5 + 2 × 4.3) × 4.4 / 4.5 | **$51.00** |
| Dividend discount | dividend $0.64 × (1 + 2.5%) / (9.0% − 2.5%) | **$10.09** |
| Earnings power | EPS $3.05 / 9.0% | **$33.89** |

`fair_value_est` is the median, $31.53, a +5% gap to the $30 price. A median rather than a mean
so that one model's extreme — the $10.09 here — can't set the answer.

Six values from $10 to $51 for the same company is the point, not a defect: the spread *is* the
uncertainty, and a tight cluster deserves more weight than a high median. But six models agreeing
is not six independent opinions — four of them (Lynch, both Grahams, earnings power) multiply the
same EPS, so they mostly restate that one number at different multiples. The DCF (cash flow) and
the dividend discount (cash actually paid out) are the two carrying separate evidence.

The screen adds two credibility rules on top: at least `MIN_MODELS` (3) models must have produced
a number, and an upside beyond `MAX_UPSIDE` (+500%) is treated as a data error rather than a
bargain. And backtested, the widest gaps have *trailed* the average steady earner — these are
rough, assumption-sensitive estimates for deciding what to research, not investment advice.

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
  tickers.py       CIK <-> ticker map (primary = the SEC's first-listed security)
  fundamentals.py  build/load the flat fundamentals.parquet + coverage report
  rawtags.py       SIC + debt / PP&E / goodwill / share counts straight from the raw SEC files
  sectors.py       SIC -> division and Fama-French industry, and the financials / utilities exclusions
  prices.py        yfinance cache: total-return + split-adjusted panels, split history (resumable)
  metrics.py       P/E, P/B, PEG, EBIT/EV, ROIC, debt/equity, ROE, margins, growth, ...
  history.py       five years of filings, point in time: normalized EPS/FCF, consistency, growth
  valuation.py     intrinsic-value models (DCF, Lynch, Graham, DDM, EPV) + rank_undervalued
                   MODEL_DOCS explains each equation; explain() prints it with a company's numbers
  report.py        render the undervalued list to static index.html / .json / .csv
  pit.py           point-in-time snapshots: split-correct, operating companies, priced
  ranking.py       ScreenSpec + composite percentile-rank selection
  backtest.py      annual-rebalance engine, universe benchmark, rebalance-month spread
  frictions.py     trading costs and taxes: a portfolio as tax lots, rebalanced and marked forward
  rolling.py       reruns the backtest over every N-year window in the price history
  performance.py   CAGR / drawdown / Sharpe / hit rate / turnover
  progress.py      `lti progress` per-stage pipeline dashboard
  cli.py           `lti` command-line entry point
  factor.py        cross-sectional IC of each metric (and composites) vs forward return, Newey-West t
  study.py         the pre-registered factor test: hypotheses fixed in code, 2011-18 chooses, 2019-25 judges
  track.py         the forward track record: append-only daily holdings of each strategy, scored later
  journal.py       the decision journal: append-only decisions, each scored against SPY since
  stock.py         one company's annual fundamentals + valuation time series
  app/             Streamlit UI
    Home.py        entry point: page config, theme, navigation
    theme.py       palette + Plotly chrome + page furniture (import this, not raw styling)
    views/         Undervalued, Stock, Screener, Backtest, Rolling backtest, Factor analysis,
                   Track record, Journal, Data health
deploy/            nightly publish of the public "undervalued today" snapshot (see deploy/README.md)
tests/             pure-logic unit tests (no network / SEC data)
```

`data/` (gitignored) holds everything generated: `data/sec/` (secfsdstools),
`data/derived/` (fundamentals, ticker map), `data/prices/` (price panels + split history),
`data/track/` (the track record and the decision journal — the one part that can't be rebuilt).

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
- Trading costs are one flat rate per dollar traded, not a spread per stock, and don't
  grow with the size of the order; taxes are federal-style, with no state tax.
- Normalized earnings assume the last five years are a fair guide: a business in lasting
  decline, or a cycle longer than five years, still fools them.
- Survivorship bias (see above) — a proper point-in-time delisting map needs paid data.
