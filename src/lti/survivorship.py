"""How much do the companies that left the market bend the backtests?

Every backtest here can only hold companies that still trade: the SEC's ticker
list names today's registrants, and Yahoo keeps no prices for the rest. A company
that was acquired, taken private or went bankrupt is in the filings — it is
simply never bought. No free source has those prices, so the bias can't be
removed. It can be **measured and bounded**, because the filings themselves say
who was there and when they stopped:

* :func:`census` — each April, every operating company with listed equity and
  revenue over a floor, split into those the backtests can price and those they
  can't, and for the ones that can't, whether this was their last year and what
  shape they were in when they went.
* :func:`universe_bounds` — the equal-weighted universe's return with the missing
  put back at an assumed return (:data:`SCENARIOS`), next to the survivors-only
  figure the backtests report.
* :func:`factor_bounds` — the same for the best fifth on each factor that needs
  no price, against the universe: does a signal's edge survive, or was it an
  edge at avoiding companies the data had already deleted?

**What is assumed.** A missing company's last year is the twelve months after
its final 10-K. Its return that year is the scenario's: one number for a company
that left *healthy* (profitable, positive equity — overwhelmingly takeovers, at a
premium) and one for a company that left *distressed* (a loss or negative equity
— some were bought cheaply, some wiped out). In the years before its last, it is
assumed to have earned what the priced companies in the same group earned. The
scenarios are deliberately wide; the truth for any one company is unknowable
here, and the point is the range.

**What can't be done.** Factors that need a price — every valuation ratio,
shareholder yield, momentum — can't be ranked for a company with no price, so
their backtests can't be bounded this way. Size is by revenue, not market cap,
for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lti import metrics, pit, prices as prices_mod, sectors


@dataclass(frozen=True)
class Scenario:
    name: str
    healthy: float  # the last-year return of a company that left in good shape
    distressed: float  # ... and of one that left making a loss or with negative equity
    note: str


# Takeover premiums average 30-45% over the pre-bid price (Betton, Eckbo & Thorburn
# 2008); stocks delisted for cause lose about 30% at delisting on top of the fall
# before it (Shumway 1997), and most of a year's value by the time they go.
SCENARIOS: list[Scenario] = [
    Scenario("harsh", 0.0, -1.0, "every distressed exit a wipe-out, no premium on the rest"),
    Scenario("central", 0.25, -0.60, "a takeover premium on healthy exits, most of the value lost on distressed ones"),
    Scenario("kind", 0.40, -0.30, "a full premium, and distressed exits mostly bought out rather than wiped out"),
]

# factors that need no price, with the direction the factor study registered (or,
# for the two it didn't test, the obvious one): +1 means higher is better
PRICE_FREE_FACTORS: dict[str, int] = {
    "share_growth": -1,
    "asset_growth": -1,
    "f_score": +1,
    "op_profitability": +1,
    "cash_profitability": +1,
    "accruals": -1,
    "roic": +1,
}
REVENUE_MIN = 500e6
GONE_AFTER_DAYS = 550  # no 10-K for this long before the data ends: the company has stopped filing
STATUSES = ["priced", "later_gone", "exit_healthy", "exit_distressed"]


@dataclass
class SurvivorshipResult:
    census: pd.DataFrame  # one row per April
    universe: pd.DataFrame  # one row per April: the universe's return, survivors-only and per scenario
    universe_cagr: pd.Series  # the same, compounded
    factors: pd.DataFrame  # one row per factor: exits in its best fifth, and its edge per scenario
    warnings: list[str]


def classify(fund: pd.DataFrame, px: prices_mod.PriceData, asof, *, revenue_min: float = REVENUE_MIN) -> pd.DataFrame:
    """Every company in the true universe at ``asof``, with a ``status`` (:data:`STATUSES`),
    its ticker where it has one, and the price-free factors.

    Left out, and counted in ``attrs["excluded"]``: filers that report no EPS
    (subsidiaries with public debt and no listed shares), companies still filing
    today that have no ticker (unlisted, or listed where the SEC's list doesn't
    reach), and companies whose last 10-K is already more than a year old.
    """
    asof = pd.Timestamp(asof)
    fund = pit.annual(fund)
    snap = pit.snapshot_asof(fund, asof)
    snap = snap[pit.operating_mask(snap)]
    snap = sectors.drop_financials(snap) if "sic" in snap.columns else snap
    snap = snap[metrics._col(snap, "revenues") >= revenue_min]
    no_equity = metrics._col(snap, "eps").isna()

    last_filed = fund.groupby("cik")["filed"].max().reindex(snap.index)
    gone = last_filed < fund["filed"].max() - pd.Timedelta(days=GONE_AFTER_DAYS)
    price = prices_mod.prices_asof(px.close, snap["ticker"], asof)
    priced = price.notna() & (price > 0)
    stale = ~priced & gone & (last_filed <= asof - pd.DateOffset(years=1))
    unlisted = ~priced & ~gone

    status = pd.Series("later_gone", index=snap.index)
    status[priced] = "priced"
    exiting = ~priced & gone & (last_filed <= asof)
    # the shape it left in, from the last annual report it ever filed — which is this one
    distressed = (metrics._col(snap, "net_income") < 0) | (metrics._col(snap, "equity") < 0)
    status[exiting & distressed] = "exit_distressed"
    status[exiting & ~distressed] = "exit_healthy"

    keep = ~(no_equity | stale | unlisted)
    out = metrics.add_fundamental_metrics(snap[keep])
    nothing = pd.Series(np.nan, index=out.index)
    out = metrics.add_price_metrics(out, price=nothing, market_cap=nothing)  # for the growth and F-score columns
    out["status"] = status[keep]
    out["price"] = price[keep]
    out.attrs["excluded"] = {
        "no_listed_equity": int(no_equity.sum()),
        "unlisted_still_filing": int((unlisted & ~no_equity).sum()),
        "already_gone": int((stale & ~no_equity).sum()),
    }
    return out


def _scenario_return(counts: pd.Series, priced_return: float, s: Scenario) -> float:
    """The group's return with its missing members put back at the scenario's returns."""
    n = counts.reindex(STATUSES).fillna(0.0)
    if n.sum() <= 0 or np.isnan(priced_return):
        return float("nan")
    held_like_priced = n["priced"] + n["later_gone"]
    return float(
        (held_like_priced * priced_return + n["exit_healthy"] * s.healthy + n["exit_distressed"] * s.distressed) / n.sum()
    )


def _best_fifth(u: pd.DataFrame, factor: str, sign: int) -> pd.Index:
    x = (u[factor] * sign).replace([np.inf, -np.inf], np.nan).dropna()
    return x.index[x >= x.quantile(0.8)] if len(x) >= 25 else x.index[:0]


def _cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((1.0 + r).prod() ** (1.0 / len(r)) - 1.0) if len(r) else float("nan")


def analyze(
    fund: pd.DataFrame | None = None,
    px: prices_mod.PriceData | None = None,
    *,
    start: int = 2011,
    end: int | None = None,
    rebalance_month: int = 4,
    revenue_min: float = REVENUE_MIN,
) -> SurvivorshipResult:
    """The census, the universe's bounds and each price-free factor's, one April at a time."""
    if fund is None:
        from lti.fundamentals import load_fundamentals

        fund = load_fundamentals()
    if px is None:
        px = prices_mod.load_price_data()
    latest = px.adj.index.max()
    end = end or (latest - pd.DateOffset(years=1)).year
    warnings: list[str] = []

    census_rows, universe_rows = [], []
    factor_rows: dict[str, list[dict]] = {f: [] for f in PRICE_FREE_FACTORS}
    for year in range(start, end + 1):
        asof = pd.Timestamp(year=year, month=rebalance_month, day=1)
        exit_date = asof + pd.DateOffset(years=1)
        if exit_date > latest:
            break
        u = classify(fund, px, asof, revenue_min=revenue_min)
        if len(u) < 50:
            warnings.append(f"{year}: only {len(u)} companies; skipped")
            continue
        fwd = prices_mod.forward_returns(px.adj, u["ticker"], asof, exit_date).where(u["status"] == "priced")
        counts = u["status"].value_counts()
        share = (counts / len(u)).reindex(STATUSES).fillna(0.0)
        census_rows.append({"year": year, "companies": len(u), **share.to_dict(), **u.attrs["excluded"]})

        base = float(fwd.mean())
        row = {"year": year, "survivors": base}
        row.update({s.name: _scenario_return(counts, base, s) for s in SCENARIOS})
        universe_rows.append(row)

        for factor, sign in PRICE_FREE_FACTORS.items():
            if factor not in u.columns:
                continue
            best = _best_fifth(u, factor, sign)
            if len(best) < 10:
                continue
            q = u.loc[best]
            q_counts = q["status"].value_counts()
            q_ret = float(fwd.reindex(best).mean())
            frow = {
                "year": year, "names": len(q),
                "exit_healthy": q_counts.get("exit_healthy", 0) / len(q) - share["exit_healthy"],
                "exit_distressed": q_counts.get("exit_distressed", 0) / len(q) - share["exit_distressed"],
                "survivors": q_ret - base,
            }
            frow.update({s.name: _scenario_return(q_counts, q_ret, s) - row[s.name] for s in SCENARIOS})
            factor_rows[factor].append(frow)

    census = pd.DataFrame(census_rows).set_index("year") if census_rows else pd.DataFrame()
    universe = pd.DataFrame(universe_rows).set_index("year") if universe_rows else pd.DataFrame()
    cagr = universe.apply(_cagr) if not universe.empty else pd.Series(dtype=float)
    factors = pd.DataFrame(
        {f: pd.DataFrame(rows).drop(columns="year").mean() for f, rows in factor_rows.items() if rows}
    ).T
    return SurvivorshipResult(census, universe, cagr, factors, warnings)
