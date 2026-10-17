"""Point-in-time snapshots (no look-ahead).

:func:`snapshot_asof` is the filing view — what had been filed by a date: each
company's latest 10-K, or the trailing-twelve-month row of a later 10-Q
(:mod:`lti.quarterly`) where the fundamentals table carries one.
:func:`company_snapshot` and :func:`priced_snapshot` build the investable
universe on top of it: per-share figures restated onto the price panels' share
basis, non-operating filers dropped, metrics and prices attached. The screener,
backtest, factor analysis and valuation all go through ``priced_snapshot``, so
they all see the same universe in the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lti import metrics, prices as prices_mod, sectors


def annual(fund: pd.DataFrame) -> pd.DataFrame:
    """``fund`` without its trailing-twelve-month 10-Q rows: one row per fiscal year.

    For whatever counts years — the five-year history, a company's annual
    record — rather than wanting the freshest numbers.
    """
    if "form" not in fund.columns or not (fund["form"] == "10-Q").any():
        return fund
    return fund[fund["form"] != "10-Q"]


def snapshot_asof(
    fund: pd.DataFrame,
    asof: pd.Timestamp,
    max_staleness_days: int = 550,
) -> pd.DataFrame:
    """The fundamentals an investor could have known at ``asof``.

    1. keep only filings with ``filed <= asof``
    2. per ``cik``, take the most recent ``period_end``; if several filings cover
       that period (restatement / 10-K/A), keep the one with the latest ``filed``
    3. drop rows whose ``period_end`` is older than ``asof - max_staleness_days``
       (company stopped filing / went dark)
    4. one row per cik, indexed by cik
    """
    asof = pd.Timestamp(asof)
    known = fund[fund["filed"] <= asof]
    if known.empty:
        return known.set_index("cik") if "cik" in known.columns else known

    known = known.sort_values(["cik", "period_end", "filed"])
    latest = known.drop_duplicates("cik", keep="last").copy()

    cutoff = asof - pd.Timedelta(days=max_staleness_days)
    latest = latest[latest["period_end"] >= cutoff]

    latest["asof"] = asof
    latest["days_since_period_end"] = (asof - latest["period_end"]).dt.days
    return latest.set_index("cik")


# --- share basis -------------------------------------------------------------


def split_factor_after(
    tickers: pd.Series,
    dates: pd.Series,
    splits: pd.DataFrame | None,
) -> pd.Series:
    """Per row, the product of the ticker's split ratios strictly after its date (1.0 if none).

    Multiply a share count reported on that date's basis by this — or divide a
    per-share figure by it — to put it on today's basis, the one the price
    panels are back-adjusted to. ``splits`` is a ``ticker, date, ratio`` table
    (a 4-for-1 split has ratio 4, a 1-for-10 reverse split 0.1).
    """
    out = np.ones(len(tickers))
    if splits is None or splits.empty or len(tickers) == 0:
        return pd.Series(out, index=tickers.index)

    sp = splits.loc[splits["ratio"] > 0, ["ticker", "date", "ratio"]].copy()
    sp["ticker"] = sp["ticker"].astype(str).str.upper()
    sp["date"] = pd.to_datetime(sp["date"]).astype("datetime64[ns]")
    # newest first, so a running product is "this split and every later one"
    sp = sp.sort_values(["ticker", "date"], ascending=[True, False])
    sp["after"] = sp.groupby("ticker")["ratio"].cumprod()

    q = pd.DataFrame(
        {
            "ticker": tickers.astype("string").str.upper().to_numpy(),
            "date": pd.to_datetime(pd.Series(dates.to_numpy())).astype("datetime64[ns]").to_numpy(),
            "_pos": np.arange(len(tickers)),
        }
    ).dropna(subset=["ticker", "date"])
    if q.empty:
        return pd.Series(out, index=tickers.index)
    q["ticker"] = q["ticker"].astype(str)

    hit = pd.merge_asof(
        q.sort_values("date"),
        sp[["ticker", "date", "after"]].sort_values("date"),
        on="date",
        by="ticker",
        direction="forward",  # the first split after the date ...
        allow_exact_matches=False,  # ... strictly after: one on the filing day is already in it
    )
    out[hit["_pos"].to_numpy()] = hit["after"].fillna(1.0).to_numpy()
    return pd.Series(out, index=tickers.index)


def restate_per_share(snap: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Put a snapshot's as-reported per-share figures on the price panels' share basis.

    A 10-K reports EPS and share counts on the share basis of the day it was
    filed (a split before filing is already restated into it). The price
    panels are back-adjusted for every split since, so paired as-is, a company
    that split *later* looks cheaper by the split ratio — and companies tend to
    split after their stock has run up, so a cheapness screen quietly selects
    future winners. Dividing per-share figures (and multiplying share counts)
    by the splits after each filing removes that look-ahead.

    ``eps_prev`` came from the previous period's filing, so it is restated by
    the splits after ``filed_prev`` instead.
    """
    if splits is None or splits.empty or snap.empty or "ticker" not in snap.columns:
        return snap
    out = snap.copy()
    factor = split_factor_after(out["ticker"], out["filed"], splits)
    out["split_factor"] = factor
    if "eps" in out.columns:
        out["eps"] = out["eps"] / factor
    if "shares_outstanding" in out.columns:
        out["shares_outstanding"] = out["shares_outstanding"] * factor
    if "eps_prev" in out.columns:
        if "filed_prev" not in out.columns:
            raise KeyError(
                "cannot restate eps_prev without 'filed_prev' — load the table with "
                "lti.fundamentals.load_fundamentals(), which adds it"
            )
        out["eps_prev"] = out["eps_prev"] / split_factor_after(out["ticker"], out["filed_prev"], splits)
    return out


# --- the universe ------------------------------------------------------------


def operating_mask(snap: pd.DataFrame) -> pd.Series:
    """Operating businesses: positive revenue, and not a commodity pool.

    Commodity and crypto trusts, blank-check shells and liquidating trusts all
    file 10-Ks, but there is no business to value — and trusts report the
    mark-to-market on their holdings as earnings, which sorts them to the top of
    any cheapness ranking at the peak of a gold rally.
    """
    mask = (metrics._col(snap, "revenues") > 0).astype(bool)
    if {"sic", "company"} <= set(snap.columns):
        mask &= ~sectors.is_commodity_pool(snap["sic"], snap["company"])
    return mask


def company_snapshot(
    fund: pd.DataFrame,
    asof,
    splits: pd.DataFrame | None = None,
    *,
    operating_only: bool = True,
    max_staleness_days: int = 550,
) -> pd.DataFrame:
    """The companies an investor could have screened at ``asof``, before prices.

    :func:`snapshot_asof`, then: companies with a ticker; optionally operating
    businesses only (:func:`operating_mask`); per-share figures on the price
    panels' basis (:func:`restate_per_share`); fundamental metrics.
    """
    snap = snapshot_asof(fund, pd.Timestamp(asof), max_staleness_days)
    if snap.empty:
        return snap
    snap = snap[snap["ticker"].notna()]
    if operating_only:
        snap = snap[operating_mask(snap)]
    snap = restate_per_share(snap, splits)
    return metrics.add_fundamental_metrics(snap)


def priced_snapshot(
    fund: pd.DataFrame,
    asof,
    px: prices_mod.PriceData,
    *,
    operating_only: bool = True,
    max_staleness_days: int = 550,
    with_history: bool = False,
) -> pd.DataFrame:
    """:func:`company_snapshot`, priced at ``asof``: price, market cap and price metrics.

    The price is the split-adjusted close on or just before ``asof``, never the
    dividend-adjusted one — that is lower by every dividend paid since, which
    would flatter past dividend payers. A company with no recent close keeps a
    NaN price (and so a NaN market cap and P/E).

    Dividends known at ``asof`` come with it: ``dps_ttm`` (the last year's
    payments), ``dividend_yield``, ``dividend_growth_5y`` and ``payout_ratio``.

    ``with_history`` adds what takes several years of filings: the normalized
    and consistency metrics (:data:`lti.metrics.HISTORY_METRICS`, via
    :mod:`lti.history`) and the fair-value upsides built on them
    (:data:`lti.metrics.VALUATION_METRICS`), at the interest rates of ``asof``
    where ``px`` carries them. It costs a pass over the history,
    so it's only worth asking for when something ranks on those.
    """
    asof = pd.Timestamp(asof)
    snap = company_snapshot(
        fund, asof, px.splits, operating_only=operating_only, max_staleness_days=max_staleness_days
    )
    if snap.empty:
        return snap
    price = prices_mod.prices_asof(px.close, snap["ticker"], asof)
    shares = metrics._col(snap, "shares_outstanding")
    snap = metrics.add_price_metrics(snap, price=price, market_cap=price * shares)
    # 12-1 momentum (Jegadeesh & Titman): the total return from a year ago to a
    # month ago, skipping the last month's short-term reversal
    then = prices_mod.prices_asof(px.adj, snap["ticker"], asof - pd.DateOffset(months=12))
    recent = prices_mod.prices_asof(px.adj, snap["ticker"], asof - pd.DateOffset(months=1))
    snap["momentum_12_1"] = recent / then.where(then > 0) - 1.0
    # What the company actually paid per share over the last year, from the
    # dividend events rather than the cash-flow statement: the SEC tag covers a
    # quarter of filers, lags by up to a year, and lumps preferred in with
    # common. Both sides of the yield are on today's share basis.
    dps = prices_mod.trailing_dividends(px.dividends, snap["ticker"], asof)
    snap["dps_ttm"] = dps.where(price.notna())  # unpriced means unfetched, not unpaid
    snap["dividend_yield"] = metrics._safe_div(snap["dps_ttm"], price)
    snap["dividend_growth_5y"] = prices_mod.dividend_growth(px.dividends, snap["ticker"], asof)
    eps = metrics._col(snap, "eps")
    snap["payout_ratio"] = metrics._safe_div(snap["dps_ttm"], eps.where(eps > 0))
    if with_history:
        from lti import history, valuation  # both build on this module

        snap = history.add_history(snap, fund, asof, px.splits)
        # valued at the rates of the day, not today's or a fixed 9%
        snap = valuation.add_fair_value_metrics(snap, valuation.market_assumptions(asof, px.rates))
    return snap
