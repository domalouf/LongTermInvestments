"""Single-stock history: annual fundamentals + a valuation time series.

The fundamentals table keeps every filing (restatements included); here we
collapse it to one row per fiscal period (latest ``filed`` wins) and attach the
derived metrics, then build a price-vs-valuation series by carrying each 10-K's
EPS / book value forward from its filing date (``merge_asof``, no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lti import metrics as metrics_mod, pit

# annual line items worth charting, in statement order
INCOME_ITEMS = ["revenues", "gross_profit", "operating_income", "net_income"]
BALANCE_ITEMS = ["assets", "liabilities", "equity"]
CASHFLOW_ITEMS = ["cfo", "capex", "free_cash_flow"]
MARGIN_METRICS = ["gross_margin", "net_margin", "fcf_margin", "roe"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper()


def resolve(fund: pd.DataFrame, symbol: str) -> tuple[int | None, str]:
    """Map a user-entered symbol to ``(cik, price_symbol)``.

    Matches the primary ``ticker`` column first, then the full ``tickers_all``
    list (so e.g. ``JPM`` resolves even when the alphabetical primary pick is a
    baby-bond ticker).
    """
    sym = symbol.upper().strip()
    if not sym:
        return None, sym

    exact = fund[_norm(fund["ticker"]) == sym]
    if not exact.empty:
        return int(exact["cik"].iloc[0]), sym

    if "tickers_all" in fund.columns:
        contains = _norm(fund["tickers_all"]).str.split(",").apply(
            lambda lst: isinstance(lst, list) and sym in lst
        )
        hit = fund[contains]
        if not hit.empty:
            row = hit.sort_values("filed").iloc[-1]
            return int(row["cik"]), sym

    return None, sym


def list_tickers(fund: pd.DataFrame) -> list[str]:
    """Sorted primary tickers that have at least one filing."""
    return sorted(_norm(fund.loc[fund["ticker"].notna(), "ticker"]).unique())


def company_name(fund: pd.DataFrame, cik: int) -> str | None:
    sub = fund.loc[fund["cik"] == cik]
    if sub.empty or "company" not in sub.columns:
        return None
    return str(sub.sort_values("filed")["company"].iloc[-1])


def price_symbol(fund: pd.DataFrame, cik: int, panel: pd.DataFrame, preferred: str) -> str | None:
    """The column in ``panel`` that holds this company's prices, if any."""
    if preferred in panel.columns:
        return preferred
    row = fund.loc[fund["cik"] == cik].sort_values("filed")
    if not row.empty:
        prim = row["ticker"].dropna()
        if not prim.empty and str(prim.iloc[-1]).upper() in panel.columns:
            return str(prim.iloc[-1]).upper()
    return None


def annual_fundamentals(fund: pd.DataFrame, cik: int) -> pd.DataFrame:
    """One row per fiscal period for ``cik`` (latest restatement), metrics added."""
    sub = fund[fund["cik"] == cik].copy()
    if sub.empty:
        return sub
    sub = (
        sub.sort_values(["period_end", "filed"])
        .drop_duplicates("period_end", keep="last")
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    return metrics_mod.add_fundamental_metrics(sub)


def _events_for(events: pd.DataFrame | None, symbol: str, value: str) -> pd.Series:
    """One ticker's ``{date: value}`` rows from a cached ``ticker, date, <value>`` table."""
    if events is None or events.empty:
        return pd.Series(dtype="float64")
    sub = events[events["ticker"].astype(str).str.upper() == symbol.upper().strip()]
    return pd.Series(sub[value].to_numpy(dtype="float64"), index=pd.to_datetime(sub["date"])).sort_index()


def splits_for(splits: pd.DataFrame, symbol: str) -> pd.Series:
    """One ticker's ``{date: ratio}`` split history from the cached split table."""
    return _events_for(splits, symbol, "ratio")


def dividends_for(dividends: pd.DataFrame, symbol: str) -> pd.Series:
    """One ticker's ``{ex-date: amount}`` dividend history from the cached table.

    Already on today's share basis, like the split-adjusted close — no restating
    of its own, so a per-share amount here divides straight into a price there.
    """
    return _events_for(dividends, symbol, "amount")


def dividends_by_year(paid: pd.Series) -> pd.DataFrame:
    """A ``{ex-date: amount}`` series as ``year, dividends`` rows, newest last.

    Calendar years, not fiscal ones: this is the cash a holder received, and a
    part-year at either end is real rather than a gap to fill.
    """
    if paid is None or paid.empty:
        return pd.DataFrame(columns=["year", "dividends"])
    by = paid.groupby(paid.index.year).sum()
    return pd.DataFrame({"year": by.index.astype(int), "dividends": by.to_numpy(dtype="float64")})


def _split_divisor(filed_dates: pd.Series, splits: pd.Series | None) -> np.ndarray:
    """Per-filing factor that restates as-reported per-share figures onto today's
    share basis: the product of every split ratio that took effect *after* the
    filing. Matches the split-adjusted price panel."""
    n = len(filed_dates)
    if splits is None or len(splits) == 0:
        return np.ones(n)
    table = pd.DataFrame({"ticker": "_", "date": pd.to_datetime(splits.index), "ratio": splits.to_numpy(dtype="float64")})
    rows = pd.Series("_", index=range(n))
    return pit.split_factor_after(rows, pd.Series(pd.to_datetime(filed_dates).to_numpy()), table).to_numpy()


def valuation_history(
    annual: pd.DataFrame,
    panel: pd.DataFrame,
    symbol: str,
    freq: str = "W",
    splits: pd.Series | None = None,
) -> pd.DataFrame:
    """``date, price, eps_ttm, bvps, pe, pb`` on a regular grid.

    Each row uses the most recent 10-K known at that date (its filing date), so
    the P/E and P/B series step when a new filing lands. As-reported EPS and book
    value per share are restated onto today's share count using ``splits`` (a
    ``{date: ratio}`` series — :func:`splits_for`) so they line up with the
    split-adjusted price panel; without it, ratios across a stock split are wrong.
    Pass the split-adjusted ``close`` panel, not the dividend-adjusted one: that
    sits below the traded price by every dividend paid since, which drags the
    early part of the P/E history down.
    """
    symbol = symbol.upper().strip()
    if annual.empty or symbol not in panel.columns:
        return pd.DataFrame()

    px = panel[symbol].dropna()
    if px.empty:
        return pd.DataFrame()

    grid = pd.date_range(px.index.min().normalize(), px.index.max().normalize(), freq=freq)
    price = px.reindex(px.index.union(grid)).ffill().reindex(grid)

    cols = [c for c in ["filed", "eps", "book_value_per_share", "fiscal_year"] if c in annual.columns]
    asof = annual[cols].dropna(subset=["filed"]).sort_values("filed").rename(columns={"filed": "date"})
    asof["filed_date"] = asof["date"]

    out = pd.DataFrame({"date": grid, "price": price.to_numpy()})
    out = pd.merge_asof(out, asof, on="date", direction="backward")
    out = out.rename(columns={"eps": "eps_ttm", "book_value_per_share": "bvps"})

    div = _split_divisor(out["filed_date"], splits)
    if "eps_ttm" in out:
        out["eps_ttm"] = out["eps_ttm"] / div
    if "bvps" in out:
        out["bvps"] = out["bvps"] / div

    out["pe"] = out["price"] / out["eps_ttm"].where(out.get("eps_ttm", np.nan) > 0)
    out["pb"] = out["price"] / out["bvps"].where(out.get("bvps", np.nan) > 0)
    return out.drop(columns=["filed_date"])
