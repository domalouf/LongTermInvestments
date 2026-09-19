"""Several years of each company's filings, as they were known on a date.

One year's earnings is a poor guide to what a business earns. It can be a
cyclical peak (Cal-Maine in an avian-flu year), a one-off (Lyft releasing a
tax-asset allowance, Uniti booking a merger gain), or the one good year of a
chronic loss-maker — and every intrinsic-value model multiplies it. Graham
valued companies on average earnings over several years for exactly this
reason.

This module builds that history point in time — each fiscal year as its latest
filing *on or before* the as-of date reported it, so a restatement filed later
isn't known yet — and summarizes it per company:

``eps_norm``
    normalized EPS: the median of the years' EPS, each restated onto today's
    share basis for the splits since its own filing. A median, because one
    outlying year shouldn't move it.
``fcf_norm``
    median free cash flow (a total; per share it uses today's share count).
``profit_years`` / ``history_years``
    how many of the years were profitable, out of how many are on file. A
    year counts only if every figure reported for it is positive — EPS, net
    income *and* operating income: a one-off gain on an operating loss isn't a
    profitable year (Lyft's 2025), and neither is a filing whose losses came
    through the standardizer as profits (The Metals Company's).
``revenue_cagr``
    the revenue trend across the window — the growth input for the models,
    steadier than any single year's change.
``fcf_conversion``
    free cash flow over net income, summed across the years: how much of the
    reported profit turned into cash.
``roic_median``
    the typical return on capital across the years.
``eps_vs_norm``
    the latest year's EPS against the normal one; far above 1 is a peak or a
    one-off, far below a trough.

And from the latest year against the one before it:

``share_growth``
    the change in the share count, restated for splits (Pontiff & Woodgate,
    2008: issuers underperform, buyers-back outperform).
``asset_growth``
    the change in total assets (Cooper, Gulen & Schill, 2008: fast asset
    growers underperform).
``f_score``
    Piotroski's (2000) nine-point financial-strength score. Operating margin
    stands in for gross margin (unreliable in this data), and leverage is
    non-current liabilities over assets, scored as a pass when it didn't rise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lti import metrics, pit

HISTORY_YEARS = 5
MIN_YEARS = 3  # fewer years than this and a median isn't a norm


def history_asof(fund: pd.DataFrame, asof, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Every company's last ``years`` fiscal periods, as known at ``asof``.

    One row per ``(cik, period_end)``: the latest filing for that period filed
    on or before ``asof``. Sorted by cik, then period.
    """
    known = fund[fund["filed"] <= pd.Timestamp(asof)]
    per_period = known.sort_values(["cik", "period_end", "filed"]).drop_duplicates(["cik", "period_end"], keep="last")
    return per_period.groupby("cik", sort=False).tail(years)


def summarize_history(hist: pd.DataFrame, splits: pd.DataFrame | None = None, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per-company summary of a :func:`history_asof` frame (see the module docstring), indexed by cik."""
    if hist.empty:
        return pd.DataFrame(index=pd.Index([], name="cik"))

    h = hist.sort_values(["cik", "period_end"]).copy()
    eps = h["eps"].astype("float64") if "eps" in h.columns else pd.Series(np.nan, index=h.index)
    if "ticker" in h.columns and "filed" in h.columns:
        eps = eps / pit.split_factor_after(h["ticker"], h["filed"], splits)
    h["eps_restated"] = eps
    ni = h["net_income"].astype("float64") if "net_income" in h.columns else pd.Series(np.nan, index=h.index)
    op = (
        h["operating_income_reported"].astype("float64")
        if "operating_income_reported" in h.columns
        else pd.Series(np.nan, index=h.index)
    )
    # profitable only if every signal reported for the year is positive; NaN if none is
    signals = pd.concat([(x > 0).astype("float64").where(x.notna()) for x in (eps, ni, op)], axis=1)
    h["profitable"] = signals.min(axis=1)
    if "roic" not in h.columns:
        h = metrics.add_capital_metrics(h)

    g = h.groupby("cik", sort=False)

    def median_of(col: str) -> pd.Series:
        if col not in h.columns:
            return pd.Series(np.nan, index=g.size().index)
        return g[col].median().where(g[col].count() >= min_years)

    out = pd.DataFrame(index=g.size().index)
    out["history_years"] = g["profitable"].count()
    out["profit_years"] = g["profitable"].sum().where(out["history_years"] > 0)
    out["eps_norm"] = median_of("eps_restated")
    out["fcf_norm"] = median_of("free_cash_flow")
    out["roic_median"] = median_of("roic")

    if "free_cash_flow" in h.columns:
        both = h[h["free_cash_flow"].notna() & ni.notna()]
        gb = both.groupby("cik")
        fcf_sum, ni_sum = gb["free_cash_flow"].sum(), gb["net_income"].sum()
        enough = gb.size() >= min_years
        out["fcf_conversion"] = (fcf_sum / ni_sum.where(ni_sum > 0)).where(enough)

    if "revenues" in h.columns:
        rev = h[h["revenues"] > 0]
        gr = rev.groupby("cik")
        first, last = gr[["revenues", "period_end"]].first(), gr[["revenues", "period_end"]].last()
        span = (last["period_end"] - first["period_end"]).dt.days / 365.25
        cagr = (last["revenues"] / first["revenues"]) ** (1.0 / span.where(span >= min_years - 1.1)) - 1.0
        out["revenue_cagr"] = cagr

    latest = h.drop_duplicates("cik", keep="last").set_index("cik")["eps_restated"]  # the latest year, even if blank
    out["eps_vs_norm"] = latest / out["eps_norm"].where(out["eps_norm"] > 0)

    year = _latest_two_years(h, splits)
    out["share_growth"] = year["shares"] / year["shares_prev"] - 1.0
    out["asset_growth"] = year["assets"] / year["assets_prev"] - 1.0
    out["f_score"] = _piotroski(year)
    return out


_CHANGE_COLS = [
    "shares", "assets", "net_income", "cfo", "liabilities_noncurrent",
    "assets_current", "liabilities_current", "revenues", "ebit",
]


def _latest_two_years(h: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Each company's latest fiscal year, with ``<col>_prev`` from the year before —
    NaN unless the two periods are a year apart."""
    h = h.copy()
    h["shares"] = (
        h["shares_outstanding"] * pit.split_factor_after(h["ticker"], h["filed"], splits)
        if {"shares_outstanding", "ticker", "filed"} <= set(h.columns)
        else np.nan
    )
    if "ebit" not in h.columns and {"operating_income_reported", "operating_income"} & set(h.columns):
        h["ebit"] = metrics.ebit(h)
    for col in _CHANGE_COLS:
        if col not in h.columns:
            h[col] = np.nan
    prev = h.groupby("cik", sort=False)[[*_CHANGE_COLS, "period_end"]].shift(1)
    a_year = (h["period_end"] - prev["period_end"]).dt.days.between(300, 430)
    for col in _CHANGE_COLS:
        h[f"{col}_prev"] = prev[col].where(a_year)
    return h.drop_duplicates("cik", keep="last").set_index("cik")


def _piotroski(y: pd.DataFrame) -> pd.Series:
    """Piotroski's F-score from :func:`_latest_two_years`; NaN when fewer than
    seven of the nine signals can be scored."""
    def ratio(a, b):
        return a / b.where(b > 0)

    roa, roa_prev = ratio(y["net_income"], y["assets"]), ratio(y["net_income_prev"], y["assets_prev"])
    lev, lev_prev = ratio(y["liabilities_noncurrent"], y["assets"]), ratio(y["liabilities_noncurrent_prev"], y["assets_prev"])
    cr, cr_prev = ratio(y["assets_current"], y["liabilities_current"]), ratio(y["assets_current_prev"], y["liabilities_current_prev"])
    margin, margin_prev = ratio(y["ebit"], y["revenues"]), ratio(y["ebit_prev"], y["revenues_prev"])
    turn, turn_prev = ratio(y["revenues"], y["assets"]), ratio(y["revenues_prev"], y["assets_prev"])

    def signal(passed: pd.Series, *inputs: pd.Series) -> pd.Series:
        known = pd.concat(inputs, axis=1).notna().all(axis=1)
        return passed.astype("float64").where(known)

    signals = pd.concat(
        [
            signal(roa > 0, roa),
            signal(y["cfo"] > 0, y["cfo"]),
            signal(roa > roa_prev, roa, roa_prev),
            signal(y["cfo"] > y["net_income"], y["cfo"], y["net_income"]),
            signal(lev <= lev_prev, lev, lev_prev),
            signal(cr > cr_prev, cr, cr_prev),
            signal(y["shares"] <= y["shares_prev"], y["shares"], y["shares_prev"]),
            signal(margin > margin_prev, margin, margin_prev),
            signal(turn > turn_prev, turn, turn_prev),
        ],
        axis=1,
    )
    return signals.sum(axis=1).where(signals.notna().sum(axis=1) >= 7)


def add_history(
    snap: pd.DataFrame,
    fund: pd.DataFrame,
    asof,
    splits: pd.DataFrame | None = None,
    years: int = HISTORY_YEARS,
) -> pd.DataFrame:
    """Attach :func:`summarize_history` to a snapshot (indexed by cik), plus the
    per-share and price-based normalized metrics: ``fcf_ps_norm``, ``pe_norm``,
    ``earnings_yield_norm`` and ``fcf_yield_norm``.

    Per-share values use the snapshot's share count, already on today's basis.
    """
    summary = summarize_history(history_asof(fund[fund["cik"].isin(snap.index)], asof, years), splits)
    out = snap.join(summary, how="left")
    shares = out["shares_outstanding"] if "shares_outstanding" in out.columns else pd.Series(np.nan, index=out.index)
    out["fcf_ps_norm"] = metrics._safe_div(out["fcf_norm"], shares)
    if "price" in out.columns:
        price = out["price"]
        out["pe_norm"] = metrics._safe_div(price, out["eps_norm"].where(out["eps_norm"] > 0))
        out["earnings_yield_norm"] = metrics._safe_div(out["eps_norm"], price)
        out["fcf_yield_norm"] = metrics._safe_div(out["fcf_ps_norm"], price)
    return out
