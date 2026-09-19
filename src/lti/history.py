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
    return out


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
