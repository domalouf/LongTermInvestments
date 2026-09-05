"""Per-stock metric computation.

Fundamental metrics depend only on the ``fundamentals`` frame. Price metrics
(P/E, P/B, earnings yield, market cap) additionally need a price and an as-of
date, so they are computed separately — the backtest and screener pass in the
price that was known at the relevant date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_METRICS = [
    "debt_to_equity",
    "current_ratio",
    "roe",
    "net_margin",
    "gross_margin",
    "revenue_growth_1y",
    "eps_growth_1y",
    "fcf_margin",
]

PRICE_METRICS = ["pe", "pb", "earnings_yield", "peg"]

# lower value = "better" (used as the default sort direction in ranking)
LOWER_IS_BETTER = {"pe", "pb", "debt_to_equity", "peg"}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_fundamental_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    equity_pos = df["equity"].where(df["equity"] > 0) if "equity" in df else pd.Series(np.nan, index=df.index)

    if {"liabilities", "equity"} <= set(df.columns):
        df["debt_to_equity"] = _safe_div(df["liabilities"], equity_pos)
    if {"assets_current", "liabilities_current"} <= set(df.columns):
        df["current_ratio"] = _safe_div(df["assets_current"], df["liabilities_current"])
    if {"net_income", "equity"} <= set(df.columns):
        df["roe"] = _safe_div(df["net_income"], equity_pos)
    if {"net_income", "revenues"} <= set(df.columns):
        df["net_margin"] = _safe_div(df["net_income"], df["revenues"])
    if {"gross_profit", "revenues"} <= set(df.columns):
        df["gross_margin"] = _safe_div(df["gross_profit"], df["revenues"])
    if {"free_cash_flow", "revenues"} <= set(df.columns):
        df["fcf_margin"] = _safe_div(df["free_cash_flow"], df["revenues"])
    if {"revenues", "revenues_prev"} <= set(df.columns):
        df["revenue_growth_1y"] = _safe_div(df["revenues"], df["revenues_prev"].where(df["revenues_prev"] > 0)) - 1.0
    if {"eps", "eps_prev"} <= set(df.columns):
        # only meaningful when both years are positive
        both_pos = (df["eps"] > 0) & (df["eps_prev"] > 0)
        df["eps_growth_1y"] = np.where(both_pos, df["eps"] / df["eps_prev"] - 1.0, np.nan)

    if {"equity", "shares_outstanding"} <= set(df.columns):
        df["book_value_per_share"] = _safe_div(df["equity"], df["shares_outstanding"])

    return df


def add_price_metrics(
    df: pd.DataFrame,
    price: pd.Series,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    """``price`` / ``market_cap`` are indexed the same way as ``df`` (typically by cik)."""
    df = df.copy()
    price = price.reindex(df.index)
    df["price"] = price

    if "eps" in df.columns:
        eps_pos = df["eps"].where(df["eps"] > 0)
        df["pe"] = _safe_div(price, eps_pos)
        df["earnings_yield"] = _safe_div(df["eps"], price)
        # PEG: P/E over the earnings growth rate in percentage points (>0 only)
        if "eps_growth_1y" in df.columns:
            growth_pts = (df["eps_growth_1y"] * 100).where(df["eps_growth_1y"] > 0)
            df["peg"] = _safe_div(df["pe"], growth_pts)
    if "book_value_per_share" in df.columns:
        bvps_pos = df["book_value_per_share"].where(df["book_value_per_share"] > 0)
        df["pb"] = _safe_div(price, bvps_pos)

    if market_cap is not None:
        df["market_cap"] = market_cap.reindex(df.index)
    elif "shares_outstanding" in df.columns:
        df["market_cap"] = price * df["shares_outstanding"]

    return df
